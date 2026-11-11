"""SLOWWM basis builder (PREREG_V8F_SLOWWM, sha256 7c2775e3...94ca): the SLOW (temporal CCA, lag 1.0 s) and PCA-B bases of the frozen DINOv3 token mean ``nm``, their per-frame caches, and control 3.

``nm`` [T, 1024] = ``tanitad.data.v8f_wm_windows.norm_mean_tokens(feats)`` = the token mean of the PARAMETER-FREE layer-normalised features: the very tensor ``V8FMini.pooled_adapt_mean`` consumes (its docstring), adapter-independent.
It is NOT re-implemented here. The only inputs of the builder are ``nm`` of the ``pool4x10`` grid, the ``ok`` flag and the frame index (clip-time adjacency): no ego, action, label or situation array is read (control 4; the audit is in
``controls_slowwm.py``).

Definitions (card, "Bases"): x = nm - mean over the TRAIN-window frames; C = E[x x^T]; S = 1/2 (E[x_t x_{t+5}^T] + E[x_{t+5} x_t^T]) over same-clip pairs with both frames in the fit set (lag 5 steps = 1.0 s; the step is 0.2 s);
ridge = 1e-3 tr(C)/1024; SLOW = the top-256 generalised eigenvectors of S v = mu (C + ridge I) v, V^T (C + ridge I) V = I, u = V^T x; PCA-B = the top-256 eigenvectors of C scaled by 1/sqrt(variance), u = Lambda^-1/2 U^T x.
FIT SET = the frames covered by at least one window of the ladder's train window list: the clips of the train manifest minus ``--exclude`` (sha12 rank slice, as ``build_data``), every full ``--hmax``-frame window (``wm_windows``,
stride 1) whose frames are all ``ok`` (the same ``ok`` as ``theory/ladder/zcache.py``, so the window list is the trainer's, 40,358 windows on box 2: ``--expect-windows``). Nothing is re-selected.

Statistics are streamed per clip in float64 (raw moments, centred at the end): RAM is a few MB, never the frames.

Modes:
  fit    : <out>/fit_stats.npz (n, mean, C, S, S_xclip), basis_slow.npz, basis_pca.npz, SPECTRA.json, FIT_RECORD.json. ``--spool DIR`` keeps nm per clip so ``cache`` does not read the raw features again.
  cache  : <out>/<kind>/<split>/<clip>.npz (zbasis f32 [T, 256], ok bool [T], t_s f32 [T], basis_sha str) + BASIS_MANIFEST.json + basis.npz beside them (the trainer's --wm-zcache is that directory).
  check3 : control 3 on the clips of --clips-file: cached u vs the basis applied to nm recomputed by the LIVE path (ClipStore.get -> norm_mean_tokens), cov(u) = I, SLOW's top mu vs PCA-B's top direction's lag-1 s autocorrelation,
           a shuffled-time basis must fail that comparison, a wrong-kind basis must fail the identity.
Pure numerical helpers (``Acc``, ``solve_slow``, ``solve_pca``, ``apply_basis`` ...) need only numpy. ``ZC_REPO`` points at the stack tree (as zcache.py); ``ZCACHE_DIR`` at theory/ladder (only ``--compare-zcache``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = os.environ.get("ZC_REPO", "D:/tanitad/repo")
for _p in (REPO + "/stack", REPO + "/taniteval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DIM_NM, K_OUT, LAG, RIDGE_REL, HMAX, DT = 1024, 256, 5, 1e-3, 91, 0.2
GRID = "pool4x10"
FORMAT = "zbasis-v1"
KINDS = ("slow", "pca")


# ---------------------------------------------------------------------------------------------------- hashing
def sha_file(p, blk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def basis_sha256(arrays: dict) -> str:
    """sha256 over name + shape + dtype + bytes of the sorted arrays ``kind``, ``mean``, ``proj``: the SAME formula as ``v8f_mini_train.basis_sha256_of_file`` (pinned by a test)."""
    h = hashlib.sha256()
    for k in sorted(("kind", "mean", "proj")):
        t = np.ascontiguousarray(arrays[k])
        h.update(k.encode() + str(tuple(t.shape)).encode() + str(t.dtype).encode() + t.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- numerics (numpy only)
def fit_mask(ok: np.ndarray, F: int) -> np.ndarray:
    """Frames covered by at least one FULL ``F``-frame window ``[t-F+1, t]`` (t in [F-1, T-1]) whose frames are all ``ok``: the frames of the ladder's train windows (``wm_windows`` + ``ok_windows``)."""
    T = len(ok)
    m = np.zeros(T, bool)
    if T < F:
        return m
    cs = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    t = np.arange(F - 1, T)
    good = (cs[t + 1] - cs[t - F + 1]) == F
    for te in t[good]:
        m[te - F + 1:te + 1] = True
    return m


def n_windows(ok: np.ndarray, F: int) -> int:
    T = len(ok)
    if T < F:
        return 0
    cs = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    t = np.arange(F - 1, T)
    return int(((cs[t + 1] - cs[t - F + 1]) == F).sum())


class Acc:
    """Streaming raw moments over the fit frames: sum x, sum x x^T, and for same-clip pairs (t, t+lag) with both frames in the fit set sum a, sum b, sum a b^T. The SHUFFLED-time pair is (t, (t + shift) mod T) with
    the pair (frame t of this clip, a pseudo-random frame of the PREVIOUS clip): no time adjacency and no shared scene, the planted shuffled-time control for the temporal statistic (control 3; a within-clip shuffle keeps the clip's static scene
    component, which is slow at any lag, and so is not a shuffle of what the statistic measures). float64; centred in :meth:`finalize`."""

    def __init__(self, d: int, lag: int = LAG):
        self.d, self.lag = d, lag
        self.n = 0
        self.sx = np.zeros(d)
        self.sxx = np.zeros((d, d))
        self.np_ = 0
        self.sa, self.sb, self.sab = np.zeros(d), np.zeros(d), np.zeros((d, d))
        self.nsh = 0
        self.sa_h, self.sb_h, self.sab_h = np.zeros(d), np.zeros(d), np.zeros((d, d))
        self._prev = None

    def add_clip(self, nm: np.ndarray, fit: np.ndarray):
        x = np.asarray(nm, np.float64)
        T = len(x)
        X = x[fit]
        if len(X):
            self.n += len(X)
            self.sx += X.sum(0)
            self.sxx += X.T @ X
        lag = self.lag
        if T > lag:
            sel = fit[:T - lag] & fit[lag:]
            A, B = x[:T - lag][sel], x[lag:][sel]
            if len(A):
                self.np_ += len(A)
                self.sa += A.sum(0)
                self.sb += B.sum(0)
                self.sab += A.T @ B
        if self._prev is not None:
            px, pfit = self._prev
            n_ = min(T, len(px))
            pi = (np.arange(n_) * 37 + 11) % len(px)                  # a fixed pseudo-random frame of the previous clip (the same-index pairing keeps the frame-index-dependent structure the clips share)
            sel = fit[:n_] & pfit[pi]
            A, B = x[:n_][sel], px[pi][sel]
            if len(A):
                self.nsh += len(A)
                self.sa_h += A.sum(0)
                self.sb_h += B.sum(0)
                self.sab_h += A.T @ B
        self._prev = (x, fit)

    def finalize(self) -> dict:
        n = max(self.n, 1)
        m = self.sx / n
        C = self.sxx / n - np.outer(m, m)

        def cross(npair, sa, sb, sab):
            k = max(npair, 1)
            M = sab / k - np.outer(sa / k, m) - np.outer(m, sb / k) + np.outer(m, m)
            return 0.5 * (M + M.T)
        return {"n": self.n, "n_pairs": self.np_, "n_pairs_xclip": self.nsh, "mean": m, "C": 0.5 * (C + C.T),
                "S": cross(self.np_, self.sa, self.sb, self.sab), "S_xclip": cross(self.nsh, self.sa_h, self.sb_h, self.sab_h)}


def _sign_fix(W: np.ndarray) -> np.ndarray:
    i = np.abs(W).argmax(0)
    s = np.sign(W[i, np.arange(W.shape[1])])
    s[s == 0] = 1.0
    return W * s


def solve_slow(C: np.ndarray, S: np.ndarray, k: int, ridge_rel: float = RIDGE_REL):
    """Top-``k`` generalised eigenvectors of S v = mu (C + ridge I) v, normalised V^T (C + ridge I) V = I. Returns (V [d, k], mu [k] descending, ridge)."""
    d = C.shape[0]
    ridge = ridge_rel * float(np.trace(C)) / d
    B = C + ridge * np.eye(d)
    L = np.linalg.cholesky(B)
    Li = np.linalg.inv(L)
    M = Li @ S @ Li.T
    w, Q = np.linalg.eigh(0.5 * (M + M.T))
    o = np.argsort(w)[::-1][:k]
    V = _sign_fix(Li.T @ Q[:, o])
    return V, w[o], ridge


def solve_pca(C: np.ndarray, k: int):
    """Top-``k`` eigenvectors of C, each scaled to unit variance: U Lambda^-1/2. Returns (P [d, k], variance [k] descending)."""
    w, Q = np.linalg.eigh(C)
    o = np.argsort(w)[::-1][:k]
    lam = w[o]
    return _sign_fix(Q[:, o]) / np.sqrt(lam)[None, :], lam


def autocorr_of(W: np.ndarray, C: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Lag autocorrelation of every direction (column of W): w^T S w / w^T C w, from the fit statistics (a unit-variance direction: its lag covariance)."""
    return np.einsum("dk,de,ek->k", W, S, W) / np.einsum("dk,de,ek->k", W, C, W)


def cov_u(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """cov(u) of u = W^T x on the fit frames: W^T C W."""
    return W.T @ C @ W


def slow_vs_pca(stats: dict, V: np.ndarray, mu: np.ndarray, P: np.ndarray) -> dict:
    """Control 3's ordering: the measured lag-1 s autocorrelation of SLOW's top direction and its stored top mu vs the measured lag-1 s autocorrelation of PCA-B's top direction. Pass = both exceed."""
    ac_s = float(autocorr_of(V[:, :1], stats["C"], stats["S"])[0])
    ac_p = float(autocorr_of(P[:, :1], stats["C"], stats["S"])[0])
    return {"slow_top_mu": float(mu[0]), "slow_top_autocorr_measured": ac_s, "pca_top_autocorr_measured": ac_p, "pass": bool(ac_s > ac_p and float(mu[0]) > ac_p)}


def apply_basis(nm: np.ndarray, mean: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """u = (nm - mean) @ proj, float32 in, float32 out: the ONE expression of the cache writer, the check and the read hook."""
    return ((np.asarray(nm, np.float32) - np.asarray(mean, np.float32)) @ np.asarray(proj, np.float32)).astype(np.float32)


def save_basis(path: str, kind: str, mean: np.ndarray, proj: np.ndarray, spectrum: np.ndarray, meta: dict) -> str:
    arr = {"kind": np.array(kind), "mean": np.asarray(mean, np.float32), "proj": np.asarray(proj, np.float32)}
    sha = basis_sha256(arr)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, spectrum=np.asarray(spectrum, np.float64), meta_json=np.array(json.dumps(meta, sort_keys=True)), basis_sha256=np.array(sha), **arr)
    os.replace(tmp, path)
    return sha


def load_basis(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        arr = {"kind": np.asarray(z["kind"]), "mean": np.asarray(z["mean"]), "proj": np.asarray(z["proj"])}
        out = {"kind": str(arr["kind"]), "mean": arr["mean"], "proj": arr["proj"], "spectrum": np.asarray(z["spectrum"]), "sha": basis_sha256(arr), "stored_sha": str(z["basis_sha256"])}
    if out["sha"] != out["stored_sha"]:
        raise ValueError(f"{path}: stored sha {out['stored_sha'][:12]} != recomputed {out['sha'][:12]}")
    return out


# ---------------------------------------------------------------------------------------------------- data (needs the stack)
def _data():
    from tanitad.data import v8f_mini_data as D
    from tanitad.data import v8f_wm_windows as WW
    return D, WW


def nm_of(feats16) -> np.ndarray:
    """feats16 [T, 40, 1024] -> nm [T, 1024] float32 through the model's own parameter-free LayerNorm token mean (``norm_mean_tokens``)."""
    _, WW = _data()
    return WW.norm_mean_tokens(feats16).numpy().astype(np.float32)


class Source:
    """One shared ClipStore; ``inputs(clip)`` = (nm f32 [T, 1024], ok bool [T]) with the ``ok`` and the grid read exactly as ``theory/ladder/zcache.py clip_inputs`` (its lines, one store instead of one per clip)."""

    def __init__(self, root, split, tracks_path, spool=None):
        import torch
        self.D, _ = _data()
        self.torch = torch
        self.root, self.split, self.spool = root, split, spool
        self.tracks = self.D.load_native_tracks(tracks_path)
        self.store = self.D.ClipStore(root, split, lru=1, grid=GRID, kin_mode="yaw", tracks=self.tracks)

    def clips(self) -> list:
        return [c for c, _ in self.D.read_manifest(self.root, self.split)]

    def inputs(self, clip):
        sp = os.path.join(self.spool, f"{clip}.npz") if self.spool else None
        if sp and os.path.exists(sp):
            with np.load(sp) as z:
                return np.asarray(z["nm"], np.float32), np.asarray(z["ok"], bool)
        D, torch = self.D, self.torch
        kin = self.store.get_kin(clip)
        kin3, T = kin["kin3"], len(kin["pose"])
        p10 = np.full((T, 8), np.nan, np.float32)
        p10[1:] = D._fk().kin_features(kin3[1:]).astype(np.float32)
        ok = np.isfinite(p10).all(1) | (np.arange(T) < 1)
        path = f"{self.root}/{self.split}/clips/{clip}.pt"
        try:
            o = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except (RuntimeError, TypeError):
            o = torch.load(path, map_location="cpu", weights_only=True)
        feats = o[GRID].to(torch.float16).clone()
        del o
        assert feats.shape[0] == T, (clip, feats.shape, T)
        nm = nm_of(feats)
        if sp:
            tmp = sp + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, nm=nm, ok=ok)
            os.replace(tmp, sp)
        return nm, ok


def train_clip_list(src: Source, exclude: str, clips_file: str | None, limit: int | None) -> list:
    """The ladder's train clips: manifest order minus the ``--exclude lo:hi`` slice of the sha12-sorted list (``build_data``)."""
    D = src.D
    clips_all = src.clips()
    if exclude:
        lo, hi = (int(x) for x in exclude.split(":"))
        ex = set(sorted(clips_all, key=D.sha12)[lo:hi])
        clips_all = [c for c in clips_all if c not in ex]
    if clips_file:
        sel = json.load(open(clips_file, encoding="utf-8"))
        keep = set(sel[src.split] if isinstance(sel, dict) else sel)
        clips_all = [c for c in clips_all if c in keep]
    if limit:
        clips_all = clips_all[:limit]
    return clips_all


# ---------------------------------------------------------------------------------------------------- fit
def cmd_fit(a):
    src = Source(a.root, a.split, a.tracks, a.spool)
    if a.spool:
        os.makedirs(a.spool, exist_ok=True)
    clips = train_clip_list(src, a.exclude, a.clips_file, a.limit)
    if a.half:                                              # SMOKE only: every second clip of the list
        clips = clips[::2]
    acc = Acc(DIM_NM, a.lag)
    nwin, t0 = 0, time.time()
    for i, c in enumerate(clips):
        nm, ok = src.inputs(c)
        nwin += n_windows(ok, a.hmax)
        acc.add_clip(nm, fit_mask(ok, a.hmax))
        if (i + 1) % 100 == 0 or i == len(clips) - 1:
            print(f"{time.strftime('%H:%M:%S')} fit {i + 1}/{len(clips)} frames={acc.n} windows={nwin} {(time.time() - t0) / (i + 1):.2f} s/clip", flush=True)
    if a.expect_windows is not None and nwin != a.expect_windows:
        raise SystemExit(f"ZZFIT-REFUSEDZZ the ok-window count is {nwin}, expected {a.expect_windows}: the clip list is not the ladder's")
    st = acc.finalize()
    os.makedirs(a.out, exist_ok=True)
    V, mu, ridge = solve_slow(st["C"], st["S"], a.k)
    if a.norm == "c":                      # OPTION, not the card's normalisation: symmetric (Loewdin) whitening of the 256-d subspace so that cov(u) = I exactly on the fit frames; V^T (C + ridge I) V = I no longer holds
        w_, Q_ = np.linalg.eigh(cov_u(V, st["C"]))
        V = V @ (Q_ * (1.0 / np.sqrt(w_))[None, :]) @ Q_.T
    P, lam = solve_pca(st["C"], a.k)
    Vs, mu_sh, _ = solve_slow(st["C"], st["S_xclip"], a.k)                # the planted shuffled-time basis (control 3's must-fail arm)
    B = st["C"] + ridge * np.eye(DIM_NM)
    meta = {"card_sha256": "7c2775e39f9b1271342a0c588780d3da8b4d63dfb56d06de9fef4c50bc8594ca", "lag_steps": a.lag, "lag_s": a.lag * DT, "ridge": ridge, "ridge_rel": RIDGE_REL, "k": a.k, "hmax": a.hmax, "exclude": a.exclude,
            "n_frames": st["n"], "n_pairs": st["n_pairs"], "norm": a.norm, "n_windows_ok": nwin, "n_clips": len(clips), "smoke": bool(a.half or a.limit or a.clips_file)}
    shas = {}
    shas["slow"] = save_basis(os.path.join(a.out, "basis_slow.npz"), "slow", st["mean"], V, mu, meta)
    shas["pca"] = save_basis(os.path.join(a.out, "basis_pca.npz"), "pca", st["mean"], P, lam, meta)
    tmp = os.path.join(a.out, "fit_stats.npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, n=st["n"], n_pairs=st["n_pairs"], mean=st["mean"], C=st["C"], S=st["S"], S_xclip=st["S_xclip"], V_xclip=Vs, mu_xclip=mu_sh)
    os.replace(tmp, os.path.join(a.out, "fit_stats.npz"))
    ac_slow, ac_pca = autocorr_of(V, st["C"], st["S"]), autocorr_of(P, st["C"], st["S"])
    cu_s, cu_p = cov_u(V, st["C"]), cov_u(P, st["C"])
    I = np.eye(a.k)
    spectra = {"slow_mu": mu.tolist(), "pca_variance": lam.tolist(), "slow_autocorr_measured": ac_slow.tolist(), "pca_autocorr_measured": ac_pca.tolist(), "trace_C": float(np.trace(st["C"])), "ridge": ridge,
               "slow_mu_top": float(mu[0]), "slow_mu_256": float(mu[-1]), "pca_var_top": float(lam[0]), "pca_var_256": float(lam[-1]), "pca_var_256_over_ridge": float(lam[-1] / ridge)}
    json.dump(spectra, open(os.path.join(a.out, "SPECTRA.json"), "w"), indent=1)
    rec = {"meta": meta, "basis_sha256": shas, "clips_sha256": hashlib.sha256("\n".join(clips).encode("utf-8")).hexdigest(),
           "vbv_max_dev_slow": float(np.abs(V.T @ B @ V - I).max()), "cov_u_max_dev_slow": float(np.abs(cu_s - I).max()), "cov_u_max_dev_pca": float(np.abs(cu_p - I).max()),
           "cov_u_diag_min_slow": float(np.diag(cu_s).min()), "slow_vs_pca": slow_vs_pca(st, V, mu, P), "xclip_shuffled_vs_pca": slow_vs_pca(st, Vs, mu_sh, P), "fit_stats_sha256": sha_file(os.path.join(a.out, "fit_stats.npz")),
           "wall_s": time.time() - t0}
    json.dump(rec, open(os.path.join(a.out, "FIT_RECORD.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "meta"}), flush=True)
    print("ZZFIT-OK", flush=True)


# ---------------------------------------------------------------------------------------------------- cache
def cmd_cache(a):
    src = Source(a.root, a.split, a.tracks, a.spool)
    B = load_basis(a.basis)
    kind = B["kind"]
    clips = train_clip_list(src, "", a.clips_file, a.limit)                  # EVERY clip of the split (zcache.py: nothing re-selected); the trainer reads the clips of its own list
    dst = os.path.join(a.out, kind, a.split)
    os.makedirs(dst, exist_ok=True)
    import shutil
    bname = "basis.npz"
    shutil.copyfile(a.basis, os.path.join(dst, bname))
    s1, s2, n, t0, files, Ts = np.zeros(K_OUT), np.zeros((K_OUT, K_OUT)), 0, time.time(), {}, []
    for i, c in enumerate(clips):
        nm, ok = src.inputs(c)
        u = apply_basis(nm, B["mean"], B["proj"])
        assert np.isfinite(u).all(), c
        f = os.path.join(dst, f"{c}.npz")
        tmp = f + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, zbasis=u, ok=ok, t_s=(np.arange(len(u)) * DT).astype(np.float32), basis_sha=np.array(B["sha"]))
        os.replace(tmp, f)
        m = fit_mask(ok, a.hmax)
        if m.any():
            x = u[m].astype(np.float64)
            n += len(x)
            s1 += x.sum(0)
            s2 += x.T @ x
        Ts.append(len(u))
        if (i + 1) % 100 == 0 or i == len(clips) - 1:
            print(f"{time.strftime('%H:%M:%S')} cache {kind} {a.split} {i + 1}/{len(clips)} {(time.time() - t0) / (i + 1):.2f} s/clip", flush=True)
    mean_u = s1 / max(n, 1)
    cov = s2 / max(n, 1) - np.outer(mean_u, mean_u)
    for c in clips:
        files[c] = sha_file(os.path.join(dst, f"{c}.npz"))
    man = {"format": FORMAT, "kind": kind, "dim": K_OUT, "dtype": "float32", "key": "zbasis", "grid": "pool4x10 token mean of the parameter-free LayerNorm (nm, 1024-d) -> (nm - mean) @ proj", "split": a.split,
           "basis_sha256": B["sha"], "basis_file": bname, "basis_file_sha256": sha_file(os.path.join(dst, bname)), "n_clips": len(clips), "T_min": int(min(Ts)), "T_max": int(max(Ts)), "frames": int(sum(Ts)),
           "fit_frames_in_cache": int(n), "cov_u_max_dev_from_I_on_fit_frames": float(np.abs(cov - np.eye(K_OUT)).max()), "cov_u_diag_min_on_fit_frames": float(np.diag(cov).min()),
           "dt_s": DT, "source_cache": a.root, "code_commit": a.commit, "slow_basis_py_sha256": sha_file(os.path.abspath(__file__)), "files_sha256": files, "smoke": bool(a.smoke)}
    json.dump(man, open(os.path.join(dst, "BASIS_MANIFEST.json"), "w"), indent=1)
    print(f"ZZCACHE-OK {kind} {a.split} clips={len(clips)} basis={B['sha'][:12]}", flush=True)


# ---------------------------------------------------------------------------------------------------- check3
def cmd_check3(a):
    D, WW = _data()
    src = Source(a.root, a.split, a.tracks)
    sel = json.load(open(a.clips_file, encoding="utf-8"))
    clips = sel[a.split] if isinstance(sel, dict) else sel
    bases = {k: load_basis(os.path.join(a.basis_dir, f"basis_{k}.npz")) for k in KINDS}
    fs = np.load(os.path.join(a.basis_dir, "fit_stats.npz"))
    st = {k: fs[k] for k in ("C", "S", "S_xclip")}
    res = {"tol_identity": 1e-4, "tol_cov": 1e-3, "split": a.split, "clips": list(clips), "identity": {}, "SMOKE": bool(a.smoke)}
    ident_ok, wrong_fail = True, True
    for k in KINDS:
        cd = os.path.join(a.cache_dir, k, a.split)
        md = 0.0
        md_wrong = np.inf
        other = bases["pca" if k == "slow" else "slow"]
        for c in clips:
            with np.load(os.path.join(cd, f"{c}.npz")) as z:
                u = np.asarray(z["zbasis"], np.float32)
                assert str(z["basis_sha"]) == bases[k]["sha"], (k, c)
            live = WW.norm_mean_tokens(src.store.get(c)["feats"]).numpy()            # the LIVE path: ClipStore.get -> norm_mean_tokens (not the cache builder's get_kin / mmap path)
            ref = apply_basis(live, bases[k]["mean"], bases[k]["proj"])
            md = max(md, float(np.abs(u - ref).max()))
            md_wrong = min(md_wrong, float(np.abs(u - apply_basis(live, other["mean"], other["proj"])).max()))
        res["identity"][k] = {"max_abs_diff": md, "pass": md <= res["tol_identity"], "wrong_kind_basis_max_abs_diff": md_wrong, "wrong_kind_fails_identity": md_wrong > res["tol_identity"]}
        ident_ok &= md <= res["tol_identity"]
        wrong_fail &= md_wrong > res["tol_identity"]
    I = np.eye(K_OUT)
    cs, cp = cov_u(bases["slow"]["proj"].astype(np.float64), st["C"]), cov_u(bases["pca"]["proj"].astype(np.float64), st["C"])
    res["cov_u_fit_frames"] = {"pca_max_dev": float(np.abs(cp - I).max()), "pca_pass": bool(np.abs(cp - I).max() <= res["tol_cov"]), "slow_max_dev": float(np.abs(cs - I).max()), "slow_pass": bool(np.abs(cs - I).max() <= res["tol_cov"]),
                               "slow_diag_min": float(np.diag(cs).min()), "slow_diag_max": float(np.diag(cs).max()),
                               "note": "SLOW is normalised V^T (C + ridge I) V = I as the card states, so cov(u) = I - ridge V^T V; the deviation is ridge / (variance + ridge) per direction"}
    V, mu, P = bases["slow"]["proj"].astype(np.float64), bases["slow"]["spectrum"], bases["pca"]["proj"].astype(np.float64)
    res["slow_vs_pca"] = slow_vs_pca(st, V, mu, P)
    Vs, mus = fs["V_xclip"], fs["mu_xclip"]
    res["shuffled_time_basis"] = {**slow_vs_pca(st, Vs, mus, P), "must_fail": True}
    res["pass"] = bool(ident_ok and wrong_fail and res["cov_u_fit_frames"]["pca_pass"] and res["slow_vs_pca"]["pass"] and not res["shuffled_time_basis"]["pass"])
    res["pass_cov_slow_card_tolerance"] = res["cov_u_fit_frames"]["slow_pass"]
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("clips",)}), flush=True)
    print("ZZCHECK3-OK" if res["pass"] else "ZZCHECK3-FAIL", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--root", required=True, help="cache dir holding <split>/manifest.jsonl and <split>/clips/")
    f.add_argument("--split", default="train")
    f.add_argument("--tracks", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--exclude", default="", help="lo:hi of the sha12-sorted clip list, as the trainer's --exclude (box 2: 540:600)")
    f.add_argument("--hmax", type=int, default=HMAX)
    f.add_argument("--lag", type=int, default=LAG)
    f.add_argument("--k", type=int, default=K_OUT)
    f.add_argument("--norm", choices=("b", "c"), default="b", help="b (default, the card) = V^T (C + ridge I) V = I; c = OPTION: symmetric whitening in the 256-d subspace, cov(u) = I exactly (see IMPL_CHOICES_SLOWWM)")
    f.add_argument("--expect-windows", type=int, default=None, help="box 2: 40358 (refused otherwise)")
    f.add_argument("--clips-file", default=None)
    f.add_argument("--limit", type=int, default=None)
    f.add_argument("--half", action="store_true", help="SMOKE: every second clip")
    f.add_argument("--spool", default=None, help="a directory that keeps nm per clip (float32), so cache does not re-read the features")
    c = sub.add_parser("cache")
    c.add_argument("--root", required=True)
    c.add_argument("--split", required=True, choices=("train", "eval"))
    c.add_argument("--tracks", required=True)
    c.add_argument("--basis", required=True, help="basis_slow.npz or basis_pca.npz of a fit")
    c.add_argument("--out", required=True)
    c.add_argument("--hmax", type=int, default=HMAX)
    c.add_argument("--clips-file", default=None)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--spool", default=None)
    c.add_argument("--commit", default="")
    c.add_argument("--smoke", action="store_true")
    k = sub.add_parser("check3")
    k.add_argument("--root", required=True)
    k.add_argument("--split", default="eval")
    k.add_argument("--tracks", required=True)
    k.add_argument("--basis-dir", required=True)
    k.add_argument("--cache-dir", required=True)
    k.add_argument("--clips-file", required=True)
    k.add_argument("--out-json", required=True)
    k.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "cache": cmd_cache, "check3": cmd_check3}[a.cmd](a)


if __name__ == "__main__":
    main()
