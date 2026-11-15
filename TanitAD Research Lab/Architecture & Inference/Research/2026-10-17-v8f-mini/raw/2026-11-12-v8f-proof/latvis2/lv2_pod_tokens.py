"""lv2_pod_tokens.py: the POD-SIDE (or any CPU) token-level read of the tool `latvis2` (card PREREG_V8F_PROOF_ADD38, sections 4 and 5). Read-only for the model; CPU.

What it does, per window (the 1,948 windows of LATENTS_CTX_<ARM>_R1.npz; frame f of a window = state-file frame / cache step g0 - 4 + f; frame 4 = t, frame 14 = t + 2.0 s):
  * rolls the stack out from the state of frame 4 (HJEPAStack.rollout, level 1, 10 steps, logged actions, the predictor's own context): the IMAGINED state at 2.0 s;
  * per fold of episodes (the pack's `fold`): fits on the frames of the four other folds (a) a per-cell PCA of the pooled tokens of the eval token cache (40 cells x F numbers -> 64 components)
    and (b) a ridge map from the level-1 state (640 = 40 cells x 16 numbers) to the 40 x 64 targets of the SAME frame; applies it out of fold to (i) the imagined state, (ii) the present state
    (frame 4), (iii) the true future state (frame 14) and scores each against the true tokens of frame 14 (squared error per cell, summed over the 64 components);
  * stores per window and cell: the three errors, the true token change between t and t + 2.0 s, and the squared distance of the present and of the imagined state to the true future state
    (cell-major cells of 16 numbers, level L3); the skills of the map (k2), the same with permuted states (k1); the k5 difference to a banked rollout; the k6 ids;
  * for the panel windows: the 3-component colours of the true grids (t, t + 2.0 s) and of the three decoded grids (level L2).
Tier T0 (a world-model diagnostic on logged frames), never driving performance. Writes --out (npz) and --out.json (checkpoint sha256, argv, counts, file:line facts).
Memory: tokens of the used frames are copied once to a float16 memmap in --workdir (about 0.8 GB for 9,460 frames x 40 x 1024); the per-cell covariances are float64 (40 x F x F = 0.34 GB at F 1024).
imag_dump.py (of the tool `imaghead`) must be importable: it is looked up next to this file, then in ../imaghead, then in --imag-dir."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "imaghead")):
    if os.path.exists(os.path.join(_p, "imag_dump.py")) and _p not in sys.path:
        sys.path.append(_p)

S0, FRAMES, K = 4, 16, 10
NCELL, CELLW = 40, 16
GRID_ROWS, GRID_COLS = 4, 10
FACTS = {   # file:line facts of the patched tree (re-asserted against the source by facts_check)
    "grid_state_cell_major": ("stack/tanitad/models/v8f_mini.py", 880, "return p.reshape(*p.shape[:2], p.shape[2] * p.shape[3])"),
    "cell_dim": ("stack/tanitad/models/v8f_mini.py", 231, "wm_cell_dim: int = 16"),
    "token_order_row_major": ("stack/tanitad/data/v8f_stopaux.py", 10, "row-major token = r x 10 + c"),
    "rollout": ("stack/tanitad/models/hjepa.py", 919, "def rollout(self, level"),
    "pool4x10_cache": ("stack/tanitad/data/v8f_mini_data.py", 4, "pool4x10"),
}
TOL_K5 = 1e-4


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def facts_check(tree):
    out = {}
    for k, (rel, ln, txt) in FACTS.items():
        with open(os.path.join(tree, rel), encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        out[k] = {"file": rel, "line": ln, "text": lines[ln - 1].strip()[:140], "ok": bool(txt in lines[ln - 1])}
    return out


# ------------------------------------------------------------------------------------------------ windows, frames
def unique_frames(eid, g0):
    """-> keys (list of (eid, step)), fidx [n, 16] row of frame f of every window in the unique-frame table (first occurrence order)."""
    key, keys = {}, []
    fidx = np.zeros((len(eid), FRAMES), np.int64)
    for j, (e, g) in enumerate(zip(eid, g0)):
        for f in range(FRAMES):
            k = (str(e), int(g) - S0 + f)
            if k not in key:
                key[k] = len(keys)
                keys.append(k)
            fidx[j, f] = key[k]
    return keys, fidx


def frame_states(Zw, fidx, n_frames):
    """Zw [n, 16, D] -> Xf [n_frames, D] (state of every unique frame) and the largest difference between copies of the same frame in different windows."""
    Xf = np.zeros((n_frames, Zw.shape[2]), np.float32)
    seen = np.zeros(n_frames, bool)
    worst = 0.0
    for j in range(fidx.shape[0]):
        for f in range(FRAMES):
            r = fidx[j, f]
            if seen[r]:
                worst = max(worst, float(np.abs(Xf[r] - Zw[j, f]).max()))
            else:
                Xf[r] = Zw[j, f]
                seen[r] = True
    return Xf, worst


def read_tokens_to_memmap(keys, cache, split, path, feat_dim=None):
    """Copy the pooled tokens of every unique frame (clip, step) from the eval cache (<cache>/<split>/clips/<clip>.pt, key pool4x10 [T, 40, F] fp8) to a float16 memmap [N, 40, F]."""
    import torch
    by_clip = {}
    for r, (e, s) in enumerate(keys):
        by_clip.setdefault(e, []).append((r, s))
    mm = None
    for e, rows in by_clip.items():
        d = torch.load(os.path.join(cache, split, "clips", e + ".pt"), map_location="cpu")
        x = d["pool4x10"]
        if x.shape[1] != NCELL:
            raise SystemExit(f"REFUSED: {e} pool4x10 has {x.shape[1]} cells, expected {NCELL}")
        if mm is None:
            F = int(x.shape[2])
            mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(keys), NCELL, F))
        steps = [s for _, s in rows]
        if min(steps) < 0 or max(steps) >= x.shape[0]:
            raise SystemExit(f"REFUSED: clip {e} has {x.shape[0]} frames, windows need steps {min(steps)}..{max(steps)}")
        # cast the whole clip before indexing: fp8 tensors cannot be indexed on the CPU in every torch build (pod: "index_cpu" not implemented for Float8_e4m3fn)
        mm[[r for r, _ in rows]] = x.float()[steps].numpy().astype(np.float16)
        del d, x
    mm.flush()
    return mm


# ------------------------------------------------------------------------------------------------ the token map
def pca_fit(mm, rows, nc, chunk=256):
    """Per-cell PCA of the tokens of `rows` of the memmap. -> mu [40, F], P [40, F, nc_eff] (columns = leading eigenvectors, sign fixed), evr [40] (explained variance ratio of the nc), colour basis (mbar [F], U [F, 3]) of the pooled cells."""
    Nc, F = mm.shape[1], mm.shape[2]
    S1 = np.zeros((Nc, F))
    S2 = np.zeros((Nc, F, F))
    rows = np.sort(np.asarray(rows))
    for a in range(0, len(rows), chunk):
        T = np.asarray(mm[rows[a:a + chunk]], np.float64)
        S1 += T.sum(0)
        for c in range(Nc):
            S2[c] += T[:, c].T @ T[:, c]
    n = len(rows)
    mu = S1 / n
    ncomp = min(nc, F)
    P = np.zeros((Nc, F, ncomp))
    evr = np.zeros(Nc)
    Cg = np.zeros((F, F))
    mbar = mu.mean(0)
    for c in range(Nc):
        C = S2[c] / n - np.outer(mu[c], mu[c])
        Cg += C + np.outer(mu[c] - mbar, mu[c] - mbar)
        w, V = np.linalg.eigh(C)
        o = np.argsort(w)[::-1][:ncomp]
        Vc = V[:, o]
        sgn = np.sign(Vc[np.argmax(np.abs(Vc), 0), np.arange(ncomp)])
        P[c] = Vc * np.where(sgn == 0, 1.0, sgn)
        evr[c] = float(w[o].sum() / max(w.sum(), 1e-30))
    wg, Vg = np.linalg.eigh(Cg / Nc)
    U = Vg[:, np.argsort(wg)[::-1][:3]]
    sgn = np.sign(U[np.argmax(np.abs(U), 0), np.arange(U.shape[1])])
    return mu, P, evr, (mbar, U * np.where(sgn == 0, 1.0, sgn))


def project(mm, mu, P, chunk=256):
    """All frames of the memmap -> Y [N, 40 x nc] float32 (PCA coordinates, cell-major)."""
    N, Nc = mm.shape[0], mm.shape[1]
    Y = np.zeros((N, Nc * P.shape[2]), np.float32)
    for a in range(0, N, chunk):
        T = np.asarray(mm[a:a + chunk], np.float64) - mu[None]
        Y[a:a + chunk] = np.einsum("mcf,cfk->mck", T, P).reshape(T.shape[0], -1)
    return Y


def ridge_fit(X, Y, lam_rel):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    xm, ym = X.mean(0), Y.mean(0)
    Xc = X - xm
    G = Xc.T @ Xc
    lam = lam_rel * float(np.trace(G)) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ (Y - ym))
    return {"W": W, "xm": xm, "ym": ym, "lam": lam}


def ridge_apply(m, X):
    return (np.asarray(X, np.float64) - m["xm"]) @ m["W"] + m["ym"]


def skill(Yhat, Y, ybar):
    """1 - SSE / SST over frames and all targets, SST against the mean of the TRAINING targets (a held-out skill that the training mean already gets is 0)."""
    sst = float(((Y - ybar) ** 2).sum())
    return float(1.0 - ((Yhat - Y) ** 2).sum() / max(sst, 1e-30))


def cell_sq(a, b):
    """a, b [n, 40 x w] cell-major -> [n, 40] squared distance per cell."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return (d ** 2).reshape(d.shape[0], NCELL, -1).sum(2)


def colours(tok, mbar, U):
    """tok [..., 40, F] -> [..., 40, 3] projection on the colour basis."""
    return (np.asarray(tok, np.float64) - mbar) @ U


def decode_tokens(Yhat, mu, P):
    """Yhat [n, 40 x nc] PCA coordinates -> [n, 40, F] tokens (mean + components)."""
    n = Yhat.shape[0]
    return mu[None] + np.einsum("cfk,nck->ncf", P, np.asarray(Yhat, np.float64).reshape(n, NCELL, -1))


def run_fold(k, mm, Xf, frame_fold, fidx, win_fold, Zim, nc, lam_rel, panel_pos, chunk=256, mutate=None, perm_seed=0, keep_Y=None):
    """One held-out fold k. -> (per-window dict for the windows of the fold, fold record, panel dict)."""
    tr = np.flatnonzero(frame_fold != k)
    ho = np.flatnonzero(frame_fold == k)
    wk = np.flatnonzero(win_fold == k)
    mu, P, evr, (mbar, U) = pca_fit(mm, tr, nc, chunk)
    Y = project(mm, mu, P, chunk)
    ybar = Y[tr].mean(0, dtype=np.float64)
    if keep_Y is not None:
        keep_Y["Y"] = Y
    mp = ridge_fit(Xf[tr], Y[tr], lam_rel)
    f4, f14 = fidx[wk, S0], fidx[wk, 14]
    tr4 = np.unique(fidx[np.flatnonzero(win_fold != k), S0])
    ho_rows = tr if mutate == "skill_in_sample" else ho                       # mutation: the held-out skill computed on training frames
    rec = {"fold": int(k), "n_train_frames": int(len(tr)), "n_heldout_frames": int(len(ho)), "n_heldout_windows": int(len(wk)), "ridge_lambda": mp["lam"], "ncomp": int(P.shape[2]),
           "evr_mean": float(evr.mean()),
           "skill_train": skill(ridge_apply(mp, Xf[tr]), Y[tr], ybar),
           "skill_heldout": skill(ridge_apply(mp, Xf[ho_rows]), Y[ho_rows], ybar),
           "skill_train_present": skill(ridge_apply(mp, Xf[tr4]), Y[tr4], ybar),
           "skill_heldout_present": skill(ridge_apply(mp, Xf[np.unique(f4)]), Y[np.unique(f4)], ybar)}
    # k1: the states permuted across frames (one fixed permutation of all frames; fit on the permuted training pairs, applied to the permuted held-out states)
    perm = np.random.default_rng(perm_seed + int(k)).permutation(Xf.shape[0])
    if mutate == "identity_perm":
        perm = np.arange(Xf.shape[0])
    Xp = Xf[perm]
    mq = ridge_fit(Xp[tr], Y[tr], lam_rel)
    rec["skill_heldout_permuted"] = skill(ridge_apply(mq, Xp[ho]), Y[ho], ybar)
    rec["skill_train_permuted"] = skill(ridge_apply(mq, Xp[tr]), Y[tr], ybar)
    # the three reads of the windows of the fold
    Zp, Zt = Xf[f4], Xf[f14]
    Zi = Zim[wk]
    if mutate == "imagined_is_present":
        Zi = Zp
    Yt = Y[f14]
    Ypres, Yimag, Ytru = ridge_apply(mp, Zp), ridge_apply(mp, Zi), ridge_apply(mp, Zt)
    out = {"win": wk,
           "err_imag": cell_sq(Yimag, Yt).astype(np.float32), "err_pres": cell_sq(Ypres, Yt).astype(np.float32), "err_true": cell_sq(Ytru, Yt).astype(np.float32),
           "tok_change": cell_sq(Yt, Y[f4]).astype(np.float32),
           "dist_pres": cell_sq(Zp, Zt).astype(np.float32), "dist_imag": cell_sq(Zi, Zt).astype(np.float32)}
    pan = {}
    sel = [(i, int(w)) for i, w in enumerate(wk) if int(w) in panel_pos]
    if sel:
        ii = np.array([i for i, _ in sel])
        t4 = np.asarray(mm[f4[ii]], np.float64)
        t14 = np.asarray(mm[f14[ii]], np.float64)
        pan = {"win": np.array([w for _, w in sel]),
               "col_true": np.stack([colours(t4, mbar, U), colours(t14, mbar, U)], 1).astype(np.float32),
               "col_dec": np.stack([colours(decode_tokens(Yh[ii], mu, P), mbar, U) for Yh in (Yimag, Ypres, Ytru)], 1).astype(np.float32)}
    return out, rec, pan


FC_INPUTS = ("P", "PA", "I", "PAI", "T")
FC_LAMS = (0.01, 0.1, 1.0, 10.0, 100.0)


def forecast_block(k, Xf, fidx, win_fold, Zim, A, Y, mutate=None, perm_seed=0, lams=FC_LAMS):
    """Note PREREG_V8F_PROOF_ADD38_N3. For the held-out fold k: ridge forecasts of the true token coordinates of frame 14, fitted on the WINDOWS of the other folds,
    from five inputs (P present state, PA + the 10 logged actions, I imagined state, PAI, T true future state). The penalty of each input is chosen by a
    cross-validation over the training folds. -> ({fc_<input>: [n_k, 40] squared error per window and cell}, record).  mutate: 'imagined_is_present'."""
    win_fold = np.asarray(win_fold)
    wtr, wk = np.flatnonzero(win_fold != k), np.flatnonzero(win_fold == k)

    def feats(w):
        P_ = np.asarray(Xf[fidx[w, S0]], np.float64)
        T_ = np.asarray(Xf[fidx[w, 14]], np.float64)
        I_ = P_ if mutate == "imagined_is_present" else np.asarray(Zim[w], np.float64)
        Aa = np.asarray(A[w], np.float64).reshape(len(w), -1)
        return {"P": P_, "PA": np.concatenate([P_, Aa], 1), "I": I_, "PAI": np.concatenate([P_, Aa, I_], 1), "T": T_}

    Ftr, Fho = feats(wtr), feats(wk)
    Ytr, Yho = np.asarray(Y[fidx[wtr, 14]], np.float64), np.asarray(Y[fidx[wk, 14]], np.float64)
    ybar = Ytr.mean(0)
    inner = win_fold[wtr]
    out, rec = {}, {"fold": int(k), "n_train_windows": int(len(wtr)), "n_heldout_windows": int(len(wk)), "lams": list(lams)}
    for name in FC_INPUTS:
        sse = []
        for lam in lams:
            tot = 0.0
            for j in np.unique(inner):
                a, b = inner != j, inner == j
                m = ridge_fit(Ftr[name][a], Ytr[a], lam)
                tot += float(((ridge_apply(m, Ftr[name][b]) - Ytr[b]) ** 2).sum())
            sse.append(tot)
        lam = float(lams[int(np.argmin(sse))])
        m = ridge_fit(Ftr[name], Ytr, lam)
        Yh = ridge_apply(m, Fho[name])
        out["fc_" + name] = cell_sq(Yh, Yho).astype(np.float32)
        rec["lam_" + name] = lam
        rec["skill_heldout_" + name] = skill(Yh, Yho, ybar)
        rec["skill_train_" + name] = skill(ridge_apply(m, Ftr[name]), Ytr, ybar)
    rng = np.random.default_rng(perm_seed + 1000 + int(k))
    mq = ridge_fit(Ftr["I"][rng.permutation(len(wtr))], Ytr, rec["lam_I"])
    rec["skill_heldout_permuted_I"] = skill(ridge_apply(mq, Fho["I"][rng.permutation(len(wk))]), Yho, ybar)
    out["win"] = wk
    return out, rec


# ------------------------------------------------------------------------------------------------ the whole read
def compute(Zw, A, eid, g0, win_fold, mm, keys, fidx, imagine, nc=64, lam_rel=1e-2, panel_windows=(), chunk=256, mutate=None, log=lambda *a: None, forecast=False, fc_out=None):
    """Zw [n,16,640], A [n,10,2] (unused here, the rollout is done by `imagine`), tokens in the memmap `mm`. `imagine(z0 [n,640]) -> [n,640]` = the imagined state at 2.0 s.
    -> result arrays and fold records."""
    n = len(eid)
    Xf, copy_diff = frame_states(Zw, fidx, len(keys))
    ep_of_frame = np.array([k[0] for k in keys])
    fold_of_ep = {}
    for e, f in zip(eid, win_fold):
        if fold_of_ep.setdefault(str(e), int(f)) != int(f):
            raise SystemExit(f"REFUSED: episode {e} lies in two folds")
    frame_fold = np.array([fold_of_ep[e] for e in ep_of_frame])
    Zim = imagine(Zw[:, S0])
    panel_pos = set(int(w) for w in panel_windows)
    res = {k: np.full((n, NCELL), np.nan, np.float32) for k in ("err_imag", "err_pres", "err_true", "tok_change", "dist_pres", "dist_imag")}
    recs, pans = [], []
    for k in sorted(set(fold_of_ep.values())):
        t0 = time.time()
        hold = {} if forecast else None
        out, rec, pan = run_fold(k, mm, Xf, frame_fold, fidx, np.asarray(win_fold), Zim, nc, lam_rel, panel_pos, chunk, mutate, keep_Y=hold)
        for key in res:
            res[key][out["win"]] = out[key]
        recs.append(rec)
        if forecast:
            fo, frec = forecast_block(k, Xf, fidx, win_fold, Zim, A, hold["Y"], mutate=mutate)
            for nm in FC_INPUTS:
                fc_out.setdefault("fc_" + nm, np.full((n, NCELL), np.nan, np.float32))[fo["win"]] = fo["fc_" + nm]
            fc_out.setdefault("recs", []).append(frec)
            log(f"fold {k}: forecast held-out skill " + " ".join(f"{nm} {frec['skill_heldout_' + nm]:.4f} (lam {frec['lam_' + nm]:g})" for nm in FC_INPUTS) + f" permuted I {frec['skill_heldout_permuted_I']:.4f}")
            del hold
        if pan:
            pans.append(pan)
        log(f"fold {k}: skill train {rec['skill_train']:.4f} held-out {rec['skill_heldout']:.4f} permuted {rec['skill_heldout_permuted']:.4f} ({time.time() - t0:.0f} s)")
    if pans:
        order = np.argsort(np.concatenate([p["win"] for p in pans]))
        pw = np.concatenate([p["win"] for p in pans])[order]
        col_true = np.concatenate([p["col_true"] for p in pans])[order]
        col_dec = np.concatenate([p["col_dec"] for p in pans])[order]
    else:
        pw, col_true, col_dec = np.zeros(0, np.int64), np.zeros((0, 2, NCELL, 3), np.float32), np.zeros((0, 3, NCELL, 3), np.float32)
    return res, recs, {"panel_win": pw, "col_true": col_true, "col_dec": col_dec}, Zim, copy_diff


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--pack", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--states")
    g.add_argument("--window-states")
    ap.add_argument("--cache", required=True, help="eval token cache root: <cache>/<split>/clips/<clip>.pt with pool4x10 fp8 [T, 40, F]")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--banked-roll", default=None, help="k5: LATENTS_CTX_<ARM>_R3g.npz of the same checkpoint (the context-3 rollout; R1 is context 1)")
    ap.add_argument("--panel-windows", default=None, help="json list of window indices (rows of --windows) for which colours are stored")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None, help="token memmap lives here (default: next to --out)")
    ap.add_argument("--ncomp", type=int, default=64)
    ap.add_argument("--lam-rel", type=float, default=1e-2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=0, help="tests: the first N windows")
    ap.add_argument("--imag-dir", default=None)
    ap.add_argument("--forecast", action="store_true", help="note ADD38_N3: also the forecasts of the future tokens from five inputs at equal calibration")
    a = ap.parse_args(argv)
    if a.imag_dir:
        sys.path.insert(0, a.imag_dir)
    import imag_dump as ID
    t0 = time.time()
    torch, M, tree = ID.import_stack(a.tree)
    torch.set_num_threads(a.threads)
    cfg, model, st, ck = ID.load_model(torch, M, a.ckpt, require_head=False)
    W = np.load(a.windows, allow_pickle=True)
    P = np.load(a.pack, allow_pickle=True)
    eid, g0 = W["eid"].astype(str), W["g0"].astype(np.int64)
    if not (np.array_equal(eid, P["eid"].astype(str)) and np.array_equal(g0, P["g0"].astype(np.int64))):
        raise SystemExit("REFUSED: the window list and the pack do not hold the same windows in the same order")
    fold = np.asarray(P["fold"], np.int64)
    A_all = torch.as_tensor(np.asarray(W["act_scaled"], np.float32))
    if a.states:
        Z_all = ID.assemble_states(torch, a.states, eid, g0)
        in_path, in_kind = a.states, "states"
    else:
        Z_all = torch.as_tensor(np.asarray(np.load(a.window_states, allow_pickle=True)["Z1"])).float()
        in_path, in_kind = a.window_states, "window-states"
    if a.max_windows:
        eid, g0, fold, A_all, Z_all = eid[:a.max_windows], g0[:a.max_windows], fold[:a.max_windows], A_all[:a.max_windows], Z_all[:a.max_windows]
    if Z_all.shape[:2] != (len(eid), FRAMES) or Z_all.shape[2] != NCELL * CELLW:
        raise SystemExit(f"REFUSED: states {tuple(Z_all.shape)} for {len(eid)} windows (need [n, 16, {NCELL * CELLW}], the grid state)")
    ctx = ID.predictor_context(st)

    def imagine(z0):
        out = []
        for i in range(0, z0.shape[0], a.chunk):
            R = ID.roll_states(torch, st, torch.as_tensor(z0[i:i + a.chunk]).float(), A_all[i:i + a.chunk], ctx)
            out.append(R[:, K - 1].numpy())
        return np.concatenate(out).astype(np.float32)

    keys, fidx = unique_frames(eid, g0)
    wd = a.workdir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "work_tokens")
    os.makedirs(wd, exist_ok=True)
    mmp = os.path.join(wd, "tokens_f16.npy")
    print(f"lv2_pod_tokens: {len(eid)} windows, {len(keys)} unique frames; reading tokens ...", flush=True)
    read_tokens_to_memmap(keys, a.cache, a.split, mmp)
    mm = np.load(mmp, mmap_mode="r")
    panel = json.load(open(a.panel_windows, encoding="utf-8")) if a.panel_windows else []
    if isinstance(panel, dict):
        panel = panel.get("window_indices", panel.get("windows", []))
    fc = {}
    res, recs, pan, Zim, copy_diff = compute(Z_all.numpy(), A_all.numpy(), eid, g0, fold, mm, keys, fidx, imagine, a.ncomp, a.lam_rel, panel, a.chunk, log=lambda *s: print(*s, flush=True),
                                             forecast=a.forecast, fc_out=fc)
    fc_recs = fc.pop("recs", [])
    fc_arr = dict(fc)
    for nm in (FC_INPUTS if fc_recs else ()):
        fc_arr["fc_skill_heldout_" + nm] = np.array([r["skill_heldout_" + nm] for r in fc_recs])
        fc_arr["fc_skill_train_" + nm] = np.array([r["skill_train_" + nm] for r in fc_recs])
        fc_arr["fc_lam_" + nm] = np.array([r["lam_" + nm] for r in fc_recs])
    if fc_recs:
        fc_arr["fc_skill_heldout_permuted_I"] = np.array([r["skill_heldout_permuted_I"] for r in fc_recs])
    k5 = None
    if a.banked_roll:
        b = np.load(a.banked_roll, allow_pickle=True)
        m = len(eid)
        if not (np.array_equal(b["eid"].astype(str)[:m], eid) and np.array_equal(b["g0"][:m].astype(np.int64), g0)):
            raise SystemExit("REFUSED: the banked rollout file holds other windows")
        k5 = {"file": a.banked_roll.replace(os.sep, "/"), "sha256": sha256_file(a.banked_roll), "n": int(m), "tol": TOL_K5,
              "max_abs_pred_2s": float(np.abs(Zim - np.asarray(b["pred_2s"][:m], np.float32)).max()),
              "max_abs_z0": float(np.abs(Z_all[:, S0].numpy() - np.asarray(b["z0"][:m], np.float32)).max()) if "z0" in b.files else None,
              "max_abs_true_2s": float(np.abs(Z_all[:, 14].numpy() - np.asarray(b["true_2s"][:m], np.float32)).max()) if "true_2s" in b.files else None}
    ck_sha = sha256_file(a.ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, eid=eid, g0=g0, fold=fold, row=np.arange(len(eid)), **res, **pan, **fc_arr,
                        skill_train=np.array([r["skill_train"] for r in recs]), skill_heldout=np.array([r["skill_heldout"] for r in recs]),
                        skill_heldout_permuted=np.array([r["skill_heldout_permuted"] for r in recs]), skill_train_present=np.array([r["skill_train_present"] for r in recs]),
                        skill_heldout_present=np.array([r["skill_heldout_present"] for r in recs]), fold_ids=np.array([r["fold"] for r in recs]),
                        k5_max_abs=np.float64(np.nan if k5 is None else k5["max_abs_pred_2s"]), ckpt_sha256=ck_sha)
    meta = {"ckpt": a.ckpt.replace(os.sep, "/"), "ckpt_sha256": ck_sha, "argv": list(argv if argv is not None else sys.argv[1:]), "tree": tree,
            "tanitad_file": __import__("tanitad").__file__.replace(os.sep, "/"), "n_windows": int(len(eid)), "n_unique_frames": int(len(keys)), "n_episodes": int(len(set(eid))),
            "token_shape": list(mm.shape), "ncomp": int(a.ncomp), "ridge_lambda_rel": a.lam_rel, "predictor_context": int(ctx), "rollout_steps": K, "folds": recs, "forecast_folds": fc_recs,
            "frame_copy_max_abs_diff": copy_diff, "k5": k5, "panel_windows": [int(x) for x in panel],
            "inputs": {"windows": a.windows.replace(os.sep, "/"), "windows_sha256": sha256_file(a.windows), "pack": a.pack.replace(os.sep, "/"), "pack_sha256": sha256_file(a.pack),
                       in_kind: in_path.replace(os.sep, "/"), in_kind + "_sha256": sha256_file(in_path), "cache": a.cache.replace(os.sep, "/")},
            "facts": facts_check(tree), "cell_layout": "state[c*16:(c+1)*16] = cell c; cell c = row c // 10, column c % 10 of the 4 x 10 grid; row 0 top (sky) .. row 3 bottom (hood)",
            "device": a.device, "seconds": round(time.time() - t0, 1)}
    with open(a.out + ".json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    print(f"lv2_pod_tokens: {len(eid)} windows -> {a.out}; k5 {None if k5 is None else k5['max_abs_pred_2s']}; {meta['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
