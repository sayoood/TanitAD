"""PRH_SEED probe (PREREG_PRH_SEED.md, sha256 411ddb72...; first wording a616e439...): CPU only, zero training.

Stages: (1) extract per-frame nm (1024-d token-mean of the parameter-free layer-normalised frozen features) and
zmean (256-d, the model's own adapter code) for TR0/TR1 (trained) and RI0/RI1 (random-init, torch seeds 100/101) on the
eval-141 clips; (2) CCA metric M, episode-cluster bootstrap, controls, outcome.
Choices the card leaves open are listed in CHOICES and written into the result file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

CHOICES = {
    "cca_ridge_M": "1e-6 x mean eigenvalue of each side (numerical only; float64). Sensitivity row with 1e-3 reported beside.",
    "ridge_beside": "1e-3 x mean eigenvalue for the whitening used by Procrustes, mutual 10-NN, per-direction profile.",
    "slow_ridge": "1e-3 x mean eigenvalue (card), applied to each side (x_t and x_t+5) of the temporal CCA; SLOW rep = (nm - mean) @ top-256 x-side canonical directions.",
    "slow_pairs": "pairs (t, t+5) in the same clip, both frames ok, fit-half clips only.",
    "corr_definition": "per-direction Pearson correlation on the score half (score-half means and variances), directions fixed from the fit half.",
    "ok": "ok[t] = isfinite(p10f[t]).all() | t<1 from ClipStore.get_kin + kin_features (as theory/ladder/zcache.py clip_inputs).",
    "separated": "95 % percentile CI of the paired difference (same resample counts) has lower bound > 0.",
    "bootstrap": "B=2000, rng seed 0, clips of the score half resampled with replacement, fit-half directions fixed; one set of counts shared by every M, so differences are paired.",
    "split": "eval clip ids sorted; even rank fit, odd rank score.",
    "random_init": "V8FMini built from the checkpoint cfg under torch.manual_seed(100/101), adapter and pos at init, no weights loaded.",
    "pca_slow_rep": "reps are the projections of nm onto the fit-half directions (centred by the fit-half mean).",
    "knn": "mutual 10-NN: cosine in each rep's fit-half-whitened centred space, score frames with t%5==0 and ok (1 per second), overlap/10 averaged; chance 10/N.",
    "profile": "for SLOW direction k: least-squares readout of SLOW_k from whitened A (fit half); apply it to A and to Procrustes-rotated B on the score half; Pearson correlation of the two readouts. Ordered by SLOW's fit-half canonical correlation.",
    "D-MIX": "random affine map of nm = nm @ G + b, G ~ N(0,1) [1024,256] seed 202, scaled so its total fit-half variance equals TR0's; D-MIX = 0.5*TR0 + 0.5*that. Pass if the paired CI of M(TR0,DMIX) - M(TR0,TR1) is separated from 0 (either sign).",
    "C-SHUF": "TR1 frames permuted (rng seed 5) across clips within the fit half and within the score half (pairing broken in both), CCA refit on the permuted fit half. Pass if M < M(RI0,RI1) (point) and mutual 10-NN <= 2 x chance.",
    "C-ROT": "random orthogonal Q (seed 7, QR of a Gaussian) applied to TR1.",
}
RIDGE_M = 1e-6
RIDGE_B = 1e-3
LAG = 5
K = 256
B_BOOT = 2000


def sha_file(p, blk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- linear algebra
def whitener(C, ridge):
    """Symmetric (C + ridge*mean_eig*I)^(-1/2), float64."""
    w, V = np.linalg.eigh(C)
    lam = ridge * float(w.mean())
    w = np.maximum(w, 0.0) + lam
    return (V / np.sqrt(w)) @ V.T


def fit_cca(A, B, ridge=RIDGE_M, k=None):
    """CCA fit on (A, B) [n, da], [n, db]: returns dict(ma, mb, Pa, Pb, s). Projections (A-ma)@Pa, (B-mb)@Pb have unit variance (up to ridge) and per-direction correlation s."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    ma, mb = A.mean(0), B.mean(0)
    Ac, Bc = A - ma, B - mb
    n = len(A)
    Wa = whitener(Ac.T @ Ac / n, ridge)
    Wb = whitener(Bc.T @ Bc / n, ridge)
    U, s, Vt = np.linalg.svd(Wa @ (Ac.T @ Bc / n) @ Wb)
    k = k or min(A.shape[1], B.shape[1])
    return {"ma": ma, "mb": mb, "Pa": Wa @ U[:, :k], "Pb": Wb @ Vt.T[:, :k], "s": s[:k]}


def pearson_cols(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def cca_M(A, B, fit_idx, score_idx, ridge=RIDGE_M):
    """Mean of the canonical correlations, directions fit on fit_idx rows, correlations on score_idx rows."""
    f = fit_cca(A[fit_idx], B[fit_idx], ridge)
    return float(pearson_cols((A[score_idx] - f["ma"]) @ f["Pa"], (B[score_idx] - f["mb"]) @ f["Pb"]).mean())


def clip_stats(a, b, cid, n_clips):
    """Per-clip sufficient statistics of the paired projections a, b [n, k] (cid [n] in 0..n_clips-1): [C,k] each plus counts [C]."""
    k = a.shape[1]
    st = {x: np.zeros((n_clips, k)) for x in ("Sa", "Sb", "Saa", "Sbb", "Sab")}
    for x, v in (("Sa", a), ("Sb", b), ("Saa", a * a), ("Sbb", b * b), ("Sab", a * b)):
        np.add.at(st[x], cid, v)
    st["n"] = np.bincount(cid, minlength=n_clips).astype(np.float64)
    return st


def corr_from_stats(st, counts):
    """counts [R, C] resample multiplicities -> per-direction correlation [R, k]."""
    n = counts @ st["n"]
    ma, mb = (counts @ st["Sa"]) / n[:, None], (counts @ st["Sb"]) / n[:, None]
    va = (counts @ st["Saa"]) / n[:, None] - ma ** 2
    vb = (counts @ st["Sbb"]) / n[:, None] - mb ** 2
    cv = (counts @ st["Sab"]) / n[:, None] - ma * mb
    return cv / np.sqrt(va * vb)


def pair_boot(A, B, fit_idx, score_idx, cid_score, n_clips, counts, ridge=RIDGE_M):
    f = fit_cca(A[fit_idx], B[fit_idx], ridge)
    a = (A[score_idx] - f["ma"]) @ f["Pa"]
    b = (B[score_idx] - f["mb"]) @ f["Pb"]
    st = clip_stats(a, b, cid_score, n_clips)
    boot = corr_from_stats(st, counts).mean(1)
    point = float(corr_from_stats(st, np.ones((1, n_clips)))[0].mean())
    return point, boot


def ci95(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def is_sep_pos(lo, hi):
    """Separated and positive: the 95 % interval lies above zero."""
    return lo > 0


def is_sep_any(lo, hi):
    return lo > 0 or hi < 0


def outcome(p1, p2_all):
    """p1: bool, p2_all: bool (all four differences separated and positive)."""
    if not p1:
        return "O3 NOT-PLATONIC"
    return "O1 PLATONIC-SLOW" if p2_all else "O2 PLATONIC-OTHER"


# --------------------------------------------------------------------------------------------- beside rows
def beside(A, B, fit_idx, score_idx, score_sub_idx, slow_S, ridge=RIDGE_B):
    """Procrustes R^2, mutual 10-NN and per-SLOW-direction profile for the pair (A, B). slow_S [n, 256] is the SLOW rep (columns ordered)."""
    fA = fit_cca  # noqa: F841  (kept for readers: whitening below is per side, not CCA)
    Af, Bf = A[fit_idx], B[fit_idx]
    ma, mb = Af.mean(0), Bf.mean(0)
    Wa = whitener(np.cov((Af - ma).T, bias=True), ridge)
    Wb = whitener(np.cov((Bf - mb).T, bias=True), ridge)
    Za_f, Zb_f = (Af - ma) @ Wa, (Bf - mb) @ Wb
    U, _, Vt = np.linalg.svd(Zb_f.T @ Za_f)
    R = U @ Vt                                                   # Zb R ~ Za
    Za_s, Zb_s = (A[score_idx] - ma) @ Wa, (B[score_idx] - mb) @ Wb
    r2 = float(1.0 - ((Za_s - Zb_s @ R) ** 2).sum() / (Za_s ** 2).sum())
    # mutual 10-NN on the 1-per-second subsample
    Xa, Xb = (A[score_sub_idx] - ma) @ Wa, (B[score_sub_idx] - mb) @ Wb
    N = len(Xa)

    def knn(X):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, 10, axis=1)[:, :10]
    na, nb = knn(Xa), knn(Xb)
    ov = float(np.mean([len(set(na[i]) & set(nb[i])) / 10.0 for i in range(N)]))
    # per-direction profile along SLOW's directions
    Wr, *_ = np.linalg.lstsq(Za_f, slow_S[fit_idx] - slow_S[fit_idx].mean(0), rcond=None)
    prof = pearson_cols(Za_s @ Wr, (Zb_s @ R) @ Wr)
    return {"procrustes_R2": r2, "mutual_10NN": ov, "chance_10NN": 10.0 / N, "N_knn": N, "profile_along_SLOW": [float(x) for x in prof]}


# --------------------------------------------------------------------------------------------- extraction
def extract(a):
    import torch
    from tanitad.data import v8f_mini_data as D
    from tanitad.data.v8f_wm_windows import norm_mean_tokens
    from tanitad.models import v8f_mini as M
    import tanitad
    assert os.path.normcase(os.path.abspath(tanitad.__file__)).startswith(os.path.normcase(os.path.abspath(a.tree))), tanitad.__file__
    torch.set_num_threads(2)
    os.makedirs(a.work, exist_ok=True)
    cache = os.path.join(a.work, "extract.npz")
    if os.path.exists(cache) and not a.refresh:
        return dict(np.load(cache, allow_pickle=False))

    prior = M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30))

    def load_trained(p):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = M.V8FMiniConfig(**ck["cfg"])
        m = M.V8FMini(cfg, prior)
        r = m.load_state_dict(ck["model"], strict=False)
        assert not r.missing_keys and set(r.unexpected_keys) <= {"hacc_w"}, (r.missing_keys, r.unexpected_keys)
        m.eval()
        return m, ck

    tr0, ck0 = load_trained(a.ckpt0)
    tr1, ck1 = load_trained(a.ckpt1)
    ri = []
    for seed in (100, 101):
        torch.manual_seed(seed)
        ri.append(M.V8FMini(M.V8FMiniConfig(**ck0["cfg"]), prior).eval())
    models = {"tr0": tr0, "tr1": tr1, "ri0": ri[0], "ri1": ri[1]}
    tracks = D.load_native_tracks(a.tracks)
    store = D.ClipStore(a.root, "eval", lru=1, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    clips = sorted(os.path.basename(p)[:-3] for p in __import__("glob").glob(f"{a.root}/eval/clips/*.pt"))
    out = {k: [] for k in ("nm", "tr0", "tr1", "ri0", "ri1", "ok", "t", "cid")}
    aff_err = 0.0
    t0 = time.time()
    for ci, clip in enumerate(clips):
        kin = store.get_kin(clip)
        T = len(kin["pose"])
        p10 = np.full((T, 8), np.nan, np.float32)
        p10[1:] = D._fk().kin_features(kin["kin3"][1:]).astype(np.float32)
        idx = np.arange(T)
        ok = np.isfinite(p10).all(1) | (idx < 1)
        path = f"{a.root}/eval/clips/{clip}.pt"
        try:
            o = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except (RuntimeError, TypeError):
            o = torch.load(path, map_location="cpu", weights_only=True)
        feats = o["pool4x10"].to(torch.float16).clone()
        del o
        assert feats.shape[0] == T
        with torch.no_grad():
            nm = torch.cat([norm_mean_tokens(feats[s:s + 16].float()) for s in range(0, T, 16)])
            for name, m in models.items():
                z = []
                for s in range(0, T, 16):
                    f = feats[s:s + 16].float()[None]
                    z.append((m.adapter(f) + m.pos[None, None]).mean(2)[0])        # V8FMini._encode(...).mean(2), the tensor WMAux.z1 projects
                z = torch.cat(z)
                out[name].append(z.numpy().astype(np.float32))
                if name == "tr0":
                    aff_err = max(aff_err, float((m.pooled_adapt_mean(nm) - z).abs().max()))
        out["nm"].append(nm.numpy().astype(np.float32))
        out["ok"].append(ok)
        out["t"].append(idx)
        out["cid"].append(np.full(T, ci))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["clips"] = np.array(clips)
    res["aff_err_tr0"] = np.array(aff_err)
    res["adapter_sha_note"] = np.array(f"{sha_file(a.ckpt0)[:16]},{sha_file(a.ckpt1)[:16]}")
    res["extract_s"] = np.array(time.time() - t0)
    np.savez(cache, **res)
    return res


# --------------------------------------------------------------------------------------------- the probe
def run_probe(X, a_cli, B=B_BOOT):
    clips = [str(c) for c in X["clips"]]
    C = len(clips)
    ok = X["ok"].astype(bool)
    cid, tt = X["cid"], X["t"]
    nm = X["nm"].astype(np.float64)
    reps = {k: X[k].astype(np.float64) for k in ("tr0", "tr1", "ri0", "ri1")}
    fit_clip = np.arange(C) % 2 == 0
    is_fit = fit_clip[cid]
    fit_idx = np.where(ok & is_fit)[0]
    score_idx = np.where(ok & ~is_fit)[0]
    score_clips = np.where(~fit_clip)[0]
    remap = -np.ones(C, int)
    remap[score_clips] = np.arange(len(score_clips))
    cid_score = remap[cid[score_idx]]
    nC = len(score_clips)
    rng = np.random.default_rng(0)
    counts = rng.multinomial(nC, np.ones(nC) / nC, size=B).astype(np.float64)
    sub_idx = score_idx[(tt[score_idx] % 5) == 0]

    # SLOW, PCA on the fit half
    n_all = len(nm)
    nxt = np.arange(n_all) + LAG
    valid = (nxt < n_all)
    valid[valid] &= (cid[nxt[valid]] == cid[valid.nonzero()[0]]) & ok[nxt[valid]]
    pair_idx = np.where(valid & ok & is_fit)[0]
    fs = fit_cca(nm[pair_idx], nm[pair_idx + LAG], RIDGE_B, k=K)
    SLOW = (nm - fs["ma"]) @ fs["Pa"]
    mpca = nm[fit_idx].mean(0)
    w, V = np.linalg.eigh(np.cov((nm[fit_idx] - mpca).T, bias=True))
    PCA = (nm - mpca) @ V[:, ::-1][:, :K]

    # D-MIX
    g = np.random.default_rng(202)
    G, b0 = g.standard_normal((nm.shape[1], K)), g.standard_normal(K)
    Rr = nm @ G + b0
    Rr *= np.sqrt(reps["tr0"][fit_idx].var(0).sum() / Rr[fit_idx].var(0).sum())
    DMIX = 0.5 * reps["tr0"] + 0.5 * Rr

    def pb(A, Bm, fi=fit_idx, si=score_idx, cs=cid_score, ridge=RIDGE_M):
        return pair_boot(A, Bm, fi, si, cs, nC, counts, ridge)

    Mv = {}
    for name, (A, Bm) in {
        "M(TR0,TR1)": (reps["tr0"], reps["tr1"]), "M(RI0,RI1)": (reps["ri0"], reps["ri1"]),
        "M(TR0,SLOW)": (reps["tr0"], SLOW), "M(TR1,SLOW)": (reps["tr1"], SLOW),
        "M(RI0,SLOW)": (reps["ri0"], SLOW), "M(RI1,SLOW)": (reps["ri1"], SLOW),
        "M(TR0,PCA)": (reps["tr0"], PCA), "M(TR1,PCA)": (reps["tr1"], PCA),
        "M(TR0,DMIX)": (reps["tr0"], DMIX), "M(TR0,RANDAFF)[diagnostic, no bar]": (reps["tr0"], Rr), "M(TR0,TR0)": (reps["tr0"], reps["tr0"]),
    }.items():
        Mv[name] = pb(A, Bm)
    out = {"M": {k: {"point": v[0], "ci95": ci95(v[1])} for k, v in Mv.items()}}

    def diff(x, y):
        d = Mv[x][1] - Mv[y][1]
        p = Mv[x][0] - Mv[y][0]
        lo, hi = ci95(d)
        return {"point": p, "ci95": [lo, hi], "separated_positive": bool(is_sep_pos(lo, hi))}

    D_ = {"P1: M(TR0,TR1)-M(RI0,RI1)": diff("M(TR0,TR1)", "M(RI0,RI1)")}
    for s in (0, 1):
        D_[f"P2a_s{s}: M(TR{s},SLOW)-M(RI{s},SLOW)"] = diff(f"M(TR{s},SLOW)", f"M(RI{s},SLOW)")
        D_[f"P2b_s{s}: M(TR{s},SLOW)-M(TR{s},PCA)"] = diff(f"M(TR{s},SLOW)", f"M(TR{s},PCA)")
    out["differences"] = D_
    p1 = D_["P1: M(TR0,TR1)-M(RI0,RI1)"]["separated_positive"]
    p2 = all(v["separated_positive"] for k, v in D_.items() if k.startswith("P2"))
    out["P1"], out["P2"] = bool(p1), bool(p2)
    out["outcome"] = outcome(p1, p2)

    # sensitivity (point only, no bar)
    out["sensitivity_ridge_1e-3_point"] = {
        "M(TR0,TR1)": cca_M(reps["tr0"], reps["tr1"], fit_idx, score_idx, 1e-3),
        "M(RI0,RI1)": cca_M(reps["ri0"], reps["ri1"], fit_idx, score_idx, 1e-3),
    }

    # beside rows
    out["beside"] = {
        "TR0_vs_TR1": beside(reps["tr0"], reps["tr1"], fit_idx, score_idx, sub_idx, SLOW),
        "RI0_vs_RI1": beside(reps["ri0"], reps["ri1"], fit_idx, score_idx, sub_idx, SLOW),
    }
    out["slow_canonical_corr_fit_half_first10_last3"] = [float(x) for x in list(fs["s"][:10]) + list(fs["s"][-3:])]

    # controls
    ctl = {}
    ctl["C-SELF"] = {"M": Mv["M(TR0,TR0)"][0], "pass": bool(abs(Mv["M(TR0,TR0)"][0] - 1) <= 1e-6)}
    Q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((K, K)))
    m_rot = cca_M(reps["tr0"], reps["tr1"] @ Q, fit_idx, score_idx)
    ctl["C-ROT"] = {"M_rotated": m_rot, "M_plain": Mv["M(TR0,TR1)"][0], "absdiff": abs(m_rot - Mv["M(TR0,TR1)"][0]), "pass": bool(abs(m_rot - Mv["M(TR0,TR1)"][0]) <= 1e-6)}
    perm = reps["tr1"].copy()
    pr = np.random.default_rng(5)
    for idxs in (fit_idx, score_idx):
        perm[idxs] = reps["tr1"][pr.permutation(idxs)]
    shuf_pt, shuf_boot = pair_boot(reps["tr0"], perm, fit_idx, score_idx, cid_score, nC, counts)
    sh_beside = beside(reps["tr0"], perm, fit_idx, score_idx, sub_idx, SLOW)
    ctl["C-SHUF"] = {"M": shuf_pt, "ci95": ci95(shuf_boot), "M(RI0,RI1)": Mv["M(RI0,RI1)"][0], "mutual_10NN": sh_beside["mutual_10NN"], "chance": sh_beside["chance_10NN"],
                     "pass": bool(shuf_pt < Mv["M(RI0,RI1)"][0] and sh_beside["mutual_10NN"] <= 2 * sh_beside["chance_10NN"])}
    dd = Mv["M(TR0,DMIX)"][1] - Mv["M(TR0,TR1)"][1]
    lo, hi = ci95(dd)
    dpt = Mv["M(TR0,DMIX)"][0] - Mv["M(TR0,TR1)"][0]
    ctl["D-MIX"] = {"M(TR0,DMIX)": Mv["M(TR0,DMIX)"][0], "M(TR0,TR1)": Mv["M(TR0,TR1)"][0], "delta": dpt, "ci95": [lo, hi], "reported_change": bool(abs(dpt) > 0), "pass": bool(is_sep_any(lo, hi))}
    ctl["C-AFFINE(extra)"] = {"max_abs_pooled_adapt_mean_nm_vs_zmean_tr0": float(X["aff_err_tr0"]), "note": "nm-affine identity of the card; no bar in the card"}
    ctl["C-ZC"] = c_zc(X, a_cli)
    out["controls"] = ctl
    out["controls_all_pass"] = bool(all(v.get("pass", False) for k, v in ctl.items() if "pass" in v))
    out["n"] = {"clips": C, "fit_clips": int(fit_clip.sum()), "score_clips": nC, "fit_frames": int(len(fit_idx)), "score_frames": int(len(score_idx)), "slow_pairs": int(len(pair_idx)), "B": B}
    return out


def c_zc(X, a):
    """Compare local zmean with box-2 cached eval npz for the clips present in --zc-dir/M2b_s{0,1}/eval/."""
    if not a.zc_dir or not os.path.isdir(a.zc_dir):
        return {"pass": False, "note": "zc dir absent: control NOT RUN (read unusable)"}
    clips = [str(c) for c in X["clips"]]
    res = {}
    for s in (0, 1):
        d = os.path.join(a.zc_dir, f"M2b_s{s}")
        rows = []
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not f.endswith(".npz"):
                continue
            c = f[:-4]
            z = np.load(os.path.join(d, f))
            m = X["cid"] == clips.index(c)
            loc = X[f"tr{s}"][m]
            rows.append({"clip": c, "T": int(m.sum()), "max_abs_diff": float(np.abs(loc - z["zmean"]).max()), "ok_equal": bool((X["ok"][m] == z["ok"]).all())})
        res[f"s{s}"] = rows
    allr = [r for v in res.values() for r in v]
    res["pass"] = bool(len(allr) >= 6 and all(r["max_abs_diff"] <= 1e-4 and r["ok_equal"] for r in allr))
    return res


def write_md(R, path):
    L = [f"# RESULT_PRH_SEED ({R['meta']['utc']})", "", f"**Outcome: {R['outcome']}**  (P1={R['P1']}, P2={R['P2']}; controls all pass: {R['controls_all_pass']})", ""]
    if not R["controls_all_pass"]:
        L += ["**A CONTROL FAILED: the read is UNUSABLE (instrument failure); do not interpret the outcome.**", ""]
    L += ["Evidence class: MEASURED, ours; artifact `RESULT_PRH_SEED.json` beside this file. Representation probe, no driving tier.", "", "## M (mean canonical correlation, score half, 95 % episode-cluster bootstrap CI, B=" + str(R["n"]["B"]) + ")", "", "| M | point | CI95 |", "|---|---|---|"]
    for k, v in R["M"].items():
        L.append(f"| {k} | {v['point']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |")
    L += ["", "## Paired differences (prediction rows)", "", "| diff | point | CI95 | separated, positive |", "|---|---|---|---|"]
    for k, v in R["differences"].items():
        L.append(f"| {k} | {v['point']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['separated_positive']} |")
    L += ["", "## Controls", "", "```json", json.dumps(R["controls"], indent=1), "```", "", "## Beside rows (no bar)", "", "```json",
          json.dumps({k: {kk: (vv if kk != "profile_along_SLOW" else "see JSON (256 values)") for kk, vv in v.items()} for k, v in R["beside"].items()}, indent=1), "```", ""]
    for k, v in R["beside"].items():
        p = np.array(v["profile_along_SLOW"])
        L.append(f"Profile {k}: mean over SLOW directions 1-16 {p[:16].mean():.3f}, 17-64 {p[16:64].mean():.3f}, 65-256 {p[64:].mean():.3f}.")
    L += ["", f"Ridge 1e-3 sensitivity (point): {json.dumps(R['sensitivity_ridge_1e-3_point'])}", "", f"n: {json.dumps(R['n'])}", "", "## Choices the card left open", ""]
    L += [f"* **{k}**: {v}" for k, v in R["choices"].items()]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="F:/TanitAD-scratch/otband_read/tree")
    ap.add_argument("--root", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/cache")
    ap.add_argument("--tracks", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt")
    ap.add_argument("--ckpt0", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s0/ckpt_last.pt")
    ap.add_argument("--ckpt1", default="E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s1/ckpt_last.pt")
    ap.add_argument("--work", default="F:/TanitAD-scratch/prh_seed/work")
    ap.add_argument("--zc-dir", default="F:/TanitAD-scratch/prh_seed/work/zc")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    t0 = time.time()
    X = extract(a)
    t1 = time.time()
    R = run_probe(X, a)
    R["choices"] = CHOICES
    R["meta"] = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "extract_wall_s": float(X["extract_s"]) if "extract_s" in X else None, "total_wall_s": time.time() - t0, "analysis_wall_s": time.time() - t1,
                 "ckpt_sha256": [sha_file(a.ckpt0), sha_file(a.ckpt1)], "card_sha256": "411ddb7215e279213664c88118488a2f448a60f755dbb42d9780713f645388ec",
                 "card_sha256_first_frozen_wording": "a616e4395ac343f93f70fea3cf0c108f5b1d765cbc5dcf90d5ae1dfdadf07a65 (two words changed, no rule, bar or control)"}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    write_md(R, a.md)
    print(R["outcome"], "controls_all_pass", R["controls_all_pass"])


if __name__ == "__main__":
    main()
