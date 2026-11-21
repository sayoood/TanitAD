"""T1 — LINEAR PROBE on the FROZEN v1 visual state, target `NOT_T_seen`.

H2_CLASSIFIER.md §9 recommendation 5 (escalation #3):
    "a linear (ridge/logistic) probe on the frozen 2048-d state — the lowest-variance reader;
     if this fails, no head rescues the representation."

This script implements exactly that, and nothing else. It reuses, verbatim and by import where
possible, H2's substrate: the same feature files, the same `tau*`, the same TRAIN/HELDOUT split,
the same chunk-grouped 5-fold CV, the same AP definition and the same paired episode-cluster
bootstrap. The only new object is the READER: a linear one instead of a 2 M-parameter attention
head.

Two independent readers are fit so a null cannot be blamed on an optimiser:
    RIDGE     closed form (Cholesky on the regularised Gram) — no optimisation at all
    LOGISTIC  full-batch LBFGS on a convex objective

Bi-directional harness validation (the `e1c_selftest` pattern):
    FIDELITY  the loader must reproduce H2's published substrate counts exactly
    POSITIVE  an ego-only probe must reproduce the separated effect H2 measured (3.74x base)
    NEGATIVE  a feature-shuffled probe and a constant score must NOT separate

usage (eval pod):
  python3 t1_probe.py --feats /workspace/h2probe/feats --bundle /workspace/h2probe/bundle \
                      --out /workspace/h2probe/artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

TAU = 0.5                  # imported value, NEVER re-swept (h2c_train.py:38)
W = 8                      # 0.8 s window (h2c_train.py:39)
SPEED_SCALE = 10.0         # v1's hard contract (MODEL_REGISTRY 1.2)
ACC_SCALE = 5.0
FOLDS = 5
SEED = 0

# H2's published substrate counts — the FIDELITY target. INHERITED from
# H2_CLASSIFIER.md §5.1 + §7 (artifacts/c12_fix.json). Reproducing them is the check that this
# loader is reading the same thing H2's head read.
FIDELITY = {
    "train_rows": 31032, "train_pos": 836,
    "heldout_rows": 50119, "heldout_pos": 1642,
    "heldout_clips": 322, "heldout_pos_clips": 101,
    "train_clips": 198,
}


# ------------------------------------------------------------------ substrate
def load_side(feats_dir, meta, side):
    """Assemble the frame-level probe inputs for one side of the pre-registered split.

    Mirrors `h2c_train.load_split` exactly on the window construction and the ego scaling; the
    target is the frame-level `NOT_T_seen = (a_req_seen_res >= tau*)`, which is `1 - EX[:,1]` in
    H2's layout and is exactly what `h2c_c12fix.py` evaluates.
    """
    IMG_T, IMG_MEAN, IMG_FLAT, EGO_T, EGO_WIN, Y, CLIP, CHUNK = [], [], [], [], [], [], [], []
    used = []
    for m in meta:
        if m["side"] != side:
            continue
        p = os.path.join(feats_dir, f"clip_{m['k']:04d}.npz")
        if not os.path.exists(p):
            continue                                       # dropped by the alignment floor
        z = np.load(p)
        F = z["feats"].astype(np.float32)                  # [T, 2048]
        j = z["j"].astype(np.int64)
        idx = np.clip(j[:, None] - np.arange(W - 1, -1, -1)[None, :], 0, F.shape[0] - 1)
        win = F[idx]                                       # [n, W, 2048]
        IMG_T.append(win[:, -1, :].copy())
        IMG_MEAN.append(win.mean(1))
        IMG_FLAT.append(win.reshape(win.shape[0], -1).astype(np.float16))
        v = z["ego_v"].astype(np.float32)
        apre = z["alon_pre"].astype(np.float32)
        r = np.arange(len(j))
        ridx = np.clip(r[:, None] - np.arange(W - 1, -1, -1)[None, :], 0, len(j) - 1)
        ew = np.stack([v[ridx] / SPEED_SCALE, apre[ridx] / ACC_SCALE], -1)     # [n, W, 2]
        EGO_WIN.append(ew.reshape(len(j), -1))
        EGO_T.append(ew[:, -1, :].copy())
        se = z["areq_seen_res"].astype(np.float32)
        Y.append((se >= TAU).astype(np.float32))           # NOT_T_seen, frame level
        CLIP.append(np.full(len(j), m["k"], np.int64))
        CHUNK.append(np.full(len(j), int(m["chunk"]), np.int64))
        used.append(m)
    cat = lambda L: np.concatenate(L)                                          # noqa: E731
    return dict(img_t=cat(IMG_T), img_win_mean=cat(IMG_MEAN), img_win_flat=cat(IMG_FLAT),
                ego_t=cat(EGO_T), ego_win=cat(EGO_WIN), y=cat(Y), clip=cat(CLIP),
                chunk=cat(CHUNK), meta=used)


# ------------------------------------------------------------------ metrics
def average_precision(y, s):
    """Step-interpolated AP, character-identical to `h2c_stats.average_precision`."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yt = y[o]
    tp = np.cumsum(yt)
    fp = np.cumsum(1.0 - yt)
    P = tp / np.maximum(tp + fp, 1e-12)
    R = tp / yt.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], R])) * P))


def roc_auc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    n1, n0 = float(y.sum()), float((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ readers
def _fit_ridge_all_lams(X, y_pm, lams, dev):
    """Closed-form ridge for every lambda, reusing one Gram. Returns {lam: (w, b)}.

    No optimiser is involved, so a null from this reader cannot be an optimisation failure.
    """
    Xd = torch.as_tensor(X, dtype=torch.float64, device=dev)
    yd = torch.as_tensor(y_pm, dtype=torch.float64, device=dev)
    n, d = Xd.shape
    ones = torch.ones(n, 1, dtype=torch.float64, device=dev)
    Xa = torch.cat([Xd, ones], 1)                       # intercept as an unpenalised column
    G = Xa.T @ Xa
    rhs = Xa.T @ yd
    P = torch.eye(d + 1, dtype=torch.float64, device=dev)
    P[d, d] = 0.0                                       # do NOT penalise the intercept
    out = {}
    for lam in lams:
        A = G + lam * P
        try:
            L = torch.linalg.cholesky(A)
            wa = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
        except Exception:
            wa = torch.linalg.solve(A + 1e-6 * torch.eye(d + 1, dtype=torch.float64, device=dev),
                                    rhs)
        out[lam] = (wa[:d].clone(), wa[d].clone())
    del Xd, Xa, G
    return out


def _score_linear(X, wb, dev, chunk=200000):
    w, b = wb
    outs = []
    for i in range(0, X.shape[0], chunk):
        xb = torch.as_tensor(X[i:i + chunk], dtype=torch.float64, device=dev)
        outs.append((xb @ w + b).cpu().numpy())
    return np.concatenate(outs)


def _fit_logistic(X, y, lam, pos_weight, dev, max_iter=300):
    """L2-regularised logistic regression, full-batch LBFGS on a convex objective.

    objective = sum_i wgt_i * BCE(sigma(x_i w + b), y_i) + lam * ||w||^2   (intercept unpenalised)
    """
    Xd = torch.as_tensor(X, dtype=torch.float32, device=dev)
    yd = torch.as_tensor(y, dtype=torch.float32, device=dev)
    d = Xd.shape[1]
    w = torch.zeros(d, dtype=torch.float32, device=dev, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, device=dev, requires_grad=True)
    pw = torch.tensor(float(pos_weight), device=dev)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        z = Xd @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            z, yd, pos_weight=pw, reduction="sum") + lam * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (w.detach().to(torch.float64), b.detach().to(torch.float64)[0])


# ------------------------------------------------------------------ CV + selection
def cv_select(Xtr, ytr, foldid, reader, lams, dev, pos_weights=(1.0,)):
    """5-fold CV grouped by CHUNK inside TRAIN; select on CV-AP. Never reads the held-out side."""
    best = None
    curve = []
    for pw in pos_weights:
        oof = {lam: np.zeros(len(ytr)) for lam in lams}
        for f in range(FOLDS):
            m_va = foldid == f
            m_tr = ~m_va
            if reader == "ridge":
                fits = _fit_ridge_all_lams(Xtr[m_tr], (ytr[m_tr] * 2.0 - 1.0), lams, dev)
                for lam in lams:
                    oof[lam][m_va] = _score_linear(Xtr[m_va], fits[lam], dev)
            else:
                for lam in lams:
                    wb = _fit_logistic(Xtr[m_tr], ytr[m_tr], lam, pw, dev)
                    oof[lam][m_va] = _score_linear(Xtr[m_va], wb, dev)
        for lam in lams:
            a = average_precision(ytr, oof[lam])
            curve.append({"reader": reader, "lam": float(lam), "pos_weight": float(pw),
                          "cv_ap": round(float(a), 6)})
            if best is None or a > best["cv_ap"]:
                best = {"reader": reader, "lam": float(lam), "pos_weight": float(pw),
                        "cv_ap": float(a)}
    return best, curve


def fit_final_and_score(Xtr, ytr, Xho, sel, dev):
    if sel["reader"] == "ridge":
        fits = _fit_ridge_all_lams(Xtr, (ytr * 2.0 - 1.0), [sel["lam"]], dev)
        wb = fits[sel["lam"]]
    else:
        wb = _fit_logistic(Xtr, ytr, sel["lam"], sel["pos_weight"], dev)
    return _score_linear(Xho, wb, dev)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stats-path", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.stats_path)
    from taniteval.ci import _draws, episode_index      # noqa: E402  the program's own machinery
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = False       # exactness over speed for the Gram
    t0 = time.time()

    meta = json.load(open(os.path.join(args.bundle, "h2c_meta.json")))
    TR = load_side(args.feats, meta, "TRAIN")
    HO = load_side(args.feats, meta, "HELDOUT")

    # ---------------------------------------------------------------- FIDELITY (direction 1)
    got = {"train_rows": int(len(TR["y"])), "train_pos": int(TR["y"].sum()),
           "heldout_rows": int(len(HO["y"])), "heldout_pos": int(HO["y"].sum()),
           "heldout_clips": int(len(np.unique(HO["clip"]))),
           "heldout_pos_clips": int(len(np.unique(HO["clip"][HO["y"] > 0]))),
           "train_clips": int(len(np.unique(TR["clip"])))}
    fidelity = {"expected_INHERITED_from_H2": FIDELITY, "measured": got,
                "match": {k: bool(got[k] == v) for k, v in FIDELITY.items()},
                "all_match": bool(all(got[k] == v for k, v in FIDELITY.items()))}
    print("[fidelity]", json.dumps(fidelity["match"]), "ALL_MATCH=", fidelity["all_match"],
          flush=True)

    y_ho = HO["y"].astype(float)
    eid = HO["clip"]
    base = float(y_ho.mean())

    # ---------------------------------------------------------------- representations
    mu = TR["img_t"].mean(0)
    sd = TR["img_t"].std(0) + 1e-5
    Ztr_t = (TR["img_t"] - mu) / sd
    Zho_t = (HO["img_t"] - mu) / sd

    mu_m = TR["img_win_mean"].mean(0)
    sd_m = TR["img_win_mean"].std(0) + 1e-5
    Ztr_m = (TR["img_win_mean"] - mu_m) / sd_m
    Zho_m = (HO["img_win_mean"] - mu_m) / sd_m

    # PCA fit on TRAIN rows of the standardised img_t (eigendecomposition of the covariance)
    Xd = torch.as_tensor(Ztr_t, dtype=torch.float64, device=dev)
    C = (Xd.T @ Xd) / Xd.shape[0]
    evals, evecs = torch.linalg.eigh(C)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    evr = (evals / evals.sum()).cpu().numpy()
    del Xd, C

    def pca(Z, k):
        Xq = torch.as_tensor(Z, dtype=torch.float64, device=dev)
        return (Xq @ evecs[:, :k]).float().cpu().numpy()

    reps = {}
    reps["img_t"] = (Ztr_t, Zho_t)
    reps["img_win_mean"] = (Ztr_m, Zho_m)
    for k in (16, 64, 256):
        reps[f"img_pca{k}"] = (pca(Ztr_t, k), pca(Zho_t, k))
    reps["ego_t"] = (TR["ego_t"], HO["ego_t"])
    reps["ego_win"] = (TR["ego_win"], HO["ego_win"])
    for k in (16, 64, 256):
        reps[f"ego_win+img_pca{k}"] = (
            np.concatenate([TR["ego_win"], reps[f"img_pca{k}"][0]], 1),
            np.concatenate([HO["ego_win"], reps[f"img_pca{k}"][1]], 1))
    # NEGATIVE CONTROL (direction 2): features permuted across rows, labels untouched.
    rng = np.random.default_rng(SEED)
    perm_tr = rng.permutation(len(TR["y"]))
    perm_ho = rng.permutation(len(HO["y"]))
    reps["img_t_SHUFFLED"] = (Ztr_t[perm_tr], Zho_t[perm_ho])
    # the head's exact input, read linearly (float16 -> float32 once, memory-aware)
    reps["img_win_flat"] = ((TR["img_win_flat"].astype(np.float32)
                             - np.tile(mu, W)) / np.tile(sd, W),
                            (HO["img_win_flat"].astype(np.float32)
                             - np.tile(mu, W)) / np.tile(sd, W))

    chunks = sorted({int(m["chunk"]) for m in TR["meta"]})
    fold_of = {c: i % FOLDS for i, c in enumerate(chunks)}
    foldid = np.array([fold_of[c] for c in TR["chunk"]])

    LAMS = [float(10.0 ** e) for e in np.arange(-1.0, 7.5, 0.5)]
    LAMS_LOG = [float(10.0 ** e) for e in np.arange(-1.0, 7.5, 1.0)]
    npos, nneg = float(TR["y"].sum()), float(len(TR["y"]) - TR["y"].sum())
    PWS = (1.0, nneg / max(npos, 1.0))

    results = {
        "target": "NOT_T_seen = (a_req_seen_res >= tau*), tau* = 0.5 m/s^2, FRAME level",
        "tau": TAU, "window": W, "folds": FOLDS, "seed": SEED,
        "fidelity_check": fidelity,
        "heldout": {"n_frames": int(len(y_ho)), "n_positives": int(y_ho.sum()),
                    "base_rate": base, "n_clips": got["heldout_clips"],
                    "n_positive_clips": got["heldout_pos_clips"]},
        "train": {"n_frames": int(len(TR["y"])), "n_positives": int(TR["y"].sum()),
                  "base_rate": float(TR["y"].mean()), "n_clips": got["train_clips"]},
        "pca_explained_variance_ratio": {
            "top1": float(evr[0]), "cum16": float(evr[:16].sum()),
            "cum64": float(evr[:64].sum()), "cum256": float(evr[:256].sum())},
        "folds_by_chunk": {str(f): [c for c in chunks if fold_of[c] == f] for f in range(FOLDS)},
        "lam_grid": LAMS, "lam_grid_logistic": LAMS_LOG, "pos_weights_logistic": list(PWS),
        "estimator": "paired episode-cluster bootstrap (taniteval.ci._draws), "
                     f"B={args.boot}, seed={SEED}, unit = clip cluster",
        "arms": {}, "cv_curves": [],
    }

    # ---------------------------------------------------------------- the bootstrap
    uniq, idx_by_ep = episode_index(eid)
    draws = list(_draws(uniq, idx_by_ep, args.boot, SEED))
    ok_draw = np.array([bool(y_ho[s_].sum() > 0) for s_ in draws])
    zero = np.zeros_like(y_ho)
    # ⚠️ AMENDMENT A1 (declared in SITUATION_SEMANTICS.md, found in the B=30 smoke run BEFORE the
    # primary was read at B=2000). A CONSTANT score is fully TIED, so under `average_precision`'s
    # stable sort its AP is the AP of the ROW ORDER — not the base rate. In the full sample the
    # rows are in a fixed clip order, so that value is biased (measured: 0.046 vs a 0.0328 base
    # rate); inside a bootstrap draw the clip order is randomised by `_draws`, so the draws are
    # ~unbiased. The consequence is a biased POINT estimate with an approximately correct CI —
    # which is why H2's published table can show a point estimate sitting outside its own
    # interval. Both conventions are computed: `const` reproduces H2 exactly, `rand` (a uniform
    # random ranker, seed 0) is the unbiased one. **The verdict is read off the CI, which both
    # share.**
    rng_c = np.random.default_rng(SEED)
    rand_score = rng_c.random(len(y_ho))

    def _ap_draws(s):
        out = np.empty(len(draws))
        for i, sel in enumerate(draws):
            out[i] = average_precision(y_ho[sel], s[sel]) if ok_draw[i] else np.nan
        return out

    ch_const_draws = _ap_draws(zero)
    ch_rand_draws = _ap_draws(rand_score)
    ch_const_point = average_precision(y_ho, zero)
    ch_rand_point = average_precision(y_ho, rand_score)

    def _pct(d):
        d = d[np.isfinite(d)]
        if d.size <= 50:
            return float("nan"), float("nan"), int(d.size)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(lo), float(hi), int(d.size)

    def eval_arm(s):
        a_draws = _ap_draws(s)
        pa = average_precision(y_ho, s)
        lo, hi, n = _pct(a_draws)
        res = {"AP": {"point": round(pa, 6), "lo": round(lo, 6), "hi": round(hi, 6),
                      "n_draws_used": n}}
        for tag, cd, cp in (("const", ch_const_draws, ch_const_point),
                            ("rand", ch_rand_draws, ch_rand_point)):
            d = a_draws - cd
            lo, hi, n = _pct(d)
            res[f"paired_AP_vs_chance_{tag}"] = {
                "AP_chance": round(cp, 6), "delta": round(pa - cp, 6),
                "delta_boot_median": round(float(np.nanmedian(d)), 6),
                "lo": round(lo, 6), "hi": round(hi, 6),
                "separated": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                "above_chance": bool(np.isfinite(lo) and lo > 0), "n_draws_used": n}
        res["paired_AP_vs_chance"] = res["paired_AP_vs_chance_const"]   # H2's convention, primary
        return res

    # ---------------------------------------------------------------- run the ladder
    # ORDER MATTERS: the controls (ego = ceiling, shuffled = floor) are computed and written
    # FIRST, per PRE_REGISTRATION_T1 §4 — the power ceiling is established before the primary
    # is read.
    ORDER = ["ego_t", "ego_win", "img_t_SHUFFLED",
             "img_t", "img_win_mean", "img_win_flat",
             "img_pca16", "img_pca64", "img_pca256",
             "ego_win+img_pca16", "ego_win+img_pca64", "ego_win+img_pca256"]
    for name in ORDER:
        Xtr, Xho = reps[name]
        t1 = time.time()
        arm = {"dim": int(Xtr.shape[1])}
        for reader in ("ridge", "logistic"):
            lams = LAMS if reader == "ridge" else LAMS_LOG
            pws = (1.0,) if reader == "ridge" else PWS
            sel, curve = cv_select(Xtr, TR["y"], foldid, reader, lams, dev, pws)
            results["cv_curves"].extend([{**c, "rep": name} for c in curve])
            s = fit_final_and_score(Xtr, TR["y"], Xho, sel, dev)
            arm[reader] = {"selected": sel, **eval_arm(s),
                           "AP_over_base": round(average_precision(y_ho, s) / max(base, 1e-12), 4),
                           "AUROC": round(roc_auc(y_ho, s), 6)}
            np.save(os.path.join(args.out, f"score_{name}_{reader}.npy"), s.astype(np.float32))
        results["arms"][name] = arm
        print(f"[{name}] d={arm['dim']} "
              f"ridge AP={arm['ridge']['AP']['point']:.5f} ({arm['ridge']['AP_over_base']}x) "
              f"dAP={arm['ridge']['paired_AP_vs_chance']['delta']:+.5f} "
              f"[{arm['ridge']['paired_AP_vs_chance']['lo']:+.5f},"
              f"{arm['ridge']['paired_AP_vs_chance']['hi']:+.5f}] "
              f"sep={arm['ridge']['paired_AP_vs_chance']['above_chance']} | "
              f"logit AP={arm['logistic']['AP']['point']:.5f} "
              f"({arm['logistic']['AP_over_base']}x) "
              f"sep={arm['logistic']['paired_AP_vs_chance']['above_chance']} "
              f"({time.time()-t1:.0f}s)", flush=True)
        json.dump(results, open(os.path.join(args.out, "t1_probe.json"), "w"),
                  indent=2, default=float)     # bank incrementally

    # the two chance arms themselves, for completeness and as the floor of record
    results["chance_arms"] = {
        "constant_rowtie": {"AP_point": round(ch_const_point, 6),
                            "AP_boot_median": round(float(np.nanmedian(ch_const_draws)), 6)},
        "uniform_random": {"AP_point": round(ch_rand_point, 6),
                           "AP_boot_median": round(float(np.nanmedian(ch_rand_draws)), 6)},
        "analytic_base_rate": round(base, 6)}
    results["arms"]["constant"] = {"dim": 0, "ridge": {
        "selected": None, **eval_arm(zero), "AP_over_base": round(ch_const_point / base, 4),
        "AUROC": 0.5}}
    results["wallclock_s"] = round(time.time() - t0, 1)
    json.dump(results, open(os.path.join(args.out, "t1_probe.json"), "w"), indent=2, default=float)
    print(f"[t1] done in {results['wallclock_s']:.0f}s -> {args.out}/t1_probe.json", flush=True)


if __name__ == "__main__":
    main()
