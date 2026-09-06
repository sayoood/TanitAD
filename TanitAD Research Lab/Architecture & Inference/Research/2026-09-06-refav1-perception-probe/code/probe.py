"""P1 — the perception probe on refav1's FROZEN trunk.

⭐ THE QUESTION
    A perfect goal makes refav1's planner 2.03x WORSE and the search optimises
    *better* while driving worse => the cost is misspecified. But a cost is
    defined IN THE LATENT. If the lead vehicle is not decodable from the latent,
    no cost in that space CAN express "keep distance" -- the objective is
    UNREPRESENTABLE, not mistuned. This separates the two.

⛔ EVERY CONTROL MUST READ A KNOWN VALUE (the 2026-08-22 lesson: four confident
   wrong numbers in one afternoon, each caught only by a control).

    constant     -> R2_skill EXACTLY 0.000000 by construction (it IS the
                    denominator's predictor). If it is not exactly 0, the
                    metric is mis-implemented and nothing else is readable.
    pixels       -> the FLOOR. A learned representation that does not beat raw
                    pixels has added NOTHING.
    dinov3       -> the EXTERNAL REFERENCE (frozen encoder, no refav1 training).
    shuffle_wc   -> targets time-shuffled WITHIN EACH CLIP. This is the control
                    that matters here: `gap` is strongly autocorrelated within a
                    clip, so a probe can score well merely by RECOGNISING THE
                    CLIP. Structure surviving this shuffle is clip identity,
                    not perception. Must read ~0.
    within-clip  -> every arm is ALSO scored on clip-mean-centred targets, which
                    is the honest "does it track the lead as it MOVES" number.

⛔ NO HYPERPARAMETER EVER SEES THE SCORED SPLIT. Both PCA stages and the ridge
   lambda are fit on the FIT split only, lambda by clip-grouped inner CV.
   (Selecting lambda on the test split once produced +0.0000 with a zero-width
   CI that beat every honest estimate.)

⚠️ FUNCTION CLASS: ridge = LINEAR. A negative from a linear probe is NOT a
   negative about learnability. `--mlp` adds a nonlinear head for the decisive
   cell so the verdict can state both.

ESTIMATOR: episode-cluster bootstrap over the SCORED split's clips -- it answers
"would another draw of EPISODES say this?". It does NOT answer "would another
training run say this?" (H-ESTIM-SEED-1). Here the trunk is frozen and the head
is a CLOSED-FORM ridge solve, so there is no training stochasticity to price:
the only randomness is the episode draw and the split seed, and --seeds sweeps
the latter.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

POOL = 160          # 8 x 20 pooled cells
CHAN_K = 32         # stage-A channel PCA
FINAL_K = 128       # stage-B final PCA (= d of every arm)
LAMBDAS = np.logspace(-3, 6, 19)
D_RFF = 1024        # nonlinear arm width (constants from rangeprobe_rff.py)


def rff_map(Z, fit_rows, seed):
    """Random Fourier Features = an RBF kernel ridge that stays CONVEX and
    CLOSED-FORM -- so there is no optimiser, LR or early-stopping to get wrong.
    ⛔ Bandwidth (median heuristic) and the RFF draw are fixed on the FIT SPLIT
    ONLY; the scored split is transformed by them, never consulted for them.
    (The MLP variant of this probe is superseded: its time-shuffled control read
    -14.00 and its pixel floor -27.98, where a no-information predictor must
    read ~0.)"""
    rng = np.random.RandomState(seed + 4242)
    F = Z[fit_rows]
    sub = F[rng.choice(len(F), min(2000, len(F)), replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    sigma2 = med / 2.0 if med > 0 else 1.0
    W = rng.normal(0, 1.0 / np.sqrt(sigma2), size=(Z.shape[1], D_RFF))
    b = rng.uniform(0, 2 * np.pi, size=D_RFF)
    return np.sqrt(2.0 / D_RFF) * np.cos(Z @ W + b)


# --------------------------------------------------------------------------
def r2_skill(y_true, y_pred, fit_mean):
    """1 - SSE_model / SSE_fitmean. The constant arm reads EXACTLY 0."""
    sse_m = float(np.sum((y_true - y_pred) ** 2))
    sse_c = float(np.sum((y_true - fit_mean) ** 2))
    if sse_c == 0.0:
        return float("nan")
    return 1.0 - sse_m / sse_c


def ridge_fit(X, y, lam):
    """Closed form with an intercept; features are pre-centred/scaled."""
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d, dtype=np.float64)
    b = X.T @ y
    return np.linalg.solve(A, b)


def pick_lambda(X, y, groups, n_folds=5, seed=0):
    """Clip-grouped inner CV on the FIT SPLIT ONLY. Never sees the scored split."""
    uq = np.unique(groups)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uq))
    fold_of = {g: (perm[i] % n_folds) for i, g in enumerate(uq)}
    fid = np.array([fold_of[g] for g in groups])
    best, best_sse = None, np.inf
    for lam in LAMBDAS:
        sse = 0.0
        for f in range(n_folds):
            tr, va = fid != f, fid == f
            if va.sum() == 0 or tr.sum() <= X.shape[1]:
                continue
            w = ridge_fit(X[tr], y[tr] - y[tr].mean(), lam)
            pred = X[va] @ w + y[tr].mean()
            sse += float(np.sum((y[va] - pred) ** 2))
        if sse < best_sse:
            best_sse, best = sse, lam
    return best


def boot_ci(y, pred, fit_mean, clip_ids, n_boot=2000, seed=0):
    """Episode-CLUSTER bootstrap: resample CLIPS with replacement."""
    uq = np.unique(clip_ids)
    idx_of = {c: np.where(clip_ids == c)[0] for c in uq}
    rng = np.random.RandomState(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(uq), len(uq))
        ii = np.concatenate([idx_of[uq[p]] for p in pick])
        vals[b] = r2_skill(y[ii], pred[ii], fit_mean)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_boot_delta(y, pred_a, pred_b, fit_mean, clip_ids, n_boot=2000, seed=0):
    """PAIRED episode-cluster bootstrap on R2_skill(a) - R2_skill(b): the SAME
    resampled clips score BOTH arms in every draw. This is the estimator that
    answers 'does arm a beat arm b', which two separate one-arm intervals do
    NOT -- overlapping marginal CIs are not a null result.
    ⚠️ It answers "would another draw of EPISODES say this?" and nothing else:
    not another training run, not another inference run. Here the trunk is
    frozen and the head is a closed-form solve, so the episode draw and the
    split seed are the only randomness -- the seed sweep prices the latter."""
    uq = np.unique(clip_ids)
    idx_of = {c: np.where(clip_ids == c)[0] for c in uq}
    rng = np.random.RandomState(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.randint(0, len(uq), len(uq))
        ii = np.concatenate([idx_of[uq[p]] for p in pick])
        vals[i] = (r2_skill(y[ii], pred_a[ii], fit_mean)
                   - r2_skill(y[ii], pred_b[ii], fit_mean))
    d = r2_skill(y, pred_a, fit_mean) - r2_skill(y, pred_b, fit_mean)
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return {"delta": float(d), "ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------
def load_bank(bank, arms):
    files = sorted(glob.glob(os.path.join(bank, "*.npz")))
    clips, data = [], []
    for p in files:
        z = np.load(p, allow_pickle=True)
        clips.append(os.path.basename(p)[:-4])
        data.append(z)
    return clips, data


def stage_a_channel_pca(data, fit_mask, key, rng, max_tok=300000):
    """PCA over CHANNEL vectors, fit on FIT-split tokens only."""
    toks = []
    budget = max_tok
    for z, isfit in zip(data, fit_mask):
        if not isfit or budget <= 0:
            continue
        a = z[key]                       # [n, POOL, C] fp16
        a = a.reshape(-1, a.shape[-1])
        take = min(len(a), max(1, budget // 8))
        sel = rng.choice(len(a), take, replace=False)
        toks.append(a[sel].astype(np.float32))
        budget -= take
    T = np.concatenate(toks, 0)
    mu = T.mean(0)
    T = T - mu
    # randomized SVD via covariance eig (C is 1024 -> cheap and exact enough)
    C = (T.T @ T) / max(len(T) - 1, 1)
    w, V = np.linalg.eigh(C.astype(np.float64))
    order = np.argsort(w)[::-1][:CHAN_K]
    return mu, V[:, order].astype(np.float32)


def build_arm(data, arm, fit_mask, rng):
    """-> X [n_rows, D_raw] float32 for one arm, with stage-A applied."""
    if arm == "pix":
        return np.concatenate(
            [z["pix"].reshape(len(z["pix"]), -1).astype(np.float32) / 255.0
             for z in data], 0)
    key = {"field": "field", "dino": "dino"}[arm]
    mu, P = stage_a_channel_pca(data, fit_mask, key, rng)
    out = []
    for z in data:
        a = z[key].astype(np.float32)                 # [n, POOL, C]
        a = (a - mu) @ P                              # [n, POOL, CHAN_K]
        out.append(a.reshape(len(a), -1))
    return np.concatenate(out, 0)


def stage_b(X, fit_rows, k):
    """PCA to k comps, FIT ROWS ONLY, then unit-variance scaling."""
    mu = X[fit_rows].mean(0)
    Xc = X - mu
    F = Xc[fit_rows]
    k = min(k, F.shape[0] - 1, F.shape[1])
    # economical: eig of the smaller Gram
    if F.shape[1] <= F.shape[0]:
        C = (F.T @ F) / max(len(F) - 1, 1)
        w, V = np.linalg.eigh(C.astype(np.float64))
        P = V[:, np.argsort(w)[::-1][:k]].astype(np.float32)
    else:
        G = (F @ F.T) / max(len(F) - 1, 1)
        w, U = np.linalg.eigh(G.astype(np.float64))
        o = np.argsort(w)[::-1][:k]
        U = U[:, o].astype(np.float32)
        P = F.T @ U
        P /= (np.linalg.norm(P, axis=0, keepdims=True) + 1e-8)
    Z = Xc @ P
    sd = Z[fit_rows].std(0) + 1e-6
    return Z / sd, k


# --------------------------------------------------------------------------
def run_cell(Z, y, rows_ok, fit_rows, score_rows, clip_ids, seed, n_boot):
    fit = fit_rows & rows_ok
    sco = score_rows & rows_ok
    if fit.sum() < 50 or sco.sum() < 50:
        return None
    Xf, yf = Z[fit], y[fit]
    Xs, ys = Z[sco], y[sco]
    fm = float(yf.mean())
    lam = pick_lambda(Xf, yf, clip_ids[fit], seed=seed)
    w = ridge_fit(Xf, yf - fm, lam)
    pred = Xs @ w + fm
    r2 = r2_skill(ys, pred, fm)
    lo, hi = boot_ci(ys, pred, fm, clip_ids[sco], n_boot=n_boot, seed=seed)
    # within-clip (clip-mean-centred) skill: does it track the lead as it MOVES?
    yc = ys.copy().astype(np.float64)
    pc = pred.copy().astype(np.float64)
    for c in np.unique(clip_ids[sco]):
        m = clip_ids[sco] == c
        yc[m] -= ys[m].mean()
        pc[m] -= pred[m].mean()
    r2_wc = r2_skill(yc, pc, 0.0)
    return {"r2": r2, "ci": [lo, hi], "lam": float(lam),
            "n_fit": int(fit.sum()), "n_score": int(sco.sum()),
            "d": int(Z.shape[1]), "r2_within_clip": r2_wc,
            "n_clips_score": int(len(np.unique(clip_ids[sco]))),
            "_pred": pred, "_fm": fm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="/home/nvidia/percprobe/bank")
    ap.add_argument("--out", default="/home/nvidia/percprobe/raw/probe_results.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--final-k", type=int, default=FINAL_K)
    ap.add_argument("--offset-sweep", action="store_true")
    ap.add_argument("--rff", action="store_true",
                    help="add the NONLINEAR (RFF/RBF kernel ridge) arms")
    ap.add_argument("--dump-pred", default="",
                    help="npz of per-frame DECODED lead (cx,cy) on the SCORED "
                         "split -- the input the video renderer draws")
    args = ap.parse_args()

    clips, data = load_bank(args.bank, None)
    print("clips banked: %d" % len(clips), flush=True)

    n_per = np.array([len(z["gap"]) for z in data])
    clip_ids = np.concatenate([[c] * n for c, n in zip(clips, n_per)])
    order = np.argsort(clips)

    # --- episode-disjoint split (seeded, by CLIP) --------------------------
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(clips))
    n_fit_clips = int(round(0.6 * len(clips)))
    fit_clips = set(np.array(clips)[perm[:n_fit_clips]])
    fit_mask = [c in fit_clips for c in clips]
    fit_rows = np.array([c in fit_clips for c in clip_ids])
    score_rows = ~fit_rows
    print("fit clips %d / score clips %d" % (len(fit_clips), len(clips) - len(fit_clips)),
          flush=True)

    # --- targets -----------------------------------------------------------
    def cat(k):
        return np.concatenate([z[k].astype(np.float64) for z in data])

    gap_unc = cat("gap_unc")
    lat_unc = cat("lat_unc")
    # ⚠️ counts are log1p'd (the convention in spatialenv.py): raw agent counts
    # are heavy-tailed, and on the raw scale a handful of dense clips dominate
    # the SSE so every interval blows up. MEASURED on the 30-clip smoke: raw
    # `n_agents_visible` gave field CI [-1.4809, +0.5454].
    targets = {
        "lead_gap_m_cap30": (gap_unc, np.isfinite(gap_unc) & (gap_unc <= 30.0)),
        "lead_gap_m_cap80": (gap_unc, np.isfinite(gap_unc) & (gap_unc <= 80.0)),
        "lead_lat_m_cap30": (lat_unc, np.isfinite(gap_unc) & (gap_unc <= 30.0)),
        "log1p_n_agents_visible": (np.log1p(cat("n_vis")),
                                   np.ones(len(gap_unc), bool)),
        "lead_present_cap30": (np.isfinite(gap_unc) & (gap_unc <= 30.0),
                               np.ones(len(gap_unc), bool)),
    }

    # --- arms --------------------------------------------------------------
    results = {"_meta": {"seed": args.seed, "n_clips": len(clips),
                         "n_rows": int(n_per.sum()),
                         "n_fit_clips": len(fit_clips),
                         "final_k": args.final_k, "chan_k": CHAN_K,
                         "pool_cells": POOL, "n_boot": args.n_boot,
                         "function_class": "ridge (LINEAR)",
                         "estimator": "episode-cluster bootstrap over SCORED clips"}}

    Zs = {}
    for arm in ("pix", "dino", "field"):
        rng2 = np.random.RandomState(args.seed + 17)
        X = build_arm(data, arm, fit_mask, rng2)
        Z, k = stage_b(X, fit_rows, args.final_k)
        Zs[arm] = Z
        print("arm %-6s raw_d=%-7d -> d=%d" % (arm, X.shape[1], k), flush=True)
        del X
    if args.rff:
        for arm in ("pix", "dino", "field"):
            Zs[arm + "_rff"] = rff_map(Zs[arm], fit_rows, args.seed)
            print("arm %-10s (NONLINEAR RFF) d=%d"
                  % (arm + "_rff", Zs[arm + "_rff"].shape[1]), flush=True)
    ARMS = ["pix", "dino", "field"] + (
        ["pix_rff", "dino_rff", "field_rff"] if args.rff else [])
    results["_meta"]["arms"] = ARMS

    for tname, (y, ok) in targets.items():
        y = np.asarray(y, np.float64)
        results[tname] = {}
        # ---- CONSTANT control: MUST read exactly 0.000000 ----------------
        fit = fit_rows & ok
        sco = score_rows & ok
        if fit.sum() < 50 or sco.sum() < 50:
            results[tname]["_skipped"] = "n too small (fit=%d score=%d)" % (
                fit.sum(), sco.sum())
            print("SKIP %s (n too small)" % tname, flush=True)
            continue
        fm = float(y[fit].mean())
        const_pred = np.full(int(sco.sum()), fm)
        results[tname]["constant"] = {
            "r2": r2_skill(y[sco], const_pred, fm), "d": 0,
            "n_score": int(sco.sum())}

        for arm in ARMS:
            r = run_cell(Zs[arm], y, ok, fit_rows, score_rows, clip_ids,
                         args.seed, args.n_boot)
            results[tname][arm] = r

        # ---- within-clip time-shuffle control (targets permuted in-clip) --
        ysh = y.copy()
        rsh = np.random.RandomState(args.seed + 99)
        for c in clips:
            m = (clip_ids == c) & ok
            if m.sum() > 1:
                v = ysh[m]
                ysh[m] = v[rsh.permutation(len(v))]
        results[tname]["shuffle_within_clip"] = run_cell(
            Zs["field"], ysh, ok, fit_rows, score_rows, clip_ids,
            args.seed, args.n_boot)
        if args.rff:
            results[tname]["shuffle_within_clip_rff"] = run_cell(
                Zs["field_rff"], ysh, ok, fit_rows, score_rows, clip_ids,
                args.seed, args.n_boot)

        # ---- report -------------------------------------------------------
        print("\n=== %s ===" % tname, flush=True)
        for a in (["constant"] + ARMS + ["shuffle_within_clip",
                                         "shuffle_within_clip_rff"]):
            r = results[tname].get(a)
            if r is None:
                continue
            if "ci" in r:
                print("  %-20s R2=%+.4f  CI[%+.4f,%+.4f]  wc=%+.4f  n=%d d=%d lam=%.3g"
                      % (a, r["r2"], r["ci"][0], r["ci"][1], r["r2_within_clip"],
                         r["n_score"], r["d"], r["lam"]), flush=True)
            else:
                print("  %-20s R2=%+.6f  (control; d=%d n=%d)"
                      % (a, r["r2"], r["d"], r["n_score"]), flush=True)

        # ---- PAIRED deltas: the decision-relevant comparison ---------------
        # (overlapping one-arm CIs are NOT a null result)
        sco_clips = clip_ids[sco]
        ys = y[sco]
        pairs = [("field", "pix"), ("field", "dino"), ("dino", "pix")]
        if args.rff:
            pairs += [("field_rff", "pix_rff"), ("field_rff", "dino_rff"),
                      ("field_rff", "field")]
        results[tname]["_paired"] = {}
        for a, b in pairs:
            ra, rb = results[tname].get(a), results[tname].get(b)
            if not ra or not rb:
                continue
            d = paired_boot_delta(ys, ra["_pred"], rb["_pred"], ra["_fm"],
                                  sco_clips, n_boot=args.n_boot, seed=args.seed)
            results[tname]["_paired"]["%s_minus_%s" % (a, b)] = d
            print("  PAIRED %-22s d=%+.4f  CI[%+.4f,%+.4f]  %s"
                  % ("%s - %s" % (a, b), d["delta"], d["ci"][0], d["ci"][1],
                     "EXCLUDES 0" if d["excludes_zero"] else "spans 0"),
                  flush=True)

    # ---- ALIGNMENT SWEEP: best offset must be 0 --------------------------
    if args.offset_sweep:
        print("\n=== alignment sweep (field / lead_gap_m_cap30) ===", flush=True)
        sweep = {}
        for off in (-2, -1, 0, 1, 2):
            ysh = np.full(len(gap_unc), np.nan)
            pos = 0
            for z in data:
                n = len(z["gap"])
                g = z["gap_unc"].astype(np.float64)
                src = np.arange(n) + off
                good = (src >= 0) & (src < n)
                tmp = np.full(n, np.nan)
                tmp[good] = g[src[good]]
                ysh[pos:pos + n] = tmp
                pos += n
            ok = np.isfinite(ysh) & (ysh <= 30.0)
            r = run_cell(Zs["field"], ysh, ok, fit_rows, score_rows, clip_ids,
                         args.seed, 200)
            sweep[str(off)] = r
            if r:
                print("  offset %+d rows (%+0.1f s): R2=%+.4f  n=%d"
                      % (off, off * 0.2, r["r2"], r["n_score"]), flush=True)
        results["_alignment_sweep"] = sweep

    # ---- per-frame decoded lead, for the video ---------------------------
    if args.dump_pred:
        frame_idx = np.concatenate([z["frame_idx"] for z in data])
        ok30 = np.isfinite(gap_unc) & (gap_unc <= 30.0)
        dump = {"clip": clip_ids, "frame_idx": frame_idx,
                "gap_true": gap_unc, "lat_true": lat_unc,
                "in_score_split": score_rows, "labelled_lead": ok30}
        for tname, key in (("lead_gap_m_cap30", "gap"),
                           ("lead_lat_m_cap30", "lat")):
            y = np.asarray(targets[tname][0], np.float64)
            fit = fit_rows & ok30
            fm = float(y[fit].mean())
            lam = pick_lambda(Zs["field"][fit], y[fit], clip_ids[fit], seed=args.seed)
            w = ridge_fit(Zs["field"][fit], y[fit] - fm, lam)
            pred_all = Zs["field"] @ w + fm      # every row; mask at draw time
            dump[key + "_pred_field"] = pred_all
            # the pixel FLOOR's prediction, so the video can show it too
            lam_p = pick_lambda(Zs["pix"][fit], y[fit], clip_ids[fit], seed=args.seed)
            wp = ridge_fit(Zs["pix"][fit], y[fit] - fm, lam_p)
            dump[key + "_pred_pix"] = Zs["pix"] @ wp + fm
        np.savez_compressed(args.dump_pred, **dump)
        print("wrote per-frame decode -> %s" % args.dump_pred, flush=True)

    # strip the prediction arrays (kept only to build the paired deltas)
    for t in results:
        if not isinstance(results[t], dict):
            continue
        for a in results[t]:
            if isinstance(results[t][a], dict):
                results[t][a].pop("_pred", None)
                results[t][a].pop("_fm", None)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("\nwrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
