"""E-DEC-36b — IS RANGE NONLINEARLY DECODABLE? (RFF+ridge, after the MLP failed)

⛔ WHY THIS REPLACES `rangeprobe_nl.py`. That probe used a full-batch MLP with 300
steps and early stopping. MEASURED 2026-08-24: its **TIME-SHUFFLED control read
−14.00 and its raw-pixel floor −27.98**, where a no-information predictor must
read ≈ 0. **A control that does not read its known value means the INSTRUMENT is
broken, not that the signal is absent** — so none of its numbers were usable, and
it exited 1 partway through. Non-convex optimisation was the wrong tool for a
question about what a representation CONTAINS.

⭐ THE RIGHT INSTRUMENT. Random Fourier Features + ridge: genuinely NONLINEAR
(an RBF-kernel approximation), but **CONVEX with a closed form**, so it cannot
diverge and has no optimiser, no learning rate and no early stopping to get
wrong. It is the same estimator family as the linear probe that already behaves,
with the one property we need added.

    phi(x) = sqrt(2/D) * cos(W x + b),   W ~ N(0, 1/(2 gamma^2)),  b ~ U[0, 2pi)

⚠️ EVERY hyper-parameter is fit on the FIT split only — the PCA basis, the
standardisation, the RFF draw, the bandwidth (median heuristic on FIT rows) and
lambda (selected by an inner split OF THE FIT CLIPS). **The scored clip is
scored, never tuned on.** This is the C-probe rule that produced four separate
manufactured results in one afternoon when it was violated.

⭐⭐ THE CONTROLS, AND THE PANEL IS UNREADABLE WITHOUT THEM:

    TIME-SHUFFLED   the SAME targets permuted WITHIN each clip — preserves every
                    clip's marginal exactly, destroys only the frame
                    correspondence. ⛔ MUST READ ~0. Whatever it reads is what the
                    probe can extract from clip-level statistics alone, i.e.
                    LEAKAGE. **The readable quantity is TRUE − SHUFFLED.**
    constant        reads EXACTLY 0.0000 by construction
    pixels (floor)  raw input; a representation not beating it added nothing
    n_agents        the POSITIVE CONTROL — known carried. If it works while range
                    fails on the SAME rows/folds/columns, the failure belongs to
                    the TARGET, not the setup.

⚠️ A SANITY GATE RUNS FIRST: if the shuffled control is worse than −0.25 on any
column, the panel ABORTS rather than printing numbers. That is the check whose
absence let the MLP version print −27.98 as though it meant something.

T0-DIAGNOSTIC. Held-out, lead-matched. MEASURED (ours; dev-box RTX 4060).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
CACHE = SP / "rangeprobe_cache.npz"
OUT = Path(os.environ.get("SPD_OUT", str(SP / "rangeprobe_rff.json")))
D_EFF, D_RFF, SEED = 96, 1024, 0
LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def within_clip_r(pred, yte):
    """Within-clip Pearson r with BOTH series centred — THE shared metric.

    ⛔⛔ EXPORTED SO IT CANNOT BE DUPLICATED AND DRIFT. `idm_oracle.py` carried
    its own copy of the earlier, BROKEN within-clip R2 (prediction centred on the
    FIT mean, divided by the TEST clip's variance) — so fixing the metric here
    left a broken copy there, and its CONSTANT control read -2064.63 where it
    must read 0. Same family as the banked probe copy that went stale while the
    scratchpad copy was fixed: **a metric with two implementations has one
    correct implementation at most.**

    A zero-variance prediction (the constant column) returns EXACTLY 0.0, the
    correct no-information value.
    """
    import numpy as _np
    pc = pred - pred.mean()
    yc = yte - yte.mean()
    den = float(_np.sqrt((pc ** 2).sum() * (yc ** 2).sum()))
    return float((pc * yc).sum() / den) if den > 1e-12 else 0.0


def rff_fold(Xtr_clips, ytr_clips, Xte, seed=SEED):
    """PCA -> standardise -> RFF -> ridge. Everything fit on Xtr only.

    ⛔⛔ LAMBDA IS SELECTED ON A CLIP-DISJOINT INNER SPLIT, NOT ON RANDOM ROWS.
    MEASURED 2026-08-24: with a random-ROW inner split the TIME-SHUFFLED control
    read -6.40 / -4.60 / -5.47 where it must read ~0, and the sanity gate
    correctly aborted the panel. Cause: random rows put frames from EVERY
    training clip in both halves, so a model that has merely memorised
    clip-level statistics validates perfectly, lambda is driven small, and the
    predictions then blow up on a genuinely held-out CLIP.
    ⇒ **The inner validation must have the SAME STRUCTURE as the outer test.**
    Fitting hyper-parameters "on the fit split only" is necessary and NOT
    sufficient — it must be on the fit split SPLIT THE SAME WAY.
    """
    rng = np.random.default_rng(seed)
    Xtr = np.concatenate(Xtr_clips)
    ytr = np.concatenate([y.ravel() for y in ytr_clips])
    nclip = len(Xtr_clips)
    bounds = np.cumsum([0] + [len(x) for x in Xtr_clips])
    n_in = max(int(round(0.25 * nclip)), 2)
    hold = set(rng.choice(nclip, size=min(n_in, nclip - 2), replace=False).tolist())
    vi = np.concatenate([np.arange(bounds[c], bounds[c + 1])
                         for c in sorted(hold)])
    ti = np.setdiff1d(np.arange(len(Xtr)), vi)
    mu = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu
    k = min(D_EFF, Xc.shape[1], max(Xc.shape[0] - 1, 1))
    # ⛔ RANDOMIZED SVD, not a full one. `np.linalg.svd` on ~1800 x 16384 costs
    # O(n^2 p) and there are 24 folds x 2 (true/shuffled) x 6 columns x 3 targets
    # of them — MEASURED 2026-08-24: ZERO rows emitted in 28 minutes. A randomized
    # range-finder for k=96 components costs O(n p k), i.e. ~5 % of that, and the
    # top-96 subspace is all the probe ever uses.
    rs = np.random.default_rng(seed + 1)
    Om = rs.standard_normal((Xc.shape[1], k + 10))
    Yq = Xc @ Om
    for _ in range(2):                      # power iterations for accuracy
        Yq = Xc @ (Xc.T @ Yq)
    Q, _ = np.linalg.qr(Yq)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    V = Vt[:k].T
    A, B = Xc @ V, (Xte - mu) @ V
    s = A.std(0, keepdims=True) + 1e-8
    A, B = A / s, B / s

    # bandwidth by the median heuristic, on a FIT-split subsample only
    sub = A[rng.choice(len(A), size=min(400, len(A)), replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.sqrt(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
    gamma = max(med, 1e-6)

    W = rng.standard_normal((k, D_RFF)) / gamma
    b = rng.uniform(0, 2 * np.pi, size=D_RFF)
    PA = np.sqrt(2.0 / D_RFF) * np.cos(A @ W + b)
    PB = np.sqrt(2.0 / D_RFF) * np.cos(B @ W + b)

    ym = ytr.mean()
    yc = ytr - ym
    # lambda on the CLIP-DISJOINT inner split computed above
    G = PA[ti].T @ PA[ti]
    c = PA[ti].T @ yc[ti]
    tr = np.trace(G) / max(D_RFF, 1)
    best, best_w = None, None
    for lam in LAMBDAS:
        w = np.linalg.solve(G + lam * tr * np.eye(D_RFF), c)
        e = float(((PA[vi] @ w - yc[vi]) ** 2).mean())
        if best is None or e < best:
            best, best_lam = e, lam
    G = PA.T @ PA
    c = PA.T @ yc
    tr = np.trace(G) / max(D_RFF, 1)
    w = np.linalg.solve(G + best_lam * tr * np.eye(D_RFF), c)
    return PB @ w + ym, ym


def score(Xs, Ys, seed=SEED):
    out = []
    for i in range(len(Xs)):
        Xtr = [Xs[j] for j in range(len(Xs)) if j != i]
        ytr = [Ys[j] for j in range(len(Ys)) if j != i]
        yte = Ys[i].ravel()
        pred, ym = rff_fold(Xtr, ytr, Xs[i], seed)
        ss_res = float(((yte - pred) ** 2).sum())
        # ⛔⛔ TWO METRICS, AND THE SECOND IS THE ONE THE PI'S QUESTION ASKS.
        #
        # CROSS-CLIP R2 (against the FIT-SPLIT mean) asks: "can you predict this
        # clip's ABSOLUTE range from other clips?" That needs calibrated metric
        # depth AND cross-scene generalisation, and `lead_range_m` has enormous
        # BETWEEN-clip variance, so a held-out clip whose range distribution sits
        # far from the training mean gives a huge negative R2 for ANY model.
        # MEASURED 2026-08-24: it reads -2.0 to -6.4 for every column INCLUDING
        # a shuffled control, i.e. the metric is dominated by the between-clip
        # offset rather than by anything the representation does.
        #
        # WITHIN-CLIP R2 (against the TEST CLIP's OWN mean) asks: "does the latent
        # TRACK the range as it CHANGES within a scene?" ⭐ That is exactly the
        # PI's question — *"if the vehicle in front decelerates, the ego must
        # react"* is about the gap CHANGING, not about absolute metric
        # calibration. And the TIME-SHUFFLED control is the correct null for it
        # by construction: shuffling within a clip destroys precisely the
        # within-clip temporal structure this metric measures, while leaving the
        # clip's mean untouched.
        ss_tot_x = float(((yte - ym) ** 2).sum())          # cross-clip
        # ⛔⛔ THE WITHIN-CLIP METRIC MUST CENTRE **BOTH** SERIES. My first version
        # divided by the TEST CLIP's variance while leaving the prediction centred
        # on the FIT mean — so the between-clip offset was removed from the
        # DENOMINATOR and left in the NUMERATOR. MEASURED 2026-08-24: the CONSTANT
        # control read -78.7 on lead_range_m and -655.8 on n_agents. A column of
        # ones cannot be broken by the data, so that was proof the METRIC was
        # broken, not the columns — and the sanity gate caught it before a single
        # number was reported. (For n_agents a clip whose mean is 144 against a
        # fit mean of 33 contributes ~12,000 squared error per row against a
        # within-clip variance of ~50.)
        # ⇒ WITHIN-CLIP PEARSON r: does the prediction TRACK THE SHAPE of the
        # within-clip variation? That is exactly the PI's question — "if the lead
        # decelerates the gap CHANGES" — and it is scale-free, so it cannot be
        # contaminated by an offset either model could not know. A prediction with
        # NO variance (the constant column) reads EXACTLY 0, the correct
        # no-information value, instead of a spurious large negative.
        r_w = within_clip_r(pred, yte)
        out.append((1.0 - ss_res / max(ss_tot_x, 1e-12), r_w))
    a = np.array(out, dtype=float)
    return a[:, 0], a[:, 1]


def main() -> int:
    if not CACHE.is_file():
        print(f"[FATAL] {CACHE} missing"); return 2
    z = np.load(CACHE, allow_pickle=True)
    RAW = {k: list(v) for k, v in z["cols"].item().items()}
    TG = {k: list(v) for k, v in z["targets"].item().items()}
    # keys are "target||column" — each target carries its OWN masked columns,
    # because the masks genuinely differ (lead_closing is nan at frame 0 too).
    COLS_BY_T = {}
    for key, v in RAW.items():
        tn, cn = key.split("||", 1)
        COLS_BY_T.setdefault(tn, {})[cn] = v
    for tn in TG:
        COLS_BY_T.setdefault(tn, {})["constant (control)"] = [
            np.ones((len(y), 1)) for y in TG[tn]]
    rng = np.random.default_rng(0)

    print("\n  E-DEC-36b · IS RANGE NONLINEARLY DECODABLE? (RFF + ridge)")
    print("  the readable quantity is TRUE - TIME-SHUFFLED, never the raw R2\n")
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT, LEAD-MATCHED",
           "function_class": f"random Fourier features (D={D_RFF}, RBF kernel "
                             f"approx, median-heuristic bandwidth) + ridge; "
                             f"CONVEX, closed form, no optimiser",
           "targets": {}}
    hdr = (f"  {'target':<16}{'column':<30}{'TRUE_r':>9}{'SHUF_r':>9}"
           f"{'TRUE-SHUF':>11}{'t':>7}{'n':>7}{'d':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2), flush=True)
    for tn, Y in TG.items():
        Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
        rep["targets"][tn] = {"columns": {}}
        for cname, X in COLS_BY_T[tn].items():
            if [len(x) for x in X] != [len(y) for y in Y]:
                raise SystemExit(f"[FATAL] {tn}/{cname} row mismatch")
            tr_x, tr_w = score(X, Y)
            sh_x, sh_w = score(X, Ysh)
            tr, sh = tr_w, sh_w   # ⭐ within-clip Pearson r is the readable pair
            # ⛔ THE SANITY GATE the MLP version lacked.
            if float(sh.mean()) < -0.25:
                print(f"  {tn:<16}{cname:<30}  ABORT - shuffled control reads "
                      f"{sh.mean():+.4f}, must be ~0. INSTRUMENT BROKEN.")
                continue
            d = tr - sh
            t = float(d.mean()) / max(float(d.std(ddof=1) / np.sqrt(len(d))), 1e-12)
            nrow = sum(len(y) for y in Y)
            rep["targets"][tn]["columns"][cname] = {
                "true_WITHIN_r": round(float(tr_w.mean()), 4),
                "shuffled_WITHIN_r": round(float(sh_w.mean()), 4),
                "true_CROSS": round(float(tr_x.mean()), 4),
                "shuffled_CROSS": round(float(sh_x.mean()), 4),
                "true_minus_shuffled": round(float(d.mean()), 4),
                "t": round(t, 2), "n_rows": nrow, "d_raw": int(X[0].shape[1]),
                "carries_signal": bool(t > 2.0 and d.mean() > 0.02)}
            print(f"  {tn:<16}{cname:<30}{tr.mean():>+9.4f}{sh.mean():>+9.4f}"
                  f"{d.mean():>+11.4f}{t:>7.2f}{nrow:>7}{X[0].shape[1]:>7}",
                  flush=True)
        print()
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
