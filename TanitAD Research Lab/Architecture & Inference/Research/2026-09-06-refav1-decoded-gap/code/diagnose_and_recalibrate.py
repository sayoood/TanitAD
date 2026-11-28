#!/usr/bin/env python3
"""D-REFAV1-DK-DECODED -- BAR-2 failed at sensitivity 12/21. This is the
diagnosis and the next lever, run rather than named.

⛔ THE FIRST QUESTION IS NOT "HOW DO I FIX THE HEAD" -- IT IS "WHAT IS THE
FIRE/FLIP METRIC ACTUALLY READING". The raw-pixel floor scores R2 -0.03 (worse
than predicting the mean) and still reproduced 10 of the oracle's 21 flips. A
metric a no-information arm half-passes is measuring something other than the
thing under test.
  ⇒ CONTROL 1: a CONSTANT gap predictor (the fit-fold mean, the same value the
    R2 skill scores against) run through the identical fire/flip pipeline. The
    barrier is `gap < d0 + tau*v0`, so a constant gap fires on every window whose
    SPEED is high enough -- with no perception at all. Whatever the constant
    scores is the floor the decoded head has to be read against.

⭐ THE MECHANISM, MEASURED NOT ASSUMED. A ridge prediction is shrunk toward the
fit mean, so it OVER-predicts small gaps and UNDER-predicts large ones. For a
symmetric error metric that is optimal; for a ONE-SIDED SAFETY BARRIER it is
exactly the wrong direction, because over-predicting a small gap means the term
does not fire where it should. Regressing truth on prediction gives the slope: a
slope > 1 IS shrinkage, and it is measured here before anything is changed.

⭐ THE LEVER: a per-fold linear recalibration `g' = alpha + beta * g_hat` whose
alpha/beta are fit on the TRAINING FOLDS ONLY -- never on the fold being scored.
⛔ This is not a free parameter tuned on the outcome: the recalibration is fit to
the training folds' own gap labels by least squares, the scored fold is untouched,
and the SAME recalibration is applied to the pixel floor so the comparison stays
matched. A lever that helped only because it was tuned on the test set would be
the 2026-08-22 lambda failure wearing a new hat.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-in", default=None)
    ap.add_argument("--head-out", default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.stack)
    from tanitad.refs import refav1_lon_cost as dk

    z = np.load(a.oof, allow_pickle=True)
    cid = z["clip_id"].astype(str)
    frm = z["frame"].astype(np.int64)
    y = z["y"].astype(np.float64)
    pf = z["pred_field"].astype(np.float64)
    pp = z["pred_pix"].astype(np.float64)
    fold = z["fold"].astype(np.int64)
    fm = float(z["fit_mean"])

    # ---- MECHANISM: is the error a SHRINKAGE? ------------------------------
    def slope(pred):
        A = np.stack([np.ones_like(pred), pred], 1)
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(b[0]), float(b[1])

    a0, b1 = slope(pf)
    res = {"task": "D-REFAV1-DK-DECODED -- BAR-2 diagnosis + recalibration lever",
           "evidence_class": "MEASURED (ours)",
           "shrinkage": {
               "regress_truth_on_pred": {"intercept": a0, "slope": b1},
               "reading": ("slope > 1 means the head's spread is SHRUNK toward the "
                           "fit mean, i.e. it over-predicts SMALL gaps -- the "
                           "direction that makes a one-sided barrier under-fire"),
               "pred_std": float(pf.std()), "truth_std": float(y.std()),
               "spread_ratio": float(pf.std() / y.std()),
               "bias_overall_m": float(np.mean(pf - y)),
               "bias_on_small_gaps_m": float(np.mean((pf - y)[y < 30])),
               "bias_on_large_gaps_m": float(np.mean((pf - y)[y >= 30]))}}
    print("shrinkage: truth = %.3f + %.3f * pred ; spread ratio %.3f ; "
          "bias overall %+.2f m, on gaps<30m %+.2f m, on gaps>=30m %+.2f m"
          % (a0, b1, res["shrinkage"]["spread_ratio"],
             res["shrinkage"]["bias_overall_m"],
             res["shrinkage"]["bias_on_small_gaps_m"],
             res["shrinkage"]["bias_on_large_gaps_m"]), flush=True)

    # ---- the LEVER: per-fold recalibration, TRAINING FOLDS ONLY ------------
    recal = {}
    pf_r = np.empty_like(pf)
    pp_r = np.empty_like(pp)
    for f in sorted(set(fold.tolist())):
        tr = fold != f
        te = fold == f
        A = np.stack([np.ones(tr.sum()), pf[tr]], 1)
        c = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pf_r[te] = c[0] + c[1] * pf[te]
        Ap = np.stack([np.ones(tr.sum()), pp[tr]], 1)
        cp = np.linalg.lstsq(Ap, y[tr], rcond=None)[0]
        pp_r[te] = cp[0] + cp[1] * pp[te]
        recal[int(f)] = {"alpha": float(c[0]), "beta": float(c[1]),
                         "alpha_pix": float(cp[0]), "beta_pix": float(cp[1]),
                         "n_fit": int(tr.sum())}
        print("  fold %d recal: g' = %+.3f + %.3f * g_hat  (fit on %d TRAINING "
              "rows, scored fold untouched)" % (f, c[0], c[1], int(tr.sum())),
              flush=True)

    def r2(pred):
        return 1.0 - float(np.sum((y - pred) ** 2)) / float(np.sum((y - fm) ** 2))

    res["recalibration"] = {
        "per_fold": recal,
        "r2_field_before": r2(pf), "r2_field_after": r2(pf_r),
        "r2_pix_before": r2(pp), "r2_pix_after": r2(pp_r),
        "mae_before": float(np.mean(np.abs(pf - y))),
        "mae_after": float(np.mean(np.abs(pf_r - y))),
        "spread_ratio_after": float(pf_r.std() / y.std()),
        "note": ("R2 is expected to move only slightly -- recalibration trades "
                 "squared error for SPREAD, which is what the one-sided barrier "
                 "needs. The readable outcome is BAR-2, not R2.")}
    print("R2 field %.4f -> %.4f ; MAE %.3f -> %.3f ; spread ratio %.3f -> %.3f"
          % (r2(pf), r2(pf_r), res["recalibration"]["mae_before"],
             res["recalibration"]["mae_after"],
             res["shrinkage"]["spread_ratio"],
             res["recalibration"]["spread_ratio_after"]), flush=True)

    # ---- BAR-2, re-scored with every arm INCLUDING the constant control ----
    manifest = json.load(open(os.path.join(a.dump, "manifest.json"),
                              encoding="utf-8"))
    by_fi = {int(e.get("file_index", i)): e for i, e in enumerate(
        manifest.get("episodes") or [])}
    files = sorted(os.path.join(a.dump, x) for x in os.listdir(a.dump)
                   if x.startswith("ep") and x.endswith(".npz"))
    blk = np.load(a.block, allow_pickle=True)
    bidx = {(c, int(fr)): i for i, (c, fr) in enumerate(
        zip(blk["clip_id"].astype(str), blk["frame"].astype(np.int64)))}
    b_gap = blk["gap0_m"].astype(np.float64)
    b_state = blk["state"].astype(str)

    cw, fw, vw = [], [], []
    for fi, f in enumerate(files):
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(int).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float64).reshape(-1)
        clip = (by_fi.get(fi) or {}).get("clip_id") or ""
        for t, v in zip(ws, v0):
            cw.append(str(clip))
            fw.append(2 * int(t))
            vw.append(float(v))
    cw, fw, vw = np.array(cw), np.array(fw, np.int64), np.array(vw)
    W = vw.size
    rr = np.array([bidx.get((c, int(x)), -1) for c, x in zip(cw, fw)])
    hv = rr >= 0
    st = np.array(["NO_LABEL"] * W, dtype=object)
    st[hv] = b_state[rr[hv]]
    go = np.full(W, np.nan)
    go[hv] = b_gap[rr[hv]]
    is_lead = np.isfinite(go) & (st == "LEAD")
    s_star = dk.DK_D0_M + dk.DK_TAU_TARGET_S * vw
    viol = is_lead & (go < s_star)
    adeq = is_lead & ~viol

    spec = dk.DistanceKeepingSpec(w_dk=1.0, gap_source=dk.GAP_SOURCE_ORACLE)
    K = 10

    def fireflip(gapmap, name):
        fires = flips = 0
        sens = spc = 0
        for i in np.flatnonzero(is_lead):
            g = gapmap.get((cw[i], int(fw[i])), np.nan)
            if not np.isfinite(g):
                continue
            g = max(0.0, float(g))
            fire = g < s_star[i]
            c0 = torch.zeros(1, K, 2, dtype=torch.float64)
            c1 = c0.clone()
            c1[..., 0] = -1.5
            flip = float(dk.distance_keeping_cost(
                c1, v0=float(vw[i]), gap0_m=g, dt=0.2, spec=spec)) < \
                float(dk.distance_keeping_cost(
                    c0, v0=float(vw[i]), gap0_m=g, dt=0.2, spec=spec))
            fires += int(fire)
            flips += int(fire and flip)
            if viol[i] and fire and flip:
                sens += 1
            if adeq[i] and not fire:
                spc += 1
        return {"arm": name, "n_fires": fires, "n_flips": flips,
                "sensitivity_n": sens, "sensitivity": sens / max(viol.sum(), 1),
                "specificity_n": spc, "specificity": spc / max(adeq.sum(), 1)}

    def mk(pred):
        return {(c, int(x)): float(p) for c, x, p in zip(cid, frm, pred)}

    arms = [
        ("oracle", {(c, int(x)): float(g) for c, x, g in
                    zip(cw, fw, go) if np.isfinite(g)}),
        ("constant_CONTROL", {(c, int(x)): fm for c, x in zip(cid, frm)}),
        ("pix_floor", mk(pp)),
        ("pix_floor_recal", mk(pp_r)),
        ("field", mk(pf)),
        ("field_recal", mk(pf_r)),
    ]
    table = [fireflip(m, n) for n, m in arms]
    res["BAR2_table"] = {"n_violating": int(viol.sum()),
                         "n_adequate": int(adeq.sum()),
                         "n_lead": int(is_lead.sum()), "rows": table}
    print("\n%-18s %6s %6s   %-14s %-14s" %
          ("arm", "fires", "flips", "SENSITIVITY", "SPECIFICITY"))
    for r in table:
        print("%-18s %6d %6d   %2d/%-2d = %.4f   %2d/%-2d = %.4f"
              % (r["arm"], r["n_fires"], r["n_flips"], r["sensitivity_n"],
                 int(viol.sum()), r["sensitivity"], r["specificity_n"],
                 int(adeq.sum()), r["specificity"]))

    best = [r for r in table if r["arm"] == "field_recal"][0]
    res["BAR2_after_lever"] = {
        "sensitivity": best["sensitivity"], "bar": 0.80,
        "specificity": best["specificity"], "bar_spec": 0.70,
        "PASS": bool(best["sensitivity"] >= 0.80 and best["specificity"] >= 0.70)}
    print("\nBAR-2 after the recalibration lever: %s"
          % ("PASS" if res["BAR2_after_lever"]["PASS"] else "FAIL"))

    # ---- fold the recalibration INTO the deployable head -------------------
    if a.head_in and a.head_out:
        b = torch.load(a.head_in, map_location="cpu", weights_only=False)
        tgt = b["targets"]["gap_all_lead"]["by_fold"]
        for f, h in tgt.items():
            r = recal[int(f)]
            # g' = alpha + beta*((x*V).sum()+bias) = (x*(beta*V)).sum()
            #      + (alpha + beta*bias)  -- still ONE dot product.
            h["V"] = h["V"] * float(r["beta"])
            h["bias"] = float(r["alpha"] + r["beta"] * h["bias"])
            h["recalibration"] = {"alpha": r["alpha"], "beta": r["beta"],
                                  "fit": "TRAINING FOLDS ONLY"}
        b["version"] = b["version"] + "+recal"
        torch.save(b, a.head_out)
        import hashlib
        res["recalibrated_bundle"] = {
            "path": a.head_out,
            "sha256": hashlib.sha256(open(a.head_out, "rb").read()).hexdigest(),
            "version": b["version"]}
        print("recalibrated bundle -> %s" % a.head_out)

    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
