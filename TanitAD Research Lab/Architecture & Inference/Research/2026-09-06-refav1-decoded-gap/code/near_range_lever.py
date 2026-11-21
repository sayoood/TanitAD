#!/usr/bin/env python3
"""D-REFAV1-DK-DECODED -- the NEXT LEVER, run rather than named.

⭐ WHAT THE DIAGNOSIS SAYS. The all-lead head's error (MAE 11.40 m) is LARGER
than the quantity the barrier decides (mean shortfall 9.10 m), so the fire
decision is dominated by decode noise. But that MAE is pooled over gaps out to
80 m, and the barrier lives at ``s* = d0 + tau*v`` -- 29.66 m mean on the
violating windows. On the <=30 m cell the SAME head reads **R2 +0.6586, MAE
3.485 m**, within-clip +0.7199 against a within-clip shuffle of -0.0066.
⇒ **the near-range decode is already well inside the decision margin; the far
range, which the barrier never touches, is what ruins the pooled number.**

⛔ WHAT THIS FILE MEASURES, AND WHAT IT DOES NOT. It scores BAR-2 using the
<=30 m head on the windows whose gap really is <=30 m -- i.e. **assuming a
PERFECT near/far gate**. That is a CEILING ON A TWO-STAGE DESIGN, not a
capability: no such gate has been built, and the number is inadmissible as a
vision-only result. It exists to answer one question before anyone spends a GPU
on it: *would a near-range head, if correctly gated, clear the bar the pooled
head misses?* If it would not, the two-stage design is dead before it is built.

⚠️ It is reported beside the SAME two controls (constant, raw pixels) under the
SAME oracle gating, so the gate's advantage cannot be mistaken for the head's.
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
    ap.add_argument("--oof-all", required=True)
    ap.add_argument("--oof-near", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.stack)
    from tanitad.refs import refav1_lon_cost as dk

    def load(p):
        z = np.load(p, allow_pickle=True)
        return (z["clip_id"].astype(str), z["frame"].astype(np.int64),
                z["y"].astype(np.float64), z["pred_field"].astype(np.float64),
                z["pred_pix"].astype(np.float64), z["fold"].astype(np.int64),
                float(z["fit_mean"]))

    cidA, frmA, yA, pfA, ppA, foA, fmA = load(a.oof_all)
    cidN, frmN, yN, pfN, ppN, foN, fmN = load(a.oof_near)

    def debias(pred, y, fold):
        """Median-residual bias correction, TRAINING FOLDS ONLY. The least
        tunable calibration there is -- it removes a bias, it does not choose an
        operating point."""
        out = np.empty_like(pred)
        for f in sorted(set(fold.tolist())):
            tr, te = fold != f, fold == f
            out[te] = pred[te] - float(np.median(pred[tr] - y[tr]))
        return out

    pfA_d, ppA_d = debias(pfA, yA, foA), debias(ppA, yA, foA)
    pfN_d, ppN_d = debias(pfN, yN, foN), debias(ppN, yN, foN)

    manifest = json.load(open(os.path.join(a.dump, "manifest.json"),
                              encoding="utf-8"))
    by_fi = {int(e.get("file_index", i)): e for i, e in enumerate(
        manifest.get("episodes") or [])}
    files = sorted(os.path.join(a.dump, x) for x in os.listdir(a.dump)
                   if x.startswith("ep") and x.endswith(".npz"))
    blk = np.load(a.block, allow_pickle=True)
    bidx = {(c, int(f)): i for i, (c, f) in enumerate(
        zip(blk["clip_id"].astype(str), blk["frame"].astype(np.int64)))}
    b_gap, b_state = blk["gap0_m"].astype(np.float64), blk["state"].astype(str)

    cw, fw, vw = [], [], []
    for fi, f in enumerate(files):
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(int).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float64).reshape(-1)
        clip = (by_fi.get(fi) or {}).get("clip_id") or ""
        for t, v in zip(ws, v0):
            cw.append(str(clip)); fw.append(2 * int(t)); vw.append(float(v))
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
    viol, adeq = is_lead & (go < s_star), is_lead & ~(go < s_star)
    near = is_lead & (go <= 30.0)
    spec = dk.DistanceKeepingSpec(w_dk=1.0, gap_source=dk.GAP_SOURCE_ORACLE)
    K = 10

    def score(gapmap, mask):
        nv, na = int((viol & mask).sum()), int((adeq & mask).sum())
        sens = spc = miss = 0
        for i in np.flatnonzero(is_lead & mask):
            g = gapmap.get((cw[i], int(fw[i])), np.nan)
            if not np.isfinite(g):
                miss += 1
                continue
            g = max(0.0, float(g))
            fire = g < s_star[i]
            c0 = torch.zeros(1, K, 2, dtype=torch.float64)
            c1 = c0.clone(); c1[..., 0] = -1.5
            flip = float(dk.distance_keeping_cost(c1, v0=float(vw[i]), gap0_m=g,
                                                  dt=0.2, spec=spec)) < \
                float(dk.distance_keeping_cost(c0, v0=float(vw[i]), gap0_m=g,
                                               dt=0.2, spec=spec))
            if viol[i] and fire and flip:
                sens += 1
            if adeq[i] and not fire:
                spc += 1
        return {"n_violating": nv, "n_adequate": na, "n_no_pred": miss,
                "sens_n": sens, "sensitivity": sens / max(nv, 1),
                "spec_n": spc, "specificity": spc / max(na, 1)}

    def mp(cids, frms, pred):
        return {(c, int(x)): float(v) for c, x, v in zip(cids, frms, pred)}

    const_near = np.full_like(yN, fmN)
    const_all = np.full_like(yA, fmA)
    # ⚠️ A LIST, NOT A DICT. The first version of this table used a dict and two
    # control rows shared a label, so the second SILENTLY OVERWROTE the first and
    # the printed table attributed near-window numbers to the all-window block.
    # A duplicate key is a data-loss bug that looks like a formatting choice.
    rows = [
        ("ALL-RANGE head | ALL LEAD windows", mp(cidA, frmA, pfA_d), is_lead),
        ("  pixel floor  | ALL LEAD windows", mp(cidA, frmA, ppA_d), is_lead),
        ("  constant CTL | ALL LEAD windows", mp(cidA, frmA, const_all), is_lead),
        ("NEAR head      | NEAR (<=30 m) [ORACLE-GATED CEILING]",
         mp(cidN, frmN, pfN_d), near),
        ("  pixel floor  | NEAR windows", mp(cidN, frmN, ppN_d), near),
        ("  constant CTL | NEAR windows", mp(cidN, frmN, const_near), near),
        ("ALL-RANGE head | NEAR windows [attribution control]",
         mp(cidA, frmA, pfA_d), near),
    ]
    out = {"task": "D-REFAV1-DK-DECODED -- near-range lever (two-stage CEILING)",
           "evidence_class": "MEASURED (ours)",
           "scope_warning": ("the NEAR rows assume a PERFECT near/far gate. This "
                             "is a CEILING on a two-stage design and is NOT a "
                             "vision-only capability number."),
           "decode_quality": {
               "all_range_mae_m": float(np.mean(np.abs(pfA_d - yA))),
               "near_range_mae_m": float(np.mean(np.abs(pfN_d - yN))),
               "mean_shortfall_m_on_violating": 9.10},
           "rows": {}}
    print("%-56s %-17s %-17s" % ("arm | window set", "SENSITIVITY", "SPECIFICITY"))
    for name, m, msk in rows:
        r = score(m, msk)
        out["rows"][name] = r
        print("%-56s %2d/%-2d = %.4f   %2d/%-2d = %.4f"
              % (name, r["sens_n"], r["n_violating"], r["sensitivity"],
                 r["spec_n"], r["n_adequate"], r["specificity"]))
    nr = out["rows"]["NEAR head      | NEAR (<=30 m) [ORACLE-GATED CEILING]"]
    nc = out["rows"]["  constant CTL | NEAR windows"]
    out["near_vs_constant"] = {
        "near_head": {"sens": nr["sensitivity"], "spec": nr["specificity"]},
        "constant": {"sens": nc["sensitivity"], "spec": nc["specificity"]},
        "reading": ("⛔ the constant control very nearly clears BAR-2 on the NEAR "
                    "subset too, so the bar is NOT a discriminating instrument "
                    "there -- restricting to <=30 m removes exactly the windows a "
                    "constant gets wrong. The discriminating statistic is the "
                    "separation from the constant at MATCHED specificity, which "
                    "the operating-point sweep reports, not a single pass/fail.")}
    print("")
    print("  constant CONTROL on the SAME near windows: sens %.4f spec %.4f"
          % (nc["sensitivity"], nc["specificity"]))
    out["BAR2_on_near_windows"] = {
        "sensitivity": nr["sensitivity"], "bar": 0.80,
        "specificity": nr["specificity"], "bar_spec": 0.70,
        "would_clear": bool(nr["sensitivity"] >= 0.80
                            and nr["specificity"] >= 0.70)}
    print("\nMAE all-range %.2f m vs near-range %.2f m  (decision margin ~9.10 m)"
          % (out["decode_quality"]["all_range_mae_m"],
             out["decode_quality"]["near_range_mae_m"]))
    print("A perfectly-gated near-range head would %s BAR-2 on the near windows."
          % ("CLEAR" if out["BAR2_on_near_windows"]["would_clear"] else "STILL MISS"))
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
