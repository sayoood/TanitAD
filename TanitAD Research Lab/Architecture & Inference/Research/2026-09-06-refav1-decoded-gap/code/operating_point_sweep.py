#!/usr/bin/env python3
"""D-REFAV1-DK-DECODED -- BAR-2 failed at a single operating point. This maps the
WHOLE sensitivity/specificity trade-off, and asks the question that decides
whether the decoded gap is worth anything HERE:

  ⛔ AT MATCHED SPECIFICITY, DOES THE DECODED GAP BEAT A PREDICTOR WITH NO
     PERCEPTION AT ALL?

The barrier is ``gap < d0 + tau*v0``. With a CONSTANT gap the firing decision
collapses to a SPEED THRESHOLD -- and the violating windows are the fast ones
(16.44 m/s mean against 8.00 in the adequate block). So a constant reproduces a
large share of the oracle's flips using `v0` alone. Every arm below is therefore
swept through the identical procedure, including the constant, and the readable
quantity is the SEPARATION BETWEEN THE CURVES, never a single arm's number.

⛔ THE OFFSET IS FIT ON THE TRAINING FOLDS ONLY. For quantile ``q`` the offset is
the ``q``-quantile of the TRAINING residual ``pred - truth``; subtracting it makes
the estimate under-predict the gap on a fraction ``q`` of training rows, which is
the conservative direction a ONE-SIDED barrier wants. The scored fold is scored,
never consulted for the offset.

⚠️ NO POINT ON THIS CURVE IS "THE ANSWER". Choosing an operating point requires a
safety criterion the programme has not committed to (how many missed
distance-keeping events are worth how many spurious decelerations). Reporting the
curve is the honest output; picking the point that clears a bar after seeing the
data is the goalpost move the rules forbid.
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
    a = ap.parse_args()
    sys.path.insert(0, a.stack)
    from tanitad.refs import refav1_lon_cost as dk

    z = np.load(a.oof, allow_pickle=True)
    cid, frm = z["clip_id"].astype(str), z["frame"].astype(np.int64)
    y = z["y"].astype(np.float64)
    pf, pp = z["pred_field"].astype(np.float64), z["pred_pix"].astype(np.float64)
    fold = z["fold"].astype(np.int64)
    fm = float(z["fit_mean"])

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
    nv, na = int(viol.sum()), int(adeq.sum())
    spec = dk.DistanceKeepingSpec(w_dk=1.0, gap_source=dk.GAP_SOURCE_ORACLE)
    K = 10

    def score(gapmap):
        sens = spc = 0
        for i in np.flatnonzero(is_lead):
            g = gapmap.get((cw[i], int(fw[i])), np.nan)
            if not np.isfinite(g):
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
        return sens / max(nv, 1), spc / max(na, 1), sens, spc

    QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    arms = {"field": pf, "pix_floor": pp, "constant_CONTROL": np.full_like(y, fm)}
    curves = {}
    for name, pred in arms.items():
        rows = []
        for q in QS:
            adj = np.empty_like(pred)
            for f in sorted(set(fold.tolist())):
                tr, te = fold != f, fold == f
                delta = float(np.quantile(pred[tr] - y[tr], q))
                adj[te] = pred[te] - delta
            s, p, sn, pn = score({(c, int(x)): float(v)
                                  for c, x, v in zip(cid, frm, adj)})
            rows.append({"q": q, "sensitivity": s, "specificity": p,
                         "sens_n": sn, "spec_n": pn,
                         "mean_offset_m": float(np.mean(pred - adj))})
            print("  %-18s q=%.2f  offset %+6.2f m  sens %2d/%-2d=%.4f  "
                  "spec %2d/%-2d=%.4f" % (name, q, rows[-1]["mean_offset_m"],
                                          sn, nv, s, pn, na, p), flush=True)
        curves[name] = rows

    # ⭐ THE DECISIVE READ: at MATCHED specificity, is `field` above `constant`?
    matched = []
    for r in curves["field"]:
        best = None
        for c in curves["constant_CONTROL"]:
            if c["specificity"] >= r["specificity"] - 1e-9:
                if best is None or c["sensitivity"] > best["sensitivity"]:
                    best = c
        if best is not None:
            matched.append({"specificity": r["specificity"],
                            "field_sens": r["sensitivity"],
                            "constant_sens_at_ge_spec": best["sensitivity"],
                            "advantage": r["sensitivity"] - best["sensitivity"]})
    res = {"task": "D-REFAV1-DK-DECODED -- operating-point sweep",
           "evidence_class": "MEASURED (ours)",
           "n_violating": nv, "n_adequate": na, "n_lead": int(is_lead.sum()),
           "offset_rule": "q-quantile of the TRAINING-fold residual (pred - truth); "
                          "the scored fold is never consulted",
           "curves": curves, "field_vs_constant_at_matched_specificity": matched,
           "bar2": {"sensitivity_bar": 0.80, "specificity_bar": 0.70},
           "any_point_clears_bar2": {
               k: [r for r in v if r["sensitivity"] >= 0.80
                   and r["specificity"] >= 0.70] for k, v in curves.items()}}
    print("\nfield vs constant at MATCHED specificity:")
    for m in matched:
        print("  spec %.4f : field %.4f vs constant %.4f  -> advantage %+.4f"
              % (m["specificity"], m["field_sens"],
                 m["constant_sens_at_ge_spec"], m["advantage"]))
    print("\npoints clearing BAR-2 (sens>=0.80 AND spec>=0.70):")
    for k, v in res["any_point_clears_bar2"].items():
        print("  %-18s %s" % (k, [("q=%.2f" % r["q"]) for r in v] or "NONE"))
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
