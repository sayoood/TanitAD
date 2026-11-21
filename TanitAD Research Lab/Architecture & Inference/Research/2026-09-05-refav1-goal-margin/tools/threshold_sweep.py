#!/usr/bin/env python
"""THE HEAD'S SEPARABILITY IS A FUNCTION OF WHAT COUNTS AS A TURN.

⭐ WHY. Every banked refav1 turn number uses `|gt_kappa| > 1e-3` as "GT turn".
That is a **1000 m radius** -- barely a curve -- while the vocabulary's `TURN`
token commands `GOAL_KAPPA_TURN = 0.08` (**R = 12.5 m**, `refa_v1.py:119`). A
score judged on separating a 1000 m curve from a straight road is being judged on
nearly-noise, and its AUC will understate what the head can do on turns that
matter. This sweeps the definition and reports the whole family, so no single
threshold can be quoted without its neighbours.

⛔ CONTROLS:
  * a SHUFFLED-LABEL AUC at every threshold, which must read ~0.5 -- a rising
    AUC could otherwise be an artefact of a shrinking positive class;
  * `n_turn` printed at every threshold, and a row with `n_turn < 10` marked
    UNDERPOWERED so it cannot be quoted as a result;
  * the `1e-3` row must reproduce the banked AUC 0.7120 / recall 0.2045 /
    false 0.0733 exactly, or this is not the banked score.

Run: python gm_threshold_sweep.py --logits <npz> --gt-kappa <npz> --out <json>
     python gm_threshold_sweep.py --intent <intent npz> --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def roc_parts(score, lab):
    o = np.argsort(-score)
    ll = lab[o]
    tp = np.cumsum(ll); fp = np.cumsum(1 - ll)
    P, N = ll.sum(), (1 - ll).sum()
    if P == 0 or N == 0:
        return None
    tpr, fpr = tp / P, fp / N
    auc = float(np.trapezoid(np.concatenate([[0], tpr]),
                             np.concatenate([[0], fpr])))
    bal = 0.5 * (tpr + (1 - fpr))
    return auc, float(bal.max()), tpr, fpr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", default=None)
    ap.add_argument("--gt-kappa", default=None)
    ap.add_argument("--intent", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.intent:
        Z = np.load(a.intent, allow_pickle=False)
        gt = Z["gt_kappa"].astype(np.float64)
        lg = Z["lat_logits"].astype(np.float64)
        names = [str(x) for x in Z["lat_names"]]
        src = a.intent
    else:
        Z = np.load(a.logits, allow_pickle=False)
        G = np.load(a.gt_kappa, allow_pickle=False)
        names = [str(x) for x in Z["lat_names"]]
        key = {(int(c), int(w)): i for i, (c, w) in
               enumerate(zip(Z["clip_index"], Z["ws"]))}
        g = np.full(len(Z["ws"]), np.nan)
        for c, w, kk in zip(G["clip_index"], G["ws"], G["gt_kappa"]):
            i = key.get((int(c), int(w)))
            if i is not None:
                g[i] = kk
        m = np.isfinite(g)
        gt = g[m]
        lg = Z["lat_logits"].astype(np.float64)[m]
        src = a.logits

    LK = names.index("LANE_KEEP")
    CURV = [i for i, t in enumerate(names)
            if t.startswith(("TURN_", "NUDGE_", "LANE_CHANGE_"))]
    TURN = [names.index("TURN_L"), names.index("TURN_R")]
    s = lg[:, CURV].max(1) - lg[:, LK]
    sT = lg[:, TURN].max(1) - lg[:, LK]
    dec = lg.argmax(-1)
    prop = np.isin(dec, CURV)
    rng = np.random.default_rng(a.seed)
    n = len(gt)

    rows = []
    for thr in (1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 8e-2):
        y = (np.abs(gt) > thr).astype(int)
        if y.sum() < 2 or (1 - y).sum() < 2:
            continue
        r = roc_parts(s, y)
        rT = roc_parts(sT, y)
        shuf = [roc_parts(s, rng.permutation(y))[0] for _ in range(100)]
        rows.append({
            "kappa_turn_threshold": thr,
            "radius_m": round(1.0 / thr, 1),
            "n_turn": int(y.sum()), "n_straight": int((1 - y).sum()),
            "base_rate": float(y.mean()),
            "UNDERPOWERED_n_turn_lt_10": bool(y.sum() < 10),
            "auc_curv_score": r[0],
            "auc_TURN_only_score": rT[0],
            "best_balanced_acc": r[1],
            "argmax_turn_recall": float(prop[y == 1].mean()),
            "argmax_false_turn_on_straight": float(prop[y == 0].mean()),
            "CONTROL_shuffled_auc_mean": float(np.mean(shuf)),
            "CONTROL_shuffled_auc_sd": float(np.std(shuf)),
            "CONTROL_shuffled_passes": bool(abs(np.mean(shuf) - 0.5) < 0.06),
        })

    base = [r for r in rows if r["kappa_turn_threshold"] == 1e-3]
    ctrl = {"CONTROL_reproduces_banked_at_1e-3": None}
    if base:
        b = base[0]
        ctrl["CONTROL_reproduces_banked_at_1e-3"] = {
            "auc": b["auc_curv_score"], "expect": 0.712020202020202,
            "recall": b["argmax_turn_recall"], "expect_recall": 0.20454545454545456,
            "false": b["argmax_false_turn_on_straight"],
            "expect_false": 0.07333333333333333,
            "passes": bool(abs(b["auc_curv_score"] - 0.712020202020202) < 1e-9
                           and abs(b["argmax_turn_recall"] - 0.20454545) < 1e-6)}

    out = {"source": src, "n_windows": int(n),
           "note": ("GOAL_KAPPA_TURN = 0.08 (R 12.5 m) is the curvature the "
                    "TURN token actually commands; 1e-3 is R 1000 m. A single "
                    "threshold is not quotable without this family."),
           "rows": rows, "controls": ctrl}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"n={n}  source={src}")
    print(f"{'thr':<8}{'R(m)':>8}{'n_turn':>8}{'AUC':>9}{'AUC_T':>9}"
          f"{'bestbal':>9}{'am_rec':>9}{'am_false':>10}{'shufAUC':>9}")
    for r in rows:
        print(f"{r['kappa_turn_threshold']:<8.0e}{r['radius_m']:>8.1f}"
              f"{r['n_turn']:>8}{r['auc_curv_score']:>9.4f}"
              f"{r['auc_TURN_only_score']:>9.4f}{r['best_balanced_acc']:>9.4f}"
              f"{r['argmax_turn_recall']:>9.4f}"
              f"{r['argmax_false_turn_on_straight']:>10.4f}"
              f"{r['CONTROL_shuffled_auc_mean']:>9.4f}"
              + ("  UNDERPOWERED" if r["UNDERPOWERED_n_turn_lt_10"] else ""))
    print("CONTROL banked-at-1e-3:", json.dumps(ctrl))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
