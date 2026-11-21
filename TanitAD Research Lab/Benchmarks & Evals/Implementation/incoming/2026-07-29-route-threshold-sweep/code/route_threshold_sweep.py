"""TASK #43 STEP 1 — sweep the route decision threshold OFFLINE. No GPU, no re-run.

THE HYPOTHESIS, pre-registered before looking at any swept number:
  The produced route is NOT a classifier. `goal_modes.scalars_to_goal` computes
      graded = tanh(curv_5s / CURV_TURN_PER_M)        # CURV_TURN_PER_M = 1/60 m
      route  = STRAIGHT ; graded >= thr -> LEFT ; graded <= -thr -> RIGHT
  with `thr = tanh(1.0) = 0.76159` HARD-CODED. `curv_5s` is the worst-fit scalar
  (R² 0.3142, RMSE 0.0123 /m), and a regressor that weak SHRINKS magnitudes toward the
  mean by roughly rho = sqrt(R²) ~ 0.56. So a turn must be ~1/0.56 = 1.78x sharper than
  the label's own definition before the threshold fires, and gentler turns round to
  STRAIGHT. Measured consequence: right recall 4.1 %, left recall 23.1 %, but left
  PRECISION 0.9074 — the head knows and will not commit.

BOTH OUTCOMES COMMITTED IN ADVANCE:
  * OUTCOME A (hypothesis supported): balanced accuracy rises materially as thr falls
    toward ~tanh(0.56) ~ 0.508, driven by turn RECALL, at a cost in turn PRECISION.
    => the collapse is a calibration artifact and the one-line fix is real.
  * OUTCOME B (refuted): balanced accuracy is flat or falls at every thr. Then the
    ordering of `curv_5s` carries too little signal for ANY threshold to recover turns,
    the shrinkage story is wrong, and the fix must be the regressor itself (step 2),
    not the threshold.
  Either way the number is reported. A flat curve is a real result, not a failed run.

PRIMARY = BALANCED ACCURACY (mean per-class recall over left/straight/right).
Plain accuracy is NOT the primary and must not be: it REWARDS the collapse, so a
threshold chosen to maximise it can "win" by predicting straight even harder.
"""
from __future__ import annotations

import json
import math
import sys

import torch

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/workspace/v4gate30k/goalagree_v4fs-30k-produced.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/v4gate30k/route_threshold_sweep.json"

CURV_TURN_PER_M = 1.0 / 60.0          # refb_labels.py:275
LEFT, STRAIGHT, RIGHT = 0, 1, 2       # refb_labels.py:76
CURRENT_THR = math.tanh(1.0)          # 0.76159...


def classify(graded: torch.Tensor, thr: float) -> torch.Tensor:
    r = torch.full_like(graded, float(STRAIGHT))
    r = torch.where(graded >= thr, torch.full_like(graded, float(LEFT)), r)
    r = torch.where(graded <= -thr, torch.full_like(graded, float(RIGHT)), r)
    return r.long()


def scores(pred: torch.Tensor, true: torch.Tensor) -> dict:
    per = {}
    recalls = []
    for c, name in ((LEFT, "left"), (STRAIGHT, "straight"), (RIGHT, "right")):
        n = int((true == c).sum())
        tp = int(((true == c) & (pred == c)).sum())
        pn = int((pred == c).sum())
        rec = tp / n if n else float("nan")
        recalls.append(rec)
        per[name] = {"n": n, "tp": tp, "pred_n": pn,
                     "recall": round(rec, 4),
                     "precision": round(tp / pn, 4) if pn else None}
    acc = int((pred == true).sum()) / int(true.numel())
    return {"balanced_accuracy": round(sum(recalls) / 3, 4),
            "plain_accuracy": round(acc, 4), "per_class": per}


def main() -> int:
    d = torch.load(DUMP, map_location="cpu", weights_only=False)
    sc, ro = d["sc_pred"], d["route_oracle"]
    if sc is None or ro is None:
        print("FATAL: dump lacks sc_pred/route_oracle"); return 1
    curv5 = sc[:, 2].double()                      # scalar_order: ttm, curv_3s, curv_5s, tspeed_5s
    graded = torch.tanh(curv5 / CURV_TURN_PER_M)
    ro = ro.long()

    # judgeable manoeuvre classes only; UNKNOWN(3)/DROPPED(4) are masked sentinels
    keep = ro <= RIGHT
    graded, true = graded[keep], ro[keep]
    print(f"n_total={ro.numel()}  judgeable={int(keep.sum())}  "
          f"(dropped {int((~keep).sum())} masked-sentinel rows)")
    maj = int((true == STRAIGHT).sum()) / int(true.numel())
    print(f"always-straight majority = {maj:.4f}   chance (3 class) = {1/3:.4f}\n")

    grid = [round(0.05 * i, 3) for i in range(1, 19)]          # 0.05 .. 0.90
    for extra in (CURRENT_THR, math.tanh(0.56)):
        if round(extra, 3) not in grid:
            grid.append(round(extra, 4))
    grid = sorted(set(grid))

    rows = []
    print(f"{'thr':>7s} {'balAcc':>8s} {'plainAcc':>9s} "
          f"{'recL':>6s} {'recS':>6s} {'recR':>6s} {'precL':>6s} {'precR':>6s}")
    for thr in grid:
        s = scores(classify(graded, thr), true)
        p = s["per_class"]
        tag = ""
        if abs(thr - CURRENT_THR) < 1e-3:
            tag = "  <-- CURRENT (tanh 1.0)"
        elif abs(thr - math.tanh(0.56)) < 1e-3:
            tag = "  <-- shrinkage-corrected prediction"
        pl = p["left"]["precision"]
        pr = p["right"]["precision"]
        print(f"{thr:7.3f} {s['balanced_accuracy']:8.4f} {s['plain_accuracy']:9.4f} "
              f"{p['left']['recall']:6.3f} {p['straight']['recall']:6.3f} "
              f"{p['right']['recall']:6.3f} "
              f"{(pl if pl is not None else float('nan')):6.3f} "
              f"{(pr if pr is not None else float('nan')):6.3f}{tag}")
        rows.append({"thr": thr, **s})

    base = next(r for r in rows if abs(r["thr"] - CURRENT_THR) < 1e-3)
    best = max(rows, key=lambda r: r["balanced_accuracy"])
    print(f"\nCURRENT thr={base['thr']:.4f}  balAcc={base['balanced_accuracy']:.4f}")
    print(f"BEST    thr={best['thr']:.4f}  balAcc={best['balanced_accuracy']:.4f}  "
          f"delta={best['balanced_accuracy'] - base['balanced_accuracy']:+.4f}")
    print("\n⚠️ This is a threshold chosen ON the evaluation split — it is an UPPER BOUND on "
          "what recalibration buys, not a validated operating point. Confirm on a held-out "
          "split before deploying it.")
    json.dump({"dump": DUMP, "current_thr": CURRENT_THR, "majority": maj,
               "primary": "balanced_accuracy (mean per-class recall, L/S/R)",
               "rows": rows, "base": base, "best": best}, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
