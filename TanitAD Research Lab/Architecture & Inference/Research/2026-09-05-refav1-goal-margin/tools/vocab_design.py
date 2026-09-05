#!/usr/bin/env python
"""HOW MUCH SUSTAINED CURVATURE DOES THE GOAL VOCABULARY NEED? — priced at zero GPU.

⭐ WHY. `ROOT_CAUSE.md` MEASURED that the lateral goal can command exactly two
SUSTAINED curvatures, 0 and +/-GOAL_KAPPA_TURN = 0.08 (R 12.5 m), while the
corpus curves at R 100-1000 m, and that `LANE_KEEP` is consequently the
VOCABULARY-OPTIMAL token on ~90 % of GT-turn windows. The fix is a finer
sustained curvature. **This prices that fix BEFORE any GPU is spent**: it asks,
for a candidate vocabulary, what fraction of the road becomes EXPRESSIBLE and how
much residual curvature error remains.

Two design axes are swept, both against the SAME banked `gt_kappa`:

  A. **the single magnitude** -- keep one TURN pair, move `GOAL_KAPPA_TURN`.
     Reported: expressible fraction, and the residual RMSE of the best-available
     token against the road. This is the CHEAPEST possible change (one constant).
  B. **the number of levels** -- a symmetric quantiser with L>=1 magnitudes
     (plus 0), sited at the L-quantiles of the corpus's own |kappa| among GT
     turns, so the sizing is DERIVED from the road rather than chosen.

⛔ THE FLOOR THAT MAKES THE NUMBERS MEAN SOMETHING. A vocabulary is scored by the
error of its BEST token against the road -- so the comparison is against:
  * `L=0` (LANE_KEEP only): every turn inexpressible, and the RMSE is just the
    road's own RMS curvature. Any vocabulary that cannot beat this is useless.
  * `CONTINUOUS` (kappa exactly = gt): residual EXACTLY 0.0. This is the ceiling,
    and it must read 0.0 or the scorer is broken.
Both are printed in every table. n printed for every cell.

⚠️ THIS IS A VOCABULARY-ADEQUACY BOUND, NOT A DRIVING RESULT. It assumes an
ORACLE that always picks the best available token. The real head's selection is
measured separately (`threshold_sweep.py`: AUC 0.8806 / recall 0.744 at the
crossover). A vocabulary that cannot express the road bounds the system from
above; it does not promise the head will use it.

Run: python gm_vocab_design.py --intent <npz> --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np

GOAL_KAPPA_TURN = 0.08          # refa_v1.py:119
GOAL_TURN_S = 4.0               # refa_v1.py:120
GOAL_KAPPA_MAX = 0.2            # refa_v1.py:118


def residual(gt, levels, duty):
    """RMS error between the road's constant curvature and the best available
    SUSTAINED token profile. `levels` are non-negative magnitudes; 0 is always
    available (LANE_KEEP). `duty` is the TURN profile's duty cycle over the
    horizon, so a token that holds kappa for only part of the window is not
    credited with holding it throughout."""
    cand = np.array(sorted({0.0} | {abs(x) for x in levels}))
    # a signed token of magnitude m contributes m*duty over the horizon in the
    # L2 sense; LANE_KEEP contributes 0 everywhere.
    eff = cand * duty
    err = np.abs(np.abs(gt)[:, None] - eff[None, :])
    j = err.argmin(1)
    return err[np.arange(len(gt)), j], cand[j]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", required=True)
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--turn-thr", type=float, default=1e-2,
                    help="what counts as a turn worth expressing (R<=100 m)")
    ap.add_argument("--min-v0", type=float, default=1.0,
                    help="⛔ EXCLUDE near-stationary windows. gt_kappa is "
                         "yaw_rate/speed.clamp_min(0.5), so below ~1 m/s it "
                         "divides by the FLOOR and is not a curvature. "
                         "MEASURED on the dense panel: 13 windows (0.27 %) "
                         "read |kappa|>0.2 and 92.3 % of them are at v0<1 m/s "
                         "(max 3.565 = R 0.28 m, physically impossible); they "
                         "carry the ENTIRE RMS (0.15827 -> 0.02192 clipped).")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Z = np.load(a.intent, allow_pickle=False)
    gt_raw = Z["gt_kappa"].astype(np.float64)
    v0 = Z["v0"].astype(np.float64)
    keep = v0 >= a.min_v0
    n_excluded = int((~keep).sum())
    gt = np.clip(gt_raw[keep], -GOAL_KAPPA_MAX, GOAL_KAPPA_MAX)
    n = len(gt)
    duty = min(1.0, GOAL_TURN_S / a.horizon_s)
    turn = np.abs(gt) > a.turn_thr

    def score(levels, name):
        r, pick = residual(gt, levels, duty)
        expressible = pick > 0
        return {"design": name,
                "levels": [round(float(x), 5) for x in sorted(set(levels))],
                "n": int(n), "n_turns": int(turn.sum()),
                "medae_on_turns": (float(np.median(r[turn]))
                                   if turn.sum() else None),
                "rmse_all": float(np.sqrt(np.mean(r ** 2))),
                "rmse_on_turns": (float(np.sqrt(np.mean(r[turn] ** 2)))
                                  if turn.sum() else None),
                "frac_turns_expressible": (float(expressible[turn].mean())
                                           if turn.sum() else None),
                "frac_straights_given_nonzero": (
                    float(expressible[~turn].mean()) if (~turn).sum() else None)}

    rows = []
    # ---- FLOOR and CEILING ------------------------------------------------ #
    rows.append(score([], "L=0  LANE_KEEP only (FLOOR)"))
    cont = {"design": "CONTINUOUS kappa (CEILING)", "levels": None, "n": int(n),
            "n_turns": int(turn.sum()), "medae_on_turns": 0.0,
            "rmse_all": 0.0, "rmse_on_turns": 0.0,
            "frac_turns_expressible": 1.0, "frac_straights_given_nonzero": None}

    # ---- A. one magnitude, swept ----------------------------------------- #
    axis_a = []
    for k in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12):
        r = score([k], f"L=1  kappa_turn={k}")
        r["is_shipped"] = bool(abs(k - GOAL_KAPPA_TURN) < 1e-12)
        axis_a.append(r)

    # ---- B. L levels at the corpus's own quantiles ------------------------ #
    axis_b = []
    ak = np.abs(gt)[turn]
    for L in (1, 2, 3, 4, 6):
        qs = [(i + 0.5) / L for i in range(L)]
        lv = [float(np.quantile(ak, q)) for q in qs]
        lv = [min(x, GOAL_KAPPA_MAX) for x in lv]
        axis_b.append(score(lv, f"L={L}  at corpus quantiles"))

    ctrl = {
        "CONTROL_continuous_is_exactly_zero": {
            "rmse": cont["rmse_all"], "passes": cont["rmse_all"] == 0.0},
        "CONTROL_floor_equals_road_rms_curvature": {
            "floor_rmse": rows[0]["rmse_all"],
            "road_rms_kappa": float(np.sqrt(np.mean(gt ** 2))),
            "passes": bool(abs(rows[0]["rmse_all"]
                               - float(np.sqrt(np.mean(gt ** 2)))) < 1e-12),
            "note": ("with only LANE_KEEP the residual IS the road's own RMS "
                     "curvature; if these differ the scorer is not measuring "
                     "what it claims")},
        "CONTROL_shipped_row_present": {
            "passes": any(r.get("is_shipped") for r in axis_a)},
    }

    ship = [r for r in axis_a if r.get("is_shipped")][0]
    best_a = min(axis_a, key=lambda r: r["medae_on_turns"])
    out = {"source": a.intent, "n_windows": int(n),
           "n_excluded_low_speed": n_excluded, "min_v0": a.min_v0,
           "target_clipped_to": GOAL_KAPPA_MAX,
           "horizon_s": a.horizon_s, "turn_duty_cycle": duty,
           "turn_threshold": a.turn_thr, "n_turns": int(turn.sum()),
           "note": ("VOCABULARY-ADEQUACY BOUND under an ORACLE token chooser. "
                    "It bounds the system from above; it does not promise the "
                    "head will select well. Head selection is measured in "
                    "threshold_sweep.json."),
           "floor": rows[0], "ceiling": cont,
           "axis_A_single_magnitude": axis_a,
           "axis_B_n_levels": axis_b,
           "shipped": ship, "best_single_magnitude": best_a,
           "controls": ctrl}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"n={n} (excluded {n_excluded} windows with v0<{a.min_v0} m/s; "
          f"target clipped to +/-{GOAL_KAPPA_MAX})  "
          f"turns(|k|>{a.turn_thr})={int(turn.sum())}  duty={duty:.3f}")
    print(f"{'design':<34}{'RMSE all':>10}{'medAE turns':>12}"
          f"{'turns expressible':>19}{'straights given k':>19}")
    def pr(r):
        print(f"{r['design']:<34}{r['rmse_all']:>10.5f}"
              f"{(r['medae_on_turns'] if r['medae_on_turns'] is not None else float('nan')):>12.5f}"
              f"{(r['frac_turns_expressible'] if r['frac_turns_expressible'] is not None else float('nan')):>19.4f}"
              f"{(r['frac_straights_given_nonzero'] if r['frac_straights_given_nonzero'] is not None else float('nan')):>19.4f}")
    pr(rows[0])
    for r in axis_a:
        pr(r)
        if r.get("is_shipped"):
            print("   ^^^ SHIPPED (GOAL_KAPPA_TURN = 0.08)")
    print()
    for r in axis_b:
        pr(r)
    print(f"{'CONTINUOUS (CEILING)':<34}{0.0:>10.5f}{0.0:>12.5f}{1.0:>19.4f}")
    print()
    print("CONTROLS: " + json.dumps({k: v["passes"] for k, v in ctrl.items()}))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
