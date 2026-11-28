#!/usr/bin/env python
"""⭐ DOES THE CONSTANT I PINNED IN CODE REALISE THE DESIGN M15 APPROVED?

A vocabulary approval is worthless if the shipped constant is not the one that
was scored. This closes that loop in two independent ways, and neither is a
re-derivation of the approver's own table — both compare the LIVE CODE against
the banked `vocab_design.json`.

  A. THE CONSTANT. Score `refa_v1.GOAL_KAPPA_TURN_LEVELS` on the same dense
     panel (`intent_stride2.npz`, 4786 windows), with the approver's own scorer
     semantics: `|kappa|` clipped to `GOAL_KAPPA_MAX`, windows at v0 < 1 m/s
     excluded (below the speed floor `yaw_rate/speed` is not a curvature), and
     each design credited only with `level x duty` where
     `duty = GOAL_TURN_S / horizon_s`. It must reproduce the banked
     `L=3 at corpus quantiles` row.

  B. ⭐ THE CODE PATH. The scorer is analytic — it assumes the profile holds
     `level` for `GOAL_TURN_S` and 0 afterwards. `canonical_controls_levels` is
     what the planner actually calls. This measures the duty cycle and the
     sustained magnitude OFF THE RETURNED TENSOR and requires them to match the
     analytic assumption, so an approval scored on paper cannot diverge from the
     code that ships. *(The `anchors.pt` units trap, prevented rather than
     repeated: the artifact must state, and here PROVE, what it is.)*

CONTROLS, each of which must read a KNOWN value or the check is void:
  * the SHIPPED single magnitude is scored in the same run and must reproduce
    its own banked row — if it does not, the panel or the scorer differs and
    nothing else in the table can be trusted;
  * the CONTINUOUS ceiling must read EXACTLY 0.0;
  * the `LANE_KEEP`-only floor must read the road's own RMS curvature;
  * `n` and the excluded count are printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

GOAL_TURN_S = 4.0
GOAL_KAPPA_MAX = 0.2


def residual(gt, levels, duty):
    """The approver's scorer, verbatim in semantics (`vocab_design.residual`)."""
    cand = np.array(sorted({0.0} | {abs(x) for x in levels}))
    eff = cand * duty
    err = np.abs(np.abs(gt)[:, None] - eff[None, :])
    j = err.argmin(1)
    return err[np.arange(len(gt)), j], cand[j]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", required=True)
    ap.add_argument("--banked", required=True, help="raw/vocab_design.json")
    ap.add_argument("--stack", default="C:/Users/Admin/tanitad-wt/stack")
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--turn-thr", type=float, default=1e-2)
    ap.add_argument("--min-v0", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.stack)
    from tanitad.refs.refa_v1 import (GOAL_KAPPA_TURN, GOAL_KAPPA_TURN_LEVELS,
                                      canonical_controls_levels,
                                      goal_kappa_vocab_id)

    Z = np.load(a.intent, allow_pickle=False)
    v0 = Z["v0"].astype(np.float64)
    keep = v0 >= a.min_v0
    gt = np.clip(Z["gt_kappa"].astype(np.float64)[keep],
                 -GOAL_KAPPA_MAX, GOAL_KAPPA_MAX)
    duty = min(1.0, GOAL_TURN_S / a.horizon_s)
    turn = np.abs(gt) > a.turn_thr

    def score(levels, name):
        r, pick = residual(gt, levels, duty)
        return {"design": name, "levels": [round(float(x), 5) for x in levels],
                "n": int(len(gt)), "n_turns": int(turn.sum()),
                "medae_on_turns": float(np.median(r[turn])),
                "rmse_all": float(np.sqrt(np.mean(r ** 2))),
                "frac_turns_expressible": float((pick[turn] > 0).mean())}

    out: dict = {
        "panel": a.intent, "n_windows_total": int(len(v0)),
        "n_excluded_low_v0": int((~keep).sum()), "duty_cycle": duty,
        "turn_threshold": a.turn_thr,
        "code_constant": {"GOAL_KAPPA_TURN": GOAL_KAPPA_TURN,
                          "GOAL_KAPPA_TURN_LEVELS": list(GOAL_KAPPA_TURN_LEVELS),
                          "vocab_id": goal_kappa_vocab_id(GOAL_KAPPA_TURN_LEVELS)},
        "scored": {"L3_from_code": score(list(GOAL_KAPPA_TURN_LEVELS),
                                         "L=3 from refa_v1 constant"),
                   "L1_shipped_CONTROL": score([GOAL_KAPPA_TURN],
                                               "L=1 shipped (CONTROL)")}}

    # ---- controls that must read known values ----------------------------- #
    r0, pick0 = residual(gt, [], duty)
    out["CONTROL_floor_L0"] = {
        "rmse_all": float(np.sqrt(np.mean(r0 ** 2))),
        "road_rms_kappa": float(np.sqrt(np.mean(gt ** 2))),
        "equal_to_1e-12": bool(abs(np.sqrt(np.mean(r0 ** 2))
                                   - np.sqrt(np.mean(gt ** 2))) < 1e-12),
        "frac_expressible": float((pick0[turn] > 0).mean()),
        "why": "with no level the best available token is LANE_KEEP, so the "
               "residual IS the road's curvature — if these differ the scorer "
               "is not measuring what it claims"}
    dense = np.abs(gt) / max(duty, 1e-12)
    rc, _ = residual(gt, list(np.unique(dense)), duty)
    out["CONTROL_ceiling_continuous"] = {
        "rmse_all": float(np.sqrt(np.mean(rc ** 2))),
        "is_exactly_zero": bool(np.max(np.abs(rc)) < 1e-12)}

    # ---- A. against the banked approval ----------------------------------- #
    with open(a.banked, encoding="utf-8") as fh:
        B = json.load(fh)
    banked_l3 = next(r for r in B["axis_B_n_levels"]
                     if r["design"].startswith("L=3"))
    banked_l1 = B["shipped"]
    cmp = {}
    for tag, mine, theirs in (("L3", out["scored"]["L3_from_code"], banked_l3),
                              ("L1_shipped", out["scored"]["L1_shipped_CONTROL"],
                               banked_l1)):
        cmp[tag] = {k: {"mine": mine[k], "banked": theirs[k],
                        "match": bool(abs(mine[k] - theirs[k]) <= a.tol)}
                    for k in ("medae_on_turns", "rmse_all",
                              "frac_turns_expressible")}
        cmp[tag]["levels_match"] = bool(
            [round(x, 5) for x in mine["levels"]] ==
            [round(x, 5) for x in theirs["levels"]])
    out["A_vs_banked_approval"] = cmp
    out["A_passes"] = bool(all(v["match"] for t in cmp.values()
                               for k, v in t.items() if isinstance(v, dict))
                           and all(t["levels_match"] for t in cmp.values()))

    # ---- B. the analytic profile against the LIVE code path --------------- #
    K, DT = 30, 0.2
    c = canonical_controls_levels("TURN_L", "CRUISE", 10.0, K, DT,
                                  GOAL_KAPPA_TURN_LEVELS).numpy()
    rows = []
    for i, lv in enumerate(GOAL_KAPPA_TURN_LEVELS):
        kap = c[i, :, 1]
        nz = int((np.abs(kap) > 0).sum())
        rows.append({
            "level": float(lv),
            "sustained_magnitude": float(np.abs(kap).max()),
            "magnitude_matches": bool(abs(np.abs(kap).max() - lv) < 1e-6),
            "n_steps_nonzero": nz, "n_steps": K,
            "measured_duty": nz / K,
            "duty_matches_analytic": bool(abs(nz / K - duty) < 1e-9),
            "mean_effective_kappa": float(np.abs(kap).mean()),
            "effective_matches_level_times_duty":
                bool(abs(float(np.abs(kap).mean()) - lv * duty) < 1e-7)})
    out["B_code_path"] = {
        "rows": rows,
        "passes": bool(all(r["magnitude_matches"] and r["duty_matches_analytic"]
                           and r["effective_matches_level_times_duty"]
                           for r in rows)),
        "why": "the approval was scored on an ANALYTIC profile; this asserts "
               "the shipped `canonical_controls_levels` produces exactly that "
               "profile, so the approved table describes the code that runs"}

    out["VERDICT"] = {
        "constant_reproduces_the_approved_row": out["A_passes"],
        "code_path_realises_the_scored_profile": out["B_code_path"]["passes"],
        "controls_pass": bool(out["CONTROL_floor_L0"]["equal_to_1e-12"]
                              and out["CONTROL_ceiling_continuous"]
                              ["is_exactly_zero"])}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("code_constant", "scored", "CONTROL_floor_L0",
                       "CONTROL_ceiling_continuous", "A_vs_banked_approval",
                       "B_code_path", "VERDICT")}, indent=1))
    print(f"[wrote] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
