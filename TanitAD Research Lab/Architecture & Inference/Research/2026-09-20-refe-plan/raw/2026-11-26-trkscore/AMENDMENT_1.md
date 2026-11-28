# PREREG_TRKSCORE, Amendment 1: CTRL-PARITY's start-state bar (one bar only)

REFe-line agent, 2026-11-26. It amends `PREREG_TRKSCORE.md` (sha256 `5cf2c3235b5a6f2d8b38d8b278e105f64a42d7389de5ebc10fc7a1d11cc2714e`).

**State when this was frozen.**
- No TRK PDMS number, sub-score or family value exists.
- The smoke read controls only (`trk_ctrl.py`), and nothing in it touches a score.
- The full run has not started.

## What failed, as written
The smoke ran on 50 tokens, C1. Its output is `E:/Projects/TanitAD-artifacts/trkscore/work/smoke/ctrl_smoke_c1.json` (the as-written file is kept).
- Six controls PASS: CTRL-SET, CTRL-ID (50/50, max |logit difference| 0.0), CTRL-DET (50/50, max |dprops| 0.0), CTRL-RED (min d0 1.000 m; 29/50 picks changed), CTRL-SUB and CTRL-SEAM.
- **CTRL-PARITY FAILED on its start-state bar.** The start state arrays differ by up to **4.48e-7** against the bar of ≤ 1e-9.
- The tracked states themselves meet their bar by more than 200×: max |Δx|, |Δy| **4.69e-7 m** and |Δheading| **1.48e-8 rad**, against 1e-4.

## Diagnosis
MEASURED with `work/tmp/s0_diag.py`, component by component, over the 50 tokens:
- **Pose is bit-identical**: x, y and heading differ by 0.0. Steering angle, steering rate, angular velocity and angular acceleration also differ by 0.0.
- **Only the velocity and acceleration differ**: vx 4.48e-7, vy 1.15e-8, ax 5.29e-8, ay 4.84e-8.
  - The relative differences are 5.38e-8, 4.96e-8, 5.69e-8 and 4.79e-8. That is float32 rounding (eps/2 = 5.96e-8).
  - Example token: the cache holds vx = 4.056971073150635, which is a float32 value. The planner's `EgoState` holds 4.056971291407112, in float64.
- ⇒ The harness's `MetricCache.ego_state` stores the kinematics rounded to float32, while the planner's own `EgoState` keeps them in float64. They are the same physical quantity at two precisions, not two different states.
- The 1e-9 bar was written without knowing how the cache stores these values.

## The change (this bar only)
- **The start-state part of CTRL-PARITY becomes:** pose (x, y, heading) exactly equal, and every other component equal to within the float32 rounding of the cache value: |Δ| ≤ 1.2e-7 × max(1, |value|).
- **Unchanged:**
  - the tracked-state bars (1e-4 m, 1e-4 rad);
  - every other control;
  - every bar in §5;
  - the rule. TRK starts from **the planner's own `EgoState`**, the inference-admissible input. It is not rounded to imitate the harness, because that would make the rule read a property of the harness's storage.
- The tracked-state agreement above (4.7e-7 m) is the measured cost of the two precisions. That is selection-irrelevant at the 1e-4 bar.
- The as-written FAIL stays printed in the read, next to the amended PASS.
