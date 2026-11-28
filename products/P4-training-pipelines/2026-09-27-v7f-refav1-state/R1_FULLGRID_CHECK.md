# R1 on the full grid — does the refav1 planner honour the clip's max speed on all 282 windows? (pre-registered)

**2026-09-27, TrainingFlyWheel; written and staged BEFORE the arm starts.** PI requirement R1 (2026-09-27): *"both of
them should get the max speed, which it must not be exceeded."* Batch `R1-VMAX` (`LANDING_READY.txt`) implements it
for refav1 at inference: `PlanConfig.v_max` caps every candidate incl. the injected baselines; `refav1_arm.py
--vmax-from-labels` reads each clip's limit from `g_tac.goals.SPEED_BAND.v_hi_ms` through
`refcv6_max_speed.speed_max_bin` / `limit_ms_of_bin`. MEASURED before this check: 147/147 eval clips carry a limit
(30 km/h 49, 50 km/h 52, 100 km/h 39, 120 km/h 7; 1 clamped from above 120).

**This is an ENGINEERING VERIFICATION, not a performance claim.** The planner is already eliminated as a lever on
this checkpoint (`…/2026-09-27-refav1-fullgrid-loncomb3/RESULT.md`); the question is only whether R1 holds.

- **Arm `A1cap`:** exactly A1 of the full-grid SPEC (`loncomb3`, `--plan-seed 0`) plus `--vmax-from-labels`, on the
  same 141 episodes / 282 windows; code = the R1-VMAX fix files on top of `b3f7ea6f` (dev copy
  `C:/Users/Admin/refav1_dev`), same checkpoint (md5 `1189bc02…`).
- **Committed expectations:**
  1. `vmax_cap.n_plan_over_limit` **= 0** (hard requirement; any other value = R1 FAILS and the fix is wrong).
  2. `vmax_cap.max_plan_excess_mps` ≤ 1e-4.
  3. `n_v0_over_limit` reported (windows where the car is ALREADY above its limit at t0; the plan must brake).
  4. Floors (`ha`, `ha0`, `ha0_ext`, `ol`) bit-identical to A1's (the cap does not touch them).
  5. `A1cap − A1` reported on all four families (paired, 141 clusters) with the reading anchored to A1's own
     inference-seed floor (`A2 − A1` ADE +0.0619): a difference smaller than that is not a cap effect.
- ⚠️ Floors are NOT capped (they are the published do-nothing references); the number of windows where a floor exceeds
  its clip's limit is reported beside them, so no floor is silently allowed a speed the arm is not.
