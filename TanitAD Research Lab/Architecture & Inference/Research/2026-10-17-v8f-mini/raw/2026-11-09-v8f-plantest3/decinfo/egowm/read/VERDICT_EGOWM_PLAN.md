# EGOWM planner half: NO-GAIN in all 16 cells; the outcome NO-EFFECT stands

Card `PREREG_V8F_EGOWM.md` (prefix sha256 `ddd6fc9b`). This is the planner half that `VERDICT_EGOWM.md` committed to report when it finished. That verdict already says it "cannot change the outcome": the deciding row is L1 IA-0 skill, and it read NO-EFFECT. Run: `run_lp_speed.py` units for E-NOW and E-KIN, seeds 0 and 1, `PLAN_SPEED4.done` rc=0 at 2026-11-12 13:39. Read by `report_speed.py` into `RESULT_EGOWM_PLAN.json` (log `report_speed.log`). The R35H rows come from the ladder harness's own runs.

**Tier T0 (oracle vision-only goal; adapter, readers and planner frozen at M2b): a world-model diagnostic, never driving performance.** Paired EC-BOOT over the 71 SCORE episodes, B 2,000 (994 windows at context 1, 923 at context 3). The bootstrap answers "another draw of episodes" only. MEASURED.

## Result
* Every cell is **NO-GAIN**: {E-NOW, E-KIN} x {c1, c3} x {PLANTEST, PLANTEST2} x {LONGITUDINAL, LATERAL}, at both seeds. Planning toward the true future is separated-WORSE than holding and separated-better than a random goal. That is R35H's pattern.
* PLANTEST c1, acceleration MAE (m/s^2; HOLD 0.3296). G-TRUE: R35H 1.1637 / 1.1753; E-NOW 1.1962 / 1.1881; E-KIN 1.1779 / 1.1540 (s0 / s1). E-KIN s0 G-TRUE minus HOLD: +0.8483 [+0.7960, +0.9008]. Yaw-rate MAE (rad/s; HOLD 0.0224): E-KIN s0 0.0929, G-TRUE minus HOLD +0.0706 [+0.0654, +0.0761].
* Over all 48 rows (three world models, every cell, both seeds) the lower CI bound of G-TRUE minus HOLD in LONGITUDINAL is at least +0.4464 m/s^2.
* **The planner lowers its own latent cost on 100 % of windows in all 48 rows (`frac_cost_lowered_L1` = 1.0).** The optimisation works. What it optimises does not lead to the logged actions: the actions that bring a 12-step level-1 rollout closest to the true future latent are worse than holding.
* Beside, against R35H (no bar; 16 rows in `beside_vs_R35H`): the acceleration differences are -0.049 to +0.008 m/s^2, at most about 0.06 of the gap to HOLD. One is separated (E-KIN c1 s1, -0.0486 [-0.0934, -0.0011]). In LATERAL, 5 of 8 rows read separated-worse, by +0.0008 to +0.0076 rad/s.
* TACTICAL: n = 0 (no decision output). STRATEGIC: PAUSED (PI item 25).

## Master Mind context (what this adds)
The measured speed given to the predictor changes neither the skill row (NO-EFFECT) nor the plan. Taken with `H-V8F-PROOF-3 WMCEIL` (2026-11-13), the failure has a mechanism. On M2b, ten composed 0.2 s steps reach 2 s at skill -0.15 / -0.17, below copy-last, while one direct 2 s jump reaches +0.12 / +0.13 (JUMP-DOMINATES). FLAT1 plans 12 knots through exactly that composition. A planner that descends a cost through a rollout worse than copying will find actions that exploit the rollout's error. A lower cost on every window, with worse actions on every row, is that signature.

**Next lever (Rule Zero):** the hierarchical planner of `PREREG_V8F_HWMPLAN` (L3 -> L2 -> L1, with level-2 sub-goals every 0.6 s on the (3, 5) ladder). It replaces the long level-1 composition by direct higher-level steps. Its T0 rows (HWM-ORACLE and FLAT-ORACLE against HOLD, on R35H, zero training, dev-box CPU) decide PLANNER-LIMITED against GOAL-LIMITED before any box-2 time goes to the goal head (G4, G5).

Plan arrays (88 files, numeric only): PRIVATE on HF, see `HF_BANK_EGOWM_PLAN.json`.
