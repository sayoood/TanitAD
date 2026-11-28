# CONTROLS_LADDER_PLANTEST (agent, 2026-11-09; tier T0, CPU, 1 thread; tolerances fixed in IMPL_CHOICES_LADDER_PLANTEST.md section 6 before any run)
Raw: `CONTROLS_LADDER_PLANTEST.json`, code `controls_lp.py` (run 2026-11-09). All eleven checks PASS; every deliberate regression goes RED.

## 1. M2b reproduction through this code path (banked PLANTEST number)
* M2b_s0, FLAT1, G-TRUE, first 64 windows, PLANTEST initialisation: banked acc MAE 1.463266, mine 1.463266; banked yaw MAE 0.111046, mine 0.111046; max |plan difference| 0.0 (tolerance 1e-3 / 1e-4 / 1e-5): PASS, bit-identical.
* PLANTEST2 path (SEED, lambda 1.0, first 64 SCORE windows): banked acc MAE 0.914901, mine 0.914901; max |plan difference| 0.0: PASS.
* Regression (planner seed 1 instead of 0): max |plan difference| 3.439 > 1e-3: RED as required.
* Latent route: whole-clip latents of M2b_s0 against `latents.pt` Z1, max abs difference 9.54e-07 (tolerance 1e-4); logged actions equal the cache's (difference 0.0).

## 2. Reachability on a real batch (8 windows per arm, 4 candidates; mean abs gradient of the level-1 goal cost with respect to the plan)
| arm | form | acc channel | yaw channel |
|---|---|---|---|
| R35_s0 | c1 | 1.182 | 2.257 |
| R35_s0 | c3 | 1.587 | 7.916 |
| R35_s1 | c1 | 1.322 | 2.110 |
| R35_s1 | c3 | 1.648 | 6.080 |
| R35H_s0 | c1 | 2.783 | 11.453 |
| R35H_s0 | c3 | 11.264 | 71.979 |
| R35H_s1 | c1 | 2.692 | 7.355 |
| R35H_s1 | c3 | 8.142 | 37.407 |
| C0_s0 | c1 | 1.898 | 3.336 |
| C0_s0 | c3 | 1.732 | 3.265 |
| C0_s1 | c1 | 1.060 | 1.667 |
| C0_s1 | c3 | 1.084 | 2.024 |
Non-zero in both channels on every arm at c1 and c3 (tolerance > 1e-6). Zero control (planted world, actions disconnected from the predictor, same rollout and cost functions): max |gradient| = 0.0 at c1 and 0.0 at c3 (exactly 0.0); the connected planted world reads 3.489.

## 3. Regressions that must go RED
* Context 3 fed a repeated latent instead of the real t-2 / t-1 history: the first predicted state differs from the direct predictor call on the real history by 0.899 (R35H_s0) and 0.245 (C0_s0); with the real history the difference is 0.0 (tolerance 1e-6). RED as required. With no history the new rollout equals `st.rollout` at context 1 and 3 exactly.
* Wrong stride: argv strides replaced by (2, 2) on R35 raises LadderMismatch; `td3_defined` flips when fed the other arm's built strides; TD3-defined reads False on R35 (built cumulative strides [1, 3, 15] frames: the level-3 step is 15 frames = 3.0 s, so the card's 2.4 s goal (12 frames) is not on level 3's grid and its top horizon of 3 steps would be 9.0 s; the level-2 step is 0.6 s so the card's 6-state sub-plan (H_2 = 6) is 3.6 s, not 1.2 s) and True on C0.
* Planner seed 1 vs 0 (reproduction), HOLD seed on vs off (reduction), REG off (limit: max deviation 3.53 prior std against 0.0), actions disconnected (planted recovery MAE 0.365 against the 0.05 tolerance), and garbage a0 (the plan changes, so the leak check can fail): all RED.

## 4. Other controls
* Reduction: `plan_all3` at c1, prior initialisation, equals `plantest_lib.plan_all` and with the HOLD seed + REG 0.3 equals `plantest2_lib.plan_all2`, bit for bit, 16 windows, FLAT1 on C0 and R35, TD3 on C0 (difference 0.0).
* Planted dynamics through `plan_topdown3` at c1 and c3: TD3 MAE 0.0036, FLAT1 0.0070 against the 0.05 tolerance and a baseline of 0.365. (The planted predictor ignores history by construction, so c3 equals c1 there; the history itself is checked in item 3.)
* Limit (lambda 1e6): plan equals HOLD, max deviation 0.0 prior std at R35 c1, R35 c3, C0 c1 (tolerance 1e-4).
* Goal plumbing, bit-repeat, no-leak (R35 c1, R35 c3, C0 c3, 16 windows): the plan equals a direct planner call on goal arrays built outside the runner, repeats bit for bit, G-TRUE differs from G-RAND; replacing the logged FUTURE actions (knots 1..15) by noise changes nothing; replacing a0 (the one logged action that enters, through HOLD) changes the plan.
* No-leak statement: logged future actions reach only the metric (`controls_lp`/`report_lp`: `window_metrics`) and the TRAIN prior fit; on eval windows the planner receives the latent at NOW, the goal latent, HOLD_c (a0 only, `lp_lib.hold_c_of`, as PLANTEST2's HOLD seed), and under c3 the past latents and past measured actions at g-2, g-1 taken from the cache, not from the window array (`lp_lib.build_ctx`).
* Split: the SCORE index set of this code equals the banked PLANTEST2 run's SCORE index set; the TUNE read asserts it touches no SCORE window (`run_lp.work`).
* TRAIN prior shortcut: macro-action embeddings are identical for zero, random and real level-1 latents (difference 0.0), so TRAIN latents of the ladder arms are not needed.
