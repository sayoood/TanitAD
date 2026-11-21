# PREREG — REFe ego-input reliance probe (T0), written 2026-11-06 BEFORE any probe forward was run

**Tier: T0** (open-loop, teacher-forced: real nuPlan frame + real ego state at t0 + route-derived goal, one `infer` call per token, no closed loop).
This is an input-reliance diagnostic of the u400 policy. It is NOT a driving-performance claim.
**Evidence class of every number it produces: MEASURED (ours, artifact `analysis/`), except where marked.**

## Question
A closed-loop sim session reports REFe "crawls": on a clear road with ax=0 it plans to lose 15-45 % of its speed in 4 s; grey images move the plan only ~20-30 %;
feeding ax=+-0.5 swings the plan strongly. MM measured one wiring defect: ego slots 4 (yaw_rate) and 5 (tire_steering_angle) are 0.0 in every nuPlan DB ego state,
so they were never trained. Is the slow-down a property of the POLICY (in-distribution) or an artefact of OUT-OF-DISTRIBUTION sim inputs?

## Fixed inputs
* Model `E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt`, required sha256 prefix `8b4dba4818b5cc96` (verified at brief time; re-verified by the script, refuses otherwise).
* Tokens: subset of the C3 tokens (`raw/2026-10-11-c3-selection-headroom/tokens.json`, 1,058 tokens / 123 logs). Subset rule in "Token subset" below; it uses ONLY the W3 export's
  logged inputs and logged future (no model output), fixed here before any forward.
* Planner path unchanged: `refe_navtest_seam` loop -> `NS.build_scenarios_for_log` -> `REFePlanner.initialize` -> `planner._image_for` -> `planner.infer` (batch 1,
  rule `navsim_v1_lane`, `select=best`, default goal fix `pdm_route`, default repair_last_heading). Scenario + images + goal are built ONCE per token; every arm runs on the same inputs.
  PK (the package) is read-only; perturbations are monkeypatches inside the probe script.

## Arms (ego vector order vx, vy, ax, ay, yaw_rate, steer, speed; images are float in [0,1], ImageNet normalisation is inside the trunk)
| arm | perturbation |
|---|---|
| A0 | real |
| A0r | replicate of A0 (fresh trunk call, no cache). DETERMINISM CONTROL: must be bit-identical to A0 (traj, scorer logits, pick) |
| AX0 | ax = ay = 0 (other slots real) |
| AXP | ax = +0.5 (ay real) |
| AXM | ax = -0.5 (ay real) |
| DEAD | yaw_rate = 0.3, steer = 0.2 (the sim's dead-slot input; other slots real) |
| GREY | every camera pixel = 128/255 |
| SWAP | all 4 camera images replaced by those of a token of a DIFFERENT log, fixed derangement: logs sorted by name; token i (sorted) of log j takes the image of token (i mod n_donor) (sorted) of log j-1 (cyclic) |

Controls (all recorded in `results_meta.json`, reported before any arm effect):
* C1 determinism: A0r == A0 bit-for-bit on picked traj, all 64 proposals, 7 scorer logits, pick index. Fail => every sub-noise delta is uninterpretable, say so.
* C2 identity: A0 reproduces `table.npz` (made by the same u400 checkpoint: `STATUS.json` ckpt = model_sft4_B_u400.pt; fidelity gate PASS there): scorer logits bit-identical, pick index equal,
  NAVSIM-grid (8-pose) executed proposals max |d| <= 1e-3 m. If the platform (GPU/driver) differs and logits are not bit-identical, the bar is: pick equal on >= 99 % and proposals <= 1e-3 m, and this is stated.
* C3 reachability (mutate, do not inspect): the ego vector actually fed to the model (`planner.last_inputs`) is recorded per arm and must show the perturbed slot values exactly
  (AX0 slots 2,3 == 0; AXP slot 2 == +0.5; AXM == -0.5; DEAD slots 4,5 == 0.3, 0.2; every other slot == A0's); GREY image std == 0 and mean == 128/255; SWAP image == donor's image bytes.
  Failure of any => arm invalid, reported as such.
* C4 trunk memo: ego-only arms reuse the image trunk output of A0 when (and only when) the input image tensor is bit-identical (`torch.equal`). A0r bypasses the memo, so C1 covers it.
* C5 ego-slot audit: fraction of tokens with non-zero real yaw_rate / steer, and the real ax, ay distribution.

## Readouts
For every arm vs A0, on the PICKED trajectory (20 poses @0.2 s, ego frame, position only):
* mean ADE between the arm's pick and A0's pick (m)
* planned speed v@T, T in {1,2,4} s, = one-step backward finite difference |p_i - p_{i-1}|/0.2 with i = T/0.2, p_0 = origin (secondary: 0.5 s window on the NAVSIM 8-pose grid)
  -> delta = v_arm - v_A0 (m/s)
* delta of the 4-s chord length |p_20| (m); delta of the max lateral offset max_i |y_i| (m)
* selection vs generation (descriptive): same deltas on the MEAN over all 64 proposals' v@T (does the picker or the generator move?) and the share of tokens whose pick index changes.
Readouts vs GT (logged future, scenario ego states; GT v@T uses the SAME one-step chord estimator with the states' true timestamps, so estimator bias cancels):
* b_arm = mean over tokens of (planned v@4s - GT v@4s) [m/s], on ALL tokens and on CRUISE
* CRUISE = v0 > 5 m/s AND |ax0| < 0.3 AND |GT v@4s - v0| < 1 m/s (v0 = real ego speed slot 6, ax0 = real slot 2). Fixed here, not tuned.
* s_arm = fraction of CRUISE tokens with planned slow-down > 15 % : (v0 - planned v@4s)/v0 > 0.15; GT's own fraction reported as the reference.

## Estimator and what the interval answers
Every interval = **paired episode(log)-cluster bootstrap**: resample the LOGS with replacement (B = 2000, seed 20261106), recompute the statistic, same resample for every arm (paired), 95 % percentile CI.
It answers *"would another draw of logs/frames say this?"* ONLY. It is blind to training variance (one checkpoint, no replicate training run) and to inference variance except via C1
(the planner is deterministic: no sampler). So no claim here is about a trained-lever effect; claims are about THIS checkpoint's input dependence.

## Families (EVAL_DOCTRINE four-family rule; per-family, never pooled)
* LONGITUDINAL: delta v@1/2/4, delta chord, b_arm (plan v@4 - GT v@4), s_arm.   * LATERAL: delta max lateral offset (and ADE between picks, reported as joint).
* TACTICAL: n/a - REFe emits 64 trajectory proposals + 7 scorer logits and has no manoeuvre-decision/class output to compare; only the pick-index change share is reported (descriptive).
* STRATEGIC: PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25) - not a pass.

## COMMITTED DECISION TEXT (thresholds fixed now; the analysis script evaluates them mechanically)
Let CI-hi(.) be the upper end of the 95 % cluster-bootstrap CI.
* **R2 (policy slow bias in distribution)** holds iff b_A0(CRUISE) <= -0.5 m/s AND CI-hi(b_A0) < 0.  => "A0 itself plans a CRUISE slow-down: the policy has a slow bias even in distribution."
* **R1 (policy leans on ego/images; sim slow-down partly policy)** holds iff |b_A0(CRUISE)| < 0.5 AND for AX0 or GREY: (b_arm - b_A0) <= -0.5 with paired CI-hi < 0.
  => "A0 tracks GT on CRUISE while AX0/GREY make it slow down."
* **R3 (sim-side inputs)** holds iff |b_A0| < 0.5 AND |b_AX0| < 0.5 (both track GT within 0.5 m/s). => "the sim's slow-down comes from sim-side inputs."
* Reported verdict: R2 if R2; else R1 if R1; else R3 if R3; else UNDECIDED (state which condition failed, with numbers). R1 and R3 can both hold (AX0 tracks, GREY slows): then both are stated.
* Added by the agent before the run (labelled ADDITIONAL, not part of the brief's three readings): **R4 (dead slots alone)**: DEAD moves b by <= -0.5 m/s with paired CI-hi < 0 => "the untrained yaw/steer slots fed non-zero are sufficient to produce part of the slow-down";
  report DEAD's ADE-from-A0 and delta v@4 regardless. And the ax sensitivity: (v@4 under AXP) - (v@4 under AXM) per m/s^2 reported as a gain.
* No threshold is changed after the data are seen. A result that misses a pre-registered bar is reported as written.

## Token subset
Preferred: all 1,058 C3 tokens. If sized GPU time exceeds ~90 min: ALL tokens that satisfy CRUISE on the W3-export inputs (v0 from `ego_statuses[-1].ego_velocity`, ax0 from `.ego_acceleration[0]`,
GT v@4 from `human_future_poses` last 0.5 s chord, same 3 conditions; CRUISE count on the 1,058: **155 tokens over 53 logs**, measured at write time) PLUS a seeded (20261106) draw of the
remaining tokens, at most 3 per log, to reach the budget. The subset file and its sha256 are written to `tokens_subset.json` before launch. Readouts on "ALL" are for the probe subset and are
enriched in CRUISE by design; the CRUISE readout is the decision readout. Strata (`zero` = banked u400 PDMS exactly 0, `random` = non-zeroed) are reported as a descriptive split.

## Limits stated now
Real nuPlan frames are in-distribution for rendering and for ego slots 0-3,6; this probe cannot test the sim's image domain gap, its actual (jittered) accel values, or any history effect
(the policy has no history input). A clean result here localises the slow-down to sim-side inputs but does not by itself identify which sim input.

## ADDENDUM A1 (2026-11-06, after the 8-token smoke, BEFORE the full run) -- instrument changes only, no decision rule touched
* Smoke 1 (8 tokens / 4 logs): C1 8/8 bit-identical, C2 8/8 logits bit-identical to `table.npz`, picks equal, executed 8-pose proposals max |d| 0.0 m. C3 flagged DEAD 0/8: a defect in the CHECK (it compared the float32 model input to the float64 literal 0.3 / 0.2), not in the arm; fixed by comparing against float32(value).
* Sizing: steady state ~6.5 s/token (4 ViT-L trunk calls at ~1.13 s: A0, A0r, GREY, SWAP) => 1,058 tokens ~ 115 min, over the ~90 min bar. Two exact savings, both guarded by controls: (i) the GREY trunk output is the same for every token (constant image), cached in its own memo slot, every 25th grey hit recomputed fresh and compared bit-for-bit (C4b, reported); (ii) the A0r determinism replicate runs on every 4th token (first token included) instead of every token (C1 reports its n). Expected ~3.6 s/token => ~65 min for all 1,058 tokens, so the ALL-tokens option of "Token subset" is used (no subsetting).
* The analysis ignores A0r as an arm (it is a control; its rows on tokens where it did not run are copies of A0).
