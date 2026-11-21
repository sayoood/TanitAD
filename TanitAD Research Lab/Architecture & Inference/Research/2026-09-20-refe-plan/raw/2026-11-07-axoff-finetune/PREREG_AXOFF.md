# PREREG_AXOFF: can REFe u400 plan without conditioning on its own acceleration? (WP YAW, stream YAX, 2026-11-07)

**Status: FROZEN when its sha256 is recorded in `PREREG_AXOFF.sha256` (written BEFORE any decision arm is evaluated). No threshold or rule below moves afterwards; an implementation deviation is written as `AMENDMENT_n.md` with its own sha256.** Code is pinned separately in `CODE_PIN.sha256` (written after the CPU gates pass, before the first full-run arm is trained; the pipeline's `pins` stage REFUSES on a mismatch). Smoke arms (4 steps, ~110 tokens) are mechanical checks of the pipeline and are NOT decision data.
Evidence classes: MEASURED (ours, artifact named) / INHERITED (MM brief, not re-verified here) / ESTIMATED / HYPOTHESIS.

## 0. Question, mechanism, tier
Closed-loop sim (TanitScenarioFlyWheel, 720p vs 720q; INHERITED from the MM brief): REFe u400 stalls at a junction because ego slot 2 (`ax`, the 7-D ego vector is vx, vy, ax, ay, yaw_rate, steer, speed; `refe/model.py:185`, `refe/planner.py:448`) feeds its own plan back: plan slows -> realised a < 0 -> next plan slows more. Withholding ax removed the stall (EPDMS 0.748 -> 0.789; INHERITED). ax gain on nuPlan: +1.10 m/s of v@4s per m/s^2 (INHERITED, `raw/2026-11-06-refe-ego-probe`).
**Question.** Does a u400 fine-tuned with ax fixed to 0.0 (in training AND at inference) keep open-loop PDMS and CRUISE speed accuracy relative to the SAME fine-tune with the real ax? If yes it is a CANDIDATE for the closed-loop test; the closed loop is the deciding read (section 6c).
**Tier.** PDMS and families6 on the 923 A5 tokens = NAVSIM v1.1 pseudo-simulation of an OPEN-LOOP plan against logged agents (T0-family, RESULT_SFT4 says so). NOT the action-closed-loop tier. Cached-frame VAL diagnostics = T0 teacher-forced. Nothing here may be quoted as "drives better" or as "the stall is fixed".
**Open-loop blindness (HYPOTHESIS).** The feedback loop that causes the stall does not exist in open loop (the logged ax is the human's), so open loop can only show what removing ax COSTS, never that it fixes the stall.

## 1. Frozen inputs
| input | value |
|---|---|
| base A0 | `E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt`, sha256 `8b4dba4818b5cc96c640fe892efe114037ace7cec9dda060c5b4cdd5fb92410d` (YF2 `g3_fingerprint.json`) |
| cache | `YF2/cache_u400_4k/` (u400's own frozen encoder, 4,000 frames), `manifest.json` sha256 `86b2f2222bcfccf762b2b6c1a7bc3aec6568289db68e138b52cade1a70a1c8da` (`YF3/CACHE_PIN.txt`, verified) |
| frame manifest / split | `YF2/manifest_yaw_4k.json` sha256 `18acb1d42645b42dcebd977c66fe7cada60ee8b01b75f87aa59c6d86158077cc`: 316/79 logs TRAIN/VAL, 3,244/756 frames, ranks 0 and 1 both trained (as YF2) |
| test tokens | the 923 A5 confirmation tokens (43 navtest logs), disjoint from the cache logs (asserted by YF2's `make_manifest_yaw4k.py`) |
| A0 reference | the BANKED A0 seam/CSV (`data/refe_navtest/seams/stage2_sft4_B_u400.npz`, PDMS 88.6473 x100), passed through the SAME per-window code. Its bit-for-bit reproduction (G6) is YF3's gate; here it is INHERITED. A0 is a descriptive row only, never in a decision |

## 2. Arms (same cache, same tokens, same seeds 0 and 1, same batch order per seed: both arms of a seed draw the identical row permutation)
| arm | ego slot 2 in training | at inference | trained |
|---|---|---|---|
| A0 | real (as u400) | real | nothing |
| **AXOFF** s0, s1 | exactly 0.0 (float32) in every row; column 2 of `ego_enc.0.weight` set to exactly 0.0 and its gradient masked to exactly 0.0, so it stays 0.0 | 0.0 by policy; ANY fed value gives the bit-identical plan because the column is 0.0. Checkpoint meta `ego_ax_policy: "zero"`, `ego_ax_slot: 2` | DEC set |
| **AXON** s0, s1 | the real ax of the bank row, untouched | real | DEC set (column 2 trains) |
Yaw slots 4/5 stay 0 as in u400 (rows carry 0). **DEC set** = `ego_enc.*` (AXOFF: minus column 2), `queries`, `dec.*`, `traj_head.*` (YF2 `yaw_common.DEC`; ~4.47 M parameters); the scorer, backbone, scene projection, goal frequencies are frozen and read from u400. Scorer frozen as in YF2.
**Optimiser (both arms, every seed):** AdamW (beta 0.9/0.999, eps 1e-8), weight decay 0, cosine to 0, no warm-up, grad-norm clip 1.0, batch 32 x accumulation 8 (256 rows/step), loss `model.wta_loss` (P2's live loss, yaw_wrap on), CPU fp32, `torch.manual_seed(seed)` and `Generator(seed)` for the permutation (YF2's code path).
**Recipe candidates (frozen):** K1 = 200 steps, lr 5e-5 (YF2's R3, verbatim). K2 = 400 steps, lr 1e-4.
**Selection (pre-registered, cache VAL only, never the test tokens):** train AXOFF s0 under K1 and K2; the recipe with the lower `val_final.loss` (VAL rows, ax = 0) is X1's recipe, unless the relative difference is < 1 % in which case K1. All four X1 arms use that one recipe.

## 3. Gates (all before any statistic; a red gate voids the read -> verdict VOID). Each has a deliberate-regression arm that must read RED
| gate | check | RED arm |
|---|---|---|
| GA-FEED | AXOFF: ego[:,2] == 0.0 exactly on every gate row; AXON: ego[:,2] equals the bank row's ax and has rms > 0.3 and std > 0 on the gate rows (else AXON == AXOFF trivially) | `--regress ax_fed_in_zero` (AXOFF fed the real ax), `--regress ax_not_fed` (AXON fed 0) |
| GA-COL | AXOFF: column 2 of `ego_enc.0.weight` == 0.0 (all 256) before and after training | `--regress keep_ax_col` (u400's column left in) |
| GB-STEP0 | AXON at step 0: the training forward on real-ax rows is `torch.equal` to u400's deployed decode on >= 512 rows | `--regress ax_not_fed` |
| G-REACH | gradient abs-sum > 0 on each declared group (ego_enc excluding column 2, queries, dec, traj_head); ZERO CONTROL: column-2 gradient with ax = 0 reads EXACTLY 0.0; POSITIVE CONTROL: column-2 gradient of u400's own column with the real ax reads > 0 (the probe can see a gain) | `--regress ax_not_fed` makes the AXON column-2 gradient 0 |
| G-DECL | every trainable tensor name starts with a DEC prefix; every other parameter has requires_grad False; after training the composed checkpoint differs from u400 ONLY in DEC keys (K1/K2 in `ax_build_ckpt.py`) | a final.pt carrying a scorer tensor -> K red |
| GA-GAIN | the COMPOSED AXOFF checkpoint file, reloaded: decode with ax in {0, +0.5, -0.5, real} is `torch.equal` on >= 512 VAL rows (gain exactly 0). POSITIVE CONTROL: u400 itself decodes differently at ax = +0.5 vs 0 | the checkpoint of a `keep_ax_col` run: not equal |
| G-CKPT | the composed file's DEC tensors `torch.equal` the trained final.pt; meta policy matches the arm | tampered file |
| G8-mini | two 5-step runs of the same arm+seed produce `torch.equal` weights (CPU determinism of this code path; the full-run rerun is NOT done: stated) | none (a failure is the finding) |

## 4. Endpoints, estimator, variance
Unit of the paired contrast: AXOFF_s minus AXON_s at the SAME seed s (isolates "no ax" from "more fine-tuning"), on the same 923 tokens. **Estimator:** paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927, one resample matrix for all arms (the SFT-4 / M6 estimator; `yf3_stats.Boot` copy). It answers "would another draw of LOGS say this" only; seeds s0/s1 answer "another training run" (floor F = |Delta(s0) - Delta(s1)|); inference is deterministic (ego probe C1, INHERITED). Not measured: another BASE training run.
* **PRIMARY (PDMS, T0-family):** Delta_s = mean over tokens of PDMS x100 (harness `score`), AXOFF_s - AXON_s. Non-inferiority margin -0.5 PDMS: holds at seed s iff the lower 95 % CI bound of Delta_s > -0.5.
* **LONGITUDINAL (guard in the rule):** CRUISE subset of the 923 tokens, the NAVSIM analogue of `analyze_ego_probe.py`'s CRUISE: `v0 > 5.0 m/s` AND `|ax0| < 0.3 m/s^2` AND `|v_gt(4 s) - v0| < 1.0`, with `v0 = hypot(ego_velocity)`, `ax0 = ego_acceleration[0]` of the last ego status in `navtest_inputs.json.gz`, and `v(4 s)` the one-step chord `|p8 - p7| / 0.5` of the 8 poses at 0.5 s (origin prepended), identically for GT (`human_future_poses`) and the plan (seam `poses`). MEASURED on the export before freezing (no model output): n_cruise = 90 tokens over 29 logs. Bias `b = v_plan(4 s) - v_gt(4 s)` (m/s, signed). Seed-wise statistic `D_lon,s = |mean_CRUISE b(AXOFF_s)| - |mean_CRUISE b(AXON_s)|`. Floor: if n_cruise < 60 or logs < 15 the condition is UNASSESSABLE and the verdict cannot be CANDIDATE. Reported with its bootstrap CI; the rule uses the POINT estimate (MM's wording has no interval), at EACH seed.
* **LATERAL:** curvature MAE (1/m), heading MAE, cross-track MAE, yaw-rate MAE via `yf3_perwindow` (families6's own geometry), paired AXOFF - AXON, with CI. Descriptive, not in the rule.
* **TACTICAL: n/a** as a test: REFe has no tactical layer; families6 derives decisions from the executed geometry (reported descriptively, `fam_adverse`). **STRATEGIC: PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25): not a pass, not omitted. Also reported: PDMS sub-scores, share of tokens whose pick differs, speed MAE (ALL), along-track MAE, headway/TTC n/a (the NAVSIM seam carries no lead-agent headway; the PDMS TTC sub-score is the proxy). Stamp on every table: tier above.

## 5. Committed rule (X1; mechanically evaluated by `ax_analyze.py`)
1. Any red gate -> **VOID** (not a verdict).
2. **REFUTED** iff the seed-averaged Delta's 95 % CI upper bound < -0.5 AND both per-seed point estimates < -0.5.
3. **CANDIDATE FOR CLOSED LOOP** iff (a) PDMS non-inferiority holds at BOTH seeds (lower bound of Delta_s > -0.5) AND (b) `D_lon,s <= +0.3 m/s` at BOTH seeds AND the CRUISE floor is met.
4. Otherwise **NOT PROVEN**, with the failing clause named.
CANDIDATE means only "removing ax costs <= 0.5 PDMS and <= 0.3 m/s CRUISE bias in open loop". It does not say the stall is fixed.
**Rule Zero (no stopping at a verdict):** if X1 is NOT PROVEN or REFUTED (gates green) the pipeline runs **X2** = the OTHER recipe candidate (AXOFF s0 under it is already trained by the selection stage; AXON s0, AXOFF s1, AXON s1 are trained) with the identical rule. If X2 also fails the chain ends with the named levers: (i) ax-noise/dropout augmentation that keeps ax but breaks its loop gain, (ii) P7 from scratch without the ax slot, (iii) a larger cache (4,000 frames is the capacity limit; ~1.1 s/frame GPU). Each blocker and what unblocks it is written to `FINAL.md`.

## 6. Closed loop (pre-registered here, run by MM in the sim session)
a. Deliverables for the sim: `ckpt/AXOFF_s0.pt` (and `AXOFF_s1.pt` as the seed replicate), meta `ego_ax_policy: "zero"`. Deploy code may feed the honest ax: the zero column makes the plan independent of it (GA-GAIN).
b. Test: T1-sim, scenario 720 (TanitScenarioFlyWheel), ax fed honestly, one run per sent checkpoint, same scenario/seed settings as the 720p/720q baseline.
c. **ADOPT iff** no stall (minimum speed > 1.0 m/s before the stop line while the arrow is green) AND EPDMS >= 0.748 (u400 with ax fed honestly). 0.789 (u400 with ax withheld) is the reference to read against, not the bar. If two checkpoints are sent, both must pass. Scope of an ADOPT: scenario 720 only (n = 1 scenario, a deterministic-sim single trajectory); broader deployment needs more junction scenarios.
d. A CANDIDATE that fails (c) is REFUTED for deployment; the next lever is (i) above.
