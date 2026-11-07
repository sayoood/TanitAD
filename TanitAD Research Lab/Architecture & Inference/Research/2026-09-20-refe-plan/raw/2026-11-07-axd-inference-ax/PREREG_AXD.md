# PREREG_AXD: break REFe u400's ax loop gain without losing open-loop PDMS (WP YAW, stream YAXD, 2026-11-07)

**Status: FROZEN when its sha256 is written to `PREREG_AXD.sha256`, before any arm is evaluated.** Smoke runs (67 + 8 tokens, whole A5 logs) are mechanics only, never decision data. Code is pinned in `CODE_PIN_AXD.json` after the smoke and before the full chain (the chain's `pins` stage REFUSES on a mismatch). Any later deviation is an `AMENDMENT_n.md` with its own sha256. Evidence classes: MEASURED (artifact named) / INHERITED / ESTIMATED / HYPOTHESIS.

## 0. Why, mechanism, tier
* Closed-loop stall (scenario 720, INHERITED from the MM brief): u400 feeds its own realised acceleration back through ego slot 2 (`ax`; `refe/planner.py:448-451` builds vx, vy, ax, ay, yaw_rate, steer, speed; `refe/model.py:185`). Withholding ax removed the stall (EPDMS 0.748 -> 0.789, INHERITED).
* ax gain (MEASURED, T0): +1.10 [+0.92, +1.30] m/s of planned v@4s per m/s^2 on CRUISE (`refe_ego_probe_20261106/RESULT.md`).
* YAX X1 REFUTED (MEASURED, `YAX/run_full/analysis/X1.md`): a decoder fine-tune with ax = 0 costs -3.73 [-5.11, -2.40] PDMS against the same fine-tune with real ax. So ax carries real information; the lever must keep it and break the LOOP GAIN.
* **Tier.** PDMS and families on the A5 tokens = NAVSIM v1.1 pseudo-simulation of an OPEN-LOOP plan against logged agents (T0-family). Open loop has no ax feedback (the logged ax is the human's). It shows what a transform COSTS and how far it cuts the plan's ax gain. **It is never proof that the stall is fixed.** The deciding read is the closed loop (section 7).

## 1. Frozen inputs
| input | value |
|---|---|
| model | u400 `E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt`, sha256 `8b4dba4818b5cc96c640fe892efe114037ace7cec9dda060c5b4cdd5fb92410d`, used in place (no new checkpoint) |
| tokens | the 923 A5 confirmation tokens `data/refe_navtest/a5_confirm_tokens.json` (43 logs), as YAX |
| seam | the production `eval/refe_navtest_seam.py::main`, `--rule navsim_v1_lane`, default goal fix and Amendment-7 repair, unchanged; harness `score_e.py` (W3, pinned by `harness_pins.require`), families6, `ax_perwindow.py`, all through YAX's `ax_eval_arm.py` functions (read-only import) |
| history | MEASURED 2026-11-07: the W3 export's NAVSIM agent input carries **4 ego statuses at 0.5 s** (t = -1.5, -1.0, -0.5, 0) with `ego_acceleration` on 923/923 A5 tokens (`refe_navtest_seam.py:93` reads the same `ego_statuses`). The seam's own ego is the nuPlan DB state at iteration 0 (`refe_navtest_seam.py:243`, `history_rows=1`) |

## 2. Arms (P1, inference only, zero training)
Every arm changes ego slot 2 ONLY, at the planner's single ego builder `REFePlanner._ego_vec` (`planner.py:440`), via a class-level wrap in `axd_seam.py`. With `r` = the planner's own float32 ax:
| arm | fed ax |
|---|---|
| A0 | `r` (the seam's own, unwrapped call, fresh trunk; the PAIRED REFERENCE) |
| IDT / IDTm | `r` through the transform machinery (fresh trunk / trunk memo): CONTROLS, must equal A0 bit-for-bit |
| AX0 | 0.0 (the sim's inherited withhold; judged by the same rule, reported as the reference transform) |
| CLAMP0 | max(r, 0): no deceleration feedback, acceleration kept |
| CLAMPm05 | max(r, -0.5): deceleration feedback capped at -0.5 m/s^2 |
| SCALE05 | 0.5 r: the whole loop gain halved, sign information kept |
| SMOOTH4 | mean of the 4 ax samples (export t = -1.5, -1.0, -0.5; DB `r` at t0). HYPOTHESIS, committed now: a low-pass does NOT cut the SUSTAINED gain, so SMOOTH4 is predicted to FAIL the gain clause |

**Gain arms (CRUISE tokens only):** for every transform T, the raw ax is perturbed by d = +-0.5 m/s^2 BEFORE T (in closed loop the perturbation enters the realised ax): `T@+`, `T@-`. SMOOTH4 also `@+s`/`@-s` (all four samples move: the sustained perturbation). IDT also `@abs+`/`@abs-` (ax SET to +-0.5, the probe's literal instrument, `refe_ego_probe.py:312`).

## 3. Gates (any red voids the affected arm(s) -> VOID). Each has a deliberate-regression arm that must read RED, run end to end on the real seam in the smoke (8-token log, `--regress`) and in `test_axd.py`
| gate | check | RED arm |
|---|---|---|
| G-SEAM | seam rc 0, misses 0, every asked token has a record | (seam's own frame-control gate) |
| G-IDT | IDT's fed ego == A0's ego (all 7 slots) and IDT's plan == A0's plan, bit-for-bit, every token; A0 record == the seam's written file. Also MEASURES Q3 (a fresh re-run reproduces A0) | `idt_perturbed` |
| G-MEMO | IDTm (memo trunk) fed ego and plan == A0 bit-for-bit | `memo_stale` |
| G-REACH:T | slot 2 fed == T(r) exactly on every token; slots 0,1,3-6 == A0's exactly; >= 1 token changed; every gain arm's fed value == T(r +- d) exactly | `transform_not_applied` |
| G-REACH:IDTgain | IDT gain arms fed r +- d and +-0.5 exactly | `transform_not_applied` |
| G-ZERO | AX0@+, AX0@- and AX0 plans bit-identical on every CRUISE token (the gain instrument reads EXACTLY 0.0 on an input-independent transform) | `ax0_leak` |
| G-POS | u400's own (IDT) down-gain CI-lo > 0.5 (the instrument can see a gain) | none (a failure is the finding) |
| G-HIST | max over tokens of abs(export t0 ax - DB ax) <= 1e-3 (SMOOTH4 only; red -> SMOOTH4 VOID) | `test_axd.py` |
| G-HARNESS:arm | harness PASS, every token scored | (harness's own) |
| R-A0BANK (reported, never voiding) | A0 here vs the banked seam `seams/stage2_sft4_B_u400.npz` (bit-identical?) and vs the banked CSV per token | - |

## 4. Endpoints and estimator
* **Estimator:** paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927 (`ax_stats.Boot`, YAX's), one resample matrix for all arms; CRUISE statistics on a Boot of the CRUISE tokens' logs. Q1 (another draw of logs) only. Q2 (another training run) does not apply: nothing is trained. Q3 (another inference run) is MEASURED by G-IDT.
* **PRIMARY, PDMS:** Delta_T = mean over tokens of PDMS x100 (harness `score`), T - A0, paired.
* **GAIN (the mechanism):** v4 = |p8 - p7| / 0.5 on NAVSIM's 8 poses, origin prepended (YAX's `plan_speed`). Per CRUISE token: down-gain `g_dn = (v4(T) - v4(T@-)) / 0.5`; also `g_up`, `g_sym = (v4(T@+) - v4(T@-)) / 1.0` (the probe's form). SMOOTH4 uses the SUSTAINED arms in `g_dn` (instant reported).
  * **Sharpening of the suggested bar, with reasons (committed before any data):** (1) the bar uses the DOWN-gain, because the stall loop runs through ax < 0 (plan slows -> realised a < 0 -> slower plan); a clamp that removes only the deceleration side cuts that loop while its symmetric gain stays near half; `g_sym` and `g_up` are reported. (2) Added RELATIVE clauses against u400's own down-gain measured in THIS run on THESE tokens, because the probe's 1.10 was measured on 154 C3 CRUISE tokens on REFe's 0.2 s grid: an absolute bar alone could be met by the definition change instead of the transform.
* **CRUISE** (YAX's definition, `ax_analyze.cruise_inputs`): v0 > 5 m/s, |ax0| < 0.3, |v_gt(4 s) - v0| < 1.0; MEASURED on the export (no model output) n = 90 tokens / 29 logs on the 923. Floor: n >= 60 and logs >= 15, else the gain clause is UNASSESSABLE (cannot be CANDIDATE).
* **LONGITUDINAL (in the rule):** `D_lon = |mean_CRUISE b(T)| - |mean_CRUISE b(A0)|`, b = v4(plan) - v4(GT) (YAX's guard). Reported: speed bias / MAE at 4 s (all), along-track MAE and bias (perwindow), headway/TTC n/a (no lead-agent headway in the seam; PDMS TTC sub-score is the proxy).
* **LATERAL (reported, paired with CI):** curvature MAE, heading MAE, yaw-rate MAE, cross-track MAE (`ax_perwindow.py`).
* **TACTICAL:** n/a as a test, REFe has no tactical layer. **STRATEGIC:** PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25), not a pass, not omitted.

## 5. Committed rule, per transform T (mechanically, `axd_analyze.py::decide`)
1. Any red gate that covers T -> **VOID**.
2. **CANDIDATE FOR CLOSED LOOP** iff ALL: (a) PDMS non-inferiority: CI-lo(Delta_T) > -0.5; (b) g_dn(T) <= 0.55; (c) CI-hi(g_dn(T)) < 0.92 (u400's probe CI-lo); (d) CI-hi(g_dn(T)) < CI-lo(g_dn(IDT)) of this run; (e) g_dn(T) <= 0.5 x g_dn(IDT); (f) D_lon <= +0.3 m/s; (g) CRUISE floor met.
3. **REFUTED** iff CI-hi(Delta_T) < -0.5 (the cost certainly exceeds the margin) OR CI-lo(g_dn(T)) > 0.55 (the gain is certainly not cut).
4. Otherwise **NOT PROVEN**, failing clauses named.
CANDIDATE means only "in open loop this transform costs <= 0.5 PDMS and cuts the deceleration gain to <= half". It does not say the stall is fixed.

## 6. Chain and Rule Zero
`axd_chain.py --tag full` starts GPU work only after `YAX/markers/yax_full.DONE` or `.FAILED`; if YAX's `status.json` reads X2 = CANDIDATE it exits SKIPPED without running. One GPU-lock hold (YAX's lock protocol, never forced) for the seam; then harness, families6, perwindow per PDMS arm (A0 + 5), then the analysis.
**If no P1 transform is CANDIDATE -> P2 (Rule Zero):** AXNOISE, the YAX recipe (X1's selected recipe K1 = 200 steps, lr 5e-5, DEC set, seeds 0 and 1, the identical row permutation per seed) with the training ax replaced by `ax + N(0, sigma^2)`, sigma = 1.0 m/s^2, drawn from a dedicated generator (seed + 7919) so the row order equals AXON's; fed honestly at inference. HYPOTHESIS (errors-in-variables attenuation): with var(ax) = 0.81^2 (MEASURED on the A5 export t0, std 0.813) the learned gain shrinks to about var/(var + sigma^2) = 0.40 of u400's. Control: YAX's banked X1 AXON s0/s1 evals (same recipe, seed, permutation). Gates: sigma = 0 reproduces AXON's 5-step weights bit-for-bit (G8-style); the training batch's fed ax minus real ax has std within 10 % of sigma while gate/VAL batches are honest (exactly 0 difference); the YAX gates (G-REACH, G-DECL, G-CKPT). Rule: section 5 with A0 replaced by AXON_s at each seed (both seeds must pass), plus the gain clauses read on the AXNOISE checkpoint itself. **P2 is NOT BUILT in this run** (named blocker: implementation; see HANDOFF).

## 7. Closed loop (MM, sim session; unchanged from PREREG_AXOFF section 6c)
For each CANDIDATE transform (applied at the planner's ego builder, ax fed honestly before the transform): T1-sim scenario 720, same settings as 720p/720q. **ADOPT iff** no stall (minimum speed > 1.0 m/s before the stop line while the arrow is green) AND EPDMS >= 0.748. 0.789 (u400, ax withheld) is the reference to read against, not the bar. Scope: scenario 720 only. A CANDIDATE that fails is REFUTED for deployment; the next lever is P2.
