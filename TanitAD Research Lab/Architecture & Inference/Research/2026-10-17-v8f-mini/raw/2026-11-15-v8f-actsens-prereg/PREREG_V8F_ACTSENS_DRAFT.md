# PREREG_V8F_ACTSENS (DRAFT, agent 2026-11-15; the Master Mind freezes it): raise the ladder world model's sensitivity to actions, then read the T0 planner rows again

**Status.** DRAFT. When this was written, no ACTSENS code, run or scored window existed. Box-2 training needs four things: the Master Mind's freeze line in the Log below, the code of `REACHABILITY_ACTSENS.md` landed on one commit, a PASS token for each run, and the box-2 go. The PI's go covers carding (MM brief 2026-11-15, INHERITED). Nothing here was launched.

**Tier.** Every planner row is **T0**: the goal is the encoded true-future latent, and the planner is frozen with zero planner training. T0 rows are a world-model diagnostic and never measure driving performance. HOLD is the T1-kin reference.

## 1 Why (measured; each number carries its artifact)
* **Planning through R35 is worse than holding the action** (`hwm_plan/VERDICT_HWMPLAN_PHASE1.md`, MEASURED, s0 / s1, 994 SCORE windows, 71 episodes):
  * HWM-ORACLE − HOLD, full 6 s, acceleration MAE: +0.5450 [+0.4907, +0.5977] / +0.4541 [+0.3914, +0.5138] m/s^2.
  * 0-2 s band, where no decode runs: +0.9907 / +0.8917 m/s^2.
  * Yaw-rate MAE, full 6 s: +0.0671 / +0.0634 rad/s.
  * FLAT-ORACLE − HOLD, full 6 s: +0.7892 / +0.7154 m/s^2.
  * The decode-free stitch `l1_full` is also worse than HOLD (`VERDICT_HWMPLAN_L1FULL.md`): +0.7664 / +0.6061 m/s^2, full 6 s.
* **Level-2 plans leave the encoder's image** (`hwm_plan/AMEND_HWMPLAN_2.md`, MEASURED). The planned macros' median nearest-neighbour distance to the TRAIN macro cloud is 7.16 / 7.21. Held-out in-image macros give 1.06 / 1.06, so the ratio is 6.77 / 6.79. 94 % of planned decodes sit at the action bounds, against about 17 % of in-image decodes.
* **The optimiser works and still loses** (`egowm/read/VERDICT_EGOWM_PLAN.md`, MEASURED). `frac_cost_lowered_L1` = 1.0 in all 48 rows. The lower CI bound of G-TRUE − HOLD (LONGITUDINAL) is at least +0.4464 m/s^2. `slowwm/read/VERDICT_SLOWWM.md` finds the same NO-GAIN over three z1 bases.
* **Level 1 barely uses the action, and composing it is below copy-last** (register rows, mirror tip `bb3825f8`, `Project Steering/GOALS_AND_CLAIMS.md`; all on M2b, eval-141, MEASURED):
  * `H-V8F-WMACT-1` ACTION-USE k1: L1 +0.0054 / +0.0049, L2 +0.0424 / +0.0462, L3 +0.1850 / +0.2000. L1 is 8x below L2 and 35x below L3.
  * The L1 step is aimed right but about 2x too large (shrink c 0.496 / 0.526).
  * `H-V8F-PROOF-3 WMCEIL`: at 2 s, ten composed 0.2 s steps give −0.1500 / −0.1724, and one direct jump gives +0.1225 / +0.1292.
* **R35's own action response (IA-3)** (`ladder_read/RESULT_LADDER_READ.md`, MEASURED, gain = decoded effect / physical effect, 1.0 = right size):
  * L1 2.0 s: lateral gain −0.038 [−0.049, −0.028] / −0.036 [−0.045, −0.027], which is zero with the wrong sign. Longitudinal gain 0.384 / 0.301, about 3x too small.
  * Lateral gain at L2 6 s: 0.142 / 0.131. At L3 12 s: 0.317 / 0.302.
  * In the planning form (context 1), no ladder arm beats Copy-Last at L1 2.0 s (`VERDICT_LADDER.md`).
* **This card's probe on R35** (`PROBE_ACTSENS_RESULT.json` beside this file, script `probe_actsens.py`, dev-box CPU; MEASURED on 2,397 TRAIN windows of 141 TRAIN clips, 31,575 transitions):
  * Level-1 forward response in the planning form: a +1 sd step of one action moves the predicted next latent by 8.7 / 8.8 % (acceleration) and 10.3 / 9.8 % (yaw rate) of the real one-step latent change. The predictor's own one-step error is larger than the real step (2.498 against 2.378 at s0).
  * The level-1 IDM head explains R^2 0.489 / 0.514 of acceleration and 0.637 / 0.629 of yaw rate.
  * The IDM targets are not unit-variance: in the WM's action units the variance is 0.1767 (acceleration) and 0.0561 (yaw rate).
* **The IDM weight R35 trained under** (`C:/Users/Admin/t2v8f_pulls/ladder_R35_s0/train_log.jsonl`, step 19,164, MEASURED):
  * gamma x idm/L1 = 4.60 of level/L1 5.10. The L2 and L3 IDM terms are 73.2 of 73.9 and 75.2 of 76.0, so L1's IDM term is 3.0 % of the world model's loss (154.98).
  * The loss identity recovers gamma = 100.0000 at every one of 384 logged steps, at all three levels, at both seeds (`gamma_from_log.py`; min 99.99998, max 100.00002).

**Reading (Master Mind, `VERDICT_HWMPLAN_L1FULL.md`).** The optimiser exploits the world model's action response. A level-1 latent that moves by about a tenth of its own error per action-sd can be steered to a goal only by extreme actions, and extreme actions are what the plans contain. The card's PLANNER-LIMITED path ends at "world-model action sensitivity (inverse-dynamics weight, action-diverse sampler)". This card runs that step.

## 2 Arms (WM-only on box 2; adapter, readers and planner frozen at M2b s{k}, exactly as R35)
Base argv: `C:/Users/Admin/t2v8f_pulls/ladder_R35_s{k}/argv.txt`. That is seed k, `--wm-k 1 --wm-strides 3 5 --wm-rollout-k 10 10 6 --wm-rollout-stride 5 5 1 --wm-ctx 3 3 3 --wm-only --wm-windows wmonly --wm-pooled on --steps 19164 --steps-mult 4 --kin-mode yaw`, with `--init-ckpt` and `--wm-zcache` of chassis M2b_s{k}. Levers live at `wm b` (beta 0.08, gamma 100). Every arm runs on ONE commit, the ACTSENS commit, with both levers default-off.

| order | tag | chassis (init-ckpt, zcache) | seed | argv change from R35_s{k} | purpose |
|---|---|---|---|---|---|
| 0 | PILOT | M2b_s0 | 0 | DIV_s0's argv + `--steps 200` | projection; measures DIVRS start-up |
| 1 | IDM_s0 | M2b_s0 | 0 | `+ --wm-idm-gamma 1000 100 100` | lever 1 |
| 2 | NULL_s2 | M2b_s0 | 2 | seed 0 → 2 only | training-run floor, chassis 0 |
| 3 | DIV_s0 | M2b_s0 | 0 | `+ --wm-sampler divrs` | lever 2 |
| 4 | IDM_s1 | M2b_s1 | 1 | `+ --wm-idm-gamma 1000 100 100` | lever 1, second seed |
| 5 | DIV_s1 | M2b_s1 | 1 | `+ --wm-sampler divrs` | lever 2, second seed |
| 6 | NULL_s3 | M2b_s1 | 3 | seed 1 → 3 only | training-run floor, chassis 1 |

* **Why a new seed for NULL.** Box-2 WM-only training is bit-identical at a fixed seed. MEASURED: R35b_s0's weights equal R35_s0's (`ladder_read/VERDICT_LADDER.md`), so a same-seed replicate reads a floor of exactly 0. NULL keeps the chassis and changes only the WM seed. That follows the egowm and slowwm replicates (seed 2 on M2b_s0).
* **Pairing.** Lever arm k is compared with R35_s{k}, which already exists and is not retrained. They share the WM initialisation, because `WMAux` is built right after `torch.manual_seed(seed)`. IDM_k also shares the data order. The comparison is paired over the same windows.
* **Run order.** If the chain dies early, the first three runs already give a complete chassis-0 contrast for lever 1 with its floor.

### Doses (fixed here, never tuned on scored windows)
* **IDM: gamma at level 1 = 1,000; levels 2 and 3 stay at 100.**
  * The code's IDM weight assumes unit-variance actions. `hjepa.IDMHead` states the collapse barrier `L_idm >= 1` "with unit-variance actions" (INHERITED from the code docstring; the paper is not banked and its numbers are not quoted).
  * Measured on R35's TRAIN windows, the level-1 target variances are 0.1767 and 0.0561. The effective per-channel weight is therefore 100 x var = 17.7 (acceleration) and 5.6 (yaw rate).
  * Restoring the design weight takes x5.66 and x17.82. The single scalar that does this on the geometric mean of the two channels is x10.04, rounded here to x10, so gamma_L1 = 1,000. The result is effective 177 (acceleration) and 56 (yaw rate), which brackets 100.
  * Why only level 1: it is the level with the smallest measured action use (ACTION-USE 1/8 of L2 and 1/35 of L3; IA-3 lateral gain about 0 at L1 against 0.13-0.32 above). It is also the level behind the 0-2 s rows. L2 and L3 IDM already make up 96 % of the loss.
  * At this dose, L1's IDM term would be about 46 at R35's end-of-run value, the same order as L2 and L3 (73-75), instead of 3 %. That is ESTIMATED from the R35 log; the realised value is in the arm's own log.
  * beta stays at 0.08, so only one lever moves.
* **DIVRS: exactly as `PREREG_V8F_DIVRS.md`, frozen `29f388e1`.**
  * A 5 x 5 grid: mean |yaw rate| with edges 0.02 / 0.05 / 0.10 / 0.20 rad/s, against mean acceleration with edges −1.0 / −0.3 / +0.3 / +1.0 m/s^2.
  * Draw probability is proportional to 1 / cell count, capped at 10x the median weight. Empty cells get no weight. The MM's ratified reading of control 1 applies.
  * There is no free dose. **A ladder caveat, MEASURED from R35's `summary.json`:** windows are 91 frames, and the start t ranges 90..104 (40,358 windows from 4,512 clips). Two windows of one clip therefore share at least 77 of 91 frames, and a DIVRS cell is in effect a clip property. The cell counts at this window length are not known and are recorded by `--selfcheck-only` before the first run (UNVERIFIED until then).

### Box-2 cost (ESTIMATED from MEASURED rates)
* **Trainer time per run:** 8,627 / 8,898 / 8,939 s (R35_s0 / R35_s1 / R35b_s0, `summary.json` seconds), which is 0.450-0.466 s/step, or 2.40-2.48 h.
* **Chain wall per run, including gate, selfcheck, post-check and bank:** 3 h 17 min to 3 h 20 min over the four steady intervals of `C:/Users/Admin/egowm_state/chain.log`. That chain ran the same recipe at wm_k 3, and its pilot read 0.4586 s/step (`egowm_state/pilot/PILOT.json`).
* **Per arm:**
  * IDM: about 3.3 h. The graph is the same as R35's; only a scalar changes.
  * NULL: about 3.3 h.
  * DIV: about 3.3 h plus the cell-table start-up. That start-up reads `a` and the native track per clip (UNVERIFIED), and the PILOT measures it.
* **Total:** pilot about 0.8 h (egowm pilot: 48 min of chain time) plus 6 runs of about 3.3 h, roughly **20.6 h ESTIMATED**. That is inside the package's 72 h rule. Without NULL_s3 it is about 17.3 h.
* **Dev-box CPU for the read:** about 2.0-2.2 h per checkpoint for HWM + FLAT at 2 inference seeds (MEASURED, `F:/TanitAD-scratch/hwm_plan/phase1_R35/phase1_R35.log`). Add the TRAIN-latent pull (405 s for two checkpoints, `collect_R35.log`) and IA-3 (341 s, `run_ladder_read.log`). Six checkpoints come to about 13.5 h ESTIMATED; the HWM rows run first.

## 3 Levers and code (full trace, probes and mutation plan in `REACHABILITY_ACTSENS.md`)
* **Neither lever is reachable from the ladder trainer today.** MEASURED by grep on the ladder tree (trainer sha256 `510b1e61`, equal to R35's `gate_token.json`):
  * `--wm-idm-gamma` and `--wm-gamma`: 0 hits; control `"--wm-strides"`: 1.
  * `wm_sampler`: 0 hits; control in the levers2 tree: 26.
* **IDM:** a new `--wm-idm-gamma G1 G2 G3` flag and a `V8FMiniConfig.wm_gamma_levels` field. WMAux builds the per-level gamma. The new G-DVB clause reads gamma from `HJEPAStack.built_levers()`, which is the LevelConfig the loss multiplies by, and not from `WMAux.gamma`. MEASURED: a builder that ignores the field passes the trainer's present gamma check (`v8f_mini_train.py:1500`).
* **DIVRS:** port `levers2/diffs/2_divrs.diff`. Its data-module and test hunks apply cleanly; 3 of its 7 trainer hunks FAIL on context (MEASURED dry-run). The 7 card mutations, left unfinished in `levers2/HANDOFF_LEVERS2.md`, run to the end on the ported tree.

## 4 Read

### 4.1 Mechanism read (per lever arm against R35 at the same chassis; NULL gives the floor)
* **IA-3** (`ladder_read/run_ladder_read.py`, code of record): gain lon, gain lat and cosine at L1 2.0 s, L2 6.0 s and L3 12.0 s.
  * **MECH-MOVED** = at both chassis, the arm's L1 2.0 s lateral gain or longitudinal gain sits above R35's, the two CIs are disjoint, and the point gap exceeds |NULL − R35| on that cell.
* **Beside, no bar:**
  * This card's `probe_actsens.py` Part A on each arm: L1 forward response per channel, and L1 IDM R^2 per channel.
  * The WMCEIL quantity on the ladder: the ladder read's planning-form L1 2.0 s skill against Copy-Last.
  * WMACT ACTION-USE was built for M2b (`wmact/wmact_lib.py`). It is reported only if it runs on ladder checkpoints unchanged; otherwise it is NOT COMPUTABLE, with the reason.
* **Health (feeds the HARM row):** the ladder read's BAND-MET (IA-0 skill against Copy-Last at the L1 2 s, L2 6 s and L3 12 s band ends) on every arm.

### 4.2 Deciding read (HWM-PLAN phase-1 rows on each arm)
* **Planning:**
  * `hwm_plan/run_hwm.py` `windows` / `plan`, code of record, with phase 1's levers: `--seed-mode prior --lam 0`, T3 2, plan-ctx 1, `stitch decode`, vocab off, constraint weight 0.
  * Arms `hwm_oracle` and `flat_oracle`, inference seeds 0 and 1.
  * Inputs: the arm's own checkpoint, and the arm's own TRAIN latents (`collect_latents_train_R35.py` pattern; a read-only sftp pull from box 2).
  * Run under the ACTSENS tree (§4 of REACHABILITY).
* **Contrast:** `F:/TanitAD-scratch/hwm_plan/phase2_l1full/contrast_l1full.py`, validated (sha256 `b7661635…`, equal to the repo copy at `bb3825f8`). `--a` = the arm's plans, `--b` = R35_s{k}'s banked phase-1 plans, and `--a hold` for the absolute rows. The plans are read from HF `Sayood/tanitad-v8f-plantest3-runs`, `hwmplan/hwmplan_phase1_R35.tar`, rev `aa2d8014`, tar sha256 `abff3a38…`, 16 files (`HF_BANK_HWMPLAN_PHASE1.json` at `bb3825f8`, MEASURED). `--windows` = R35's phase-1 windows dir, after control C6.
* **Deciding quantities:** band **0-2 s**, which is level 1's own plan and involves no decode. Two quantities: LONGITUDINAL `acc_mae` and LATERAL `yaw_mae`.
  * Δ_k = HWM-ORACLE(arm_k) − HWM-ORACLE(R35_k), paired, inference seed 0.
  * **BETTER** on a quantity = at both chassis, the CI of Δ_k lies wholly below 0 and |Δ_k| exceeds Floor_k.
  * **WORSE** = at either chassis, the CI of Δ_k lies wholly above 0 and Δ_k exceeds Floor_k.
* **Absolute row:** HWM-ORACLE(arm_k) − HOLD, 0-2 s and full 6 s.
* **Same rule, reported and not deciding:** FLAT-ORACLE rows, the 2-6 s band and the full 6 s band. The 2-6 s band passes through the off-image decode (DECODE-WEAK).
* **Off-image diagnostic** (`F:/TanitAD-scratch/hwm_plan/decode_check/decode_check.py`, sha256 `027da9ee…`, equal to the repo copy), on every arm, beside:
  * the median nearest-neighbour distance of the planned macros to the TRAIN macro cloud, over the held-out in-image distance (R35: 6.77 / 6.79);
  * the share of decodes at the bounds (R35: 94 %);
  * the in-image residual control.

### 4.3 Families (per band: 0-2 s, 2-6 s, full 6 s)
* **LONGITUDINAL:** `acc_mae`, `speed_err`, plus `hw_err` / `tg_err` / `ttc_err` on windows with a lead (n printed).
* **LATERAL:** `yaw_mae`, `head_err`, `curv_err`, `xtrack`.
* **TACTICAL:** plan-derived (`hwm_eval.TACTICAL_SCOPE`): the planned and the logged trajectory classified by one geometry rule, with kappa per band. Goal and anchor selection: n = 0, NOT COMPUTABLE (the planner has no decision output).
* **STRATEGIC:** PAUSED (PI item 25, 2026-09-19). Never a pass, never omitted.
* Each family is reported separately, never pooled.

### 4.4 Estimator and floors
* **Interval:** paired episode-cluster bootstrap (`taniteval/ci.py` through `hwm_eval.contrast`), B 2,000, seed 0, over the 994 SCORE windows in 71 episodes. It answers *another draw of episodes* and nothing else.
* **Training floor** F_T,k = |m(NULL_k) − m(R35_k)| per row, with NULL_0 = NULL_s2 and NULL_1 = NULL_s3. It answers *another training run*, once per chassis (H-ESTIM-SEED-1: one replicate pair).
* **Inference floor** F_I = max over {arm, R35} of |m(seed 1) − m(seed 0)| per row. It answers *another inference run*. Phase 1's HWM value: −0.0109 / −0.0073 m/s^2 (full 6 s).
* **Floor_k** = max(F_T,k, F_I). Every reported CI names which of the three questions it answers.

## 5 Controls (if one fails, the read is INVALID)
| # | control | known value |
|---|---|---|
| C1 | PARITY: the ACTSENS commit runs `--selfcheck-only` on R35_s0's argv. Its `live.terms` must equal R35_s0's banked `selfcheck.json` `live.terms`. Also, the flag-absent build's state_dict must equal the ladder-tree build at seed 0 (dev-box CPU, two trees). | bit-equal; a tree with a changed default goes red |
| C2 | GAMMA-LOG: `gamma_from_log.py` on every `train_log.jsonl` | R35: 100.0000 (MEASURED); NULL and DIV: 100 / 100 / 100; IDM: 1000 / 100 / 100, within 1e-4 relative at every logged step |
| C3 | IDM G-LIVE identity at the gate: same init and same batch as R35_s{k} | unweighted pred / sigreg / idm / rollout terms equal to R35's; level/L1 difference = 900 x idm/L1 (1e-5 relative) |
| C4 | DIVRS card control 1 on the built sampler, plus its deliberate regression (flag set, uniform draw) | the regression FAILS; resolvable cells within ±20 % |
| C5 | Post-training reachability: `probe_zerograd3.py` (Adam state) on every checkpoint | every `wm.*` tensor has state (R35_s0 is read first as the control: 133 WM tensors, UNVERIFIED until run) |
| C6 | WINDOWS-EQUAL: each arm's 994 SCORE windows (`eid`, `now`, logged `A`, `v0`) against R35's phase-1 windows | exactly equal |
| C7 | REPRO + SELF: `contrast_l1full.py --check` on the HF tar reproduces R35's 75 phase-1 rows, and an arm compared with itself reads 0.0 [0, 0] | exact (validated on phase 2) |
| C8 | ZERO-ACT deliberate regression (zero training, dev box): R35_s0 with the level-1 action embedding zeroed (weights and bias) | IA-3 L1 gains and Part A forward response exactly 0.0 |

## 6 Committed outcomes (each lever arm on its own; both chassis are required; each outcome names its next step)
| outcome | condition | next step (run in the same work package, Rule Zero) |
|---|---|---|
| **ACTSENS-CEILING** | HWM-ORACLE(arm) − HOLD at 0-2 s: CI wholly below 0 on `acc_mae` or `yaw_mae` at both chassis, and not WORSE on the other quantity | **The planner beats HOLD at T0 on this world model.** (1) This WM replaces R35 as the planning WM of record. (2) The full 6 s and 2-6 s rows are read at once, and any band that also beats HOLD is named. (3) HWM-PLAN G4/G5 (predicted goal, head-only, box 2) runs on this WM next: the first T1-capable step. (4) The level-2 image constraint gets its amendment for the 2-6 s band. |
| **ACTSENS-NARROWS** | BETTER on at least one deciding quantity, not WORSE on the other, not CEILING | The lever acts in the right direction. If both levers NARROW: the combined arm IDM + DIV, 2 runs. If only IDM: the exact per-channel design weight at L1 (an IDM target whitened by the TRAIN variance: gamma_c = 100 / var_c, which is 566 / 1,783). If only DIV: cap 10 → 30. Each is carded by amendment before any of its windows is scored. |
| **ACTSENS-MECH-ONLY** | MECH-MOVED, but no deciding quantity BETTER | The response grew and the plan did not improve, so the response is not the binding limit at T0. Next, zero training on dev-box CPU: the prior-regularised planner (`run_hwm.py --lam > 0`, amended dose) on this arm's WM, then the level-2 image constraint. |
| **ACTSENS-NULL** | not MECH-MOVED and no deciding quantity BETTER, with C2 / C4 holding (the lever ran) | The lever trained and did nothing. Next, box-2 training: a direct action-response objective. Either the counterfactual-action target of `PREREG_V8F_DIVRS` (window pairs with opposite lateral actions, a loss on the difference of their predicted states) or a gain-matching term toward IA-3 gain 1.0. Carded next. |
| **ACTSENS-HARM** | WORSE on a deciding quantity at either chassis, or BAND-MET lost at a level | The lever stays off. IDM next: gamma_L1 300, the bracket below. DIV next: ACTSENS-NULL's next step. |
| **SEEDS-DISAGREE** | a label holds at one chassis only | Reported as such. The next step of the less favourable label applies. |

* If both levers reach the same label, the lever with the larger mean −Δ over both chassis on 0-2 s `acc_mae` becomes the lever of record.
* An arm that misses its bar **FAILED** and is reported that way, and its next step runs (Rule Zero).
* **Done** means either (a) CEILING, with its next step started, or (b) every next step above is blocked, each blocker named with what would unblock it.
* No dose is changed after the data is seen. Every next dose is named above.

## 7 Box-2 package and stop rules
* `theory/actsens/box2/`, copied from `egowm/box2/`. The parts:
  * the pinned argvs;
  * the per-run token from `actsens_check.py gate` (the trainer's `--selfcheck-only` asserts what was BUILT);
  * the pilot and the 72 h rule;
  * the GPU lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` (dev-box stages only);
  * a WMI launch with `CREATE_NO_WINDOW`;
  * the bank to `C:/Users/Admin/t2v8f_pulls/actsens_<tag>/`, with sha256 checked against box 2's copy.
* A refused arm writes `ARM_REFUSED_<tag>.txt` and the chain continues with the next arm.
* Deletions on box 2 are limited to the run's own gate and pilot dirs, by literal path, after `BANK_VERIFIED`.
* Box 2 is a person's desktop. The chain touches only `D:\tanitad\actsens\` and `D:\tanitad\runs\v8f_actsens\`, plus the read-only inputs the argvs name.

## 8 Unchanged and out of scope
* R35's recipe in every other flag, and the data (`exclude 540:600`).
* The planner, its readers and the phase-1 levers. HWM-PLAN's card and amendments.
* The 2-6 s decode question (AMEND_HWMPLAN_2). No T1 or driving claim is made from this card.

## Log
* 2026-11-15, agent draft. No ACTSENS code, run or scored window existed. The probe numbers in §1 come from `probe_actsens.py` and `gamma_from_log.py` (dev-box CPU, read-only on checkpoints) and decide only the IDM dose, which comes from TRAIN windows, never from scored ones. Freeze line: Master Mind.
* **2026-11-15, FROZEN by the Master Mind as written above (this line is the only addition), before any ACTSENS code, run or scored window exists.** The PI's answer "Yes, card it" (2026-11-15) was to the question of carding this arm for box-2 time. Three notes, none of which moves a bar:
  * (F1) **The gate.** Each run's gate is the package checker `actsens_check.py gate` plus the trainer's `--selfcheck-only`, as for EGOWM, SLOWWM and STOPPERC. `stack/scripts/launch_gate.py` is not edited. The G-DVB clause for `--wm-idm-gamma` and `--wm-sampler` lives in that checker and the trainer's selfcheck.
  * (F2) **The code.** The code plan in `REACHABILITY_ACTSENS.md` (sha256 `dc2cd176...`) is binding: both levers default-off, and C1 parity bit-equal to the ladder tree. A deviation needs an amendment before the first run. The code lands as a diff under `raw/`, never into `stack/`.
  * (F3) **Runs.** The run order is §2's. Every arm must have a PASS token before box 2 trains it.
