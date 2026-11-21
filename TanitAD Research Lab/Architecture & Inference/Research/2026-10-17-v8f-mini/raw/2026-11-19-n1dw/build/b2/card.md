# PREREG_V8F_N1DW (FROZEN 2026-11-19 by the MM after the coordinator's review, before any N1DW code, checkpoint or number exists): a roundabout-window weight on O's trajectory loss, on top of N1D

Status: drafted by the Master Mind (MM) and reviewed by the coordinator on 2026-11-19, under the PI's delegation. The review required five changes, R1 to R5, plus one recommendation; all are applied here, and section 0a lists where each one went. The card is then frozen by the MM, with its sha256 written into `PREREG_V8F_N1DW.sha256`. At freeze no N1DW code, flag, checkpoint or number exists.
Evidence classes: MEASURED (M, artifact named), PUBLISHED (P), INHERITED (I), ESTIMATED (E), HYPOTHESIS (H).
Parents: `raw/2026-11-16-v8f-strat1/navrbd/PREREG_V8F_NAVRBD.md` (sha256 881d7f3d, the card whose bar, grid, label, floor and guard this card reuses unchanged), `raw/2026-11-16-v8f-strat1/navrbd/read/MM_READING_NAVRBD.md`, `raw/2026-11-16-v8f-strat1/rbstack/MM_READING_RBSTACK.md` and its `MM_READING_RBSTACK_ADDENDUM1.md` section 3.

## 0. PI go and scope
* **PI, 2026-11-19 ~13:30Z, verbatim:** "1 yes, 2: yes, 3: yes after tanitadruled". Item 1 was "Approve N1DW at weights 2 and 4 to replace N1D8? About 2-3 LoRA pod-h". So the go covers **4 arms: weights 2 and 4, seeds s0 and s1**, about 2 to 3 LoRA pod-h (E). Any other dose, arm or cost goes back to the PI.
* **What "weight w" means (fixed here, before any code):** a roundabout (RB) training sample's trajectory terms count **w times**; every other sample counts once. w = 2 and w = 4. The equivalent extra weight is w - 1 = 1 and 3.
* **Code base:** `0c3b374f` (branch `agent/v8f-navrbd-20261118`), the code N1D_s0 / N1D_s1 trained from. The new flag is added on a child branch of it.

## 0a. Coordinator review applied at freeze (2026-11-19)
* **R1:** LONGITUDINAL and LATERAL are printed on the deciding cell D1-HO as well (section 5.1).
* **R2:** the bit-identity controls run on CPU, and GPU checks are read against a same-code repeat (section 6).
* **R3:** two outcome gaps are closed: RBW-GAIN-ONE-COST and the cross-weight tie rule (section 7).
* **R4:** RBW-GAIN is provisional until N1DWx is read. The N1DWx read rule is fixed here. p and the clipped-step share are printed (sections 6 and 7).
* **R5:** the CPU probe of N1D's own features is pre-registered here and runs now, in parallel. Its result picks the lever after RBW-NO-GAIN. N1D8 stays parked unless the probe points at vision (section 10).
* **Recommendation:** the within-arm seed differences are printed as further floor draws, and the largest one becomes the bar (section 5.3).
* **PI ask:** the conditional go for one N1DWx arm is asked now, so that a gain does not leave the pod idle.

## 1. Question
NAVRBD (RBD-NO-GAIN, RBD-FLOOR-HIGH) showed the exit + distance channel is read (Z-USE separated at both seeds) and moves held-out roundabout TAC-LAT by +0.0291 [+0.0174, +0.0409] (s0) and +0.0171 [+0.0044, +0.0300] (s1): separated, below the 0.0424 bar (M, `navrbd/read/out/RESULT_NAVRBD.json`). RB-JOINT-K showed the same text decodes the kinematic roundabout decision far above the state alone at T0 (ED - S +0.2331, 9,236 windows / 395 train episodes; T0 decodability, never driving performance; M, `navrbd/rbjk/`). RBSTACK showed vision adds +0.023 / +0.028 over the text at T0 (STACK-WEAK).
So the inputs N1D already has carry far more about the roundabout decision at T0 than the trained arm uses. **Does a trained N1D arm that weights its roundabout samples' trajectory loss by w = 2 or 4 decide the roundabout lateral manoeuvre better than N1R, on the same held-out cell, by at least the bar?**
* HYPOTHESIS (why the weight is the lever): roundabout samples are a small share of the train loss, and a 3,000-step fine-tune at lr 1e-4 gives the O path little gradient from them. The weight raises that share without a new input or a new module.
* **The share, MEASURED on the train channel table** (`navrbd/build/b2/navtext_nav_rbd_report.json`, G-CLOCK-RBD): 29,836 of 461,518 train cache steps have c0 = 1, so **p = 0.0646**. The trajectory-loss share on RB samples is then 0.0646 at w = 1, **0.121 at w = 2** and **0.217 at w = 4** (w p / (w p + 1 - p)). This is the step share of the table. The share at the sampler's NOW ticks is measured by the build (K-SHARE) and printed beside it.
* ⚠️ The T0 figures and the T1-kin cell are different tiers and different estimands. The +0.23 is a ceiling on what the features carry for a tree, not a prediction of a T1 effect.

## 2. The lever (one new trainer flag)
* `--rb-window-w w` (float, **default 1.0** = the term is not computed and not in the total = N1D bit for bit).
* With w > 1, the total gains ONE term, built on the launch-intent pattern already in this code (`v8f_mini.py:1255-1317` at 0c3b374f):
  `total += w_traj * (w - 1) * (ADE_rb / 3.0 + 0.5 * FDE_rb / 6.0)`
  * `ADE_rb` and `FDE_rb` are ADE's and FDE's own per-sample distances (`traj_terms`, `v8f_mini.py` at 0c3b374f: the mean over the 30 knots and the last knot of the same smoothed L2 distance, both axes, times the same per-sample GT-valid flag `gok`), **summed over RB samples only and divided by the SAME denominator as `T["ade"]` / `T["fde"]`**, `gok.sum().clamp_min(1.0)` (the batch's GT-valid samples, not the RB count). So on an RB sample the trajectory terms carry weight exactly w, and every other sample's loss is unchanged. A batch with no RB sample adds exactly 0.0.
  * Nothing else is weighted: not the acceleration term (`--w-acc`), not the kinematic, world-model, DAgger or channel terms. The TAC-LAT label is a function of the plan track's geometry, which is what ADE / FDE score.
* **The RB mask** = column c0 (`state`) of the sample's `nav_rb` row at its NOW tick, the row O's nav token reads at NOW. It is the same rule that defines a D1-HO window ("state at NOW"). c0 comes from the route input (route-text table, rows at or before NOW), so the mask uses nothing an arm does not already read. It is used in training only; nothing changes at inference.
* Refusals (G-DVB entry, `declared_vs_built`): w not finite or w < 1; w > 1 with `--nav-rb off`; w > 1 with `w_traj` 0 (a weight that does nothing); a checkpoint cfg whose `rb_window_w` differs from the argv.
* DAgger swap rows hold `nav_rb` as logged (as N1D), so their mask is the logged c0.

## 3. Admissibility (binding)
* **Labels may use ego; inference is vision plus the route input.** The weight is a training-loss weight and reads the route input only. At inference N1DW is N1D's model with N1D's inputs.
* **Not the situation classifier's output**, in any form: the mask is c0 of the route-text channel, which no classifier produces.
* **No future row:** the mask at NOW reads rows at or before NOW (test: rows after NOW replaced by garbage leave the mask unchanged; a mutant reading NOW + 5 goes RED).
* Optimistic by construction, as NAVRBD section 3: the route text is GPS-matched along the driven path; every number is worded "with the route supplied along the driven path".

## 4. Arms (LoRA pod, `flock /root/gpu_queue.lock`, one job at a time; N1D's argv + one flag, 3,000 steps, fine-tune from M2b)

| priority | arm | = | paired with |
|---|---|---|---|
| 1 | **N1DW4_s0** | N1D_s0 argv (`navrbd/build/b4/argv/N1D_s0.argv.json`) + `--rb-window-w 4` | N1R_s0 (deciding), N1D_s0 (beside) |
| 2 | **N1DW4_s1** | N1D_s1 argv + `--rb-window-w 4` | N1R_s1, N1D_s1 |
| 3 | **N1DW2_s0** | N1D_s0 argv + `--rb-window-w 2` | N1R_s0, N1D_s0 |
| 4 | **N1DW2_s1** | N1D_s1 argv + `--rb-window-w 2` | N1R_s1, N1D_s1 |

* w = 4 runs first: the larger dose is the larger expected effect (RULE ZERO item 5). A killed queue still leaves a readable two-seed pair.
* Every arm names the new flag. N1D's and N1R's argv render unchanged (Z0).
* N1R_s0 / s1 and N1D_s0 / s1 are banked PRIVATE on HF `Sayood/tanitad-v8f-strat1-runs` (M, `navrbd/read/out/BANKED_NAVRBD_ARMS.json`), with their D1-HO dumps (`navrbd_read/navrbd_read_outputs.tar`, sha256 143f3b2a).
* Conditional arms (each needs the PI's go, runs only on its trigger, section 7): the deliberate-regression arm N1DWx at the gained weight, s0 (one arm, about 0.5 LoRA pod-h ESTIMATED, asked now); seeds s10 / s11; w = 8; the lateral-only weight; the scale-matched N1D arm.

## 5. Read (unchanged from NAVRBD except the arms)
**Tiers.** D1-HO and the eval-141 cells: **T1-kin** (self-action open loop, m = 10, causal bucket, nav dead-reckoned, `nav_rb` held from t0 with c13 dead-reckoned), PRIMARY. STRATEGIC: T0 at the window's first tick, as STRAT1.
**Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0, over the stratum's episodes. It answers "another draw of EPISODES" only. "Another TRAINING RUN" is carried by F_HO and the two-seed rule. "Another INFERENCE RUN" does not apply (deterministic planner, one environment).

### 5.1 Deciding cell
* **D1-HO:** kinematic TAC-LAT accuracy (NAVRBD section 0a item 4: `read_otband.labels` of the plan track against the GT track, 3 classes) on the T1-feasible held-out grid, **5,736 windows / 497 episodes** (`eval_grid_x_feasible.json`, sha256 ca289d0f). Delta = **N1DW - N1R**, paired, per seed and per weight.
* **Beside, decides attribution only:** N1DW - N1D on the same windows (the weight's own share), and D2-HO (turn recall, side given turn, false-turn share), as NAVRBD.
* **R1, beside on D1-HO, no bar:** **LONGITUDINAL** and **LATERAL** for both deltas (N1DW - N1R and N1DW - N1D), on the same 5,736 windows / 497 episodes and with the same estimator.
  * The cells are the STRAT1 cells of those two families: speed, along-track, acceleration, headway and time gap; heading, curvature, yaw rate, cross-track, and FDE at 0-2 s and 2-6 s.
  * Why: an L2 weight on roundabout samples mostly up-weights longitudinal error there, and eval-141 holds too few roundabout windows (235 / 16 episodes) to see that shift.
  * Any cell that cannot be computed on these windows (for example headway, where a held-out clip has no lead label) is printed per cell with the reason and the n. It is never dropped.
  * TACTICAL on D1-HO is the deciding cell itself. STRATEGIC is read on eval-141 (section 5.2), because the held-out cell has no route-setting reference.

### 5.2 Guard: the four families on eval-141, no cost
* The STRAT1 24 cells per seed (LONGITUDINAL, LATERAL, TACTICAL, ADE / FDE) for N1DW - N1R, with the STRAT1 cost / gain rule and floor F2. **The guard holds at a weight iff no cell is a cost at either seed and there is no ROUTE-COST** (NAVRBD's rule, as applied in its reading).
* **STRATEGIC in full:** route setting (RC-A, 6 cells), route following S_CKPT at 80 m under real, donor and zero nav, detected-information cells as in the N1D read. **Fog is unscored** (PI 2026-11-17); weather on clear / rain / snow, fog n printed.
* A weight changes what the loss favours, so a cost on non-roundabout cells is this lever's named risk. Every cell also prints N1DW - N1D.

### 5.3 Floor and bar (fixed; no new floor)
* **F_HO = 0.0424** (M, NAVRBD step 0, `step0/out/HO_FLOOR.json` sha256 cbc7d42b). **GAIN at a seed and weight:** N1DW - N1R >= max(0.0383, F_HO) = **0.0424** and the paired CI lower bound above 0.
* N1DW dumps use the same tool, grid, thread count and environment (pod CPU) as the NAVRBD dumps. If the environment differs, N1R and N1D are re-dumped beside them, F_HO is recomputed, and the larger F_HO is the bar.
* **Further floor draws, committed now:** the within-arm seed differences on D1-HO are printed beside F_HO: |N1D_s1 - N1D_s0| and |N1DW_w_s1 - N1DW_w_s0| for each weight.
  * **The bar at a weight is max(0.0383, F_HO, |N1D_s1 - N1D_s0|, |N1DW_w_s1 - N1DW_w_s0|).** A larger floor draw raises the bar; nothing lowers it below 0.0424.
  * From the rounded published levels (N1R 0.5905 / 0.6328 plus the N1D deltas), |N1D_s1 - N1D_s0| ≈ 0.0303, below F_HO. The read recomputes it exactly from the banked dumps (`navrbd_read/navrbd_read_outputs.tar`) before any N1DW number is opened. The N1DW draws exist only after the arms.

## 6. Controls (each with the value it must read)

**R2, where these run.** Z0, Z1, Z2 and Z-ID are bit-identity or exact-ratio checks. They run on **CPU** (`CUDA_VISIBLE_DEVICES=-1`, fp32, one thread), where same-code runs are bit-identical. The K-TREE read showed same-code runs on the LoRA pod GPU differ: max rel 2.0e-6 at 20 steps (`raw/2026-11-15-v8f-datascale/reps1/ktree/KTREE_RESULT.json`).
* Any check that must run on the GPU (the mint's 4-step crash / resume, G-CKPT, G-EVAL) is read against a **same-code repeat** run beside it. It passes when the N1DW - N1D difference is no larger than the repeat's own difference, term by term, at the same step.
* Z-MASK, K-SHARE and the refusals are device-independent.

| id | control | must read |
|---|---|---|
| Z0 | `--rb-window-w 1.0` build vs `0c3b374f` (state-dict keys and values, every loss term, total, outputs, RNG stream) | bit-identical |
| Z1 | w = 4 at step 0 vs w = 1, same init and batch | every output and every existing loss term bit-identical; total differs by exactly `w_traj * 3 * (ADE_rb/3 + 0.5 FDE_rb/6)` computed independently from the per-knot distances (rtol 1e-6) |
| Z2 (G-LIVE reach) | on a real batch with at least one RB sample: the extra term and the gradient it sends to O's plan parameters; then the same batch with the mask set to all zeros | term finite and > 0, gradient nonzero / term **exactly 0.0** and total's gradient equal to w = 1's bit for bit |
| Z-ID | analytic identity on one batch: w = 1 reproduces N1D's gradient bit for bit; on RB samples the per-sample trajectory gradient scales by exactly w | exact / rtol 1e-6 |
| Z-MASK | the training mask vs an independent recompute of c0 at NOW over one full sampler epoch (the ChunkedSampler replay, leftover batches included) | equal on every sample; RB sample share p printed |
| K-SHARE | realised weight share on RB samples from the run's counts, w p / (w p + 1 - p), written into each run record; the sampler's p printed **before launch** from a CPU replay of the arm's own sampler | printed; the run's RB count > 0 (no per-batch requirement: leftover batches of 1 to 4 rows can hold none) |
| K-CLIP (R4) | share of logged optimiser steps on which clipping was active, per run, for N1DW and for N1D. The trainer clips at `--clip-grad` 1.0 and logs the pre-clip `grad_norm` (`v8f_mini_train.py:4125-4135` at 0c3b374f), so a step is clipped iff its grad_norm > 1.0. N1D's share is read from its banked train log; the logged-step count is printed | printed beside every label. With the batch denominator the loss scale grows by 1 + (w - 1) p, so under clipping the step changes for every sample, and that alone can move a metric |
| K-MUT | (a) delete the extra-term line; (b) mask all ones; (c) mask read at NOW + 5; (d) divide by the RB count instead of the batch denominator; (e) the flag without its G-DVB entry | (a)-(d): the named test RED; (e): the launch gate refuses |
| G-DVB / G-EVAL | declared = built (`cfg.rb_window_w`, `built_levers`); the eval loader builds the identical model from an N1DW checkpoint | PASS; refusals of section 2 each REFUSED |
| K1 | reader self-pair (N1R_s0 vs N1R_s0) on D1-HO | exactly 0.0 |
| K-CONST | majority-class predictor on D1-HO | 0.338215 exactly |
| K-RE | N1D - N1R recomputed from the banked dumps | equal to `RESULT_NAVRBD.json` |
| Z-USE (beside) | channel real - zero on D1-HO for each N1DW arm | reported; says whether the weight raised the arm's use of the channel |

## 7. Outcomes, committed before any number (each names its next lever, RULE ZERO)
Read per weight and per seed. "Gain" and "guard" as section 5.

| outcome | condition | next |
|---|---|---|
| **RBW-GAIN (provisional)** | at one weight: D1-HO a gain at BOTH seeds and the guard holds at both seeds | **provisional until N1DWx is read (R4)**: nothing enters the recipe yet. If both weights qualify, the smaller one is carried and both are printed. Next: the regression arm **N1DWx** at that weight, s0 (PI go asked in advance, see section 0a). N1DWx is the same weight on a random mask with the same per-batch RB count, drawn from non-RB samples. Its read is below |
| **RBW-GAIN-ONE-COST** (R3a) | at one weight: D1-HO a gain at BOTH seeds, and a guard cost at exactly one seed | not carried. That weight at s10 / s11 (2 arms, PI go): it is carried only if the cost cell is not a cost at both new seeds and the gain holds under the two-seed rule. If the cost is LONGITUDINAL, the lateral-only weight below is carded beside it |
| **RBW-ONE-SEED** | at a weight, a gain at exactly one seed, guard holds | that weight at s10 / s11 (2 arms, PI go), read under the same bar |
| **RBW-SPLIT** (R3b tie rule) | no weight gains at both seeds, but w = 4 gains at one seed and w = 2 at the other | no weight is carried. The weight whose mean two-seed N1DW - N1R is larger goes to s10 / s11 (2 arms, PI go). If the two means are within 0.005, the smaller weight goes |
| **RBW-DOSE** | no gain at either weight; N1DW4 - N1D separated above 0 at both seeds and N1DW4 - N1D >= N1DW2 - N1D at both seeds | w = 8 at s0 / s1 (2 arms, PI go) |
| **RBW-NO-GAIN** | no gain at either weight and RBW-DOSE does not hold | the loss weight is not what holds the effect down. **The probe of section 10 picks the next lever** (it runs now, in parallel, and is read before this card's arms). N1D8 stays parked unless the probe reads PROBE-VISION |
| **RBW-COSTS** | at a weight, a guard cell a cost at both seeds, or D1-HO a cost at both seeds | that weight is not adopted; if the other weight holds the guard, it is read alone. **If the cost is LONGITUDINAL** (on eval-141, or on D1-HO's beside cells at both seeds), the weight line does not stop: the next lever is a **lateral-only weight** (the extra term on the cross-track component of the per-knot distance only), at the same doses, PI go. Otherwise RBW-NO-GAIN's next applies |
| **RBW-FLOOR-HIGH** | the bar of section 5.3 above 0.0383 and above every observed N1DW - N1R | the cell cannot resolve this lever at these seeds: more seeds (PI go), never a lower bar |

**N1DWx read (R4, fixed now).** N1DW_w_s0 - N1DWx_w_s0, paired on D1-HO, same estimator:
* CI lower bound above 0 ⇒ **RBW-GAIN-MASK**: the gain is specific to the roundabout mask, and the weight enters the recipe. Next: seeds s10 / s11 (PI go), then the stacked arm on the S1 route-text recipe.
* Otherwise ⇒ **RBW-GAIN-SCALE**: the gain comes from the loss scale (1 + (w - 1) p under clipping), not from the mask, and the weight does not enter the recipe. Next: an N1D arm with the loss scale matched and no mask (a global factor of 1 + (w - 1) p), to test whether a step-size change alone is the lever.
* K-CLIP is printed beside both labels.

* **Attribution, printed with every label:** where N1DW - N1R is a gain but N1DW - N1D is not separated at either seed, the label carries "the gain is not attributed to the weight over the channel" and the N1DWx arm comes first.
* Bars, controls, arms and outcomes do not change after any number is seen. A bar that fails is reported as written, then the triggered lever is built.

## 8. Cost (for the PI's go, already given)
* MEASURED: N1D_s0 trained 3,000 steps in 672 s (pod `runs/N1D_s0/logs/train.log`, DONE line). The extra term is a masked sum over tensors the loss already holds; ESTIMATED step cost unchanged within 5 %.
* ESTIMATED: 4 arms x about 12 min training, plus the per-arm launch gate (G-SUITE) and the D1-HO + eval-141 dumps on pod CPU: about 2 to 3 LoRA pod-h in all, as asked.

## 9. Known risks
* **Cost on non-roundabout cells.** The weight shifts the loss toward RB samples; the 24-cell guard and RBW-COSTS cover it. w = 2 exists so a guard failure at w = 4 still leaves a readable dose.
* **Seed noise.** F_HO 0.0424 is itself separated as an episode-draw difference; a true effect near the bar can read below it at one seed (RBW-ONE-SEED).
* **The mask is the logged route state**, not the GT manoeuvre. A roundabout sample with lane-keep GT is weighted too; that is intended (the decision includes not turning).
* **Driving side** (left- or right-hand traffic) is not in the text, as NAVRBD section 9; the weight cannot add information the inputs lack.
* 3,000 steps may be too short for the weight to act; RBW-DOSE and the N1D8 / probe levers cover a null.

## 10. PROBE-N1D-FEAT (R5): where the route text's information is lost in N1D. CPU, zero training, runs now in parallel
**Question.** At T0 the route text decodes the roundabout kinematic decision far above the state alone (RBJK, ED - S +0.2331). N1D realises +0.017 to +0.029 at T1-kin. Where along N1D's own path is the information lost?
**Evidence class:** T0 decodability of frozen features, never driving performance.

**Data.** The D1-HO grid: 5,736 windows / 497 held-out episodes, which no N1R or N1D arm trained on. The label is the GT track's 3-class kinematic TAC-LAT (`read_otband.labels`, NAVRBD section 0a item 4), with majority share 0.338215.

**Features**, all at the window's NOW tick, extracted from the banked checkpoints N1D_s0 and N1D_s1 on CPU, deterministic, fp32:
* **F-TEXT:** the 15 `nav_rb` columns the token receives. This is the reference: what the text carries into the model.
* **F-NAV:** O's nav token after the `e_nav_rb` addition (d dims).
* **F-PLAN:** the input to O's plan head at NOW, i.e. the pooled O feature the trajectory is decoded from (d dims). The build cites the `file:line` of the tensor taken.
* **F-VIS:** the mean of the NOW frame's adapted vision tokens (d dims).
* **F-PLAN+TEXT:** the two concatenated (beside).

**Probes and folds.**
* Two probes on each feature set: multinomial logistic regression (standardised, L2, C fixed at 1.0, never tuned on these data) and a gradient-boosted tree with RBJK's settings.
* 5-fold CV grouped by episode (no episode in both train and test of a fold), fold assignment seed 0. Probe seeds 0 and 1.
* ⚠️ A linear probe's negative is not a negative about learnability, so the tree decides and the linear probe is printed beside it.

**Estimator.** Paired episode-cluster bootstrap, B 2,000, seed 0, over the 497 episodes, on the per-window out-of-fold correctness. Every contrast is paired on the same windows. **Bar 0.0383** (the rig floor of the roundabout cells).

**Controls (must read)**
* **SHUF:** each feature set with the training labels permuted within each training fold. Accuracy within 0.02 of the majority share (0.338215).
* **K-CONST:** the majority predictor reads 0.338215 exactly.
* **K-DET:** features extracted twice from the same checkpoint on CPU are bit-identical.
* **Regression arm:** F-TEXT with column c0 and the exit columns zeroed must read separated below F-TEXT. This proves the probe sees the text.

**Outcomes, committed** (tree probe; each must hold at both N1D seeds and both probe seeds, otherwise the label is printed per seed and carries "one-seed"):

| outcome | condition | the lever after RBW-NO-GAIN |
|---|---|---|
| **PROBE-LOST-AT-TOKEN** | F-NAV - F-TEXT separated below 0 by at least 0.0383 | the token path: the channel enters O too weakly. The `--nav-rb-lr-mult` dose (30) or a 2-layer `e_nav_rb` (PI go) |
| **PROBE-LOST-AFTER-TOKEN** | F-NAV not separated below F-TEXT by 0.0383, and F-PLAN - F-NAV separated below 0 by at least 0.0383 | the information reaches the token but not the plan head. A route-decision auxiliary head on O's plan feature (trained on the kinematic label; labels may use ego), PI go |
| **PROBE-PRESENT-UNUSED** | F-PLAN not separated below F-TEXT by 0.0383 | the information is at the plan head and the decoded track still misses it. The lever is the decoder / rollout: a T0-against-T1 read of N1D on D1-HO (zero training) first, then the planner-side lever it points to |
| **PROBE-VISION** | F-PLAN+TEXT - F-PLAN, or F-VIS over F-TEXT, separated above 0 by at least 0.0383 | vision carries roundabout information the plan path lacks. **Only this label un-parks N1D8** (interpolated position-table init, PI go) |

* More than one label can hold; all are printed. The lever order follows the table's rows.
* The probe is read before any N1DW number is opened. Its label does not change any N1DW bar or outcome; it only picks the lever after RBW-NO-GAIN.
* Cost: CPU only, on the pod (cores outside TanitADRuled's 0-15, nice 10) or the dev box, ESTIMATED under 2 CPU-h. It reads only banked checkpoints and the D1-HO cache.

## Log
* 2026-11-19, MM: drafted after the PI's go and before any N1DW code, flag, checkpoint or number exists.
* 2026-11-19, coordinator review: R1 to R5 plus the floor-draw recommendation, applied by the MM (section 0a). Frozen after that, still before any N1DW code, flag, checkpoint or number, and before any probe feature is extracted.
