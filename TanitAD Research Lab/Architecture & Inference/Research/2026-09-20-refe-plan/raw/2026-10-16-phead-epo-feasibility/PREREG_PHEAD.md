# PREREG: PHEAD -- what fraction of R0's EPO objective gain survives if EPO may only move the progress heads (CPU feasibility, 0 A40)

Written by the Research Lab agent (Sonnet), 2026-10-16, **BEFORE any restricted-arm number was computed**. Its sha256 is in `PREREG_PHEAD.sha256`, written the same minute.
- Source of the question: `eval/PREREG_R0C.md` section 4, row O3, next lever (R0c was killed by K3 and read as O3, `raw/2026-10-16-r0c-read/RESULT.md`).
- Nothing below may change without a dated amendment written before the number it affects.

## 0. What was looked at before this file was written (disclosed)

Only the **full** arm, never a restricted one (`probe0_fullarm_baseline.py`, `probe0_fullarm_baseline.json`):
- the banked closed-loop full-arm states (`closed_P_{0.0,0.25,1.0}.pt`, K = 25 / 50) scored with the new evaluator, to learn the size and noise of the denominator.
- Regime P, K = 50, lam 1 vs lam 0, EPO-attributable DeltaJ_full: **train pool +0.0407 [0.0258, 0.0557]**, **val pool +0.0333 [0.0064, 0.0558]** (interval = log-cluster bootstrap, 10,000, seed 20260927, evaluation sets resampled, training state fixed).
- That fixed three choices below: the denominator is positive and separated from 0 on both pools at K = 50 (but not on val at K = 25), so K = 50 is the decision point; the val pool has only 62 sets / 13 logs, so its intervals are wide and the decision rule uses both pools.

## 1. Question

R0c (EPO anchored to arm A) breached g5 on TTC with a small KL, so the EPO ascent itself lowers TTC whatever the anchor (O3). O3's committed next lever is *EPO restricted to the progress heads' parameters with a frozen trunk*, with its fraction of R0's J gain on cached trunk features run first on CPU. **Question: if EPO may only move group G, what fraction f of the full-arm J gain survives at matched updates, dose and learning rate, and what happens to the safety sub-scores?**

## 2. Definitions (from source)

- **J** = `sum_i pi_i R_i`, exact over the 64 hypotheses of a scene; `pi = softmax(log(sel_score(sigmoid(logits)) + eps) / T)`; R from `rl_reward.reward` (R0's reward config). Code: `scorer_finetune.py:218-231` (`epo_loss`), `:188-191` (`epo_policy`), `:234-249` (`epo_reward`), `sel_score` closure `:750-752`. The evaluator calls these functions themselves (`phead_common.Evaluator`).
- **The scorer** = `score_q_mlp` (trajectory encoder, `model.py:560-562`) -> `score_dec` (4 CrossBlocks that attend to the visual context, `model.py:551-552`) -> `score_head = Linear(256, 7)` (`model.py:563`). `scorer_finetune.py:75` `SCORER`. Its input visual context and the proposals are DETACHED and the whole backbone is frozen in EVERY arm (`scorer_finetune.py:5-6`, `:621-626`); that is not what "frozen trunk" adds.
- **Progress head.** The scorer has NO separate progress module. The progress (EP) output is **row 2 of `score_head`** (`train.py:502` `PDM_ORDER = (nc, dac, ep, ttc, c, ddc)` = head rows 0..5; row 6 = the lane output). It enters selection through `agg_v1 = NC*DAC*(5EP + 5TTC + 2C)/12` (`scorer_finetune.py:80-85`). The question is therefore ambiguous, and the candidates are ALL run as their own arms:

| group | entries that may move | parameters | reading |
|---|---|---|---|
| **PH_A_EPROW** | `score_head.weight[2,:]`, `score_head.bias[2]` | 257 | the EP output only |
| **PH_B_SIX** | `score_head` rows 0..5 | 1,542 | the six PDM component outputs, lane row frozen |
| **PH_C_HEAD** | all of `score_head` | 1,799 | the whole head |
| *LAD_1 (rung)* | `score_head` + `score_dec.3` | 1,055,239 | NOT a reading of "progress heads": a reference rung that releases the last scoring-decoder block |
| *LAD_2 (rung)* | `score_head` + all of `score_dec` | 4,215,559 | rung: everything except `score_q_mlp` |
| FULL | all 78 tensors, 4,296,967 params | 4,296,967 | the reference (R0's regime) |

- **"Frozen trunk"** = `score_q_mlp` + `score_dec` get no gradient and are not in the optimiser or the clip (kind R), on top of the always-frozen backbone / proposals / visual context. The scorer's EPO displacement (R0 minus the floor, energy share) lives **99.1 % / 99.3 % in `score_dec`, 0.9 % / 0.7 % in `score_q_mlp`, 0.02 % in `score_head`** at u250 / u500 (`disp_real.json`, MEASURED).
- **Two kinds of restriction, both run.** Kind **R** (the brief's reading, "everything else frozen"): in arm B only G is trainable for BCE AND EPO. Kind **E** ("EPO restricted"): BCE moves every parameter, only the EPO gradient is masked to G. E is run for PH_C_HEAD only (the best-capacity head), as a reference.
- **Restricted optimiser.** Only G's entries are given to `torch.optim.AdamW` and `clip_grad_norm_(.., 1.0)`, as a trainer that sets `requires_grad False` on the rest would (`scorer_finetune.py:1360-1362`, `:1391-1395`). Weight decay therefore never touches a frozen entry. Arm A stays the full BCE-only control (`paramsA`, `:1360`). A real implementation of PH_A would need the EP row as its own parameter (a residual delta on the EP logit, zero-initialised); weight decay on the delta instead of the row differs by `lr * wd * steps = 5e-6` relative at K = 50, which is below every tolerance here.

## 3. The metric

For an arm X with trainable group G, dose lam:
- `Delta J_X = J(theta_X(lam)) - J(theta_X(0))`, where `theta_X(0)` is the SAME restriction, same draws, BCE only (kind R), or the full BCE-only run (kind E). This is the EPO-attributable J gain, the analogue of the `D(lam) - D(0)` construction in `adam_sim.py`.
- **f = Delta J_G / Delta J_FULL** at the same K, the same lam, the same regime, the same lr, the same draws (stream 0), on a fixed evaluation pool.
- A second, guard-relevant reading is reported but does not decide: **f_A = (J(theta_G(lam)) - J(A)) / Delta J_FULL**, A = the full BCE-only control (what g3-g5 compare to).
- Pools: **train** (302 sets / 30 logs; 296 of them are the sets the optimiser sees, so in-sample) and **val** (RL-val frames with a cached visual context: 62 sets / 13 logs, out-of-sample, the g5 tier). **f_min = min(f_train, f_val)** is the decision quantity.
- **Estimator.** Paired log-cluster bootstrap over the pool's logs, 10,000 resamples, seed 20260927; f is the ratio of the resampled means. It answers *"would another draw of evaluation sets say this?"* and nothing about another training run or data order. A second question is answered separately: **three bank streams** (3 different data orders) in the open-loop arms give the range of f over data order.

## 4. The simulation (reused, not reinvented)

- **Closed loop (decides).** `phead_closed.py` = `adam_closed.py`'s real trainer step on moving theta: the trainer's own `train_losses` closure, per-micro backward / accum 8, clip, AdamW, gradients recomputed at the moving theta; regime **P** (constant lr 1e-4, lam constant from step 0, cold Adam state), **K = 50 updates** of 8 micro-batches of 8, stream-0 draws, snapshots at K = 25 and 50. Full-arm reference = the banked `closed_P_{0.0,1.0}.pt` (re-run for K = 25 as control (a)).
- **Open loop (robustness).** `phead_open.py` = `adam_sim.py`'s Adam replay on the gradient bank (`bank_s{0,1,2}.pt`), restriction applied: regime P (K = 100) and regime T (the trainer's warm-up + EPO ramp, K = 250), lam in {0, 0.25, 1}, J evaluated at theta_K on the stream-0 states and, at K = 50 / lam in {0, 1}, on all three streams.
- Evidence class of every sim number: **ESTIMATED** (real scorer weights, labels and trajectories; the visual context is the substituted epoch-15 cache, within-scene V = 0.996-0.997 against the banked logits, `late_probe.json`; CPU fp32). Tier: offline selection read, not T1, not a driving claim.

## 5. Safety proxies the sim CAN compute, and the ones it cannot

All from the labelled 64-proposal sets, per set, on both pools, B minus control, x100 (the g5 convention):
- **P1 pick-TTC / pick-NC / pick-DAC**: the sub-score label of the proposal the deployed rule selects (`argmax sel_score`). Direction: higher is safer. g5's bound: **>= -0.5**. Discrete (resolution 1/62 = 1.6 points on val, 0.33 on train).
- **P2 policy-expected TTC / NC / DAC**: `sum_i pi_i label_i`, the EPO policy's own expectation. Continuous. Same bound.
- Also reported: pick-EP, pick-C, pick PDMS, KL(pi_base || pi_theta).
- **Controls the TTC condition is read against.** T-match: vs the same-restriction lam = 0 control (the EPO-attributable harm; decides). T-A: vs the full BCE-only A (g5 as registered; reported). If T-match holds and T-A fails, the row is kept and flagged: *the A40 arm must carry a matched restricted control A'*.
- **Cannot be computed by the sim:** PDMS at the full RL-val set (3,000 sets; the cache covers 62), the full-navtest endpoint, speed / headway, lateral, tactical (n = 0, no instrument in a scorer-only sim); STRATEGIC is PAUSED (PI ruling 2026-09-19, item 25). Each is reported per term in `RESULT.md`.
- **Proxy validity (control d).** The proxy is admissible only if it reads the known R0 trade: on the banked R0 states (`scorer_states_u250/500/1000.pt`, both pools pooled, 364 sets), P2 B - A TTC is **negative at >= 2 of 3 checkpoints** (the measured RL-val B - A are -1.23 / -0.63 / -1.00, `ab_vs_base.json`), AND on the C0 floor control (B is BCE-only, B ~ A) |P2 B - A| < 0.5 at every checkpoint. Failing this, the TTC condition below is **UNDECIDED IN SIM** (never "not worse") and the rows carry the suffix *(TTC unverifiable in sim; rests on K3)*.

## 6. Arms

| id | kind | group | lam | lr | runs |
|---|---|---|---|---|---|
| CL-FULL | R | FULL | 0, 1 | 1e-4 | banked `closed_P_{0.0,1.0}.pt` (+ a K = 25 re-run, control a) |
| CL-PHA, CL-PHB, CL-PHC | R | PH_A, PH_B, PH_C | 0, 1 | 1e-4 | closed loop, K = 50 -- **the three candidates; these decide** |
| CL-LAD1, CL-LAD2 | R | LAD_1, LAD_2 | 0, 1 | 1e-4 | closed loop, reference rungs |
| CL-EPHC | E | PH_C | 1 | 1e-4 | closed loop; control = banked FULL lam 0 |
| CL-PHA-lrK, CL-PHC-lrK | R | PH_A, PH_C | 0, 1 | 1e-4 x {4, 16} | closed loop; **step-size sensitivity: does NOT change the row**, it names the next lever |
| OL-* | R (E for PH_C) | all of the above | 0, 0.25, 1 | 1e-4 | open loop, 3 streams |
| CTRL-a/b/c/d | | | | | section 7 |

## 7. Identity controls (all must pass BEFORE any restricted number is reported)

- **(a) FULL through my code reproduces the banked numbers.**
  - (a1) open loop: `Restricted(FULL)` vs `adam_sim.run` on bank stream 0, regime P, lam 0 / 0.25 / 1, K = 50: per-tensor `max |diff|` **<= 1e-7** (same ops, expected exactly 0.0), and the recomputed `S_C80`, `S_all`, `cos_C80`, `U_full_norm_ratio` equal `adam_sim.json` streams[0].modes.P.at[50] to **relative 1e-6**.
  - (a2) J: `J(theta_0)` on the 256 sets of the stream-0 partition's first 32 micro-batches equals `late_probe.json` states.base.mean_J = 0.8391432501375675 to **absolute 1e-5**.
  - (a3) closed loop: FULL, lam 1, K = 25, rerun vs banked `closed_P_1.0.pt` D[25]: **relative L2 <= 1e-3** (different thread count => different float summation order; the exact figure is reported).
- **(b) EMPTY group => Delta J exactly 0.0**: open loop (lam 1 vs 0, K = 50, all three pools' per-set arrays) and a closed-loop K = 3 run: `max |theta - theta_0| == 0.0` and `Delta J == 0.0` with `==`, not a tolerance. (Weight decay on a masked-gradient entry would break this; it is the bug the check exists to catch.)
- **(c) Mutation.** `PHEAD_MUTATE_NO_RESTRICT=1` deletes the line that applies the restriction (`phead_common.Restricted.__init__`). The check `max |theta - theta_0| over entries OUTSIDE G == 0.0` must PASS unmutated and read **RED** mutated, and the mutated PH_C result must differ from the unmutated one.
- **(d) Proxy validity** as in section 5.

## 8. Bars and committed outcomes (decided on f_min, closed loop, regime P, K = 50, lam 1, matched lr; per candidate; the recommendation uses the candidate with the largest f_min)

| row | condition | meaning | next lever (Rule Zero) |
|---|---|---|---|
| **R1 CARRIES** | f_min >= 0.5 AND T-match not worse (P1 and P2 both >= -0.5 on both pools, proxy validated) | a head-only EPO keeps at least half of the J gain at no measured proxy cost | **recommend ONE A40 arm** on that group (a PI decision: ~3.5 A40-h, a parameter-group change in `scorer_finetune.py`, a matched restricted control A', a launch-gate PASS); the replicate arm per H-ESTIM-SEED-1 before any claim |
| **R2 PARTIAL** | 0.2 <= f_min < 0.5, T-match not worse | the restriction carries part of the gain | the cheapest rung that reaches f_min >= 0.5 among LAD_1 / LAD_2 and the lr x {4, 16} arms is the proposed group / lr (a PI decision); if none does, R3 |
| **R3 CANNOT CARRY** | f_min < 0.2, OR T-match worse (validated proxy) at any f | the head cannot carry the EPO gain, or the gain it carries costs TTC | read the rungs: (i) if an lr x {4, 16} arm reaches f_min >= 0.5 with T-match not worse, the limit is STEP SIZE, not capacity -> a head-only arm at that lr (PI); (ii) else if LAD_1 / LAD_2 reach it, the gain needs the scoring decoder, and the TTC trade must be broken another way: RL-S1 O2 (ii), a scorer whose safety outputs EPO cannot move (separate safety head, an architecture change, PI); (iii) else the EPO objective's gain is not reachable under any restriction tested and the next lever is the objective itself (reward, `w_prog`, not the parameter group) |
| **U (flag)** | proxy fails control (d) | the sim cannot certify TTC | the f rows stand; the TTC condition rests on the A40 arm's K3 guard |

Rows are applied to f_min exactly as computed; an f whose own interval straddles a band edge is reported with its interval and the row is NOT moved. A failed bar is reported FAILED, and the next lever is named either way.

## 9. Limits written in advance
- The J gain is measured on the trained-on pool (in-sample) and on 62 out-of-sample sets; neither is the RL-val 3,000-set read. K = 50 updates is 5 % of R0's 1,000; the ramp (T regime) is covered only by the open-loop arms.
- One data order per closed-loop arm. The three-stream open-loop arms bound the order-sensitivity of f; they do not replace a training replicate.
- The restricted arm's real gradients also depend on the real visual context, which the cache substitutes (V = 0.997).
