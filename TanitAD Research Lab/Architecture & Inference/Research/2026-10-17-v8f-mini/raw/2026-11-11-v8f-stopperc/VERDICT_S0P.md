# STOPPERC stage 0: verdict on S0-P (Master Mind, 2026-11-12)

**Outcome (committed rule): FLAT.**
* RES-ADDS does not hold, and HEAD-UNDERUSES does not hold, for the linear or the MLP probe.
* Under card §4 / §8, FLAT means **R deferred; A and S0-L stay**. Stage 1 is therefore **A alone**, and it needs the PI's go.

**Sources.**
* Card `PREREG_V8F_STOPPERC.md` (frozen, sha256 `70c75176…8877`).
* `AMENDMENT_1_S0.md` (`6c4ad83b…d123`).
* `AMENDMENT_2_S0P.md`: paired HEAD-UNDERUSES, written before any S0-P number on training data.
* Numbers: `s0p/RESULT_S0P.json` (sha256 `050d8b1f18dde5e1551e5371f4f10bd06e406bc414cbccd07a206e0f22cc1344`, MEASURED, written by the S0-P chain at 2026-11-12 08:58:48).

**Tier and estimator.**
* Tier: a linear / MLP probe on frozen encoder features, scored once on eval-141's first-tick windows. It is a diagnostic of what the inputs carry, **never driving performance**.
* T's AUROC is T1-kin first tick (STOPDEC npz `goal_prob`).
* Estimator: `taniteval.ci.stat_cluster_bootstrap` over the 139 eval episodes, B 2,000, seed 0. Paired differences are taken on the same resample.
* It answers **"another draw of episodes"** only. The probes are deterministic (lbfgs; MLP `random_state` 0), so there is no inference-seed term.
* "Another training run" for T is the BAL-NULL_s0 / s1 pair, which differs by 0.028 AUROC. That is close to the 0.037 HEAD-UNDERUSES bar.

**STRATEGIC: PAUSED** (PI item 25).

## Sample
* Train: 16,000 windows (4,000 STOP_POINT-positive) from the card's seeded sampler over STOPBAL's prevalence pool (4,512 clips).
* Eval: STOPCUE's rule on eval-141: 2,365 windows, 131 positives, 139 episodes. This re-counts STOPCUE exactly.
* PCA dimension, L2 strength and MLP size were chosen by 5-fold clip-grouped CV inside the training sample.

## Probe AUROC on eval-141 (point [95 % CI])
| arm (inputs) | linear | MLP | CV choice (linear / MLP) |
|---|---|---|---|
| E (T's 18 non-vision NOW inputs: `p10f` 8, `nav` 5, `limit` 4, `limit_ok` 1) | 0.8525 [0.8080, 0.8928] | 0.8387 [0.7852, 0.8896] | C 1.0 / h 64 |
| E−limit (13) | 0.8521 [0.8077, 0.8927] | 0.8395 [0.7869, 0.8890] | C 1.0 / h 64 |
| M+E (`mean`, PCA) | 0.8653 [0.8187, 0.9074] | 0.8472 [0.7930, 0.8930] | dim 32, C 1.0 / dim 32, h 256 |
| **V40+E** (`pool4x10`, what T reads) | **0.8615 [0.8101, 0.9073]** | 0.8352 [0.7885, 0.8763] | **dim 16, C 0.001** / dim 32, h 256 |
| V160+E (`pool8x20`) | 0.8553 [0.8048, 0.8991] | 0.8461 [0.7939, 0.8910] | dim 16, C 0.001 / dim 16, h 256 |
| **T, BAL-NULL_s0 / s1** (`goal_prob`, causal `limit`) | **0.8294 [0.7769, 0.8808] / 0.8578 [0.8105, 0.9017]** | | |
| T with `limit` off (S0-M, beside) | 0.7892 / 0.8159 | | |

## The two committed reads (paired, same windows)
| read | bar | linear | MLP | holds |
|---|---|---|---|---|
| **RES-ADDS**: V160+E − V40+E | ≥ +0.02 and CI > 0 | −0.0063 [−0.0276, +0.0157] | +0.0110 [−0.0245, +0.0459] | **no** |
| **HEAD-UNDERUSES**: V40+E − T, at BOTH BAL-NULL_s0 and _s1 | ≥ 0.037 and CI lower > 0 | s0 +0.0321 [−0.0216, +0.0861]; s1 +0.0038 [−0.0436, +0.0470] | s0 +0.0057 [−0.0437, +0.0527]; s1 −0.0226 [−0.0682, +0.0178] | **no** |

Beside, read for nothing (paired, linear / MLP):
* E − E−limit: +0.0004 [−0.0008, +0.0016] / −0.0009 [−0.0180, +0.0147].
* M+E − E: +0.0128 [−0.0160, +0.0399] / +0.0085 [−0.0478, +0.0609].
* **V40+E − E: +0.0091 [−0.0315, +0.0477]** / −0.0035 [−0.0541, +0.0474].
* V160+E − E: +0.0028 [−0.0436, +0.0461] / +0.0075 [−0.0408, +0.0562].
* V40+E − M+E: −0.0037 [−0.0269, +0.0181] / −0.0120 [−0.0589, +0.0414].

## What it says (and what it does not)
* **The stop cue that a frozen-feature readout reaches is in T's non-vision inputs, and vision adds nothing measurable to it.** E alone reads 0.852, at T's level. Adding the 4x10 grid moves it by +0.009 (CI covers 0). CV inside the training sample picked the smallest PCA dimension (16) and the strongest L2 (C 0.001) for both grids, so the training data itself said that more vision features hurt. MEASURED.
* **`limit` carries none of the probe's cue** (E − E−limit +0.0004, CI ±0.0012). This agrees with S0-M on T (−0.040 / −0.042, not separated). Q7's concern, that the cue rides on an ego-future channel, has no support at either level. MEASURED.
* **T already extracts what a probe can find.** The paired V40+E − T difference is +0.032 at s0 and +0.004 at s1. The two T seeds themselves differ by 0.028, so the s0 gap sits inside one training-run swing.
* ⛔ **It does NOT say "the stop cue is not learnable from vision"** (card §4: a linear negative says nothing about a trained adapter).
  * These are frozen features, PCA-compressed per token, NOW frame only.
  * Arm A exists to change exactly that: it puts a gradient through the adapter toward the stop location.
  * S0-C says what A must reach: T's stop decision on roughly 60-70 % of stop-point windows, against 6 % / 12 % now (`VERDICT_S0.md`).

## Four families
* **TACTICAL:** the stop-point decision's AUROC, the table above (n 2,365 / 131 positives / 139 episodes).
* **LONGITUDINAL:** n = 0. A probe emits no plan. The longitudinal reads for a stop lever are S0-C's (tactical speed −0.579* / −0.477* m/s at c 0.7, `VERDICT_S0.md`).
* **LATERAL:** n = 0, for the same reason. S0-C carries the named lateral cost of a forced stop (tactical yaw rate +2.17* / +2.18* deg/s at c 0.7).
* **STRATEGIC:** PAUSED (PI item 25).

## Controls (`RESULT_S0P.json` `controls`)
* 0 clip ids shared between train and eval.
* E speed vs the dumps' first-tick input: max |diff| 0.0.
* Second route for `limit`, `limit_ok`, `nav`, `p10f`: max |diff| 0.0 on eval and on train.
* Leak (GT appended): AUROC 1.0 (≥ 0.999).
* Label shuffle: CI covers 0.5 at E (0.5143 [0.4077, 0.6181]) and at V40+E (0.5366 [0.4391, 0.6360]).
* **Episode-shuffled labels:**
  * E 0.3833 [0.2891, 0.4755]: below 0.5, CI excludes 0.5.
  * V40+E 0.4284 [0.3356, 0.5247]: covers 0.5.
  * The card lists this control without a bar, so it decides nothing. The diagnosis is below.

### The episode-shuffle reading is one permutation draw, not a pipeline defect (POST-HOC, MEASURED, decides nothing)
1. **The ordering explanation is refuted** (`s0p/posthoc_episode_shuffle.py` → `s0p/POSTHOC_EPSHUFFLE.json`).
   * HYPOTHESIS tested: the training arrays are label-sorted within a clip, so `np.resize` truncation hands borrowed positives to short clips.
   * Finding: the arrays are already in window-start order. Labels shuffled in as-run order and in time order are identical, and their AUROC difference is exactly 0.0 [0.0, 0.0].
   * The shuffled labels are uncorrelated with the real ones: r 0.0011; P(shuffled 1 | real 1) 0.103 vs P(shuffled 1 | real 0) 0.102. Shuffled prevalence is flat across clip-size thirds (0.096 / 0.110 / 0.108).
2. **The permutation spread** (`s0p/posthoc_shuffle_seed_spread.py` → `s0p/POSTHOC_SHUFFLE_SPREAD.json`): the same linear probe and the real fit's hyperparameters, re-run over many permutation seeds. Eval AUROC per seed:

| arm | control | seeds | mean | sd | min-max | as-run |
|---|---|---|---|---|---|---|
| E | episode shuffle | 40 | 0.5625 | 0.1066 | 0.3194-0.7613 | 0.3833 (1 of 40 at or below it) |
| E | label shuffle | 40 | 0.4891 | 0.1018 | 0.2776-0.7277 | 0.5143 |
| V40+E | episode shuffle | 20 | 0.5094 | 0.0644 | 0.3936-0.5835 | 0.4284 |
| V40+E | label shuffle | 20 | 0.4997 | 0.0558 | 0.4029-0.6018 | 0.5366 |

* ⇒ A chance-level linear probe on E lands anywhere from about 0.3 to 0.75 on these 131 positives. The as-run 0.383 sits in the lower tail of that distribution.
* The bootstrap CI resamples eval episodes with the fitted probe held fixed, so it cannot see this variance. **Same class as `H-ESTIM-SEED-1`:** the interval answers "another draw of episodes", and the spread here is "another draw of the shuffle".
* ⚠️ So a **single-draw** shuffle control with a bootstrap CI is a weak check on this eval set. The committed label-shuffle bar (CI covers 0.5) passed here, but one draw in the low or high tail would fail it with nothing wrong. Read shuffle controls as a permutation distribution.
* Every real fit is far outside these null distributions: real E 0.852 vs the null maximum 0.761; V40+E 0.862 vs 0.602.
* HYPOTHESIS, not tested: the episode shuffle's mean sits slightly above 0.5 (0.5625, standard error of the mean about 0.017) because borrowed sequences keep their within-clip runs of positives.
* ⚠️ **Not measured:** the variance of the real fits over another draw of the TRAINING SAMPLE (the sampler seed is fixed by the card). All committed reads have CIs wide enough to cover 0, so this does not change the outcome.

## Next
1. **Stage 1, A alone** (card §5, Q1, Q3). This is the PI's go.
   * NULL_s0 at A's commit with `--stop-aux off`. If its checkpoint sha256 equals BAL-NULL_s0's, BAL-NULL_s1 and BAL-NULL_rep are reused.
   * NULL_rep2 (`--seed 3`).
   * A_s0 and A_s1.
   * ~40 + 40 + 2 × 44 min ≈ 2.8 box-2 h, ESTIMATED from the card's per-run costs, plus ship, gates and pulls. If the sha check fails, NULL_s1 and NULL_rep add 80 min.
   * Box 2 is free after the SLOWWM chain (ESTIMATED ~23:30 tonight).
   * A's code is being built on a new branch off `715ef8fc` (BAL-NULL's commit). It needs a launch-gate PASS before any run.
2. **S0-L** once the pod's L_s0 is banked (ESTIMATED Nov 14 evening).
3. **R stays deferred** (FLAT). Under §8 it was approved only on RES-ADDS.
