# PREREG_S1C_DRAFT: does a longer training budget lift the S1 drivable-area head to K2? (DRAFT: committed before the A40 run, not yet registered by the Master Mind)

Date 2026-10-16. Companion of `S1C_DESIGN.md` (same folder). Parent: `eval/PREREG_AUX_DRIVABLE_S1.md` (K2, P1, the estimator, the four-family rule). Tier: **T0-like** (head perception on teacher-forced frames; never driving performance). STRATEGIC family: **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` item 25) throughout.

## 1. Question

S1 (400 updates) read K2 FIRES on all three H1 seeds: navtest P1 IoU [0,40) **0.4957 / 0.5211 / 0.4864** on the 1,058 C3 tokens (0.561 / 0.577 / 0.554 re-weighted to the natural navtest population; held-out navtrain 0.631 / 0.643 / 0.621) against the 0.80 bar (`raw/2026-10-15-s1-read/S1_READ.md`, `raw/2026-10-16-aux-shift-locate/SHIFT_LOCATE.md`, MEASURED). S1's head was still learning at update 400 (`aux_bce` 0.569 -> 0.432, still falling). The dev-box probe (in-domain, NOT a prediction) read 0.635 at 2,000 updates and 0.675 at 10,000 (`raw/2026-10-16-scene-tap-a2/A2_RESULT.md`, MEASURED).

**Q: with the SAME head, trunk, labels, learning rates and natural sampling as S1's H1 arm, does training for 10,000 updates (80,000 rows) instead of 400 (25,600 rows) lift navtest P1 [0,40) to the K2 bar on BOTH the C3 set and the natural navtest population?**

Hypotheses (all HYPOTHESIS until read):
* **HS1C-1 (the registered claim under test):** the S1 head was budget-limited; at 10,000 x 8 it reads P1 >= 0.80 on C3 AND natural navtest.
* **HS1C-0 (what I expect):** it improves but does not reach 0.80 (the probe's in-domain best is 0.675; S1c is read out-of-domain). Stated before the run so that a "miss" is not a surprise and a "hit" is not a goalpost move.

## 2. Arm, comparators, controls

| | |
|---|---|
| **Arm (the only one that runs)** | `H1u10k-s0`: `--max-updates 10000 --batch 8 --accum 1 --eval-every 5000`, seed 0, H1 (feed off), profile `refe_aux_s1c`, argv in `S1C_DESIGN.md` section 1 / `ARGV_DIFF.json`. ~4.3 A40-h. |
| Registered alternative (NOT run unless an outcome below calls for it) | `H1u2k-s0`: 2,000 x 64, ~6.2 A40-h. |
| **Comparator of record** | S1's `H1-s0` (400 updates, same seed, same init, same frozen u400 trunk, same labels, same frames): the **budget contrast** S1c minus H1-s0, paired on the same frames. |
| Not a comparator | S1's N arms (they have no head and trained 400 updates). The scorer pair (arms A, B) is trained too (no head-only mode exists) but is budget-confounded and is NOT read for any claim here. |
| **C1 identity (free)** | S1c's `step0` held-out `aux_eval` IoU = S1 H1-s0's **0.1586791569036157** (+-1e-3). A mismatch means the trunk, data or init differ from S1: the arm is **not paired with S1** and no budget claim is made (the init floor and absolute K2 may still be reported). |
| C2 zero control | G12's weight-0 control reads exactly 0.0 (a launch-gate check, already required for PASS). |
| C3 schedule | `run-check`: update N of N at lr 0.0, all eval tags and dumps (a launch-chain check). |
| C4 init floor | A2: trained minus the seed-0 INITIAL head on the same frames, 95 % lower bound > 0 (`--pred-init`). |

## 3. Reads (all on navtest; navtest governs, as in S1)

* **P1 [0,40) m, all frames** on (a) **C3** (the 1,058 tokens, failure-enriched) and (b) the **natural navtest population** (SHIFT_LOCATE weights: failure stratum 4.36 %, navtest's own city mix, ESS reported). **K2 is decided on the point estimate against 0.80, as S1's rule words it; the interval is context.**
* **Budget contrast** Delta_C3, Delta_NAT = P1(S1c) - P1(H1-s0) with its paired log-cluster 95 % CI (`head_fit_pairs["H1u10k-s0-minus-H1-s0"]`).
* Secondary, no decision: P1 bands [0,20)/[20,40)/[40,72), the u400-DAC-zero split, P2 (SDF MAE near the boundary), P3 (DAC AUC on C3) beside the scorer's, held-out navtrain P1, per-city P1 when the driver exists.
* Reference floor for "beyond the rig's noise": **F = the largest pairwise |difference| between S1's three H1 seeds on the same population = 0.035 (C3), 0.023 (NAT)** (MEASURED from the numbers in section 1). It is a proxy for training-run variance at 400 updates; the variance at 10,000 updates is unknown (HYPOTHESIS that it is no larger).

## 4. Committed outcomes and the next lever for each (Rule Zero: no outcome ends at a verdict)

Decided by the arm's reads in this order. "Separated" = the paired CI excludes 0 **and** |Delta| > F; with one seed it is **necessary, not sufficient** (section 5).

| outcome | rule | what it means | NEXT lever, executed in the same run of work | what blocks it |
|---|---|---|---|---|
| **O1 K2 CLEARED** | P1_C3 >= 0.80 **and** P1_NAT >= 0.80 | the frozen visual tap carries the road at K2 precision once the head is trained long enough; S1's K2 firing was a budget artefact | (1) replicate seed 1 (`H1u10k-s1`, ~4.3 A40-h) because one seed is not a lever claim; (2) **S1c-H2**: the S1 question (does the head's SDF feed lift selection / P3) re-asked with an adequate head, as its own prereg; (3) hand the head to S2 planning | PI: replicate and H2 are A40 spend |
| **O2 PARTIAL** | Delta_C3 and Delta_NAT both separated and > 0, but P1_C3 < 0.80 or P1_NAT < 0.80 | the budget is a real lever that has not saturated or has saturated short of the bar | rank by the measured remaining gap `0.80 - P1` on the worse population. **No extrapolation beyond 2x the fitted range (400 -> 10,000 is already 25x): do not project to 100k.** Candidates, each its own registered pair: (a) **more label logs** for the head (the probe's own recommendation, `A2_RESULT.md`); (b) the **trunk lever**: a head on P6's trained trunk / the scene-tap graft (A3r: the frozen trunk may be the limit) | (a) a label build (CPU/labels; size and cost not scoped); (b) PI decision and P6's checkpoint |
| **O3 NO GAIN** | Delta_C3 or Delta_NAT not separated (CI includes 0 or |Delta| <= F) on either population, and C1 holds | the training budget is **not** the lever for this tap at >= 80,000 rows; the representation (frozen trunk) or the label coverage limits the head | **close the budget lever for the visual tap**; go to the trunk lever (head on P6's trained trunk / graft; PI decision) and the label-coverage lever. A null from ONE seed closes nothing alone: if the PI wants it closed, the replicate (seed 1) is the price | PI: trunk lever; P6's checkpoint |
| **O4 WORSE** | Delta_C3 or Delta_NAT separated and < 0 | not expected; batch-8 noise is the one thing that changed | defect hunt first (C1, C2, C3, the per-eval trajectory step0 -> 5000 -> 10000), then the registered alternative **`H1u2k-s0` (batch 64, ~6.2 A40-h)** as the cheapest arm that separates "budget" from "batch" | PI: A40 spend |
| **INVALID** | C1 fails, or the chain's run-check refused | no S1c read | find the cause (drift of trunk / data / environment); a re-run is a Master Mind / PI decision (~4.3 A40-h) | PI |

If P1_C3 and P1_NAT disagree on K2, **the C3 read governs** (as S1 prereg 4.1 wrote for navtest), and both numbers are reported with their population named.

## 5. Seed policy (H-ESTIM-SEED-1)

* **ONE arm, ONE seed (0)**, by the PI's scope for this window. The registered replicate is `H1u10k-s1` (a one-line change: `launch_gate.AUXS1C_SEEDS = (0, 1)` + the chain arm + the test tables); running it is a **follow-up PI decision**, not part of this launch.
* Until the replicate exists, **every S1c-vs-S1 difference is stamped "separated, single seed, not yet a lever claim"**, the label S1's harness already uses. The paired log-cluster bootstrap resamples LOGS with the models held fixed; two runs differing in nothing cleared such an interval on the v7-tiny rig (9.5 % false-positive rate), so a separated CI is **necessary, not sufficient**, for a claim that the BUDGET moved the metric. The sufficient form is the replicate against the rig's own run-to-run noise (F above is the cheap proxy until then).
* A K2 verdict (O1 / not O1) is an absolute read of one trained head against a fixed bar; it does not need a replicate to be reported, but "the budget is the lever" does.

## 6. Estimator

**Paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927** (`proxy_eval.boot`'s draw scheme; ratios of sums use the same resamples; `eval/aux_s1_readout.py`). It answers **"would another draw of LOGS say this?" ONLY**: not another training run (section 5), not another inference run (the head is deterministic: ASSERTED, not measured; the navtest forward runs in fp32 on the planner path, `aux_navtest_forward.py --head-ctx planner`). The navtest read has few logs; the intervals will be wide (SHIFT_LOCATE's held-out side was 24 logs and its gap interval [0.041, 0.214]).

## 7. Invalid-run and kill rules

* The launch gate, the chain's run-check (`s1aux_checks.run_problems` at the S1c budget) and C1-C4 above decide validity. A run that ends early (the TL-1 epoch cap), whose last logged row is not update N of N at lr 0, or whose eval tags / dumps are incomplete is **refused by the chain and banks nothing**.
* Operational check (not outcome-bearing): the logged `s_per_update` at accum 1 should read 1.24 to ~1.8 s and the train wall clock 3.4 to 4.7 h (`COST_MODEL.json`). A reading outside it means the cost model, not the claim, is wrong.
* No kill rule on the K2 trajectory: K6 (S1's early-stop on the BCE) is not carried over; with 10,000 updates a stall would be visible in the 5,000-update eval and the arm is cheap enough to finish.

## 8. Not computed, non-claims

* **The four families** (LONGITUDINAL, LATERAL, TACTICAL) are **NOT COMPUTED as paired reads**: there is no budget-matched N arm (arm B's scorer trained 25x the updates at batch 8, so a difference is not attributable to the head). Arm B's pick is read **descriptively** against the three 400-update N arms and stamped budget-confounded. An `N-u10k` control (~4.2 A40-h) would unlock them and is a PI decision. **STRATEGIC: PAUSED**, with the ruling above; never a pass, never omitted, never extended to the other three.
* No E1/E2/E3 on the full navtest (the planner seam does not apply a feed), no driving-performance claim, no claim about H2.
* The probe's numbers are in-domain and are cited only to rank the budget lever.

## 9. Decisions this draft asks of the PI / Master Mind

1. Register this prereg (or amend the thresholds) **before** the arm launches; the thresholds above were written before any S1c number exists.
2. Authorise the launch window: after R0c logs `ZZRLS1_QUEUE_CLEAR`, ~4.3 A40-h (up to 5.4).
3. Decide now, so it is not decided after the data: the replicate (seed 1), the `N-u10k` control, and whether O2/O3 trigger the trunk lever automatically.
