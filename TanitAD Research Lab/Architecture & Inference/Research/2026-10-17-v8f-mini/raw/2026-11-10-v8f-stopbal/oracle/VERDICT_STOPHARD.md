# VERDICT: PREREAD_V8F_STOPHARD = HARD-TRADES for BAL-G and for BAL-L (Master Mind, 2026-11-11 ~19:04)

Card frozen 18:47:10 (sha256 `bc9ac911`, re-hashed equal after the run). Read by a Sonnet agent (`stophard_read.py` on `stoporc_read.run_arm`, imported unchanged; `RESULT_STOPHARD.json`). Re-checked by the Master Mind: `test_stophard.py` 24 passed; cells, floors and decision counts read from the JSON. Tier: **T1-kin first tick, never driving performance**. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 ("another draw of episodes" only). TACTICAL: n = 0 (the transformed input). STRATEGIC: **PAUSED** (PI item 25). All numbers MEASURED unless marked.

Controls (all seven arms, `controls_ok` true): an unreachable threshold gives a code bit-equal to OWN and a plan difference of 0.0 m in all 12 cells; HARD changes only the named entries and only where the decision fires; the decision counts equal STOPDEC's per window and STOPBAL's M counts (e.g. BAL-L s0 lon: 34 of 117 stop-point windows, 103 of 2,234 others; BAL-G s1 goal: 34 / 113). The card names one floor rule per cell; it is applied per variant (HARD-GOAL for BAL-G, HARD-LON for BAL-L): |delta(NULL_rep) - delta(NULL_s0)|.

## Cells: tactical ADE (knots 11-30), HARD minus OWN (m)
| arm (decision) | fires on stop-point / other windows | S: 117 stop-point windows [CI] | A: 2,351 windows [CI] |
|---|---|---|---|
| BAL-G s0 (goal) | 26 / 62 | **-0.052** [-0.094, -0.019], floor 0.022: gain | **+0.0059** [0.0009, 0.0122], floor 0.001: cost |
| BAL-G s1 (goal) | 34 / 113 | **-0.042** [-0.081, -0.009]: gain | **+0.0096** [0.0016, 0.0204]: cost |
| BAL-L s0 (lon) | 34 / 103 | **-0.088** [-0.156, -0.034], floor 0.021: gain | **+0.0076** [0.0000, 0.0154], floor 0.007: cost |
| BAL-L s1 (lon) | 46 / 160 | **-0.081** [-0.179, -0.009]: gain | **+0.0173** [0.0059, 0.0299]: cost |
| BAL-NULL s0 (both, beside) | 10 / 35 | -0.014, not separated | +0.0013, not separated |

## Outcome, as committed: **HARD-TRADES** for both levers
S is a gain and A a cost at both seeds. Hardening also hardens the false stops. Committed next step: **a hardening threshold set on TRAINING windows only, pre-registered; the second dose waits on it.**

## The size, which the label does not carry
* **Hardening recovers 3-6 % of the ceiling.** S moves by 0.04-0.09 m; STOPORC's forced GT decision moved the same windows by 1.48-1.54 m. ESTIMATED from the means: per window where the decision fires, hardening gains about 0.2-0.3 m (BAL-L s0: -0.088 x 117 / 34 = -0.30 m; BAL-G s0: -0.052 x 117 / 26 = -0.23 m).
* **So most of the ceiling sits in the windows T does not flag.** At BAL-L s0, 83 of 117 stop-point windows have no stop decision, and hardening cannot touch them. The confidence of the decision is a small part of the gap; its coverage is the large part. That points at the levers that raise coverage without raising false positives: the discrimination (perception) card, ahead of more weight on the same ROC.
* The A costs are small in absolute terms (6-17 mm) and clear floors of 1-7 mm, because the NULL arms seldom fire. They are costs by the committed rule, reported as such.

## What follows (Rule Zero)
1. **Committed: the training-threshold card.** `PREREAD_V8F_STOPTAU.md` is written beside this verdict and frozen before any training-window number exists. Its ceiling is the 0.04-0.09 m above, so it is a small lever, run because it was committed and because it is cheap.
2. **The largest measured lever is coverage.** The second dose moves the operating point along an unchanged ROC (STOPBAL), and its committed prediction is BAL-FP-COSTS. The perception card (stop-region resolution, LoRA arm L's read adds STOPCUE, a stop-line auxiliary) is the lever that can raise AUROC. It needs a training slot and possibly an architecture change: **a PI decision**, put with this verdict.
