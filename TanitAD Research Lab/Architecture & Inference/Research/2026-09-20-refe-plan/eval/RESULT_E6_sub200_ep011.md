# RESULT E-6 -- selection diagnosis, `sub200_ep011` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch011.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 91.2 [89.2, 93.1] |
| **planner's pick** | **43.8 [37.4, 50.6]** |
| random proposal (mean of 64) | 43.2 [38.9, 47.7] |
| medoid, no scorer (pre-registered comparison) | 43.7 [37.2, 50.4] |
| logitsum (EXPLORATORY) | 39.9 [33.8, 45.9] |
| violation_only (EXPLORATORY) | 47.9 [42.1, 53.7] |
| aggregate_without_EP (EXPLORATORY) | 48.5 [42.7, 54.4] |

Paired: pick - random **0.6 [-4.3, 5.9]**, oracle - pick **47.4 [41.3, 53.4]**, medoid - pick **-0.1 [-6.7, 6.1]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.013 [-0.089, 0.121]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 86.2 | 81.2 | 100.0 |
| DAC | 63.0 | 66.9 | 99.0 |
| DDC | 88.0 | 90.5 | 94.0 |
| EP | 47.2 | 44.2 | 89.0 |
| TTC | 74.0 | 69.4 | 98.5 |
| C | 51.5 | 51.4 | 81.5 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 198 tokens whose PDMS varies: **0.129 [0.080, 0.180]**; pooled over all (token, proposal): 0.086
- the pick is a best proposal on **15.0 %** of tokens; a random pick would be 14.6 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.780 | 0.724 [0.683, 0.761] (n 120) | 60 % | 80 % | 80 % | no |
| DAC | 0.653 | 0.545 [0.501, 0.591] (n 143) | 72 % | 97 % | 67 % | no |
| DDC | 0.665 | 0.533 [0.466, 0.603] (n 105) | 52 % | 94 % | 86 % | no |
| TTC | 0.747 | 0.653 [0.613, 0.691] (n 142) | 71 % | 60 % | 69 % | no |
| C | 0.559 | 0.488 [0.451, 0.526] (n 159) | 80 % | 86 % | 51 % | **yes** |
| EP (Spearman) | -0.095 | 0.128 [0.046, 0.204] | 99 % | 94 % | 44 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **1.046 [0.998, 1.091]**; its percentile among the 64: 0.602 [0.545, 0.657]
- within-token Spearman with path length -- scorer aggregate **0.105 [0.012, 0.195]**, predicted EP 0.869 [0.855, 0.884], TRUE PDMS **-0.041 [-0.125, 0.039]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **20.0**, median 15.5 of 64; 88.5 % of tokens have at least one, 50.0 % have at least 16
- the pick scores 0 on **46.0 %** of tokens; all 64 proposals score 0 on only 1.0 %
- the pick leaves the drivable area on **37.0 %** of tokens; 99.0 % of tokens have at least one proposal that stays inside it

## Reading

The proposal head is not the problem: in a typical scene a quarter to a third of the 64 proposals score 80 or more, and the best one averages 91.2 PDMS. The scorer is: its pick is indistinguishable from a random proposal (skill 0.013), it believes 97 % of proposals stay on the road when 67 % do, and within a scene its drivable-area, driving-direction and comfort outputs rank proposals at chance (within-token AUC 0.54 / 0.53 / 0.49), while its progress output follows path length (within-token Spearman 0.87) although the true PDMS does not (-0.04). Pooled AUCs look better than within-token ones because whole scenes differ in difficulty; choosing needs the within-token skill. MECHANISM (consistent with every number here, not yet shown causal): `refe/train.py` supervises the scorer with 8-9 FIXED candidates per frame, each attached to its NEAREST proposal a few metres away (`assign_d` 2.5-5.7 m on the latest micro-batches), so the labels a proposal's scorer output learns are the scores of a different path, and drivable area / comfort change within one metre -- a declared departure from DriveZero, which scores the student's OWN proposals. The fix is a recipe decision for the PI.

## Verdict (rule fixed before the table was read)

**SELECTION_BOUND**; failing scorer outputs: **C**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep011/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep011/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep011_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
