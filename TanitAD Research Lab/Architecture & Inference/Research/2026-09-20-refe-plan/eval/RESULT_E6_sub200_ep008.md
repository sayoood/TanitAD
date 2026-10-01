# RESULT E-6 -- selection diagnosis, `sub200_ep008` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch008.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 90.4 [87.7, 92.8] |
| **planner's pick** | **49.3 [42.4, 56.6]** |
| random proposal (mean of 64) | 41.3 [37.3, 45.5] |
| medoid, no scorer (pre-registered comparison) | 41.5 [35.2, 47.9] |
| logitsum (EXPLORATORY) | 38.7 [32.9, 45.0] |
| violation_only (EXPLORATORY) | 47.9 [42.0, 53.7] |
| aggregate_without_EP (EXPLORATORY) | 50.4 [44.3, 57.0] |

Paired: pick - random **7.9 [1.8, 14.7]**, oracle - pick **41.1 [34.3, 47.5]**, medoid - pick **-7.7 [-16.4, 0.0]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.162 [0.038, 0.296]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 88.2 | 79.4 | 99.5 |
| DAC | 69.5 | 67.6 | 97.5 |
| DDC | 89.8 | 89.8 | 90.2 |
| EP | 46.1 | 42.8 | 86.2 |
| TTC | 81.0 | 67.5 | 99.5 |
| C | 59.0 | 49.0 | 85.0 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 195 tokens whose PDMS varies: **0.130 [0.073, 0.188]**; pooled over all (token, proposal): 0.106
- the pick is a best proposal on **15.0 %** of tokens; a random pick would be 13.4 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.769 | 0.717 [0.673, 0.759] (n 115) | 57 % | 78 % | 79 % | no |
| DAC | 0.631 | 0.519 [0.476, 0.562] (n 144) | 72 % | 97 % | 68 % | no |
| DDC | 0.716 | 0.610 [0.553, 0.670] (n 109) | 55 % | 95 % | 85 % | no |
| TTC | 0.758 | 0.671 [0.625, 0.715] (n 141) | 70 % | 56 % | 67 % | no |
| C | 0.497 | 0.504 [0.463, 0.544] (n 176) | 88 % | 87 % | 49 % | **yes** |
| EP (Spearman) | -0.051 | 0.108 [0.031, 0.179] | 98 % | 96 % | 43 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.824 [0.773, 0.875]**; its percentile among the 64: 0.375 [0.324, 0.432]
- within-token Spearman with path length -- scorer aggregate **-0.176 [-0.264, -0.090]**, predicted EP 0.877 [0.864, 0.889], TRUE PDMS **-0.076 [-0.157, -0.003]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **18.0**, median 13.5 of 64; 84.5 % of tokens have at least one, 46.0 % have at least 16
- the pick scores 0 on **38.5 %** of tokens; all 64 proposals score 0 on only 2.5 %
- the pick leaves the drivable area on **30.5 %** of tokens; 98.0 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 21 % (median) to 28 % (mean) of the 64 proposals score 80 or more, and the best one averages 90.4 PDMS. The planner's pick is better than a random proposal: it realises 16 % of the gain over a random pick (4 % to 30 %). The scorer believes 97 % of proposals stay on the road when 68 % do. Within a scene, AUC per output: NC 0.72 (better than chance), DAC 0.52 (at chance), DDC 0.61 (better than chance), TTC 0.67 (better than chance), C 0.50 (at chance). Its progress output's within-token rank correlation with path length is 0.88 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is -0.08. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**UNDETERMINED**; failing scorer outputs: **C**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep008/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep008/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep008_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
