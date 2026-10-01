# RESULT E-6 -- selection diagnosis, `sub200_ep019` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch019.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 98.5 [98.0, 99.0] |
| **planner's pick** | **81.6 [77.3, 85.6]** |
| random proposal (mean of 64) | 70.4 [67.1, 73.8] |
| medoid, no scorer (pre-registered comparison) | 78.8 [74.0, 83.2] |
| logitsum (EXPLORATORY) | 77.5 [73.5, 81.5] |
| violation_only (EXPLORATORY) | 80.1 [75.5, 84.3] |
| aggregate_without_EP (EXPLORATORY) | 78.0 [74.0, 82.0] |

Paired: pick - random **11.1 [6.8, 15.5]**, oracle - pick **17.0 [13.1, 21.1]**, medoid - pick **-2.7 [-8.3, 2.5]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.396 [0.252, 0.530]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 98.5 | 91.3 | 100.0 |
| DAC | 92.5 | 85.0 | 100.0 |
| DDC | 97.5 | 96.4 | 97.0 |
| EP | 71.9 | 66.6 | 97.0 |
| TTC | 94.5 | 83.7 | 99.5 |
| C | 100.0 | 100.0 | 100.0 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 195 tokens whose PDMS varies: **0.194 [0.110, 0.276]**; pooled over all (token, proposal): 0.343
- the pick is a best proposal on **25.0 %** of tokens; a random pick would be 22.6 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.872 | 0.886 [0.845, 0.924] (n 77) | 38 % | 86 % | 91 % | no |
| DAC | 0.811 | 0.787 [0.741, 0.838] (n 97) | 48 % | 67 % | 85 % | no |
| DDC | 0.806 | 0.659 [0.581, 0.729] (n 63) | 32 % | 98 % | 95 % | no |
| TTC | 0.839 | 0.842 [0.796, 0.882] (n 109) | 55 % | 56 % | 84 % | no |
| C | 0.627 | 0.935 [0.933, 0.936] (n 2) | 1 % | 57 % | 100 % | no |
| EP (Spearman) | 0.138 | 0.379 [0.295, 0.459] | 98 % | 91 % | 67 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.848 [0.821, 0.875]**; its percentile among the 64: 0.277 [0.242, 0.313]
- within-token Spearman with path length -- scorer aggregate **-0.298 [-0.371, -0.227]**, predicted EP 0.990 [0.987, 0.992], TRUE PDMS **0.282 [0.182, 0.372]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **43.1**, median 46.5 of 64; 99.5 % of tokens have at least one, 90.0 % have at least 16
- the pick scores 0 on **9.0 %** of tokens; all 64 proposals score 0 on only 0.0 %
- the pick leaves the drivable area on **7.5 %** of tokens; 100.0 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 73 % (median) to 67 % (mean) of the 64 proposals score 80 or more, and the best one averages 98.5 PDMS. The planner's pick is better than a random proposal: it realises 40 % of the gain over a random pick (25 % to 53 %). The scorer believes 67 % of proposals stay on the road when 85 % do. Within a scene, AUC per output: NC 0.89 (better than chance), DAC 0.79 (better than chance), DDC 0.66 (better than chance), TTC 0.84 (better than chance), C 0.93 (better than chance). Its progress output's within-token rank correlation with path length is 0.99 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is 0.28. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**NOT_SELECTION_BOUND**; failing scorer outputs: **none**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep019/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep019/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep019_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
