# RESULT E-6 -- selection diagnosis, `sub200_ep016` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch016.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 96.2 [93.7, 98.3] |
| **planner's pick** | **76.5 [71.2, 82.0]** |
| random proposal (mean of 64) | 60.2 [55.8, 64.7] |
| medoid, no scorer (pre-registered comparison) | 64.2 [58.1, 70.4] |
| logitsum (EXPLORATORY) | 74.2 [69.5, 79.1] |
| violation_only (EXPLORATORY) | 76.1 [71.3, 80.8] |
| aggregate_without_EP (EXPLORATORY) | 75.6 [71.5, 79.5] |

Paired: pick - random **16.3 [10.3, 22.7]**, oracle - pick **19.7 [14.6, 24.8]**, medoid - pick **-12.3 [-19.5, -5.7]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.452 [0.301, 0.598]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 94.2 | 84.4 | 99.0 |
| DAC | 90.5 | 78.7 | 97.0 |
| DDC | 94.2 | 92.0 | 95.8 |
| EP | 66.3 | 58.0 | 95.1 |
| TTC | 90.5 | 75.5 | 98.5 |
| C | 100.0 | 99.9 | 100.0 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 191 tokens whose PDMS varies: **0.234 [0.142, 0.328]**; pooled over all (token, proposal): 0.470
- the pick is a best proposal on **29.0 %** of tokens; a random pick would be 26.0 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.869 | 0.849 [0.797, 0.894] (n 93) | 46 % | 79 % | 84 % | no |
| DAC | 0.817 | 0.828 [0.771, 0.891] (n 105) | 52 % | 68 % | 79 % | no |
| DDC | 0.755 | 0.557 [0.453, 0.655] (n 91) | 46 % | 99 % | 90 % | no |
| TTC | 0.864 | 0.821 [0.767, 0.869] (n 120) | 60 % | 50 % | 76 % | no |
| C | 0.385 | 0.923 [0.822, 0.995] (n 3) | 2 % | 55 % | 100 % | no |
| EP (Spearman) | 0.020 | 0.242 [0.151, 0.328] | 96 % | 93 % | 58 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.769 [0.738, 0.801]**; its percentile among the 64: 0.235 [0.206, 0.266]
- within-token Spearman with path length -- scorer aggregate **-0.414 [-0.478, -0.352]**, predicted EP 0.989 [0.985, 0.991], TRUE PDMS **0.154 [0.052, 0.250]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **36.2**, median 38.0 of 64; 97.0 % of tokens have at least one, 80.0 % have at least 16
- the pick scores 0 on **14.0 %** of tokens; all 64 proposals score 0 on only 3.0 %
- the pick leaves the drivable area on **9.5 %** of tokens; 97.5 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 59 % (median) to 56 % (mean) of the 64 proposals score 80 or more, and the best one averages 96.2 PDMS. The planner's pick is better than a random proposal: it realises 45 % of the gain over a random pick (30 % to 60 %). The scorer believes 68 % of proposals stay on the road when 79 % do. Within a scene, AUC per output: NC 0.85 (better than chance), DAC 0.83 (better than chance), DDC 0.56 (better than chance), TTC 0.82 (better than chance), C 0.92 (better than chance). Its progress output's within-token rank correlation with path length is 0.99 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is 0.15. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**NOT_SELECTION_BOUND**; failing scorer outputs: **none**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep016/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep016/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep016_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
