# RESULT E-6 -- selection diagnosis, `sub200_final` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/model_final.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 98.3 [97.1, 99.1] |
| **planner's pick** | **84.2 [80.5, 88.0]** |
| random proposal (mean of 64) | 75.7 [72.6, 78.7] |
| medoid, no scorer (pre-registered comparison) | 82.4 [78.1, 86.5] |
| logitsum (EXPLORATORY) | 82.3 [78.8, 85.9] |
| violation_only (EXPLORATORY) | 82.8 [79.2, 86.5] |
| aggregate_without_EP (EXPLORATORY) | 81.5 [78.1, 85.1] |

Paired: pick - random **8.5 [5.0, 12.1]**, oracle - pick **14.1 [10.6, 17.7]**, medoid - pick **-1.7 [-6.5, 2.8]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.375 [0.232, 0.517]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 97.0 | 92.5 | 100.0 |
| DAC | 95.0 | 89.0 | 99.5 |
| DDC | 98.2 | 97.7 | 98.5 |
| EP | 75.5 | 72.4 | 96.5 |
| TTC | 93.5 | 85.6 | 100.0 |
| C | 100.0 | 100.0 | 100.0 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 194 tokens whose PDMS varies: **0.168 [0.097, 0.235]**; pooled over all (token, proposal): 0.309
- the pick is a best proposal on **28.5 %** of tokens; a random pick would be 27.3 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.881 | 0.864 [0.817, 0.906] (n 67) | 34 % | 87 % | 92 % | no |
| DAC | 0.815 | 0.737 [0.696, 0.783] (n 85) | 42 % | 69 % | 89 % | no |
| DDC | 0.827 | 0.632 [0.516, 0.738] (n 39) | 20 % | 99 % | 97 % | no |
| TTC | 0.852 | 0.797 [0.740, 0.845] (n 98) | 49 % | 58 % | 86 % | no |
| C | 0.661 | 1.000 [1.000, 1.000] (n 1) | 0 % | 58 % | 100 % | no |
| EP (Spearman) | 0.143 | 0.430 [0.342, 0.510] | 96 % | 92 % | 72 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.883 [0.855, 0.912]**; its percentile among the 64: 0.326 [0.285, 0.370]
- within-token Spearman with path length -- scorer aggregate **-0.149 [-0.218, -0.078]**, predicted EP 0.989 [0.986, 0.992], TRUE PDMS **0.338 [0.236, 0.431]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **47.0**, median 52.5 of 64; 99.0 % of tokens have at least one, 91.0 % have at least 16
- the pick scores 0 on **7.5 %** of tokens; all 64 proposals score 0 on only 0.5 %
- the pick leaves the drivable area on **5.0 %** of tokens; 99.5 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 82 % (median) to 73 % (mean) of the 64 proposals score 80 or more, and the best one averages 98.3 PDMS. The planner's pick is better than a random proposal: it realises 38 % of the gain over a random pick (23 % to 52 %). The scorer believes 69 % of proposals stay on the road when 89 % do. Within a scene, AUC per output: NC 0.86 (better than chance), DAC 0.74 (better than chance), DDC 0.63 (better than chance), TTC 0.80 (better than chance), C 1.00 (better than chance). Its progress output's within-token rank correlation with path length is 0.99 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is 0.34. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**UNDETERMINED**; failing scorer outputs: **none**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_final/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_final/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_final_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
