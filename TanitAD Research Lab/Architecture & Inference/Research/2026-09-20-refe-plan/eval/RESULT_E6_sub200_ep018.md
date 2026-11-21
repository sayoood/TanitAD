# RESULT E-6 -- selection diagnosis, `sub200_ep018` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch018.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 99.3 [99.0, 99.6] |
| **planner's pick** | **83.7 [80.5, 86.8]** |
| random proposal (mean of 64) | 70.0 [66.5, 73.6] |
| medoid, no scorer (pre-registered comparison) | 77.3 [71.5, 82.6] |
| logitsum (EXPLORATORY) | 75.6 [71.2, 79.7] |
| violation_only (EXPLORATORY) | 82.6 [79.4, 85.8] |
| aggregate_without_EP (EXPLORATORY) | 80.2 [77.1, 83.4] |

Paired: pick - random **13.7 [10.2, 17.0]**, oracle - pick **15.6 [12.5, 18.7]**, medoid - pick **-6.4 [-11.4, -1.7]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.468 [0.369, 0.560]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 98.0 | 89.7 | 100.0 |
| DAC | 95.5 | 85.5 | 100.0 |
| DDC | 96.5 | 96.6 | 98.0 |
| EP | 74.0 | 67.3 | 98.2 |
| TTC | 93.5 | 81.6 | 100.0 |
| C | 100.0 | 100.0 | 100.0 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 197 tokens whose PDMS varies: **0.245 [0.169, 0.318]**; pooled over all (token, proposal): 0.337
- the pick is a best proposal on **25.0 %** of tokens; a random pick would be 27.3 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.865 | 0.886 [0.848, 0.921] (n 84) | 42 % | 84 % | 89 % | no |
| DAC | 0.805 | 0.811 [0.763, 0.861] (n 101) | 50 % | 66 % | 85 % | no |
| DDC | 0.789 | 0.567 [0.474, 0.663] (n 64) | 32 % | 97 % | 95 % | no |
| TTC | 0.848 | 0.832 [0.786, 0.874] (n 115) | 57 % | 54 % | 82 % | no |
| C | 0.459 | 0.920 [0.806, 1.000] (n 3) | 2 % | 52 % | 100 % | no |
| EP (Spearman) | 0.102 | 0.309 [0.217, 0.396] | 98 % | 91 % | 67 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.817 [0.787, 0.847]**; its percentile among the 64: 0.262 [0.226, 0.300]
- within-token Spearman with path length -- scorer aggregate **-0.323 [-0.389, -0.259]**, predicted EP 0.989 [0.986, 0.991], TRUE PDMS **0.195 [0.094, 0.288]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **42.4**, median 45.0 of 64; 99.5 % of tokens have at least one, 92.0 % have at least 16
- the pick scores 0 on **6.5 %** of tokens; all 64 proposals score 0 on only 0.0 %
- the pick leaves the drivable area on **4.5 %** of tokens; 100.0 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 70 % (median) to 66 % (mean) of the 64 proposals score 80 or more, and the best one averages 99.3 PDMS. The planner's pick is better than a random proposal: it realises 47 % of the gain over a random pick (37 % to 56 %). The scorer believes 66 % of proposals stay on the road when 85 % do. Within a scene, AUC per output: NC 0.89 (better than chance), DAC 0.81 (better than chance), DDC 0.57 (better than chance), TTC 0.83 (better than chance), C 0.92 (better than chance). Its progress output's within-token rank correlation with path length is 0.99 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is 0.20. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**NOT_SELECTION_BOUND**; failing scorer outputs: **none**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep018/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep018/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep018_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
