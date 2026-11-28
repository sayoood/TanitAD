# RESULT E-6 -- selection diagnosis, `sub200_ep020` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch020.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 99.0 [98.6, 99.3] |
| **planner's pick** | **84.8 [81.6, 88.1]** |
| random proposal (mean of 64) | 71.9 [68.6, 75.2] |
| medoid, no scorer (pre-registered comparison) | 80.1 [75.3, 84.3] |
| logitsum (EXPLORATORY) | 81.3 [77.2, 85.1] |
| violation_only (EXPLORATORY) | 82.6 [79.0, 85.9] |
| aggregate_without_EP (EXPLORATORY) | 81.8 [78.4, 85.2] |

Paired: pick - random **13.0 [9.5, 16.7]**, oracle - pick **14.1 [10.9, 17.3]**, medoid - pick **-4.8 [-10.1, 0.2]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.479 [0.366, 0.588]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 97.5 | 92.0 | 100.0 |
| DAC | 96.0 | 85.5 | 100.0 |
| DDC | 98.2 | 96.0 | 96.8 |
| EP | 75.7 | 68.7 | 97.5 |
| TTC | 94.5 | 84.2 | 100.0 |
| C | 100.0 | 100.0 | 100.0 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 197 tokens whose PDMS varies: **0.206 [0.138, 0.274]**; pooled over all (token, proposal): 0.360
- the pick is a best proposal on **29.5 %** of tokens; a random pick would be 25.0 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.866 | 0.861 [0.820, 0.900] (n 75) | 38 % | 86 % | 92 % | no |
| DAC | 0.834 | 0.773 [0.723, 0.827] (n 97) | 48 % | 69 % | 86 % | no |
| DDC | 0.810 | 0.627 [0.553, 0.696] (n 64) | 32 % | 99 % | 94 % | no |
| TTC | 0.840 | 0.814 [0.766, 0.857] (n 114) | 57 % | 56 % | 84 % | no |
| C | 0.652 | 0.888 [0.847, 0.929] (n 2) | 1 % | 58 % | 100 % | no |
| EP (Spearman) | 0.135 | 0.376 [0.299, 0.450] | 98 % | 92 % | 69 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.863 [0.836, 0.889]**; its percentile among the 64: 0.309 [0.273, 0.347]
- within-token Spearman with path length -- scorer aggregate **-0.215 [-0.285, -0.144]**, predicted EP 0.989 [0.986, 0.991], TRUE PDMS **0.275 [0.188, 0.354]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **44.4**, median 49.5 of 64; 100.0 % of tokens have at least one, 91.0 % have at least 16
- the pick scores 0 on **6.5 %** of tokens; all 64 proposals score 0 on only 0.0 %
- the pick leaves the drivable area on **4.0 %** of tokens; 100.0 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 77 % (median) to 69 % (mean) of the 64 proposals score 80 or more, and the best one averages 99.0 PDMS. The planner's pick is better than a random proposal: it realises 48 % of the gain over a random pick (37 % to 59 %). The scorer believes 69 % of proposals stay on the road when 86 % do. Within a scene, AUC per output: NC 0.86 (better than chance), DAC 0.77 (better than chance), DDC 0.63 (better than chance), TTC 0.81 (better than chance), C 0.89 (better than chance). Its progress output's within-token rank correlation with path length is 0.99 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is 0.27. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**NOT_SELECTION_BOUND**; failing scorer outputs: **none**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep020/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep020/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep020_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
