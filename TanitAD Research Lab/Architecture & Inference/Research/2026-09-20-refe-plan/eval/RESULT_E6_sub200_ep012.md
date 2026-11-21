# RESULT E-6 -- selection diagnosis, `sub200_ep012` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch012.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 91.3 [89.3, 93.2] |
| **planner's pick** | **47.9 [41.7, 54.4]** |
| random proposal (mean of 64) | 44.0 [39.6, 48.4] |
| medoid, no scorer (pre-registered comparison) | 47.4 [41.3, 53.2] |
| logitsum (EXPLORATORY) | 46.3 [40.6, 52.0] |
| violation_only (EXPLORATORY) | 50.4 [44.3, 56.7] |
| aggregate_without_EP (EXPLORATORY) | 52.4 [46.7, 58.0] |

Paired: pick - random **3.9 [-1.5, 9.6]**, oracle - pick **43.4 [37.8, 49.0]**, medoid - pick **-0.5 [-7.3, 5.8]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.082 [-0.033, 0.198]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 89.8 | 82.9 | 100.0 |
| DAC | 63.5 | 65.0 | 99.0 |
| DDC | 88.2 | 91.2 | 97.2 |
| EP | 48.3 | 43.6 | 85.3 |
| TTC | 77.0 | 72.2 | 99.5 |
| C | 59.0 | 56.4 | 89.5 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 198 tokens whose PDMS varies: **0.088 [0.038, 0.142]**; pooled over all (token, proposal): 0.075
- the pick is a best proposal on **17.5 %** of tokens; a random pick would be 12.5 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.767 | 0.710 [0.663, 0.754] (n 112) | 56 % | 82 % | 82 % | no |
| DAC | 0.663 | 0.530 [0.477, 0.581] (n 151) | 76 % | 98 % | 65 % | no |
| DDC | 0.703 | 0.531 [0.445, 0.611] (n 96) | 48 % | 95 % | 87 % | no |
| TTC | 0.739 | 0.658 [0.612, 0.702] (n 140) | 70 % | 56 % | 72 % | no |
| C | 0.572 | 0.514 [0.481, 0.548] (n 149) | 74 % | 82 % | 56 % | **yes** |
| EP (Spearman) | -0.034 | 0.129 [0.055, 0.201] | 99 % | 95 % | 44 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.979 [0.938, 1.017]**; its percentile among the 64: 0.546 [0.499, 0.593]
- within-token Spearman with path length -- scorer aggregate **-0.056 [-0.124, 0.007]**, predicted EP 0.716 [0.688, 0.745], TRUE PDMS **-0.012 [-0.090, 0.066]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **20.7**, median 15.5 of 64; 85.5 % of tokens have at least one, 50.0 % have at least 16
- the pick scores 0 on **42.0 %** of tokens; all 64 proposals score 0 on only 1.0 %
- the pick leaves the drivable area on **36.5 %** of tokens; 99.0 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 24 % (median) to 32 % (mean) of the 64 proposals score 80 or more, and the best one averages 91.3 PDMS. The planner's pick is indistinguishable from a random proposal: it realises 8 % of the gain over a random pick (-3 % to 20 %). The scorer believes 98 % of proposals stay on the road when 65 % do. Within a scene, AUC per output: NC 0.71 (better than chance), DAC 0.53 (at chance), DDC 0.53 (at chance), TTC 0.66 (better than chance), C 0.51 (at chance). Its progress output's within-token rank correlation with path length is 0.72 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is -0.01. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**SELECTION_BOUND**; failing scorer outputs: **C**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep012/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep012/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep012_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
