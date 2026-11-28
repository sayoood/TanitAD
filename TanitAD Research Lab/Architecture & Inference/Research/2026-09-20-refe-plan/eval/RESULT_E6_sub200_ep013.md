# RESULT E-6 -- selection diagnosis, `sub200_ep013` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch013.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 86.3 [82.4, 89.9] |
| **planner's pick** | **49.4 [43.6, 55.5]** |
| random proposal (mean of 64) | 46.5 [42.1, 51.0] |
| medoid, no scorer (pre-registered comparison) | 50.1 [43.7, 56.1] |
| logitsum (EXPLORATORY) | 40.9 [35.2, 46.6] |
| violation_only (EXPLORATORY) | 54.6 [49.1, 60.3] |
| aggregate_without_EP (EXPLORATORY) | 56.1 [50.5, 61.9] |

Paired: pick - random **2.9 [-1.9, 8.0]**, oracle - pick **36.9 [31.3, 42.6]**, medoid - pick **0.7 [-6.0, 6.9]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.073 [-0.047, 0.199]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 89.0 | 82.2 | 98.5 |
| DAC | 70.5 | 69.1 | 94.5 |
| DDC | 88.5 | 90.0 | 93.0 |
| EP | 48.1 | 48.2 | 85.9 |
| TTC | 78.0 | 72.0 | 96.0 |
| C | 53.5 | 48.7 | 72.0 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 189 tokens whose PDMS varies: **0.115 [0.043, 0.192]**; pooled over all (token, proposal): 0.355
- the pick is a best proposal on **21.5 %** of tokens; a random pick would be 21.3 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.736 | 0.730 [0.662, 0.793] (n 104) | 52 % | 86 % | 81 % | no |
| DAC | 0.767 | 0.650 [0.594, 0.713] (n 126) | 63 % | 71 % | 69 % | no |
| DDC | 0.779 | 0.534 [0.463, 0.609] (n 93) | 46 % | 98 % | 85 % | no |
| TTC | 0.783 | 0.745 [0.689, 0.797] (n 129) | 64 % | 51 % | 72 % | no |
| C | 0.409 | 0.389 [0.340, 0.443] (n 128) | 64 % | 39 % | 49 % | **yes** |
| EP (Spearman) | -0.045 | 0.159 [0.075, 0.235] | 92 % | 93 % | 48 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.801 [0.773, 0.830]**; its percentile among the 64: 0.255 [0.227, 0.289]
- within-token Spearman with path length -- scorer aggregate **-0.534 [-0.596, -0.468]**, predicted EP 0.982 [0.978, 0.986], TRUE PDMS **0.002 [-0.083, 0.082]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **21.5**, median 15.0 of 64; 80.0 % of tokens have at least one, 49.5 % have at least 16
- the pick scores 0 on **37.5 %** of tokens; all 64 proposals score 0 on only 5.5 %
- the pick leaves the drivable area on **29.5 %** of tokens; 95.0 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 23 % (median) to 34 % (mean) of the 64 proposals score 80 or more, and the best one averages 86.3 PDMS. The planner's pick is indistinguishable from a random proposal: it realises 7 % of the gain over a random pick (-5 % to 20 %). The scorer believes 71 % of proposals stay on the road when 69 % do. Within a scene, AUC per output: NC 0.73 (better than chance), DAC 0.65 (better than chance), DDC 0.53 (at chance), TTC 0.74 (better than chance), C 0.39 (INVERTED). Its progress output's within-token rank correlation with path length is 0.98 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is 0.00. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**SELECTION_BOUND**; failing scorer outputs: **C**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep013/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep013/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep013_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
