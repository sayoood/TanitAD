# RESULT E-6 -- selection diagnosis, `sub200_ep015` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 84.2 [79.9, 88.2] |
| **planner's pick** | **61.2 [55.7, 66.6]** |
| random proposal (mean of 64) | 44.1 [39.3, 48.9] |
| medoid, no scorer (pre-registered comparison) | 49.2 [42.5, 55.6] |
| logitsum (EXPLORATORY) | 58.1 [52.7, 63.6] |
| violation_only (EXPLORATORY) | 59.0 [53.0, 64.9] |
| aggregate_without_EP (EXPLORATORY) | 62.8 [57.0, 68.5] |

Paired: pick - random **17.1 [12.5, 21.8]**, oracle - pick **23.0 [19.0, 27.0]**, medoid - pick **-12.0 [-18.4, -6.1]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.426 [0.328, 0.522]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 89.8 | 81.2 | 96.8 |
| DAC | 78.5 | 64.4 | 91.5 |
| DDC | 92.0 | 91.3 | 95.2 |
| EP | 58.2 | 46.3 | 83.2 |
| TTC | 83.5 | 69.2 | 95.5 |
| C | 60.5 | 51.0 | 72.5 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 182 tokens whose PDMS varies: **0.258 [0.186, 0.329]**; pooled over all (token, proposal): 0.539
- the pick is a best proposal on **33.5 %** of tokens; a random pick would be 25.0 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.796 | 0.773 [0.718, 0.824] (n 95) | 48 % | 83 % | 80 % | no |
| DAC | 0.807 | 0.751 [0.703, 0.796] (n 124) | 62 % | 65 % | 64 % | no |
| DDC | 0.747 | 0.534 [0.443, 0.621] (n 90) | 45 % | 99 % | 86 % | no |
| TTC | 0.843 | 0.755 [0.699, 0.810] (n 123) | 62 % | 51 % | 69 % | no |
| C | 0.870 | 0.743 [0.676, 0.806] (n 112) | 56 % | 57 % | 51 % | no |
| EP (Spearman) | -0.113 | 0.145 [0.054, 0.232] | 90 % | 93 % | 46 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.831 [0.801, 0.863]**; its percentile among the 64: 0.278 [0.238, 0.322]
- within-token Spearman with path length -- scorer aggregate **-0.401 [-0.468, -0.331]**, predicted EP 0.986 [0.983, 0.990], TRUE PDMS **0.016 [-0.075, 0.104]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **22.0**, median 14.0 of 64; 79.0 % of tokens have at least one, 48.5 % have at least 16
- the pick scores 0 on **29.0 %** of tokens; all 64 proposals score 0 on only 9.0 %
- the pick leaves the drivable area on **21.5 %** of tokens; 92.0 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 22 % (median) to 34 % (mean) of the 64 proposals score 80 or more, and the best one averages 84.2 PDMS. The planner's pick is better than a random proposal: it realises 43 % of the gain over a random pick (33 % to 52 %). The scorer believes 65 % of proposals stay on the road when 64 % do. Within a scene, AUC per output: NC 0.77 (better than chance), DAC 0.75 (better than chance), DDC 0.53 (at chance), TTC 0.75 (better than chance), C 0.74 (better than chance). Its progress output's within-token rank correlation with path length is 0.99 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is 0.02. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**NOT_SELECTION_BOUND**; failing scorer outputs: **none**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep015/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep015_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
