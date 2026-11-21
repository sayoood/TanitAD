# RESULT E-6 -- selection diagnosis, `sub200_ep014` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch014.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 88.0 [83.6, 91.6] |
| **planner's pick** | **58.9 [52.7, 65.3]** |
| random proposal (mean of 64) | 44.7 [39.7, 49.9] |
| medoid, no scorer (pre-registered comparison) | 46.0 [39.3, 53.0] |
| logitsum (EXPLORATORY) | 44.5 [38.6, 50.8] |
| violation_only (EXPLORATORY) | 60.3 [54.8, 65.8] |
| aggregate_without_EP (EXPLORATORY) | 60.2 [53.9, 66.4] |

Paired: pick - random **14.3 [9.7, 19.1]**, oracle - pick **29.0 [24.1, 33.8]**, medoid - pick **-13.0 [-20.1, -6.5]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.329 [0.233, 0.431]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 91.0 | 82.5 | 98.5 |
| DAC | 78.0 | 65.2 | 95.0 |
| DDC | 94.8 | 92.1 | 95.5 |
| EP | 56.2 | 46.5 | 86.3 |
| TTC | 82.5 | 69.9 | 97.0 |
| C | 62.0 | 51.3 | 78.5 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 190 tokens whose PDMS varies: **0.240 [0.172, 0.309]**; pooled over all (token, proposal): 0.489
- the pick is a best proposal on **25.5 %** of tokens; a random pick would be 20.9 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.791 | 0.788 [0.739, 0.834] (n 104) | 52 % | 83 % | 82 % | no |
| DAC | 0.793 | 0.718 [0.677, 0.763] (n 127) | 64 % | 62 % | 65 % | no |
| DDC | 0.756 | 0.563 [0.494, 0.630] (n 93) | 46 % | 97 % | 87 % | no |
| TTC | 0.843 | 0.760 [0.707, 0.810] (n 129) | 64 % | 54 % | 70 % | no |
| C | 0.566 | 0.504 [0.456, 0.552] (n 136) | 68 % | 38 % | 51 % | **yes** |
| EP (Spearman) | -0.124 | 0.120 [0.031, 0.207] | 94 % | 91 % | 46 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.786 [0.753, 0.821]**; its percentile among the 64: 0.235 [0.201, 0.271]
- within-token Spearman with path length -- scorer aggregate **-0.524 [-0.587, -0.460]**, predicted EP 0.988 [0.985, 0.991], TRUE PDMS **-0.029 [-0.118, 0.056]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **21.5**, median 13.0 of 64; 83.5 % of tokens have at least one, 48.0 % have at least 16
- the pick scores 0 on **29.0 %** of tokens; all 64 proposals score 0 on only 5.0 %
- the pick leaves the drivable area on **22.0 %** of tokens; 96.0 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 20 % (median) to 34 % (mean) of the 64 proposals score 80 or more, and the best one averages 88.0 PDMS. The planner's pick is better than a random proposal: it realises 33 % of the gain over a random pick (23 % to 43 %). The scorer believes 62 % of proposals stay on the road when 65 % do. Within a scene, AUC per output: NC 0.79 (better than chance), DAC 0.72 (better than chance), DDC 0.56 (better than chance), TTC 0.76 (better than chance), C 0.50 (at chance). Its progress output's within-token rank correlation with path length is 0.99 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is -0.03. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**UNDETERMINED**; failing scorer outputs: **C**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep014/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep014/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep014_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
