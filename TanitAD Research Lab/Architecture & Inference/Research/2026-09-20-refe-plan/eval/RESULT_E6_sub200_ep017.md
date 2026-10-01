# RESULT E-6 -- selection diagnosis, `sub200_ep017` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch017.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 98.8 [98.4, 99.1] |
| **planner's pick** | **82.6 [78.8, 86.2]** |
| random proposal (mean of 64) | 70.2 [66.8, 73.6] |
| medoid, no scorer (pre-registered comparison) | 78.5 [73.3, 83.2] |
| logitsum (EXPLORATORY) | 79.7 [74.9, 84.1] |
| violation_only (EXPLORATORY) | 81.8 [78.2, 85.3] |
| aggregate_without_EP (EXPLORATORY) | 80.0 [76.8, 83.1] |

Paired: pick - random **12.4 [8.4, 16.4]**, oracle - pick **16.1 [12.7, 19.9]**, medoid - pick **-4.1 [-9.3, 0.6]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.435 [0.305, 0.550]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 96.8 | 90.6 | 100.0 |
| DAC | 96.0 | 85.4 | 100.0 |
| DDC | 94.0 | 95.6 | 96.8 |
| EP | 71.6 | 66.6 | 97.0 |
| TTC | 93.0 | 83.5 | 100.0 |
| C | 100.0 | 99.9 | 100.0 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 198 tokens whose PDMS varies: **0.161 [0.093, 0.230]**; pooled over all (token, proposal): 0.399
- the pick is a best proposal on **23.0 %** of tokens; a random pick would be 23.1 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.876 | 0.867 [0.824, 0.907] (n 80) | 40 % | 84 % | 90 % | no |
| DAC | 0.831 | 0.792 [0.735, 0.856] (n 101) | 50 % | 69 % | 85 % | no |
| DDC | 0.805 | 0.632 [0.542, 0.725] (n 65) | 32 % | 98 % | 94 % | no |
| TTC | 0.850 | 0.800 [0.738, 0.857] (n 116) | 58 % | 56 % | 83 % | no |
| C | 0.435 | 0.945 [0.842, 1.000] (n 3) | 2 % | 57 % | 100 % | no |
| EP (Spearman) | 0.145 | 0.372 [0.289, 0.451] | 98 % | 92 % | 67 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.810 [0.782, 0.837]**; its percentile among the 64: 0.233 [0.204, 0.266]
- within-token Spearman with path length -- scorer aggregate **-0.406 [-0.464, -0.348]**, predicted EP 0.989 [0.987, 0.992], TRUE PDMS **0.260 [0.166, 0.344]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **43.6**, median 48.0 of 64; 100.0 % of tokens have at least one, 91.0 % have at least 16
- the pick scores 0 on **7.0 %** of tokens; all 64 proposals score 0 on only 0.0 %
- the pick leaves the drivable area on **4.0 %** of tokens; 100.0 % of tokens have at least one proposal that stays inside it

## Reading

In a typical scene 75 % (median) to 68 % (mean) of the 64 proposals score 80 or more, and the best one averages 98.8 PDMS. The planner's pick is better than a random proposal: it realises 44 % of the gain over a random pick (31 % to 55 %). The scorer believes 69 % of proposals stay on the road when 85 % do. Within a scene, AUC per output: NC 0.87 (better than chance), DAC 0.79 (better than chance), DDC 0.63 (better than chance), TTC 0.80 (better than chance), C 0.95 (better than chance). Its progress output's within-token rank correlation with path length is 0.99 (+1 = longer paths predicted to progress more, -1 = the reverse); the true PDMS's correlation with path length is 0.26. Pooled AUCs mix in between-scene difficulty; choosing needs the within-token skill.

## Verdict (rule fixed before the table was read)

**NOT_SELECTION_BOUND**; failing scorer outputs: **none**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep017/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep017/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep017_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
