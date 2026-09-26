# RESULT E-6 -- selection diagnosis, `sub200_ep005` (SPEC_NAVTEST Amendments 3 + 3a)

**Checkpoint:** `D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch005.pt` · **tokens:** 200 over 93 logs (`A1_sub200_tokens.json`) · **proposals:** 64 per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED (12,800 scored plans) · **bootstrap:** log clusters, 10,000 resamples · **evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).

## Gates

- **G1** the dump re-run reproduces the landed seam: max |d pose| **0.0**, 200 of 200 tokens identical -> PASS
- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| **0.0** over 200 tokens -> PASS
- **G3** 64 of 64 single-proposal runs PASS with every token valid -> PASS
- transcribed aggregate reproduces the planner's pick on **100.0 %** of tokens

## (a) PDMS by how the plan is chosen

| choice | PDMS [95 %] |
|---|---|
| best of 64 (oracle) | 81.6 [77.2, 85.5] |
| **planner's pick** | **53.5 [48.7, 58.0]** |
| random proposal (mean of 64) | 29.6 [25.5, 33.8] |
| medoid, no scorer (pre-registered comparison) | 25.7 [19.9, 31.9] |
| logitsum (EXPLORATORY) | 52.4 [47.3, 57.2] |
| violation_only (EXPLORATORY) | 48.2 [43.4, 52.8] |
| aggregate_without_EP (EXPLORATORY) | 47.2 [42.5, 51.9] |

Paired: pick - random **23.9 [18.8, 29.0]**, oracle - pick **28.1 [24.3, 32.0]**, medoid - pick **-27.7 [-34.5, -21.0]** PDMS points. **Selection skill** (pick - random) / (oracle - random) = **0.460 [0.376, 0.537]**.

Sub-scores (x100) at each choice:

| component | pick | random | best proposal |
|---|---|---|---|
| NC | 91.0 | 73.6 | 96.8 |
| DAC | 78.0 | 52.8 | 93.5 |
| DDC | 90.8 | 67.1 | 83.0 |
| EP | 37.8 | 29.5 | 74.4 |
| TTC | 82.5 | 59.9 | 94.0 |
| C | 77.0 | 40.9 | 79.0 |

## (b) Ranking skill within a token

- Spearman(planner aggregate, true PDMS) over the 186 tokens whose PDMS varies: **0.233 [0.181, 0.287]**; pooled over all (token, proposal): 0.381
- the pick is a best proposal on **22.0 %** of tokens; a random pick would be 12.3 %

## (c) Each scorer output against the harness's verdict on the same proposals

| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |
|---|---|---|---|---|---|---|
| NC | 0.418 | 0.267 [0.218, 0.316] (n 132) | 66 % | 71 % | 72 % | **yes** |
| DAC | 0.680 | 0.685 [0.639, 0.731] (n 164) | 82 % | 35 % | 53 % | no |
| DDC | 0.814 | 0.774 [0.720, 0.822] (n 165) | 82 % | 45 % | 62 % | no |
| TTC | 0.394 | 0.330 [0.275, 0.388] (n 156) | 78 % | 66 % | 60 % | **yes** |
| C | 0.613 | 0.737 [0.689, 0.782] (n 191) | 96 % | 54 % | 41 % | no |
| EP (Spearman) | 0.334 | 0.055 [-0.026, 0.134] | 93 % | 23 % | 30 % (mean) | - |

## (d) Path length

- pick's path length / mean of 64: **0.515 [0.472, 0.560]**; its percentile among the 64: 0.113 [0.084, 0.143]
- within-token Spearman with path length -- scorer aggregate **-0.850 [-0.903, -0.797]**, predicted EP -0.995 [-0.996, -0.994], TRUE PDMS **-0.215 [-0.279, -0.151]**

## Descriptive (not pre-registered)

- proposals scoring PDMS >= 80 per token: mean **12.2**, median 5.5 of 64; 72.5 % of tokens have at least one, 29.5 % have at least 16
- the pick scores 0 on **27.5 %** of tokens; all 64 proposals score 0 on only 7.0 %
- the pick leaves the drivable area on **22.0 %** of tokens; 95.0 % of tokens have at least one proposal that stays inside it

## Reading

The proposal head is not the problem: in a typical scene a quarter to a third of the 64 proposals score 80 or more, and the best one averages 81.6 PDMS. The scorer is: its pick is indistinguishable from a random proposal (skill 0.460), it believes 35 % of proposals stay on the road when 53 % do, and within a scene its drivable-area, driving-direction and comfort outputs rank proposals at chance (within-token AUC 0.69 / 0.77 / 0.74), while its progress output follows path length (within-token Spearman -1.00) although the true PDMS does not (-0.22). Pooled AUCs look better than within-token ones because whole scenes differ in difficulty; choosing needs the within-token skill. MECHANISM (consistent with every number here, not yet shown causal): `refe/train.py` supervises the scorer with 8-9 FIXED candidates per frame, each attached to its NEAREST proposal a few metres away (`assign_d` 2.5-5.7 m on the latest micro-batches), so the labels a proposal's scorer output learns are the scores of a different path, and drivable area / comfort change within one metre -- a declared departure from DriveZero, which scores the student's OWN proposals. The fix is a recipe decision for the PI.

## Verdict (rule fixed before the table was read)

**NOT_SELECTION_BOUND**; failing scorer outputs: **NC, TTC**. Rule: AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens. Amendment 3's original clause (unused, reported): False.

## Files

`D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep005/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, `logs/`; single-proposal seams `D:/Projects/TanitAD/data/refe_navtest/seams/proptable/sub200_ep005/`; scores `D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep005_pNN/`. Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), this file by `eval/write_result_e6.py`.
