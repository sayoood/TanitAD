# RESULT -- like-for-like PDMS readout for REFe snapshot 016 (015 re-read with Amendment 7's repair)

*Every number is read from the unchanged harness's per-token scores; none is typed. Written by `eval/like_for_like_016.py` (CPU only; torch never imported). Full record: `eval/raw/like_for_like_016/like_for_like_016.json`.*

**Tokens:** W3's 200-token paired subset `A1_sub200_tokens.json` (md5 `114deb5bf6d2631a9cdd6e8c0724273d`), 93 logs -- the learning curve's own tokens, NOT the published split. **Tier:** NAVSIM v1 PDMS (ego pseudo-simulation of an open-loop plan against logged agents). **Repair:** on the planner's native [20, 3] output, heading[19] := heading[18] for the executed plan (`refe/planner.py repair_last_heading`); it acts after selection, so each token's pick is unchanged.

## Headline

| comparison (same 200 tokens) | PDMS | difference [95 %] | separated | better / worse / tied |
|---|---|---|---|---|
| **015 repaired -> 016 repaired (LIKE-FOR-LIKE: the epoch of training)** | 77.94 -> 76.52 | **-1.41 [-6.45, +3.88]** | no | 40 / 103 / 57 |
| 015 shipped -> 015 repaired (the repair alone, this subset) | 61.18 -> 77.94 | **+16.76 [+11.72, +22.13]** | yes | 109 / 24 / 67 |
| 015 shipped -> 016 repaired (the curve's published step: repair + training) | 61.18 -> 76.52 | **+15.34 [+9.97, +21.41]** | yes | 91 / 58 / 51 |

Differences are means of per-token differences at full precision (PDMS 61.18, 77.94, 76.52 are rounded), so a difference can sit 0.01 from the difference of the rounded PDMS: the like-for-like is -1.4138.

Where the like-for-like difference comes from (descriptive; NAVSIM v1 PDMS = NC x DAC x (5 EP + 5 TTC + 2 C) / 12, DDC weight 0): a multiplier moved (NC or DAC): 33 tokens (17 better / 16 worse), +0.58; only ego progress moved: 104 tokens (19 better / 85 worse), -2.17; tied: 57 tokens (0 better / 0 worse), +0.00; TTC or comfort moved, multipliers unchanged: 6 tokens (4 better / 2 worse), +0.18. The contributions sum to the mean difference.

The published +15.34 step decomposes exactly (same tokens, same draws) into the repair's +16.76 and the like-for-like -1.41. For comparison, Amendment 7's confirmation read the repair at +16.20 [+12.83, +19.52] on 923 fresh tokens (43 logs, none in these 200) at the same snapshot; here, on the curve's 200 tokens, the repair reads +16.76 [+11.72, +22.13].

**What the interval answers:** EPISODES only (another draw of logs). One checkpoint per point, one deterministic forward: NOT training variance (another run of epoch 16 -- no replicate exists) and NOT inference variance. Estimator: paired log-cluster bootstrap over the 93 logs, 10000 resamples, percentile 95 %, seed 20260927 (eval/snapshot_pair_under_rule.py's estimator); all three comparisons on the SAME draws. No decision rides on it.

## The rule each point was picked with

| point | selection rule (seam report / proposal dump) | select | last-heading repair (report / dump) |
|---|---|---|---|
| sub200_ep015 | navsim_v1 / navsim_v1 | best / best | ABSENT (the report predates Amendment 7) / ABSENT (the dump predates Amendment 7) |
| sub200_ep016 | navsim_v1 / navsim_v1 | best / best | True / True |

One selection rule for both points (gate R: PASS), so the comparison is like-for-like in selection; the repair is the only executed-plan difference, and this readout removes it. 015's shipped seam carries no repair flag because it predates Amendment 7; gate A proves it executed the UNREPAIRED plan (the unrepaired conversion matches it bit-for-bit on every token, the repaired one on 0).

Apart from the repair, both points ran one inference path (reported; net diff of today's files against their last committed versions on the push mirror): `refe/model.py` vs 4ac00b4: identical; `refe/planner.py` vs 7e9ccdd: +50/-4 lines; `eval/refe_navtest_seam.py` vs 4ac00b4: +14/-5 lines. The changed lines (listed in the JSON) are Amendment 5's rule plumbing -- both points were picked with navsim_v1 -- and Amendment 7's repair with its metadata. A net diff cannot exclude a change made and reverted in between; a GPU re-forward of 015 under today's code with the repair off would be the direct test.

## Validity gates (all must pass before the statistic is read)

| gate | result |
|---|---|
| A: each token's shipped pick recovered from the ep015 native dump by a UNIQUE exact match, to_navsim(traj[token, k]) == shipped seam bit-for-bit | PASS: 200/200 unique; 0-match 0, >1-match 0 |
| R: the same selection rule for both points | PASS: [['navsim_v1', 'best']] |
| B: repaired x, y bit-identical on all 8 poses; headings 0.5-3.5 s identical; the 4.0 s heading == native heading[18] | PASS: x/y True, h 0.5-3.5 s True, h 4.0 s True; changed on 200/200 tokens |
| C: the rebuilt UNREPAIRED seam, scored today, reproduces the pipeline's ep015 scores token-for-token | PASS: PDMS 61.183 vs pipeline 61.183; max per-token |diff| PDMS 0.0, sub-scores 0.0 |
| D: every harness run PASS, every token valid | PASS: refe_lfl016_ep015_unrepaired PASS 200/200; refe_lfl016_ep015_repaired PASS 200/200; pipeline refe_sub200_ep015 PASS 200/200; pipeline refe_sub200_ep016 PASS 200/200 |

Cross-checks (reported): recovered pick == the pipeline's own recorded pick 200/200; == the dump's pick 200/200; the dump's second forward repeated its poses to 0.0 (a deterministic forward). The repair ran as planner.py's own function (compiled from its source, sha256 `8f15c9f62a91`), equal bit-for-bit to an independent literal: True.

CSV-to-seam ties (the harness's own call log names the seam file it served; the file must be unmodified since): REFe_sub200_ep015 PASS (200 seam calls, scored 2026-09-27T09:48:12); REFe_sub200_ep016 PASS (200 seam calls, scored 2026-09-27T16:41:45); REFe_lfl016_ep015_unrepaired PASS (200 seam calls, scored 2026-09-27T17:25:23); REFe_lfl016_ep015_repaired PASS (200 seam calls, scored 2026-09-27T17:25:23).

Estimator control: the same bootstrap on Amendment 7's own CSVs re-reads its banked +16.20 [+12.83, +19.52] as +16.20 [+12.83, +19.52] (exact).

## Reported, not gating

NAVSIM sub-scores x100: 015_shipped NC 89.75, DAC 78.50, EP 58.24, TTC 83.50, C 60.50, DDC 92.00; 015_repaired NC 92.25, DAC 92.00, EP 72.05, TTC 88.00, C 100.00, DDC 91.75; 016_repaired NC 94.25, DAC 90.50, EP 66.31, TTC 90.50, C 100.00, DDC 94.25.

Sub-score deltas x100, like-for-like (016 repaired - 015 repaired): NC +2.00, DAC -1.50, EP -5.74, TTC +2.50, C +0.00, DDC +2.50; the repair alone: NC +2.50, DAC +13.50, EP +13.81, TTC +4.50, C +39.50, DDC -0.25.

Heading at t = 4.0 s, share |heading| > pi: 015 shipped 58.0 %, 015 repaired 0.0 %, 016 shipped 0.0 % -- consistent with 016's report that it executed the repaired plan.

The exploration that motivated Amendment 7 (same 200 tokens, `slow_copies.json` 'hold') banked 77.94 PDMS, +16.76 [11.72, 22.13]; its repaired seam is bit-identical to this one: True; per-token max |PDMS diff| vs this run 0.0.

**The four metric families are NOT recomputed, and need not be.** They read positions only (the NAVSIM family adapter drops the heading column; its heading_mae_deg is the path tangent) -- Amendment 7 ran `families6.py` on a shipped and a repaired seam and found headline numbers identical = True, full blocks identical apart from run labels = True -- and gate B proves 015-repaired has 015-shipped's positions bit-for-bit. So the pipeline's existing per-snapshot family readouts (`D:/Projects/TanitAD/data/refe_navtest/points/sub200_ep015/4_families.json`, on the unrepaired seam; `D:/Projects/TanitAD/data/refe_navtest/points/sub200_ep016/4_families.json`, on the repaired seam) are already like-for-like. Strategic: unavailable in NAVSIM.

## Extension -- all 64 proposals: ep015 repaired vs ep016 (best-of-64, mean-of-64, share >= 80)

*The PI's question: did the planner's BEST proposal degenerate or stabilise? All 64 of ep015's proposals are re-read with the repair (built from the same native dump, each of the 64 scored by `score_navtest_refe.py` UNCHANGED, 2 workers), so the comparison with ep016 -- whose 64 the pipeline already executed with the repair -- is like-for-like.*

| pair (same 200 tokens) | best of 64 | mean of 64 | share of the 64 with PDMS >= 80 (%) |
|---|---|---|---|
| 015 unrepaired -> 016 (as the two E-6 tables stand) | 84.19 -> 96.23: **+12.03 [+7.70, +16.58]** separated | 44.13 -> 60.25: **+16.12 [+11.76, +20.53]** separated | 34.39 -> 56.50: **+22.11 [+17.15, +27.06]** separated |
| **015 repaired -> 016 (LIKE-FOR-LIKE)** | 98.42 -> 96.23: **-2.19 [-4.90, -0.02]** separated | 63.05 -> 60.25: **-2.81 [-6.59, +0.66]** | 58.91 -> 56.50: **-2.41 [-6.29, +1.29]** |
| 015 unrepaired -> 015 repaired (the repair on all 64) | 84.19 -> 98.42: **+14.22 [+10.20, +18.54]** separated | 44.13 -> 63.05: **+18.93 [+15.12, +23.00]** separated | 34.39 -> 58.91: **+24.52 [+19.99, +29.22]** separated |

Like-for-like, best-of-64 moved -2.19 [-4.90, -0.02] (separated), mean-of-64 -2.81 [-6.59, +0.66] (not separated), share >= 80 -2.41 [-6.29, +1.29] points (not separated). ep015 unrepaired, from its existing E-6 table: best-of-64 84.19, mean-of-64 44.13.

**How far a separated like-for-like interval here reaches:** best_of_64 -2.19 [-4.90, -0.02] -- the bound nearest zero is 0.02 points from it. The interval answers another draw of EPISODES only; one training run and two consecutive snapshots cannot say whether another run of epoch 16 would show the same, so it is necessary, not sufficient, for calling this an effect of training (H-ESTIM-SEED-1: a same-flags replicate cleared 'separated' on 9.5 % of cells on the tiny rig).

Estimator: `eval/oracle_trend.py` UNCHANGED (sha256 `ff8253735991`), `--pairs 015:016 015_repaired:016 015:015_repaired --boot 10000`, reading this script's table `proptable/sub200_ep015_repaired/table.npz` (proposal_table.py's E-6 format: ep015's logits, pick and rule, the repaired poses and their harness scores) beside the pipeline's ep015 and ep016 tables. paired log-cluster bootstrap over the logs, percentile 95 %, seed 20260927 (oracle_trend.py, unchanged; snapshot_pair_under_rule.py's estimator). It answers EPISODES only (another draw of logs); one checkpoint each, deterministic forward.

Gates: (a) 12800/12800 unrepaired rebuilds bit-identical to the pipeline's E-6 poses (which equal its proposals.npz: True) -- PASS; (b) x, y bit-identical True, headings 0.5-3.5 s identical True, 4.0 s heading == native heading[18] True (changed on 12800/12800) -- PASS; the pick column == the pick readout's repaired seam True, table pick == recovered pick 200/200 -- PASS; 64/64 harness runs PASS with every token valid, each tied to its seam file -- PASS; the table's pick column reproduces the pick readout's repaired scores (max |diff| PDMS 0.0, sub-scores 0.0; PDMS 77.9381) -- PASS. The reader oracle_trend.py uses (`snapshot_pair_under_rule.load`, which asserts each table's own rule reproduces its pick) accepts all three: sub200_ep015 True, sub200_ep015_repaired True, sub200_ep016 True.

The ep016 table: its own gates {'G1': True, 'G3': True, 'G2': True}; its dump records repair_last_heading = True; its pick column == the ep016 seam True. Share of the 64 proposals with |heading(4.0 s)| > pi: ep016 0.0 %, ep015 unrepaired 60.1 %, ep015 repaired 0.0 % -- ep016's 64 are the repaired ones.

**Position-only oracle ADE needs no re-read:** 0.4189 m at 015 -> 0.3241 m at 016 (the seam reports' `proposals.ade_oracle_m`, best of 64 vs the logged human future). It is computed from x, y only, and gate (b) proves the repair moves no position, so 015's value is 015-repaired's.

## What this does not answer

One training run, two consecutive snapshots, one deterministic forward each: the interval is over episodes only. Whether another run of epoch 16 would land elsewhere (training variance) is not measured -- there is no replicate. The 200 tokens are the learning curve's paired subset, not the published navtest split.

Artifacts and checksums: `eval/raw/like_for_like_016/MANIFEST.md`.
