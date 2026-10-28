# RESULT_V8F_LAUNCH_INTENT: O's launch-sample along-track weight on DAG-OACC's recipe, aimed at the crawl rows. Outcome **HARMS** (arm A, W 1.58926, not adopted)

Claim row `H-V8F-LAUNCH-INTENT-1`. Pre-registration `Project Steering/PREREG_V8F_LAUNCH_INTENT.md`, promoted at `77234276` (13:49:00; sha256 `469bac1f…`), with **Amendment 1** (`A1-2026-10-28`, the K-BUILD data-module clause), landed at `e28516e8` (15:24:46; sha256 of the amended text `6ae79f2f…`) before any row was read. Read 2026-10-28 by the pre-registered reader of record `launch_intent_read.py` (sha256 `fb3d2d7d…`, the Amendment 1 reader): all **206 reader controls PASS**, wall 341.6 s. Output `raw/2026-10-28-launch-intent-read/read_A1/LAUNCH_INTENT_READ.json` (sha256 `e451e1a6…`), banked at `d7dc9f11` (17:11:44).

* **The word, as registered.** **HARMS.** Arm A is not adopted. It failed its pre-registered bar: the decisive R-MOVE is false at both seeds, and the guard R-ADE fails at both seeds. Nothing below softens that.
* **Tier.** T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan; plain arms). Never 'closed loop'. The R-T0 points and the slow-draw cells are at T0: T0 (teacher-forced own-code, eval-141, ages cycle: box 2's standard eval). Never 'driving performance'.
* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed and per pair, paired, on the same windows and matched cells (JSON `estimator`). Intervals are 95 % (INHERITED: the convention of `RESULT_V8F_DAG_ORC.md`; the JSON stores the bounds only). It answers *"would another draw of EPISODES say this?"*. The question *"would another training run say this?"* is answered by the replicate pair at each seed: a lever effect is credited at a seed only when both pairs show it (`H-ESTIM-SEED-1`). The planner is deterministic, so no inference-seed question arises. A T0 point carries no interval: the JSON stores the point only.
* **Arms** (box 2, launch commit `d8c9f7bf`, 13:51:52; four gates PASS, four run records PASS, four K-SHARE PASS). A_s = DAG-OACC_s's training argv verbatim plus `--launch-intent-w` 1.58926 (one third of the along-track weight on the launch samples). A_rep,s = REP_s's argv (trainer seed 10 / 11) plus the same flag. CTRL_s = DAG-OACC_s and REP_s are banked, so no control was retrained. Checkpoint sha256 prefixes: A `deb8f28b…` (s0) / `abe0fd24…` (s1); A_rep `b86f7d43…` / `2900ff73…`. **Arm B (W 0.726175) was not trained and is not read.**
* **Coverage.** The four families at T1-kin are in section 3. **The T0 four families are NOT RUN** (section 3.6). **M-LAUNCH, the anti-echo block and the M2 / L2 launch-subset rows are NOT BUILT** by the reader (section 3.7). They are work items, and the outcome above does not use them.

**Summary** (each value is in the section named; evidence class and tier as stated there).

* **HARMS (section 1).** Controls pass. R-ADE fails at s0 in the replicate pair (ADE6, in m: +0.0431 [+0.0137, +0.0711] W) and at s1 in the main pair (ADE6, in m: +0.0483 [+0.0105, +0.0884] W). R-TLAT, R-TLON, R-T0 and R-NEW hold at both seeds. R-KEEP and R-NOBASE hold at both seeds.
* **R-MOVE is false at both seeds (section 2.3).** 2 of the 16 crawl cells are separated-better and 1 clears its bar: L2 curvature in the replicate pair at s0, -0.0215 [-0.0423, -0.0048] B against a bar of 0.0136. Its main-pair partner reads -0.0075 [-0.0212, +0.0025] –. No crawl row moves by its bar in both pairs at either seed.
* **R-CRAWL is false at both seeds (section 2.4).** All 16 X − P10 crawl cells (4 rows × 2 pairs × 2 seeds) are separated-worse.
* **Reading, decides nothing (section 5).** At each seed the separated ADE6 cells flip sign between the main pair and the replicate pair.
* **Next (section 6).** Committed: B (W 0.726175, 4 box-2 runs) with its own PI go. NEXT-LAUNCH (PREREG section 11) routes by M-LAUNCH, which is built separately; its slot is open in section 6.2.

## 1. The outcome, as the table decides it

Evidence class MEASURED (ours). Tier and estimator as in the header: T1-kin unless a cell says T0; paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. Source: `rows.A.<seed>.pairs.<pair>` and `rows.A.<seed>` in the JSON. X is A (main pair) or A_rep (replicate pair); Y is the pair's control, CTRL = DAG-OACC or REP.

| predicate (PREREG section 10) | s0 main (A − CTRL) | s0 replicate (A_rep − REP) | **s0, both pairs** | s1 main (A − CTRL) | s1 replicate (A_rep − REP) | **s1, both pairs** |
|---|---|---|---|---|---|---|
| controls (reader, 206) | PASS | PASS | PASS | PASS | PASS | PASS |
| R-TLAT (T head macro-F1, ticks >= 1, X − Y not separated-worse) | held | held | **held** | held | held | **held** |
| R-TLON (T head macro-F1, ticks >= 1, X − Y not separated-worse) | held | held | **held** | held | held | **held** |
| R-ADE (T1-kin ADE6, X − Y, not separated-worse) | held: -0.0294 B | FAILS: +0.0431 W | **FAILS** | FAILS: +0.0483 W | held: -0.0341 B | **FAILS** |
| R-T0 (T0 own-code ADE6, X − Y, point <= +0.0604 m) | held: +0.0021 | held: +0.0179 | **held** | held: -0.0214 | held: -0.0015 | **held** |
| R-NEW (no row of D fires) | held: 0 of 15 rows | held: 0 of 15 rows | **held** | held: 0 of 15 rows | held: 0 of 15 rows | **held** |
| R-MOVE (every crawl row clears its bar) | false: 0 of 4 rows clear | false: 1 of 4 rows clear | **false** | false: 0 of 4 rows clear | false: 0 of 4 rows clear | **false** |
| R-CRAWL (every crawl row of X − P10 not separated-worse) | false: 0 of 4 rows | false: 0 of 4 rows | **false** | false: 0 of 4 rows | false: 0 of 4 rows | **false** |
| R-KEEP (T1-kin ADE6, X − Y, point <= +0.0604 m) | true: -0.0294 | true: +0.0431 | **true** | true: +0.0483 | true: -0.0341 | **true** |
| R-NOBASE | true: 0 of 15 rows violate | true: 0 of 15 rows violate | **true** | true: 0 of 15 rows violate | true: 0 of 15 rows violate | **true** |

**HARMS.** The table's condition, verbatim from the JSON: *"controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed"*. A guard fails at a seed when it fails in either pair (PREREG section 10), so R-ADE fails at s0 and at s1. REFUSED is decided first and HARMS second; the controls pass, so the R-MOVE and R-CRAWL flags do not change the word. Committed next, verbatim from the JSON: *"not adopted; read B under this table (train B first, with its own PI go)."* Cascade: `cascade.final` reads HARMS, and `cascade.b_status` reads *"PENDING: train B (4 runs, share 1/4) through its gate, with its own PI go, and read it under this same table"*.

⚠ **Reported as written, beside the word.** R-MOVE is false at both seeds. R-CRAWL is false at both seeds. The registered guard failure is R-ADE; R-TLAT, R-TLON, R-T0 and R-NEW hold, and R-KEEP and R-NOBASE hold.

The word is the reader's output (`matching_rows_in_table_order`: one row, HARMS). `check_result.py` recomputes R-ADE, R-MOVE, R-CRAWL, R-NEW, R-NOBASE, R-T0 and R-KEEP per pair and per seed from the JSON's cells, floors and points, derives the word from them, and finds every flag equal to the reader's.

## 2. The decisive rows

Tier T1-kin unless a heading says T0. Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. Evidence class MEASURED (ours): every cell is a copy of `read_a.<contrast>.<seed>.<row>` in the JSON. Cells read point [95 % CI] and a flag: B = separated-better, W = separated-worse, – = not separated. P10 is the kinematic floor. The `n` of a cell is windows / episodes.

### 2.1 The floors and the bar (`floors`)

`bar_r = 2 × max(F_r, RF0_r)`, one value per row, applied at both seeds and in both pairs, inclusive. F is 0.1207 m on ADE6 and 0 on every other row. RF0 is the frozen same-seed cross-build floor (`P0_RF0.json`, sha256 `1608d70b…`). RF is the larger of two REP − DAG-OACC draws, reported beside every row. It is not a bar term (PREREG section 6).

| row | better | F | RF0 | **bar** = 2 × max(F, RF0) | RF (beside, not a bar term) | RF / bar |
|---|---|---|---|---|---|---|
| ADE6 | lower | 0.1207 | 0.0138 | **0.2414** | 0.4025 | 1.67 |
| speed MAE | lower | 0.0000 | 0.0796 | **0.1592** | 0.3373 | 2.12 |
| along-track MAE | lower | 0.0000 | 0.0201 | **0.0402** | 0.4596 | 11.43 |
| accel MAE | lower | 0.0000 | 0.1323 | **0.2646** | 0.5603 | 2.12 |
| DK-VIOL | lower | 0.0000 | 0.0050 | **0.0100** | 0.0017 | 0.17 |
| DK-HW | lower | 0.0000 | 0.1308 | **0.2616** | 0.2540 | 0.97 |
| DK-TG | lower | 0.0000 | 0.0041 | **0.0082** | 0.1042 | 12.71 |
| heading MAE | lower | 0.0000 | 0.3801 | **0.7602** | 2.5074 | 3.30 |
| yaw-rate MAE | lower | 0.0000 | 0.5535 | **1.1070** | 15.8553 | 14.32 |
| curvature MAE | lower | 0.0000 | 0.0263 | **0.0526** | 0.1324 | 2.52 |
| cross-track MAE | lower | 0.0000 | 0.0320 | **0.0640** | 0.0224 | 0.35 |
| L1 lateral-accel MAE | lower | 0.0000 | 0.3985 | **0.7970** | 1.2242 | 1.54 |
| L2 curvature MAE (GT>=5.2) | lower | 0.0000 | 0.0068 | **0.0136** | 0.0870 | 6.40 |
| TAC-LAT | higher | 0.0000 | 0.0183 | **0.0366** | 0.0421 | 1.15 |
| TAC-LON | higher | 0.0000 | 0.0125 | **0.0250** | 0.0950 | 3.80 |
| S-POS@30 | lower | 0.0000 | 0.1877 | **0.3754** | 0.2592 | 0.69 |
| S-POS@50 | lower | 0.0000 | 0.4449 | **0.8898** | 0.3912 | 0.44 |
| S-POS@80 | lower | 0.0000 | 1.0082 | **2.0164** | 0.5170 | 0.26 |
| M1 crawl share | lower | 0.0000 | 0.0048 | **0.0096** | 0.0388 | 4.04 |

### 2.2 ADE6: the guard R-ADE and the adoption row R-KEEP (T1-kin)

| seed | pair | row | X − Y | mean X | mean Y | X − P10 | Y − P10 | n (windows / episodes) |
|---|---|---|---|---|---|---|---|---|
| s0 | main (A − CTRL) | ADE6 | **-0.0294 [-0.0522, -0.0077] B** | 2.2128 | 2.2421 | -1.2305 [-1.4302, -1.0448] B | -1.2012 [-1.3945, -1.0186] B | 2399/141 |
| s0 | replicate (A_rep − REP) | ADE6 | **+0.0431 [+0.0137, +0.0711] W** | 2.6877 | 2.6446 | -0.7556 [-1.0164, -0.4950] B | -0.7987 [-1.0586, -0.5428] B | 2399/141 |
| s1 | main (A − CTRL) | ADE6 | **+0.0483 [+0.0105, +0.0884] W** | 2.3684 | 2.3201 | -1.0749 [-1.2743, -0.8735] B | -1.1231 [-1.3301, -0.9159] B | 2399/141 |
| s1 | replicate (A_rep − REP) | ADE6 | **-0.0341 [-0.0655, -0.0027] B** | 2.2976 | 2.3317 | -1.1457 [-1.3570, -0.9417] B | -1.1116 [-1.3180, -0.9119] B | 2399/141 |

R-ADE needs X − Y not separated-worse. It fails in the replicate pair at s0 and in the main pair at s1. R-KEEP reads the point against +0.0604 m and holds in all four pairs. R-ADE and R-KEEP read one contrast, as the PREREG discloses.

### 2.3 R-MOVE: the crawl rows, X − Y against the bar (T1-kin)

R-MOVE needs, in each pair, every crawl row separated-better with an improvement of at least the bar. Improvement is −delta.

| seed | pair | crawl row | X − Y | improvement | bar | improvement / bar | clears (R-MOVE row) |
|---|---|---|---|---|---|---|---|
| s0 | main (A − CTRL) | yaw-rate MAE | -0.7305 [-1.7042, +0.3980] – | +0.7305 | 1.1070 | 66.0 % | false |
| s0 | main (A − CTRL) | curvature MAE | -0.0007 [-0.0109, +0.0113] – | +0.0007 | 0.0526 | 1.3 % | false |
| s0 | main (A − CTRL) | L2 curvature MAE (GT>=5.2) | -0.0075 [-0.0212, +0.0025] – | +0.0075 | 0.0136 | 55.1 % | false |
| s0 | main (A − CTRL) | M1 crawl share | -0.0008 [-0.0037, +0.0022] – | +0.0008 | 0.0096 | 8.3 % | false |
| s0 | replicate (A_rep − REP) | yaw-rate MAE | +0.5729 [-0.6672, +1.7987] – | -0.5729 | 1.1070 | -51.8 % | false |
| s0 | replicate (A_rep − REP) | curvature MAE | -0.0073 [-0.0273, +0.0122] – | +0.0073 | 0.0526 | 13.9 % | false |
| s0 | replicate (A_rep − REP) | L2 curvature MAE (GT>=5.2) | **-0.0215 [-0.0423, -0.0048] B** | +0.0215 | 0.0136 | 158.1 % | true |
| s0 | replicate (A_rep − REP) | M1 crawl share | +0.0042 [-0.0001, +0.0083] – | -0.0042 | 0.0096 | -43.8 % | false |
| s1 | main (A − CTRL) | yaw-rate MAE | +0.2594 [-0.8091, +1.3427] – | -0.2594 | 1.1070 | -23.4 % | false |
| s1 | main (A − CTRL) | curvature MAE | **-0.0387 [-0.0679, -0.0160] B** | +0.0387 | 0.0526 | 73.6 % | false |
| s1 | main (A − CTRL) | L2 curvature MAE (GT>=5.2) | -0.0068 [-0.0204, +0.0063] – | +0.0068 | 0.0136 | 50.0 % | false |
| s1 | main (A − CTRL) | M1 crawl share | +0.0016 [-0.0028, +0.0060] – | -0.0016 | 0.0096 | -16.7 % | false |
| s1 | replicate (A_rep − REP) | yaw-rate MAE | -0.2191 [-1.7774, +1.2541] – | +0.2191 | 1.1070 | 19.8 % | false |
| s1 | replicate (A_rep − REP) | curvature MAE | +0.0101 [-0.0051, +0.0291] – | -0.0101 | 0.0526 | -19.2 % | false |
| s1 | replicate (A_rep − REP) | L2 curvature MAE (GT>=5.2) | +0.0178 [-0.0059, +0.0474] – | -0.0178 | 0.0136 | -130.9 % | false |
| s1 | replicate (A_rep − REP) | M1 crawl share | +0.0021 [-0.0032, +0.0070] – | -0.0021 | 0.0096 | -21.9 % | false |

2 of the 16 cells are separated-better and 1 clears its bar. No pair clears all four rows, and no main-pair cell clears.

### 2.4 R-CRAWL: the crawl rows, X − P10 (T1-kin)

The share closed is (Y − P10 − (X − P10)) / (Y − P10) from the point deltas; a negative share means the gap to P10 widens. MEASURED (arithmetic on the banked JSON).

| seed | pair | crawl row | X − P10 | Y − P10 | X − P10 not separated-worse (R-CRAWL row) | share of the Y − P10 gap closed |
|---|---|---|---|---|---|---|
| s0 | main (A − CTRL) | yaw-rate MAE | **+2.7420 [+0.3388, +5.2598] W** | **+3.4726 [+0.6601, +6.3308] W** | false | 21.0 % |
| s0 | main (A − CTRL) | curvature MAE | **+0.0847 [+0.0549, +0.1200] W** | **+0.0854 [+0.0580, +0.1175] W** | false | 0.8 % |
| s0 | main (A − CTRL) | L2 curvature MAE (GT>=5.2) | **+0.0375 [+0.0201, +0.0588] W** | **+0.0443 [+0.0232, +0.0744] W** | false | 15.3 % |
| s0 | main (A − CTRL) | M1 crawl share | **+0.0564 [+0.0374, +0.0788] W** | **+0.0568 [+0.0378, +0.0798] W** | false | 0.7 % |
| s0 | replicate (A_rep − REP) | yaw-rate MAE | **+19.9019 [+14.5378, +25.7101] W** | **+19.3282 [+13.9956, +25.0836] W** | false | -3.0 % |
| s0 | replicate (A_rep − REP) | curvature MAE | **+0.2107 [+0.1535, +0.2708] W** | **+0.2180 [+0.1586, +0.2848] W** | false | 3.3 % |
| s0 | replicate (A_rep − REP) | L2 curvature MAE (GT>=5.2) | **+0.1071 [+0.0671, +0.1558] W** | **+0.1313 [+0.0797, +0.1961] W** | false | 18.4 % |
| s0 | replicate (A_rep − REP) | M1 crawl share | **+0.0990 [+0.0727, +0.1284] W** | **+0.0957 [+0.0690, +0.1258] W** | false | -3.4 % |
| s1 | main (A − CTRL) | yaw-rate MAE | **+8.9439 [+5.3604, +12.9899] W** | **+8.6844 [+5.1690, +12.5740] W** | false | -3.0 % |
| s1 | main (A − CTRL) | curvature MAE | **+0.0964 [+0.0650, +0.1321] W** | **+0.1352 [+0.0889, +0.1915] W** | false | 28.7 % |
| s1 | main (A − CTRL) | L2 curvature MAE (GT>=5.2) | **+0.0462 [+0.0260, +0.0716] W** | **+0.0529 [+0.0281, +0.0847] W** | false | 12.7 % |
| s1 | main (A − CTRL) | M1 crawl share | **+0.0620 [+0.0415, +0.0870] W** | **+0.0604 [+0.0391, +0.0864] W** | false | -2.6 % |
| s1 | replicate (A_rep − REP) | yaw-rate MAE | **+9.3556 [+5.7951, +13.2237] W** | **+9.5748 [+5.8282, +13.7093] W** | false | 2.3 % |
| s1 | replicate (A_rep − REP) | curvature MAE | **+0.1445 [+0.0951, +0.2028] W** | **+0.1343 [+0.0858, +0.1912] W** | false | -7.6 % |
| s1 | replicate (A_rep − REP) | L2 curvature MAE (GT>=5.2) | **+0.0767 [+0.0413, +0.1232] W** | **+0.0589 [+0.0334, +0.0896] W** | false | -30.2 % |
| s1 | replicate (A_rep − REP) | M1 crawl share | **+0.0672 [+0.0457, +0.0923] W** | **+0.0652 [+0.0440, +0.0900] W** | false | -3.1 % |

All 16 X − P10 crawl cells are separated-worse. So are the Y − P10 cells in every pair: that is how the crawl set C was fixed.

### 2.5 R-NEW and R-NOBASE (T1-kin; domain D is every row not separated-worse for Y − P10, no exempt rows)

R-NEW fires on a row of D when three things hold: (1) X − P10 is separated-worse, (2) Y − P10 is not (the row is in D), and (3) X − Y is separated-worse with a worsening of at least the bar. R-NOBASE reads (1) and (3).

| seed | pair | control Y | rows in D | X − Y separated-worse | of those, worsening >= bar | of those, X − P10 separated-worse (R-NEW fires) |
|---|---|---|---|---|---|---|
| s0 | main (A − CTRL) | CTRL − P10 not separated-worse on D | 15 | none | none | none |
| s0 | replicate (A_rep − REP) | REP − P10 not separated-worse on D | 15 | ADE6, along-track MAE | along-track MAE | none |
| s1 | main (A − CTRL) | CTRL − P10 not separated-worse on D | 15 | ADE6, speed MAE, accel MAE, cross-track MAE, L1 lateral-accel MAE | speed MAE, accel MAE, cross-track MAE, L1 lateral-accel MAE | none |
| s1 | replicate (A_rep − REP) | REP − P10 not separated-worse on D | 15 | none | none | none |

**R-NEW and R-NOBASE hold because condition (1) fails, not because condition (3) does.** In the main pair at s1, four rows meet (3): speed MAE +0.1797 [+0.0998, +0.2726] W, accel MAE +0.7391 [+0.4366, +1.0952] W, cross-track MAE +0.0720 [+0.0403, +0.1109] W and L1 lateral-accel MAE +1.4444 [+0.6003, +2.4872] W. None of the four has X − P10 separated-worse. In the replicate pair at s0, along-track MAE meets (3): +0.0493 [+0.0170, +0.0801] W. The registered predicate is applied as written.

### 2.6 R-T0, R-KEEP, R-TLAT and R-TLON

R-T0 is at tier T0 (a point, no interval in the JSON). R-KEEP is T1-kin.

| seed | pair | T0 own-code ADE6, X − Y (point; no interval in the JSON) | R-T0 (<= +0.0604 m) | T1-kin ADE6, X − Y (point) | R-KEEP (<= +0.0604 m) |
|---|---|---|---|---|---|
| s0 | main (A − CTRL) | +0.0021 | held | -0.0294 | true |
| s0 | replicate (A_rep − REP) | +0.0179 | held | +0.0431 | true |
| s1 | main (A − CTRL) | -0.0214 | held | +0.0483 | true |
| s1 | replicate (A_rep − REP) | -0.0015 | held | -0.0341 | true |

R-TLAT and R-TLON are booleans in the JSON and read *held* in all four pairs. The macro-F1 values behind them are not carried (UNVERIFIED as numbers).

## 3. The four families (T1-kin; per family, per seed, per pair, never pooled)

Tier: T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan; plain arms). Never 'closed loop'. Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, per pair. Evidence class MEASURED (ours); source `four_families.A.<seed>.<pair>.<family>.rows`. Each family block carries the tier and estimator strings, and its leaves equal the `read_a` cells cell for cell (204 cells, asserted by `check_result.py`). Each table shows X − Y, the one-lever contrast of the pair. The X − P10 and Y − P10 cells are in section 3.8. STRATEGIC is **not paused** for v8F (PI ruling 2026-10-17).

### 3.1 Separated cells per family, per seed, per pair (B / W / not separated; X − Y)

| family (rows) | s0 main: B / W / not separated | s0 replicate: B / W / not separated | s1 main: B / W / not separated | s1 replicate: B / W / not separated |
|---|---|---|---|---|
| LONGITUDINAL (6) | 2 / 0 / 4 | 0 / 1 / 5 | 0 / 2 / 4 | 2 / 0 / 4 |
| LATERAL (6) | 2 / 0 / 4 | 1 / 0 / 5 | 1 / 2 / 3 | 1 / 0 / 5 |
| TACTICAL (2) | 0 / 0 / 2 | 0 / 0 / 2 | 0 / 0 / 2 | 1 / 0 / 1 |
| STRATEGIC (3) | 3 / 0 / 0 | 0 / 0 / 3 | 0 / 0 / 3 | 1 / 0 / 2 |

### 3.2 LONGITUDINAL (read first: the lever acts on along-track pressure)

| row | better | A − CTRL s0 | A_rep − REP s0 | A − CTRL s1 | A_rep − REP s1 | n, s0 main / s0 replicate / s1 main / s1 replicate |
|---|---|---|---|---|---|---|
| speed MAE | lower | **-0.0201 [-0.0393, -0.0031] B** | +0.0074 [-0.0176, +0.0310] – | **+0.1797 [+0.0998, +0.2726] W** | **-0.0744 [-0.1220, -0.0337] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| along-track MAE | lower | -0.0185 [-0.0388, +0.0021] – | **+0.0493 [+0.0170, +0.0801] W** | +0.0057 [-0.0249, +0.0346] – | -0.0250 [-0.0521, +0.0059] – | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| accel MAE | lower | **-0.0568 [-0.1137, -0.0041] B** | +0.0425 [-0.0186, +0.1079] – | **+0.7391 [+0.4366, +1.0952] W** | **-0.2560 [-0.4105, -0.1222] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| DK-VIOL | lower | +0.0050 [-0.0092, +0.0212] – | -0.0050 [-0.0160, +0.0037] – | -0.0034 [-0.0161, +0.0089] – | +0.0034 [-0.0039, +0.0117] – | 595/70 ; 595/70 ; 595/70 ; 595/70 |
| DK-HW | lower | +0.0387 [-0.0882, +0.1748] – | +0.0092 [-0.0740, +0.0952] – | +0.0590 [-0.0277, +0.1582] – | -0.0382 [-0.1706, +0.0776] – | 712/72 ; 710/72 ; 710/73 ; 712/72 |
| DK-TG | lower | +0.0070 [-0.0359, +0.0563] – | -0.0245 [-0.0605, +0.0016] – | -0.0094 [-0.0479, +0.0204] – | -0.0062 [-0.0363, +0.0189] – | 654/72 ; 652/72 ; 653/72 ; 654/72 |

Not built for this family: M-LAUNCH, the anti-echo block and the M2 / L2 launch-subset rows (section 3.7).

### 3.3 LATERAL

| row | better | A − CTRL s0 | A_rep − REP s0 | A − CTRL s1 | A_rep − REP s1 | n, s0 main / s0 replicate / s1 main / s1 replicate |
|---|---|---|---|---|---|---|
| heading MAE | lower | **-0.4211 [-0.6883, -0.1899] B** | +0.0892 [-0.2326, +0.3963] – | +0.1038 [-0.2848, +0.5761] – | -0.1749 [-0.5225, +0.1333] – | 2373/141 ; 2373/141 ; 2373/141 ; 2373/141 |
| yaw-rate MAE | lower | -0.7305 [-1.7042, +0.3980] – | +0.5729 [-0.6672, +1.7987] – | +0.2594 [-0.8091, +1.3427] – | -0.2191 [-1.7774, +1.2541] – | 2366/141 ; 2367/141 ; 2366/141 ; 2367/141 |
| curvature MAE | lower | -0.0007 [-0.0109, +0.0113] – | -0.0073 [-0.0273, +0.0122] – | **-0.0387 [-0.0679, -0.0160] B** | +0.0101 [-0.0051, +0.0291] – | 2366/141 ; 2367/141 ; 2366/141 ; 2367/141 |
| cross-track MAE | lower | **-0.0191 [-0.0365, -0.0030] B** | +0.0053 [-0.0144, +0.0234] – | **+0.0720 [+0.0403, +0.1109] W** | -0.0270 [-0.0572, +0.0001] – | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| L1 lateral-accel MAE | lower | -0.0438 [-0.1565, +0.0921] – | +0.0906 [-0.0774, +0.2676] – | **+1.4444 [+0.6003, +2.4872] W** | **-0.3908 [-0.7329, -0.0748] B** | 2366/141 ; 2367/141 ; 2366/141 ; 2367/141 |
| L2 curvature MAE (GT>=5.2) | lower | -0.0075 [-0.0212, +0.0025] – | **-0.0215 [-0.0423, -0.0048] B** | -0.0068 [-0.0204, +0.0063] – | +0.0178 [-0.0059, +0.0474] – | 2105/135 ; 2104/135 ; 2106/135 ; 2106/135 |

### 3.4 TACTICAL

| row | better | A − CTRL s0 | A_rep − REP s0 | A − CTRL s1 | A_rep − REP s1 | n, s0 main / s0 replicate / s1 main / s1 replicate |
|---|---|---|---|---|---|---|
| TAC-LAT | higher | +0.0063 [-0.0042, +0.0158] – | -0.0075 [-0.0221, +0.0071] – | +0.0000 [-0.0142, +0.0150] – | **+0.0217 [+0.0079, +0.0371] B** | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |
| TAC-LON | higher | +0.0150 [-0.0004, +0.0333] – | -0.0021 [-0.0158, +0.0117] – | +0.0013 [-0.0121, +0.0146] – | -0.0025 [-0.0204, +0.0171] – | 2399/141 ; 2399/141 ; 2399/141 ; 2399/141 |

TAC-LAT and TAC-LON are the plan-derived agreements (higher is better). The guards R-TLAT and R-TLON are in section 2.6. T1-kin goal-head F1 and head accuracy are **not computable** here: the paired tool reads eval dumps, not tick dumps (JSON note).

### 3.5 STRATEGIC

| row | better | A − CTRL s0 | A_rep − REP s0 | A − CTRL s1 | A_rep − REP s1 | n, s0 main / s0 replicate / s1 main / s1 replicate |
|---|---|---|---|---|---|---|
| S-POS@30 | lower | **-0.1265 [-0.2605, -0.0056] B** | -0.0083 [-0.1748, +0.1592] – | +0.1814 [-0.1398, +0.5613] – | **-0.2141 [-0.4437, -0.0057] B** | 2321/141 ; 2321/141 ; 2321/141 ; 2321/141 |
| S-POS@50 | lower | **-0.2295 [-0.4581, -0.0215] B** | -0.1178 [-0.4129, +0.1810] – | -0.1536 [-0.5723, +0.2938] – | -0.0448 [-0.3621, +0.2538] – | 2313/141 ; 2313/141 ; 2313/141 ; 2313/141 |
| S-POS@80 | lower | **-0.4006 [-0.7914, -0.0184] B** | -0.1193 [-0.6477, +0.4084] – | -0.3352 [-1.1408, +0.5368] – | -0.0523 [-0.5763, +0.4724] – | 2290/141 ; 2290/141 ; 2290/141 ; 2290/141 |

S-POS@30 / @50 / @80 (lower is better). The X − Y cell is the difference of the two runs' (S − CTRV, ticks ≥ 1) contrasts on the same windows. Goal and route-setting quality is not a Read-A row; whether another tool computes it for these arms is UNVERIFIED here.

### 3.6 T0 (box 2's standard eval): the four families are NOT RUN

| family | what the JSON names | status | reason |
|---|---|---|---|
| LONGITUDINAL | the T0 LONGITUDINAL rows and distance keeping at T0 | NOT RUN | a build item the reader did not carry (`not_built.t0_four_families`) |
| LATERAL | the T0 LATERAL rows | NOT RUN | same |
| TACTICAL | the T0 TACTICAL rows | NOT RUN | same |
| STRATEGIC | not named in the JSON; PREREG section 8 puts the box-2 families at T0 (`orc_families.py`, `v8f_paired_families.py`) | NOT RUN | same; the family is not paused for v8F |

`not_built.t0_four_families` reads: *"orc_families.py pointed at the arms and their controls (the T0 LONGITUDINAL / LATERAL / TACTICAL rows and distance keeping at T0): NOT RUN, a build item, never a silent omission"*. The only T0 numbers in this read are the R-T0 points (section 2.6) and the slow-draw cells (section 4.2). The inputs exist: each run's box-2 T0 eval JSON is in the run record (section 10), and its `.npz` is in the pulled run directory on E: (listed in the run's `SHA256SUMS.txt`). **Nothing is claimed for T0 LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC.**

### 3.7 Registered rows that are NOT BUILT or not computable

Verbatim from the JSON `not_built`:

* `M-LAUNCH`: *"X - Y of the mean (v_own - v_GT) at ticks 1-2 in the GT-launch windows, from the tick dumps, in each pair: NOT BUILT (routes NEXT-LAUNCH by hand)"*
* `anti_echo_block`: *"longitudinal_claim_admissible, hold-v0, copy detector (RESULT_V8F_ORECOVER 3.1): NOT BUILT"*
* `M2_and_L2_launch_subset`: *"along-track MAE, speed MAE and L2 on the GT-launch subset; M2: NOT BUILT"*

By family: LONGITUDINAL carries all three (the mechanism row, the anti-echo block, the launch-subset rows). TACTICAL: goal-head F1 and head accuracy are not computable at T1-kin, and the R-TLAT / R-TLON macro-F1 values are not in the JSON. LATERAL and STRATEGIC: every registered Read-A row is computed.

### 3.8 Against P10: X − P10 and Y − P10, every row (T1-kin)

For the three S-POS rows the P10 contrast is the run's own S − CTRV and the JSON labels its `n` *"ticks>=1, unpaired vs CTRV (DAG_READ.json)"*: those cells are not paired across runs.

**s0**

| row | better | A − P10 s0 | CTRL − P10 s0 | A_rep − P10 s0 | REP − P10 s0 |
|---|---|---|---|---|---|
| ADE6 | lower | **-1.2305 [-1.4302, -1.0448] B** | **-1.2012 [-1.3945, -1.0186] B** | **-0.7556 [-1.0164, -0.4950] B** | **-0.7987 [-1.0586, -0.5428] B** |
| speed MAE | lower | **-0.7962 [-1.0078, -0.6137] B** | **-0.7761 [-0.9806, -0.5996] B** | **-0.4314 [-0.6628, -0.2286] B** | **-0.4388 [-0.6617, -0.2409] B** |
| along-track MAE | lower | **-0.6994 [-0.8611, -0.5457] B** | **-0.6809 [-0.8443, -0.5214] B** | -0.1720 [-0.4096, +0.0640] – | -0.2213 [-0.4578, +0.0079] – |
| accel MAE | lower | **-0.8740 [-1.2470, -0.5533] B** | **-0.8172 [-1.1770, -0.4998] B** | -0.2144 [-0.5718, +0.1072] – | -0.2569 [-0.5998, +0.0569] – |
| DK-VIOL | lower | +0.0067 [-0.0164, +0.0314] – | +0.0017 [-0.0231, +0.0272] – | -0.0050 [-0.0338, +0.0219] – | +0.0000 [-0.0248, +0.0246] – |
| DK-HW | lower | -0.2519 [-0.6506, +0.0829] – | **-0.3318 [-0.6942, -0.0242] B** | +0.0017 [-0.4459, +0.4482] – | -0.0980 [-0.5251, +0.3139] – |
| DK-TG | lower | -0.0537 [-0.1320, +0.0207] – | -0.0568 [-0.1232, +0.0163] – | +0.0033 [-0.1074, +0.1580] – | +0.0279 [-0.0978, +0.2093] – |
| heading MAE | lower | **-3.0131 [-3.7375, -2.3511] B** | **-2.5920 [-3.2391, -1.9751] B** | +0.0045 [-0.9179, +0.9648] – | -0.0846 [-0.9878, +0.8799] – |
| yaw-rate MAE | lower | **+2.7420 [+0.3388, +5.2598] W** | **+3.4726 [+0.6601, +6.3308] W** | **+19.9019 [+14.5378, +25.7101] W** | **+19.3282 [+13.9956, +25.0836] W** |
| curvature MAE | lower | **+0.0847 [+0.0549, +0.1200] W** | **+0.0854 [+0.0580, +0.1175] W** | **+0.2107 [+0.1535, +0.2708] W** | **+0.2180 [+0.1586, +0.2848] W** |
| cross-track MAE | lower | **-0.9811 [-1.1264, -0.8486] B** | **-0.9620 [-1.1077, -0.8319] B** | **-0.9733 [-1.1219, -0.8367] B** | **-0.9786 [-1.1252, -0.8423] B** |
| L1 lateral-accel MAE | lower | **-2.5056 [-3.8661, -1.3292] B** | **-2.4618 [-3.8097, -1.2924] B** | **-1.1458 [-2.4394, -0.0147] B** | **-1.2365 [-2.4350, -0.1556] B** |
| L2 curvature MAE (GT>=5.2) | lower | **+0.0375 [+0.0201, +0.0588] W** | **+0.0443 [+0.0232, +0.0744] W** | **+0.1071 [+0.0671, +0.1558] W** | **+0.1313 [+0.0797, +0.1961] W** |
| TAC-LAT | higher | **+0.1967 [+0.1580, +0.2378] B** | **+0.1905 [+0.1508, +0.2319] B** | **+0.1409 [+0.0985, +0.1861] B** | **+0.1484 [+0.1072, +0.1921] B** |
| TAC-LON | higher | **+0.1546 [+0.0912, +0.2194] B** | **+0.1396 [+0.0771, +0.1997] B** | +0.0425 [-0.0292, +0.1096] – | +0.0446 [-0.0292, +0.1130] – |
| S-POS@30 | lower | **-1.8693 [-2.4537, -1.3756] B** | **-1.7428 [-2.2737, -1.2861] B** | **-1.4919 [-1.9634, -1.0845] B** | **-1.4837 [-1.9587, -1.0595] B** |
| S-POS@50 | lower | **-4.7626 [-5.9590, -3.7194] B** | **-4.5331 [-5.6695, -3.5422] B** | **-4.2598 [-5.2985, -3.3374] B** | **-4.1419 [-5.1720, -3.1830] B** |
| S-POS@80 | lower | **-9.9630 [-12.0882, -7.9003] B** | **-9.5624 [-11.6843, -7.6124] B** | **-9.4896 [-11.4905, -7.6284] B** | **-9.3703 [-11.3441, -7.5010] B** |
| M1 crawl share | lower | **+0.0564 [+0.0374, +0.0788] W** | **+0.0568 [+0.0378, +0.0798] W** | **+0.0990 [+0.0727, +0.1284] W** | **+0.0957 [+0.0690, +0.1258] W** |

**s1**

| row | better | A − P10 s1 | CTRL − P10 s1 | A_rep − P10 s1 | REP − P10 s1 |
|---|---|---|---|---|---|
| ADE6 | lower | **-1.0749 [-1.2743, -0.8735] B** | **-1.1231 [-1.3301, -0.9159] B** | **-1.1457 [-1.3570, -0.9417] B** | **-1.1116 [-1.3180, -0.9119] B** |
| speed MAE | lower | **-0.5233 [-0.6966, -0.3707] B** | **-0.7030 [-0.9087, -0.5230] B** | **-0.7336 [-0.9525, -0.5426] B** | **-0.6592 [-0.8610, -0.4801] B** |
| along-track MAE | lower | **-0.5819 [-0.7575, -0.4109] B** | **-0.5876 [-0.7702, -0.4119] B** | **-0.6009 [-0.7823, -0.4214] B** | **-0.5758 [-0.7559, -0.4022] B** |
| accel MAE | lower | +0.1377 [-0.2067, +0.5134] – | **-0.6014 [-0.9063, -0.3345] B** | **-0.6569 [-1.0141, -0.3371] B** | **-0.4009 [-0.7081, -0.1184] B** |
| DK-VIOL | lower | -0.0067 [-0.0302, +0.0162] – | -0.0034 [-0.0243, +0.0181] – | +0.0000 [-0.0249, +0.0254] – | -0.0034 [-0.0270, +0.0207] – |
| DK-HW | lower | **-0.4007 [-0.7827, -0.0973] B** | **-0.4466 [-0.8200, -0.1686] B** | **-0.5136 [-0.8890, -0.1944] B** | **-0.4713 [-0.8540, -0.1546] B** |
| DK-TG | lower | **-0.0817 [-0.1396, -0.0318] B** | **-0.0724 [-0.1384, -0.0085] B** | **-0.1051 [-0.1677, -0.0483] B** | **-0.0951 [-0.1535, -0.0468] B** |
| heading MAE | lower | **-1.9501 [-2.6594, -1.2479] B** | **-2.0539 [-2.7092, -1.3705] B** | **-1.9614 [-2.6722, -1.2710] B** | **-1.7865 [-2.5844, -1.0282] B** |
| yaw-rate MAE | lower | **+8.9439 [+5.3604, +12.9899] W** | **+8.6844 [+5.1690, +12.5740] W** | **+9.3556 [+5.7951, +13.2237] W** | **+9.5748 [+5.8282, +13.7093] W** |
| curvature MAE | lower | **+0.0964 [+0.0650, +0.1321] W** | **+0.1352 [+0.0889, +0.1915] W** | **+0.1445 [+0.0951, +0.2028] W** | **+0.1343 [+0.0858, +0.1912] W** |
| cross-track MAE | lower | **-0.9021 [-1.0493, -0.7638] B** | **-0.9742 [-1.1152, -0.8405] B** | **-0.9788 [-1.1217, -0.8478] B** | **-0.9518 [-1.0961, -0.8185] B** |
| L1 lateral-accel MAE | lower | -0.4029 [-1.3012, +0.4388] – | **-1.8473 [-2.9617, -0.8501] B** | **-1.9161 [-3.1968, -0.8210] B** | **-1.5251 [-2.5967, -0.5934] B** |
| L2 curvature MAE (GT>=5.2) | lower | **+0.0462 [+0.0260, +0.0716] W** | **+0.0529 [+0.0281, +0.0847] W** | **+0.0767 [+0.0413, +0.1232] W** | **+0.0589 [+0.0334, +0.0896] W** |
| TAC-LAT | higher | **+0.1972 [+0.1593, +0.2363] B** | **+0.1972 [+0.1600, +0.2364] B** | **+0.1926 [+0.1517, +0.2331] B** | **+0.1709 [+0.1297, +0.2141] B** |
| TAC-LON | higher | **+0.1384 [+0.0746, +0.2006] B** | **+0.1371 [+0.0751, +0.2001] B** | **+0.1542 [+0.0962, +0.2114] B** | **+0.1567 [+0.0967, +0.2155] B** |
| S-POS@30 | lower | **-1.6440 [-2.0440, -1.2864] B** | **-1.8255 [-2.3433, -1.3880] B** | **-1.9276 [-2.4624, -1.4663] B** | **-1.7135 [-2.1974, -1.3049] B** |
| S-POS@50 | lower | **-4.7627 [-5.7624, -3.8251] B** | **-4.6092 [-5.6305, -3.6576] B** | **-4.8929 [-5.9895, -3.8792] B** | **-4.8481 [-5.9214, -3.8649] B** |
| S-POS@80 | lower | **-10.2601 [-12.4419, -8.1915] B** | **-9.9249 [-11.9750, -7.9686] B** | **-10.4943 [-12.6644, -8.4336] B** | **-10.4419 [-12.5718, -8.3877] B** |
| M1 crawl share | lower | **+0.0620 [+0.0415, +0.0870] W** | **+0.0604 [+0.0391, +0.0864] W** | **+0.0672 [+0.0457, +0.0923] W** | **+0.0652 [+0.0440, +0.0900] W** |

## 4. Beside, deciding nothing

### 4.1 RF beside every row

The RF column of section 2.1 is this block (`beside.A.RF_beside_every_row`): RF, RF0 and the bar for each of the 19 rows. RF is not a bar term.

### 4.2 The slow-draw flag (T0; the five speed cells, per-window paired difference, episode-cluster bootstrap, B = 2,000, seed 0)

The rule (PREREG section 7): an arm is a **slow-T draw** when all five cells are separated from the pair's control and each shift exceeds 0.5 m/s in magnitude. The flag decides nothing.

| arm and pair (T0, m/s) | lon_v_target_ms | v_a_ms | v_end_ms | v_lo_ms | v_hi_ms | label |
|---|---|---|---|---|---|---|
| s0 main (A − CTRL) | +0.0718 [+0.0195, +0.1256] | +0.0505 [+0.0012, +0.0992] | +0.0682 [+0.0181, +0.1170] | +0.0090 [-0.0465, +0.0624] | +0.0421 [-0.0157, +0.0979] | none |
| s0 replicate (A_rep − REP) | -0.0792 [-0.1340, -0.0274] | -0.0873 [-0.1516, -0.0212] | +0.0523 [+0.0069, +0.0963] | -0.0517 [-0.1105, +0.0066] | -0.0917 [-0.1449, -0.0388] | none |
| s1 main (A − CTRL) | +0.0901 [+0.0312, +0.1509] | +0.0842 [+0.0339, +0.1378] | +0.0109 [-0.0450, +0.0670] | +0.0731 [+0.0157, +0.1303] | +0.0890 [+0.0341, +0.1422] | none |
| s1 replicate (A_rep − REP) | +0.1627 [+0.1074, +0.2155] | +0.0344 [-0.0179, +0.0867] | +0.0487 [+0.0009, +0.0970] | +0.0295 [-0.0280, +0.0882] | +0.0485 [-0.0046, +0.1005] | none |

No arm is labelled. The largest shift in magnitude over the 20 cells is 0.1627 m/s, and 13 of the 20 cell intervals exclude 0.

### 4.3 The lever reached the data: K-SHARE and K-MASK (MEASURED, box-2 chain log)

The realised weight share is recomputed by the run record from `launch_n`, `launch_eligible` and the flag. The target is 1/3 ± 0.02. Rows equal eligible rows in all four runs.

| run | trainer seed | launch rows / rows | launch-row fraction (K-MASK: 0.1619 +- 0.02) | weight share (K-SHARE: 1/3 +- 0.02) | argv sha16 | checkpoint sha256 (12 hex) |
|---|---|---|---|---|---|---|
| A_0 (M2b_s0_launchintent) | 0 | 30727 / 191203 | 0.160704 | 0.331451 | `012d6f3467d0f8a5` | `deb8f28b4ea9` |
| A_1 (M2b_s1_launchintent) | 1 | 30674 / 191396 | 0.160265 | 0.330729 | `2d5ceaf32bd7f6ff` | `abe0fd24e6f9` |
| A_rep,0 (M2b_s0_launchintentrep) | 10 | 30068 / 191333 | 0.15715 | 0.325586 | `abd825ae017a954a` | `b86f7d4353b9` |
| A_rep,1 (M2b_s1_launchintentrep) | 11 | 30367 / 191391 | 0.158665 | 0.328093 | `06117e7b7fab5385` | `2900ff73cd85` |

The four shares equal the PREREG's ESTIMATED shares for trainer seeds 0 / 1 / 10 / 11 (0.3315, 0.3307, 0.3256, 0.3281, simulated from the sampler stream) at four decimals.

### 4.4 The launch gates (MEASURED, box-2 chain log)

Each run went through its own launch gate before training: G-DVB, the swap probe, the acc probe, the launch probe, and the one-lever check against its pair control's banked argv. `head_o grad diff` is the gradient the launch term puts on O's head; it is non-zero in all four.

| run | W | trainer seed | probe batch launch rows | launch probe contribution | head_o grad diff | swap probe grad diff |
|---|---|---|---|---|---|---|
| A_0 | 1.58926 | 0 | 11 / 64 | 0.03420829772949219 | 11.032110214233398 | 14267.4716796875 |
| A_1 | 1.58926 | 1 | 8 / 64 | 0.03153038024902344 | 10.921006202697754 | 10972.4638671875 |
| A_rep,0 | 1.58926 | 10 | 11 / 64 | 0.03741455078125 | 10.58016586303711 | 10940.1328125 |
| A_rep,1 | 1.58926 | 11 | 10 / 64 | 0.02295684814453125 | 9.637072563171387 | 46505.515625 |

## 5. Reading (decides nothing)

**Label: MEASURED facts from the tables above, and one mechanism statement. The outcome is HARMS, as registered. Nothing in this section changes a flag, a bar or the word.**

**Fact 1. At each seed, the separated ADE6 cells flip sign between the main pair and the replicate pair.**

| seed | trainer seed (main pair / replicate pair) | main pair, A − CTRL | replicate pair, A_rep − REP |
|---|---|---|---|
| s0 | 0 / 10 | **-0.0294 [-0.0522, -0.0077] B** | **+0.0431 [+0.0137, +0.0711] W** |
| s1 | 1 / 11 | **+0.0483 [+0.0105, +0.0884] W** | **-0.0341 [-0.0655, -0.0027] B** |

All four cells are separated. The lever is the same in both pairs (W 1.58926). The two pairs of a seed differ in the trainer seed and the control. The reader's K-PAIR control passes for both. Beyond ADE6, the rows whose separated X − Y cells have opposite signs between the two pairs of the same seed are: at s0, ADE6; at s1, ADE6, speed MAE, accel MAE, L1 lateral-accel MAE. At s1 that includes accel MAE: main pair +0.7391 [+0.4366, +1.0952] W, replicate pair -0.2560 [-0.4105, -0.1222] B.

**Fact 2. No crawl row moved by its bar in both pairs at any seed.** The crawl cells that are separated-better:

| seed | pair | crawl row | X − Y | improvement / bar | clears (R-MOVE row) |
|---|---|---|---|---|---|
| s0 | replicate (A_rep − REP) | L2 curvature MAE (GT>=5.2) | **-0.0215 [-0.0423, -0.0048] B** | 158.1 % | true |
| s1 | main (A − CTRL) | curvature MAE | **-0.0387 [-0.0679, -0.0160] B** | 73.6 % | false |

The only cell that clears its bar is in the replicate pair at s0. Its main-pair partner is -0.0075 [-0.0212, +0.0025] –, not separated. Every main-pair crawl cell is below its bar.

**Why this is the pattern the pair design exists to expose.** The episode-cluster bootstrap resamples EPISODES with the models held fixed (`H-ESTIM-SEED-1`). It answers *"would another draw of EPISODES say this?"*. It cannot answer *"would another training run say this?"*. The replicate pair asks that second question: the same lever on the same recipe, another trainer seed. Here the answer on ADE6 is the opposite sign at each seed. A read on either pair alone would have called the lever better on ADE6 at one seed and worse at the other, depending on which pair was run. Each separated cell is right about its episodes and silent about its training run. That is why a separated cell from one pair is necessary and not sufficient, and why the registered predicates hold only when they hold in both pairs.

The sizes are small. The four ADE6 points lie between 0.0294 and 0.0483 m, all inside ±0.0604 m, which is why R-KEEP holds in all four. The registered table does not use this reading: R-ADE counts as failed at a seed when it fails in either pair, so the word stays HARMS.

## 6. Next (Rule Zero)

### 6.1 Committed next: B, with its own PI go

PREREG section 10, verbatim from the JSON: *"not adopted; read B under this table (train B first, with its own PI go)."* B is the same lever at a lower dose: W 0.726175, a launch-sample weight share of one quarter (K-SHARE band 1/4 ± 0.02; ESTIMATED realised shares 0.2484 / 0.2478 / 0.2435 / 0.2456 at trainer seeds 0 / 1 / 10 / 11, PREREG section 3). It is four box-2 runs (B_0, B_1, B_rep,0, B_rep,1) in the same pair design, read under the same table with its own seeds, and never pooled with A. B's HARMS routes to NEXT-LAUNCH (section 6.2).

**Cost (ESTIMATED from a MEASURED span).** The A chain ran 13:54 to 16:48 on box 2 (174 min, 2.9 h; MEASURED, `chain_v8f_launchintent.out`), against the PREREG's ESTIMATED 2.8 h (section 13). B is about the same.

**Blocked on:** the PI's go for box-2 time. The A chain's closing line says *"B is not queued"*. What unblocks it: the go, and a B chain on the pattern of the A chain (0 GPU to build; UNVERIFIED that nothing else is needed).

### 6.2 NEXT-LAUNCH (PREREG section 11) and the M-LAUNCH slot

LAUNCH-INTENT is the last lever `PREREG_V8F_DAG_ORC` section 8 ranks. If the crawl does not move, the routing variable is M-LAUNCH: X − Y of the mean (v_own − v_GT) at ticks 1–2 in the GT-launch windows, from the tick dumps, in each pair, at T1-kin. It is read on A, or on B when B is the arm that sends the programme here. *Moved* means separated toward 0 in both pairs at both seeds.

| M-LAUNCH | what it says | committed next |
|---|---|---|
| moved | the plan's launch speed was repaired and the crawl rows did not follow: the crawl is not in O's plan profile | **LOOP-STATE** (zero training, dev-box CPU): decompose the crawl by plan index in X's tick dumps; its read names the loop term |
| not moved | the lever did not reach O's launch behaviour at this dose | **LAUNCH-INTENT-HI**: the same lever at a launch-sample share of one half (W_HI 4.17852), under the section 10 table, with its own PI go |

This reader does not build M-LAUNCH (`not_built.M-LAUNCH`: NOT BUILT). It is built separately, and its result goes in the slot below.

**M-LAUNCH: NOT MOVED** (MEASURED 2026-10-28, `raw/2026-10-28-launch-intent-mlaunch/MLAUNCH.json`, sha256 `f5ebb4b2…`; T1-kin, causal lane; paired episode-cluster bootstrap, B = 2,000, seed 0, over the 49 episodes of the 327 GT-launch windows, the `hmc_read.py` definition). Mean (v_own − v_GT) at ticks 1–2, m/s; a negative value is slower than the human.

| pair | seed | X − Y [95 % CI] | separated |
|---|---|---|---|
| main (A − DAG-OACC) | s0 | +0.0348 [−0.0299, +0.0923] | no |
| main (A − DAG-OACC) | s1 | +0.0282 [−0.0247, +0.0836] | no |
| replicate (A_rep − REP) | s0 | +0.0243 [−0.0281, +0.0846] | no |
| replicate (A_rep − REP) | s1 | −0.0146 [−0.0607, +0.0286] | no |

Section 11 label: **not moved** (moved needs all four cells separated toward 0). Route: **LAUNCH-INTENT-HI** (W_HI 4.17852, four runs, its own PI go). Its controls read their known values: X = Y gives exactly 0.0; tick 0 gives exactly 0 in all four pairs; shuffled GT, the complement mask and a permuted Y move the number; a ±0.5 m/s injection reads back exactly. Beside, deciding nothing: the training draw alone (REP − DAG-OACC at s0) is −0.5560 [−0.6994, −0.3975] m/s, about 16 times the lever's largest point. The readings M-LAUNCH's builder had to choose (tick indexing, the v_GT rule, "toward 0" as the |X| − |Y| interval below 0, "moved" in all four cells) are in its NOTES.md; the Master Mind confirmed them. Its first control pass was badly posed and refused; the cell numbers did not change, and the redesign is disclosed in its NOTES.md.

Both routes need a PI go: B, registered after HARMS (section 10), and HI, the branch M-LAUNCH selects (section 11). The PI is asked which, if either, box 2 runs.

### 6.3 What is blocked, and on what (Rule Zero: every remaining lever)

| lever | status | blocker, and what unblocks it |
|---|---|---|
| B (W 0.726175, 4 runs) | committed next | **the PI's go** for box-2 time; then a B chain |
| M-LAUNCH (the section 11 routing variable) | NOT BUILT in the reader; being built separately | its build; it reads the tick dumps, which are single-copy on F: (section 8, item 6) |
| LOOP-STATE (zero training) | routed from M-LAUNCH *moved* | M-LAUNCH |
| LAUNCH-INTENT-HI (W 4.17852) | routed from M-LAUNCH *not moved* | M-LAUNCH, and its own PI go for box 2 |
| T0 four families | NOT RUN | a build item: `orc_families.py` pointed at the four arms and their controls; inputs: each run's eval JSON (repo) and `.npz` (E:) |
| anti-echo block; M2 / L2 launch subset | NOT BUILT | build items for the reader |

If both NEXT-LAUNCH branches are exhausted, the programme has no registered lever for the crawl rows on DAG-OACC (PREREG section 11). The blockers then are the ones that section names: route extent and nav timing (eliminated as the lever by N0, INHERITED), the state-channel levers (DAgger iteration 2, INHERITED), and the zero-training post-map (it moves the crawl but failed KEEP at s0, INHERITED). What would unblock: a new diagnosis of the loop term, or a PI decision on an architecture change that gives O a launch-intent input.

## 7. What this read cannot claim

* **It is not closed loop.** T1-kin is a self-action open loop: the lead follows its logged future and the cameras are logged. T0 is teacher-forced and is not driving performance.
* **One training run per arm, pair and seed.** The two pairs of a seed and the two seeds are four training draws of the lever. They disagree on ADE6 (section 5). No row here is an effect of the lever on its own (`H-ESTIM-SEED-1`).
* **It attributes to a weighting, not to a mechanism.** The lever is one label-defined weight on one term. The read does not say which of the 12 knots or which launch windows carried any cell.
* **The mask uses future GT speed.** That is admissible for labels. A result here says nothing about a deployable launch detector. Nothing new enters at inference.
* **The share is a training-weight fact.** K-SHARE and K-MASK are measured on the TRAIN windows of each run. They are not an eval fact.
* **B and HI are not read.** Nothing here says what a lower or a higher dose does.
* **The T0 four families, M-LAUNCH, the anti-echo block and the M2 / L2 launch-subset rows are not computed** (sections 3.6 and 3.7). The T-head macro-F1 behind R-TLAT / R-TLON is not in the JSON.

## 8. Disclosures

1. **Amendment 1 (`A1-2026-10-28`), before the read.**
   * **What was wrong.** The PREREG's K-BUILD row states a rule. Against `aeb54468`, the commit DAG-OACC trained at, each of the three files is byte-identical, or every changed hunk sits behind a default-off flag whose bit-identity test passes. The row then adds a fact: `v8f_mini_data.py` is byte-identical. That fact had been measured against `d7208328`, not against `aeb54468`. The reader coded it as a rule, so it refused the launch commit on the data module.
   * **What changed.** K-BUILD is read by its registered rule, with the data module pinned to DAG-ORC r2's build (sha256 `c117e1cf…`) and its twelve flag-gated hunks listed with a PASSing bit-identity test. The reader changed in that clause alone. No bar, row, outcome, predicate or other control changed.
   * **What had been seen: only controls.** These were the T1-kin chain's dry run (where K-BUILD REFUSES), A_0's run record and K-SHARE (both PASS), and the four launch-gate lines. The statement, and its time "written ~15:10", are the Amendment's own (INHERITED).
   * **Timing.** It landed at `e28516e8` (15:24:46). The read of record started after that (first console stamp 17:02:21). A_0's three T1-kin steps ran from 15:13:04 to 15:22:14, between the writing and the landing. Each fresh-T0 log prints one ADE6 summary line. The Master Mind states that before the landing it read only run-record, K-SHARE, gate and rollout start / end lines, and no metric line. A log cannot prove that negative, so it stays a statement (INHERITED from the Master Mind). The amendment's content is a code-bytes control that no metric value can change.
   * **Both readers are on record.** The launch-time reader (sha256 `5e950dcf…`) and the PREREG as promoted (sha256 `469bac1f…`) are what the launch-gate tokens bind. The reader of record (sha256 `fb3d2d7d…`) ran with the amended PREREG (sha256 `6ae79f2f…`).
   * **The T1-kin chain's own read with the launch-time reader REFUSED** at 17:01:36 (chain log), with one failed control, K-BUILD, and wrote nothing: *"stack/tanitad/data/v8f_mini_data.py differs from aeb54468: this build must leave it byte-identical"*. That read is banked as `lit1_logs/launch_intent_read_REFUSED.out`. The read of record ran the amended reader on the same manifest (sha256 `98d2cd3d…`).
   * **Effect on the pairs.** The replicate pair is unaffected: A_rep and REP have the same data module. The main pair carries DAG-ORC's flag-gated data hunks on the A side, as DAG-ORC's own main pair did.
2. **The T1-kin chain was stopped once, on its own RAM guard.**
   * **What happened.** It was stopped at 15:11, still waiting on the guard, and before any rollout. The stopped launch's last line, 15:07:07, reads *"LOW RAM available 3040 MB < 4096 MB"*, and its log holds no `start` line for a rollout. It was relaunched with a 2,048 MB guard.
   * **Where the numbers come from.** The 15:11 stop time and the 2,048 MB value are from the Master Mind's brief (INHERITED). The relaunch's log does not echo the guard value.
   * **What it did not change.** The relaunch started at 15:12:51 and its first rollout at 15:13:04. All eight rollouts in the relaunch's log show attempt 1. The template proofs (rollout argv equal to the banked argv, token for token) pass in both launches. The guard gates dev-box memory. It is not a measurement gate.
   * **Logs.** The stopped launch: `lit1_logs/chain.out.prev_1028151251`. The relaunch: `lit1_logs/chain.out`.
3. **Pin slip (dated note).**
   * **The slip.** `PINS_FILLED.txt` (in `raw/2026-10-28-launch-intent-chain/`) says the chain pins were filled ~14:10. They were filled ~13:50, before commit `d8c9f7bf` (13:51:52) and after the PREREG was promoted at `77234276` (13:49:00).
   * **Where the numbers come from.** The ~13:50 is from the Master Mind's brief (INHERITED). The bracket between the two commit times is MEASURED from the commits. The box-2 pins line is stamped 13:54.
   * **The pins are the promoted ones.** The PREREG blob and the reader blob at `77234276` hash to the `469bac1f…` and `5e950dcf…` that the pins carry (re-measured here).
4. **A label inside the reader.** The JSON's `provenance.prereg` reads *"Project Steering/PREREG_V8F_LAUNCH_INTENT.md (DRAFT v2, MM_REDESIGN_2026-10-28)"*. That is a string inside the reader. The reader of record was run with `--prereg-sha` equal to the amended text's sha256 `6ae79f2f…` (`read_A1/run_read_A1.sh`, `read_A1.out`).
5. **The bar and arms were redesigned before promotion.** The PREREG's disclosure, verbatim: *"The bar and arms were changed by the Master Mind before promotion, after DAG-ORC's floors and the REP s0 diagnosis were known, and before any LAUNCH-INTENT number existed."* The bar is 2 × max(F, RF0), with RF beside.
6. **Single copies.** The per-episode dumps and tick arrays are under `F:/TanitAD-scratch/launch_intent_t1_out/` (the chain's `--out-dir` and `--dump-ticks`). The repo listing at `d7dc9f11` holds the rollout JSONs under `t1/`, the fresh-T0 evals under `t0fresh/` and each run's record, and no tick arrays. M-LAUNCH reads tick dumps, so it needs F:. The checkpoints are on E: only.
7. **Disclosed coupling.** DAG-ORC and LAUNCH-INTENT share the REP_s runs. In DAG-ORC they set the floor RF; here they are the control of the replicate pair, and RF is not a bar term. Neither read's result changes the other's bar.
8. **What `check_result.py` is and is not.** It re-finds every quoted number in the banked JSON, logs and PREREG, recomputes the decision from the cells, and checks the document for the banned words. It does **not** recompute the cells from the per-window dumps or re-run the bootstrap (the dumps are a single copy on F:). A defect inside the estimator or the reader's heavy layer would not show there.

## 9. Run record

Evidence class MEASURED (ours): the box-2 chain log and the T1-kin chain logs, as banked.

**Box-2 chain** (`chain_v8f_launchintent.out`; tip `d8c9f7bf`). The chain waited for the DAG-ORC r2 chain's DONE marker, then ran: refresh, four launch gates (13:57 to 14:02), then train and T0 eval in the order A_0, A_1, A_rep,0, A_rep,1, with a pull and a run record after each. First stamped line 13:54, last 16:48: 174 min, 2.9 h. REP-LICENCE conditions 1 and 2 passed at both seeds in the chain; the reader checks conditions 3 and 4 (controls `K-REP-LICENCE:s0` and `K-REP-LICENCE:s1`, both PASS in the JSON). The closing line: *"B is not queued"*

| run | train: launch minute | train: ZZJOB-END | T0 eval: launch minute | T0 eval: ZZJOB-END |
|---|---|---|---|---|
| A_0 (M2b_s0_launchintent) | 14:02 | 14:41:35 | 14:41 | 14:43:37 |
| A_1 (M2b_s1_launchintent) | 14:44 | 15:22:52 | 15:22 | 15:24:52 |
| A_rep,0 (M2b_s0_launchintentrep) | 15:25 | 16:04:14 | 16:04 | 16:06:27 |
| A_rep,1 (M2b_s1_launchintentrep) | 16:06 | 16:45:37 | 16:45 | 16:47:37 |

**T1-kin chain** (`lit1_chain`, dev box CPU, one process at a time). Twelve steps (four runs × causal rollout, logged rollout, fresh-T0 eval); the logged rollouts and fresh-T0 evals are controls (K2-logged, K4), not rows. Step wall times:

| run | causal rollout (s) | logged rollout (s) | fresh-T0 eval (s) |
|---|---|---|---|
| A_0 | 218 | 218 | 88 |
| A_1 | 222 | 178 | 50 |
| A_rep,0 | 208 | 139 | 41 |
| A_rep,1 | 208 | 140 | 46 |

The twelve steps sum to 1756 s. The chain's own read ran 16:55:55 to 17:01:36 and REFUSED (disclosure 1). The read of record's console ends with *"ZZLAUNCH-INTENT-READ-DONE HARMS -> F:/TanitAD-scratch/launch_intent_t1_out/read_A1/LAUNCH_INTENT_READ.json"* and *"ZZREAD-A1-RC 0"*.

## 10. Provenance (artifact path, sha256)

Evidence class MEASURED (ours). Every sha256 below is of the bytes at `d7dc9f11`, re-measured for this document. The run directories are under `raw/2026-10-28-launch-intent-read/box2/`.

| role | repo path (under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-28-launch-intent-read/`) | sha256 |
|---|---|---|
| read of record (reader output) | `read_A1/LAUNCH_INTENT_READ.json` | `e451e1a6ba44e1d30e97561456476dc4223cd916bd3be96163cec4b02844bf23` |
| read of record, console | `read_A1/read_A1.out` | `2fd51cc833e1d3e85f3a2bb74319bcf9fc2ab736684bf87468c2bf22cf0adba7` |
| read of record, launcher | `read_A1/run_read_A1.sh` | `6c77c388adbfba6d88894886bc4d6ca31e57b97925a9599aa4f57b6198f5945f` |
| T1-kin chain's own read, REFUSED (launch-time reader) | `lit1_logs/launch_intent_read_REFUSED.out` | `c9028122ab0a4848c97ebd0800229cb5477b2625497fcdd0e9c4e3455ab1b543` |
| T1-kin chain log (relaunch) | `lit1_logs/chain.out` | `8bc3a51d4f58ffea73b002fc0e96a20b33efda01b0d6f8387d720bfe62cf2782` |
| T1-kin chain log (stopped launch) | `lit1_logs/chain.out.prev_1028151251` | `f7487872786af66f3522aa56a4667ab5c93578be9610012eeee5c5f22d865401` |
| T1-kin chain launch log (both launches) | `lit1_logs/lit1_chain.launch.out` | `1bf5706af86b642076b55196842e4d0e6b5c3c6c5984278d1f48bab3802ca630` |
| manifest the reader ran on | `runs.json` | `98d2cd3d3c18263396c26a12f54b5f60e7fd50e85b3b7edac6cc7eab1a4991e2` |
| box-2 chain log | `box2/chain_v8f_launchintent.out` | `ddae7b9e3ac82f86bfe65461944e7a55923701e685ee043a66f78362466bc5da` |
| box-2 pins | `box2/PINS.txt` | `030b77d39ed28dbd54878109a4dcc1e0e838e29fe3d37b0ec326e113ca6ef2f4` |
| K-BUILD record | `box2/K-BUILD.txt` | `568271d43133ba25eaf0cb90a2b5a1fd118a4ad8ed95bc04b6cd9721e78b28bd` |
| launch commit | `LAUNCH_COMMIT.txt` | `86b67b7a04f0507edecaa21b50420bdfb9a58e23ce2791f2d84892dbba2c0973` |
| chain-file pins | `PINS_BOX2.txt` | `20a5b48ccabce9b71b54398c4f14a746da6acc7d0f47415eb34a15fa358de018` |
| pin-fill note | `../2026-10-28-launch-intent-chain/PINS_FILLED.txt` | `fa825de89d77e6d826957bb317f068b886e2909a67382a372a1f1b6a199be6f1` |
| PREREG as amended (read by the reader of record) | `Project Steering/PREREG_V8F_LAUNCH_INTENT.md` at `e28516e8` and `d7dc9f11` | `6ae79f2f451868e8da9d270214d6132f2fdc7c9f6ff608c0ec05c35a58ac1d34` |
| run record A_0 | `box2/M2b_s0_launchintent/summary.json` | `93c83ed897363fea404916903baa5549bd66e229477a023c0b42c5b9b0df7da1` |
| run record A_0 | `box2/M2b_s0_launchintent/argv.json` | `8fb3609130b9db5fd4c951386f3ac2157772f61fd9a9029aaa79ff2bca6c5a72` |
| run record A_0 | `box2/M2b_s0_launchintent/launch_gate_token.json` | `4bf0c43cd3f94d9fc4961f3658cafcbcff09de1571f89c7e4a2836755b7f3c06` |
| run record A_0 | `box2/M2b_s0_launchintent/SHA256SUMS.txt` | `9c5efa778784eb08c3698a7522dc83084db91271a4c2c790bd6e0de8c9f31d69` |
| run record A_0 | `box2/M2b_s0_launchintent/eval_M2b-launchintent_s0.json` | `f7394b3b5ab0254bbddc07225ad07c39390be6ae934fd4bf6eafea9280d11105` |
| run record A_1 | `box2/M2b_s1_launchintent/summary.json` | `3e00ede39cc598e5a72b7edfe8da9290ecf32eb42be48362916e07027c18be91` |
| run record A_1 | `box2/M2b_s1_launchintent/argv.json` | `65f8373ebd1555f2d94b43a0ab8ca2a5847c03a997f25133f302a6c1109c5456` |
| run record A_1 | `box2/M2b_s1_launchintent/launch_gate_token.json` | `ac9ad2920aadd1821a657d06d008ccd8cc17be4d8f9b795c1ed7cc2998c991b4` |
| run record A_1 | `box2/M2b_s1_launchintent/SHA256SUMS.txt` | `de719392cdfb702c2156d4b71028c2e09b1add672c16ca25789ccbc4473bef0c` |
| run record A_1 | `box2/M2b_s1_launchintent/eval_M2b-launchintent_s1.json` | `68d7914904425ced660b4ebfbed1542885132b4f9a39bda2e5e42e881490b2d3` |
| run record A_rep,0 | `box2/M2b_s0_launchintentrep/summary.json` | `e124dfae208e4cf15d1d158dcbf21aea1e638ecd7bd8e4b2b702c57749937432` |
| run record A_rep,0 | `box2/M2b_s0_launchintentrep/argv.json` | `7d41fe6742e6c971fe19ea628aa8a5683e58d6339946d9432fdfef96bc6c0633` |
| run record A_rep,0 | `box2/M2b_s0_launchintentrep/launch_gate_token.json` | `4342775a42d441c1069f9534bb14598dc5a1bf7115cec2f8b43578113e1389d0` |
| run record A_rep,0 | `box2/M2b_s0_launchintentrep/SHA256SUMS.txt` | `03ad65e5b5fa5eaec6d9816781a6f9f4d87eebe729ab3216a0ccc82ef290f824` |
| run record A_rep,0 | `box2/M2b_s0_launchintentrep/eval_M2b-launchintentrep_s0.json` | `c6c52006e3efceb509d0bcac039d2972c313ee1658fbec12539ccace998d8599` |
| run record A_rep,1 | `box2/M2b_s1_launchintentrep/summary.json` | `b5b357ceac8e164419273a5e2d44cd2264dc6f06cac31faeb46091e928f2203d` |
| run record A_rep,1 | `box2/M2b_s1_launchintentrep/argv.json` | `8bbb121d7b989d235df942ab794dd7b1c53f9e2320cc08854e129c69fe84f5ff` |
| run record A_rep,1 | `box2/M2b_s1_launchintentrep/launch_gate_token.json` | `bd7ca2c9176b98db76d591e3213530b874dcc960b694efbcb8e5dd06547c48cc` |
| run record A_rep,1 | `box2/M2b_s1_launchintentrep/SHA256SUMS.txt` | `207fdd062c626d7f10a5cc634b8dfb0571e503864dc71e8e9039f206c2b6edf3` |
| run record A_rep,1 | `box2/M2b_s1_launchintentrep/eval_M2b-launchintentrep_s1.json` | `e9a7a56b64fb2637bb20029bfa7364b4412d39844b13ab73fd786563ab7edebb` |
| checkpoint A_0 (not staged; E: only) | `E:/Projects/TanitAD-artifacts/v8f-launch-intent/M2b_s0_launchintent/ckpt_last.pt` | `deb8f28b4ea9b2b1571f7ebea00c286922443e1dcc15f14f62e8cbcd7e143776` |
| checkpoint A_1 (not staged; E: only) | `E:/Projects/TanitAD-artifacts/v8f-launch-intent/M2b_s1_launchintent/ckpt_last.pt` | `abe0fd24e6f9906bb3545b5b61ad14ae916c606d969c83ef431f484dc7adc556` |
| checkpoint A_rep,0 (not staged; E: only) | `E:/Projects/TanitAD-artifacts/v8f-launch-intent/M2b_s0_launchintentrep/ckpt_last.pt` | `b86f7d4353b9fbedaa80d8034df7e9c0f9ae16cccd95884bdf358f2c40540f3c` |
| checkpoint A_rep,1 (not staged; E: only) | `E:/Projects/TanitAD-artifacts/v8f-launch-intent/M2b_s1_launchintentrep/ckpt_last.pt` | `2900ff73cd85571888aee2f3f76a6c46361d52b86ca4a755eee0dc5fd309c81e` |

The reader's pins (`provenance.rf0_sha256`, `provenance.reader_sha256`, `provenance.manifest_sha256`) are in the JSON. The PREREG read for this document is the file at `d7dc9f11`, which is the file at `e28516e8` (sha256 `6ae79f2f…`).
