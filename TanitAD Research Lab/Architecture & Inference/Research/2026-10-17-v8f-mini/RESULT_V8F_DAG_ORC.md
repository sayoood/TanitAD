# RESULT_V8F_DAG_ORC: O-RECOVER's launch-recovery perturbation composed into DAG-OACC's recipe. Outcome **HARMS** (arm A, p = 0.3, not adopted)

Claim row `H-V8F-DAG-ORC-1`. Pre-registration `Project Steering/PREREG_V8F_DAG_ORC.md` with **Amendment 1** (`A1-2026-10-27`, re-based on DAG-OACC, landed `6cd741b6`, before any DAG-ORC run) and **Amendment 2** (`A2-2026-10-28`, the composed perturbation's own random stream, landed `8c24823f`, before any DAG-ORC row). Read 2026-10-28 07:58 by the pre-registered reader (`raw/2026-10-28-dag-orc-build/rebase/read/dag_orc_read.py`, sha256 `2e5f9f17…`; started 07:54:18, rc 0 at 07:58:54, 275.4 s), all **162 reader controls PASS**. Output `raw/2026-10-28-dag-orc-result/read/DAGORC_READ.json` (sha256 `70a7a3a4…`) and `raw/2026-10-28-dag-orc-result/read/RF_REP.json` (sha256 `64c9444d…`, written before any A contrast was computed).

* **Tier.** T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan; plain arms). Never 'closed loop'. R-T0 is at T0: T0 (teacher-forced own-code, eval-141, ages cycle: box 2's standard eval). Never 'driving performance'.
* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed, paired, on the same windows and matched cells. It answers *"would another draw of EPISODES say this?"*. The control is a different training run (DAG-OACC), so the training-run question is carried by the two seeds and the replicate floors RF0 / RF (`H-ESTIM-SEED-1`). The planner is deterministic, so no inference-seed question arises.
* **Arms (box 2, r2c chain pinned at `8c24823f`; four gates PASS, four run records PASS, four K-REACH PASS).** A_s = DAG-OACC_s's training argv verbatim plus the six perturbation flags at the registered law (`--ego-perturb-p 0.3`) and `--ego-perturb-with-dagger`. REP_s = DAG-OACC_s's argv with `--seed 10` / `--seed 11` (same init M2b_s). CTRL = DAG-OACC_s, banked (no control was retrained). Checkpoints, sha256 prefixes: A `380b230d…` (s0) / `870dded7…` (s1); REP `b03fd6ef…` / `77853e56…`; CTRL `67265ec1…` / `90b01474…`. **Arm B (p = 0.1) was not trained and is not read.**
* **Coverage.** The four families at T1-kin are in section 3.1. **The T0 four families are NOT RUN** (section 3.2), and three registered beside rows are NOT BUILT (section 3.3). They are work items, and the outcome above does not use them.

## 1. The outcome, as the table decides it

| condition (PREREG section 7, as amended) | s0 | s1 |
|---|---|---|
| controls (reader, 162) | PASS | PASS |
| guard R-TLAT (T head macro-F1, ticks ≥ 1, A − CTRL not separated-worse) | held | held |
| guard R-TLON (T head macro-F1, ticks ≥ 1, A − CTRL not separated-worse) | held | held |
| guard R-ADE (T1-kin ADE6, A − CTRL, not separated-worse) | **FAILS**: +0.1434 [+0.0538, +0.2342] m, separated-worse | held: +0.0565 [−0.0115, +0.1255] m, not separated |
| guard R-T0 (T0 own-code ADE6, A − CTRL, point ≤ +0.0604 m) | **FAILS**: +0.1106 m | **FAILS**: +0.0722 m |
| guard R-NEW (no row of D_s fires) | held: 0 of 15 rows | held: 0 of 15 rows |
| decisive R-MOVE (every row of C clears its bar) | false: 0 of 4 rows clear | false: 0 of 4 rows clear |
| decisive R-CRAWL (every row of C not separated-worse vs P10) | false: 1 of 4 rows (yaw-rate only) | false: 0 of 4 rows |
| adoption R-KEEP (T1-kin ADE6, A − DAG-OACC, point ≤ +0.0604 m) | false: +0.1434 m | true: +0.0565 m |
| adoption R-NOBASE | true: 0 of 15 rows violate | true: 0 of 15 rows violate |

R-ADE and R-KEEP read one contrast, A − DAG-OACC (Amendment 1 section 2: DAG = CTRL = DAG-OACC), so the two rows of ADE6 above are the same cells.

**HARMS.** The table's condition, verbatim: *"controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed"*. Three guard cells fail: R-ADE at s0, and R-T0 at s0 and at s1. REFUSED is decided first and HARMS second (PREREG section 7, *Precedence*), so the R-MOVE flags do not change the word. Committed next, verbatim from the JSON: *"A: not adopted; read B (p 0.1) under this same table (B is NOT queued: it needs its own PI go, A1 section 4)."*

⚠ **Reported as written, beside the word.** R-MOVE is false at both seeds, with 0 of 4 crawl rows clearing the bar at either seed (section 2.4). R-CRAWL is false at both seeds. The guard failures are the registered conditions; the table does not test whether the cost is the lever's (section 4 (c)).

The word is the reader's output (`matching_rows_in_table_order`: one row, `HARMS`). The script that built this document recomputed R-MOVE, R-CRAWL, R-NEW, R-NOBASE, R-ADE, R-T0 and R-KEEP per seed from the reader's own deltas and floors, and all of them agree with the reader's flags (`audit/gen_result.py`).

## 2. The rows (T1-kin; point [95 % CI]; B / W = separated-better / separated-worse)

The contrasts are the reader's own `read_a`. **A − DAG is identical to A − CTRL, cell for cell**, because Amendment 1 sets DAG = DAG-OACC = CTRL (`read_a.A_DAG` equals `read_a.A_CTRL` and `read_a.DAG_P10` equals `read_a.CTRL_P10` in the JSON). P10 is the kinematic floor. Row units are the reader's own. The last column of 2.1 is the `n` field (windows / episodes) of the A − CTRL cell at s0 and s1. For the three S-POS rows the P10 contrast is the run's own S − CTRV, and the JSON labels its `n` *"ticks>=1, unpaired vs CTRV (DAG_READ.json)"*: those cells are not paired across runs.

### 2.1 The floors and the bar (`floors`)

`bar_r = 2 × max(F_r, RF0_r, RF_r)`, one value per row, applied at both seeds, inclusive. F is 0.1207 m on ADE6 and 0 elsewhere. RF0 is from the banked DAG-0 / ORC-0 pair (frozen `P0_RF0.json`, sha256 `1608d70b…`). RF is max over s of |REP − CTRL| (`RF_REP.json`, written before any A contrast).

| row | better | F | RF0 | RF | bar | the max term | RF set by | n, A − CTRL (s0 ; s1) |
|---|---|---|---|---|---|---|---|---|
| ADE6 | lower | 0.1207 | 0.0138 | 0.4025 | 0.8050 | RF | REP s0 | 2399 / 141 |
| speed MAE | lower | 0.0000 | 0.0796 | 0.3373 | 0.6746 | RF | REP s0 | 2399 / 141 |
| along-track MAE | lower | 0.0000 | 0.0201 | 0.4596 | 0.9192 | RF | REP s0 | 2399 / 141 |
| accel MAE | lower | 0.0000 | 0.1323 | 0.5603 | 1.1206 | RF | REP s0 | 2399 / 141 |
| DK-VIOL | lower | 0.0000 | 0.0050 | 0.0017 | 0.0100 | RF0 | REP s0 | 595 / 70 |
| DK-HW | lower | 0.0000 | 0.1308 | 0.2540 | 0.5080 | RF | REP s0 | 709 / 72 ; 710 / 73 |
| DK-TG | lower | 0.0000 | 0.0041 | 0.1042 | 0.2084 | RF | REP s0 | 654 / 72 ; 652 / 72 |
| heading MAE | lower | 0.0000 | 0.3801 | 2.5074 | 5.0148 | RF | REP s0 | 2373 / 141 |
| yaw-rate MAE | lower | 0.0000 | 0.5535 | 15.8553 | 31.7106 | RF | REP s0 | 2366 / 141 |
| curvature MAE | lower | 0.0000 | 0.0263 | 0.1324 | 0.2648 | RF | REP s0 | 2366 / 141 |
| cross-track MAE | lower | 0.0000 | 0.0320 | 0.0224 | 0.0640 | RF0 | REP s1 | 2399 / 141 |
| L1 lateral-accel MAE | lower | 0.0000 | 0.3985 | 1.2242 | 2.4484 | RF | REP s0 | 2366 / 141 |
| L2 curvature MAE (GT>=5.2) | lower | 0.0000 | 0.0068 | 0.0870 | 0.1740 | RF | REP s0 | 2105 / 135 ; 2106 / 135 |
| TAC-LAT | higher | 0.0000 | 0.0183 | 0.0421 | 0.0842 | RF | REP s0 | 2399 / 141 |
| TAC-LON | higher | 0.0000 | 0.0125 | 0.0950 | 0.1900 | RF | REP s0 | 2399 / 141 |
| S-POS@30 | lower | 0.0000 | 0.1877 | 0.2592 | 0.5184 | RF | REP s0 | 2321 / 141 |
| S-POS@50 | lower | 0.0000 | 0.4449 | 0.3912 | 0.8898 | RF0 | REP s0 | 2313 / 141 |
| S-POS@80 | lower | 0.0000 | 1.0082 | 0.5170 | 2.0164 | RF0 | REP s1 | 2290 / 141 |
| M1 crawl share | lower | 0.0000 | 0.0048 | 0.0388 | 0.0776 | RF | REP s0 | 2105 / 135 ; 2106 / 135 |

The REP pair sets the bar on 15 of 19 rows (every row of C among them); RF0 sets it on 4 rows. REP s0 sets RF on 17 of 19 rows; REP s1 sets it on cross-track MAE and S-POS@80.

### 2.2 A − CTRL (the one-lever contrast; = A − DAG-OACC) and REP − CTRL (the replicate)

| row | A − CTRL s0 | A − CTRL s1 | REP − CTRL s0 | REP − CTRL s1 |
|---|---|---|---|---|
| ADE6 | **+0.1434 [+0.0538, +0.2342] W** | +0.0565 [−0.0115, +0.1255] – | **+0.4025 [+0.2761, +0.5343] W** | +0.0116 [−0.0531, +0.0765] – |
| speed MAE | +0.0371 [−0.0027, +0.0782] – | +0.0139 [−0.0215, +0.0500] – | **+0.3373 [+0.2661, +0.4080] W** | +0.0438 [−0.0147, +0.1046] – |
| along-track MAE | **+0.1188 [+0.0323, +0.2168] W** | +0.0391 [−0.0272, +0.1071] – | **+0.4596 [+0.3335, +0.5913] W** | +0.0117 [−0.0472, +0.0665] – |
| accel MAE | +0.0494 [−0.0342, +0.1415] – | **+0.1064 [+0.0031, +0.2287] W** | **+0.5603 [+0.4375, +0.6842] W** | **+0.2005 [+0.0279, +0.3860] W** |
| DK-VIOL | **+0.0185 [+0.0039, +0.0352] W** | −0.0101 [−0.0227, +0.0000] – | −0.0017 [−0.0243, +0.0214] – | +0.0000 [−0.0158, +0.0167] – |
| DK-HW | +0.0123 [−0.1747, +0.1739] – | −0.0175 [−0.1399, +0.1017] – | **+0.2540 [+0.0200, +0.5249] W** | −0.0208 [−0.1592, +0.1182] – |
| DK-TG | +0.0037 [−0.0935, +0.0826] – | −0.0286 [−0.0758, +0.0120] – | **+0.1042 [+0.0060, +0.2537] W** | −0.0227 [−0.0751, +0.0219] – |
| heading MAE | −0.4492 [−0.7852, −0.0952] B | −0.8053 [−1.3937, −0.1649] B | **+2.5074 [+1.7272, +3.4192] W** | +0.2674 [−0.1926, +0.7560] – |
| yaw-rate MAE | −2.7748 [−4.8396, −0.4768] B | −5.8026 [−7.8146, −4.0038] B | **+15.8553 [+11.9088, +20.2953] W** | +0.8903 [−0.5695, +2.5371] – |
| curvature MAE | −0.0447 [−0.0615, −0.0298] B | −0.0700 [−0.1027, −0.0419] B | **+0.1324 [+0.0904, +0.1795] W** | −0.0008 [−0.0335, +0.0292] – |
| cross-track MAE | **+0.0371 [+0.0016, +0.0732] W** | **+0.0296 [+0.0003, +0.0602] W** | −0.0166 [−0.0633, +0.0298] – | +0.0224 [−0.0220, +0.0649] – |
| L1 lateral-accel MAE | +0.2156 [−0.0419, +0.5572] – | −0.1608 [−0.3946, +0.1082] – | **+1.2242 [+0.9141, +1.5672] W** | +0.3222 [−0.1466, +0.8169] – |
| L2 curvature MAE (GT>=5.2) | −0.0254 [−0.0468, −0.0102] B | −0.0311 [−0.0508, −0.0152] B | **+0.0870 [+0.0503, +0.1344] W** | +0.0060 [−0.0092, +0.0216] – |
| TAC-LAT | +0.0213 [+0.0096, +0.0342] B | +0.0242 [+0.0083, +0.0425] B | **−0.0421 [−0.0621, −0.0225] W** | **−0.0263 [−0.0454, −0.0075] W** |
| TAC-LON | +0.0158 [−0.0017, +0.0338] – | +0.0046 [−0.0142, +0.0229] – | **−0.0950 [−0.1384, −0.0538] W** | +0.0196 [−0.0121, +0.0546] – |
| S-POS@30 | **+0.2321 [+0.0609, +0.4341] W** | +0.1340 [−0.0629, +0.3354] – | **+0.2592 [+0.0537, +0.4573] W** | +0.1120 [−0.1269, +0.3535] – |
| S-POS@50 | +0.3142 [−0.0246, +0.6741] – | +0.2622 [−0.1258, +0.6623] – | +0.3912 [−0.0284, +0.8180] – | −0.2390 [−0.6295, +0.1314] – |
| S-POS@80 | +0.5049 [−0.1379, +1.2090] – | +0.2001 [−0.4531, +0.8437] – | +0.1921 [−0.5499, +0.9344] – | −0.5170 [−1.2818, +0.1862] – |
| M1 crawl share | −0.0262 [−0.0356, −0.0178] B | −0.0166 [−0.0237, −0.0101] B | **+0.0388 [+0.0270, +0.0519] W** | +0.0048 [−0.0019, +0.0117] – |

REP differs from CTRL in the trainer seed alone. Even so, REP − CTRL reads separated on 15 of 19 rows at s0 and 2 of 19 at s1.

### 2.3 A − P10 and CTRL − P10 (DAG-OACC − P10, the banked control)

| row | A − P10 s0 | A − P10 s1 | CTRL − P10 s0 | CTRL − P10 s1 |
|---|---|---|---|---|
| ADE6 | −1.0578 [−1.2673, −0.8601] B | −1.0667 [−1.2806, −0.8552] B | −1.2012 [−1.3945, −1.0186] B | −1.1231 [−1.3301, −0.9159] B |
| speed MAE | −0.7390 [−0.9388, −0.5695] B | −0.6890 [−0.8867, −0.5148] B | −0.7761 [−0.9806, −0.5996] B | −0.7030 [−0.9087, −0.5230] B |
| along-track MAE | −0.5622 [−0.7350, −0.3877] B | −0.5485 [−0.7288, −0.3652] B | −0.6809 [−0.8443, −0.5214] B | −0.5876 [−0.7702, −0.4119] B |
| accel MAE | −0.7678 [−1.1205, −0.4615] B | −0.4950 [−0.7872, −0.2275] B | −0.8172 [−1.1770, −0.4998] B | −0.6014 [−0.9063, −0.3345] B |
| DK-VIOL | +0.0202 [−0.0080, +0.0536] – | −0.0134 [−0.0356, +0.0070] – | +0.0017 [−0.0231, +0.0272] – | −0.0034 [−0.0243, +0.0181] – |
| DK-HW | −0.2853 [−0.6770, +0.0241] – | −0.4630 [−0.8266, −0.1609] B | −0.3318 [−0.6942, −0.0242] B | −0.4466 [−0.8200, −0.1686] B |
| DK-TG | −0.0629 [−0.1275, −0.0077] B | −0.1031 [−0.1650, −0.0532] B | −0.0568 [−0.1232, +0.0163] – | −0.0724 [−0.1384, −0.0085] B |
| heading MAE | −3.0412 [−3.7939, −2.3379] B | −2.8592 [−3.6395, −2.0969] B | −2.5920 [−3.2391, −1.9751] B | −2.0539 [−2.7092, −1.3705] B |
| yaw-rate MAE | +0.6976 [−2.3078, +3.7030] – | **+2.8732 [+0.1261, +5.7756] W** | **+3.4726 [+0.6601, +6.3308] W** | **+8.6844 [+5.1690, +12.5740] W** |
| curvature MAE | **+0.0409 [+0.0205, +0.0656] W** | **+0.0649 [+0.0400, +0.0947] W** | **+0.0854 [+0.0580, +0.1175] W** | **+0.1352 [+0.0889, +0.1915] W** |
| cross-track MAE | −0.9248 [−1.0731, −0.7904] B | −0.9446 [−1.0903, −0.8090] B | −0.9620 [−1.1077, −0.8319] B | −0.9742 [−1.1152, −0.8405] B |
| L1 lateral-accel MAE | −2.2451 [−3.5098, −1.1596] B | −2.0073 [−3.0828, −1.0292] B | −2.4618 [−3.8097, −1.2924] B | −1.8473 [−2.9617, −0.8501] B |
| L2 curvature MAE (GT>=5.2) | **+0.0193 [+0.0092, +0.0316] W** | **+0.0218 [+0.0111, +0.0356] W** | **+0.0443 [+0.0232, +0.0744] W** | **+0.0529 [+0.0281, +0.0847] W** |
| TAC-LAT | +0.2118 [+0.1735, +0.2520] B | +0.2213 [+0.1856, +0.2606] B | +0.1905 [+0.1508, +0.2319] B | +0.1972 [+0.1600, +0.2364] B |
| TAC-LON | +0.1555 [+0.0971, +0.2123] B | +0.1417 [+0.0804, +0.2026] B | +0.1396 [+0.0771, +0.1997] B | +0.1371 [+0.0751, +0.2001] B |
| S-POS@30 | −1.5107 [−2.0061, −1.0815] B | −1.6914 [−2.1591, −1.2925] B | −1.7428 [−2.2737, −1.2861] B | −1.8255 [−2.3433, −1.3880] B |
| S-POS@50 | −4.2189 [−5.3198, −3.2295] B | −4.3469 [−5.3021, −3.4420] B | −4.5331 [−5.6695, −3.5422] B | −4.6092 [−5.6305, −3.6576] B |
| S-POS@80 | −9.0575 [−11.2001, −7.0842] B | −9.7248 [−11.7856, −7.7220] B | −9.5624 [−11.6843, −7.6124] B | −9.9249 [−11.9750, −7.9686] B |
| M1 crawl share | **+0.0307 [+0.0145, +0.0498] W** | **+0.0437 [+0.0240, +0.0669] W** | **+0.0568 [+0.0378, +0.0798] W** | **+0.0604 [+0.0391, +0.0864] W** |

### 2.4 The decisive rows: the crawl set C (yaw-rate, curvature, L2, M1 at both seeds)

R-MOVE: A − CTRL separated-better AND the improvement (−delta) ≥ bar. R-CRAWL: A − P10 not separated-worse.

| seed | row | A − CTRL | improvement | bar | improvement / bar | clears (R-MOVE row) | A − P10 | not separated-worse (R-CRAWL row) |
|---|---|---|---|---|---|---|---|---|
| s0 | yaw-rate MAE | −2.7748 [−4.8396, −0.4768] B | 2.7748 | 31.7106 | 8.8 % | false | +0.6976 [−2.3078, +3.7030] – | true |
| s0 | curvature MAE | −0.0447 [−0.0615, −0.0298] B | 0.0447 | 0.2648 | 16.9 % | false | **+0.0409 [+0.0205, +0.0656] W** | false |
| s0 | L2 curvature MAE (GT>=5.2) | −0.0254 [−0.0468, −0.0102] B | 0.0254 | 0.1740 | 14.6 % | false | **+0.0193 [+0.0092, +0.0316] W** | false |
| s0 | M1 crawl share | −0.0262 [−0.0356, −0.0178] B | 0.0262 | 0.0776 | 33.8 % | false | **+0.0307 [+0.0145, +0.0498] W** | false |
| s1 | yaw-rate MAE | −5.8026 [−7.8146, −4.0038] B | 5.8026 | 31.7106 | 18.3 % | false | **+2.8732 [+0.1261, +5.7756] W** | false |
| s1 | curvature MAE | −0.0700 [−0.1027, −0.0419] B | 0.0700 | 0.2648 | 26.4 % | false | **+0.0649 [+0.0400, +0.0947] W** | false |
| s1 | L2 curvature MAE (GT>=5.2) | −0.0311 [−0.0508, −0.0152] B | 0.0311 | 0.1740 | 17.9 % | false | **+0.0218 [+0.0111, +0.0356] W** | false |
| s1 | M1 crawl share | −0.0166 [−0.0237, −0.0101] B | 0.0166 | 0.0776 | 21.4 % | false | **+0.0437 [+0.0240, +0.0669] W** | false |

Every one of the eight cells is separated-better against CTRL, and every one is below its bar: the largest is 33.8 % of the bar (M1 at s0) and the smallest 8.8 % (yaw-rate at s0).

### 2.5 The guard rows R-NEW and R-NOBASE (domain D_s: 15 rows at each seed, no exempt rows)

R-NEW fires on a row when (1) A − P10 is separated-worse, (2) DAG-OACC − P10 is not, and (3) A − CTRL is separated-worse with the worsening ≥ bar. R-NOBASE: (1) and (3).

| seed | rows with A − CTRL separated-worse | of those, worsening ≥ bar (3) | of those, A − P10 separated-worse (1) | fires |
|---|---|---|---|---|
| s0 | ADE6, along-track MAE, DK-VIOL, cross-track MAE, S-POS@30 | DK-VIOL | none | none |
| s1 | accel MAE, cross-track MAE | none | none | none |

The closest approach is DK-VIOL at s0: A − CTRL +0.0185 is separated-worse and above its bar (0.0100). R-NEW does not fire there because A − P10 on DK-VIOL is not separated-worse (+0.0202 [−0.0080, +0.0536] –).

## 3. The four families

### 3.1 T1-kin (the primary tier), per seed, per family, never pooled

Tier: T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan; plain arms). Never 'closed loop'. Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, per seed. Source `four_families.A.<seed>.<family>.rows` (each family block carries this same tier and estimator string, and its leaves equal `read_a` cell for cell; the build script asserts both).

| family | row | A − CTRL s0 | A − CTRL s1 | A − P10 s0 | A − P10 s1 |
|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE | +0.0371 [−0.0027, +0.0782] – | +0.0139 [−0.0215, +0.0500] – | −0.7390 [−0.9388, −0.5695] B | −0.6890 [−0.8867, −0.5148] B |
|  | along-track MAE | **+0.1188 [+0.0323, +0.2168] W** | +0.0391 [−0.0272, +0.1071] – | −0.5622 [−0.7350, −0.3877] B | −0.5485 [−0.7288, −0.3652] B |
|  | accel MAE | +0.0494 [−0.0342, +0.1415] – | **+0.1064 [+0.0031, +0.2287] W** | −0.7678 [−1.1205, −0.4615] B | −0.4950 [−0.7872, −0.2275] B |
|  | DK-VIOL | **+0.0185 [+0.0039, +0.0352] W** | −0.0101 [−0.0227, +0.0000] – | +0.0202 [−0.0080, +0.0536] – | −0.0134 [−0.0356, +0.0070] – |
|  | DK-HW | +0.0123 [−0.1747, +0.1739] – | −0.0175 [−0.1399, +0.1017] – | −0.2853 [−0.6770, +0.0241] – | −0.4630 [−0.8266, −0.1609] B |
|  | DK-TG | +0.0037 [−0.0935, +0.0826] – | −0.0286 [−0.0758, +0.0120] – | −0.0629 [−0.1275, −0.0077] B | −0.1031 [−0.1650, −0.0532] B |
| LATERAL | heading MAE | −0.4492 [−0.7852, −0.0952] B | −0.8053 [−1.3937, −0.1649] B | −3.0412 [−3.7939, −2.3379] B | −2.8592 [−3.6395, −2.0969] B |
|  | yaw-rate MAE | −2.7748 [−4.8396, −0.4768] B | −5.8026 [−7.8146, −4.0038] B | +0.6976 [−2.3078, +3.7030] – | **+2.8732 [+0.1261, +5.7756] W** |
|  | curvature MAE | −0.0447 [−0.0615, −0.0298] B | −0.0700 [−0.1027, −0.0419] B | **+0.0409 [+0.0205, +0.0656] W** | **+0.0649 [+0.0400, +0.0947] W** |
|  | cross-track MAE | **+0.0371 [+0.0016, +0.0732] W** | **+0.0296 [+0.0003, +0.0602] W** | −0.9248 [−1.0731, −0.7904] B | −0.9446 [−1.0903, −0.8090] B |
|  | L1 lateral-accel MAE | +0.2156 [−0.0419, +0.5572] – | −0.1608 [−0.3946, +0.1082] – | −2.2451 [−3.5098, −1.1596] B | −2.0073 [−3.0828, −1.0292] B |
|  | L2 curvature MAE (GT>=5.2) | −0.0254 [−0.0468, −0.0102] B | −0.0311 [−0.0508, −0.0152] B | **+0.0193 [+0.0092, +0.0316] W** | **+0.0218 [+0.0111, +0.0356] W** |
| TACTICAL | TAC-LAT | +0.0213 [+0.0096, +0.0342] B | +0.0242 [+0.0083, +0.0425] B | +0.2118 [+0.1735, +0.2520] B | +0.2213 [+0.1856, +0.2606] B |
|  | TAC-LON | +0.0158 [−0.0017, +0.0338] – | +0.0046 [−0.0142, +0.0229] – | +0.1555 [+0.0971, +0.2123] B | +0.1417 [+0.0804, +0.2026] B |
| | R-TLAT / R-TLON (T-head macro-F1, A − CTRL) | held | held | n/a (P10 has no T head) | n/a (P10 has no T head) |
| STRATEGIC | S-POS@30 | **+0.2321 [+0.0609, +0.4341] W** | +0.1340 [−0.0629, +0.3354] – | −1.5107 [−2.0061, −1.0815] B | −1.6914 [−2.1591, −1.2925] B |
|  | S-POS@50 | +0.3142 [−0.0246, +0.6741] – | +0.2622 [−0.1258, +0.6623] – | −4.2189 [−5.3198, −3.2295] B | −4.3469 [−5.3021, −3.4420] B |
|  | S-POS@80 | +0.5049 [−0.1379, +1.2090] – | +0.2001 [−0.4531, +0.8437] – | −9.0575 [−11.2001, −7.0842] B | −9.7248 [−11.7856, −7.7220] B |

STRATEGIC is reported, not paused, for v8F (`four_families.A.s0.STRATEGIC.ruling`: *"NOT paused for v8F (PI ruling 2026-10-17)"*). Its rows are S-POS@30 / @50 / @80 (lower is better); the A − CTRL leaf is the difference of the two runs' (S − CTRV, ticks ≥ 1) contrasts on the same windows (`read_a.A_CTRL.s0["S-POS@30"].kind`: *"difference of the two runs' (S - CTRV, ticks>=1) contrasts on the same windows"*). The T-head macro-F1 values behind R-TLAT / R-TLON are not carried in the JSON, only the booleans (UNVERIFIED as numbers).

### 3.2 T0 (box 2's standard eval): the four families are NOT RUN

`t0_four_families.status` reads **NOT RUN**. The PREREG (section 5) registered the T0 families as a build item of the reader (*"DAGGER2 left the T0 families NOT RUN. Here they are a build item of the reader"*); the reader did not carry it. Work item, verbatim from the JSON: *"orc_families.py pointed at A / CTRL (section 5): the T0 LONGITUDINAL / LATERAL / TACTICAL rows and distance keeping at T0; a build item, never a silent omission"*. The only T0 numbers in this read are the R-T0 guard points, T0 own-code ADE6 A − CTRL +0.1106 m (s0) and +0.0722 m (s1), with no interval in the JSON. Inputs exist: the pulled T0 evals (`eval_M2b-dagoaccorc_s*.json` / `.npz`) of A and of CTRL, on E:. **Nothing is claimed for T0 LONGITUDINAL, LATERAL or TACTICAL here.**

### 3.3 Registered beside rows that are NOT BUILT

* `M-LAUNCH`: X - CTRL of the mean (v_own - v_GT) at ticks 1-2 in the GT-launch windows, from the tick dumps: NOT BUILT
* `anti_echo_block`: longitudinal_claim_admissible, hold-v0, copy detector (RESULT_V8F_ORECOVER 3.1): NOT BUILT
* `M2_and_L2_launch_subset`: NOT BUILT

Not computable by design (PREREG section 5): T1-kin goal-head F1 and head accuracy (the paired tool reads eval `run` dumps, not tick dumps); anchor selection (one path per window).

## 4. Beside, deciding nothing

### (a) A − CTRL is separated-better on every crawl row and on heading, at both seeds. Re-verified from the JSON:

| row | s0 | s1 |
|---|---|---|
| yaw-rate MAE | −2.7748 [−4.8396, −0.4768] B | −5.8026 [−7.8146, −4.0038] B |
| curvature MAE | −0.0447 [−0.0615, −0.0298] B | −0.0700 [−0.1027, −0.0419] B |
| L2 curvature MAE (GT>=5.2) | −0.0254 [−0.0468, −0.0102] B | −0.0311 [−0.0508, −0.0152] B |
| M1 crawl share | −0.0262 [−0.0356, −0.0178] B | −0.0166 [−0.0237, −0.0101] B |
| heading MAE | −0.4492 [−0.7852, −0.0952] B | −0.8053 [−1.3937, −0.1649] B |

All 10 cells are separated-better (CI upper bound below 0). The sizes against the bar are in section 2.4: the effect is real at the episode level and small against the replicate floor (c).

### (b) The share of the CTRL − P10 crawl gap that A closes: (CTRL−P10 − (A−P10)) / (CTRL−P10), from the reader's deltas

| seed | row | CTRL − P10 | A − P10 | share closed | matched-cell form −(A−CTRL) / (CTRL−P10) |
|---|---|---|---|---|---|
| s0 | yaw-rate MAE | +3.4726 | +0.6976 | 79.9 % | 79.9 % |
| s0 | curvature MAE | +0.0854 | +0.0409 | 52.1 % | 52.3 % |
| s0 | L2 curvature MAE (GT>=5.2) | +0.0443 | +0.0193 | 56.4 % | 57.3 % |
| s0 | M1 crawl share | +0.0568 | +0.0307 | 46.0 % | 46.1 % |
| s1 | yaw-rate MAE | +8.6844 | +2.8732 | 66.9 % | 66.8 % |
| s1 | curvature MAE | +0.1352 | +0.0649 | 52.0 % | 51.8 % |
| s1 | L2 curvature MAE (GT>=5.2) | +0.0529 | +0.0218 | 58.8 % | 58.8 % |
| s1 | M1 crawl share | +0.0604 | +0.0437 | 27.6 % | 27.5 % |

A closes 27.6 % to 79.9 % of the gap that DAG-OACC leaves to P10 on the crawl rows (largest: yaw-rate at s0; smallest: M1 at s1). The two forms differ by at most 0.9 points, because the CTRL − P10 and A − P10 cells differ by one window on 6 of the 8 rows (the `n` fields). At s0 the yaw-rate row ends not separated-worse against P10; the other 7 rows stay separated-worse.

### (c) The replicate floor is set by REP s0; REP differs from CTRL only in the trainer seed

K-PAIR (reader control) PASS at both seeds: argv(REP_s) equals argv(DAG-OACC_s) except `--out` and `--seed` (10 vs 0, 11 vs 1). So REP − CTRL is a measured draw of **training variance** on this rig (`H-ESTIM-SEED-1`), the one thing the episode-cluster bootstrap does not see.

| row | REP − CTRL s0 | REP − CTRL s1 | RF (max over s of the absolute REP − CTRL) | set by | RF0 | bar (2 × max) |
|---|---|---|---|---|---|---|
| ADE6 | **+0.4025 [+0.2761, +0.5343] W** | +0.0116 [−0.0531, +0.0765] – | 0.4025 | REP s0 | 0.0138 | 0.8050 |
| curvature MAE | **+0.1324 [+0.0904, +0.1795] W** | −0.0008 [−0.0335, +0.0292] – | 0.1324 | REP s0 | 0.0263 | 0.2648 |
| yaw-rate MAE | **+15.8553 [+11.9088, +20.2953] W** | +0.8903 [−0.5695, +2.5371] – | 15.8553 | REP s0 | 0.5535 | 31.7106 |
| M1 crawl share | **+0.0388 [+0.0270, +0.0519] W** | +0.0048 [−0.0019, +0.0117] – | 0.0388 | REP s0 | 0.0048 | 0.0776 |

The A − CTRL effects, set beside the floor they had to clear (size = |A − CTRL|; the bar is 2 × RF on all four crawl rows):

| row | A − CTRL s0 | as share of RF | of bar | A − CTRL s1 | as share of RF | of bar |
|---|---|---|---|---|---|---|
| yaw-rate MAE | −2.7748 | 17.5 % | 8.8 % | −5.8026 | 36.6 % | 18.3 % |
| curvature MAE | −0.0447 | 33.8 % | 16.9 % | −0.0700 | 52.9 % | 26.4 % |
| L2 curvature MAE (GT>=5.2) | −0.0254 | 29.2 % | 14.6 % | −0.0311 | 35.7 % | 17.9 % |
| M1 crawl share | −0.0262 | 67.5 % | 33.8 % | −0.0166 | 42.8 % | 21.4 % |
| ADE6 | +0.1434 | 35.6 % | 17.8 % | +0.0565 | 14.0 % | 7.0 % |

Two readings, arithmetic only. (i) On the four crawl rows A − CTRL is 17.5 % to 67.5 % of RF, so each is smaller than the REP s0 draw on the same row. (ii) REP s0 − CTRL on ADE6 is +0.4025 m, 3.33 × F (0.1207 m); the s0 cost A − CTRL on ADE6 that failed R-ADE, +0.1434 m, is 36 % of that replicate draw. REP s1 − CTRL on ADE6 is +0.0116 m. The bar is one value per row from the larger draw by registration (PREREG section 3, *Two choices*), so a small s1 draw does not lower it.

### (d) The longitudinal cost at s0 (A − CTRL)

| row | A − CTRL s0 | A − CTRL s1 | RF | bar | means s0 (A vs CTRL) |
|---|---|---|---|---|---|
| ADE6 | **+0.1434 [+0.0538, +0.2342] W** | +0.0565 [−0.0115, +0.1255] – | 0.4025 | 0.8050 | 2.3855 vs 2.2421 |
| along-track MAE | **+0.1188 [+0.0323, +0.2168] W** | +0.0391 [−0.0272, +0.1071] – | 0.4596 | 0.9192 | 1.8960 vs 1.7772 |
| DK-VIOL | **+0.0185 [+0.0039, +0.0352] W** | −0.0101 [−0.0227, +0.0000] – | 0.0017 | 0.0100 | n/a |

At s0 all three are separated-worse against CTRL; at s1 none is. Against the floors: ADE6 +0.1434 is below its RF (0.4025) and its bar (0.8050); along-track +0.1188 is below its RF (0.4596); DK-VIOL +0.0185 is above its RF (0.0017) and above its bar (0.0100, which RF0 sets), and R-NEW did not fire on it (section 2.5). At s1 the longitudinal family carries one separated-worse row, accel MAE (**+0.1064 [+0.0031, +0.2287] W**).

### What this read cannot claim

* **It is not closed loop.** T1-kin is a self-action open loop: the lead follows its logged future and the cameras are logged (PREREG section 9).
* **One training run per arm and seed.** The two seeds and the REP pair carry the training-run question; no row here is an effect of the lever on its own (`H-ESTIM-SEED-1`).
* **No lateral / longitudinal split.** The perturbation has two components and neither half is credited.
* **B is not read.** Nothing here says what p = 0.1 does.

## 5. The run history (facts; the notes and logs are in the package)

**Launches and stops, from the box-2 chain logs (first to last timestamped line, minute resolution).**

| launch | log | first → last stamp | span | what ended it |
|---|---|---|---|---|
| r1 | `chain_v8f_dagoaccorc.out` | 01:16 → 02:09 | 53 min | four gates PASS, A s0 trained, **K-REACH FAIL at the run record**: perturbation 19,161 / 76,898 = 0.2492, outside 0.3 ± 0.02. Chain stopped (`ZZDAGOACCORC-CONTROL-FAIL`). Cause and fix: Amendment 2 (the swap and the perturbation drew from one random stream) |
| r2, first launch | `…r2.out_r2a_marker_stop` | 03:27 → 03:39 | 12 min | **refused at its first run record**: *"the run record carries no levers.ego_perturb_stream: the separated perturbation stream is not recorded on A"*. A shared aside marker (`r2_aside_done.txt`, written by DAG-OACC's r2 chain into the same runs root) made the aside a no-op, the queue skipped A s0's training because r1's `summary.json` was still there, and r1's checkpoint (`6ea4d6ba…`, the same sha256 as r1's pull) was pulled again. Note: `raw/2026-10-28-dag-orc-r2a/R2A_NOTE.md` |
| r2, second launch | `…r2.out_r2b_queue_edit_stop` | 03:48 → 06:04 | 136 min | A s0 and REP s0 trained and passed their run records; A s1 trained (checkpoint `b245c8e7…`), then was **refused at its run record**: *"summary.json init_ckpt not applied / read back: False / None"*. Per `R2C_NOTE.md`, a self-test run by another agent reached the real box 2 at 05:56:16 and replaced the trainer, the model file and the queue `.cmd` while A s1 was training; cmd.exe resumed the new bytes at the old offset and ran the trainer again (the box-2 train log shows a second `train M2b_s1_dagoaccorc` at 05:59:36). Note: `notes/R2C_NOTE.md` |
| disk floor | `…r2.out_r2c_diskfloor_stop` | no timestamp | 0 min | the first relaunch stopped before any job on the E: floor: pulling 805,306,368 B would leave less than 1,073,741,824 B (free 1,858,076,672 B). Per `R2C_NOTE.md`, two aside dirs and two dead lander holds (414 MB) were moved to F: |
| r2c (final) | `chain_v8f_dagoaccorc_r2.out` | 06:09 → 07:44 | 95 min | **`ZZV8FMINI-DONE`**. Pinned to `8c24823f` through `OAF_BR=agent/dagorc-r2-pin` (`run_chain_v8f_dagoaccorc_r2c_pin.cmd`; per `R2C_NOTE.md` the remote tip had moved to `94a47f4b`, a SPEEDSUB commit with no trainer, model, data or queue change). Four gates PASS; A s0 and REP s0 were skipped by the queue (already trained, jobs ended 04:35 and 05:17 in r2b, checkpoint sha256 re-verified); A s1 and REP s1 trained 06:21–07:00 and 07:03–07:42 |

**Box-2 cost.** Summed over the four launches the logs span **296 min = 4.93 h** (53 + 12 + 136 + 95), against the **~3 h** the PI's go was asked as (Amendment 1 section 4 planned 2.8 h) and against the **about 4.8 h** that `R2C_NOTE.md` states (an estimate that adds up the notes' own figures: `R2A_NOTE.md` 3.9 h after the first r2 launch, plus the 0.2 h and the 0.7 h that `R2C_NOTE.md` adds). The overrun beyond the ~3 h is 1.9 h by the logs. The four kept trainings account for 2.52 h of it (A s0 2286 s, REP s0 2240 s, A s1 2293 s, REP s1 2237 s; `summary.json`). The rest is the refused r1 A s0 training, the r2 first launch, the refused A s1 training and the gates, evals and pulls of each launch.

**Training jobs (log minutes; launch stamp to `ZZJOB-END`).**

| run | launch | start → end | span | kept in the read |
|---|---|---|---|---|
| M2b_s0_dagoaccorc | r1 | 01:27 → 02:06 | 40 min | no: refused, never rolled out |
| M2b_s0_dagoaccorc | r2b | 03:56 → 04:35 | 40 min | yes |
| M2b_s0_dagoaccorcrep | r2b | 04:38 → 05:17 | 39 min | yes |
| M2b_s1_dagoaccorc | r2b | 05:20 → 06:01 | 41 min | no: refused, never rolled out |
| M2b_s0_dagoaccorc | r2c | 06:18 → 06:18 | 1 min | skipped job (checkpoint already trained) |
| M2b_s0_dagoaccorcrep | r2c | 06:20 → 06:20 | 0 min | skipped job (checkpoint already trained) |
| M2b_s1_dagoaccorc | r2c | 06:21 → 07:00 | 40 min | yes |
| M2b_s1_dagoaccorcrep | r2c | 07:03 → 07:42 | 39 min | yes |

**The final four runs** (box-2 chain log `chain_v8f_dagoaccorc_r2.out`, the `summary:` line and the per-run lines; checkpoint sha256 from `SHA256SUMS.txt`, verified by `sha256sum -c`; the T1-kin rollout done-markers carry the same sha256).

> `10-28 07:44 summary: M2b_s0_dagoaccorc=380b230d0603 M2b_s0_dagoaccorcrep=b03fd6efebec M2b_s1_dagoaccorc=870dded74c71 M2b_s1_dagoaccorcrep=77853e564f52` (L119)

| run | trainer seed | argv sha16 | checkpoint sha256 | run record | K-REACH: swap n / eligible | K-REACH: perturbation n / eligible | overlap | stream | gate token (commit, argv sha256 prefix) |
|---|---|---|---|---|---|---|---|---|---|
| M2b_s0_dagoaccorc | 0 | `b2e18d0d8af33e42` | `380b230d0603e21d874cc08ad67400041ddf1d7920af1ab2dab36e7e075fc3df` | PASS (L59) | 14,688 / 145,037 = 0.1013 | 22,963 / 76,898 = 0.2986 | 0 | 1 | `8c24823f`, `0c8cf774f05fe7ee` |
| M2b_s0_dagoaccorcrep | 10 | `7c6d0c0e8f3cf78c` | `b03fd6efebec285f747673fa90d093cb9202a649d560ba8f5c016475f1fb4bca` | PASS (L78) | 14,659 / 144,643 = 0.1013 | 0 / 82,118 = 0.0000 | n/a (flag off) | 0 | `8c24823f`, `4fefcc322cd5dcc1` |
| M2b_s1_dagoaccorc | 1 | `cecf3000297fd4ff` | `870dded74c71b30ab47248e14dc0af881ef417b49f24da28aa7491b5d29ed290` | PASS (L97) | 14,839 / 147,199 = 0.1008 | 22,970 / 76,435 = 0.3005 | 0 | 1 | `8c24823f`, `8792c141ce28bcab` |
| M2b_s1_dagoaccorcrep | 11 | `dee95c3d5666232b` | `77853e564f52ff029cfc241042b733d75dba8191a99a7be0fc59778d511aeca2` | PASS (L116) | 14,749 / 147,639 = 0.0999 | 0 / 81,072 = 0.0000 | n/a (flag off) | 0 | `8c24823f`, `c7341cf55e6db151` |

Run-record and K-REACH lines, verbatim from `chain_v8f_dagoaccorc_r2.out`:

```
DAG-ORC (DAG-OACC base) run record: PASS | dagoaccorc s0 | trainer seed 0 | argv sha16 b2e18d0d8af33e42 | steps 3000 | ego_perturb_p 0.3 | w_acc 49.6969
DAG-ORC (DAG-OACC base) K-REACH: PASS | dagoaccorc | swap 14688/145037 = 0.1013 (must lie in 0.1 +- 0.02) | ego 22963/76898 = 0.2986 (A: 0.3 +- 0.02; REP: exactly 0) | overlap 0 (A: exactly 0) | flag True
DAG-ORC (DAG-OACC base) run record: PASS | dagoaccorcrep s0 | trainer seed 10 | argv sha16 7c6d0c0e8f3cf78c | steps 3000 | ego_perturb_p 0.0 | w_acc 49.6969
DAG-ORC (DAG-OACC base) K-REACH: PASS | dagoaccorcrep | swap 14659/144643 = 0.1013 (must lie in 0.1 +- 0.02) | ego 0/82118 (A: 0.0 +- 0.02; REP: exactly 0) | overlap 0 (A: exactly 0) | flag False
DAG-ORC (DAG-OACC base) run record: PASS | dagoaccorc s1 | trainer seed 1 | argv sha16 cecf3000297fd4ff | steps 3000 | ego_perturb_p 0.3 | w_acc 49.6969
DAG-ORC (DAG-OACC base) K-REACH: PASS | dagoaccorc | swap 14839/147199 = 0.1008 (must lie in 0.1 +- 0.02) | ego 22970/76435 = 0.3005 (A: 0.3 +- 0.02; REP: exactly 0) | overlap 0 (A: exactly 0) | flag True
DAG-ORC (DAG-OACC base) run record: PASS | dagoaccorcrep s1 | trainer seed 11 | argv sha16 dee95c3d5666232b | steps 3000 | ego_perturb_p 0.0 | w_acc 49.6969
DAG-ORC (DAG-OACC base) K-REACH: PASS | dagoaccorcrep | swap 14749/147639 = 0.0999 (must lie in 0.1 +- 0.02) | ego 0/81072 (A: 0.0 +- 0.02; REP: exactly 0) | overlap 0 (A: exactly 0) | flag False
```

The K-REACH bands are 0.3 ± 0.02 for A, exactly 0 for REP, and 0.1 ± 0.02 for the swap; the overlap of swapped and perturbed samples is exactly 0 on A. All four fall inside. The same gate tokens (same commit, same argv sha256) were written in the second launch (r2b) and the final one (r2c).

**K-BUILD** (launch commit `8c24823f` against `aeb54468`, the commit DAG-OACC trained at): 37 changed hunks in 2 files (`v8f_mini_train.py` 25, `v8f_mini_data.py` 12), every one behind `--ego-perturb-with-dagger`, default off, bit-identity test PASS; `v8f_mini.py` and `v8f_dagger.py` byte-identical. Record: `kbuild/kbuild_r2_launch_8c24823f.json` (sha256 `b1c5c36b…`).

## 6. Next (Rule Zero): A is not adopted; the committed next is B, blocked on the PI's go

* **Committed next (the table, verbatim from the JSON):** *"A: not adopted; read B (p 0.1) under this same table (B is NOT queued: it needs its own PI go, A1 section 4)."* `cascade.b_status`: *"PENDING: train B (p = 0.1) through its gate and read it under this same table"*.
* **Blocked on:** the PI's go for B (A1 section 4: *B is not queued*). What unblocks it: the go, then box 2 for two runs through the launch gate. ESTIMATED ≈ 1.4 h of box 2: A's two trainings took 2286 s and 2293 s (`summary.json`), plus about 4 min per run for gate and eval (log minutes). The T1-kin rollouts and the read are dev-box CPU.

**The Master Mind's ranking, beside the table (arithmetic, not a forecast).**

1. **What B must clear.** R-MOVE needs, on every row of C at a seed, an improvement ≥ bar. The bars are defined from RF0 and the REP pair (PREREG section 3), neither of which involves the dose, and the committed next reads B *"under this same table"*; the factors below assume the same bars (PREREG section 7 also says B's claim *"carries B's own floors and seeds"*; if the reader recomputes floors for B, the factors change). A's crawl improvements stayed below RF and below the bar (section 4 (c)); the bar divided by A's improvement is the factor B's improvement would have to exceed A's by:

| row | s0: bar / A's improvement | s1: bar / A's improvement |
|---|---|---|
| yaw-rate MAE | 11.4 × | 5.5 × |
| curvature MAE | 5.9 × | 3.8 × |
| L2 curvature MAE (GT>=5.2) | 6.9 × | 5.6 × |
| M1 crawl share | 3.0 × | 4.7 × |

   B's dose is a third of A's (p = 0.1 against 0.3). **So B cannot reach R-MOVE unless its crawl effect exceeds A's by 3.0–11.4 ×,** at a third of the dose.
2. **Where B's outcomes lead.** Under PREREG section 7 and A1 section 3, HARMS of B, NO EFFECT and PARTIAL / ONE SEED all route to NEXT-CRAWL, which on DAG-OACC's states is LAUNCH-INTENT (N0 read NOT NAVMOVE; COMPOSE is not met, since DAG-OACC − DAG is not separated-better on the rows of C, `RESULT_V8F_DAG_OACC.md` section 3). Only R-MOVE at both seeds routes elsewhere. So B changes the route only if it clears the factors above.
3. **The registered sequel is NEXT-CRAWL → LAUNCH-INTENT** (draft: `dagland/launch_intent/PREREG_V8F_LAUNCH_INTENT.md`, on E:). Its draft reuses this REP pair, and therefore these floors: the bars in section 2.1 are the bars a LAUNCH-INTENT crawl effect has to clear unless its PREREG sets its own, and A's measured effects (section 2.4) are the reference size. Its REP-reuse licence is conditional (its section 4); this read shows the REP run records PASS (swap share 0.1013 / 0.0999, perturbation exactly 0, composition flag absent, stream 0). Its other conditions were not checked here (UNVERIFIED).

**Unblocked now, 0 GPU, each a registered item of this PREREG that did not get built (work items, not done in this package):**

* **M-LAUNCH** (PREREG section 4): A − CTRL of the mean (v_own − v_GT) at ticks 1–2 in the GT-launch windows, from the tick dumps on F: (`F:/TanitAD-scratch/dagorc_t1_out/ticks/`). It is the cheapest discriminating row for the ranking: *"If R-MOVE fails while M-LAUNCH moves, the speed deficit was repaired and the crawl rows did not follow. That outcome points away from the state channel."*
* **The T0 four families** (`orc_families.py` on the pulled `eval_*.npz` of A and CTRL) and the **anti-echo block** and **M2 / L2 on the GT-launch subset** (section 3.3).
* **Single copy.** The per-episode dumps and tick arrays are on F: scratch only (`F:/TanitAD-scratch/dagorc_t1_out/`); the checkpoints and T0 evals are on E:. None is in the repo.

## 7. Provenance (artifact path, sha256)

Repo paths are under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/` and are staged by the hold (`landing/mk_hold_dagorc_result.py`). Source paths are where the bytes were read from.

| role | source path | repo path | sha256 |
|---|---|---|---|
| read JSON (reader output) | `F:/TanitAD-scratch/dagorc_t1_out/read/DAGORC_READ.json` | `raw/2026-10-28-dag-orc-result/read/DAGORC_READ.json` | `70a7a3a46e520311ade0cf2e1c71203010a25185ef5eb82b9d34b988b9548112` |
| replicate floors RF | `F:/TanitAD-scratch/dagorc_t1_out/read/RF_REP.json` | `raw/2026-10-28-dag-orc-result/read/RF_REP.json` | `64c9444d486939be8dbbf61ef67028c35aa56b13c5d4bdbbc71d6e17e6fc497a` |
| T1-kin chain log | `E:/Projects/TanitAD-artifacts/dagorc_t1/logs/chain.out` | `raw/2026-10-28-dag-orc-result/logs/chain.out` | `db5fa5a813818082d61956b937fdd42b17ffbeddd076df5acdd3a06161200a8d` |
| reader log | `E:/Projects/TanitAD-artifacts/dagorc_t1/logs/dagorc_read.out` | `raw/2026-10-28-dag-orc-result/logs/dagorc_read.out` | `5a8679b7db7076acd8df874a437deddb6d2aedafbfb8442a16f666e170c5563b` |
| box-2 chain log, r2c (final; the `summary:` line) | `C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out` | `raw/2026-10-28-dag-orc-result/box2/logs/chain_v8f_dagoaccorc_r2.out` | `d81f2d7470d7984fbff6f0789e2f048bba9169af76d8fa003ee7a098fc26f1b2` |
| box-2 chain log, r1 (K-REACH stop) | `C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc.out` | `raw/2026-10-28-dag-orc-result/box2/logs/chain_v8f_dagoaccorc.out` | `4693eb137355d1fc6c603c485b4ec6906ed834ff86592a2c9b8f5064f8a0cb56` |
| box-2 chain log, r2 first launch (marker stop) | `C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out_r2a_marker_stop` | `raw/2026-10-28-dag-orc-result/box2/logs/chain_v8f_dagoaccorc_r2.out_r2a_marker_stop` | `8fa871b371a2d334cda01fc9c6a09b8099dfa90434716cc9f509446bf207b618` |
| box-2 chain log, r2 second launch (queue-edit stop) | `C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out_r2b_queue_edit_stop` | `raw/2026-10-28-dag-orc-result/box2/logs/chain_v8f_dagoaccorc_r2.out_r2b_queue_edit_stop` | `2ebc94f971f73501193e90715282fa882eb72c63b33eba6af1989870b26d83a3` |
| box-2 chain log, disk-floor stop | `C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out_r2c_diskfloor_stop` | `raw/2026-10-28-dag-orc-result/box2/logs/chain_v8f_dagoaccorc_r2.out_r2c_diskfloor_stop` | `ebd5f0517f35e0db8aa0ba510351b091df8fed8d2329e9668f4081f4460b3539` |
| box-2 train log of the A s1 incident (second `train` at 05:59:36) | `C:/Users/Admin/box2_ship/logs/v8fdagoaccorc10280348_t_M2b_s1_dagoaccorc.log` | `raw/2026-10-28-dag-orc-result/box2/logs/v8fdagoaccorc10280348_t_M2b_s1_dagoaccorc.log` | `9d077655415d6fc5bb3ab8fab78e0c468f2f29ddfda2bbe3b909e50d50236e7f` |
| pin launcher of the final launch | `C:/Users/Admin/box2_ship/run_chain_v8f_dagoaccorc_r2c_pin.cmd` | `raw/2026-10-28-dag-orc-result/box2/run_chain_v8f_dagoaccorc_r2c_pin.cmd` | `75fa3d9370a19cc48b68e91f24e1ccd24125b52777bea6d3885990cac6e363a4` |
| note: A s1 refused (self-test replaced the queue) | `E:/Projects/TanitAD-artifacts/dagland/dagorc_r2a/R2C_NOTE.md` | `raw/2026-10-28-dag-orc-result/notes/R2C_NOTE.md` | `df02ff2f8b46ab05f45d437bd32d6a69596ea49b07d968f731c8b72cc9a98fac` |
| note: r2 first launch refused (shared aside marker) | `E:/Projects/TanitAD-artifacts/dagland/dagorc_r2a/R2A_NOTE.md` | `raw/2026-10-28-dag-orc-r2a/R2A_NOTE.md (already in the repo)` | `825fa0856b91b04937cc906b1cae70b443b1615ef8c07c4ceebf7ebc116823c3` |
| K-BUILD record, launch commit `8c24823f` | `E:/Projects/TanitAD-artifacts/dagorc_t1/kbuild_r2_launch_8c24823f.json` | `raw/2026-10-28-dag-orc-result/kbuild/kbuild_r2_launch_8c24823f.json` | `b1c5c36b6c3cb483f7d735546ac5f104fedcc6e8cf96df499da06a6f5fba1d65` |
| frozen RF0 (P-0) | `E:/Projects/TanitAD-artifacts/dagorc_rebase/read/P0_RF0.json` | `raw/2026-10-28-dag-orc-build/rebase/read/P0_RF0.json (already in the repo)` | `1608d70b21b6b6a45136a6d96828f70b5bbabd9c0ceb84db136927593c79ad0f` |
| reader | `E:/Projects/TanitAD-artifacts/dagorc_rebase/read/dag_orc_read.py` | `raw/2026-10-28-dag-orc-build/rebase/read/dag_orc_read.py (already in the repo)` | `2e5f9f1778e104615d4480f51c6246703c9e3bb8f6c0eeaa3fc0c5621bd8adcf` |
| run record M2b_s0_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorc/summary.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorc/summary.json` | `5f6ddd9c43c10280f640562c9cf33ba047641cf594e4f14d0da9bacfdb736dc9` |
| run record M2b_s0_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorc/selfcheck.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorc/selfcheck.json` | `2107b9bd68d272c23a527820ceabc4a12a0c87231f5ec8a20d5259c2410dbcd8` |
| run record M2b_s0_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorc/argv.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorc/argv.json` | `c2f495baf388604059be7cbdc2144e46ef13768bbbbf3faeea7359a5bf1be38b` |
| run record M2b_s0_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorc/launch_gate_token.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorc/launch_gate_token.json` | `3070de22d62544274c9d3ca1bc6a9f378c92a1237651928737d3780baa9b766d` |
| run record M2b_s0_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorc/SHA256SUMS.txt` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorc/SHA256SUMS.txt` | `9cd44139793e3ac2b6dece4344256ad9b2f7e6ec9fd7108ac516db5dae16187c` |
| checkpoint M2b_s0_dagoaccorc (not staged) | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorc/ckpt_last.pt` | none (E: only) | `380b230d0603e21d874cc08ad67400041ddf1d7920af1ab2dab36e7e075fc3df` |
| run record M2b_s0_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/summary.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorcrep/summary.json` | `eb091bfd3c0471f4f14386917d0062a23434abd9cb2ef59b44a0bf2fc057a661` |
| run record M2b_s0_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/selfcheck.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorcrep/selfcheck.json` | `b3ec93d7ba8e6947066acd3f0256eb40c62dc9aa64cc319d149832f793bb37c5` |
| run record M2b_s0_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/argv.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorcrep/argv.json` | `6a0e5bdf1dde12b995d7f1ef0faba51f63d268934b8b5c0ff4d43ea33a449fcf` |
| run record M2b_s0_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/launch_gate_token.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorcrep/launch_gate_token.json` | `711c4f7f1ed6d6757143f5ac2aff4ef9562f7895ecf093e16e52c4f2071e03ea` |
| run record M2b_s0_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/SHA256SUMS.txt` | `raw/2026-10-28-dag-orc-result/runs/M2b_s0_dagoaccorcrep/SHA256SUMS.txt` | `8e1b3a7c251fe3354cd8d9eb6caa29448b3baf00ce5404dfa8dacf1b7ae9c895` |
| checkpoint M2b_s0_dagoaccorcrep (not staged) | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/ckpt_last.pt` | none (E: only) | `b03fd6efebec285f747673fa90d093cb9202a649d560ba8f5c016475f1fb4bca` |
| run record M2b_s1_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorc/summary.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorc/summary.json` | `155ba1c93ff958df5c9659a7391685ce2c32503559d859f23c23a1d964117f11` |
| run record M2b_s1_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorc/selfcheck.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorc/selfcheck.json` | `246179d276295323baaada6e82cd22586a4e8d1e4e4693f4f4ac1e6fbb042f39` |
| run record M2b_s1_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorc/argv.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorc/argv.json` | `446958d6fd6571e0804385af5926020abeb2656b7532e83193a75183311ddc87` |
| run record M2b_s1_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorc/launch_gate_token.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorc/launch_gate_token.json` | `b9db4917f0bddc24817bd91d35195e471f73942c8d4fa4ca0cccdf9e27d8e466` |
| run record M2b_s1_dagoaccorc | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorc/SHA256SUMS.txt` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorc/SHA256SUMS.txt` | `10b8de9eaf61aca4fa4f8a98448e7bfab1eec9992c1507b78df07f2331ad38e3` |
| checkpoint M2b_s1_dagoaccorc (not staged) | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorc/ckpt_last.pt` | none (E: only) | `870dded74c71b30ab47248e14dc0af881ef417b49f24da28aa7491b5d29ed290` |
| run record M2b_s1_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/summary.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorcrep/summary.json` | `d2309e2eeb1dcd43920168d482721873d33e7326e2abfa2062a9b7afee7eac43` |
| run record M2b_s1_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/selfcheck.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorcrep/selfcheck.json` | `5692bc29de385efb10441fb06837ac448a8c2b99f9b234a1b884c5b9a6ab6d5f` |
| run record M2b_s1_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/argv.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorcrep/argv.json` | `ac26f32fa0e51ca27a6f86a0dba7bd0eaca57b1b4cd5f460b2d9d740fe817fb9` |
| run record M2b_s1_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/launch_gate_token.json` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorcrep/launch_gate_token.json` | `31114c4a947d21f6f264aeeaea4b2c2ea18e7c88cc2cf433799653eaee6122de` |
| run record M2b_s1_dagoaccorcrep | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/SHA256SUMS.txt` | `raw/2026-10-28-dag-orc-result/runs/M2b_s1_dagoaccorcrep/SHA256SUMS.txt` | `d61840793a8236a14d778ee3e956b5e41abd2417f378277de69d2aae3ee47b9d` |
| checkpoint M2b_s1_dagoaccorcrep (not staged) | `E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/ckpt_last.pt` | none (E: only) | `77853e564f52ff029cfc241042b733d75dba8191a99a7be0fc59778d511aeca2` |

The reader's pins (`provenance.ref_pins`), the PREREG sha256 it read (`38bc688b…`, the file at `3b7b404e`) and Amendment 1's (`f2f96157…`) are in the JSON under `provenance`; Amendment 2 changes the build and the re-run, not a row, a bar or a floor (its section 2). The PREREG read for this document is the file at `4c412181` (sha256 `d8b8885b…`, Amendments 1 and 2 included).
