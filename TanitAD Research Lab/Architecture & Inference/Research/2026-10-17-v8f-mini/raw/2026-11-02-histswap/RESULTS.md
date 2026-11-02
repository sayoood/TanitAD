# RESULTS: H-HISTSWAP-1, does O follow its own slowed ego history? (zero training, ORACLE diagnostic), DAG-OACC and SOUP at s0 / s1

Card `PREREG_HISTSWAP_FROZEN.md` sha256 `7e91a7c742a82365bc73d3709cfd789ffb514eff7f27b063d9c80b339412278b` (frozen 2026-11-02T17:57:30+0100). Reader `code/histswap_t1_read.py` (rule `decide`, tested in `tests/test_histswap_read_rule.py`). Read clock 2026-11-02T18:31:45+0100. Raw `raw/HISTSWAP_T1.json`, per-window arrays `raw/arrays/*.npz`. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class **MEASURED (ours; artifacts in MANIFEST.txt)**.
Estimator of every interval below: **paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only; X and Y share the checkpoint, so no training-seed variance is in X - Y**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. X and Y use the same checkpoint, the planner is deterministic, and the two seeds are the rig's replicates, not a variance estimate (`H-ESTIM-SEED-1` not addressed).

## 0. The committed reading: H SUPPORTED, and X is an ORACLE

**MEASURED.** The card's rule reads **SUPPORTED**: M1 crawl share X - Y is separated-better at **4 of 4 cells** (both seeds on both bases) and the mean share of Y - P10's M1 excess closed is **0.939** (bar 0.5; pooled over the four cells 0.937). H (a HYPOTHESIS before this read): O follows its own slowed ego history.

> **ORACLE CAVEAT (the card's, binding on every sentence of this file).** Arm X reads the LOGGED ego history at every tick; the logged ego history after tick 0 carries the human's speed at that tick. X is therefore an ORACLE and is **not deployable**. **SUPPORTED shows that the executed-speed drift enters through the ego-history channel (the network's ego-history input), and does NOT show a deployable fix.** X's absolute rows are NOT performance and are never quoted as such; only X - Y on M1 and the row structure are read. X also leaks the human's realised motion at ticks 1 and 2, so a positive reading cannot tell *the executed lag is the cause* from *any human-motion history fixes it*; a negative reading would not be subject to that leak. **A leak-free arm is the next step** (being built in `../histswap_lf/`; its result is not part of this file).

| cell | M1 X - Y (X = oracle history) | Y - P10 (the excess) | share of the excess closed | ADE6 X - Y (m; oracle, not performance) |
|---|---|---|---|---|
| DAG-OACC s0 | -0.0476 [-0.0664, -0.0315] SB | +0.0483 [+0.0302, +0.0698] SW | 0.9844 (98.4 %) | -1.2325 [-1.3530, -1.1168] SB |
| DAG-OACC s1 | -0.0523 [-0.0739, -0.0338] SB | +0.0565 [+0.0347, +0.0829] SW | 0.9247 (92.5 %) | -1.2352 [-1.3687, -1.1032] SB |
| SOUP s0 | -0.0612 [-0.0832, -0.0408] SB | +0.0666 [+0.0428, +0.0927] SW | 0.9178 (91.8 %) | -1.2971 [-1.4285, -1.1693] SB |
| SOUP s1 | -0.0537 [-0.0754, -0.0355] SB | +0.0576 [+0.0364, +0.0823] SW | 0.9309 (93.1 %) | -1.2806 [-1.4169, -1.1517] SB |

Rule (card section 2): `better` = the M1 row of X - Y is separated and delta < 0; `share closed` = -(X - Y) / (Y - P10) on the M1 row; SUPPORTED = 4 of 4 better and mean share closed >= 0.5; NOT SUPPORTED = fewer than 3 of 4 better; MIXED otherwise.

## 1. Controls (computed before any row; any failure refuses the read)

* **KY_reused_Y_equals_frozen_pins**: PASS
* **KA1_fresh_off_rerun_equals_reused_Y**: PASS
* **KS0_tick0_equals_Y_tick0**: PASS
* **KS1_tick1_executed_state_equals_Y**: PASS
* **KS2_v0_in_is_the_logged_speed_and_differs_after_tick0**: PASS
* **KF_floor_window_check_vs_fresh_Y**: PASS
* **K3_Y_banked_minus_P10_reproduces_SOUP_READ**: PASS
* **KB1_P10_side_equals_landed_banked_b**: PASS

**KY**: 8 / 8 reused Y files hash-equal to the pins of `FROZEN_SHA.txt`. **KA1** (the fresh `--histswap off` rerun `SREP4_DAGOACC_s0` equals the reused `SREP_DAGOACC_s0`): dump bit-identical True (max abs dev 0.000e+00 m), ticks bit-identical True (max abs dev 0.000e+00), differing keys none, key-set difference none.

**KS0** (tick 0 of every X equals Y's tick 0, tolerance 1e-4): max abs deviation over tau / P / plan / plan_raw / v0 / pose / gt = 0.0e+00 on 2399/2399/2399/2399 tick-0 rows per cell. **KS1** (tick 1's executed pose and v0 equal Y's): max abs deviation 0.0e+00. **KS2** (v0_in is the logged speed):

| cell | v0_in = v0 at tick 0 (max abs) | share of later ticks with v0_in != v0 by > 0.05 m/s | mean abs v0_in - v0, later ticks (m/s) | mean executed v0 / logged v0_in, later ticks (m/s) |
|---|---|---|---|---|
| CTRL_s0 | 0.0e+00 | 0.9193 | 0.5586 | 11.155 / 11.297 |
| CTRL_s1 | 0.0e+00 | 0.9202 | 0.5863 | 11.158 / 11.297 |
| SOUP_s0 | 0.0e+00 | 0.9271 | 0.5976 | 11.025 / 11.297 |
| SOUP_s1 | 0.0e+00 | 0.9193 | 0.5751 | 11.156 / 11.297 |

**KF**: windows X = fresh Y at 4 / 4 cells, floor X vs Y max abs 0.0e+00 m. **K3**: Y banked - P10 reproduces `SOUP_READ.json` on 76 / 76 rows. **KB1**: the P10 side of every extracted X and Y equals the landed banked `b` arrays: 8 / 8 runs pass. Tool tests `logs/pytest_histswap.log` (8 passed, 3 mutations RED) and `logs/pytest_read_rule.log` (6 passed); 32-window smoke `logs/smoke_histswap.out`: run when the tool was built; not re-run in this read.

## 2. Read A for X (vs P10), X - Y and Y - P10, all 19 rows, per cell (X = ORACLE; absolute rows are not performance)

Every cell is `point [lo, hi] mark`. `read_a` and `read_xy` are the FEAS-family reads applied to X; because X reads the human's history, **its `DOMINANT` flag is an oracle artefact and is never a claim about a deployable arm**. S-POS rows of X mix the logged poses the S head read with the executed pose the extractor frames them in: **UNVERIFIED**, outside the reading.

### DAG-OACC s0

* **read_a (X vs P10)**: D1 True, D2 True, D3 True ({'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}), DOMINANT-flag True (oracle artefact); separated-worse rows: none; separated-better rows: 16 of 19.
* **read_xy (X - Y)**: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}; ADE6 X - Y -1.2325 (within guard: True); non-crawl rows separated-worse: none; rows separated-worse on X - Y, all 19: 0; MOVED-seed True, HARMS-seed False.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -2.4881 [-2.7105, -2.2753] SB | -1.2325 [-1.3530, -1.1168] SB | -1.2556 [-1.4503, -1.0678] SB |
| speed MAE | LONGITUDINAL | -1.6157 [-1.8650, -1.3955] SB | -0.7827 [-0.8872, -0.6907] SB | -0.8330 [-1.0377, -0.6583] SB |
| along-track MAE | LONGITUDINAL | -1.8786 [-2.0466, -1.7122] SB | -1.1593 [-1.2692, -1.0495] SB | -0.7192 [-0.8806, -0.5627] SB |
| accel MAE | LONGITUDINAL | -1.3357 [-1.8358, -0.9180] SB | -0.3966 [-0.5710, -0.2543] SB | -0.9390 [-1.3078, -0.6221] SB |
| DK-VIOL | LONGITUDINAL | -0.0168 [-0.0462, +0.0085] ns | -0.0336 [-0.0790, +0.0018] ns | +0.0168 [-0.0157, +0.0625] ns |
| DK-HW | LONGITUDINAL | -1.0200 [-1.5366, -0.5729] SB | -0.7328 [-1.1950, -0.3065] SB | -0.2604 [-0.6426, +0.0675] ns |
| DK-TG | LONGITUDINAL | -0.1635 [-0.2633, -0.0752] SB | -0.1251 [-0.2151, -0.0544] SB | -0.0588 [-0.1226, -0.0011] SB |
| heading MAE | LATERAL | -4.0950 [-4.8319, -3.3543] SB | -0.8649 [-1.4667, -0.2077] SB | -3.4520 [-4.2140, -2.7529] SB |
| yaw-rate MAE | LATERAL | -10.8131 [-15.4242, -6.9020] SB | -9.6674 [-13.2778, -6.7383] SB | -1.3762 [-3.9121, +1.0259] ns |
| curvature MAE | LATERAL | +0.0062 [-0.0047, +0.0174] ns | -0.0510 [-0.0765, -0.0299] SB | +0.0578 [+0.0364, +0.0821] SW |
| cross-track MAE | LATERAL | -1.3135 [-1.4785, -1.1557] SB | -0.3126 [-0.3809, -0.2532] SB | -1.0009 [-1.1468, -0.8671] SB |
| L1 lateral-accel MAE | LATERAL | -3.4203 [-4.7970, -2.1593] SB | -0.9606 [-1.3978, -0.6062] SB | -2.7327 [-4.0998, -1.5826] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0078 [-0.0113, -0.0048] SB | -0.0403 [-0.0677, -0.0207] SB | +0.0325 [+0.0135, +0.0590] SW |
| TAC-LAT | TACTICAL | +0.2730 [+0.2347, +0.3139] SB | +0.0692 [+0.0429, +0.0955] SB | +0.2038 [+0.1639, +0.2450] SB |
| TAC-LON | TACTICAL | +0.4756 [+0.4215, +0.5285] SB | +0.3005 [+0.2637, +0.3407] SB | +0.1751 [+0.1138, +0.2389] SB |
| S-POS@30 | STRATEGIC | -2.2898 [-2.9797, -1.6939] SB | -1.9915 [-2.3898, -1.6325] SB | -1.6443 [-2.1654, -1.1864] SB |
| S-POS@50 | STRATEGIC | -5.6316 [-7.1231, -4.3042] SB | -2.8514 [-3.5820, -2.2156] SB | -4.5745 [-5.8089, -3.5098] SB |
| S-POS@80 | STRATEGIC | -11.3070 [-14.0322, -8.7986] SB | -3.7918 [-5.0576, -2.6931] SB | -9.8284 [-12.1173, -7.7691] SB |
| M1 crawl share | crawl | +0.0017 [-0.0046, +0.0084] ns | -0.0476 [-0.0664, -0.0315] SB | +0.0483 [+0.0302, +0.0698] SW |

### DAG-OACC s1

* **read_a (X vs P10)**: D1 True, D2 True, D3 True ({'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}), DOMINANT-flag True (oracle artefact); separated-worse rows: none; separated-better rows: 16 of 19.
* **read_xy (X - Y)**: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}; ADE6 X - Y -1.2352 (within guard: True); non-crawl rows separated-worse: none; rows separated-worse on X - Y, all 19: 0; MOVED-seed True, HARMS-seed False.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -2.4255 [-2.6496, -2.2102] SB | -1.2352 [-1.3687, -1.1032] SB | -1.1903 [-1.3986, -0.9782] SB |
| speed MAE | LONGITUDINAL | -1.6086 [-1.8582, -1.3872] SB | -0.7945 [-0.8898, -0.7089] SB | -0.8141 [-1.0377, -0.6191] SB |
| along-track MAE | LONGITUDINAL | -1.8429 [-2.0111, -1.6805] SB | -1.2127 [-1.3390, -1.0894] SB | -0.6302 [-0.8196, -0.4510] SB |
| accel MAE | LONGITUDINAL | -1.3118 [-1.8160, -0.8961] SB | -0.3398 [-0.4749, -0.2213] SB | -0.9719 [-1.3431, -0.6572] SB |
| DK-VIOL | LONGITUDINAL | -0.0202 [-0.0501, +0.0050] ns | -0.0202 [-0.0445, +0.0017] ns | +0.0000 [-0.0196, +0.0212] ns |
| DK-HW | LONGITUDINAL | -1.1213 [-1.6148, -0.6988] SB | -0.6134 [-1.0556, -0.2155] SB | -0.5171 [-0.9057, -0.2118] SB |
| DK-TG | LONGITUDINAL | -0.1978 [-0.2856, -0.1248] SB | -0.1100 [-0.2097, -0.0318] SB | -0.0869 [-0.1542, -0.0278] SB |
| heading MAE | LATERAL | -3.5015 [-4.3606, -2.5758] SB | -0.7822 [-1.5407, +0.0641] ns | -2.8936 [-3.6062, -2.1968] SB |
| yaw-rate MAE | LATERAL | -10.8294 [-15.4283, -6.8709] SB | -15.2357 [-21.5652, -9.8799] SB | +4.1412 [+1.1665, +7.3821] SW |
| curvature MAE | LATERAL | +0.0063 [-0.0042, +0.0173] ns | -0.1134 [-0.1663, -0.0694] SB | +0.1212 [+0.0757, +0.1729] SW |
| cross-track MAE | LATERAL | -1.2705 [-1.4353, -1.1163] SB | -0.2487 [-0.3260, -0.1793] SB | -1.0217 [-1.1686, -0.8803] SB |
| L1 lateral-accel MAE | LATERAL | -3.4133 [-4.7890, -2.1462] SB | -1.1046 [-1.6429, -0.6717] SB | -2.6082 [-3.8410, -1.5151] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0077 [-0.0112, -0.0047] SB | -0.0583 [-0.0882, -0.0342] SB | +0.0505 [+0.0269, +0.0791] SW |
| TAC-LAT | TACTICAL | +0.2684 [+0.2319, +0.3067] SB | +0.0559 [+0.0333, +0.0808] SB | +0.2126 [+0.1709, +0.2540] SB |
| TAC-LON | TACTICAL | +0.4744 [+0.4207, +0.5273] SB | +0.3105 [+0.2761, +0.3485] SB | +0.1638 [+0.1021, +0.2225] SB |
| S-POS@30 | STRATEGIC | -2.3491 [-2.9782, -1.7959] SB | -2.1057 [-2.6247, -1.6598] SB | -1.7416 [-2.2209, -1.3273] SB |
| S-POS@50 | STRATEGIC | -5.9298 [-7.4428, -4.6011] SB | -2.9878 [-3.8632, -2.2268] SB | -4.7600 [-5.8592, -3.7484] SB |
| S-POS@80 | STRATEGIC | -12.3672 [-15.2029, -9.7902] SB | -4.1693 [-5.5832, -2.9536] SB | -10.2999 [-12.3713, -8.3307] SB |
| M1 crawl share | crawl | +0.0043 [-0.0025, +0.0117] ns | -0.0523 [-0.0739, -0.0338] SB | +0.0565 [+0.0347, +0.0829] SW |

### SOUP s0

* **read_a (X vs P10)**: D1 True, D2 True, D3 True ({'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}), DOMINANT-flag True (oracle artefact); separated-worse rows: none; separated-better rows: 16 of 19.
* **read_xy (X - Y)**: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}; ADE6 X - Y -1.2971 (within guard: True); non-crawl rows separated-worse: none; rows separated-worse on X - Y, all 19: 0; MOVED-seed True, HARMS-seed False.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -2.4899 [-2.7166, -2.2733] SB | -1.2971 [-1.4285, -1.1693] SB | -1.1928 [-1.4078, -0.9880] SB |
| speed MAE | LONGITUDINAL | -1.6068 [-1.8557, -1.3868] SB | -0.8252 [-0.9206, -0.7383] SB | -0.7817 [-0.9919, -0.5956] SB |
| along-track MAE | LONGITUDINAL | -1.8698 [-2.0390, -1.7036] SB | -1.2407 [-1.3586, -1.1263] SB | -0.6291 [-0.8126, -0.4582] SB |
| accel MAE | LONGITUDINAL | -1.2797 [-1.7802, -0.8610] SB | -0.4005 [-0.5517, -0.2786] SB | -0.8791 [-1.2581, -0.5583] SB |
| DK-VIOL | LONGITUDINAL | -0.0134 [-0.0442, +0.0144] ns | -0.0202 [-0.0453, +0.0038] ns | +0.0067 [-0.0196, +0.0346] ns |
| DK-HW | LONGITUDINAL | -1.0182 [-1.5469, -0.5568] SB | -0.7026 [-1.1180, -0.2970] SB | -0.2040 [-0.5885, +0.1367] ns |
| DK-TG | LONGITUDINAL | -0.1738 [-0.2694, -0.0873] SB | -0.1144 [-0.2093, -0.0307] SB | -0.0560 [-0.1173, -0.0018] SB |
| heading MAE | LATERAL | -4.0466 [-4.8089, -3.2560] SB | -1.3639 [-2.0751, -0.6298] SB | -2.9614 [-3.7496, -2.1787] SB |
| yaw-rate MAE | LATERAL | -10.6882 [-15.2875, -6.7785] SB | -12.9514 [-17.6919, -9.1174] SB | +1.9270 [-1.0592, +5.0043] ns |
| curvature MAE | LATERAL | +0.0088 [-0.0027, +0.0211] ns | -0.0826 [-0.1216, -0.0484] SB | +0.0919 [+0.0584, +0.1301] SW |
| cross-track MAE | LATERAL | -1.3217 [-1.4851, -1.1657] SB | -0.2951 [-0.3644, -0.2348] SB | -1.0265 [-1.1729, -0.8943] SB |
| L1 lateral-accel MAE | LATERAL | -3.4178 [-4.7943, -2.1563] SB | -1.0738 [-1.5292, -0.7108] SB | -2.6420 [-4.0010, -1.5149] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0076 [-0.0110, -0.0046] SB | -0.0704 [-0.1158, -0.0355] SB | +0.0628 [+0.0279, +0.1076] SW |
| TAC-LAT | TACTICAL | +0.2764 [+0.2390, +0.3150] SB | +0.0917 [+0.0626, +0.1214] SB | +0.1847 [+0.1417, +0.2274] SB |
| TAC-LON | TACTICAL | +0.4585 [+0.4027, +0.5133] SB | +0.3118 [+0.2728, +0.3503] SB | +0.1467 [+0.0792, +0.2138] SB |
| S-POS@30 | STRATEGIC | -2.2665 [-2.9408, -1.6827] SB | -2.0847 [-2.4664, -1.7198] SB | -1.6062 [-2.1525, -1.1336] SB |
| S-POS@50 | STRATEGIC | -5.7663 [-7.3157, -4.3849] SB | -3.0017 [-3.7207, -2.3631] SB | -4.5825 [-5.8449, -3.4774] SB |
| S-POS@80 | STRATEGIC | -11.7889 [-14.6915, -9.1854] SB | -4.0503 [-5.3460, -2.9250] SB | -10.1918 [-12.6081, -8.0751] SB |
| M1 crawl share | crawl | +0.0055 [-0.0017, +0.0134] ns | -0.0612 [-0.0832, -0.0408] SB | +0.0666 [+0.0428, +0.0927] SW |

### SOUP s1

* **read_a (X vs P10)**: D1 True, D2 True, D3 True ({'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}), DOMINANT-flag True (oracle artefact); separated-worse rows: none; separated-better rows: 16 of 19.
* **read_xy (X - Y)**: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}; ADE6 X - Y -1.2806 (within guard: True); non-crawl rows separated-worse: none; rows separated-worse on X - Y, all 19: 0; MOVED-seed True, HARMS-seed False.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -2.4701 [-2.6926, -2.2590] SB | -1.2806 [-1.4169, -1.1517] SB | -1.1895 [-1.3938, -0.9847] SB |
| speed MAE | LONGITUDINAL | -1.6147 [-1.8637, -1.3937] SB | -0.8410 [-0.9418, -0.7457] SB | -0.7737 [-0.9864, -0.5882] SB |
| along-track MAE | LONGITUDINAL | -1.8616 [-2.0299, -1.6990] SB | -1.2356 [-1.3583, -1.1151] SB | -0.6260 [-0.8049, -0.4497] SB |
| accel MAE | LONGITUDINAL | -1.3264 [-1.8280, -0.9103] SB | -0.5299 [-0.7340, -0.3528] SB | -0.7964 [-1.1397, -0.4913] SB |
| DK-VIOL | LONGITUDINAL | -0.0151 [-0.0456, +0.0108] ns | -0.0084 [-0.0275, +0.0109] ns | -0.0067 [-0.0265, +0.0122] ns |
| DK-HW | LONGITUDINAL | -1.1507 [-1.6671, -0.7431] SB | -0.5955 [-1.0189, -0.2079] SB | -0.5481 [-0.9299, -0.2218] SB |
| DK-TG | LONGITUDINAL | -0.1890 [-0.2788, -0.1167] SB | -0.1039 [-0.1921, -0.0322] SB | -0.0863 [-0.1467, -0.0356] SB |
| heading MAE | LATERAL | -3.7276 [-4.5932, -2.8141] SB | -0.9743 [-1.7057, -0.1606] SB | -2.9491 [-3.6329, -2.3131] SB |
| yaw-rate MAE | LATERAL | -10.8539 [-15.4083, -6.8988] SB | -14.1474 [-19.8526, -9.3302] SB | +3.0944 [+0.2216, +6.2120] SW |
| curvature MAE | LATERAL | +0.0062 [-0.0039, +0.0164] ns | -0.0965 [-0.1425, -0.0575] SB | +0.1067 [+0.0666, +0.1533] SW |
| cross-track MAE | LATERAL | -1.3064 [-1.4660, -1.1546] SB | -0.2875 [-0.3639, -0.2222] SB | -1.0189 [-1.1607, -0.8829] SB |
| L1 lateral-accel MAE | LATERAL | -3.4189 [-4.7951, -2.1554] SB | -1.1425 [-1.6175, -0.7273] SB | -2.5744 [-3.8473, -1.4601] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0078 [-0.0112, -0.0047] SB | -0.0522 [-0.0810, -0.0286] SB | +0.0454 [+0.0224, +0.0739] SW |
| TAC-LAT | TACTICAL | +0.2764 [+0.2392, +0.3147] SB | +0.0625 [+0.0375, +0.0891] SB | +0.2138 [+0.1747, +0.2548] SB |
| TAC-LON | TACTICAL | +0.4773 [+0.4237, +0.5303] SB | +0.3189 [+0.2822, +0.3554] SB | +0.1584 [+0.0975, +0.2172] SB |
| S-POS@30 | STRATEGIC | -2.3800 [-3.0318, -1.7999] SB | -2.0053 [-2.5012, -1.5743] SB | -1.7972 [-2.2609, -1.3948] SB |
| S-POS@50 | STRATEGIC | -6.0173 [-7.5628, -4.6746] SB | -2.8720 [-3.7009, -2.1406] SB | -5.0252 [-6.1549, -3.9713] SB |
| S-POS@80 | STRATEGIC | -12.5196 [-15.3612, -9.9728] SB | -3.9833 [-5.2961, -2.8454] SB | -10.7619 [-12.9488, -8.6791] SB |
| M1 crawl share | crawl | +0.0035 [-0.0023, +0.0101] ns | -0.0537 [-0.0754, -0.0355] SB | +0.0576 [+0.0364, +0.0823] SW |

## 3. The M1 row by situation class (descriptive, not a decision; classes L launch / S stay / O other-stopped / M moving; weight `w` = share of the row's windows)

| cell | contrast | row delta | L: n, delta, contribution | M: n, delta, contribution | S, O windows | L share of the row's delta |
|---|---|---|---|---|---|---|
| DAG-OACC s0 | X - Y | -0.0476 [-0.0664, -0.0315] | 253, -0.2677 [-0.3474, -0.1905], -0.0322 | 1850, -0.0175 [-0.0286, -0.0088], -0.0154 | 0, 0 | 68 % |
| DAG-OACC s0 | Y - P10 | +0.0483 [+0.0302, +0.0698] | 254, +0.2614 [+0.1650, +0.3607], +0.0316 | 1850, +0.0191 [+0.0097, +0.0310], +0.0168 | 0, 0 | 65 % |
| DAG-OACC s0 | X - P10 | +0.0017 [-0.0046, +0.0084] | 254, -0.0013 [-0.0436, +0.0398], -0.0002 | 1851, +0.0021 [-0.0020, +0.0063], +0.0019 | 0, 0 | n/a (row delta near 0) |
| DAG-OACC s1 | X - Y | -0.0523 [-0.0739, -0.0338] | 254, -0.2596 [-0.3469, -0.1695], -0.0313 | 1851, -0.0238 [-0.0373, -0.0137], -0.0209 | 0, 0 | 60 % |
| DAG-OACC s1 | Y - P10 | +0.0565 [+0.0347, +0.0829] | 255, +0.2717 [+0.1600, +0.3902], +0.0329 | 1851, +0.0269 [+0.0154, +0.0424], +0.0236 | 0, 0 | 58 % |
| DAG-OACC s1 | X - P10 | +0.0043 [-0.0025, +0.0117] | 254, +0.0133 [-0.0341, +0.0580], +0.0016 | 1851, +0.0030 [-0.0010, +0.0076], +0.0027 | 0, 0 | n/a (row delta near 0) |
| SOUP s0 | X - Y | -0.0612 [-0.0832, -0.0408] | 253, -0.3035 [-0.3961, -0.2121], -0.0365 | 1851, -0.0280 [-0.0420, -0.0167], -0.0247 | 0, 0 | 60 % |
| SOUP s0 | Y - P10 | +0.0666 [+0.0428, +0.0927] | 254, +0.3152 [+0.2023, +0.4337], +0.0380 | 1851, +0.0325 [+0.0201, +0.0478], +0.0286 | 0, 0 | 57 % |
| SOUP s0 | X - P10 | +0.0055 [-0.0017, +0.0134] | 254, +0.0129 [-0.0339, +0.0590], +0.0016 | 1851, +0.0045 [+0.0001, +0.0093], +0.0040 | 0, 0 | n/a (row delta near 0) |
| SOUP s1 | X - Y | -0.0537 [-0.0754, -0.0355] | 254, -0.2594 [-0.3452, -0.1742], -0.0313 | 1851, -0.0254 [-0.0387, -0.0152], -0.0224 | 0, 0 | 58 % |
| SOUP s1 | Y - P10 | +0.0576 [+0.0364, +0.0823] | 255, +0.2691 [+0.1620, +0.3842], +0.0326 | 1851, +0.0285 [+0.0169, +0.0427], +0.0251 | 0, 0 | 57 % |
| SOUP s1 | X - P10 | +0.0035 [-0.0023, +0.0101] | 254, +0.0068 [-0.0383, +0.0502], +0.0008 | 1851, +0.0031 [-0.0004, +0.0070], +0.0027 | 0, 0 | n/a (row delta near 0) |

The S and O classes carry no windows in the masked M1 pool at any cell. Launch-class windows are 12.1 %-12.1 % of the M1 row's windows (weight `w`) and carry 57 %-65 % of the Y - P10 excess and 58 %-68 % of the X - Y movement (contribution / row delta).

## 4. Swap bookkeeping (`swap_bucket_differs / swap_ticks`, from each run's `t1v8f_HSW_*.json`)

| cell | swap_ticks | swap_bucket_differs | share |
|---|---|---|---|
| DAG-OACC s0 | 7197 | 121 | 1.68 % |
| DAG-OACC s1 | 7197 | 118 | 1.64 % |
| SOUP s0 | 7197 | 118 | 1.64 % |
| SOUP s1 | 7197 | 116 | 1.61 % |

The swap includes the causal speed bucket (a function of the ego speed history); its separate contribution to M1 is not attributed here (card section 6).

## 5. The prediction next to the result

Card (ESTIMATED, written before any HISTSWAP number): P(SUPPORTED) 0.60, P(MIXED) 0.25, P(NOT SUPPORTED) 0.15. Read: **SUPPORTED** (the card gave it 0.60). Basis of the estimate: the oracle history carries the human's realised speeds, so M1 was expected to close largely through the swap whatever the mechanism. The card also expected, not committed, that X's ADE6 improves vs Y by leak: measured ADE6 X - Y -1.2325, -1.2352, -1.2971, -1.2806 m at the four cells (CTRL s0, s1, SOUP s0, s1), all separated-better and not performance.

## 6. Ceiling pre-check (card section 1; SEEN Y data, `raw/PRECHECK_CEILING.json`) beside the read

Slow GT-fast pairs of Y minus those of P10, by executed-step block [0-10) / [10-20) / [20-30): DAGOACC_s0 -25 / +583 / +1158; DAGOACC_s1 +6 / +765 / +1414; SOUP_s0 +24 / +934 / +1579; SOUP_s1 -6 / +716 / +1497. The excess starts after tick 0, which the swap cannot change, so the swap is not ceiling-limited; the pattern does not discriminate H from a plan that decays with the horizon (card).

## 7. The card's committed next step (SUPPORTED), and what stays open

* **Committed:** the recovery fine-tune of O (`DRAFT_RECOVERY_CARD.md`, a DRAFT, not frozen, not edited here) goes to the PI for a box-2 go, **together with the leak caveat of section 0 and the leak-free separating arm named as its unknown**; no GPU job is started by this card.
* **What the read does not say:** SUPPORTED does not say which part of the logged history carries the effect (the human's realised speeds at ticks 1 and 2, the speed bucket, or the executed lag itself), and does not say a model trained or run without the human's history recovers any of the 91.8 %-98.4 % closed. The leak-free arm (`../histswap_lf/`, in progress) is the experiment that separates them.
* **Not decided here:** adoption into any K-1 / WM-1 claim; the Master Mind registers the claim row `H-HISTSWAP-1` (a proposed id).

## 8. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)

* **CTRL_s0**: X {"N": 2359, "E": 0.0062, "partition_sum_minus_E": 0.0, "n_dip_windows": 19, "n_highk_steps": 2901, "n_highk_steps_masked_in_row": 2738, "highk_share_from_steps_under_2mps": 0.9155} | Y {"N": 2364, "E": 0.0578, "partition_sum_minus_E": 0.0, "n_dip_windows": 46, "n_highk_steps": 4613, "n_highk_steps_masked_in_row": 3411, "highk_share_from_steps_under_2mps": 0.5549}
* **CTRL_s1**: X {"N": 2360, "E": 0.0063, "partition_sum_minus_E": 0.0, "n_dip_windows": 19, "n_highk_steps": 2981, "n_highk_steps_masked_in_row": 2769, "highk_share_from_steps_under_2mps": 0.9339} | Y {"N": 2365, "E": 0.1212, "partition_sum_minus_E": 0.0, "n_dip_windows": 63, "n_highk_steps": 4962, "n_highk_steps_masked_in_row": 3704, "highk_share_from_steps_under_2mps": 0.5851}
* **SOUP_s0**: X {"N": 2359, "E": 0.0088, "partition_sum_minus_E": 0.0, "n_dip_windows": 21, "n_highk_steps": 2969, "n_highk_steps_masked_in_row": 2825, "highk_share_from_steps_under_2mps": 0.9139} | Y {"N": 2364, "E": 0.0919, "partition_sum_minus_E": 0.0, "n_dip_windows": 74, "n_highk_steps": 5468, "n_highk_steps_masked_in_row": 4202, "highk_share_from_steps_under_2mps": 0.534}
* **SOUP_s1**: X {"N": 2360, "E": 0.0062, "partition_sum_minus_E": -0.0, "n_dip_windows": 16, "n_highk_steps": 2980, "n_highk_steps_masked_in_row": 2793, "highk_share_from_steps_under_2mps": 0.9267} | Y {"N": 2365, "E": 0.1067, "partition_sum_minus_E": 0.0, "n_dip_windows": 55, "n_highk_steps": 5282, "n_highk_steps_masked_in_row": 3936, "highk_share_from_steps_under_2mps": 0.618}

Gloss (from `gate_kappa.partition`): N = windows finite on both sides; E = the curvature MAE point (X - P10 for X, Y - P10 for Y); n_dip_windows = windows with an unwarranted speed dip; n_highk_steps = steps with |kappa| of the executed path above 0.2 1/m.

## 9. Limits

* **X is an ORACLE** (section 0): no row of X is performance and nothing here is a deployable-fix claim.
* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); X and Y share the checkpoint and the planner is deterministic; the two seeds per base are replicates of the rig, not a variance estimate. A separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`).
* Tier T1-kin: executed state, logged cameras, own plan applied open-loop in the scene; it says nothing about re-perception.
* S-POS rows of X are UNVERIFIED (logged poses read by the S head, executed pose in the extractor frame). The swap's causal speed bucket is not separated from the rest of the history swap.
