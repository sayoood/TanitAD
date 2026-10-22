# R8-PICKER: refcv8 EMA-48,852 plus a GT-free picker, against the shipped refcv8 + RETIME

* **Pre-registration.** `PREREG_R8_PICKER.md`, sha256 `e6ec30e7bc03b3dc…`, plus Addendum 1 `994bd84910dd920d…` and Addendum 2 `a4982749af9b2ada…` (the K1 rule). All three seals were verified at read time: **True**.
* **How this file was written.** Mechanically, by `code/analyze_picker_r8.py`, at 2026-10-22 06:55:36 +0200.
* **Tier.** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).
* **Evidence class.** MEASURED.
* **Estimator.** paired episode-cluster bootstrap (taniteval.ci via pair_read_r8.paired_cell), B 2000, seed 0, 95 % percentile; models and RETIME gains FIXED: answers ONLY 'another draw of EPISODES'; ONE training seed; inference seeds 0 and 1 are the inference replicate; picker - SHIP is within one DDIM draw.
* **Rolls.** R0 = seed 0 **FULL ARM**, which reproduces the banked s0 draw. R1 = seed 1 `--os-only`. Both ran on Thor.

## Verdict

**Recommendation: NONE ADOPTABLE**

Every blocking control passed: **True**.

| arm | verdict | A1: SHIP-R8 | A2: no family cell separated worse vs SHIP | A3: S2 ADE − SHIP, s0 / s1 |
|---|---|---|---|---|
| MBR32+RT | **FAIL (A2: separated worse than SHIP: LATERAL:S6:LAT_heading_mae_deg@s0,s1 +4 more)** | PASS | False (5 cells) | -0.0099 [-0.0160, -0.0047] sep / -0.0107 [-0.0158, -0.0060] sep |
| BASE+RT | **FAIL (A2: separated worse than SHIP: LATERAL:S6:LAT_curvature_mae_1pm_valid@s0)** | PASS | False (1 cells) | -0.0079 [-0.0135, -0.0034] sep / -0.0066 [-0.0108, -0.0029] sep |
| MBR32 | **NOT RAW-ELIGIBLE (E1 PASS (single t, E3 PASS (single t, NR {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}, A2 False, A3 False)** | n/a (raw arm) | False (6 cells) | -0.0003 [-0.0066, +0.0054] / -0.0012 [-0.0064, +0.0038] |
| BASE | **NOT RAW-ELIGIBLE (E1 PASS (single t, E3 PASS (single t, NR {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'}, A2 False, A3 False)** | n/a (raw arm) | False (9 cells) | +0.0100 [+0.0034, +0.0161] sep / +0.0124 [+0.0069, +0.0182] sep |

The SHIP inference floor (|S2 ADE s0 − s1| on compliant windows) is 0.0010 m. BASE's adoption is a PI decision whatever its verdict, because it removes the E9 goal-distance graft and the drivable critic.

**Next levers (prereg section 8, evaluated mechanically):**

* L3 TRIGGERED (BASE+RT): a LATERAL / TACTICAL cell is separated worse (or clause 4/5 failed) -> refcv9 selector item (rank the extras). BLOCKED on training compute + a spec
* L3 TRIGGERED (MBR32+RT): a LATERAL / TACTICAL cell is separated worse (or clause 4/5 failed) -> refcv9 selector item (rank the extras). BLOCKED on training compute + a spec

## SHIP-R8 (AMENDMENT 3, registered table 2aacc9ff…) per tree

| tree | overall | per clause | E1 s0 / s1 | E3 s0 / s1 |
|---|---|---|---|---|
| SHIP (pkship_pf) | **PASS** | {'1': 'PASS', '1b': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'PASS', '6': 'N/A', '7': 'PASS'} | -0.0340 [-0.0471, -0.0211] sep / -0.0329 [-0.0452, -0.0207] sep | -0.8672 [-1.1957, -0.5368] sep / -0.8684 [-1.1913, -0.5447] sep |
| BASE (pkbase_pf) | **PASS** | {'1': 'PASS', '1b': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'PASS', '6': 'N/A', '7': 'PASS'} | -0.0419 [-0.0534, -0.0313] sep / -0.0396 [-0.0507, -0.0289] sep | -1.0278 [-1.3450, -0.7297] sep / -1.0244 [-1.3397, -0.7228] sep |
| MBR32 (pkmbr32_pf) | **PASS** | {'1': 'PASS', '1b': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'PASS', '6': 'N/A', '7': 'PASS'} | -0.0439 [-0.0546, -0.0337] sep / -0.0436 [-0.0540, -0.0334] sep | -1.0844 [-1.3899, -0.7834] sep / -1.0792 [-1.3819, -0.7819] sep |

**Raw-plan bars.** These are capped, on compliant windows, minus `ha0_ext_pose`; NR is vs refcv7-50,400.

| raw arm | E1 | E3 | NR per family |
|---|---|---|---|
| SHIP_raw | FAILED (s0 -0.0065, s1 -0.0047) | PASS (single training seed; training-seed floor unmeasured) (s0 -0.7275, s1 -0.7292) | {'ADE': 'FAIL: S6:ade_m', 'LONGITUDINAL': 'FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m', 'LATERAL': 'PASS'} |
| BASE_raw | PASS (single training seed; training-seed floor unmeasured) (s0 -0.024, s1 -0.0205) | PASS (single training seed; training-seed floor unmeasured) (s0 -0.9302, s1 -0.927) | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} |
| MBR32_raw | PASS (single training seed; training-seed floor unmeasured) (s0 -0.0343, s1 -0.0341) | PASS (single training seed; training-seed floor unmeasured) (s0 -1.0097, s1 -1.0009) | {'ADE': 'PASS', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} |

## The four metric families, arm − SHIP (paired, compliant windows)

Each cell is delta [lo, hi]. `sep` marks a separated interval; ⚠ marks a cell that is separated WORSE. STRATEGIC is **NOT APPLICABLE, n = 0**: the run's argv carries `--no-strategic`, so the route head is untrained and unscored, and this battery has no route or goal readout.

### S2 (compliant n = 4159 windows)

| family | metric | MBR32+RT s0 | MBR32+RT s1 | BASE+RT s0 | BASE+RT s1 | MBR32_raw_capped s0 | MBR32_raw_capped s1 | BASE_raw_capped s0 | BASE_raw_capped s1 |
|---|---|---|---|---|---|---|---|---|---|
| ADE | ade_m | -0.0099 [-0.0160, -0.0047] sep | -0.0107 [-0.0158, -0.0060] sep | -0.0079 [-0.0135, -0.0034] sep | -0.0066 [-0.0108, -0.0029] sep | -0.0003 [-0.0066, +0.0054] | -0.0012 [-0.0064, +0.0038] | ⚠ +0.0100 [+0.0034, +0.0161] sep | ⚠ +0.0124 [+0.0069, +0.0182] sep |
| ADE | fde_m | -0.0220 [-0.0358, -0.0106] sep | -0.0237 [-0.0349, -0.0131] sep | -0.0184 [-0.0311, -0.0081] sep | -0.0154 [-0.0253, -0.0068] sep | +0.0032 [-0.0106, +0.0166] | +0.0012 [-0.0109, +0.0127] | ⚠ +0.0228 [+0.0079, +0.0369] sep | ⚠ +0.0276 [+0.0146, +0.0407] sep |
| LONGITUDINAL | LON_speed_mae_mps | -0.0127 [-0.0187, -0.0074] sep | -0.0134 [-0.0190, -0.0084] sep | -0.0099 [-0.0155, -0.0050] sep | -0.0082 [-0.0130, -0.0039] sep | +0.0053 [-0.0009, +0.0115] | +0.0050 [-0.0007, +0.0105] | ⚠ +0.0170 [+0.0103, +0.0238] sep | ⚠ +0.0192 [+0.0128, +0.0257] sep |
| LONGITUDINAL | LON_along_mae_m | -0.0109 [-0.0164, -0.0061] sep | -0.0116 [-0.0166, -0.0072] sep | -0.0090 [-0.0140, -0.0046] sep | -0.0075 [-0.0117, -0.0036] sep | -0.0009 [-0.0063, +0.0045] | -0.0012 [-0.0063, +0.0037] | ⚠ +0.0100 [+0.0037, +0.0164] sep | ⚠ +0.0130 [+0.0072, +0.0190] sep |
| LONGITUDINAL | LON_along_abs_final_m | -0.0244 [-0.0370, -0.0133] sep | -0.0257 [-0.0372, -0.0159] sep | -0.0207 [-0.0325, -0.0104] sep | -0.0176 [-0.0277, -0.0087] sep | +0.0026 [-0.0102, +0.0153] | +0.0020 [-0.0100, +0.0132] | ⚠ +0.0226 [+0.0082, +0.0367] sep | ⚠ +0.0284 [+0.0155, +0.0418] sep |
| LONGITUDINAL | LON_speed_within_0p5 | +0.0127 [+0.0081, +0.0178] sep | +0.0113 [+0.0069, +0.0159] sep | +0.0082 [+0.0041, +0.0126] sep | +0.0068 [+0.0028, +0.0107] sep | -0.0023 [-0.0090, +0.0041] | -0.0048 [-0.0115, +0.0017] | ⚠ -0.0118 [-0.0180, -0.0056] sep | ⚠ -0.0154 [-0.0218, -0.0093] sep |
| LONGITUDINAL | LON_accel_mae_mps2 | -0.0144 [-0.0216, -0.0081] sep | -0.0146 [-0.0213, -0.0089] sep | -0.0109 [-0.0176, -0.0053] sep | -0.0092 [-0.0149, -0.0042] sep | ⚠ +0.0420 [+0.0333, +0.0506] sep | ⚠ +0.0412 [+0.0318, +0.0503] sep | ⚠ +0.0547 [+0.0458, +0.0639] sep | ⚠ +0.0561 [+0.0473, +0.0648] sep |
| LATERAL | LAT_heading_mae_deg | +0.0085 [-0.0189, +0.0357] | +0.0032 [-0.0192, +0.0274] | +0.0050 [-0.0113, +0.0223] | -0.0061 [-0.0206, +0.0099] | +0.0069 [-0.0222, +0.0356] | +0.0026 [-0.0206, +0.0276] | +0.0062 [-0.0119, +0.0261] | -0.0061 [-0.0215, +0.0110] |
| LATERAL | LAT_curvature_mae_1pm_valid | ⚠ +0.0002 [+0.0000, +0.0004] sep | ⚠ +0.0002 [+0.0001, +0.0004] sep | +0.0001 [-0.0000, +0.0002] | -0.0000 [-0.0001, +0.0001] | ⚠ +0.0002 [+0.0000, +0.0004] sep | ⚠ +0.0002 [+0.0001, +0.0004] sep | +0.0001 [-0.0000, +0.0002] | -0.0000 [-0.0001, +0.0001] |
| LATERAL | LAT_yaw_rate_mae_radps_valid | ⚠ +0.0006 [+0.0001, +0.0012] sep | ⚠ +0.0007 [+0.0002, +0.0012] sep | +0.0001 [-0.0002, +0.0004] | -0.0001 [-0.0004, +0.0002] | ⚠ +0.0006 [+0.0001, +0.0012] sep | ⚠ +0.0007 [+0.0003, +0.0013] sep | +0.0003 [-0.0001, +0.0006] | +0.0001 [-0.0003, +0.0004] |
| LATERAL | LAT_cross_mae_m | -0.0003 [-0.0034, +0.0024] | -0.0003 [-0.0029, +0.0023] | -0.0004 [-0.0026, +0.0012] | -0.0008 [-0.0026, +0.0009] | +0.0002 [-0.0032, +0.0033] | +0.0002 [-0.0028, +0.0031] | +0.0005 [-0.0019, +0.0025] | +0.0001 [-0.0020, +0.0023] |
| LATERAL | LAT_cross_abs_final_m | -0.0007 [-0.0076, +0.0053] | -0.0006 [-0.0062, +0.0054] | -0.0009 [-0.0055, +0.0026] | -0.0018 [-0.0056, +0.0020] | +0.0005 [-0.0071, +0.0076] | +0.0009 [-0.0060, +0.0079] | +0.0019 [-0.0034, +0.0066] | +0.0009 [-0.0040, +0.0063] |
| TACTICAL | TAC_traj_lat_correct | -0.0005 [-0.0048, +0.0034] | -0.0024 [-0.0063, +0.0014] | +0.0007 [-0.0012, +0.0031] | +0.0000 [-0.0022, +0.0020] | -0.0010 [-0.0053, +0.0031] | -0.0036 [-0.0078, +0.0002] | -0.0005 [-0.0028, +0.0019] | -0.0014 [-0.0038, +0.0007] |
| TACTICAL | TAC_traj_lon_correct | +0.0149 [+0.0065, +0.0241] sep | +0.0135 [+0.0057, +0.0218] sep | +0.0087 [+0.0029, +0.0149] sep | +0.0060 [+0.0017, +0.0110] sep | +0.0041 [-0.0039, +0.0122] | -0.0002 [-0.0087, +0.0081] | -0.0060 [-0.0144, +0.0024] | -0.0094 [-0.0200, +0.0007] |

### S6 (compliant n = 3251 windows)

| family | metric | MBR32+RT s0 | MBR32+RT s1 | BASE+RT s0 | BASE+RT s1 | MBR32_raw_capped s0 | MBR32_raw_capped s1 | BASE_raw_capped s0 | BASE_raw_capped s1 |
|---|---|---|---|---|---|---|---|---|---|
| ADE | ade_m | -0.2173 [-0.3152, -0.1317] sep | -0.2108 [-0.2991, -0.1332] sep | -0.1606 [-0.2413, -0.0942] sep | -0.1561 [-0.2322, -0.0940] sep | -0.1425 [-0.2321, -0.0644] sep | -0.1325 [-0.2152, -0.0615] sep | -0.0630 [-0.1387, -0.0021] sep | -0.0587 [-0.1287, +0.0032] |
| ADE | fde_m | -0.5770 [-0.8455, -0.3411] sep | -0.5636 [-0.8073, -0.3514] sep | -0.4360 [-0.6568, -0.2586] sep | -0.4360 [-0.6428, -0.2713] sep | -0.4165 [-0.6624, -0.1981] sep | -0.3970 [-0.6254, -0.2001] sep | -0.2448 [-0.4505, -0.0836] sep | -0.2449 [-0.4338, -0.0883] sep |
| LONGITUDINAL | LON_speed_mae_mps | -0.1123 [-0.1513, -0.0787] sep | -0.1065 [-0.1436, -0.0734] sep | -0.0730 [-0.1088, -0.0439] sep | -0.0707 [-0.1028, -0.0451] sep | -0.0863 [-0.1221, -0.0547] sep | -0.0801 [-0.1149, -0.0495] sep | -0.0422 [-0.0751, -0.0161] sep | -0.0396 [-0.0694, -0.0148] sep |
| LONGITUDINAL | LON_along_mae_m | -0.2424 [-0.3324, -0.1592] sep | -0.2322 [-0.3228, -0.1537] sep | -0.1681 [-0.2491, -0.1000] sep | -0.1582 [-0.2330, -0.0966] sep | -0.1655 [-0.2477, -0.0931] sep | -0.1510 [-0.2308, -0.0824] sep | -0.0719 [-0.1471, -0.0111] sep | -0.0608 [-0.1297, -0.0019] sep |
| LONGITUDINAL | LON_along_abs_final_m | -0.6534 [-0.8991, -0.4315] sep | -0.6237 [-0.8730, -0.4156] sep | -0.4526 [-0.6724, -0.2693] sep | -0.4322 [-0.6336, -0.2692] sep | -0.4930 [-0.7174, -0.2935] sep | -0.4580 [-0.6803, -0.2667] sep | -0.2696 [-0.4780, -0.0994] sep | -0.2468 [-0.4343, -0.0895] sep |
| LONGITUDINAL | LON_speed_within_0p5 | +0.0197 [+0.0088, +0.0305] sep | +0.0152 [+0.0049, +0.0255] sep | +0.0105 [+0.0059, +0.0151] sep | +0.0096 [+0.0047, +0.0147] sep | +0.0070 [-0.0044, +0.0184] | +0.0059 [-0.0052, +0.0168] | -0.0034 [-0.0091, +0.0022] | -0.0030 [-0.0095, +0.0034] |
| LONGITUDINAL | LON_accel_mae_mps2 | -0.0404 [-0.0547, -0.0277] sep | -0.0372 [-0.0512, -0.0253] sep | -0.0272 [-0.0412, -0.0163] sep | -0.0254 [-0.0375, -0.0157] sep | -0.0342 [-0.0483, -0.0216] sep | -0.0313 [-0.0451, -0.0196] sep | -0.0226 [-0.0359, -0.0123] sep | -0.0209 [-0.0327, -0.0115] sep |
| LATERAL | LAT_heading_mae_deg | ⚠ +0.1670 [+0.0471, +0.3009] sep | ⚠ +0.1485 [+0.0349, +0.2693] sep | +0.0510 [-0.0125, +0.1228] | -0.0103 [-0.0889, +0.0658] | ⚠ +0.1842 [+0.0673, +0.3203] sep | ⚠ +0.1572 [+0.0440, +0.2782] sep | +0.0449 [-0.0186, +0.1173] | -0.0124 [-0.0829, +0.0566] |
| LATERAL | LAT_curvature_mae_1pm_valid | ⚠ +0.0003 [+0.0000, +0.0007] sep | ⚠ +0.0003 [+0.0000, +0.0007] sep | ⚠ +0.0009 [+0.0000, +0.0025] sep | +0.0007 [-0.0001, +0.0024] | ⚠ +0.0003 [+0.0001, +0.0007] sep | ⚠ +0.0003 [+0.0001, +0.0007] sep | ⚠ +0.0001 [+0.0000, +0.0003] sep | -0.0000 [-0.0001, +0.0001] |
| LATERAL | LAT_yaw_rate_mae_radps_valid | ⚠ +0.0013 [+0.0003, +0.0025] sep | ⚠ +0.0011 [+0.0002, +0.0022] sep | +0.0007 [-0.0000, +0.0018] | +0.0002 [-0.0006, +0.0014] | ⚠ +0.0016 [+0.0005, +0.0028] sep | ⚠ +0.0013 [+0.0003, +0.0024] sep | ⚠ +0.0005 [+0.0001, +0.0010] sep | -0.0001 [-0.0006, +0.0004] |
| LATERAL | LAT_cross_mae_m | +0.0139 [-0.0180, +0.0451] | +0.0152 [-0.0157, +0.0465] | -0.0068 [-0.0241, +0.0087] | -0.0174 [-0.0388, +0.0052] | +0.0206 [-0.0123, +0.0525] | +0.0229 [-0.0081, +0.0559] | +0.0070 [-0.0120, +0.0249] | -0.0051 [-0.0269, +0.0184] |
| LATERAL | LAT_cross_abs_final_m | +0.0576 [-0.0305, +0.1474] | +0.0494 [-0.0405, +0.1414] | -0.0174 [-0.0611, +0.0235] | -0.0560 [-0.1122, +0.0001] | +0.0813 [-0.0087, +0.1704] | +0.0741 [-0.0164, +0.1619] | +0.0168 [-0.0299, +0.0611] | -0.0231 [-0.0773, +0.0325] |
| TACTICAL | TAC_traj_lat_correct | -0.0012 [-0.0100, +0.0075] | -0.0055 [-0.0145, +0.0031] | -0.0009 [-0.0041, +0.0022] | +0.0003 [-0.0032, +0.0037] | -0.0031 [-0.0116, +0.0057] | -0.0077 [-0.0169, +0.0015] | -0.0006 [-0.0040, +0.0025] | -0.0003 [-0.0040, +0.0033] |
| TACTICAL | TAC_traj_lon_correct | +0.0221 [+0.0057, +0.0383] sep | +0.0145 [-0.0026, +0.0308] | +0.0154 [+0.0077, +0.0234] sep | +0.0178 [+0.0094, +0.0264] sep | +0.0197 [+0.0027, +0.0366] sep | +0.0102 [-0.0059, +0.0267] | +0.0114 [+0.0024, +0.0208] sep | +0.0145 [+0.0055, +0.0229] sep |

## Speed cap

**R5-i COMPLIANCE.** A window counts as exceeding when the full 8-slot emitted plan is above the fed ceiling (strict). Rates are on compliant windows; arm − SHIP is paired.

| arm | s0 rate | s1 rate | − SHIP s0 | − SHIP s1 |
|---|---|---|---|---|
| SHIP | 0.0053 [0.0027, 0.0085] | 0.0063 [0.0035, 0.0098] | n/a | n/a |
| SHIP_raw_uncapped | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0053 [-0.0085, -0.0027] sep | -0.0063 [-0.0098, -0.0035] sep |
| SHIP_raw_capped | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0053 [-0.0085, -0.0027] sep | -0.0063 [-0.0098, -0.0035] sep |
| BASE+RT | 0.0058 [0.0026, 0.0098] | 0.0058 [0.0031, 0.0093] | +0.0005 [-0.0010, +0.0025] | -0.0005 [-0.0017, +0.0005] |
| BASE_raw_uncapped | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0053 [-0.0085, -0.0027] sep | -0.0063 [-0.0098, -0.0035] sep |
| BASE_raw_capped | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0053 [-0.0085, -0.0027] sep | -0.0063 [-0.0098, -0.0035] sep |
| MBR32+RT | 0.0034 [0.0014, 0.0060] | 0.0034 [0.0012, 0.0060] | -0.0019 [-0.0038, -0.0002] sep | -0.0029 [-0.0053, -0.0007] sep |
| MBR32_raw_uncapped | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0053 [-0.0085, -0.0027] sep | -0.0063 [-0.0098, -0.0035] sep |
| MBR32_raw_capped | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0053 [-0.0085, -0.0027] sep | -0.0063 [-0.0098, -0.0035] sep |

**R5-ii.** The fraction of windows whose pick lies outside the model's own `e9_keep`: s0: ship 0.0000, base 0.0000, mbr32 0.0002; s1: ship 0.0000, base 0.0000, mbr32 0.0002

## Controls (prereg section 6 and Addendum 1)

| control | value | pass |
|---|---|---|
| K1_smoke | | True |
| K8_identity_splice | | True |
| K9_lane_reproduces_bank | | True |
| K13_ship_minus_ship_zero | | True |
| K11_md5_R0 | | True |
| K3_R0 | | True |
| K4_R0 | | True |
| K5_R0 | | True |
| K6b_R0 | | True |
| K6c_R0 | | True |
| roll_pf_ok_R0 | | True |
| roll_form_R0 | | True |
| K12_gt_S2_s0 | | True |
| K12_gt_S6_s0 | | True |
| K11_md5_R1 | | True |
| K3_R1 | | True |
| K4_R1 | | True |
| K5_R1 | | True |
| K6b_R1 | | True |
| K6c_R1 | | True |
| roll_pf_ok_R1 | | True |
| roll_form_R1 | | True |
| K2_R1_vs_fanceil_thor | | True |
| K12_gt_S2_s1 | | True |
| K12_gt_S6_s1 | | True |
| K10_ship_verdict_PASS | | True |
| R0 detail | K3a 0.0, K4 disagreements 0 (ties 0), K5 disagreements 0 (ties 0), K6a n 0, K6b n_nonbinding 1365, K7 vs bank 0.9939 | |
| R1 detail | K3a 0.0, K4 disagreements 0 (ties 0), K5 disagreements 0 (ties 0), K6a n 0, K6b n_nonbinding 1348, K7 vs bank 0.9958, K2 vs fanceil-Thor 1.0000 / max traj 0.0 | |
| K10 drift vs the banked verdict (not blocking) | {'E1|s0': {'ship_tree': -0.034, 'bank': -0.0342, 'abs_diff': 0.0001999999999999988, 'tol': 0.005, 'within': True}, 'E1|s1': {'ship_tree': -0.0329, 'bank': -0.0329, 'abs_diff': 0.0, 'tol': 0.005, 'within': True}, 'E3|s0': {'ship_tree': -0.8672, 'bank': -0.8682, 'abs_diff': 0.0010000000000000009, 'tol': 0.02, 'within': True}, 'E3|s1': {'ship_tree': -0.8684, 'bank': -0.8676, 'abs_diff': 0.0007999999999999119, 'tol': 0.02, 'within': True}} | |

## Artifacts

* `PICKER_R8.json` holds every number in this file.
* The trees are `archive/pk{ship,base,mbr32}_pf`, and their lanes are `lane/<tree>/` (pfpkg, ship_r8, feed_parity).
* The captures are `runs/R{0,1}/` (Thor originals: `/home/nvidia/r8ns/picker/out/R{0,1}`).
* The picks are `archive/PICKS_s{0,1}.npz`.
