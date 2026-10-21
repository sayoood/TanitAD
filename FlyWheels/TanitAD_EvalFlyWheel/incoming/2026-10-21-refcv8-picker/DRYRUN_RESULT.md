# R8-PICKER: refcv8 EMA-48,852 plus a GT-free picker, against the shipped refcv8 + RETIME

* **Pre-registration.** `PREREG_R8_PICKER.md`, sha256 `e6ec30e7bc03b3dc…`, plus Addendum 1 `994bd84910dd920d…` and Addendum 2 `a4982749af9b2ada…` (the K1 rule). All three seals were verified at read time: **True**.
* **How this file was written.** Mechanically, by `code/analyze_picker_r8.py`, at 2026-10-21 22:21:27 +0200.
* **Tier.** T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).
* **Evidence class.** MEASURED.
* **Estimator.** paired episode-cluster bootstrap (taniteval.ci via pair_read_r8.paired_cell), B 2000, seed 0, 95 % percentile; models and RETIME gains FIXED: answers ONLY 'another draw of EPISODES'; ONE training seed; inference seeds 0 and 1 are the inference replicate; picker - SHIP is within one DDIM draw.
* **Rolls.** R0 = seed 0 **FULL ARM**, which reproduces the banked s0 draw. R1 = seed 1 `--os-only`. Both ran on Thor.

## Verdict

**Recommendation: NONE ADOPTABLE**

Every blocking control passed: **False**.

| arm | verdict | A1: SHIP-R8 | A2: no family cell separated worse vs SHIP | A3: S2 ADE − SHIP, s0 / s1 |
|---|---|---|---|---|
| MBR32+RT | **INCOMPLETE** | PASS | False (5 cells) | +0.0000 [+0.0000, +0.0000] / -0.0109 [-0.0157, -0.0064] sep |
| BASE+RT | **INCOMPLETE** | PASS | True (0 cells) | +0.0000 [+0.0000, +0.0000] / -0.0066 [-0.0108, -0.0029] sep |
| MBR32 | **INCOMPLETE** | n/a (raw arm) | False (25 cells) | +0.0274 [+0.0210, +0.0342] sep / -0.0009 [-0.0060, +0.0041] |
| BASE | **INCOMPLETE** | n/a (raw arm) | False (21 cells) | +0.0274 [+0.0210, +0.0342] sep / +0.0125 [+0.0068, +0.0184] sep |

The SHIP inference floor (|S2 ADE s0 − s1| on compliant windows) is 0.0013 m. BASE's adoption is a PI decision whatever its verdict, because it removes the E9 goal-distance graft and the drivable critic.

**Next levers (prereg section 8, evaluated mechanically):**

* L3 TRIGGERED (MBR32+RT): a LATERAL / TACTICAL cell is separated worse (or clause 4/5 failed) -> refcv9 selector item (rank the extras). BLOCKED on training compute + a spec

## SHIP-R8 (AMENDMENT 3, registered table 2aacc9ff…) per tree

| tree | overall | per clause | E1 s0 / s1 | E3 s0 / s1 |
|---|---|---|---|---|
| SHIP (pvship_pf) | **PASS** | {'1': 'PASS', '1b': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'PASS', '6': 'N/A', '7': 'PASS'} | -0.0342 [-0.0474, -0.0216] sep / -0.0329 [-0.0452, -0.0207] sep | -0.8682 [-1.1959, -0.5380] sep / -0.8684 [-1.1913, -0.5447] sep |
| BASE (pvbase_pf) | **PASS** | {'1': 'PASS', '1b': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'PASS', '6': 'N/A', '7': 'PASS'} | -0.0342 [-0.0474, -0.0216] sep / -0.0395 [-0.0506, -0.0288] sep | -0.8682 [-1.1959, -0.5380] sep / -1.0225 [-1.3376, -0.7212] sep |
| MBR32 (pvmbrpx_pf) | **PASS** | {'1': 'PASS', '1b': 'PASS', '2': 'PASS', '3': 'PASS', '4': 'PASS', '5': 'PASS', '6': 'N/A', '7': 'PASS'} | -0.0342 [-0.0474, -0.0216] sep / -0.0438 [-0.0543, -0.0336] sep | -0.8682 [-1.1959, -0.5380] sep / -1.0672 [-1.3696, -0.7724] sep |

**Raw-plan bars.** These are capped, on compliant windows, minus `ha0_ext_pose`; NR is vs refcv7-50,400.

| raw arm | E1 | E3 | NR per family |
|---|---|---|---|
| SHIP_raw | FAILED (s0 -0.0069, s1 -0.0047) | PASS (single training seed; training-seed floor unmeasured) (s0 -0.7285, s1 -0.7292) | {'ADE': 'FAIL: S6:ade_m', 'LONGITUDINAL': 'FAIL: S2:LON_speed_mae_mps, S2:LON_along_abs_final_m', 'LATERAL': 'PASS'} |
| BASE_raw | FAILED (s0 -0.0069, s1 -0.0205) | PASS (single training seed; training-seed floor unmeasured) (s0 -0.7285, s1 -0.9256) | {'ADE': 'FAIL: S6:ade_m', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} |
| MBR32_raw | FAILED (s0 -0.0069, s1 -0.0338) | PASS (single training seed; training-seed floor unmeasured) (s0 -0.7285, s1 -0.9865) | {'ADE': 'FAIL: S6:ade_m', 'LONGITUDINAL': 'PASS', 'LATERAL': 'PASS'} |

## The four metric families, arm − SHIP (paired, compliant windows)

Each cell is delta [lo, hi]. `sep` marks a separated interval; ⚠ marks a cell that is separated WORSE. STRATEGIC is **NOT APPLICABLE, n = 0**: the run's argv carries `--no-strategic`, so the route head is untrained and unscored, and this battery has no route or goal readout.

### S2 (compliant n = 4159 windows)

| family | metric | MBR32+RT s0 | MBR32+RT s1 | BASE+RT s0 | BASE+RT s1 | MBR32_raw_capped s0 | MBR32_raw_capped s1 | BASE_raw_capped s0 | BASE_raw_capped s1 |
|---|---|---|---|---|---|---|---|---|---|
| ADE | ade_m | +0.0000 [+0.0000, +0.0000] | -0.0109 [-0.0157, -0.0064] sep | +0.0000 [+0.0000, +0.0000] | -0.0066 [-0.0108, -0.0029] sep | ⚠ +0.0274 [+0.0210, +0.0342] sep | -0.0009 [-0.0060, +0.0041] | ⚠ +0.0274 [+0.0210, +0.0342] sep | ⚠ +0.0125 [+0.0068, +0.0184] sep |
| ADE | fde_m | +0.0000 [+0.0000, +0.0000] | -0.0236 [-0.0345, -0.0133] sep | +0.0000 [+0.0000, +0.0000] | -0.0153 [-0.0251, -0.0068] sep | ⚠ +0.0667 [+0.0505, +0.0840] sep | +0.0026 [-0.0096, +0.0139] | ⚠ +0.0667 [+0.0505, +0.0840] sep | ⚠ +0.0277 [+0.0148, +0.0408] sep |
| LONGITUDINAL | LON_speed_mae_mps | +0.0000 [+0.0000, +0.0000] | -0.0134 [-0.0188, -0.0085] sep | +0.0000 [+0.0000, +0.0000] | -0.0081 [-0.0128, -0.0038] sep | ⚠ +0.0394 [+0.0305, +0.0485] sep | +0.0057 [-0.0003, +0.0114] | ⚠ +0.0394 [+0.0305, +0.0485] sep | ⚠ +0.0194 [+0.0129, +0.0259] sep |
| LONGITUDINAL | LON_along_mae_m | +0.0000 [+0.0000, +0.0000] | -0.0119 [-0.0168, -0.0077] sep | +0.0000 [+0.0000, +0.0000] | -0.0074 [-0.0116, -0.0035] sep | ⚠ +0.0292 [+0.0225, +0.0365] sep | -0.0011 [-0.0063, +0.0038] | ⚠ +0.0292 [+0.0225, +0.0365] sep | ⚠ +0.0131 [+0.0073, +0.0192] sep |
| LONGITUDINAL | LON_along_abs_final_m | +0.0000 [+0.0000, +0.0000] | -0.0260 [-0.0372, -0.0164] sep | +0.0000 [+0.0000, +0.0000] | -0.0173 [-0.0274, -0.0083] sep | ⚠ +0.0705 [+0.0534, +0.0886] sep | +0.0029 [-0.0093, +0.0141] | ⚠ +0.0705 [+0.0534, +0.0886] sep | ⚠ +0.0287 [+0.0156, +0.0421] sep |
| LONGITUDINAL | LON_speed_within_0p5 | +0.0000 [+0.0000, +0.0000] | +0.0112 [+0.0070, +0.0158] sep | +0.0000 [+0.0000, +0.0000] | +0.0067 [+0.0027, +0.0107] sep | ⚠ -0.0281 [-0.0354, -0.0210] sep | -0.0054 [-0.0121, +0.0010] | ⚠ -0.0281 [-0.0354, -0.0210] sep | ⚠ -0.0154 [-0.0219, -0.0093] sep |
| LONGITUDINAL | LON_accel_mae_mps2 | +0.0000 [+0.0000, +0.0000] | -0.0139 [-0.0205, -0.0084] sep | +0.0000 [+0.0000, +0.0000] | -0.0090 [-0.0146, -0.0039] sep | ⚠ +0.0817 [+0.0684, +0.0949] sep | ⚠ +0.0431 [+0.0339, +0.0522] sep | ⚠ +0.0817 [+0.0684, +0.0949] sep | ⚠ +0.0563 [+0.0475, +0.0652] sep |
| LATERAL | LAT_heading_mae_deg | +0.0000 [+0.0000, +0.0000] | +0.0045 [-0.0190, +0.0294] | +0.0000 [+0.0000, +0.0000] | -0.0056 [-0.0202, +0.0104] | -0.0005 [-0.0054, +0.0043] | +0.0037 [-0.0200, +0.0287] | -0.0005 [-0.0054, +0.0043] | -0.0055 [-0.0211, +0.0117] |
| LATERAL | LAT_curvature_mae_1pm_valid | +0.0000 [+0.0000, +0.0000] | ⚠ +0.0002 [+0.0001, +0.0004] sep | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0001, +0.0001] | +0.0000 [-0.0000, +0.0001] | ⚠ +0.0002 [+0.0001, +0.0004] sep | +0.0000 [-0.0000, +0.0001] | -0.0000 [-0.0001, +0.0001] |
| LATERAL | LAT_yaw_rate_mae_radps_valid | +0.0000 [+0.0000, +0.0000] | ⚠ +0.0007 [+0.0002, +0.0013] sep | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0004, +0.0002] | +0.0002 [-0.0000, +0.0004] | ⚠ +0.0008 [+0.0003, +0.0013] sep | +0.0002 [-0.0000, +0.0004] | +0.0001 [-0.0003, +0.0004] |
| LATERAL | LAT_cross_mae_m | +0.0000 [+0.0000, +0.0000] | -0.0004 [-0.0030, +0.0023] | +0.0000 [+0.0000, +0.0000] | -0.0008 [-0.0026, +0.0009] | ⚠ +0.0009 [+0.0001, +0.0019] sep | +0.0002 [-0.0027, +0.0032] | ⚠ +0.0009 [+0.0001, +0.0019] sep | +0.0001 [-0.0021, +0.0023] |
| LATERAL | LAT_cross_abs_final_m | +0.0000 [+0.0000, +0.0000] | -0.0005 [-0.0062, +0.0057] | +0.0000 [+0.0000, +0.0000] | -0.0019 [-0.0056, +0.0019] | ⚠ +0.0029 [+0.0003, +0.0057] sep | +0.0014 [-0.0054, +0.0086] | ⚠ +0.0029 [+0.0003, +0.0057] sep | +0.0008 [-0.0042, +0.0063] |
| TACTICAL | TAC_traj_lat_correct | +0.0000 [+0.0000, +0.0000] | -0.0024 [-0.0062, +0.0012] | +0.0000 [+0.0000, +0.0000] | -0.0002 [-0.0024, +0.0019] | -0.0010 [-0.0024, +0.0002] | -0.0038 [-0.0080, +0.0000] | -0.0010 [-0.0024, +0.0002] | -0.0017 [-0.0040, +0.0005] |
| TACTICAL | TAC_traj_lon_correct | +0.0000 [+0.0000, +0.0000] | +0.0127 [+0.0052, +0.0211] sep | +0.0000 [+0.0000, +0.0000] | +0.0055 [+0.0012, +0.0106] sep | ⚠ -0.0255 [-0.0360, -0.0156] sep | -0.0022 [-0.0111, +0.0067] | ⚠ -0.0255 [-0.0360, -0.0156] sep | -0.0101 [-0.0207, +0.0003] |

### S6 (compliant n = 3251 windows)

| family | metric | MBR32+RT s0 | MBR32+RT s1 | BASE+RT s0 | BASE+RT s1 | MBR32_raw_capped s0 | MBR32_raw_capped s1 | BASE_raw_capped s0 | BASE_raw_capped s1 |
|---|---|---|---|---|---|---|---|---|---|
| ADE | ade_m | +0.0000 [+0.0000, +0.0000] | -0.1988 [-0.2847, -0.1234] sep | +0.0000 [+0.0000, +0.0000] | -0.1541 [-0.2301, -0.0918] sep | ⚠ +0.1397 [+0.1110, +0.1695] sep | -0.1181 [-0.1973, -0.0482] sep | ⚠ +0.1397 [+0.1110, +0.1695] sep | -0.0572 [-0.1269, +0.0039] |
| ADE | fde_m | +0.0000 [+0.0000, +0.0000] | -0.5309 [-0.7667, -0.3272] sep | +0.0000 [+0.0000, +0.0000] | -0.4315 [-0.6367, -0.2656] sep | ⚠ +0.2654 [+0.2134, +0.3189] sep | -0.3617 [-0.5798, -0.1690] sep | ⚠ +0.2654 [+0.2134, +0.3189] sep | -0.2417 [-0.4299, -0.0834] sep |
| LONGITUDINAL | LON_speed_mae_mps | +0.0000 [+0.0000, +0.0000] | -0.1020 [-0.1382, -0.0704] sep | +0.0000 [+0.0000, +0.0000] | -0.0703 [-0.1026, -0.0447] sep | ⚠ +0.0430 [+0.0344, +0.0524] sep | -0.0751 [-0.1082, -0.0463] sep | ⚠ +0.0430 [+0.0344, +0.0524] sep | -0.0394 [-0.0692, -0.0147] sep |
| LONGITUDINAL | LON_along_mae_m | +0.0000 [+0.0000, +0.0000] | -0.2215 [-0.3096, -0.1478] sep | +0.0000 [+0.0000, +0.0000] | -0.1566 [-0.2304, -0.0942] sep | ⚠ +0.1398 [+0.1104, +0.1691] sep | -0.1379 [-0.2131, -0.0738] sep | ⚠ +0.1398 [+0.1104, +0.1691] sep | -0.0595 [-0.1287, -0.0008] sep |
| LONGITUDINAL | LON_along_abs_final_m | +0.0000 [+0.0000, +0.0000] | -0.5901 [-0.8319, -0.3925] sep | +0.0000 [+0.0000, +0.0000] | -0.4291 [-0.6313, -0.2682] sep | ⚠ +0.2588 [+0.2066, +0.3123] sep | -0.4210 [-0.6356, -0.2373] sep | ⚠ +0.2588 [+0.2066, +0.3123] sep | -0.2449 [-0.4334, -0.0882] sep |
| LONGITUDINAL | LON_speed_within_0p5 | +0.0000 [+0.0000, +0.0000] | +0.0140 [+0.0038, +0.0241] sep | +0.0000 [+0.0000, +0.0000] | +0.0099 [+0.0048, +0.0151] sep | ⚠ -0.0165 [-0.0206, -0.0123] sep | +0.0037 [-0.0070, +0.0145] | ⚠ -0.0165 [-0.0206, -0.0123] sep | -0.0025 [-0.0090, +0.0038] |
| LONGITUDINAL | LON_accel_mae_mps2 | +0.0000 [+0.0000, +0.0000] | -0.0353 [-0.0490, -0.0238] sep | +0.0000 [+0.0000, +0.0000] | -0.0253 [-0.0372, -0.0155] sep | ⚠ +0.0058 [+0.0033, +0.0084] sep | -0.0300 [-0.0431, -0.0185] sep | ⚠ +0.0058 [+0.0033, +0.0084] sep | -0.0208 [-0.0325, -0.0113] sep |
| LATERAL | LAT_heading_mae_deg | +0.0000 [+0.0000, +0.0000] | ⚠ +0.1489 [+0.0289, +0.2777] sep | +0.0000 [+0.0000, +0.0000] | -0.0090 [-0.0872, +0.0667] | +0.0080 [-0.0183, +0.0309] | ⚠ +0.1584 [+0.0390, +0.2810] sep | +0.0080 [-0.0183, +0.0309] | -0.0111 [-0.0813, +0.0582] |
| LATERAL | LAT_curvature_mae_1pm_valid | +0.0000 [+0.0000, +0.0000] | +0.0003 [-0.0000, +0.0006] | +0.0000 [+0.0000, +0.0000] | +0.0008 [-0.0001, +0.0024] | ⚠ +0.0001 [+0.0000, +0.0001] sep | ⚠ +0.0003 [+0.0000, +0.0007] sep | ⚠ +0.0001 [+0.0000, +0.0001] sep | +0.0000 [-0.0001, +0.0001] |
| LATERAL | LAT_yaw_rate_mae_radps_valid | +0.0000 [+0.0000, +0.0000] | ⚠ +0.0011 [+0.0001, +0.0022] sep | +0.0000 [+0.0000, +0.0000] | +0.0002 [-0.0006, +0.0014] | ⚠ +0.0003 [+0.0001, +0.0005] sep | ⚠ +0.0013 [+0.0003, +0.0024] sep | ⚠ +0.0003 [+0.0001, +0.0005] sep | -0.0000 [-0.0005, +0.0005] |
| LATERAL | LAT_cross_mae_m | +0.0000 [+0.0000, +0.0000] | +0.0177 [-0.0125, +0.0486] | +0.0000 [+0.0000, +0.0000] | -0.0169 [-0.0385, +0.0058] | ⚠ +0.0151 [+0.0065, +0.0243] sep | +0.0262 [-0.0053, +0.0585] | ⚠ +0.0151 [+0.0065, +0.0243] sep | -0.0047 [-0.0266, +0.0191] |
| LATERAL | LAT_cross_abs_final_m | +0.0000 [+0.0000, +0.0000] | +0.0531 [-0.0351, +0.1432] | +0.0000 [+0.0000, +0.0000] | -0.0542 [-0.1107, +0.0028] | ⚠ +0.0387 [+0.0190, +0.0589] sep | +0.0795 [-0.0110, +0.1672] | ⚠ +0.0387 [+0.0190, +0.0589] sep | -0.0211 [-0.0760, +0.0347] |
| TACTICAL | TAC_traj_lat_correct | +0.0000 [+0.0000, +0.0000] | -0.0049 [-0.0141, +0.0037] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0033, +0.0033] | +0.0000 [+0.0000, +0.0000] | -0.0071 [-0.0164, +0.0016] | +0.0000 [+0.0000, +0.0000] | -0.0003 [-0.0037, +0.0030] |
| TACTICAL | TAC_traj_lon_correct | +0.0000 [+0.0000, +0.0000] | +0.0154 [-0.0003, +0.0320] | +0.0000 [+0.0000, +0.0000] | +0.0169 [+0.0083, +0.0254] sep | -0.0028 [-0.0089, +0.0034] | +0.0123 [-0.0036, +0.0283] | -0.0028 [-0.0089, +0.0034] | +0.0141 [+0.0053, +0.0227] sep |

## Speed cap

**R5-i COMPLIANCE.** A window counts as exceeding when the full 8-slot emitted plan is above the fed ceiling (strict). Rates are on compliant windows; arm − SHIP is paired.

| arm | s0 rate | s1 rate | − SHIP s0 | − SHIP s1 |
|---|---|---|---|---|
| SHIP | 0.0055 [0.0028, 0.0087] | 0.0063 [0.0035, 0.0098] | n/a | n/a |
| SHIP_raw_uncapped | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0055 [-0.0087, -0.0028] sep | -0.0063 [-0.0098, -0.0035] sep |
| SHIP_raw_capped | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0055 [-0.0087, -0.0028] sep | -0.0063 [-0.0098, -0.0035] sep |
| BASE+RT | 0.0055 [0.0028, 0.0087] | 0.0060 [0.0032, 0.0096] | +0.0000 [+0.0000, +0.0000] | -0.0002 [-0.0015, +0.0010] |
| BASE_raw_uncapped | 0.0000 [0.0000, 0.0000] | 0.0005 [0.0000, 0.0012] | -0.0055 [-0.0087, -0.0028] sep | -0.0058 [-0.0094, -0.0029] sep |
| BASE_raw_capped | 0.0000 [0.0000, 0.0000] | 0.0005 [0.0000, 0.0012] | -0.0055 [-0.0087, -0.0028] sep | -0.0058 [-0.0094, -0.0029] sep |
| MBR32+RT | 0.0055 [0.0028, 0.0087] | 0.0178 [0.0098, 0.0275] | +0.0000 [+0.0000, +0.0000] | +0.0115 [+0.0042, +0.0209] sep |
| MBR32_raw_uncapped | 0.0000 [0.0000, 0.0000] | 0.0161 [0.0077, 0.0265] | -0.0055 [-0.0087, -0.0028] sep | +0.0099 [+0.0015, +0.0202] sep |
| MBR32_raw_capped | 0.0000 [0.0000, 0.0000] | 0.0106 [0.0046, 0.0180] | -0.0055 [-0.0087, -0.0028] sep | +0.0043 [-0.0024, +0.0122] |

**R5-ii.** The fraction of windows whose pick lies outside the model's own `e9_keep`: s0: ship 0.0000, base 0.0000, mbr32 0.0000; s1: ship 0.0000, base 0.0000, mbr32 0.0000

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
| K2_R1_vs_fanceil_thor | | False |
| K12_gt_S2_s1 | | True |
| K12_gt_S6_s1 | | True |
| K10_ship_verdict_PASS | | True |
| R0 detail | K3a 0.0, K4 disagreements 0 (ties 0), K5 disagreements 0 (ties 0), K6a n 0, K6b n_nonbinding 3, K7 vs bank 1.0000 | |
| R1 detail | K3a 0.0, K4 disagreements 0 (ties 0), K5 disagreements 0 (ties 0), K6a n 0, K6b n_nonbinding 4, K7 vs bank 0.8750, K2 vs fanceil-Thor 0.8750 / max traj 1.3894901275634766 | |
| K10 drift vs the banked verdict (not blocking) | {'E1|s0': {'ship_tree': -0.0342, 'bank': -0.0342, 'abs_diff': 0.0, 'tol': 0.005, 'within': True}, 'E1|s1': {'ship_tree': -0.0329, 'bank': -0.0329, 'abs_diff': 0.0, 'tol': 0.005, 'within': True}, 'E3|s0': {'ship_tree': -0.8682, 'bank': -0.8682, 'abs_diff': 0.0, 'tol': 0.02, 'within': True}, 'E3|s1': {'ship_tree': -0.8684, 'bank': -0.8676, 'abs_diff': 0.0007999999999999119, 'tol': 0.02, 'within': True}} | |

## Artifacts

* `PICKER_R8.json` holds every number in this file.
* The trees are `archive/pk{ship,base,mbr32}_pf`, and their lanes are `lane/<tree>/` (pfpkg, ship_r8, feed_parity).
* The captures are `runs/R{0,1}/` (Thor originals: `/home/nvidia/r8ns/picker/out/R{0,1}`).
* The picks are `archive/PICKS_s{0,1}.npz`.
