# RESULTS2: H-LEVELPROBE-2 (zero training; ORACLE diagnostics)

Card `PREREG_LEVELPROBE2.md` sha256 `8f1208a583469ad7fbf26f1de5c1937538ecc421030a81367ec758610f416b39` (frozen before any level file and any rollout); reader `code/levelprobe2_read.py`; raw `raw/LEVELPROBE2.json`; level files `raw/levels_meta.json`. Every number: tier **T1-kin** / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers another draw of EPISODES only; X and Y share the checkpoint) / evidence class **MEASURED**.

## 0. The committed outcome: **PROFILE**

| arm | level | mean share of the M1 excess closed | per-cell shares | cells >= 0.30 | M1 reading |
|---|---|---|---|---|---|
| MEAN6 | mean logged speed over (t0, t0 + 6 s] | -0.461 | -0.695, -0.441, -0.290, -0.417 | 0 of 4 | NOT CLOSES (0 of 4 cells separated-better) |
| FUT3 | logged speed at t0 + 3 s | -0.416 | -0.643, -0.395, -0.255, -0.369 | 0 of 4 | NOT CLOSES (0 of 4 cells separated-better) |
| FUT1 | logged speed at t0 + 1 s | -0.354 | -0.556, -0.341, -0.188, -0.333 | 0 of 4 | NOT CLOSES (0 of 4 cells separated-better) |

Horizon pattern (mean share closed): FUT1 -0.354, FUT3 -0.416, MEAN6 -0.461. TARGET-SPEED SUPPORTED: False; PROFILE: True.
Reference rows read from files: HISTSWAP-LF HLVL (per-tick logged speed) 0.599 to 0.633 per cell, HCV -0.559 to -0.301 (`histswap_lf/raw/HISTSWAP_LF_T1.json`); HLVL-T0 (logged speed at t0) mean -0.386 (`raw/LEVELPROBE.json`).

## 1. Per arm and cell: M1 X - Y, the excess, ADE6 X - Y, HARMS flag, outcome label

| arm | cell | M1 X - Y | Y - P10 (the excess) | share closed | ADE6 X - Y (m) | outcome label |
|---|---|---|---|---|---|---|
| MEAN6 | CTRL_s0 | +0.0336 [+0.0099, +0.0559] SW | +0.0483 | -0.6950 | -0.1650 [-0.2757, -0.0522] SB | HARMS |
| MEAN6 | CTRL_s1 | +0.0249 [+0.0004, +0.0486] SW | +0.0565 | -0.4415 | -0.2185 [-0.3599, -0.0846] SB | HARMS |
| MEAN6 | SOUP_s0 | +0.0193 [-0.0046, +0.0425] ns | +0.0666 | -0.2900 | -0.2559 [-0.3930, -0.1208] SB | HARMS |
| MEAN6 | SOUP_s1 | +0.0241 [-0.0006, +0.0471] ns | +0.0576 | -0.4174 | -0.2460 [-0.3804, -0.1169] SB | HARMS |
| FUT3 | CTRL_s0 | +0.0311 [+0.0075, +0.0524] SW | +0.0483 | -0.6435 | -0.0295 [-0.1408, +0.0810] ns | HARMS |
| FUT3 | CTRL_s1 | +0.0223 [-0.0023, +0.0445] ns | +0.0565 | -0.3946 | -0.0790 [-0.2154, +0.0480] ns | HARMS |
| FUT3 | SOUP_s0 | +0.0170 [-0.0067, +0.0394] ns | +0.0666 | -0.2552 | -0.1142 [-0.2455, +0.0211] ns | HARMS |
| FUT3 | SOUP_s1 | +0.0213 [-0.0028, +0.0429] ns | +0.0576 | -0.3688 | -0.1062 [-0.2331, +0.0163] ns | HARMS |
| FUT1 | CTRL_s0 | +0.0269 [+0.0079, +0.0457] SW | +0.0483 | -0.5556 | -0.2055 [-0.2938, -0.1120] SB | HARMS |
| FUT1 | CTRL_s1 | +0.0193 [-0.0015, +0.0384] ns | +0.0565 | -0.3407 | -0.2495 [-0.3648, -0.1392] SB | HARMS |
| FUT1 | SOUP_s0 | +0.0125 [-0.0075, +0.0321] ns | +0.0666 | -0.1881 | -0.2892 [-0.3948, -0.1783] SB | HARMS |
| FUT1 | SOUP_s1 | +0.0192 [-0.0012, +0.0390] ns | +0.0576 | -0.3328 | -0.2808 [-0.3866, -0.1756] SB | HARMS |

HARMS flag per arm and base (outcome label == HARMS): {'MEAN6': {'DAG-OACC': True, 'SOUP': True}, 'FUT3': {'DAG-OACC': True, 'SOUP': True}, 'FUT1': {'DAG-OACC': True, 'SOUP': True}}.

## 2. All 19 rows, X - Y per cell, by family (S-POS rows UNVERIFIED; STRATEGIC family PAUSED, ITEM 25; oracle absolute rows are not performance)

### MEAN6

| row | family | CTRL_s0 | CTRL_s1 | SOUP_s0 | SOUP_s1 |
|---|---|---|---|---|---|
| ADE6 | PRIMARY | -0.1650 [-0.2757, -0.0522] SB | -0.2185 [-0.3599, -0.0846] SB | -0.2559 [-0.3930, -0.1208] SB | -0.2460 [-0.3804, -0.1169] SB |
| speed MAE | LONGITUDINAL | -0.2885 [-0.3638, -0.2189] SB | -0.3195 [-0.3925, -0.2505] SB | -0.3316 [-0.4013, -0.2630] SB | -0.3612 [-0.4365, -0.2911] SB |
| along-track MAE | LONGITUDINAL | -0.1612 [-0.2723, -0.0524] SB | -0.2513 [-0.4055, -0.1101] SB | -0.2577 [-0.3985, -0.1206] SB | -0.2486 [-0.3939, -0.1092] SB |
| accel MAE | LONGITUDINAL | -0.2199 [-0.3716, -0.0963] SB | -0.2061 [-0.3142, -0.1117] SB | -0.2007 [-0.3248, -0.0991] SB | -0.3894 [-0.5737, -0.2343] SB |
| DK-VIOL | LONGITUDINAL | -0.0370 [-0.0830, -0.0016] SB | -0.0235 [-0.0520, +0.0016] ns | -0.0269 [-0.0561, -0.0018] SB | -0.0218 [-0.0420, -0.0049] SB |
| DK-HW | LONGITUDINAL | -0.1877 [-0.7005, +0.3034] ns | -0.1889 [-0.6616, +0.2608] ns | -0.2475 [-0.7814, +0.2352] ns | -0.0541 [-0.4525, +0.3361] ns |
| DK-TG | LONGITUDINAL | -0.0214 [-0.1660, +0.1302] ns | -0.0125 [-0.1446, +0.1260] ns | -0.0043 [-0.1252, +0.1303] ns | -0.0157 [-0.1428, +0.1070] ns |
| heading MAE | LATERAL | -1.5520 [-2.1953, -1.0006] SB | -2.1107 [-2.9155, -1.4037] SB | -1.7081 [-2.4053, -1.1067] SB | -1.9394 [-2.7101, -1.2509] SB |
| yaw-rate MAE | LATERAL | -10.0555 [-13.1278, -7.2881] SB | -14.2723 [-19.5202, -9.7432] SB | -12.7198 [-16.5338, -9.3405] SB | -12.5573 [-16.7563, -8.7910] SB |
| curvature MAE | LATERAL | -0.0397 [-0.0664, -0.0153] SB | -0.0810 [-0.1201, -0.0461] SB | -0.0628 [-0.0989, -0.0289] SB | -0.0437 [-0.0771, -0.0123] SB |
| cross-track MAE | LATERAL | -0.1201 [-0.1809, -0.0613] SB | -0.0986 [-0.1651, -0.0333] SB | -0.1219 [-0.1811, -0.0631] SB | -0.1383 [-0.2004, -0.0779] SB |
| L1 lateral-accel MAE | LATERAL | -1.0586 [-1.5147, -0.6862] SB | -1.1551 [-1.6962, -0.7245] SB | -1.1413 [-1.5790, -0.7800] SB | -1.1952 [-1.6767, -0.7891] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0381 [-0.0649, -0.0185] SB | -0.0555 [-0.0847, -0.0317] SB | -0.0677 [-0.1124, -0.0336] SB | -0.0499 [-0.0781, -0.0271] SB |
| TAC-LAT | TACTICAL | +0.0146 [-0.0050, +0.0342] ns | +0.0079 [-0.0196, +0.0371] ns | +0.0313 [+0.0112, +0.0529] SB | +0.0229 [-0.0038, +0.0480] ns |
| TAC-LON | TACTICAL | -0.1125 [-0.1593, -0.0642] SW | -0.0863 [-0.1376, -0.0321] SW | -0.0871 [-0.1392, -0.0334] SW | -0.0871 [-0.1359, -0.0325] SW |
| S-POS@30 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +0.4723 [+0.1296, +0.8470] SW | +0.8560 [+0.3974, +1.3541] SW | +0.4013 [-0.0175, +0.8725] ns | +0.8440 [+0.3615, +1.3416] SW |
| S-POS@50 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +0.6475 [+0.0607, +1.3019] SW | +1.1906 [+0.4632, +2.0066] SW | +0.5794 [-0.1667, +1.4338] ns | +1.1971 [+0.4628, +1.9804] SW |
| S-POS@80 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +0.8592 [-0.2077, +2.0647] ns | +1.5065 [+0.2991, +2.8985] SW | +0.7814 [-0.4912, +2.2398] ns | +1.5365 [+0.4145, +2.7973] SW |
| M1 crawl share | crawl | +0.0336 [+0.0099, +0.0559] SW | +0.0249 [+0.0004, +0.0486] SW | +0.0193 [-0.0046, +0.0425] ns | +0.0241 [-0.0006, +0.0471] ns |

### FUT3

| row | family | CTRL_s0 | CTRL_s1 | SOUP_s0 | SOUP_s1 |
|---|---|---|---|---|---|
| ADE6 | PRIMARY | -0.0295 [-0.1408, +0.0810] ns | -0.0790 [-0.2154, +0.0480] ns | -0.1142 [-0.2455, +0.0211] ns | -0.1062 [-0.2331, +0.0163] ns |
| speed MAE | LONGITUDINAL | -0.2825 [-0.3576, -0.2149] SB | -0.3127 [-0.3842, -0.2459] SB | -0.3242 [-0.3928, -0.2565] SB | -0.3541 [-0.4290, -0.2856] SB |
| along-track MAE | LONGITUDINAL | -0.0213 [-0.1320, +0.0856] ns | -0.1079 [-0.2506, +0.0298] ns | -0.1093 [-0.2431, +0.0231] ns | -0.1076 [-0.2426, +0.0284] ns |
| accel MAE | LONGITUDINAL | -0.1984 [-0.3458, -0.0791] SB | -0.1840 [-0.2882, -0.0947] SB | -0.1777 [-0.2958, -0.0792] SB | -0.3668 [-0.5461, -0.2137] SB |
| DK-VIOL | LONGITUDINAL | -0.0286 [-0.0695, +0.0036] ns | -0.0134 [-0.0388, +0.0099] ns | -0.0218 [-0.0472, +0.0000] ns | -0.0134 [-0.0302, +0.0016] ns |
| DK-HW | LONGITUDINAL | -0.0848 [-0.5516, +0.3731] ns | +0.0132 [-0.3808, +0.4235] ns | -0.1685 [-0.6681, +0.2873] ns | +0.1060 [-0.2401, +0.4456] ns |
| DK-TG | LONGITUDINAL | +0.0106 [-0.1237, +0.1522] ns | +0.0241 [-0.0946, +0.1554] ns | +0.0179 [-0.0978, +0.1440] ns | +0.0096 [-0.1112, +0.1293] ns |
| heading MAE | LATERAL | -0.0847 [-0.5934, +0.4696] ns | -0.6216 [-1.1867, -0.0721] SB | -0.4324 [-1.0756, +0.1852] ns | -0.3197 [-0.8412, +0.2182] ns |
| yaw-rate MAE | LATERAL | -8.2688 [-11.1113, -5.6213] SB | -12.8252 [-17.7746, -8.5359] SB | -11.1378 [-14.5672, -7.9969] SB | -11.2420 [-15.3124, -7.6371] SB |
| curvature MAE | LATERAL | -0.0192 [-0.0433, +0.0074] ns | -0.0617 [-0.0977, -0.0309] SB | -0.0484 [-0.0812, -0.0189] SB | -0.0266 [-0.0564, -0.0014] SB |
| cross-track MAE | LATERAL | -0.1130 [-0.1742, -0.0539] SB | -0.0879 [-0.1576, -0.0233] SB | -0.1174 [-0.1757, -0.0578] SB | -0.1239 [-0.1846, -0.0643] SB |
| L1 lateral-accel MAE | LATERAL | -1.0139 [-1.4656, -0.6461] SB | -1.1208 [-1.6521, -0.6981] SB | -1.0763 [-1.4829, -0.7376] SB | -1.1583 [-1.6421, -0.7566] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0331 [-0.0596, -0.0133] SB | -0.0495 [-0.0782, -0.0252] SB | -0.0627 [-0.1072, -0.0284] SB | -0.0431 [-0.0713, -0.0209] SB |
| TAC-LAT | TACTICAL | +0.0108 [-0.0063, +0.0284] ns | +0.0025 [-0.0233, +0.0279] ns | +0.0292 [+0.0092, +0.0504] SB | +0.0167 [-0.0083, +0.0412] ns |
| TAC-LON | TACTICAL | -0.1096 [-0.1574, -0.0600] SW | -0.0817 [-0.1330, -0.0279] SW | -0.0842 [-0.1359, -0.0317] SW | -0.0884 [-0.1388, -0.0338] SW |
| S-POS@30 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +0.9650 [+0.4764, +1.5148] SW | +1.3388 [+0.7829, +1.9795] SW | +0.8913 [+0.3444, +1.5101] SW | +1.3598 [+0.7679, +2.0241] SW |
| S-POS@50 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +1.4350 [+0.6329, +2.3768] SW | +1.9615 [+1.0636, +3.0041] SW | +1.3589 [+0.4317, +2.4606] SW | +2.0228 [+1.0757, +3.0875] SW |
| S-POS@80 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +1.9699 [+0.6490, +3.4777] SW | +2.5941 [+1.1730, +4.2875] SW | +1.8749 [+0.3842, +3.6840] SW | +2.7320 [+1.3460, +4.4714] SW |
| M1 crawl share | crawl | +0.0311 [+0.0075, +0.0524] SW | +0.0223 [-0.0023, +0.0445] ns | +0.0170 [-0.0067, +0.0394] ns | +0.0213 [-0.0028, +0.0429] ns |

### FUT1

| row | family | CTRL_s0 | CTRL_s1 | SOUP_s0 | SOUP_s1 |
|---|---|---|---|---|---|
| ADE6 | PRIMARY | -0.2055 [-0.2938, -0.1120] SB | -0.2495 [-0.3648, -0.1392] SB | -0.2892 [-0.3948, -0.1783] SB | -0.2808 [-0.3866, -0.1756] SB |
| speed MAE | LONGITUDINAL | -0.0958 [-0.1716, -0.0208] SB | -0.1154 [-0.1909, -0.0414] SB | -0.1406 [-0.2169, -0.0654] SB | -0.1578 [-0.2323, -0.0844] SB |
| along-track MAE | LONGITUDINAL | -0.2095 [-0.2989, -0.1175] SB | -0.3053 [-0.4239, -0.1883] SB | -0.2995 [-0.4106, -0.1871] SB | -0.3045 [-0.4214, -0.1865] SB |
| accel MAE | LONGITUDINAL | -0.2068 [-0.3588, -0.0833] SB | -0.1819 [-0.2879, -0.0902] SB | -0.1859 [-0.3120, -0.0843] SB | -0.3662 [-0.5488, -0.2123] SB |
| DK-VIOL | LONGITUDINAL | +0.0168 [-0.0136, +0.0472] ns | +0.0387 [+0.0132, +0.0675] SW | +0.0252 [+0.0000, +0.0532] ns | +0.0370 [+0.0091, +0.0698] SW |
| DK-HW | LONGITUDINAL | -0.0048 [-0.3070, +0.3159] ns | +0.1603 [-0.1289, +0.4760] ns | -0.0184 [-0.3520, +0.3194] ns | +0.1142 [-0.1299, +0.3662] ns |
| DK-TG | LONGITUDINAL | -0.0058 [-0.1129, +0.0886] ns | +0.0201 [-0.0658, +0.1152] ns | -0.0008 [-0.0860, +0.0897] ns | +0.0065 [-0.0694, +0.0885] ns |
| heading MAE | LATERAL | -0.2712 [-0.6990, +0.2024] ns | -0.2100 [-0.9261, +0.5462] ns | -0.4901 [-1.1047, +0.1659] ns | -0.0319 [-0.7235, +0.7374] ns |
| yaw-rate MAE | LATERAL | -8.3470 [-10.9474, -5.8751] SB | -12.4314 [-16.9153, -8.6849] SB | -10.7833 [-14.0494, -7.8316] SB | -11.5768 [-15.4651, -8.1241] SB |
| curvature MAE | LATERAL | -0.0229 [-0.0454, -0.0009] SB | -0.0583 [-0.0939, -0.0278] SB | -0.0470 [-0.0800, -0.0167] SB | -0.0343 [-0.0662, -0.0053] SB |
| cross-track MAE | LATERAL | -0.0545 [-0.1068, -0.0051] SB | -0.0061 [-0.0653, +0.0534] ns | -0.0540 [-0.1059, -0.0039] SB | -0.0522 [-0.1090, +0.0025] ns |
| L1 lateral-accel MAE | LATERAL | -1.0032 [-1.4429, -0.6447] SB | -1.0998 [-1.6133, -0.6864] SB | -1.0554 [-1.4433, -0.7255] SB | -1.1574 [-1.6275, -0.7633] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0304 [-0.0563, -0.0123] SB | -0.0376 [-0.0636, -0.0138] SB | -0.0561 [-0.1005, -0.0213] SB | -0.0332 [-0.0615, -0.0091] SB |
| TAC-LAT | TACTICAL | +0.0050 [-0.0113, +0.0213] ns | -0.0033 [-0.0304, +0.0230] ns | +0.0196 [+0.0037, +0.0371] SB | +0.0067 [-0.0179, +0.0296] ns |
| TAC-LON | TACTICAL | -0.0696 [-0.1126, -0.0246] SW | -0.0600 [-0.1120, -0.0079] SW | -0.0429 [-0.0912, +0.0067] ns | -0.0617 [-0.1120, -0.0083] SW |
| S-POS@30 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +0.8874 [+0.3969, +1.4508] SW | +1.2239 [+0.5745, +1.9354] SW | +0.6675 [+0.1580, +1.2448] SW | +1.2678 [+0.6445, +1.9502] SW |
| S-POS@50 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +1.4685 [+0.6195, +2.4229] SW | +1.9863 [+0.9200, +3.1719] SW | +1.1646 [+0.2760, +2.1522] SW | +2.0712 [+1.0585, +3.1794] SW |
| S-POS@80 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +2.0925 [+0.7615, +3.7164] SW | +2.7866 [+1.1676, +4.6866] SW | +1.6828 [+0.3461, +3.2547] SW | +2.9277 [+1.3953, +4.6433] SW |
| M1 crawl share | crawl | +0.0269 [+0.0079, +0.0457] SW | +0.0193 [-0.0015, +0.0384] ns | +0.0125 [-0.0075, +0.0321] ns | +0.0192 [-0.0012, +0.0390] ns |

## 3. Controls (computed before any row; a failure refuses the read)

| control | PASS |
|---|---|
| KY_reused_Y_equals_frozen_pins | True |
| KA1_fresh_off_rerun_equals_reused_Y | True |
| KT0_level_file_of_t0_speeds_reproduces_HLVL_T0_bit_for_bit | True |
| KC1_flag_reached_the_run_and_v0_in_is_as_the_card_says | True |
| KC2_the_CV_history_reached_the_network | True |
| KC3_plan_committed_to_the_executed_state | True |
| KF_floor_window_check_vs_fresh_Y | True |
| K3_Y_banked_minus_P10_reproduces_SOUP_READ | True |
| KB1_P10_side_equals_landed_banked_b | True |

## 4. Scope
* Oracles are diagnostics, not deployable; the arms hold a per-window constant level computed from the logged speed after t0.
* One checkpoint per cell; the 4 cells are the rig's replicates, not a variance estimate (`H-ESTIM-SEED-1`). T1-kin cameras follow the logged path.
* A SUPPORTED outcome opens a separately pre-registered predicted mean-speed / goal input (vision + nav, not derived from the situation classifier); a PROFILE outcome closes the scalar target-speed family for M1.

