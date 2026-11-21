# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 16, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LC:LFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LO:LFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LCO:LFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LFTb:LFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LCO:LO", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LCO:LC", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LC minus LFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LO minus LFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LCO minus LFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LFTb minus LFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LCO minus LO over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LCO minus LC over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LC minus LFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0178 / -0.0160 | -0.0169 | [-0.0318, -0.0033] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0047 / -0.0156 | -0.0101 | [-0.0247, -0.0008] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0296 / +0.0405 | +0.0350 | [+0.0146, +0.0544] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0136 / +0.0362 | +0.0249 | [+0.0070, +0.0418] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.3401 / +0.3206 | +0.3303 | [+0.2145, +0.4595] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.3323 / +1.2871 | +1.3097 | [+0.7833, +1.8683] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0018 / -0.0135 | -0.0077 | [-0.0180, +0.0020] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.1312 / +0.1173 | +0.1242 | [+0.0397, +0.2079] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +0.8795 / +0.9458 | +0.9126 | [+0.4385, +1.4064] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0097 / +0.0516 | +0.0307 | [-0.0171, +0.0861] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.2389 / +0.3913 | +0.3151 | [+0.1645, +0.5374] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LO minus LFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0534 / -0.0569 | -0.0552 | [-0.0921, -0.0250] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0405 / -0.0545 | -0.0475 | [-0.0760, -0.0252] | 0.0421 (0.0171) | BETTER |
| F falls behind by more than 10 m at the last scored step | +0.2259 / +0.3349 | +0.2804 | [+0.1913, +0.3761] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.1253 / +2.3720 | +2.2486 | [+1.4806, +3.1495] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +4.5503 / +8.2578 | +6.4041 | [+3.6482, +10.1852] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +9.8026 / +16.5424 | +13.1725 | [+8.0613, +20.3663] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.6835 / +1.9227 | +1.8031 | [+1.0358, +2.6787] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +3.4580 / +6.5838 | +5.0209 | [+2.6403, +8.3827] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +7.6535 / +13.3772 | +10.5154 | [+5.8671, +16.9609] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.9185 / +1.4904 | +1.2044 | [+0.7472, +1.8606] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +0.7534 / +2.4567 | +1.6051 | [+0.6130, +3.2903] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LCO minus LFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0712 / -0.0552 | -0.0632 | [-0.0978, -0.0319] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0389 / -0.0421 | -0.0405 | [-0.0649, -0.0190] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.2555 / +0.3131 | +0.2843 | [+0.2201, +0.3564] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.1848 / +2.3662 | +2.2755 | [+1.4769, +3.1786] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +6.7133 / +7.5742 | +7.1437 | [+4.7588, +9.9461] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +13.0978 / +14.7586 | +13.9282 | [+9.7539, +19.0670] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.7373 / +1.9349 | +1.8361 | [+1.0597, +2.7181] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +5.5397 / +6.0369 | +5.7883 | [+3.6410, +8.3099] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +10.6968 / +11.6752 | +11.1860 | [+7.0041, +15.9301] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.8727 / +1.3503 | +1.1115 | [+0.6976, +1.6813] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +1.4153 / +1.6269 | +1.5211 | [+0.9849, +2.2607] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LFTb minus LFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0214 / -0.0160 | -0.0187 | [-0.0348, -0.0051] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0016 / -0.0093 | -0.0039 | [-0.0157, +0.0043] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0234 / +0.0234 | +0.0234 | [+0.0108, +0.0359] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0045 / +0.0114 | +0.0079 | [-0.0001, +0.0165] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.1419 / +0.1218 | +0.1319 | [+0.0379, +0.2210] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +0.5540 / +0.7223 | +0.6381 | [+0.2509, +1.0739] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.0005 / -0.0089 | -0.0042 | [-0.0123, +0.0031] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0348 / -0.0043 | +0.0152 | [-0.0691, +0.0884] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.1693 / +0.7226 | +0.4460 | [-0.0113, +0.9162] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0065 / -0.0400 | -0.0168 | [-0.0804, +0.0214] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.1515 / +0.1959 | +0.1737 | [+0.0696, +0.3091] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LCO minus LO

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0178 / +0.0018 | -0.0080 | [-0.0279, +0.0119] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / +0.0125 | +0.0070 | [-0.0054, +0.0172] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0296 / -0.0218 | +0.0039 | [-0.0360, +0.0495] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0595 / -0.0058 | +0.0269 | [-0.2122, +0.2578] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +2.1630 / -0.6836 | +0.7397 | [-1.1103, +2.7423] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +3.2952 / -1.7837 | +0.7558 | [-2.4988, +4.1821] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0538 / +0.0122 | +0.0330 | [-0.1984, +0.2743] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +2.0817 / -0.5469 | +0.7674 | [-0.9928, +2.6831] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +3.0433 / -1.7020 | +0.6706 | [-2.3794, +3.9382] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.0458 / -0.1401 | -0.0930 | [-0.2805, +0.0672] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.6619 / -0.8298 | -0.0840 | [-1.2107, +0.8062] | 1.7849 (0.8089) | OPEN |

## LCO minus LC

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0534 / -0.0391 | -0.0463 | [-0.0830, -0.0122] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0343 / -0.0265 | -0.0304 | [-0.0518, -0.0121] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.2259 / +0.2726 | +0.2492 | [+0.1886, +0.3174] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.1712 / +2.3300 | +2.2506 | [+1.4611, +3.1401] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +6.3732 / +7.2536 | +6.8134 | [+4.4285, +9.5898] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +11.7655 / +13.4716 | +12.6185 | [+8.3830, +17.8894] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.7391 / +1.9485 | +1.8438 | [+1.0662, +2.7303] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +5.4085 / +5.9196 | +5.6640 | [+3.4813, +8.2185] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +9.8173 / +10.7294 | +10.2734 | [+6.2690, +15.0460] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.8630 / +1.2987 | +1.0808 | [+0.6921, +1.6280] | 0.8769 (0.3180) | WORSE |
| time-gap error over the scored steps (s) | +1.1765 / +1.2356 | +1.2060 | [+0.5958, +1.9532] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LC (0,1) | 0.1246 [0.0809, 0.1708] | 0.0530 [0.0295, 0.0793] | 0.1807 [0.1295, 0.2356] | -0.1975 [-0.2479, -0.1477] | 0.8489 [0.3548, 1.3336] | 4.5461 [2.8669, 6.2820] | 0.4755 [0.4254, 0.5289] | 2.6780 [2.3775, 3.0159] | 7.9414 [6.9331, 9.0713] | 0.9822 [0.6653, 1.3655] | 2.3237 [1.6840, 3.0560] |
| LCO (0,1) | 0.0783 [0.0490, 0.1114] | 0.0226 [0.0105, 0.0354] | 0.4299 [0.3645, 0.4948] | 2.0531 [1.2651, 2.9565] | 7.6624 [5.3794, 10.2552] | 17.1647 [12.9244, 21.7622] | 2.3193 [1.5291, 3.1990] | 8.3420 [6.0987, 10.9234] | 18.2148 [14.1357, 22.7485] | 2.0630 [1.4692, 2.8140] | 3.5297 [2.8312, 4.4338] |
| LFT (0,1) | 0.1415 [0.0965, 0.1878] | 0.0631 [0.0370, 0.0918] | 0.1456 [0.0958, 0.1980] | -0.2224 [-0.2672, -0.1763] | 0.5186 [0.0504, 0.9670] | 3.2365 [1.7611, 4.7609] | 0.4832 [0.4338, 0.5358] | 2.5537 [2.2424, 2.8619] | 7.0288 [6.1826, 7.9892] | 0.9515 [0.6423, 1.3322] | 2.0086 [1.4447, 2.7032] |
| LFTb (0,1) | 0.1228 [0.0803, 0.1688] | 0.0592 [0.0329, 0.0888] | 0.1690 [0.1155, 0.2241] | -0.2145 [-0.2600, -0.1676] | 0.6505 [0.1691, 1.1206] | 3.8746 [2.3139, 5.4224] | 0.4790 [0.4285, 0.5320] | 2.5690 [2.2800, 2.8754] | 7.4748 [6.6152, 8.3724] | 0.9348 [0.6433, 1.2732] | 2.1823 [1.5739, 2.8929] |
| LO (0,1) | 0.0863 [0.0542, 0.1203] | 0.0156 [0.0033, 0.0300] | 0.4260 [0.3440, 0.5045] | 2.0263 [1.2562, 2.9294] | 6.9227 [4.2823, 10.4875] | 16.4089 [11.7976, 22.6707] | 2.2863 [1.5438, 3.1573] | 7.5746 [5.0730, 11.0222] | 17.5442 [13.1896, 23.5785] | 2.1560 [1.5184, 2.9748] | 3.6136 [2.3252, 5.4031] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
