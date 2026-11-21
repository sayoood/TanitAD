# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 12, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRT:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFTb:GFT", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GFT:G", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "GRT:G", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, GRT minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GFTb minus GFT over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GFT minus G over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, GRT minus G over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## GRT minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0036 / -0.0036 | +0.0000 | [-0.0187, +0.0164] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0016 / -0.0031 | -0.0008 | [-0.0114, +0.0099] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0732 / +0.0592 | +0.0662 | [+0.0280, +0.1108] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0745 / +0.0570 | +0.0657 | [+0.0410, +0.0967] | 0.0689 (0.0315) | INSIDE-SPREAD |
| gap bias 4.0 s (m) | +0.6375 / +0.5218 | +0.5796 | [+0.2073, +1.0604] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.8920 / +1.9179 | +1.9049 | [+0.5679, +3.3978] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.0032 / +0.0005 | -0.0013 | [-0.0147, +0.0120] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.1668 / +0.2013 | +0.1841 | [-0.0270, +0.4367] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +1.6393 / +1.0365 | +1.3379 | [+0.3617, +2.6359] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0015 / +0.0621 | +0.0303 | [-0.0389, +0.0900] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.3452 / +0.1530 | -0.0961 | [-0.4807, +0.2018] | 1.7849 (0.8089) | OPEN |

## GFTb minus GFT

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0036 | +0.0018 | [-0.0147, +0.0185] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0125 / +0.0031 | -0.0047 | [-0.0199, +0.0081] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0483 / +0.0047 | +0.0265 | [-0.0054, +0.0723] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.1090 / +0.0367 | -0.0361 | [-0.1300, +0.0453] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.6828 / +0.1919 | +0.4373 | [+0.0599, +0.9951] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +2.5168 / +0.2247 | +1.3708 | [-0.1896, +3.5977] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0348 / -0.0050 | +0.0149 | [-0.0116, +0.0578] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.2566 / +0.1823 | +0.2195 | [-0.0462, +0.6192] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +1.4834 / +0.0912 | +0.7873 | [-0.2242, +2.5768] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.1727 / +0.1407 | -0.0160 | [-0.2274, +0.2639] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.2648 / +0.0949 | +0.1799 | [+0.0126, +0.3770] | 1.7849 (0.8089) | INSIDE-SPREAD |

## GFT minus G

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0125 / -0.0178 | -0.0151 | [-0.0371, +0.0055] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0078 / -0.0234 | -0.0078 | [-0.0329, +0.0148] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | -0.0545 / -0.0249 | -0.0397 | [-0.0821, +0.0049] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0830 / -0.0196 | +0.0317 | [-0.0331, +0.0949] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | -0.6914 / -0.1124 | -0.4019 | [-0.8661, +0.0806] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | -1.3858 / -0.2716 | -0.8287 | [-2.2196, +1.0080] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0316 / +0.0002 | -0.0157 | [-0.0390, +0.0111] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.4466 / +0.0414 | -0.2026 | [-0.5797, +0.1874] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | -1.0538 / -1.1566 | -1.1052 | [-2.3265, -0.1904] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.2140 / +0.0859 | -0.0641 | [-0.3014, +0.2540] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.6180 / +0.3013 | -0.1583 | [-0.7679, +0.4702] | 1.7849 (0.8089) | OPEN |

## GRT minus G

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0089 / -0.0214 | -0.0151 | [-0.0340, +0.0034] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0093 / -0.0265 | -0.0086 | [-0.0336, +0.0144] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0187 / +0.0343 | +0.0265 | [-0.0331, +0.0762] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1575 / +0.0375 | +0.0975 | [+0.0159, +0.1764] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | -0.0539 / +0.4094 | +0.1777 | [-0.4226, +0.6498] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.5062 / +1.6463 | +1.0762 | [-0.8468, +2.8899] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | -0.0348 / +0.0007 | -0.0170 | [-0.0480, +0.0131] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | -0.2798 / +0.2427 | -0.0185 | [-0.5260, +0.3746] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.5855 / -0.1201 | +0.2327 | [-1.3258, +1.7374] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | -0.2155 / +0.1480 | -0.0338 | [-0.3148, +0.3023] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | -0.9631 / +0.4542 | -0.2544 | [-1.1889, +0.6316] | 1.7849 (0.8089) | OPEN |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| GFT (0,1) | 0.1121 [0.0711, 0.1570] | 0.0405 [0.0212, 0.0648] | 0.1931 [0.1119, 0.2846] | -0.1611 [-0.2359, -0.0881] | 1.0823 [0.4765, 1.7433] | 5.2278 [2.7695, 7.6887] | 0.4680 [0.4239, 0.5183] | 2.7143 [2.3027, 3.2118] | 7.9076 [6.3657, 9.6000] | 1.2414 [0.7961, 1.7429] | 2.3332 [1.5451, 3.2287] |
| GFTb (0,1) | 0.1139 [0.0746, 0.1575] | 0.0358 [0.0148, 0.0639] | 0.2196 [0.1162, 0.3360] | -0.1972 [-0.2667, -0.1386] | 1.5197 [0.6401, 2.4813] | 6.5986 [3.0172, 10.6094] | 0.4829 [0.4326, 0.5378] | 2.9338 [2.4471, 3.5291] | 8.6949 [6.3608, 11.6601] | 1.2253 [0.8514, 1.6538] | 2.5131 [1.6507, 3.4819] |
| GRT (0,1) | 0.1121 [0.0751, 0.1520] | 0.0397 [0.0224, 0.0590] | 0.2593 [0.1678, 0.3553] | -0.0953 [-0.1798, -0.0122] | 1.6620 [0.9858, 2.3766] | 7.1328 [4.7012, 9.7188] | 0.4666 [0.4233, 0.5201] | 2.8984 [2.4723, 3.3656] | 9.2455 [7.2804, 11.4222] | 1.2717 [0.8459, 1.7530] | 2.2371 [1.6207, 2.9295] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
