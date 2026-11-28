# VERDICT HEADWAY: distance keeping in the T1-kin rollout (card PREREG_V8F_PROOF_ADD25)

Card: PREREG_V8F_PROOF_ADD25.md. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps + label join; files in 'dumps' and 'label_join').
Estimator: paired window differences A minus B, two-level bootstrap seeds then episodes (B 2000, rng 0), engine of q3_read.py / hierrep_read.py; rule of ADD11 with same sign in every seed.
Label source: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz. Alignment reference (ego speed): F:/TanitAD-scratch/v8proof/res/cache/frames.npz.

## Controls (first)

* h5_counts: PASS  {"n_lead_windows": 642, "n_episodes_with_lead": 56, "enough": true, "min_windows": 200, "min_episodes": 20}
* h4_alignment: PASS  {"n_checked": 642, "n_unchecked": 0, "n_over_tol": 0, "max_abs_diff_mps": 0.0, "tol_mps": 0.05, "shift5_check": {"PASS_expected": false, "n_over_tol": 470, "max_abs_diff_mps": 2.872589588165283, "n_unchecked": 72, "check_fails_as_required": true}}
* h6_same_windows: PASS  {"n_dumps": 32, "n_differing": 0, "regression_form_fails": true}
* h1_identity: PASS  {"max_abs_gap_diff_m": 2.2737367544323206e-13, "n_defined_U": 562, "U_mean": 0.0, "max_abs_bias_m": [1.1368683772161603e-13, 1.1368683772161603e-13, 2.842170943040401e-14], "regression_form_advance_0.1mps_max_abs_gap_diff_m": 0.6000000000000227, "regression_form_fails": true}
* h2_planted_regression: PASS  {"mps": 0.5, "max_dev_bias_2s_from_-1.0": 7.105427357601002e-15, "max_dev_bias_6s_from_-3.0": 2.842170943040401e-14, "mean_bias_2s": -1.0, "mean_bias_6s": -3.0, "n_2s": 642, "n_6s": 446, "planted_vs_logged_U": {"D": 0.1298932384341637, "A_lo": 0.09283080641493707, "A_hi": 0.1654724818442084, "spread_max": 0.04626334519572954, "verdict": "WORSE"}, "named_worse_on_U": true, "regression_form_zero_advance_dev_from_-1.0": 1.0000000000001137, "regression_form_fails": true}
* h3_swap_self: PASS  {"pairs": [{"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMC:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMT:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMG:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSML:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMH:LSRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMC:LSDRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMT:LSDRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMG:LSDRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDML:LSDRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSDMH:LSDRH", "seeds": [0, 1]}, {"PASS": true, "max_abs_self_D": 0.0, "max_abs_swap_dev": 0.0, "regression_form_fails": true, "pair": "LSMR:LSRH", "seeds": [0, 1]}]}
* ALL_PASS: True

## Counts (h5, before any comparison)

Lead windows: 642 of 2399 windows; episodes with at least one lead window: 56 of 141. Bars of the card: at least 200 windows and 20 episodes; enough: True.
Join against the bank on their common rows (decides nothing): {"bank_rows": 9460, "presence_mismatch_rows": 1164, "rows_finite_in_both": 2489, "max_abs_gap_diff_m": 1.8367528915405273}.

## Spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True.

## Names of section 3 of the card

* None, cell U, LSMC minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMT minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMG minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSML minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMH minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMC minus LSDRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMT minus LSDRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMG minus LSDRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDML minus LSDRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSDMH minus LSDRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)
* None, cell U, LSMR minus LSRH over seeds [0, 1] of [0, 1]: **NOT NAMED (no question of the card for this pair)** (decides nothing)

## LSMC minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0178 / -0.0445 | -0.0311 | [-0.0567, -0.0038] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0156 / -0.0171 | -0.0164 | [-0.0296, -0.0058] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0701 / +0.1215 | +0.0958 | [+0.0454, +0.1461] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6611 / +1.0695 | +0.8653 | [+0.5568, +1.2179] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.0099 / +2.8143 | +2.4121 | [+1.5217, +3.3977] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.1083 / +4.9353 | +4.0218 | [+2.2729, +6.1200] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4993 / +0.8951 | +0.6972 | [+0.3949, +1.0437] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.7879 / +2.3879 | +2.0879 | [+1.2903, +2.9680] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.8579 / +4.2255 | +3.5417 | [+1.9913, +5.4271] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2168 / +0.4851 | +0.3510 | [+0.1782, +0.5795] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0005 / +1.3906 | +1.1956 | [+0.8071, +1.6463] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSMT minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 1.7849 (0.8089) | OPEN |

## LSMG minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0178 / -0.0445 | -0.0311 | [-0.0567, -0.0038] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0156 / -0.0171 | -0.0164 | [-0.0296, -0.0058] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0685 / +0.1215 | +0.0950 | [+0.0444, +0.1461] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.6611 / +1.0695 | +0.8653 | [+0.5568, +1.2179] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.0099 / +2.8143 | +2.4121 | [+1.5217, +3.3977] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.0670 / +4.9353 | +4.0011 | [+2.2524, +6.1200] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.4993 / +0.8951 | +0.6972 | [+0.3949, +1.0437] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +1.7879 / +2.3879 | +2.0879 | [+1.2903, +2.9680] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +2.8229 / +4.2255 | +3.5242 | [+1.9790, +5.4271] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.2168 / +0.4851 | +0.3510 | [+0.1782, +0.5795] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +1.0012 / +1.3906 | +1.1959 | [+0.8073, +1.6463] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSML minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0053 / -0.0107 | -0.0080 | [-0.0210, +0.0021] | 0.0463 (0.0214) | OPEN |
| C contact rate | -0.0031 / -0.0016 | -0.0023 | [-0.0055, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0327 / +0.0109 | +0.0218 | [+0.0029, +0.0469] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.1734 / +0.1115 | +0.1424 | [+0.0808, +0.2157] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.7085 / +0.1326 | +0.4206 | [-0.0987, +0.9664] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +1.3382 / +0.0994 | +0.7188 | [-0.4432, +1.7930] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.1290 / +0.0887 | +0.1089 | [+0.0556, +0.1730] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.6548 / +0.0607 | +0.3578 | [-0.1477, +0.9065] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +1.3363 / +0.0008 | +0.6685 | [-0.5040, +1.7749] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0194 / +0.0268 | +0.0231 | [-0.0011, +0.0503] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.1139 / +0.1077 | +0.1108 | [+0.0277, +0.1916] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSMH minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0552 / -0.0712 | -0.0632 | [-0.0963, -0.0342] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0187 / -0.0249 | -0.0218 | [-0.0405, -0.0072] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.1900 / +0.2539 | +0.2220 | [+0.1553, +0.2866] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.1085 / +2.5283 | +2.3184 | [+1.8999, +2.7645] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +5.6145 / +6.5980 | +6.1063 | [+4.7661, +7.4890] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +9.0716 / +9.6303 | +9.3509 | [+7.1414, +11.4334] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +1.7455 / +2.1770 | +1.9612 | [+1.5281, +2.4151] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +5.0368 / +5.8584 | +5.4476 | [+4.1418, +6.8005] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +8.2088 / +8.5763 | +8.3926 | [+6.2187, +10.5057] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.7612 / +0.8387 | +0.7999 | [+0.6221, +0.9915] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +2.6603 / +3.0764 | +2.8684 | [+2.2760, +3.5002] | 1.7849 (0.8089) | WORSE |

## LSDMC minus LSDRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0569 / -0.0747 | -0.0658 | [-0.0971, -0.0374] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0125 / -0.0358 | -0.0241 | [-0.0487, -0.0058] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0779 / +0.0903 | +0.0841 | [+0.0578, +0.1135] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.9401 / +0.8777 | +0.9089 | [+0.6910, +1.1259] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.3964 / +2.3980 | +2.3972 | [+1.7390, +3.1127] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.4379 / +3.9646 | +3.7013 | [+2.5465, +5.0182] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.7786 / +0.7631 | +0.7708 | [+0.5687, +0.9834] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.1870 / +1.9966 | +2.0918 | [+1.4262, +2.7903] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.1768 / +3.3274 | +3.2521 | [+2.1063, +4.5096] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.3462 / +0.3347 | +0.3404 | [+0.2161, +0.5027] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.9788 / +0.8283 | +0.9035 | [+0.6101, +1.2490] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDMT minus LSDRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0018 / -0.0036 | -0.0009 | [-0.0069, +0.0033] | 0.0463 (0.0214) | OPEN |
| C contact rate | +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0421 (0.0171) | OPEN |
| F falls behind by more than 10 m at the last scored step | +0.0016 / +0.0016 | +0.0016 | [+0.0000, +0.0040] | 0.1604 (0.0654) | OPEN |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.0044 / +0.0066 | +0.0055 | [+0.0000, +0.0149] | 0.0689 (0.0315) | OPEN |
| gap bias 4.0 s (m) | +0.0035 / +0.0403 | +0.0219 | [+0.0000, +0.0756] | 1.6421 (0.6283) | OPEN |
| gap bias 6.0 s (m) | +0.0149 / +0.0029 | +0.0089 | [+0.0000, +0.0297] | 5.2724 (2.2479) | OPEN |
| gap error 2.0 s (m) | +0.0033 / +0.0066 | +0.0049 | [+0.0000, +0.0143] | 0.0426 (0.0189) | OPEN |
| gap error 4.0 s (m) | +0.0065 / +0.0231 | +0.0148 | [-0.0086, +0.0600] | 1.0764 (0.6913) | OPEN |
| gap error 6.0 s (m) | +0.0523 / +0.0029 | +0.0276 | [+0.0000, +0.1039] | 3.3507 (1.4831) | OPEN |
| time-gap error 2.0 s (s) | +0.0003 / +0.0004 | +0.0004 | [+0.0000, +0.0009] | 0.8769 (0.3180) | OPEN |
| time-gap error over the scored steps (s) | +0.0009 / +0.0026 | +0.0017 | [-0.0002, +0.0058] | 1.7849 (0.8089) | OPEN |

## LSDMG minus LSDRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0552 / -0.0676 | -0.0614 | [-0.0912, -0.0347] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0125 / -0.0358 | -0.0241 | [-0.0487, -0.0058] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0779 / +0.0888 | +0.0833 | [+0.0578, +0.1112] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.9226 / +0.8650 | +0.8938 | [+0.6766, +1.1122] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +2.3907 / +2.3242 | +2.3574 | [+1.6933, +3.0523] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +3.4420 / +3.8847 | +3.6634 | [+2.5365, +4.9402] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.7643 / +0.7503 | +0.7573 | [+0.5581, +0.9707] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +2.1848 / +1.9916 | +2.0882 | [+1.4162, +2.7943] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +3.1896 / +3.3327 | +3.2611 | [+2.1119, +4.4977] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.3447 / +0.3340 | +0.3393 | [+0.2154, +0.5016] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.9775 / +0.8237 | +0.9006 | [+0.6042, +1.2484] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDML minus LSDRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0338 / -0.0391 | -0.0365 | [-0.0550, -0.0201] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | -0.0047 / -0.0202 | -0.0125 | [-0.0276, -0.0017] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.0312 / +0.0202 | +0.0257 | [+0.0122, +0.0410] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +0.2902 / +0.2460 | +0.2681 | [+0.1783, +0.3649] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +0.8163 / +0.4437 | +0.6300 | [+0.3250, +1.0472] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | +1.2388 / +1.0619 | +1.1504 | [+0.5874, +1.7508] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | +0.2061 / +0.2080 | +0.2070 | [+0.1245, +0.3017] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +0.7338 / +0.3129 | +0.5233 | [+0.1992, +0.9413] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | +1.1187 / +0.7881 | +0.9534 | [+0.3694, +1.5838] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | +0.0506 / +0.0734 | +0.0620 | [+0.0304, +0.1079] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +0.3662 / +0.1365 | +0.2513 | [+0.0925, +0.4900] | 1.7849 (0.8089) | INSIDE-SPREAD |

## LSDMH minus LSDRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | -0.0801 / -0.1068 | -0.0934 | [-0.1375, -0.0559] | 0.0463 (0.0214) | BETTER |
| C contact rate | -0.0140 / -0.0498 | -0.0319 | [-0.0670, -0.0071] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | +0.2025 / +0.2445 | +0.2235 | [+0.1742, +0.2759] | 0.1604 (0.0654) | WORSE |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | +2.3235 / +2.3390 | +2.3312 | [+1.9634, +2.6948] | 0.0689 (0.0315) | WORSE |
| gap bias 4.0 s (m) | +5.6853 / +6.2810 | +5.9832 | [+4.8295, +7.1780] | 1.6421 (0.6283) | WORSE |
| gap bias 6.0 s (m) | +8.2386 / +8.4214 | +8.3300 | [+6.5563, +9.9975] | 5.2724 (2.2479) | WORSE |
| gap error 2.0 s (m) | +2.0026 / +1.9859 | +1.9943 | [+1.6291, +2.3533] | 0.0426 (0.0189) | WORSE |
| gap error 4.0 s (m) | +5.2192 / +5.3811 | +5.3002 | [+4.2178, +6.3579] | 1.0764 (0.6913) | WORSE |
| gap error 6.0 s (m) | +7.8503 / +7.1913 | +7.5208 | [+5.6483, +9.2196] | 3.3507 (1.4831) | WORSE |
| time-gap error 2.0 s (s) | +0.9213 / +0.7654 | +0.8433 | [+0.6088, +1.1096] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | +2.8662 / +2.2859 | +2.5760 | [+1.9521, +3.2885] | 1.7849 (0.8089) | WORSE |

## LSMR minus LSRH

Paired difference A minus B, windows with both arms defined. Below 0: the first arm is lower (for U, C, F, gap error, time-gap error lower is better; for the signed gap bias below 0 = the first arm closes in more than the second). Names are only on U.

| cell | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| U model-induced unsafe rate (primary) | +0.0320 / +0.0160 | +0.0240 | [+0.0058, +0.0486] | 0.0463 (0.0214) | INSIDE-SPREAD |
| C contact rate | +0.0125 / +0.0125 | +0.0125 | [+0.0042, +0.0219] | 0.0421 (0.0171) | INSIDE-SPREAD |
| F falls behind by more than 10 m at the last scored step | -0.0483 / -0.0312 | -0.0397 | [-0.0661, -0.0146] | 0.1604 (0.0654) | INSIDE-SPREAD |
| gap bias 2.0 s (m, below 0 = closes in more than the driver) | -0.2087 / -0.1031 | -0.1559 | [-0.3116, -0.0507] | 0.0689 (0.0315) | BETTER |
| gap bias 4.0 s (m) | -1.1060 / -0.6521 | -0.8790 | [-1.5596, -0.3539] | 1.6421 (0.6283) | INSIDE-SPREAD |
| gap bias 6.0 s (m) | -2.6419 / -1.4641 | -2.0530 | [-4.0136, -0.8000] | 5.2724 (2.2479) | INSIDE-SPREAD |
| gap error 2.0 s (m) | -0.1604 / -0.0721 | -0.1162 | [-0.2500, -0.0297] | 0.0426 (0.0189) | BETTER |
| gap error 4.0 s (m) | -0.5076 / -0.3420 | -0.4248 | [-0.7524, -0.1157] | 1.0764 (0.6913) | INSIDE-SPREAD |
| gap error 6.0 s (m) | -1.0404 / -0.4230 | -0.7317 | [-1.4875, -0.0357] | 3.3507 (1.4831) | INSIDE-SPREAD |
| time-gap error 2.0 s (s) | -0.0108 / -0.0085 | -0.0097 | [-0.0174, -0.0032] | 0.8769 (0.3180) | INSIDE-SPREAD |
| time-gap error over the scored steps (s) | -0.0541 / -0.0866 | -0.0703 | [-0.1504, -0.0091] | 1.7849 (0.8089) | INSIDE-SPREAD |

## Level table: every arm beside the logged driver (mean over seeds, two-level interval, T1-kin)

| arm (seeds) | U | C | F | bias_2.0 | bias_4.0 | bias_6.0 | abs_2.0 | abs_4.0 | abs_6.0 | tg_2.0 | tg_all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G (0,1,2,3,4,5) | 0.1296 [0.1007, 0.1582] | 0.0452 [0.0290, 0.0622] | 0.2160 [0.1679, 0.2690] | -0.2223 [-0.2556, -0.1892] | 1.2412 [0.7824, 1.7313] | 5.6287 [4.0030, 7.2245] | 0.4890 [0.4603, 0.5185] | 2.8195 [2.4725, 3.2163] | 8.5064 [7.4751, 9.5738] | 1.1895 [0.8884, 1.5269] | 2.3819 [1.7664, 3.0287] |
| LSDMC (0,1) | 0.0516 [0.0277, 0.0823] | 0.0093 [0.0000, 0.0261] | 0.4097 [0.3379, 0.4816] | 1.1278 [0.7191, 1.5873] | 5.4852 [3.9211, 7.1340] | 13.6403 [10.7057, 16.7188] | 1.5814 [1.2109, 2.0053] | 6.3777 [4.9718, 7.8509] | 14.6549 [12.0072, 17.5735] | 1.8397 [1.1472, 2.5956] | 3.6730 [2.6322, 4.7572] |
| LSDMG (0,1) | 0.0560 [0.0301, 0.0912] | 0.0093 [0.0000, 0.0261] | 0.4089 [0.3371, 0.4816] | 1.1127 [0.7039, 1.5713] | 5.4454 [3.8589, 7.0812] | 13.6024 [10.6442, 16.6801] | 1.5679 [1.1919, 1.9927] | 6.3741 [4.9779, 7.8531] | 14.6640 [12.0121, 17.5946] | 1.8386 [1.1472, 2.5930] | 3.6701 [2.6248, 4.7552] |
| LSDMH (0,1) | 0.0240 [0.0113, 0.0380] | 0.0016 [0.0000, 0.0051] | 0.5491 [0.4748, 0.6103] | 2.5501 [2.0222, 3.1061] | 9.0712 [7.3524, 10.7473] | 18.2690 [14.9898, 21.3254] | 2.8048 [2.2958, 3.3284] | 9.5860 [7.9233, 11.2412] | 18.9236 [15.7737, 21.8890] | 2.3426 [1.5567, 3.2238] | 5.3455 [4.1002, 6.7545] |
| LSDML (0,1) | 0.0810 [0.0470, 0.1248] | 0.0210 [0.0042, 0.0477] | 0.3512 [0.2702, 0.4347] | 0.4870 [0.1888, 0.8437] | 3.7180 [2.3627, 5.0573] | 11.0894 [8.5720, 13.5416] | 1.0176 [0.7520, 1.3406] | 4.8092 [3.7034, 5.9425] | 12.3563 [10.2458, 14.5408] | 1.5613 [0.9175, 2.2785] | 3.0208 [1.9841, 4.1141] |
| LSDMT (0,1) | 0.1165 [0.0746, 0.1633] | 0.0335 [0.0072, 0.0708] | 0.3271 [0.2509, 0.4032] | 0.2244 [-0.0011, 0.5060] | 3.1099 [2.0245, 4.2360] | 9.9479 [7.5280, 12.1979] | 0.8155 [0.6068, 1.0734] | 4.3006 [3.4383, 5.2380] | 11.4305 [9.5443, 13.5194] | 1.4996 [0.8583, 2.2313] | 2.7712 [1.8700, 3.7206] |
| LSDRH (0,1) | 0.1174 [0.0734, 0.1670] | 0.0335 [0.0072, 0.0708] | 0.3255 [0.2493, 0.4022] | 0.2189 [-0.0074, 0.5021] | 3.0880 [1.9867, 4.2275] | 9.9390 [7.5229, 12.1905] | 0.8106 [0.6022, 1.0655] | 4.2859 [3.4176, 5.2197] | 11.4028 [9.5253, 13.4774] | 1.4992 [0.8581, 2.2308] | 2.7695 [1.8682, 3.7196] |
| LSMC (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4058 [0.3276, 0.4795] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5297 [10.7494, 16.4204] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6582 [12.0491, 17.4352] | 1.7826 [1.3123, 2.2538] | 4.1701 [3.3443, 5.0752] |
| LSMG (0,1) | 0.0534 [0.0312, 0.0772] | 0.0070 [0.0000, 0.0177] | 0.4050 [0.3275, 0.4776] | 1.1719 [0.5946, 1.8411] | 5.6125 [4.1334, 7.2170] | 13.5090 [10.7492, 16.4188] | 1.6390 [1.0908, 2.2664] | 6.4847 [5.0902, 8.0342] | 14.6407 [12.0428, 17.4009] | 1.7826 [1.3123, 2.2538] | 4.1704 [3.3447, 5.0753] |
| LSMH (0,1) | 0.0214 [0.0121, 0.0331] | 0.0016 [0.0000, 0.0053] | 0.5319 [0.4595, 0.5982] | 2.6250 [1.9439, 3.3623] | 9.3066 [7.4985, 11.3251] | 18.8588 [15.4101, 22.0929] | 2.9031 [2.2414, 3.6215] | 9.8444 [8.0697, 11.8615] | 19.5090 [16.2516, 22.6837] | 2.2315 [1.7206, 2.7620] | 5.8429 [4.9456, 6.7637] |
| LSML (0,1) | 0.0765 [0.0471, 0.1121] | 0.0210 [0.0074, 0.0395] | 0.3318 [0.2356, 0.4329] | 0.4491 [0.1264, 0.8341] | 3.6209 [2.4838, 4.8864] | 10.2267 [7.5575, 13.1692] | 1.0507 [0.7526, 1.4125] | 4.7545 [3.8012, 5.8417] | 11.7850 [9.4929, 14.3969] | 1.4547 [1.0080, 1.9167] | 3.0853 [2.2624, 3.9878] |
| LSMR (0,1) | 0.1085 [0.0720, 0.1471] | 0.0358 [0.0177, 0.0574] | 0.2702 [0.1956, 0.3563] | 0.1507 [-0.1065, 0.5367] | 2.3214 [1.5134, 3.3066] | 7.4549 [5.4368, 9.5641] | 0.8256 [0.5876, 1.1818] | 3.9720 [3.2597, 4.9426] | 10.3847 [8.5795, 12.3461] | 1.4219 [0.9703, 1.8963] | 2.9042 [2.0605, 3.8330] |
| LSMT (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| LSRH (0,1) | 0.0845 [0.0527, 0.1232] | 0.0234 [0.0086, 0.0426] | 0.3100 [0.2259, 0.4009] | 0.3066 [0.0047, 0.6708] | 3.2004 [2.2391, 4.2372] | 9.5079 [7.0923, 11.8941] | 0.9418 [0.6696, 1.2884] | 4.3968 [3.5541, 5.3752] | 11.1165 [9.1719, 13.2105] | 1.4316 [0.9801, 1.9035] | 2.9745 [2.1468, 3.8918] |
| logged driver | U 0 by construction (windows where it has no unsafe step) | contact 0.0000 | F 0 | bias 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Logged driver, unrestricted over the 642 lead windows (same rule): unsafe-step share 0.1246, contact share 0.0000.

Four-family line: LONGITUDINAL: this read (distance keeping, T1-kin); LATERAL: not measured by this read; TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)

Limits: The lead does not react to the rolled ego (log replay): a gap below the bound is a statement about the plan against the logged lead, not about a collision in traffic. The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping. One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled. A separated interval is necessary and not sufficient, which is why the rule also asks for the six-seed spread of the baseline.
