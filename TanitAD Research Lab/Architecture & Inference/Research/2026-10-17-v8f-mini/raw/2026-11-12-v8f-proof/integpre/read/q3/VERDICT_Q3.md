# VERDICT Q3: the four metric families at T1-kin, SPATIAL stage 1

Card: PREREG_V8F_PROOF_ADD12.md, section 4, Q3. Tier: T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class: MEASURED (ours, T1-kin dumps).
Estimator: two-level bootstrap seeds then episodes (B 2000, rng 0), paired window differences A minus B; engine of hierrep_read.py unchanged.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16).

## Controls (first)

* q5_same_windows: PASS  {"n_dumps": 38}
* q1_window_counts: PASS  {"card": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}, "read": {"n_windows": 2399, "n_episodes": 141, "n_crit": 1305, "n_bulk": 1094}}
* q2_self_is_zero: PASS  {"max_abs_D": 0.0, "arm": "IRQ_RLAm"}
* q3_swap_is_minus: PASS  {"max_abs_dev": 0.0}
* q4_planted_regression: PASS  {"planted_mps": 0.8079339416045079, "D_read": 0.8079339416045082, "verdict": "WORSE", "name": "COST-LONGITUDINAL"}
* ALL_PASS: True

## Run-to-run spread of the baseline

Arm G, seeds present [0, 1, 2, 3, 4, 5] of requested [0, 1, 2, 3, 4, 5]; complete for the card (six seeds): True. A difference inside the spread is not an effect.

## IRQ RLAm minus IR0 RLFT

Seeds read [0, 1, 2, 3] of [0, 1, 2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1446 / +0.1860 / +0.1688 / +0.0340 | +0.1334 | [+0.0375, +0.2292] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.6289 / +0.6520 / +0.6377 / +0.4600 | +0.5947 | [+0.4079, +0.7943] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0335 / +0.0355 / -0.0359 / +0.0222 | -0.0029 | [-0.0421, +0.0394] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.8389 / +0.7428 / +1.0808 / +1.2378 | +0.9751 | [+0.6390, +1.3090] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0327 / +0.0116 / +0.0271 / +0.0252 | +0.0242 | [+0.0130, +0.0370] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +2.5521 / +2.2930 / +3.4983 / +1.6378 | +2.4953 | [+1.1190, +3.9174] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0083 / +0.0208 / +0.0096 / -0.0154 | +0.0058 | [-0.0105, +0.0214] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0017 / +0.0263 / +0.0484 / +0.0371 | +0.0283 | [+0.0073, +0.0483] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | +0.0083 / +0.0188 / +0.0254 / +0.0021 | +0.0137 | [+0.0009, +0.0266] | 0.0492 (0.0263) | INSIDE-SPREAD |
| TACTICAL:ade_tac_m | +0.7914 / +0.8834 / +0.7813 / +0.5976 | +0.7634 | [+0.5129, +1.0329] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2808 / +0.3907 / +0.3397 / +0.3648 | +0.3440 | [+0.1968, +0.5139] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.9156 / +0.9294 / +0.8307 / +0.5293 | +0.8013 | [+0.5073, +1.1149] | 0.3535 | WORSE |
| ade6_m|all | +0.5703 / +0.6364 / +0.5636 / +0.4398 | +0.5525 | [+0.3748, +0.7439] | 0.4002 | WORSE |
| ade_tac_m|all | +0.7914 / +0.8834 / +0.7813 / +0.5976 | +0.7634 | [+0.5129, +1.0329] | 0.5955 | WORSE |

## IRQ RLAm minus IR0 RLAm

Seeds read [0, 1, 2, 3] of [0, 1, 2, 3]. **Q3: COST-LONGITUDINAL COST-TACTICAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.1554 / +0.1195 / +0.1369 / +0.0677 | +0.1199 | [+0.0341, +0.2092] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.6178 / +0.6429 / +0.6279 / +0.5090 | +0.5994 | [+0.4239, +0.7936] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0332 / +0.0113 / -0.0469 / -0.0175 | -0.0216 | [-0.0553, +0.0121] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.9134 / +0.7694 / +0.8729 / +0.7597 | +0.8288 | [+0.5896, +1.0693] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0265 / +0.0255 / +0.0191 / +0.0333 | +0.0261 | [+0.0171, +0.0369] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.8331 / +1.7722 / +2.4465 / +1.6584 | +1.9276 | [+0.7969, +3.0956] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | -0.0104 / +0.0125 / +0.0088 / +0.0025 | +0.0033 | [-0.0090, +0.0155] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0025 / +0.0304 / +0.0300 / +0.0067 | +0.0174 | [+0.0009, +0.0338] | 0.1042 (0.0534) | INSIDE-SPREAD |
| TACTICAL:joint_err_rate | -0.0083 / +0.0158 / +0.0188 / +0.0067 | +0.0082 | [-0.0044, +0.0211] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.7789 / +0.8540 / +0.7560 / +0.6277 | +0.7541 | [+0.5102, +1.0171] | 0.5955 (0.2199) | WORSE |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.3034 / +0.3818 / +0.3350 / +0.3436 | +0.3409 | [+0.1927, +0.5102] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.8694 / +0.8958 / +0.7997 / +0.6018 | +0.7917 | [+0.5255, +1.0904] | 0.3535 | WORSE |
| ade6_m|all | +0.5615 / +0.6162 / +0.5469 / +0.4613 | +0.5465 | [+0.3740, +0.7355] | 0.4002 | WORSE |
| ade_tac_m|all | +0.7789 / +0.8540 / +0.7560 / +0.6277 | +0.7541 | [+0.5102, +1.0171] | 0.5955 | WORSE |

## IRQ RLAm minus IRL RLAm

Seeds read [0, 1, 2, 3] of [0, 1, 2, 3]. **Q3: COST-LONGITUDINAL**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0981 / +0.1274 / +0.1237 / +0.0707 | +0.1050 | [+0.0539, +0.1584] | 0.2693 (0.1381) | INSIDE-SPREAD |
| LONGITUDINAL:along_mae_m | +0.4001 / +0.4142 / +0.4036 / +0.3046 | +0.3806 | [+0.2765, +0.4912] | 0.3530 (0.2400) | WORSE |
| LATERAL:cross_mae_m | -0.0069 / +0.0203 / +0.0097 / +0.0120 | +0.0088 | [-0.0138, +0.0323] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.4080 / +0.5747 / +0.5513 / +0.5603 | +0.5236 | [+0.3365, +0.7160] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0201 / +0.0179 / +0.0123 / +0.0218 | +0.0180 | [+0.0099, +0.0276] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.0715 / +1.5335 / +1.9511 / +1.4387 | +1.4987 | [+0.6277, +2.4463] | 15.8479 (7.7515) | INSIDE-SPREAD |
| TACTICAL:lon_err_rate | +0.0017 / +0.0150 / +0.0063 / +0.0054 | +0.0071 | [-0.0005, +0.0152] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0013 / +0.0163 / +0.0208 / +0.0038 | +0.0105 | [-0.0002, +0.0218] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0042 / +0.0096 / +0.0125 / +0.0046 | +0.0056 | [-0.0023, +0.0137] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.5248 / +0.5625 / +0.5335 / +0.4067 | +0.5069 | [+0.3609, +0.6610] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.2041 / +0.2720 / +0.2582 / +0.2670 | +0.2503 | [+0.1438, +0.3667] | 0.4517 | INSIDE-SPREAD |
| ade6_m|bulk | +0.5714 / +0.5546 / +0.5204 / +0.3249 | +0.4928 | [+0.3004, +0.7025] | 0.3535 | WORSE |
| ade6_m|all | +0.3716 / +0.4009 / +0.3778 / +0.2934 | +0.3609 | [+0.2574, +0.4697] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.5248 / +0.5625 / +0.5335 / +0.4067 | +0.5069 | [+0.3609, +0.6610] | 0.5955 | INSIDE-SPREAD |

## IRQ RLAm minus IRQ RLFT

Seeds read [0, 1, 2, 3] of [0, 1, 2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0008 / +0.0443 / +0.0100 / -0.0806 | -0.0068 | [-0.0582, +0.0422] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0016 / -0.0272 / -0.0121 / -0.1321 | -0.0425 | [-0.1143, +0.0228] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0221 / +0.0148 / +0.0098 / +0.0389 | +0.0103 | [-0.0149, +0.0361] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.0519 / +0.0590 / +0.2376 / +0.4866 | +0.1828 | [-0.0729, +0.4396] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0130 / -0.0104 / -0.0131 / -0.0074 | -0.0045 | [-0.0162, +0.0090] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.5226 / +0.4329 / -0.5499 / -0.6334 | -0.0569 | [-1.0013, +0.8659] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0200 / +0.0075 / +0.0104 / -0.0175 | +0.0051 | [-0.0104, +0.0193] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0046 / +0.0000 / +0.0129 / +0.0317 | +0.0100 | [-0.0048, +0.0263] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0113 / +0.0054 / +0.0158 / -0.0050 | +0.0069 | [-0.0035, +0.0168] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | -0.0047 / -0.0265 / -0.0115 / -0.1525 | -0.0488 | [-0.1399, +0.0353] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0539 / +0.0023 / -0.0226 / -0.0130 | -0.0218 | [-0.0718, +0.0257] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0517 / -0.0422 / +0.0022 / -0.2180 | -0.0516 | [-0.1855, +0.0671] | 0.3535 | OPEN |
| ade6_m|all | -0.0057 / -0.0180 / -0.0113 / -0.1065 | -0.0354 | [-0.0984, +0.0228] | 0.4002 | OPEN |
| ade_tac_m|all | -0.0047 / -0.0265 / -0.0115 / -0.1525 | -0.0488 | [-0.1399, +0.0353] | 0.5955 | OPEN |

## IR0 RLAm minus IR0 RLFT

Seeds read [0, 1, 2, 3] of [0, 1, 2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | -0.0108 / +0.0666 / +0.0319 / -0.0336 | +0.0135 | [-0.0280, +0.0588] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0111 / +0.0091 / +0.0097 / -0.0489 | -0.0047 | [-0.0391, +0.0243] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | -0.0003 / +0.0243 / +0.0111 / +0.0396 | +0.0187 | [-0.0001, +0.0385] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | -0.1843 / +0.0309 / +0.2182 / +0.5952 | +0.1650 | [-0.1203, +0.4699] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0062 / -0.0144 / +0.0063 / -0.0075 | -0.0023 | [-0.0131, +0.0071] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.1903 / +0.5327 / +0.4176 / -0.1544 | +0.2466 | [-0.5207, +1.0957] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0188 / +0.0083 / +0.0008 / -0.0179 | +0.0025 | [-0.0120, +0.0168] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | -0.0008 / -0.0042 / +0.0183 / +0.0304 | +0.0109 | [-0.0047, +0.0265] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0167 / +0.0029 / +0.0067 / -0.0046 | +0.0054 | [-0.0042, +0.0152] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0125 / +0.0295 / +0.0253 / -0.0301 | +0.0093 | [-0.0299, +0.0452] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | -0.0225 / +0.0089 / +0.0047 / +0.0212 | +0.0031 | [-0.0273, +0.0330] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0461 / +0.0336 / +0.0310 / -0.0725 | +0.0096 | [-0.0445, +0.0574] | 0.3535 | OPEN |
| ade6_m|all | +0.0088 / +0.0202 / +0.0167 / -0.0215 | +0.0060 | [-0.0209, +0.0305] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0125 / +0.0295 / +0.0253 / -0.0301 | +0.0093 | [-0.0299, +0.0452] | 0.5955 | OPEN |

## IRL RLAm minus IR0 RLAm

Seeds read [0, 1, 2, 3] of [0, 1, 2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0573 / -0.0079 / +0.0131 / -0.0031 | +0.0149 | [-0.0449, +0.0762] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2177 / +0.2287 / +0.2243 / +0.2043 | +0.2188 | [+0.1079, +0.3484] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0263 / -0.0090 / -0.0566 / -0.0294 | -0.0303 | [-0.0563, -0.0063] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.5277 / +0.1649 / +0.3673 / +0.1986 | +0.3146 | [+0.1257, +0.5276] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0122 / +0.0148 / +0.0119 / +0.0164 | +0.0138 | [+0.0091, +0.0196] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +1.0369 / -0.3137 / +0.6546 / +0.1885 | +0.3916 | [-0.4122, +1.1592] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0121 / -0.0025 / +0.0025 / -0.0029 | -0.0038 | [-0.0115, +0.0051] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0013 / +0.0142 / +0.0092 / +0.0029 | +0.0069 | [-0.0020, +0.0163] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0042 / +0.0063 / +0.0063 / +0.0021 | +0.0026 | [-0.0040, +0.0099] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2541 / +0.2915 / +0.2225 / +0.2209 | +0.2473 | [+0.0977, +0.4175] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0993 / +0.1098 / +0.0768 / +0.0766 | +0.0906 | [-0.0117, +0.2068] | 0.4517 | OPEN |
| ade6_m|bulk | +0.2980 / +0.3412 / +0.2793 / +0.2769 | +0.2989 | [+0.1626, +0.4615] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.1899 / +0.2153 / +0.1691 / +0.1679 | +0.1856 | [+0.0765, +0.3107] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2541 / +0.2915 / +0.2225 / +0.2209 | +0.2473 | [+0.0977, +0.4175] | 0.5955 | INSIDE-SPREAD |

## IRL RLFT minus IR0 RLFT

Seeds read [0, 1, 2, 3] of [0, 1, 2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0615 / +0.0029 / +0.0295 / -0.0005 | +0.0234 | [-0.0337, +0.0830] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.2251 / +0.2442 / +0.2305 / +0.1923 | +0.2230 | [+0.1111, +0.3557] | 0.3530 (0.2400) | INSIDE-SPREAD |
| LATERAL:cross_mae_m | -0.0318 / -0.0147 / -0.0516 / -0.0245 | -0.0306 | [-0.0528, -0.0088] | 0.3667 (0.1723) | INSIDE-SPREAD |
| LATERAL:heading_mae_deg | +0.4936 / +0.2690 / +0.2725 / +0.2014 | +0.3091 | [+0.1509, +0.4959] | 4.5570 (1.9414) | INSIDE-SPREAD |
| LATERAL:curvature_mae_1pm | +0.0101 / +0.0136 / +0.0139 / +0.0123 | +0.0125 | [+0.0082, +0.0178] | 0.1528 (0.0760) | INSIDE-SPREAD |
| LATERAL:yaw_rate_mae_degps | +0.9495 / -0.1115 / +0.5876 / +0.1670 | +0.3981 | [-0.3518, +1.1330] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | -0.0067 / -0.0004 / -0.0054 / -0.0029 | -0.0039 | [-0.0100, +0.0027] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0004 / +0.0113 / +0.0046 / +0.0063 | +0.0056 | [-0.0022, +0.0146] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | -0.0013 / +0.0038 / -0.0004 / +0.0013 | +0.0008 | [-0.0043, +0.0066] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.2566 / +0.3025 / +0.2414 / +0.2104 | +0.2527 | [+0.1040, +0.4243] | 0.5955 (0.2199) | INSIDE-SPREAD |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.1102 / +0.1332 / +0.0831 / +0.0752 | +0.1004 | [-0.0035, +0.2205] | 0.4517 | OPEN |
| ade6_m|bulk | +0.2885 / +0.3294 / +0.2993 / +0.2628 | +0.2950 | [+0.1582, +0.4555] | 0.3535 | INSIDE-SPREAD |
| ade6_m|all | +0.1915 / +0.2227 / +0.1817 / +0.1607 | +0.1892 | [+0.0810, +0.3159] | 0.4002 | INSIDE-SPREAD |
| ade_tac_m|all | +0.2566 / +0.3025 / +0.2414 / +0.2104 | +0.2527 | [+0.1040, +0.4243] | 0.5955 | INSIDE-SPREAD |

## IR0 RLAm minus IPox RLAm

Seeds read [0, 1, 2, 3] of [0, 1, 2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.3535 | OPEN |
| ade6_m|all | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.5955 | OPEN |

## IR0 RLFT minus IPox RLFT

Seeds read [0, 1, 2, 3] of [0, 1, 2, 3]. **Q3: NO-COST**

| cell (all windows; every row is an error, below 0 is better) | per seed | D | two-level interval | spread max (median) | rule |
|---|---|---|---|---|---|
| LONGITUDINAL:speed_mae_mps | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.2693 (0.1381) | OPEN |
| LONGITUDINAL:along_mae_m | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.3530 (0.2400) | OPEN |
| LATERAL:cross_mae_m | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.3667 (0.1723) | OPEN |
| LATERAL:heading_mae_deg | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 4.5570 (1.9414) | OPEN |
| LATERAL:curvature_mae_1pm | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1528 (0.0760) | OPEN |
| LATERAL:yaw_rate_mae_degps | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 15.8479 (7.7515) | OPEN |
| TACTICAL:lon_err_rate | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0838 (0.0204) | OPEN |
| TACTICAL:lat_err_rate | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.1042 (0.0534) | OPEN |
| TACTICAL:joint_err_rate | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.0492 (0.0263) | OPEN |
| TACTICAL:ade_tac_m | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.5955 (0.2199) | OPEN |

Beside (not in the name):

| cell | per seed | D | two-level interval | spread max | rule |
|---|---|---|---|---|---|
| ade6_m|critical | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.4517 | OPEN |
| ade6_m|bulk | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.3535 | OPEN |
| ade6_m|all | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.4002 | OPEN |
| ade_tac_m|all | +0.0000 / +0.0000 / +0.0000 / +0.0000 | +0.0000 | [+0.0000, +0.0000] | 0.5955 | OPEN |

## Arm levels (mean over windows)

| arm | ADE6 all (m) | ADE6 critical (m) | speed MAE (m/s) | cross-track MAE (m) | joint class error rate |
|---|---|---|---|---|---|
| G_s0 | 3.0920 | 3.6802 | 1.9118 | 1.2271 | 0.7386 |
| G_s1 | 2.8385 | 3.2400 | 1.7317 | 1.3217 | 0.7040 |
| G_s2 | 3.2029 | 3.6918 | 1.9674 | 1.4299 | 0.7366 |
| G_s3 | 2.8027 | 3.2461 | 1.7737 | 1.1494 | 0.7349 |
| G_s4 | 2.9384 | 3.3460 | 1.7226 | 1.5161 | 0.6895 |
| G_s5 | 2.8064 | 3.2591 | 1.6981 | 1.1967 | 0.7303 |
| IPox_RLAm_s0 | 2.7508 | 3.2052 | 1.6757 | 1.2574 | 0.7257 |
| IPox_RLAm_s1 | 2.7484 | 3.2450 | 1.8212 | 1.1837 | 0.6336 |
| IPox_RLAm_s2 | 2.6990 | 3.2300 | 1.5717 | 1.2703 | 0.6173 |
| IPox_RLAm_s3 | 2.8184 | 3.2943 | 1.7745 | 1.2825 | 0.6820 |
| IPox_RLFT_s0 | 2.7420 | 3.2277 | 1.6865 | 1.2578 | 0.7090 |
| IPox_RLFT_s1 | 2.7283 | 3.2361 | 1.7547 | 1.1595 | 0.6307 |
| IPox_RLFT_s2 | 2.6823 | 3.2253 | 1.5398 | 1.2592 | 0.6107 |
| IPox_RLFT_s3 | 2.8400 | 3.2731 | 1.8081 | 1.2428 | 0.6865 |
| IR0_RLAm_s0 | 2.7508 | 3.2052 | 1.6757 | 1.2574 | 0.7257 |
| IR0_RLAm_s1 | 2.7484 | 3.2450 | 1.8212 | 1.1837 | 0.6336 |
| IR0_RLAm_s2 | 2.6990 | 3.2300 | 1.5717 | 1.2703 | 0.6173 |
| IR0_RLAm_s3 | 2.8184 | 3.2943 | 1.7745 | 1.2825 | 0.6820 |
| IR0_RLFT_s0 | 2.7420 | 3.2277 | 1.6865 | 1.2578 | 0.7090 |
| IR0_RLFT_s1 | 2.7283 | 3.2361 | 1.7547 | 1.1595 | 0.6307 |
| IR0_RLFT_s2 | 2.6823 | 3.2253 | 1.5398 | 1.2592 | 0.6107 |
| IR0_RLFT_s3 | 2.8400 | 3.2731 | 1.8081 | 1.2428 | 0.6865 |
| IRL_RLAm_s0 | 2.9407 | 3.3044 | 1.7331 | 1.2311 | 0.7216 |
| IRL_RLAm_s1 | 2.9637 | 3.3548 | 1.8133 | 1.1747 | 0.6398 |
| IRL_RLAm_s2 | 2.8682 | 3.3068 | 1.5848 | 1.2136 | 0.6236 |
| IRL_RLAm_s3 | 2.9864 | 3.3708 | 1.7714 | 1.2530 | 0.6840 |
| IRL_RLFT_s0 | 2.9335 | 3.3379 | 1.7480 | 1.2260 | 0.7078 |
| IRL_RLFT_s1 | 2.9509 | 3.3693 | 1.7576 | 1.1448 | 0.6344 |
| IRL_RLFT_s2 | 2.8640 | 3.3084 | 1.5693 | 1.2076 | 0.6103 |
| IRL_RLFT_s3 | 3.0007 | 3.3483 | 1.8076 | 1.2184 | 0.6878 |
| IRQ_RLAm_s0 | 3.3123 | 3.5085 | 1.8312 | 1.2242 | 0.7174 |
| IRQ_RLAm_s1 | 3.3646 | 3.6268 | 1.9407 | 1.1950 | 0.6494 |
| IRQ_RLAm_s2 | 3.2459 | 3.5650 | 1.7085 | 1.2233 | 0.6361 |
| IRQ_RLAm_s3 | 3.2798 | 3.6378 | 1.8421 | 1.2650 | 0.6886 |
| IRQ_RLFT_s0 | 3.3181 | 3.5624 | 1.8320 | 1.2464 | 0.7061 |
| IRQ_RLFT_s1 | 3.3827 | 3.6245 | 1.8965 | 1.1802 | 0.6440 |
| IRQ_RLFT_s2 | 3.2572 | 3.5876 | 1.6985 | 1.2135 | 0.6203 |
| IRQ_RLFT_s3 | 3.3863 | 3.6508 | 1.9227 | 1.2261 | 0.6936 |

## Not measured by this read

* LONGITUDINAL headway and time gap: GAP (R14a) no per-window lead block in the T1-kin dumps and none on the pod
* TACTICAL goal-setting (stop-goal decision): GAP (R14b) no per-arm goal label file
* STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and this reader has no strategic instrument; PI ruling of 2026-11-16)

Limits: one training run per arm per seed; the two-level interval resamples the seeds present, and with two seeds it is wide by construction;
a separated interval is necessary and not sufficient, which is why the rule also asks for the spread of the baseline seeds.
