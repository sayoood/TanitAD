# World Understanding Card — v8 flagship RH_s0..s2 (card ADD155, Phase A)

MEASURED, tier T0 (world-model diagnostics on logged frames, never driving performance); eval-141 deciding; three training seeds (values s0 / s1 / s2; the interval shown is the widest of the three per-seed 95 % episode-cluster bootstrap intervals, B 2,000). Four families: not applicable (no section scores a rollout). Labels applied by `wua_card.py` exactly as frozen in `PREREG_V8F_PROOF_ADD155.md`.

## A1 ENCODED — selectivity over the ego-only control (dR^2, dAUROC or d balanced accuracy; linear probe)

| variable | L1 | L2 | L3 | TOK-MEAN | TOK-GRID |
|---|---|---|---|---|---|
| `ag1_pos` (bar 0.10) | ENCODED-BEYOND-EGO (AM3)<br>+0.116 / +0.106 / +0.137<br>[+0.059, +0.190] | PARTIAL (AM3)<br>+0.073 / +0.086 / +0.091<br>[+0.027, +0.140] | PARTIAL (AM3)<br>+0.067 / +0.091 / +0.082<br>[+0.024, +0.141] | ENCODED-BEYOND-EGO (AM3)<br>+0.132 / +0.122 / +0.117<br>[+0.057, +0.196] | NOT-ENCODED (AM3)<br>+0.069 / +0.111 / +0.078<br>[-0.007, +0.173] |
| `lead_gap` (bar 0.10) | NOT-ENCODED (AM3)<br>+0.081 / +0.219 / +0.163<br>[-0.194, +0.379] | ENCODED-BEYOND-EGO (AM3)<br>+0.247 / +0.275 / +0.167<br>[+0.011, +0.433] | ENCODED-BEYOND-EGO (AM3)<br>+0.206 / +0.253 / +0.306<br>[+0.056, +0.436] | NOT-ENCODED (AM3)<br>+0.068 / +0.074 / +0.092<br>[-0.168, +0.296] | NOT-ENCODED (AM3)<br>+0.094 / +0.175 / +0.037<br>[-0.210, +0.338] |
| `ag1_vel` | INVALID (uncontrolled)<br>+0.070 / -0.014 / +0.093<br>[-0.117, +0.199] | INVALID (uncontrolled)<br>-0.029 / -0.097 / +0.077<br>[-0.289, +0.232] | INVALID (uncontrolled)<br>-0.039 / -0.078 / +0.088<br>[-0.383, +0.214] | INVALID (uncontrolled)<br>+0.007 / -0.037 / +0.027<br>[-0.192, +0.148] | INVALID (uncontrolled)<br>-0.059 / -0.125 / -0.068<br>[-0.321, +0.065] |
| `ag2_pos` | ENCODED-BEYOND-EGO (AM3)<br>+0.056 / +0.072 / +0.114<br>[+0.005, +0.165] | ENCODED-BEYOND-EGO (AM3)<br>+0.048 / +0.062 / +0.064<br>[+0.006, +0.118] | NOT-ENCODED (AM3)<br>+0.058 / +0.064 / +0.040<br>[-0.006, +0.113] | NOT-ENCODED (AM3)<br>+0.038 / +0.060 / +0.064<br>[-0.022, +0.126] | NOT-ENCODED (AM3)<br>-0.003 / +0.030 / +0.013<br>[-0.080, +0.092] |
| `ag3_pos` | INVALID (uncontrolled)<br>+0.008 / +0.038 / +0.071<br>[-0.066, +0.131] | INVALID (uncontrolled)<br>+0.006 / +0.007 / -0.005<br>[-0.063, +0.070] | INVALID (uncontrolled)<br>+0.010 / -0.025 / +0.001<br>[-0.073, +0.062] | INVALID (uncontrolled)<br>+0.019 / +0.055 / +0.040<br>[-0.061, +0.122] | INVALID (uncontrolled)<br>+0.000 / +0.046 / -0.010<br>[-0.084, +0.106] |
| `curv_ahead` | INVALID (uncontrolled)<br>-0.085 / -0.055 / -0.079<br>[-0.141, +0.001] | INVALID (uncontrolled)<br>-0.112 / -0.098 / -0.108<br>[-0.183, -0.033] | INVALID (uncontrolled)<br>-0.121 / -0.099 / -0.089<br>[-0.208, -0.016] | INVALID (uncontrolled)<br>-0.134 / -0.117 / -0.134<br>[-0.228, -0.047] | INVALID (uncontrolled)<br>-0.364 / -0.311 / -0.386<br>[-0.597, -0.198] |
| `gap_change_2s` | INVALID (uncontrolled)<br>-0.056 / -0.054 / -0.073<br>[-0.181, +0.009] | INVALID (uncontrolled)<br>-0.088 / -0.063 / -0.073<br>[-0.214, +0.015] | INVALID (uncontrolled)<br>-0.066 / -0.072 / -0.076<br>[-0.179, -0.004] | INVALID (uncontrolled)<br>-0.055 / -0.074 / -0.080<br>[-0.216, +0.018] | INVALID (uncontrolled)<br>-0.203 / -0.170 / -0.148<br>[-0.407, -0.031] |
| `lat_cls` | INVALID (uncontrolled)<br>+0.033 / +0.020 / +0.019<br>[-0.002, +0.065] | INVALID (uncontrolled)<br>+0.054 / +0.030 / +0.038<br>[+0.004, +0.111] | INVALID (uncontrolled)<br>+0.051 / +0.031 / +0.035<br>[+0.005, +0.104] | INVALID (uncontrolled)<br>+0.053 / +0.034 / +0.031<br>[-0.000, +0.111] | INVALID (uncontrolled)<br>+0.026 / +0.016 / +0.009<br>[-0.004, +0.057] |
| `lead_closing` | INVALID (uncontrolled)<br>-0.042 / -0.078 / -0.093<br>[-0.238, +0.065] | INVALID (uncontrolled)<br>-0.022 / -0.068 / -0.056<br>[-0.247, +0.112] | INVALID (uncontrolled)<br>-0.032 / -0.104 / -0.039<br>[-0.245, +0.082] | INVALID (uncontrolled)<br>-0.143 / -0.166 / -0.232<br>[-0.508, +0.021] | INVALID (uncontrolled)<br>-0.235 / -0.176 / -0.126<br>[-0.492, -0.006] |
| `lead_present` | ENCODED-BEYOND-EGO (AM3)<br>+0.255 / +0.265 / +0.264<br>[+0.177, +0.345] | ENCODED-BEYOND-EGO (AM3)<br>+0.247 / +0.239 / +0.266<br>[+0.159, +0.343] | ENCODED-BEYOND-EGO (AM3)<br>+0.212 / +0.225 / +0.245<br>[+0.131, +0.322] | ENCODED-BEYOND-EGO (AM3)<br>+0.253 / +0.246 / +0.267<br>[+0.163, +0.348] | ENCODED-BEYOND-EGO (AM3)<br>+0.207 / +0.219 / +0.223<br>[+0.133, +0.298] |
| `lead_rel_acc` | INVALID (uncontrolled)<br>-0.024 / -0.020 / -0.011<br>[-0.049, +0.002] | INVALID (uncontrolled)<br>-0.020 / -0.020 / -0.010<br>[-0.043, +0.004] | INVALID (uncontrolled)<br>-0.011 / -0.022 / -0.011<br>[-0.045, +0.002] | INVALID (uncontrolled)<br>-0.048 / -0.042 / -0.037<br>[-0.092, -0.017] | INVALID (uncontrolled)<br>-0.105 / -0.098 / -0.099<br>[-0.181, -0.057] |
| `lon7` | INVALID (uncontrolled)<br>+0.006 / +0.007 / +0.016<br>[-0.056, +0.058] | INVALID (uncontrolled)<br>+0.020 / -0.006 / +0.021<br>[-0.070, +0.062] | INVALID (uncontrolled)<br>+0.015 / -0.017 / +0.026<br>[-0.079, +0.066] | INVALID (uncontrolled)<br>+0.008 / +0.008 / +0.016<br>[-0.057, +0.059] | INVALID (uncontrolled)<br>+0.013 / +0.016 / +0.023<br>[-0.050, +0.066] |
| `tl_red_vs_green` | INVALID (uncontrolled)<br>-0.154 / -0.154 / -0.267<br>[-0.484, +0.115] | INVALID (uncontrolled)<br>-0.036 / -0.077 / -0.062<br>[-0.307, +0.175] | INVALID (uncontrolled)<br>+0.026 / -0.039 / -0.053<br>[-0.316, +0.219] | INVALID (uncontrolled)<br>-0.140 / -0.193 / -0.254<br>[-0.518, +0.133] | INVALID (uncontrolled)<br>+0.076 / +0.037 / +0.091<br>[-0.094, +0.245] |

Temporal MLP (state at t-4..t + ego history vs MLP on ego history):

| variable | L1 | L2 | L3 | TOK-MEAN |
|---|---|---|---|---|
| `ag1_pos` | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.527 / -0.599 / -0.583<br>[-0.832, -0.346] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.523 / -0.583 / -0.666<br>[-0.896, -0.367] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.561 / -0.654 / -0.440<br>[-0.851, -0.293] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.501 / -0.451 / -0.642<br>[-0.932, -0.284] |
| `gap_change_2s` | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-1.189 / -1.441 / -0.791<br>[-2.613, -0.445] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.892 / -1.185 / -1.340<br>[-2.284, -0.451] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.954 / -1.186 / -0.981<br>[-2.190, -0.502] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-1.991 / -1.930 / -1.574<br>[-3.566, -0.814] |
| `lead_closing` | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.267 / -0.412 / -0.395<br>[-0.878, -0.013] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.289 / -0.455 / -0.339<br>[-0.898, +0.007] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.507 / -0.751 / -0.353<br>[-1.340, -0.109] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.377 / -0.588 / -0.486<br>[-1.049, -0.094] |
| `lead_gap` | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.148 / -0.636 / -0.224<br>[-1.162, +0.144] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>+0.028 / -0.390 / -0.282<br>[-0.805, +0.235] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.357 / -0.176 / -0.441<br>[-0.978, +0.168] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.278 / -0.239 / -0.213<br>[-0.686, +0.076] |
| `lead_rel_acc` | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.416 / -0.478 / -0.609<br>[-1.092, -0.231] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.769 / -0.658 / -0.539<br>[-1.388, -0.321] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.655 / -0.826 / -0.356<br>[-1.670, -0.212] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.773 / -0.770 / -0.891<br>[-1.510, -0.436] |
| `speed` | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.005 / -0.007 / -0.007<br>[-0.010, -0.004] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.014 / -0.010 / -0.011<br>[-0.018, -0.008] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.008 / -0.008 / -0.007<br>[-0.011, -0.005] | INSTRUMENT-LIMITED (card rule: NOT-ENCODED)<br>-0.015 / -0.021 / -0.028<br>[-0.039, -0.011] |

Linear temporal probe P-TLIN (Amendment 4; state at t-4..t + ego history vs ego history; two-part rule with 19 derangements):

| variable | L1 | L2 | L3 | TOK-MEAN |
|---|---|---|---|---|
| `ag1_pos` | ENCODED-BEYOND-EGO (AM4)<br>+0.112 / +0.117 / +0.131<br>[+0.064, +0.191] | NOT-ENCODED (AM4)<br>+0.021 / +0.047 / +0.043<br>[-0.033, +0.113] | NOT-ENCODED (AM4)<br>+0.040 / +0.082 / +0.049<br>[-0.011, +0.145] | NOT-ENCODED (AM4)<br>+0.078 / +0.070 / +0.052<br>[-0.021, +0.150] |
| `gap_change_2s` | NOT-ENCODED (AM4)<br>-0.098 / -0.110 / -0.133<br>[-0.294, -0.021] | NOT-ENCODED (AM4)<br>-0.232 / -0.181 / -0.222<br>[-0.493, -0.051] | NOT-ENCODED (AM4)<br>-0.198 / -0.177 / -0.214<br>[-0.392, -0.068] | NOT-ENCODED (AM4)<br>-0.135 / -0.184 / -0.225<br>[-0.482, -0.024] |
| `lead_closing` | NOT-ENCODED (AM4)<br>-0.103 / -0.139 / -0.183<br>[-0.403, +0.045] | NOT-ENCODED (AM4)<br>-0.112 / -0.184 / -0.157<br>[-0.467, +0.088] | NOT-ENCODED (AM4)<br>-0.176 / -0.233 / -0.180<br>[-0.445, -0.013] | NOT-ENCODED (AM4)<br>-0.337 / -0.397 / -0.482<br>[-0.945, -0.087] |
| `lead_gap` | NOT-ENCODED (AM4)<br>+0.018 / +0.135 / +0.098<br>[-0.287, +0.332] | NOT-ENCODED (AM4)<br>+0.220 / +0.148 / +0.009<br>[-0.221, +0.395] | NOT-ENCODED (AM4)<br>+0.101 / +0.174 / +0.237<br>[-0.088, +0.390] | NOT-ENCODED (AM4)<br>-0.200 / -0.143 / -0.076<br>[-0.617, +0.240] |
| `lead_rel_acc` | NOT-ENCODED (AM4)<br>-0.046 / -0.037 / -0.026<br>[-0.082, -0.009] | NOT-ENCODED (AM4)<br>-0.061 / -0.065 / -0.046<br>[-0.115, -0.023] | NOT-ENCODED (AM4)<br>-0.037 / -0.044 / -0.040<br>[-0.080, -0.013] | NOT-ENCODED (AM4)<br>-0.130 / -0.108 / -0.103<br>[-0.237, -0.051] |

## A2 IMAGINED — ego-matched counterfactual pairs (pair-discrimination AUC of gap change over 2 s)

| scorer | label | AUC s0 / s1 / s2 | minus EGO s0 / s1 / s2 | SAME-pair AUC s0 / s1 / s2 |
|---|---|---|---|---|
| IMAG-L1 | BLIND | 0.572 [0.448, 0.706] / 0.530 [0.413, 0.644] / 0.528 [0.400, 0.663] | +0.091 [-0.039, +0.235] / +0.049 [-0.089, +0.188] / +0.047 [-0.111, +0.214] | 0.545 / 0.573 / 0.524 |
| IMAG-L2 | BLIND | 0.469 [0.353, 0.602] / 0.518 [0.406, 0.642] / 0.479 [0.357, 0.612] | -0.011 [-0.168, +0.148] / +0.037 [-0.116, +0.204] / -0.001 [-0.158, +0.166] | 0.528 / 0.524 / 0.492 |
| ENC-L1 | BLIND | 0.601 [0.450, 0.738] / 0.634 [0.511, 0.737] / 0.570 [0.416, 0.692] | +0.120 [-0.024, +0.250] / +0.153 [+0.016, +0.277] / +0.089 [-0.066, +0.214] | 0.431 / 0.500 / 0.427 |
| ORACLE-L1 | BLIND | 0.568 [0.456, 0.690] / 0.572 [0.457, 0.689] / 0.584 [0.473, 0.701] | +0.087 [-0.087, +0.263] / +0.091 [-0.062, +0.243] / +0.103 [-0.061, +0.266] | 0.423 / 0.480 / 0.443 |
| EGO | (control) | 0.480 [0.380, 0.584] / 0.480 [0.380, 0.584] / 0.480 [0.380, 0.584] | - / - / - | 0.553 / 0.553 / 0.553 |

Controls: {"c5_same_contains_half": true, "c6_ego_auc_le_0.60": true, "c7_swap": true, "n_pairs": [{"DIFF": 999, "SAME": 247}, {"DIFF": 999, "SAME": 247}, {"DIFF": 999, "SAME": 247}]}

## A3 IMAGINED — skill over copy-last (probe on the rollout vs on the present state)

| level, k (h) | variable | label | skill s0 / s1 / s2 | oracle R^2 | copy - imag MSE [CI] per seed |
|---|---|---|---|---|---|
| L1 k10 (2.0 s) | `ag1_pos` | COPIES | +0.004 / -0.001 / +0.011 | 0.14 / 0.13 / 0.16 | +0.65 [-3.27, +4.41] / -0.187 [-3.6, +3.06] / +1.99 [-1.51, +5.33] |
| L1 k10 (2.0 s) | `lead_closing` | UNREADABLE | -0.017 / -0.018 / +0.019 | -0.02 / -0.04 / -0.07 | -0.197 [-0.714, +0.351] / -0.221 [-0.654, +0.205] / +0.239 [-0.384, +1.02] |
| L1 k10 (2.0 s) | `lead_gap` | COPIES | +0.031 / +0.012 / +0.013 | 0.20 / 0.26 / 0.20 | +11.9 [-20, +44.9] / +4.36 [-25.6, +32] / +5.03 [-18.5, +29.9] |
| L1 k10 (2.0 s) | `speed` | COPIES | +0.079 / +0.012 / +0.043 | 0.72 / 0.75 / 0.75 | +1.48 [-0.576, +3.39] / +0.202 [-1.18, +1.56] / +0.692 [-0.863, +2.24] |
| L1 k5 (1.0 s) | `ag1_pos` | COPIES | -0.000 / -0.001 / +0.005 | 0.15 / 0.13 / 0.16 | -0.0616 [-2.44, +2.14] / -0.196 [-2.02, +1.61] / +0.838 [-1.19, +2.76] |
| L1 k5 (1.0 s) | `lead_closing` | UNREADABLE | -0.009 / -0.008 / +0.013 | -0.02 / -0.04 / -0.06 | -0.107 [-0.385, +0.151] / -0.0929 [-0.318, +0.138] / +0.158 [-0.198, +0.581] |
| L1 k5 (1.0 s) | `lead_gap` | COPIES | +0.039 / +0.014 / +0.018 | 0.22 / 0.27 / 0.21 | +14.3 [-1.82, +31.1] / +4.99 [-10.1, +20.7] / +6.86 [-6.24, +20.5] |
| L1 k5 (1.0 s) | `speed` | COPIES | +0.067 / +0.016 / +0.050 | 0.71 / 0.74 / 0.75 | +1.2 [-0.0463, +2.39] / +0.251 [-0.546, +1.03] / +0.768 [-0.121, +1.65] |
| L2 k15 (6.0 s) | `ag1_pos` | SEEDS-DISAGREE | +0.061 / +0.021 / +0.040 | 0.11 / 0.10 / 0.13 | +12.3 [+6.98, +17.8] / +4.16 [-2.95, +11.2] / +7.74 [+1.47, +13.9] |
| L2 k15 (6.0 s) | `lead_closing` | UNREADABLE | +0.085 / +0.057 / +0.025 | -0.12 / -0.06 / -0.01 | +1.13 [-0.346, +2.76] / +0.763 [-0.822, +2.64] / +0.323 [-1.95, +3.05] |
| L2 k15 (6.0 s) | `lead_gap` | COPIES | -0.034 / -0.099 / +0.096 | 0.30 / 0.29 / 0.17 | -13.5 [-84.2, +48.9] / -41.9 [-114, +17.8] / +42.9 [-8.19, +95] |
| L2 k15 (6.0 s) | `speed` | IMAGINES | +0.319 / +0.307 / +0.310 | 0.73 / 0.76 / 0.74 | +8.22 [+5.07, +11.5] / +7.61 [+4.74, +10.7] / +8.2 [+4.8, +11.6] |
| L2 k5 (2.0 s) | `ag1_pos` | SEEDS-DISAGREE | +0.030 / +0.014 / +0.020 | 0.10 / 0.11 / 0.13 | +5.51 [+2.95, +8.18] / +2.58 [-0.743, +5.86] / +3.65 [+0.723, +6.52] |
| L2 k5 (2.0 s) | `lead_closing` | UNREADABLE | +0.021 / -0.002 / +0.001 | -0.08 / -0.05 / -0.01 | +0.262 [-0.312, +0.852] / -0.0236 [-0.728, +0.699] / +0.0166 [-0.733, +0.766] |
| L2 k5 (2.0 s) | `lead_gap` | COPIES | -0.026 / -0.033 / +0.037 | 0.30 / 0.30 / 0.19 | -8.58 [-32.3, +13.9] / -11.5 [-35.7, +11.6] / +14 [-3.41, +31.4] |
| L2 k5 (2.0 s) | `speed` | IMAGINES | +0.129 / +0.119 / +0.115 | 0.71 / 0.74 / 0.73 | +2.48 [+1.44, +3.63] / +2.1 [+1.08, +3.24] / +2.11 [+0.869, +3.39] |
| L3 k15 (12.0 s) | `ag1_pos` | SEEDS-DISAGREE | +0.088 / +0.021 / +0.036 | 0.11 / 0.11 / 0.11 | +18.3 [+10.7, +26.6] / +4.32 [-6.49, +15.8] / +7.09 [-1.25, +16.5] |
| L3 k15 (12.0 s) | `lead_closing` | UNREADABLE | +0.170 / +0.162 / +0.210 | -0.05 / -0.12 / -0.10 | +2.71 [+0.465, +5.25] / +2.72 [-0.442, +6.2] / +3.67 [-0.411, +9.01] |
| L3 k15 (12.0 s) | `lead_gap` | SEEDS-DISAGREE | +0.086 / +0.147 / +0.107 | 0.22 / 0.26 / 0.27 | +47.1 [-68.6, +163] / +94.4 [+7.47, +200] / +55.4 [-21.8, +141] |
| L3 k15 (12.0 s) | `speed` | IMAGINES | +0.506 / +0.444 / +0.450 | 0.80 / 0.82 / 0.78 | +17 [+11, +23.8] / +16 [+9.87, +22.6] / +16.8 [+9.97, +24.5] |
| L3 k5 (4.0 s) | `ag1_pos` | IMAGINES | +0.049 / +0.033 / +0.029 | 0.10 / 0.12 / 0.11 | +9.55 [+5.32, +14] / +6.18 [+1.89, +10.6] / +5.48 [+1.45, +9.69] |
| L3 k5 (4.0 s) | `lead_closing` | UNREADABLE | +0.105 / +0.079 / +0.069 | -0.07 / -0.12 / 0.01 | +1.41 [+0.389, +2.67] / +1.09 [-0.114, +2.44] / +0.849 [-0.451, +2.49] |
| L3 k5 (4.0 s) | `lead_gap` | COPIES | +0.044 / +0.041 / +0.056 | 0.21 / 0.24 / 0.31 | +17.6 [-18.8, +53.4] / +16.8 [-18.2, +55.4] / +19.8 [-7.43, +48.1] |
| L3 k5 (4.0 s) | `speed` | IMAGINES | +0.230 / +0.190 / +0.185 | 0.77 / 0.77 / 0.75 | +4.26 [+2.69, +5.77] / +3.66 [+1.96, +5.27] / +3.73 [+1.51, +6.01] |

## A4 IMAGINED under the ego's own action — slope of the imagined response against the kinematic expectation (1 = exact)

| level, k (h) | offset | variable | label | slope [CI] s0 / s1 / s2 |
|---|---|---|---|---|
| L1 k10 (2.0 s) | lat | `ag1_pos` | WEAK | +0.07 [+0.06, +0.09] / +0.02 [+0.02, +0.03] / +0.06 [+0.06, +0.07] |
| L1 k10 (2.0 s) | lon | `lead_gap` | WRONG-SIGN | -0.49 [-0.64, -0.35] / -1.13 [-1.29, -0.98] / -0.67 [-0.81, -0.55] |
| L1 k10 (2.0 s) | lon | `speed` | WEAK | +0.34 [+0.30, +0.37] / +0.45 [+0.42, +0.49] / +0.36 [+0.33, +0.39] |
| L2 k5 (2.0 s) | lat | `ag1_pos` | WEAK | +0.01 [+0.00, +0.02] / +0.04 [+0.03, +0.06] / +0.03 [+0.02, +0.04] |
| L2 k5 (2.0 s) | lon | `lead_gap` | WRONG-SIGN | -0.96 [-1.08, -0.83] / -0.95 [-1.06, -0.83] / -0.70 [-0.80, -0.59] |
| L2 k5 (2.0 s) | lon | `speed` | WEAK | +0.39 [+0.36, +0.42] / +0.37 [+0.34, +0.40] / +0.35 [+0.32, +0.38] |
| L3 k3 (2.4 s) | lat | `ag1_pos` | INERT | +0.00 [-0.00, +0.00] / +0.01 [+0.01, +0.02] / +0.03 [+0.02, +0.03] |
| L3 k3 (2.4 s) | lon | `lead_gap` | WRONG-SIGN | -0.64 [-0.74, -0.54] / -0.74 [-0.89, -0.60] / -0.53 [-0.62, -0.44] |
| L3 k3 (2.4 s) | lon | `speed` | WEAK | +0.33 [+0.31, +0.35] / +0.34 [+0.31, +0.37] / +0.32 [+0.30, +0.35] |

## A5 USED

| seed | abs dv LEAD - abs dv CONTROL (mean-token mask) [CI] | signed dv LEAD | signed dv CONTROL | C-CENTRE abs dv LEAD - centre-matched [CI] | signed dv centre-matched |
|---|---|---|---|---|---|
| s0 | +0.388 [+0.271, +0.517] | +0.245 [+0.071, +0.415] | +0.022 [+0.002, +0.046] | +0.163 [+0.050, +0.296] | -0.023 [-0.114, +0.065] |
| s1 | +0.416 [+0.311, +0.534] | +0.310 [+0.154, +0.467] | +0.012 [+0.003, +0.024] | +0.143 [+0.024, +0.266] | -0.036 [-0.142, +0.061] |
| s2 | +0.293 [+0.224, +0.368] | +0.178 [+0.056, +0.294] | +0.016 [+0.003, +0.029] | +0.067 [-0.012, +0.148] | -0.051 [-0.147, +0.033] |

A5 label: {"FINAL": "USED", "c11_all_seeds_pass": true} ; c10 pooled lead minus control logit drop +0.938 [+0.757, +1.102]
Amendment 2 name: A5-CENTRE-NOT-EXCLUDED

Controls: {"A1_s0": ["c1_shuffle|L1|lead_gap", "c1_shuffle|L1|ag1_pos", "c1_shuffle|L1|speed", "c1_shuffle|TOK-GRID|lead_gap", "c1_shuffle|TOK-GRID|ag1_pos", "c1_shuffle|TOK-GRID|speed"], "A1_s1": ["c1_shuffle|L1|lead_gap", "c1_shuffle|L1|ag1_pos", "c1_shuffle|L1|speed", "c1_shuffle|TOK-GRID|lead_gap", "c1_shuffle|TOK-GRID|ag1_pos", "c1_shuffle|TOK-GRID|speed"], "A1_s2": ["c1_shuffle|L1|lead_gap", "c1_shuffle|L1|ag1_pos", "c1_shuffle|L1|speed", "c1_shuffle|TOK-GRID|lead_gap", "c1_shuffle|TOK-GRID|ag1_pos", "c1_shuffle|TOK-GRID|speed"], "A4_c9": [{"L1|k10": 0.0, "L2|k5": 0.0, "L3|k3": 0.0}, {"L1|k10": 0.0, "L2|k5": 0.0, "L3|k3": 0.0}, {"L1|k10": 0.0, "L2|k5": 0.0, "L3|k3": 0.0}]}
