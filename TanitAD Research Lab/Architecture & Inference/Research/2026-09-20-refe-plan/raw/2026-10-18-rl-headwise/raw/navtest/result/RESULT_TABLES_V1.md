## Levels (mean x100, all 12,146 tokens)

| point | PDMS | NC | DAC | EP | TTC | C | DDC | zero-PDMS picks |
|---|---|---|---|---|---|---|---|---|
| BASE | 88.2388 | 98.94 | 96.57 | 81.19 | 95.63 | 99.97 | 97.66 | 530 |
| C0A | 87.3604 | 98.98 | 96.35 | 79.35 | 95.64 | 99.98 | 98.69 | 549 |
| R0B | 88.7855 | 98.69 | 96.53 | 83.00 | 95.23 | 99.98 | 97.55 | 566 |
| V1 | 88.3339 | 98.82 | 96.45 | 81.72 | 95.50 | 99.97 | 97.52 | 560 |

**Each point minus BASE (paired, x100)**

| point | PDMS | NC | DAC | EP | TTC | C | DDC | better/worse/tied | n logs |
|---|---|---|---|---|---|---|---|---|---|
| V_vs_BASE | **+0.10 [-0.07, +0.26]** | -0.12 [-0.24, -0.02] ADV | -0.12 [-0.26, +0.01] | +0.53 [+0.37, +0.69] | -0.13 [-0.32, +0.06] | +0.00 [+0.00, +0.00] | -0.14 [-0.20, -0.09] ADV | 2181/418/9547 | 136 |
| R0B_vs_BASE | **+0.55 [+0.24, +0.84]** | -0.25 [-0.45, -0.06] ADV | -0.04 [-0.26, +0.15] | +1.81 [+1.50, +2.13] | -0.40 [-0.71, -0.09] ADV | +0.02 [+0.00, +0.05] | -0.12 [-0.21, -0.03] ADV | 4308/951/6887 | 136 |
| C0A_vs_BASE | **-0.88 [-1.47, -0.14]** | +0.04 [-0.18, +0.24] | -0.21 [-0.76, +0.48] | -1.84 [-2.58, -1.04] ADV | +0.02 [-0.44, +0.45] | +0.02 [+0.00, +0.05] | +1.03 [+0.72, +1.37] | 3626/4088/4432 | 136 |

**Lever view (paired, x100)**

| point | PDMS | NC | DAC | EP | TTC | C | DDC | better/worse/tied | n logs |
|---|---|---|---|---|---|---|---|---|---|
| V-R0B | **-0.45 [-0.72, -0.17]** | +0.13 [-0.05, +0.32] | -0.07 [-0.26, +0.12] | -1.28 [-1.56, -1.00] ADV | +0.26 [+0.01, +0.53] | -0.02 [-0.05, +0.00] | -0.02 [-0.12, +0.08] | 1041/3531/7574 | 136 |
| V-C0A | **+0.97 [+0.14, +1.62]** | -0.16 [-0.40, +0.08] | +0.10 [-0.71, +0.69] | +2.37 [+1.52, +3.16] | -0.15 [-0.60, +0.33] | -0.02 [-0.05, +0.00] | -1.17 [-1.54, -0.83] ADV | 4434/3166/4546 | 136 |

## DEPLOYABLE-SAFE (vs BASE)

* V_vs_BASE: **False** {'PDMS_lo_gt_0': False, 'NC_hi_ge_0': False, 'DAC_hi_ge_0': True, 'TTC_hi_ge_0': True}
* R0B_vs_BASE: **False** {'PDMS_lo_gt_0': True, 'NC_hi_ge_0': False, 'DAC_hi_ge_0': True, 'TTC_hi_ge_0': False}
* C0A_vs_BASE: **False** {'PDMS_lo_gt_0': False, 'NC_hi_ge_0': True, 'DAC_hi_ge_0': True, 'TTC_hi_ge_0': True}

## OUTCOME: **N-BOTH** (adverse: ['NC']); gates {'harness_PASS': True, 'rows_12146': True, 'identity_a_same_pick_rows_equal': True}

## Families

**LONGITUDINAL** (families6, x = point estimate [episode-cluster 95 % interval], unpaired)

| metric | BASE | R0B | V1 |
|---|---|---|---|
| longitudinal.speed_mae_mps | 0.7397 [0.7112, 0.7713] | 0.7307 [0.7001, 0.7617] | 0.7265 [0.697, 0.757] |
| longitudinal.speed_bias_mps | -0.2303 [-0.3258, -0.1369] | -0.0638 [-0.1561, 0.0282] | -0.1705 [-0.265, -0.0759] |
| longitudinal.along_mae_m | 1.034 [0.9879, 1.083] | 1.036 [0.9902, 1.084] | 1.02 [0.9714, 1.068] |
| longitudinal.along_final_bias_m | -0.8263 [-1.19, -0.4778] | -0.195 [-0.5492, 0.1529] | -0.6057 [-0.9737, -0.2535] |
| longitudinal.accel_mae_mps2 | 0.5193 [0.5032, 0.5385] | 0.508 [0.4913, 0.5269] | 0.5112 [0.4952, 0.5295] |
| longitudinal.target_speed_acc.within_1.0_mps | 0.7428 [0.7279, 0.7567] | 0.749 [0.735, 0.7634] | 0.7492 [0.7352, 0.7634] |

**LATERAL** (families6, x = point estimate [episode-cluster 95 % interval], unpaired)

| metric | BASE | R0B | V1 |
|---|---|---|---|
| lateral.heading_mae_deg | 2.532 [2.314, 2.729] | 2.48 [2.272, 2.67] | 2.533 [2.312, 2.737] |
| lateral.yaw_rate_mae_degps | 2.983 [2.8, 3.165] | 2.925 [2.739, 3.107] | 2.967 [2.784, 3.149] |
| lateral.curvature_mae_1pm | 0.01987 [0.01834, 0.02138] | 0.01891 [0.01741, 0.02035] | 0.01955 [0.01806, 0.02101] |
| lateral.cross_mae_m | 0.3779 [0.3513, 0.4045] | 0.3739 [0.3469, 0.3998] | 0.3777 [0.3504, 0.4036] |
| lateral.cross_final_mae_m | 0.865 [0.7945, 0.9332] | 0.8616 [0.7891, 0.93] | 0.8676 [0.7964, 0.9361] |

**TACTICAL** (families6; rows WITHOUT an interval, reported not tested)

| decision | BASE | R0B | V1 |
|---|---|---|---|
| lateral_decision | acc 0.9252, kappa 0.8628 | acc 0.9266, kappa 0.8654 | acc 0.9253, kappa 0.8632 |
| longitudinal_decision | acc 0.6789, kappa 0.5157 | acc 0.6872, kappa 0.5292 | acc 0.6824, kappa 0.5214 |
| maneuver_5way_collapsed | acc 0.7831, kappa 0.7182 | acc 0.7877, kappa 0.7256 | acc 0.7854, kappa 0.7218 |

**STRATEGIC: PAUSED** (PI 2026-09-19, PI_DECISION_QUEUE item 25) and UNAVAILABLE in NAVSIM (no strategic decision is scored). Reported as paused, never as a pass.