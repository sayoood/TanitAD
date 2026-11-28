
#### G3 S6: paired delta (arm - A REAL), episode-cluster bootstrap B=2000, n 1253 windows / 139 episodes

| family | cell | REAL (mean) | A-rep (seed 1) | B HOLD | C NAVSIM-ST | D NAVSIM-NT |
|---|---|---|---|---|---|---|
| ADE | ade_m | 2.6886 | +0.049 [+0.008, +0.093] SEP | +0.016 [-0.032, +0.071] | +0.040 [-0.022, +0.102] | +0.039 [-0.005, +0.089] |
| ADE | fde_m | 6.5725 | +0.124 [+0.013, +0.238] SEP | +0.044 [-0.066, +0.170] | +0.102 [-0.051, +0.254] | +0.091 [-0.021, +0.211] |
| LONGITUDINAL | LON_speed_mae_mps | 0.9714 | +0.0140 [-0.0024, +0.0307] | +0.0031 [-0.0099, +0.0157] | +0.0159 [-0.0012, +0.0337] | +0.0071 [-0.0075, +0.0222] |
| LONGITUDINAL | LON_along_mae_m | 2.3155 | +0.0436 [+0.0009, +0.0854] SEP | +0.0136 [-0.0219, +0.0505] | +0.0384 [-0.0082, +0.0841] | +0.0278 [-0.0125, +0.0688] |
| LONGITUDINAL | LON_accel_mae_mps2 | 0.4398 | +0.0038 [-0.0016, +0.0092] | +0.0022 [-0.0019, +0.0063] | +0.0064 [+0.0011, +0.0122] SEP | +0.0032 [-0.0013, +0.0076] |
| LATERAL | LAT_cross_mae_m | 0.8345 | +0.0201 [-0.0024, +0.0490] | +0.0091 [-0.0288, +0.0510] | +0.0064 [-0.0397, +0.0505] | +0.0249 [-0.0031, +0.0608] |
| LATERAL | LAT_heading_mae_deg | 3.4063 | +0.111 [-0.018, +0.277] | +0.118 [-0.051, +0.327] | +0.114 [-0.062, +0.319] | +0.165 [+0.014, +0.369] SEP |
| LATERAL | LAT_yaw_rate_mae_radps_valid | 0.0287 | +0.00050 [-0.00030, +0.00150] | +0.00060 [-0.00000, +0.00130] | +0.00060 [-0.00010, +0.00140] | +0.00100 [+0.00020, +0.00200] SEP |
| TACTICAL | TAC_traj_lat_correct | 0.8795 | -0.0024 [-0.0104, +0.0056] | +0.0016 [-0.0056, +0.0088] | +0.0024 [-0.0048, +0.0096] | +0.0032 [-0.0048, +0.0104] |
| TACTICAL | TAC_traj_lon_correct | 0.6018 | +0.0016 [-0.0159, +0.0184] | -0.0056 [-0.0175, +0.0064] | -0.0104 [-0.0271, +0.0056] | -0.0064 [-0.0184, +0.0056] |
| STRATEGIC | - | NOT APPLICABLE (strategic layer OFF, PI R5), n = 0 | | | | |

#### G3 S2: paired delta (arm - A REAL), episode-cluster bootstrap B=2000, n 1253 windows / 139 episodes

| family | cell | REAL (mean) | A-rep (seed 1) | B HOLD | C NAVSIM-ST | D NAVSIM-NT |
|---|---|---|---|---|---|---|
| ADE | ade_m | 0.3051 | +0.003 [-0.002, +0.008] | +0.002 [-0.003, +0.007] | +0.003 [-0.002, +0.009] | +0.004 [+0.000, +0.009] SEP |
| ADE | fde_m | 0.6547 | +0.010 [-0.001, +0.021] | +0.005 [-0.006, +0.016] | +0.008 [-0.006, +0.021] | +0.009 [-0.000, +0.021] |
| LONGITUDINAL | LON_speed_mae_mps | 0.2813 | +0.0033 [-0.0025, +0.0091] | +0.0017 [-0.0019, +0.0052] | +0.0041 [-0.0005, +0.0085] | +0.0032 [-0.0007, +0.0076] |
| LONGITUDINAL | LON_along_mae_m | 0.2499 | +0.0030 [-0.0023, +0.0082] | +0.0018 [-0.0013, +0.0047] | +0.0034 [-0.0004, +0.0069] | +0.0027 [-0.0007, +0.0063] |
| LONGITUDINAL | LON_accel_mae_mps2 | 0.3927 | +0.0061 [-0.0006, +0.0130] | +0.0026 [-0.0020, +0.0073] | +0.0060 [+0.0002, +0.0121] SEP | +0.0050 [+0.0003, +0.0102] SEP |
| LATERAL | LAT_cross_mae_m | 0.1075 | -0.0002 [-0.0030, +0.0029] | +0.0016 [-0.0027, +0.0067] | +0.0012 [-0.0036, +0.0065] | +0.0027 [-0.0003, +0.0072] |
| LATERAL | LAT_heading_mae_deg | 1.7053 | +0.015 [-0.023, +0.066] | +0.016 [-0.033, +0.072] | +0.017 [-0.034, +0.072] | +0.030 [-0.003, +0.078] |
| LATERAL | LAT_yaw_rate_mae_radps_valid | 0.0187 | +0.00020 [-0.00050, +0.00120] | +0.00020 [-0.00060, +0.00130] | +0.00020 [-0.00070, +0.00120] | +0.00050 [-0.00010, +0.00140] |
| TACTICAL | TAC_traj_lat_correct | 0.9593 | -0.0032 [-0.0072, +0.0000] | +0.0008 [-0.0032, +0.0048] | +0.0024 [-0.0024, +0.0072] | -0.0016 [-0.0064, +0.0024] |
| TACTICAL | TAC_traj_lon_correct | 0.8356 | -0.0112 [-0.0208, -0.0016] SEP | +0.0000 [-0.0064, +0.0064] | -0.0064 [-0.0152, +0.0016] | -0.0024 [-0.0104, +0.0048] |
| STRATEGIC | - | NOT APPLICABLE (strategic layer OFF, PI R5), n = 0 | | | | |

#### G3 LEAD-VIOL S6 (lead windows 392; at-risk = human keeps >= 1.0 s; paired X - A REAL)

| arm | LEAD-VIOL share (all lead) | share_of_at_risk | n_viol/n_lead | progress ratio | paired at-risk delta |
|---|---|---|---|---|---|
| A REAL | 0.043367 | 0.059649 | 17/None | 1.101 | - |
| A-rep (seed 1) | 0.043367 | 0.059649 | 17/None | 1.107 | +0.0000 [-0.0170, +0.0154] |
| B HOLD | 0.043367 | 0.059649 | 17/None | 1.096 | +0.0000 [-0.0102, +0.0102] |
| C NAVSIM-ST | 0.043367 | 0.059649 | 17/None | 1.100 | +0.0000 [-0.0136, +0.0141] |
| D NAVSIM-NT | 0.043367 | 0.059649 | 17/None | 1.098 | +0.0000 [-0.0102, +0.0102] |
| ha0_ext | 0.035714 | 0.049123 | 14/None | 0.942 | - |
| stop | 0.0 | 0.0 | 0/None | 0.000 | - |

#### G3 LEAD-VIOL S2 (lead windows 392; at-risk = human keeps >= 1.0 s; paired X - A REAL)

| arm | LEAD-VIOL share (all lead) | share_of_at_risk | n_viol/n_lead | progress ratio | paired at-risk delta |
|---|---|---|---|---|---|
| A REAL | 0.005102 | 0.006472 | 2/None | 1.093 | - |
| A-rep (seed 1) | 0.005102 | 0.006472 | 2/None | 1.091 | +0.0000 [+0.0000, +0.0000] |
| B HOLD | 0.002551 | 0.003236 | 1/None | 1.114 | -0.0032 [-0.0101, +0.0000] |
| C NAVSIM-ST | 0.002551 | 0.003236 | 1/None | 1.118 | -0.0032 [-0.0101, +0.0000] |
| D NAVSIM-NT | 0.002551 | 0.003236 | 1/None | 1.085 | -0.0032 [-0.0101, +0.0000] |
| ha0_ext | 0.005102 | 0.006472 | 2/None | 0.950 | - |
| stop | 0.0 | 0.0 | 0/None | 0.000 | - |

#### G3 PREREG s5 verdict (S6 primaries)

* **A-rep (seed 1): MIXED** -- P1 +0.0493 [+0.0081,+0.0929] (margin 0.0538, floor 0.0493) INDETERMINATE; P2 +0.0436 [+0.0009,+0.0854] (margin 0.0463, floor 0.0436) INDETERMINATE; P3 +0.0201 [-0.0024,+0.0490] (margin 0.0167, floor 0.0201) INDETERMINATE; P4 +0.0016 [-0.0159,+0.0184] (margin 0.0100, floor 0.0016) INDETERMINATE; P5 +0.0000 [-0.0170,+0.0154] (margin 0.0100, floor 0.0000) INDETERMINATE
* **B HOLD: MIXED** -- P1 +0.0156 [-0.0316,+0.0706] (margin 0.0538, floor 0.0493) INDETERMINATE; P2 +0.0136 [-0.0219,+0.0505] (margin 0.0463, floor 0.0436) INDETERMINATE; P3 +0.0091 [-0.0288,+0.0510] (margin 0.0167, floor 0.0201) INDETERMINATE; P4 -0.0056 [-0.0175,+0.0064] (margin 0.0100, floor 0.0016) INDETERMINATE; P5 +0.0000 [-0.0102,+0.0102] (margin 0.0100, floor 0.0000) INDETERMINATE
* **C NAVSIM-ST: MIXED** -- P1 +0.0401 [-0.0218,+0.1016] (margin 0.0538, floor 0.0493) INDETERMINATE; P2 +0.0384 [-0.0082,+0.0841] (margin 0.0463, floor 0.0436) INDETERMINATE; P3 +0.0064 [-0.0397,+0.0505] (margin 0.0167, floor 0.0201) INDETERMINATE; P4 -0.0104 [-0.0271,+0.0056] (margin 0.0100, floor 0.0016) INDETERMINATE; P5 +0.0000 [-0.0136,+0.0141] (margin 0.0100, floor 0.0000) INDETERMINATE
* **D NAVSIM-NT: MIXED** -- P1 +0.0389 [-0.0052,+0.0886] (margin 0.0538, floor 0.0493) INDETERMINATE; P2 +0.0278 [-0.0125,+0.0688] (margin 0.0463, floor 0.0436) INDETERMINATE; P3 +0.0249 [-0.0031,+0.0608] (margin 0.0167, floor 0.0201) INDETERMINATE; P4 -0.0064 [-0.0184,+0.0056] (margin 0.0100, floor 0.0016) INDETERMINATE; P5 +0.0000 [-0.0102,+0.0102] (margin 0.0100, floor 0.0000) INDETERMINATE
