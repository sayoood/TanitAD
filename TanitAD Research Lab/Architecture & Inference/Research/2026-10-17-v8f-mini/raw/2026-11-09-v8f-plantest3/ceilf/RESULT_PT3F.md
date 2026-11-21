# RESULT_PT3F: READ-END with readers fitted on every trained clip (N4565)
Card `PREREG_V8F_PT3F.md`. Tier **T0**, an oracle-goal ceiling with no world-model rollout and no planner, never a driving result. MEASURED: `RESULT_PT3F.json`, `runs/`. Paired EC-BOOT over the 71 SCORE episodes (994 windows), B 2,000. TACTICAL: not computed (n = 0, no decision output). STRATEGIC: PAUSED (PI item 25).

**Controls: all pass.**

* M2b_s0: control 1 speed_ridge_m4 +0.7551 against +0.7551, speed_mlp_m4 +0.7785 against +0.7785, yaw_mlp_m4 +0.5102 against +0.5102; controls 2 and 3 largest window difference c2_pt3m_tie_acc_mae 0.0e+00, c3_oracle_tie_acc_mae 0.0e+00, c3_hold_tie_acc_mae 0.0e+00, c2_pt3m_tie_yaw_mae 0.0e+00, c3_oracle_tie_yaw_mae 0.0e+00, c3_hold_tie_yaw_mae 0.0e+00
* M2b_s0: held-out R2 speed ridge +0.755, speed MLP +0.778 (deciding form: mlp); yaw MLP +0.510; N1053 yaw MLP at 1e-3 +0.570; absolute read error at knot 12 of F-DECIDE: speed 2.30 m/s, yaw rate 0.0318 rad/s
* M2b_s1: control 1 speed_ridge_m4 +0.6956 against +0.6956, speed_mlp_m4 +0.6876 against +0.6876, yaw_mlp_m4 +0.4633 against +0.4633; controls 2 and 3 largest window difference c2_pt3m_tie_acc_mae 0.0e+00, c3_oracle_tie_acc_mae 0.0e+00, c3_hold_tie_acc_mae 0.0e+00, c2_pt3m_tie_yaw_mae 0.0e+00, c3_oracle_tie_yaw_mae 0.0e+00, c3_hold_tie_yaw_mae 0.0e+00
* M2b_s1: held-out R2 speed ridge +0.696, speed MLP +0.688 (deciding form: ridge); yaw MLP +0.463; N1053 yaw MLP at 1e-3 +0.441; absolute read error at knot 12 of F-DECIDE: speed 2.57 m/s, yaw rate 0.0316 rad/s

## LONGITUDINAL (m/s^2; MAE against the 12 logged actions)
| seed | arm | HOLD | arm MAE | arm - HOLD | separated |
|---|---|---|---|---|---|
| M2b_s0 | F-DECIDE (deciding) | 0.3348 | 1.0049 [0.8545, 1.1710] | +0.6701 [+0.5138, +0.8426] | worse |
| M2b_s0 | F-OTHER (beside) | 0.3348 | 1.1500 [0.9819, 1.3257] | +0.8152 [+0.6429, +1.0024] | worse |
| M2b_s0 | N1053-LR1E3 (beside) | 0.3348 | 1.1510 [0.9938, 1.3277] | +0.8162 [+0.6495, +0.9964] | worse |
| M2b_s0 | PT3M (tie) | 0.3348 | 1.1510 [0.9938, 1.3277] | +0.8162 [+0.6495, +0.9964] | worse |
| M2b_s0 | F-DECIDE, read difference (beside; acceleration only) | 0.3348 | 0.8032 [0.7157, 0.9044] | +0.4684 [+0.3735, +0.5762] | worse |
| M2b_s0 | ORACLE-END (beside) | 0.3348 | 0.2161 [0.1855, 0.2472] | -0.1187 [-0.1364, -0.1014] | better |
| M2b_s1 | F-DECIDE (deciding) | 0.3348 | 1.0988 [0.9543, 1.2557] | +0.7640 [+0.6121, +0.9311] | worse |
| M2b_s1 | F-OTHER (beside) | 0.3348 | 1.1161 [0.9483, 1.3083] | +0.7813 [+0.6042, +0.9788] | worse |
| M2b_s1 | N1053-LR1E3 (beside) | 0.3348 | 1.1147 [0.9694, 1.2771] | +0.7799 [+0.6254, +0.9495] | worse |
| M2b_s1 | PT3M (tie) | 0.3348 | 1.1147 [0.9694, 1.2771] | +0.7799 [+0.6254, +0.9495] | worse |
| M2b_s1 | F-DECIDE, read difference (beside; acceleration only) | 0.3348 | 0.7739 [0.7211, 0.8319] | +0.4391 [+0.3725, +0.5082] | worse |
| M2b_s1 | ORACLE-END (beside) | 0.3348 | 0.2161 [0.1855, 0.2472] | -0.1187 [-0.1364, -0.1014] | better |

## LATERAL (rad/s; MAE against the 12 logged actions)
| seed | arm | HOLD | arm MAE | arm - HOLD | separated |
|---|---|---|---|---|---|
| M2b_s0 | F-DECIDE (deciding) | 0.0256 | 0.0228 [0.0181, 0.0279] | -0.0028 [-0.0061, +0.0002] | none |
| M2b_s0 | F-OTHER (beside) | 0.0256 | 0.0228 [0.0181, 0.0279] | -0.0028 [-0.0061, +0.0002] | none |
| M2b_s0 | N1053-LR1E3 (beside) | 0.0256 | 0.0255 [0.0202, 0.0313] | -0.0001 [-0.0034, +0.0027] | none |
| M2b_s0 | PT3M (tie) | 0.0256 | 0.0255 [0.0204, 0.0313] | -0.0001 [-0.0030, +0.0026] | none |
| M2b_s0 | F-DECIDE, read difference (beside; acceleration only) | 0.0256 | 0.0228 [0.0181, 0.0279] | -0.0028 [-0.0061, +0.0002] | none |
| M2b_s0 | ORACLE-END (beside) | 0.0256 | 0.0142 [0.0113, 0.0172] | -0.0114 [-0.0151, -0.0082] | better |
| M2b_s1 | F-DECIDE (deciding) | 0.0256 | 0.0228 [0.0180, 0.0280] | -0.0028 [-0.0058, +0.0001] | none |
| M2b_s1 | F-OTHER (beside) | 0.0256 | 0.0228 [0.0180, 0.0280] | -0.0028 [-0.0058, +0.0001] | none |
| M2b_s1 | N1053-LR1E3 (beside) | 0.0256 | 0.0258 [0.0207, 0.0316] | +0.0002 [-0.0029, +0.0033] | none |
| M2b_s1 | PT3M (tie) | 0.0256 | 0.0261 [0.0212, 0.0317] | +0.0005 [-0.0026, +0.0036] | none |
| M2b_s1 | F-DECIDE, read difference (beside; acceleration only) | 0.0256 | 0.0228 [0.0180, 0.0280] | -0.0028 [-0.0058, +0.0001] | none |
| M2b_s1 | ORACLE-END (beside) | 0.0256 | 0.0142 [0.0113, 0.0172] | -0.0114 [-0.0151, -0.0082] | better |

## Beside, no bar: at 2.4 s (EC-BOOT mean)
| seed | plan | ADE [m] | speed err [m/s] | heading err [rad] |
|---|---|---|---|---|
| M2b_s0 | hold | 0.2284 | 0.6903 | 0.0572 |
| M2b_s0 | oracle | 0.2066 | 0.0000 | 0.0296 |
| M2b_s0 | f_decide | 0.9958 | 2.3044 | 0.0499 |
| M2b_s0 | pt3m | 1.1272 | 2.6757 | 0.0565 |
| M2b_s1 | hold | 0.2284 | 0.6903 | 0.0572 |
| M2b_s1 | oracle | 0.2066 | 0.0000 | 0.0296 |
| M2b_s1 | f_decide | 1.0633 | 2.5318 | 0.0497 |
| M2b_s1 | pt3m | 1.0763 | 2.5560 | 0.0578 |

## Outcome per family (F-DECIDE, both seeds)
* **LONGITUDINAL: NOT-READ**
* **LATERAL: NOT-READ**
