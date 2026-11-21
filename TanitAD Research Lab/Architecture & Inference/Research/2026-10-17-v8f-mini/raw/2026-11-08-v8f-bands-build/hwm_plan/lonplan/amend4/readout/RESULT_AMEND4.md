# RESULT_AMEND4 (skeleton generated from RESULT_AMEND4.json; the MM writes the reading)

Tier T0 (oracle goal): a world-model diagnostic, never driving performance. Estimator: paired episode-cluster bootstrap, B 2,000 seed 0 (another draw of EPISODES only); F_I answers another INFERENCE run.

## Labels (section 6)

| arm | s0 | s1 | label | effective (next step applies) | 2-6 s band | HOLD-collapsed | abs dH < 2 F_I |
|---|---|---|---|---|---|---|---|
| P | NULL | HARM | L2IMG-SEEDS-DISAGREE | L2IMG-HARM | BAND-HURTS (BAND-HURTS / BAND-HURTS) | {'s0': False, 's1': False} | {'s0': False, 's1': False} |
| P23 | HARM | HARM | L23IMG-HARM | L23IMG-HARM | BAND-HURTS (BAND-HURTS / BAND-HURTS) | {'s0': False, 's1': False} | {'s0': False, 's1': False} |

Base solver of the RA read: **A0**. Multiplicity guard applies: False.

## Chassis R35_s0: 994 windows, 71 episodes (windows sha256 371c753bcc3f)

Deciding cells, 0-2 s, with the family tables in the contrast files named below. Rows: delta [lo, hi] label.

### P_A0  (c_P_A0_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.0295 | [-0.0635, +0.0038] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0443 | [-0.1066, +0.0164] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.1083 | [-0.0706, +0.3254] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0243 | [-0.0143, +0.0685] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.4634 | [-0.1281, +1.1388] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1340 | [+0.0912, +0.1738] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0754 | [-0.1072, +0.2638] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.3468 | [-0.8514, +0.1377] | none | 89 | 20 |
| LONGITUDINAL | 2-6s | tg_err | -0.5370 | [-1.7648, +0.1490] | none | 65 | 19 |
| LONGITUDINAL | 2-6s | ttc_err | -0.7561 | [-2.2153, +0.9412] | none | 89 | 20 |
| LONGITUDINAL | full6s | acc_mae | +0.0795 | [+0.0489, +0.1095] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0754 | [-0.1072, +0.2638] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.2019 | [-0.4598, +0.8921] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.1209 | [-0.0021, +0.2736] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +0.6401 | [-0.8271, +2.1721] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0043 | [-0.0000, +0.0089] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | -0.0082 | [-0.0169, +0.0005] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0002 | [-0.0005, +0.0009] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0985 | [-0.2804, +0.0422] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0405 | [-0.1059, +0.0145] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0248 | [+0.0193, +0.0298] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0187 | [+0.0021, +0.0346] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0033 | [+0.0020, +0.0047] | worse | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.4307 | [-1.5330, +0.3565] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.3196 | [-0.9335, +0.1461] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0180 | [+0.0136, +0.0219] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0187 | [+0.0021, +0.0346] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0023 | [+0.0013, +0.0034] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | -0.4307 | [-1.5330, +0.3565] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.2266 | [-0.6575, +0.0959] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0040 | [-0.0473, +0.0392] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | -0.0262 | [-0.0724, +0.0171] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0091 | [-0.0493, +0.0302] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0835 | [+0.0332, +0.1318] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.1278 | [+0.0724, +0.1821] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0302 | [-0.0221, +0.0825] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0161 | [-0.0312, +0.0604] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0493 | [-0.0060, +0.0996] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0000 | [-0.0503, +0.0503] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.0833 | [-0.2000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -4.9549 | [-12.8947, +4.4758] | none | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0456 | [+0.0152, +0.0775] | higher | 856 | 69 |

X - HOLD (the CEILING cell and its family):

| family | band | metric | X - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.9356 | [+0.8673, +1.0032] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.8698 | [+0.7531, +0.9838] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.2118 | [+0.6152, +2.0344] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1722 | [+0.0957, +0.2805] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +3.7540 | [+2.3911, +5.3722] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.3920 | [+0.3112, +0.4741] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.0710 | [-0.4296, +0.2875] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +1.5270 | [+0.5777, +2.4322] | worse | 115 |
| LONGITUDINAL | 2-6s | tg_err | +0.7453 | [-0.3235, +2.6731] | none | 90 |
| LONGITUDINAL | 2-6s | ttc_err | +3.2942 | [+1.3044, +5.5334] | worse | 115 |
| LONGITUDINAL | full6s | acc_mae | +0.5732 | [+0.5110, +0.6366] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.0710 | [-0.4296, +0.2875] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.9630 | [+0.9454, +3.1420] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.3076 | [+0.1625, +0.4906] | worse | 182 |
| LONGITUDINAL | full6s | ttc_err | +3.9582 | [+2.2926, +5.7848] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0981 | [+0.0905, +0.1059] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.1000 | [+0.0881, +0.1120] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0158 | [+0.0117, +0.0207] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2166 | [+1.0127, +1.4324] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7582 | [+0.6757, +0.8372] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0713 | [+0.0615, +0.0808] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0571 | [+0.0226, +0.0903] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0106 | [+0.0073, +0.0141] | worse | 967 |
| LATERAL | 2-6s | xtrack | +4.7649 | [+3.6093, +6.0040] | worse | 994 |
| LATERAL | 2-6s | ade | +4.1334 | [+3.4529, +4.8418] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0802 | [+0.0733, +0.0868] | worse | 994 |
| LATERAL | full6s | head_err | +0.0571 | [+0.0226, +0.0903] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0131 | [+0.0094, +0.0171] | worse | 974 |
| LATERAL | full6s | xtrack | +4.7649 | [+3.6093, +6.0040] | worse | 994 |
| LATERAL | full6s | ade | +3.0083 | [+2.5332, +3.5035] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4507 | [+0.3994, +0.5000] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2445 | [+0.1982, +0.2928] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3692 | [+0.3159, +0.4225] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1911 | [+0.1127, +0.2737] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.2103 | [+0.1479, +0.2676] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.1127 | [+0.0322, +0.1972] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.1962 | [+0.1288, +0.2646] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2646 | [+0.1942, +0.3330] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0815 | [+0.0070, +0.1570] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.5000 | [-1.0000, +0.2000] | none | 8 |
| STRATEGIC | full6s | s_adh_deg | -51.6388 | [-96.6433, -6.6344] | worse | 8 |
| STRATEGIC | full6s | s_false | -0.0399 | [-0.0953, +0.0118] | none | 853 |

### P23_A0  (c_P23_A0_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.0121 | [-0.0477, +0.0222] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0025 | [-0.0813, +0.0738] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0482 | [-0.1672, +0.2232] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0160 | [-0.0269, +0.0596] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.5997 | [+0.0131, +1.2654] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1505 | [+0.1050, +0.1917] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.0644 | [-0.1152, +0.2558] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.6836 | [-2.1523, +0.2911] | none | 80 | 20 |
| LONGITUDINAL | 2-6s | tg_err | +0.2175 | [-1.2266, +1.8677] | none | 57 | 19 |
| LONGITUDINAL | 2-6s | ttc_err | -1.5635 | [-3.7557, +0.8434] | none | 80 | 20 |
| LONGITUDINAL | full6s | acc_mae | +0.0963 | [+0.0637, +0.1274] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.0644 | [-0.1152, +0.2558] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.2523 | [-0.5203, +1.1512] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0767 | [-0.0423, +0.2289] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +0.6071 | [-0.8434, +1.9537] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0087 | [+0.0039, +0.0139] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | -0.0021 | [-0.0117, +0.0084] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0016 | [+0.0004, +0.0028] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.0684 | [-0.2356, +0.0828] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0152 | [-0.0763, +0.0391] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0267 | [+0.0215, +0.0314] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0249 | [+0.0082, +0.0408] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0038 | [+0.0024, +0.0053] | worse | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.1434 | [-1.2360, +0.7922] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.0564 | [-0.6576, +0.4713] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0207 | [+0.0165, +0.0245] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0249 | [+0.0082, +0.0408] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0030 | [+0.0017, +0.0044] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | -0.1434 | [-1.2360, +0.7922] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.0427 | [-0.4612, +0.3259] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0181 | [-0.0231, +0.0604] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | -0.0070 | [-0.0523, +0.0382] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0010 | [-0.0382, +0.0433] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0795 | [+0.0312, +0.1278] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0946 | [+0.0382, +0.1489] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0342 | [-0.0201, +0.0885] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0272 | [-0.0201, +0.0755] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0463 | [-0.0111, +0.0986] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0000 | [-0.0493, +0.0523] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [-0.2000, +0.2500] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -1.9225 | [-5.2083, +4.0307] | none | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0421 | [+0.0108, +0.0742] | higher | 856 | 69 |

X - HOLD (the CEILING cell and its family):

| family | band | metric | X - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.9530 | [+0.8864, +1.0189] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.9117 | [+0.7996, +1.0293] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.1517 | [+0.6331, +1.7908] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1639 | [+0.0907, +0.2538] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +3.8903 | [+2.4070, +5.6198] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.4085 | [+0.3250, +0.4929] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | -0.0819 | [-0.4585, +0.2727] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +1.8131 | [+0.8954, +2.8395] | worse | 109 |
| LONGITUDINAL | 2-6s | tg_err | +0.5607 | [-0.2703, +1.8633] | none | 82 |
| LONGITUDINAL | 2-6s | ttc_err | +2.7632 | [+0.5662, +4.8864] | worse | 109 |
| LONGITUDINAL | full6s | acc_mae | +0.5900 | [+0.5248, +0.6530] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | -0.0819 | [-0.4585, +0.2727] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +2.0133 | [+0.7346, +3.6084] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.2634 | [+0.1049, +0.4562] | worse | 182 |
| LONGITUDINAL | full6s | ttc_err | +3.9251 | [+2.0128, +5.9029] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.1024 | [+0.0949, +0.1101] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.1060 | [+0.0947, +0.1181] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0172 | [+0.0128, +0.0226] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2468 | [+1.0460, +1.4643] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7835 | [+0.7076, +0.8625] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0732 | [+0.0632, +0.0825] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0633 | [+0.0287, +0.0975] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0111 | [+0.0077, +0.0148] | worse | 967 |
| LATERAL | 2-6s | xtrack | +5.0522 | [+3.7966, +6.3736] | worse | 994 |
| LATERAL | 2-6s | ade | +4.3966 | [+3.7378, +5.1028] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0830 | [+0.0760, +0.0895] | worse | 994 |
| LATERAL | full6s | head_err | +0.0633 | [+0.0287, +0.0975] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0137 | [+0.0099, +0.0180] | worse | 974 |
| LATERAL | full6s | xtrack | +5.0522 | [+3.7966, +6.3736] | worse | 994 |
| LATERAL | full6s | ade | +3.1922 | [+2.7262, +3.6872] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4728 | [+0.4306, +0.5191] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2636 | [+0.2183, +0.3119] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3793 | [+0.3340, +0.4246] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1871 | [+0.1036, +0.2746] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.1771 | [+0.1167, +0.2364] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.1167 | [+0.0302, +0.2042] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.2072 | [+0.1428, +0.2767] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2616 | [+0.1972, +0.3270] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.0815 | [+0.0070, +0.1590] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.5000 | [-1.0000, +0.2000] | none | 8 |
| STRATEGIC | full6s | s_adh_deg | -47.7377 | [-89.7084, -5.7670] | worse | 8 |
| STRATEGIC | full6s | s_false | -0.0422 | [-0.0981, +0.0092] | none | 853 |

### RS_A0  (c_RS_A0_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.3310 | [+0.2318, +0.4292] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +1.0865 | [+0.8673, +1.3103] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.2725 | [-0.3594, +0.8134] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0399 | [-0.0299, +0.1175] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +1.4309 | [+0.0298, +2.8738] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1340 | [+0.0912, +0.1738] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.7101 | [+0.3938, +1.0051] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +2.2414 | [+0.9118, +3.7410] | worse | 41 | 13 |
| LONGITUDINAL | 2-6s | tg_err | +1.8888 | [-0.2688, +4.5115] | none | 26 | 12 |
| LONGITUDINAL | 2-6s | ttc_err | +2.0872 | [-3.2267, +7.9584] | none | 41 | 13 |
| LONGITUDINAL | full6s | acc_mae | +0.1997 | [+0.1612, +0.2377] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.7101 | [+0.3938, +1.0051] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.3506 | [-0.6188, +1.2226] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0020 | [-0.1336, +0.1193] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +1.1057 | [-1.0191, +3.3277] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0417 | [+0.0310, +0.0530] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0730 | [+0.0541, +0.0915] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0022 | [-0.0003, +0.0043] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.0905 | [+0.8465, +1.3614] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.5793 | [+0.4792, +0.6773] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0248 | [+0.0193, +0.0298] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0802 | [+0.0556, +0.1049] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0010 | [-0.0008, +0.0029] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +5.4998 | [+4.2005, +6.8245] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +4.8068 | [+4.0647, +5.5720] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0305 | [+0.0254, +0.0354] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0802 | [+0.0556, +0.1049] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0011 | [-0.0012, +0.0032] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +5.4998 | [+4.2005, +6.8245] | worse | 994 | 71 |
| LATERAL | full6s | ade | +3.3977 | [+2.8744, +3.9414] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.2193 | [+0.1630, +0.2757] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2435 | [+0.1770, +0.3099] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.2495 | [+0.1911, +0.3068] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0926 | [+0.0433, +0.1429] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.1348 | [+0.0805, +0.1891] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0513 | [-0.0010, +0.1016] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0966 | [+0.0463, +0.1469] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1650 | [+0.1036, +0.2234] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0855 | [+0.0221, +0.1479] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.0833 | [-0.2000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -9.1543 | [-16.0848, -1.8459] | worse | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.1415 | [+0.0991, +0.1884] | higher | 848 | 68 |

### RS_P  (c_RS_P_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.3605 | [+0.2654, +0.4506] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +1.1308 | [+0.9128, +1.3509] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.1642 | [-0.5149, +0.7402] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0156 | [-0.0620, +0.0970] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.9675 | [-0.4559, +2.4344] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.6348 | [+0.3013, +0.9394] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +2.1638 | [+0.4689, +4.0301] | worse | 48 | 17 |
| LONGITUDINAL | 2-6s | tg_err | +1.6691 | [+0.0288, +3.6623] | worse | 34 | 16 |
| LONGITUDINAL | 2-6s | ttc_err | +3.2241 | [-3.5749, +10.3262] | none | 48 | 17 |
| LONGITUDINAL | full6s | acc_mae | +0.1202 | [+0.0885, +0.1502] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.6348 | [+0.3013, +0.9394] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.1487 | [-1.2186, +1.4472] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.1229 | [-0.3802, +0.0896] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +0.4655 | [-1.5184, +2.5466] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0373 | [+0.0255, +0.0489] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0811 | [+0.0619, +0.1008] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0020 | [-0.0004, +0.0040] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.1890 | [+0.8906, +1.5436] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.6198 | [+0.5095, +0.7409] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0615 | [+0.0404, +0.0825] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0023 | [-0.0043, -0.0003] | better | 967 | 71 |
| LATERAL | 2-6s | xtrack | +5.9305 | [+4.2397, +7.8657] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +5.1264 | [+4.1492, +6.2189] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0124 | [+0.0085, +0.0163] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0615 | [+0.0404, +0.0825] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0012 | [-0.0037, +0.0009] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +5.9305 | [+4.2397, +7.8657] | worse | 994 | 71 |
| LATERAL | full6s | ade | +3.6242 | [+2.9419, +4.3976] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.2233 | [+0.1640, +0.2797] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2696 | [+0.1972, +0.3410] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.2586 | [+0.1922, +0.3219] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0091 | [-0.0030, +0.0241] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0070 | [-0.0040, +0.0221] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0211 | [+0.0060, +0.0392] | worse | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0805 | [+0.0322, +0.1258] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1157 | [+0.0543, +0.1740] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0855 | [+0.0252, +0.1439] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 13 | 4 |
| STRATEGIC | full6s | s_adh_deg | -4.3353 | [-8.1442, +0.1085] | none | 13 | 4 |
| STRATEGIC | full6s | s_false | +0.0979 | [+0.0579, +0.1394] | higher | 848 | 68 |

### P23_P  (c_P23_P_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0174 | [-0.0006, +0.0357] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0418 | [-0.0108, +0.0924] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0601 | [-0.3198, +0.1543] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0083 | [-0.0390, +0.0149] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.1364 | [-0.4630, +0.7663] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0165 | [+0.0006, +0.0309] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0109 | [-0.0770, +0.0528] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.2699 | [-0.1997, +0.7085] | none | 108 | 22 |
| LONGITUDINAL | 2-6s | tg_err | +0.4302 | [-0.2237, +1.5018] | none | 85 | 22 |
| LONGITUDINAL | 2-6s | ttc_err | -0.2358 | [-2.0589, +1.1461] | none | 108 | 22 |
| LONGITUDINAL | full6s | acc_mae | +0.0168 | [+0.0036, +0.0287] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0109 | [-0.0770, +0.0528] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.0503 | [-0.4593, +0.6212] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0442 | [-0.1574, +0.0512] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.0331 | [-0.9662, +0.9444] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0043 | [+0.0015, +0.0073] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0060 | [+0.0007, +0.0115] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0014 | [+0.0004, +0.0025] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0302 | [-0.0286, +0.0907] | none | 994 | 71 |
| LATERAL | 0-2s | ade | +0.0253 | [+0.0033, +0.0471] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0019 | [+0.0005, +0.0033] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0062 | [-0.0030, +0.0155] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0004 | [-0.0002, +0.0013] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.2873 | [-0.1172, +0.7042] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.2633 | [+0.0574, +0.4724] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0027 | [+0.0013, +0.0042] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0062 | [-0.0030, +0.0155] | none | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0006 | [+0.0000, +0.0014] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +0.2873 | [-0.1172, +0.7042] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.1839 | [+0.0413, +0.3284] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0221 | [-0.0060, +0.0503] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0191 | [-0.0070, +0.0463] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0101 | [-0.0231, +0.0433] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0040 | [-0.0272, +0.0201] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0332 | [-0.0624, -0.0040] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0040 | [-0.0211, +0.0282] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0111 | [-0.0131, +0.0342] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0030 | [-0.0312, +0.0262] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0000 | [-0.0211, +0.0201] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0769 | [+0.0000, +0.2500] | none | 13 | 4 |
| STRATEGIC | full6s | s_adh_deg | +2.4075 | [-1.6216, +5.8610] | none | 13 | 4 |
| STRATEGIC | full6s | s_false | -0.0035 | [-0.0178, +0.0115] | none | 856 | 69 |

### P_A1  (c_P_A1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.9291 | [+0.8605, +0.9949] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.8883 | [+0.7698, +0.9995] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +1.0783 | [+0.5525, +1.7907] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1621 | [+0.0915, +0.2595] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +3.2629 | [+2.0550, +4.6404] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1340 | [+0.0912, +0.1738] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.1874 | [-0.0555, +0.4352] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +1.5374 | [+0.8623, +2.2763] | worse | 116 | 21 |
| LONGITUDINAL | 2-6s | tg_err | +0.9439 | [-0.0529, +2.7136] | none | 92 | 21 |
| LONGITUDINAL | 2-6s | ttc_err | +2.7239 | [+0.1557, +5.7975] | worse | 116 | 21 |
| LONGITUDINAL | full6s | acc_mae | +0.3991 | [+0.3599, +0.4358] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.1874 | [-0.0555, +0.4352] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +1.7534 | [+0.8647, +2.7728] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.2175 | [+0.0805, +0.3507] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +3.4510 | [+2.1045, +4.8963] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0954 | [+0.0878, +0.1032] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0964 | [+0.0856, +0.1077] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0153 | [+0.0113, +0.0202] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.1719 | [+0.9730, +1.3864] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.7259 | [+0.6456, +0.8064] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0248 | [+0.0193, +0.0298] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0677 | [+0.0476, +0.0868] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0038 | [+0.0021, +0.0055] | worse | 967 | 71 |
| LATERAL | 2-6s | xtrack | +4.1743 | [+3.2469, +5.1207] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +4.0405 | [+3.4371, +4.6614] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0484 | [+0.0434, +0.0529] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0677 | [+0.0476, +0.0868] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0079 | [+0.0055, +0.0105] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +4.1743 | [+3.2469, +5.1207] | worse | 994 | 71 |
| LATERAL | full6s | ade | +2.9357 | [+2.5061, +3.3717] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.4567 | [+0.4064, +0.5060] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2495 | [+0.2032, +0.2978] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3722 | [+0.3209, +0.4245] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0855 | [+0.0362, +0.1348] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.1288 | [+0.0734, +0.1831] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0352 | [-0.0171, +0.0876] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.1187 | [+0.0734, +0.1650] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1740 | [+0.1137, +0.2364] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0423 | [-0.0171, +0.1006] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.3333 | [-0.6667, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -16.4624 | [-29.7159, -3.2089] | worse | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0188 | [-0.0233, +0.0563] | none | 853 | 69 |

### P23_A1  (c_P23_A1_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.9465 | [+0.8787, +1.0123] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.9301 | [+0.8153, +1.0474] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +1.0183 | [+0.5419, +1.6118] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1538 | [+0.0843, +0.2424] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +3.3993 | [+2.0638, +4.7981] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1505 | [+0.1050, +0.1917] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.1765 | [-0.0688, +0.4324] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +1.3684 | [+0.5867, +2.1477] | worse | 110 | 23 |
| LONGITUDINAL | 2-6s | tg_err | -0.0226 | [-2.5967, +1.7878] | none | 85 | 23 |
| LONGITUDINAL | 2-6s | ttc_err | +2.1845 | [+0.1705, +4.8362] | worse | 110 | 23 |
| LONGITUDINAL | full6s | acc_mae | +0.4158 | [+0.3748, +0.4556] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.1765 | [-0.0688, +0.4324] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +1.8037 | [+0.7128, +3.1968] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.1733 | [+0.0049, +0.3408] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +3.4179 | [+2.1812, +4.7538] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0998 | [+0.0919, +0.1075] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.1025 | [+0.0918, +0.1138] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0167 | [+0.0123, +0.0220] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.2020 | [+1.0090, +1.4168] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.7512 | [+0.6740, +0.8300] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0267 | [+0.0215, +0.0314] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0740 | [+0.0561, +0.0927] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0043 | [+0.0025, +0.0061] | worse | 967 | 71 |
| LATERAL | 2-6s | xtrack | +4.4616 | [+3.5046, +5.5155] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +4.3038 | [+3.7150, +4.9479] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0511 | [+0.0465, +0.0555] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0740 | [+0.0561, +0.0927] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0085 | [+0.0060, +0.0113] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +4.4616 | [+3.5046, +5.5155] | worse | 994 | 71 |
| LATERAL | full6s | ade | +3.1196 | [+2.7089, +3.5691] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.4789 | [+0.4366, +0.5241] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2686 | [+0.2223, +0.3169] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3823 | [+0.3380, +0.4276] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0815 | [+0.0312, +0.1308] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0956 | [+0.0432, +0.1479] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0392 | [-0.0141, +0.0936] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.1298 | [+0.0835, +0.1771] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1710 | [+0.1127, +0.2324] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0423 | [-0.0131, +0.1016] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.2500 | [-0.6667, +0.1818] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | -13.4300 | [-22.0296, -4.8305] | worse | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0164 | [-0.0258, +0.0535] | none | 853 | 69 |

### P2-0_P  (c_P2-0_P_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.2052 | [-0.2740, -0.1369] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0078 | [-0.0925, +0.1158] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.4156 | [-0.9716, -0.0273] | better | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0305 | [-0.0923, +0.0281] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.3377 | [-1.5117, +0.9052] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.2334 | [+0.0933, +0.3920] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.2797 | [-0.9390, +0.5552] | none | 83 | 20 |
| LONGITUDINAL | 2-6s | tg_err | +1.2358 | [+0.2125, +3.1101] | worse | 59 | 20 |
| LONGITUDINAL | 2-6s | ttc_err | -2.5643 | [-6.7444, +0.4310] | none | 83 | 20 |
| LONGITUDINAL | full6s | acc_mae | -0.0684 | [-0.0913, -0.0456] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.2334 | [+0.0933, +0.3920] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.7215 | [-1.8156, +0.2032] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0158 | [-0.2151, +0.2229] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.8666 | [-2.2020, +0.4718] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0079 | [-0.0005, +0.0164] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | -0.0186 | [-0.0306, -0.0069] | better | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0001 | [-0.0024, +0.0020] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.2117 | [-0.3892, -0.0572] | better | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0839 | [-0.1524, -0.0195] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0101 | [-0.0047, +0.0250] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0006 | [-0.0014, +0.0002] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0094 | [-0.9008, +0.8076] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1166 | [-0.6022, +0.3581] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0026 | [-0.0002, +0.0055] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0101 | [-0.0047, +0.0250] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0003 | [-0.0014, +0.0007] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0094 | [-0.9008, +0.8076] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.1057 | [-0.4494, +0.2280] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0121 | [-0.0634, +0.0402] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | -0.0533 | [-0.1046, -0.0030] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0231 | [-0.0342, +0.0795] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0070 | [-0.0161, +0.0000] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0040 | [-0.0080, -0.0010] | better | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0010 | [-0.0080, +0.0101] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0241 | [-0.0080, +0.0544] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0131 | [-0.0362, +0.0563] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0272 | [-0.0030, +0.0604] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 12 | 4 |
| STRATEGIC | full6s | s_adh_deg | +0.1112 | [-0.2497, +0.7718] | none | 12 | 4 |
| STRATEGIC | full6s | s_false | +0.0258 | [-0.0069, +0.0613] | none | 854 | 69 |

### P23-2-0_P23  (c_P23-2-0_P23_s0.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.1974 | [-0.2654, -0.1268] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0184 | [-0.0865, +0.1284] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.1863 | [-0.6559, +0.1698] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0247 | [-0.0465, +0.1199] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.3588 | [-1.4440, +0.6836] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.2433 | [+0.1042, +0.4076] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.1175 | [-0.5939, +0.8513] | none | 76 | 21 |
| LONGITUDINAL | 2-6s | tg_err | +0.9503 | [+0.0367, +2.0271] | worse | 49 | 18 |
| LONGITUDINAL | 2-6s | ttc_err | -0.3690 | [-2.7093, +2.3483] | none | 76 | 21 |
| LONGITUDINAL | full6s | acc_mae | -0.0658 | [-0.0885, -0.0423] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.2433 | [+0.1042, +0.4076] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.4439 | [-1.2647, +0.1895] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0380 | [-0.1139, +0.2142] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.5392 | [-1.7583, +0.5801] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0038 | [-0.0041, +0.0117] | none | 994 | 71 |
| LATERAL | 0-2s | head_err | -0.0189 | [-0.0307, -0.0074] | better | 994 | 71 |
| LATERAL | 0-2s | curv_err | -0.0012 | [-0.0037, +0.0009] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.1866 | [-0.3609, -0.0340] | better | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0843 | [-0.1497, -0.0213] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0051 | [-0.0093, +0.0194] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0006 | [-0.0015, +0.0002] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0347 | [-1.0281, +0.8960] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1422 | [-0.6707, +0.3627] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0013 | [-0.0014, +0.0039] | none | 994 | 71 |
| LATERAL | full6s | head_err | +0.0051 | [-0.0093, +0.0194] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0007 | [-0.0021, +0.0005] | none | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0347 | [-1.0281, +0.8960] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.1229 | [-0.4972, +0.2338] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0312 | [-0.0765, +0.0141] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | -0.0503 | [-0.1036, +0.0020] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0252 | [-0.0292, +0.0775] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0040 | [-0.0141, +0.0040] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0010 | [+0.0000, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0020 | [-0.0141, +0.0080] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0231 | [-0.0121, +0.0543] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0171 | [-0.0262, +0.0594] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0392 | [+0.0070, +0.0755] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.0909 | [-0.2500, +0.0000] | none | 11 | 4 |
| STRATEGIC | full6s | s_adh_deg | -2.9446 | [-7.0009, -0.6268] | worse | 11 | 4 |
| STRATEGIC | full6s | s_false | +0.0106 | [-0.0125, +0.0338] | none | 853 | 69 |

### Mechanism rows (beside, no bar)

| arm | REACH median cost1 ratio vs A0 | REACH mean log-ratio [CI] | COST2 median ratio (batch 0) | image resid level 2 max | level 3 max |
|---|---|---|---|---|---|
| P | 0.803 | -0.2474 [-0.2910, -0.2060] | 2.083 | 1.75e-07 | None |
| P23 | 0.819 | -0.2097 [-0.2563, -0.1654] | 1.854 | 2.38e-07 | 6.506289196295256e-07 |
| RS | 11.157 | +2.3790 [+2.2678, +2.4864] | 2.083 | 1.75e-07 | None |

### DIR-ACC (beside, no bar)

| arm | r | CI | n windows | n episodes |
|---|---|---|---|---|
| A0 | +0.3795 | [+0.3088, +0.4475] | 994 | 71 |
| P | +0.4086 | [+0.3406, +0.4727] | 994 | 71 |
| P23 | +0.3939 | [+0.3275, +0.4538] | 994 | 71 |
| RS | +0.2656 | [+0.2120, +0.3140] | 994 | 71 |
| P2-0 | +0.4046 | [+0.3260, +0.4741] | 994 | 71 |
| P23-2-0 | +0.4123 | [+0.3415, +0.4795] | 994 | 71 |

C-RS on SCORE (RS - A0, 0-2 s acc_mae): {'delta': 0.331, 'lo': 0.2318, 'hi': 0.4292, 'label': 'worse', 'ok': True}

STRATEGIC / TACTICAL blocks: see `families` in RESULT_AMEND4.json (strategic_block with n per cell and the not-computed entries with reasons).

## Chassis R35_s1: 994 windows, 71 episodes (windows sha256 8833e2923c17)

Deciding cells, 0-2 s, with the family tables in the contrast files named below. Rows: delta [lo, hi] label.

### P_A0  (c_P_A0_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0288 | [-0.0037, +0.0631] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0532 | [-0.1158, +0.0148] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.1119 | [-0.4053, +0.1500] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0152 | [-0.0729, +0.0398] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.5669 | [-1.4232, +0.0427] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1682 | [+0.1243, +0.2103] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.4559 | [+0.2464, +0.6923] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.3118 | [-0.0645, +0.8226] | none | 96 | 21 |
| LONGITUDINAL | 2-6s | tg_err | -0.5549 | [-1.8321, +0.6556] | none | 64 | 19 |
| LONGITUDINAL | 2-6s | ttc_err | +0.8660 | [-0.2620, +2.2944] | none | 96 | 21 |
| LONGITUDINAL | full6s | acc_mae | +0.1217 | [+0.0925, +0.1515] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.4559 | [+0.2464, +0.6923] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.3601 | [-1.1711, +0.2734] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | -0.0525 | [-0.1784, +0.0599] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.3635 | [-1.3080, +0.3836] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0076 | [+0.0035, +0.0121] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | -0.0003 | [-0.0081, +0.0076] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0015 | [+0.0005, +0.0025] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0047 | [-0.0998, +0.1081] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0026 | [-0.0400, +0.0358] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0279 | [+0.0239, +0.0318] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0304 | [+0.0128, +0.0486] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0050 | [+0.0032, +0.0071] | worse | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.3726 | [-0.3065, +1.0567] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.3110 | [-0.0060, +0.6454] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0211 | [+0.0178, +0.0245] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0304 | [+0.0128, +0.0486] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0039 | [+0.0025, +0.0055] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +0.3726 | [-0.3065, +1.0567] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.2065 | [-0.0135, +0.4378] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0282 | [-0.0664, +0.0111] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | -0.0070 | [-0.0392, +0.0252] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0161 | [-0.0513, +0.0221] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.1056 | [+0.0604, +0.1519] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.1298 | [+0.0805, +0.1891] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0493 | [-0.0040, +0.1067] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0644 | [+0.0231, +0.1046] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0654 | [+0.0161, +0.1157] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0634 | [+0.0171, +0.1097] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.1429 | [-0.3750, +0.0000] | none | 7 | 4 |
| STRATEGIC | full6s | s_adh_deg | -7.8565 | [-12.7511, -4.1855] | worse | 7 | 4 |
| STRATEGIC | full6s | s_false | +0.0187 | [-0.0092, +0.0495] | none | 855 | 70 |

X - HOLD (the CEILING cell and its family):

| family | band | metric | X - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.8974 | [+0.8231, +0.9710] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.8684 | [+0.7504, +0.9980] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.0988 | [+0.5528, +1.9234] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1423 | [+0.0756, +0.2488] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +3.0487 | [+1.7806, +4.6432] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.3476 | [+0.2612, +0.4334] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | +0.2294 | [-0.1657, +0.6277] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.2200 | [+1.1055, +3.5049] | worse | 104 |
| LONGITUDINAL | 2-6s | tg_err | +2.2667 | [+0.7100, +4.4093] | worse | 70 |
| LONGITUDINAL | 2-6s | ttc_err | +0.7143 | [-1.9034, +3.3111] | none | 104 |
| LONGITUDINAL | full6s | acc_mae | +0.5309 | [+0.4594, +0.5988] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | +0.2294 | [-0.1657, +0.6277] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.2470 | [+0.2879, +2.4158] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1008 | [-0.0823, +0.2760] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.4970 | [+0.7418, +4.3524] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0927 | [+0.0841, +0.1004] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.0969 | [+0.0845, +0.1096] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0151 | [+0.0113, +0.0199] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2011 | [+0.9965, +1.4054] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7636 | [+0.6885, +0.8398] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0728 | [+0.0633, +0.0816] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0600 | [+0.0286, +0.0919] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0127 | [+0.0085, +0.0175] | worse | 967 |
| LATERAL | 2-6s | xtrack | +4.4827 | [+3.3532, +5.6239] | worse | 994 |
| LATERAL | 2-6s | ade | +4.3186 | [+3.6309, +5.0502] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0794 | [+0.0731, +0.0856] | worse | 994 |
| LATERAL | full6s | head_err | +0.0600 | [+0.0286, +0.0919] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0142 | [+0.0100, +0.0191] | worse | 974 |
| LATERAL | full6s | xtrack | +4.4827 | [+3.3532, +5.6239] | worse | 994 |
| LATERAL | full6s | ade | +3.1336 | [+2.6595, +3.6401] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4366 | [+0.3783, +0.4970] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2294 | [+0.1821, +0.2807] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3541 | [+0.2998, +0.4074] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1932 | [+0.1147, +0.2696] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.2264 | [+0.1640, +0.2918] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0845 | [+0.0040, +0.1700] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.2354 | [+0.1700, +0.3008] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2626 | [+0.2072, +0.3169] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.1499 | [+0.0795, +0.2224] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.4286 | [-0.7500, -0.2000] | worse | 7 |
| STRATEGIC | full6s | s_adh_deg | -45.6194 | [-83.6221, -17.1173] | worse | 7 |
| STRATEGIC | full6s | s_false | -0.0704 | [-0.1272, -0.0182] | lower | 852 |

### P23_A0  (c_P23_A0_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0344 | [-0.0011, +0.0737] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0319 | [-0.1013, +0.0413] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.1257 | [-0.4306, +0.1507] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | -0.0138 | [-0.0725, +0.0421] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.2053 | [-1.1187, +0.5850] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1807 | [+0.1397, +0.2214] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.3912 | [+0.1910, +0.6105] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.2624 | [-0.4686, +1.1604] | none | 92 | 20 |
| LONGITUDINAL | 2-6s | tg_err | +0.6581 | [-1.0871, +3.3199] | none | 59 | 17 |
| LONGITUDINAL | 2-6s | ttc_err | -0.4991 | [-2.1129, +0.6184] | none | 92 | 20 |
| LONGITUDINAL | full6s | acc_mae | +0.1319 | [+0.1028, +0.1613] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.3912 | [+0.1910, +0.6105] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | -0.0586 | [-0.8307, +0.7512] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0117 | [-0.1064, +0.1604] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.2042 | [-1.1977, +0.7699] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0114 | [+0.0072, +0.0158] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0050 | [-0.0038, +0.0139] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0023 | [+0.0011, +0.0039] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0498 | [-0.0545, +0.1559] | none | 994 | 71 |
| LATERAL | 0-2s | ade | +0.0146 | [-0.0243, +0.0543] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0304 | [+0.0267, +0.0342] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0346 | [+0.0174, +0.0520] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0056 | [+0.0037, +0.0079] | worse | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.3586 | [-0.3715, +1.0598] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.4025 | [+0.0295, +0.7844] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0241 | [+0.0210, +0.0273] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0346 | [+0.0174, +0.0520] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0048 | [+0.0033, +0.0068] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +0.3586 | [-0.3715, +1.0598] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.2732 | [+0.0154, +0.5378] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0050 | [-0.0332, +0.0433] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0312 | [-0.0040, +0.0664] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0020 | [-0.0372, +0.0412] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0915 | [+0.0473, +0.1378] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.1127 | [+0.0644, +0.1670] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0473 | [+0.0000, +0.0956] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0594 | [+0.0171, +0.0996] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0594 | [+0.0101, +0.1087] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0654 | [+0.0171, +0.1167] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.1111 | [-0.7500, +0.2727] | none | 9 | 4 |
| STRATEGIC | full6s | s_adh_deg | +0.3679 | [-8.0115, +7.0714] | none | 9 | 4 |
| STRATEGIC | full6s | s_false | +0.0257 | [+0.0000, +0.0529] | none | 857 | 70 |

X - HOLD (the CEILING cell and its family):

| family | band | metric | X - HOLD | CI | label | n |
|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.9029 | [+0.8320, +0.9789] | worse | 994 |
| LONGITUDINAL | 0-2s | speed_err | +0.8897 | [+0.7746, +1.0165] | worse | 994 |
| LONGITUDINAL | 0-2s | hw_err | +1.0850 | [+0.5391, +1.9377] | worse | 214 |
| LONGITUDINAL | 0-2s | tg_err | +0.1437 | [+0.0707, +0.2689] | worse | 182 |
| LONGITUDINAL | 0-2s | ttc_err | +3.4104 | [+1.9546, +5.2022] | worse | 214 |
| LONGITUDINAL | 2-6s | acc_mae | +0.3601 | [+0.2718, +0.4449] | worse | 994 |
| LONGITUDINAL | 2-6s | speed_err | +0.1647 | [-0.2286, +0.5723] | none | 994 |
| LONGITUDINAL | 2-6s | hw_err | +2.3401 | [+1.2841, +3.5958] | worse | 104 |
| LONGITUDINAL | 2-6s | tg_err | +1.8683 | [+0.5073, +4.7393] | worse | 73 |
| LONGITUDINAL | 2-6s | ttc_err | -0.1980 | [-2.8301, +2.3381] | none | 104 |
| LONGITUDINAL | full6s | acc_mae | +0.5411 | [+0.4670, +0.6103] | worse | 994 |
| LONGITUDINAL | full6s | speed_err | +0.1647 | [-0.2286, +0.5723] | none | 994 |
| LONGITUDINAL | full6s | hw_err | +1.5484 | [+0.4221, +2.9413] | worse | 214 |
| LONGITUDINAL | full6s | tg_err | +0.1650 | [-0.0793, +0.4437] | none | 182 |
| LONGITUDINAL | full6s | ttc_err | +2.6563 | [+0.7400, +4.7542] | worse | 214 |
| LATERAL | 0-2s | yaw_mae | +0.0965 | [+0.0882, +0.1041] | worse | 994 |
| LATERAL | 0-2s | head_err | +0.1022 | [+0.0889, +0.1160] | worse | 994 |
| LATERAL | 0-2s | curv_err | +0.0159 | [+0.0120, +0.0206] | worse | 950 |
| LATERAL | 0-2s | xtrack | +1.2462 | [+1.0385, +1.4483] | worse | 994 |
| LATERAL | 0-2s | ade | +0.7807 | [+0.7080, +0.8550] | worse | 994 |
| LATERAL | 2-6s | yaw_mae | +0.0754 | [+0.0658, +0.0843] | worse | 994 |
| LATERAL | 2-6s | head_err | +0.0643 | [+0.0312, +0.0980] | worse | 994 |
| LATERAL | 2-6s | curv_err | +0.0134 | [+0.0089, +0.0184] | worse | 967 |
| LATERAL | 2-6s | xtrack | +4.4687 | [+3.2920, +5.6499] | worse | 994 |
| LATERAL | 2-6s | ade | +4.4101 | [+3.7308, +5.0918] | worse | 994 |
| LATERAL | full6s | yaw_mae | +0.0824 | [+0.0762, +0.0885] | worse | 994 |
| LATERAL | full6s | head_err | +0.0643 | [+0.0312, +0.0980] | worse | 994 |
| LATERAL | full6s | curv_err | +0.0151 | [+0.0106, +0.0203] | worse | 974 |
| LATERAL | full6s | xtrack | +4.4687 | [+3.2920, +5.6499] | worse | 994 |
| LATERAL | full6s | ade | +3.2003 | [+2.7313, +3.6708] | worse | 994 |
| TACTICAL | 0-2s | tac_man_dis | +0.4698 | [+0.4145, +0.5232] | worse | 994 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2676 | [+0.2122, +0.3199] | worse | 994 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3722 | [+0.3249, +0.4195] | worse | 994 |
| TACTICAL | 2-6s | tac_man_dis | +0.1791 | [+0.1026, +0.2515] | worse | 994 |
| TACTICAL | 2-6s | tac_lat_dis | +0.2093 | [+0.1509, +0.2706] | worse | 994 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0825 | [+0.0040, +0.1630] | worse | 994 |
| TACTICAL | full6s | tac_man_dis | +0.2304 | [+0.1620, +0.2998] | worse | 994 |
| TACTICAL | full6s | tac_lat_dis | +0.2565 | [+0.1952, +0.3179] | worse | 994 |
| TACTICAL | full6s | tac_lon_dis | +0.1519 | [+0.0744, +0.2294] | worse | 994 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 |
| STRATEGIC | full6s | s_adh | -0.5000 | [-1.0000, +0.2000] | none | 8 |
| STRATEGIC | full6s | s_adh_deg | -43.3147 | [-82.7536, -3.8758] | worse | 8 |
| STRATEGIC | full6s | s_false | -0.0632 | [-0.1200, -0.0115] | lower | 854 |

### RS_A0  (c_RS_A0_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.3287 | [+0.2282, +0.4304] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.9113 | [+0.6713, +1.1246] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.5002 | [-0.0676, +1.0479] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1164 | [+0.0197, +0.2397] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +1.2251 | [-0.2196, +2.6740] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1682 | [+0.1243, +0.2103] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.7595 | [+0.4241, +1.0998] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +1.5206 | [-1.1513, +3.4508] | none | 38 | 11 |
| LONGITUDINAL | 2-6s | tg_err | -5.5456 | [-9.9922, -0.6362] | better | 18 | 10 |
| LONGITUDINAL | 2-6s | ttc_err | +9.4876 | [+5.9528, +12.3767] | worse | 38 | 11 |
| LONGITUDINAL | full6s | acc_mae | +0.2217 | [+0.1802, +0.2655] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.7595 | [+0.4241, +1.0998] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.7120 | [-1.0317, +2.1940] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.1847 | [-0.1231, +0.5143] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +1.4195 | [+0.0113, +2.8820] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0471 | [+0.0344, +0.0601] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0961 | [+0.0740, +0.1185] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0036 | [+0.0015, +0.0062] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.4227 | [+1.0523, +1.7941] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.6381 | [+0.5079, +0.7697] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0279 | [+0.0239, +0.0318] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.1083 | [+0.0813, +0.1367] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0013 | [-0.0013, +0.0038] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | +7.2065 | [+5.1789, +9.3227] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +5.3714 | [+4.1957, +6.5864] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0343 | [+0.0289, +0.0397] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.1083 | [+0.0813, +0.1367] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0019 | [-0.0005, +0.0045] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +7.2065 | [+5.1789, +9.3227] | worse | 994 | 71 |
| LATERAL | full6s | ade | +3.7936 | [+2.9720, +4.6417] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.2133 | [+0.1459, +0.2847] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.3169 | [+0.2314, +0.4004] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.2626 | [+0.1942, +0.3300] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.1227 | [+0.0734, +0.1731] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.1358 | [+0.0855, +0.1952] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0714 | [+0.0151, +0.1298] | worse | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.1097 | [+0.0644, +0.1550] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.2153 | [+0.1469, +0.2827] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0825 | [+0.0231, +0.1388] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.3000 | [-0.6667, +0.0000] | none | 10 | 5 |
| STRATEGIC | full6s | s_adh_deg | -9.4813 | [-29.3314, +7.5641] | none | 10 | 5 |
| STRATEGIC | full6s | s_false | +0.1779 | [+0.1339, +0.2258] | higher | 849 | 68 |

### RS_P  (c_RS_P_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.2998 | [+0.1962, +0.4065] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.9645 | [+0.7251, +1.1868] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.6121 | [+0.1046, +1.1581] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1316 | [+0.0381, +0.2582] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +1.7920 | [+0.2100, +3.4396] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.3036 | [-0.0076, +0.6188] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +2.2078 | [+1.1864, +4.0781] | worse | 33 | 7 |
| LONGITUDINAL | 2-6s | tg_err | -3.8474 | [-7.4713, +0.1040] | none | 12 | 6 |
| LONGITUDINAL | 2-6s | ttc_err | +10.3172 | [+7.8977, +12.4322] | worse | 33 | 7 |
| LONGITUDINAL | full6s | acc_mae | +0.0999 | [+0.0654, +0.1355] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.3036 | [-0.0076, +0.6188] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +1.0721 | [-0.1616, +2.4121] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.2372 | [+0.0062, +0.5212] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +1.7830 | [+0.2221, +3.4816] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0395 | [+0.0267, +0.0529] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0964 | [+0.0736, +0.1184] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0022 | [-0.0004, +0.0047] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.4180 | [+1.0509, +1.7935] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.6407 | [+0.5130, +0.7693] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0779 | [+0.0517, +0.1051] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0037 | [-0.0070, -0.0008] | better | 967 | 71 |
| LATERAL | 2-6s | xtrack | +6.8339 | [+4.6967, +9.0013] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +5.0604 | [+3.8292, +6.2756] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0132 | [+0.0089, +0.0176] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0779 | [+0.0517, +0.1051] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0020 | [-0.0050, +0.0007] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +6.8339 | [+4.6967, +9.0013] | worse | 994 | 71 |
| LATERAL | full6s | ade | +3.5871 | [+2.7304, +4.4313] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.2414 | [+0.1761, +0.3089] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.3239 | [+0.2414, +0.4034] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.2787 | [+0.2082, +0.3431] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0171 | [+0.0020, +0.0362] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0060 | [+0.0000, +0.0121] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0221 | [+0.0060, +0.0412] | worse | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0453 | [-0.0080, +0.0966] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1499 | [+0.0774, +0.2203] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0191 | [-0.0423, +0.0785] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.2500 | [-0.5000, +0.0000] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | -7.4137 | [-25.4752, +3.4231] | none | 8 | 4 |
| STRATEGIC | full6s | s_false | +0.1580 | [+0.1135, +0.2076] | higher | 848 | 69 |

### P23_P  (c_P23_P_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.0056 | [-0.0157, +0.0278] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.0213 | [-0.0289, +0.0711] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0138 | [-0.1163, +0.1266] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0014 | [-0.0186, +0.0314] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.3617 | [-0.2565, +1.1253] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0125 | [-0.0017, +0.0261] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0647 | [-0.1321, +0.0021] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +0.0780 | [-0.6134, +0.9157] | none | 100 | 21 |
| LONGITUDINAL | 2-6s | tg_err | +0.8698 | [-0.0580, +2.6508] | none | 66 | 19 |
| LONGITUDINAL | 2-6s | ttc_err | -0.2342 | [-1.7972, +1.0232] | none | 100 | 21 |
| LONGITUDINAL | full6s | acc_mae | +0.0102 | [-0.0016, +0.0230] | none | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0647 | [-0.1321, +0.0021] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.3014 | [-0.2159, +1.0272] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0642 | [-0.0366, +0.2032] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +0.1593 | [-0.5712, +0.9912] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0038 | [+0.0012, +0.0065] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0053 | [-0.0005, +0.0110] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0008 | [+0.0001, +0.0016] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +0.0451 | [-0.0065, +0.0945] | none | 994 | 71 |
| LATERAL | 0-2s | ade | +0.0171 | [-0.0058, +0.0404] | none | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0026 | [+0.0010, +0.0041] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0043 | [-0.0065, +0.0146] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0006 | [-0.0001, +0.0014] | none | 967 | 71 |
| LATERAL | 2-6s | xtrack | -0.0140 | [-0.4408, +0.4046] | none | 994 | 71 |
| LATERAL | 2-6s | ade | +0.0915 | [-0.1142, +0.2845] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0030 | [+0.0016, +0.0043] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0043 | [-0.0065, +0.0146] | none | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0010 | [+0.0003, +0.0019] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | -0.0140 | [-0.4408, +0.4046] | none | 994 | 71 |
| LATERAL | full6s | ade | +0.0667 | [-0.0766, +0.1991] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.0332 | [+0.0040, +0.0614] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.0382 | [+0.0101, +0.0654] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.0181 | [-0.0141, +0.0503] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | -0.0141 | [-0.0402, +0.0141] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0171 | [-0.0503, +0.0151] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | -0.0020 | [-0.0292, +0.0241] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | -0.0050 | [-0.0342, +0.0231] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | -0.0060 | [-0.0473, +0.0322] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0020 | [-0.0211, +0.0262] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [-0.3333, +0.3333] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | +7.4282 | [+0.7068, +14.0957] | better | 8 | 4 |
| STRATEGIC | full6s | s_false | +0.0082 | [-0.0139, +0.0281] | none | 856 | 70 |

### P_A1  (c_P_A1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.8671 | [+0.7987, +0.9368] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.8535 | [+0.7400, +0.9761] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.9458 | [+0.4642, +1.6728] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1194 | [+0.0660, +0.2078] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +2.6489 | [+1.5299, +3.9643] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1682 | [+0.1243, +0.2103] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.5660 | [+0.2951, +0.8545] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +1.9453 | [+1.0899, +2.8841] | worse | 104 | 20 |
| LONGITUDINAL | 2-6s | tg_err | +0.9785 | [-2.5694, +3.7080] | none | 74 | 19 |
| LONGITUDINAL | 2-6s | ttc_err | +1.3362 | [-0.3493, +3.0326] | none | 104 | 20 |
| LONGITUDINAL | full6s | acc_mae | +0.4012 | [+0.3604, +0.4394] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.5660 | [+0.2951, +0.8545] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +1.1643 | [+0.4460, +2.0940] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.1494 | [+0.0558, +0.2764] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +2.5210 | [+1.4947, +3.6827] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0886 | [+0.0803, +0.0962] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0911 | [+0.0792, +0.1030] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0142 | [+0.0106, +0.0187] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.1299 | [+0.9353, +1.3245] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.7027 | [+0.6289, +0.7754] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0279 | [+0.0239, +0.0318] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0609 | [+0.0414, +0.0821] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0055 | [+0.0032, +0.0081] | worse | 967 | 71 |
| LATERAL | 2-6s | xtrack | +3.9442 | [+3.0632, +4.8108] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +4.1917 | [+3.6041, +4.8028] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0481 | [+0.0444, +0.0518] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0609 | [+0.0414, +0.0821] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0085 | [+0.0059, +0.0113] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +3.9442 | [+3.0632, +4.8108] | worse | 994 | 71 |
| LATERAL | full6s | ade | +3.0287 | [+2.6117, +3.4560] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.4235 | [+0.3642, +0.4869] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2223 | [+0.1750, +0.2736] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3350 | [+0.2817, +0.3893] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.1036 | [+0.0583, +0.1509] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.1308 | [+0.0805, +0.1891] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0473 | [-0.0070, +0.1046] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.1539 | [+0.1087, +0.1992] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1670 | [+0.1147, +0.2193] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.1137 | [+0.0654, +0.1640] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.2857 | [-0.7500, +0.0000] | none | 7 | 4 |
| STRATEGIC | full6s | s_adh_deg | -15.6060 | [-25.5877, -11.5413] | worse | 7 | 4 |
| STRATEGIC | full6s | s_false | -0.0141 | [-0.0581, +0.0266] | none | 852 | 70 |

### P23_A1  (c_P23_A1_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | +0.8727 | [+0.8055, +0.9423] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | +0.8748 | [+0.7654, +0.9920] | worse | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.9320 | [+0.4476, +1.6688] | worse | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.1208 | [+0.0626, +0.2206] | worse | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +3.0105 | [+1.7079, +4.5373] | worse | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.1807 | [+0.1397, +0.2214] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | +0.5013 | [+0.2406, +0.7799] | worse | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | +2.1355 | [+1.3103, +3.2178] | worse | 104 | 19 |
| LONGITUDINAL | 2-6s | tg_err | +1.8293 | [+0.4979, +4.7131] | worse | 74 | 18 |
| LONGITUDINAL | 2-6s | ttc_err | +0.4918 | [-1.3058, +2.0144] | none | 104 | 19 |
| LONGITUDINAL | full6s | acc_mae | +0.4114 | [+0.3708, +0.4495] | worse | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | +0.5013 | [+0.2406, +0.7799] | worse | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +1.4658 | [+0.5792, +2.6692] | worse | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.2136 | [+0.0494, +0.4482] | worse | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +2.6803 | [+1.4015, +4.1154] | worse | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0923 | [+0.0843, +0.1000] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | +0.0964 | [+0.0839, +0.1094] | worse | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0151 | [+0.0113, +0.0196] | worse | 950 | 71 |
| LATERAL | 0-2s | xtrack | +1.1750 | [+0.9795, +1.3659] | worse | 994 | 71 |
| LATERAL | 0-2s | ade | +0.7198 | [+0.6452, +0.7916] | worse | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0304 | [+0.0267, +0.0342] | worse | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0651 | [+0.0448, +0.0882] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | +0.0062 | [+0.0038, +0.0089] | worse | 967 | 71 |
| LATERAL | 2-6s | xtrack | +3.9302 | [+3.0220, +4.8689] | worse | 994 | 71 |
| LATERAL | 2-6s | ade | +4.2832 | [+3.7051, +4.8740] | worse | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0511 | [+0.0476, +0.0546] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0651 | [+0.0448, +0.0882] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | +0.0094 | [+0.0066, +0.0125] | worse | 974 | 71 |
| LATERAL | full6s | xtrack | +3.9302 | [+3.0220, +4.8689] | worse | 994 | 71 |
| LATERAL | full6s | ade | +3.0954 | [+2.6872, +3.5129] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | +0.4567 | [+0.4024, +0.5121] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | +0.2606 | [+0.2072, +0.3129] | worse | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | +0.3531 | [+0.3068, +0.3994] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0895 | [+0.0443, +0.1368] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.1137 | [+0.0654, +0.1680] | worse | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0453 | [-0.0020, +0.0956] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.1489 | [+0.0986, +0.2012] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.1610 | [+0.1056, +0.2183] | worse | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.1157 | [+0.0603, +0.1731] | worse | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.3750 | [-1.0000, +0.2500] | none | 8 | 4 |
| STRATEGIC | full6s | s_adh_deg | -8.7395 | [-18.7686, +2.8448] | none | 8 | 4 |
| STRATEGIC | full6s | s_false | -0.0070 | [-0.0513, +0.0337] | none | 854 | 70 |

### P2-0_P  (c_P2-0_P_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.1797 | [-0.2482, -0.1137] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0816 | [-0.1957, +0.0244] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | -0.0867 | [-0.7135, +0.3136] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0368 | [-0.0252, +0.1085] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | -0.0703 | [-0.8937, +0.6942] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0026 | [-0.1279, +0.1160] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.6384 | [-1.7269, +0.1947] | none | 70 | 18 |
| LONGITUDINAL | 2-6s | tg_err | -1.8685 | [-4.6693, +0.0298] | none | 38 | 17 |
| LONGITUDINAL | 2-6s | ttc_err | +0.7828 | [-1.8007, +3.1650] | none | 70 | 18 |
| LONGITUDINAL | full6s | acc_mae | -0.0599 | [-0.0827, -0.0379] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0026 | [-0.1279, +0.1160] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.2915 | [-0.4252, +0.9715] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0951 | [-0.0008, +0.2258] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | -0.2462 | [-1.0068, +0.5214] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0101 | [+0.0025, +0.0174] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | -0.0094 | [-0.0236, +0.0041] | none | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0010 | [-0.0005, +0.0024] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.1431 | [-0.3075, +0.0322] | none | 994 | 71 |
| LATERAL | 0-2s | ade | -0.0958 | [-0.1599, -0.0295] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0243 | [+0.0086, +0.0394] | worse | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0009 | [-0.0017, -0.0001] | better | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.6327 | [-0.3299, +1.6595] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.1291 | [-0.6773, +0.4890] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0034 | [+0.0008, +0.0058] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0243 | [+0.0086, +0.0394] | worse | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0000 | [-0.0009, +0.0009] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.6327 | [-0.3299, +1.6595] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.1180 | [-0.5026, +0.3116] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0342 | [-0.0855, +0.0191] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | -0.0553 | [-0.1127, -0.0010] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0080 | [-0.0564, +0.0412] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0010 | [-0.0020, +0.0050] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | +0.0000 | [-0.0030, +0.0030] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0020 | [-0.0020, +0.0070] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0101 | [-0.0211, +0.0402] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0302 | [-0.0101, +0.0704] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0030 | [-0.0221, +0.0272] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | -0.1429 | [-0.6000, +0.0000] | none | 7 | 4 |
| STRATEGIC | full6s | s_adh_deg | +3.2923 | [-4.7404, +8.6971] | none | 7 | 4 |
| STRATEGIC | full6s | s_false | +0.0433 | [+0.0142, +0.0734] | higher | 855 | 69 |

### P23-2-0_P23  (c_P23-2-0_P23_s1.json; 994 windows / 71 episodes)

| family | band | metric | a - b | CI | label | n windows | n episodes |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | 0-2s | acc_mae | -0.1596 | [-0.2254, -0.0973] | better | 994 | 71 |
| LONGITUDINAL | 0-2s | speed_err | -0.0649 | [-0.1716, +0.0431] | none | 994 | 71 |
| LONGITUDINAL | 0-2s | hw_err | +0.0456 | [-0.6500, +0.5393] | none | 214 | 32 |
| LONGITUDINAL | 0-2s | tg_err | +0.0277 | [-0.0513, +0.1019] | none | 182 | 31 |
| LONGITUDINAL | 0-2s | ttc_err | +0.0236 | [-1.1511, +1.0886] | none | 214 | 32 |
| LONGITUDINAL | 2-6s | acc_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | speed_err | -0.0334 | [-0.1481, +0.0739] | none | 994 | 71 |
| LONGITUDINAL | 2-6s | hw_err | -0.5831 | [-1.6737, +0.2945] | none | 74 | 20 |
| LONGITUDINAL | 2-6s | tg_err | +0.4001 | [-0.6915, +2.1140] | none | 44 | 18 |
| LONGITUDINAL | 2-6s | ttc_err | +1.4977 | [+0.0181, +3.0186] | worse | 74 | 20 |
| LONGITUDINAL | full6s | acc_mae | -0.0532 | [-0.0751, -0.0324] | better | 994 | 71 |
| LONGITUDINAL | full6s | speed_err | -0.0334 | [-0.1481, +0.0739] | none | 994 | 71 |
| LONGITUDINAL | full6s | hw_err | +0.1056 | [-1.2097, +1.3557] | none | 214 | 32 |
| LONGITUDINAL | full6s | tg_err | +0.0060 | [-0.1852, +0.1775] | none | 182 | 31 |
| LONGITUDINAL | full6s | ttc_err | +0.4314 | [-0.9135, +1.5973] | none | 214 | 32 |
| LATERAL | 0-2s | yaw_mae | +0.0086 | [+0.0012, +0.0167] | worse | 994 | 71 |
| LATERAL | 0-2s | head_err | -0.0146 | [-0.0285, -0.0007] | better | 994 | 71 |
| LATERAL | 0-2s | curv_err | +0.0007 | [-0.0010, +0.0024] | none | 950 | 71 |
| LATERAL | 0-2s | xtrack | -0.1637 | [-0.3196, -0.0073] | better | 994 | 71 |
| LATERAL | 0-2s | ade | -0.1011 | [-0.1594, -0.0417] | better | 994 | 71 |
| LATERAL | 2-6s | yaw_mae | +0.0000 | [+0.0000, +0.0000] | none | 994 | 71 |
| LATERAL | 2-6s | head_err | +0.0102 | [-0.0065, +0.0271] | none | 994 | 71 |
| LATERAL | 2-6s | curv_err | -0.0012 | [-0.0020, -0.0005] | better | 967 | 71 |
| LATERAL | 2-6s | xtrack | +0.3634 | [-0.6625, +1.4498] | none | 994 | 71 |
| LATERAL | 2-6s | ade | -0.2049 | [-0.7370, +0.3324] | none | 994 | 71 |
| LATERAL | full6s | yaw_mae | +0.0029 | [+0.0004, +0.0056] | worse | 994 | 71 |
| LATERAL | full6s | head_err | +0.0102 | [-0.0065, +0.0271] | none | 994 | 71 |
| LATERAL | full6s | curv_err | -0.0006 | [-0.0015, +0.0003] | none | 974 | 71 |
| LATERAL | full6s | xtrack | +0.3634 | [-0.6625, +1.4498] | none | 994 | 71 |
| LATERAL | full6s | ade | -0.1703 | [-0.5383, +0.2052] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_man_dis | -0.0473 | [-0.0946, +0.0030] | none | 994 | 71 |
| TACTICAL | 0-2s | tac_lat_dis | -0.0915 | [-0.1479, -0.0342] | better | 994 | 71 |
| TACTICAL | 0-2s | tac_lon_dis | -0.0010 | [-0.0493, +0.0463] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_man_dis | +0.0020 | [-0.0040, +0.0091] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lat_dis | -0.0030 | [-0.0070, +0.0010] | none | 994 | 71 |
| TACTICAL | 2-6s | tac_lon_dis | +0.0010 | [-0.0060, +0.0080] | none | 994 | 71 |
| TACTICAL | full6s | tac_man_dis | +0.0201 | [-0.0131, +0.0554] | none | 994 | 71 |
| TACTICAL | full6s | tac_lat_dis | +0.0091 | [-0.0372, +0.0584] | none | 994 | 71 |
| TACTICAL | full6s | tac_lon_dis | +0.0030 | [-0.0262, +0.0332] | none | 994 | 71 |
| STRATEGIC | 0-2s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 0-2s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_adh_deg | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | 2-6s | s_false | n/a | [n/a, n/a] | n/a | 0 | 0 |
| STRATEGIC | full6s | s_adh | +0.0000 | [+0.0000, +0.0000] | none | 10 | 4 |
| STRATEGIC | full6s | s_adh_deg | +2.9132 | [-1.6329, +9.3410] | none | 10 | 4 |
| STRATEGIC | full6s | s_false | +0.0188 | [-0.0103, +0.0484] | none | 853 | 70 |

### Mechanism rows (beside, no bar)

| arm | REACH median cost1 ratio vs A0 | REACH mean log-ratio [CI] | COST2 median ratio (batch 0) | image resid level 2 max | level 3 max |
|---|---|---|---|---|---|
| P | 0.834 | -0.1800 [-0.2221, -0.1381] | 2.145 | 2.49e-07 | None |
| P23 | 0.848 | -0.1326 [-0.1782, -0.0854] | 1.872 | 2.65e-07 | 9.70393898569455e-07 |
| RS | 12.374 | +2.5002 [+2.3919, +2.6098] | 2.145 | 2.49e-07 | None |

### DIR-ACC (beside, no bar)

| arm | r | CI | n windows | n episodes |
|---|---|---|---|---|
| A0 | +0.4109 | [+0.3502, +0.4682] | 994 | 71 |
| P | +0.3976 | [+0.3367, +0.4556] | 994 | 71 |
| P23 | +0.3993 | [+0.3365, +0.4601] | 994 | 71 |
| RS | +0.2729 | [+0.2079, +0.3313] | 994 | 71 |
| P2-0 | +0.4067 | [+0.3420, +0.4719] | 994 | 71 |
| P23-2-0 | +0.4098 | [+0.3493, +0.4698] | 994 | 71 |

C-RS on SCORE (RS - A0, 0-2 s acc_mae): {'delta': 0.3287, 'lo': 0.2282, 'hi': 0.4304, 'label': 'worse', 'ok': True}

STRATEGIC / TACTICAL blocks: see `families` in RESULT_AMEND4.json (strategic_block with n per cell and the not-computed entries with reasons).

