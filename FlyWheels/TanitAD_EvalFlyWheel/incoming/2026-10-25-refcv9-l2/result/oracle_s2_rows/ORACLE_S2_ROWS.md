# L2 ORACLE-class arm vs BASE -- S2 (0.5-2 s) four-family rows

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: 'another draw of episodes' only; NOT another training run, NOT another inference run (floor block)
evidence class: MEASURED. Controls: {"K1_S2_stage_bit_equal_banked": true, "K2_S2_compliant_n": true, "K3_oracle_trace_final_n": true, "K4_identity_BASEos_vs_itself_max_abs_delta": {"S2_compliant|stage": 0.0, "S2_compliant|nostage": 0.0, "S2_all|stage": 0.0, "S2_all|nostage": 0.0}}
Not computed: LONGITUDINAL distance-keeping (no lead track in the banked dumps; n = 0); STRATEGIC (--no-strategic; n = 0); false/missed-decel (defined on the 6 s horizon (last chord 5->6 s): see L2_RESULT.md S6 rows)

## S2_compliant (4159 windows / 138 episodes), stage: oracle vs BASE-os (delta = oracle - BASE-os; FLOOR = BASE-os s1 - s0)

| family | metric | oracle | BASE-os | delta [lo, hi] | sep | worse | FLOOR delta [lo, hi] |
|---|---|---|---|---|---|---|---|
| ADE | ADE (mean over the surface's instants) (m) | 0.2542 | 0.2521 | +0.0021 [-0.0035, +0.0074] | False | False | +0.0008 [-0.0016, +0.0031] |
| ADE | FDE (final instant) (m) | 0.5419 | 0.5403 | +0.0015 [-0.0110, +0.0136] | False | False | +0.0010 [-0.0043, +0.0059] |
| LONGITUDINAL | speed MAE (m/s) | 0.2281 | 0.2289 | -0.0008 [-0.0066, +0.0045] | False | False | +0.0002 [-0.0019, +0.0023] |
| LONGITUDINAL | along-track MAE (m) | 0.2063 | 0.2054 | +0.0009 [-0.0045, +0.0057] | False | False | +0.0004 [-0.0017, +0.0025] |
| LONGITUDINAL | |along-track| at the final instant (m) | 0.4477 | 0.4497 | -0.0020 [-0.0144, +0.0091] | False | False | +0.0001 [-0.0046, +0.0046] |
| LONGITUDINAL | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0.8754 | 0.8737 | +0.0017 [-0.0029, +0.0066] | False | False | +0.0003 [-0.0025, +0.0033] |
| LONGITUDINAL | accel MAE (m/s2) | 0.2980 | 0.3023 | -0.0043 [-0.0114, +0.0022] | False | False | -0.0008 [-0.0032, +0.0015] |
| LATERAL | heading MAE (deg) | 1.5306 | 1.5028 | +0.0278 [+0.0007, +0.0579] | True | True | +0.0025 [-0.0099, +0.0148] |
| LATERAL | curvature MAE (masked) (1/m) | 0.0040 | 0.0039 | +0.0001 [-0.0000, +0.0003] | False | False | -0.0000 [-0.0001, +0.0001] |
| LATERAL | yaw-rate MAE (masked) (rad/s) | 0.0184 | 0.0177 | +0.0007 [+0.0001, +0.0013] | True | True | -0.0001 [-0.0004, +0.0001] |
| LATERAL | cross-track MAE (m) | 0.0971 | 0.0954 | +0.0017 [-0.0012, +0.0046] | False | False | +0.0004 [-0.0012, +0.0020] |
| LATERAL | |cross-track| at the final instant (m) | 0.1976 | 0.1936 | +0.0040 [-0.0023, +0.0104] | False | False | +0.0009 [-0.0029, +0.0044] |
| TACTICAL | trajectory-derived lateral decision correct (frac) | 0.9598 | 0.9632 | -0.0034 [-0.0077, +0.0005] | False | False | +0.0010 [-0.0014, +0.0034] |
| TACTICAL | trajectory-derived longitudinal decision correct (frac) | 0.8521 | 0.8545 | -0.0024 [-0.0098, +0.0051] | False | False | +0.0012 [-0.0035, +0.0059] |

E1-style ADE(0-2 s) minus echo `ha0_ext`: oracle -0.0265 [-0.0377, -0.0156]; BASE-os -0.0286 [-0.0410, -0.0162]; BASE-bat -0.0291 [-0.0416, -0.0169]

## S2_compliant (4159 windows / 138 episodes), nostage: oracle vs BASE-os (delta = oracle - BASE-os; FLOOR = BASE-os s1 - s0)

| family | metric | oracle | BASE-os | delta [lo, hi] | sep | worse | FLOOR delta [lo, hi] |
|---|---|---|---|---|---|---|---|
| ADE | ADE (mean over the surface's instants) (m) | 0.2795 | 0.2805 | -0.0009 [-0.0099, +0.0070] | False | False | +0.0006 [-0.0030, +0.0040] |
| ADE | FDE (final instant) (m) | 0.6009 | 0.6091 | -0.0082 [-0.0290, +0.0105] | False | False | -0.0002 [-0.0079, +0.0074] |
| LONGITUDINAL | speed MAE (m/s) | 0.2633 | 0.2689 | -0.0057 [-0.0157, +0.0035] | False | False | -0.0001 [-0.0038, +0.0035] |
| LONGITUDINAL | along-track MAE (m) | 0.2336 | 0.2354 | -0.0018 [-0.0108, +0.0059] | False | False | +0.0005 [-0.0028, +0.0039] |
| LONGITUDINAL | |along-track| at the final instant (m) | 0.5114 | 0.5218 | -0.0104 [-0.0314, +0.0081] | False | False | -0.0003 [-0.0078, +0.0076] |
| LONGITUDINAL | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0.8453 | 0.8438 | +0.0015 [-0.0049, +0.0085] | False | False | -0.0011 [-0.0050, +0.0031] |
| LONGITUDINAL | accel MAE (m/s2) | 0.3696 | 0.3839 | -0.0143 [-0.0272, -0.0027] | True | False | -0.0018 [-0.0061, +0.0026] |
| LATERAL | heading MAE (deg) | 1.5299 | 1.5017 | +0.0282 [+0.0029, +0.0560] | True | True | +0.0032 [-0.0096, +0.0159] |
| LATERAL | curvature MAE (masked) (1/m) | 0.0039 | 0.0038 | +0.0002 [+0.0000, +0.0003] | True | True | +0.0000 [-0.0001, +0.0001] |
| LATERAL | yaw-rate MAE (masked) (rad/s) | 0.0185 | 0.0178 | +0.0007 [+0.0001, +0.0013] | True | True | -0.0001 [-0.0004, +0.0002] |
| LATERAL | cross-track MAE (m) | 0.0978 | 0.0964 | +0.0014 [-0.0014, +0.0043] | False | False | +0.0006 [-0.0011, +0.0022] |
| LATERAL | |cross-track| at the final instant (m) | 0.1995 | 0.1966 | +0.0030 [-0.0031, +0.0092] | False | False | +0.0013 [-0.0025, +0.0049] |
| TACTICAL | trajectory-derived lateral decision correct (frac) | 0.9582 | 0.9620 | -0.0038 [-0.0078, -0.0005] | True | True | +0.0010 [-0.0017, +0.0034] |
| TACTICAL | trajectory-derived longitudinal decision correct (frac) | 0.8329 | 0.8271 | +0.0058 [-0.0036, +0.0160] | False | False | -0.0002 [-0.0066, +0.0061] |

E1-style ADE(0-2 s) minus echo `ha0_ext`: oracle -0.0012 [-0.0158, +0.0136]; BASE-os -0.0002 [-0.0174, +0.0180]; BASE-bat -0.0017 [-0.0190, +0.0160]

## S2_all (4754 windows / 139 episodes), stage: oracle vs BASE-os (delta = oracle - BASE-os; FLOOR = BASE-os s1 - s0)

| family | metric | oracle | BASE-os | delta [lo, hi] | sep | worse | FLOOR delta [lo, hi] |
|---|---|---|---|---|---|---|---|
| ADE | ADE (mean over the surface's instants) (m) | 0.2683 | 0.2646 | +0.0037 [-0.0021, +0.0093] | False | False | +0.0009 [-0.0015, +0.0033] |
| ADE | FDE (final instant) (m) | 0.5708 | 0.5662 | +0.0045 [-0.0082, +0.0172] | False | False | +0.0014 [-0.0036, +0.0062] |
| LONGITUDINAL | speed MAE (m/s) | 0.2395 | 0.2406 | -0.0011 [-0.0068, +0.0039] | False | False | +0.0007 [-0.0015, +0.0028] |
| LONGITUDINAL | along-track MAE (m) | 0.2165 | 0.2157 | +0.0008 [-0.0044, +0.0056] | False | False | +0.0007 [-0.0015, +0.0029] |
| LONGITUDINAL | |along-track| at the final instant (m) | 0.4683 | 0.4709 | -0.0026 [-0.0148, +0.0081] | False | False | +0.0011 [-0.0038, +0.0055] |
| LONGITUDINAL | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0.8676 | 0.8657 | +0.0019 [-0.0024, +0.0066] | False | False | -0.0006 [-0.0035, +0.0022] |
| LONGITUDINAL | accel MAE (m/s2) | 0.3082 | 0.3129 | -0.0047 [-0.0114, +0.0013] | False | False | +0.0000 [-0.0024, +0.0024] |
| LATERAL | heading MAE (deg) | 1.4745 | 1.4304 | +0.0441 [+0.0104, +0.0783] | True | True | -0.0008 [-0.0128, +0.0117] |
| LATERAL | curvature MAE (masked) (1/m) | 0.0039 | 0.0037 | +0.0002 [+0.0000, +0.0003] | True | True | -0.0000 [-0.0001, +0.0000] |
| LATERAL | yaw-rate MAE (masked) (rad/s) | 0.0192 | 0.0182 | +0.0011 [+0.0004, +0.0018] | True | True | -0.0002 [-0.0005, +0.0000] |
| LATERAL | cross-track MAE (m) | 0.1052 | 0.1014 | +0.0038 [+0.0004, +0.0072] | True | True | -0.0000 [-0.0016, +0.0016] |
| LATERAL | |cross-track| at the final instant (m) | 0.2140 | 0.2057 | +0.0083 [+0.0010, +0.0158] | True | True | -0.0003 [-0.0038, +0.0032] |
| TACTICAL | trajectory-derived lateral decision correct (frac) | 0.9590 | 0.9626 | -0.0036 [-0.0082, +0.0015] | False | False | +0.0008 [-0.0017, +0.0034] |
| TACTICAL | trajectory-derived longitudinal decision correct (frac) | 0.8452 | 0.8464 | -0.0013 [-0.0080, +0.0055] | False | False | +0.0015 [-0.0027, +0.0059] |

E1-style ADE(0-2 s) minus echo `ha0_ext`: oracle -0.0203 [-0.0314, -0.0087]; BASE-os -0.0240 [-0.0362, -0.0111]; BASE-bat -0.0247 [-0.0369, -0.0120]

## S2_all (4754 windows / 139 episodes), nostage: oracle vs BASE-os (delta = oracle - BASE-os; FLOOR = BASE-os s1 - s0)

| family | metric | oracle | BASE-os | delta [lo, hi] | sep | worse | FLOOR delta [lo, hi] |
|---|---|---|---|---|---|---|---|
| ADE | ADE (mean over the surface's instants) (m) | 0.2977 | 0.2976 | +0.0001 [-0.0091, +0.0085] | False | False | +0.0009 [-0.0026, +0.0045] |
| ADE | FDE (final instant) (m) | 0.6386 | 0.6454 | -0.0068 [-0.0285, +0.0129] | False | False | +0.0009 [-0.0069, +0.0085] |
| LONGITUDINAL | speed MAE (m/s) | 0.2808 | 0.2872 | -0.0065 [-0.0168, +0.0026] | False | False | +0.0004 [-0.0033, +0.0039] |
| LONGITUDINAL | along-track MAE (m) | 0.2491 | 0.2515 | -0.0023 [-0.0115, +0.0057] | False | False | +0.0009 [-0.0026, +0.0044] |
| LONGITUDINAL | |along-track| at the final instant (m) | 0.5437 | 0.5561 | -0.0123 [-0.0340, +0.0068] | False | False | +0.0010 [-0.0071, +0.0087] |
| LONGITUDINAL | target-speed accuracy (+-0.5 m/s, fraction of steps) (frac) | 0.8321 | 0.8288 | +0.0033 [-0.0030, +0.0097] | False | False | +0.0003 [-0.0039, +0.0044] |
| LONGITUDINAL | accel MAE (m/s2) | 0.3858 | 0.4009 | -0.0151 [-0.0281, -0.0040] | True | False | -0.0008 [-0.0054, +0.0036] |
| LATERAL | heading MAE (deg) | 1.4721 | 1.4280 | +0.0441 [+0.0128, +0.0772] | True | True | -0.0002 [-0.0118, +0.0119] |
| LATERAL | curvature MAE (masked) (1/m) | 0.0038 | 0.0036 | +0.0002 [+0.0001, +0.0004] | True | True | +0.0000 [-0.0001, +0.0001] |
| LATERAL | yaw-rate MAE (masked) (rad/s) | 0.0193 | 0.0182 | +0.0011 [+0.0004, +0.0018] | True | True | -0.0002 [-0.0004, +0.0001] |
| LATERAL | cross-track MAE (m) | 0.1052 | 0.1018 | +0.0034 [+0.0002, +0.0067] | True | True | +0.0001 [-0.0016, +0.0017] |
| LATERAL | |cross-track| at the final instant (m) | 0.2142 | 0.2072 | +0.0070 [+0.0003, +0.0141] | True | True | -0.0003 [-0.0038, +0.0035] |
| TACTICAL | trajectory-derived lateral decision correct (frac) | 0.9579 | 0.9619 | -0.0040 [-0.0084, +0.0006] | False | False | +0.0006 [-0.0019, +0.0032] |
| TACTICAL | trajectory-derived longitudinal decision correct (frac) | 0.8229 | 0.8164 | +0.0065 [-0.0021, +0.0158] | False | False | -0.0002 [-0.0063, +0.0061] |

E1-style ADE(0-2 s) minus echo `ha0_ext`: oracle +0.0091 [-0.0051, +0.0244]; BASE-os +0.0090 [-0.0076, +0.0262]; BASE-bat +0.0070 [-0.0097, +0.0249]
