# NAV_BRAKE — H-R9-NAV-BRAKE-1: **NAV-DRIVES-BRAKE (SINGLE-INFERENCE-SEED)** (tree step40000ema_unc; prereg sha256 2ea9049b8be6…)

tier: T1 open loop, banked plans, no forward; estimator: paired episode-cluster bootstrap B 2000 (pair_read_r8.paired_cell); answers another draw of EPISODES only; GT-cruise = S6 compliant windows with GT class cruise (thr 1.5 m/s, dt 1 s)

## step40000ema_unc — label NAV-DRIVES-BRAKE (SINGLE-INFERENCE-SEED) (controls ALL_PASS)
* seed 0: GT-cruise 1856 (TURN 579, FOLLOW 1277); TURN share of r8 false-brake 0.647 vs of GT-cruise 0.312; nav_zero releases {'TURN': 0.7133333333333334, 'FOLLOW': 0.036585365853658534}
  * dFD(nav_zero) = FD(nav_true) − FD(nav_zero): TURN +0.3575 [+0.2282, +0.4690] n 579 | FOLLOW -0.1323 [-0.1844, -0.0882] n 1277 | ALL +0.0205 [-0.0488, +0.0934] n 1856
  * dFD(nav_flipped) = FD(nav_true) − FD(nav_flipped): TURN -0.0345 [-0.0710, +0.0016] n 579 | FOLLOW +0.0000 [-0.0094, +0.0100] n 1277 | ALL -0.0108 [-0.0254, +0.0022] n 1856
  * dFD(nav_shuffled) = FD(nav_true) − FD(nav_shuffled): TURN +0.2712 [+0.1815, +0.3519] n 579 | FOLLOW -0.2216 [-0.2612, -0.1825] n 1277 | ALL -0.0679 [-0.1309, -0.0033] n 1856
  * TURN|STRAIGHT6: n 382, FD r8 0.5418848167539267, r7 0.09162303664921466, r8 nav_zero 0.15968586387434555; r8 − r7 +0.4503 [+0.2924, +0.5939]
  * TURN|CURVED6: n 197, FD r8 0.4720812182741117, r7 0.15736040609137056, r8 nav_zero 0.16243654822335024; r8 − r7 +0.3147 [+0.1636, +0.4420]
  * FOLLOW|STRAIGHT6: n 700, FD r8 0.10714285714285714, r7 0.06, r8 nav_zero 0.2057142857142857; r8 − r7 +0.0471 [+0.0142, +0.0882]
  * FOLLOW|CURVED6: n 577, FD r8 0.15424610051993068, r7 0.10571923743500866, r8 nav_zero 0.3275563258232236; r8 − r7 +0.0485 [+0.0038, +0.0935]
* seed 1: GT-cruise 1856 (TURN 579, FOLLOW 1277); TURN share of r8 false-brake 0.630 vs of GT-cruise 0.312; nav_zero releases None
  * TURN|STRAIGHT6: n 382, FD r8 0.5418848167539267, r7 0.10471204188481675, r8 nav_zero None; r8 − r7 +0.4372 [+0.2701, +0.5897]
  * TURN|CURVED6: n 197, FD r8 0.4416243654822335, r7 0.17258883248730963, r8 nav_zero None; r8 − r7 +0.2690 [+0.1000, +0.4055]
  * FOLLOW|STRAIGHT6: n 700, FD r8 0.11571428571428571, r7 0.05714285714285714, r8 nav_zero None; r8 − r7 +0.0586 [+0.0207, +0.1025]
  * FOLLOW|CURVED6: n 577, FD r8 0.15944540727902945, r7 0.10398613518197573, r8 nav_zero None; r8 − r7 +0.0555 [+0.0145, +0.1047]

## step30000ema_unc — label NAV-DRIVES-BRAKE (SINGLE-INFERENCE-SEED) (controls ALL_PASS)
* seed 0: GT-cruise 1856 (TURN 579, FOLLOW 1277); TURN share of r8 false-brake 0.696 vs of GT-cruise 0.312; nav_zero releases {'TURN': 0.685131195335277, 'FOLLOW': 0.04666666666666667}
  * dFD(nav_zero) = FD(nav_true) − FD(nav_zero): TURN +0.3886 [+0.2611, +0.5029] n 579 | FOLLOW -0.1182 [-0.1596, -0.0829] n 1277 | ALL +0.0399 [-0.0275, +0.1098] n 1856
  * dFD(nav_flipped) = FD(nav_true) − FD(nav_flipped): TURN -0.0363 [-0.0679, -0.0061] n 579 | FOLLOW +0.0086 [-0.0024, +0.0205] n 1277 | ALL -0.0054 [-0.0186, +0.0075] n 1856
  * dFD(nav_shuffled) = FD(nav_true) − FD(nav_shuffled): TURN +0.3022 [+0.2109, +0.3849] n 579 | FOLLOW -0.2318 [-0.2713, -0.1909] n 1277 | ALL -0.0652 [-0.1332, +0.0027] n 1856
  * TURN|STRAIGHT6: n 382, FD r8 0.612565445026178, r7 0.09162303664921466, r8 nav_zero 0.19109947643979058; r8 − r7 +0.5209 [+0.3527, +0.6676]
  * TURN|CURVED6: n 197, FD r8 0.5532994923857868, r7 0.15736040609137056, r8 nav_zero 0.22842639593908629; r8 − r7 +0.3959 [+0.2088, +0.5407]
  * FOLLOW|STRAIGHT6: n 700, FD r8 0.09, r7 0.06, r8 nav_zero 0.17142857142857143; r8 − r7 +0.0300 [+0.0101, +0.0543]
  * FOLLOW|CURVED6: n 577, FD r8 0.15077989601386482, r7 0.10571923743500866, r8 nav_zero 0.31369150779896016; r8 − r7 +0.0451 [+0.0103, +0.0835]
* seed 1: GT-cruise 1856 (TURN 579, FOLLOW 1277); TURN share of r8 false-brake 0.688 vs of GT-cruise 0.312; nav_zero releases None
  * TURN|STRAIGHT6: n 382, FD r8 0.5863874345549738, r7 0.10471204188481675, r8 nav_zero None; r8 − r7 +0.4817 [+0.3189, +0.6316]
  * TURN|CURVED6: n 197, FD r8 0.5329949238578681, r7 0.17258883248730963, r8 nav_zero None; r8 − r7 +0.3604 [+0.1778, +0.5000]
  * FOLLOW|STRAIGHT6: n 700, FD r8 0.09428571428571429, r7 0.05714285714285714, r8 nav_zero None; r8 − r7 +0.0371 [+0.0151, +0.0625]
  * FOLLOW|CURVED6: n 577, FD r8 0.1438474870017331, r7 0.10398613518197573, r8 nav_zero None; r8 − r7 +0.0399 [+0.0037, +0.0794]

