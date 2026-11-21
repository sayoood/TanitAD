# FAM4: L_s0 minus BPRIME_s0 (T1-kin, first tick, eval-141; a diagnostic, never driving performance)

Counts: 2399 windows, 2365 with has, 2351 ok, 117 stop-point windows, 139 episodes (has).
delta = L_s0 - BPRIME_s0; errors lower-better, TAC accuracies / AUROC / hit higher-better. CI = paired episode-cluster bootstrap, B 2,000, seed 0 (another draw of EPISODES only). One training seed per arm: no replicate floor available for this pair. separated is necessary, not sufficient; no cell is a lever effect.

Controls: self-pair exact zero True (52 cells); window counts True; deliberate regression True.

## LONGITUDINAL

| cell | set | delta | CI | separated | side | n win | n ep |
|---|---|---|---|---|---|---|---|
| operative_0_2s accel_mae_mps2 | all | -0.2586 | [-0.3546, -0.1678] | yes | L better | 2399 | 141 |
| operative_0_2s along_mae_m | all | -0.0234 | [-0.0362, -0.0145] | yes | L better | 2399 | 141 |
| operative_0_2s dk_hw_abs_err_m | all | 0.0174 | [-0.0449, 0.0992] | no | L worse | 670 | 72 |
| operative_0_2s dk_tg_abs_err_s | all | -0.0102 | [-0.0246, 0.0034] | no | L better | 647 | 72 |
| operative_0_2s speed_mae_mps | all | -0.0542 | [-0.0776, -0.0339] | yes | L better | 2399 | 141 |
| tactical_2_6s accel_mae_mps2 | all | 0.2913 | [-0.0810, 0.7025] | no | L worse | 2399 | 141 |
| tactical_2_6s along_mae_m | all | -0.1631 | [-0.3467, 0.0290] | no | L better | 2399 | 141 |
| tactical_2_6s dk_hw_abs_err_m | all | 0.1578 | [-0.1466, 0.4723] | no | L worse | 531 | 59 |
| tactical_2_6s dk_tg_abs_err_s | all | 0.0581 | [-0.0005, 0.1249] | no | L worse | 519 | 59 |
| tactical_2_6s speed_mae_mps | all | -0.0354 | [-0.2011, 0.1392] | no | L better | 2399 | 141 |
| operative_0_2s accel_mae_mps2 | sp | 0.0853 | [-0.3126, 0.4739] | no | L worse | 117 | 25 |
| operative_0_2s along_mae_m | sp | 0.0138 | [-0.0112, 0.0387] | no | L worse | 117 | 25 |
| operative_0_2s dk_hw_abs_err_m | sp | 0.3635 | [-0.0922, 1.1057] | no | L worse | 46 | 14 |
| operative_0_2s dk_tg_abs_err_s | sp | 0.0456 | [-0.0052, 0.1217] | no | L worse | 46 | 14 |
| operative_0_2s speed_mae_mps | sp | 0.0246 | [-0.0412, 0.0837] | no | L worse | 117 | 25 |
| tactical_2_6s accel_mae_mps2 | sp | 1.0894 | [-0.4545, 2.9052] | no | L worse | 117 | 25 |
| tactical_2_6s along_mae_m | sp | 0.9094 | [0.2286, 1.5314] | yes | L worse | 117 | 25 |
| tactical_2_6s dk_hw_abs_err_m | sp | 0.0414 | [-0.5748, 0.6664] | no | L worse | 34 | 11 |
| tactical_2_6s dk_tg_abs_err_s | sp | 0.0181 | [-0.0650, 0.1114] | no | L worse | 33 | 11 |
| tactical_2_6s speed_mae_mps | sp | 0.6411 | [0.0175, 1.4858] | yes | L worse | 117 | 25 |

## LATERAL

| cell | set | delta | CI | separated | side | n win | n ep |
|---|---|---|---|---|---|---|---|
| operative_0_2s cross_mae_m | all | 0.0002 | [-0.0030, 0.0034] | no | L worse | 2399 | 141 |
| operative_0_2s curvature_mae_1pm | all | -0.0020 | [-0.0048, 0.0008] | no | L better | 2304 | 141 |
| operative_0_2s heading_mae_deg | all | -0.0184 | [-0.0632, 0.0218] | no | L better | 2313 | 141 |
| operative_0_2s yaw_rate_mae_degps | all | -0.1247 | [-0.2535, -0.0087] | yes | L better | 2304 | 141 |
| tactical_2_6s cross_mae_m | all | -0.0892 | [-0.2023, 0.0134] | no | L better | 2399 | 141 |
| tactical_2_6s curvature_mae_1pm | all | 0.0578 | [-0.0121, 0.1325] | no | L worse | 2338 | 141 |
| tactical_2_6s heading_mae_deg | all | 0.0476 | [-1.2958, 1.5043] | no | L worse | 2348 | 141 |
| tactical_2_6s yaw_rate_mae_degps | all | 9.5968 | [3.2816, 16.9620] | yes | L worse | 2338 | 141 |
| operative_0_2s cross_mae_m | sp | 0.0074 | [-0.0032, 0.0180] | no | L worse | 117 | 25 |
| operative_0_2s curvature_mae_1pm | sp | -0.0123 | [-0.0214, -0.0046] | yes | L better | 116 | 25 |
| operative_0_2s heading_mae_deg | sp | -0.0265 | [-0.1625, 0.1349] | no | L better | 116 | 25 |
| operative_0_2s yaw_rate_mae_degps | sp | -0.6164 | [-1.0088, -0.2359] | yes | L better | 116 | 25 |
| tactical_2_6s cross_mae_m | sp | -0.0877 | [-0.4575, 0.2591] | no | L better | 117 | 25 |
| tactical_2_6s curvature_mae_1pm | sp | 0.7493 | [0.2522, 1.3447] | yes | L worse | 113 | 25 |
| tactical_2_6s heading_mae_deg | sp | 8.8916 | [1.8087, 16.6114] | yes | L worse | 115 | 25 |
| tactical_2_6s yaw_rate_mae_degps | sp | 70.3101 | [26.3954, 120.8797] | yes | L worse | 113 | 25 |

## TACTICAL

| cell | set | delta | CI | separated | side | n win | n ep |
|---|---|---|---|---|---|---|---|
| tactical_decision TAC-LAT | all | 0.0421 | [0.0158, 0.0684] | yes | L better | 2399 | 141 |
| tactical_decision TAC-LON | all | 0.0296 | [-0.0033, 0.0643] | no | L better | 2399 | 141 |
| tactical_decision TAC-LAT | sp | 0.0171 | [-0.1000, 0.1683] | no | L better | 117 | 25 |
| tactical_decision TAC-LON | sp | -0.1368 | [-0.2308, -0.0476] | yes | L worse | 117 | 25 |
| cue|auroc p(STOP_POINT) vs stop_hit (has) |  | 0.0095 | [-0.0289, 0.0525] | no | L better | 2365 | 139 |
| cue|auroc lon-STOP vs regime stop (has) |  | 0.0110 | [-0.0198, 0.0429] | no | L better | 2365 | 139 |
| cue|goal-stop decision hit rate (stop-point windows, higher better) |  | 0.0769 | [-0.0094, 0.1717] | no | L better | 117 | 25 |
| cue|goal-stop decision fpr (other has-and-present, lower better) |  | 0.0246 | [0.0098, 0.0416] | yes | L worse | 2234 | 139 |

## PRIMARY (ADE beside)

| cell | set | delta | CI | separated | side | n win | n ep |
|---|---|---|---|---|---|---|---|
| operative_0_2s ade_m | all | -0.0229 | [-0.0362, -0.0137] | yes | L better | 2399 | 141 |
| operative_0_2s fde_m | all | -0.0363 | [-0.0692, -0.0138] | yes | L better | 2399 | 141 |
| tactical_2_6s ade_m | all | -0.2017 | [-0.3949, 0.0029] | no | L better | 2399 | 141 |
| tactical_2_6s fde_m | all | -0.7727 | [-1.3467, -0.2156] | yes | L better | 2399 | 141 |
| operative_0_2s ade_m | sp | 0.0163 | [-0.0079, 0.0401] | no | L worse | 117 | 25 |
| operative_0_2s fde_m | sp | 0.0080 | [-0.0811, 0.1026] | no | L worse | 117 | 25 |
| tactical_2_6s ade_m | sp | 0.8171 | [0.1396, 1.4410] | yes | L worse | 117 | 25 |
| tactical_2_6s fde_m | sp | 1.4390 | [-0.3321, 3.1192] | no | L worse | 117 | 25 |

## STRATEGIC

PAUSED (PI item 25); not a pass.

SEPARATED_LIST: [["all", "operative_0_2s", "ade_m", "PRIMARY", -0.02286, "L better"], ["all", "operative_0_2s", "fde_m", "PRIMARY", -0.03626, "L better"], ["all", "operative_0_2s", "speed_mae_mps", "LONGITUDINAL", -0.05417, "L better"], ["all", "operative_0_2s", "along_mae_m", "LONGITUDINAL", -0.02337, "L better"], ["all", "operative_0_2s", "accel_mae_mps2", "LONGITUDINAL", -0.25857, "L better"], ["all", "operative_0_2s", "yaw_rate_mae_degps", "LATERAL", -0.12469, "L better"], ["all", "tactical_2_6s", "fde_m", "PRIMARY", -0.77271, "L better"], ["all", "tactical_2_6s", "yaw_rate_mae_degps", "LATERAL", 9.59684, "L worse"], ["all", "tactical_decision", "TAC-LAT", "TACTICAL", 0.0421, "L better"], ["sp", "operative_0_2s", "yaw_rate_mae_degps", "LATERAL", -0.61641, "L better"], ["sp", "operative_0_2s", "curvature_mae_1pm", "LATERAL", -0.01235, "L better"], ["sp", "tactical_2_6s", "ade_m", "PRIMARY", 0.8171, "L worse"], ["sp", "tactical_2_6s", "speed_mae_mps", "LONGITUDINAL", 0.64109, "L worse"], ["sp", "tactical_2_6s", "along_mae_m", "LONGITUDINAL", 0.90939, "L worse"], ["sp", "tactical_2_6s", "heading_mae_deg", "LATERAL", 8.89156, "L worse"], ["sp", "tactical_2_6s", "yaw_rate_mae_degps", "LATERAL", 70.31007, "L worse"], ["sp", "tactical_2_6s", "curvature_mae_1pm", "LATERAL", 0.74926, "L worse"], ["sp", "tactical_decision", "TAC-LON", "TACTICAL", -0.13675, "L worse"]]

## What it says

- One training seed per arm and no replicate floor for this pair: the CIs answer "another draw of EPISODES" only, not "another training run". Separated is necessary, not sufficient; no cell below is called a lever effect.
- All-window operative (0-2 s) band: L_s0 reads lower error than BPRIME_s0 on ADE, FDE, speed, along-track, acceleration and yaw rate, all separated but small (e.g. speed -0.054 m/s, ADE -0.023 m). Cross-track, heading and curvature are not separated.
- All-window tactical (2-6 s) band: only FDE (-0.77 m, better) and yaw rate (+9.6 deg/s, worse) separate; tactical ADE, speed, along-track, acceleration and distance-keeping do not. The tactical yaw-rate cell is large and its CI is wide.
- Stop-point subset (117 windows, 25 episodes): the 2-6 s longitudinal and lateral errors are worse for L_s0 (ADE +0.82 m, speed +0.64 m/s, along +0.91 m, heading, curvature, yaw rate), while 0-2 s yaw rate and curvature are better. Stop-point TAC-LON accuracy is lower (-0.137); all-window TAC-LAT is higher (+0.042).
- Stop cue: AUROC deltas not separated (+0.010, +0.011; this pair's L-vs-B-prime result matches the S0-L NO-GAIN verdict); goal-stop hit rate +0.077 not separated, false-positive rate +0.025 (worse) separated. STRATEGIC: PAUSED (PI item 25).
