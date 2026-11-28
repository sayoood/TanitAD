FAIL (G_strict)

# SELFSHARP RESULT (mechanical; PREREG_SELFSHARP sha256 cb2dc8ea5c2a)

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'would another draw of EPISODES say this?' ONLY; NOT another training run (one refcv8 training seed), NOT another inference run (one inference seed, 0)

MD (S6 compliant, stage on): BASE-os 0.4695 -> SELFSHARP 0.3897 (ORACLE 0.3412)
FD: BASE-os 0.1395 -> SELFSHARP 0.1503 (ORACLE 0.1379)
ceiling share 0.622 [0.344, 0.903] (descriptive)

| bar | value | cell |
|---|---|---|
| P1 MD separated BELOW BASE-os | True | -0.0798 [-0.1317, -0.0357] separated=True |
| P2 FD NOT separated above BASE-os | True | +0.0108 [-0.0165, +0.0358] separated=False |
| G-strict no S6 row separated worse | False | LAT_cross_abs_final_m, LAT_cross_mae_m, LAT_curvature_mae_1pm_valid, LAT_heading_mae_deg, LAT_yaw_rate_mae_radps_valid, TAC_traj_lat_correct |
| (G-channel) rows ORACLE also degrades | False | LAT_curvature_mae_1pm_valid, LAT_heading_mae_deg, LAT_yaw_rate_mae_radps_valid |

controls failed: none
pending: none
families not computed: {'LONGITUDINAL distance-keeping (headway/TTC)': 'no lead-agent track in the banked refcv8 dumps; n = 0; named work item', 'STRATEGIC': 'refcv8 runs --no-strategic; n = 0', 'S2 rows': 'not computed in this run (prereg: PENDING)'}

NEXT LEVER: sharpening degrades rows the ORACLE force does not: the cost is in the head's wrong choices, not the channel -- gate the sharpen on head margin.

ZZSSVERDICTENDZZ