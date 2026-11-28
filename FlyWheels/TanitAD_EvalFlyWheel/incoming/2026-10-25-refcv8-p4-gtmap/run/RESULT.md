# P4 GT-MAP ORACLE -- does refcv8's planner gain from a TRUE map in its BEV?

* Rendered 2026-10-25 22:20:59 +0100; prereg `PREREG_P4.md` sha256 `d9fc8245d14ff492bb9e0748204d5469c6f6c18be360ba0806b64cb0d0a9c6dc` (seal `d9fc8245d14ff492bb9e0748204d5469c6f6c18be360ba0806b64cb0d0a9c6dc`).
* **Tier: T0-ORACLE (privileged SAM3 GT map injected at inference) on the SHIP-R8 T1 self-action open-loop roll: a BOUND, never driving.**
* **Interval:** paired episode-cluster bootstrap (taniteval.ci), B 2000, seed 0, models FIXED: 'another draw of EPISODES' only; NOT another training run (one training seed, H-ESTIM-SEED-1); inference: seeds 0 and 1 + floors F (BASE s1-s0) and F' (contrast s1-s0).
* Model: refcv8 EMA-48,852 (md5 b4364029b68f29a33eeb4aebbbbd002b), per-frame nav feed, eval139 (4,754 windows / 139 episodes), os-only, RTX 4060.

## Verdict (PREREG s6, mechanical): **INCONCLUSIVE-INERT (a WRONG map injected the same way does not hurt the planner: the channel cannot carry map content)**

| seed | S6 ADE TRUE-OWNMAP | effect beyond floor | share of BEV_ZERO (perc2plan) |
|---|---|---|---|
| s0 | +0.0096 [-0.0213, +0.0418] | False | -0.008 |
| s1 | +0.0079 [-0.0247, +0.0438] | False | -0.007 |

### Floors (inference seed)

* S2: F (BASE s1-s0) = 0.0009 m +0.0009 [-0.0026, +0.0045]; F' (contrast s1-s0) = 0.0015 m; used max = **0.0015 m**
* S6: F (BASE s1-s0) = 0.0170 m +0.0170 [-0.0158, +0.0518]; F' (contrast s1-s0) = 0.0017 m; used max = **0.0170 m**

### Gates

```
{
 "validity": {
  "C1_s1": true,
  "C2": true,
  "C3": true,
  "C6": true,
  "C7": true,
  "C8": true,
  "applied_once": true
 },
 "reach": {
  "pass": true,
  "true_miou": 0.8484925579058097,
  "own0_miou": 0.22446229804453927,
  "gain": 0.6240302598612704,
  "thin": {
   "band0:lane": 0.8219626501204943,
   "band0:crosswalk": 0.8437518603369443,
   "band0:non-drv-edge": 0.6021949520588065,
   "band1:lane": 0.855946993595864,
   "band1:crosswalk": 0.9134960343255286,
   "band1:non-drv-edge": 0.6809168983749139
  },
  "bars": {
   "miou_min": 0.8,
   "thin_min": 0.5,
   "thin_bands": [
    0,
    1
   ],
   "gain_min": 0.2
  }
 },
 "liveness": {
  "0": {
   "cell": {
    "delta": 0.0175,
    "lo": -0.0196,
    "hi": 0.0564,
    "ci95": 0.038,
    "p_delta_gt0": 0.826,
    "separated": false,
    "reducer": "mean",
    "n_windows": 3668,
    "n_episodes": 139,
    "n_boot": 2000,
    "estimator": "paired_episode_cluster_bootstrap",
    "n_dropped_nonfinite": 0,
    "mean_new": 2.8248205602809766,
    "mean_ref": 2.8073516713371913
   },
   "separated_worse": false
  },
  "1": {
   "cell": {
    "delta": 0.0214,
    "lo": -0.0198,
    "hi": 0.0641,
    "ci95": 0.042,
    "p_delta_gt0": 0.846,
    "separated": false,
    "reducer": "mean",
    "n_windows": 3668,
    "n_episodes": 139,
    "n_boot": 2000,
    "estimator": "paired_episode_cluster_bootstrap",
    "n_dropped_nonfinite": 0,
    "mean_new": 2.840816343746226,
    "mean_ref": 2.819455975306796
   },
   "separated_worse": false
  }
 }
}
```

## Seed 0

### Controls

```
{
 "C2_base_capture_eq_dump": {
  "n": 4754,
  "max_abs": 0.0,
  "pass": true
 },
 "C1_base_reproduces": {
  "against": "C:\\Users\\Admin\\refcv8_eval_archive\\battery\\step48852ema_pf\\dump_s0",
  "n": 4754,
  "max_abs_os": 3.7817821502685547,
  "max_abs_plan8": 35.30845642089844,
  "tol": 1e-05,
  "binding": false,
  "pass": null,
  "note": "seed 0: the banked s0 is the FULL arm-set roll (another sampler stream)"
 },
 "C3_recompute_identity": {
  "checked": 4754,
  "equal": 4754,
  "maxdiff": 0.0,
  "own_calls": 4754,
  "pass": true
 },
 "C3b_no_target_windows_are_base": {
  "n_by_arm": {
   "TRUE": 68,
   "OWNMAP": 68,
   "SHUF": 0
  },
  "pass": true
 },
 "C8_gt_item_eq_store": {
  "checked": 50,
  "equal": 50,
  "pass": true
 },
 "applied_once": {
  "arm_forwards": 14126,
  "applied_not_once": 0,
  "cap_not_once": 0,
  "mh_memo_hit": 14126,
  "mh_memo_miss": 0,
  "pass": true
 },
 "refine_repro": {
  "maxdiff_fp32_decode_vs_model_logits": 0.0,
  "argmax_agree_min": 1.0,
  "note": "CPU fp32 smoke: must be ~0; CUDA run: the model's logits ran under bf16 autocast -- recorded, not a gate"
 },
 "C7_gt_parity": {
  "n": 4754,
  "max_abs_g": 0.0,
  "pass": true
 }
}
```

### Census

```
{
 "forced": {
  "TRUE": 4686,
  "OWNMAP": 4686,
  "SHUF": 4754
 },
 "gt_label_windows": 4686,
 "rel_norm_median": {
  "TRUE": null,
  "OWNMAP": null,
  "SHUF": null
 },
 "loss0_mean": {
  "TRUE": null,
  "OWNMAP": null,
  "SHUF": null
 },
 "loss1_mean": {
  "TRUE": null,
  "OWNMAP": null,
  "SHUF": null
 },
 "inv_wall_mean_s": null,
 "wall_by_arm_s": {
  "OWN": 1964.4499999999998,
  "TRUE": 1157.6770000000001,
  "OWNMAP": 1152.2559999999999,
  "SHUF": 1169.625
 },
 "roll_wall_s": 5955.8
}
```

### S2 -- n_windows 4754, episodes 139 (all four families; arm - ref)

| family.metric | BASE mean | TRUE-OWNMAP | SHUF-OWNMAP | OWNMAP-BASE | TRUE-BASE | SHUF-BASE |
|---|---|---|---|---|---|---|
| ADE.ade_m | 0.2976 | +0.0012 [-0.0020, +0.0045] | +0.0042 [+0.0008, +0.0077] **worse** | +0.0080 [+0.0023, +0.0137] **worse** | +0.0092 [+0.0034, +0.0153] **worse** | +0.0122 [+0.0063, +0.0185] **worse** |
| ADE.fde_m | 0.64542 | +0.0012 [-0.0053, +0.0079] | +0.0073 [-0.0004, +0.0154] | +0.0152 [+0.0027, +0.0274] **worse** | +0.0164 [+0.0037, +0.0295] **worse** | +0.0225 [+0.0095, +0.0366] **worse** |
| LONGITUDINAL.LON_speed_mae_mps | 0.28721 | +0.0022 [-0.0009, +0.0055] | +0.0006 [-0.0028, +0.0041] | +0.0007 [-0.0043, +0.0057] | +0.0029 [-0.0020, +0.0080] | +0.0014 [-0.0043, +0.0067] |
| LONGITUDINAL.LON_along_mae_m | 0.25148 | +0.0019 [-0.0011, +0.0053] | +0.0002 [-0.0029, +0.0034] | +0.0018 [-0.0030, +0.0062] | +0.0037 [-0.0010, +0.0085] | +0.0019 [-0.0031, +0.0067] |
| LONGITUDINAL.LON_along_abs_final_m | 0.55606 | +0.0033 [-0.0026, +0.0101] | -0.0016 [-0.0088, +0.0055] | +0.0014 [-0.0088, +0.0114] | +0.0048 [-0.0056, +0.0154] | -0.0002 [-0.0117, +0.0104] |
| LONGITUDINAL.LON_speed_within_0p5 | 0.82878 | -0.0023 [-0.0051, +0.0005] | -0.0014 [-0.0050, +0.0025] | +0.0001 [-0.0041, +0.0045] | -0.0022 [-0.0065, +0.0020] | -0.0013 [-0.0059, +0.0031] |
| LONGITUDINAL.LON_accel_mae_mps2 | 0.40092 | +0.0020 [-0.0012, +0.0055] | +0.0005 [-0.0036, +0.0046] | -0.0015 [-0.0076, +0.0041] | +0.0005 [-0.0052, +0.0066] | -0.0010 [-0.0082, +0.0060] |
| LATERAL.LAT_heading_mae_deg | 1.4274 | -0.0012 [-0.0220, +0.0225] | +0.0394 [+0.0126, +0.0675] **worse** | +0.0817 [+0.0441, +0.1195] **worse** | +0.0805 [+0.0414, +0.1228] **worse** | +0.1216 [+0.0802, +0.1700] **worse** |
| LATERAL.LAT_curvature_mae_1pm_valid | 0.0036464 | -0.0000 [-0.0001, +0.0001] | +0.0001 [+0.0000, +0.0003] **worse** | +0.0002 [+0.0001, +0.0004] **worse** | +0.0002 [+0.0001, +0.0004] **worse** | +0.0004 [+0.0002, +0.0006] **worse** |
| LATERAL.LAT_yaw_rate_mae_radps_valid | 0.018255 | +0.0000 [-0.0003, +0.0004] | +0.0009 [+0.0004, +0.0015] **worse** | +0.0009 [+0.0004, +0.0015] **worse** | +0.0009 [+0.0004, +0.0016] **worse** | +0.0018 [+0.0011, +0.0026] **worse** |
| LATERAL.LAT_cross_mae_m | 0.10179 | -0.0007 [-0.0026, +0.0013] | +0.0055 [+0.0029, +0.0082] **worse** | +0.0088 [+0.0047, +0.0130] **worse** | +0.0081 [+0.0039, +0.0125] **worse** | +0.0143 [+0.0098, +0.0190] **worse** |
| LATERAL.LAT_cross_abs_final_m | 0.20724 | -0.0021 [-0.0063, +0.0024] | +0.0120 [+0.0064, +0.0181] **worse** | +0.0198 [+0.0110, +0.0291] **worse** | +0.0177 [+0.0089, +0.0272] **worse** | +0.0317 [+0.0220, +0.0421] **worse** |
| TACTICAL.TAC_traj_lat_correct | 0.96193 | +0.0011 [-0.0017, +0.0038] | -0.0044 [-0.0080, -0.0011] **worse** | -0.0034 [-0.0074, +0.0013] | -0.0023 [-0.0072, +0.0023] | -0.0078 [-0.0131, -0.0023] **worse** |
| TACTICAL.TAC_traj_lon_correct | 0.81637 | +0.0002 [-0.0046, +0.0046] | +0.0013 [-0.0046, +0.0074] | -0.0023 [-0.0090, +0.0046] | -0.0021 [-0.0090, +0.0046] | -0.0011 [-0.0088, +0.0071] |

* STRATEGIC: NOT APPLICABLE (n 0): SHIP-R8 clause (6) --no-strategic; route checkpoint off (--r8-no-rc)
* LONGITUDINAL distance-keeping: NOT COMPUTED (n 0): the battery's per-window components carry no lead-agent headway/TTC instrument on the os windows (pair_read_r8.METRICS); a named gap, not an omission

### S6 -- n_windows 3668, episodes 139 (all four families; arm - ref)

| family.metric | BASE mean | TRUE-OWNMAP | SHUF-OWNMAP | OWNMAP-BASE | TRUE-BASE | SHUF-BASE |
|---|---|---|---|---|---|---|
| ADE.ade_m | 2.7807 | +0.0096 [-0.0213, +0.0418] | +0.0175 [-0.0196, +0.0564] | +0.0266 [-0.0196, +0.0729] | +0.0362 [-0.0158, +0.0934] | +0.0441 [-0.0094, +0.1001] |
| ADE.fde_m | 6.835 | +0.0201 [-0.0611, +0.0974] | +0.0462 [-0.0469, +0.1434] | +0.0455 [-0.0587, +0.1547] | +0.0657 [-0.0596, +0.2011] | +0.0917 [-0.0405, +0.2322] |
| LONGITUDINAL.LON_speed_mae_mps | 1.0323 | +0.0040 [-0.0081, +0.0159] | -0.0032 [-0.0176, +0.0117] | +0.0028 [-0.0136, +0.0189] | +0.0068 [-0.0120, +0.0272] | -0.0004 [-0.0201, +0.0201] |
| LONGITUDINAL.LON_along_mae_m | 2.4308 | +0.0107 [-0.0192, +0.0432] | -0.0077 [-0.0422, +0.0281] | +0.0092 [-0.0358, +0.0527] | +0.0199 [-0.0316, +0.0755] | +0.0015 [-0.0513, +0.0590] |
| LONGITUDINAL.LON_along_abs_final_m | 5.9525 | +0.0149 [-0.0599, +0.0913] | -0.0125 [-0.1019, +0.0758] | +0.0264 [-0.0856, +0.1380] | +0.0413 [-0.0869, +0.1793] | +0.0139 [-0.1197, +0.1616] |
| LONGITUDINAL.LON_speed_within_0p5 | 0.50491 | -0.0018 [-0.0055, +0.0016] | +0.0030 [-0.0022, +0.0081] | -0.0007 [-0.0053, +0.0040] | -0.0025 [-0.0077, +0.0025] | +0.0023 [-0.0039, +0.0080] |
| LONGITUDINAL.LON_accel_mae_mps2 | 0.46869 | -0.0005 [-0.0045, +0.0035] | -0.0018 [-0.0072, +0.0033] | -0.0006 [-0.0058, +0.0045] | -0.0011 [-0.0073, +0.0055] | -0.0024 [-0.0094, +0.0050] |
| LATERAL.LAT_heading_mae_deg | 3.152 | +0.0149 [-0.0490, +0.0821] | +0.1189 [+0.0218, +0.2221] **worse** | +0.1113 [+0.0431, +0.1852] **worse** | +0.1184 [+0.0335, +0.2074] **worse** | +0.2319 [+0.1240, +0.3536] **worse** |
| LATERAL.LAT_curvature_mae_1pm_valid | 0.0051185 | +0.0001 [-0.0000, +0.0002] | +0.0000 [-0.0001, +0.0002] | +0.0002 [+0.0000, +0.0003] **worse** | +0.0002 [+0.0001, +0.0003] **worse** | +0.0002 [+0.0000, +0.0004] **worse** |
| LATERAL.LAT_yaw_rate_mae_radps_valid | 0.027992 | +0.0002 [-0.0002, +0.0007] | +0.0008 [+0.0002, +0.0014] **worse** | +0.0005 [-0.0000, +0.0011] | +0.0007 [+0.0001, +0.0014] **worse** | +0.0013 [+0.0005, +0.0021] **worse** |
| LATERAL.LAT_cross_mae_m | 0.84445 | +0.0013 [-0.0156, +0.0191] | +0.0364 [+0.0097, +0.0651] **worse** | +0.0343 [+0.0104, +0.0572] **worse** | +0.0356 [+0.0085, +0.0638] **worse** | +0.0707 [+0.0381, +0.1042] **worse** |
| LATERAL.LAT_cross_abs_final_m | 2.1133 | +0.0081 [-0.0360, +0.0551] | +0.0837 [+0.0148, +0.1561] **worse** | +0.0516 [-0.0051, +0.1087] | +0.0597 [-0.0075, +0.1294] | +0.1353 [+0.0523, +0.2199] **worse** |
| TACTICAL.TAC_traj_lat_correct | 0.88222 | -0.0008 [-0.0055, +0.0035] | -0.0027 [-0.0096, +0.0038] | +0.0014 [-0.0041, +0.0068] | +0.0005 [-0.0063, +0.0071] | -0.0014 [-0.0104, +0.0071] |
| TACTICAL.TAC_traj_lon_correct | 0.60278 | -0.0025 [-0.0085, +0.0041] | +0.0000 [-0.0087, +0.0085] | -0.0093 [-0.0163, -0.0022] **worse** | -0.0117 [-0.0201, -0.0035] **worse** | -0.0093 [-0.0191, +0.0011] |

* STRATEGIC: NOT APPLICABLE (n 0): SHIP-R8 clause (6) --no-strategic; route checkpoint off (--r8-no-rc)
* LONGITUDINAL distance-keeping: NOT COMPUTED (n 0): the battery's per-window components carry no lead-agent headway/TTC instrument on the os windows (pair_read_r8.METRICS); a named gap, not an omission

## Seed 1

### Controls

```
{
 "C2_base_capture_eq_dump": {
  "n": 4754,
  "max_abs": 0.0,
  "pass": true
 },
 "C1_base_reproduces": {
  "against": "C:\\Users\\Admin\\refcv8_eval_archive\\battery\\step48852ema_pf\\dump_s1",
  "n": 4754,
  "max_abs_os": 0.0,
  "max_abs_plan8": 0.0,
  "tol": 1e-05,
  "binding": true,
  "pass": true,
  "note": null
 },
 "C3_recompute_identity": {
  "checked": 4754,
  "equal": 4754,
  "maxdiff": 0.0,
  "own_calls": 4754,
  "pass": true
 },
 "C3b_no_target_windows_are_base": {
  "n_by_arm": {
   "TRUE": 68,
   "OWNMAP": 68,
   "SHUF": 0
  },
  "pass": true
 },
 "C8_gt_item_eq_store": {
  "checked": 50,
  "equal": 50,
  "pass": true
 },
 "applied_once": {
  "arm_forwards": 14126,
  "applied_not_once": 0,
  "cap_not_once": 0,
  "mh_memo_hit": 14126,
  "mh_memo_miss": 0,
  "pass": true
 },
 "refine_repro": {
  "maxdiff_fp32_decode_vs_model_logits": 0.0,
  "argmax_agree_min": 1.0,
  "note": "CPU fp32 smoke: must be ~0; CUDA run: the model's logits ran under bf16 autocast -- recorded, not a gate"
 },
 "C7_gt_parity": {
  "n": 4754,
  "max_abs_g": 0.0,
  "pass": true
 }
}
```

### Census

```
{
 "forced": {
  "TRUE": 4686,
  "OWNMAP": 4686,
  "SHUF": 4754
 },
 "gt_label_windows": 4686,
 "rel_norm_median": {
  "TRUE": 0.7891412377357483,
  "OWNMAP": 0.7341323792934418,
  "SHUF": 0.9066692888736725
 },
 "loss0_mean": {
  "TRUE": 0.9679275307258988,
  "OWNMAP": 0.5201500031812497,
  "SHUF": 2.057766880766404
 },
 "loss1_mean": {
  "TRUE": 0.07539249449468294,
  "OWNMAP": 0.041105105589146713,
  "SHUF": 0.09968546977627914
 },
 "inv_wall_mean_s": 3.2805482097468457,
 "wall_by_arm_s": {
  "OWN": 2070.5829999999996,
  "TRUE": 1233.863,
  "OWNMAP": 1205.986,
  "SHUF": 1221.1509999999998
 },
 "roll_wall_s": 21745.9
}
```

### Map reach (pooled IoU on the planner crop, target-seen cells)

| pair | mIoU 1-7 | drivable | lane | crosswalk | non-drv-edge |
|---|---|---|---|---|---|
| own0_vs_gt | 0.224 | 0.624 | 0.206 | 0.094 | 0.040 |
| TRUE_vs_gt | 0.848 | 0.981 | 0.844 | 0.884 | 0.652 |
| OWNMAP_vs_gt | 0.224 | 0.623 | 0.205 | 0.094 | 0.041 |
| SHUF_vs_gt | 0.095 | 0.360 | 0.021 | 0.003 | 0.005 |
| SHUF_vs_donor | 0.822 | 0.978 | 0.826 | 0.848 | 0.604 |
| OWNMAP_vs_own0 | 0.938 | 0.994 | 0.965 | 0.955 | 0.923 |
| TRUEpool_vs_gt | 0.752 | 0.962 | 0.703 | 0.758 | 0.414 |
| SHUFpool_vs_donor | 0.729 | 0.961 | 0.688 | 0.723 | 0.387 |
| OWNMAPpool_vs_own0 | 0.880 | 0.980 | 0.893 | 0.917 | 0.793 |

### S2 -- n_windows 4754, episodes 139 (all four families; arm - ref)

| family.metric | BASE mean | TRUE-OWNMAP | SHUF-OWNMAP | OWNMAP-BASE | TRUE-BASE | SHUF-BASE |
|---|---|---|---|---|---|---|
| ADE.ade_m | 0.29854 | +0.0027 [-0.0011, +0.0067] | +0.0045 [+0.0009, +0.0080] **worse** | +0.0082 [+0.0029, +0.0140] **worse** | +0.0109 [+0.0056, +0.0166] **worse** | +0.0127 [+0.0071, +0.0186] **worse** |
| ADE.fde_m | 0.64632 | +0.0034 [-0.0043, +0.0112] | +0.0074 [-0.0005, +0.0153] | +0.0166 [+0.0052, +0.0292] **worse** | +0.0200 [+0.0090, +0.0320] **worse** | +0.0239 [+0.0122, +0.0364] **worse** |
| LONGITUDINAL.LON_speed_mae_mps | 0.28758 | +0.0031 [-0.0007, +0.0071] | +0.0018 [-0.0019, +0.0054] | +0.0012 [-0.0034, +0.0063] | +0.0043 [-0.0000, +0.0087] | +0.0030 [-0.0018, +0.0081] |
| LONGITUDINAL.LON_along_mae_m | 0.25243 | +0.0028 [-0.0011, +0.0067] | +0.0011 [-0.0024, +0.0044] | +0.0020 [-0.0025, +0.0070] | +0.0048 [+0.0006, +0.0091] **worse** | +0.0031 [-0.0015, +0.0080] |
| LONGITUDINAL.LON_along_abs_final_m | 0.5571 | +0.0042 [-0.0038, +0.0121] | -0.0002 [-0.0080, +0.0070] | +0.0027 [-0.0075, +0.0143] | +0.0068 [-0.0023, +0.0163] | +0.0025 [-0.0075, +0.0129] |
| LONGITUDINAL.LON_speed_within_0p5 | 0.82904 | -0.0052 [-0.0081, -0.0021] **worse** | -0.0041 [-0.0077, -0.0004] **worse** | +0.0017 [-0.0028, +0.0060] | -0.0034 [-0.0083, +0.0010] | -0.0024 [-0.0072, +0.0025] |
| LONGITUDINAL.LON_accel_mae_mps2 | 0.40013 | +0.0029 [-0.0010, +0.0071] | +0.0008 [-0.0032, +0.0049] | -0.0013 [-0.0067, +0.0045] | +0.0016 [-0.0037, +0.0071] | -0.0005 [-0.0061, +0.0055] |
| LATERAL.LAT_heading_mae_deg | 1.428 | -0.0039 [-0.0245, +0.0188] | +0.0419 [+0.0158, +0.0715] **worse** | +0.0885 [+0.0554, +0.1246] **worse** | +0.0844 [+0.0439, +0.1284] **worse** | +0.1302 [+0.0864, +0.1812] **worse** |
| LATERAL.LAT_curvature_mae_1pm_valid | 0.0036637 | -0.0000 [-0.0001, +0.0001] | +0.0001 [+0.0000, +0.0003] **worse** | +0.0003 [+0.0001, +0.0004] **worse** | +0.0002 [+0.0001, +0.0004] **worse** | +0.0004 [+0.0002, +0.0006] **worse** |
| LATERAL.LAT_yaw_rate_mae_radps_valid | 0.018078 | -0.0001 [-0.0005, +0.0003] | +0.0009 [+0.0003, +0.0014] **worse** | +0.0012 [+0.0007, +0.0018] **worse** | +0.0011 [+0.0005, +0.0018] **worse** | +0.0021 [+0.0013, +0.0029] **worse** |
| LATERAL.LAT_cross_mae_m | 0.10185 | +0.0004 [-0.0017, +0.0024] | +0.0052 [+0.0025, +0.0079] **worse** | +0.0086 [+0.0049, +0.0124] **worse** | +0.0090 [+0.0049, +0.0133] **worse** | +0.0138 [+0.0094, +0.0186] **worse** |
| LATERAL.LAT_cross_abs_final_m | 0.20698 | -0.0004 [-0.0049, +0.0042] | +0.0110 [+0.0049, +0.0171] **worse** | +0.0201 [+0.0122, +0.0287] **worse** | +0.0197 [+0.0112, +0.0286] **worse** | +0.0311 [+0.0215, +0.0418] **worse** |
| TACTICAL.TAC_traj_lat_correct | 0.96256 | +0.0011 [-0.0017, +0.0038] | -0.0027 [-0.0067, +0.0008] | -0.0046 [-0.0090, -0.0004] **worse** | -0.0036 [-0.0073, +0.0002] | -0.0074 [-0.0124, -0.0025] **worse** |
| TACTICAL.TAC_traj_lon_correct | 0.81615 | -0.0029 [-0.0069, +0.0013] | -0.0002 [-0.0057, +0.0059] | -0.0017 [-0.0088, +0.0057] | -0.0046 [-0.0116, +0.0027] | -0.0019 [-0.0086, +0.0048] |

* STRATEGIC: NOT APPLICABLE (n 0): SHIP-R8 clause (6) --no-strategic; route checkpoint off (--r8-no-rc)
* LONGITUDINAL distance-keeping: NOT COMPUTED (n 0): the battery's per-window components carry no lead-agent headway/TTC instrument on the os windows (pair_read_r8.METRICS); a named gap, not an omission

### S6 -- n_windows 3668, episodes 139 (all four families; arm - ref)

| family.metric | BASE mean | TRUE-OWNMAP | SHUF-OWNMAP | OWNMAP-BASE | TRUE-BASE | SHUF-BASE |
|---|---|---|---|---|---|---|
| ADE.ade_m | 2.7977 | +0.0079 [-0.0247, +0.0438] | +0.0214 [-0.0198, +0.0641] | +0.0217 [-0.0278, +0.0719] | +0.0296 [-0.0144, +0.0766] | +0.0431 [-0.0109, +0.0978] |
| ADE.fde_m | 6.8851 | +0.0081 [-0.0741, +0.0969] | +0.0374 [-0.0682, +0.1452] | +0.0256 [-0.0920, +0.1415] | +0.0337 [-0.0717, +0.1452] | +0.0630 [-0.0706, +0.1942] |
| LONGITUDINAL.LON_speed_mae_mps | 1.0413 | +0.0070 [-0.0050, +0.0204] | +0.0021 [-0.0153, +0.0198] | -0.0078 [-0.0258, +0.0101] | -0.0008 [-0.0160, +0.0145] | -0.0057 [-0.0251, +0.0128] |
| LONGITUDINAL.LON_along_mae_m | 2.4491 | +0.0148 [-0.0162, +0.0493] | -0.0031 [-0.0452, +0.0399] | -0.0004 [-0.0467, +0.0474] | +0.0144 [-0.0242, +0.0546] | -0.0035 [-0.0528, +0.0459] |
| LONGITUDINAL.LON_along_abs_final_m | 5.9998 | +0.0263 [-0.0491, +0.1109] | -0.0158 [-0.1223, +0.0948] | -0.0007 [-0.1138, +0.1123] | +0.0256 [-0.0692, +0.1251] | -0.0165 [-0.1428, +0.1134] |
| LONGITUDINAL.LON_speed_within_0p5 | 0.50418 | -0.0049 [-0.0087, -0.0012] **worse** | -0.0013 [-0.0064, +0.0038] | +0.0004 [-0.0050, +0.0058] | -0.0045 [-0.0102, +0.0013] | -0.0010 [-0.0069, +0.0055] |
| LONGITUDINAL.LON_accel_mae_mps2 | 0.47048 | +0.0010 [-0.0033, +0.0057] | -0.0004 [-0.0070, +0.0060] | -0.0037 [-0.0099, +0.0020] | -0.0027 [-0.0081, +0.0024] | -0.0041 [-0.0111, +0.0025] |
| LATERAL.LAT_heading_mae_deg | 3.1438 | -0.0235 [-0.1003, +0.0531] | +0.1134 [+0.0324, +0.2023] **worse** | +0.1519 [+0.0790, +0.2343] **worse** | +0.1216 [+0.0302, +0.2254] **worse** | +0.2550 [+0.1509, +0.3772] **worse** |
| LATERAL.LAT_curvature_mae_1pm_valid | 0.0051206 | +0.0000 [-0.0001, +0.0001] | +0.0001 [-0.0001, +0.0002] | +0.0002 [+0.0001, +0.0003] **worse** | +0.0002 [+0.0001, +0.0003] **worse** | +0.0002 [+0.0001, +0.0004] **worse** |
| LATERAL.LAT_yaw_rate_mae_radps_valid | 0.027915 | -0.0000 [-0.0005, +0.0005] | +0.0009 [+0.0003, +0.0014] **worse** | +0.0007 [+0.0002, +0.0012] **worse** | +0.0006 [+0.0000, +0.0013] **worse** | +0.0015 [+0.0008, +0.0023] **worse** |
| LATERAL.LAT_cross_mae_m | 0.8444 | -0.0121 [-0.0335, +0.0095] | +0.0324 [+0.0038, +0.0603] **worse** | +0.0415 [+0.0141, +0.0686] **worse** | +0.0294 [+0.0005, +0.0606] **worse** | +0.0739 [+0.0386, +0.1111] **worse** |
| LATERAL.LAT_cross_abs_final_m | 2.1243 | -0.0357 [-0.0946, +0.0230] | +0.0711 [+0.0048, +0.1366] **worse** | +0.0611 [-0.0080, +0.1265] | +0.0254 [-0.0489, +0.1006] | +0.1322 [+0.0534, +0.2149] **worse** |
| TACTICAL.TAC_traj_lat_correct | 0.88304 | +0.0005 [-0.0044, +0.0057] | -0.0003 [-0.0071, +0.0068] | -0.0035 [-0.0098, +0.0025] | -0.0030 [-0.0101, +0.0038] | -0.0038 [-0.0112, +0.0030] |
| TACTICAL.TAC_traj_lon_correct | 0.59678 | -0.0014 [-0.0076, +0.0046] | -0.0019 [-0.0109, +0.0066] | +0.0019 [-0.0071, +0.0117] | +0.0005 [-0.0082, +0.0098] | +0.0000 [-0.0106, +0.0115] |

* STRATEGIC: NOT APPLICABLE (n 0): SHIP-R8 clause (6) --no-strategic; route checkpoint off (--r8-no-rc)
* LONGITUDINAL distance-keeping: NOT COMPUTED (n 0): the battery's per-window components carry no lead-agent headway/TTC instrument on the os windows (pair_read_r8.METRICS); a named gap, not an omission

