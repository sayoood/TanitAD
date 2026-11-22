# K2DC02s1 dose hypothesis: **DOSE-VOID**

between the bars: next step = a replicate arm (same flags, seed + 1), NOT a reinterpretation

*DIAGNOSTIC (one training run per arm, H-ESTIM-SEED-1).* Pre-registration: `TanitAD-artifacts/dagland/e2/PREREG_K2DC_2026-11-12.md`. Tier: T1 (self-action open loop on logged eval139 frames; the `os` arm). Windows: DEV (DEV_HOLD_SPLIT.json); HOLD is never read.

Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES (clips); delta = A - B; it answers 'another draw of EPISODES', not 'another training run' (H-ESTIM-SEED-1) and not 'another inference draw' (each seed is one call).

In-training validity (`dose_validity`): **VOID** (>= 2 probes at step >= 3000 inside [0.01, 0.03]; 1 of 13 late probes inside the band).

## Pre-registered rows (delta = A - B, metres; lower is better)

| seed | K2DC-K0 LAT_cross_mae | separated worse? | K2DC-K2 LAT_cross_mae | <= -2.5 mm | K2DC-K2 LAT_cross_abs_final | K2DC-K2 jitter abs(d e) | jitter < 0 | supported | refuted |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.00390 [+0.00100, +0.00680] | YES | +0.00070 [-0.00250, +0.00390] | no | +0.00160 [-0.00490, +0.00810] | +0.00140 [-0.00860, +0.01050] | no | no | no |
| 1 | +0.00170 [-0.00160, +0.00480] | no | +0.00060 [-0.00200, +0.00320] | no | -0.00030 [-0.00590, +0.00560] | -0.00120 [-0.01010, +0.00700] | yes | no | yes |

## All four families (K2DC - K2 | K2DC - K0), per seed, never pooled

### seed 0

| family | row | K2DC - K2 | sep | K2DC - K0 | sep |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | -0.00580 |  | -0.00180 |  |
| LONGITUDINAL | LON_along_mae_m | -0.00600 |  | -0.00190 |  |
| LONGITUDINAL | LON_along_abs_final_m | -0.01100 |  | -0.00270 |  |
| LONGITUDINAL | LON_speed_within_0p5 | +0.00350 |  | +0.00450 |  |
| LONGITUDINAL | LON_accel_mae_mps2 | -0.00600 |  | +0.00190 |  |
| LATERAL | LAT_heading_mae_deg | -0.01290 |  | +0.03150 | * |
| LATERAL | LAT_curvature_mae_1pm_valid | -0.00030 |  | +0.00020 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | -0.00070 |  | +0.00060 |  |
| LATERAL | LAT_cross_mae_m | +0.00070 |  | +0.00390 | * |
| LATERAL | LAT_cross_abs_final_m | +0.00160 |  | +0.00900 | * |
| TACTICAL | TAC_traj_lat_correct | +0.00130 |  | -0.00250 |  |
| TACTICAL | TAC_traj_lon_correct | +0.00130 |  | +0.00080 |  |
| TACTICAL | TAC_traj_man5_correct | -0.00170 |  | -0.00080 |  |
| STRATEGIC | - | PAUSED | | PAUSED | |

### seed 1

| family | row | K2DC - K2 | sep | K2DC - K0 | sep |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | -0.00120 |  | +0.00050 |  |
| LONGITUDINAL | LON_along_mae_m | -0.00130 |  | +0.00030 |  |
| LONGITUDINAL | LON_along_abs_final_m | -0.00010 |  | +0.00220 |  |
| LONGITUDINAL | LON_speed_within_0p5 | -0.00040 |  | -0.00260 |  |
| LONGITUDINAL | LON_accel_mae_mps2 | -0.00260 |  | +0.00490 |  |
| LATERAL | LAT_heading_mae_deg | +0.00740 |  | +0.01600 |  |
| LATERAL | LAT_curvature_mae_1pm_valid | -0.00010 |  | +0.00010 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | -0.00020 |  | +0.00030 |  |
| LATERAL | LAT_cross_mae_m | +0.00060 |  | +0.00170 |  |
| LATERAL | LAT_cross_abs_final_m | -0.00030 |  | +0.00370 |  |
| TACTICAL | TAC_traj_lat_correct | +0.00040 |  | -0.00170 |  |
| TACTICAL | TAC_traj_lon_correct | +0.00080 |  | -0.00460 |  |
| TACTICAL | TAC_traj_man5_correct | +0.00040 |  | -0.00540 |  |
| STRATEGIC | - | PAUSED | | PAUSED | |

## Beside (no committed direction)

```
{
 "status": "REPORTED",
 "direction": "none committed (PREREG: 'map primary K2DC - K2 reported with no committed direction')",
 "tier": "T0 label level (map, box); T1 (S6c)",
 "map_primary_nf_swedge_prec_lv_5_20": {
  "k2dc": 0.4602483873827187,
  "k2": 0.4081785213017287,
  "delta": 0.05206986608099001
 },
 "box_ap2m_all": {
  "k2dc": 0.3244873582858472,
  "k2": 0.3200722521812886,
  "delta": 0.004415106104558586
 },
 "s6c_sidewalk_rate_s0": {
  "k2dc": 0.19382716049382717,
  "k2": 0.1845679012345679,
  "delta": 0.009259259259259273,
  "better": "lower"
 },
 "s6c_sidewalk_rate_s1": {
  "k2dc": 0.19320987654320987,
  "k2": 0.18641975308641975,
  "delta": 0.006790123456790115,
  "better": "lower"
 }
}
```

## Resolutions of PREREG wording

* 'at both seeds' governs the whole SUPPORTED sentence
* REFUTED 'lateral rows' = LAT_cross_mae_m and LAT_cross_abs_final_m (the lateral rows that are lengths)
* 'separated worse' = the paired CI excludes zero AND delta > 0 (lower is better)
* jitter clause = the POINT delta of the frame-to-frame |d e| (K2DC - K2) < 0; its interval is reported, no CI is required by the PREREG
