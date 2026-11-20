# K2DC02 dose hypothesis: **INCONCLUSIVE**

between the bars: next step = a replicate arm (same flags, seed + 1), NOT a reinterpretation

*DIAGNOSTIC (one training run per arm, H-ESTIM-SEED-1).* Pre-registration: `TanitAD-artifacts/dagland/e2/PREREG_K2DC_2026-11-12.md`. Tier: T1 (self-action open loop on logged eval139 frames; the `os` arm). Windows: DEV (DEV_HOLD_SPLIT.json); HOLD is never read.

Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES (clips); delta = A - B; it answers 'another draw of EPISODES', not 'another training run' (H-ESTIM-SEED-1) and not 'another inference draw' (each seed is one call).

In-training validity (`dose_validity`): **HELD** (>= 2 probes at step >= 3000 inside [0.01, 0.03]; 3 of 13 late probes inside the band).

## Pre-registered rows (delta = A - B, metres; lower is better)

| seed | K2DC-K0 LAT_cross_mae | separated worse? | K2DC-K2 LAT_cross_mae | <= -2.5 mm | K2DC-K2 LAT_cross_abs_final | K2DC-K2 jitter abs(d e) | jitter < 0 | supported | refuted |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.00750 [+0.00240, +0.01260] | YES | -0.00510 [-0.00950, -0.00110] | yes | -0.01150 [-0.02160, -0.00250] | +0.00440 [-0.00560, +0.01370] | no | no | no |
| 1 | +0.00830 [+0.00240, +0.01470] | YES | -0.00080 [-0.00510, +0.00340] | no | -0.00050 [-0.01030, +0.00910] | +0.00460 [-0.00390, +0.01320] | no | no | yes |

## All four families (K2DC - K2 | K2DC - K0), per seed, never pooled

### seed 0

| family | row | K2DC - K2 | sep | K2DC - K0 | sep |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | +0.00610 |  | +0.00720 |  |
| LONGITUDINAL | LON_along_mae_m | +0.00520 |  | +0.00870 |  |
| LONGITUDINAL | LON_along_abs_final_m | +0.01270 |  | +0.01490 |  |
| LONGITUDINAL | LON_speed_within_0p5 | -0.00170 |  | -0.00060 |  |
| LONGITUDINAL | LON_accel_mae_mps2 | +0.00550 |  | +0.00790 |  |
| LATERAL | LAT_heading_mae_deg | -0.05050 | * | +0.05890 |  |
| LATERAL | LAT_curvature_mae_1pm_valid | -0.00030 | * | +0.00010 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | -0.00100 | * | +0.00120 |  |
| LATERAL | LAT_cross_mae_m | -0.00510 | * | +0.00750 | * |
| LATERAL | LAT_cross_abs_final_m | -0.01150 | * | +0.01730 | * |
| TACTICAL | TAC_traj_lat_correct | +0.00170 |  | -0.00540 |  |
| TACTICAL | TAC_traj_lon_correct | -0.00420 |  | +0.00750 |  |
| TACTICAL | TAC_traj_man5_correct | -0.00170 |  | +0.00250 |  |
| STRATEGIC | - | PAUSED | | PAUSED | |

### seed 1

| family | row | K2DC - K2 | sep | K2DC - K0 | sep |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | +0.00910 | * | +0.00900 |  |
| LONGITUDINAL | LON_along_mae_m | +0.00788 | * | +0.00880 |  |
| LONGITUDINAL | LON_along_abs_final_m | +0.01850 | * | +0.01590 |  |
| LONGITUDINAL | LON_speed_within_0p5 | -0.00830 | * | -0.00970 |  |
| LONGITUDINAL | LON_accel_mae_mps2 | +0.00850 |  | +0.01400 | * |
| LATERAL | LAT_heading_mae_deg | -0.00470 |  | +0.07590 | * |
| LATERAL | LAT_curvature_mae_1pm_valid | -0.00010 |  | +0.00030 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | -0.00020 |  | +0.00150 | * |
| LATERAL | LAT_cross_mae_m | -0.00080 |  | +0.00830 | * |
| LATERAL | LAT_cross_abs_final_m | -0.00050 |  | +0.01980 | * |
| TACTICAL | TAC_traj_lat_correct | -0.00170 |  | -0.00880 | * |
| TACTICAL | TAC_traj_lon_correct | -0.00630 |  | -0.00250 |  |
| TACTICAL | TAC_traj_man5_correct | -0.00710 |  | -0.00500 |  |
| STRATEGIC | - | PAUSED | | PAUSED | |

## Beside (no committed direction)

```
{
 "status": "REPORTED",
 "direction": "none committed (PREREG: 'map primary K2DC - K2 reported with no committed direction')",
 "tier": "T0 label level (map, box); T1 (S6c)",
 "map_primary_nf_swedge_prec_lv_5_20": {
  "k2dc": 0.47415488198038874,
  "k2": 0.4849510123294613,
  "delta": -0.010796130349072564
 },
 "box_ap2m_all": {
  "k2dc": 0.31849390732258637,
  "k2": 0.30762299437913837,
  "delta": 0.010870912943447997
 },
 "s6c_sidewalk_rate_s0": {
  "k2dc": 0.18518518518518517,
  "k2": 0.1765432098765432,
  "delta": 0.00864197530864197,
  "better": "lower"
 },
 "s6c_sidewalk_rate_s1": {
  "k2dc": 0.18209876543209877,
  "k2": 0.18518518518518517,
  "delta": -0.0030864197530864057,
  "better": "lower"
 }
}
```

## Resolutions of PREREG wording

* 'at both seeds' governs the whole SUPPORTED sentence
* REFUTED 'lateral rows' = LAT_cross_mae_m and LAT_cross_abs_final_m (the lateral rows that are lengths)
* 'separated worse' = the paired CI excludes zero AND delta > 0 (lower is better)
* jitter clause = the POINT delta of the frame-to-frame |d e| (K2DC - K2) < 0; its interval is reported, no CI is required by the PREREG
