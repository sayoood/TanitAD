# K2DC02s1 dose hypothesis: **DOSE-VOID**

between the bars: next step = a replicate arm (same flags, seed + 1), NOT a reinterpretation

*DIAGNOSTIC (one training run per arm, H-ESTIM-SEED-1).* Pre-registration: `TanitAD-artifacts/dagland/e2/PREREG_K2DC_2026-11-12.md`. Tier: T1 (self-action open loop on logged eval139 frames; the `os` arm). Windows: DEV (DEV_HOLD_SPLIT.json); HOLD is never read.

Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES (clips); delta = A - B; it answers 'another draw of EPISODES', not 'another training run' (H-ESTIM-SEED-1) and not 'another inference draw' (each seed is one call).

In-training validity (`dose_validity`): **VOID** (>= 2 probes at step >= 3000 inside [0.01, 0.03]; 1 of 13 late probes inside the band).

## Pre-registered rows (delta = A - B, metres; lower is better)

| seed | K2DC-K0 LAT_cross_mae | separated worse? | K2DC-K2 LAT_cross_mae | <= -2.5 mm | K2DC-K2 LAT_cross_abs_final | K2DC-K2 jitter abs(d e) | jitter < 0 | supported | refuted |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.00740 [+0.00270, +0.01220] | YES | -0.00520 [-0.01280, +0.00100] | yes | -0.01230 [-0.03000, +0.00190] | +0.00470 [-0.00570, +0.01520] | no | no | no |
| 1 | +0.00590 [+0.00110, +0.01130] | YES | -0.00320 [-0.01050, +0.00290] | yes | -0.00740 [-0.02440, +0.00650] | +0.00100 [-0.01090, +0.01260] | no | no | no |

## All four families (K2DC - K2 | K2DC - K0), per seed, never pooled

### seed 0

| family | row | K2DC - K2 | sep | K2DC - K0 | sep |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | -0.00290 |  | -0.00180 |  |
| LONGITUDINAL | LON_along_mae_m | -0.00510 |  | -0.00150 |  |
| LONGITUDINAL | LON_along_abs_final_m | -0.00610 |  | -0.00400 |  |
| LONGITUDINAL | LON_speed_within_0p5 | +0.00420 |  | +0.00520 |  |
| LONGITUDINAL | LON_accel_mae_mps2 | +0.00100 |  | +0.00340 |  |
| LATERAL | LAT_heading_mae_deg | -0.06100 |  | +0.04850 |  |
| LATERAL | LAT_curvature_mae_1pm_valid | -0.00020 |  | +0.00020 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | -0.00110 |  | +0.00110 |  |
| LATERAL | LAT_cross_mae_m | -0.00520 |  | +0.00740 | * |
| LATERAL | LAT_cross_abs_final_m | -0.01230 |  | +0.01650 | * |
| TACTICAL | TAC_traj_lat_correct | +0.00170 |  | -0.00540 |  |
| TACTICAL | TAC_traj_lon_correct | -0.00710 |  | +0.00460 |  |
| TACTICAL | TAC_traj_man5_correct | -0.00330 |  | +0.00080 |  |
| STRATEGIC | - | PAUSED | | PAUSED | |

### seed 1

| family | row | K2DC - K2 | sep | K2DC - K0 | sep |
|---|---|---|---|---|---|
| LONGITUDINAL | LON_speed_mae_mps | -0.00390 |  | -0.00400 |  |
| LONGITUDINAL | LON_along_mae_m | -0.00510 |  | -0.00420 |  |
| LONGITUDINAL | LON_along_abs_final_m | -0.00560 |  | -0.00820 |  |
| LONGITUDINAL | LON_speed_within_0p5 | +0.00550 |  | +0.00420 |  |
| LONGITUDINAL | LON_accel_mae_mps2 | -0.00290 |  | +0.00250 |  |
| LATERAL | LAT_heading_mae_deg | -0.03200 |  | +0.04890 |  |
| LATERAL | LAT_curvature_mae_1pm_valid | -0.00020 |  | +0.00030 |  |
| LATERAL | LAT_yaw_rate_mae_radps_valid | -0.00070 |  | +0.00100 |  |
| LATERAL | LAT_cross_mae_m | -0.00320 |  | +0.00590 | * |
| LATERAL | LAT_cross_abs_final_m | -0.00740 |  | +0.01290 | * |
| TACTICAL | TAC_traj_lat_correct | +0.00170 |  | -0.00540 |  |
| TACTICAL | TAC_traj_lon_correct | +0.00130 |  | +0.00500 |  |
| TACTICAL | TAC_traj_man5_correct | +0.00000 |  | +0.00210 |  |
| STRATEGIC | - | PAUSED | | PAUSED | |

## Beside (no committed direction)

```
{
 "status": "REPORTED",
 "direction": "none committed (PREREG: 'map primary K2DC - K2 reported with no committed direction')",
 "tier": "T0 label level (map, box); T1 (S6c)",
 "map_primary_nf_swedge_prec_lv_5_20": {
  "k2dc": 0.4877989770561121,
  "k2": 0.4849510123294613,
  "delta": 0.0028479647266507757
 },
 "box_ap2m_all": {
  "k2dc": 0.31680842829901423,
  "k2": 0.30762299437913837,
  "delta": 0.009185433919875863
 },
 "s6c_sidewalk_rate_s0": {
  "k2dc": 0.18271604938271604,
  "k2": 0.1765432098765432,
  "delta": 0.006172839506172839,
  "better": "lower"
 },
 "s6c_sidewalk_rate_s1": {
  "k2dc": 0.17716049382716048,
  "k2": 0.18518518518518517,
  "delta": -0.008024691358024694,
  "better": "lower"
 }
}
```

## Resolutions of PREREG wording

* 'at both seeds' governs the whole SUPPORTED sentence
* REFUTED 'lateral rows' = LAT_cross_mae_m and LAT_cross_abs_final_m (the lateral rows that are lengths)
* 'separated worse' = the paired CI excludes zero AND delta > 0 (lower is better)
* jitter clause = the POINT delta of the frame-to-frame |d e| (K2DC - K2) < 0; its interval is reported, no CI is required by the PREREG
