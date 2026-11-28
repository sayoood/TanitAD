# OBD seed-3 BAND read: BAND-UNSETTLED

Amendment `AMEND_OBD_SEED3.md` (sha256 `a8a5848f`), card `PREREG_V8F_OBD.md` (sha256 `2d42cd51`), code of record `cdbff971`. Existing read `RESULT_OBD.json` sha256 `e05e1f5f`. Tier T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner). Evidence class MEASURED (ours; dumps, zero training). Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0; ci.py from F:/TanitAD-scratch/otband_read/tree/taniteval/taniteval/ci.py. It answers 'another draw of EPISODES'; 'another TRAINING run' is floor G (OBD-BAND_rep vs OBD-BAND_s0) and the four-seed combination rule.

STRATEGIC: **PAUSED** (PI item 25, `PI_DECISION_QUEUE.md`, 2026-09-19), reported as paused, not as a pass.

## Band outcome
**BAND-UNSETTLED**

Rule (amendment): the label held by at least 3 of the 4 per-seed labels, with at least one of them at each init (M2b s0 and M2b s1), else BAND-UNSETTLED.

| per-seed label | init | label | source |
|---|---|---|---|
| s0 | M2b s0 | BAND-ADDS | existing RESULT_OBD.json |
| s1 | M2b s1 | BAND-COSTS | existing RESULT_OBD.json |
| s10 | M2b s0 | BAND-ADDS | this read |
| s11 | M2b s1 | BAND-NEUTRAL | this read |

**Recipe of record:** OBD-CTL
**Triggered:** OBD-CTL's argv becomes the recipe of record, the conservative branch: four training runs could not show the band loss consistently. The band's operative-only gain stays recorded beside, with the code-drift caveat (VERDICT_OBD.md section 3).

* s10: BAND-ADDS (24 cells read)
    * gain operative_0_2s:ade_m: -0.0020 [-0.0035, -0.0005], floor +0.0013
    * gain operative_0_2s:fde_m: -0.0094 [-0.0140, -0.0050], floor +0.0002
    * gain operative_0_2s:along_mae_m: -0.0014 [-0.0028, -0.0001], floor +0.0014
    * gain operative_0_2s:cross_mae_m: -0.0014 [-0.0024, -0.0004], floor +0.0005
* s11: BAND-NEUTRAL (24 cells read)

## Family PRIMARY: BAND minus CTL, per seed (floor G)

| cell | floor G | s0 | s1 | s10 | s11 |
|---|---|---|---|---|---|
| operative_0_2s:ade_m | +0.0013 | -0.0019 [-0.0030, -0.0009] gain | -0.0014 [-0.0027, -0.0001] gain | -0.0020 [-0.0035, -0.0005] gain | -0.0003 [-0.0014, +0.0008] |
| operative_0_2s:fde_m | +0.0002 | -0.0058 [-0.0092, -0.0026] gain | -0.0051 [-0.0094, -0.0009] gain | -0.0094 [-0.0140, -0.0050] gain | -0.0026 [-0.0063, +0.0011] |
| tactical_2_6s:ade_m | +0.0688 | -0.0314 [-0.0646, +0.0017] | +0.0487 [+0.0108, +0.0870] sep, sub-floor | +0.0195 [-0.0152, +0.0543] | +0.0374 [-0.0023, +0.0778] |
| tactical_2_6s:fde_m | +0.2464 | -0.0871 [-0.1897, +0.0096] | +0.1989 [+0.0944, +0.2985] sep, sub-floor | +0.1114 [+0.0022, +0.2198] sep, sub-floor | +0.1392 [+0.0167, +0.2606] sep, sub-floor |

## Family LONGITUDINAL: BAND minus CTL, per seed (floor G)

| cell | floor G | s0 | s1 | s10 | s11 |
|---|---|---|---|---|---|
| operative_0_2s:speed_mae_mps | +0.0040 | +0.0001 [-0.0012, +0.0014] | +0.0010 [-0.0009, +0.0031] | -0.0006 [-0.0024, +0.0011] | +0.0003 [-0.0010, +0.0016] |
| operative_0_2s:along_mae_m | +0.0014 | -0.0011 [-0.0021, -0.0001] sep, sub-floor | -0.0003 [-0.0015, +0.0009] | -0.0014 [-0.0028, -0.0001] gain | -0.0002 [-0.0011, +0.0007] |
| operative_0_2s:accel_mae_mps2 | +0.0228 | +0.0022 [-0.0004, +0.0051] | +0.0062 [+0.0020, +0.0105] sep, sub-floor | +0.0049 [+0.0013, +0.0086] sep, sub-floor | +0.0031 [+0.0006, +0.0057] sep, sub-floor |
| operative_0_2s:dk_hw_abs_err_m | +0.0104 | -0.0050 [-0.0127, +0.0012] | +0.0185 [-0.0049, +0.0510] | -0.0026 [-0.0180, +0.0079] | -0.0055 [-0.0118, +0.0010] |
| operative_0_2s:dk_tg_abs_err_s | +0.0111 | -0.0017 [-0.0055, +0.0008] | +0.0058 [+0.0001, +0.0143] sep, sub-floor | +0.0003 [-0.0012, +0.0016] | -0.0007 [-0.0025, +0.0011] |
| tactical_2_6s:speed_mae_mps | +0.0972 | -0.0297 [-0.0640, +0.0016] | +0.0537 [+0.0245, +0.0841] sep, sub-floor | +0.0322 [-0.0005, +0.0630] | +0.0561 [+0.0156, +0.0978] sep, sub-floor |
| tactical_2_6s:along_mae_m | +0.0996 | -0.0028 [-0.0344, +0.0268] | +0.0434 [+0.0059, +0.0818] sep, sub-floor | +0.0206 [-0.0177, +0.0599] | +0.0296 [-0.0051, +0.0640] |
| tactical_2_6s:accel_mae_mps2 | +0.2884 | -0.1070 [-0.2026, -0.0209] sep, sub-floor | +0.0797 [-0.0481, +0.2075] | +0.0430 [-0.0616, +0.1531] | +0.2600 [+0.0951, +0.4374] sep, sub-floor |
| tactical_2_6s:dk_hw_abs_err_m | +0.1744 | -0.0123 [-0.1682, +0.1201] | +0.0294 [-0.0545, +0.1051] | +0.0862 [-0.0066, +0.1958] | -0.0660 [-0.1709, +0.0373] |
| tactical_2_6s:dk_tg_abs_err_s | +0.0006 | +0.0213 [-0.0094, +0.0537] | +0.0081 [-0.0093, +0.0265] | -0.0050 [-0.0415, +0.0259] | +0.0215 [-0.0128, +0.0608] |

## Family LATERAL: BAND minus CTL, per seed (floor G)

| cell | floor G | s0 | s1 | s10 | s11 |
|---|---|---|---|---|---|
| operative_0_2s:cross_mae_m | +0.0005 | -0.0013 [-0.0023, -0.0003] gain | -0.0021 [-0.0032, -0.0012] gain | -0.0014 [-0.0024, -0.0004] gain | -0.0003 [-0.0012, +0.0008] |
| operative_0_2s:heading_mae_deg | +0.0086 | +0.0009 [-0.0117, +0.0157] | -0.0155 [-0.0335, +0.0007] | -0.0103 [-0.0274, +0.0077] | -0.0032 [-0.0202, +0.0123] |
| operative_0_2s:yaw_rate_mae_degps | +0.0542 | +0.0212 [-0.0046, +0.0507] | +0.0455 [-0.0075, +0.1180] | +0.0361 [-0.0068, +0.0905] | -0.0312 [-0.1257, +0.0367] |
| operative_0_2s:curvature_mae_1pm | +0.0020 | +0.0004 [-0.0003, +0.0010] | +0.0012 [-0.0004, +0.0031] | +0.0013 [-0.0003, +0.0032] | -0.0004 [-0.0019, +0.0012] |
| tactical_2_6s:cross_mae_m | +0.0255 | -0.0398 [-0.0706, -0.0089] gain | +0.0172 [-0.0050, +0.0400] | -0.0061 [-0.0280, +0.0160] | +0.0215 [-0.0100, +0.0529] |
| tactical_2_6s:heading_mae_deg | +1.0255 | -0.0467 [-0.3417, +0.2804] | +1.1491 [+0.5761, +1.8852] COST | +0.1943 [-0.2319, +0.5656] | +0.0996 [-0.2131, +0.4044] |
| tactical_2_6s:yaw_rate_mae_degps | +5.8542 | +0.7761 [-0.3280, +1.9788] | +3.0470 [+0.8834, +5.4723] sep, sub-floor | +1.1554 [-0.7678, +2.9002] | +0.8940 [-1.0219, +2.4122] |
| tactical_2_6s:curvature_mae_1pm | +0.0441 | +0.0157 [+0.0030, +0.0303] sep, sub-floor | +0.0221 [+0.0021, +0.0429] sep, sub-floor | +0.0212 [-0.0009, +0.0423] | -0.0249 [-0.0637, +0.0078] |

## Family TACTICAL: BAND minus CTL, per seed (floor G)

| cell | floor G | s0 | s1 | s10 | s11 |
|---|---|---|---|---|---|
| TAC-LAT | +0.0104 | -0.0075 [-0.0200, +0.0046] | -0.0171 [-0.0329, -0.0021] COST | -0.0033 [-0.0150, +0.0100] | +0.0013 [-0.0096, +0.0121] |
| TAC-LON | +0.0254 | -0.0067 [-0.0221, +0.0088] | -0.0175 [-0.0346, -0.0025] sep, sub-floor | -0.0146 [-0.0338, +0.0037] | +0.0063 [-0.0138, +0.0234] |

## STRATEGIC

PAUSED (PI item 25, 2026-09-19): reported as paused, never as a pass.

## Beside (no bar)

### The four labels

s0: BAND-ADDS, s1: BAND-COSTS, s10: BAND-ADDS, s11: BAND-NEUTRAL

### Per-cell spread of BAND minus CTL across the four seeds

| cell | s0 | s1 | s10 | s11 | range |
|---|---|---|---|---|---|
| operative_0_2s:ade_m | -0.0019 | -0.0014 | -0.0020 | -0.0003 | +0.0017 |
| operative_0_2s:fde_m | -0.0058 | -0.0051 | -0.0094 | -0.0026 | +0.0068 |
| operative_0_2s:speed_mae_mps | +0.0001 | +0.0010 | -0.0006 | +0.0003 | +0.0016 |
| operative_0_2s:along_mae_m | -0.0011 | -0.0003 | -0.0014 | -0.0002 | +0.0012 |
| operative_0_2s:accel_mae_mps2 | +0.0022 | +0.0062 | +0.0049 | +0.0031 | +0.0039 |
| operative_0_2s:cross_mae_m | -0.0013 | -0.0021 | -0.0014 | -0.0003 | +0.0019 |
| operative_0_2s:heading_mae_deg | +0.0009 | -0.0155 | -0.0103 | -0.0032 | +0.0163 |
| operative_0_2s:yaw_rate_mae_degps | +0.0212 | +0.0455 | +0.0361 | -0.0312 | +0.0767 |
| operative_0_2s:curvature_mae_1pm | +0.0004 | +0.0012 | +0.0013 | -0.0004 | +0.0017 |
| operative_0_2s:dk_hw_abs_err_m | -0.0050 | +0.0185 | -0.0026 | -0.0055 | +0.0240 |
| operative_0_2s:dk_tg_abs_err_s | -0.0017 | +0.0058 | +0.0003 | -0.0007 | +0.0074 |
| tactical_2_6s:ade_m | -0.0314 | +0.0487 | +0.0195 | +0.0374 | +0.0801 |
| tactical_2_6s:fde_m | -0.0871 | +0.1989 | +0.1114 | +0.1392 | +0.2861 |
| tactical_2_6s:speed_mae_mps | -0.0297 | +0.0537 | +0.0322 | +0.0561 | +0.0859 |
| tactical_2_6s:along_mae_m | -0.0028 | +0.0434 | +0.0206 | +0.0296 | +0.0462 |
| tactical_2_6s:accel_mae_mps2 | -0.1070 | +0.0797 | +0.0430 | +0.2600 | +0.3670 |
| tactical_2_6s:cross_mae_m | -0.0398 | +0.0172 | -0.0061 | +0.0215 | +0.0614 |
| tactical_2_6s:heading_mae_deg | -0.0467 | +1.1491 | +0.1943 | +0.0996 | +1.1958 |
| tactical_2_6s:yaw_rate_mae_degps | +0.7761 | +3.0470 | +1.1554 | +0.8940 | +2.2708 |
| tactical_2_6s:curvature_mae_1pm | +0.0157 | +0.0221 | +0.0212 | -0.0249 | +0.0470 |
| tactical_2_6s:dk_hw_abs_err_m | -0.0123 | +0.0294 | +0.0862 | -0.0660 | +0.1522 |
| tactical_2_6s:dk_tg_abs_err_s | +0.0213 | +0.0081 | -0.0050 | +0.0215 | +0.0265 |
| TAC-LAT | -0.0075 | -0.0171 | -0.0033 | +0.0013 | +0.0183 |
| TAC-LON | -0.0067 | -0.0175 | -0.0146 | +0.0063 | +0.0238 |

### Code-drift row (NULL minus DAG-OACC, from the existing read), for scale

* s0: 7 of 24 cells separated; largest |delta| per family: PRIMARY tactical_2_6s:fde_m +0.0658; LONGITUDINAL tactical_2_6s:accel_mae_mps2 +0.0840; LATERAL tactical_2_6s:yaw_rate_mae_degps +0.4352; TACTICAL TAC-LAT +0.0104
* s1: 7 of 24 cells separated; largest |delta| per family: PRIMARY tactical_2_6s:fde_m +0.2209; LONGITUDINAL tactical_2_6s:accel_mae_mps2 +0.2418; LATERAL tactical_2_6s:heading_mae_deg +0.3353; TACTICAL TAC-LON -0.0108

### Operative-only gain (BAND-UNSETTLED branch)

the band's operative-only gain stays recorded beside, with the code-drift caveat (VERDICT_OBD.md section 3); operative ADE delta per seed: s0 -0.0018891560845865352, s1 -0.0014004960604120653, s10 -0.0019918405415473295, s11 -0.0002700412993106305
