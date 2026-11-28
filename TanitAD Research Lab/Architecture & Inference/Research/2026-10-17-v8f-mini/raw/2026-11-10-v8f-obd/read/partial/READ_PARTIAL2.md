# OBD read: DETACH: NOT-YET: per-seed labels so far {0: 'DETACH-KEEPS'} decide nothing (the card needs both seeds) | BAND: SEEDS-DISAGREE (s0 BAND-ADDS; s1 BAND-COSTS)

Card `PREREG_V8F_OBD.md` (frozen, sha256 `2d42cd51`), code of record `cdbff971`. Tier T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner). Evidence class MEASURED (ours; dumps, zero training). Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0; ci.py from F:/TanitAD-scratch/otband_read/tree/taniteval/taniteval/ci.py. It answers 'another draw of EPISODES'; 'another TRAINING run' is the floors F (OBD-NULL_rep vs OBD-NULL_s0) and G (OBD-BAND_rep vs OBD-BAND_s0).

NULL parity state: **fails**. STRATEGIC: **PAUSED** (PI item 25 (PI_DECISION_QUEUE.md ITEM 25, 2026-09-19): reported as PAUSED, never as a pass, never omitted.)

## DETACH question: OBD-CTL against its reference, floor F
**Outcome: NOT-YET: per-seed labels so far {0: 'DETACH-KEEPS'} decide nothing (the card needs both seeds)**

* seed 0: DETACH-KEEPS (reference OBD-NULL_s0 (NULL parity failed), 24 cells read)
* seed 1: None (reference None, 0 cells read)

| family | cell | s0 delta [CI] | s0 floor | s0 | s1 delta [CI] | s1 | 
|---|---|---|---|---|---|---|
| PRIMARY | operative_0_2s:ade_m | -0.0010 [-0.0045, +0.0018] | +0.0020 |  | n/a |
| PRIMARY | operative_0_2s:fde_m | +0.0041 [-0.0055, +0.0126] | +0.0020 |  | n/a |
| LONGITUDINAL | operative_0_2s:speed_mae_mps | -0.0035 [-0.0082, +0.0008] | +0.0050 |  | n/a |
| LONGITUDINAL | operative_0_2s:along_mae_m | -0.0011 [-0.0046, +0.0020] | +0.0022 |  | n/a |
| LONGITUDINAL | operative_0_2s:accel_mae_mps2 | -0.0171 [-0.0245, -0.0101] | +0.0261 | sep, sub-floor | n/a |
| LATERAL | operative_0_2s:cross_mae_m | -0.0004 [-0.0015, +0.0006] | +0.0001 |  | n/a |
| LATERAL | operative_0_2s:heading_mae_deg | -0.0043 [-0.0199, +0.0106] | +0.0118 |  | n/a |
| LATERAL | operative_0_2s:yaw_rate_mae_degps | +0.0538 [-0.0246, +0.1499] | +0.1334 |  | n/a |
| LATERAL | operative_0_2s:curvature_mae_1pm | +0.0018 [-0.0002, +0.0044] | +0.0039 |  | n/a |
| LONGITUDINAL | operative_0_2s:dk_hw_abs_err_m | +0.0121 [-0.0045, +0.0340] | +0.0105 |  | n/a |
| LONGITUDINAL | operative_0_2s:dk_tg_abs_err_s | +0.0008 [-0.0033, +0.0053] | +0.0129 |  | n/a |
| PRIMARY | tactical_2_6s:ade_m | +0.0303 [-0.0393, +0.0984] | +0.0937 |  | n/a |
| PRIMARY | tactical_2_6s:fde_m | +0.0086 [-0.1925, +0.2046] | +0.2286 |  | n/a |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps | +0.0036 [-0.0547, +0.0566] | +0.0711 |  | n/a |
| LONGITUDINAL | tactical_2_6s:along_mae_m | +0.0072 [-0.0641, +0.0756] | +0.1185 |  | n/a |
| LONGITUDINAL | tactical_2_6s:accel_mae_mps2 | +0.0360 [-0.1119, +0.1615] | +0.2725 |  | n/a |
| LATERAL | tactical_2_6s:cross_mae_m | +0.0280 [-0.0087, +0.0642] | +0.0101 |  | n/a |
| LATERAL | tactical_2_6s:heading_mae_deg | -0.0968 [-0.5670, +0.3191] | +1.3916 |  | n/a |
| LATERAL | tactical_2_6s:yaw_rate_mae_degps | -1.1162 [-3.3823, +1.0450] | +7.9718 |  | n/a |
| LATERAL | tactical_2_6s:curvature_mae_1pm | -0.0192 [-0.0387, -0.0021] | +0.0598 | sep, sub-floor | n/a |
| LONGITUDINAL | tactical_2_6s:dk_hw_abs_err_m | +0.0104 [-0.1002, +0.1301] | +0.1131 |  | n/a |
| LONGITUDINAL | tactical_2_6s:dk_tg_abs_err_s | +0.0142 [-0.0300, +0.0705] | +0.0169 |  | n/a |
| TACTICAL | TAC-LAT | +0.0017 [-0.0121, +0.0163] | +0.0179 |  | n/a |
| TACTICAL | TAC-LON | +0.0108 [-0.0154, +0.0396] | +0.0242 |  | n/a |

## BAND question: OBD-BAND against OBD-CTL, floor G
**Outcome: SEEDS-DISAGREE (s0 BAND-ADDS; s1 BAND-COSTS)**

* seed 0: BAND-ADDS (reference OBD-CTL_s0, 24 cells read)
    * gain operative_0_2s:ade_m: -0.0019 [-0.0030, -0.0009], floor +0.0013
    * gain operative_0_2s:fde_m: -0.0058 [-0.0092, -0.0026], floor +0.0002
    * gain operative_0_2s:cross_mae_m: -0.0013 [-0.0023, -0.0003], floor +0.0005
    * gain tactical_2_6s:cross_mae_m: -0.0398 [-0.0706, -0.0089], floor +0.0255
* seed 1: BAND-COSTS (reference OBD-CTL_s1, 24 cells read)
    * COST tactical_2_6s:heading_mae_deg: +1.1491 [+0.5761, +1.8852], floor +1.0255
    * COST TAC-LAT: -0.0171 [-0.0329, -0.0021], floor +0.0104
    * gain operative_0_2s:ade_m: -0.0014 [-0.0027, -0.0001], floor +0.0013
    * gain operative_0_2s:fde_m: -0.0051 [-0.0094, -0.0009], floor +0.0002
    * gain operative_0_2s:cross_mae_m: -0.0021 [-0.0032, -0.0012], floor +0.0005

| family | cell | s0 delta [CI] | s0 floor | s0 | s1 delta [CI] | s1 | 
|---|---|---|---|---|---|---|
| PRIMARY | operative_0_2s:ade_m | -0.0019 [-0.0030, -0.0009] | +0.0013 | gain | -0.0014 [-0.0027, -0.0001] | gain |
| PRIMARY | operative_0_2s:fde_m | -0.0058 [-0.0092, -0.0026] | +0.0002 | gain | -0.0051 [-0.0094, -0.0009] | gain |
| LONGITUDINAL | operative_0_2s:speed_mae_mps | +0.0001 [-0.0012, +0.0014] | +0.0040 |  | +0.0010 [-0.0009, +0.0031] |  |
| LONGITUDINAL | operative_0_2s:along_mae_m | -0.0011 [-0.0021, -0.0001] | +0.0014 | sep, sub-floor | -0.0003 [-0.0015, +0.0009] |  |
| LONGITUDINAL | operative_0_2s:accel_mae_mps2 | +0.0022 [-0.0004, +0.0051] | +0.0228 |  | +0.0062 [+0.0020, +0.0105] | sep, sub-floor |
| LATERAL | operative_0_2s:cross_mae_m | -0.0013 [-0.0023, -0.0003] | +0.0005 | gain | -0.0021 [-0.0032, -0.0012] | gain |
| LATERAL | operative_0_2s:heading_mae_deg | +0.0009 [-0.0117, +0.0157] | +0.0086 |  | -0.0155 [-0.0335, +0.0007] |  |
| LATERAL | operative_0_2s:yaw_rate_mae_degps | +0.0212 [-0.0046, +0.0507] | +0.0542 |  | +0.0455 [-0.0075, +0.1180] |  |
| LATERAL | operative_0_2s:curvature_mae_1pm | +0.0004 [-0.0003, +0.0010] | +0.0020 |  | +0.0012 [-0.0004, +0.0031] |  |
| LONGITUDINAL | operative_0_2s:dk_hw_abs_err_m | -0.0050 [-0.0127, +0.0012] | +0.0104 |  | +0.0185 [-0.0049, +0.0510] |  |
| LONGITUDINAL | operative_0_2s:dk_tg_abs_err_s | -0.0017 [-0.0055, +0.0008] | +0.0111 |  | +0.0058 [+0.0001, +0.0143] | sep, sub-floor |
| PRIMARY | tactical_2_6s:ade_m | -0.0314 [-0.0646, +0.0017] | +0.0688 |  | +0.0487 [+0.0108, +0.0870] | sep, sub-floor |
| PRIMARY | tactical_2_6s:fde_m | -0.0871 [-0.1897, +0.0096] | +0.2464 |  | +0.1989 [+0.0944, +0.2985] | sep, sub-floor |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps | -0.0297 [-0.0640, +0.0016] | +0.0972 |  | +0.0537 [+0.0245, +0.0841] | sep, sub-floor |
| LONGITUDINAL | tactical_2_6s:along_mae_m | -0.0028 [-0.0344, +0.0268] | +0.0996 |  | +0.0434 [+0.0059, +0.0818] | sep, sub-floor |
| LONGITUDINAL | tactical_2_6s:accel_mae_mps2 | -0.1070 [-0.2026, -0.0209] | +0.2884 | sep, sub-floor | +0.0797 [-0.0481, +0.2075] |  |
| LATERAL | tactical_2_6s:cross_mae_m | -0.0398 [-0.0706, -0.0089] | +0.0255 | gain | +0.0172 [-0.0050, +0.0400] |  |
| LATERAL | tactical_2_6s:heading_mae_deg | -0.0467 [-0.3417, +0.2804] | +1.0255 |  | +1.1491 [+0.5761, +1.8852] | COST |
| LATERAL | tactical_2_6s:yaw_rate_mae_degps | +0.7761 [-0.3280, +1.9788] | +5.8542 |  | +3.0470 [+0.8834, +5.4723] | sep, sub-floor |
| LATERAL | tactical_2_6s:curvature_mae_1pm | +0.0157 [+0.0030, +0.0303] | +0.0441 | sep, sub-floor | +0.0221 [+0.0021, +0.0429] | sep, sub-floor |
| LONGITUDINAL | tactical_2_6s:dk_hw_abs_err_m | -0.0123 [-0.1682, +0.1201] | +0.1744 |  | +0.0294 [-0.0545, +0.1051] |  |
| LONGITUDINAL | tactical_2_6s:dk_tg_abs_err_s | +0.0213 [-0.0094, +0.0537] | +0.0006 |  | +0.0081 [-0.0093, +0.0265] |  |
| TACTICAL | TAC-LAT | -0.0075 [-0.0200, +0.0046] | +0.0104 |  | -0.0171 [-0.0329, -0.0021] | COST |
| TACTICAL | TAC-LON | -0.0067 [-0.0221, +0.0088] | +0.0254 |  | -0.0175 [-0.0346, -0.0025] | sep, sub-floor |

## What the card commits to do next
* NULL parity FAILED: the code of record changed the training path: OBD-NULL_s1 is trained, every row 'against DAG-OACC' is read against OBD-NULL of the same seed instead, and NULL minus DAG-OACC is reported as a code-drift row; the rules are unchanged.
* BAND SEEDS-DISAGREE: reported per seed; the read is repeated with a third training seed per init (--seed 10 / 11) for the arms concerned before any outcome is quoted.
