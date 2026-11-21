# OBD read: SMOKE (code check, never a result): DETACH DETACH-COSTS; BAND BAND-COSTS

Card `PREREG_V8F_OBD.md` (frozen, sha256 `2d42cd51`), code of record `cdbff971`. Tier T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner). Evidence class MEASURED (ours; dumps, zero training). Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0; ci.py from F:/TanitAD-scratch/otband_read/tree/taniteval/taniteval/ci.py. It answers 'another draw of EPISODES'; 'another TRAINING run' is the floors F (OBD-NULL_rep vs OBD-NULL_s0) and G (OBD-BAND_rep vs OBD-BAND_s0).

NULL parity state: **holds**. STRATEGIC: **PAUSED** (PI item 25 (PI_DECISION_QUEUE.md ITEM 25, 2026-09-19): reported as PAUSED, never as a pass, never omitted.)

## DETACH question: OBD-CTL against its reference, floor F
**Outcome: DETACH-COSTS**

* seed 0: DETACH-COSTS (reference DAG-OACC, 24 cells read)
    * COST operative_0_2s:ade_m: +0.0384 [+0.0198, +0.0711], floor +0.0031
    * COST operative_0_2s:fde_m: +0.0545 [+0.0089, +0.1390], floor +0.0250
    * COST operative_0_2s:speed_mae_mps: +0.1370 [+0.1058, +0.1877], floor +0.0296
    * COST operative_0_2s:along_mae_m: +0.0437 [+0.0249, +0.0763], floor +0.0021
    * COST operative_0_2s:accel_mae_mps2: +0.8312 [+0.6935, +1.0309], floor +0.1080
    * COST tactical_2_6s:ade_m: +0.7341 [+0.5047, +1.0293], floor +0.6021
    * COST tactical_2_6s:fde_m: +2.2062 [+1.5965, +2.9477], floor +1.6794
    * COST tactical_2_6s:speed_mae_mps: +0.9135 [+0.7089, +1.1421], floor +0.4911
    * COST tactical_2_6s:accel_mae_mps2: +1.9840 [+1.6150, +2.3841], floor +0.8005
    * COST tactical_2_6s:cross_mae_m: +0.2977 [+0.1862, +0.4228], floor +0.0259
    * COST TAC-LAT: -0.0729 [-0.1026, -0.0450], floor +0.0417
    * COST TAC-LON: -0.0992 [-0.1389, -0.0604], floor +0.0950
    * gain operative_0_2s:cross_mae_m: -0.0038 [-0.0067, -0.0013], floor +0.0019
* seed 1: DETACH-COSTS (reference DAG-OACC, 24 cells read)
    * COST operative_0_2s:ade_m: +0.0325 [+0.0147, +0.0627], floor +0.0031
    * COST operative_0_2s:speed_mae_mps: +0.1208 [+0.0907, +0.1694], floor +0.0296
    * COST operative_0_2s:along_mae_m: +0.0369 [+0.0189, +0.0669], floor +0.0021
    * COST operative_0_2s:accel_mae_mps2: +0.7685 [+0.6413, +0.9488], floor +0.1080
    * COST operative_0_2s:dk_hw_abs_err_m: +0.0625 [+0.0167, +0.1074], floor +0.0241
    * COST tactical_2_6s:ade_m: +0.6852 [+0.4716, +0.9651], floor +0.6021
    * COST tactical_2_6s:fde_m: +2.2660 [+1.5952, +3.0740], floor +1.6794
    * COST tactical_2_6s:speed_mae_mps: +0.7465 [+0.5449, +0.9816], floor +0.4911
    * COST tactical_2_6s:accel_mae_mps2: +1.5602 [+1.1889, +1.9681], floor +0.8005
    * COST tactical_2_6s:cross_mae_m: +0.5096 [+0.3678, +0.6607], floor +0.0259
    * COST tactical_2_6s:heading_mae_deg: +4.6548 [+3.2844, +6.1709], floor +3.6929
    * COST TAC-LAT: -0.1409 [-0.1765, -0.1076], floor +0.0417
    * COST TAC-LON: -0.1155 [-0.1617, -0.0717], floor +0.0950

| family | cell | s0 delta [CI] | s0 floor | s0 | s1 delta [CI] | s1 | 
|---|---|---|---|---|---|---|
| PRIMARY | operative_0_2s:ade_m | +0.0384 [+0.0198, +0.0711] | +0.0031 | COST | +0.0325 [+0.0147, +0.0627] | COST |
| PRIMARY | operative_0_2s:fde_m | +0.0545 [+0.0089, +0.1390] | +0.0250 | COST | +0.0431 [-0.0018, +0.1219] |  |
| LONGITUDINAL | operative_0_2s:speed_mae_mps | +0.1370 [+0.1058, +0.1877] | +0.0296 | COST | +0.1208 [+0.0907, +0.1694] | COST |
| LONGITUDINAL | operative_0_2s:along_mae_m | +0.0437 [+0.0249, +0.0763] | +0.0021 | COST | +0.0369 [+0.0189, +0.0669] | COST |
| LONGITUDINAL | operative_0_2s:accel_mae_mps2 | +0.8312 [+0.6935, +1.0309] | +0.1080 | COST | +0.7685 [+0.6413, +0.9488] | COST |
| LATERAL | operative_0_2s:cross_mae_m | -0.0038 [-0.0067, -0.0013] | +0.0019 | gain | -0.0032 [-0.0069, +0.0002] |  |
| LATERAL | operative_0_2s:heading_mae_deg | -0.0114 [-0.0421, +0.0196] | +0.0603 |  | +0.0185 [-0.0328, +0.0744] |  |
| LATERAL | operative_0_2s:yaw_rate_mae_degps | +0.1952 [+0.0593, +0.3413] | +0.2891 | sep, sub-floor | +0.1379 [-0.0599, +0.3627] |  |
| LATERAL | operative_0_2s:curvature_mae_1pm | +0.0032 [-0.0008, +0.0075] | +0.0066 |  | +0.0008 [-0.0047, +0.0068] |  |
| LONGITUDINAL | operative_0_2s:dk_hw_abs_err_m | -0.0342 [-0.1572, +0.0452] | +0.0241 |  | +0.0625 [+0.0167, +0.1074] | COST |
| LONGITUDINAL | operative_0_2s:dk_tg_abs_err_s | +0.0065 [-0.0101, +0.0210] | +0.0017 |  | +0.0009 [-0.0248, +0.0176] |  |
| PRIMARY | tactical_2_6s:ade_m | +0.7341 [+0.5047, +1.0293] | +0.6021 | COST | +0.6852 [+0.4716, +0.9651] | COST |
| PRIMARY | tactical_2_6s:fde_m | +2.2062 [+1.5965, +2.9477] | +1.6794 | COST | +2.2660 [+1.5952, +3.0740] | COST |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps | +0.9135 [+0.7089, +1.1421] | +0.4911 | COST | +0.7465 [+0.5449, +0.9816] | COST |
| LONGITUDINAL | tactical_2_6s:along_mae_m | +0.6302 [+0.4200, +0.9217] | +0.6884 | sep, sub-floor | +0.4229 [+0.2025, +0.7375] | sep, sub-floor |
| LONGITUDINAL | tactical_2_6s:accel_mae_mps2 | +1.9840 [+1.6150, +2.3841] | +0.8005 | COST | +1.5602 [+1.1889, +1.9681] | COST |
| LATERAL | tactical_2_6s:cross_mae_m | +0.2977 [+0.1862, +0.4228] | +0.0259 | COST | +0.5096 [+0.3678, +0.6607] | COST |
| LATERAL | tactical_2_6s:heading_mae_deg | +3.2524 [+2.2481, +4.3288] | +3.6929 | sep, sub-floor | +4.6548 [+3.2844, +6.1709] | COST |
| LATERAL | tactical_2_6s:yaw_rate_mae_degps | +3.7989 [+0.5048, +6.9962] | +24.4451 | sep, sub-floor | +5.4143 [+1.8319, +9.4729] | sep, sub-floor |
| LATERAL | tactical_2_6s:curvature_mae_1pm | -0.0039 [-0.0305, +0.0227] | +0.2066 |  | +0.0171 [-0.0309, +0.0637] |  |
| LONGITUDINAL | tactical_2_6s:dk_hw_abs_err_m | -0.0061 [-0.1829, +0.1430] | +0.3268 |  | -0.0037 [-0.1995, +0.1992] |  |
| LONGITUDINAL | tactical_2_6s:dk_tg_abs_err_s | +0.0149 [-0.0343, +0.0686] | +0.1231 |  | -0.0100 [-0.0719, +0.0470] |  |
| TACTICAL | TAC-LAT | -0.0729 [-0.1026, -0.0450] | +0.0417 | COST | -0.1409 [-0.1765, -0.1076] | COST |
| TACTICAL | TAC-LON | -0.0992 [-0.1389, -0.0604] | +0.0950 | COST | -0.1155 [-0.1617, -0.0717] | COST |

## BAND question: OBD-BAND against OBD-CTL, floor G
**Outcome: BAND-COSTS**

* seed 0: BAND-COSTS (reference OBD-CTL_s0, 24 cells read)
    * COST operative_0_2s:heading_mae_deg: +0.0311 [+0.0059, +0.0620], floor +0.0176
    * gain operative_0_2s:ade_m: -0.0168 [-0.0275, -0.0098], floor +0.0074
    * gain operative_0_2s:fde_m: -0.0393 [-0.0678, -0.0214], floor +0.0008
    * gain operative_0_2s:along_mae_m: -0.0173 [-0.0283, -0.0102], floor +0.0088
    * gain operative_0_2s:dk_hw_abs_err_m: -0.0427 [-0.0783, -0.0053], floor +0.0417
    * gain operative_0_2s:dk_tg_abs_err_s: -0.0245 [-0.0602, -0.0030], floor +0.0160
* seed 1: BAND-COSTS (reference OBD-CTL_s1, 24 cells read)
    * COST operative_0_2s:heading_mae_deg: +0.0313 [+0.0003, +0.0681], floor +0.0176
    * gain operative_0_2s:ade_m: -0.0149 [-0.0202, -0.0105], floor +0.0074
    * gain operative_0_2s:fde_m: -0.0267 [-0.0420, -0.0140], floor +0.0008
    * gain operative_0_2s:along_mae_m: -0.0157 [-0.0210, -0.0112], floor +0.0088
    * gain tactical_2_6s:speed_mae_mps: -0.2527 [-0.3612, -0.1435], floor +0.2081
    * gain tactical_2_6s:accel_mae_mps2: -0.4279 [-0.6768, -0.1972], floor +0.3820
    * gain tactical_2_6s:yaw_rate_mae_degps: -4.3512 [-7.6631, -1.4572], floor +3.1765
    * gain TAC-LAT: +0.0371 [+0.0141, +0.0630], floor +0.0271

| family | cell | s0 delta [CI] | s0 floor | s0 | s1 delta [CI] | s1 | 
|---|---|---|---|---|---|---|
| PRIMARY | operative_0_2s:ade_m | -0.0168 [-0.0275, -0.0098] | +0.0074 | gain | -0.0149 [-0.0202, -0.0105] | gain |
| PRIMARY | operative_0_2s:fde_m | -0.0393 [-0.0678, -0.0214] | +0.0008 | gain | -0.0267 [-0.0420, -0.0140] | gain |
| LONGITUDINAL | operative_0_2s:speed_mae_mps | -0.0184 [-0.0362, -0.0056] | +0.0390 | sep, sub-floor | -0.0258 [-0.0335, -0.0184] | sep, sub-floor |
| LONGITUDINAL | operative_0_2s:along_mae_m | -0.0173 [-0.0283, -0.0102] | +0.0088 | gain | -0.0157 [-0.0210, -0.0112] | gain |
| LONGITUDINAL | operative_0_2s:accel_mae_mps2 | -0.0450 [-0.1123, +0.0074] | +0.2447 |  | -0.1057 [-0.1379, -0.0746] | sep, sub-floor |
| LATERAL | operative_0_2s:cross_mae_m | -0.0006 [-0.0023, +0.0009] | +0.0003 |  | -0.0006 [-0.0020, +0.0007] |  |
| LATERAL | operative_0_2s:heading_mae_deg | +0.0311 [+0.0059, +0.0620] | +0.0176 | COST | +0.0313 [+0.0003, +0.0681] | COST |
| LATERAL | operative_0_2s:yaw_rate_mae_degps | +0.1339 [+0.0172, +0.2912] | +0.1971 | sep, sub-floor | +0.0705 [-0.0104, +0.1609] |  |
| LATERAL | operative_0_2s:curvature_mae_1pm | +0.0035 [+0.0001, +0.0078] | +0.0062 | sep, sub-floor | +0.0025 [-0.0000, +0.0054] |  |
| LONGITUDINAL | operative_0_2s:dk_hw_abs_err_m | -0.0427 [-0.0783, -0.0053] | +0.0417 | gain | +0.0073 [-0.0468, +0.0939] |  |
| LONGITUDINAL | operative_0_2s:dk_tg_abs_err_s | -0.0245 [-0.0602, -0.0030] | +0.0160 | gain | -0.0012 [-0.0096, +0.0107] |  |
| PRIMARY | tactical_2_6s:ade_m | -0.0235 [-0.1147, +0.0672] | +0.2869 |  | -0.0600 [-0.1539, +0.0306] |  |
| PRIMARY | tactical_2_6s:fde_m | -0.0030 [-0.3080, +0.3190] | +0.8925 |  | -0.5278 [-0.8690, -0.2178] | sep, sub-floor |
| LONGITUDINAL | tactical_2_6s:speed_mae_mps | -0.0139 [-0.1126, +0.0842] | +0.2081 |  | -0.2527 [-0.3612, -0.1435] | gain |
| LONGITUDINAL | tactical_2_6s:along_mae_m | +0.0081 [-0.0813, +0.0985] | +0.3542 |  | -0.0391 [-0.1331, +0.0554] |  |
| LONGITUDINAL | tactical_2_6s:accel_mae_mps2 | +0.0864 [-0.1110, +0.3000] | +0.3820 |  | -0.4279 [-0.6768, -0.1972] | gain |
| LATERAL | tactical_2_6s:cross_mae_m | -0.0353 [-0.0877, +0.0152] | +0.0331 |  | -0.0273 [-0.0984, +0.0366] |  |
| LATERAL | tactical_2_6s:heading_mae_deg | +0.3938 [-0.1904, +0.9773] | +1.9076 |  | -0.2212 [-0.9609, +0.4723] |  |
| LATERAL | tactical_2_6s:yaw_rate_mae_degps | -1.9620 [-4.5450, +0.2681] | +3.1765 |  | -4.3512 [-7.6631, -1.4572] | gain |
| LATERAL | tactical_2_6s:curvature_mae_1pm | -0.0088 [-0.0305, +0.0112] | +0.0354 |  | -0.0234 [-0.0621, +0.0109] |  |
| LONGITUDINAL | tactical_2_6s:dk_hw_abs_err_m | -0.1449 [-0.2941, +0.0125] | +0.1868 |  | +0.0983 [-0.0644, +0.2458] |  |
| LONGITUDINAL | tactical_2_6s:dk_tg_abs_err_s | -0.0420 [-0.1016, +0.0048] | +0.0259 |  | +0.0122 [-0.0200, +0.0425] |  |
| TACTICAL | TAC-LAT | +0.0229 [+0.0017, +0.0455] | +0.0271 | sep, sub-floor | +0.0371 [+0.0141, +0.0630] | gain |
| TACTICAL | TAC-LON | -0.0050 [-0.0250, +0.0142] | +0.0425 |  | +0.0208 [-0.0029, +0.0421] |  |

## What the card commits to do next
* DETACH-COSTS: the base cannot meet R-T2O at this recipe without the named cost. Next levers, cheapest first, under this card's rules: (1) T's own loss weight --w-tac x2 (the detach removes O's gradient into T, so T's own term carries T alone), (2) T's own DAgger term on the DAgger rows (already present; read its weight), (3) report the costing cells to the PI: R-T2O is the PI's ruling, and accepting a measured cost of it is the PI's decision. (band question read beside: BAND-COSTS; the card commits no combined action)
