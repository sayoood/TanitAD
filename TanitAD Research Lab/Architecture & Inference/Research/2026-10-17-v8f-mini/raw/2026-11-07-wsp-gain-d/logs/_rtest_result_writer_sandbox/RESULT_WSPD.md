# RESULT_WSPD: E-GAIN-D (the weight edit aimed at the data's own longitudinal gains), card PREREG_WSP_GAIN_D.md

**Verdict by the card's section 4 table: HARMS.** Next step it commits: the data-matched point joins P3's chain beside WSP-GAIN's; no T2 stage; the perception lever (LoRA) stays the route.

Tiers: READ 1 [M] T0-class mechanism check; READ 2 [M] T1-kin on eval-141 (scene `reindexed`), STRATEGIC family PAUSED (PI item 25). Estimator: READ 2: paired episode-cluster bootstrap B 2,000 seed 0 (another draw of EPISODES only; the edit is a deterministic function of C0, so no training seed is in a D - C0 contrast).

## Fit [M] (20,000 TRAIN fit windows)

g_data: ds/dv0 0.9303 [0.9126090951755731, 0.9483152496713488], ds/da0 1.3637 [1.3373736850998392, 1.3901126456684132]; P5 beside 0.9355 / 1.3841.

| seed | g_model (v0, a0) | delta-g (v0, a0) | delta-m m/s | target shift m/s | ckpt sha256 |
|---|---|---|---|---|---|
| s0 | 0.7705, 1.2080 | +0.1598, +0.1557 | +0.4687 | +0.3383 | `30bdd83ce3c037ee` |
| s1 | 0.7536, 1.1310 | +0.1767, +0.2328 | +0.4784 | +0.3682 | `8d89c61086bb032a` |

## READ 1 [M] (TRAIN fit windows, +-10 % of g_data at both seeds and both gains): **MECH-OK**

| seed | gain | C0 | edited | g_data | rel err | within |
|---|---|---|---|---|---|---|
| s0 | dS/dv0 | 0.7705 | 0.8533 | 0.9303 | 0.083 | True |
| s0 | dS/da0 | 1.2080 | 1.3168 | 1.3637 | 0.034 | True |
| s1 | dS/dv0 | 0.7536 | 0.8427 | 0.9303 | 0.094 | True |
| s1 | dS/da0 | 1.1310 | 1.2980 | 1.3637 | 0.048 | True |

s0: mean plan-speed shift vs C0 +0.3383 m/s (target +0.3383); edited mean plan speed 5.5875 vs realised mean 5.5875.

s1: mean plan-speed shift vs C0 +0.3682 m/s (target +0.3682); edited mean plan speed 5.5875 vs realised mean 5.5875.

## READ 2 [M] T1-kin eval-141, E-GAIN-D vs C0 (RIC0) and vs P10: **HARMS**

M1 `share_closed` = -(D - C0)[M1] / (C0 - P10)[M1].

| seed | M1 D-C0 [CI] | M1 C0-P10 | share_closed | n worse vs P10 (D / C0) | new separated-worse rows |
|---|---|---|---|---|---|
| s0 | -0.0132 [-0.0204, -0.0065] | +0.0533 | 0.247 | 3 / 3 | ADE6, DK-TG, DK-VIOL, TAC-LON, accel MAE, along-track MAE, cross-track MAE, curvature MAE, heading MAE, speed MAE |
| s1 | -0.0146 [-0.0231, -0.0070] | +0.0596 | 0.246 | 3 / 3 | ADE6, DK-VIOL, TAC-LON, accel MAE, along-track MAE, cross-track MAE, curvature MAE, heading MAE, speed MAE |

Every row, both seeds (delta [CI] = D - C0; status: better / worse = CI excludes 0 in that direction, ns otherwise; vs P10 status in the last columns):

| family | row | s0 D-C0 | s0 vs C0 | s0 vs P10 | s1 D-C0 | s1 vs C0 | s1 vs P10 |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE | +0.2365 [+0.1580, +0.3161] | worse | better | +0.3109 [+0.2377, +0.3830] | worse | better |
| LONGITUDINAL | along-track MAE | +0.4506 [+0.3398, +0.5697] | worse | better | +0.5049 [+0.4105, +0.6082] | worse | ns |
| LONGITUDINAL | accel MAE | +0.6084 [+0.4725, +0.7289] | worse | ns | +0.5410 [+0.4381, +0.6407] | worse | better |
| LONGITUDINAL | DK-VIOL | +0.0168 [+0.0050, +0.0326] | worse | ns | +0.0134 [+0.0018, +0.0275] | worse | ns |
| LONGITUDINAL | DK-HW | +0.2946 [-0.0085, +0.6657] | ns | ns | +0.2066 [-0.0253, +0.4602] | ns | ns |
| LONGITUDINAL | DK-TG | +0.0397 [+0.0022, +0.0815] | worse | better | -0.0037 [-0.0637, +0.0439] | ns | better |
| LATERAL | heading MAE | +1.5668 [+1.0318, +2.2060] | worse | ns | +1.6141 [+1.0693, +2.2753] | worse | ns |
| LATERAL | yaw-rate MAE | +0.7249 [-1.1074, +2.4036] | ns | worse | +0.8112 [-1.4349, +2.8946] | ns | worse |
| LATERAL | curvature MAE | +0.0335 [+0.0170, +0.0521] | worse | worse | +0.0474 [+0.0240, +0.0752] | worse | worse |
| LATERAL | cross-track MAE | +0.0436 [+0.0146, +0.0743] | worse | better | +0.0310 [+0.0035, +0.0588] | worse | better |
| LATERAL | L1 lateral-accel MAE | -0.2294 [-0.5974, +0.0757] | ns | better | -0.2242 [-0.5694, +0.0719] | ns | better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0115 [-0.0312, +0.0015] | ns | worse | -0.0067 [-0.0227, +0.0088] | ns | worse |
| TACTICAL | TAC-LAT | -0.0071 [-0.0217, +0.0071] | ns | better | +0.0163 [+0.0021, +0.0362] | better | better |
| TACTICAL | TAC-LON | -0.0855 [-0.1484, -0.0250] | worse | better | -0.0959 [-0.1502, -0.0425] | worse | ns |
| STRATEGIC (PAUSED, PI item 25) | S-POS@30 | +0.1710 [-0.0456, +0.4423] | ns | better | +0.6027 [+0.2523, +1.0192] | worse | better |
| STRATEGIC (PAUSED, PI item 25) | S-POS@50 | +0.2250 [-0.1510, +0.6820] | ns | better | +0.9590 [+0.3702, +1.6759] | worse | better |
| STRATEGIC (PAUSED, PI item 25) | S-POS@80 | +0.2986 [-0.3015, +1.0228] | ns | better | +1.3133 [+0.3726, +2.4080] | worse | better |
| ADE and M1 | ADE6 | +0.4242 [+0.3238, +0.5315] | worse | better | +0.4659 [+0.3744, +0.5607] | worse | better |
| ADE and M1 | M1 crawl share | -0.0132 [-0.0204, -0.0065] | better | worse | -0.0146 [-0.0231, -0.0070] | better | worse |

Beside, banked WSP-GAIN alpha 1.0 (`wsp_edit/raw/READ2.json`): HARMS; M1 delta s0 -0.0132, s1 -0.0146; new separated-worse rows s0 10, s1 9.

STRATEGIC family: PAUSED (PI item 25); its rows are listed beside and decide nothing. Raw: `raw/READ1_D.json`, `raw/READ1_D_EVAL.json`, `raw/READ2_D.json`. Card: `PREREG_WSP_GAIN_D.md` section 6.
