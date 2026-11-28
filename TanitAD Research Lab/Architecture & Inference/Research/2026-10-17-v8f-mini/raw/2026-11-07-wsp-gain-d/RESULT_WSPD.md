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

READ 1 beside, eval-141 band windows (3-8 m/s, n 586; never deciding):

| seed | C0 (v0, a0) | edited (v0, a0) | rel err vs g_data (v0, a0) | mean plan-speed shift |
|---|---|---|---|---|
| s0 | 0.8058, 1.2614 | 0.8747, 1.3531 | 0.060, 0.008 | +0.3130 |
| s1 | 0.7616, 1.1133 | 0.8443, 1.2731 | 0.092, 0.066 | +0.3302 |

## READ 2 [M] T1-kin eval-141, E-GAIN-D vs C0 (RIC0) and vs P10: **HARMS**

M1 `share_closed` = -(D - C0)[M1] / (C0 - P10)[M1].

| seed | M1 D-C0 [CI] | M1 C0-P10 | share_closed | n worse vs P10 (D / C0) | new separated-worse rows |
|---|---|---|---|---|---|
| s0 | -0.0175 [-0.0241, -0.0117] | +0.0533 | 0.329 | 3 / 3 | ADE6, DK-HW, DK-VIOL, TAC-LON, accel MAE, along-track MAE, cross-track MAE, heading MAE, speed MAE |
| s1 | -0.0172 [-0.0242, -0.0104] | +0.0596 | 0.289 | 2 / 3 | ADE6, DK-HW, DK-VIOL, TAC-LON, accel MAE, along-track MAE, cross-track MAE, heading MAE, speed MAE |

Every row, both seeds (delta [CI] = D - C0; status: better / worse = CI excludes 0 in that direction, ns otherwise; vs P10 status in the last columns):

| family | row | s0 D-C0 | s0 vs C0 | s0 vs P10 | s1 D-C0 | s1 vs C0 | s1 vs P10 |
|---|---|---|---|---|---|---|---|
| LONGITUDINAL | speed MAE | +0.0976 [+0.0323, +0.1659] | worse | better | +0.1789 [+0.1202, +0.2384] | worse | better |
| LONGITUDINAL | along-track MAE | +0.1969 [+0.1108, +0.2897] | worse | better | +0.3130 [+0.2286, +0.4014] | worse | better |
| LONGITUDINAL | accel MAE | +0.1712 [+0.0837, +0.2541] | worse | better | +0.2036 [+0.1358, +0.2739] | worse | better |
| LONGITUDINAL | DK-VIOL | +0.0118 [+0.0047, +0.0200] | worse | ns | +0.0101 [+0.0016, +0.0208] | worse | ns |
| LONGITUDINAL | DK-HW | +0.2355 [+0.0025, +0.5309] | worse | ns | +0.1764 [+0.0056, +0.3720] | worse | ns |
| LONGITUDINAL | DK-TG | +0.0222 [-0.0037, +0.0505] | ns | better | -0.0095 [-0.0601, +0.0275] | ns | better |
| LATERAL | heading MAE | +0.5824 [+0.0861, +1.2122] | worse | better | +1.2969 [+0.4588, +2.3244] | worse | ns |
| LATERAL | yaw-rate MAE | +0.2192 [-1.2330, +1.8796] | ns | worse | -2.6944 [-4.8217, -1.0317] | better | ns |
| LATERAL | curvature MAE | +0.0029 [-0.0092, +0.0155] | ns | worse | -0.0122 [-0.0254, -0.0006] | better | worse |
| LATERAL | cross-track MAE | +0.0233 [+0.0032, +0.0430] | worse | better | +0.0239 [+0.0021, +0.0459] | worse | better |
| LATERAL | L1 lateral-accel MAE | +0.1855 [-0.1784, +0.6258] | ns | better | -0.1683 [-0.4195, +0.0606] | ns | better |
| LATERAL | L2 curvature MAE (GT>=5.2) | -0.0188 [-0.0375, -0.0054] | better | worse | -0.0162 [-0.0299, -0.0059] | better | worse |
| TACTICAL | TAC-LAT | -0.0008 [-0.0129, +0.0096] | ns | better | +0.0142 [+0.0008, +0.0300] | better | better |
| TACTICAL | TAC-LON | -0.0604 [-0.1217, -0.0025] | worse | better | -0.0763 [-0.1287, -0.0242] | worse | better |
| STRATEGIC (PAUSED, PI item 25) | S-POS@30 | -0.0570 [-0.1565, +0.0611] | ns | better | +0.4089 [+0.1309, +0.7586] | worse | better |
| STRATEGIC (PAUSED, PI item 25) | S-POS@50 | -0.1755 [-0.3732, +0.0478] | ns | better | +0.6572 [+0.1514, +1.2411] | worse | better |
| STRATEGIC (PAUSED, PI item 25) | S-POS@80 | -0.3556 [-0.7075, +0.0409] | ns | better | +0.8550 [+0.0534, +1.8178] | worse | better |
| ADE and M1 | ADE6 | +0.1789 [+0.1023, +0.2593] | worse | better | +0.2846 [+0.2060, +0.3624] | worse | better |
| ADE and M1 | M1 crawl share | -0.0175 [-0.0241, -0.0117] | better | worse | -0.0172 [-0.0242, -0.0104] | better | worse |

Beside, banked WSP-GAIN alpha 1.0 (`wsp_edit/raw/READ2.json`): HARMS; M1 delta s0 -0.0132, s1 -0.0146; new separated-worse rows s0 10, s1 9.

## Distance keeping beside (dk141, T1-class open loop; see `raw/DK141_READ_D*.json`)

Present: 1 file(s). Table: `raw/DK141_TABLE_D.md`.

STRATEGIC family: PAUSED (PI item 25); its rows are listed beside and decide nothing. Raw: `raw/READ1_D.json`, `raw/READ1_D_EVAL.json`, `raw/READ2_D.json`. Card: `PREREG_WSP_GAIN_D.md` section 6.
