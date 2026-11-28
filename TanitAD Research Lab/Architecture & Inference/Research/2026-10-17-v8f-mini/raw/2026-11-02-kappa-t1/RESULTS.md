# RESULTS: H-KAPPA-T1-1 Part 2, X = S + curvature-bounded decode (box), Y = S rerun, DAG-OACC and SOUP at s0 / s1

Card `PREREG_FROZEN.md` sha256 `9564d56836e065290f27d4f641077008232b76ec4a7465eabe9f93654046f72c` (frozen 2026-11-02T15:44:58+0100). Reader `code/kappa_t1_read.py` under `READER_FIX_FLOOR.md` (an extractor-guard fix; not an amendment). Read clock 2026-11-02T16:19:43+0100. Raw `raw/KAPPA_T1.json`, per-window arrays `raw/arrays/*.npz`. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class **MEASURED (ours)**.
Estimator of every interval below: **paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. One training run per seed and a deterministic planner: the intervals do not answer *another training run* (`H-ESTIM-SEED-1`) nor *another inference run*.

## 0. Outcome (the card's rule, precedence DOMINANT > HARMS > MOVED > NO EFFECT)

| arm | outcome | seed 0: D1 / D2 / D3 | seed 1: D1 / D2 / D3 | seed 0: MOVED / HARMS | seed 1: MOVED / HARMS |
|---|---|---|---|---|---|
| DAG-OACC | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| SOUP | **HARMS** | True / False / True | True / False / True | False / True | False / True |

Headline arm DAG-OACC; SOUP stated separately. `moved_and_harms` flags (HARMS precedence hides a MOVED condition): DAG-OACC False, SOUP False.

### What triggers HARMS and what blocks DOMINANT, per arm and seed (generated from the rows)

* **DAG-OACC s0**: HARMS triggers: S-POS@30 X - Y +0.4995 SW; S-POS@50 X - Y +0.7836 SW; S-POS@80 X - Y +1.0942 SW; heading MAE X - Y +1.6946 SW. DOMINANT blocked by: D2 rows M1 crawl share (D1 True, D3 True). Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not separated-better: M1 crawl share.
* **DAG-OACC s1**: HARMS triggers: ADE6 X - Y +0.0606 > guard +0.0604; ADE6 X - Y +0.0606 SW; along-track MAE X - Y +0.0607 SW; heading MAE X - Y +1.3399 SW. DOMINANT blocked by: D2 rows M1 crawl share (D1 True, D3 True). Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not separated-better: M1 crawl share.
* **SOUP s0**: HARMS triggers: S-POS@30 X - Y +0.6435 SW; S-POS@50 X - Y +1.0833 SW; S-POS@80 X - Y +1.5711 SW; heading MAE X - Y +1.6201 SW. DOMINANT blocked by: D2 rows M1 crawl share (D1 True, D3 True). Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not separated-better: M1 crawl share.
* **SOUP s1**: HARMS triggers: S-POS@30 X - Y +0.3612 SW; S-POS@50 X - Y +0.5467 SW; S-POS@80 X - Y +0.7712 SW; along-track MAE X - Y +0.0373 SW; heading MAE X - Y +1.4407 SW. DOMINANT blocked by: D2 rows M1 crawl share (D1 True, D3 True). Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not separated-better: M1 crawl share.

## 1. Known answers and controls (all computed before any row)

* **KA1_Y_reproduces_out_m16_S**: PASS
* **KA2_issued_plan_is_box_of_S**: PASS
* **KF_floor_window_check_vs_fresh_Y**: PASS
* **K3_Y_banked_minus_P10_reproduces_SOUP_READ**: PASS
* **KB1_P10_side_equals_landed_banked_b**: PASS

**KA1** (Y equals `out_m16/SMOOTH_*`, bit for bit; tolerance 1e-4 m):

| run | dump bit-identical | dump max abs dev (m) | ticks bit-identical | ticks max abs dev |
|---|---|---|---|---|
| CTRL_s0 | True | 0.000e+00 | True | 0.000e+00 |
| CTRL_s1 | True | 0.000e+00 | True | 0.000e+00 |
| SOUP_s0 | True | 0.000e+00 | True | 0.000e+00 |
| SOUP_s1 | True | 0.000e+00 | True | 0.000e+00 |

**KA2** (X's issued `plan` = box(P + SG(plan_raw - P)); tick-0 `plan_raw` = Y's; floor = Y's; tolerance 1e-4 m):

| run | plan_s vs P+SG (m) | plan vs box(plan_s) (m) | tick-0 plan_raw vs Y (m) | floor X vs Y (m) | box moved the issued plan by up to (m) |
|---|---|---|---|---|---|
| CTRL_s0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 227.217 |
| CTRL_s1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 126.545 |
| SOUP_s0 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 218.270 |
| SOUP_s1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 105.187 |

**KF** (READER_FIX_FLOOR.md: the floor / window check against the fresh Y; gross guard 1.0 m against the banked CTRL):

| run | windows X = fresh Y | windows fresh Y = banked CTRL | floor X vs fresh Y (m) | floor fresh Y vs banked CTRL max / mean (m) | floor X vs banked CTRL max (m) |
|---|---|---|---|---|---|
| CTRL_s0 | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 | 1.711e-02 |
| CTRL_s1 | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 | 1.711e-02 |
| SOUP_s0 | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 | 1.711e-02 |
| SOUP_s1 | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 | 1.711e-02 |

**K3**: Y banked - P10 from the landed arrays reproduces the pinned `SOUP_READ.json` on 76 / 76 rows (tolerance 6e-05). **KB1**: the P10 side of every extracted X and Y equals the landed banked `b` arrays on the non-S-POS rows: 8 / 8 runs pass.

The last KA2 column is the maximum over every tick and knot of |plan - plan_s| (the box's displacement of the issued plan); the gate's 32-window tool smoke read a maximum of 10.2 m.

**KA3** (pytest on the patched tool, off returns the same object, round trip, planted plan inside the box, deleting the call turns RED): from `logs/pytest_feasible_smooth.log` and `logs/pytest_existing_t1.log`, run when the tool was built (HANDOFF.md); not re-run here.

## 2. Read A against P10 and X - Y, per arm and seed

### DAG-OACC s0

* **PRIMARY ADE6 (m)**: X - P10 -1.2441 [-1.4510, -1.0498] SB (D1 bar point <= -0.2415 and SB: **True**); X - Y +0.0115 [-0.0251, +0.0514] ns (guard point <= +0.0604: **True**); Y - P10 -1.2556 [-1.4503, -1.0678] SB.
* **D2** (no row of 19 separated-worse, X - P10): **False**; separated-worse rows: M1 crawl share.
* **D3** (a separated-better row per family, X - P10): {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: ADE6, DK-HW, DK-TG, L1 lateral-accel MAE, S-POS@30, S-POS@50, S-POS@80, TAC-LAT, TAC-LON, accel MAE, along-track MAE, cross-track MAE, curvature MAE, heading MAE, speed MAE, yaw-rate MAE.
* **The four crawl rows, X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, heading MAE. Rows separated-worse on X - Y, all 19: 5.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2441 [-1.4510, -1.0498] SB | +0.0115 [-0.0251, +0.0514] ns | -1.2556 [-1.4503, -1.0678] SB |
| speed MAE | LONGITUDINAL | -0.9406 [-1.1732, -0.7388] SB | -0.1076 [-0.1716, -0.0580] SB | -0.8330 [-1.0377, -0.6583] SB |
| along-track MAE | LONGITUDINAL | -0.6965 [-0.8662, -0.5336] SB | +0.0227 [-0.0141, +0.0614] ns | -0.7192 [-0.8806, -0.5627] SB |
| accel MAE | LONGITUDINAL | -1.2790 [-1.7527, -0.8840] SB | -0.3399 [-0.4948, -0.2184] SB | -0.9390 [-1.3078, -0.6221] SB |
| DK-VIOL | LONGITUDINAL | +0.0151 [-0.0133, +0.0479] ns | -0.0017 [-0.0207, +0.0164] ns | +0.0168 [-0.0157, +0.0625] ns |
| DK-HW | LONGITUDINAL | -0.4168 [-0.7965, -0.1086] SB | -0.0685 [-0.1405, +0.0037] ns | -0.2604 [-0.6426, +0.0675] ns |
| DK-TG | LONGITUDINAL | -0.1085 [-0.1750, -0.0552] SB | -0.0271 [-0.0606, +0.0023] ns | -0.0588 [-0.1226, -0.0011] SB |
| heading MAE | LATERAL | -1.7574 [-2.8826, -0.4239] SB | +1.6946 [+0.3667, +3.3224] SW | -3.4520 [-4.2140, -2.7529] SB |
| yaw-rate MAE | LATERAL | -12.8450 [-18.0767, -8.3421] SB | -11.6134 [-15.7105, -8.0623] SB | -1.3762 [-3.9121, +1.0259] ns |
| curvature MAE | LATERAL | -0.0353 [-0.0517, -0.0212] SB | -0.0949 [-0.1327, -0.0633] SB | +0.0578 [+0.0364, +0.0821] SW |
| cross-track MAE | LATERAL | -1.0158 [-1.1716, -0.8733] SB | -0.0148 [-0.0515, +0.0236] ns | -1.0009 [-1.1468, -0.8671] SB |
| L1 lateral-accel MAE | LATERAL | -3.6447 [-5.1756, -2.3182] SB | -0.9568 [-1.3933, -0.5973] SB | -2.7327 [-4.0998, -1.5826] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0010 [-0.0039, +0.0022] ns | -0.0292 [-0.0536, -0.0137] SB | +0.0325 [+0.0135, +0.0590] SW |
| TAC-LAT | TACTICAL | +0.2180 [+0.1802, +0.2584] SB | +0.0142 [+0.0029, +0.0259] SB | +0.2038 [+0.1639, +0.2450] SB |
| TAC-LON | TACTICAL | +0.1784 [+0.1129, +0.2431] SB | +0.0033 [-0.0129, +0.0208] ns | +0.1751 [+0.1138, +0.2389] SB |
| S-POS@30 | STRATEGIC | -0.8101 [-1.1147, -0.5170] SB | +0.4995 [+0.1678, +0.9471] SW | -1.6443 [-2.1654, -1.1864] SB |
| S-POS@50 | STRATEGIC | -2.9607 [-3.8941, -2.1055] SB | +0.7836 [+0.1805, +1.5747] SW | -4.5745 [-5.8089, -3.5098] SB |
| S-POS@80 | STRATEGIC | -7.5920 [-9.6109, -5.6213] SB | +1.0942 [+0.0516, +2.4176] SW | -9.8284 [-12.1173, -7.7691] SB |
| M1 crawl share | crawl | +0.0528 [+0.0340, +0.0753] SW | +0.0048 [+0.0019, +0.0084] SW | +0.0483 [+0.0302, +0.0698] SW |

### DAG-OACC s1

* **PRIMARY ADE6 (m)**: X - P10 -1.1296 [-1.3370, -0.9226] SB (D1 bar point <= -0.2415 and SB: **True**); X - Y +0.0606 [+0.0261, +0.0995] SW (guard point <= +0.0604: **False**); Y - P10 -1.1903 [-1.3986, -0.9782] SB.
* **D2** (no row of 19 separated-worse, X - P10): **False**; separated-worse rows: M1 crawl share.
* **D3** (a separated-better row per family, X - P10): {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: ADE6, DK-HW, DK-TG, L1 lateral-accel MAE, S-POS@30, S-POS@50, S-POS@80, TAC-LAT, TAC-LON, accel MAE, along-track MAE, cross-track MAE, curvature MAE, heading MAE, speed MAE, yaw-rate MAE.
* **The four crawl rows, X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, along-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 4.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1296 [-1.3370, -0.9226] SB | +0.0606 [+0.0261, +0.0995] SW | -1.1903 [-1.3986, -0.9782] SB |
| speed MAE | LONGITUDINAL | -0.8745 [-1.1079, -0.6701] SB | -0.0604 [-0.1013, -0.0306] SB | -0.8141 [-1.0377, -0.6191] SB |
| along-track MAE | LONGITUDINAL | -0.5695 [-0.7551, -0.3943] SB | +0.0607 [+0.0223, +0.1042] SW | -0.6302 [-0.8196, -0.4510] SB |
| accel MAE | LONGITUDINAL | -1.2574 [-1.7283, -0.8642] SB | -0.2855 [-0.3931, -0.1928] SB | -0.9719 [-1.3431, -0.6572] SB |
| DK-VIOL | LONGITUDINAL | -0.0050 [-0.0255, +0.0154] ns | -0.0050 [-0.0187, +0.0100] ns | +0.0000 [-0.0196, +0.0212] ns |
| DK-HW | LONGITUDINAL | -0.5927 [-0.9981, -0.2870] SB | -0.0875 [-0.1750, -0.0069] SB | -0.5171 [-0.9057, -0.2118] SB |
| DK-TG | LONGITUDINAL | -0.1329 [-0.2062, -0.0753] SB | -0.0491 [-0.1281, +0.0088] ns | -0.0869 [-0.1542, -0.0278] SB |
| heading MAE | LATERAL | -1.5536 [-2.6252, -0.3273] SB | +1.3399 [+0.1946, +2.6951] SW | -2.8936 [-3.6062, -2.1968] SB |
| yaw-rate MAE | LATERAL | -12.8375 [-18.2495, -8.3385] SB | -16.9787 [-23.6484, -11.2331] SB | +4.1412 [+1.1665, +7.3821] SW |
| curvature MAE | LATERAL | -0.0333 [-0.0496, -0.0191] SB | -0.1545 [-0.2149, -0.1013] SB | +0.1212 [+0.0757, +0.1729] SW |
| cross-track MAE | LATERAL | -1.0035 [-1.1548, -0.8575] SB | +0.0182 [-0.0075, +0.0499] ns | -1.0217 [-1.1686, -0.8803] SB |
| L1 lateral-accel MAE | LATERAL | -3.6811 [-5.2510, -2.3413] SB | -1.0729 [-1.5892, -0.6513] SB | -2.6082 [-3.8410, -1.5151] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0011 [-0.0025, +0.0053] ns | -0.0459 [-0.0704, -0.0256] SB | +0.0505 [+0.0269, +0.0791] SW |
| TAC-LAT | TACTICAL | +0.2126 [+0.1720, +0.2525] SB | +0.0000 [-0.0104, +0.0100] ns | +0.2126 [+0.1709, +0.2540] SB |
| TAC-LON | TACTICAL | +0.1672 [+0.1050, +0.2263] SB | +0.0033 [-0.0042, +0.0117] ns | +0.1638 [+0.1021, +0.2225] SB |
| S-POS@30 | STRATEGIC | -1.1885 [-1.4960, -0.9191] SB | +0.2400 [-0.0168, +0.5255] ns | -1.7416 [-2.2209, -1.3273] SB |
| S-POS@50 | STRATEGIC | -3.7780 [-4.6793, -2.9789] SB | +0.3663 [-0.0681, +0.8296] ns | -4.7600 [-5.8592, -3.7484] SB |
| S-POS@80 | STRATEGIC | -9.3415 [-11.3993, -7.4349] SB | +0.4863 [-0.2260, +1.2065] ns | -10.2999 [-12.3713, -8.3307] SB |
| M1 crawl share | crawl | +0.0627 [+0.0395, +0.0902] SW | +0.0071 [+0.0029, +0.0122] SW | +0.0565 [+0.0347, +0.0829] SW |

### SOUP s0

* **PRIMARY ADE6 (m)**: X - P10 -1.1844 [-1.4050, -0.9726] SB (D1 bar point <= -0.2415 and SB: **True**); X - Y +0.0084 [-0.0327, +0.0509] ns (guard point <= +0.0604: **True**); Y - P10 -1.1928 [-1.4078, -0.9880] SB.
* **D2** (no row of 19 separated-worse, X - P10): **False**; separated-worse rows: M1 crawl share.
* **D3** (a separated-better row per family, X - P10): {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: ADE6, DK-HW, DK-TG, L1 lateral-accel MAE, S-POS@30, S-POS@50, S-POS@80, TAC-LAT, TAC-LON, accel MAE, along-track MAE, cross-track MAE, curvature MAE, speed MAE, yaw-rate MAE.
* **The four crawl rows, X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, heading MAE. Rows separated-worse on X - Y, all 19: 4.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1844 [-1.4050, -0.9726] SB | +0.0084 [-0.0327, +0.0509] ns | -1.1928 [-1.4078, -0.9880] SB |
| speed MAE | LONGITUDINAL | -0.8819 [-1.1123, -0.6703] SB | -0.1002 [-0.1486, -0.0605] SB | -0.7817 [-0.9919, -0.5956] SB |
| along-track MAE | LONGITUDINAL | -0.6037 [-0.7910, -0.4253] SB | +0.0254 [-0.0181, +0.0678] ns | -0.6291 [-0.8126, -0.4582] SB |
| accel MAE | LONGITUDINAL | -1.2034 [-1.6687, -0.8120] SB | -0.3243 [-0.4434, -0.2244] SB | -0.8791 [-1.2581, -0.5583] SB |
| DK-VIOL | LONGITUDINAL | +0.0050 [-0.0223, +0.0331] ns | -0.0017 [-0.0127, +0.0122] ns | +0.0067 [-0.0196, +0.0346] ns |
| DK-HW | LONGITUDINAL | -0.3948 [-0.7731, -0.1068] SB | -0.0726 [-0.1556, -0.0038] SB | -0.2040 [-0.5885, +0.1367] ns |
| DK-TG | LONGITUDINAL | -0.1012 [-0.1618, -0.0511] SB | -0.0451 [-0.1124, +0.0051] ns | -0.0560 [-0.1173, -0.0018] SB |
| heading MAE | LATERAL | -1.3412 [-2.4923, +0.0010] ns | +1.6201 [+0.2740, +3.2205] SW | -2.9614 [-3.7496, -2.1787] SB |
| yaw-rate MAE | LATERAL | -12.2311 [-17.3757, -7.7697] SB | -14.2102 [-19.1308, -9.9723] SB | +1.9270 [-1.0592, +5.0043] ns |
| curvature MAE | LATERAL | -0.0307 [-0.0471, -0.0164] SB | -0.1244 [-0.1702, -0.0842] SB | +0.0919 [+0.0584, +0.1301] SW |
| cross-track MAE | LATERAL | -1.0484 [-1.2045, -0.9059] SB | -0.0219 [-0.0567, +0.0153] ns | -1.0265 [-1.1729, -0.8943] SB |
| L1 lateral-accel MAE | LATERAL | -3.6436 [-5.2073, -2.3087] SB | -1.0017 [-1.4307, -0.6491] SB | -2.6420 [-4.0010, -1.5149] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0025 [-0.0014, +0.0068] ns | -0.0558 [-0.0973, -0.0257] SB | +0.0628 [+0.0279, +0.1076] SW |
| TAC-LAT | TACTICAL | +0.2059 [+0.1637, +0.2465] SB | +0.0213 [+0.0079, +0.0350] SB | +0.1847 [+0.1417, +0.2274] SB |
| TAC-LON | TACTICAL | +0.1588 [+0.0904, +0.2294] SB | +0.0121 [+0.0017, +0.0238] SB | +0.1467 [+0.0792, +0.2138] SB |
| S-POS@30 | STRATEGIC | -0.8487 [-1.1397, -0.5814] SB | +0.6435 [+0.2376, +1.1259] SW | -1.6062 [-2.1525, -1.1336] SB |
| S-POS@50 | STRATEGIC | -3.0612 [-3.9563, -2.2520] SB | +1.0833 [+0.3644, +1.9230] SW | -4.5825 [-5.8449, -3.4774] SB |
| S-POS@80 | STRATEGIC | -7.9468 [-9.8983, -6.1129] SB | +1.5711 [+0.4469, +2.9337] SW | -10.1918 [-12.6081, -8.0751] SB |
| M1 crawl share | crawl | +0.0658 [+0.0431, +0.0915] SW | +0.0004 [-0.0032, +0.0035] ns | +0.0666 [+0.0428, +0.0927] SW |

### SOUP s1

* **PRIMARY ADE6 (m)**: X - P10 -1.1642 [-1.3719, -0.9593] SB (D1 bar point <= -0.2415 and SB: **True**); X - Y +0.0253 [-0.0081, +0.0664] ns (guard point <= +0.0604: **True**); Y - P10 -1.1895 [-1.3938, -0.9847] SB.
* **D2** (no row of 19 separated-worse, X - P10): **False**; separated-worse rows: M1 crawl share.
* **D3** (a separated-better row per family, X - P10): {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: ADE6, DK-HW, DK-TG, L1 lateral-accel MAE, S-POS@30, S-POS@50, S-POS@80, TAC-LAT, TAC-LON, accel MAE, along-track MAE, cross-track MAE, curvature MAE, heading MAE, speed MAE, yaw-rate MAE.
* **The four crawl rows, X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, along-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 5.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1642 [-1.3719, -0.9593] SB | +0.0253 [-0.0081, +0.0664] ns | -1.1895 [-1.3938, -0.9847] SB |
| speed MAE | LONGITUDINAL | -0.8851 [-1.1156, -0.6802] SB | -0.1114 [-0.1600, -0.0680] SB | -0.7737 [-0.9864, -0.5882] SB |
| along-track MAE | LONGITUDINAL | -0.5887 [-0.7738, -0.4118] SB | +0.0373 [+0.0037, +0.0748] SW | -0.6260 [-0.8049, -0.4497] SB |
| accel MAE | LONGITUDINAL | -1.2550 [-1.7178, -0.8587] SB | -0.4586 [-0.6376, -0.3039] SB | -0.7964 [-1.1397, -0.4913] SB |
| DK-VIOL | LONGITUDINAL | -0.0034 [-0.0249, +0.0185] ns | +0.0034 [-0.0035, +0.0126] ns | -0.0067 [-0.0265, +0.0122] ns |
| DK-HW | LONGITUDINAL | -0.6035 [-0.9787, -0.2925] SB | -0.0368 [-0.1114, +0.0387] ns | -0.5481 [-0.9299, -0.2218] SB |
| DK-TG | LONGITUDINAL | -0.1127 [-0.1841, -0.0518] SB | -0.0235 [-0.0946, +0.0436] ns | -0.0863 [-0.1467, -0.0356] SB |
| heading MAE | LATERAL | -1.5040 [-2.5141, -0.3554] SB | +1.4407 [+0.3782, +2.7298] SW | -2.9491 [-3.6329, -2.3131] SB |
| yaw-rate MAE | LATERAL | -12.6520 [-17.9978, -8.1859] SB | -15.8216 [-21.6460, -10.6314] SB | +3.0944 [+0.2216, +6.2120] SW |
| curvature MAE | LATERAL | -0.0336 [-0.0504, -0.0192] SB | -0.1423 [-0.1998, -0.0925] SB | +0.1067 [+0.0666, +0.1533] SW |
| cross-track MAE | LATERAL | -1.0325 [-1.1834, -0.8869] SB | -0.0137 [-0.0422, +0.0209] ns | -1.0189 [-1.1607, -0.8829] SB |
| L1 lateral-accel MAE | LATERAL | -3.6602 [-5.2197, -2.3289] SB | -1.0870 [-1.5474, -0.6854] SB | -2.5744 [-3.8473, -1.4601] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0015 [-0.0021, +0.0057] ns | -0.0428 [-0.0678, -0.0227] SB | +0.0454 [+0.0224, +0.0739] SW |
| TAC-LAT | TACTICAL | +0.2063 [+0.1657, +0.2474] SB | -0.0075 [-0.0209, +0.0058] ns | +0.2138 [+0.1747, +0.2548] SB |
| TAC-LON | TACTICAL | +0.1705 [+0.1088, +0.2318] SB | +0.0121 [+0.0037, +0.0217] SB | +0.1584 [+0.0975, +0.2172] SB |
| S-POS@30 | STRATEGIC | -1.1996 [-1.5153, -0.9180] SB | +0.3612 [+0.0986, +0.6472] SW | -1.7972 [-2.2609, -1.3948] SB |
| S-POS@50 | STRATEGIC | -3.8291 [-4.7635, -2.9764] SB | +0.5467 [+0.1073, +1.0056] SW | -5.0252 [-6.1549, -3.9713] SB |
| S-POS@80 | STRATEGIC | -9.5390 [-11.7064, -7.4891] SB | +0.7712 [+0.0880, +1.5057] SW | -10.7619 [-12.9488, -8.6791] SB |
| M1 crawl share | crawl | +0.0609 [+0.0392, +0.0862] SW | +0.0036 [-0.0000, +0.0082] ns | +0.0576 [+0.0364, +0.0823] SW |

## 3. The prediction next to the result

Card, written before any number (ESTIMATED): gate lever (a) at 55 %; given (a), **DOMINANT 10 %, MOVED 40 %, NO EFFECT 30 %, HARMS 20 %**, and the stated expectation *the clamp caps the tail, not the ordinary steps: curvature MAE X - Y separated-better at both seeds; M1 and L2 move less*.

| arm | outcome read | curvature MAE X - Y separated-better s0 / s1 | share of the Y - P10 excess closed by X - Y: curvature MAE, L2, M1, yaw-rate (s0 / s1) |
|---|---|---|---|
| DAG-OACC | **HARMS** | True / True | s0: curvature 164%, L2 90%, M1 -10%, yaw-rate n/a (Y - P10 <= 0); s1: curvature 127%, L2 91%, M1 -13%, yaw-rate 410% |
| SOUP | **HARMS** | True / True | s0: curvature 135%, L2 89%, M1 -1%, yaw-rate 737%; s1: curvature 133%, L2 94%, M1 -6%, yaw-rate 511% |

## 4. The card's committed next arm for the outcome (Rule Zero)

* **DAG-OACC, HARMS**: the kappa-only projection (`a_max` = inf) as a further flag value of the same tool, to attribute the harm to the accel clamp or the curvature clamp.
* **SOUP, HARMS**: the kappa-only projection (`a_max` = inf) as a further flag value of the same tool, to attribute the harm to the accel clamp or the curvature clamp.

Each next arm is a zero-training flag value of the same tool and needs its own frozen card before any rollout; none is launched by this read.

## 5. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)

* **CTRL_s0**: X {"N": 2363, "E": -0.0353, "partition_sum_minus_E": -0.0, "n_dip_windows": 31, "n_highk_steps": 2029, "n_highk_steps_masked_in_row": 1612, "highk_share_from_steps_under_2mps": 1.0765} | Y {"N": 2364, "E": 0.0578, "partition_sum_minus_E": 0.0, "n_dip_windows": 46, "n_highk_steps": 4613, "n_highk_steps_masked_in_row": 3411, "highk_share_from_steps_under_2mps": 0.5549}
* **CTRL_s1**: X {"N": 2365, "E": -0.0333, "partition_sum_minus_E": 0.0, "n_dip_windows": 53, "n_highk_steps": 2232, "n_highk_steps_masked_in_row": 1783, "highk_share_from_steps_under_2mps": 1.081} | Y {"N": 2365, "E": 0.1212, "partition_sum_minus_E": 0.0, "n_dip_windows": 63, "n_highk_steps": 4962, "n_highk_steps_masked_in_row": 3704, "highk_share_from_steps_under_2mps": 0.5851}
* **SOUP_s0**: X {"N": 2364, "E": -0.0307, "partition_sum_minus_E": -0.0, "n_dip_windows": 64, "n_highk_steps": 2463, "n_highk_steps_masked_in_row": 2027, "highk_share_from_steps_under_2mps": 1.1502} | Y {"N": 2364, "E": 0.0919, "partition_sum_minus_E": 0.0, "n_dip_windows": 74, "n_highk_steps": 5468, "n_highk_steps_masked_in_row": 4202, "highk_share_from_steps_under_2mps": 0.534}
* **SOUP_s1**: X {"N": 2364, "E": -0.0336, "partition_sum_minus_E": 0.0, "n_dip_windows": 42, "n_highk_steps": 2356, "n_highk_steps_masked_in_row": 1892, "highk_share_from_steps_under_2mps": 1.0672} | Y {"N": 2365, "E": 0.1067, "partition_sum_minus_E": 0.0, "n_dip_windows": 55, "n_highk_steps": 5282, "n_highk_steps_masked_in_row": 3936, "highk_share_from_steps_under_2mps": 0.618}

Gloss (from `gate_kappa.partition`): N = windows finite on both sides; E = the curvature MAE point (X - P10 for X, Y - P10 for Y); n_dip_windows = windows with an unwarranted speed dip; n_highk_steps = steps with |kappa| of the executed path above 0.2 1/m; highk_share_from_steps_under_2mps above 1 means those steps carry more than the HIGHK total (other HIGHK steps carry the opposite sign).

## 5b. Reading (HYPOTHESIS, not tested here; the committed next arm discriminates it)

* The box closes the curvature, L2 curvature and yaw-rate excess (section 3) and leaves M1 crawl share separated-worse, moving it the wrong way. The rows it harms are position and heading rows (heading MAE, S-POS, along-track, ADE6 at DAG-OACC s1): the projection re-integrates positions.
* HYPOTHESIS: the acceleration clamp (`a_max` 4.0, which holds a plan's speed down when it launches above the entry speed; GATE_RESULT.md tool smoke) drives M1, along-track and ADE6; the curvature clamp drives curvature, yaw-rate, heading and S-POS. The kappa-only projection (`a_max` = inf) is the arm that separates them.

## 6. Limits

* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); a separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`; replicate arm not run).
* X - P10 and Y - P10 use the banked P10 side (SMOOTH-T1's reference choice); the fresh floor differs from it by the KF deviation above. X - Y does not use P10.
* Tier T1-kin: executed state, logged cameras, own plan applied open-loop in the scene; it says nothing about re-perception.
* Adoption into any K-1 / WM-1 claim is not decided here; the Master Mind registers the claim row.
