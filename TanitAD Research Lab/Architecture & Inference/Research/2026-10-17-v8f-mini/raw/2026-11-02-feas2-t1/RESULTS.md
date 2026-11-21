# RESULTS: H-FEAS2-T1-1, the box split into its clamps after S (kbox / abox / kabox), DAG-OACC and SOUP at s0 / s1

Card `PREREG_FROZEN.md` sha256 `ecf9bb5d6b48595086656a470f69534d22d5be5d9882573504c1ab4df4ac77b2` (frozen 2026-11-02T16:43:48+0100). Reader `code/feas2_t1_read.py` under KAPPA-T1's `READER_FIX_FLOOR.md` (extractor guard 1.0 m; the floor / window check KF runs against the reused fresh Y). Read clock 2026-11-02T17:20:31+0100. Raw `raw/FEAS2_T1.json`, per-window arrays `raw/arrays/*.npz`. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class **MEASURED (ours; artifacts in MANIFEST.txt)**.
Estimator of every interval below: **paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. One training run per seed and a deterministic planner: the intervals answer *another draw of episodes* only, not *another training run* (`H-ESTIM-SEED-1`) nor *another inference run*.

## 0. Outcome (the card's rule, precedence DOMINANT > HARMS > MOVED > NO EFFECT)

| lever | arm | outcome | seed 0: D1 / D2 / D3 | seed 1: D1 / D2 / D3 | seed 0: MOVED / HARMS | seed 1: MOVED / HARMS |
|---|---|---|---|---|---|---|
| kbox | DAG-OACC | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| kbox | SOUP | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| abox | DAG-OACC | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| abox | SOUP | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| kabox | DAG-OACC | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| kabox | SOUP | **HARMS** | True / False / True | True / False / True | False / True | False / True |

Headline arm DAG-OACC; SOUP stated separately. `moved_and_harms` flags (HARMS precedence hides a MOVED condition): kbox/DAG-OACC False, kbox/SOUP False, abox/DAG-OACC False, abox/SOUP False, kabox/DAG-OACC False, kabox/SOUP False.

**MEASURED headline.** 6 of 6 lever x arm outcomes read HARMS (DOMINANT, MOVED and NO EFFECT: 0). D1 (ADE6 X - P10) holds at 12 of 12 cells, D3 at 12 of 12, and D2 fails at 12 of 12.

## 1. What triggers HARMS and what blocks DOMINANT, per lever, arm and seed (generated from the rows)

* **kbox DAG-OACC s0**: HARMS triggers: S-POS@30 X - Y +0.5048 SW; S-POS@50 X - Y +0.7883 SW; S-POS@80 X - Y +1.0915 SW; heading MAE X - Y +1.7413 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kbox DAG-OACC s1**: HARMS triggers: ADE6 X - Y +0.0753 > guard +0.0604; ADE6 X - Y +0.0753 SW; S-POS@30 X - Y +0.2572 SW; along-track MAE X - Y +0.0718 SW; heading MAE X - Y +1.3432 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kbox SOUP s0**: HARMS triggers: S-POS@30 X - Y +0.6854 SW; S-POS@50 X - Y +1.1340 SW; S-POS@80 X - Y +1.6440 SW; along-track MAE X - Y +0.0492 SW; heading MAE X - Y +1.6052 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kbox SOUP s1**: HARMS triggers: ADE6 X - Y +0.0446 SW; S-POS@30 X - Y +0.3731 SW; S-POS@50 X - Y +0.5627 SW; S-POS@80 X - Y +0.7949 SW; along-track MAE X - Y +0.0502 SW; heading MAE X - Y +1.4220 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **abox DAG-OACC s0**: HARMS triggers: S-POS@30 X - Y +0.4806 SW; S-POS@50 X - Y +0.7708 SW; S-POS@80 X - Y +1.1173 SW; heading MAE X - Y +1.8684 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE. Crawl rows X - Y separated-better: none; not: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share.
* **abox DAG-OACC s1**: HARMS triggers: S-POS@30 X - Y +0.4436 SW; S-POS@50 X - Y +0.7233 SW; S-POS@80 X - Y +1.0803 SW; heading MAE X - Y +0.9396 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE. Crawl rows X - Y separated-better: none; not: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share.
* **abox SOUP s0**: HARMS triggers: S-POS@30 X - Y +0.6141 SW; S-POS@50 X - Y +1.0038 SW; S-POS@80 X - Y +1.4689 SW; heading MAE X - Y +1.7787 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE. Crawl rows X - Y separated-better: none; not: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share.
* **abox SOUP s1**: HARMS triggers: S-POS@30 X - Y +0.2217 SW; S-POS@50 X - Y +0.3506 SW; S-POS@80 X - Y +0.4061 SW; TAC-LAT X - Y -0.0096 SW; heading MAE X - Y +1.2698 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE. Crawl rows X - Y separated-better: none; not: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share.
* **kabox DAG-OACC s0**: HARMS triggers: S-POS@30 X - Y +0.5058 SW; S-POS@50 X - Y +0.7966 SW; S-POS@80 X - Y +1.1223 SW; heading MAE X - Y +1.7364 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kabox DAG-OACC s1**: HARMS triggers: ADE6 X - Y +0.0356 SW; S-POS@30 X - Y +0.2660 SW; along-track MAE X - Y +0.0357 SW; heading MAE X - Y +1.1877 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kabox SOUP s0**: HARMS triggers: S-POS@30 X - Y +0.6577 SW; S-POS@50 X - Y +1.1096 SW; S-POS@80 X - Y +1.6220 SW; heading MAE X - Y +1.4556 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kabox SOUP s1**: HARMS triggers: S-POS@30 X - Y +0.3621 SW; S-POS@50 X - Y +0.5431 SW; S-POS@80 X - Y +0.7728 SW; heading MAE X - Y +1.0169 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.

### D2 failing rows, count per lever (4 cells: DAG-OACC s0 / s1, SOUP s0 / s1)

| lever | M1 crawl share | curvature MAE | L2 curvature MAE (GT>=5.2) | yaw-rate MAE | any other row |
|---|---|---|---|---|---|
| kbox | 4 / 4 | 0 / 4 | 0 / 4 | 0 / 4 | none |
| abox | 4 / 4 | 4 / 4 | 4 / 4 | 3 / 4 | none |
| kabox | 4 / 4 | 0 / 4 | 1 / 4 | 0 / 4 | none |

## 2. Known answers and controls (all six groups, computed before any row)

* **KY_reused_Y_equals_frozen_pins**: PASS
* **KA1_fresh_rerun_equals_reused_Y**: PASS
* **KA2_issued_plan_is_lever_of_S**: PASS
* **KF_floor_window_check_vs_fresh_Y**: PASS
* **K3_Y_banked_minus_P10_reproduces_SOUP_READ**: PASS
* **KB1_P10_side_equals_landed_banked_b**: PASS

**KY** (the reused Y files equal the pins of `FROZEN_SHA.txt`): 8 / 8 files hash-equal.

**KA1** (the fresh rerun `SREP2_DAGOACC_s0` equals the reused `SREP_DAGOACC_s0`; tolerance 1e-4 m): dump bit-identical True (max abs dev 0.000e+00 m); ticks bit-identical True (max abs dev 0.000e+00); differing keys none; key-set difference none.

**KA2** (X's issued `plan` = lever(P + SG(plan_raw - P), tick v0); tick-0 `plan_raw` = Y's; floor = Y's; tolerance 1e-4 m):

| run | plan_s vs P+SG (m) | plan vs lever(plan_s) (m) | tick-0 plan_raw vs Y (m) | floor X vs Y (m) | lever moved the issued plan by up to (m) |
|---|---|---|---|---|---|
| CTRL_s0:kbox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1231.289 |
| CTRL_s0:abox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 2141.024 |
| CTRL_s0:kabox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 2014.771 |
| CTRL_s1:kbox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 5524.202 |
| CTRL_s1:abox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 355.723 |
| CTRL_s1:kabox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 127.176 |
| SOUP_s0:kbox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 948.237 |
| SOUP_s0:abox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1552.326 |
| SOUP_s0:kabox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1461.505 |
| SOUP_s1:kbox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 270.261 |
| SOUP_s1:abox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 231.344 |
| SOUP_s1:kabox | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 167.839 |

The last KA2 column is the maximum over every tick and knot of |plan - plan_s|; the pre-check read the same quantity on the far knots (105-227 m on unexecuted knots 11+), so only executed knots 1-10 reach the rows.

**KF** (floor / window check against the fresh Y; gross guard 1.0 m against the banked CTRL):

| run | windows X = Y | windows Y = banked CTRL | floor X vs Y (m) | floor Y vs banked CTRL max / mean (m) |
|---|---|---|---|---|
| CTRL_s0:kbox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s0:abox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s0:kabox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s1:kbox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s1:abox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s1:kabox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s0:kbox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s0:abox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s0:kabox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s1:kbox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s1:abox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s1:kabox | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |

**K3**: Y banked - P10 from the landed arrays reproduces the pinned `SOUP_READ.json` on 76 / 76 rows (tolerance 6e-05). **KB1**: the P10 side of every extracted X and Y equals the landed banked `b` arrays on the non-S-POS rows: 16 / 16 runs pass.

**KA3** (`tests/test_v8f_t1_feas2.py`, 16 passed, `logs/pytest_feas2.log`): run when the tool was built; not re-run in this read.

## 3. Read A against P10 (D1 / D2 / D3) and X - Y, per lever, arm and seed

The 19 rows of `extract_smooth.py`; every cell is `point [lo, hi] mark`. D1 bar: ADE6 X - P10 separated-better with point <= -0.2415 m. HARMS guard: ADE6 X - Y point <= +0.0604 m.

### kbox DAG-OACC s0

* **PRIMARY ADE6 (m)**: X - P10 -1.2326 [-1.4374, -1.0372] SB (D1: **True**); X - Y +0.0230 [-0.0196, +0.0695] ns (guard: **True**); Y - P10 -1.2556 [-1.4503, -1.0678] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 16 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, heading MAE. Rows separated-worse on X - Y, all 19: 5.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2326 [-1.4374, -1.0372] SB | +0.0230 [-0.0196, +0.0695] ns | -1.2556 [-1.4503, -1.0678] SB |
| speed MAE | LONGITUDINAL | -0.9208 [-1.1551, -0.7168] SB | -0.0878 [-0.1464, -0.0424] SB | -0.8330 [-1.0377, -0.6583] SB |
| along-track MAE | LONGITUDINAL | -0.6819 [-0.8527, -0.5174] SB | +0.0373 [-0.0044, +0.0799] ns | -0.7192 [-0.8806, -0.5627] SB |
| accel MAE | LONGITUDINAL | -1.2072 [-1.6621, -0.8229] SB | -0.2681 [-0.3992, -0.1646] SB | -0.9390 [-1.3078, -0.6221] SB |
| DK-VIOL | LONGITUDINAL | +0.0235 [-0.0119, +0.0695] ns | +0.0067 [-0.0032, +0.0215] ns | +0.0168 [-0.0157, +0.0625] ns |
| DK-HW | LONGITUDINAL | -0.3825 [-0.7721, -0.0642] SB | -0.0343 [-0.0895, +0.0279] ns | -0.2604 [-0.6426, +0.0675] ns |
| DK-TG | LONGITUDINAL | -0.1053 [-0.1727, -0.0512] SB | -0.0240 [-0.0579, +0.0056] ns | -0.0588 [-0.1226, -0.0011] SB |
| heading MAE | LATERAL | -1.7108 [-2.8338, -0.3571] SB | +1.7413 [+0.4096, +3.3880] SW | -3.4520 [-4.2140, -2.7529] SB |
| yaw-rate MAE | LATERAL | -12.7263 [-17.9808, -8.2574] SB | -11.4947 [-15.5472, -7.9321] SB | -1.3762 [-3.9121, +1.0259] ns |
| curvature MAE | LATERAL | -0.0352 [-0.0516, -0.0212] SB | -0.0948 [-0.1326, -0.0632] SB | +0.0578 [+0.0364, +0.0821] SW |
| cross-track MAE | LATERAL | -1.0126 [-1.1669, -0.8711] SB | -0.0117 [-0.0487, +0.0279] ns | -1.0009 [-1.1468, -0.8671] SB |
| L1 lateral-accel MAE | LATERAL | -3.5531 [-5.1037, -2.2205] SB | -0.8652 [-1.2228, -0.5691] SB | -2.7327 [-4.0998, -1.5826] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0012 [-0.0041, +0.0019] ns | -0.0293 [-0.0537, -0.0138] SB | +0.0325 [+0.0135, +0.0590] SW |
| TAC-LAT | TACTICAL | +0.2172 [+0.1792, +0.2577] SB | +0.0133 [+0.0013, +0.0258] SB | +0.2038 [+0.1639, +0.2450] SB |
| TAC-LON | TACTICAL | +0.1888 [+0.1240, +0.2531] SB | +0.0138 [+0.0025, +0.0296] SB | +0.1751 [+0.1138, +0.2389] SB |
| S-POS@30 | STRATEGIC | -0.8427 [-1.1531, -0.5519] SB | +0.5048 [+0.1671, +0.9556] SW | -1.6443 [-2.1654, -1.1864] SB |
| S-POS@50 | STRATEGIC | -3.0202 [-3.9329, -2.1574] SB | +0.7883 [+0.1822, +1.5861] SW | -4.5745 [-5.8089, -3.5098] SB |
| S-POS@80 | STRATEGIC | -7.6927 [-9.7151, -5.7095] SB | +1.0915 [+0.0382, +2.4018] SW | -9.8284 [-12.1173, -7.7691] SB |
| M1 crawl share | crawl | +0.0515 [+0.0330, +0.0735] SW | +0.0040 [+0.0013, +0.0073] SW | +0.0483 [+0.0302, +0.0698] SW |

### kbox DAG-OACC s1

* **PRIMARY ADE6 (m)**: X - P10 -1.1150 [-1.3221, -0.9058] SB (D1: **True**); X - Y +0.0753 [+0.0352, +0.1225] SW (guard: **False**); Y - P10 -1.1903 [-1.3986, -0.9782] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 16 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, along-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 4.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1150 [-1.3221, -0.9058] SB | +0.0753 [+0.0352, +0.1225] SW | -1.1903 [-1.3986, -0.9782] SB |
| speed MAE | LONGITUDINAL | -0.8451 [-1.0720, -0.6423] SB | -0.0310 [-0.0545, -0.0096] SB | -0.8141 [-1.0377, -0.6191] SB |
| along-track MAE | LONGITUDINAL | -0.5584 [-0.7445, -0.3814] SB | +0.0718 [+0.0284, +0.1175] SW | -0.6302 [-0.8196, -0.4510] SB |
| accel MAE | LONGITUDINAL | -1.1745 [-1.6212, -0.7983] SB | -0.2025 [-0.2871, -0.1302] SB | -0.9719 [-1.3431, -0.6572] SB |
| DK-VIOL | LONGITUDINAL | -0.0034 [-0.0247, +0.0179] ns | -0.0034 [-0.0165, +0.0114] ns | +0.0000 [-0.0196, +0.0212] ns |
| DK-HW | LONGITUDINAL | -0.5869 [-0.9903, -0.2837] SB | -0.0817 [-0.1694, -0.0025] SB | -0.5171 [-0.9057, -0.2118] SB |
| DK-TG | LONGITUDINAL | -0.1326 [-0.2059, -0.0752] SB | -0.0488 [-0.1277, +0.0091] ns | -0.0869 [-0.1542, -0.0278] SB |
| heading MAE | LATERAL | -1.5504 [-2.6097, -0.3281] SB | +1.3432 [+0.2024, +2.6876] SW | -2.8936 [-3.6062, -2.1968] SB |
| yaw-rate MAE | LATERAL | -12.6408 [-18.0235, -8.1479] SB | -16.7821 [-23.3655, -11.0983] SB | +4.1412 [+1.1665, +7.3821] SW |
| curvature MAE | LATERAL | -0.0335 [-0.0498, -0.0193] SB | -0.1547 [-0.2153, -0.1014] SB | +0.1212 [+0.0757, +0.1729] SW |
| cross-track MAE | LATERAL | -0.9949 [-1.1473, -0.8494] SB | +0.0268 [-0.0000, +0.0590] ns | -1.0217 [-1.1686, -0.8803] SB |
| L1 lateral-accel MAE | LATERAL | -3.5119 [-5.0666, -2.1902] SB | -0.9037 [-1.2826, -0.5855] SB | -2.6082 [-3.8410, -1.5151] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0010 [-0.0025, +0.0052] ns | -0.0460 [-0.0705, -0.0256] SB | +0.0505 [+0.0269, +0.0791] SW |
| TAC-LAT | TACTICAL | +0.2088 [+0.1685, +0.2495] SB | -0.0038 [-0.0150, +0.0067] ns | +0.2126 [+0.1709, +0.2540] SB |
| TAC-LON | TACTICAL | +0.1634 [+0.1017, +0.2228] SB | -0.0004 [-0.0071, +0.0067] ns | +0.1638 [+0.1021, +0.2225] SB |
| S-POS@30 | STRATEGIC | -1.2180 [-1.5273, -0.9423] SB | +0.2572 [+0.0019, +0.5411] SW | -1.7416 [-2.2209, -1.3273] SB |
| S-POS@50 | STRATEGIC | -3.8355 [-4.7300, -3.0325] SB | +0.3925 [-0.0419, +0.8639] ns | -4.7600 [-5.8592, -3.7484] SB |
| S-POS@80 | STRATEGIC | -9.4194 [-11.4916, -7.5032] SB | +0.5132 [-0.2056, +1.2478] ns | -10.2999 [-12.3713, -8.3307] SB |
| M1 crawl share | crawl | +0.0592 [+0.0367, +0.0858] SW | +0.0036 [-0.0001, +0.0082] ns | +0.0565 [+0.0347, +0.0829] SW |

### kbox SOUP s0

* **PRIMARY ADE6 (m)**: X - P10 -1.1591 [-1.3760, -0.9503] SB (D1: **True**); X - Y +0.0337 [-0.0098, +0.0797] ns (guard: **True**); Y - P10 -1.1928 [-1.4078, -0.9880] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 16 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, along-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 5.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1591 [-1.3760, -0.9503] SB | +0.0337 [-0.0098, +0.0797] ns | -1.1928 [-1.4078, -0.9880] SB |
| speed MAE | LONGITUDINAL | -0.8439 [-1.0699, -0.6432] SB | -0.0622 [-0.0975, -0.0302] SB | -0.7817 [-0.9919, -0.5956] SB |
| along-track MAE | LONGITUDINAL | -0.5799 [-0.7707, -0.4013] SB | +0.0492 [+0.0008, +0.0975] SW | -0.6291 [-0.8126, -0.4582] SB |
| accel MAE | LONGITUDINAL | -1.0930 [-1.5283, -0.7213] SB | -0.2139 [-0.2965, -0.1412] SB | -0.8791 [-1.2581, -0.5583] SB |
| DK-VIOL | LONGITUDINAL | +0.0084 [-0.0207, +0.0392] ns | +0.0017 [-0.0083, +0.0154] ns | +0.0067 [-0.0196, +0.0346] ns |
| DK-HW | LONGITUDINAL | -0.3769 [-0.7541, -0.0915] SB | -0.0546 [-0.1368, +0.0128] ns | -0.2040 [-0.5885, +0.1367] ns |
| DK-TG | LONGITUDINAL | -0.0995 [-0.1614, -0.0489] SB | -0.0434 [-0.1112, +0.0074] ns | -0.0560 [-0.1173, -0.0018] SB |
| heading MAE | LATERAL | -1.3562 [-2.4825, -0.0380] SB | +1.6052 [+0.2657, +3.2017] SW | -2.9614 [-3.7496, -2.1787] SB |
| yaw-rate MAE | LATERAL | -12.1367 [-17.3481, -7.6514] SB | -14.1158 [-19.0073, -9.9389] SB | +1.9270 [-1.0592, +5.0043] ns |
| curvature MAE | LATERAL | -0.0309 [-0.0472, -0.0167] SB | -0.1247 [-0.1710, -0.0847] SB | +0.0919 [+0.0584, +0.1301] SW |
| cross-track MAE | LATERAL | -1.0403 [-1.1960, -0.8975] SB | -0.0137 [-0.0480, +0.0232] ns | -1.0265 [-1.1729, -0.8943] SB |
| L1 lateral-accel MAE | LATERAL | -3.5542 [-5.1064, -2.2364] SB | -0.9124 [-1.2987, -0.6068] SB | -2.6420 [-4.0010, -1.5149] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0024 [-0.0016, +0.0068] ns | -0.0559 [-0.0973, -0.0258] SB | +0.0628 [+0.0279, +0.1076] SW |
| TAC-LAT | TACTICAL | +0.2013 [+0.1587, +0.2422] SB | +0.0167 [+0.0029, +0.0304] SB | +0.1847 [+0.1417, +0.2274] SB |
| TAC-LON | TACTICAL | +0.1530 [+0.0838, +0.2232] SB | +0.0063 [-0.0025, +0.0154] ns | +0.1467 [+0.0792, +0.2138] SB |
| S-POS@30 | STRATEGIC | -0.8671 [-1.1543, -0.5993] SB | +0.6854 [+0.2773, +1.1672] SW | -1.6062 [-2.1525, -1.1336] SB |
| S-POS@50 | STRATEGIC | -3.0941 [-3.9597, -2.2861] SB | +1.1340 [+0.4064, +1.9736] SW | -4.5825 [-5.8449, -3.4774] SB |
| S-POS@80 | STRATEGIC | -7.9972 [-9.9400, -6.1691] SB | +1.6440 [+0.5116, +2.9983] SW | -10.1918 [-12.6081, -8.0751] SB |
| M1 crawl share | crawl | +0.0648 [+0.0426, +0.0902] SW | -0.0006 [-0.0043, +0.0025] ns | +0.0666 [+0.0428, +0.0927] SW |

### kbox SOUP s1

* **PRIMARY ADE6 (m)**: X - P10 -1.1449 [-1.3538, -0.9384] SB (D1: **True**); X - Y +0.0446 [+0.0042, +0.0925] SW (guard: **True**); Y - P10 -1.1895 [-1.3938, -0.9847] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 16 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, S-POS@50, S-POS@80, along-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 6.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1449 [-1.3538, -0.9384] SB | +0.0446 [+0.0042, +0.0925] SW | -1.1895 [-1.3938, -0.9847] SB |
| speed MAE | LONGITUDINAL | -0.8516 [-1.0816, -0.6478] SB | -0.0779 [-0.1182, -0.0389] SB | -0.7737 [-0.9864, -0.5882] SB |
| along-track MAE | LONGITUDINAL | -0.5758 [-0.7650, -0.3974] SB | +0.0502 [+0.0129, +0.0924] SW | -0.6260 [-0.8049, -0.4497] SB |
| accel MAE | LONGITUDINAL | -1.1644 [-1.6014, -0.7912] SB | -0.3680 [-0.5320, -0.2316] SB | -0.7964 [-1.1397, -0.4913] SB |
| DK-VIOL | LONGITUDINAL | -0.0034 [-0.0249, +0.0185] ns | +0.0034 [-0.0035, +0.0126] ns | -0.0067 [-0.0265, +0.0122] ns |
| DK-HW | LONGITUDINAL | -0.5971 [-0.9772, -0.2859] SB | -0.0303 [-0.1048, +0.0440] ns | -0.5481 [-0.9299, -0.2218] SB |
| DK-TG | LONGITUDINAL | -0.1124 [-0.1838, -0.0511] SB | -0.0232 [-0.0944, +0.0443] ns | -0.0863 [-0.1467, -0.0356] SB |
| heading MAE | LATERAL | -1.5226 [-2.5137, -0.3965] SB | +1.4220 [+0.3766, +2.6896] SW | -2.9491 [-3.6329, -2.3131] SB |
| yaw-rate MAE | LATERAL | -12.4453 [-17.8313, -7.9681] SB | -15.6150 [-21.4489, -10.4684] SB | +3.0944 [+0.2216, +6.2120] SW |
| curvature MAE | LATERAL | -0.0339 [-0.0506, -0.0194] SB | -0.1426 [-0.2001, -0.0927] SB | +0.1067 [+0.0666, +0.1533] SW |
| cross-track MAE | LATERAL | -1.0190 [-1.1701, -0.8716] SB | -0.0001 [-0.0313, +0.0365] ns | -1.0189 [-1.1607, -0.8829] SB |
| L1 lateral-accel MAE | LATERAL | -3.4888 [-5.0529, -2.1675] SB | -0.9157 [-1.2992, -0.5663] SB | -2.5744 [-3.8473, -1.4601] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0016 [-0.0020, +0.0059] ns | -0.0429 [-0.0679, -0.0229] SB | +0.0454 [+0.0224, +0.0739] SW |
| TAC-LAT | TACTICAL | +0.2080 [+0.1676, +0.2494] SB | -0.0058 [-0.0188, +0.0067] ns | +0.2138 [+0.1747, +0.2548] SB |
| TAC-LON | TACTICAL | +0.1663 [+0.1047, +0.2269] SB | +0.0079 [+0.0004, +0.0154] SB | +0.1584 [+0.0975, +0.2172] SB |
| S-POS@30 | STRATEGIC | -1.2196 [-1.5390, -0.9296] SB | +0.3731 [+0.1077, +0.6610] SW | -1.7972 [-2.2609, -1.3948] SB |
| S-POS@50 | STRATEGIC | -3.8791 [-4.8291, -3.0300] SB | +0.5627 [+0.1117, +1.0290] SW | -5.0252 [-6.1549, -3.9713] SB |
| S-POS@80 | STRATEGIC | -9.6079 [-11.7597, -7.5811] SB | +0.7949 [+0.0859, +1.5354] SW | -10.7619 [-12.9488, -8.6791] SB |
| M1 crawl share | crawl | +0.0597 [+0.0379, +0.0848] SW | +0.0024 [-0.0020, +0.0078] ns | +0.0576 [+0.0364, +0.0823] SW |

### abox DAG-OACC s0

* **PRIMARY ADE6 (m)**: X - P10 -1.2938 [-1.5066, -1.0948] SB (D1: **True**); X - Y -0.0382 [-0.0763, -0.0055] SB (guard: **True**); Y - P10 -1.2556 [-1.4503, -1.0678] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 14 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': False, 'curvature MAE': False, 'L2 curvature MAE (GT>=5.2)': False, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, heading MAE. Rows separated-worse on X - Y, all 19: 7.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2938 [-1.5066, -1.0948] SB | -0.0382 [-0.0763, -0.0055] SB | -1.2556 [-1.4503, -1.0678] SB |
| speed MAE | LONGITUDINAL | -0.9435 [-1.1779, -0.7412] SB | -0.1105 [-0.1726, -0.0615] SB | -0.8330 [-1.0377, -0.6583] SB |
| along-track MAE | LONGITUDINAL | -0.7336 [-0.9058, -0.5666] SB | -0.0144 [-0.0422, +0.0128] ns | -0.7192 [-0.8806, -0.5627] SB |
| accel MAE | LONGITUDINAL | -1.2159 [-1.6679, -0.8396] SB | -0.2769 [-0.4114, -0.1717] SB | -0.9390 [-1.3078, -0.6221] SB |
| DK-VIOL | LONGITUDINAL | +0.0084 [-0.0186, +0.0401] ns | -0.0084 [-0.0235, +0.0000] ns | +0.0168 [-0.0157, +0.0625] ns |
| DK-HW | LONGITUDINAL | -0.3842 [-0.7534, -0.0701] SB | -0.0863 [-0.1694, -0.0175] SB | -0.2604 [-0.6426, +0.0675] ns |
| DK-TG | LONGITUDINAL | -0.0635 [-0.1293, -0.0028] SB | -0.0046 [-0.0149, +0.0060] ns | -0.0588 [-0.1226, -0.0011] SB |
| heading MAE | LATERAL | -1.5836 [-2.3698, -0.6929] SB | +1.8684 [+0.9526, +3.0070] SW | -3.4520 [-4.2140, -2.7529] SB |
| yaw-rate MAE | LATERAL | +0.6170 [-1.9005, +2.9465] ns | +1.8491 [+0.7702, +3.1321] SW | -1.3762 [-3.9121, +1.0259] ns |
| curvature MAE | LATERAL | +0.1105 [+0.0717, +0.1554] SW | +0.0525 [+0.0309, +0.0776] SW | +0.0578 [+0.0364, +0.0821] SW |
| cross-track MAE | LATERAL | -1.0383 [-1.1895, -0.9006] SB | -0.0373 [-0.0630, -0.0132] SB | -1.0009 [-1.1468, -0.8671] SB |
| L1 lateral-accel MAE | LATERAL | -3.1755 [-4.6005, -1.9419] SB | -0.4705 [-0.8212, -0.2072] SB | -2.7327 [-4.0998, -1.5826] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0467 [+0.0217, +0.0840] SW | +0.0106 [-0.0000, +0.0236] ns | +0.0325 [+0.0135, +0.0590] SW |
| TAC-LAT | TACTICAL | +0.2026 [+0.1628, +0.2439] SB | -0.0013 [-0.0079, +0.0054] ns | +0.2038 [+0.1639, +0.2450] SB |
| TAC-LON | TACTICAL | +0.1692 [+0.1055, +0.2310] SB | -0.0058 [-0.0204, +0.0079] ns | +0.1751 [+0.1138, +0.2389] SB |
| S-POS@30 | STRATEGIC | -1.3729 [-1.7954, -0.9995] SB | +0.4806 [+0.2299, +0.8102] SW | -1.6443 [-2.1654, -1.1864] SB |
| S-POS@50 | STRATEGIC | -4.0474 [-5.0672, -3.1047] SB | +0.7708 [+0.3419, +1.3413] SW | -4.5745 [-5.8089, -3.5098] SB |
| S-POS@80 | STRATEGIC | -8.9266 [-10.9163, -6.9861] SB | +1.1173 [+0.3913, +2.0495] SW | -9.8284 [-12.1173, -7.7691] SB |
| M1 crawl share | crawl | +0.0547 [+0.0356, +0.0774] SW | +0.0059 [+0.0028, +0.0095] SW | +0.0483 [+0.0302, +0.0698] SW |

### abox DAG-OACC s1

* **PRIMARY ADE6 (m)**: X - P10 -1.2204 [-1.4386, -0.9979] SB (D1: **True**); X - Y -0.0301 [-0.0524, -0.0099] SB (guard: **True**); Y - P10 -1.1903 [-1.3986, -0.9782] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 14 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': False, 'curvature MAE': False, 'L2 curvature MAE (GT>=5.2)': False, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, heading MAE. Rows separated-worse on X - Y, all 19: 6.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2204 [-1.4386, -0.9979] SB | -0.0301 [-0.0524, -0.0099] SB | -1.1903 [-1.3986, -0.9782] SB |
| speed MAE | LONGITUDINAL | -0.8931 [-1.1297, -0.6857] SB | -0.0790 [-0.1246, -0.0450] SB | -0.8141 [-1.0377, -0.6191] SB |
| along-track MAE | LONGITUDINAL | -0.6360 [-0.8256, -0.4540] SB | -0.0059 [-0.0230, +0.0120] ns | -0.6302 [-0.8196, -0.4510] SB |
| accel MAE | LONGITUDINAL | -1.2036 [-1.6570, -0.8270] SB | -0.2317 [-0.3287, -0.1504] SB | -0.9719 [-1.3431, -0.6572] SB |
| DK-VIOL | LONGITUDINAL | -0.0067 [-0.0252, +0.0108] ns | -0.0067 [-0.0174, +0.0017] ns | +0.0000 [-0.0196, +0.0212] ns |
| DK-HW | LONGITUDINAL | -0.5415 [-0.9303, -0.2280] SB | -0.0567 [-0.1046, -0.0179] SB | -0.5171 [-0.9057, -0.2118] SB |
| DK-TG | LONGITUDINAL | -0.0941 [-0.1609, -0.0346] SB | -0.0044 [-0.0151, +0.0056] ns | -0.0869 [-0.1542, -0.0278] SB |
| heading MAE | LATERAL | -1.9540 [-2.6541, -1.2346] SB | +0.9396 [+0.4948, +1.5201] SW | -2.8936 [-3.6062, -2.1968] SB |
| yaw-rate MAE | LATERAL | +3.4531 [+0.9391, +6.4218] SW | -0.6881 [-1.5787, +0.1091] ns | +4.1412 [+1.1665, +7.3821] SW |
| curvature MAE | LATERAL | +0.1500 [+0.0956, +0.2157] SW | +0.0288 [+0.0127, +0.0488] SW | +0.1212 [+0.0757, +0.1729] SW |
| cross-track MAE | LATERAL | -1.0555 [-1.2030, -0.9089] SB | -0.0338 [-0.0517, -0.0187] SB | -1.0217 [-1.1686, -0.8803] SB |
| L1 lateral-accel MAE | LATERAL | -3.1408 [-4.5454, -1.9166] SB | -0.5326 [-0.9082, -0.2465] SB | -2.6082 [-3.8410, -1.5151] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0555 [+0.0317, +0.0850] SW | +0.0050 [-0.0038, +0.0137] ns | +0.0505 [+0.0269, +0.0791] SW |
| TAC-LAT | TACTICAL | +0.2109 [+0.1693, +0.2523] SB | -0.0017 [-0.0083, +0.0050] ns | +0.2126 [+0.1709, +0.2540] SB |
| TAC-LON | TACTICAL | +0.1651 [+0.1025, +0.2236] SB | +0.0013 [-0.0075, +0.0104] ns | +0.1638 [+0.1021, +0.2225] SB |
| S-POS@30 | STRATEGIC | -1.5140 [-1.8875, -1.1716] SB | +0.4436 [+0.2011, +0.7400] SW | -1.7416 [-2.2209, -1.3273] SB |
| S-POS@50 | STRATEGIC | -4.2785 [-5.2264, -3.3797] SB | +0.7233 [+0.3286, +1.2032] SW | -4.7600 [-5.8592, -3.7484] SB |
| S-POS@80 | STRATEGIC | -9.5665 [-11.6092, -7.5931] SB | +1.0803 [+0.4656, +1.8325] SW | -10.2999 [-12.3713, -8.3307] SB |
| M1 crawl share | crawl | +0.0641 [+0.0410, +0.0918] SW | +0.0076 [+0.0035, +0.0123] SW | +0.0565 [+0.0347, +0.0829] SW |

### abox SOUP s0

* **PRIMARY ADE6 (m)**: X - P10 -1.2515 [-1.4767, -1.0319] SB (D1: **True**); X - Y -0.0586 [-0.0991, -0.0225] SB (guard: **True**); Y - P10 -1.1928 [-1.4078, -0.9880] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 13 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': False, 'curvature MAE': False, 'L2 curvature MAE (GT>=5.2)': False, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, heading MAE. Rows separated-worse on X - Y, all 19: 8.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2515 [-1.4767, -1.0319] SB | -0.0586 [-0.0991, -0.0225] SB | -1.1928 [-1.4078, -0.9880] SB |
| speed MAE | LONGITUDINAL | -0.8958 [-1.1296, -0.6835] SB | -0.1141 [-0.1661, -0.0739] SB | -0.7817 [-0.9919, -0.5956] SB |
| along-track MAE | LONGITUDINAL | -0.6570 [-0.8508, -0.4764] SB | -0.0279 [-0.0604, -0.0001] SB | -0.6291 [-0.8126, -0.4582] SB |
| accel MAE | LONGITUDINAL | -1.1522 [-1.6059, -0.7708] SB | -0.2731 [-0.3801, -0.1890] SB | -0.8791 [-1.2581, -0.5583] SB |
| DK-VIOL | LONGITUDINAL | -0.0017 [-0.0272, +0.0234] ns | -0.0084 [-0.0173, +0.0000] ns | +0.0067 [-0.0196, +0.0346] ns |
| DK-HW | LONGITUDINAL | -0.3649 [-0.7338, -0.0617] SB | -0.0683 [-0.1466, +0.0013] ns | -0.2040 [-0.5885, +0.1367] ns |
| DK-TG | LONGITUDINAL | -0.0576 [-0.1267, +0.0155] ns | -0.0163 [-0.0452, +0.0022] ns | -0.0560 [-0.1173, -0.0018] SB |
| heading MAE | LATERAL | -1.1827 [-2.1248, -0.1420] SB | +1.7787 [+0.8951, +2.8226] SW | -2.9614 [-3.7496, -2.1787] SB |
| yaw-rate MAE | LATERAL | +4.5829 [+1.4733, +7.9125] SW | +2.6547 [+1.2460, +4.3492] SW | +1.9270 [-1.0592, +5.0043] ns |
| curvature MAE | LATERAL | +0.1562 [+0.1001, +0.2272] SW | +0.0643 [+0.0345, +0.1044] SW | +0.0919 [+0.0584, +0.1301] SW |
| cross-track MAE | LATERAL | -1.0768 [-1.2256, -0.9378] SB | -0.0503 [-0.0771, -0.0255] SB | -1.0265 [-1.1729, -0.8943] SB |
| L1 lateral-accel MAE | LATERAL | -3.0577 [-4.4783, -1.8254] SB | -0.4155 [-0.6785, -0.2103] SB | -2.6420 [-4.0010, -1.5149] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0848 [+0.0444, +0.1355] SW | +0.0202 [+0.0044, +0.0431] SW | +0.0628 [+0.0279, +0.1076] SW |
| TAC-LAT | TACTICAL | +0.1805 [+0.1378, +0.2236] SB | -0.0042 [-0.0104, +0.0017] ns | +0.1847 [+0.1417, +0.2274] SB |
| TAC-LON | TACTICAL | +0.1613 [+0.0930, +0.2319] SB | +0.0146 [+0.0071, +0.0234] SB | +0.1467 [+0.0792, +0.2138] SB |
| S-POS@30 | STRATEGIC | -1.2705 [-1.6757, -0.9201] SB | +0.6141 [+0.2790, +1.0357] SW | -1.6062 [-2.1525, -1.1336] SB |
| S-POS@50 | STRATEGIC | -3.9092 [-4.8883, -3.0376] SB | +1.0038 [+0.4374, +1.7077] SW | -4.5825 [-5.8449, -3.4774] SB |
| S-POS@80 | STRATEGIC | -9.0973 [-11.0979, -7.2384] SB | +1.4689 [+0.5773, +2.5737] SW | -10.1918 [-12.6081, -8.0751] SB |
| M1 crawl share | crawl | +0.0725 [+0.0482, +0.0987] SW | +0.0054 [+0.0021, +0.0092] SW | +0.0666 [+0.0428, +0.0927] SW |

### abox SOUP s1

* **PRIMARY ADE6 (m)**: X - P10 -1.2373 [-1.4504, -1.0243] SB (D1: **True**); X - Y -0.0478 [-0.0767, -0.0224] SB (guard: **True**); Y - P10 -1.1895 [-1.3938, -0.9847] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 14 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': False, 'curvature MAE': False, 'L2 curvature MAE (GT>=5.2)': False, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, TAC-LAT, heading MAE. Rows separated-worse on X - Y, all 19: 8.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2373 [-1.4504, -1.0243] SB | -0.0478 [-0.0767, -0.0224] SB | -1.1895 [-1.3938, -0.9847] SB |
| speed MAE | LONGITUDINAL | -0.8911 [-1.1232, -0.6846] SB | -0.1175 [-0.1709, -0.0746] SB | -0.7737 [-0.9864, -0.5882] SB |
| along-track MAE | LONGITUDINAL | -0.6453 [-0.8306, -0.4633] SB | -0.0193 [-0.0433, +0.0039] ns | -0.6260 [-0.8049, -0.4497] SB |
| accel MAE | LONGITUDINAL | -1.1994 [-1.6514, -0.8200] SB | -0.4030 [-0.5675, -0.2641] SB | -0.7964 [-1.1397, -0.4913] SB |
| DK-VIOL | LONGITUDINAL | -0.0101 [-0.0313, +0.0103] ns | -0.0034 [-0.0104, +0.0033] ns | -0.0067 [-0.0265, +0.0122] ns |
| DK-HW | LONGITUDINAL | -0.5307 [-0.9137, -0.2016] SB | -0.0514 [-0.0932, -0.0157] SB | -0.5481 [-0.9299, -0.2218] SB |
| DK-TG | LONGITUDINAL | -0.0987 [-0.1605, -0.0464] SB | -0.0131 [-0.0272, -0.0015] SB | -0.0863 [-0.1467, -0.0356] SB |
| heading MAE | LATERAL | -1.6793 [-2.4157, -0.9317] SB | +1.2698 [+0.6766, +1.9896] SW | -2.9491 [-3.6329, -2.3131] SB |
| yaw-rate MAE | LATERAL | +3.3559 [+0.7591, +6.2066] SW | +0.2155 [-0.7605, +1.2059] ns | +3.0944 [+0.2216, +6.2120] SW |
| curvature MAE | LATERAL | +0.1351 [+0.0860, +0.1940] SW | +0.0292 [+0.0128, +0.0485] SW | +0.1067 [+0.0666, +0.1533] SW |
| cross-track MAE | LATERAL | -1.0639 [-1.2083, -0.9206] SB | -0.0450 [-0.0641, -0.0256] SB | -1.0189 [-1.1607, -0.8829] SB |
| L1 lateral-accel MAE | LATERAL | -3.0565 [-4.4383, -1.8571] SB | -0.5086 [-0.8166, -0.2546] SB | -2.5744 [-3.8473, -1.4601] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0563 [+0.0317, +0.0870] SW | +0.0110 [+0.0012, +0.0224] SW | +0.0454 [+0.0224, +0.0739] SW |
| TAC-LAT | TACTICAL | +0.2043 [+0.1635, +0.2465] SB | -0.0096 [-0.0175, -0.0025] SW | +0.2138 [+0.1747, +0.2548] SB |
| TAC-LON | TACTICAL | +0.1622 [+0.1016, +0.2213] SB | +0.0038 [-0.0025, +0.0108] ns | +0.1584 [+0.0975, +0.2172] SB |
| S-POS@30 | STRATEGIC | -1.6251 [-2.0185, -1.2768] SB | +0.2217 [+0.0871, +0.3696] SW | -1.7972 [-2.2609, -1.3948] SB |
| S-POS@50 | STRATEGIC | -4.7554 [-5.8160, -3.7841] SB | +0.3506 [+0.1328, +0.5916] SW | -5.0252 [-6.1549, -3.9713] SB |
| S-POS@80 | STRATEGIC | -10.5711 [-12.7059, -8.5732] SB | +0.4061 [+0.0947, +0.7627] SW | -10.7619 [-12.9488, -8.6791] SB |
| M1 crawl share | crawl | +0.0640 [+0.0415, +0.0904] SW | +0.0064 [+0.0025, +0.0112] SW | +0.0576 [+0.0364, +0.0823] SW |

### kabox DAG-OACC s0

* **PRIMARY ADE6 (m)**: X - P10 -1.2576 [-1.4674, -1.0634] SB (D1: **True**); X - Y -0.0020 [-0.0402, +0.0379] ns (guard: **True**); Y - P10 -1.2556 [-1.4503, -1.0678] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 16 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, heading MAE. Rows separated-worse on X - Y, all 19: 5.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2576 [-1.4674, -1.0634] SB | -0.0020 [-0.0402, +0.0379] ns | -1.2556 [-1.4503, -1.0678] SB |
| speed MAE | LONGITUDINAL | -0.9512 [-1.1886, -0.7468] SB | -0.1183 [-0.1867, -0.0647] SB | -0.8330 [-1.0377, -0.6583] SB |
| along-track MAE | LONGITUDINAL | -0.7110 [-0.8846, -0.5477] SB | +0.0082 [-0.0276, +0.0444] ns | -0.7192 [-0.8806, -0.5627] SB |
| accel MAE | LONGITUDINAL | -1.2881 [-1.7628, -0.8899] SB | -0.3491 [-0.5080, -0.2247] SB | -0.9390 [-1.3078, -0.6221] SB |
| DK-VIOL | LONGITUDINAL | +0.0168 [-0.0121, +0.0507] ns | +0.0000 [-0.0178, +0.0182] ns | +0.0168 [-0.0157, +0.0625] ns |
| DK-HW | LONGITUDINAL | -0.4018 [-0.7890, -0.0911] SB | -0.0535 [-0.1158, +0.0155] ns | -0.2604 [-0.6426, +0.0675] ns |
| DK-TG | LONGITUDINAL | -0.1058 [-0.1725, -0.0518] SB | -0.0244 [-0.0576, +0.0048] ns | -0.0588 [-0.1226, -0.0011] SB |
| heading MAE | LATERAL | -1.7156 [-2.8610, -0.3251] SB | +1.7364 [+0.3783, +3.4017] SW | -3.4520 [-4.2140, -2.7529] SB |
| yaw-rate MAE | LATERAL | -12.3444 [-17.3584, -7.9688] SB | -11.1130 [-15.1266, -7.6631] SB | -1.3762 [-3.9121, +1.0259] ns |
| curvature MAE | LATERAL | -0.0317 [-0.0476, -0.0181] SB | -0.0914 [-0.1290, -0.0602] SB | +0.0578 [+0.0364, +0.0821] SW |
| cross-track MAE | LATERAL | -1.0210 [-1.1787, -0.8762] SB | -0.0200 [-0.0588, +0.0217] ns | -1.0009 [-1.1468, -0.8671] SB |
| L1 lateral-accel MAE | LATERAL | -3.6222 [-5.1446, -2.3050] SB | -0.9344 [-1.3558, -0.5869] SB | -2.7327 [-4.0998, -1.5826] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0010 [-0.0017, +0.0041] ns | -0.0271 [-0.0508, -0.0121] SB | +0.0325 [+0.0135, +0.0590] SW |
| TAC-LAT | TACTICAL | +0.2159 [+0.1764, +0.2572] SB | +0.0121 [-0.0017, +0.0258] ns | +0.2038 [+0.1639, +0.2450] SB |
| TAC-LON | TACTICAL | +0.1797 [+0.1143, +0.2440] SB | +0.0046 [-0.0121, +0.0225] ns | +0.1751 [+0.1138, +0.2389] SB |
| S-POS@30 | STRATEGIC | -0.8257 [-1.1368, -0.5319] SB | +0.5058 [+0.1681, +0.9528] SW | -1.6443 [-2.1654, -1.1864] SB |
| S-POS@50 | STRATEGIC | -2.9745 [-3.9111, -2.1111] SB | +0.7966 [+0.1854, +1.5945] SW | -4.5745 [-5.8089, -3.5098] SB |
| S-POS@80 | STRATEGIC | -7.6123 [-9.6397, -5.6498] SB | +1.1223 [+0.0658, +2.4472] SW | -9.8284 [-12.1173, -7.7691] SB |
| M1 crawl share | crawl | +0.0536 [+0.0345, +0.0760] SW | +0.0056 [+0.0025, +0.0093] SW | +0.0483 [+0.0302, +0.0698] SW |

### kabox DAG-OACC s1

* **PRIMARY ADE6 (m)**: X - P10 -1.1547 [-1.3643, -0.9420] SB (D1: **True**); X - Y +0.0356 [+0.0059, +0.0702] SW (guard: **True**); Y - P10 -1.1903 [-1.3986, -0.9782] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 16 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, along-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 5.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1547 [-1.3643, -0.9420] SB | +0.0356 [+0.0059, +0.0702] SW | -1.1903 [-1.3986, -0.9782] SB |
| speed MAE | LONGITUDINAL | -0.8922 [-1.1305, -0.6830] SB | -0.0781 [-0.1239, -0.0441] SB | -0.8141 [-1.0377, -0.6191] SB |
| along-track MAE | LONGITUDINAL | -0.5944 [-0.7794, -0.4171] SB | +0.0357 [+0.0067, +0.0668] SW | -0.6302 [-0.8196, -0.4510] SB |
| accel MAE | LONGITUDINAL | -1.2696 [-1.7448, -0.8723] SB | -0.2976 [-0.4091, -0.2003] SB | -0.9719 [-1.3431, -0.6572] SB |
| DK-VIOL | LONGITUDINAL | -0.0050 [-0.0255, +0.0154] ns | -0.0050 [-0.0187, +0.0100] ns | +0.0000 [-0.0196, +0.0212] ns |
| DK-HW | LONGITUDINAL | -0.5911 [-0.9970, -0.2862] SB | -0.0859 [-0.1734, -0.0032] SB | -0.5171 [-0.9057, -0.2118] SB |
| DK-TG | LONGITUDINAL | -0.1321 [-0.2057, -0.0741] SB | -0.0483 [-0.1281, +0.0102] ns | -0.0869 [-0.1542, -0.0278] SB |
| heading MAE | LATERAL | -1.7059 [-2.6902, -0.6428] SB | +1.1877 [+0.1603, +2.4098] SW | -2.8936 [-3.6062, -2.1968] SB |
| yaw-rate MAE | LATERAL | -12.1884 [-17.2471, -7.9512] SB | -16.3296 [-22.5344, -10.8565] SB | +4.1412 [+1.1665, +7.3821] SW |
| curvature MAE | LATERAL | -0.0276 [-0.0427, -0.0141] SB | -0.1487 [-0.2078, -0.0979] SB | +0.1212 [+0.0757, +0.1729] SW |
| cross-track MAE | LATERAL | -1.0103 [-1.1648, -0.8617] SB | +0.0114 [-0.0178, +0.0456] ns | -1.0217 [-1.1686, -0.8803] SB |
| L1 lateral-accel MAE | LATERAL | -3.6643 [-5.2179, -2.3245] SB | -1.0560 [-1.5741, -0.6352] SB | -2.6082 [-3.8410, -1.5151] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0023 [-0.0013, +0.0064] ns | -0.0448 [-0.0694, -0.0241] SB | +0.0505 [+0.0269, +0.0791] SW |
| TAC-LAT | TACTICAL | +0.2093 [+0.1685, +0.2504] SB | -0.0033 [-0.0142, +0.0067] ns | +0.2126 [+0.1709, +0.2540] SB |
| TAC-LON | TACTICAL | +0.1655 [+0.1037, +0.2247] SB | +0.0017 [-0.0058, +0.0100] ns | +0.1638 [+0.1021, +0.2225] SB |
| S-POS@30 | STRATEGIC | -1.1982 [-1.5106, -0.9192] SB | +0.2660 [+0.0077, +0.5500] SW | -1.7416 [-2.2209, -1.3273] SB |
| S-POS@50 | STRATEGIC | -3.7414 [-4.6443, -2.9357] SB | +0.4139 [-0.0154, +0.8899] ns | -4.7600 [-5.8592, -3.7484] SB |
| S-POS@80 | STRATEGIC | -9.2243 [-11.2540, -7.3199] SB | +0.5558 [-0.1499, +1.2979] ns | -10.2999 [-12.3713, -8.3307] SB |
| M1 crawl share | crawl | +0.0634 [+0.0401, +0.0913] SW | +0.0077 [+0.0032, +0.0131] SW | +0.0565 [+0.0347, +0.0829] SW |

### kabox SOUP s0

* **PRIMARY ADE6 (m)**: X - P10 -1.2111 [-1.4336, -0.9961] SB (D1: **True**); X - Y -0.0183 [-0.0608, +0.0254] ns (guard: **True**); Y - P10 -1.1928 [-1.4078, -0.9880] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 16 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, heading MAE. Rows separated-worse on X - Y, all 19: 5.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2111 [-1.4336, -0.9961] SB | -0.0183 [-0.0608, +0.0254] ns | -1.1928 [-1.4078, -0.9880] SB |
| speed MAE | LONGITUDINAL | -0.9016 [-1.1406, -0.6854] SB | -0.1199 [-0.1770, -0.0742] SB | -0.7817 [-0.9919, -0.5956] SB |
| along-track MAE | LONGITUDINAL | -0.6293 [-0.8177, -0.4520] SB | -0.0002 [-0.0394, +0.0384] ns | -0.6291 [-0.8126, -0.4582] SB |
| accel MAE | LONGITUDINAL | -1.2182 [-1.6895, -0.8231] SB | -0.3390 [-0.4634, -0.2362] SB | -0.8791 [-1.2581, -0.5583] SB |
| DK-VIOL | LONGITUDINAL | +0.0084 [-0.0203, +0.0379] ns | +0.0017 [-0.0096, +0.0174] ns | +0.0067 [-0.0196, +0.0346] ns |
| DK-HW | LONGITUDINAL | -0.3865 [-0.7608, -0.1003] SB | -0.0642 [-0.1509, +0.0114] ns | -0.2040 [-0.5885, +0.1367] ns |
| DK-TG | LONGITUDINAL | -0.0963 [-0.1594, -0.0455] SB | -0.0402 [-0.1076, +0.0113] ns | -0.0560 [-0.1173, -0.0018] SB |
| heading MAE | LATERAL | -1.5058 [-2.5614, -0.2418] SB | +1.4556 [+0.1851, +2.9114] SW | -2.9614 [-3.7496, -2.1787] SB |
| yaw-rate MAE | LATERAL | -11.3955 [-16.2325, -7.1631] SB | -13.3749 [-18.0190, -9.2907] SB | +1.9270 [-1.0592, +5.0043] ns |
| curvature MAE | LATERAL | -0.0246 [-0.0401, -0.0111] SB | -0.1183 [-0.1629, -0.0792] SB | +0.0919 [+0.0584, +0.1301] SW |
| cross-track MAE | LATERAL | -1.0611 [-1.2172, -0.9136] SB | -0.0346 [-0.0732, +0.0071] ns | -1.0265 [-1.1729, -0.8943] SB |
| L1 lateral-accel MAE | LATERAL | -3.6129 [-5.1617, -2.2845] SB | -0.9710 [-1.3911, -0.6275] SB | -2.6420 [-4.0010, -1.5149] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0057 [+0.0018, +0.0102] SW | -0.0526 [-0.0935, -0.0228] SB | +0.0628 [+0.0279, +0.1076] SW |
| TAC-LAT | TACTICAL | +0.2043 [+0.1619, +0.2452] SB | +0.0196 [+0.0050, +0.0342] SB | +0.1847 [+0.1417, +0.2274] SB |
| TAC-LON | TACTICAL | +0.1626 [+0.0938, +0.2345] SB | +0.0158 [+0.0050, +0.0283] SB | +0.1467 [+0.0792, +0.2138] SB |
| S-POS@30 | STRATEGIC | -0.8567 [-1.1427, -0.5901] SB | +0.6577 [+0.2458, +1.1374] SW | -1.6062 [-2.1525, -1.1336] SB |
| S-POS@50 | STRATEGIC | -3.0487 [-3.9321, -2.2407] SB | +1.1096 [+0.3701, +1.9473] SW | -4.5825 [-5.8449, -3.4774] SB |
| S-POS@80 | STRATEGIC | -7.9092 [-9.8574, -6.0715] SB | +1.6220 [+0.4728, +3.0089] SW | -10.1918 [-12.6081, -8.0751] SB |
| M1 crawl share | crawl | +0.0700 [+0.0462, +0.0960] SW | +0.0046 [+0.0011, +0.0087] SW | +0.0666 [+0.0428, +0.0927] SW |

### kabox SOUP s1

* **PRIMARY ADE6 (m)**: X - P10 -1.1953 [-1.4093, -0.9851] SB (D1: **True**); X - Y -0.0058 [-0.0362, +0.0291] ns (guard: **True**); Y - P10 -1.1895 [-1.3938, -0.9847] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 16 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: S-POS@30, S-POS@50, S-POS@80, heading MAE. Rows separated-worse on X - Y, all 19: 5.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1953 [-1.4093, -0.9851] SB | -0.0058 [-0.0362, +0.0291] ns | -1.1895 [-1.3938, -0.9847] SB |
| speed MAE | LONGITUDINAL | -0.9055 [-1.1429, -0.6940] SB | -0.1318 [-0.1899, -0.0836] SB | -0.7737 [-0.9864, -0.5882] SB |
| along-track MAE | LONGITUDINAL | -0.6208 [-0.8021, -0.4435] SB | +0.0052 [-0.0196, +0.0299] ns | -0.6260 [-0.8049, -0.4497] SB |
| accel MAE | LONGITUDINAL | -1.2683 [-1.7421, -0.8718] SB | -0.4719 [-0.6546, -0.3124] SB | -0.7964 [-1.1397, -0.4913] SB |
| DK-VIOL | LONGITUDINAL | -0.0034 [-0.0249, +0.0185] ns | +0.0034 [-0.0035, +0.0126] ns | -0.0067 [-0.0265, +0.0122] ns |
| DK-HW | LONGITUDINAL | -0.6084 [-0.9844, -0.2949] SB | -0.0417 [-0.1136, +0.0340] ns | -0.5481 [-0.9299, -0.2218] SB |
| DK-TG | LONGITUDINAL | -0.1141 [-0.1852, -0.0540] SB | -0.0249 [-0.0958, +0.0414] ns | -0.0863 [-0.1467, -0.0356] SB |
| heading MAE | LATERAL | -1.9278 [-2.7892, -1.0340] SB | +1.0169 [+0.2159, +1.9341] SW | -2.9491 [-3.6329, -2.3131] SB |
| yaw-rate MAE | LATERAL | -11.5758 [-16.4940, -7.4680] SB | -14.7458 [-20.2453, -9.8737] SB | +3.0944 [+0.2216, +6.2120] SW |
| curvature MAE | LATERAL | -0.0253 [-0.0413, -0.0116] SB | -0.1340 [-0.1888, -0.0863] SB | +0.1067 [+0.0666, +0.1533] SW |
| cross-track MAE | LATERAL | -1.0382 [-1.1903, -0.8925] SB | -0.0193 [-0.0509, +0.0172] ns | -1.0189 [-1.1607, -0.8829] SB |
| L1 lateral-accel MAE | LATERAL | -3.6206 [-5.1516, -2.3012] SB | -1.0474 [-1.5038, -0.6599] SB | -2.5744 [-3.8473, -1.4601] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0037 [-0.0001, +0.0079] ns | -0.0408 [-0.0658, -0.0206] SB | +0.0454 [+0.0224, +0.0739] SW |
| TAC-LAT | TACTICAL | +0.2088 [+0.1681, +0.2496] SB | -0.0050 [-0.0192, +0.0096] ns | +0.2138 [+0.1747, +0.2548] SB |
| TAC-LON | TACTICAL | +0.1701 [+0.1087, +0.2315] SB | +0.0117 [+0.0029, +0.0221] SB | +0.1584 [+0.0975, +0.2172] SB |
| S-POS@30 | STRATEGIC | -1.1817 [-1.4987, -0.8908] SB | +0.3621 [+0.1010, +0.6440] SW | -1.7972 [-2.2609, -1.3948] SB |
| S-POS@50 | STRATEGIC | -3.7407 [-4.6630, -2.8730] SB | +0.5431 [+0.1070, +0.9847] SW | -5.0252 [-6.1549, -3.9713] SB |
| S-POS@80 | STRATEGIC | -9.3859 [-11.5290, -7.3224] SB | +0.7728 [+0.1076, +1.4796] SW | -10.7619 [-12.9488, -8.6791] SB |
| M1 crawl share | crawl | +0.0625 [+0.0403, +0.0887] SW | +0.0052 [+0.0012, +0.0102] SW | +0.0576 [+0.0364, +0.0823] SW |

## 4. KAPPA-T1's box beside the three clamps (X - Y, point and mark; `../kappa_t1/raw/KAPPA_T1.json`, same Y, same windows)

### DAG-OACC s0, X - Y

| row | box (KAPPA-T1) | kbox | abox | kabox |
|---|---|---|---|---|
| ADE6 | +0.0115 ns | +0.0230 ns | -0.0382 SB | -0.0020 ns |
| heading MAE | +1.6946 SW | +1.7413 SW | +1.8684 SW | +1.7364 SW |
| S-POS@30 | +0.4995 SW | +0.5048 SW | +0.4806 SW | +0.5058 SW |
| S-POS@50 | +0.7836 SW | +0.7883 SW | +0.7708 SW | +0.7966 SW |
| S-POS@80 | +1.0942 SW | +1.0915 SW | +1.1173 SW | +1.1223 SW |
| along-track MAE | +0.0227 ns | +0.0373 ns | -0.0144 ns | +0.0082 ns |
| curvature MAE | -0.0949 SB | -0.0948 SB | +0.0525 SW | -0.0914 SB |
| L2 curvature MAE (GT>=5.2) | -0.0292 SB | -0.0293 SB | +0.0106 ns | -0.0271 SB |
| yaw-rate MAE | -11.6134 SB | -11.4947 SB | +1.8491 SW | -11.1130 SB |
| M1 crawl share | +0.0048 SW | +0.0040 SW | +0.0059 SW | +0.0056 SW |

### DAG-OACC s1, X - Y

| row | box (KAPPA-T1) | kbox | abox | kabox |
|---|---|---|---|---|
| ADE6 | +0.0606 SW | +0.0753 SW | -0.0301 SB | +0.0356 SW |
| heading MAE | +1.3399 SW | +1.3432 SW | +0.9396 SW | +1.1877 SW |
| S-POS@30 | +0.2400 ns | +0.2572 SW | +0.4436 SW | +0.2660 SW |
| S-POS@50 | +0.3663 ns | +0.3925 ns | +0.7233 SW | +0.4139 ns |
| S-POS@80 | +0.4863 ns | +0.5132 ns | +1.0803 SW | +0.5558 ns |
| along-track MAE | +0.0607 SW | +0.0718 SW | -0.0059 ns | +0.0357 SW |
| curvature MAE | -0.1545 SB | -0.1547 SB | +0.0288 SW | -0.1487 SB |
| L2 curvature MAE (GT>=5.2) | -0.0459 SB | -0.0460 SB | +0.0050 ns | -0.0448 SB |
| yaw-rate MAE | -16.9787 SB | -16.7821 SB | -0.6881 ns | -16.3296 SB |
| M1 crawl share | +0.0071 SW | +0.0036 ns | +0.0076 SW | +0.0077 SW |

### SOUP s0, X - Y

| row | box (KAPPA-T1) | kbox | abox | kabox |
|---|---|---|---|---|
| ADE6 | +0.0084 ns | +0.0337 ns | -0.0586 SB | -0.0183 ns |
| heading MAE | +1.6201 SW | +1.6052 SW | +1.7787 SW | +1.4556 SW |
| S-POS@30 | +0.6435 SW | +0.6854 SW | +0.6141 SW | +0.6577 SW |
| S-POS@50 | +1.0833 SW | +1.1340 SW | +1.0038 SW | +1.1096 SW |
| S-POS@80 | +1.5711 SW | +1.6440 SW | +1.4689 SW | +1.6220 SW |
| along-track MAE | +0.0254 ns | +0.0492 SW | -0.0279 SB | -0.0002 ns |
| curvature MAE | -0.1244 SB | -0.1247 SB | +0.0643 SW | -0.1183 SB |
| L2 curvature MAE (GT>=5.2) | -0.0558 SB | -0.0559 SB | +0.0202 SW | -0.0526 SB |
| yaw-rate MAE | -14.2102 SB | -14.1158 SB | +2.6547 SW | -13.3749 SB |
| M1 crawl share | +0.0004 ns | -0.0006 ns | +0.0054 SW | +0.0046 SW |

### SOUP s1, X - Y

| row | box (KAPPA-T1) | kbox | abox | kabox |
|---|---|---|---|---|
| ADE6 | +0.0253 ns | +0.0446 SW | -0.0478 SB | -0.0058 ns |
| heading MAE | +1.4407 SW | +1.4220 SW | +1.2698 SW | +1.0169 SW |
| S-POS@30 | +0.3612 SW | +0.3731 SW | +0.2217 SW | +0.3621 SW |
| S-POS@50 | +0.5467 SW | +0.5627 SW | +0.3506 SW | +0.5431 SW |
| S-POS@80 | +0.7712 SW | +0.7949 SW | +0.4061 SW | +0.7728 SW |
| along-track MAE | +0.0373 SW | +0.0502 SW | -0.0193 ns | +0.0052 ns |
| curvature MAE | -0.1423 SB | -0.1426 SB | +0.0292 SW | -0.1340 SB |
| L2 curvature MAE (GT>=5.2) | -0.0428 SB | -0.0429 SB | +0.0110 SW | -0.0408 SB |
| yaw-rate MAE | -15.8216 SB | -15.6150 SB | +0.2155 ns | -14.7458 SB |
| M1 crawl share | +0.0036 ns | +0.0024 ns | +0.0064 SW | +0.0052 SW |

## 5. The prediction next to the result

Card (ESTIMATED, written before any X number of these arms), per lever: P(outcome) in %; read = the outcome at DAG-OACC / SOUP.

| lever | DOMINANT | HARMS | MOVED | NO EFFECT | read (DAG-OACC / SOUP) | probability the card gave the outcome read |
|---|---|---|---|---|---|---|
| kbox | 2 % | 70 % | 3 % | 25 % | **HARMS / HARMS** | 70 % / 70 % |
| abox | 1 % | 25 % | 9 % | 65 % | **HARMS / HARMS** | 25 % / 25 % |
| kabox | 5 % | 60 % | 8 % | 27 % | **HARMS / HARMS** | 60 % / 60 % |

M1 (crawl share X - Y; share of the Y - P10 excess closed = -(X - Y) / (Y - P10); negative = M1 moved the wrong way). Card: kbox unmoved (< 5 % of its excess); abox and kabox move < 15 % of its excess, P(separated-better at both seeds) 15 %. P(M1 keeps D2 failing under all three levers) = 85 %.

| lever | cell | M1 X - Y | Y - P10 | share of excess closed | X - Y separated-better | M1 fails D2 (X - P10) |
|---|---|---|---|---|---|---|
| kbox | DAG-OACC s0 | +0.0040 | +0.0483 | -8 % | False | True |
| kbox | DAG-OACC s1 | +0.0036 | +0.0565 | -6 % | False | True |
| kbox | SOUP s0 | -0.0006 | +0.0666 | 1 % | False | True |
| kbox | SOUP s1 | +0.0024 | +0.0576 | -4 % | False | True |
| abox | DAG-OACC s0 | +0.0059 | +0.0483 | -12 % | False | True |
| abox | DAG-OACC s1 | +0.0076 | +0.0565 | -13 % | False | True |
| abox | SOUP s0 | +0.0054 | +0.0666 | -8 % | False | True |
| abox | SOUP s1 | +0.0064 | +0.0576 | -11 % | False | True |
| kabox | DAG-OACC s0 | +0.0056 | +0.0483 | -11 % | False | True |
| kabox | DAG-OACC s1 | +0.0077 | +0.0565 | -14 % | False | True |
| kabox | SOUP s0 | +0.0046 | +0.0666 | -7 % | False | True |
| kabox | SOUP s1 | +0.0052 | +0.0576 | -9 % | False | True |

**MEASURED: M1 crawl share fails D2 at 12 of 12 cells** (the card gave 85 % that it fails under all three levers). kbox read HARMS as predicted (70 %) and kabox read HARMS (60 %). abox was predicted NO EFFECT (65 %) and read HARMS at both arms (25 %): abox is separated-worse on heading MAE and S-POS@30/50/80 X - Y at every cell, which the plan-level pre-check (section 6) did not show.

## 6. The pre-check attribution next to the rollouts (`raw/PRECHECK_ATTRIBUTION.json`; a per-tick counterfactual on KAPPA-T1's issued S plans, not a rollout)

Heading MAE proxy delta vs plan_s in degrees (plan-level, executed-knot steps vs the logged future; the card's table), against the rollout row `heading MAE` X - Y (T1-kin; paired episode-cluster bootstrap; the extractor's units).

| cell | pre-check kbox | pre-check abox | pre-check kabox | rollout kbox X - Y | rollout abox X - Y | rollout kabox X - Y |
|---|---|---|---|---|---|---|
| DAG-OACC s0 | +0.315 [-0.090, +0.784] | -0.127 [-0.282, -0.003] | +0.329 [-0.084, +0.814] | +1.7413 [+0.4096, +3.3880] SW | +1.8684 [+0.9526, +3.0070] SW | +1.7364 [+0.3783, +3.4017] SW |
| DAG-OACC s1 | +0.317 [-0.114, +0.749] | -0.121 [-0.268, +0.000] | +0.279 [-0.144, +0.704] | +1.3432 [+0.2024, +2.6876] SW | +0.9396 [+0.4948, +1.5201] SW | +1.1877 [+0.1603, +2.4098] SW |
| SOUP s0 | +0.181 [-0.312, +0.685] | -0.087 [-0.244, +0.053] | +0.145 [-0.336, +0.626] | +1.6052 [+0.2657, +3.2017] SW | +1.7787 [+0.8951, +2.8226] SW | +1.4556 [+0.1851, +2.9114] SW |
| SOUP s1 | +0.501 [+0.088, +0.926] | -0.094 [-0.249, +0.042] | +0.394 [+0.018, +0.811] | +1.4220 [+0.3766, +2.6896] SW | +1.2698 [+0.6766, +1.9896] SW | +1.0169 [+0.2159, +1.9341] SW |

Reading: the pre-check located the heading harm at the curvature clamp (kbox = box at the plan level) and read abox as improving heading at the plan level; the rollouts agree on kbox and kabox and do NOT agree on abox, which reads separated-worse on heading MAE at all four cells. A plan-level counterfactual on issued plans cannot see what a clamp changes in later ticks (the executed state feeds the next plan). HYPOTHESIS, not tested here: the entry bound (`clamp_entry`, binding on 1.9-2.8 % of ticks in the pre-check) and the accel clamp perturb the executed first knots and the heading read moves with them.

Crawl proxy at the plan level (step speed < GT step speed - 2 m/s, v0 3-8 m/s): plan_s sat below the P10 prior's share in all four cells and the clamps moved it by at most +0.002 (pre-check, plan level); the rollout M1 row above reads the executed rows, where the excess lives.

## 7. The card's committed next arm per outcome (Rule Zero), and what the read selects

* **HARMS (all six outcomes)**: on heading / S-POS: `kbox2` = `project_feasible(a_max=inf, kappa_max=0.2, min_speed=2.0)`; with M1 still separated-worse at every cell: `lonfloor`; and their composition `kbox2+lonfloor`. All three are the FEAS3 arms (PREREG_FEAS3_FROZEN.md).
* **kbox and kabox** carry heading MAE + S-POS@30/50/80 (+ along-track and ADE6 at some cells) as HARMS triggers and fail D2 on M1 only at 7 of 8 cells (L2 curvature also at SOUP s0 kabox): `kbox2` is aimed at the heading / S-POS harm (the curvature cap stops acting below 2 m/s), `lonfloor` at M1, and `kbox2+lonfloor` is the composition that could reach DOMINANT.
* **abox** reads HARMS on the same rows (heading MAE, S-POS) and leaves curvature, L2 and (3 of 4 cells) yaw-rate separated-worse vs P10: the card's committed next arms do not act on abox's own harm, so abox is not carried into FEAS3.

Each next arm is a zero-training flag value of the same tool and has its own frozen card (`PREREG_FEAS3_FROZEN.md`) before any FEAS3 number.

## 8. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)

* **kbox CTRL_s0**: X {"N": 2363, "E": -0.0352, "partition_sum_minus_E": -0.0, "n_dip_windows": 48, "n_highk_steps": 1842, "n_highk_steps_masked_in_row": 1438, "highk_share_from_steps_under_2mps": 1.0676} | Y {"N": 2364, "E": 0.0578, "partition_sum_minus_E": 0.0, "n_dip_windows": 46, "n_highk_steps": 4613, "n_highk_steps_masked_in_row": 3411, "highk_share_from_steps_under_2mps": 0.5549}
* **kbox CTRL_s1**: X {"N": 2365, "E": -0.0335, "partition_sum_minus_E": 0.0, "n_dip_windows": 63, "n_highk_steps": 2120, "n_highk_steps_masked_in_row": 1681, "highk_share_from_steps_under_2mps": 1.0688} | Y {"N": 2365, "E": 0.1212, "partition_sum_minus_E": 0.0, "n_dip_windows": 63, "n_highk_steps": 4962, "n_highk_steps_masked_in_row": 3704, "highk_share_from_steps_under_2mps": 0.5851}
* **kbox SOUP_s0**: X {"N": 2364, "E": -0.0309, "partition_sum_minus_E": 0.0, "n_dip_windows": 84, "n_highk_steps": 2290, "n_highk_steps_masked_in_row": 1856, "highk_share_from_steps_under_2mps": 1.1506} | Y {"N": 2364, "E": 0.0919, "partition_sum_minus_E": 0.0, "n_dip_windows": 74, "n_highk_steps": 5468, "n_highk_steps_masked_in_row": 4202, "highk_share_from_steps_under_2mps": 0.534}
* **kbox SOUP_s1**: X {"N": 2364, "E": -0.0339, "partition_sum_minus_E": 0.0, "n_dip_windows": 58, "n_highk_steps": 2200, "n_highk_steps_masked_in_row": 1751, "highk_share_from_steps_under_2mps": 1.1526} | Y {"N": 2365, "E": 0.1067, "partition_sum_minus_E": 0.0, "n_dip_windows": 55, "n_highk_steps": 5282, "n_highk_steps_masked_in_row": 3936, "highk_share_from_steps_under_2mps": 0.618}
* **abox CTRL_s0**: X {"N": 2363, "E": 0.1105, "partition_sum_minus_E": 0.0, "n_dip_windows": 26, "n_highk_steps": 5105, "n_highk_steps_masked_in_row": 3875, "highk_share_from_steps_under_2mps": 0.6482} | Y {"N": 2364, "E": 0.0578, "partition_sum_minus_E": 0.0, "n_dip_windows": 46, "n_highk_steps": 4613, "n_highk_steps_masked_in_row": 3411, "highk_share_from_steps_under_2mps": 0.5549}
* **abox CTRL_s1**: X {"N": 2365, "E": 0.15, "partition_sum_minus_E": 0.0, "n_dip_windows": 44, "n_highk_steps": 5194, "n_highk_steps_masked_in_row": 3994, "highk_share_from_steps_under_2mps": 0.7323} | Y {"N": 2365, "E": 0.1212, "partition_sum_minus_E": 0.0, "n_dip_windows": 63, "n_highk_steps": 4962, "n_highk_steps_masked_in_row": 3704, "highk_share_from_steps_under_2mps": 0.5851}
* **abox SOUP_s0**: X {"N": 2364, "E": 0.1562, "partition_sum_minus_E": -0.0, "n_dip_windows": 57, "n_highk_steps": 5784, "n_highk_steps_masked_in_row": 4555, "highk_share_from_steps_under_2mps": 0.6329} | Y {"N": 2364, "E": 0.0919, "partition_sum_minus_E": 0.0, "n_dip_windows": 74, "n_highk_steps": 5468, "n_highk_steps_masked_in_row": 4202, "highk_share_from_steps_under_2mps": 0.534}
* **abox SOUP_s1**: X {"N": 2364, "E": 0.1351, "partition_sum_minus_E": 0.0, "n_dip_windows": 29, "n_highk_steps": 5637, "n_highk_steps_masked_in_row": 4363, "highk_share_from_steps_under_2mps": 0.6573} | Y {"N": 2365, "E": 0.1067, "partition_sum_minus_E": 0.0, "n_dip_windows": 55, "n_highk_steps": 5282, "n_highk_steps_masked_in_row": 3936, "highk_share_from_steps_under_2mps": 0.618}
* **kabox CTRL_s0**: X {"N": 2363, "E": -0.0317, "partition_sum_minus_E": 0.0, "n_dip_windows": 31, "n_highk_steps": 2436, "n_highk_steps_masked_in_row": 1929, "highk_share_from_steps_under_2mps": 1.3123} | Y {"N": 2364, "E": 0.0578, "partition_sum_minus_E": 0.0, "n_dip_windows": 46, "n_highk_steps": 4613, "n_highk_steps_masked_in_row": 3411, "highk_share_from_steps_under_2mps": 0.5549}
* **kabox CTRL_s1**: X {"N": 2365, "E": -0.0276, "partition_sum_minus_E": -0.0, "n_dip_windows": 53, "n_highk_steps": 2708, "n_highk_steps_masked_in_row": 2108, "highk_share_from_steps_under_2mps": 2.478} | Y {"N": 2365, "E": 0.1212, "partition_sum_minus_E": 0.0, "n_dip_windows": 63, "n_highk_steps": 4962, "n_highk_steps_masked_in_row": 3704, "highk_share_from_steps_under_2mps": 0.5851}
* **kabox SOUP_s0**: X {"N": 2364, "E": -0.0246, "partition_sum_minus_E": 0.0, "n_dip_windows": 64, "n_highk_steps": 3028, "n_highk_steps_masked_in_row": 2464, "highk_share_from_steps_under_2mps": 2.1209} | Y {"N": 2364, "E": 0.0919, "partition_sum_minus_E": 0.0, "n_dip_windows": 74, "n_highk_steps": 5468, "n_highk_steps_masked_in_row": 4202, "highk_share_from_steps_under_2mps": 0.534}
* **kabox SOUP_s1**: X {"N": 2364, "E": -0.0253, "partition_sum_minus_E": 0.0, "n_dip_windows": 42, "n_highk_steps": 2854, "n_highk_steps_masked_in_row": 2256, "highk_share_from_steps_under_2mps": 9.392} | Y {"N": 2365, "E": 0.1067, "partition_sum_minus_E": 0.0, "n_dip_windows": 55, "n_highk_steps": 5282, "n_highk_steps_masked_in_row": 3936, "highk_share_from_steps_under_2mps": 0.618}

Gloss (from `gate_kappa.partition`): N = windows finite on both sides; E = the curvature MAE point (X - P10 for X, Y - P10 for Y); n_dip_windows = windows with an unwarranted speed dip; n_highk_steps = steps with |kappa| of the executed path above 0.2 1/m; highk_share_from_steps_under_2mps above 1 means those steps carry more than the HIGHK total (other HIGHK steps carry the opposite sign).

## 9. Limits

* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); a separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`; replicate arm not run). The planner is deterministic, so *another inference run* is not in play for these arms.
* X - P10 and Y - P10 use the banked P10 side (SMOOTH-T1's reference choice); the fresh floor differs from it by the KF deviation above. X - Y does not use P10.
* Tier T1-kin: executed state, logged cameras, own plan applied open-loop in the scene; it says nothing about re-perception. Inference inputs: vision plus measured ego history.
* Adoption into any K-1 / WM-1 claim is not decided here; the Master Mind registers the claim row `H-FEAS2-T1-1` (a proposed id).
