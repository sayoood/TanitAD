# RESULTS_FEAS3: H-FEAS3-T1-1, kbox2 / lonfloor / kbox2lf after S, DAG-OACC and SOUP at s0 / s1

Card `PREREG_FEAS3_FROZEN.md` sha256 `e817abcfcf51088a9a72a4f49444462e1588ee587c4974b4454de7f8569c04a7` (frozen 2026-11-02T17:43:04+0100). Reader `code/feas3_t1_read.py` (the import-path fix of section 8 applies). Read clock 2026-11-02T18:21:23+0100 (the re-run read). Raw `raw/FEAS3_T1.json`, `raw/PRECHECK_FEAS3.json`, per-window arrays `raw/arrays/*.npz`. Tier **T1-kin** (own executed state, logged cameras; a self-action open loop). Evidence class **MEASURED (ours; artifacts in MANIFEST.txt)**.
Estimator of every interval below: **paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. One training run per seed and a deterministic planner: the intervals answer *another draw of episodes* only, not *another training run* (`H-ESTIM-SEED-1`) nor *another inference run*. X = S + lever (S = `--smooth sg72`); Y = KAPPA-T1's fresh S runs, reused (pins in `FROZEN_SHA_FEAS3.txt`); P10 = the prior floor.

## 0. Outcome (the card's rule, precedence DOMINANT > HARMS > MOVED > NO EFFECT)

| lever | arm | outcome | seed 0: D1 / D2 / D3 | seed 1: D1 / D2 / D3 | seed 0: MOVED / HARMS | seed 1: MOVED / HARMS |
|---|---|---|---|---|---|---|
| kbox2 | DAG-OACC | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| kbox2 | SOUP | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| lonfloor | DAG-OACC | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| lonfloor | SOUP | **HARMS** | True / False / True | True / False / True | True / True | True / True |
| kbox2lf | DAG-OACC | **HARMS** | True / False / True | True / False / True | False / True | False / True |
| kbox2lf | SOUP | **HARMS** | True / False / True | True / False / True | False / True | False / True |

Headline arm DAG-OACC; SOUP stated separately. `moved_and_harms` flags (HARMS precedence hides a MOVED condition): kbox2/DAG-OACC False, kbox2/SOUP False, lonfloor/DAG-OACC False, lonfloor/SOUP True, kbox2lf/DAG-OACC False, kbox2lf/SOUP False.

**MEASURED headline.** 6 of 6 lever x arm outcomes read **HARMS** (DOMINANT, MOVED and NO EFFECT: 0). D1 (ADE6 X - P10) holds at 12 of 12 cells, D3 at 12 of 12, and D2 fails at 12 of 12 (on M1 crawl share at 12 of 12).

**The one MOVED condition hidden by precedence: lonfloor on SOUP.** At both SOUP seeds all four crawl rows (yaw-rate MAE, curvature MAE, L2 curvature MAE, M1 crawl share) are separated-better on X - Y and ADE6 X - Y is within the guard (-0.0004; +0.0104 m), so the MOVED condition holds; HARMS wins because a non-crawl row of 15 is separated-worse on X - Y (accel MAE, cross-track MAE at s0; L1 lateral-accel MAE at s1). The outcome word is HARMS. On DAG-OACC lonfloor's M1 X - Y is separated-better at both seeds but the MOVED condition does not hold (ADE6 X - Y +0.0389 / +0.0162 m; non-crawl rows separated-worse: ADE6, accel MAE, along-track MAE, cross-track MAE, speed MAE / ADE6, accel MAE, cross-track MAE, speed MAE).

## 1. What triggers HARMS and what blocks DOMINANT, per lever, arm and seed (generated from the rows)

* **kbox2 DAG-OACC s0**: HARMS triggers: ADE6 X - Y +0.1283 > guard +0.0604; ADE6 X - Y +0.1283 SW; S-POS@30 X - Y +0.5542 SW; S-POS@50 X - Y +0.8515 SW; S-POS@80 X - Y +1.1924 SW; along-track MAE X - Y +0.1051 SW; cross-track MAE X - Y +0.0508 SW; heading MAE X - Y +2.4906 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kbox2 DAG-OACC s1**: HARMS triggers: ADE6 X - Y +0.1442 > guard +0.0604; ADE6 X - Y +0.1442 SW; S-POS@30 X - Y +0.4027 SW; S-POS@50 X - Y +0.5957 SW; S-POS@80 X - Y +0.8102 SW; along-track MAE X - Y +0.1154 SW; cross-track MAE X - Y +0.0739 SW; heading MAE X - Y +2.1163 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kbox2 SOUP s0**: HARMS triggers: ADE6 X - Y +0.1342 > guard +0.0604; ADE6 X - Y +0.1342 SW; S-POS@30 X - Y +0.7093 SW; S-POS@50 X - Y +1.1454 SW; S-POS@80 X - Y +1.6317 SW; along-track MAE X - Y +0.1161 SW; cross-track MAE X - Y +0.0485 SW; heading MAE X - Y +2.4643 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kbox2 SOUP s1**: HARMS triggers: ADE6 X - Y +0.1287 > guard +0.0604; ADE6 X - Y +0.1287 SW; S-POS@30 X - Y +0.4786 SW; S-POS@50 X - Y +0.7178 SW; S-POS@80 X - Y +0.9651 SW; along-track MAE X - Y +0.1133 SW; cross-track MAE X - Y +0.0478 SW; heading MAE X - Y +2.0449 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **lonfloor DAG-OACC s0**: HARMS triggers: ADE6 X - Y +0.0389 SW; accel MAE X - Y +0.0277 SW; along-track MAE X - Y +0.0318 SW; cross-track MAE X - Y +0.0168 SW; speed MAE X - Y +0.0421 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE. Crawl rows X - Y separated-better: curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share; not: yaw-rate MAE.
* **lonfloor DAG-OACC s1**: HARMS triggers: ADE6 X - Y +0.0162 SW; accel MAE X - Y +0.0444 SW; cross-track MAE X - Y +0.0139 SW; speed MAE X - Y +0.0208 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, M1 crawl share; not: L2 curvature MAE (GT>=5.2).
* **lonfloor SOUP s0**: HARMS triggers: accel MAE X - Y +0.0285 SW; cross-track MAE X - Y +0.0063 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share; not: none.
* **lonfloor SOUP s1**: HARMS triggers: L1 lateral-accel MAE X - Y +0.1196 SW. D2 (X - P10) failing rows: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share; not: none.
* **kbox2lf DAG-OACC s0**: HARMS triggers: ADE6 X - Y +0.1716 > guard +0.0604; ADE6 X - Y +0.1716 SW; S-POS@30 X - Y +0.5441 SW; S-POS@50 X - Y +0.8283 SW; S-POS@80 X - Y +1.1543 SW; along-track MAE X - Y +0.1445 SW; cross-track MAE X - Y +0.0635 SW; heading MAE X - Y +2.4824 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kbox2lf DAG-OACC s1**: HARMS triggers: ADE6 X - Y +0.1833 > guard +0.0604; ADE6 X - Y +0.1833 SW; S-POS@30 X - Y +0.3847 SW; S-POS@50 X - Y +0.5597 SW; S-POS@80 X - Y +0.7643 SW; along-track MAE X - Y +0.1414 SW; cross-track MAE X - Y +0.0988 SW; heading MAE X - Y +2.1251 SW; speed MAE X - Y +0.0339 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.
* **kbox2lf SOUP s0**: HARMS triggers: ADE6 X - Y +0.1415 > guard +0.0604; ADE6 X - Y +0.1415 SW; S-POS@30 X - Y +0.6834 SW; S-POS@50 X - Y +1.0889 SW; S-POS@80 X - Y +1.5376 SW; along-track MAE X - Y +0.1168 SW; cross-track MAE X - Y +0.0571 SW; heading MAE X - Y +2.3627 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share; not: none.
* **kbox2lf SOUP s1**: HARMS triggers: ADE6 X - Y +0.1640 > guard +0.0604; ADE6 X - Y +0.1640 SW; S-POS@30 X - Y +0.4578 SW; S-POS@50 X - Y +0.6803 SW; S-POS@80 X - Y +0.9170 SW; along-track MAE X - Y +0.1383 SW; cross-track MAE X - Y +0.0660 SW; heading MAE X - Y +2.0102 SW. D2 (X - P10) failing rows: M1 crawl share. Crawl rows X - Y separated-better: yaw-rate MAE, curvature MAE, L2 curvature MAE (GT>=5.2); not: M1 crawl share.

### D2 failing rows, count per lever (4 cells: DAG-OACC s0 / s1, SOUP s0 / s1)

| lever | M1 crawl share | curvature MAE | L2 curvature MAE (GT>=5.2) | yaw-rate MAE | any other row |
|---|---|---|---|---|---|
| kbox2 | 4 / 4 | 0 / 4 | 1 / 4 | 0 / 4 | none |
| lonfloor | 4 / 4 | 4 / 4 | 4 / 4 | 1 / 4 | none |
| kbox2lf | 4 / 4 | 0 / 4 | 0 / 4 | 0 / 4 | none |

### HARMS triggers, count per lever (cells at which the row is separated-worse on X - Y, or ADE6 beyond the guard)

| lever | ADE6 beyond guard | ADE6 | L1 lateral-accel MAE | S-POS@30 | S-POS@50 | S-POS@80 | accel MAE | along-track MAE | cross-track MAE | heading MAE | speed MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kbox2 | 4 / 4 | 4 / 4 | 0 / 4 | 4 / 4 | 4 / 4 | 4 / 4 | 0 / 4 | 4 / 4 | 4 / 4 | 4 / 4 | 0 / 4 |
| lonfloor | 0 / 4 | 2 / 4 | 1 / 4 | 0 / 4 | 0 / 4 | 0 / 4 | 3 / 4 | 1 / 4 | 3 / 4 | 0 / 4 | 2 / 4 |
| kbox2lf | 4 / 4 | 4 / 4 | 0 / 4 | 4 / 4 | 4 / 4 | 4 / 4 | 0 / 4 | 4 / 4 | 4 / 4 | 4 / 4 | 1 / 4 |

## 2. Known answers and controls (all six groups, computed before any row)

* **KY_reused_Y_equals_frozen_pins**: PASS
* **KA1_fresh_rerun_equals_reused_Y**: PASS
* **KA2_issued_plan_is_lever_of_S**: PASS
* **KF_floor_window_check_vs_fresh_Y**: PASS
* **K3_Y_banked_minus_P10_reproduces_SOUP_READ**: PASS
* **KB1_P10_side_equals_landed_banked_b**: PASS

**KY** (the reused Y files equal the pins of `FROZEN_SHA_FEAS3.txt`): 8 / 8 files hash-equal.

**KA1** (the fresh rerun `SREP3_DAGOACC_s0` with this tool and `--feasible off` equals the reused `SREP_DAGOACC_s0`; tolerance 1e-4 m): dump bit-identical True (max abs dev 0.000e+00 m); ticks bit-identical True (max abs dev 0.000e+00); differing keys none; key-set difference none.

**KA2** (X's issued `plan` = lever(P + SG(plan_raw - P), tick v0), the operators re-spelled independently of the tool; tick-0 `plan_raw` = Y's; floor = Y's; tolerance 1e-4 m):

| run | plan_s vs P+SG (m) | plan vs lever(plan_s) (m) | tick-0 plan_raw vs Y (m) | floor X vs Y (m) | lever moved the issued plan by up to (m) |
|---|---|---|---|---|---|
| CTRL_s0:kbox2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1231.289 |
| CTRL_s0:lonfloor | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 84.926 |
| CTRL_s0:kbox2lf | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 497.546 |
| CTRL_s1:kbox2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 5524.202 |
| CTRL_s1:lonfloor | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 147.382 |
| CTRL_s1:kbox2lf | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1867.586 |
| SOUP_s0:kbox2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 948.237 |
| SOUP_s0:lonfloor | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 83.641 |
| SOUP_s0:kbox2lf | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1217.692 |
| SOUP_s1:kbox2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 270.261 |
| SOUP_s1:lonfloor | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 138.684 |
| SOUP_s1:kbox2lf | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 501.343 |

The last KA2 column is the maximum over every tick and knot of |plan - plan_s| (far unexecuted knots included); only executed knots 1-10 reach the rows.

**KF** (floor / window check against the fresh Y; gross guard 1.0 m against the banked CTRL):

| run | windows X = Y | windows Y = banked CTRL | floor X vs Y (m) | floor Y vs banked CTRL max / mean (m) |
|---|---|---|---|---|
| CTRL_s0:kbox2 | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s0:lonfloor | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s0:kbox2lf | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s1:kbox2 | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s1:lonfloor | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| CTRL_s1:kbox2lf | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s0:kbox2 | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s0:lonfloor | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s0:kbox2lf | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s1:kbox2 | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s1:lonfloor | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |
| SOUP_s1:kbox2lf | True | True | 0.000e+00 | 1.711e-02 / 3.416e-05 |

**K3**: Y banked - P10 from the landed arrays reproduces the pinned `SOUP_READ.json` on 76 / 76 rows (tolerance 6e-05). **KB1**: the P10 side of every extracted X and Y equals the landed banked `b` arrays on the non-S-POS rows: 16 / 16 runs pass.

**KM1** (the disclosure reader reproduces the M1 row's per-window values on Y and on X exactly): 16 / 16 cells True. **KA3** (`tests/test_v8f_t1_feas3.py`, 16 passed, `logs/pytest_feas3.log`; real-checkpoint smoke `logs/smoke_feas3.out`): run when the tool was built; not re-run in this read.

## 3. Read A against P10 (D1 / D2 / D3) and X - Y, per lever, arm and seed

The 19 rows of `extract_smooth.py`; every cell is `point [lo, hi] mark`. D1 bar: ADE6 X - P10 separated-better with point <= -0.2415 m. HARMS guard: ADE6 X - Y point <= +0.0604 m.

### kbox2 DAG-OACC s0

* **PRIMARY ADE6 (m)**: X - P10 -1.1272 [-1.3258, -0.9384] SB (D1: **True**); X - Y +0.1283 [+0.0731, +0.1945] SW (guard: **False**); Y - P10 -1.2556 [-1.4503, -1.0678] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 15 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, S-POS@50, S-POS@80, along-track MAE, cross-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 7.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1272 [-1.3258, -0.9384] SB | +0.1283 [+0.0731, +0.1945] SW | -1.2556 [-1.4503, -1.0678] SB |
| speed MAE | LONGITUDINAL | -0.8655 [-1.0771, -0.6801] SB | -0.0325 [-0.0739, -0.0017] SB | -0.8330 [-1.0377, -0.6583] SB |
| along-track MAE | LONGITUDINAL | -0.6142 [-0.7835, -0.4579] SB | +0.1051 [+0.0496, +0.1718] SW | -0.7192 [-0.8806, -0.5627] SB |
| accel MAE | LONGITUDINAL | -1.0651 [-1.4774, -0.7271] SB | -0.1260 [-0.2193, -0.0576] SB | -0.9390 [-1.3078, -0.6221] SB |
| DK-VIOL | LONGITUDINAL | +0.0235 [-0.0119, +0.0695] ns | +0.0067 [-0.0032, +0.0215] ns | +0.0168 [-0.0157, +0.0625] ns |
| DK-HW | LONGITUDINAL | -0.4159 [-0.7939, -0.1036] SB | -0.1706 [-0.3489, -0.0448] SB | -0.2604 [-0.6426, +0.0675] ns |
| DK-TG | LONGITUDINAL | -0.1217 [-0.2034, -0.0549] SB | -0.0568 [-0.1396, -0.0032] SB | -0.0588 [-0.1226, -0.0011] SB |
| heading MAE | LATERAL | -0.9614 [-2.1624, +0.5043] ns | +2.4906 [+1.0417, +4.3102] SW | -3.4520 [-4.2140, -2.7529] SB |
| yaw-rate MAE | LATERAL | -11.9046 [-16.7517, -7.7470] SB | -10.5775 [-14.3118, -7.3217] SB | -1.3762 [-3.9121, +1.0259] ns |
| curvature MAE | LATERAL | -0.0303 [-0.0446, -0.0178] SB | -0.0898 [-0.1255, -0.0604] SB | +0.0578 [+0.0364, +0.0821] SW |
| cross-track MAE | LATERAL | -0.9502 [-1.0970, -0.8094] SB | +0.0508 [+0.0169, +0.0885] SW | -1.0009 [-1.1468, -0.8671] SB |
| L1 lateral-accel MAE | LATERAL | -3.4683 [-4.9939, -2.1701] SB | -0.7358 [-1.0340, -0.4892] SB | -2.7327 [-4.0998, -1.5826] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0001 [-0.0030, +0.0038] ns | -0.0301 [-0.0536, -0.0142] SB | +0.0325 [+0.0135, +0.0590] SW |
| TAC-LAT | TACTICAL | +0.2072 [+0.1651, +0.2490] SB | +0.0033 [-0.0113, +0.0171] ns | +0.2038 [+0.1639, +0.2450] SB |
| TAC-LON | TACTICAL | +0.1788 [+0.1169, +0.2431] SB | +0.0038 [-0.0038, +0.0117] ns | +0.1751 [+0.1138, +0.2389] SB |
| S-POS@30 | STRATEGIC | -0.9597 [-1.2692, -0.6559] SB | +0.5542 [+0.2237, +0.9803] SW | -1.6443 [-2.1654, -1.1864] SB |
| S-POS@50 | STRATEGIC | -3.2298 [-4.0848, -2.4233] SB | +0.8515 [+0.2784, +1.6022] SW | -4.5745 [-5.8089, -3.5098] SB |
| S-POS@80 | STRATEGIC | -8.0776 [-9.9922, -6.2251] SB | +1.1924 [+0.2333, +2.4550] SW | -9.8284 [-12.1173, -7.7691] SB |
| M1 crawl share | crawl | +0.0460 [+0.0282, +0.0666] SW | -0.0020 [-0.0051, +0.0006] ns | +0.0483 [+0.0302, +0.0698] SW |

### kbox2 DAG-OACC s1

* **PRIMARY ADE6 (m)**: X - P10 -1.0460 [-1.2503, -0.8381] SB (D1: **True**); X - Y +0.1442 [+0.0857, +0.2109] SW (guard: **False**); Y - P10 -1.1903 [-1.3986, -0.9782] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 15 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, S-POS@50, S-POS@80, along-track MAE, cross-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 7.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.0460 [-1.2503, -0.8381] SB | +0.1442 [+0.0857, +0.2109] SW | -1.1903 [-1.3986, -0.9782] SB |
| speed MAE | LONGITUDINAL | -0.8203 [-1.0335, -0.6280] SB | -0.0061 [-0.0245, +0.0125] ns | -0.8141 [-1.0377, -0.6191] SB |
| along-track MAE | LONGITUDINAL | -0.5148 [-0.6986, -0.3433] SB | +0.1154 [+0.0609, +0.1774] SW | -0.6302 [-0.8196, -0.4510] SB |
| accel MAE | LONGITUDINAL | -1.0353 [-1.4270, -0.7025] SB | -0.0633 [-0.1157, -0.0159] SB | -0.9719 [-1.3431, -0.6572] SB |
| DK-VIOL | LONGITUDINAL | -0.0034 [-0.0247, +0.0179] ns | -0.0034 [-0.0165, +0.0114] ns | +0.0000 [-0.0196, +0.0212] ns |
| DK-HW | LONGITUDINAL | -0.5951 [-0.9989, -0.2844] SB | -0.1407 [-0.3147, -0.0091] SB | -0.5171 [-0.9057, -0.2118] SB |
| DK-TG | LONGITUDINAL | -0.1328 [-0.2160, -0.0683] SB | -0.0460 [-0.1270, +0.0160] ns | -0.0869 [-0.1542, -0.0278] SB |
| heading MAE | LATERAL | -0.7773 [-2.0004, +0.6958] ns | +2.1163 [+0.7849, +3.7185] SW | -2.8936 [-3.6062, -2.1968] SB |
| yaw-rate MAE | LATERAL | -11.6786 [-16.6535, -7.5136] SB | -15.8198 [-22.1263, -10.4602] SB | +4.1412 [+1.1665, +7.3821] SW |
| curvature MAE | LATERAL | -0.0248 [-0.0384, -0.0125] SB | -0.1460 [-0.2036, -0.0956] SB | +0.1212 [+0.0757, +0.1729] SW |
| cross-track MAE | LATERAL | -0.9478 [-1.0994, -0.8021] SB | +0.0739 [+0.0430, +0.1132] SW | -1.0217 [-1.1686, -0.8803] SB |
| L1 lateral-accel MAE | LATERAL | -3.3998 [-4.9049, -2.0972] SB | -0.7916 [-1.1108, -0.5229] SB | -2.6082 [-3.8410, -1.5151] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0053 [-0.0004, +0.0124] ns | -0.0452 [-0.0684, -0.0263] SB | +0.0505 [+0.0269, +0.0791] SW |
| TAC-LAT | TACTICAL | +0.2088 [+0.1696, +0.2502] SB | -0.0038 [-0.0154, +0.0079] ns | +0.2126 [+0.1709, +0.2540] SB |
| TAC-LON | TACTICAL | +0.1634 [+0.1017, +0.2238] SB | -0.0004 [-0.0071, +0.0067] ns | +0.1638 [+0.1021, +0.2225] SB |
| S-POS@30 | STRATEGIC | -1.2712 [-1.5974, -0.9729] SB | +0.4027 [+0.1611, +0.6865] SW | -1.7416 [-2.2209, -1.3273] SB |
| S-POS@50 | STRATEGIC | -3.8743 [-4.7285, -3.0675] SB | +0.5957 [+0.1938, +1.0511] SW | -4.7600 [-5.8592, -3.7484] SB |
| S-POS@80 | STRATEGIC | -9.5216 [-11.5766, -7.6600] SB | +0.8102 [+0.1674, +1.5083] SW | -10.2999 [-12.3713, -8.3307] SB |
| M1 crawl share | crawl | +0.0587 [+0.0363, +0.0855] SW | +0.0022 [-0.0008, +0.0053] ns | +0.0565 [+0.0347, +0.0829] SW |

### kbox2 SOUP s0

* **PRIMARY ADE6 (m)**: X - P10 -1.0586 [-1.2688, -0.8544] SB (D1: **True**); X - Y +0.1342 [+0.0770, +0.2005] SW (guard: **False**); Y - P10 -1.1928 [-1.4078, -0.9880] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 15 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, S-POS@50, S-POS@80, along-track MAE, cross-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 7.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.0586 [-1.2688, -0.8544] SB | +0.1342 [+0.0770, +0.2005] SW | -1.1928 [-1.4078, -0.9880] SB |
| speed MAE | LONGITUDINAL | -0.7938 [-1.0041, -0.6070] SB | -0.0121 [-0.0307, +0.0069] ns | -0.7817 [-0.9919, -0.5956] SB |
| along-track MAE | LONGITUDINAL | -0.5130 [-0.6950, -0.3369] SB | +0.1161 [+0.0554, +0.1840] SW | -0.6291 [-0.8126, -0.4582] SB |
| accel MAE | LONGITUDINAL | -0.9373 [-1.3250, -0.6138] SB | -0.0582 [-0.1065, -0.0126] SB | -0.8791 [-1.2581, -0.5583] SB |
| DK-VIOL | LONGITUDINAL | +0.0067 [-0.0223, +0.0374] ns | +0.0000 [-0.0105, +0.0142] ns | +0.0067 [-0.0196, +0.0346] ns |
| DK-HW | LONGITUDINAL | -0.4062 [-0.7858, -0.1169] SB | -0.0898 [-0.1923, +0.0189] ns | -0.2040 [-0.5885, +0.1367] ns |
| DK-TG | LONGITUDINAL | -0.1345 [-0.2193, -0.0662] SB | -0.0613 [-0.1332, -0.0023] SB | -0.0560 [-0.1173, -0.0018] SB |
| heading MAE | LATERAL | -0.4970 [-1.8082, +1.0676] ns | +2.4643 [+0.9360, +4.2514] SW | -2.9614 [-3.7496, -2.1787] SB |
| yaw-rate MAE | LATERAL | -11.1712 [-16.0066, -7.0054] SB | -13.1441 [-17.6580, -9.2443] SB | +1.9270 [-1.0592, +5.0043] ns |
| curvature MAE | LATERAL | -0.0228 [-0.0376, -0.0099] SB | -0.1164 [-0.1589, -0.0788] SB | +0.0919 [+0.0584, +0.1301] SW |
| cross-track MAE | LATERAL | -0.9780 [-1.1305, -0.8354] SB | +0.0485 [+0.0174, +0.0872] SW | -1.0265 [-1.1729, -0.8943] SB |
| L1 lateral-accel MAE | LATERAL | -3.4533 [-4.9838, -2.1418] SB | -0.8114 [-1.1590, -0.5375] SB | -2.6420 [-4.0010, -1.5149] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0048 [+0.0001, +0.0106] SW | -0.0580 [-0.0986, -0.0272] SB | +0.0628 [+0.0279, +0.1076] SW |
| TAC-LAT | TACTICAL | +0.2001 [+0.1576, +0.2410] SB | +0.0154 [+0.0017, +0.0300] SB | +0.1847 [+0.1417, +0.2274] SB |
| TAC-LON | TACTICAL | +0.1467 [+0.0799, +0.2148] SB | +0.0000 [-0.0075, +0.0071] ns | +0.1467 [+0.0792, +0.2138] SB |
| S-POS@30 | STRATEGIC | -0.8940 [-1.2098, -0.6118] SB | +0.7093 [+0.2991, +1.1942] SW | -1.6062 [-2.1525, -1.1336] SB |
| S-POS@50 | STRATEGIC | -3.1209 [-3.9271, -2.3953] SB | +1.1454 [+0.4289, +1.9887] SW | -4.5825 [-5.8449, -3.4774] SB |
| S-POS@80 | STRATEGIC | -8.1893 [-10.0827, -6.4275] SB | +1.6317 [+0.5014, +2.9591] SW | -10.1918 [-12.6081, -8.0751] SB |
| M1 crawl share | crawl | +0.0652 [+0.0418, +0.0906] SW | -0.0015 [-0.0034, +0.0003] ns | +0.0666 [+0.0428, +0.0927] SW |

### kbox2 SOUP s1

* **PRIMARY ADE6 (m)**: X - P10 -1.0608 [-1.2629, -0.8611] SB (D1: **True**); X - Y +0.1287 [+0.0723, +0.1918] SW (guard: **False**); Y - P10 -1.1895 [-1.3938, -0.9847] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 15 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, S-POS@50, S-POS@80, along-track MAE, cross-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 7.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.0608 [-1.2629, -0.8611] SB | +0.1287 [+0.0723, +0.1918] SW | -1.1895 [-1.3938, -0.9847] SB |
| speed MAE | LONGITUDINAL | -0.8071 [-1.0172, -0.6184] SB | -0.0334 [-0.0636, -0.0071] SB | -0.7737 [-0.9864, -0.5882] SB |
| along-track MAE | LONGITUDINAL | -0.5127 [-0.6906, -0.3407] SB | +0.1133 [+0.0610, +0.1744] SW | -0.6260 [-0.8049, -0.4497] SB |
| accel MAE | LONGITUDINAL | -0.9996 [-1.3864, -0.6728] SB | -0.2031 [-0.3312, -0.0942] SB | -0.7964 [-1.1397, -0.4913] SB |
| DK-VIOL | LONGITUDINAL | -0.0034 [-0.0258, +0.0190] ns | +0.0034 [-0.0031, +0.0104] ns | -0.0067 [-0.0265, +0.0122] ns |
| DK-HW | LONGITUDINAL | -0.6074 [-0.9919, -0.2937] SB | -0.0415 [-0.1247, +0.0320] ns | -0.5481 [-0.9299, -0.2218] SB |
| DK-TG | LONGITUDINAL | -0.1478 [-0.2413, -0.0739] SB | -0.0595 [-0.1559, +0.0123] ns | -0.0863 [-0.1467, -0.0356] SB |
| heading MAE | LATERAL | -0.9042 [-2.0809, +0.5050] ns | +2.0449 [+0.8302, +3.5291] SW | -2.9491 [-3.6329, -2.3131] SB |
| yaw-rate MAE | LATERAL | -11.5807 [-16.6042, -7.3860] SB | -14.7208 [-20.1456, -9.9248] SB | +3.0944 [+0.2216, +6.2120] SW |
| curvature MAE | LATERAL | -0.0253 [-0.0397, -0.0128] SB | -0.1336 [-0.1873, -0.0870] SB | +0.1067 [+0.0666, +0.1533] SW |
| cross-track MAE | LATERAL | -0.9711 [-1.1217, -0.8251] SB | +0.0478 [+0.0150, +0.0873] SW | -1.0189 [-1.1607, -0.8829] SB |
| L1 lateral-accel MAE | LATERAL | -3.4213 [-4.9640, -2.1222] SB | -0.8469 [-1.2111, -0.5175] SB | -2.5744 [-3.8473, -1.4601] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0049 [-0.0004, +0.0110] ns | -0.0405 [-0.0651, -0.0214] SB | +0.0454 [+0.0224, +0.0739] SW |
| TAC-LAT | TACTICAL | +0.2072 [+0.1665, +0.2481] SB | -0.0067 [-0.0188, +0.0050] ns | +0.2138 [+0.1747, +0.2548] SB |
| TAC-LON | TACTICAL | +0.1605 [+0.0996, +0.2200] SB | +0.0021 [-0.0037, +0.0083] ns | +0.1584 [+0.0975, +0.2172] SB |
| S-POS@30 | STRATEGIC | -1.3161 [-1.6400, -1.0190] SB | +0.4786 [+0.2294, +0.7643] SW | -1.7972 [-2.2609, -1.3948] SB |
| S-POS@50 | STRATEGIC | -3.9387 [-4.8634, -3.1100] SB | +0.7178 [+0.3301, +1.1481] SW | -5.0252 [-6.1549, -3.9713] SB |
| S-POS@80 | STRATEGIC | -9.5954 [-11.6972, -7.6316] SB | +0.9651 [+0.3788, +1.6282] SW | -10.7619 [-12.9488, -8.6791] SB |
| M1 crawl share | crawl | +0.0602 [+0.0384, +0.0861] SW | +0.0026 [-0.0014, +0.0075] ns | +0.0576 [+0.0364, +0.0823] SW |

### lonfloor DAG-OACC s0

* **PRIMARY ADE6 (m)**: X - P10 -1.2166 [-1.4054, -1.0365] SB (D1: **True**); X - Y +0.0389 [+0.0224, +0.0575] SW (guard: **True**); Y - P10 -1.2556 [-1.4503, -1.0678] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 12 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': False, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}. **Non-crawl rows separated-worse on X - Y**: ADE6, accel MAE, along-track MAE, cross-track MAE, speed MAE. Rows separated-worse on X - Y, all 19: 5.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2166 [-1.4054, -1.0365] SB | +0.0389 [+0.0224, +0.0575] SW | -1.2556 [-1.4503, -1.0678] SB |
| speed MAE | LONGITUDINAL | -0.7909 [-0.9839, -0.6225] SB | +0.0421 [+0.0198, +0.0683] SW | -0.8330 [-1.0377, -0.6583] SB |
| along-track MAE | LONGITUDINAL | -0.6875 [-0.8449, -0.5377] SB | +0.0318 [+0.0145, +0.0513] SW | -0.7192 [-0.8806, -0.5627] SB |
| accel MAE | LONGITUDINAL | -0.9113 [-1.2711, -0.6068] SB | +0.0277 [+0.0017, +0.0576] SW | -0.9390 [-1.3078, -0.6221] SB |
| DK-VIOL | LONGITUDINAL | +0.0185 [-0.0157, +0.0679] ns | +0.0017 [+0.0000, +0.0053] ns | +0.0168 [-0.0157, +0.0625] ns |
| DK-HW | LONGITUDINAL | -0.2387 [-0.6291, +0.1041] ns | +0.0216 [-0.0196, +0.0740] ns | -0.2604 [-0.6426, +0.0675] ns |
| DK-TG | LONGITUDINAL | -0.0584 [-0.1224, +0.0001] ns | +0.0004 [-0.0016, +0.0023] ns | -0.0588 [-0.1226, -0.0011] SB |
| heading MAE | LATERAL | -3.4733 [-4.2287, -2.7806] SB | -0.0212 [-0.0851, +0.0198] ns | -3.4520 [-4.2140, -2.7529] SB |
| yaw-rate MAE | LATERAL | -1.6256 [-4.0819, +0.7241] ns | -0.2493 [-0.7410, +0.0604] ns | -1.3762 [-3.9121, +1.0259] ns |
| curvature MAE | LATERAL | +0.0551 [+0.0347, +0.0789] SW | -0.0027 [-0.0057, -0.0003] SB | +0.0578 [+0.0364, +0.0821] SW |
| cross-track MAE | LATERAL | -0.9841 [-1.1239, -0.8516] SB | +0.0168 [+0.0071, +0.0282] SW | -1.0009 [-1.1468, -0.8671] SB |
| L1 lateral-accel MAE | LATERAL | -2.6583 [-3.9757, -1.5472] SB | +0.0744 [-0.0767, +0.1995] ns | -2.7327 [-4.0998, -1.5826] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0338 [+0.0143, +0.0598] SW | -0.0045 [-0.0106, -0.0011] SB | +0.0325 [+0.0135, +0.0590] SW |
| TAC-LAT | TACTICAL | +0.2063 [+0.1675, +0.2476] SB | +0.0025 [-0.0017, +0.0071] ns | +0.2038 [+0.1639, +0.2450] SB |
| TAC-LON | TACTICAL | +0.1697 [+0.1110, +0.2294] SB | -0.0054 [-0.0288, +0.0167] ns | +0.1751 [+0.1138, +0.2389] SB |
| S-POS@30 | STRATEGIC | -1.6616 [-2.1808, -1.1988] SB | -0.0223 [-0.0482, -0.0019] SB | -1.6443 [-2.1654, -1.1864] SB |
| S-POS@50 | STRATEGIC | -4.6167 [-5.8564, -3.5448] SB | -0.0465 [-0.0932, -0.0080] SB | -4.5745 [-5.8089, -3.5098] SB |
| S-POS@80 | STRATEGIC | -9.9025 [-12.2090, -7.8287] SB | -0.0774 [-0.1578, -0.0139] SB | -9.8284 [-12.1173, -7.7691] SB |
| M1 crawl share | crawl | +0.0478 [+0.0297, +0.0698] SW | -0.0010 [-0.0021, -0.0001] SB | +0.0483 [+0.0302, +0.0698] SW |

### lonfloor DAG-OACC s1

* **PRIMARY ADE6 (m)**: X - P10 -1.1740 [-1.3783, -0.9723] SB (D1: **True**); X - Y +0.0162 [+0.0008, +0.0339] SW (guard: **True**); Y - P10 -1.1903 [-1.3986, -0.9782] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE, yaw-rate MAE. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 14 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': False, 'M1 crawl share': True}. **Non-crawl rows separated-worse on X - Y**: ADE6, accel MAE, cross-track MAE, speed MAE. Rows separated-worse on X - Y, all 19: 4.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1740 [-1.3783, -0.9723] SB | +0.0162 [+0.0008, +0.0339] SW | -1.1903 [-1.3986, -0.9782] SB |
| speed MAE | LONGITUDINAL | -0.7933 [-1.0043, -0.6047] SB | +0.0208 [+0.0029, +0.0413] SW | -0.8141 [-1.0377, -0.6191] SB |
| along-track MAE | LONGITUDINAL | -0.6218 [-0.8057, -0.4441] SB | +0.0084 [-0.0063, +0.0250] ns | -0.6302 [-0.8196, -0.4510] SB |
| accel MAE | LONGITUDINAL | -0.9275 [-1.2871, -0.6234] SB | +0.0444 [+0.0116, +0.0839] SW | -0.9719 [-1.3431, -0.6572] SB |
| DK-VIOL | LONGITUDINAL | +0.0000 [-0.0181, +0.0182] ns | +0.0000 [-0.0079, +0.0082] ns | +0.0000 [-0.0196, +0.0212] ns |
| DK-HW | LONGITUDINAL | -0.4764 [-0.8641, -0.1709] SB | +0.0356 [-0.0102, +0.0881] ns | -0.5171 [-0.9057, -0.2118] SB |
| DK-TG | LONGITUDINAL | -0.0792 [-0.1461, -0.0215] SB | +0.0049 [-0.0029, +0.0141] ns | -0.0869 [-0.1542, -0.0278] SB |
| heading MAE | LATERAL | -2.9351 [-3.6198, -2.2706] SB | -0.0416 [-0.1259, +0.0344] ns | -2.8936 [-3.6062, -2.1968] SB |
| yaw-rate MAE | LATERAL | +3.6119 [+0.8296, +6.5813] SW | -0.5293 [-1.1732, -0.0224] SB | +4.1412 [+1.1665, +7.3821] SW |
| curvature MAE | LATERAL | +0.1144 [+0.0721, +0.1649] SW | -0.0068 [-0.0144, -0.0013] SB | +0.1212 [+0.0757, +0.1729] SW |
| cross-track MAE | LATERAL | -1.0078 [-1.1520, -0.8651] SB | +0.0139 [+0.0062, +0.0221] SW | -1.0217 [-1.1686, -0.8803] SB |
| L1 lateral-accel MAE | LATERAL | -2.5210 [-3.7173, -1.4836] SB | +0.0873 [-0.0399, +0.2246] ns | -2.6082 [-3.8410, -1.5151] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0451 [+0.0241, +0.0697] SW | -0.0054 [-0.0138, +0.0009] ns | +0.0505 [+0.0269, +0.0791] SW |
| TAC-LAT | TACTICAL | +0.2159 [+0.1755, +0.2568] SB | +0.0033 [-0.0008, +0.0075] ns | +0.2126 [+0.1709, +0.2540] SB |
| TAC-LON | TACTICAL | +0.1638 [+0.1012, +0.2232] SB | +0.0000 [-0.0038, +0.0038] ns | +0.1638 [+0.1021, +0.2225] SB |
| S-POS@30 | STRATEGIC | -1.7479 [-2.2293, -1.3357] SB | -0.0200 [-0.0404, -0.0028] SB | -1.7416 [-2.2209, -1.3273] SB |
| S-POS@50 | STRATEGIC | -4.7740 [-5.8823, -3.7638] SB | -0.0399 [-0.0778, -0.0072] SB | -4.7600 [-5.8592, -3.7484] SB |
| S-POS@80 | STRATEGIC | -10.3271 [-12.3758, -8.3537] SB | -0.0517 [-0.1134, -0.0022] SB | -10.2999 [-12.3713, -8.3307] SB |
| M1 crawl share | crawl | +0.0544 [+0.0332, +0.0804] SW | -0.0021 [-0.0044, -0.0002] SB | +0.0565 [+0.0347, +0.0829] SW |

### lonfloor SOUP s0

* **PRIMARY ADE6 (m)**: X - P10 -1.1932 [-1.3990, -1.0013] SB (D1: **True**); X - Y -0.0004 [-0.0220, +0.0202] ns (guard: **True**); Y - P10 -1.1928 [-1.4078, -0.9880] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 13 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}. **Non-crawl rows separated-worse on X - Y**: accel MAE, cross-track MAE. Rows separated-worse on X - Y, all 19: 2.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1932 [-1.3990, -1.0013] SB | -0.0004 [-0.0220, +0.0202] ns | -1.1928 [-1.4078, -0.9880] SB |
| speed MAE | LONGITUDINAL | -0.7755 [-0.9792, -0.5928] SB | +0.0062 [-0.0113, +0.0241] ns | -0.7817 [-0.9919, -0.5956] SB |
| along-track MAE | LONGITUDINAL | -0.6314 [-0.8081, -0.4695] SB | -0.0023 [-0.0250, +0.0194] ns | -0.6291 [-0.8126, -0.4582] SB |
| accel MAE | LONGITUDINAL | -0.8506 [-1.2228, -0.5350] SB | +0.0285 [+0.0046, +0.0559] SW | -0.8791 [-1.2581, -0.5583] SB |
| DK-VIOL | LONGITUDINAL | +0.0067 [-0.0196, +0.0346] ns | +0.0000 [+0.0000, +0.0000] ns | +0.0067 [-0.0196, +0.0346] ns |
| DK-HW | LONGITUDINAL | -0.1840 [-0.5759, +0.1692] ns | +0.0135 [-0.0304, +0.0594] ns | -0.2040 [-0.5885, +0.1367] ns |
| DK-TG | LONGITUDINAL | -0.0564 [-0.1185, -0.0021] SB | -0.0004 [-0.0048, +0.0033] ns | -0.0560 [-0.1173, -0.0018] SB |
| heading MAE | LATERAL | -3.1733 [-3.9455, -2.4462] SB | -0.2119 [-0.3885, -0.0776] SB | -2.9614 [-3.7496, -2.1787] SB |
| yaw-rate MAE | LATERAL | +0.5336 [-2.0636, +3.0625] ns | -1.3928 [-2.4141, -0.6074] SB | +1.9270 [-1.0592, +5.0043] ns |
| curvature MAE | LATERAL | +0.0746 [+0.0477, +0.1057] SW | -0.0173 [-0.0301, -0.0073] SB | +0.0919 [+0.0584, +0.1301] SW |
| cross-track MAE | LATERAL | -1.0202 [-1.1662, -0.8897] SB | +0.0063 [+0.0008, +0.0118] SW | -1.0265 [-1.1729, -0.8943] SB |
| L1 lateral-accel MAE | LATERAL | -2.6122 [-3.9407, -1.5056] SB | +0.0298 [-0.0320, +0.1133] ns | -2.6420 [-4.0010, -1.5149] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0421 [+0.0199, +0.0748] SW | -0.0207 [-0.0443, -0.0059] SB | +0.0628 [+0.0279, +0.1076] SW |
| TAC-LAT | TACTICAL | +0.1959 [+0.1548, +0.2372] SB | +0.0113 [+0.0054, +0.0179] SB | +0.1847 [+0.1417, +0.2274] SB |
| TAC-LON | TACTICAL | +0.1484 [+0.0846, +0.2146] SB | +0.0017 [-0.0129, +0.0138] ns | +0.1467 [+0.0792, +0.2138] SB |
| S-POS@30 | STRATEGIC | -1.6325 [-2.1698, -1.1584] SB | -0.0634 [-0.1188, -0.0195] SB | -1.6062 [-2.1525, -1.1336] SB |
| S-POS@50 | STRATEGIC | -4.6226 [-5.8819, -3.5114] SB | -0.1204 [-0.2147, -0.0418] SB | -4.5825 [-5.8449, -3.4774] SB |
| S-POS@80 | STRATEGIC | -10.2356 [-12.6289, -8.1073] SB | -0.1988 [-0.3629, -0.0626] SB | -10.1918 [-12.6081, -8.0751] SB |
| M1 crawl share | crawl | +0.0636 [+0.0408, +0.0891] SW | -0.0031 [-0.0051, -0.0014] SB | +0.0666 [+0.0428, +0.0927] SW |

### lonfloor SOUP s1

* **PRIMARY ADE6 (m)**: X - P10 -1.1791 [-1.3786, -0.9808] SB (D1: **True**); X - Y +0.0104 [-0.0061, +0.0302] ns (guard: **True**); Y - P10 -1.1895 [-1.3938, -0.9847] SB.
* **D2**: **False**; separated-worse rows (X - P10): L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 14 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}. **Non-crawl rows separated-worse on X - Y**: L1 lateral-accel MAE. Rows separated-worse on X - Y, all 19: 1.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1791 [-1.3786, -0.9808] SB | +0.0104 [-0.0061, +0.0302] ns | -1.1895 [-1.3938, -0.9847] SB |
| speed MAE | LONGITUDINAL | -0.7618 [-0.9676, -0.5815] SB | +0.0119 [-0.0130, +0.0386] ns | -0.7737 [-0.9864, -0.5882] SB |
| along-track MAE | LONGITUDINAL | -0.6184 [-0.7965, -0.4438] SB | +0.0076 [-0.0067, +0.0241] ns | -0.6260 [-0.8049, -0.4497] SB |
| accel MAE | LONGITUDINAL | -0.8352 [-1.1797, -0.5404] SB | -0.0388 [-0.1138, +0.0279] ns | -0.7964 [-1.1397, -0.4913] SB |
| DK-VIOL | LONGITUDINAL | -0.0050 [-0.0257, +0.0145] ns | +0.0017 [-0.0035, +0.0079] ns | -0.0067 [-0.0265, +0.0122] ns |
| DK-HW | LONGITUDINAL | -0.5317 [-0.9123, -0.2121] SB | +0.0163 [-0.0264, +0.0556] ns | -0.5481 [-0.9299, -0.2218] SB |
| DK-TG | LONGITUDINAL | -0.0826 [-0.1432, -0.0326] SB | +0.0037 [-0.0040, +0.0127] ns | -0.0863 [-0.1467, -0.0356] SB |
| heading MAE | LATERAL | -3.0210 [-3.6916, -2.4171] SB | -0.0719 [-0.1576, +0.0105] ns | -2.9491 [-3.6329, -2.3131] SB |
| yaw-rate MAE | LATERAL | +2.3673 [-0.2707, +5.0679] ns | -0.7268 [-1.3599, -0.2078] SB | +3.0944 [+0.2216, +6.2120] SW |
| curvature MAE | LATERAL | +0.0983 [+0.0599, +0.1423] SW | -0.0084 [-0.0143, -0.0036] SB | +0.1067 [+0.0666, +0.1533] SW |
| cross-track MAE | LATERAL | -1.0161 [-1.1570, -0.8800] SB | +0.0028 [-0.0071, +0.0128] ns | -1.0189 [-1.1607, -0.8829] SB |
| L1 lateral-accel MAE | LATERAL | -2.4548 [-3.7088, -1.3805] SB | +0.1196 [+0.0359, +0.2211] SW | -2.5744 [-3.8473, -1.4601] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0374 [+0.0170, +0.0626] SW | -0.0080 [-0.0151, -0.0025] SB | +0.0454 [+0.0224, +0.0739] SW |
| TAC-LAT | TACTICAL | +0.2188 [+0.1798, +0.2590] SB | +0.0050 [-0.0008, +0.0109] ns | +0.2138 [+0.1747, +0.2548] SB |
| TAC-LON | TACTICAL | +0.1567 [+0.0959, +0.2159] SB | -0.0017 [-0.0063, +0.0033] ns | +0.1584 [+0.0975, +0.2172] SB |
| S-POS@30 | STRATEGIC | -1.7949 [-2.2559, -1.3964] SB | -0.0238 [-0.0472, -0.0054] SB | -1.7972 [-2.2609, -1.3948] SB |
| S-POS@50 | STRATEGIC | -5.0102 [-6.1341, -3.9646] SB | -0.0423 [-0.0841, -0.0090] SB | -5.0252 [-6.1549, -3.9713] SB |
| S-POS@80 | STRATEGIC | -10.7322 [-12.8907, -8.6722] SB | -0.0557 [-0.1242, -0.0027] SB | -10.7619 [-12.9488, -8.6791] SB |
| M1 crawl share | crawl | +0.0538 [+0.0335, +0.0781] SW | -0.0038 [-0.0066, -0.0017] SB | +0.0576 [+0.0364, +0.0823] SW |

### kbox2lf DAG-OACC s0

* **PRIMARY ADE6 (m)**: X - P10 -1.0840 [-1.2791, -0.9024] SB (D1: **True**); X - Y +0.1716 [+0.1084, +0.2466] SW (guard: **False**); Y - P10 -1.2556 [-1.4503, -1.0678] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 15 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, S-POS@50, S-POS@80, along-track MAE, cross-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 7.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.0840 [-1.2791, -0.9024] SB | +0.1716 [+0.1084, +0.2466] SW | -1.2556 [-1.4503, -1.0678] SB |
| speed MAE | LONGITUDINAL | -0.8299 [-1.0338, -0.6493] SB | +0.0031 [-0.0420, +0.0424] ns | -0.8330 [-1.0377, -0.6583] SB |
| along-track MAE | LONGITUDINAL | -0.5747 [-0.7358, -0.4209] SB | +0.1445 [+0.0826, +0.2180] SW | -0.7192 [-0.8806, -0.5627] SB |
| accel MAE | LONGITUDINAL | -1.0303 [-1.4252, -0.6959] SB | -0.0913 [-0.2005, -0.0030] SB | -0.9390 [-1.3078, -0.6221] SB |
| DK-VIOL | LONGITUDINAL | +0.0252 [-0.0119, +0.0747] ns | +0.0084 [-0.0018, +0.0234] ns | +0.0168 [-0.0157, +0.0625] ns |
| DK-HW | LONGITUDINAL | -0.3922 [-0.7778, -0.0609] SB | -0.1469 [-0.3303, -0.0052] SB | -0.2604 [-0.6426, +0.0675] ns |
| DK-TG | LONGITUDINAL | -0.1213 [-0.2048, -0.0537] SB | -0.0563 [-0.1394, -0.0026] SB | -0.0588 [-0.1226, -0.0011] SB |
| heading MAE | LATERAL | -0.9696 [-2.1665, +0.4940] ns | +2.4824 [+1.0110, +4.2989] SW | -3.4520 [-4.2140, -2.7529] SB |
| yaw-rate MAE | LATERAL | -11.9863 [-16.7813, -7.8365] SB | -10.6592 [-14.4463, -7.3910] SB | -1.3762 [-3.9121, +1.0259] ns |
| curvature MAE | LATERAL | -0.0312 [-0.0458, -0.0187] SB | -0.0907 [-0.1266, -0.0610] SB | +0.0578 [+0.0364, +0.0821] SW |
| cross-track MAE | LATERAL | -0.9375 [-1.0838, -0.7961] SB | +0.0635 [+0.0265, +0.1040] SW | -1.0009 [-1.1468, -0.8671] SB |
| L1 lateral-accel MAE | LATERAL | -3.4765 [-5.0088, -2.1969] SB | -0.7440 [-1.0839, -0.4705] SB | -2.7327 [-4.0998, -1.5826] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0002 [-0.0034, +0.0036] ns | -0.0305 [-0.0540, -0.0145] SB | +0.0325 [+0.0135, +0.0590] SW |
| TAC-LAT | TACTICAL | +0.2105 [+0.1694, +0.2518] SB | +0.0067 [-0.0088, +0.0208] ns | +0.2038 [+0.1639, +0.2450] SB |
| TAC-LON | TACTICAL | +0.1726 [+0.1129, +0.2343] SB | -0.0025 [-0.0271, +0.0209] ns | +0.1751 [+0.1138, +0.2389] SB |
| S-POS@30 | STRATEGIC | -0.9713 [-1.2854, -0.6651] SB | +0.5441 [+0.2175, +0.9748] SW | -1.6443 [-2.1654, -1.1864] SB |
| S-POS@50 | STRATEGIC | -3.2542 [-4.1229, -2.4472] SB | +0.8283 [+0.2638, +1.5755] SW | -4.5745 [-5.8089, -3.5098] SB |
| S-POS@80 | STRATEGIC | -8.1106 [-10.0383, -6.2614] SB | +1.1543 [+0.1931, +2.4329] SW | -9.8284 [-12.1173, -7.7691] SB |
| M1 crawl share | crawl | +0.0456 [+0.0276, +0.0666] SW | -0.0029 [-0.0064, +0.0001] ns | +0.0483 [+0.0302, +0.0698] SW |

### kbox2lf DAG-OACC s1

* **PRIMARY ADE6 (m)**: X - P10 -1.0070 [-1.2070, -0.8059] SB (D1: **True**); X - Y +0.1833 [+0.1131, +0.2624] SW (guard: **False**); Y - P10 -1.1903 [-1.3986, -0.9782] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 15 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, S-POS@50, S-POS@80, along-track MAE, cross-track MAE, heading MAE, speed MAE. Rows separated-worse on X - Y, all 19: 8.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.0070 [-1.2070, -0.8059] SB | +0.1833 [+0.1131, +0.2624] SW | -1.1903 [-1.3986, -0.9782] SB |
| speed MAE | LONGITUDINAL | -0.7802 [-0.9789, -0.5985] SB | +0.0339 [+0.0073, +0.0689] SW | -0.8141 [-1.0377, -0.6191] SB |
| along-track MAE | LONGITUDINAL | -0.4888 [-0.6683, -0.3169] SB | +0.1414 [+0.0760, +0.2167] SW | -0.6302 [-0.8196, -0.4510] SB |
| accel MAE | LONGITUDINAL | -0.9761 [-1.3498, -0.6543] SB | -0.0042 [-0.0706, +0.0723] ns | -0.9719 [-1.3431, -0.6572] SB |
| DK-VIOL | LONGITUDINAL | -0.0017 [-0.0219, +0.0179] ns | -0.0017 [-0.0161, +0.0142] ns | +0.0000 [-0.0196, +0.0212] ns |
| DK-HW | LONGITUDINAL | -0.5569 [-0.9589, -0.2478] SB | -0.1025 [-0.2824, +0.0376] ns | -0.5171 [-0.9057, -0.2118] SB |
| DK-TG | LONGITUDINAL | -0.1281 [-0.2123, -0.0625] SB | -0.0413 [-0.1230, +0.0211] ns | -0.0869 [-0.1542, -0.0278] SB |
| heading MAE | LATERAL | -0.7685 [-2.0074, +0.7010] ns | +2.1251 [+0.7690, +3.7432] SW | -2.8936 [-3.6062, -2.1968] SB |
| yaw-rate MAE | LATERAL | -11.6746 [-16.6241, -7.5937] SB | -15.8158 [-22.1003, -10.5251] SB | +4.1412 [+1.1665, +7.3821] SW |
| curvature MAE | LATERAL | -0.0264 [-0.0406, -0.0138] SB | -0.1476 [-0.2060, -0.0965] SB | +0.1212 [+0.0757, +0.1729] SW |
| cross-track MAE | LATERAL | -0.9229 [-1.0739, -0.7790] SB | +0.0988 [+0.0617, +0.1423] SW | -1.0217 [-1.1686, -0.8803] SB |
| L1 lateral-accel MAE | LATERAL | -3.3126 [-4.8070, -2.0515] SB | -0.7044 [-1.0150, -0.4371] SB | -2.6082 [-3.8410, -1.5151] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0044 [-0.0010, +0.0109] ns | -0.0461 [-0.0698, -0.0268] SB | +0.0505 [+0.0269, +0.0791] SW |
| TAC-LAT | TACTICAL | +0.2097 [+0.1712, +0.2501] SB | -0.0029 [-0.0146, +0.0075] ns | +0.2126 [+0.1709, +0.2540] SB |
| TAC-LON | TACTICAL | +0.1626 [+0.1005, +0.2212] SB | -0.0013 [-0.0083, +0.0071] ns | +0.1638 [+0.1021, +0.2225] SB |
| S-POS@30 | STRATEGIC | -1.2842 [-1.6084, -0.9813] SB | +0.3847 [+0.1419, +0.6644] SW | -1.7416 [-2.2209, -1.3273] SB |
| S-POS@50 | STRATEGIC | -3.8931 [-4.7460, -3.0842] SB | +0.5597 [+0.1507, +1.0069] SW | -4.7600 [-5.8592, -3.7484] SB |
| S-POS@80 | STRATEGIC | -9.5483 [-11.6001, -7.6572] SB | +0.7643 [+0.1019, +1.4774] SW | -10.2999 [-12.3713, -8.3307] SB |
| M1 crawl share | crawl | +0.0564 [+0.0347, +0.0830] SW | -0.0001 [-0.0033, +0.0030] ns | +0.0565 [+0.0347, +0.0829] SW |

### kbox2lf SOUP s0

* **PRIMARY ADE6 (m)**: X - P10 -1.0513 [-1.2587, -0.8587] SB (D1: **True**); X - Y +0.1415 [+0.0772, +0.2159] SW (guard: **False**); Y - P10 -1.1928 [-1.4078, -0.9880] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 15 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, S-POS@50, S-POS@80, along-track MAE, cross-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 7.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.0513 [-1.2587, -0.8587] SB | +0.1415 [+0.0772, +0.2159] SW | -1.1928 [-1.4078, -0.9880] SB |
| speed MAE | LONGITUDINAL | -0.7792 [-0.9788, -0.6030] SB | +0.0025 [-0.0243, +0.0308] ns | -0.7817 [-0.9919, -0.5956] SB |
| along-track MAE | LONGITUDINAL | -0.5123 [-0.6890, -0.3458] SB | +0.1168 [+0.0540, +0.1886] SW | -0.6291 [-0.8126, -0.4582] SB |
| accel MAE | LONGITUDINAL | -0.9026 [-1.2723, -0.5891] SB | -0.0235 [-0.0823, +0.0367] ns | -0.8791 [-1.2581, -0.5583] SB |
| DK-VIOL | LONGITUDINAL | +0.0067 [-0.0223, +0.0374] ns | +0.0000 [-0.0105, +0.0142] ns | +0.0067 [-0.0196, +0.0346] ns |
| DK-HW | LONGITUDINAL | -0.3936 [-0.7803, -0.0879] SB | -0.0772 [-0.1951, +0.0411] ns | -0.2040 [-0.5885, +0.1367] ns |
| DK-TG | LONGITUDINAL | -0.1350 [-0.2203, -0.0660] SB | -0.0618 [-0.1340, -0.0030] SB | -0.0560 [-0.1173, -0.0018] SB |
| heading MAE | LATERAL | -0.5987 [-1.8952, +0.9925] ns | +2.3627 [+0.8324, +4.1529] SW | -2.9614 [-3.7496, -2.1787] SB |
| yaw-rate MAE | LATERAL | -11.3077 [-16.1492, -7.1669] SB | -13.2805 [-17.8958, -9.3347] SB | +1.9270 [-1.0592, +5.0043] ns |
| curvature MAE | LATERAL | -0.0252 [-0.0397, -0.0124] SB | -0.1187 [-0.1623, -0.0805] SB | +0.0919 [+0.0584, +0.1301] SW |
| cross-track MAE | LATERAL | -0.9694 [-1.1216, -0.8240] SB | +0.0571 [+0.0227, +0.0970] SW | -1.0265 [-1.1729, -0.8943] SB |
| L1 lateral-accel MAE | LATERAL | -3.4259 [-4.9403, -2.1334] SB | -0.7840 [-1.1272, -0.5148] SB | -2.6420 [-4.0010, -1.5149] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0027 [-0.0014, +0.0075] ns | -0.0601 [-0.1019, -0.0285] SB | +0.0628 [+0.0279, +0.1076] SW |
| TAC-LAT | TACTICAL | +0.2059 [+0.1647, +0.2465] SB | +0.0213 [+0.0063, +0.0367] SB | +0.1847 [+0.1417, +0.2274] SB |
| TAC-LON | TACTICAL | +0.1492 [+0.0846, +0.2151] SB | +0.0025 [-0.0146, +0.0167] ns | +0.1467 [+0.0792, +0.2138] SB |
| S-POS@30 | STRATEGIC | -0.9011 [-1.2212, -0.6143] SB | +0.6834 [+0.2754, +1.1757] SW | -1.6062 [-2.1525, -1.1336] SB |
| S-POS@50 | STRATEGIC | -3.1272 [-3.9427, -2.3896] SB | +1.0889 [+0.3722, +1.9525] SW | -4.5825 [-5.8449, -3.4774] SB |
| S-POS@80 | STRATEGIC | -8.1809 [-10.0690, -6.4237] SB | +1.5376 [+0.3983, +2.8757] SW | -10.1918 [-12.6081, -8.0751] SB |
| M1 crawl share | crawl | +0.0614 [+0.0394, +0.0864] SW | -0.0057 [-0.0093, -0.0027] SB | +0.0666 [+0.0428, +0.0927] SW |

### kbox2lf SOUP s1

* **PRIMARY ADE6 (m)**: X - P10 -1.0255 [-1.2286, -0.8310] SB (D1: **True**); X - Y +0.1640 [+0.0960, +0.2420] SW (guard: **False**); Y - P10 -1.1895 [-1.3938, -0.9847] SB.
* **D2**: **False**; separated-worse rows (X - P10): M1 crawl share. **D3**: {'LONGITUDINAL': True, 'LATERAL': True, 'TACTICAL': True, 'STRATEGIC': True}; separated-better rows: 15 of 19.
* **Crawl rows X - Y separated-better**: {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}. **Non-crawl rows separated-worse on X - Y**: ADE6, S-POS@30, S-POS@50, S-POS@80, along-track MAE, cross-track MAE, heading MAE. Rows separated-worse on X - Y, all 19: 7.

| row | family | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|
| ADE6 | PRIMARY | -1.0255 [-1.2286, -0.8310] SB | +0.1640 [+0.0960, +0.2420] SW | -1.1895 [-1.3938, -0.9847] SB |
| speed MAE | LONGITUDINAL | -0.7757 [-0.9716, -0.6002] SB | -0.0021 [-0.0393, +0.0389] ns | -0.7737 [-0.9864, -0.5882] SB |
| along-track MAE | LONGITUDINAL | -0.4877 [-0.6637, -0.3229] SB | +0.1383 [+0.0772, +0.2101] SW | -0.6260 [-0.8049, -0.4497] SB |
| accel MAE | LONGITUDINAL | -0.9865 [-1.3599, -0.6695] SB | -0.1900 [-0.3307, -0.0673] SB | -0.7964 [-1.1397, -0.4913] SB |
| DK-VIOL | LONGITUDINAL | -0.0034 [-0.0242, +0.0176] ns | +0.0034 [-0.0031, +0.0107] ns | -0.0067 [-0.0265, +0.0122] ns |
| DK-HW | LONGITUDINAL | -0.5947 [-0.9836, -0.2806] SB | -0.0288 [-0.1213, +0.0587] ns | -0.5481 [-0.9299, -0.2218] SB |
| DK-TG | LONGITUDINAL | -0.1438 [-0.2373, -0.0689] SB | -0.0555 [-0.1529, +0.0175] ns | -0.0863 [-0.1467, -0.0356] SB |
| heading MAE | LATERAL | -0.9389 [-2.1060, +0.4840] ns | +2.0102 [+0.7887, +3.4941] SW | -2.9491 [-3.6329, -2.3131] SB |
| yaw-rate MAE | LATERAL | -11.7091 [-16.7190, -7.5432] SB | -14.8492 [-20.3399, -10.0167] SB | +3.0944 [+0.2216, +6.2120] SW |
| curvature MAE | LATERAL | -0.0271 [-0.0417, -0.0146] SB | -0.1354 [-0.1895, -0.0883] SB | +0.1067 [+0.0666, +0.1533] SW |
| cross-track MAE | LATERAL | -0.9529 [-1.1024, -0.8103] SB | +0.0660 [+0.0299, +0.1090] SW | -1.0189 [-1.1607, -0.8829] SB |
| L1 lateral-accel MAE | LATERAL | -3.3886 [-4.9043, -2.1069] SB | -0.8143 [-1.1703, -0.5002] SB | -2.5744 [-3.8473, -1.4601] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0036 [-0.0014, +0.0093] ns | -0.0418 [-0.0666, -0.0224] SB | +0.0454 [+0.0224, +0.0739] SW |
| TAC-LAT | TACTICAL | +0.2105 [+0.1719, +0.2511] SB | -0.0033 [-0.0146, +0.0079] ns | +0.2138 [+0.1747, +0.2548] SB |
| TAC-LON | TACTICAL | +0.1588 [+0.0980, +0.2188] SB | +0.0004 [-0.0071, +0.0083] ns | +0.1584 [+0.0975, +0.2172] SB |
| S-POS@30 | STRATEGIC | -1.3204 [-1.6483, -1.0285] SB | +0.4578 [+0.2085, +0.7443] SW | -1.7972 [-2.2609, -1.3948] SB |
| S-POS@50 | STRATEGIC | -3.9373 [-4.8618, -3.1094] SB | +0.6803 [+0.2909, +1.1098] SW | -5.0252 [-6.1549, -3.9713] SB |
| S-POS@80 | STRATEGIC | -9.5818 [-11.7025, -7.6011] SB | +0.9170 [+0.3036, +1.5968] SW | -10.7619 [-12.9488, -8.6791] SB |
| M1 crawl share | crawl | +0.0570 [+0.0357, +0.0825] SW | -0.0006 [-0.0051, +0.0040] ns | +0.0576 [+0.0364, +0.0823] SW |

## 4. FEAS2's kbox beside the three FEAS3 levers (X - Y, point and mark; `raw/FEAS2_T1.json`, same Y, same windows)

`kbox` = `project_feasible(a_max=inf, kappa_max=0.2, mu=None, clamp_entry=False)` (module default min_speed 0.5); `kbox2` = the same with min_speed 2.0.

### DAG-OACC s0, X - Y

| row | kbox (FEAS2) | kbox2 | lonfloor | kbox2lf |
|---|---|---|---|---|
| ADE6 | +0.0230 ns | +0.1283 SW | +0.0389 SW | +0.1716 SW |
| heading MAE | +1.7413 SW | +2.4906 SW | -0.0212 ns | +2.4824 SW |
| S-POS@30 | +0.5048 SW | +0.5542 SW | -0.0223 SB | +0.5441 SW |
| S-POS@50 | +0.7883 SW | +0.8515 SW | -0.0465 SB | +0.8283 SW |
| S-POS@80 | +1.0915 SW | +1.1924 SW | -0.0774 SB | +1.1543 SW |
| along-track MAE | +0.0373 ns | +0.1051 SW | +0.0318 SW | +0.1445 SW |
| cross-track MAE | -0.0117 ns | +0.0508 SW | +0.0168 SW | +0.0635 SW |
| speed MAE | -0.0878 SB | -0.0325 SB | +0.0421 SW | +0.0031 ns |
| accel MAE | -0.2681 SB | -0.1260 SB | +0.0277 SW | -0.0913 SB |
| curvature MAE | -0.0948 SB | -0.0898 SB | -0.0027 SB | -0.0907 SB |
| L2 curvature MAE (GT>=5.2) | -0.0293 SB | -0.0301 SB | -0.0045 SB | -0.0305 SB |
| yaw-rate MAE | -11.4947 SB | -10.5775 SB | -0.2493 ns | -10.6592 SB |
| M1 crawl share | +0.0040 SW | -0.0020 ns | -0.0010 SB | -0.0029 ns |

### DAG-OACC s1, X - Y

| row | kbox (FEAS2) | kbox2 | lonfloor | kbox2lf |
|---|---|---|---|---|
| ADE6 | +0.0753 SW | +0.1442 SW | +0.0162 SW | +0.1833 SW |
| heading MAE | +1.3432 SW | +2.1163 SW | -0.0416 ns | +2.1251 SW |
| S-POS@30 | +0.2572 SW | +0.4027 SW | -0.0200 SB | +0.3847 SW |
| S-POS@50 | +0.3925 ns | +0.5957 SW | -0.0399 SB | +0.5597 SW |
| S-POS@80 | +0.5132 ns | +0.8102 SW | -0.0517 SB | +0.7643 SW |
| along-track MAE | +0.0718 SW | +0.1154 SW | +0.0084 ns | +0.1414 SW |
| cross-track MAE | +0.0268 ns | +0.0739 SW | +0.0139 SW | +0.0988 SW |
| speed MAE | -0.0310 SB | -0.0061 ns | +0.0208 SW | +0.0339 SW |
| accel MAE | -0.2025 SB | -0.0633 SB | +0.0444 SW | -0.0042 ns |
| curvature MAE | -0.1547 SB | -0.1460 SB | -0.0068 SB | -0.1476 SB |
| L2 curvature MAE (GT>=5.2) | -0.0460 SB | -0.0452 SB | -0.0054 ns | -0.0461 SB |
| yaw-rate MAE | -16.7821 SB | -15.8198 SB | -0.5293 SB | -15.8158 SB |
| M1 crawl share | +0.0036 ns | +0.0022 ns | -0.0021 SB | -0.0001 ns |

### SOUP s0, X - Y

| row | kbox (FEAS2) | kbox2 | lonfloor | kbox2lf |
|---|---|---|---|---|
| ADE6 | +0.0337 ns | +0.1342 SW | -0.0004 ns | +0.1415 SW |
| heading MAE | +1.6052 SW | +2.4643 SW | -0.2119 SB | +2.3627 SW |
| S-POS@30 | +0.6854 SW | +0.7093 SW | -0.0634 SB | +0.6834 SW |
| S-POS@50 | +1.1340 SW | +1.1454 SW | -0.1204 SB | +1.0889 SW |
| S-POS@80 | +1.6440 SW | +1.6317 SW | -0.1988 SB | +1.5376 SW |
| along-track MAE | +0.0492 SW | +0.1161 SW | -0.0023 ns | +0.1168 SW |
| cross-track MAE | -0.0137 ns | +0.0485 SW | +0.0063 SW | +0.0571 SW |
| speed MAE | -0.0622 SB | -0.0121 ns | +0.0062 ns | +0.0025 ns |
| accel MAE | -0.2139 SB | -0.0582 SB | +0.0285 SW | -0.0235 ns |
| curvature MAE | -0.1247 SB | -0.1164 SB | -0.0173 SB | -0.1187 SB |
| L2 curvature MAE (GT>=5.2) | -0.0559 SB | -0.0580 SB | -0.0207 SB | -0.0601 SB |
| yaw-rate MAE | -14.1158 SB | -13.1441 SB | -1.3928 SB | -13.2805 SB |
| M1 crawl share | -0.0006 ns | -0.0015 ns | -0.0031 SB | -0.0057 SB |

### SOUP s1, X - Y

| row | kbox (FEAS2) | kbox2 | lonfloor | kbox2lf |
|---|---|---|---|---|
| ADE6 | +0.0446 SW | +0.1287 SW | +0.0104 ns | +0.1640 SW |
| heading MAE | +1.4220 SW | +2.0449 SW | -0.0719 ns | +2.0102 SW |
| S-POS@30 | +0.3731 SW | +0.4786 SW | -0.0238 SB | +0.4578 SW |
| S-POS@50 | +0.5627 SW | +0.7178 SW | -0.0423 SB | +0.6803 SW |
| S-POS@80 | +0.7949 SW | +0.9651 SW | -0.0557 SB | +0.9170 SW |
| along-track MAE | +0.0502 SW | +0.1133 SW | +0.0076 ns | +0.1383 SW |
| cross-track MAE | -0.0001 ns | +0.0478 SW | +0.0028 ns | +0.0660 SW |
| speed MAE | -0.0779 SB | -0.0334 SB | +0.0119 ns | -0.0021 ns |
| accel MAE | -0.3680 SB | -0.2031 SB | -0.0388 ns | -0.1900 SB |
| curvature MAE | -0.1426 SB | -0.1336 SB | -0.0084 SB | -0.1354 SB |
| L2 curvature MAE (GT>=5.2) | -0.0429 SB | -0.0405 SB | -0.0080 SB | -0.0418 SB |
| yaw-rate MAE | -15.6150 SB | -14.7208 SB | -0.7268 SB | -14.8492 SB |
| M1 crawl share | +0.0024 ns | +0.0026 ns | -0.0038 SB | -0.0006 ns |

**MEASURED.** ADE6 X - Y is beyond the guard (+0.0604 m) at 1 of 4 cells for kbox and at 4 of 4 cells for kbox2. Heading MAE X - Y (extractor units), kbox / kbox2 per cell: CTRL_s0 +1.7413 / +2.4906; CTRL_s1 +1.3432 / +2.1163; SOUP_s0 +1.6052 / +2.4643; SOUP_s1 +1.4220 / +2.0449. The card's pre-check (a plan-level counterfactual on issued plans, `raw/PRECHECK_FEAS3.json`) described kbox2 as a small change to kbox: the executed knots differ on 8.8-10.5 % of ticks, by at most 0.53 m; the rollouts do not read it as small on ADE6, and heading MAE X - Y stays separated-worse at 4 of 4 kbox2 cells.

## 5. The prediction next to the result

Card (ESTIMATED, written before any X number of these arms), per lever: P(outcome) in %; read = the outcome at DAG-OACC / SOUP.

| lever | DOMINANT | HARMS | MOVED | NO EFFECT | read (DAG-OACC / SOUP) | probability the card gave the outcome read |
|---|---|---|---|---|---|---|
| kbox2 | 2 % | 70 % | 3 % | 25 % | **HARMS / HARMS** | 70 % / 70 % |
| lonfloor | 1 % | 12 % | 2 % | 85 % | **HARMS / HARMS** | 12 % / 12 % |
| kbox2lf | 2 % | 70 % | 2 % | 26 % | **HARMS / HARMS** | 70 % / 70 % |

M1 (crawl share X - Y; share of the Y - P10 excess closed = -(X - Y) / (Y - P10); negative = M1 moved the wrong way). Card: kbox2 unmoved (< 5 % of its excess); lonfloor moves < 5 % of its excess, P(separated-better at both seeds, both arms) 8 %; kbox2lf as lonfloor. P(M1 keeps D2 failing under all three arms) = 92 %. P(any arm DOMINANT) = 4 %.

| lever | cell | M1 X - Y | Y - P10 | share of excess closed | X - Y separated-better | M1 fails D2 (X - P10) |
|---|---|---|---|---|---|---|
| kbox2 | DAG-OACC s0 | -0.0020 | +0.0483 | 4.2 % | False | True |
| kbox2 | DAG-OACC s1 | +0.0022 | +0.0565 | -3.9 % | False | True |
| kbox2 | SOUP s0 | -0.0015 | +0.0666 | 2.2 % | False | True |
| kbox2 | SOUP s1 | +0.0026 | +0.0576 | -4.5 % | False | True |
| lonfloor | DAG-OACC s0 | -0.0010 | +0.0483 | 2.0 % | True | True |
| lonfloor | DAG-OACC s1 | -0.0021 | +0.0565 | 3.8 % | True | True |
| lonfloor | SOUP s0 | -0.0031 | +0.0666 | 4.6 % | True | True |
| lonfloor | SOUP s1 | -0.0038 | +0.0576 | 6.6 % | True | True |
| kbox2lf | DAG-OACC s0 | -0.0029 | +0.0483 | 5.9 % | False | True |
| kbox2lf | DAG-OACC s1 | -0.0001 | +0.0565 | 0.2 % | False | True |
| kbox2lf | SOUP s0 | -0.0057 | +0.0666 | 8.5 % | True | True |
| kbox2lf | SOUP s1 | -0.0006 | +0.0576 | 1.1 % | False | True |

* **kbox2**: M1 X - Y separated-better at 0 of 4 cells; share of the excess closed -4.5 % to 4.2 % (mean -0.5 %); below 5 % at 4 of 4 cells.
* **lonfloor**: M1 X - Y separated-better at 4 of 4 cells; share of the excess closed 2.0 % to 6.6 % (mean 4.3 %); below 5 % at 3 of 4 cells.
* **kbox2lf**: M1 X - Y separated-better at 1 of 4 cells; share of the excess closed 0.2 % to 8.5 % (mean 3.9 %); below 5 % at 2 of 4 cells.

**MEASURED:** M1 crawl share fails D2 at 12 of 12 cells (the card gave 92 % that it fails under all three arms; held). Any arm DOMINANT: 0 of 6 (the card gave 4 %). kbox2 read HARMS at both arms (card 70 %) and kbox2lf read HARMS at both arms (card 70 %). **lonfloor was predicted NO EFFECT at 85 % and read HARMS at both arms (the card gave HARMS 12 %)**; the card's 8 % event (M1 X - Y separated-better at both seeds on both arms) occurred for lonfloor (4 of 4 cells); the card's size estimate for lonfloor (< 5 % of its excess closed) held at 3 of 4 cells (largest 6.6 %). The HARMS triggers of lonfloor are rows of small magnitude (speed MAE, accel MAE, cross-track MAE, along-track MAE, ADE6 within the guard at 4 of 4 cells), listed per cell in section 1.

## 6. Disclosure (i): M1 at 1 / 2 / 3 m/s margins and the row definition (pre-registered beside the outcome; changes no outcome word)

`row def (5.2 m/s)` = the M1 row of the 19 (`rr_lib.m1_window`: per window, among GT-fast pairs, GT pair speed >= 5.2 m/s, the share whose own pair speed is < 5.2 m/s). `margin m` = per window, among the GT-fast pairs, the share whose OWN pair speed is < GT pair speed - m. Mean over windows of the share, for X, Y and P10 (P10 = the fresh Y run's own P10 floor run); X - P10, X - Y, Y - P10 with the paired bootstrap. Lower is better.

### kbox2 DAG-OACC s0

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0689 | 0.0713 | 0.0230 | +0.0460 [+0.0282, +0.0666] SW | -0.0020 [-0.0051, +0.0006] ns | +0.0483 [+0.0302, +0.0698] SW |
| margin 1 m/s | 0.1758 | 0.1786 | 0.0480 | +0.1277 [+0.1050, +0.1524] SW | -0.0025 [-0.0057, -0.0001] SB | +0.1306 [+0.1074, +0.1558] SW |
| margin 2 m/s | 0.0782 | 0.0801 | 0.0142 | +0.0640 [+0.0487, +0.0809] SW | -0.0015 [-0.0047, +0.0013] ns | +0.0659 [+0.0501, +0.0840] SW |
| margin 3 m/s | 0.0387 | 0.0398 | 0.0055 | +0.0333 [+0.0230, +0.0458] SW | -0.0006 [-0.0032, +0.0014] ns | +0.0343 [+0.0233, +0.0469] SW |

Windows 2102 (X - P10, X - Y), 2104 (Y - P10); episodes in the pool 135.

### kbox2 DAG-OACC s1

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0816 | 0.0795 | 0.0230 | +0.0587 [+0.0363, +0.0855] SW | +0.0022 [-0.0008, +0.0053] ns | +0.0565 [+0.0347, +0.0829] SW |
| margin 1 m/s | 0.1857 | 0.1833 | 0.0480 | +0.1377 [+0.1116, +0.1644] SW | +0.0024 [-0.0006, +0.0056] ns | +0.1353 [+0.1095, +0.1619] SW |
| margin 2 m/s | 0.0933 | 0.0910 | 0.0142 | +0.0792 [+0.0595, +0.1007] SW | +0.0023 [-0.0004, +0.0053] ns | +0.0768 [+0.0576, +0.0979] SW |
| margin 3 m/s | 0.0493 | 0.0481 | 0.0055 | +0.0438 [+0.0297, +0.0591] SW | +0.0011 [-0.0019, +0.0041] ns | +0.0427 [+0.0291, +0.0572] SW |

Windows 2106 (X - P10, X - Y), 2106 (Y - P10); episodes in the pool 135.

### kbox2 SOUP s0

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0881 | 0.0896 | 0.0230 | +0.0652 [+0.0418, +0.0906] SW | -0.0015 [-0.0034, +0.0003] ns | +0.0666 [+0.0428, +0.0927] SW |
| margin 1 m/s | 0.2281 | 0.2298 | 0.0480 | +0.1801 [+0.1533, +0.2077] SW | -0.0018 [-0.0041, +0.0003] ns | +0.1818 [+0.1542, +0.2094] SW |
| margin 2 m/s | 0.1110 | 0.1089 | 0.0142 | +0.0968 [+0.0756, +0.1191] SW | +0.0021 [-0.0007, +0.0051] ns | +0.0947 [+0.0740, +0.1168] SW |
| margin 3 m/s | 0.0604 | 0.0581 | 0.0055 | +0.0549 [+0.0391, +0.0714] SW | +0.0028 [-0.0004, +0.0062] ns | +0.0526 [+0.0379, +0.0690] SW |

Windows 2104 (X - P10, X - Y), 2105 (Y - P10); episodes in the pool 135.

### kbox2 SOUP s1

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0832 | 0.0806 | 0.0230 | +0.0602 [+0.0384, +0.0861] SW | +0.0026 [-0.0014, +0.0075] ns | +0.0576 [+0.0364, +0.0823] SW |
| margin 1 m/s | 0.1784 | 0.1761 | 0.0480 | +0.1304 [+0.1038, +0.1579] SW | +0.0024 [-0.0014, +0.0072] ns | +0.1281 [+0.1027, +0.1549] SW |
| margin 2 m/s | 0.0928 | 0.0916 | 0.0142 | +0.0786 [+0.0588, +0.1009] SW | +0.0012 [-0.0028, +0.0063] ns | +0.0774 [+0.0580, +0.0987] SW |
| margin 3 m/s | 0.0501 | 0.0488 | 0.0055 | +0.0446 [+0.0301, +0.0612] SW | +0.0012 [-0.0026, +0.0053] ns | +0.0434 [+0.0300, +0.0580] SW |

Windows 2106 (X - P10, X - Y), 2106 (Y - P10); episodes in the pool 135.

### lonfloor DAG-OACC s0

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0708 | 0.0713 | 0.0230 | +0.0478 [+0.0297, +0.0698] SW | -0.0010 [-0.0021, -0.0001] SB | +0.0483 [+0.0302, +0.0698] SW |
| margin 1 m/s | 0.1716 | 0.1786 | 0.0480 | +0.1236 [+0.1005, +0.1489] SW | -0.0074 [-0.0141, -0.0028] SB | +0.1306 [+0.1074, +0.1558] SW |
| margin 2 m/s | 0.0765 | 0.0801 | 0.0142 | +0.0623 [+0.0469, +0.0794] SW | -0.0041 [-0.0076, -0.0015] SB | +0.0659 [+0.0501, +0.0840] SW |
| margin 3 m/s | 0.0379 | 0.0398 | 0.0055 | +0.0325 [+0.0221, +0.0442] SW | -0.0023 [-0.0043, -0.0008] SB | +0.0343 [+0.0233, +0.0469] SW |

Windows 2104 (X - P10, X - Y), 2104 (Y - P10); episodes in the pool 135.

### lonfloor DAG-OACC s1

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0773 | 0.0795 | 0.0230 | +0.0544 [+0.0332, +0.0804] SW | -0.0021 [-0.0044, -0.0002] SB | +0.0565 [+0.0347, +0.0829] SW |
| margin 1 m/s | 0.1798 | 0.1833 | 0.0480 | +0.1318 [+0.1071, +0.1579] SW | -0.0035 [-0.0067, -0.0010] SB | +0.1353 [+0.1095, +0.1619] SW |
| margin 2 m/s | 0.0888 | 0.0910 | 0.0142 | +0.0746 [+0.0562, +0.0954] SW | -0.0023 [-0.0042, -0.0007] SB | +0.0768 [+0.0576, +0.0979] SW |
| margin 3 m/s | 0.0469 | 0.0481 | 0.0055 | +0.0414 [+0.0283, +0.0558] SW | -0.0013 [-0.0028, +0.0003] ns | +0.0427 [+0.0291, +0.0572] SW |

Windows 2106 (X - P10, X - Y), 2106 (Y - P10); episodes in the pool 135.

### lonfloor SOUP s0

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0866 | 0.0896 | 0.0230 | +0.0636 [+0.0408, +0.0891] SW | -0.0031 [-0.0051, -0.0014] SB | +0.0666 [+0.0428, +0.0927] SW |
| margin 1 m/s | 0.2209 | 0.2298 | 0.0480 | +0.1728 [+0.1466, +0.1985] SW | -0.0090 [-0.0152, -0.0045] SB | +0.1818 [+0.1542, +0.2094] SW |
| margin 2 m/s | 0.1018 | 0.1089 | 0.0142 | +0.0876 [+0.0681, +0.1078] SW | -0.0071 [-0.0115, -0.0036] SB | +0.0947 [+0.0740, +0.1168] SW |
| margin 3 m/s | 0.0520 | 0.0581 | 0.0055 | +0.0465 [+0.0338, +0.0606] SW | -0.0061 [-0.0108, -0.0026] SB | +0.0526 [+0.0379, +0.0690] SW |

Windows 2105 (X - P10, X - Y), 2105 (Y - P10); episodes in the pool 135.

### lonfloor SOUP s1

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0768 | 0.0806 | 0.0230 | +0.0538 [+0.0335, +0.0781] SW | -0.0038 [-0.0066, -0.0017] SB | +0.0576 [+0.0364, +0.0823] SW |
| margin 1 m/s | 0.1712 | 0.1761 | 0.0480 | +0.1232 [+0.0986, +0.1492] SW | -0.0048 [-0.0079, -0.0024] SB | +0.1281 [+0.1027, +0.1549] SW |
| margin 2 m/s | 0.0885 | 0.0916 | 0.0142 | +0.0743 [+0.0557, +0.0945] SW | -0.0031 [-0.0051, -0.0015] SB | +0.0774 [+0.0580, +0.0987] SW |
| margin 3 m/s | 0.0462 | 0.0488 | 0.0055 | +0.0407 [+0.0280, +0.0546] SW | -0.0026 [-0.0043, -0.0013] SB | +0.0434 [+0.0300, +0.0580] SW |

Windows 2106 (X - P10, X - Y), 2106 (Y - P10); episodes in the pool 135.

### kbox2lf DAG-OACC s0

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0685 | 0.0713 | 0.0230 | +0.0456 [+0.0276, +0.0666] SW | -0.0029 [-0.0064, +0.0001] ns | +0.0483 [+0.0302, +0.0698] SW |
| margin 1 m/s | 0.1690 | 0.1786 | 0.0480 | +0.1209 [+0.0985, +0.1445] SW | -0.0097 [-0.0173, -0.0041] SB | +0.1306 [+0.1074, +0.1558] SW |
| margin 2 m/s | 0.0755 | 0.0801 | 0.0142 | +0.0613 [+0.0460, +0.0781] SW | -0.0046 [-0.0093, -0.0008] SB | +0.0659 [+0.0501, +0.0840] SW |
| margin 3 m/s | 0.0372 | 0.0398 | 0.0055 | +0.0317 [+0.0215, +0.0438] SW | -0.0026 [-0.0057, +0.0000] ns | +0.0343 [+0.0233, +0.0469] SW |

Windows 2102 (X - P10, X - Y), 2104 (Y - P10); episodes in the pool 135.

### kbox2lf DAG-OACC s1

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0793 | 0.0795 | 0.0230 | +0.0564 [+0.0347, +0.0830] SW | -0.0001 [-0.0033, +0.0030] ns | +0.0565 [+0.0347, +0.0829] SW |
| margin 1 m/s | 0.1819 | 0.1833 | 0.0480 | +0.1339 [+0.1086, +0.1598] SW | -0.0014 [-0.0054, +0.0027] ns | +0.1353 [+0.1095, +0.1619] SW |
| margin 2 m/s | 0.0907 | 0.0910 | 0.0142 | +0.0765 [+0.0577, +0.0973] SW | -0.0003 [-0.0032, +0.0025] ns | +0.0768 [+0.0576, +0.0979] SW |
| margin 3 m/s | 0.0479 | 0.0481 | 0.0055 | +0.0424 [+0.0288, +0.0572] SW | -0.0003 [-0.0033, +0.0026] ns | +0.0427 [+0.0291, +0.0572] SW |

Windows 2106 (X - P10, X - Y), 2106 (Y - P10); episodes in the pool 135.

### kbox2lf SOUP s0

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0844 | 0.0896 | 0.0230 | +0.0614 [+0.0394, +0.0864] SW | -0.0057 [-0.0093, -0.0027] SB | +0.0666 [+0.0428, +0.0927] SW |
| margin 1 m/s | 0.2190 | 0.2298 | 0.0480 | +0.1710 [+0.1448, +0.1961] SW | -0.0112 [-0.0177, -0.0061] SB | +0.1818 [+0.1542, +0.2094] SW |
| margin 2 m/s | 0.1032 | 0.1089 | 0.0142 | +0.0890 [+0.0693, +0.1095] SW | -0.0061 [-0.0106, -0.0023] SB | +0.0947 [+0.0740, +0.1168] SW |
| margin 3 m/s | 0.0536 | 0.0581 | 0.0055 | +0.0481 [+0.0349, +0.0630] SW | -0.0045 [-0.0094, -0.0003] SB | +0.0526 [+0.0379, +0.0690] SW |

Windows 2105 (X - P10, X - Y), 2105 (Y - P10); episodes in the pool 135.

### kbox2lf SOUP s1

| threshold | X | Y | P10 | X - P10 | X - Y | Y - P10 |
|---|---|---|---|---|---|---|
| row def (5.2 m/s) | 0.0800 | 0.0806 | 0.0230 | +0.0570 [+0.0357, +0.0825] SW | -0.0006 [-0.0051, +0.0040] ns | +0.0576 [+0.0364, +0.0823] SW |
| margin 1 m/s | 0.1738 | 0.1761 | 0.0480 | +0.1258 [+0.1003, +0.1524] SW | -0.0022 [-0.0072, +0.0028] ns | +0.1281 [+0.1027, +0.1549] SW |
| margin 2 m/s | 0.0889 | 0.0916 | 0.0142 | +0.0747 [+0.0555, +0.0960] SW | -0.0027 [-0.0076, +0.0022] ns | +0.0774 [+0.0580, +0.0987] SW |
| margin 3 m/s | 0.0481 | 0.0488 | 0.0055 | +0.0426 [+0.0289, +0.0584] SW | -0.0007 [-0.0045, +0.0030] ns | +0.0434 [+0.0300, +0.0580] SW |

Windows 2106 (X - P10, X - Y), 2106 (Y - P10); episodes in the pool 135.

### Where M1 X - Y is separated-better, per cell (SB = separated-better; the card's THRESHOLD-LOCAL reading)

| lever | cell | 5.2 row | 1 m/s | 2 m/s | 3 m/s | THRESHOLD-LOCAL (literal: SB on 5.2 or 2, SB on neither 1 nor 3) | broader reading: SB on 5.2 or 2 but not on both 1 and 3 |
|---|---|---|---|---|---|---|---|
| kbox2 | DAG-OACC s0 | - | SB | - | - | no | no |
| kbox2 | DAG-OACC s1 | - | - | - | - | no | no |
| kbox2 | SOUP s0 | - | - | - | - | no | no |
| kbox2 | SOUP s1 | - | - | - | - | no | no |
| lonfloor | DAG-OACC s0 | SB | SB | SB | SB | no | no |
| lonfloor | DAG-OACC s1 | SB | SB | SB | - | no | yes |
| lonfloor | SOUP s0 | SB | SB | SB | SB | no | no |
| lonfloor | SOUP s1 | SB | SB | SB | SB | no | no |
| kbox2lf | DAG-OACC s0 | - | SB | SB | - | no | yes |
| kbox2lf | DAG-OACC s1 | - | - | - | - | no | no |
| kbox2lf | SOUP s0 | SB | SB | SB | SB | no | no |
| kbox2lf | SOUP s1 | - | - | - | - | no | no |

Cells at which M1 X - Y is separated-better, per lever (5.2 row / 1 m/s / 2 m/s / 3 m/s, of 4): kbox2 0 / 1 / 0 / 0; lonfloor 4 / 4 / 4 / 3; kbox2lf 1 / 2 / 2 / 1.

**Headline reading.** THRESHOLD-LOCAL under the card's wording read literally (separated-better on the 5.2 row or the 2 m/s row, and on neither the 1 m/s nor the 3 m/s row): no cell. Under the broader reading (not separated-better on both the 1 m/s and the 3 m/s row): lonfloor:CTRL_s1, kbox2lf:CTRL_s0. The card's wording admits both; both are printed, the outcome words do not change either way.

**What the margins show.** lonfloor's M1 X - Y is separated-better on the 5.2 row at 4 of 4 cells and on the 2 m/s margin at 4 of 4, so its small M1 effect is not an artefact of the 5.2 definition; the largest M1 X - Y on the 5.2 row is -0.0038 (share of the Y - P10 excess closed 2.0 %-6.6 %, section 5). The Y - P10 excess at margin 1 is larger than on the 5.2 row at every cell (0.1281-0.1818 against 0.0483-0.0666); the share of the margin-1 excess closed is 2.6 %-5.7 % for lonfloor, -1.9 %-1.9 % for kbox2 and 1.0 %-7.5 % for kbox2lf.

## 7. Disclosure (ii): the hard-braking windows (pre-registered; changes no outcome word)

Windows in the top 5 % of GT peak deceleration (peak of -(d vG / dt) over the GT pair speeds; fixed by GT alone, one set for every arm): 120 windows from 19 episodes, threshold 2.372 m/s^2 on the 2399 windows with a finite deceleration; identical at all four cells: True. X - Y on those windows, **19 episodes make the intervals wide: a null here is not evidence of safety**.

### kbox2 DAG-OACC s0

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.0136 [-0.0815, +0.1015] ns | 120 | 19 |
| S-POS@30 | +0.1053 [-0.1920, +0.5144] ns | 114 | 19 |
| S-POS@50 | +0.2582 [-0.0609, +0.7778] ns | 111 | 19 |
| S-POS@80 | +0.3874 [-0.0739, +1.1763] ns | 105 | 18 |

### kbox2 DAG-OACC s1

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.0758 [+0.0026, +0.1868] SW | 120 | 19 |
| S-POS@30 | +0.0942 [-0.1697, +0.4694] ns | 114 | 19 |
| S-POS@50 | +0.0177 [-0.6689, +0.7468] ns | 111 | 19 |
| S-POS@80 | -0.0839 [-1.2746, +1.1573] ns | 105 | 18 |

### kbox2 SOUP s0

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.0856 [+0.0257, +0.1633] SW | 120 | 19 |
| S-POS@30 | -0.0288 [-0.5894, +0.6012] ns | 114 | 19 |
| S-POS@50 | +0.1224 [-0.9044, +1.1313] ns | 111 | 19 |
| S-POS@80 | +0.1725 [-1.5969, +1.8700] ns | 105 | 18 |

### kbox2 SOUP s1

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.0224 [-0.0373, +0.0896] ns | 120 | 19 |
| S-POS@30 | +0.1852 [-0.1230, +0.5836] ns | 114 | 19 |
| S-POS@50 | +0.2531 [-0.2617, +0.8503] ns | 111 | 19 |
| S-POS@80 | +0.4036 [-0.2981, +1.2341] ns | 105 | 18 |

### lonfloor DAG-OACC s0

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.1750 [+0.0645, +0.3047] SW | 120 | 19 |
| S-POS@30 | -0.0098 [-0.1011, +0.0646] ns | 114 | 19 |
| S-POS@50 | -0.0092 [-0.1251, +0.0872] ns | 111 | 19 |
| S-POS@80 | +0.0731 [-0.0879, +0.2311] ns | 105 | 18 |

### lonfloor DAG-OACC s1

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.2261 [+0.0825, +0.3893] SW | 120 | 19 |
| S-POS@30 | -0.0767 [-0.2560, +0.0524] ns | 114 | 19 |
| S-POS@50 | -0.2054 [-0.6052, +0.0763] ns | 111 | 19 |
| S-POS@80 | -0.1917 [-0.9668, +0.3149] ns | 105 | 18 |

### lonfloor SOUP s0

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.1871 [+0.0500, +0.3399] SW | 120 | 19 |
| S-POS@30 | -0.0264 [-0.2037, +0.1462] ns | 114 | 19 |
| S-POS@50 | -0.0923 [-0.3891, +0.1690] ns | 111 | 19 |
| S-POS@80 | -0.0739 [-0.4517, +0.2996] ns | 105 | 18 |

### lonfloor SOUP s1

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.2036 [+0.0361, +0.3841] SW | 120 | 19 |
| S-POS@30 | -0.0164 [-0.1644, +0.1289] ns | 114 | 19 |
| S-POS@50 | -0.0963 [-0.4790, +0.2091] ns | 111 | 19 |
| S-POS@80 | -0.0184 [-0.8427, +0.5330] ns | 105 | 18 |

### kbox2lf DAG-OACC s0

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.1873 [+0.0325, +0.3517] SW | 120 | 19 |
| S-POS@30 | +0.1077 [-0.2170, +0.5148] ns | 114 | 19 |
| S-POS@50 | +0.2689 [-0.0889, +0.7975] ns | 111 | 19 |
| S-POS@80 | +0.4640 [-0.0526, +1.2722] ns | 105 | 18 |

### kbox2lf DAG-OACC s1

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.2826 [+0.1259, +0.4630] SW | 120 | 19 |
| S-POS@30 | +0.0274 [-0.3017, +0.4263] ns | 114 | 19 |
| S-POS@50 | -0.1721 [-0.9835, +0.6142] ns | 111 | 19 |
| S-POS@80 | -0.2517 [-1.6786, +1.0850] ns | 105 | 18 |

### kbox2lf SOUP s0

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.2627 [+0.1185, +0.4279] SW | 120 | 19 |
| S-POS@30 | -0.0431 [-0.6635, +0.6445] ns | 114 | 19 |
| S-POS@50 | +0.0490 [-1.0425, +1.1043] ns | 111 | 19 |
| S-POS@80 | +0.1295 [-1.7211, +1.8370] ns | 105 | 18 |

### kbox2lf SOUP s1

| row | X - Y on the braking set | windows | episodes |
|---|---|---|---|
| along-track MAE | +0.2472 [+0.0780, +0.4272] SW | 120 | 19 |
| S-POS@30 | +0.1754 [-0.1769, +0.5952] ns | 114 | 19 |
| S-POS@50 | +0.1644 [-0.4961, +0.8486] ns | 111 | 19 |
| S-POS@80 | +0.3936 [-0.7197, +1.3749] ns | 105 | 18 |

### Separated-worse rows on the braking set, per lever (cells of 4)

| lever | along-track MAE | S-POS@30 | S-POS@50 | S-POS@80 | outcome word | BRAKING-BLOCKED applies (MOVED or DOMINANT read) |
|---|---|---|---|---|---|---|
| kbox2 | 2 / 4 | 0 / 4 | 0 / 4 | 0 / 4 | HARMS / HARMS | no: no arm reads MOVED or DOMINANT |
| lonfloor | 4 / 4 | 0 / 4 | 0 / 4 | 0 / 4 | HARMS / HARMS | no: no arm reads MOVED or DOMINANT |
| kbox2lf | 4 / 4 | 0 / 4 | 0 / 4 | 0 / 4 | HARMS / HARMS | no: no arm reads MOVED or DOMINANT |

**Headline reading.** No arm reads MOVED or DOMINANT, so BRAKING-BLOCKED (the card's label for a MOVED or DOMINANT arm that is separated-worse on any of the four rows on this set) is not applied to any arm. Beside the outcome: along-track MAE X - Y on the braking set is separated-worse at 2 of 4 cells for kbox2, 4 of 4 for lonfloor and 4 of 4 for kbox2lf; S-POS@30/50/80 X - Y is separated-worse at 0 of 36 cells (3 rows x 4 cells x 3 levers). The one MOVED condition hidden by precedence (lonfloor on SOUP) has along-track MAE separated-worse on the braking set at both seeds (+0.1871, +0.2036 m); had that arm read MOVED the card's label would have been BRAKING-BLOCKED. This is the direction the card's pre-check anticipated for lonfloor (a floor built on a prior that does not predict braking sits above real deceleration; plan-level step-speed MAE against the logged future on the 311 top-5 % ticks of each run, plan_s -> lonfloor in m/s per run: 6.43 -> 6.47, 7.41 -> 7.45, 7.70 -> 7.76, 7.77 -> 7.83); the rollout read is on windows, the pre-check on ticks.

## 8. The reader fix (an import-path edit after the first read stopped; `READER_FIX_FEAS3.md`)

* **What failed.** The 12 rollouts finished (nfail 0); the first read stopped at rc 1 inside `disclosures()` (the card's beside checks) with `taniteval.stack_guard.StackShadowError`: `lever()` had put the arm tree's stack root (`F:/TanitAD-scratch/feas2_t1/tree_dag/stack`) on `sys.path` to import `tanitad.refs.feasible_decode`, so a second `tanitad` root preceded the pinned one when `rr_lib` imported `taniteval`; the guard refused as designed. Marker kept: `raw/FEAS3.fail.first_read` (`2026-11-02T18:18:57 read rc=1 (see logs/read_feas3.out) 1`). `logs/read_feas3.out` holds the traceback and the 12 per-cell lines the read had printed before the stop; no `FEAS3_T1.json` existed until the re-run.
* **The fix (path only).** `lever()` loads `tree_dag/stack/tanitad/refs/feasible_decode.py` by file path (`importlib.util.spec_from_file_location`) instead of adding the arm tree's stack root to `sys.path`; that file imports only `math` and `torch` and is byte-identical to the pinned tree's copy (`READER_FIX_FEAS3.md`: `diff -q`, no difference). The diff of the reader is that one hunk (3 lines replaced).
* **Reader shas.** `FROZEN_SHA_FEAS3.txt` pins `code/feas3_t1_read.py` at `7c6df0e68eeb1914c8926f3956e0f1f818ab8f0d5320c30b7d4a448379a57dc5`; the pre-fix file kept at `F:/TanitAD-scratch/feas2_t1/feas3_t1_read.pre_pathfix.py` hashes to `7c6df0e68eeb1914c8926f3956e0f1f818ab8f0d5320c30b7d4a448379a57dc5` (equal to the pin); the reader used for the re-run (current `code/feas3_t1_read.py`) hashes to `2f91cfd9cbcfbe8eaedf46b649a1afc9468636b14e21ed452b065ea12ddd0260`.
* **Nothing else moved.** No row, threshold, arm, control, outcome rule or bootstrap setting changed and no rollout was re-run. The per-cell outcome lines of the first read and of the re-run are compared line by line here: 12 of 12 identical (identical). Marker of the re-run: `raw/FEAS3.done` (`2026-11-02T18:21:57 FEAS3_T1.json written by the re-run read (READER_FIX_FEAS3.md)`).

## 9. The card's committed next arm per outcome (Rule Zero), and what the read selects

The card's lines: **HARMS on heading / S-POS at kbox2 and kbox2lf**: the decode-time box family (box, kbox, abox, kabox, kbox2) is closed as a heading / S-POS lever (the clamp acts on executed steps the logged future truly turns on); remaining levers, each with its blocker: (c) the lateral shrink `P + diag(1, alpha)(S(plan) - P)` with alpha chosen on TRAIN windows (blocked: the tool rolls out eval-141 only; whether DAgger's TRAIN-window rollout inputs exist on disk is **UNVERIFIED**); the clamp as a loss term or output layer at training time (blocked: the PI's go and compute).

* **kbox2 and kbox2lf (HARMS at 4 of 4 cells each)**: the trigger the card names, heading MAE X - Y separated-worse, is present at 4 of 4 and 4 of 4 cells, S-POS@30/50/80 at 4 of 4 and 4 of 4 (all three rows); ADE6 X - Y is beyond the guard at 4 of 4 and 4 of 4. The card's line applies: box family closed as a heading / S-POS lever; remaining levers (c) and the training-time clamp, blockers as above.
* **lonfloor (HARMS at 4 of 4 cells; MOVED condition met on SOUP)**: the card committed lines for DOMINANT, MOVED, HARMS-on-heading (kbox2 and kbox2lf) and NO EFFECT on lonfloor; it committed none for HARMS at lonfloor. The nearest committed lines are NO EFFECT-with-M1-separated-worse (partition M1's executed excess by window position, speed band and whether the preceding executed step was already below the prior) and MOVED (name the rows still separated-worse vs P10 per seed: here M1 crawl share, curvature MAE and L2 curvature MAE at every lonfloor cell, yaw-rate MAE at DAG-OACC s1). No such partition is run in this deliverable; the executed-state question was taken up by the separate package `../histswap` (ORACLE diagnostic, `../histswap/RESULTS.md`).
* **No arm is DOMINANT**, so the card's DOMINANT line (re-read on a fresh seed or window set; the PI's go for the training lever) is not triggered.

## 10. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)

* **kbox2 CTRL_s0**: X {"N": 2364, "E": -0.0303, "partition_sum_minus_E": 0.0, "n_dip_windows": 46, "n_highk_steps": 3110, "n_highk_steps_masked_in_row": 2273, "highk_share_from_steps_under_2mps": 0.9889} | Y {"N": 2364, "E": 0.0578, "partition_sum_minus_E": 0.0, "n_dip_windows": 46, "n_highk_steps": 4613, "n_highk_steps_masked_in_row": 3411, "highk_share_from_steps_under_2mps": 0.5549}
* **kbox2 CTRL_s1**: X {"N": 2365, "E": -0.0248, "partition_sum_minus_E": 0.0, "n_dip_windows": 60, "n_highk_steps": 3425, "n_highk_steps_masked_in_row": 2589, "highk_share_from_steps_under_2mps": 0.9955} | Y {"N": 2365, "E": 0.1212, "partition_sum_minus_E": 0.0, "n_dip_windows": 63, "n_highk_steps": 4962, "n_highk_steps_masked_in_row": 3704, "highk_share_from_steps_under_2mps": 0.5851}
* **kbox2 SOUP_s0**: X {"N": 2364, "E": -0.0228, "partition_sum_minus_E": 0.0, "n_dip_windows": 83, "n_highk_steps": 3577, "n_highk_steps_masked_in_row": 2721, "highk_share_from_steps_under_2mps": 1.0987} | Y {"N": 2364, "E": 0.0919, "partition_sum_minus_E": 0.0, "n_dip_windows": 74, "n_highk_steps": 5468, "n_highk_steps_masked_in_row": 4202, "highk_share_from_steps_under_2mps": 0.534}
* **kbox2 SOUP_s1**: X {"N": 2365, "E": -0.0253, "partition_sum_minus_E": 0.0, "n_dip_windows": 53, "n_highk_steps": 3536, "n_highk_steps_masked_in_row": 2698, "highk_share_from_steps_under_2mps": 1.0586} | Y {"N": 2365, "E": 0.1067, "partition_sum_minus_E": 0.0, "n_dip_windows": 55, "n_highk_steps": 5282, "n_highk_steps_masked_in_row": 3936, "highk_share_from_steps_under_2mps": 0.618}
* **lonfloor CTRL_s0**: X {"N": 2364, "E": 0.0551, "partition_sum_minus_E": -0.0, "n_dip_windows": 40, "n_highk_steps": 4552, "n_highk_steps_masked_in_row": 3368, "highk_share_from_steps_under_2mps": 0.5786} | Y {"N": 2364, "E": 0.0578, "partition_sum_minus_E": 0.0, "n_dip_windows": 46, "n_highk_steps": 4613, "n_highk_steps_masked_in_row": 3411, "highk_share_from_steps_under_2mps": 0.5549}
* **lonfloor CTRL_s1**: X {"N": 2365, "E": 0.1144, "partition_sum_minus_E": -0.0, "n_dip_windows": 53, "n_highk_steps": 4885, "n_highk_steps_masked_in_row": 3612, "highk_share_from_steps_under_2mps": 0.5935} | Y {"N": 2365, "E": 0.1212, "partition_sum_minus_E": 0.0, "n_dip_windows": 63, "n_highk_steps": 4962, "n_highk_steps_masked_in_row": 3704, "highk_share_from_steps_under_2mps": 0.5851}
* **lonfloor SOUP_s0**: X {"N": 2364, "E": 0.0746, "partition_sum_minus_E": 0.0, "n_dip_windows": 49, "n_highk_steps": 5328, "n_highk_steps_masked_in_row": 4070, "highk_share_from_steps_under_2mps": 0.5303} | Y {"N": 2364, "E": 0.0919, "partition_sum_minus_E": 0.0, "n_dip_windows": 74, "n_highk_steps": 5468, "n_highk_steps_masked_in_row": 4202, "highk_share_from_steps_under_2mps": 0.534}
* **lonfloor SOUP_s1**: X {"N": 2365, "E": 0.0983, "partition_sum_minus_E": -0.0, "n_dip_windows": 44, "n_highk_steps": 5092, "n_highk_steps_masked_in_row": 3754, "highk_share_from_steps_under_2mps": 0.6294} | Y {"N": 2365, "E": 0.1067, "partition_sum_minus_E": 0.0, "n_dip_windows": 55, "n_highk_steps": 5282, "n_highk_steps_masked_in_row": 3936, "highk_share_from_steps_under_2mps": 0.618}
* **kbox2lf CTRL_s0**: X {"N": 2364, "E": -0.0312, "partition_sum_minus_E": 0.0, "n_dip_windows": 40, "n_highk_steps": 3015, "n_highk_steps_masked_in_row": 2202, "highk_share_from_steps_under_2mps": 0.9911} | Y {"N": 2364, "E": 0.0578, "partition_sum_minus_E": 0.0, "n_dip_windows": 46, "n_highk_steps": 4613, "n_highk_steps_masked_in_row": 3411, "highk_share_from_steps_under_2mps": 0.5549}
* **kbox2lf CTRL_s1**: X {"N": 2365, "E": -0.0264, "partition_sum_minus_E": 0.0, "n_dip_windows": 51, "n_highk_steps": 3221, "n_highk_steps_masked_in_row": 2448, "highk_share_from_steps_under_2mps": 0.99} | Y {"N": 2365, "E": 0.1212, "partition_sum_minus_E": 0.0, "n_dip_windows": 63, "n_highk_steps": 4962, "n_highk_steps_masked_in_row": 3704, "highk_share_from_steps_under_2mps": 0.5851}
* **kbox2lf SOUP_s0**: X {"N": 2364, "E": -0.0252, "partition_sum_minus_E": 0.0, "n_dip_windows": 58, "n_highk_steps": 3398, "n_highk_steps_masked_in_row": 2554, "highk_share_from_steps_under_2mps": 1.0647} | Y {"N": 2364, "E": 0.0919, "partition_sum_minus_E": 0.0, "n_dip_windows": 74, "n_highk_steps": 5468, "n_highk_steps_masked_in_row": 4202, "highk_share_from_steps_under_2mps": 0.534}
* **kbox2lf SOUP_s1**: X {"N": 2365, "E": -0.0271, "partition_sum_minus_E": 0.0, "n_dip_windows": 43, "n_highk_steps": 3278, "n_highk_steps_masked_in_row": 2485, "highk_share_from_steps_under_2mps": 1.0229} | Y {"N": 2365, "E": 0.1067, "partition_sum_minus_E": 0.0, "n_dip_windows": 55, "n_highk_steps": 5282, "n_highk_steps_masked_in_row": 3936, "highk_share_from_steps_under_2mps": 0.618}

Gloss (from `gate_kappa.partition`): N = windows finite on both sides; E = the curvature MAE point (X - P10 for X, Y - P10 for Y); n_dip_windows = windows with an unwarranted speed dip; n_highk_steps = steps with |kappa| of the executed path above 0.2 1/m; highk_share_from_steps_under_2mps above 1 means those steps carry more than the HIGHK total (other HIGHK steps carry the opposite sign).

## 11. Limits

* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); a separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`; replicate arm not run). The planner is deterministic, so *another inference run* is not in play for these arms.
* X - P10 and Y - P10 use the banked P10 side (SMOOTH-T1's reference choice); the fresh floor differs from it by the KF deviation above. X - Y does not use P10.
* Tier T1-kin: executed state, logged cameras, own plan applied open-loop in the scene; it says nothing about re-perception. Inference inputs: vision plus measured ego history.
* The braking-set intervals rest on 19 episodes (section 7). The threshold-local wording is read two ways (section 6).
* The reader was edited after data in one import path (section 8); no number changed.
* Adoption into any K-1 / WM-1 claim is not decided here; the Master Mind registers the claim row `H-FEAS3-T1-1` (a proposed id).
