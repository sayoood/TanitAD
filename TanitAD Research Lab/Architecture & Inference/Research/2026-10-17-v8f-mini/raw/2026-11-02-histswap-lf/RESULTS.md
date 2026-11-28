# RESULTS: H-HISTSWAP-LF-1, is HISTSWAP's closure inertia or level? (zero training; HCV deployable arm, HLVL level-only ORACLE arm), DAG-OACC and SOUP at s0 / s1

Card `PREREG_HISTSWAP_LF_FROZEN.md` sha256 `d98db98087e9175fb2dfae83cd68f9a39de72bb82be2410399b2a5c4d49e6a1b` (frozen 2026-11-02T18:50:07+0100, before any HCV / HLVL number). Reader `code/histswap_lf_read.py` (rules tested in `tests/test_histswap_lf_read_rule.py`). Read clock 2026-11-02T19:15:12+0100. Raw `raw/HISTSWAP_LF_T1.json`, per-window arrays `raw/arrays/*.npz`. Generator of this file: `code/make_results_hlf.py`. Tier **T1-kin**. Evidence class **MEASURED (ours; artifacts in MANIFEST.txt)**.
Estimator of every interval below: **paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only; X and Y share the checkpoint, so no training-seed variance is in X - Y**. Marks: SB = separated-better, SW = separated-worse (direction-aware), ns = interval spans 0. The planner is deterministic (KA1), and the two seeds are the rig's replicates, not a variance estimate (`H-ESTIM-SEED-1` is not addressed). In the card's tags `CTRL` is DAG-OACC.

## 0. The committed reading: (B) LEVEL

Every number in this section: tier **T1-kin** (self-action open loop; own executed state, logged cameras) / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers *another draw of EPISODES* only) / evidence class **MEASURED** (ours; artifacts in MANIFEST.txt).

**MEASURED.** The card's rule reads **B_LEVEL** (reading **(B) level**; `hcv_closes` False, `hlvl_closes` True, `hlvl_discordant` False). The outcome label is **HARMS for both arms on both bases**; the label and its triggering rows are in section 1.

* **HCV: NOT CLOSES.** M1 crawl share X - Y is **separated-WORSE at 4 of 4 cells** (+0.0201 to +0.0270), mean share of Y - P10's M1 excess closed **-0.425** (cells -0.559 to -0.301; bar 0.5; pooled -0.415), ADE6 X - Y **+0.5867 to +0.6527 m** (guard +0.0604 m).
* **HLVL: CLOSES.** M1 crawl share X - Y is **separated-better at 4 of 4 cells** (-0.0422 to -0.0289), mean share closed **0.610** (cells 0.599 to 0.633; pooled 0.612), ADE6 X - Y **-0.6165 to -0.5171 m**.
* **Committed interpretation (card section 2, verbatim):** HCV CLOSES => **(A) inertia**, and HCV is a deployable inference lever. HCV does not close and HLVL CLOSES => **(B) level**: the speed level must come from perception or training, not from the history form. Neither CLOSES => **(C) interaction**: the trend-times-level structure of HISTSWAP's full history carries it. (MIXED counts as "does not close" and is flagged beside the reading.)
* **Committed next step under (B) (card section 5, verbatim):** the recovery fine-tune with re-expressed targets (`dagland/histswap/DRAFT_RECOVERY_CARD.md`) goes to the PI for a box-2 go, with the level source (perception or training) named as its unknown; no history-form lever is pursued.

> **ORACLE CAVEAT (the card's, binding on every sentence of this file).** HLVL reads the LOGGED measured speed at every tick, which is the human's speed LEVEL; it is **not deployable** and its absolute rows are not performance. Its closure is **near-mechanical** (card section 4): the plan is `P + O's delta` and the executor tracks it perfectly, so a prior anchored on the human's speed puts the executed speed near the human's at once; HLVL alone cannot tell *O needs the level* from *the prior's anchor is the input speed* (section 5 is the descriptive block that says which moved). **HCV is the discriminating arm** and it is deployable. HCV also removes ALL acceleration information (the tick row's own acceleration is 0 as well), which is stronger than *no trend*: its negative reading excludes *a history without acceleration recovers M1*, not a trend-only statement (HTRND, card section 5 (C), was not run).

| cell | arm | M1 X - Y | Y - P10 (the excess) | share of the excess closed | ADE6 X - Y (m) |
|---|---|---|---|---|---|
| DAG-OACC s0 | HCV | +0.0270 [+0.0121, +0.0432] SW | +0.0483 [+0.0302, +0.0698] SW | -0.5594 (-55.9 %) | +0.6527 [+0.5132, +0.7898] SW |
| DAG-OACC s0 | HLVL | -0.0289 [-0.0456, -0.0157] SB | +0.0483 [+0.0302, +0.0698] SW | 0.5988 (59.9 %) | -0.5171 [-0.6063, -0.4266] SB |
| DAG-OACC s1 | HCV | +0.0239 [+0.0099, +0.0390] SW | +0.0565 [+0.0347, +0.0829] SW | -0.4232 (-42.3 %) | +0.6141 [+0.4584, +0.7585] SW |
| DAG-OACC s1 | HLVL | -0.0344 [-0.0542, -0.0187] SB | +0.0565 [+0.0347, +0.0829] SW | 0.6095 (61.0 %) | -0.5696 [-0.6786, -0.4620] SB |
| SOUP s0 | HCV | +0.0201 [+0.0050, +0.0357] SW | +0.0666 [+0.0428, +0.0927] SW | -0.3012 (-30.1 %) | +0.5867 [+0.4374, +0.7366] SW |
| SOUP s0 | HLVL | -0.0422 [-0.0610, -0.0259] SB | +0.0666 [+0.0428, +0.0927] SW | 0.6330 (63.3 %) | -0.6165 [-0.7199, -0.5180] SB |
| SOUP s1 | HCV | +0.0240 [+0.0079, +0.0415] SW | +0.0576 [+0.0364, +0.0823] SW | -0.4156 (-41.6 %) | +0.5867 [+0.4316, +0.7365] SW |
| SOUP s1 | HLVL | -0.0346 [-0.0533, -0.0191] SB | +0.0576 [+0.0364, +0.0823] SW | 0.5994 (59.9 %) | -0.6009 [-0.7038, -0.5018] SB |

Rules (card section 2): `share closed` = -(X - Y) / (Y - P10) on the M1 row (undefined = 0); **CLOSES** = separated-better at all 4 cells AND mean share closed >= 0.5; **NOT CLOSES** = separated-better at fewer than 3 cells; else MIXED. HCV: 0 of 4 cells separated-better; HLVL: 4 of 4.

## 1. The outcome label: which rows trigger HARMS

Every number in this section: tier **T1-kin** (self-action open loop; own executed state, logged cameras) / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers *another draw of EPISODES* only) / evidence class **MEASURED** (ours; artifacts in MANIFEST.txt).

Rule (card section 2.1, `smooth_t1_read.arm_outcome`, precedence DOMINANT, HARMS, MOVED, NO EFFECT): HARMS at a cell = ADE6 X - Y > +0.0604 m **or** a non-crawl row of X - Y separated-worse; the base is HARMS if either seed is. MOVED = all four crawl rows separated-better and ADE6 X - Y <= +0.0604 m. The four crawl rows are `yaw-rate MAE`, `curvature MAE`, `L2 curvature MAE (GT>=5.2)`, `M1 crawl share`. The label is reported as the rule gives it.

| cell | arm | ADE6 X - Y (m) vs guard | crawl rows separated-better | non-crawl rows separated-worse (X - Y) | HARMS seed | MOVED seed |
|---|---|---|---|---|---|---|
| DAG-OACC s0 | HCV | +0.6527 (OVER guard) | 3 of 4 | ADE6, DK-HW, DK-VIOL, S-POS@30, S-POS@50, S-POS@80, TAC-LON, along-track MAE, speed MAE | True | False |
| DAG-OACC s1 | HCV | +0.6141 (OVER guard) | 3 of 4 | ADE6, DK-TG, DK-VIOL, S-POS@30, S-POS@50, S-POS@80, along-track MAE, speed MAE | True | False |
| SOUP s0 | HCV | +0.5867 (OVER guard) | 3 of 4 | ADE6, DK-VIOL, S-POS@30, S-POS@50, S-POS@80, along-track MAE, speed MAE | True | False |
| SOUP s1 | HCV | +0.5867 (OVER guard) | 1 of 4 | ADE6, DK-VIOL, S-POS@30, S-POS@50, S-POS@80, along-track MAE | True | False |
| DAG-OACC s0 | HLVL | -0.5171 (within guard) | 4 of 4 | S-POS@30, S-POS@50, S-POS@80 | True | True |
| DAG-OACC s1 | HLVL | -0.5696 (within guard) | 4 of 4 | S-POS@30, S-POS@50, S-POS@80 | True | True |
| SOUP s0 | HLVL | -0.6165 (within guard) | 4 of 4 | S-POS@30, S-POS@50, S-POS@80 | True | True |
| SOUP s1 | HLVL | -0.6009 (within guard) | 4 of 4 | S-POS@30, S-POS@50, S-POS@80 | True | True |

* **HCV triggers.** ADE6 over the guard at 4 of 4 cells; non-crawl rows separated-worse at some cell: `ADE6`, `DK-HW`, `DK-TG`, `DK-VIOL`, `S-POS@30`, `S-POS@50`, `S-POS@80`, `TAC-LON`, `along-track MAE`, `speed MAE`; at all 4 cells: `ADE6`, `DK-VIOL`, `S-POS@30`, `S-POS@50`, `S-POS@80`, `along-track MAE`. By family (rows separated-worse at some cell): LONGITUDINAL: DK-HW, DK-TG, DK-VIOL, along-track MAE, speed MAE; PRIMARY: ADE6; STRATEGIC: S-POS@30, S-POS@50, S-POS@80; TACTICAL: TAC-LON. Outcome labels: DAG-OACC **HARMS**, SOUP **HARMS** (`moved_and_harms`: DAG-OACC False, SOUP False).
* **HLVL triggers.** ADE6 over the guard at 0 of 4 cells; non-crawl rows separated-worse at some cell: `S-POS@30`, `S-POS@50`, `S-POS@80`; at all 4 cells: `S-POS@30`, `S-POS@50`, `S-POS@80`. By family (rows separated-worse at some cell): STRATEGIC: S-POS@30, S-POS@50, S-POS@80. Outcome labels: DAG-OACC **HARMS**, SOUP **HARMS** (`moved_and_harms`: DAG-OACC True, SOUP True).

⚠ **HLVL's HARMS label rests on the S-POS rows alone.** Its ADE6 X - Y is within the guard at all four cells, all four crawl rows are separated-better at all four cells (MOVED holds at both bases), and the only non-crawl rows separated-worse are `S-POS@30`, `S-POS@50`, `S-POS@80`. The card marks the S-POS rows of both arms **UNVERIFIED** (they mix the CV pose the S head read with the executed pose the extractor frames them in; card section 6), and they belong to the STRATEGIC family, which is PAUSED by the PI's ruling (`PI_DECISION_QUEUE.md` ITEM 25, 2026-09-19; reported as paused, not as a pass, and not extended to the other families). The label stands as the rule gave it; it is flagged here so that nobody reads it as a longitudinal or lateral harm. HLVL is an oracle in any case.

**HCV's HARMS is not a rows-on-the-margin result.** ADE6 X - Y is over the guard at every cell (+0.5867 to +0.6527 m) and the separated-worse rows include the longitudinal ones (`DK-HW`, `DK-TG`, `DK-VIOL`, `along-track MAE`, `speed MAE`). Leaving out the UNVERIFIED S-POS rows, the non-crawl rows separated-worse at all 4 cells are `ADE6`, `DK-VIOL`, `along-track MAE`. HCV is a deployable arm; under the card's committed rule (section 5, HARMS at either arm) the named non-crawl rows above are the next lever's target before any other reading is used.

## 2. Read A (X vs P10): D1 / D2 / D3 (HLVL is an ORACLE; its rows are not performance)

Every number in this section: tier **T1-kin** (self-action open loop; own executed state, logged cameras) / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers *another draw of EPISODES* only) / evidence class **MEASURED** (ours; artifacts in MANIFEST.txt).

Definitions (`smooth_t1_read.read_a`, smooth_t1_read.py:71-80): **D1** = ADE6 X - P10 separated-better and its point <= -0.2415 m; **D2** = no row of X - P10 separated-worse; **D3** = every one of the four families has at least one separated-better row; DOMINANT = D1 and D2 and D3. P10 is the programme's P10 floor. The STRATEGIC family is PAUSED (ITEM 25): its D3 entry is a measured flag of the rule, not a pass claim.

| cell | arm | D1 | D2 | D3 (L / Lat / T / S) | DOMINANT | ADE6 X - P10 (m) | rows separated-worse (X - P10) | rows separated-better |
|---|---|---|---|---|---|---|---|---|
| DAG-OACC s0 | HCV | True | False | True (True / True / True / True) | False | -0.6029 [-0.8347, -0.3719] SB | DK-VIOL, M1 crawl share | 12 of 19 |
| DAG-OACC s0 | HLVL | True | False | True (True / True / True / True) | False | -1.7726 [-1.9546, -1.5960] SB | M1 crawl share | 16 of 19 |
| DAG-OACC s1 | HCV | True | False | True (True / True / True / True) | False | -0.5762 [-0.8082, -0.3476] SB | DK-VIOL, L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE | 12 of 19 |
| DAG-OACC s1 | HLVL | True | False | True (True / True / True / True) | False | -1.7598 [-1.9404, -1.5914] SB | M1 crawl share | 15 of 19 |
| SOUP s0 | HCV | True | False | True (True / True / True / True) | False | -0.6061 [-0.8528, -0.3605] SB | DK-VIOL, L2 curvature MAE (GT>=5.2), M1 crawl share | 12 of 19 |
| SOUP s0 | HLVL | True | False | True (True / True / True / True) | False | -1.8093 [-1.9955, -1.6322] SB | M1 crawl share | 15 of 19 |
| SOUP s1 | HCV | True | False | True (True / True / True / True) | False | -0.6028 [-0.8398, -0.3693] SB | L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE | 12 of 19 |
| SOUP s1 | HLVL | True | False | True (True / True / True / True) | False | -1.7904 [-1.9707, -1.6220] SB | M1 crawl share, curvature MAE | 15 of 19 |

D1 and D3 hold and D2 fails at all 8 cells of both arms; no cell is DOMINANT. The row(s) that keep D2 failing are listed per cell. Y itself fails D2 on `M1 crawl share` (the programme's starting point; HISTSWAP card section 1).

## 3. All 19 rows, per cell: Y - P10, X - Y and X - P10 for both arms (HLVL absolute rows are not performance)

Every number in this section: tier **T1-kin** (self-action open loop; own executed state, logged cameras) / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers *another draw of EPISODES* only) / evidence class **MEASURED** (ours; artifacts in MANIFEST.txt). Every cell is `point [lo, hi] mark`. S-POS rows: **UNVERIFIED** (card section 6); STRATEGIC family PAUSED (ITEM 25).

### DAG-OACC s0

* HCV read_xy: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}; ADE6 X - Y +0.6527 (within guard: False); rows separated-worse on X - Y, all 19: 10. HLVL read_xy: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}; ADE6 X - Y -0.5171 (within guard: True); rows separated-worse on X - Y, all 19: 3.

| row | family | Y - P10 | HCV X - Y | HLVL X - Y | HCV X - P10 | HLVL X - P10 |
|---|---|---|---|---|---|---|
| ADE6 | PRIMARY | -1.2556 [-1.4503, -1.0678] SB | +0.6527 [+0.5132, +0.7898] SW | -0.5171 [-0.6063, -0.4266] SB | -0.6029 [-0.8347, -0.3719] SB | -1.7726 [-1.9546, -1.5960] SB |
| speed MAE | LONGITUDINAL | -0.8330 [-1.0377, -0.6583] SB | +0.1043 [+0.0311, +0.1772] SW | -0.5230 [-0.6040, -0.4518] SB | -0.7287 [-0.9598, -0.5303] SB | -1.3560 [-1.5825, -1.1526] SB |
| along-track MAE | LONGITUDINAL | -0.7192 [-0.8806, -0.5627] SB | +0.7059 [+0.5579, +0.8502] SW | -0.5105 [-0.5961, -0.4161] SB | -0.0134 [-0.2209, +0.1945] ns | -1.2297 [-1.3661, -1.0952] SB |
| accel MAE | LONGITUDINAL | -0.9390 [-1.3078, -0.6221] SB | -0.3573 [-0.5127, -0.2352] SB | -0.0673 [-0.2113, +0.0506] ns | -1.2964 [-1.7690, -0.9018] SB | -1.0064 [-1.4580, -0.6316] SB |
| DK-VIOL | LONGITUDINAL | +0.0168 [-0.0157, +0.0625] ns | +0.0286 [+0.0017, +0.0608] SW | +0.0000 [-0.0294, +0.0239] ns | +0.0454 [+0.0052, +0.0932] SW | +0.0168 [-0.0089, +0.0444] ns |
| DK-HW | LONGITUDINAL | -0.2604 [-0.6426, +0.0675] ns | +0.4371 [+0.1192, +0.7664] SW | -0.2322 [-0.5495, +0.1230] ns | +0.1069 [-0.3704, +0.5373] ns | -0.5214 [-0.9750, -0.1322] SB |
| DK-TG | LONGITUDINAL | -0.0588 [-0.1226, -0.0011] SB | +0.0821 [-0.0442, +0.2348] ns | -0.0618 [-0.1867, +0.0447] ns | -0.0063 [-0.1187, +0.0968] ns | -0.1240 [-0.2440, -0.0197] SB |
| heading MAE | LATERAL | -3.4520 [-4.2140, -2.7529] SB | -0.6815 [-1.1057, -0.2701] SB | -0.6590 [-1.0517, -0.2641] SB | -4.1450 [-4.9588, -3.4160] SB | -4.1225 [-4.9198, -3.4072] SB |
| yaw-rate MAE | LATERAL | -1.3762 [-3.9121, +1.0259] ns | -10.5769 [-13.8866, -7.6848] SB | -10.1100 [-13.2909, -7.3328] SB | -12.0139 [-15.9779, -8.5769] SB | -11.5474 [-15.3037, -8.2429] SB |
| curvature MAE | LATERAL | +0.0578 [+0.0364, +0.0821] SW | -0.0651 [-0.0897, -0.0436] SB | -0.0587 [-0.0820, -0.0378] SB | -0.0091 [-0.0231, +0.0053] ns | -0.0027 [-0.0175, +0.0135] ns |
| cross-track MAE | LATERAL | -1.0009 [-1.1468, -0.8671] SB | -0.0174 [-0.0620, +0.0267] ns | -0.1057 [-0.1629, -0.0531] SB | -1.0184 [-1.1680, -0.8839] SB | -1.1066 [-1.2549, -0.9701] SB |
| L1 lateral-accel MAE | LATERAL | -2.7327 [-4.0998, -1.5826] SB | -1.0353 [-1.4782, -0.6725] SB | -1.0030 [-1.4408, -0.6481] SB | -3.7646 [-5.3266, -2.4414] SB | -3.7323 [-5.2816, -2.4229] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0325 [+0.0135, +0.0590] SW | -0.0320 [-0.0568, -0.0152] SB | -0.0367 [-0.0635, -0.0179] SB | +0.0008 [-0.0033, +0.0056] ns | -0.0041 [-0.0072, -0.0010] SB |
| TAC-LAT | TACTICAL | +0.2038 [+0.1639, +0.2450] SB | +0.0125 [-0.0054, +0.0317] ns | +0.0113 [-0.0038, +0.0263] ns | +0.2163 [+0.1759, +0.2577] SB | +0.2151 [+0.1773, +0.2546] SB |
| TAC-LON | TACTICAL | +0.1751 [+0.1138, +0.2389] SB | -0.0383 [-0.0750, -0.0008] SW | +0.2193 [+0.1851, +0.2530] SB | +0.1367 [+0.0671, +0.2060] SB | +0.3943 [+0.3378, +0.4498] SB |
| S-POS@30 | STRATEGIC (PAUSED, ITEM 25) | -1.6443 [-2.1654, -1.1864] SB | +0.9631 [+0.4065, +1.6005] SW | +0.9863 [+0.4234, +1.6364] SW | -1.1294 [-1.5379, -0.7538] SB | -1.2457 [-1.6150, -0.8917] SB |
| S-POS@50 | STRATEGIC (PAUSED, ITEM 25) | -4.5745 [-5.8089, -3.5098] SB | +1.6025 [+0.6451, +2.7148] SW | +1.5608 [+0.5755, +2.7192] SW | -3.4286 [-4.4571, -2.5103] SB | -3.5187 [-4.4984, -2.5947] SB |
| S-POS@80 | STRATEGIC (PAUSED, ITEM 25) | -9.8284 [-12.1173, -7.7691] SB | +2.2796 [+0.7425, +4.1566] SW | +2.1888 [+0.5723, +4.1279] SW | -8.2093 [-10.2683, -6.1767] SB | -8.1464 [-10.2217, -6.1115] SB |
| M1 crawl share | crawl | +0.0483 [+0.0302, +0.0698] SW | +0.0270 [+0.0121, +0.0432] SW | -0.0289 [-0.0456, -0.0157] SB | +0.0763 [+0.0513, +0.1039] SW | +0.0203 [+0.0104, +0.0314] SW |

### DAG-OACC s1

* HCV read_xy: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}; ADE6 X - Y +0.6141 (within guard: False); rows separated-worse on X - Y, all 19: 9. HLVL read_xy: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}; ADE6 X - Y -0.5696 (within guard: True); rows separated-worse on X - Y, all 19: 3.

| row | family | Y - P10 | HCV X - Y | HLVL X - Y | HCV X - P10 | HLVL X - P10 |
|---|---|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1903 [-1.3986, -0.9782] SB | +0.6141 [+0.4584, +0.7585] SW | -0.5696 [-0.6786, -0.4620] SB | -0.5762 [-0.8082, -0.3476] SB | -1.7598 [-1.9404, -1.5914] SB |
| speed MAE | LONGITUDINAL | -0.8141 [-1.0377, -0.6191] SB | +0.0886 [+0.0197, +0.1578] SW | -0.5555 [-0.6321, -0.4858] SB | -0.7255 [-0.9547, -0.5240] SB | -1.3696 [-1.5939, -1.1654] SB |
| along-track MAE | LONGITUDINAL | -0.6302 [-0.8196, -0.4510] SB | +0.6268 [+0.4621, +0.7869] SW | -0.6143 [-0.7269, -0.5038] SB | -0.0034 [-0.2182, +0.2150] ns | -1.2444 [-1.3849, -1.1125] SB |
| accel MAE | LONGITUDINAL | -0.9719 [-1.3431, -0.6572] SB | -0.3317 [-0.4400, -0.2392] SB | -0.0491 [-0.1423, +0.0304] ns | -1.3036 [-1.7697, -0.9146] SB | -1.0210 [-1.4679, -0.6491] SB |
| DK-VIOL | LONGITUDINAL | +0.0000 [-0.0196, +0.0212] ns | +0.0303 [+0.0090, +0.0546] SW | +0.0101 [-0.0101, +0.0319] ns | +0.0303 [+0.0035, +0.0603] SW | +0.0101 [-0.0166, +0.0374] ns |
| DK-HW | LONGITUDINAL | -0.5171 [-0.9057, -0.2118] SB | +0.2614 [-0.0196, +0.5720] ns | -0.2230 [-0.5481, +0.1236] ns | -0.2539 [-0.7375, +0.1342] ns | -0.7283 [-1.2008, -0.3503] SB |
| DK-TG | LONGITUDINAL | -0.0869 [-0.1542, -0.0278] SB | +0.0634 [+0.0019, +0.1379] SW | -0.0099 [-0.0968, +0.0888] ns | -0.0250 [-0.1028, +0.0468] ns | -0.0983 [-0.1881, -0.0175] SB |
| heading MAE | LATERAL | -2.8936 [-3.6062, -2.1968] SB | -0.3111 [-1.0616, +0.4400] ns | -0.7776 [-1.3523, -0.2317] SB | -3.2047 [-3.9144, -2.5451] SB | -3.6711 [-4.2892, -3.1246] SB |
| yaw-rate MAE | LATERAL | +4.1412 [+1.1665, +7.3821] SW | -12.8727 [-17.3151, -9.1652] SB | -14.3030 [-19.5152, -9.8771] SB | -8.7144 [-11.5683, -6.2805] SB | -10.1436 [-13.4282, -7.3139] SB |
| curvature MAE | LATERAL | +0.1212 [+0.0757, +0.1729] SW | -0.0626 [-0.0980, -0.0276] SB | -0.0984 [-0.1388, -0.0635] SB | +0.0569 [+0.0154, +0.1088] SW | +0.0212 [-0.0019, +0.0477] ns |
| cross-track MAE | LATERAL | -1.0217 [-1.1686, -0.8803] SB | +0.0151 [-0.0377, +0.0717] ns | -0.0608 [-0.1241, +0.0002] ns | -1.0067 [-1.1553, -0.8623] SB | -1.0825 [-1.2305, -0.9476] SB |
| L1 lateral-accel MAE | LATERAL | -2.6082 [-3.8410, -1.5151] SB | -1.1168 [-1.6329, -0.7042] SB | -1.0891 [-1.5889, -0.6885] SB | -3.6635 [-5.1670, -2.3633] SB | -3.6358 [-5.1360, -2.3507] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0505 [+0.0269, +0.0791] SW | -0.0245 [-0.0467, -0.0039] SB | -0.0513 [-0.0785, -0.0282] SB | +0.0260 [+0.0067, +0.0483] SW | -0.0008 [-0.0045, +0.0036] ns |
| TAC-LAT | TACTICAL | +0.2126 [+0.1709, +0.2540] SB | -0.0042 [-0.0296, +0.0200] ns | +0.0029 [-0.0221, +0.0267] ns | +0.2084 [+0.1679, +0.2510] SB | +0.2155 [+0.1783, +0.2583] SB |
| TAC-LON | TACTICAL | +0.1638 [+0.1021, +0.2225] SB | -0.0225 [-0.0651, +0.0242] ns | +0.2409 [+0.2031, +0.2807] SB | +0.1413 [+0.0692, +0.2149] SB | +0.4048 [+0.3495, +0.4579] SB |
| S-POS@30 | STRATEGIC (PAUSED, ITEM 25) | -1.7416 [-2.2209, -1.3273] SB | +1.4237 [+0.6911, +2.2526] SW | +1.3055 [+0.6240, +2.0648] SW | -0.8681 [-1.3151, -0.4018] SB | -0.9111 [-1.3398, -0.4516] SB |
| S-POS@50 | STRATEGIC (PAUSED, ITEM 25) | -4.7600 [-5.8592, -3.7484] SB | +2.2862 [+1.0776, +3.6707] SW | +2.0294 [+0.9386, +3.3274] SW | -3.0066 [-4.1702, -1.8591] SB | -2.9097 [-4.0130, -1.7325] SB |
| S-POS@80 | STRATEGIC (PAUSED, ITEM 25) | -10.2999 [-12.3713, -8.3307] SB | +3.2571 [+1.4110, +5.4413] SW | +2.8368 [+1.1113, +4.8726] SW | -7.5087 [-9.8200, -5.1390] SB | -7.6088 [-9.8859, -5.2661] SB |
| M1 crawl share | crawl | +0.0565 [+0.0347, +0.0829] SW | +0.0239 [+0.0099, +0.0390] SW | -0.0344 [-0.0542, -0.0187] SB | +0.0804 [+0.0531, +0.1113] SW | +0.0221 [+0.0124, +0.0334] SW |

### SOUP s0

* HCV read_xy: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': False}; ADE6 X - Y +0.5867 (within guard: False); rows separated-worse on X - Y, all 19: 8. HLVL read_xy: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}; ADE6 X - Y -0.6165 (within guard: True); rows separated-worse on X - Y, all 19: 3.

| row | family | Y - P10 | HCV X - Y | HLVL X - Y | HCV X - P10 | HLVL X - P10 |
|---|---|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1928 [-1.4078, -0.9880] SB | +0.5867 [+0.4374, +0.7366] SW | -0.6165 [-0.7199, -0.5180] SB | -0.6061 [-0.8528, -0.3605] SB | -1.8093 [-1.9955, -1.6322] SB |
| speed MAE | LONGITUDINAL | -0.7817 [-0.9919, -0.5956] SB | +0.0706 [+0.0008, +0.1408] SW | -0.5678 [-0.6411, -0.5000] SB | -0.7111 [-0.9455, -0.5044] SB | -1.3495 [-1.5754, -1.1453] SB |
| along-track MAE | LONGITUDINAL | -0.6291 [-0.8126, -0.4582] SB | +0.6349 [+0.4807, +0.7910] SW | -0.6147 [-0.7205, -0.5103] SB | +0.0059 [-0.2120, +0.2237] ns | -1.2438 [-1.3844, -1.1053] SB |
| accel MAE | LONGITUDINAL | -0.8791 [-1.2581, -0.5583] SB | -0.3787 [-0.5041, -0.2737] SB | -0.0648 [-0.1862, +0.0331] ns | -1.2578 [-1.7307, -0.8620] SB | -0.9439 [-1.3972, -0.5625] SB |
| DK-VIOL | LONGITUDINAL | +0.0067 [-0.0196, +0.0346] ns | +0.0286 [+0.0052, +0.0549] SW | +0.0101 [-0.0097, +0.0297] ns | +0.0353 [+0.0037, +0.0741] SW | +0.0168 [-0.0113, +0.0476] ns |
| DK-HW | LONGITUDINAL | -0.2040 [-0.5885, +0.1367] ns | +0.3042 [-0.0079, +0.6231] ns | -0.2750 [-0.5718, +0.0640] ns | +0.0122 [-0.4487, +0.4130] ns | -0.5796 [-1.0111, -0.1967] SB |
| DK-TG | LONGITUDINAL | -0.0560 [-0.1173, -0.0018] SB | +0.0681 [-0.0217, +0.1859] ns | -0.0324 [-0.1379, +0.0890] ns | -0.0088 [-0.0882, +0.0630] ns | -0.1013 [-0.1930, -0.0145] SB |
| heading MAE | LATERAL | -2.9614 [-3.7496, -2.1787] SB | -1.1740 [-1.6435, -0.7215] SB | -1.0115 [-1.4825, -0.5894] SB | -4.1470 [-4.9155, -3.4442] SB | -3.9845 [-4.6684, -3.3467] SB |
| yaw-rate MAE | LATERAL | +1.9270 [-1.0592, +5.0043] ns | -12.7992 [-16.5707, -9.4312] SB | -12.5933 [-16.2489, -9.1579] SB | -10.9369 [-14.7141, -7.7036] SB | -10.7311 [-14.2698, -7.6354] SB |
| curvature MAE | LATERAL | +0.0919 [+0.0584, +0.1301] SW | -0.0811 [-0.1110, -0.0543] SB | -0.0792 [-0.1128, -0.0492] SB | +0.0090 [-0.0095, +0.0314] ns | +0.0109 [-0.0075, +0.0322] ns |
| cross-track MAE | LATERAL | -1.0265 [-1.1729, -0.8943] SB | -0.0217 [-0.0673, +0.0255] ns | -0.1118 [-0.1677, -0.0591] SB | -1.0483 [-1.2021, -0.9086] SB | -1.1383 [-1.2875, -0.9971] SB |
| L1 lateral-accel MAE | LATERAL | -2.6420 [-4.0010, -1.5149] SB | -1.1160 [-1.5382, -0.7594] SB | -1.0785 [-1.4941, -0.7335] SB | -3.7546 [-5.3141, -2.4321] SB | -3.7172 [-5.2641, -2.4109] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0628 [+0.0279, +0.1076] SW | -0.0515 [-0.0921, -0.0213] SB | -0.0649 [-0.1094, -0.0305] SB | +0.0116 [+0.0013, +0.0248] SW | -0.0018 [-0.0052, +0.0022] ns |
| TAC-LAT | TACTICAL | +0.1847 [+0.1417, +0.2274] SB | +0.0367 [+0.0217, +0.0529] SB | +0.0350 [+0.0184, +0.0530] SB | +0.2213 [+0.1814, +0.2614] SB | +0.2197 [+0.1825, +0.2587] SB |
| TAC-LON | TACTICAL | +0.1467 [+0.0792, +0.2138] SB | -0.0021 [-0.0450, +0.0413] ns | +0.2205 [+0.1846, +0.2561] SB | +0.1446 [+0.0717, +0.2178] SB | +0.3672 [+0.3099, +0.4258] SB |
| S-POS@30 | STRATEGIC (PAUSED, ITEM 25) | -1.6062 [-2.1525, -1.1336] SB | +0.8353 [+0.2811, +1.4527] SW | +0.7766 [+0.2173, +1.4041] SW | -1.4160 [-1.8424, -1.0304] SB | -1.4206 [-1.8131, -1.0568] SB |
| S-POS@50 | STRATEGIC (PAUSED, ITEM 25) | -4.5825 [-5.8449, -3.4774] SB | +1.4375 [+0.4810, +2.5150] SW | +1.2805 [+0.2841, +2.3784] SW | -3.8480 [-4.8703, -2.8864] SB | -3.7639 [-4.7332, -2.8705] SB |
| S-POS@80 | STRATEGIC (PAUSED, ITEM 25) | -10.1918 [-12.6081, -8.0751] SB | +2.0516 [+0.5462, +3.7837] SW | +1.8062 [+0.2894, +3.5410] SW | -8.7015 [-10.7309, -6.7443] SB | -8.4173 [-10.4652, -6.4722] SB |
| M1 crawl share | crawl | +0.0666 [+0.0428, +0.0927] SW | +0.0201 [+0.0050, +0.0357] SW | -0.0422 [-0.0610, -0.0259] SB | +0.0872 [+0.0599, +0.1173] SW | +0.0249 [+0.0141, +0.0374] SW |

### SOUP s1

* HCV read_xy: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': False, 'L2 curvature MAE (GT>=5.2)': False, 'M1 crawl share': False}; ADE6 X - Y +0.5867 (within guard: False); rows separated-worse on X - Y, all 19: 7. HLVL read_xy: crawl rows separated-better {'yaw-rate MAE': True, 'curvature MAE': True, 'L2 curvature MAE (GT>=5.2)': True, 'M1 crawl share': True}; ADE6 X - Y -0.6009 (within guard: True); rows separated-worse on X - Y, all 19: 3.

| row | family | Y - P10 | HCV X - Y | HLVL X - Y | HCV X - P10 | HLVL X - P10 |
|---|---|---|---|---|---|---|
| ADE6 | PRIMARY | -1.1895 [-1.3938, -0.9847] SB | +0.5867 [+0.4316, +0.7365] SW | -0.6009 [-0.7038, -0.5018] SB | -0.6028 [-0.8398, -0.3693] SB | -1.7904 [-1.9707, -1.6220] SB |
| speed MAE | LONGITUDINAL | -0.7737 [-0.9864, -0.5882] SB | +0.0472 [-0.0263, +0.1187] ns | -0.5942 [-0.6736, -0.5192] SB | -0.7265 [-0.9569, -0.5267] SB | -1.3679 [-1.5911, -1.1640] SB |
| along-track MAE | LONGITUDINAL | -0.6260 [-0.8049, -0.4497] SB | +0.6292 [+0.4613, +0.7941] SW | -0.6152 [-0.7230, -0.5084] SB | +0.0032 [-0.2097, +0.2207] ns | -1.2412 [-1.3838, -1.1095] SB |
| accel MAE | LONGITUDINAL | -0.7964 [-1.1397, -0.4913] SB | -0.5111 [-0.6968, -0.3552] SB | -0.2296 [-0.4054, -0.0838] SB | -1.3075 [-1.7738, -0.9190] SB | -1.0260 [-1.4780, -0.6506] SB |
| DK-VIOL | LONGITUDINAL | -0.0067 [-0.0265, +0.0122] ns | +0.0336 [+0.0122, +0.0599] SW | +0.0050 [-0.0109, +0.0217] ns | +0.0269 [+0.0000, +0.0598] ns | -0.0017 [-0.0229, +0.0193] ns |
| DK-HW | LONGITUDINAL | -0.5481 [-0.9299, -0.2218] SB | +0.2195 [-0.0003, +0.4328] ns | -0.2076 [-0.4860, +0.0856] ns | -0.3244 [-0.7603, +0.0469] ns | -0.7612 [-1.1773, -0.3976] SB |
| DK-TG | LONGITUDINAL | -0.0863 [-0.1467, -0.0356] SB | +0.0487 [-0.0007, +0.0998] ns | -0.0044 [-0.0889, +0.1034] ns | -0.0348 [-0.1055, +0.0307] ns | -0.1020 [-0.1856, -0.0219] SB |
| heading MAE | LATERAL | -2.9491 [-3.6329, -2.3131] SB | -0.0020 [-0.6800, +0.8120] ns | -0.7157 [-1.1500, -0.2912] SB | -2.9511 [-3.7489, -2.0932] SB | -3.6648 [-4.3193, -3.0644] SB |
| yaw-rate MAE | LATERAL | +3.0944 [+0.2216, +6.2120] SW | -11.1998 [-14.9710, -7.9899] SB | -12.7491 [-17.0275, -8.9450] SB | -8.1470 [-10.7936, -5.7966] SB | -9.6957 [-12.7977, -7.0527] SB |
| curvature MAE | LATERAL | +0.1067 [+0.0666, +0.1533] SW | -0.0255 [-0.0667, +0.0191] ns | -0.0670 [-0.1017, -0.0366] SB | +0.0799 [+0.0238, +0.1504] SW | +0.0385 [+0.0067, +0.0780] SW |
| cross-track MAE | LATERAL | -1.0189 [-1.1607, -0.8829] SB | -0.0264 [-0.0790, +0.0287] ns | -0.1053 [-0.1639, -0.0524] SB | -1.0453 [-1.1923, -0.9001] SB | -1.1242 [-1.2704, -0.9867] SB |
| L1 lateral-accel MAE | LATERAL | -2.5744 [-3.8473, -1.4601] SB | -1.1599 [-1.6218, -0.7626] SB | -1.1303 [-1.5762, -0.7441] SB | -3.7322 [-5.2741, -2.4150] SB | -3.7025 [-5.2528, -2.3981] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | +0.0454 [+0.0224, +0.0739] SW | -0.0164 [-0.0395, +0.0061] ns | -0.0446 [-0.0715, -0.0232] SB | +0.0288 [+0.0083, +0.0533] SW | +0.0008 [-0.0039, +0.0067] ns |
| TAC-LAT | TACTICAL | +0.2138 [+0.1747, +0.2548] SB | +0.0108 [-0.0129, +0.0325] ns | +0.0129 [-0.0100, +0.0338] ns | +0.2247 [+0.1862, +0.2661] SB | +0.2268 [+0.1904, +0.2677] SB |
| TAC-LON | TACTICAL | +0.1584 [+0.0975, +0.2172] SB | -0.0321 [-0.0747, +0.0117] ns | +0.2363 [+0.1972, +0.2752] SB | +0.1263 [+0.0542, +0.1983] SB | +0.3947 [+0.3407, +0.4486] SB |
| S-POS@30 | STRATEGIC (PAUSED, ITEM 25) | -1.7972 [-2.2609, -1.3948] SB | +1.4466 [+0.7351, +2.2581] SW | +1.3024 [+0.6485, +2.0222] SW | -0.9403 [-1.3574, -0.5247] SB | -0.9336 [-1.3524, -0.5419] SB |
| S-POS@50 | STRATEGIC (PAUSED, ITEM 25) | -5.0252 [-6.1549, -3.9713] SB | +2.3689 [+1.1959, +3.6916] SW | +2.0599 [+1.0102, +3.2267] SW | -3.0592 [-4.0999, -2.0251] SB | -3.0476 [-4.0818, -2.0514] SB |
| S-POS@80 | STRATEGIC (PAUSED, ITEM 25) | -10.7619 [-12.9488, -8.6791] SB | +3.3228 [+1.5542, +5.3940] SW | +2.8877 [+1.3329, +4.6793] SW | -7.5644 [-9.7700, -5.3408] SB | -7.5761 [-9.7804, -5.4210] SB |
| M1 crawl share | crawl | +0.0576 [+0.0364, +0.0823] SW | +0.0240 [+0.0079, +0.0415] SW | -0.0346 [-0.0533, -0.0191] SB | +0.0812 [+0.0549, +0.1113] SW | +0.0231 [+0.0132, +0.0345] SW |

## 4. The M1 row by situation class (descriptive, not a decision; classes L launch / S stay / O other-stopped / M moving; weight `w` = share of the row's windows)

Every number in this section: tier **T1-kin** (self-action open loop; own executed state, logged cameras) / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers *another draw of EPISODES* only) / evidence class **MEASURED** (ours; artifacts in MANIFEST.txt).

| cell | contrast | row delta | L: n, delta, contribution | M: n, delta, contribution | S, O windows | L share of the row's delta |
|---|---|---|---|---|---|---|
| DAG-OACC s0 | HCV X - Y | +0.0270 [+0.0121, +0.0432] | 254, +0.2849 [+0.2104, +0.3587], +0.0344 | 1850, -0.0084 [-0.0168, -0.0014], -0.0074 | 0, 0 | 127 % |
| DAG-OACC s0 | HLVL X - Y | -0.0289 [-0.0456, -0.0157] | 253, -0.1182 [-0.1909, -0.0468], -0.0142 | 1850, -0.0167 [-0.0287, -0.0075], -0.0147 | 0, 0 | 49 % |
| DAG-OACC s0 | Y - P10 (shared) | +0.0483 [+0.0302, +0.0698] | 254, +0.2614 [+0.1650, +0.3607], +0.0316 | 1850, +0.0191 [+0.0097, +0.0310], +0.0168 | 0, 0 | 65 % |
| DAG-OACC s1 | HCV X - Y | +0.0239 [+0.0099, +0.0390] | 255, +0.2762 [+0.1990, +0.3460], +0.0334 | 1851, -0.0108 [-0.0177, -0.0047], -0.0095 | 0, 0 | 140 % |
| DAG-OACC s1 | HLVL X - Y | -0.0344 [-0.0542, -0.0187] | 255, -0.1138 [-0.1968, -0.0359], -0.0138 | 1851, -0.0235 [-0.0379, -0.0126], -0.0207 | 0, 0 | 40 % |
| DAG-OACC s1 | Y - P10 (shared) | +0.0565 [+0.0347, +0.0829] | 255, +0.2717 [+0.1600, +0.3902], +0.0329 | 1851, +0.0269 [+0.0154, +0.0424], +0.0236 | 0, 0 | 58 % |
| SOUP s0 | HCV X - Y | +0.0201 [+0.0050, +0.0357] | 254, +0.2552 [+0.1733, +0.3368], +0.0308 | 1851, -0.0122 [-0.0204, -0.0049], -0.0107 | 0, 0 | 153 % |
| SOUP s0 | HLVL X - Y | -0.0422 [-0.0610, -0.0259] | 254, -0.1417 [-0.2252, -0.0645], -0.0171 | 1851, -0.0285 [-0.0433, -0.0163], -0.0251 | 0, 0 | 41 % |
| SOUP s0 | Y - P10 (shared) | +0.0666 [+0.0428, +0.0927] | 254, +0.3152 [+0.2023, +0.4337], +0.0380 | 1851, +0.0325 [+0.0201, +0.0478], +0.0286 | 0, 0 | 57 % |
| SOUP s1 | HCV X - Y | +0.0240 [+0.0079, +0.0415] | 254, +0.3079 [+0.2223, +0.3899], +0.0371 | 1851, -0.0150 [-0.0233, -0.0077], -0.0132 | 0, 0 | 155 % |
| SOUP s1 | HLVL X - Y | -0.0346 [-0.0533, -0.0191] | 255, -0.1008 [-0.1821, -0.0197], -0.0122 | 1851, -0.0254 [-0.0400, -0.0147], -0.0223 | 0, 0 | 35 % |
| SOUP s1 | Y - P10 (shared) | +0.0576 [+0.0364, +0.0823] | 255, +0.2691 [+0.1620, +0.3842], +0.0326 | 1851, +0.0285 [+0.0169, +0.0427], +0.0251 | 0, 0 | 57 % |

The S and O classes carry no windows in the masked M1 pool at any cell. Launch-class windows carry 57 %-65 % of the Y - P10 excess, 35 %-49 % of HLVL's X - Y movement and 127 %-155 % of HCV's X - Y movement (contribution / row delta; the weight of L is 12.1 % of the row's windows).
Per class (X - Y delta of the class's own M1 value): **HCV** L class +0.2552 to +0.3079, M class -0.0150 to -0.0084; **HLVL** L class -0.1417 to -0.1008, M class -0.0285 to -0.0167; for scale, the shared Y - P10 excess is L +0.2614 to +0.3152, M +0.0191 to +0.0325. HCV's harm sits in the launch class (its launch-class X - Y is 0.81-1.14 of Y's own launch-class excess over P10), while the moving class improves slightly; HLVL improves both.

## 5. Prior anchor versus O's delta (descriptive; the card's `o_residual_x30` / `P_x30` block)

Tier T1-kin, evidence class MEASURED, **point means with no interval** (mean over windows of the per-tick 6 s along-track value, ticks >= 1; the tick-0 plan is excluded). `P` = the prior the network computed in the same call (it reads the swapped history), `O delta` = plan_raw - P at 6 s, `plan` = P + O delta. Y is the fresh `SREP_*` run.

| cell | arm | P (Y, X, X - Y) m | O delta (Y, X, X - Y) m | plan (Y, X, X - Y) m | P + O delta - plan (X, m) |
|---|---|---|---|---|---|
| DAG-OACC s0 | HCV | 64.270, 64.362, +0.092 | +0.171, +0.120, -0.051 | 64.446, 64.482, +0.036 | -6.3e-04 |
| DAG-OACC s0 | HLVL | 64.270, 65.100, +0.830 | +0.171, -0.134, -0.306 | 64.446, 64.967, +0.520 | -6.6e-04 |
| DAG-OACC s1 | HCV | 64.372, 64.077, -0.295 | -0.345, -0.276, +0.069 | 64.028, 63.802, -0.226 | -1.2e-03 |
| DAG-OACC s1 | HLVL | 64.372, 65.064, +0.692 | -0.345, -0.677, -0.332 | 64.028, 64.386, +0.358 | +9.5e-04 |
| SOUP s0 | HCV | 61.543, 63.134, +1.591 | -0.382, -0.769, -0.387 | 61.164, 62.364, +1.200 | +6.5e-04 |
| SOUP s0 | HLVL | 61.543, 65.015, +3.472 | -0.382, -1.298, -0.916 | 61.164, 63.716, +2.552 | +1.1e-03 |
| SOUP s1 | HCV | 64.605, 64.179, -0.426 | -0.565, -0.170, +0.395 | 64.044, 64.013, -0.032 | -3.7e-03 |
| SOUP s1 | HLVL | 64.605, 65.133, +0.528 | -0.565, -0.578, -0.014 | 64.044, 64.556, +0.512 | -1.6e-03 |

Identity check: max |P + O delta - plan| over the table = 4.7e-03 m (the three columns are separate means, so equality is approximate; the residual is not investigated).
* **HLVL:** the prior's 6 s end point rises vs Y at 4 of 4 cells (X - Y +0.528 to +3.472 m), O's own delta moves **down** at 4 of 4 cells (X - Y -0.916 to -0.014 m), and the plan's end point rises at 4 of 4 cells (X - Y +0.358 to +2.552 m). The movement of the plan's end point has the sign of the movement of P: the prior's anchor carries it, and O's delta works against it. This is consistent with the card's section 4 caution that HLVL's closure is near-mechanical (it cannot by itself show that O needs the level). It is a consistency check, not a decomposition of the M1 closure: the block averages over all windows and ticks >= 1, while M1 lives in the launch-class windows (section 4).
* **HCV:** the prior's end point moves up at 2 of 4 cells (X - Y -0.426 to +1.591 m), O's delta moves up at 2 of 4 (-0.387 to +0.395 m), the plan's end point up at 2 of 4 (-0.226 to +1.200 m). The signs are not uniform, so the mean 6 s end point does not carry HCV's M1 and ADE6 harm; those are windowed and class-specific (section 4: launch-class windows), and this block is not targeted at them.

## 6. Controls (computed before any row; any failure refuses the read)

* **KY_reused_Y_equals_frozen_pins**: PASS
* **KA1_fresh_off_rerun_equals_reused_Y**: PASS
* **KC1_flag_reached_the_run_and_v0_in_is_as_the_card_says**: PASS
* **KC2_the_CV_history_reached_the_network**: PASS
* **KC3_plan_committed_to_the_executed_state**: PASS
* **KF_floor_window_check_vs_fresh_Y**: PASS
* **K3_Y_banked_minus_P10_reproduces_SOUP_READ**: PASS
* **KB1_P10_side_equals_landed_banked_b**: PASS

**KY**: 8 / 8 reused Y files hash-equal to the pins of `FROZEN_SHA.txt`.
**KA1 (the SREP5 bit-identity rerun):** the fresh LF-tool `--histswap off` rerun `SREP5_DAGOACC_s0` against the reused `SREP_DAGOACC_s0`: dump bit-identical **True** (max abs dev 0.000e+00 m), ticks bit-identical **True** (max abs dev 0.000e+00), differing keys none, key-set difference none. The planner is deterministic, so the thread-order floor is 0 on this rig.

**KC1** (the flag reached the run; `v0_in` = the speed the network read, `v0` = the executed speed):

| run | `--histswap` | max abs v0_in - v0, all ticks (m/s) | max abs at tick 0 | share of later ticks with v0_in != v0 by > 0.05 m/s | mean executed / network speed, later ticks (m/s) |
|---|---|---|---|---|---|
| CTRL_s0:HCV | cv_exec | 0.000 | 0.0e+00 | 0.0000 | 11.179 / 11.179 |
| CTRL_s0:HLVL | cv_logged | 6.670 | 0.0e+00 | 0.9471 | 11.208 / 11.297 |
| CTRL_s1:HCV | cv_exec | 0.000 | 0.0e+00 | 0.0000 | 11.129 / 11.129 |
| CTRL_s1:HLVL | cv_logged | 6.834 | 0.0e+00 | 0.9483 | 11.166 / 11.297 |
| SOUP_s0:HCV | cv_exec | 0.000 | 0.0e+00 | 0.0000 | 10.969 / 10.969 |
| SOUP_s0:HLVL | cv_logged | 7.023 | 0.0e+00 | 0.9460 | 11.047 / 11.297 |
| SOUP_s1:HCV | cv_exec | 0.000 | 0.0e+00 | 0.0000 | 11.137 / 11.137 |
| SOUP_s1:HLVL | cv_logged | 6.787 | 0.0e+00 | 0.9544 | 11.167 / 11.297 |

**KC2** (the CV history reached the network: the tick-0 plan differs from Y's by more than 1e-3 m): max abs tick-0 plan_raw deviation 42.5 to 50.6 m over 2399 tick-0 rows per run, 8 / 8 runs pass. **KC3** (the plan is committed to the executed state: tick 1's executed position equals the tick-0 pose composed with plan knot 10, tolerance 5e-3 m, >= 99.9 % of windows): max deviation 1.1e-13 m, share within tolerance 1.0000 at 8 / 8 runs. **KF**: windows X = fresh Y at 8 / 8 runs, floor X vs fresh Y max abs 0.0e+00 m. **K3**: Y banked - P10 reproduces `SOUP_READ.json` on 76 / 76 rows. **KB1**: the P10 side of every extracted X and Y equals the landed banked `b` arrays: 12 / 12 runs pass.

Tool tests `logs/pytest_histswap_lf.log`: **9 passed** (three mutations RED: cv_exec reading the logged speed, a ramp in the track, the item builder on the executed dict); reader-rule tests `logs/pytest_read_rule.log`: **7 passed** (three mutations RED); 32-window mechanical smoke `logs/smoke_lf.out`: **SMOKELF OK**. Rollouts: 9 (8 arm runs + the SREP5 rerun), `logs/chain.out` records `ZZHLF-ROLLOUTS-DONE nfail=0`.

**Swap bookkeeping** (`swap_bucket_differs / swap_ticks`: how often the rebuilt causal speed bucket differs from the executed-history bucket):

| cell | HCV | HLVL |
|---|---|---|
| DAG-OACC s0 | 626 / 7197 (8.70 %) | 1005 / 7197 (13.96 %) |
| DAG-OACC s1 | 644 / 7197 (8.95 %) | 998 / 7197 (13.87 %) |
| SOUP s0 | 668 / 7197 (9.28 %) | 1006 / 7197 (13.98 %) |
| SOUP s1 | 640 / 7197 (8.89 %) | 997 / 7197 (13.85 %) |

The bucket is rebuilt from the CV history in both arms; its separate contribution to M1 is not attributed (card section 1 (iii)).

## 7. The card's ESTIMATED probabilities next to the result

| quantity | card (ESTIMATED, written before any number) | result (MEASURED) | probability the card gave the result |
|---|---|---|---|
| HCV M1 reading | CLOSES 0.10 / MIXED 0.15 / NOT CLOSES 0.75 | **NOT CLOSES** (0 of 4 cells separated-better; mean share -0.425) | 0.75 |
| HLVL M1 reading | CLOSES 0.80 / MIXED 0.12 / NOT CLOSES 0.08 | **CLOSES** (4 of 4 cells separated-better; mean share 0.610) | 0.80 |
| interpretation | (A) 0.10 / (B) 0.70 / (C) 0.20 | **B_LEVEL** | 0.70 |

One read is not a calibration. What the card's three-way split could not express: HCV's NOT CLOSES is not a null: it is **separated-worse** than Y on M1 at 4 of 4 cells and ADE6 is over the guard at 4 of 4, so the executed-speed constant-twist history is itself a harmful input, not only an uninformative one (whether the harm is the missing acceleration information or the off-distribution input, a history that is exactly constant at every row, is UNVERIFIED).

## 8. Against HISTSWAP's full logged history (H-HISTSWAP-1, `dagland/histswap/raw/HISTSWAP_T1.json`)

Every number in this section: tier **T1-kin** (self-action open loop; own executed state, logged cameras) / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers *another draw of EPISODES* only) / evidence class **MEASURED** (ours; artifacts in MANIFEST.txt). HISTSWAP (the full LOGGED history swap: speed profile, acceleration, trend, yaw / pose rows and bucket) is an ORACLE too; same Y, same windows, same checkpoints.

| cell | share closed: HISTSWAP / HLVL / HCV | M1 X - Y: HISTSWAP | M1 X - Y: HLVL | ADE6 X - Y (m): HISTSWAP | ADE6 X - Y (m): HLVL |
|---|---|---|---|---|---|
| DAG-OACC s0 | 0.984 / 0.599 / -0.559 | -0.0476 [-0.0664, -0.0315] SB | -0.0289 [-0.0456, -0.0157] SB | -1.2325 [-1.3530, -1.1168] SB | -0.5171 [-0.6063, -0.4266] SB |
| DAG-OACC s1 | 0.925 / 0.610 / -0.423 | -0.0523 [-0.0739, -0.0338] SB | -0.0344 [-0.0542, -0.0187] SB | -1.2352 [-1.3687, -1.1032] SB | -0.5696 [-0.6786, -0.4620] SB |
| SOUP s0 | 0.918 / 0.633 / -0.301 | -0.0612 [-0.0832, -0.0408] SB | -0.0422 [-0.0610, -0.0259] SB | -1.2971 [-1.4285, -1.1693] SB | -0.6165 [-0.7199, -0.5180] SB |
| SOUP s1 | 0.931 / 0.599 / -0.416 | -0.0537 [-0.0754, -0.0355] SB | -0.0346 [-0.0533, -0.0191] SB | -1.2806 [-1.4169, -1.1517] SB | -0.6009 [-0.7038, -0.5018] SB |

* Mean share closed: **HISTSWAP 0.939** (outcome SUPPORTED, 4 of 4 cells), **HLVL 0.610**, HCV -0.425. Mean ADE6 X - Y: HISTSWAP -1.261 m, HLVL -0.576 m, HCV +0.610 m. The level alone, as an oracle, recovers 65 % of what the full logged history recovers on M1 (share closed 0.610 / 0.939).
* **HISTSWAP - HLVL is what the rest of the logged history adds on top of the speed level**: 0.329 of the share closed (0.939 - 0.610) and -0.685 m of ADE6 (mean X - Y). The two differ in more than acceleration and trend: HISTSWAP swaps every ego-row-built key from the logged arrays (card of H-HISTSWAP-1, section 1), so it also carries the logged yaw / pose rows, which HLVL holds at the executed yaw rate (card section 1 (i)). The split of the 0.33 between acceleration, trend and yaw rows is not measured here.
* Interval overlap, a check and **not an estimator of the difference** (the two arms were not bootstrapped against each other): HLVL's and HISTSWAP's X - Y intervals overlap on M1 at 4 of 4 cells and on ADE6 at 0 of 4 cells.

## 9. What this read decides, and what it does not

* **Reading (B) level; the committed next step is the card's:** the recovery fine-tune (`dagland/histswap/DRAFT_RECOVERY_CARD.md`, a DRAFT; v2 in `DRAFT_RECOVERY_CARD_v2.md`) goes to the PI for a box-2 go, with the **level source (perception or training) named as its unknown**; no history-form lever is pursued. This package starts no training and touches no box.
* **The trend copy is not supported as the carrier:** HCV, which holds speed and yaw rate constant over the history and sets acceleration to 0, does not close M1; it is separated-worse. HLVL, which differs from HCV in exactly one number (the speed, logged instead of executed), closes 60 %-63 % of the excess at the same 4 cells. By construction (card section 1) that one number, the speed, is the only input difference between a harmful arm and a closing arm.
* **What the read does not say.** (i) HLVL is an oracle: nothing here shows that a model trained or run without the human's speed recovers any of the 60 %-63 %. (ii) Over all windows the movement of HLVL's plan end point comes through the prior's anchor, with O's delta moving against it (section 5), as the card's near-mechanical caution anticipated. (iii) HCV removes all acceleration information, so *trend alone* is untested (HTRND not run). (iv) Tick 0 of both arms differs from Y by construction (the CV history differs from the logged one at t0), so part of the ADE6 movement can originate at tick 0 (KC2). (v) T1-kin: the cameras are the logged ones, so a model that reads the human's speed from frame motion can pass M1 here and not in driving; the frame-freeze control that separates this is pre-registered in `DRAFT_RECOVERY_CARD_v2.md` section 8, and it was **not run on any arm of this read**.
* **Not decided here:** adoption into any K-1 / WM-1 claim; the Master Mind registers `H-HISTSWAP-LF-1` (the id the card used).

## 10. Curvature-excess partition (descriptive; the gate's partition applied to this read's X and Y dumps)

`E` = the curvature MAE point (X - P10 for X, Y - P10 for Y); `n_dip` = windows with an unwarranted speed dip; `highk share <2 m/s` = share of the high-curvature steps' excess from steps under 2 m/s. Point values, no interval.

| cell | arm | N | E X | n_dip X | highk share <2 m/s X | E Y | n_dip Y | highk share <2 m/s Y |
|---|---|---|---|---|---|---|---|---|
| DAG-OACC s0 | HCV | 2367 | -0.0091 | 14 | 0.851 | +0.0578 | 46 | 0.555 |
| DAG-OACC s0 | HLVL | 2367 | -0.0027 | 7 | 0.931 | +0.0578 | 46 | 0.555 |
| DAG-OACC s1 | HCV | 2366 | +0.0569 | 41 | 1.003 | +0.1212 | 63 | 0.585 |
| DAG-OACC s1 | HLVL | 2366 | +0.0212 | 8 | 0.954 | +0.1212 | 63 | 0.585 |
| SOUP s0 | HCV | 2367 | +0.0090 | 25 | 0.873 | +0.0919 | 74 | 0.534 |
| SOUP s0 | HLVL | 2367 | +0.0109 | 8 | 0.924 | +0.0919 | 74 | 0.534 |
| SOUP s1 | HCV | 2367 | +0.0799 | 38 | 1.010 | +0.1067 | 55 | 0.618 |
| SOUP s1 | HLVL | 2367 | +0.0385 | 9 | 0.965 | +0.1067 | 55 | 0.618 |

## 11. Limits

* **HLVL is an ORACLE** (section 0): no HLVL row is performance and nothing here is a deployable-fix claim. HCV is deployable and does not close.
* Intervals answer *another draw of episodes* only (paired episode-cluster bootstrap, B 2,000, seed 0); X and Y share the checkpoint and the planner is deterministic; the two seeds per base are replicates of the rig, not a variance estimate. A separated interval is necessary, not sufficient, for a lever claim (`H-ESTIM-SEED-1`). Y is the reused `SREP_*` runs (KY, KA1).
* Tier T1-kin: executed state, logged cameras, own plan applied in the scene as a self-action open loop; it says nothing about re-perception, and the camera frames advance at the human's pace (section 9 (v)).
* S-POS rows of both arms are UNVERIFIED (card section 6); the STRATEGIC family is PAUSED (ITEM 25). The causal speed bucket is rebuilt from the CV history and is not separated from the rest of the history change.
* HCV removes all acceleration information, not only the trend; HCV's history (exactly constant at every row) is off the training distribution, and whether its harm comes from that or from the missing information is UNVERIFIED.
