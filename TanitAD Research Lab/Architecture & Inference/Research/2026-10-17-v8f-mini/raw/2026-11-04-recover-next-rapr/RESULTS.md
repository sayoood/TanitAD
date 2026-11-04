# RECOVER-NEXT, arm RAPR (`--recover-dev-law apriori`): read result and the committed next arm, 2026-11-04

Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `01046fe6…`) section 3, the RECOVER-FT section 6 / 7 rules with R := RAPR; section 4 (the combiner's `next_arm`), addendum 1 (`PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md`, frozen 2026-11-04 01:02) and addendum 2 (`PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md`, frozen 2026-11-04 14:35:47, before any RAPR checkpoint). Tier **T1-kin** (self-action open loop on logged cameras; not closed loop). Estimator: paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only. Reader sha256 `aeee6b1b1351`, mode `production`, run clock 2026-11-04T16:16:51+0100. Combiner sha256 `c6b4f31551a6`, run clock 2026-11-04T16:18:05+0100. Evidence class MEASURED (ours): `RECOVER_READ_RAPR.json`, `RECOVER_NEXT_after_RAPR.json` in this folder.

## Verdict

* **Label: HARMS** (H-RECOVER-NEXT-1, arm RAPR). Controls: 20 run, 0 failed; KA1 bit-identity rerun of RAPR_s0: PASS=True. Guards PASS (both seeds): False.
* **Composite RAPRC = RAPR + `--smooth sg72 --feasible kbox` vs P10: NOT DOMINANT** (counted: False); failing terms s0: ['D2 (separated-worse: DK-VIOL, M1 crawl share)'], s1: ['D2 (separated-worse: M1 crawl share)']. D1 = True / True, D3 (four families, STRATEGIC included) = True / True; STRATEGIC = True / True.
* Frame-freeze reading **PARTIAL** (share closed under hold_now s0 0.4826, s1 0.3021; robust bar 0.5).
* **RAPR secondary** (RAPR's M1 share closed exceeds R's (0.220 / 0.132 in the RECOVER-FT read, read from its JSON) at each seed): exceeds per seed {'s0': True, 's1': True}; at both True, at neither False.
* **Committed next arm: addendum 2 row E: RAPR-P01: RAPR's frozen argv with `--recover-p 0.1` (2 runs).** Combiner (card section 4): SELECTED; matching rows ['row3']. Q1 (from the RTL and RP01 reads): neither.

## RAPR against R, RTL and RP01 (each changes one setting of R)

| quantity | bar | R s0 | R s1 | RTL s0 | RTL s1 | RP01 s0 | RP01 s1 | RAPR s0 | RAPR s1 |
|---|---|---|---|---|---|---|---|---|---|
| M1 crawl share closed vs C0 | >= 0.5 | 0.2198 | 0.1318 | 0.2131 | 0.1933 | 0.1473 | 0.1342 | 0.4447 | 0.3175 |
| ADE6 X − C0 (m) | <= +0.0604 | +0.1892 | +0.0977 | +0.1492 | +0.0925 | +0.0017 | +0.0008 | +0.3504 | +0.3202 |
| non-crawl rows separated-worse vs C0 | none | ADE6, DK-HW, DK-VIOL, TAC-LAT, accel MAE, along-track MAE, cross-track MAE | DK-VIOL, along-track MAE | ADE6, along-track MAE | DK-HW, DK-VIOL, along-track MAE | DK-VIOL | DK-VIOL, accel MAE | ADE6, DK-HW, DK-TG, DK-VIOL, along-track MAE, speed MAE | ADE6, DK-HW, DK-VIOL, along-track MAE, speed MAE |
| composite M1 crawl share XC − P10 | not separated-worse | +0.0482 [+0.0282, +0.0715] | +0.0571 [+0.0340, +0.0853] | +0.0484 [+0.0282, +0.0715] | +0.0540 [+0.0318, +0.0805] | +0.0513 [+0.0319, +0.0738] | +0.0559 [+0.0340, +0.0824] | +0.0343 [+0.0171, +0.0550] | +0.0425 [+0.0227, +0.0671] |
| kbox's own M1 effect XC − X | not separated-worse | +0.0080 [+0.0032, +0.0142] | +0.0062 [+0.0019, +0.0121] | +0.0066 [+0.0027, +0.0112] | +0.0070 [+0.0025, +0.0127] | +0.0068 [+0.0022, +0.0128] | +0.0050 [+0.0009, +0.0101] | +0.0048 [+0.0012, +0.0092] | +0.0024 [-0.0011, +0.0057] |
| composite ADE6 XC − P10 (m) | <= -0.2415, separated | -0.9566 | -0.9344 | -0.9766 | -0.9353 | -1.1647 | -1.0226 | -0.7950 | -0.7253 |
| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | 0.0117 vs 0.0362 | 0.0078 vs 0.0038 | 0.0113 vs 0.0362 | 0.0115 vs 0.0038 | 0.0078 vs 0.0362 | 0.0080 vs 0.0038 | 0.0237 vs 0.0362 | 0.0189 vs 0.0038 |
| frame-freeze share closed | robust bar | 0.1674 | 0.1568 | 0.1860 | 0.1899 | 0.1546 | 0.1265 | 0.4826 | 0.3021 |

**Composite RAPRC − P10, all 19 rows (Read A; D2 = no row separated-worse)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | -0.7950 [-1.0087, -0.5683] | yes | -0.7253 [-0.9530, -0.4948] | yes |
| LONGITUDINAL | speed MAE | lower | -0.7380 [-0.9601, -0.5434] | yes | -0.7002 [-0.9249, -0.5000] | yes |
| LONGITUDINAL | along-track MAE | lower | -0.1986 [-0.3911, -0.0026] | yes | -0.1309 [-0.3401, +0.0734] | no |
| LONGITUDINAL | accel MAE | lower | -1.0678 [-1.5168, -0.6940] | yes | -1.1189 [-1.5670, -0.7460] | yes |
| LONGITUDINAL | DK-VIOL | lower | +0.0487 [+0.0092, +0.1017] | yes | +0.0437 [+0.0000, +0.1007] | no |
| LONGITUDINAL | DK-HW | lower | +0.0829 [-0.4174, +0.6014] | no | -0.0064 [-0.4890, +0.5136] | no |
| LONGITUDINAL | DK-TG | lower | -0.0490 [-0.1426, +0.0294] | no | -0.0872 [-0.1779, -0.0176] | yes |
| LATERAL | heading MAE | lower | -2.7848 [-3.8519, -1.5727] | yes | -2.3451 [-3.3791, -1.1121] | yes |
| LATERAL | yaw-rate MAE | lower | -13.0598 [-18.4435, -8.5110] | yes | -13.1595 [-18.4534, -8.6282] | yes |
| LATERAL | curvature MAE | lower | -0.0406 [-0.0576, -0.0255] | yes | -0.0415 [-0.0589, -0.0264] | yes |
| LATERAL | cross-track MAE | lower | -1.0203 [-1.1821, -0.8690] | yes | -0.9915 [-1.1497, -0.8454] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -3.6006 [-5.1679, -2.2555] | yes | -3.6094 [-5.1462, -2.2854] | yes |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0024 [-0.0057, +0.0011] | no | -0.0025 [-0.0059, +0.0011] | no |
| TACTICAL | TAC-LAT | higher | +0.2197 [+0.1793, +0.2600] | yes | +0.2063 [+0.1668, +0.2463] | yes |
| TACTICAL | TAC-LON | higher | +0.1467 [+0.0938, +0.1989] | yes | +0.1509 [+0.0962, +0.2029] | yes |
| STRATEGIC | S-POS@30 | lower | -0.6707 [-0.9492, -0.4052] | yes | -0.8617 [-1.1701, -0.5882] | yes |
| STRATEGIC | S-POS@50 | lower | -2.7554 [-3.6177, -1.9281] | yes | -3.2511 [-4.1109, -2.4493] | yes |
| STRATEGIC | S-POS@80 | lower | -7.3925 [-9.2839, -5.5507] | yes | -8.5778 [-10.5430, -6.7508] | yes |
| crawl | M1 crawl share | lower | +0.0343 [+0.0171, +0.0550] | yes | +0.0425 [+0.0227, +0.0671] | yes |

**RAPR − C0, all 19 rows (the label's guards read the non-crawl rows)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | +0.3504 [+0.2163, +0.4820] | yes | +0.3202 [+0.1809, +0.4504] | yes |
| LONGITUDINAL | speed MAE | lower | +0.1138 [+0.0555, +0.1723] | yes | +0.1082 [+0.0471, +0.1658] | yes |
| LONGITUDINAL | along-track MAE | lower | +0.4128 [+0.2734, +0.5523] | yes | +0.3955 [+0.2582, +0.5314] | yes |
| LONGITUDINAL | accel MAE | lower | +0.0319 [-0.0220, +0.0817] | no | +0.0096 [-0.0477, +0.0667] | no |
| LONGITUDINAL | DK-VIOL | lower | +0.0538 [+0.0179, +0.1025] | yes | +0.0588 [+0.0179, +0.1131] | yes |
| LONGITUDINAL | DK-HW | lower | +0.5825 [+0.2492, +0.9977] | yes | +0.5053 [+0.1454, +0.9499] | yes |
| LONGITUDINAL | DK-TG | lower | +0.0763 [+0.0068, +0.1496] | yes | +0.0009 [-0.1176, +0.0812] | no |
| LATERAL | heading MAE | lower | -1.2936 [-1.7854, -0.7690] | yes | -2.0080 [-2.6517, -1.4186] | yes |
| LATERAL | yaw-rate MAE | lower | -4.6536 [-6.3383, -3.1140] | yes | -8.1734 [-11.3855, -5.4667] | yes |
| LATERAL | curvature MAE | lower | -0.0564 [-0.0770, -0.0377] | yes | -0.0723 [-0.1014, -0.0465] | yes |
| LATERAL | cross-track MAE | lower | -0.0585 [-0.1141, -0.0026] | yes | -0.0728 [-0.1538, -0.0054] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -0.2183 [-0.4470, -0.0149] | yes | -0.6013 [-0.9040, -0.3422] | yes |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0242 [-0.0413, -0.0118] | yes | -0.0308 [-0.0528, -0.0120] | yes |
| TACTICAL | TAC-LAT | higher | +0.0263 [+0.0138, +0.0387] | yes | +0.0634 [+0.0408, +0.0913] | yes |
| TACTICAL | TAC-LON | higher | -0.0025 [-0.0388, +0.0342] | no | +0.0021 [-0.0300, +0.0367] | no |
| STRATEGIC | S-POS@30 | lower | -0.5725 [-0.8041, -0.3525] | yes | -0.5793 [-0.8175, -0.3587] | yes |
| STRATEGIC | S-POS@50 | lower | -0.9559 [-1.3476, -0.5771] | yes | -0.8821 [-1.2783, -0.5011] | yes |
| STRATEGIC | S-POS@80 | lower | -1.4443 [-2.1093, -0.8203] | yes | -1.2996 [-1.9220, -0.7014] | yes |
| crawl | M1 crawl share | lower | -0.0237 [-0.0326, -0.0159] | yes | -0.0189 [-0.0276, -0.0112] | yes |

## What this establishes, and what it cannot

* The a-priori law (dv ~ U[−3, 0] m/s, heavier on the slow side) against the bank law at the same dose p 0.3: M1 crawl share closed RAPR 0.4447 / 0.3175, R 0.2198 / 0.1318 (bar 0.5).
* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the training-run question is the C0rep floor row above. T1-kin is not T2: nothing here is a driving claim beyond T1-kin.

## Next (Rule Zero)

* addendum 2 row E decides: RAPR-P01: RAPR's frozen argv with `--recover-p 0.1` (2 runs). Any new box-2 arm needs the PI's go (box 2 runs one job at a time, SAFE MODE).
