# RECOVER-NEXT, arm RP01 (`--recover-p 0.1`): read result, Q1 and the committed next arm, 2026-11-04

Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `01046fe6…`) section 3, the RECOVER-FT section 6 / 7 rules with R := RP01; addendum `PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md` (frozen 2026-11-04 01:02, before any RTL or RP01 rollout). Tier **T1-kin** (self-action open loop on logged cameras; not closed loop). Estimator: paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only. Reader sha256 `aeee6b1b1351`, mode `production`, run clock 2026-11-04T05:45:04+0100. Combiner sha256 `c6b4f31551a6`, run clock 2026-11-04T05:46:22+0100. Evidence class MEASURED (ours): `RECOVER_READ_RP01.json`, `RECOVER_NEXT_after_RP01.json` in this folder; RTL's read is `../2026-11-04-recover-next-rtl/RECOVER_READ_RTL.json`.

## Verdict

* **Label: HARMS** (H-RECOVER-NEXT-1, arm RP01). Controls: 20 run, 0 failed; KA1 bit-identity rerun of RP01_s0: PASS=True. Guards PASS (both seeds): False.
* **Composite RP01C = RP01 + `--smooth sg72 --feasible kbox` vs P10: NOT DOMINANT**; failing terms s0: ['D2 (separated-worse: M1 crawl share)'], s1: ['D2 (separated-worse: M1 crawl share)']. D1 = True / True, D3 (four families, STRATEGIC included) = True / True.
* Frame-freeze reading **PARTIAL** (share closed under hold_now s0 0.1546, s1 0.1265; robust bar 0.5).
* Strategic family: reported, D3_by_family STRATEGIC = True / True; for v8F STRATEGIC is reported and counted in D3.
* **Q1 (did the rejoin blend or the dose harm?): neither** (rule: blend harmed iff RTL's guards PASS at both seeds; dose harmed iff RP01's guards PASS at both seeds; both / neither otherwise; blend harmed False, dose harmed False).
* **Committed next arm: addendum row D: RAPR as frozen (RAPR_0 / RAPR_1, argv already in `argv/`, unchanged).** Composite DOMINANT and counted: RTLC False, RP01C False. The combiner's own card-section-4 table reads: UNCOVERED: no row of card section 4 matches (RAPR has no usable read); the addendum rows decide.

## RP01 against R and RTL (each changes one flag of R)

| quantity | bar | R s0 | R s1 | RTL s0 | RTL s1 | RP01 s0 | RP01 s1 |
|---|---|---|---|---|---|---|---|
| M1 crawl share closed vs C0 | >= 0.5 | 0.2198 | 0.1318 | 0.2131 | 0.1933 | 0.1473 | 0.1342 |
| ADE6 X − C0 (m) | <= +0.0604 | +0.1892 | +0.0977 | +0.1492 | +0.0925 | +0.0017 | +0.0008 |
| non-crawl rows separated-worse vs C0 | none | ADE6, DK-HW, DK-VIOL, TAC-LAT, accel MAE, along-track MAE, cross-track MAE | DK-VIOL, along-track MAE | ADE6, along-track MAE | DK-HW, DK-VIOL, along-track MAE | DK-VIOL | DK-VIOL, accel MAE |
| composite M1 crawl share XC − P10 | not separated-worse | +0.0482 [+0.0282, +0.0715] | +0.0571 [+0.0340, +0.0853] | +0.0484 [+0.0282, +0.0715] | +0.0540 [+0.0318, +0.0805] | +0.0513 [+0.0319, +0.0738] | +0.0559 [+0.0340, +0.0824] |
| kbox's own M1 effect XC − X | not separated-worse | +0.0080 [+0.0032, +0.0142] | +0.0062 [+0.0019, +0.0121] | +0.0066 [+0.0027, +0.0112] | +0.0070 [+0.0025, +0.0127] | +0.0068 [+0.0022, +0.0128] | +0.0050 [+0.0009, +0.0101] |
| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | 0.0117 vs 0.0362 | 0.0078 vs 0.0038 | 0.0113 vs 0.0362 | 0.0115 vs 0.0038 | 0.0078 vs 0.0362 | 0.0080 vs 0.0038 |
| frame-freeze share closed | robust bar | 0.1674 | 0.1568 | 0.1860 | 0.1899 | 0.1546 | 0.1265 |

**Composite RP01C − P10, all 19 rows (Read A; D2 = no row separated-worse)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | -1.1647 [-1.3754, -0.9645] | yes | -1.0226 [-1.2422, -0.8039] | yes |
| LONGITUDINAL | speed MAE | lower | -0.8850 [-1.1240, -0.6794] | yes | -0.8034 [-1.0310, -0.6049] | yes |
| LONGITUDINAL | along-track MAE | lower | -0.6269 [-0.8083, -0.4560] | yes | -0.4548 [-0.6302, -0.2809] | yes |
| LONGITUDINAL | accel MAE | lower | -1.1631 [-1.6176, -0.7801] | yes | -1.1012 [-1.5390, -0.7402] | yes |
| LONGITUDINAL | DK-VIOL | lower | +0.0202 [-0.0112, +0.0526] | no | +0.0067 [-0.0166, +0.0314] | no |
| LONGITUDINAL | DK-HW | lower | -0.4175 [-0.8164, -0.0489] | yes | -0.3927 [-0.7579, -0.1070] | yes |
| LONGITUDINAL | DK-TG | lower | -0.1047 [-0.1760, -0.0460] | yes | -0.0929 [-0.1766, -0.0294] | yes |
| LATERAL | heading MAE | lower | -2.3031 [-3.2333, -1.2146] | yes | -1.6447 [-2.7652, -0.3573] | yes |
| LATERAL | yaw-rate MAE | lower | -12.8064 [-18.1870, -8.2845] | yes | -12.4083 [-17.6390, -8.0108] | yes |
| LATERAL | curvature MAE | lower | -0.0350 [-0.0511, -0.0211] | yes | -0.0343 [-0.0504, -0.0202] | yes |
| LATERAL | cross-track MAE | lower | -0.9816 [-1.1304, -0.8462] | yes | -0.9923 [-1.1570, -0.8376] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -3.5528 [-5.1246, -2.2016] | yes | -3.4556 [-4.9956, -2.1320] | yes |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0010 [-0.0043, +0.0024] | no | -0.0003 [-0.0036, +0.0032] | no |
| TACTICAL | TAC-LAT | higher | +0.1863 [+0.1502, +0.2247] | yes | +0.1905 [+0.1500, +0.2334] | yes |
| TACTICAL | TAC-LON | higher | +0.1513 [+0.0917, +0.2079] | yes | +0.1592 [+0.1005, +0.2168] | yes |
| STRATEGIC | S-POS@30 | lower | -0.6673 [-0.9306, -0.4274] | yes | -0.9518 [-1.2298, -0.6969] | yes |
| STRATEGIC | S-POS@50 | lower | -2.8860 [-3.6805, -2.1583] | yes | -3.3881 [-4.2217, -2.6228] | yes |
| STRATEGIC | S-POS@80 | lower | -7.7606 [-9.5173, -6.1333] | yes | -8.7818 [-10.6607, -6.9614] | yes |
| crawl | M1 crawl share | lower | +0.0513 [+0.0319, +0.0738] | yes | +0.0559 [+0.0340, +0.0824] | yes |

**RP01 − C0, all 19 rows (the label's guards read the non-crawl rows)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | +0.0017 [-0.0490, +0.0528] | no | +0.0008 [-0.0910, +0.0754] | no |
| LONGITUDINAL | speed MAE | lower | -0.0177 [-0.0543, +0.0173] | no | +0.0196 [-0.0101, +0.0497] | no |
| LONGITUDINAL | along-track MAE | lower | +0.0045 [-0.0440, +0.0543] | no | +0.0499 [-0.0030, +0.1052] | no |
| LONGITUDINAL | accel MAE | lower | +0.0537 [-0.0236, +0.1408] | no | +0.0933 [+0.0285, +0.1648] | yes |
| LONGITUDINAL | DK-VIOL | lower | +0.0235 [+0.0051, +0.0475] | yes | +0.0202 [+0.0052, +0.0374] | yes |
| LONGITUDINAL | DK-HW | lower | +0.0419 [-0.1167, +0.1992] | no | +0.1116 [-0.0376, +0.2568] | no |
| LONGITUDINAL | DK-TG | lower | +0.0016 [-0.0253, +0.0265] | no | -0.0270 [-0.1401, +0.0445] | no |
| LATERAL | heading MAE | lower | -0.3635 [-0.6836, -0.0268] | yes | -0.8565 [-1.3070, -0.4342] | yes |
| LATERAL | yaw-rate MAE | lower | -1.7137 [-2.9985, -0.4691] | yes | -3.5909 [-5.5935, -1.9147] | yes |
| LATERAL | curvature MAE | lower | -0.0098 [-0.0237, +0.0050] | no | -0.0383 [-0.0556, -0.0223] | yes |
| LATERAL | cross-track MAE | lower | -0.0187 [-0.0582, +0.0226] | no | -0.0812 [-0.1706, -0.0096] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -0.0874 [-0.4674, +0.2406] | no | -0.1071 [-0.3519, +0.1567] | no |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0078 [-0.0188, +0.0009] | no | -0.0111 [-0.0249, -0.0005] | yes |
| TACTICAL | TAC-LAT | higher | -0.0117 [-0.0329, +0.0063] | no | +0.0396 [+0.0196, +0.0650] | yes |
| TACTICAL | TAC-LON | higher | -0.0054 [-0.0242, +0.0125] | no | +0.0096 [-0.0083, +0.0292] | no |
| STRATEGIC | S-POS@30 | lower | -0.3006 [-0.4948, -0.1260] | yes | -0.6275 [-0.8610, -0.4161] | yes |
| STRATEGIC | S-POS@50 | lower | -0.4900 [-0.8209, -0.1849] | yes | -1.0081 [-1.3707, -0.6663] | yes |
| STRATEGIC | S-POS@80 | lower | -0.8062 [-1.3608, -0.2889] | yes | -1.4781 [-2.0754, -0.9241] | yes |
| crawl | M1 crawl share | lower | -0.0078 [-0.0125, -0.0036] | yes | -0.0080 [-0.0130, -0.0036] | yes |

## What this establishes, and what it cannot

* Lowering the dose to p 0.1 does not remove RECOVER-FT's harm by the card's guards: ADE6 RP01 − C0 +0.0017 / +0.0008 m against the +0.0604 m guard. With RTL's guards failing at both seeds, Q1 = `neither`.
* M1 crawl share closed vs C0: RP01 0.1473 / 0.1342, RTL 0.2131 / 0.1933, R 0.2198 / 0.1318 (bar 0.5).
* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the training-run question is the C0rep floor row above.

## Next (Rule Zero; addendum rows, frozen before any RTL or RP01 rollout)

* Row D decides: RAPR as frozen (RAPR_0 / RAPR_1, argv already in `argv/`, unchanged). It needs the PI's box-2 go (box 2 runs one job at a time, SAFE MODE).
