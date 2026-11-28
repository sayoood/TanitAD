# RECOVER-NEXT, arm RPT (R's bank law at p 0.3 with `--recover-dev-shape pertick`): read result and the committed next arm

Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `01046fe6…`) section 3, the RECOVER-FT section 6 / 7 rules with R := RPT; addendum 4 (`PREREG_RECOVER_NEXT_ADDENDUM4_RPT.md`, frozen before any RPT run) rows M, I, N, J, K1, K2 decide the next arm. Tier **T1-kin** (self-action open loop on logged cameras: the model's own actions re-integrate the ego state). Estimator: paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only. Reader sha256 `de5cfbf572ba`, mode `production`, run clock 2026-11-05T08:40:17+0100. Evidence class MEASURED (ours): `RECOVER_READ_RPT.json` in this folder.

## Verdict

* **Label: HARMS** (H-RECOVER-NEXT-1, arm RPT). Controls: 20 run, 0 failed; KA1 bit-identity rerun: PASS=True. Guards per seed: s0 False, s1 False (reader guards.PASS False). M1 share closed RPT 0.0725 / 0.1508 against R 0.2198 / 0.1318 (exceeds R: s0 False, s1 True; bar 0.5).
* **Composite RPTC = RPT + `--smooth sg72 --feasible kbox` vs P10: NOT DOMINANT** (DOMINANT at both seeds: False; counted: False); failing terms s0: ['D2 (separated-worse: M1 crawl share)'], s1: ['D2 (separated-worse: M1 crawl share)']. D1 = True / True, D2 = False / False, D3 (four families, STRATEGIC included) = True / True; STRATEGIC = True / True.
* Frame-freeze reading **PARTIAL** (share closed under hold_now s0 0.1022, s1 0.1536; robust bar 0.5).
* **Committed next arm: addendum 4 row K2: the per-tick shape adds no closure at the dose measured: next arm is option (a), the level-from-image auxiliary of the decision item, as a PI-decision item with its costs; it does not start without the PI's decision and box-2 go.**

## RPT against R (same law, same dose, constant shape); RAPR (the largest closure before RPT) for reference (reported, not deciding)

| quantity | bar | R s0 | R s1 | RAPR s0 | RAPR s1 | RPT s0 | RPT s1 |
|---|---|---|---|---|---|---|---|
| M1 crawl share closed vs C0 | >= 0.5 | 0.2198 | 0.1318 | 0.4447 | 0.3175 | 0.0725 | 0.1508 |
| ADE6 X − C0 (m) | <= +0.0604 | +0.1892 | +0.0977 | +0.3504 | +0.3202 | +0.1668 | +0.0576 |
| non-crawl rows separated-worse vs C0 | none | ADE6, DK-HW, DK-VIOL, TAC-LAT, accel MAE, along-track MAE, cross-track MAE | DK-VIOL, along-track MAE | ADE6, DK-HW, DK-TG, DK-VIOL, along-track MAE, speed MAE | ADE6, DK-HW, DK-VIOL, along-track MAE, speed MAE | ADE6, DK-VIOL, TAC-LAT, along-track MAE | DK-HW, DK-VIOL, accel MAE, along-track MAE |
| composite M1 crawl share XC − P10 | not separated-worse | +0.0482 [+0.0282, +0.0715] | +0.0571 [+0.0340, +0.0853] | +0.0343 [+0.0171, +0.0550] | +0.0425 [+0.0227, +0.0671] | +0.0540 [+0.0325, +0.0781] | +0.0553 [+0.0328, +0.0822] |
| kbox's own M1 effect XC − X | not separated-worse | +0.0080 [+0.0032, +0.0142] | +0.0062 [+0.0019, +0.0121] | +0.0048 [+0.0012, +0.0092] | +0.0024 [-0.0011, +0.0057] | +0.0049 [+0.0018, +0.0089] | +0.0057 [+0.0022, +0.0102] |
| composite ADE6 XC − P10 (m) | <= -0.2415, separated | -0.9566 | -0.9344 | -0.7950 | -0.7253 | -0.9472 | -0.9619 |
| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | 0.0117 vs 0.0362 | 0.0078 vs 0.0038 | 0.0237 vs 0.0362 | 0.0189 vs 0.0038 | 0.0039 vs 0.0362 | 0.0090 vs 0.0038 |
| frame-freeze share closed | robust bar | 0.1674 | 0.1568 | 0.4826 | 0.3021 | 0.1022 | 0.1536 |

Closure-need estimate of the decision item (ESTIMATED, post hoc, not a rule; `rowk_decision/d2_closure_need.json`): about 0.68 (s0) / 0.60 (s1) of the M1 excess for the composite to stop being separated-worse on M1; RPT closes 0.0725 / 0.1508.

**The per-tick table** (RPT's launch-gate selfcheck, `levers.recover_table`; NOW columns compared with R's gate table at the same seed: expected equal when no record is dropped, since the NOW row is R's offset bit for bit)

| seed | n | kept | dropped (per-row deviation not finite) | ticks | R's n | NOW columns equal R's |
|---|---|---|---|---|---|---|
| s0 | 416254 | 416254 | 17047 | 16 | 433301 | mean_dv, mean_da, mean_dom, mean_dy, mean_dpsi differ |
| s1 | 407696 | 407696 | 16785 | 16 | 424481 | mean_dv, mean_da, mean_dom, mean_dy, mean_dpsi differ |

**Composite RPTC − P10, all 19 rows (Read A; D2 = no row separated-worse)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | -0.9472 [-1.1568, -0.7444] | yes | -0.9619 [-1.1867, -0.7320] | yes |
| LONGITUDINAL | speed MAE | lower | -0.8446 [-1.0816, -0.6331] | yes | -0.8230 [-1.0542, -0.6237] | yes |
| LONGITUDINAL | along-track MAE | lower | -0.4731 [-0.6699, -0.2836] | yes | -0.4126 [-0.6055, -0.2180] | yes |
| LONGITUDINAL | accel MAE | lower | -1.1160 [-1.5467, -0.7475] | yes | -1.1332 [-1.5896, -0.7571] | yes |
| LONGITUDINAL | DK-VIOL | lower | +0.0134 [-0.0108, +0.0380] | no | +0.0101 [-0.0155, +0.0372] | no |
| LONGITUDINAL | DK-HW | lower | -0.4054 [-0.8105, -0.0727] | yes | -0.3240 [-0.7160, -0.0072] | yes |
| LONGITUDINAL | DK-TG | lower | -0.1302 [-0.2225, -0.0567] | yes | -0.0999 [-0.1990, -0.0154] | yes |
| LATERAL | heading MAE | lower | -1.7259 [-2.7186, -0.6148] | yes | -1.7566 [-2.8939, -0.3721] | yes |
| LATERAL | yaw-rate MAE | lower | -12.3645 [-17.5446, -7.9106] | yes | -12.8012 [-18.2057, -8.3121] | yes |
| LATERAL | curvature MAE | lower | -0.0339 [-0.0498, -0.0199] | yes | -0.0375 [-0.0542, -0.0227] | yes |
| LATERAL | cross-track MAE | lower | -0.8679 [-1.0067, -0.7398] | yes | -0.9755 [-1.1404, -0.8216] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -3.4583 [-5.0159, -2.1254] | yes | -3.5650 [-5.1425, -2.2174] | yes |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | +0.0005 [-0.0029, +0.0044] | no | -0.0006 [-0.0040, +0.0033] | no |
| TACTICAL | TAC-LAT | higher | +0.1613 [+0.1175, +0.2052] | yes | +0.2172 [+0.1776, +0.2572] | yes |
| TACTICAL | TAC-LON | higher | +0.1667 [+0.1064, +0.2295] | yes | +0.1526 [+0.0934, +0.2101] | yes |
| STRATEGIC | S-POS@30 | lower | -1.0422 [-1.3493, -0.7558] | yes | -1.2572 [-1.5507, -0.9912] | yes |
| STRATEGIC | S-POS@50 | lower | -3.5877 [-4.4807, -2.7647] | yes | -3.9225 [-4.8160, -3.1093] | yes |
| STRATEGIC | S-POS@80 | lower | -9.0648 [-10.9934, -7.1920] | yes | -9.5506 [-11.5362, -7.7250] | yes |
| crawl | M1 crawl share | lower | +0.0540 [+0.0325, +0.0781] | yes | +0.0553 [+0.0328, +0.0822] | yes |

**RPT − C0, all 19 rows (the label's guards read the non-crawl rows)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | +0.1668 [+0.0903, +0.2467] | yes | +0.0576 [-0.0594, +0.1794] | no |
| LONGITUDINAL | speed MAE | lower | -0.0226 [-0.0720, +0.0218] | no | -0.0005 [-0.0462, +0.0495] | no |
| LONGITUDINAL | along-track MAE | lower | +0.1374 [+0.0510, +0.2295] | yes | +0.1083 [+0.0104, +0.2162] | yes |
| LONGITUDINAL | accel MAE | lower | -0.0161 [-0.0837, +0.0468] | no | +0.0871 [+0.0124, +0.1684] | yes |
| LONGITUDINAL | DK-VIOL | lower | +0.0134 [+0.0017, +0.0266] | yes | +0.0218 [+0.0030, +0.0457] | yes |
| LONGITUDINAL | DK-HW | lower | +0.0301 [-0.1554, +0.2202] | no | +0.2770 [+0.0279, +0.5314] | yes |
| LONGITUDINAL | DK-TG | lower | +0.0076 [-0.0489, +0.0580] | no | +0.0218 [-0.1101, +0.1290] | no |
| LATERAL | heading MAE | lower | -0.3790 [-0.8043, +0.0588] | no | -0.8879 [-1.3604, -0.4002] | yes |
| LATERAL | yaw-rate MAE | lower | -2.3710 [-3.9632, -0.9743] | yes | -2.8205 [-5.4157, -0.5820] | yes |
| LATERAL | curvature MAE | lower | -0.0190 [-0.0315, -0.0078] | yes | -0.0258 [-0.0445, -0.0082] | yes |
| LATERAL | cross-track MAE | lower | +0.0454 [-0.0056, +0.0996] | no | -0.0928 [-0.1916, -0.0097] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -0.2754 [-0.5979, +0.0063] | no | -0.1370 [-0.4799, +0.2302] | no |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0054 [-0.0149, +0.0024] | no | -0.0055 [-0.0141, +0.0026] | no |
| TACTICAL | TAC-LAT | higher | -0.0350 [-0.0700, -0.0038] | yes | +0.0704 [+0.0454, +0.0980] | yes |
| TACTICAL | TAC-LON | higher | +0.0108 [-0.0167, +0.0405] | no | +0.0100 [-0.0192, +0.0417] | no |
| STRATEGIC | S-POS@30 | lower | -0.4705 [-0.6975, -0.2506] | yes | -1.0910 [-1.4382, -0.7556] | yes |
| STRATEGIC | S-POS@50 | lower | -0.8268 [-1.2413, -0.4391] | yes | -1.7836 [-2.3736, -1.2444] | yes |
| STRATEGIC | S-POS@80 | lower | -1.2383 [-1.9513, -0.5537] | yes | -2.5296 [-3.5144, -1.6725] | yes |
| crawl | M1 crawl share | lower | -0.0039 [-0.0103, +0.0025] | no | -0.0090 [-0.0141, -0.0044] | yes |

**RPT − C0 against R − C0, all rows** (two separate paired reads against C0; there is no paired RPT − R interval, and none is claimed)

| family | row | better | R s0 | RPT s0 | R s1 | RPT s1 |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | +0.1892 sep | +0.1668 sep | +0.0977 | +0.0576 |
| LONGITUDINAL | speed MAE | lower | +0.0203 | -0.0226 | -0.0121 | -0.0005 |
| LONGITUDINAL | along-track MAE | lower | +0.1360 sep | +0.1374 sep | +0.1498 sep | +0.1083 sep |
| LONGITUDINAL | accel MAE | lower | +0.1169 sep | -0.0161 | +0.0237 | +0.0871 sep |
| LONGITUDINAL | DK-VIOL | lower | +0.0202 sep | +0.0134 sep | +0.0235 sep | +0.0218 sep |
| LONGITUDINAL | DK-HW | lower | +0.2087 sep | +0.0301 | +0.1916 | +0.2770 sep |
| LONGITUDINAL | DK-TG | lower | +0.0135 | +0.0076 | -0.0167 | +0.0218 |
| LATERAL | heading MAE | lower | -0.5185 sep | -0.3790 | -1.0389 sep | -0.8879 sep |
| LATERAL | yaw-rate MAE | lower | -2.2547 sep | -2.3710 sep | -2.8226 sep | -2.8205 sep |
| LATERAL | curvature MAE | lower | -0.0263 sep | -0.0190 sep | -0.0149 | -0.0258 sep |
| LATERAL | cross-track MAE | lower | +0.0593 sep | +0.0454 | -0.0769 | -0.0928 sep |
| LATERAL | L1 lateral-accel MAE | lower | -0.0018 | -0.2754 | -0.2182 | -0.1370 |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0081 | -0.0054 | -0.0043 | -0.0055 |
| TACTICAL | TAC-LAT | higher | -0.0484 sep | -0.0350 sep | +0.0654 sep | +0.0704 sep |
| TACTICAL | TAC-LON | higher | -0.0133 | +0.0108 | +0.0208 | +0.0100 |
| STRATEGIC | S-POS@30 | lower | -0.4072 sep | -0.4705 sep | -0.8136 sep | -1.0910 sep |
| STRATEGIC | S-POS@50 | lower | -0.7248 sep | -0.8268 sep | -1.2966 sep | -1.7836 sep |
| STRATEGIC | S-POS@80 | lower | -1.2410 sep | -1.2383 sep | -1.8015 sep | -2.5296 sep |
| crawl | M1 crawl share | lower | -0.0117 sep | -0.0039 | -0.0078 sep | -0.0090 sep |

## What this establishes, and what it cannot

* RPT and R share the bank law, the dose p 0.3, the random stream and the drawn record; they differ only in the shape of the deviation over the history window. The comparison is two separate paired reads against C0, not a paired RPT − R read.
* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the training-run question is the C0rep floor row above. T1-kin is not T2: nothing here is a driving claim beyond T1-kin.
* Every row M, I, N, J, K1, K2 is reached by a mutation test (`test_gen_rpt.py`), written and run before the read.

## Next (Rule Zero)

* addendum 4 row K2 decides: the per-tick shape adds no closure at the dose measured: next arm is option (a), the level-from-image auxiliary of the decision item, as a PI-decision item with its costs; it does not start without the PI's decision and box-2 go. Any new box-2 arm needs the PI's go (box 2 runs one job at a time, SAFE MODE).
