# RECOVER-NEXT, arm RAPR-P01 (`--recover-dev-law apriori --recover-p 0.1`): read result and the committed next arm, 2026-11-04

Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `01046fe6…`) section 3, the RECOVER-FT section 6 / 7 rules with R := RAPR-P01; addendum 3 (`PREREG_RECOVER_NEXT_ADDENDUM3_RAPRP01.md`, sha256 `9bea3fa1…`, frozen 2026-11-04 16:29:29, before any RAPR-P01 run) rows M, I, N, J, K decide the next arm. Tier **T1-kin** (self-action open loop on logged cameras; not closed loop). Estimator: paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only. Reader sha256 `a99104733397`, mode `production`, run clock 2026-11-04T20:37:46+0100. Evidence class MEASURED (ours): `RECOVER_READ_RAPRP01.json` in this folder.

## Verdict

* **Label: HARMS** (H-RECOVER-NEXT-1, arm RAPR-P01). Controls: 20 run, 0 failed; KA1 bit-identity rerun: PASS=True. Guards per seed: s0 False, s1 False (reader guards.PASS False).
* **Composite RAPRP01C = RAPR-P01 + `--smooth sg72 --feasible kbox` vs P10: NOT DOMINANT** (DOMINANT at both seeds: False; counted: False); failing terms s0: ['D2 (separated-worse: M1 crawl share)'], s1: ['D2 (separated-worse: M1 crawl share)']. D1 = True / True, D2 = False / False, D3 (four families, STRATEGIC included) = True / True; STRATEGIC = True / True.
* Frame-freeze reading **PARTIAL** (share closed under hold_now s0 0.2344, s1 0.1055; robust bar 0.5).
* **Committed next arm: addendum 3 row K: the one-offset-per-window recovery recipe is closed at the laws and doses measured (bank law at p 0.3 and 0.1, a-priori law at p 0.3 and 0.1); next arm: a PI-decision item between (a) card row 4's level-from-image auxiliary (architecture change) and (b) a per-tick deviation recipe (a new trainer lever with its own launch-gate entry); neither starts without the PI's decision and box-2 go.**

## RAPR-P01 against RP01 (bank law, same dose) and RAPR (same law, p 0.3); R for reference (reported, not deciding)

| quantity | bar | R s0 | R s1 | RP01 s0 | RP01 s1 | RAPR s0 | RAPR s1 | RAPR-P01 s0 | RAPR-P01 s1 |
|---|---|---|---|---|---|---|---|---|---|
| M1 crawl share closed vs C0 | >= 0.5 | 0.2198 | 0.1318 | 0.1473 | 0.1342 | 0.4447 | 0.3175 | 0.2147 | 0.1391 |
| ADE6 X − C0 (m) | <= +0.0604 | +0.1892 | +0.0977 | +0.0017 | +0.0008 | +0.3504 | +0.3202 | +0.0371 | +0.0429 |
| non-crawl rows separated-worse vs C0 | none | ADE6, DK-HW, DK-VIOL, TAC-LAT, accel MAE, along-track MAE, cross-track MAE | DK-VIOL, along-track MAE | DK-VIOL | DK-VIOL, accel MAE | ADE6, DK-HW, DK-TG, DK-VIOL, along-track MAE, speed MAE | ADE6, DK-HW, DK-VIOL, along-track MAE, speed MAE | DK-TG, DK-VIOL, along-track MAE | DK-VIOL, along-track MAE |
| composite M1 crawl share XC − P10 | not separated-worse | +0.0482 [+0.0282, +0.0715] | +0.0571 [+0.0340, +0.0853] | +0.0513 [+0.0319, +0.0738] | +0.0559 [+0.0340, +0.0824] | +0.0343 [+0.0171, +0.0550] | +0.0425 [+0.0227, +0.0671] | +0.0478 [+0.0300, +0.0693] | +0.0522 [+0.0304, +0.0783] |
| kbox's own M1 effect XC − X | not separated-worse | +0.0080 [+0.0032, +0.0142] | +0.0062 [+0.0019, +0.0121] | +0.0068 [+0.0022, +0.0128] | +0.0050 [+0.0009, +0.0101] | +0.0048 [+0.0012, +0.0092] | +0.0024 [-0.0011, +0.0057] | +0.0060 [+0.0022, +0.0110] | +0.0024 [-0.0012, +0.0064] |
| composite ADE6 XC − P10 (m) | <= -0.2415, separated | -0.9566 | -0.9344 | -1.1647 | -1.0226 | -0.7950 | -0.7253 | -1.1141 | -0.9657 |
| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | 0.0117 vs 0.0362 | 0.0078 vs 0.0038 | 0.0078 vs 0.0362 | 0.0080 vs 0.0038 | 0.0237 vs 0.0362 | 0.0189 vs 0.0038 | 0.0114 vs 0.0362 | 0.0083 vs 0.0038 |
| frame-freeze share closed | robust bar | 0.1674 | 0.1568 | 0.1546 | 0.1265 | 0.4826 | 0.3021 | 0.2344 | 0.1055 |

**Composite RAPRP01C − P10, all 19 rows (Read A; D2 = no row separated-worse)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | -1.1141 [-1.3076, -0.9186] | yes | -0.9657 [-1.1670, -0.7680] | yes |
| LONGITUDINAL | speed MAE | lower | -0.8487 [-1.0762, -0.6530] | yes | -0.7843 [-1.0073, -0.5924] | yes |
| LONGITUDINAL | along-track MAE | lower | -0.5659 [-0.7333, -0.4042] | yes | -0.4297 [-0.6054, -0.2605] | yes |
| LONGITUDINAL | accel MAE | lower | -1.1415 [-1.5945, -0.7629] | yes | -1.1673 [-1.6125, -0.7885] | yes |
| LONGITUDINAL | DK-VIOL | lower | +0.0235 [-0.0054, +0.0569] | no | +0.0034 [-0.0242, +0.0343] | no |
| LONGITUDINAL | DK-HW | lower | -0.3410 [-0.7332, -0.0127] | yes | -0.3658 [-0.7362, -0.0505] | yes |
| LONGITUDINAL | DK-TG | lower | -0.0933 [-0.1621, -0.0370] | yes | -0.0897 [-0.1741, -0.0173] | yes |
| LATERAL | heading MAE | lower | -1.9085 [-2.9358, -0.7150] | yes | -1.6918 [-2.7614, -0.5488] | yes |
| LATERAL | yaw-rate MAE | lower | -12.9573 [-18.3344, -8.3959] | yes | -12.8914 [-18.1885, -8.4174] | yes |
| LATERAL | curvature MAE | lower | -0.0366 [-0.0533, -0.0223] | yes | -0.0365 [-0.0528, -0.0225] | yes |
| LATERAL | cross-track MAE | lower | -0.9879 [-1.1412, -0.8471] | yes | -0.9253 [-1.0722, -0.7920] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -3.6246 [-5.1743, -2.2843] | yes | -3.6015 [-5.1482, -2.2785] | yes |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0011 [-0.0043, +0.0023] | no | -0.0013 [-0.0047, +0.0022] | no |
| TACTICAL | TAC-LAT | higher | +0.2076 [+0.1681, +0.2470] | yes | +0.1863 [+0.1467, +0.2284] | yes |
| TACTICAL | TAC-LON | higher | +0.1567 [+0.0975, +0.2132] | yes | +0.1467 [+0.0913, +0.1982] | yes |
| STRATEGIC | S-POS@30 | lower | -0.6893 [-0.9681, -0.4398] | yes | -0.8667 [-1.1681, -0.5965] | yes |
| STRATEGIC | S-POS@50 | lower | -2.7986 [-3.6347, -2.0228] | yes | -3.3122 [-4.1645, -2.5335] | yes |
| STRATEGIC | S-POS@80 | lower | -7.4551 [-9.2403, -5.6819] | yes | -8.7720 [-10.7664, -6.9409] | yes |
| crawl | M1 crawl share | lower | +0.0478 [+0.0300, +0.0693] | yes | +0.0522 [+0.0304, +0.0783] | yes |

**RAPR-P01 − C0, all 19 rows (the label's guards read the non-crawl rows)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | +0.0371 [-0.0154, +0.0886] | no | +0.0429 [-0.0169, +0.0991] | no |
| LONGITUDINAL | speed MAE | lower | +0.0223 [-0.0119, +0.0565] | no | +0.0166 [-0.0190, +0.0509] | no |
| LONGITUDINAL | along-track MAE | lower | +0.0632 [+0.0053, +0.1176] | yes | +0.0786 [+0.0164, +0.1400] | yes |
| LONGITUDINAL | accel MAE | lower | +0.0089 [-0.0462, +0.0580] | no | -0.0481 [-0.0944, -0.0074] | yes |
| LONGITUDINAL | DK-VIOL | lower | +0.0218 [+0.0049, +0.0466] | yes | +0.0252 [+0.0075, +0.0492] | yes |
| LONGITUDINAL | DK-HW | lower | +0.1004 [-0.0688, +0.2672] | no | +0.1291 [-0.0147, +0.2872] | no |
| LONGITUDINAL | DK-TG | lower | +0.0291 [+0.0026, +0.0614] | yes | -0.0075 [-0.0731, +0.0490] | no |
| LATERAL | heading MAE | lower | -0.6998 [-1.0257, -0.3627] | yes | -1.5092 [-2.1124, -0.9946] | yes |
| LATERAL | yaw-rate MAE | lower | -2.6587 [-3.7145, -1.6851] | yes | -5.5332 [-8.4690, -3.2827] | yes |
| LATERAL | curvature MAE | lower | -0.0266 [-0.0391, -0.0159] | yes | -0.0446 [-0.0680, -0.0244] | yes |
| LATERAL | cross-track MAE | lower | -0.0365 [-0.0695, -0.0043] | yes | -0.0369 [-0.0779, -0.0013] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -0.1771 [-0.4213, +0.0200] | no | -0.4300 [-0.7062, -0.2100] | yes |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0116 [-0.0180, -0.0064] | yes | -0.0208 [-0.0372, -0.0076] | yes |
| TACTICAL | TAC-LAT | higher | +0.0129 [+0.0025, +0.0234] | yes | +0.0479 [+0.0300, +0.0696] | yes |
| TACTICAL | TAC-LON | higher | +0.0013 [-0.0213, +0.0229] | no | +0.0004 [-0.0250, +0.0250] | no |
| STRATEGIC | S-POS@30 | lower | -0.3250 [-0.4761, -0.1847] | yes | -0.4814 [-0.7008, -0.2767] | yes |
| STRATEGIC | S-POS@50 | lower | -0.5279 [-0.7945, -0.2763] | yes | -0.7127 [-1.0803, -0.3742] | yes |
| STRATEGIC | S-POS@80 | lower | -0.7830 [-1.2334, -0.3502] | yes | -0.9875 [-1.6084, -0.4074] | yes |
| crawl | M1 crawl share | lower | -0.0114 [-0.0164, -0.0068] | yes | -0.0083 [-0.0126, -0.0039] | yes |

## What this establishes, and what it cannot

* Dose of the a-priori law: M1 crawl share closed RAPR-P01 0.2147 / 0.1391 at p 0.1 against RAPR 0.4447 / 0.3175 at p 0.3; the bank law at p 0.1 (RP01) 0.1473 / 0.1342 (bar 0.5).
* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the training-run question is the C0rep floor row above. T1-kin is not T2: nothing here is a driving claim beyond T1-kin.
* Instrument note: the RAPR results writer compared the composite status to the string `DOMINANT`, which the reader never emits (it writes `DOMINANT, COUNTED` and three other forms). RAPR's composite was NOT DOMINANT at both seeds, so its row was not affected; this writer reads the reader's `DOMINANT` and `counts` booleans, and every row M, I, N, J, K is pinned by a mutation test (`test_gen_raprp01.py`).

## Next (Rule Zero)

* addendum 3 row K decides: the one-offset-per-window recovery recipe is closed at the laws and doses measured (bank law at p 0.3 and 0.1, a-priori law at p 0.3 and 0.1); next arm: a PI-decision item between (a) card row 4's level-from-image auxiliary (architecture change) and (b) a per-tick deviation recipe (a new trainer lever with its own launch-gate entry); neither starts without the PI's decision and box-2 go. Any new box-2 arm needs the PI's go (box 2 runs one job at a time, SAFE MODE).
