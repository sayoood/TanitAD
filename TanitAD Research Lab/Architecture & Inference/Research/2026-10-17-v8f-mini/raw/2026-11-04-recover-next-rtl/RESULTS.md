# RECOVER-NEXT, arm RTL (`--recover-target logged`): read result, 2026-11-04

Card `PREREG_RECOVER_NEXT_FROZEN.md` (sha256 `01046fe6…`) section 3, the RECOVER-FT section 6 / 7 rules with R := RTL; addendum `PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md` (frozen 2026-11-04 01:02, before any RTL rollout). Tier **T1-kin** (self-action open loop on logged cameras; not closed loop). Estimator: paired episode-cluster bootstrap, B=2000, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only. Reader sha256 `aeee6b1b1351`, mode `production`, run clock 2026-11-04T01:31:51+0100. Evidence class MEASURED (ours): `RECOVER_READ_RTL.json` in this folder.

## Verdict

* **Label: HARMS** (H-RECOVER-NEXT-1, arm RTL). Controls: 20 run, 0 failed; KA1 bit-identity rerun of RTL_s0: PASS=True.
* **Composite RTLC = RTL + `--smooth sg72 --feasible kbox` vs P10: NOT DOMINANT** at both seeds; failing terms s0: ['D2 (separated-worse: M1 crawl share)'], s1: ['D2 (separated-worse: M1 crawl share)']. D1 = True / True, D3 (four families, STRATEGIC included) = True / True.
* Frame-freeze reading **PARTIAL** (share closed under hold_now s0 0.1860, s1 0.1899; robust bar 0.5).
* Strategic family: reported, D3_by_family STRATEGIC = True / True (separated-better rows ['S-POS@30', 'S-POS@50', 'S-POS@80'] on RTL vs P10, s0). The reader file also carries the note: "PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict…"; for v8F STRATEGIC is reported and counted in D3.

## RTL against R (the RECOVER-FT arm it changes one flag of)

| quantity | bar | R s0 | R s1 | RTL s0 | RTL s1 |
|---|---|---|---|---|---|
| M1 crawl share closed vs C0 | >= 0.5 | 0.2198 | 0.1318 | 0.2131 | 0.1933 |
| ADE6 X − C0 (m) | <= +0.0604 | +0.1892 | +0.0977 | +0.1492 | +0.0925 |
| non-crawl rows separated-worse vs C0 | none | ADE6, DK-HW, DK-VIOL, TAC-LAT, accel MAE, along-track MAE, cross-track MAE | DK-VIOL, along-track MAE | ADE6, along-track MAE | DK-HW, DK-VIOL, along-track MAE |
| composite M1 crawl share XC − P10 | not separated-worse | +0.0482 [+0.0282, +0.0715] | +0.0571 [+0.0340, +0.0853] | +0.0484 [+0.0282, +0.0715] | +0.0540 [+0.0318, +0.0805] |
| kbox's own M1 effect XC − X | not separated-worse | +0.0080 [+0.0032, +0.0142] | +0.0062 [+0.0019, +0.0121] | +0.0066 [+0.0027, +0.0112] | +0.0070 [+0.0025, +0.0127] |
| floor: abs(X − C0) M1 vs abs(C0rep − C0) | exceeds | 0.0117 vs 0.0362 | 0.0078 vs 0.0038 | 0.0113 vs 0.0362 | 0.0115 vs 0.0038 |

**Composite RTLC − P10, all 19 rows (Read A; D2 = no row separated-worse)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | -0.9766 [-1.1894, -0.7718] | yes | -0.9353 [-1.1699, -0.6955] | yes |
| LONGITUDINAL | speed MAE | lower | -0.8298 [-1.0715, -0.6230] | yes | -0.8168 [-1.0554, -0.6064] | yes |
| LONGITUDINAL | along-track MAE | lower | -0.4803 [-0.6760, -0.2896] | yes | -0.3744 [-0.5720, -0.1779] | yes |
| LONGITUDINAL | accel MAE | lower | -1.1329 [-1.5759, -0.7579] | yes | -1.1964 [-1.6553, -0.8071] | yes |
| LONGITUDINAL | DK-VIOL | lower | +0.0151 [-0.0099, +0.0401] | no | +0.0151 [-0.0094, +0.0413] | no |
| LONGITUDINAL | DK-HW | lower | -0.2999 [-0.7179, +0.0427] | no | -0.1719 [-0.5697, +0.1550] | no |
| LONGITUDINAL | DK-TG | lower | -0.0786 [-0.1483, -0.0224] | yes | -0.0466 [-0.1055, +0.0036] | no |
| LATERAL | heading MAE | lower | -2.1146 [-3.0688, -1.0241] | yes | -1.6908 [-2.8115, -0.3911] | yes |
| LATERAL | yaw-rate MAE | lower | -12.5446 [-17.7544, -8.1355] | yes | -12.8704 [-18.2312, -8.3300] | yes |
| LATERAL | curvature MAE | lower | -0.0343 [-0.0498, -0.0205] | yes | -0.0375 [-0.0543, -0.0230] | yes |
| LATERAL | cross-track MAE | lower | -0.8952 [-1.0376, -0.7606] | yes | -0.9711 [-1.1417, -0.8156] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -3.5045 [-5.0653, -2.1626] | yes | -3.5212 [-5.0936, -2.1824] | yes |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0009 [-0.0042, +0.0025] | no | -0.0010 [-0.0043, +0.0026] | no |
| TACTICAL | TAC-LAT | higher | +0.1917 [+0.1533, +0.2318] | yes | +0.2126 [+0.1701, +0.2544] | yes |
| TACTICAL | TAC-LON | higher | +0.1484 [+0.0851, +0.2104] | yes | +0.1663 [+0.1014, +0.2316] | yes |
| STRATEGIC | S-POS@30 | lower | -0.6751 [-0.9631, -0.4143] | yes | -1.0527 [-1.3572, -0.7784] | yes |
| STRATEGIC | S-POS@50 | lower | -2.8344 [-3.6618, -2.0843] | yes | -3.4905 [-4.4027, -2.6725] | yes |
| STRATEGIC | S-POS@80 | lower | -7.7092 [-9.4899, -5.9894] | yes | -8.7958 [-10.8197, -6.8822] | yes |
| crawl | M1 crawl share | lower | +0.0484 [+0.0282, +0.0715] | yes | +0.0540 [+0.0318, +0.0805] | yes |

**RTL − C0, all 19 rows (the label's guards read the non-crawl rows)** (T1-kin, paired episode-cluster bootstrap B = 2000, seed 0; `sep` = 95 % interval excludes 0)

| family | row | better | s0 delta [lo, hi] | s0 sep | s1 delta [lo, hi] | s1 sep |
|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | lower | +0.1492 [+0.0710, +0.2294] | yes | +0.0925 [-0.0443, +0.2152] | no |
| LONGITUDINAL | speed MAE | lower | +0.0009 [-0.0406, +0.0422] | no | +0.0077 [-0.0423, +0.0583] | no |
| LONGITUDINAL | along-track MAE | lower | +0.1446 [+0.0596, +0.2356] | yes | +0.1538 [+0.0462, +0.2677] | yes |
| LONGITUDINAL | accel MAE | lower | +0.0200 [-0.0503, +0.0940] | no | +0.0513 [-0.0319, +0.1473] | no |
| LONGITUDINAL | DK-VIOL | lower | +0.0168 [+0.0000, +0.0394] | no | +0.0269 [+0.0064, +0.0540] | yes |
| LONGITUDINAL | DK-HW | lower | +0.1535 [-0.0467, +0.3624] | no | +0.2964 [+0.0421, +0.5806] | yes |
| LONGITUDINAL | DK-TG | lower | +0.0298 [-0.0066, +0.0740] | no | +0.0072 [-0.0627, +0.0695] | no |
| LATERAL | heading MAE | lower | -0.4845 [-0.9380, -0.0292] | yes | -1.5211 [-2.1023, -1.0076] | yes |
| LATERAL | yaw-rate MAE | lower | -3.1940 [-4.6818, -1.8099] | yes | -4.5291 [-6.8986, -2.4961] | yes |
| LATERAL | curvature MAE | lower | -0.0304 [-0.0448, -0.0167] | yes | -0.0453 [-0.0679, -0.0269] | yes |
| LATERAL | cross-track MAE | lower | +0.0171 [-0.0308, +0.0671] | no | -0.0902 [-0.1923, -0.0078] | yes |
| LATERAL | L1 lateral-accel MAE | lower | -0.2670 [-0.5428, -0.0361] | yes | -0.1354 [-0.4539, +0.2006] | no |
| LATERAL | L2 curvature MAE (GT>=5.2) | lower | -0.0140 [-0.0272, -0.0028] | yes | -0.0204 [-0.0346, -0.0088] | yes |
| TACTICAL | TAC-LAT | higher | -0.0088 [-0.0321, +0.0125] | no | +0.0813 [+0.0555, +0.1113] | yes |
| TACTICAL | TAC-LON | higher | -0.0021 [-0.0275, +0.0217] | no | +0.0175 [-0.0088, +0.0454] | no |
| STRATEGIC | S-POS@30 | lower | -0.3760 [-0.6379, -0.1328] | yes | -0.8400 [-1.1710, -0.5185] | yes |
| STRATEGIC | S-POS@50 | lower | -0.6535 [-1.1401, -0.2232] | yes | -1.3238 [-1.8681, -0.7995] | yes |
| STRATEGIC | S-POS@80 | lower | -1.0417 [-1.8704, -0.3064] | yes | -1.8762 [-2.7631, -1.0392] | yes |
| crawl | M1 crawl share | lower | -0.0113 [-0.0173, -0.0057] | yes | -0.0115 [-0.0177, -0.0062] | yes |

## What this establishes, and what it cannot

* Removing the rejoin blend (target logged) does not remove RECOVER-FT's harm: ADE6 RTL − C0 stays above the +0.0604 m guard at both seeds. Under the card's Q1 rule (blend harmed iff RTL's guards PASS at both seeds) the blend is not the whole harm; Q1 itself needs RP01's read (box 2, running).
* The composite's only failing term is the M1 crawl share, at the same size as R's composite; every other row of RTLC − P10 is not separated-worse, and ADE6 is separated-better by about 0.9 m at both seeds.
* kbox itself raises the M1 crawl share (XC − X separated-worse at both seeds, for R and for RTL).
* One training run per seed (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only; the training-run question is the C0rep floor, and at s0 abs(RTL − C0) on M1 sits inside it.

## Next (Rule Zero; addendum rows, frozen before this read)

* RTL's guards fail, so Q1 cannot be `blend` or `both`: addendum row C (RAPR-TL) is unreachable. If RP01C is not DOMINANT (counted), row D decides: **RAPR as frozen** (argv already shipped on box 2), which needs the PI's box-2 go. If RP01C is DOMINANT and counted, row A: evidence videos and the PI scale signal.
* The larger lever by measured effect is the speed level (§7.35: an oracle swap of the logged history closes 0.918 to 0.984 of the M1 excess, the level alone 0.599 to 0.633), against 0.13 to 0.22 for the RECOVER family; the card's section 4 names the level-from-image auxiliary as the PI-decision item for it.
