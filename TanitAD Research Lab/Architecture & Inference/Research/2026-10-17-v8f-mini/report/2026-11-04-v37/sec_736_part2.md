### 7.36 RECOVER-FT and its attribution arms (2026-11-03 and 2026-11-04): a recovery fine-tune moves M1 the right way and fails its guards, neither the rejoin blend nor the dose alone carries the harm, and an a-priori deviation law doubles the closure at a larger position cost

**7.36.4 RAPR: the a-priori deviation law (`H-RECOVER-NEXT-1`, arm RAPR).**

* **Setup [M].** RAPR is R's argv with `--recover-dev-law apriori` and without the two bank flags; every other flag is R_s's (p 0.3, the rejoin target, K = 10). Each perturbed sample's offset is drawn from the law the RECOVER-FT card fixed in its section 3 before any data: dv ~ U[−3.0, 0.0] m/s, da ~ U[−1.0, +0.5] m/s², dy ~ N(0, 0.5 m), dψ ~ N(0, 3°) and a yaw-rate offset ~ N(0, 2°/s), heavier on the slow side where the crawl lives. No deviation table is built and no bank is read (control `KR3_apriori_law_builds_no_table_and_reads_no_bank`). The PI's box-2 go came 2026-11-04 14:30; addendum 2 (`PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md`, sha256 `a292eac5…`, frozen 2026-11-04 14:35:47, before any RAPR checkpoint) committed a next arm for the outcomes the card's section 4 leaves open. The read ran 2026-11-04T16:16:51+0100 with the same reader, comparators, bars and estimator as RTL and RP01; 20 controls, 0 failed, and the KA1 bit-identity rerun of RAPR_s0 passes. The full comparison with the three arms before it:

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

* **Reading [M].** RAPR reads **HARMS** and its composite is **NOT DOMINANT**. Its M1 closure is the largest of the four recovery arms and above R's at both seeds: 0.4447 / 0.3175 of the C0 − P10 excess (bar 0.5; R 0.2198 / 0.1318), with M1 RAPR − C0 −0.0237 [−0.0326, −0.0159] / −0.0189 [−0.0276, −0.0112]. It pays in position and speed: ADE6 RAPR − C0 +0.3504 / +0.3202 m against the +0.0604 m guard (R +0.1892 / +0.0977 m), with speed MAE, along-track MAE, DK-HW and DK-VIOL separated-worse at both seeds and DK-TG at s0, while every LATERAL row improves (heading MAE −1.2936 / −2.0080, yaw-rate MAE −4.6536 / −8.1734, curvature MAE −0.0564 / −0.0723, all separated) and TAC-LAT rises (+0.0263 / +0.0634, separated). STRATEGIC improves (S-POS@80 −1.4443 / −1.2996 m, separated). All 19 rows of RAPR − C0, per family, never pooled:

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

* **The composite RAPRC = RAPR + S (sg72) + kbox vs P10 [M].** D1 holds (ADE6 −0.7950 / −0.7253 m, the smallest D1 margin of the four recovery composites) and D3 holds over the four families, STRATEGIC included. D2 fails on the M1 crawl share at both seeds (+0.0343 [+0.0171, +0.0550] / +0.0425 [+0.0227, +0.0671], the smallest M1 excess of the four composites; C0C +0.0587 / +0.0604) and on DK-VIOL at s0 (+0.0487 [+0.0092, +0.1017]). All 19 rows of RAPRC − P10 (Read A):

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

* **Frame-freeze and the floor [M].** The frame-freeze reading is PARTIAL (bar 0.5), but RAPR's share closed with every history row's image features replaced by the current frame's (`--scene hold_now`) is 0.4826 / 0.3021, against 0.4447 / 0.3175 on the logged frames: with frames that carry no motion RAPR closes about as much as with them, so its closure is not the logged frames' motion. R kept 0.1674 / 0.1568 of its 0.2198 / 0.1318. The training-run floor splits as for every recovery arm: abs(RAPR − C0) on M1 is 0.0237 against the replicate's 0.0362 at s0 (inside) and 0.0189 against 0.0038 at s1 (outside).
* **Committed next [M].** The combiner selects card section 4 row 3 (RAPRC fails D2 on a non-M1 row, DK-VIOL at s0), whose setting is the harmless target or dose; with Q1 = neither no setting is known not to harm, and addendum 2 row E decides: **RAPR-P01**, RAPR's frozen argv with `--recover-p 0.1`, 2 runs. In the addendum's words, *"RP01's dose is the one measured setting that kept ADE6 inside its guard at both seeds."*

**7.36.5 What the four recovery arms establish, and what they cannot.**

* **Establishes [M].** (1) A recovery fine-tune moves M1 the right way at both seeds in every arm (R, RTL, RP01, RAPR), and the deviation law sets the size of the move: the a-priori law closes 0.4447 / 0.3175, about twice R's bank law at the same dose. (2) No arm closes M1 within its guards, and across the four arms the arm with the most closure (RAPR) carries the most position harm, while the one arm without the ADE6 harm (RP01) has the least closure and still a distance-keeping harm. (3) RAPR's closure is kept under frozen frames, so it is not an effect of the logged frames' motion. (4) Every composite keeps D1 and D3 and fails D2 on M1, RAPRC with the smallest excess (+0.0343 / +0.0425).
* **Cannot [M].** One training run per seed per arm (`H-ESTIM-SEED-1`): at s0 every arm's M1 effect sits inside the replicate's spread (RAPR 0.0237 against 0.0362), at s1 every arm's sits outside it. T1-kin is not T2: no number here is a driving claim, and no T2 number exists. The longitudinal harm of RAPR is not attributed between the law's speed offset, its acceleration offset and its lateral offsets.
* **[H] (untested), read with §7.35.4.** The oracle closure needs a per-tick speed consistent with the frames, and every recovery arm offsets the history by ONE constant per window. A law that offsets the speed further toward slow teaches O to plan faster after a slow history, which would close the crawl and cost along-track and speed accuracy on windows with no deviation; the RAPR rows are consistent with that and do not test it.

**7.36.6 Next (Rule Zero).** RAPR-P01 is the committed next arm: it asks whether the a-priori law at the dose that removed RP01's ADE6 harm keeps part of RAPR's closure. Its argv is RAPR's frozen argv with one token changed, and the reader needs an `ARM_SPEC` entry with a known answer before its read (instrument work, no box-2 time); the runs need the PI's box-2 go. The dominance bar is unmet: no counted DOMINANT exists, so no evidence video and no scale signal follow from this section. **[E]** The card, written before any of these runs, gave P(any XC DOMINANT, counted) about 0.12 and RAPR about 0.10.

**7.36.7 Provenance [M].** Under `R/raw/` at the tip `{{TP}}`: `2026-11-03-recover-read-result/RESULTS.md` (sha256 `{{RES_RF}}…`), `2026-11-04-recover-next-rtl/RESULTS.md` (`{{RES_RT}}…`), `2026-11-04-recover-next-rp01/RESULTS.md` (`{{RES_RP}}…`) and `2026-11-04-recover-next-rapr/RESULTS.md` (`{{RES_RA}}…`, with `RECOVER_READ_RAPR.json` and `RECOVER_NEXT_after_RAPR.json`); cards `2026-11-02-recover-ft/PREREG_RECOVER_FT_FROZEN.md` (`57caba3b…`), `2026-11-02-recover-read/PREREG_COMPOSITE_FROZEN.md` (`18ed37bc…`), `2026-11-03-recover-next/PREREG_RECOVER_NEXT_FROZEN.md` (`01046fe6…`), addendum 1 `2026-11-04-recover-next-rtl/PREREG_RECOVER_NEXT_ADDENDUM_RTLRP01.md` (`8c62b4da…`) and addendum 2 `2026-11-04-recover-next-rapr/PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md` (`a292eac5…`); claim rows `H-RECOVER-FT-1`, `H-RECOVER-COMPOSITE-1` and `H-RECOVER-NEXT-1` in `Project Steering/GOALS_AND_CLAIMS.md`. The build is `patch_v37.py` (anchored edits on the landed v36 blobs; every table line asserted to be a line of its RESULTS file), checked by `checks_v37.py` and re-measured by `audit_v37.py`.
