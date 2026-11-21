# RECOVER-READ results: H-RECOVER-FT-1 and H-RECOVER-COMPOSITE-1 at T1-kin

**Stamp.** Evidence class **MEASURED** (ours; artifact `raw/RECOVER_READ.json`, mode `production`, run_clock 2026-11-03T05:25:34+0100). Tier **T1-kin** = self-action open loop on logged cameras (plain causal bucket, nav dead-reckoned). Estimator = **paired episode-cluster bootstrap, B = 2,000, seed 0** (141 eval episodes, card section 6); it answers "another draw of EPISODES" only and does NOT answer training variance. The training-variance check is the **C0rep floor** (`H-ESTIM-SEED-1`); the planner is deterministic (KA1 bit-identity rerun of R_s0: ticks_max_abs_dev 0.0). All 18 controls PASS, `controls_failed = []`, `refused = null`. Cards: `recover_ft/PREREG_RECOVER_FT_FROZEN.md` (+ `AMENDMENT_1.md`) and `PREREG_COMPOSITE_FROZEN.md`. Every number below is pasted from `raw/RESULTS_TABLES.md`, which `code/make_results_tables.py` generates from the JSON; card constants are quoted with their section.

## Verdict
1. **H-RECOVER-FT-1 = HARMS.** GUARDS fail at both seeds (ADE6 R - C0 point above the +0.0604 m bar at s0 and s1, plus separated-worse non-crawl rows); PRIMARY fails (M1 R - C0 is separated-better at both seeds, mean share closed 0.1758 against the 0.5 bar); FLOOR PASS = false (exceeds_floor false at s0, true at s1); frame-freeze reads PARTIAL.
2. **Read A of R vs P10: DOMINANT = false at both seeds.** D1 and D3 hold; D2 fails (separated-worse: L2 curvature MAE, M1 crawl share, curvature MAE).
3. **H-RECOVER-COMPOSITE-1 (RC = R + S + kbox): NOT DOMINANT, counts = false.** D1 and D3 hold at both seeds; D2 fails at M1 crawl share alone at both seeds (C0C, the same composite on C0, fails at the same single row).

## 1. H-RECOVER-FT-1: label `HARMS`
Label as the JSON gives it: `HARMS`. Card section 6 defines HARMS as "G fails (ADE6 R - C0 > +0.0604 m at a seed, or a separated-worse non-crawl row)". Cells are `delta [lo, hi] marker` (markers: see section 5).

| item | s0 | s1 |
|---|---|---|
| M1 R - C0, logged frames | -0.0117 [-0.0185, -0.0050] sep-B | -0.0078 [-0.0143, -0.0019] sep-B |
| M1 separated-better (m1_cell_logged.better) | true | true |
| M1 C0 - P10, the excess (m1_cell_logged.yp_delta) | +0.0532 | +0.0594 |
| M1 C0 - P10 separated-worse (m1_cell_logged.yp_separated_worse) | true | true |
| share closed (primary.share_closed) | 0.2198 | 0.1318 |
| ADE6 R - C0 | +0.1892 [+0.1060, +0.2739] sep-W | +0.0977 [-0.0337, +0.2218] ns |
| ADE6 within the guard (guards.per_seed.ade6_within_guard) | false | false |
| non-crawl separated-worse rows in R - C0 | ADE6, DK-HW, DK-VIOL, TAC-LAT, accel MAE, along-track MAE, cross-track MAE | DK-VIOL, along-track MAE |
| FLOOR abs(R - C0) on M1 (point) | 0.0117 | 0.0078 |
| FLOOR abs(C0rep - C0) on M1 (point) | 0.0362 | 0.0038 |
| FLOOR exceeds_floor | false | true |
| FREEZE M1 FZ-R - FZ-C0 (xy_delta) | -0.0090 | -0.0093 |
| FREEZE M1 FZ-R - FZ-C0, cell with interval | -0.0090 [-0.0162, -0.0021] sep | -0.0093 [-0.0156, -0.0035] sep |
| FREEZE separated-better (m1_cells_freeze.better) | true | true |
| FREEZE M1 FZ-C0 - P10, the excess (yp_delta) | +0.0537 | +0.0593 |
| FREEZE share closed | 0.1674 | 0.1568 |

* PRIMARY: n_seeds_separated_better = 2; mean_share_closed = 0.1758; share_bar = 0.5; PASS = false
* GUARDS: ade_guard_m = 0.0604; PASS = false
* FLOOR: PASS = false (the per-seed rule is |R - C0| > |C0rep - C0| on M1, point)
* FRAME-FREEZE: reading = `PARTIAL`; n_seeds_separated_better = 2; mean_share_closed = 0.1621; robust_bar = 0.5; artefact_bar = 0.25; P_logged_holds = false; applies_to_label = false
* freeze_cells_note (JSON): G_F = M1 share of FZ-R - FZ-C0 against FZ-C0 - P10 (each frame mode against its own excess); G_L = primary.share_closed
* Freeze controls: KFZ4 (FZ-HLVL keeps its closure, read from FZ_CONTROLS.json): share_closed s0 = 0.5690, s1 = 0.6080; mean = 0.5885; share_bar = 0.5; PASS = true

## 2. Read A of R vs P10 (reported beside the label, not part of it: card section 6)
Card constants: D1 = ADE6 separated-better with point <= -0.2415 m; D2 = no row of the 19 separated-worse; D3 = every family has a separated-better row.

| item | s0 | s1 |
|---|---|---|
| ADE6 R - P10 | -0.9254 [-1.1261, -0.7276] sep-B | -0.9647 [-1.1969, -0.7286] sep-B |
| D1 | true | true |
| D2 | false | false |
| D3 (four families) | true | true |
| D3 by family | LONGITUDINAL true, LATERAL true, TACTICAL true, STRATEGIC true | LONGITUDINAL true, LATERAL true, TACTICAL true, STRATEGIC true |
| D3 (three families, STRATEGIC set aside) | true | true |
| DOMINANT (four-family D3) | false | false |
| DOMINANT (three-family D3) | false | false |
| separated-worse rows | L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE | L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE |
| STRATEGIC rows separated-better | S-POS@30, S-POS@50, S-POS@80 | S-POS@30, S-POS@50, S-POS@80 |
| failing_terms | D2 (separated-worse: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE) | D2 (separated-worse: L2 curvature MAE (GT>=5.2), M1 crawl share, curvature MAE) |

* DOMINANT_both_seeds = false; DOMINANT_three_families_strategic_paused_both_seeds = false; reported_beside_the_label_not_part_of_it = true

## 3. H-RECOVER-COMPOSITE-1 (RC = R + S (sg72) + kbox vs P10; zero training)
* status = `NOT DOMINANT`; counts = false; DOMINANT = false; DOMINANT (three-family) = false; R_freeze_reading = `PARTIAL`; attribution = `R-SPECIFIC`; C0C_dominant_seeds = 0
* freeze_reading_note (JSON): PARTIAL by default: R's own PRIMARY did not hold, so ARTEFACT is not defined by the RECOVER-FT card; no frame-freeze arm of RC exists
* The card counts a DOMINANT only if R's freeze reading is ROBUST or PARTIAL (composite card section 3); here the status is NOT DOMINANT, so `counts = false` and the attribution tag is reported beside it, not as part of it.

| item | s0 | s1 |
|---|---|---|
| RC vs P10: ADE6 | -0.9566 [-1.1678, -0.7514] sep | -0.9344 [-1.1692, -0.6864] sep |
| RC vs P10: M1 crawl share | +0.0482 [+0.0282, +0.0715] sep-W | +0.0571 [+0.0340, +0.0853] sep-W |
| RC vs P10: D1 | true | true |
| RC vs P10: D2 | false | false |
| RC vs P10: D3 (four families) | true | true |
| RC vs P10: D3 (three families) | not in the read JSON | not in the read JSON |
| RC vs P10: DOMINANT (four-family D3) | false | false |
| RC vs P10: DOMINANT (three-family D3) | false | false |
| RC vs P10: separated-worse rows | M1 crawl share | M1 crawl share |
| C0C vs P10: ADE6 | -1.1204 [-1.3191, -0.9330] sep-B | -1.0086 [-1.2061, -0.8149] sep-B |
| C0C vs P10: M1 crawl share | +0.0587 [+0.0401, +0.0812] sep-W | +0.0604 [+0.0383, +0.0868] sep-W |
| C0C vs P10: D1 | true | true |
| C0C vs P10: D2 | false | false |
| C0C vs P10: D3 (four families) | true | true |
| C0C vs P10: D3 (three families) | true | true |
| C0C vs P10: DOMINANT (four-family D3) | false | false |
| C0C vs P10: DOMINANT (three-family D3) | false | false |
| C0C vs P10: separated-worse rows | M1 crawl share | M1 crawl share |

`not in the read JSON` for RC means the JSON carries RC's `DOMINANT` three-family flag but not the three-family D3 flag itself; D3 by family for RC is in `raw/RESULTS_TABLES.md` A3.

**kbox's own effect on the recovery base (RC - R)**, the JSON's six rows (all 19 are in `raw/RESULTS_TABLES.md` part C):

| row | s0 | s1 |
|---|---|---|
| ADE6 | -0.0312 [-0.0697, +0.0095] ns | +0.0303 [-0.0041, +0.0702] ns |
| M1 crawl share | +0.0080 [+0.0032, +0.0142] sep | +0.0062 [+0.0019, +0.0121] sep |
| heading MAE | +0.3736 [-0.6477, +1.5793] ns | +0.5085 [-0.4311, +1.6263] ns |
| curvature MAE | -0.0932 [-0.1309, -0.0613] sep | -0.1207 [-0.1719, -0.0749] sep |
| yaw-rate MAE | -12.7951 [-17.9209, -8.5794] sep | -13.0791 [-18.4481, -8.4111] sep |
| S-POS@30 | +0.1777 [-0.1402, +0.5765] ns | +0.1814 [-0.0853, +0.4884] ns |

## 4. Replicate floor (C0rep vs C0: same recipe and init, training seed 10+s)
The two training draws of the control, straight from the JSON, with R - C0 on the same axis:

| item | s0 | s1 |
|---|---|---|
| ADE6: C0rep - C0 | +0.3061 [+0.1937, +0.4233] sep | +0.0066 [-0.0630, +0.0787] ns |
| M1 crawl share: C0rep - C0 | +0.0362 [+0.0240, +0.0498] sep | +0.0038 [-0.0039, +0.0112] ns |
| ADE6: R - C0 (same contrast axis, for scale) | +0.1892 [+0.1060, +0.2739] sep-W | +0.0977 [-0.0337, +0.2218] ns |
| M1 crawl share: R - C0 (same contrast axis, for scale) | -0.0117 [-0.0185, -0.0050] sep-B | -0.0078 [-0.0143, -0.0019] sep-B |

* FLOOR per seed (M1, point): s0 |R - C0| 0.0117 against |C0rep - C0| 0.0362 (exceeds_floor false); s1 |R - C0| 0.0078 against |C0rep - C0| 0.0038 (exceeds_floor true).
* A separated CI between two arms is necessary, not sufficient (`H-ESTIM-SEED-1`): the bootstrap resamples episodes with the models held fixed. The block C0rep - C0 is the card's same-init replicate floor (section 2 of RECOVER-FT: `--seed 10+s`, identical init); its 19 cells are in part C of `raw/RESULTS_TABLES.md`. In the JSON the replicate is separated from C0 on both ADE6 and M1 at s0, and on neither at s1.

## 5. Four-family tables (ADE6 on top; point and marker, both seeds; intervals in `raw/RESULTS_TABLES.md` parts C and D)
* Families: the JSON does not assign rows to families (it carries only the four family names as `D3_by_family` keys); the grouping is the brief's and equals the pinned `smooth_t1_read.py` FAMILY map except that `M1 crawl share` is shown under LONGITUDINAL (the pinned map tags it `crawl`, so the reader's own D3 does not count M1 toward LONGITUDINAL).
* Markers: `ns` = JSON `separated` false; `sep-B` / `sep-W` = separated and listed by the JSON as separated-better / separated-worse for that block; `sep` = separated, direction not listed by the JSON for that block. The JSON lists directions for R - P10 and C0C - P10 (both lists), RC - P10 (worse list only), R - C0 (crawl rows better, non-crawl rows worse); every other block shows `sep` / `ns` only.
* STRATEGIC (the JSON's `strategic_family` note, verbatim): PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it D3 above is the four-family verdict; the three-family D3 is reported beside it. The S-POS rows are in every table.

### 5a. R - C0, C0rep - C0, R - P10
| row | R - C0 s0 | R - C0 s1 | C0rep - C0 s0 | C0rep - C0 s1 | R - P10 s0 | R - P10 s1 |
|---|---|---|---|---|---|---|
| **PRIMARY (ADE6 on top)** | | | | | | |
| ADE6 | +0.1892 sep-W | +0.0977 ns | +0.3061 sep | +0.0066 ns | -0.9254 sep-B | -0.9647 sep-B |
| **LONGITUDINAL** | | | | | | |
| speed MAE | +0.0203 ns | -0.0121 ns | +0.1733 sep | +0.0026 ns | -0.7154 sep-B | -0.7699 sep-B |
| along-track MAE | +0.1360 sep-W | +0.1498 sep-W | +0.3540 sep | +0.0293 ns | -0.5157 sep-B | -0.4343 sep-B |
| accel MAE | +0.1169 sep-W | +0.0237 ns | +0.1576 sep | +0.0261 ns | -0.6847 sep-B | -0.9199 sep-B |
| DK-VIOL | +0.0202 sep-W | +0.0235 sep-W | -0.0034 ns | +0.0034 ns | +0.0101 ns | +0.0050 ns |
| DK-HW | +0.2087 sep-W | +0.1916 ns | +0.1321 ns | -0.0155 ns | -0.1426 ns | -0.3175 ns |
| DK-TG | +0.0135 ns | -0.0167 ns | +0.0493 sep | -0.0289 ns | -0.1125 sep-B | -0.0978 sep-B |
| M1 crawl share | -0.0117 sep-B | -0.0078 sep-B | +0.0362 sep | +0.0038 ns | +0.0411 sep-W | +0.0516 sep-W |
| **LATERAL** | | | | | | |
| heading MAE | -0.5185 sep | -1.0389 sep | +1.5217 sep | -0.5237 sep | -2.3895 sep-B | -2.8885 sep-B |
| yaw-rate MAE | -2.2547 sep-B | -2.8226 sep-B | +9.1575 sep | +0.5219 ns | +0.2943 ns | +0.0790 ns |
| curvature MAE | -0.0263 sep-B | -0.0149 ns | +0.1043 sep | +0.0055 ns | +0.0568 sep-W | +0.0835 sep-W |
| cross-track MAE | +0.0593 sep-W | -0.0769 ns | -0.0226 ns | -0.0218 ns | -0.8005 sep-B | -0.9420 sep-B |
| L1 lateral-accel MAE | -0.0018 ns | -0.2182 ns | +0.2599 ns | +0.0697 ns | -2.2422 sep-B | -2.7661 sep-B |
| L2 curvature MAE (GT>=5.2) | -0.0081 ns | -0.0043 ns | +0.0784 sep | +0.0037 ns | +0.0307 sep-W | +0.0373 sep-W |
| **TACTICAL** | | | | | | |
| TAC-LAT | -0.0484 sep-W | +0.0654 sep | -0.0438 sep | +0.0288 sep | +0.1371 sep-B | +0.2026 sep-B |
| TAC-LON | -0.0133 ns | +0.0208 ns | -0.0592 sep | -0.0050 ns | +0.1313 sep-B | +0.1617 sep-B |
| **STRATEGIC** | | | | | | |
| S-POS@30 | -0.4072 sep | -0.8136 sep | +0.4786 sep | -0.1081 ns | -1.1648 sep-B | -1.4827 sep-B |
| S-POS@50 | -0.7248 sep | -1.2966 sep | +0.8049 sep | -0.1236 ns | -4.0454 sep-B | -4.3337 sep-B |
| S-POS@80 | -1.2410 sep | -1.8015 sep | +1.0246 sep | -0.3049 ns | -9.6011 sep-B | -9.9107 sep-B |

### 5b. RC - P10, C0C - P10, RC - R
| row | RC - P10 s0 | RC - P10 s1 | C0C - P10 s0 | C0C - P10 s1 | RC - R s0 | RC - R s1 |
|---|---|---|---|---|---|---|
| **PRIMARY (ADE6 on top)** | | | | | | |
| ADE6 | -0.9566 sep | -0.9344 sep | -1.1204 sep-B | -1.0086 sep-B | -0.0312 ns | +0.0303 ns |
| **LONGITUDINAL** | | | | | | |
| speed MAE | -0.8380 sep | -0.8180 sep | -0.8485 sep-B | -0.8069 sep-B | -0.1226 sep | -0.0481 sep |
| along-track MAE | -0.4922 sep | -0.3851 sep | -0.6289 sep-B | -0.5094 sep-B | +0.0236 ns | +0.0491 sep |
| accel MAE | -1.1129 sep | -1.1653 sep | -1.1289 sep-B | -1.1432 sep-B | -0.4282 sep | -0.2454 sep |
| DK-VIOL | +0.0218 ns | +0.0101 ns | -0.0034 ns | -0.0134 ns | +0.0118 ns | +0.0050 ns |
| DK-HW | -0.2666 ns | -0.3767 sep | -0.4783 sep-B | -0.5366 sep-B | -0.0625 ns | -0.0113 ns |
| DK-TG | -0.1091 sep | -0.1129 sep | -0.1139 sep-B | -0.1333 sep-B | +0.0038 ns | -0.0131 ns |
| M1 crawl share | +0.0482 sep-W | +0.0571 sep-W | +0.0587 sep-W | +0.0604 sep-W | +0.0080 sep | +0.0062 sep |
| **LATERAL** | | | | | | |
| heading MAE | -2.0159 sep | -2.3800 sep | -1.6135 sep-B | -1.1123 sep-B | +0.3736 ns | +0.5085 ns |
| yaw-rate MAE | -12.4450 sep | -13.0001 sep | -12.6952 sep-B | -12.3358 sep-B | -12.7951 sep | -13.0791 sep |
| curvature MAE | -0.0355 sep | -0.0372 sep | -0.0347 sep-B | -0.0320 sep-B | -0.0932 sep | -0.1207 sep |
| cross-track MAE | -0.8757 sep | -0.9637 sep | -0.9014 sep-B | -0.8804 sep-B | -0.0753 sep | -0.0217 ns |
| L1 lateral-accel MAE | -3.3932 sep | -3.5136 sep | -3.5456 sep-B | -3.5423 sep-B | -1.1883 sep | -0.7475 sep |
| L2 curvature MAE (GT>=5.2) | -0.0009 ns | -0.0004 ns | +0.0003 ns | +0.0009 ns | -0.0303 sep | -0.0372 sep |
| **TACTICAL** | | | | | | |
| TAC-LAT | +0.1534 sep | +0.2113 sep | +0.1859 sep-B | +0.1455 sep-B | +0.0163 sep | +0.0088 ns |
| TAC-LON | +0.1388 sep | +0.1672 sep | +0.1517 sep-B | +0.1459 sep-B | +0.0075 ns | +0.0054 ns |
| **STRATEGIC** | | | | | | |
| S-POS@30 | -0.7867 sep | -1.0680 sep | -0.7577 sep-B | -0.8399 sep-B | +0.1777 ns | +0.1814 ns |
| S-POS@50 | -3.0969 sep | -3.5536 sep | -3.0093 sep-B | -3.2929 sep-B | +0.3002 ns | +0.2614 ns |
| S-POS@80 | -8.2885 sep | -9.1106 sep | -7.8724 sep-B | -8.9137 sep-B | +0.5254 ns | +0.2995 ns |

## 6. Disclosures
* **READER_FIX_RREAD1** (`READER_FIX_RREAD1.md`, before any R rollout): the first chain launch stopped at pre-flight `PRE_R_s0_kbox` (rc 7, `V8FMiniConfig.__init__() got an unexpected keyword argument 'launch_intent_w'`); the dev-box eval tree predated a config field in the box-2 training tree. Only `stack/tanitad/models/v8f_mini.py` in the eval tree was swapped for the `ad0bcb72` blob (original kept as `v8f_mini.py.pre_launch_intent`); the diff is 45 lines, all additions, none on the forward path, and all six RECOVER-FT checkpoints carry `launch_intent_w = 0.0`. Bit-identity control (`raw/SWAPCTL.txt`): the 35 data arrays (8,625 elements) of DAG-OACC s0, 6 windows, are byte-equal before and after. The comparator as written printed FAIL because it also compared the `tag` array, which holds each run's own name (`BEFORE` / `AFTER`); that is the comparator's self-name array, disclosed in the control file.
* **READER_FIX_RREAD2** (`READER_FIX_RREAD2.md`, rollouts only; no read had run): relaunch 2 finished 9 of 15 rollouts and lost 6 (5 with `OSError: [Errno 28] No space left on device`, 1 with a numpy `_ArrayMemoryError`) because the shared F: scratch drive was full (0 MB free of 1,907,727 MB, measured 05:03; outside this package). The write root moved to `C:/Users/Admin/rr_scratch` via the optional `RREAD_W` (`out/` copied with its 9 done markers; the 6 lost rollouts re-ran from scratch). Chain sha256 c6a9a699... -> 8c5f444d... (recorded in the note).
* **Neither fix changed the tool, the reader, the tests or the cards.** The chain's gate 0 re-checked the card hashes, the tool sha `0721fc01...` and the pinned reader/tests, and ran the unit tests: 27 passed in 171.07 s (`logs/chain.out`, `ZZRREAD-GATE0`). The KA1 bit-identity rerun and every done marker's checkpoint sha are re-checked by the reader (controls above).
* No frame-freeze arm of RC exists; RC's freeze reading is PARTIAL by default (JSON note in section 3).
* Nothing in this package is committed; the Master Mind lands. T1-kin is not T2: nothing here is a driving claim.

## 7. Committed next arm, quoted from the cards (no recommendation beyond them)
**For the label HARMS, RECOVER-FT card section 8:** "the dose arm `--recover-p 0.1` and the attribution arm `--recover-target logged` (same law, no rejoin blend): which of dose and blend harmed." Same section: "Cost and blockers per arm: each is 2 box-2 runs per seed (about 1.3 h at both seeds) and needs a PI box-2 go; none is started by this card."

**For the composite outcome NOT DOMINANT, composite card section 4:** "the JSON names the failing terms per seed. D2 failing at M1 alone (the FEAS2 signature): the residual is R's own M1 shortfall, so the next arm is RECOVER-FT section 8's arm for R's label (R-APR for NOT MOVED, dose / logged-target for HARMS, R-rep for INSIDE-FLOOR), then this composite on that arm." The JSON's failing term for RC vs P10 at both seeds is `D2 (separated-worse: M1 crawl share)` (section 3), which is the M1-alone clause; with R's label HARMS the arm is the one quoted above, then the composite on it.

## 8. Files
`raw/RECOVER_READ.json` (the only source of numbers), `raw/RESULTS_TABLES.md` (generated), `code/make_results_tables.py`, `MANIFEST.txt` (path | bytes | sha256, refreshed after this document). Nothing here trains, queues or touches box 2.
