# RESULTS: H-LEVELPROBE-1 (zero training)

Card `PREREG_LEVELPROBE.md` sha256 `25c66e938d1ee6deee08bb538446d06224eaa9dff17d9e78aea64f58abe7eef4`; reader `code/levelprobe_read.py`; raw `raw/LEVELPROBE.json`, `raw/STAGE1.json`. Modes read: ['HLT0'].

## 0. The committed outcome

**Card outcome as the reader gives it: NOT DEFINED.** HLVL-PRED not run: Stage 1's features cover the v0 < 6 windows only (F1, PREREG_LEVELPROBE.md section C)
Stage 1 beside it (T0-class, v0 < 6 windows only, `STAGE1.md`): eval RMSE 1.416 m/s, R^2 0.489 (608 windows); on the 313 M1 windows with v0 < 6 RMSE 1.591, R^2 0.117.

## 1. HLVL-T0 (ORACLE, the logged speed at t0 held constant): tier **T1-kin** / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers another draw of EPISODES only; X and Y share the checkpoint) / evidence class **MEASURED**

HLVL-T0 mean share of the M1 excess closed **-0.386** (cells -0.597, -0.377, -0.205, -0.364); M1 reading `NOT CLOSES` (0 of 4 cells separated-better). Context (HISTSWAP-LF, per-tick logged speed, MEASURED in `histswap_lf/raw/HISTSWAP_LF_T1.json`): HLVL 0.599 to 0.633 per cell; HCV -0.559 to -0.301. HLVL-T0 differs from HLVL by holding the t0 speed instead of re-reading the logged speed at each tick (PREREG B7).

| cell | M1 X - Y | Y - P10 (the excess) | share closed | ADE6 X - Y (m) | outcome label |
|---|---|---|---|---|---|
| CTRL_s0 | +0.0289 [+0.0084, +0.0496] SW | +0.0483 | -0.5970 | +0.8045 [+0.6264, +0.9917] SW | HARMS |
| CTRL_s1 | +0.0213 [+0.0002, +0.0420] SW | +0.0565 | -0.3775 | +0.7654 [+0.5613, +0.9623] SW | HARMS |
| SOUP_s0 | +0.0137 [-0.0077, +0.0348] ns | +0.0666 | -0.2051 | +0.7181 [+0.5267, +0.9222] SW | HARMS |
| SOUP_s1 | +0.0210 [-0.0011, +0.0429] ns | +0.0576 | -0.3639 | +0.7372 [+0.5396, +0.9382] SW | HARMS |

## 2. All 19 rows, X - Y per cell for HLVL-T0 (S-POS rows UNVERIFIED; STRATEGIC family PAUSED; HLVL-T0 absolute rows are not performance)

| row | family | CTRL_s0 | CTRL_s1 | SOUP_s0 | SOUP_s1 |
|---|---|---|---|---|---|
| ADE6 | PRIMARY | +0.8045 [+0.6264, +0.9917] SW | +0.7654 [+0.5613, +0.9623] SW | +0.7181 [+0.5267, +0.9222] SW | +0.7372 [+0.5396, +0.9382] SW |
| speed MAE | LONGITUDINAL | +0.1888 [+0.0966, +0.2825] SW | +0.1696 [+0.0771, +0.2626] SW | +0.1405 [+0.0478, +0.2368] SW | +0.1277 [+0.0382, +0.2195] SW |
| along-track MAE | LONGITUDINAL | +0.8451 [+0.6631, +1.0322] SW | +0.7475 [+0.5319, +0.9540] SW | +0.7419 [+0.5405, +0.9484] SW | +0.7497 [+0.5410, +0.9615] SW |
| accel MAE | LONGITUDINAL | -0.2017 [-0.3551, -0.0780] SB | -0.1755 [-0.2830, -0.0802] SB | -0.1818 [-0.3108, -0.0781] SB | -0.3598 [-0.5438, -0.2048] SB |
| DK-VIOL | LONGITUDINAL | +0.0353 [+0.0047, +0.0699] SW | +0.0555 [+0.0238, +0.0909] SW | +0.0471 [+0.0129, +0.0876] SW | +0.0538 [+0.0212, +0.0908] SW |
| DK-HW | LONGITUDINAL | +0.4280 [+0.1040, +0.7658] SW | +0.5036 [+0.1454, +0.8783] SW | +0.4135 [+0.0701, +0.7795] SW | +0.4452 [+0.1306, +0.7548] SW |
| DK-TG | LONGITUDINAL | +0.0598 [-0.0464, +0.1559] ns | +0.1045 [+0.0336, +0.1865] SW | +0.0812 [+0.0019, +0.1634] SW | +0.0757 [+0.0151, +0.1421] SW |
| heading MAE | LATERAL | +0.1643 [-0.4130, +0.8110] ns | +0.2850 [-0.6050, +1.2597] ns | +0.0307 [-0.7515, +0.8646] ns | +0.4767 [-0.4188, +1.4593] ns |
| yaw-rate MAE | LATERAL | -7.9620 [-10.6310, -5.4767] SB | -11.9807 [-16.3350, -8.3921] SB | -10.4885 [-13.7112, -7.5012] SB | -11.1452 [-15.0299, -7.8126] SB |
| curvature MAE | LATERAL | -0.0208 [-0.0456, +0.0040] ns | -0.0540 [-0.0928, -0.0203] SB | -0.0437 [-0.0783, -0.0105] SB | -0.0307 [-0.0685, +0.0058] ns |
| cross-track MAE | LATERAL | +0.0209 [-0.0338, +0.0769] ns | +0.0864 [+0.0225, +0.1561] SW | +0.0318 [-0.0287, +0.0955] ns | +0.0348 [-0.0263, +0.0989] ns |
| L1 lateral-accel MAE | LATERAL | -0.9807 [-1.4189, -0.6255] SB | -1.0743 [-1.5810, -0.6674] SB | -1.0573 [-1.4635, -0.7139] SB | -1.1323 [-1.5933, -0.7447] SB |
| L2 curvature MAE (GT>=5.2) | LATERAL | -0.0195 [-0.0465, +0.0016] ns | -0.0201 [-0.0496, +0.0076] ns | -0.0437 [-0.0900, -0.0059] SB | -0.0169 [-0.0475, +0.0128] ns |
| TAC-LAT | TACTICAL | +0.0058 [-0.0125, +0.0250] ns | -0.0033 [-0.0305, +0.0230] ns | +0.0250 [+0.0091, +0.0421] SB | +0.0058 [-0.0200, +0.0304] ns |
| TAC-LON | TACTICAL | -0.0596 [-0.1031, -0.0154] SW | -0.0509 [-0.1009, +0.0008] ns | -0.0329 [-0.0812, +0.0159] ns | -0.0529 [-0.1028, +0.0000] ns |
| S-POS@30 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +0.9482 [+0.3985, +1.5789] SW | +1.3494 [+0.6714, +2.1343] SW | +0.7399 [+0.1815, +1.3492] SW | +1.3745 [+0.6839, +2.1340] SW |
| S-POS@50 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +1.5859 [+0.6296, +2.6980] SW | +2.2290 [+1.0976, +3.5375] SW | +1.2872 [+0.3070, +2.3507] SW | +2.2728 [+1.1511, +3.5426] SW |
| S-POS@80 | STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED) | +2.2305 [+0.7314, +4.0746] SW | +3.2073 [+1.4417, +5.3500] SW | +1.8280 [+0.3515, +3.4988] SW | +3.2001 [+1.5291, +5.1417] SW |
| M1 crawl share | crawl | +0.0289 [+0.0084, +0.0496] SW | +0.0213 [+0.0002, +0.0420] SW | +0.0137 [-0.0077, +0.0348] ns | +0.0210 [-0.0011, +0.0429] ns |

## 3. Controls (computed before any row; a failure REFUSES the read)

| control | PASS |
|---|---|
| KY_reused_Y_equals_frozen_pins | True |
| KA1_fresh_off_rerun_equals_reused_Y | True |
| KC1_flag_reached_the_run_and_v0_in_is_as_the_card_says | True |
| KC2_the_CV_history_reached_the_network | True |
| KC3_plan_committed_to_the_executed_state | True |
| KF_floor_window_check_vs_fresh_Y | True |
| K3_Y_banked_minus_P10_reproduces_SOUP_READ | True |
| KB1_P10_side_equals_landed_banked_b | True |

HARMS flag (outcome label == HARMS, per base): {'HLT0': {'DAG-OACC': True, 'SOUP': True}}. Outcome labels: {'DAG-OACC': {'outcome': 'HARMS', 'DOMINANT': False, 'HARMS_condition': True, 'MOVED_condition': False, 'moved_and_harms': False}, 'SOUP': {'outcome': 'HARMS', 'DOMINANT': False, 'HARMS_condition': True, 'MOVED_condition': False, 'moved_and_harms': False}}.

## 4. Scope
* HLVL-T0 is an ORACLE (the human's speed at t0); its absolute rows are not performance. HLVL-PRED is not read here (F1; `AMEND_A1_PROPOSED.md`).
* One checkpoint per cell; the 4 cells are the rig's replicates, not a variance estimate (`H-ESTIM-SEED-1`). T1-kin cameras follow the logged path.

## 5. What the HLVL-T0 reading means (MEASURED numbers above; the inferences are labelled)
* **MEASURED: the logged speed AT t0, held constant, does not close the M1 excess** (mean share closed -0.386, 0 of 4 cells separated-better, ADE6 X - Y over the +0.0604 m guard at all 4 cells). HISTSWAP-LF's HLVL (the logged speed re-read at every tick) closed 0.599 to 0.633 at 4 of 4 cells on the same cells and comparator; HCV (the executed speed held) read -0.559 to -0.301. HLVL-T0 reads like HCV, not like HLVL.
* **INFERENCE (not separately tested):** what carried HLVL's closure is not the speed level at t0 but the logged speed AFTER t0 (the human's speed profile over the rollout), which no t0 frame supplies. A level-from-image auxiliary that only reproduces the t0 level has, on this reading, nothing to gain here.
* **HYPOTHESIS, unmeasured (F1 blocks it):** HLVL-PRED is expected not to close either. It is NOT bounded by HLVL-T0: the image can carry scene context that predicts speed beyond v(t0) (a green light, a clearing lead), so a prediction could differ from v(t0) in a helpful direction. Only the amended run can say.
* **A hole in the card (reported, the rule is not changed):** SUPPORTED requires HLVL-PRED's mean closure >= 0.5 x HLVL-T0's mean closure. With HLVL-T0's mean at -0.386 that threshold is -0.193, so the second clause is satisfied by any prediction above it; only the first clause (>= 0.30 at >= 3 of 4 cells) binds. If A1 is approved this clause should be re-stated before it is frozen.
* **Next lever (Rule Zero, proposals for a card, none run):** (i) HLVL-FUT, an oracle holding the logged speed at t0 + 1 s (and t0 + 3 s) constant, to see how much of the profile is needed; (ii) HLVL-MEAN6, the mean logged speed over the 6 s horizon held constant. Both are zero-training, use the same tool path (a per-window constant via `--level-file`), and tell whether the closure is a short-horizon or a profile effect before any architecture decision.

