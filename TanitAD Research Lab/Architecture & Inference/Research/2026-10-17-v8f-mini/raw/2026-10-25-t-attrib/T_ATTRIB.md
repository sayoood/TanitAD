# T_ATTRIB: where v8F-mini's tactical-layer (T) constraint error lives, and what each lever could close (2026-10-25)

**EXPLORATORY. This decides nothing; it ranks the next pre-registration.**
* **Evidence class:** MEASURED unless a line says otherwise (ESTIMATED / HYPOTHESIS are marked in place).
* **Data:** eval-141 = 2,399 windows over 141 episodes. 2,365 windows (139 episodes) have a v9 label row and carry 26,056 present constraint cells; 34 windows (two episodes, 15 and 30) have no row and are excluded from every T-error number.
* **Arms:** M2b s0 and s1 (primary); GRID 8x20, HIST 10, PGT, WC, DF50 (all seed 0) for the cross-arm reads.
* **Tier:** the T-vs-GT error is a diagnostic of T's own code (no manipulation of O). The ADE6 gap numbers are **T0** (O is fed the GT code by the banked `-tacorc*` evals). Own-code ADE6 is the T1-class open-loop eval.
* **Estimator:** every share and every arm delta carries an **episode-cluster bootstrap, B = 2,000, over the 141 episodes** (ratio of sums for shares, paired for arm deltas), implemented directly in the scripts (`taniteval/ci.py` was not imported; the resampling is the same). It answers *another draw of EPISODES* only (`H-ESTIM-SEED-1`). Single-seed lever arms: a separated cell is necessary, not sufficient.
* **Units, verified from source and by reproduction:**
  * SD units of the standardiser (`cons_stats`, `v8f_mini.py:773-781`), the units of T's head `cons_std` (`:573`) and of the Huber train loss (`:675-683`).
  * mu/sd were recovered from the banked GT-oracle code and reproduce it to 3.4e-7.
  * T's block from the base eval (`cons_pred`, raw) standardised equals the `-tacabs` `tac_code` to 6e-7.
  * Every arm's own GT block equals mine on all 2,365 windows (`s1b_gt_per_arm_check.out`).
  * Reproduced banked numbers: M2b_s0 slot-pooled RMSE latcons 0.6861 / lonspeed 0.4515 / stoplead 0.8155 and present-slot RMSE 0.5363 (WC and DSCALE results: 0.686 / 0.452 / 0.815 / 0.536).
* **R63 applies:** only PRESENT cells carry an error. T's absent-slot values are O's input, not error (shown for completeness in `s2b_episodes_absent.out`).

## HEADLINE

1. **The dominant stratum of T's constraint error is the future-timing cells, in a few heavy-tailed windows.**
   * TIME-TO-EVENT cells (turn/lane-change/reach/start times) plus the lead TIME-GAP cells hold **43 % of the squared error on 20 % of the cells** (M2b_s0: 0.266 [0.216, 0.322] + 0.166 [0.061, 0.274]).
   * The worst **10 % of windows carry 55 %** of the error (holding 15 % of the cells); the worst **14 of 141 episodes carry 44 %**. Seed 1 reads identically (0.551 and 0.444).
   * 1.5 % of the cells (|error| >= 2 SD) carry 40 % of the squared error.
2. **That error is a property of the windows, not of training.**
   * 88 % of the squared error is common to the models (93-97 % for the time cells).
   * No lever moved it: 4 of 5 lever arms made TIME-TO-EVENT worse, separated (+4.8 to +9.9 %).
   * The 7-model ensemble cuts the whole block's RMSE by only 5.4 %.
3. **But this stratum is not where the ADE gap lives, and no single T-error stratum can clear the bar.**
   * The T-to-GT ADE6 gap (1.20 / 1.05 m at s0 / s1) is diffuse. The best half of the windows by T-error (6 % of the error) hold **35 % of the gap**; the worst decile holds only 15-20 %.
   * Removing ALL time-cell error is worth an ESTIMATED **0.17 / 0.07 m** (s0 / s1), below 2F = 0.2415 m.
   * Clearing 2F by accuracy alone needs a **33 % (s0) to 55 % (s1) uniform cut of T's constraint RMSE**. Every measured lever delivered 0 to 6 %.
4. **Top-ranked next step** (ranking and arithmetic in section 4):
   * First the per-kind GT oracle (box 2), because the two ADE-weight estimators disagree about the speed cells by 10x.
   * Then the one stratum with a measured proof of existence: **turn cells in route-silent turn windows**. T is at chance there (RMSE 0.95 SD against a constant predictor's 1.0), and the same model reaches 0.40 SD when the route speaks. That is 19 % of the total error.
   * The remaining levers (LoRA, 2x data, finer pooling) have ceilings of 0.00 to 0.10 m (D), and none can clear 2F alone.

## 1. Where T's constraint error lives (Q1)

Full tables: `s2_where.out` (groups, cells, 7 strata, concentration), `s3_tail.out` (kinds, |GT| bins, the worst windows), `s2b_episodes_absent.out` (episodes by name).

**By group and by cell kind** (M2b_s0; share of total squared error, episode-bootstrap CI):

| group | n cells | RMSE | share | | kind | cells | RMSE | share |
|---|---|---|---|---|---|---|---|---|
| latcons | 6,715 | 0.686 | 0.344 [0.240, 0.455] | | TIME-TO-EVENT | 11.8 % | 0.889 | **0.266** [0.216, 0.322] |
| lonspeed | 14,839 | 0.452 | 0.330 [0.267, 0.409] | | TIME-GAP to lead | 8.1 % | 0.849 | **0.166** [0.061, 0.274] |
| stoplead | 4,502 | 0.815 | 0.326 [0.201, 0.451] | | GEOMETRY (m) | 14.3 % | 0.664 | 0.179 [0.133, 0.225] |
| | | | | | ANGLE (deg) | 12.7 % | 0.635 | 0.145 [0.092, 0.212] |
| | | | | | LEAD GAP (m) | 8.1 % | 0.739 | 0.126 [0.091, 0.170] |
| | | | | | SPEED (m/s) | 44.9 % | 0.304 | 0.118 [0.094, 0.152] |

* Seed 1 shares: latcons 0.339, lonspeed 0.339, stoplead 0.322; TIME-TO-EVENT 0.270, TIME-GAP 0.167.
* **The groups are equal; the kinds are not.** Worst cells (R2 against the train-mean predictor):
  * `lon_t_start_s` 0.102 (R2 0.11), `lon_t_reach_s` 0.095 (0.21), `lead_tg_min_s` 0.086 (0.27), `lead_tg_s` 0.081 (0.36);
  * `lat_theta_deg` 0.103 (0.52), `turn_exit_y_m` 0.041, `stop_y_m` 0.029 (0.14).
* **The lead time-gap labels are heavy-tailed** (skew 4.5-6.4, capped at 92 s).
  * 77-84 % of those two cells' error comes from cells with |GT| >= 3 SD, where T sits near 1 SD.
  * The two worst episodes (ca11a2a2, 8df0cb04: 12-14 % of the whole error) are dominated by `lead_tg_min_s`.
  * The 24 worst windows (1 %) carry 18 % of the error and come from **7 episodes**.

**Strata** (M2b_s0; share of total squared error against share of cells = lift; full tables in `s2_where.out`):
* **GT lateral class:**
  * LANE_KEEP 0.695 (lift 0.93);
  * TURN_L 0.094 (lift 2.69, **n = 45 windows**, CI [0.009, 0.232]);
  * TURN_R 0.090 (0.82).
* **GT longitudinal class:**
  * STOP 0.173 (lift 2.17), HOLD 0.053 (lift 6.7, n = 22);
  * ACCELERATE 0.224, DECELERATE 0.229 (~1.0);
  * FOLLOW 0.114 (0.60), KEEP 0.127 (0.57).
* **Ego speed band (m/s):** no concentration (lift 0.61-1.18 across `<1`, 1-5, 5-10, 10-15, `>=15`).
* **GT path heading change over 6 s:** `>=30` deg is 14 % of windows and 0.367 of the error (lift 1.62; latcons 0.584).
* **Lead present:** 0.583 of the error (lift 1.14); stoplead lives there by construction (0.992).
* **Nav token:** TURN_L token windows lift 2.02 (n = 117); the FOLLOW token holds 0.705 (lift 0.93).
* **No stratum other than the cell kinds exceeds a lift of about 2.** Concentration is in WINDOWS and EPISODES, not in a driving situation.

**Concentration** (M2b_s0 / s1):
* **Windows by window error:** worst 1 % 0.182 / 0.178; worst 5 % 0.402 / 0.403; worst 10 % **0.550 / 0.551**; worst 25 % **0.780 / 0.782**; worst 50 % 0.936 / 0.938.
* **Episodes by episode error:** worst 10 % (14 episodes) **0.446 / 0.444**; worst 25 % (35 episodes) 0.680 / 0.687.
* The worst 10 episodes together carry 0.377 / 0.364 of the error and hold 11 % of the cells.

## 2. Shared or model-specific (Q2)

`s5_shared.out`. Seven M2b-family arms: s0, s1, grid, hist, pgt, wc, df50.

| read | value |
|---|---|
| per-window total squared error, pairwise **Pearson** | 0.930-0.981, median **0.963** |
| per-window total squared error, pairwise **Spearman** | 0.865-0.930, median 0.900 |
| overlap of the worst-10 % window sets vs M2b_s0 (chance 0.10) | 0.76-0.84 |
| signed cell-residual Pearson | 0.847-0.912, median 0.881; the s0-s1 pair 0.875 |
| **shared fraction of the MSE** (mean pairwise E[r_a r_b] / mean E[r^2]) | **0.876** (7 arms), 0.862 (s0, s1) |
| ensemble-of-7 residual MSE / mean single MSE | 0.894 (**RMSE -5.4 %**) |

* **Shared fraction by kind:**
  * TIME-GAP 0.968, TIME-TO-EVENT 0.925, LEAD GAP 0.888;
  * ANGLE 0.853, GEOMETRY 0.839;
  * **SPEED 0.706**, the only kind with a real model-specific part; the ensemble cuts its RMSE by 13.5 % (0.311 to 0.269).
* **By group:** latcons 0.847, lonspeed 0.857, stoplead 0.926. Window-level Spearman across arms by group: latcons 0.71, lonspeed 0.82, stoplead 0.97 (medians).
* **Reading:** high correlation AND high concentration. The error is a property of the windows (labels or information missing from every model's inputs), not of a training run.

**What each tested lever moved** (MSE change vs M2b_s0, paired, episode bootstrap; `*` = interval excludes 0; one seed each):

| kind | s1 (seed floor) | GRID | HIST | PGT | WC | DF50 |
|---|---|---|---|---|---|---|
| ALL | +1.2 % | +2.8 % | +1.2 % | +8.4 % * | -4.9 % | +10.4 % * |
| TIME-TO-EVENT | +2.8 % | +4.8 % * | +5.8 % * | +9.9 % * | -1.1 % | +7.4 % * |
| TIME-GAP | +1.6 % | +2.5 % | +0.3 % | +9.3 % | -2.3 % | +9.3 % |
| SPEED | +6.4 % | +5.6 % | -3.3 % | +28.3 % * | -16.5 % * | +10.2 % |
| LEAD GAP | -2.8 % | -2.6 % | +4.6 % | +3.9 % | -5.7 % | +9.7 % |

(Column labels follow the script header; `s5_shared.out` Q2c has the CIs.) **No lever lowered the time cells.** WC lowered SPEED -16.5 % separated, and its ADE6 moved -0.0035 m [-0.084, +0.077], not separated (`s10_ade_deltas.out`; the banked WC read has [-0.0795, +0.0850]). The seed pair itself reads -0.1207 [-0.228, -0.018] *separated*, which is the `H-ESTIM-SEED-1` floor, not a lever effect.

## 3. Train versus eval (Q3)

`s6_trainval.out`.

**Identical units and reduction:**
* Train `cons` is the Huber (delta = 1 SD) loss, pooled over ALL present cells of a 64-window batch (`v8f_mini.py:675-683`).
* The logged value is the single batch at the logged step (`v8f_mini_train.py:770-773`), not an interval average.
* The eval equivalent is the mean Huber over all 26,056 present cells.
* My train last-10 %-of-steps means reproduce the DSCALE and WC text exactly (0.1264, 0.1209).

| arm | TRAIN Huber, last 10 % (SE, n = 40) | EVAL Huber [episode CI] | eval / train | eval MSE |
|---|---|---|---|---|
| M2b_s0 | 0.1264 (0.0044) | 0.1355 [0.118, 0.155] | **1.07** | 0.352 |
| M2b_s1 | 0.1230 (0.0036) | 0.1381 [0.120, 0.157] | 1.12 | 0.356 |
| GRID | 0.1203 (0.0046) | 0.1396 | 1.16 | 0.362 |
| HIST | 0.1261 (0.0046) | 0.1379 | 1.09 | 0.357 |
| PGT (p_gt 0.02) | 0.1343 (0.0048) | 0.1473 | 1.10 | 0.382 |
| WC | 0.1066 (0.0038) | 0.1273 | 1.19 | 0.335 |
| DF50 | 0.1209 (0.0037) | 0.1486 | 1.23 | 0.389 |

* **The baseline shows no detectable train/eval gap** in the objective's own units. The eval interval (0.118-0.155) contains the train value, and eval/train is 1.07-1.12.
* The "T fits better and generalises worse" signature in `RESULT_V8F_DSCALE.md` §2 is an ARM-level statement.
  * DF50's train loss is -4.3 % with a difference of 1 SE (z = 0.96), not separated.
  * WC's is -15.6 % (z ~ 3.4), real, against an eval change of -6.0 % (CI wide).
* **The eval error is dominated by cells the train loss barely sees.** Eval Huber is 0.136 against 0.5 x MSE = 0.176; cells with |error| >= 2 SD (1.5 %) carry 40 % of the eval squared error and enter the Huber loss linearly.
* **T is at a common floor on both sides:** train 0.123-0.126 and eval 0.136-0.138 for the baseline pair, with 88 % of the error shared between models. That is a floor, not an overfit, for the baseline.
* **Not separable on CPU with local files:**
  * The train-clip eval cache and `join_train.npz` are not under `box2_ship` and `D:\tanitad` is absent here.
  * The logged batches use p_gt about 0.24 (S's checkpoint is GT with that probability) while eval uses p_gt 0 and S's own checkpoint.
  * A clean split needs one box-2 eval of T on 141 TRAIN clips at p_gt 0 (spec in section 5).

## 4. Lever ranking, with the arithmetic (Q4)

**Admissibility and inputs (read from source):**
* T already reads ego kinematics (`p10f`, PI ruling D-PI-YAWRATE-1), the nav token with its distance, the speed limit, S's checkpoint, and the scene tokens (`v8f_mini.py:556-568`).
* No lever below adds ego state. A goal input must stay information-disjoint from the situation classifier.
* A supplied route token is the ego's own future path on PhysicalAI (optimistic); prefer a PREDICTED goal.

**Ceilings.** Two kinds, kept apart (`s7_ceilings.out`, `s9_lever_table.out`; bar 2F = 0.2415 m):
* **(E) EXACT window-stratum ceiling** [MEASURED, T0]: sum over the stratum of (own ADE6 minus the GT-constraints ADE6), divided by 2,399. It assumes T's constraints are PERFECT in those windows, so it is an outer bound. Windows are disjoint, so strata add.
* **(D) dose-model estimate for a CELL stratum** [ESTIMATED]: a lever that removes a share phi of the total squared error maps to alpha_eq = 1 - sqrt(1 - phi), then the banked response r(alpha) is read.
  * r at alpha = 0.25 / 0.5 / 0.75 is 0.145 / 0.326 / 0.650 (s0) and 0.066 / 0.183 / 0.418 (s1). Below 0.25 a power law through the origin is used.
  * **D assumes O responds to the SIZE of the error, not to which cell carries it.** That is untested, and D is demonstrably wrong in both directions: the heavy-tail lead cells give D = 0.039 m against an exact 0.004 m, while the association reads below put the SPEED cells far above D.

**Stratum and lever ceilings** (s0 / s1):

| stratum or lever | phi (share of T's error) | (D) m | (E) outer bound m |
|---|---|---|---|
| all TIME-TO-EVENT + TIME-GAP error removed | 0.43 | 0.171 / 0.069 | top-10 %-error windows: 0.164 / 0.186 |
| turn cells in the 210 **route-silent** turn windows, removed | 0.189 / 0.181 | 0.059 / 0.017 | 0.161 / 0.142 (15 % of the gap) |
| ... only up to what the model does when the route speaks (-82 % of their error) | 0.154 / 0.149 | 0.046 / 0.013 | same |
| lead cells (LEAD GAP + TIME-GAP), ALL removed (LoRA's stratum) | 0.292 / 0.288 | 0.102 / 0.035 | lead-present windows: 0.501 / 0.433 |
| ... only the privileged-lead headroom | 0.025 / 0.024 | 0.005 / 0.001 | same |
| heavy-tail lead_tg cells (label-side fix) | 0.134 | 0.039 / 0.011 | 32 windows: **0.004 / 0.010** |
| SPEED-kind cells, ALL removed | 0.118 / 0.124 | 0.033 / 0.009 | n/a |
| all cells, RMSE cut 6 % (DSCALE 2x) | 0.116 | 0.033 / 0.009 | n/a |
| all cells, RMSE cut 5.4 % (7-model ensemble) | 0.105 | 0.029 / 0.007 | n/a |
| GRID, HIST, PGT, WC, DF50 as measured | n/a | ~0 | n/a |

* **To clear 2F by accuracy alone:** alpha = 0.33 (s0) or 0.55 (s1) uniform, i.e. 55 % or 80 % of the squared error. The response is convex, and 18.0 % (s0) / 22.6 % (s1) of the windows get WORSE with GT constraints.
* **T-error share is not ADE share.**
  * Windows ranked by T-error show a monotone but weak link: mean gap per window 0.43 m (best decile) to 1.66 m (worst) at s0, 0.39 to 1.89 at s1.
  * The worst decile holds 55 % of the error and only 15-20 % of the gap.
  * Spearman between window error and gap: lonspeed +0.30 / +0.29, latcons +0.10 / +0.09, stoplead -0.01 / +0.05.
  * The leave-one-out necessities (s0): lonspeed +1.99 m, latcons +0.67 m, stoplead +0.42 m.
* **Which cells O listens to is ESTIMATED and weak.**
  * Within-model NNLS of the per-window leave-one-out necessity on kind error (R2 0.15-0.18) puts 90 % of the lonspeed necessity on the SPEED cells, all of latcons on ANGLE (`lat_theta_deg`), and none on stoplead.
  * The cross-arm panel (R2 0.18-0.23; `s7b_panel.out`) agrees on the order: SPEED 0.78-0.90 m per SD^2, ANGLE 0.20-0.23, TIME-TO-EVENT 0.12-0.24, TIME-GAP and GEOMETRY ~0.
  * Against the levers (`s10_ade_deltas.out`): the five realised ADE6 deltas all have intervals (about +-0.1 m) that contain the panel's prediction, so the levers neither confirm nor refute it. The point estimates realise 5 % (WC: predicted -0.068, realised -0.004), 25 % (PGT: +0.117, +0.029) and 93 % (DF50: +0.042, +0.039); HIST and GRID move the other way (-0.077 and -0.076 against about 0).
  * So it is an association, not a lever effect.
* **Where the missing information could come from** (`s8_probe.out`, `s8b_speedcells.out`; a GBM probe trained on the 141 local episodes only, so a probe that LOSES to T says nothing):
  * **SPEED cells:** a probe on ego + nav + limit matches T (0.293 against 0.304 SD). Vision adds nothing measurable there.
  * **v_a:** T's 1.56 / 1.68 m/s error is no better than carrying the ego speed (1.38 m/s).
  * **LEAD GAP:** privileged lead information cuts T's RMSE by 11 % (0.739 to 0.661). **TIME-GAP:** it cuts nothing (0.849 against 0.889).
  * **ANGLE, GEOMETRY, TIME-TO-EVENT:** T beats the non-vision probe by 20 %, 20 % and 8 % (probe kinds use cells with n >= 200 only). Vision plausibly matters there, but no per-kind oracle exists.

**Ranking.** Criterion: (D) central estimate and (E) outer bound for the stratum a lever can reach, weighted by the measured effect so far. Nothing here clears 2F alone, and the table says so.

| rank | lever | stratum and ceiling | evidence on the effect | note |
|---|---|---|---|---|
| **0** | **per-kind GT oracle** (instrument, box 2) | resolves the 10x disagreement between D and the association | none yet | cheapest discriminating experiment; spec in section 5 |
| **1** | **goal/route input for turn cells** (a PREDICTED goal point, information-disjoint from the situation classifier) | route-silent turn windows: (E) 0.161 / 0.142 m, (D) 0.046 / 0.013 m | strongest measured handle: the same model reaches 0.40 SD when the route speaks, 0.95 when silent (chance 1.0), independent of lead time | first a 0-GPU label audit: 210 turn-label windows have a silent nav token and 129 nav-turn windows have no turn label, so the two annotation sources disagree |
| **2** | **variance reduction on T's SPEED cells** (EMA/SWA or two-seed code average) | SPEED: (D) 0.033 / 0.009 m; HYPOTHESIS up to ~0.1 m if O weights the SPEED cells as the association says | the only kind with a model-specific part (0.294 not shared, ensemble -13.5 % RMSE) | needs rank 0 to size it |
| 3 | **encoder LoRA** | lead cells: (D) 0.102 / 0.035 m, privileged-lead headroom only 0.005 / 0.001 m; (E) lead-present windows 0.50 / 0.43 m | none measured (the LoRA artifacts are infrastructure and dry runs); prior LOW (`PREREG_V8F_LORA.md`) | its handle would be the ANGLE/GEOMETRY/TIME cells, where no ceiling exists |
| 4 | **more data (2x)** | uniform alpha about 0.05-0.06: 0.033 / 0.009 m | DF50 vs s0: ALL MSE +10.4 % separated (RMSE +5 %), ADE6 +0.039 [-0.079, +0.155] ns | cannot clear 2F by 7x |
| 5 | **label-side fix** of the heavy-tail lead_tg labels | removes 13 % of the T-error metric; driving value (E) 0.004 / 0.010 m | n/a | an instrument and metric fix; also the Huber-train/MSE-eval mismatch (section 3) |
| - | **finer pooling** (GRID 8x20) | consumed | ALL MSE +2.8 % ns, TIME-TO-EVENT +4.8 % separated worse, ADE6 -0.076 [-0.172, +0.025] ns | no further ceiling |

**The arithmetic's loudest message is about O, not T.**
* A single-digit gain in T's accuracy does not register in ADE. The first 25 % of the way to the GT buys 6-15 % of the gap.
* The levers on T are therefore worth pre-registering only if they target a >= 33 % cut on the cells O actually reads, or if O is made to consume partial accuracy (training O on mixed codes).
* The latter is outside this deliverable. The evidence is the convexity and the 18-23 % of windows that get worse with GT constraints.

## 5. Box-2 jobs this report asks for (not run here; the 4-minute rule)

1. **Per-kind GT oracle (rank 0).**
   * Add a partition of the 26 constraint cells (SPEED = cells 12, 22-25; ANGLE = 0, 5, 9; TIME = 1, 2, 10, 13, 15, 19, 21; GEOMETRY; LEAD GAP) to `CODE_GROUPS` (`v8f_mini.py:54-61`).
   * Run `v8f_mini_eval.py run --tac oracle --tac-groups <kind>` on eval-141 for M2b s0 and s1.
   * About 10 evals. Reads exactly what (D) and the association cannot: the ADE6 worth of each kind.
   * Pre-register both outcomes first (SPEED worth >= 0.2 m, or not).
2. **T on 141 TRAIN clips at p_gt 0** (Q3).
   * A 141-clip train grid with the t = 3 mod 4 rule, plus `join_train.npz` and the train cache.
   * Run `--tac absmask` and compute the same present-cell Huber and RMSE. It separates held-out clips from the p_gt and epoch effects.

## 6. UNVERIFIED and caveats

* Cell-to-ADE weights (NNLS R2 0.15-0.21, panel R2 0.18-0.23) are weak, and one lever family contradicts their size. Ranks 1-2 are conditional on rank 0.
* The probes use 141 episodes against T's 4,512 clips.
* The 34 no-label windows (two episodes) are excluded. The 141 windows with t = 3 have a_t forced to 0 by the held-age clamp (a_t counts 893 / 753 / 753); the mapping is verified by the bit-exact GT block on all 2,365 labelled windows.
* All lever arms are single-seed. The seed pair (s0, s1) is the floor; the true noise floor of a lever read is larger than the bootstrap interval (`H-ESTIM-SEED-1`).
* The dose curve is a frozen-O read (R63): it sizes a consistency check, not a forecast.
* T's absent-slot values are O's input, not error (R63). The pre-R63 reading of "absent slots carry 92 % of the error" is not reused.
* Seed 0 and seed 1 differ in how steep r(alpha) is (0.145 against 0.066 at alpha = 0.25), so every D and "alpha needed" figure is a range.

## Artifacts

All under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/t_attrib/`: `s1_build_table`, `s1b_gt_per_arm_check`, `s2_where`, `s2b_episodes_absent`, `s2c_routesilent`, `s3_tail`, `s4_gap`, `s5_shared`, `s6_trainval`, `s7_ceilings`, `s7b_panel`, `s8_probe`, `s8b_speedcells`, `s9_lever_table`, `s10_ade_deltas` (each `.py` with its `.out`), plus `tcommon.py`, `table.npz` (the per-window table every script reads) and `gap_M2b_s{0,1}.npz`. Inputs are read from `C:/Users/Admin/box2_ship/logs/v8f_*/eval/` and from `git show af3ed6af` of the repo mirror.
