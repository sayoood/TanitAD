# PREREG_P2 -- is refcv8's thin-class map "slightly misplaced" or "absent"?  (boundary-tolerant re-read of the 10 cm map)

*EvalFlyWheel, 2026-10-25 (Berlin local). Written and sha256-pinned BEFORE the scored GPU passes. The only data touched before this file was
committed: the synthetic controls (`code/test_p2_tol.py`, all PASS) and the banked perc8 tables' *shapes and keys* (never a class statistic). A <= 8-window CPU smoke of the plumbing
follows this commit (not scored, not read for any class statistic). The banked perc8 tolerance columns (`tp_pred` / `tp_gt` at 0.2 m) were NOT looked at.*

## 0. Question, evidence class, tier, estimator, n

| | |
|---|---|
| **Question** | `perc8/RESULT.md` section 3 measured refcv8's 10 cm exact-cell map IoU: lane 0.199 (0-20 m) / 0.060 (20-40 m) / ~0 beyond, crosswalk 0.149 (0-20 m) / ~0 beyond, arrow / non-drivable edge / hatched ~0. Exact 10 cm IoU on a 10-15 cm line halves at a 10 cm offset, so it cannot tell MISPLACEMENT (the model draws the line, a few cells off) from ABSENCE (the model does not draw it). This read separates the two with boundary-tolerant precision / recall / F / IoU. |
| **Evidence class** | MEASURED when run (ours): artifacts under `C:/Users/Admin/refcv8_pf/perc8tol/`. Nothing in this file is a result. |
| **Tier** | **T0** -- the perception output of the model's forward on the S2 windows (no closed loop). A perception read, NOT a driving-performance claim. |
| **Arms** | `refcv8_ema48852` = `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt` (md5 b4364029b68f29a33eeb4aebbbbd002b) -- PRIMARY. `refcv7_50400` = `E:/refcv7_eval_kit/ckpt/ckpt_50400.pt` (md5 b418d0fc4a92a6848c246a6a7c50207b) -- context / paired comparison. |
| **Windows** | the S2 list `s2_windows.json` (md5 a3a78dba889e70aa69a05271559794ad): 4,754 windows / 139 episodes; map-labelled windows expected 4,686 / 137 episodes (perc8's number). BOTH arms are re-forwarded on these identical windows on the dev box (RTX 4060), unshifted pose clock (`R8_NO_POSE_SYNC=1`, as perc8's primary read), the arm's DECLARED decision rule (prior-corrected, the run's frozen class weights). |
| **refcv7 side reuse** | `E:/refcv7_diag_bin/2026-10-04/eval_final.acc.pt` holds Chebyshev tolerance accumulators for k in {1,2,5,10} but on the `eval_diag` window set (K = 8 windows per episode over the same 139 episodes, ~1,112 windows), NOT on the S2 windows. Not the same windows => NOT used for the paired read (it may appear as an INHERITED plausibility column only). |
| **Estimator** | episode-cluster bootstrap over the map-labelled eval episodes, the `taniteval.ci` draw stream (`ci._draws`, seed 0, 1,000 draws), 95 % percentile interval; PAIRED = the same draws for both arms (equals `ci.paired_episode_cluster_bootstrap`; checked, control C6). |
| **What the CI answers** | *another draw of EPISODES only.* It does NOT answer "would another TRAINING run say this?" (`H-ESTIM-SEED-1`: ONE training seed per arm; a separated interval is necessary, not sufficient) and the forward is deterministic up to GPU numerics (no inference sampling). |

## 1. Definitions (pooled counts over all map windows; per class c, 20 m band b)

* Cells are 0.1 m. Scored cells = `gt != 255` (the programme's `_scored_mask`). Tolerance `k` = **Chebyshev** (chessboard) distance in cells, **k in {0, 1, 2, 5, 10} = {0, 0.1, 0.2, 0.5, 1.0 m}**; cells outside the grid never count (identical to `diag_metrics.dilate`).
* `pred_c`, `gt_c` = #cells predicted / labelled c. `tpp_k` = #predicted-c cells with a GT-c cell within k; `tpg_k` = #GT-c cells with a predicted-c cell within k. (`tpp_0 = tpg_0 = inter`.)
* `P_k = tpp_k / pred`, `R_k = tpg_k / gt`, `F_k = 2 P_k R_k / (P_k + R_k)`, **`IoU_k = F_k / (2 - F_k)`**. At k = 0 this IS the exact IoU `inter/union`. Conventions as `MH._f1`: both sides empty -> undefined; exactly one side empty -> F = 0.
* Distance bands: 0-20, 20-40, 40-60, 60-80, 80-100 m. All five are reported; the VERDICT is for 0-20 and 20-40 m.

## 2. The mechanical verdict (committed; applied to the POOLED POINT ESTIMATES, per arm; PRIMARY arm = refcv8)

Verdict classes: **lane / road line** and **crosswalk**, bands **0-20 m** and **20-40 m** (4 cells per arm).

1. **UNDERPOWERED** (no verdict) if the (class, band) has `gt` < 100,000 cells or GT in < 20 episodes.
2. else **ABSENT** if `R_5 < 0.20` (fewer than 20 % of the GT cells have a predicted cell of that class within 0.5 m; this is conservative for "absent" because over-prediction can only inflate R_5).
3. else **MISPLACED** if `R_5 >= 0.50` **and** `P_5 >= 0.50` **and** `IoU_5 >= 2 x IoU_0`. (The precision clause is the sharpening: without it a diffuse over-predictor passes on recall alone.)
4. else **MIXED**.

Annotation (does NOT change a verdict): **CI-FRAGILE** if the 95 % bootstrap interval of the deciding quantity straddles its threshold (`R_5` vs 0.20 for ABSENT/MIXED; `R_5`, `P_5` vs 0.50 and `IoU_5/IoU_0` vs 2 for MISPLACED).

**Descriptive only (no verdict):** arrow / text, non-drivable edge, hatched area (all bands); lane and crosswalk in the 40-100 m bands; drivable, sidewalk, "seen, no map class" (context rows); every other band x class. Each is reported with P / R / F / IoU at every k, n_gt, and the bootstrap interval.

## 3. Secondary readings (descriptive, no verdict)

* `k*` = smallest k in {0,1,2,5,10} with `R_k >= 0.5` (else ">10").
* Paired difference refcv8 - refcv7 of `R_5`, `P_5`, `IoU_0`, `IoU_5`, with the paired episode-cluster interval. One seed per arm: SEP is necessary not sufficient; no lever claim.
* **GT line-width census** (S2 windows, GT only, identical for both arms): per class x band, the thickness of every GT cell = min(maximal lateral run, maximal longitudinal run through it), in cells; median, p10 / p90, share <= 1, 2, 3, 5 cells; and the lateral / longitudinal run histograms. Plus the analytic exact-IoU-versus-offset table for parallel strips of the measured median width (`IoU_0 = (w - d)+ / (w + d)`).

## 4. Controls (each must read its known value; a FAIL is reported as a FAIL and every verdict below it is then marked UNVERIFIED -- no verdict is suppressed and no threshold is moved)

| id | control | known value | PASS criterion |
|---|---|---|---|
| C1 | GT as the prediction (synthetic random scene + the real GT of the first 20 map windows) | `P = R = F = IoU = 1` at every k, every class with GT | exact |
| C2 | all-one-class prediction (class 1 everywhere) | every other class: `tpg_k = tpp_k = inter = 0` => `IoU_k = 0` at every k | exact |
| C3 | analytic strips (width w, lateral offset d): closed form `tpp_k = rows x min(w, max(0, k+w-d))`; diagonal-offset literals; mutation (taxicab metric) must be CAUGHT | closed form | exact (`test_p2_tol.py`) |
| C4 | chessboard k=1 vs the programme's Euclidean EDT (< 0.2 m) `tp_pred` / `tp_gt`, classes 2, 3, 5, EVERY real window | identical | 0 mismatching cells; also vs the torch max-pool `dilate` instrument on random scenes (exact) |
| C5 | forward reproduction at k = 0 vs the BANKED tables (`perc8/raw/A_nosync/perc_map_table.npz` for refcv8, produced on Thor; `E:/refcv7_eval_kit/perc/step50400_a6text/perc_map_table.npz` for refcv7, produced on this dev box): (a) `n_gt` per window x class x band **exactly equal**, (b) `eid` sequence equal, (c) exact IoU per class x band (all cells with `n_gt` >= 100,000) | equal | (a),(b): exact. (c): max abs diff <= 2e-3 (cross-GPU numerics; the value is reported, and "EXACT" if <= 1e-9). A different GPU / torch can flip a few near-tie argmax cells -- 1e-9 is not a pre-registrable pass criterion for a re-forward. |
| C5b | **bank-consistency re-score** (no forward): this analysis' pooling code applied to the banked tables reproduces `MH.pooled` and perc8's `raw/map_read.json` exact IoU | equal | max abs diff <= 1e-9 against `MH.pooled`; `round(.,4)` equals the banked `mean` |
| C6 | bootstrap: the vectorised draw stream reproduces `ci.paired_episode_cluster_bootstrap` (same `ci._draws`) on >= 6 (class, band, metric) cells, both arms | equal | `round(mine, dp) == ci's value` for delta, lo, hi |
| C7 | pairing identity: the two arms' `eid` / `ws` sequences are equal, and `n_gt` is equal element-wise; the GT census is equal | equal | exact |
| C8 | identities: ckpt md5 of both arms; S2 md5; eval-cache 139 md5s vs perc8's list | as in section 0 | exact |
| C9 | LAW-frame control (zero LAW frame leaves every perception tensor unchanged, the law term differs) on 2 windows per arm | as perc8 | exact |

## 5. What each outcome means and the next lever (committed in advance -- a refutation is a waypoint, not a deliverable)

* **lane MISPLACED at 0-20 m:** the exact 10 cm IoU understates a head that localises lines to within 0.5 m. The metric to quote for line classes is the tolerant IoU at the tolerance a consumer needs (the planner reads the map at what resolution? -- to be answered, not assumed). Next lever: a localisation-sharpness arm (higher-resolution decode / line-aware loss), pre-registered separately; nothing here authorises training.
* **lane ABSENT (R_5 < 0.20):** the head does not draw the class beyond what chance proximity gives; the exact-IoU number was NOT a resolution artefact. Next lever: the thin-class loss / class-weight / data lever (the 10 cm label release), pre-registered separately.
* **MIXED:** report the k at which recall rises (k*) and the precision at that k; the next lever is whichever of the two (localisation vs presence) the P_k / R_k curves say is the larger gap -- chosen by the measured curve, not by preference.
* Whichever way it falls, the report states the verdict, the controls, the census, and the next-lever recommendation; it does not stop at the label.

## 6. What this does NOT establish

T0 only; one training seed per arm; the bootstrap answers episodes only; "misplaced / absent" is a statement about the pooled statistic on 137 episodes at <= 1 m tolerance and says nothing about driving; GPU numerics differ between Thor and the dev box (control C5 bounds it); k = 5 and 10 are post-hoc relative to perc8 but pre-registered HERE before any value was seen.
