# STAGE1: how well does the frozen image grid of the t0 frame predict the logged speed at t0? (H-LEVELPROBE-1 Stage 1, AS WRITTEN)

Card `PREREG_LEVELPROBE.md` sha256 `25c66e938d1ee6deee08bb538446d06224eaa9dff17d9e78aea64f58abe7eef4` (frozen before this fit; `FROZEN_SHA.txt`, `ERRATA_1.md`). Tier **T0-class probe** (frozen features of logged frames; NOT driving performance). Evidence class **MEASURED (ours)**; artifact `raw/STAGE1.json` (+ `raw/stage1_pred_eval608.npz`); code `code/stage1.py`; run clock 2026-11-04T01:51:06+0100.

**Coverage (F1, `raw/F1_coverage.json`): every number below is on the 608 eval windows with v0 < 6 (of the 2,399 the T1 rig rolls out) and was fitted on the 24,000 TRAIN windows of P-STOP + P-MOVE (labels in [0, 6)). It says nothing about the 1,791 windows with v0 >= 6, which hold 85.1 % of the M1 windows.**

Features: probe 3's G8 grid (DINOv3-L/16 pool8x20, current frame) mean-pooled to G2x4 = 8192 features; vision only (`assertions.ego_nav_columns_loaded_into_fit` = False). Ridge, alpha = c x trace/D, c chosen by episode-grouped 5-fold CV on the TRAIN windows (folds = `int(sha12,16) % 5`): chosen **c = 1.0** (grid edge: False); CV RMSE by c: 0.001: 1.745, 0.01: 1.536, 0.1: 1.384, 1.0: 1.338, 10.0: 1.479, 100.0: 1.767, 1000.0: 2.020. Fit and score clip sets disjoint (asserted): 2185 train clips, 67 eval clips.

## 1. The scores on the eval windows (m/s)

| subset | n | RMSE | MAE | R^2 (vs eval mean) | eval v0 mean / std |
|---|---|---|---|---|---|
| all | 608 | 1.416 | 1.125 | 0.489 | 3.07 / 1.98 |
| M1_windows | 313 | 1.591 | 1.289 | 0.117 | 3.93 / 1.69 |
| non_M1_windows | 295 | 1.203 | 0.951 | 0.581 | 2.17 / 1.86 |
| class_L | 327 | 1.491 | 1.196 | 0.431 | 2.98 / 1.98 |
| class_S | 32 | 0.886 | 0.676 | -25.433 | 0.11 / 0.17 |
| class_O | 8 | 1.194 | 1.035 | -106.743 | 0.16 / 0.12 |
| class_M | 241 | 1.376 | 1.092 | 0.308 | 3.70 / 1.65 |

K3, the train-mean predictor (label mean 1.867 m/s) on the same subsets:

| subset | n | RMSE | MAE | R^2 (vs eval mean) |
|---|---|---|---|---|
| all | 608 | 2.319 | 2.009 | -0.372 |
| M1_windows | 313 | 2.666 | 2.383 | -1.480 |
| non_M1_windows | 295 | 1.883 | 1.612 | -0.027 |
| class_L | 327 | 2.266 | 1.962 | -0.314 |
| class_S | 32 | 1.764 | 1.756 | -103.755 |
| class_O | 8 | 1.709 | 1.705 | -219.683 |
| class_M | 241 | 2.469 | 2.118 | -1.228 |

RMSE skill versus the train-mean predictor (1 - RMSE_ridge / RMSE_mean): all 0.389, M1_windows 0.403, non_M1_windows 0.361, class_L 0.342, class_S 0.498, class_O 0.301, class_M 0.443.

## 2. Controls

| control | threshold | reading | PASS |
|---|---|---|---|
| POOL equals an independent loop | max abs dev <= 1e-4 | 9.54e-08 | True |
| K1 labels shuffled within TRAIN (seed 1, same pipeline) | eval R^2 <= 0.02 | -0.3733 (RMSE 2.321, chosen c 1000.0) | True |
| K2 v_logged appended (column scale 9.69) | eval R^2 >= 0.99 | 1.0000 (RMSE 0.000, chosen c 0.1) | True |
| K3 train-mean RMSE (reported) | baseline | 2.319 m/s on all 608 | reported |

All controls pass: **True**.

## 3. What this can and cannot say
* It is a ridge on a pooled (G2x4) readout of ONE frozen encoder at ONE frame; a low R^2 is not a statement about learnability (a trained in-model head could do better), and a high one is not driving performance.
* The train mixture is 50 % stopped windows (P-STOP, v0 < 0.5) by construction; the eval windows are 101 stopped and 507 in [0.5, 6): the mixtures differ, so R^2 and RMSE are not comparable to a probe fitted on the natural mix.
* The estimator is one fit on one train set, scored on 141 eval episodes (608 windows with v0 < 6): there is no interval here; the eval windows are correlated within episodes and per-class cells are small.
* K3's RMSE skill is inflated by the mixture shift: the train-mean predictor sits at 1.87 m/s while the eval mean is 3.07 m/s. The mixture-free reference is the eval-mean predictor, whose RMSE is the eval std (1.98 m/s overall): the ridge's 1.42 m/s is 28 % below that (R^2 0.489); on the M1 windows 1.59 m/s versus an std of 1.69 m/s (R^2 0.117).
* R^2 inside the S and O classes (n = 32 and 8, eval std 0.17 and 0.12 m/s) is not interpretable (the denominator is near zero); read RMSE / MAE there.
* Stage 2's HLVL-PRED cannot use this fit (F1): it needs levels for all 2,399 windows. See `AMEND_A1_PROPOSED.md`.

