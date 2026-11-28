# PERC-BOTTLENECK — where is refcv8's map / box bottleneck? (RESULT, assembled from raw/*.json)

| | |
|---|---|
| **Evidence class** | MEASURED (ours). Every number below is read by `code/pbn_assemble.py` from `raw/RESULT_Q1.json`, `raw/RESULT_Q2.json`, `raw/RESULT_Q3.json`. |
| **Tier** | **T0** perception read (frozen forward; no closed loop). Not a driving claim. |
| **Checkpoint** | refcv8 EMA-48,852 (`E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`; md5 in `raw/capture_*.json`). ONE training seed. |
| **Prereg** | `PREREG_PERC_BOTTLENECK.md`, sealed sha256 `5489cf56...` BEFORE any probe was fit; `pbn_probe.py` refuses to score on a seal mismatch. |
| **Estimator** | paired episode-cluster bootstrap over eval clips, 1,000 draws, seed 0, 95 % percentile — answers *another draw of EPISODES*; probe seeds 0/1 beside it; NOT training-run variance (`H-ESTIM-SEED-1`). |

## Q1 — features or head?

n: FIT 139 train clips / 556 windows (28 clips held out as VAL for every hyper-parameter); SCORE 137 eval clips / 548 windows. Cells: 0.25 m centres, x 0-40 m, |y| < 3 m.

**Controls.** C1 constant: PASS (A_0_20|lv: AP 0.0585 vs rate 0.0585, AUROC 0.500; B_5_40|lv: AP 0.0919 vs rate 0.0919, AUROC 0.500). C4 head psw>=0.5 vs MAPSEL bands: PASS (0_5 m: 0.266 vs 0.251; 5_10 m: 0.736 vs 0.733; 10_20 m: 0.725 vs 0.716; 20_40 m: 0.738 vs 0.744).

**Probe zoo (selection on VAL only; d and n per C3):**

| stage / probe | d | n fit cells | wd | VAL AP sidewalk (A, lift-valid) s0 / s1 |
|---|---|---|---|---|
| enc / linear | 64 | 423050 | 0.01 | 0.582 / 0.583 |
| enc / mlp | 64 | 423050 | 0.01 | 0.593 / 0.612 |
| lift / linear | 64 | 423050 | 0.01 | 0.445 / 0.438 |
| lift / mlp | 64 | 423050 | 0.0001 | 0.448 / 0.462 |
| img / linear | 2048 | 106099 | 0.001 | 0.452 / 0.457 |
| img / mlp | 2048 | 106099 | 0.001 | 0.494 / 0.484 |
| enc+img / linear | 2112 | 106099 | 0.01 | 0.565 / 0.567 |
| enc+img / mlp | 2112 | 106099 | 0.001 | 0.618 / 0.600 |
| enc / conv3x3 | 64 | 2256756 | 0.0001 | 0.640 / 0.641 |

**Sidewalk AP on SCORE cells (head = its calibrated posterior; best probe by VAL = `enc / conv3x3`).**

| cells | n | sw rate | head AP | best-probe AP | C2 pos-prior AP | C1 AP | head decision P / R | best-probe P @ head R | C2 P @ head R | head drv AP | probe drv AP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_0_20 / lv | 223116 | 0.058 | 0.719 | 0.605 | 0.206 | 0.058 | 0.675 / 0.640 | 0.521 | 0.185 | 0.996 | 0.994 |
| B_5_40 / lv | 459315 | 0.092 | 0.714 | 0.644 | 0.196 | 0.092 | 0.682 / 0.609 | 0.601 | 0.167 | 0.992 | 0.990 |
| A0_0_5 / all | 64382 | 0.055 | 0.297 | 0.470 | 0.253 | 0.055 | 0.200 / 0.608 | 0.412 | 0.256 | 0.970 | 0.986 |
| A0_0_5 / lv | 25872 | 0.037 | 0.576 | 0.483 | 0.229 | 0.037 | 0.631 / 0.448 | 0.509 | 0.252 | 0.990 | 0.989 |
| A0_0_5 / nlv | 38510 | 0.067 | 0.248 | 0.466 | 0.257 | 0.067 | 0.171 / 0.668 | 0.369 | 0.255 | 0.953 | 0.983 |
| bin_0_5 / all | 64382 | 0.055 | 0.297 | 0.470 | 0.253 | 0.055 | 0.200 / 0.608 | 0.412 | 0.256 | 0.970 | 0.986 |
| bin_5_10 / lv | 65724 | 0.051 | 0.694 | 0.558 | 0.230 | 0.051 | 0.678 / 0.623 | 0.482 | 0.231 | 0.996 | 0.994 |
| bin_10_20 / lv | 131520 | 0.066 | 0.744 | 0.638 | 0.194 | 0.066 | 0.677 / 0.668 | 0.542 | 0.155 | 0.997 | 0.995 |
| bin_20_40 / lv | 262071 | 0.115 | 0.712 | 0.664 | 0.191 | 0.115 | 0.684 / 0.590 | 0.640 | 0.166 | 0.987 | 0.985 |
| bin_5_10 / all | 65724 | 0.051 | 0.694 | 0.558 | 0.230 | 0.051 | 0.678 / 0.623 | 0.482 | 0.231 | 0.996 | 0.994 |
| bin_10_20 / all | 131520 | 0.066 | 0.744 | 0.638 | 0.194 | 0.066 | 0.677 / 0.668 | 0.542 | 0.155 | 0.997 | 0.995 |
| bin_20_40 / all | 262071 | 0.115 | 0.712 | 0.664 | 0.191 | 0.115 | 0.684 / 0.590 | 0.640 | 0.166 | 0.987 | 0.985 |

(the best-probe column reports the better of its two probe seeds on that row — descriptive; the committed Δ below uses the VAL-selected seed.)

**Every probe on the primary cells (A 0-20 m, lift-valid):**

| arm | AP sw | AUROC sw | P @ head R | AP drv | 8-class acc |
|---|---|---|---|---|---|
| head | 0.719 | 0.973 | 0.680 | 0.996 | 0.895 |
| head_rawsoftmax | 0.711 | 0.973 | 0.674 | 0.996 | 0.873 |
| C1_constant | 0.058 | 0.500 | 0.058 | 0.905 | 0.854 |
| C2_posprior | 0.206 | 0.827 | 0.185 | 0.972 | 0.854 |
| enc / linear / s0 | 0.591 | 0.955 | 0.532 | 0.993 | 0.865 |
| enc / linear / s1 | 0.593 | 0.955 | 0.537 | 0.993 | 0.868 |
| enc / mlp / s0 | 0.591 | 0.957 | 0.537 | 0.994 | 0.874 |
| enc / mlp / s1 | 0.582 | 0.956 | 0.534 | 0.994 | 0.875 |
| lift / linear / s0 | 0.390 | 0.862 | 0.266 | 0.979 | 0.769 |
| lift / linear / s1 | 0.384 | 0.858 | 0.256 | 0.979 | 0.741 |
| lift / mlp / s0 | 0.453 | 0.886 | 0.299 | 0.984 | 0.862 |
| lift / mlp / s1 | 0.432 | 0.885 | 0.288 | 0.984 | 0.857 |
| enc / conv3x3 / s0 | 0.602 | 0.958 | 0.543 | 0.994 | 0.875 |
| enc / conv3x3 / s1 | 0.605 | 0.956 | 0.521 | 0.994 | 0.874 |
| img / linear / s0 | 0.386 | 0.872 | 0.284 | 0.979 | 0.718 |
| img / linear / s1 | 0.403 | 0.876 | 0.298 | 0.980 | 0.699 |
| img / mlp / s0 | 0.454 | 0.901 | 0.347 | 0.987 | 0.863 |
| img / mlp / s1 | 0.454 | 0.899 | 0.337 | 0.986 | 0.862 |
| enc+img / linear / s0 | 0.533 | 0.932 | 0.461 | 0.988 | 0.751 |
| enc+img / linear / s1 | 0.539 | 0.934 | 0.473 | 0.989 | 0.760 |
| enc+img / mlp / s0 | 0.577 | 0.950 | 0.507 | 0.993 | 0.875 |
| enc+img / mlp / s1 | 0.560 | 0.947 | 0.486 | 0.993 | 0.874 |

**Committed contrasts (paired bootstrap):**

| contrast | cells | point | 95 % CI | separated |
|---|---|---|---|---|
| delta_best_minus_head (`enc / conv3x3 / s1` − `head`) | A_0_20 / lv | -0.1145 | [-0.1570, -0.0768] | yes |
| delta_best_minus_head (`enc / conv3x3 / s1` − `head`) | B_5_40 / lv | -0.0701 | [-0.1066, -0.0394] | yes |
| delta_img_minus_enc (`img / mlp / s0` − `enc / conv3x3 / s1`) | A_0_20 / lv | -0.1507 | [-0.1978, -0.1014] | yes |
| delta_img_minus_enc (`img / mlp / s0` − `enc / conv3x3 / s1`) | B_5_40 / lv | -0.1442 | [-0.1867, -0.0998] | yes |
| delta_best_minus_head (`enc / conv3x3 / s1` − `head`) | A0_0_5 / all | +0.1183 | [+0.0509, +0.1933] | yes |
| delta_C2_minus_head (`C2_posprior` − `head`) | A0_0_5 / all | -0.0440 | [-0.0930, -0.0005] | yes |

**Data-limited check** (`enc|mlp` VAL AP, full vs half FIT windows): 0.593 vs 0.589, gain +0.004 -> not data-limited at this size.

**Visibility (A0, 0-5 m):** 64382 GT-seen cells, 59.8 % lift-invalid; 10717 head sidewalk flags, 93.6 % of them on lift-invalid cells -> VISIBILITY-LIMITED.

**VERDICT (committed rule):** band A 0-20 m → **O2_FEATURE_LIMITED**; band B 5-40 m → **O2_FEATURE_LIMITED**. Localisation: A → **IMAGE_TRUNK_IS_THE_CEILING**, B → **IMAGE_TRUNK_IS_THE_CEILING**. Function classes tested: linear, MLP-256, 3x3-conv (a failing probe shows only *not readable by these classes*).

## Q2 — is the GT wrong where the own map disagrees?

Windows read 528 (no bevgt 20, no label 0). LiDAR-OCCUPIED rate: FP (own sidewalk on GT-drivable, 0-20 m) **0.028** (n 9393); AGR control (own drivable on GT-drivable) 0.037 (n 213056); MIS (FP cells, ±20 frames) 0.029. By distance: 0_5 m 0.028 (n 7726); 5_10 m 0.037 (n 455); 10_20 m 0.024 (n 1212).

Committed reading: obstacle/GT-error share NOT SHOWN; upper bound on the precision correction (FP − AGR occupancy) -0.008.

Curb test: BLOCKED: the artifact has no ground-height channel (occupancy band starts at 0.30 m); raw sweeps on the dev box cover 2 clips, neither in eval139 (checked by uuid and sha12 prefix).

## Q3 — the box head (refcv8 EMA-48,852, eval139 S2 windows, from PERC8's banked dump)

n: 4532 windows / 139 episodes / 14586 VIS-1 positives. Controls: re-derived AP@2 m 0.3282 vs PERC8 banked 0.328 (PASS); vectorised bootstrap AP == programme AP (PASS).

| AP@2 m | value | 95 % CI | n pos |
|---|---|---|---|
| all | 0.328 | [0.301, 0.354] | 14586 |
| range 0_20 m | 0.539 | [0.503, 0.572] | 5606 |
| range 20_40 m | 0.234 | [0.214, 0.253] | 5399 |
| range 40_60 m | 0.092 | [0.077, 0.107] | 3581 |
| size group small (PROXY by class: person, rider, stroller, animal, protruding_object) | 0.127 | — | 3626 |
| size group medium (PROXY by class: automobile, other_vehicle) | 0.357 | — | 10589 |
| size group large (PROXY by class: heavy_truck, bus, trailer) | 0.101 | — | 371 |
| class automobile | 0.360 | — | 10495 |
| class heavy_truck | 0.144 | — | 230 |
| class bus | 0.109 | — | 67 |
| class other_vehicle | 0.018 | — | 94 |
| class trailer | 0.011 | — | 74 |
| class person | 0.177 | — | 2871 |
| class rider | 0.094 | — | 645 |
| class stroller | 0.008 | — | 19 |
| class animal | 0.079 | — | 7 |
| class protruding_object | 0.021 | — | 84 |

**Class collapse?** TP (2 m greedy, all confidences) class accuracy **0.822** [0.800, 0.844] vs C-MAJ (always 'automobile') 0.726; non-automobile TPs predicted automobile 330/3934 (8.4 %) -> **NOT COLLAPSED** (committed rule).

Confusion (rows GT, cols predicted argmax) of the 2 m greedy TPs:

| GT \ pred | automo | heavy_ | bus | other_ | traile | person | rider | stroll | animal | protru |
|---|---|---|---|---|---|---|---|---|---|---|
| automobile | 9266 | 196 | 51 | 36 | 227 | 106 | 178 | 6 | 2 | 364 |
| heavy_truck | 7 | 198 | 1 | 3 | 10 | 0 | 5 | 0 | 0 | 6 |
| bus | 1 | 1 | 58 | 0 | 1 | 3 | 0 | 1 | 0 | 0 |
| other_vehicle | 5 | 18 | 0 | 27 | 8 | 7 | 16 | 2 | 0 | 9 |
| trailer | 15 | 15 | 0 | 17 | 22 | 4 | 0 | 0 | 0 | 0 |
| person | 210 | 10 | 20 | 15 | 28 | 1745 | 498 | 145 | 41 | 17 |
| rider | 52 | 11 | 4 | 1 | 0 | 79 | 464 | 23 | 0 | 2 |
| stroller | 0 | 1 | 0 | 0 | 0 | 8 | 2 | 8 | 0 | 0 |
| animal | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 3 | 1 |
| protruding_object | 39 | 18 | 0 | 3 | 4 | 1 | 0 | 0 | 0 | 18 |

Size error |dl|+|dw| (m) on the Hungarian pairs, by GT class (pairs recovered by unique distance; 9/14586 unresolved and dropped): trailer median 0.88 (n 74); rider median 0.71 (n 643); other_vehicle median 1.50 (n 94); person median 0.09 (n 2867); automobile median 0.23 (n 10492); bus median 1.11 (n 67); protruding_object median 2.73 (n 84); heavy_truck median 1.19 (n 230); animal median 0.75 (n 7); stroller median 0.35 (n 19). C-SIZE: NOT COMPUTABLE from the banked pack (no GT l/w stored) -- disclosed deviation.

## Headline

- Q1: A O2_FEATURE_LIMITED / B O2_FEATURE_LIMITED; localisation A IMAGE_TRUNK_IS_THE_CEILING / B IMAGE_TRUNK_IS_THE_CEILING; A0 VISIBILITY-LIMITED.
- Q2: FP occupancy 0.028 vs agreement 0.037 vs mispaired 0.029 -> obstacle share NOT SHOWN; curb test BLOCKED (no ground-height channel).
- Q3: AP@2m 0.328; 0-20 / 20-40 / 40-60 m 0.539 / 0.234 / 0.092; class NOT collapsed (TP acc 0.822 vs majority 0.726).
