# RESULT - MOTIONPROBE: do cached frozen DINOv3-L token grids, stacked over frames, carry ego motion?

Evidence class: MEASURED (ours). Pre-registration: `PREREG_MOTIONPROBE.md` (sha256 in `FROZEN_SHA.txt`); numbers: `raw/MOTIONPROBE.json` (cells), `raw/cells_ridge.json`, `raw/cells_mlp.json`, `raw/cells_grid2.json`, `raw/VERDICT.json`. Tier: not a T1 driving number (T0-class diagnostic of the cached perception features; no driving model involved).

Estimator: fold mean +/- SD (ddof 1) over 5 folds BY CLIP. It answers "would another draw of CLIPS say this?" only. It does not answer "would another training run say this?" (ridge is deterministic; the MLP has one seed per fold) nor "would another inference run say this?" (nothing stochastic at inference). Controls K1/K2/K3 are the guard against reading a spurious gain as motion.

## Verdict (committed rule, PREREG section 9)

| target | verdict | s* | readout (s*) | readout (F1) | R2 band F1 | R2 band s* | gain g | gain K2 | gain K3 | paired gain by fold |
|---|---|---|---|---|---|---|---|---|---|---|
| v | **INVALID** | D3 | ridge | mlp | -0.435 | +0.975 | +1.410 | +0.218 | -0.141 | +0.995, +2.171, +1.170, +1.265, +1.446 |
| dv | **INVALID** | D3 | ridge | ridge | +0.011 | +0.012 | +0.001 | -0.015 | -0.013 | +0.001, +0.016, -0.007, +0.003, -0.006 |

* `v`: INVALID A K1 control left its known range: the pipeline has a defect and no verdict is issued.
* `dv`: INVALID A K1 control left its known range: the pipeline has a defect and no verdict is issued.

Sensitivity (cannot change the verdict): D3f (differences projected with the frame basis) best-readout band R2 minus F1 = +1.406 on v, -0.010 on dv; K2/K3 were not run for D3f.

K1 VIOLATIONS: [{"readout": "ridge", "fs": "F1mean", "target": "k1_v", "mean_r2_all": -0.5637286884502515}, {"readout": "ridge", "fs": "F1", "target": "k1_v", "mean_r2_all": -0.1381664551833845}, {"readout": "ridge", "fs": "F4", "target": "k1_v", "mean_r2_all": -0.13948783831206096}, {"readout": "ridge", "fs": "D3", "target": "k1_v", "mean_r2_all": -0.20852001876462012}, {"readout": "ridge", "fs": "D3f", "target": "k1_v", "mean_r2_all": -0.31885347419080906}, {"readout": "ridge", "fs": "F4_K2", "target": "k1_v", "mean_r2_all": -0.12437237638030188}, {"readout": "ridge", "fs": "D3_K2", "target": "k1_v", "mean_r2_all": -0.07871794294569985}, {"readout": "ridge", "fs": "F4_K3", "target": "k1_v", "mean_r2_all": -0.1384437517494611}, {"readout": "ridge", "fs": "D3_K3", "target": "k1_v", "mean_r2_all": -0.12791080762335363}, {"readout": "mlp", "fs": "F4", "target": "k1_v", "mean_r2_all": -0.13986439493762165}]

## Sample

40 clips kept (0 dropped), clip-list sha256 `86101f7ae2d35987e201fd8cfb5f6aa5fa6b066892851d064ef40332506ce122`, 5 folds by clip, windows total 1960 (band 3-8 m/s: 1139). eval-141 disjointness asserted against `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/manifest.jsonl` (0 shared ids). pool8x20 subset: 20 clips.

PCA retained variance (mean over cells and folds): frame basis 0.994, difference basis 0.099, pooled 8x20 basis 0.127.

## R2, target v(t), current speed, band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1139 | -0.445 +/- 0.651 | -0.436 +/- 0.593 |
| F1 | 1280 | 1139 | -0.557 +/- 0.497 | -0.435 +/- 0.455 |
| F4 | 5120 | 1139 | -0.360 +/- 0.421 | -0.460 +/- 0.523 |
| D3 | 3840 | 1139 | +0.975 +/- 0.003 | +0.836 +/- 0.063 |
| D3f | 3840 | 1139 | +0.972 +/- 0.006 | n/a |
| F4_K2 | 5120 | 1139 | -0.356 +/- 0.480 | -0.529 +/- 0.602 |
| D3_K2 | 3840 | 1139 | -0.216 +/- 0.342 | -0.485 +/- 0.476 |
| F4_K3 | 5120 | 1139 | -0.499 +/- 0.482 | -0.486 +/- 0.524 |
| D3_K3 | 3840 | 1139 | -0.576 +/- 0.426 | -0.819 +/- 0.589 |

## R2, target v(t), current speed, all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1960 | +0.152 +/- 0.122 | +0.165 +/- 0.175 |
| F1 | 1280 | 1960 | -0.016 +/- 0.183 | +0.022 +/- 0.193 |
| F4 | 5120 | 1960 | +0.094 +/- 0.212 | +0.002 +/- 0.204 |
| D3 | 3840 | 1960 | +0.992 +/- 0.003 | +0.922 +/- 0.022 |
| D3f | 3840 | 1960 | +0.991 +/- 0.004 | n/a |
| F4_K2 | 5120 | 1960 | +0.192 +/- 0.129 | -0.092 +/- 0.179 |
| D3_K2 | 3840 | 1960 | +0.273 +/- 0.144 | +0.231 +/- 0.150 |
| F4_K3 | 5120 | 1960 | -0.009 +/- 0.190 | -0.008 +/- 0.210 |
| D3_K3 | 3840 | 1960 | -0.100 +/- 0.154 | +0.042 +/- 0.251 |

## R2, target dv = v(t)-v(t-3), band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1139 | +0.006 +/- 0.010 | +0.001 +/- 0.010 |
| F1 | 1280 | 1139 | +0.011 +/- 0.008 | +0.005 +/- 0.015 |
| F4 | 5120 | 1139 | +0.011 +/- 0.008 | +0.005 +/- 0.014 |
| D3 | 3840 | 1139 | +0.012 +/- 0.003 | -0.034 +/- 0.044 |
| D3f | 3840 | 1139 | +0.001 +/- 0.009 | n/a |
| F4_K2 | 5120 | 1139 | +0.012 +/- 0.018 | -0.029 +/- 0.023 |
| D3_K2 | 3840 | 1139 | -0.005 +/- 0.006 | -0.032 +/- 0.026 |
| F4_K3 | 5120 | 1139 | +0.011 +/- 0.008 | +0.007 +/- 0.011 |
| D3_K3 | 3840 | 1139 | -0.002 +/- 0.003 | -0.014 +/- 0.012 |

## R2, target dv = v(t)-v(t-3), all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1960 | -0.006 +/- 0.008 | -0.017 +/- 0.017 |
| F1 | 1280 | 1960 | -0.007 +/- 0.013 | -0.012 +/- 0.022 |
| F4 | 5120 | 1960 | -0.007 +/- 0.013 | -0.020 +/- 0.019 |
| D3 | 3840 | 1960 | +0.026 +/- 0.010 | -0.024 +/- 0.029 |
| D3f | 3840 | 1960 | +0.004 +/- 0.013 | n/a |
| F4_K2 | 5120 | 1960 | -0.028 +/- 0.029 | -0.020 +/- 0.016 |
| D3_K2 | 3840 | 1960 | -0.006 +/- 0.003 | -0.031 +/- 0.021 |
| F4_K3 | 5120 | 1960 | -0.007 +/- 0.013 | -0.017 +/- 0.015 |
| D3_K3 | 3840 | 1960 | -0.001 +/- 0.004 | -0.007 +/- 0.018 |

## RMSE (m/s), band 3-8 m/s, target v

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1139 | +1.73 +/- 0.48 | +1.72 +/- 0.45 |
| F1 | 1280 | 1139 | +1.81 +/- 0.38 | +1.74 +/- 0.36 |
| F4 | 5120 | 1139 | +1.69 +/- 0.35 | +1.75 +/- 0.41 |
| D3 | 3840 | 1139 | +0.23 +/- 0.01 | +0.58 +/- 0.12 |
| D3f | 3840 | 1139 | +0.24 +/- 0.01 | n/a |
| F4_K2 | 5120 | 1139 | +1.68 +/- 0.37 | +1.78 +/- 0.43 |
| D3_K2 | 3840 | 1139 | +1.60 +/- 0.29 | +1.76 +/- 0.37 |
| F4_K3 | 5120 | 1139 | +1.78 +/- 0.38 | +1.77 +/- 0.41 |
| D3_K3 | 3840 | 1139 | +1.82 +/- 0.34 | +1.95 +/- 0.40 |

## RMSE (m/s), band 3-8 m/s, target dv

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1139 | +0.50 +/- 0.03 | +0.50 +/- 0.03 |
| F1 | 1280 | 1139 | +0.50 +/- 0.03 | +0.50 +/- 0.03 |
| F4 | 5120 | 1139 | +0.50 +/- 0.03 | +0.50 +/- 0.03 |
| D3 | 3840 | 1139 | +0.50 +/- 0.03 | +0.51 +/- 0.03 |
| D3f | 3840 | 1139 | +0.50 +/- 0.03 | n/a |
| F4_K2 | 5120 | 1139 | +0.50 +/- 0.03 | +0.51 +/- 0.02 |
| D3_K2 | 3840 | 1139 | +0.50 +/- 0.03 | +0.51 +/- 0.03 |
| F4_K3 | 5120 | 1139 | +0.50 +/- 0.03 | +0.50 +/- 0.03 |
| D3_K3 | 3840 | 1139 | +0.50 +/- 0.03 | +0.50 +/- 0.03 |

## Controls

K1 (train targets permuted; scored on true test targets; known value: all-windows R2 in [-0.05, 0.03]):

| readout | feature set | target | mean all-windows R2 |
|---|---|---|---|
| ridge | F1mean | k1_v | -0.564 |
| ridge | F1mean | k1_dv | -0.003 |
| ridge | F1 | k1_v | -0.138 |
| ridge | F1 | k1_dv | -0.003 |
| ridge | F4 | k1_v | -0.139 |
| ridge | F4 | k1_dv | -0.003 |
| ridge | D3 | k1_v | -0.209 |
| ridge | D3 | k1_dv | -0.010 |
| ridge | D3f | k1_v | -0.319 |
| ridge | D3f | k1_dv | -0.008 |
| ridge | F4_K2 | k1_v | -0.124 |
| ridge | F4_K2 | k1_dv | -0.004 |
| ridge | D3_K2 | k1_v | -0.079 |
| ridge | D3_K2 | k1_dv | -0.003 |
| ridge | F4_K3 | k1_v | -0.138 |
| ridge | F4_K3 | k1_dv | -0.003 |
| ridge | D3_K3 | k1_v | -0.128 |
| ridge | D3_K3 | k1_dv | -0.004 |
| mlp | F4 | k1_v | -0.140 |
| mlp | F4 | k1_dv | -0.011 |

K2 (frames t-3..t-1 from another clip of the same fold) and K3 (frame order shuffled per window) rows are the `_K2` and `_K3` rows of the R2 tables above; their gains over F1 are in the verdict table.

## Second grid pool8x20 (supplementary, ridge, 300-clip subset; cannot produce PRESENT)

F1_8 / D3_8 use a 160-cell pooled PCA basis, 8 dims per cell; F1_4s / D3f_4s are the matched 4x10 references on the same clips.


Target v, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 517 | -1.309 +/- 1.500 |
| D3_8 | 3840 | 517 | +0.958 +/- 0.023 |
| F1_4s | 1280 | 517 | -1.099 +/- 1.183 |
| D3f_4s | 3840 | 517 | +0.960 +/- 0.017 |

Target v, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 980 | -0.289 +/- 0.463 |
| D3_8 | 3840 | 980 | +0.986 +/- 0.002 |
| F1_4s | 1280 | 980 | -0.291 +/- 0.307 |
| D3f_4s | 3840 | 980 | +0.987 +/- 0.003 |

Target dv, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 517 | -0.150 +/- 0.194 |
| D3_8 | 3840 | 517 | -0.010 +/- 0.019 |
| F1_4s | 1280 | 517 | -0.011 +/- 0.011 |
| D3f_4s | 3840 | 517 | -0.028 +/- 0.019 |

Target dv, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 980 | -0.138 +/- 0.178 |
| D3_8 | 3840 | 980 | +0.013 +/- 0.007 |
| F1_4s | 1280 | 980 | -0.002 +/- 0.007 |
| D3f_4s | 3840 | 980 | -0.006 +/- 0.010 |

v: 8x20 gain D3_8 - F1_8 = +2.267; 4x10 matched gain D3f_4s - F1_4s = +2.060 (band R2).

dv: 8x20 gain D3_8 - F1_8 = +0.140; 4x10 matched gain D3f_4s - F1_4s = -0.017 (band R2).

## Reading guide and limits

* The band is selected on v itself, so band R2 on v is harsh (small SST); read it with the RMSE table.
* The probe is a statement about THIS reduction (per-cell PCA to 32 dims) and THESE readouts (ridge, 2 x 256 MLP). It is a floor on what a trained adapter over the full 1024-d tokens could extract, not a ceiling.
* Alpha grid edges hit are recorded per cell in `raw/cells_ridge.json` (`alpha_edge`).
* Windows overlap within a clip (every 2nd frame); folds are by clip, so the SD reflects clip draws only.
* Code sha256 and PREREG sha256 are recorded in `raw/MOTIONPROBE.json`.
