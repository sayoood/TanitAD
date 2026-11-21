# RESULT - MOTIONPROBE: do cached frozen DINOv3-L token grids, stacked over frames, carry ego motion?

Evidence class: MEASURED (ours). Pre-registration: `PREREG_MOTIONPROBE.md` (sha256 in `FROZEN_SHA.txt`); numbers: `raw/MOTIONPROBE.json` (cells), `raw/cells_ridge.json`, `raw/cells_mlp.json`, `raw/cells_grid2.json`, `raw/VERDICT.json`. Tier: not a T1 driving number (T0-class diagnostic of the cached perception features; no driving model involved).

Estimator: fold mean +/- SD (ddof 1) over 5 folds BY CLIP. It answers "would another draw of CLIPS say this?" only. It does not answer "would another training run say this?" (ridge is deterministic; the MLP has one seed per fold) nor "would another inference run say this?" (nothing stochastic at inference). Controls K1/K2/K3 are the guard against reading a spurious gain as motion.

## Verdict (committed rule, PREREG section 9)

| target | verdict | s* | readout (s*) | readout (F1) | R2 band F1 | R2 band s* | gain g | gain K2 | gain K3 | paired gain by fold |
|---|---|---|---|---|---|---|---|---|---|---|
| v | **INVALID** | D3 | ridge | mlp | -0.911 | +0.980 | +1.891 | +0.046 | -0.306 | +2.345, +2.026, +1.505, +2.070, +1.509 |
| dv | **INVALID** | D3 | ridge | mlp | +0.009 | +0.175 | +0.167 | -0.010 | -0.011 | +0.235, +0.173, +0.090, +0.206, +0.131 |

* `v`: INVALID A K1 control left its known range: the pipeline has a defect and no verdict is issued.
* `dv`: INVALID A K1 control left its known range: the pipeline has a defect and no verdict is issued.

Sensitivity (cannot change the verdict): D3f (differences projected with the frame basis) best-readout band R2 minus F1 = +1.890 on v, +0.115 on dv; K2/K3 were not run for D3f.

Sensitivity F1d (frame t read through the difference basis; cannot change the verdict), best-readout band R2 gaps: v: F1d-F1 +0.101, D3-F1d +1.790, F4-F1d -0.017; dv: F1d-F1 -0.006, D3-F1d +0.173, F4-F1d +0.009. If D3's gain over F1 is matched by F1d over F1, the gain is basis content, not motion.

K1 VIOLATIONS: [{"readout": "ridge", "fs": "D3", "target": "k1_v", "mean_r2_all": -0.23889385695777277}, {"readout": "ridge", "fs": "D3", "target": "k1_dv", "mean_r2_all": -0.1505834142060219}, {"readout": "ridge", "fs": "D3f", "target": "k1_v", "mean_r2_all": -0.3190184289991468}, {"readout": "ridge", "fs": "D3f", "target": "k1_dv", "mean_r2_all": -0.2225314536123933}]

## Sample

200 clips kept (0 dropped), clip-list sha256 `cee0b615b4a99f6f55d305694facbdcc7534f44e7468789d4d3123e824becd53`, 5 folds by clip, windows total 9800 (band 3-8 m/s: 4702). eval-141 disjointness asserted against `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/manifest.jsonl` (0 shared ids). pool8x20 subset: 100 clips.

PCA retained variance (mean over cells and folds): frame basis 0.700, difference basis 0.077, pooled 8x20 basis 0.128.

## R2, target v(t), current speed, band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 4702 | -0.844 +/- 0.179 | -1.737 +/- 0.550 |
| F1 | 1280 | 4702 | -0.937 +/- 0.362 | -0.911 +/- 0.371 |
| F4 | 5120 | 4702 | -0.940 +/- 0.323 | -0.828 +/- 0.392 |
| D3 | 3840 | 4702 | +0.980 +/- 0.001 | +0.815 +/- 0.037 |
| D3f | 3840 | 4702 | +0.979 +/- 0.001 | n/a |
| F1d | 1280 | 4702 | -0.810 +/- 0.226 | -1.176 +/- 0.605 |
| F4_K2 | 5120 | 4702 | -0.904 +/- 0.442 | -0.900 +/- 0.595 |
| D3_K2 | 3840 | 4702 | -0.865 +/- 0.331 | -1.169 +/- 0.555 |
| F4_K3 | 5120 | 4702 | -0.941 +/- 0.323 | -0.816 +/- 0.490 |
| D3_K3 | 3840 | 4702 | -1.217 +/- 0.388 | -1.642 +/- 0.578 |

## R2, target v(t), current speed, all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 9800 | +0.238 +/- 0.077 | +0.073 +/- 0.071 |
| F1 | 1280 | 9800 | +0.130 +/- 0.029 | +0.043 +/- 0.064 |
| F4 | 5120 | 9800 | +0.114 +/- 0.030 | +0.116 +/- 0.086 |
| D3 | 3840 | 9800 | +0.996 +/- 0.001 | +0.958 +/- 0.012 |
| D3f | 3840 | 9800 | +0.996 +/- 0.001 | n/a |
| F1d | 1280 | 9800 | +0.230 +/- 0.055 | +0.057 +/- 0.058 |
| F4_K2 | 5120 | 9800 | +0.158 +/- 0.029 | +0.093 +/- 0.052 |
| D3_K2 | 3840 | 9800 | +0.333 +/- 0.085 | +0.192 +/- 0.085 |
| F4_K3 | 5120 | 9800 | +0.114 +/- 0.030 | +0.116 +/- 0.099 |
| D3_K3 | 3840 | 9800 | -0.029 +/- 0.019 | +0.297 +/- 0.055 |

## R2, target dv = v(t)-v(t-3), band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 4702 | +0.001 +/- 0.008 | -0.012 +/- 0.017 |
| F1 | 1280 | 4702 | +0.000 +/- 0.005 | +0.009 +/- 0.006 |
| F4 | 5120 | 4702 | +0.001 +/- 0.005 | +0.011 +/- 0.006 |
| D3 | 3840 | 4702 | +0.175 +/- 0.057 | -0.000 +/- 0.008 |
| D3f | 3840 | 4702 | +0.123 +/- 0.032 | n/a |
| F1d | 1280 | 4702 | +0.002 +/- 0.006 | -0.003 +/- 0.008 |
| F4_K2 | 5120 | 4702 | -0.000 +/- 0.004 | +0.002 +/- 0.006 |
| D3_K2 | 3840 | 4702 | -0.002 +/- 0.005 | -0.012 +/- 0.009 |
| F4_K3 | 5120 | 4702 | +0.001 +/- 0.005 | +0.010 +/- 0.007 |
| D3_K3 | 3840 | 4702 | -0.003 +/- 0.005 | -0.013 +/- 0.008 |

## R2, target dv = v(t)-v(t-3), all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 9800 | -0.002 +/- 0.004 | -0.015 +/- 0.014 |
| F1 | 1280 | 9800 | -0.000 +/- 0.001 | -0.002 +/- 0.008 |
| F4 | 5120 | 9800 | -0.000 +/- 0.002 | -0.003 +/- 0.006 |
| D3 | 3840 | 9800 | +0.159 +/- 0.037 | +0.010 +/- 0.015 |
| D3f | 3840 | 9800 | +0.125 +/- 0.019 | n/a |
| F1d | 1280 | 9800 | -0.001 +/- 0.004 | -0.010 +/- 0.014 |
| F4_K2 | 5120 | 9800 | -0.000 +/- 0.001 | -0.002 +/- 0.004 |
| D3_K2 | 3840 | 9800 | -0.001 +/- 0.001 | -0.012 +/- 0.014 |
| F4_K3 | 5120 | 9800 | -0.000 +/- 0.002 | -0.004 +/- 0.006 |
| D3_K3 | 3840 | 9800 | -0.001 +/- 0.001 | -0.009 +/- 0.009 |

## RMSE (m/s), band 3-8 m/s, target v

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 4702 | +1.96 +/- 0.10 | +2.39 +/- 0.25 |
| F1 | 1280 | 4702 | +2.01 +/- 0.18 | +1.99 +/- 0.19 |
| F4 | 5120 | 4702 | +2.01 +/- 0.16 | +1.95 +/- 0.19 |
| D3 | 3840 | 4702 | +0.20 +/- 0.00 | +0.62 +/- 0.06 |
| D3f | 3840 | 4702 | +0.21 +/- 0.01 | n/a |
| F1d | 1280 | 4702 | +1.95 +/- 0.12 | +2.12 +/- 0.28 |
| F4_K2 | 5120 | 4702 | +1.99 +/- 0.22 | +1.98 +/- 0.30 |
| D3_K2 | 3840 | 4702 | +1.97 +/- 0.15 | +2.12 +/- 0.24 |
| F4_K3 | 5120 | 4702 | +2.01 +/- 0.16 | +1.94 +/- 0.24 |
| D3_K3 | 3840 | 4702 | +2.15 +/- 0.18 | +2.34 +/- 0.26 |

## RMSE (m/s), band 3-8 m/s, target dv

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 4702 | +0.51 +/- 0.01 | +0.52 +/- 0.01 |
| F1 | 1280 | 4702 | +0.51 +/- 0.01 | +0.51 +/- 0.01 |
| F4 | 5120 | 4702 | +0.51 +/- 0.01 | +0.51 +/- 0.01 |
| D3 | 3840 | 4702 | +0.47 +/- 0.01 | +0.51 +/- 0.01 |
| D3f | 3840 | 4702 | +0.48 +/- 0.01 | n/a |
| F1d | 1280 | 4702 | +0.51 +/- 0.01 | +0.51 +/- 0.01 |
| F4_K2 | 5120 | 4702 | +0.51 +/- 0.01 | +0.51 +/- 0.01 |
| D3_K2 | 3840 | 4702 | +0.51 +/- 0.01 | +0.52 +/- 0.01 |
| F4_K3 | 5120 | 4702 | +0.51 +/- 0.01 | +0.51 +/- 0.01 |
| D3_K3 | 3840 | 4702 | +0.51 +/- 0.01 | +0.52 +/- 0.01 |

## Controls

K1 (train targets permuted; scored on true test targets; known value: all-windows R2 in [-0.05, 0.03]):

| readout | feature set | target | mean all-windows R2 |
|---|---|---|---|
| ridge | F1mean | k1_v | -0.043 |
| ridge | F1mean | k1_dv | -0.001 |
| ridge | F1 | k1_v | -0.034 |
| ridge | F1 | k1_dv | -0.000 |
| ridge | F4 | k1_v | -0.032 |
| ridge | F4 | k1_dv | -0.001 |
| ridge | D3 | k1_v | -0.239 |
| ridge | D3 | k1_dv | -0.151 |
| ridge | D3f | k1_v | -0.319 |
| ridge | D3f | k1_dv | -0.223 |
| ridge | F1d | k1_v | -0.043 |
| ridge | F1d | k1_dv | +0.000 |
| ridge | F4_K2 | k1_v | -0.033 |
| ridge | F4_K2 | k1_dv | -0.001 |
| ridge | D3_K2 | k1_v | -0.035 |
| ridge | D3_K2 | k1_dv | -0.001 |
| ridge | F4_K3 | k1_v | -0.032 |
| ridge | F4_K3 | k1_dv | -0.001 |
| ridge | D3_K3 | k1_v | -0.029 |
| ridge | D3_K3 | k1_dv | -0.001 |
| mlp | F4 | k1_v | -0.036 |
| mlp | F4 | k1_dv | -0.001 |

K2 (frames t-3..t-1 from another clip of the same fold) and K3 (frame order shuffled per window) rows are the `_K2` and `_K3` rows of the R2 tables above; their gains over F1 are in the verdict table.

## Second grid pool8x20 (supplementary, ridge, 300-clip subset; cannot produce PRESENT)

F1_8 / D3_8 use a 160-cell pooled PCA basis, 8 dims per cell; F1_4s / D3f_4s are the matched 4x10 references on the same clips.


Target v, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 2469 | -0.785 +/- 0.217 |
| D3_8 | 3840 | 2469 | +0.969 +/- 0.002 |
| F1_4s | 1280 | 2469 | -0.735 +/- 0.388 |
| D3f_4s | 3840 | 2469 | +0.974 +/- 0.002 |

Target v, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 4900 | +0.074 +/- 0.256 |
| D3_8 | 3840 | 4900 | +0.993 +/- 0.001 |
| F1_4s | 1280 | 4900 | +0.064 +/- 0.119 |
| D3f_4s | 3840 | 4900 | +0.994 +/- 0.001 |

Target dv, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 2469 | -0.001 +/- 0.007 |
| D3_8 | 3840 | 2469 | +0.061 +/- 0.011 |
| F1_4s | 1280 | 2469 | +0.003 +/- 0.007 |
| D3f_4s | 3840 | 2469 | +0.026 +/- 0.020 |

Target dv, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 4900 | -0.003 +/- 0.006 |
| D3_8 | 3840 | 4900 | +0.070 +/- 0.013 |
| F1_4s | 1280 | 4900 | -0.002 +/- 0.007 |
| D3f_4s | 3840 | 4900 | +0.042 +/- 0.015 |

v: 8x20 gain D3_8 - F1_8 = +1.754; 4x10 matched gain D3f_4s - F1_4s = +1.709 (band R2).

dv: 8x20 gain D3_8 - F1_8 = +0.062; 4x10 matched gain D3f_4s - F1_4s = +0.023 (band R2).

## Changes after the freeze (all made before any real-data fit)

* `F1d` (frame t projected with the difference basis, 1,280 dims) was added as a SENSITIVITY row for ridge and MLP; it is not in the decision rule.
* A plumbing check on synthetic tokens (planted displacement and null) lives in `selftest/`; it reads no real token.

## Reading guide and limits

* The band is selected on v itself, so band R2 on v is harsh (small SST); read it with the RMSE table.
* The probe is a statement about THIS reduction (per-cell PCA to 32 dims) and THESE readouts (ridge, 2 x 256 MLP). It is a floor on what a trained adapter over the full 1024-d tokens could extract, not a ceiling.
* Alpha grid edges hit are recorded per cell in `raw/cells_ridge.json` (`alpha_edge`).
* Windows overlap within a clip (every 2nd frame); folds are by clip, so the SD reflects clip draws only.
* Code sha256 and PREREG sha256 are recorded in `raw/MOTIONPROBE.json`.
