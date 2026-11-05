# RESULT - MOTIONPROBE: do cached frozen DINOv3-L token grids, stacked over frames, carry ego motion?

Evidence class: MEASURED (ours). Pre-registration: `PREREG_MOTIONPROBE.md` (sha256 in `FROZEN_SHA.txt`); numbers: `raw/MOTIONPROBE.json` (cells), `raw/cells_ridge.json`, `raw/cells_mlp.json`, `raw/cells_grid2.json`, `raw/VERDICT.json`. Tier: not a T1 driving number (T0-class diagnostic of the cached perception features; no driving model involved).

Estimator: fold mean +/- SD (ddof 1) over 5 folds BY CLIP. It answers "would another draw of CLIPS say this?" only. It does not answer "would another training run say this?" (ridge is deterministic; the MLP has one seed per fold) nor "would another inference run say this?" (nothing stochastic at inference). Controls K1/K2/K3 are the guard against reading a spurious gain as motion.

## Verdict (committed rule, PREREG section 9)

| target | verdict | s* | readout (s*) | readout (F1) | R2 band F1 | R2 band s* | gain g | gain K2 | gain K3 | paired gain by fold |
|---|---|---|---|---|---|---|---|---|---|---|
| v | **INVALID** | D3 | ridge | mlp | -0.694 | -0.652 | +0.042 | +0.020 | +0.027 | +0.122, +0.174, -0.164, -0.194, +0.271 |
| dv | **INVALID** | D3 | ridge | mlp | -0.024 | -0.003 | +0.021 | +0.021 | +0.023 | -0.010, +0.004, +0.070, +0.012, +0.031 |

* `v`: INVALID A K1 control left its known range: the pipeline has a defect and no verdict is issued.
* `dv`: INVALID A K1 control left its known range: the pipeline has a defect and no verdict is issued.

Sensitivity (cannot change the verdict): D3f (differences projected with the frame basis) best-readout band R2 minus F1 = +0.030 on v, +0.023 on dv; K2/K3 were not run for D3f.

K1 VIOLATIONS: [{"readout": "ridge", "fs": "F1mean", "target": "k1_v", "mean_r2_all": -1.2138555751840259}, {"readout": "ridge", "fs": "F1", "target": "k1_v", "mean_r2_all": -1.2347997239914712}, {"readout": "ridge", "fs": "F1", "target": "k1_dv", "mean_r2_all": -0.08673905158655773}, {"readout": "ridge", "fs": "F4", "target": "k1_v", "mean_r2_all": -3.7844300635242605}, {"readout": "ridge", "fs": "F4", "target": "k1_dv", "mean_r2_all": -0.11103955062943452}, {"readout": "ridge", "fs": "D3", "target": "k1_v", "mean_r2_all": -0.11696140529441008}, {"readout": "ridge", "fs": "D3f", "target": "k1_v", "mean_r2_all": -0.11927162747896407}, {"readout": "ridge", "fs": "F4_K2", "target": "k1_v", "mean_r2_all": -0.4221779232884181}, {"readout": "ridge", "fs": "D3_K2", "target": "k1_v", "mean_r2_all": -0.13911771250995647}, {"readout": "ridge", "fs": "F4_K3", "target": "k1_v", "mean_r2_all": -3.3341865645733235}, {"readout": "ridge", "fs": "F4_K3", "target": "k1_dv", "mean_r2_all": -0.13167407056617683}, {"readout": "ridge", "fs": "D3_K3", "target": "k1_v", "mean_r2_all": -0.12125785863329663}, {"readout": "mlp", "fs": "F4", "target": "k1_v", "mean_r2_all": -0.13430378562112238}]

## Sample

40 clips kept (0 dropped), clip-list sha256 `58e9c08ce3d4c9c34aa46feefabc8003cf45612f417d39c37258917ee24c811c`, 5 folds by clip, windows total 1960 (band 3-8 m/s: 1139). eval-141 disjointness asserted against `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/manifest.jsonl` (0 shared ids). pool8x20 subset: 20 clips.

PCA retained variance (mean over cells and folds): frame basis 0.990, difference basis 0.084, pooled 8x20 basis 0.027.

## R2, target v(t), current speed, band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1139 | -2.129 +/- 2.315 | -0.684 +/- 0.985 |
| F1 | 1280 | 1139 | -1.506 +/- 0.998 | -0.694 +/- 0.665 |
| F4 | 5120 | 1139 | -5.291 +/- 5.511 | -0.794 +/- 0.728 |
| D3 | 3840 | 1139 | -0.652 +/- 0.581 | -0.752 +/- 0.921 |
| D3f | 3840 | 1139 | -0.664 +/- 0.592 | n/a |
| F4_K2 | 5120 | 1139 | -2.209 +/- 2.563 | -0.659 +/- 0.599 |
| D3_K2 | 3840 | 1139 | -0.678 +/- 0.536 | -0.674 +/- 0.751 |
| F4_K3 | 5120 | 1139 | -4.691 +/- 4.405 | -0.763 +/- 0.731 |
| D3_K3 | 3840 | 1139 | -0.667 +/- 0.616 | -0.717 +/- 0.922 |

## R2, target v(t), current speed, all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1960 | -0.706 +/- 1.010 | -0.081 +/- 0.250 |
| F1 | 1280 | 1960 | -0.529 +/- 0.464 | -0.226 +/- 0.258 |
| F4 | 5120 | 1960 | -1.506 +/- 1.652 | -0.185 +/- 0.266 |
| D3 | 3840 | 1960 | -0.116 +/- 0.196 | -0.190 +/- 0.284 |
| D3f | 3840 | 1960 | -0.123 +/- 0.198 | n/a |
| F4_K2 | 5120 | 1960 | -0.382 +/- 0.377 | -0.186 +/- 0.199 |
| D3_K2 | 3840 | 1960 | -0.106 +/- 0.156 | -0.136 +/- 0.271 |
| F4_K3 | 5120 | 1960 | -1.482 +/- 1.352 | -0.190 +/- 0.276 |
| D3_K3 | 3840 | 1960 | -0.128 +/- 0.212 | -0.183 +/- 0.284 |

## R2, target dv = v(t)-v(t-3), band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1139 | -0.004 +/- 0.004 | -0.008 +/- 0.011 |
| F1 | 1280 | 1139 | -0.067 +/- 0.098 | -0.024 +/- 0.031 |
| F4 | 5120 | 1139 | -0.089 +/- 0.107 | -0.026 +/- 0.033 |
| D3 | 3840 | 1139 | -0.003 +/- 0.003 | -0.019 +/- 0.009 |
| D3f | 3840 | 1139 | -0.002 +/- 0.004 | n/a |
| F4_K2 | 5120 | 1139 | -0.006 +/- 0.046 | -0.003 +/- 0.027 |
| D3_K2 | 3840 | 1139 | -0.003 +/- 0.003 | -0.014 +/- 0.015 |
| F4_K3 | 5120 | 1139 | -0.133 +/- 0.179 | -0.028 +/- 0.028 |
| D3_K3 | 3840 | 1139 | -0.001 +/- 0.003 | -0.004 +/- 0.006 |

## R2, target dv = v(t)-v(t-3), all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1960 | -0.003 +/- 0.002 | -0.004 +/- 0.005 |
| F1 | 1280 | 1960 | -0.055 +/- 0.080 | -0.023 +/- 0.023 |
| F4 | 5120 | 1960 | -0.078 +/- 0.119 | -0.014 +/- 0.023 |
| D3 | 3840 | 1960 | -0.003 +/- 0.002 | -0.020 +/- 0.005 |
| D3f | 3840 | 1960 | -0.004 +/- 0.003 | n/a |
| F4_K2 | 5120 | 1960 | -0.018 +/- 0.024 | -0.009 +/- 0.022 |
| D3_K2 | 3840 | 1960 | -0.004 +/- 0.002 | -0.017 +/- 0.008 |
| F4_K3 | 5120 | 1960 | -0.098 +/- 0.130 | -0.019 +/- 0.024 |
| D3_K3 | 3840 | 1960 | -0.002 +/- 0.002 | -0.008 +/- 0.010 |

## RMSE (m/s), band 3-8 m/s, target v

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1139 | +2.47 +/- 1.07 | +1.86 +/- 0.62 |
| F1 | 1280 | 1139 | +2.29 +/- 0.58 | +1.88 +/- 0.47 |
| F4 | 5120 | 1139 | +3.45 +/- 1.71 | +1.94 +/- 0.49 |
| D3 | 3840 | 1139 | +1.86 +/- 0.42 | +1.90 +/- 0.58 |
| D3f | 3840 | 1139 | +1.87 +/- 0.42 | n/a |
| F4_K2 | 5120 | 1139 | +2.47 +/- 0.84 | +1.87 +/- 0.44 |
| D3_K2 | 3840 | 1139 | +1.88 +/- 0.39 | +1.86 +/- 0.50 |
| F4_K3 | 5120 | 1139 | +3.28 +/- 1.51 | +1.92 +/- 0.49 |
| D3_K3 | 3840 | 1139 | +1.87 +/- 0.44 | +1.88 +/- 0.58 |

## RMSE (m/s), band 3-8 m/s, target dv

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 1139 | +0.50 +/- 0.03 | +0.50 +/- 0.03 |
| F1 | 1280 | 1139 | +0.52 +/- 0.05 | +0.51 +/- 0.03 |
| F4 | 5120 | 1139 | +0.52 +/- 0.05 | +0.51 +/- 0.03 |
| D3 | 3840 | 1139 | +0.50 +/- 0.03 | +0.51 +/- 0.03 |
| D3f | 3840 | 1139 | +0.50 +/- 0.03 | n/a |
| F4_K2 | 5120 | 1139 | +0.50 +/- 0.04 | +0.50 +/- 0.03 |
| D3_K2 | 3840 | 1139 | +0.50 +/- 0.03 | +0.50 +/- 0.03 |
| F4_K3 | 5120 | 1139 | +0.53 +/- 0.05 | +0.51 +/- 0.03 |
| D3_K3 | 3840 | 1139 | +0.50 +/- 0.03 | +0.50 +/- 0.03 |

## Controls

K1 (train targets permuted; scored on true test targets; known value: all-windows R2 in [-0.05, 0.03]):

| readout | feature set | target | mean all-windows R2 |
|---|---|---|---|
| ridge | F1mean | k1_v | -1.214 |
| ridge | F1mean | k1_dv | -0.002 |
| ridge | F1 | k1_v | -1.235 |
| ridge | F1 | k1_dv | -0.087 |
| ridge | F4 | k1_v | -3.784 |
| ridge | F4 | k1_dv | -0.111 |
| ridge | D3 | k1_v | -0.117 |
| ridge | D3 | k1_dv | -0.002 |
| ridge | D3f | k1_v | -0.119 |
| ridge | D3f | k1_dv | -0.003 |
| ridge | F4_K2 | k1_v | -0.422 |
| ridge | F4_K2 | k1_dv | -0.012 |
| ridge | D3_K2 | k1_v | -0.139 |
| ridge | D3_K2 | k1_dv | -0.002 |
| ridge | F4_K3 | k1_v | -3.334 |
| ridge | F4_K3 | k1_dv | -0.132 |
| ridge | D3_K3 | k1_v | -0.121 |
| ridge | D3_K3 | k1_dv | -0.001 |
| mlp | F4 | k1_v | -0.134 |
| mlp | F4 | k1_dv | -0.007 |

K2 (frames t-3..t-1 from another clip of the same fold) and K3 (frame order shuffled per window) rows are the `_K2` and `_K3` rows of the R2 tables above; their gains over F1 are in the verdict table.

## Second grid pool8x20 (supplementary, ridge, 300-clip subset; cannot produce PRESENT)

F1_8 / D3_8 use a 160-cell pooled PCA basis, 8 dims per cell; F1_4s / D3f_4s are the matched 4x10 references on the same clips.


Target v, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 517 | -2.368 +/- 3.053 |
| D3_8 | 3840 | 517 | -1.077 +/- 1.412 |
| F1_4s | 1280 | 517 | -1.465 +/- 1.511 |
| D3f_4s | 3840 | 517 | -1.020 +/- 1.381 |

Target v, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 980 | -0.767 +/- 0.980 |
| D3_8 | 3840 | 980 | -0.367 +/- 0.403 |
| F1_4s | 1280 | 980 | -0.556 +/- 0.584 |
| D3f_4s | 3840 | 980 | -0.333 +/- 0.384 |

Target dv, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 517 | -0.010 +/- 0.019 |
| D3_8 | 3840 | 517 | -0.018 +/- 0.016 |
| F1_4s | 1280 | 517 | -0.055 +/- 0.081 |
| D3f_4s | 3840 | 517 | -0.023 +/- 0.014 |

Target dv, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 980 | -0.003 +/- 0.009 |
| D3_8 | 3840 | 980 | -0.006 +/- 0.010 |
| F1_4s | 1280 | 980 | -0.024 +/- 0.040 |
| D3f_4s | 3840 | 980 | -0.008 +/- 0.009 |

v: 8x20 gain D3_8 - F1_8 = +1.291; 4x10 matched gain D3f_4s - F1_4s = +0.445 (band R2).

dv: 8x20 gain D3_8 - F1_8 = -0.008; 4x10 matched gain D3f_4s - F1_4s = +0.032 (band R2).

## Reading guide and limits

* The band is selected on v itself, so band R2 on v is harsh (small SST); read it with the RMSE table.
* The probe is a statement about THIS reduction (per-cell PCA to 32 dims) and THESE readouts (ridge, 2 x 256 MLP). It is a floor on what a trained adapter over the full 1024-d tokens could extract, not a ceiling.
* Alpha grid edges hit are recorded per cell in `raw/cells_ridge.json` (`alpha_edge`).
* Windows overlap within a clip (every 2nd frame); folds are by clip, so the SD reflects clip draws only.
* Code sha256 and PREREG sha256 are recorded in `raw/MOTIONPROBE.json`.
