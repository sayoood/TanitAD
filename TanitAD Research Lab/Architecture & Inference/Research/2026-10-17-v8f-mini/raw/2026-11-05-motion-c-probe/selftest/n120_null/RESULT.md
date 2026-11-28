# RESULT - MOTIONPROBE: do cached frozen DINOv3-L token grids, stacked over frames, carry ego motion?

Evidence class: MEASURED (ours). Pre-registration: `PREREG_MOTIONPROBE.md` (sha256 in `FROZEN_SHA.txt`); numbers: `raw/MOTIONPROBE.json` (cells), `raw/cells_ridge.json`, `raw/cells_mlp.json`, `raw/cells_grid2.json`, `raw/VERDICT.json`. Tier: not a T1 driving number (T0-class diagnostic of the cached perception features; no driving model involved).

Estimator: fold mean +/- SD (ddof 1) over 5 folds BY CLIP. It answers "would another draw of CLIPS say this?" only. It does not answer "would another training run say this?" (ridge is deterministic; the MLP has one seed per fold) nor "would another inference run say this?" (nothing stochastic at inference). Controls K1/K2/K3 are the guard against reading a spurious gain as motion.

## Verdict (committed rule, PREREG section 9)

| target | verdict | s* | readout (s*) | readout (F1) | R2 band F1 | R2 band s* | gain g | gain K2 | gain K3 | paired gain by fold |
|---|---|---|---|---|---|---|---|---|---|---|
| v | **WEAK** | D3 | mlp | mlp | -1.188 | -1.131 | +0.057 | -0.044 | +0.136 | +0.402, +0.215, +0.295, -0.251, -0.375 |
| dv | **ABSENT** | F4 | ridge | ridge | -0.002 | -0.002 | +0.000 | -0.001 | +0.000 | +0.002, +0.001, +0.000, -0.000, -0.000 |

* `v`: WEAK Something is readable but it is not shown to be motion-borne, or it is below the committed bar. The cheapest discriminator is a stronger readout (unreduced 1024-d differences, temporal conv) to see whether the PCA bottleneck hides signal, or a pixel-difference CNN control on the same windows.
* `dv`: ABSENT Under per-cell PCA(32) plus ridge/MLP, stacking or differencing cached tokens adds less than 0.05 band R2 over frame t alone; motion has to come from pixels (a small CNN on frame differences, fused with the frozen tokens) or from encoder adaptation.

Noise context (post-hoc, informational, cannot change the verdict). The band is selected on v, so band R2 is noisy between folds and the best-of-readouts selection leans upward:

| target | paired band gain mean +/- SD over folds | all-windows R2 F1 | all-windows R2 s* | gain (all windows) | gain K2 (all) | gain K3 (all) |
|---|---|---|---|---|---|---|
| v | +0.057 +/- 0.347 | -0.015 | -0.012 | +0.004 | -0.002 | +0.004 |
| dv | +0.000 +/- 0.001 | -0.000 | -0.000 | +0.000 | -0.001 | +0.000 |

Sensitivity (cannot change the verdict): D3f (differences projected with the frame basis) best-readout band R2 minus F1 = -0.023 on v, -0.002 on dv; K2/K3 were not run for D3f.

Sensitivity F1d (frame t read through the difference basis; cannot change the verdict), best-readout band R2 gaps: v: F1d-F1 -0.024, D3-F1d +0.081, F4-F1d +0.038; dv: F1d-F1 -0.004, D3-F1d +0.002, F4-F1d +0.005. If D3's gain over F1 is matched by F1d over F1, the gain is basis content, not motion.

## Sample

120 clips kept (0 dropped), clip-list sha256 `337e4040dc2fcc2a83057741fb88b0e0c465a676c33a50e902778f30f678d638`, 5 folds by clip, windows total 5880 (band 3-8 m/s: 2833). eval-141 disjointness asserted against `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/manifest.jsonl` (0 shared ids). pool8x20 subset: 60 clips.

PCA retained variance (mean over cells and folds): frame basis 0.453, difference basis 0.058, pooled 8x20 basis 0.017.

## R2, target v(t), current speed, band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 2833 | -1.498 +/- 0.748 | -1.319 +/- 0.725 |
| F1 | 1280 | 2833 | -1.240 +/- 0.611 | -1.188 +/- 0.658 |
| F4 | 5120 | 2833 | -1.254 +/- 0.608 | -1.174 +/- 0.447 |
| D3 | 3840 | 2833 | -1.214 +/- 0.640 | -1.131 +/- 0.560 |
| D3f | 3840 | 2833 | -1.211 +/- 0.642 | n/a |
| F1d | 1280 | 2833 | -1.212 +/- 0.649 | -1.352 +/- 0.715 |
| F4_K2 | 5120 | 2833 | -1.226 +/- 0.634 | -1.196 +/- 0.464 |
| D3_K2 | 3840 | 2833 | -1.232 +/- 0.702 | -1.512 +/- 0.838 |
| F4_K3 | 5120 | 2833 | -1.254 +/- 0.608 | -1.192 +/- 0.461 |
| D3_K3 | 3840 | 2833 | -1.212 +/- 0.642 | -1.052 +/- 0.516 |

## R2, target v(t), current speed, all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 5880 | -0.010 +/- 0.037 | +0.003 +/- 0.029 |
| F1 | 1280 | 5880 | -0.022 +/- 0.039 | -0.015 +/- 0.020 |
| F4 | 5120 | 5880 | -0.029 +/- 0.042 | -0.014 +/- 0.020 |
| D3 | 3840 | 5880 | -0.012 +/- 0.010 | -0.023 +/- 0.019 |
| D3f | 3840 | 5880 | -0.011 +/- 0.010 | n/a |
| F1d | 1280 | 5880 | -0.013 +/- 0.012 | -0.003 +/- 0.039 |
| F4_K2 | 5120 | 5880 | -0.018 +/- 0.013 | -0.017 +/- 0.011 |
| D3_K2 | 3840 | 5880 | -0.018 +/- 0.031 | -0.058 +/- 0.069 |
| F4_K3 | 5120 | 5880 | -0.029 +/- 0.042 | -0.014 +/- 0.020 |
| D3_K3 | 3840 | 5880 | -0.012 +/- 0.010 | -0.016 +/- 0.007 |

## R2, target dv = v(t)-v(t-3), band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 2833 | -0.009 +/- 0.008 | -0.004 +/- 0.007 |
| F1 | 1280 | 2833 | -0.002 +/- 0.005 | -0.006 +/- 0.008 |
| F4 | 5120 | 2833 | -0.002 +/- 0.005 | -0.005 +/- 0.003 |
| D3 | 3840 | 2833 | -0.004 +/- 0.006 | -0.022 +/- 0.013 |
| D3f | 3840 | 2833 | -0.004 +/- 0.006 | n/a |
| F1d | 1280 | 2833 | -0.007 +/- 0.005 | -0.007 +/- 0.007 |
| F4_K2 | 5120 | 2833 | -0.004 +/- 0.006 | -0.005 +/- 0.003 |
| D3_K2 | 3840 | 2833 | -0.004 +/- 0.006 | -0.003 +/- 0.010 |
| F4_K3 | 5120 | 2833 | -0.002 +/- 0.005 | -0.006 +/- 0.003 |
| D3_K3 | 3840 | 2833 | -0.005 +/- 0.006 | -0.015 +/- 0.015 |

## R2, target dv = v(t)-v(t-3), all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 5880 | -0.004 +/- 0.006 | -0.002 +/- 0.003 |
| F1 | 1280 | 5880 | -0.000 +/- 0.003 | -0.005 +/- 0.006 |
| F4 | 5120 | 5880 | -0.000 +/- 0.003 | -0.003 +/- 0.004 |
| D3 | 3840 | 5880 | -0.002 +/- 0.002 | -0.019 +/- 0.004 |
| D3f | 3840 | 5880 | -0.002 +/- 0.002 | n/a |
| F1d | 1280 | 5880 | -0.002 +/- 0.003 | -0.006 +/- 0.008 |
| F4_K2 | 5120 | 5880 | -0.001 +/- 0.002 | -0.003 +/- 0.004 |
| D3_K2 | 3840 | 5880 | -0.001 +/- 0.003 | -0.003 +/- 0.002 |
| F4_K3 | 5120 | 5880 | -0.000 +/- 0.003 | -0.003 +/- 0.003 |
| D3_K3 | 3840 | 5880 | -0.002 +/- 0.003 | -0.016 +/- 0.013 |

## RMSE (m/s), band 3-8 m/s, target v

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 2833 | +2.21 +/- 0.32 | +2.14 +/- 0.35 |
| F1 | 1280 | 2833 | +2.10 +/- 0.26 | +2.07 +/- 0.27 |
| F4 | 5120 | 2833 | +2.11 +/- 0.26 | +2.08 +/- 0.22 |
| D3 | 3840 | 2833 | +2.09 +/- 0.28 | +2.05 +/- 0.22 |
| D3f | 3840 | 2833 | +2.09 +/- 0.28 | n/a |
| F1d | 1280 | 2833 | +2.09 +/- 0.28 | +2.15 +/- 0.28 |
| F4_K2 | 5120 | 2833 | +2.10 +/- 0.27 | +2.09 +/- 0.23 |
| D3_K2 | 3840 | 2833 | +2.09 +/- 0.28 | +2.22 +/- 0.31 |
| F4_K3 | 5120 | 2833 | +2.11 +/- 0.26 | +2.09 +/- 0.22 |
| D3_K3 | 3840 | 2833 | +2.09 +/- 0.28 | +2.01 +/- 0.21 |

## RMSE (m/s), band 3-8 m/s, target dv

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 2833 | +0.51 +/- 0.02 | +0.51 +/- 0.02 |
| F1 | 1280 | 2833 | +0.51 +/- 0.02 | +0.51 +/- 0.02 |
| F4 | 5120 | 2833 | +0.51 +/- 0.02 | +0.51 +/- 0.02 |
| D3 | 3840 | 2833 | +0.51 +/- 0.02 | +0.52 +/- 0.02 |
| D3f | 3840 | 2833 | +0.51 +/- 0.02 | n/a |
| F1d | 1280 | 2833 | +0.51 +/- 0.02 | +0.51 +/- 0.02 |
| F4_K2 | 5120 | 2833 | +0.51 +/- 0.02 | +0.51 +/- 0.02 |
| D3_K2 | 3840 | 2833 | +0.51 +/- 0.02 | +0.51 +/- 0.02 |
| F4_K3 | 5120 | 2833 | +0.51 +/- 0.02 | +0.51 +/- 0.02 |
| D3_K3 | 3840 | 2833 | +0.51 +/- 0.02 | +0.52 +/- 0.02 |

## Controls

K1 (amendment 1: (v, dv) permuted once across ALL windows before the split, standard pipeline incl. its own alpha choice, scored on the permuted test targets; known value: all-windows R2 in [-0.05, 0.03]):

| readout | feature set | target | mean all-windows R2 |
|---|---|---|---|
| ridge | F1mean | k1_v | -0.001 |
| ridge | F1mean | k1_dv | -0.001 |
| ridge | F1 | k1_v | -0.000 |
| ridge | F1 | k1_dv | -0.001 |
| ridge | F4 | k1_v | -0.000 |
| ridge | F4 | k1_dv | -0.000 |
| ridge | D3 | k1_v | -0.001 |
| ridge | D3 | k1_dv | -0.001 |
| ridge | D3f | k1_v | -0.001 |
| ridge | D3f | k1_dv | -0.001 |
| ridge | F1d | k1_v | -0.002 |
| ridge | F1d | k1_dv | -0.003 |
| ridge | F4_K2 | k1_v | -0.000 |
| ridge | F4_K2 | k1_dv | -0.001 |
| ridge | D3_K2 | k1_v | -0.000 |
| ridge | D3_K2 | k1_dv | -0.001 |
| ridge | F4_K3 | k1_v | -0.000 |
| ridge | F4_K3 | k1_dv | -0.000 |
| ridge | D3_K3 | k1_v | -0.001 |
| ridge | D3_K3 | k1_dv | -0.001 |
| mlp | F4 | k1_v | -0.001 |
| mlp | F4 | k1_dv | -0.002 |

K2 (frames t-3..t-1 from another clip of the same fold) and K3 (frame order shuffled per window) rows are the `_K2` and `_K3` rows of the R2 tables above; their gains over F1 are in the verdict table.

## Second grid pool8x20 (supplementary, ridge, 300-clip subset; cannot produce PRESENT)

F1_8 / D3_8 use a 160-cell pooled PCA basis, 8 dims per cell; F1_4s / D3f_4s are the matched 4x10 references on the same clips.


Target v, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 1432 | -3.307 +/- 3.330 |
| D3_8 | 3840 | 1432 | -0.946 +/- 0.607 |
| F1_4s | 1280 | 1432 | -0.905 +/- 0.536 |
| D3f_4s | 3840 | 1432 | -0.982 +/- 0.605 |

Target v, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 2940 | -0.799 +/- 1.417 |
| D3_8 | 3840 | 2940 | -0.073 +/- 0.086 |
| F1_4s | 1280 | 2940 | -0.074 +/- 0.090 |
| D3f_4s | 3840 | 2940 | -0.076 +/- 0.087 |

Target dv, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 1432 | -0.017 +/- 0.018 |
| D3_8 | 3840 | 1432 | -0.015 +/- 0.020 |
| F1_4s | 1280 | 1432 | -0.015 +/- 0.019 |
| D3f_4s | 3840 | 1432 | -0.018 +/- 0.017 |

Target dv, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 2940 | -0.006 +/- 0.007 |
| D3_8 | 3840 | 2940 | -0.004 +/- 0.006 |
| F1_4s | 1280 | 2940 | -0.004 +/- 0.006 |
| D3f_4s | 3840 | 2940 | -0.006 +/- 0.006 |

v: 8x20 gain D3_8 - F1_8 = +2.361; 4x10 matched gain D3f_4s - F1_4s = -0.077 (band R2).

dv: 8x20 gain D3_8 - F1_8 = +0.001; 4x10 matched gain D3f_4s - F1_4s = -0.003 (band R2).

## Changes after the freeze (all made before any real-data fit)

* PREREG amendment 2 (K2 'vanishes' operationalised as gK2 <= max(0.02, 0.10 g) instead of gK2 <= 0.02): in the synthetic plumbing check a planted gain of +2.27 band R2 was labelled WEAK because a no-motion control sat +0.074 above F1, which is the scale of the band-R2 noise between two no-signal models (their SD across folds was 0.3-0.6). Initial texts: `PREREG_MOTIONPROBE_v0.md`, `PREREG_MOTIONPROBE_v1.md`.
* PREREG amendment 1 (K1 redefined, original text kept in `PREREG_MOTIONPROBE_v0.md`; see `FROZEN_SHA.txt`): the synthetic plumbing check showed that a K1 that reuses the real target's alpha reads strongly negative whenever the real signal is strong (low alpha overfits permuted noise), which is an overfit, not a pipeline defect. The amended K1 permutes the labels globally before the split and lets every column choose its own alpha.
* `F1d` (frame t projected with the difference basis, 1,280 dims) was added as a SENSITIVITY row for ridge and MLP; it is not in the decision rule.
* A plumbing check on synthetic tokens (planted displacement and null) lives in `selftest/`; it reads no real token.

## Reading guide and limits

* The band is selected on v itself, so band R2 on v is harsh (small SST); read it with the RMSE table.
* The probe is a statement about THIS reduction (per-cell PCA to 32 dims) and THESE readouts (ridge, 2 x 256 MLP). It is a floor on what a trained adapter over the full 1024-d tokens could extract, not a ceiling.
* Alpha grid edges hit are recorded per cell in `raw/cells_ridge.json` (`alpha_edge`).
* Windows overlap within a clip (every 2nd frame); folds are by clip, so the SD reflects clip draws only.
* Code sha256 and PREREG sha256 are recorded in `raw/MOTIONPROBE.json`.
