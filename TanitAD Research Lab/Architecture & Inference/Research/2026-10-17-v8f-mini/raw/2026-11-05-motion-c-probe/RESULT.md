# RESULT - MOTIONPROBE: do cached frozen DINOv3-L token grids, stacked over frames, carry ego motion?

Evidence class: MEASURED (ours). Pre-registration: `PREREG_MOTIONPROBE.md` (sha256 in `FROZEN_SHA.txt`); numbers: `raw/MOTIONPROBE.json` (cells), `raw/cells_ridge.json`, `raw/cells_mlp.json`, `raw/cells_grid2.json`, `raw/VERDICT.json`. Tier: not a T1 driving number (T0-class diagnostic of the cached perception features; no driving model involved).

Estimator: fold mean +/- SD (ddof 1) over 5 folds BY CLIP. It answers "would another draw of CLIPS say this?" only. It does not answer "would another training run say this?" (ridge is deterministic; the MLP has one seed per fold) nor "would another inference run say this?" (nothing stochastic at inference). Controls K1/K2/K3 are the guard against reading a spurious gain as motion.

## Verdict (committed rule, PREREG section 9)

| target | verdict | s* | readout (s*) | readout (F1) | R2 band F1 | R2 band s* | gain g | gain K2 | gain K3 | paired gain by fold |
|---|---|---|---|---|---|---|---|---|---|---|
| v | **WEAK** | F4 | mlp | mlp | -3.046 | -2.714 | +0.332 | -0.476 | +0.339 | +0.230, +0.086, +0.693, +0.080, +0.571 |
| dv | **ABSENT** | F4 | ridge | ridge | +0.080 | +0.123 | +0.043 | -0.065 | +0.009 | +0.027, +0.058, +0.037, +0.071, +0.025 |

* `v`: WEAK Something is readable but it is not shown to be motion-borne, or it is below the committed bar. The cheapest discriminator is a stronger readout (unreduced 1024-d differences, temporal conv) to see whether the PCA bottleneck hides signal, or a pixel-difference CNN control on the same windows.
* `dv`: ABSENT Under per-cell PCA(32) plus ridge/MLP, stacking or differencing cached tokens adds less than 0.05 band R2 over frame t alone; motion has to come from pixels (a small CNN on frame differences, fused with the frozen tokens) or from encoder adaptation.

Noise context (post-hoc, informational, cannot change the verdict). The band is selected on v, so band R2 is noisy between folds and the best-of-readouts selection leans upward:

| target | paired band gain mean +/- SD over folds | all-windows R2 F1 | all-windows R2 s* | gain (all windows) | gain K2 (all) | gain K3 (all) |
|---|---|---|---|---|---|---|
| v | +0.332 +/- 0.284 | +0.767 | +0.779 | +0.012 | -0.025 | +0.008 |
| dv | +0.043 +/- 0.020 | +0.042 | +0.065 | +0.023 | -0.037 | +0.003 |

Sensitivity (cannot change the verdict): D3f (differences projected with the frame basis) best-readout band R2 minus F1 = -8.434 on v, +0.054 on dv; K2/K3 were not run for D3f.

Sensitivity F1d (frame t read through the difference basis; cannot change the verdict), best-readout band R2 gaps: v: F1d-F1 -0.406, D3-F1d -2.687, F4-F1d +0.738; dv: F1d-F1 -0.004, D3-F1d +0.040, F4-F1d +0.048. If D3's gain over F1 is matched by F1d over F1, the gain is basis content, not motion.

## Sample

600 clips kept (0 dropped), clip-list sha256 `cbbe333de48eb98ab08a8fe0fbea933893ab375bdc38d18f3712c91a46fca3b8`, 5 folds by clip, windows total 29406 (band 3-8 m/s: 7772). eval-141 disjointness asserted against `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/manifest.jsonl` (0 shared ids). pool8x20 subset: 300 clips.

PCA retained variance (mean over cells and folds): frame basis 0.712, difference basis 0.465, pooled 8x20 basis 0.465.

## R2, target v(t), current speed, band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 7772 | -2.802 +/- 0.314 | -2.681 +/- 0.786 |
| F1 | 1280 | 7772 | -3.539 +/- 0.435 | -3.046 +/- 0.291 |
| F4 | 5120 | 7772 | -3.384 +/- 0.421 | -2.714 +/- 0.340 |
| D3 | 3840 | 7772 | -11.477 +/- 1.314 | -6.139 +/- 0.601 |
| D3f | 3840 | 7772 | -11.480 +/- 1.319 | n/a |
| F1d | 1280 | 7772 | -3.830 +/- 0.738 | -3.452 +/- 0.806 |
| F4_K2 | 5120 | 7772 | -3.868 +/- 0.551 | -3.522 +/- 0.676 |
| D3_K2 | 3840 | 7772 | -9.366 +/- 1.233 | -6.870 +/- 1.232 |
| F4_K3 | 5120 | 7772 | -3.410 +/- 0.430 | -2.706 +/- 0.457 |
| D3_K3 | 3840 | 7772 | -11.495 +/- 1.342 | -10.254 +/- 2.464 |

## R2, target v(t), current speed, all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 29406 | +0.793 +/- 0.020 | +0.768 +/- 0.017 |
| F1 | 1280 | 29406 | +0.767 +/- 0.031 | +0.765 +/- 0.017 |
| F4 | 5120 | 29406 | +0.776 +/- 0.031 | +0.779 +/- 0.017 |
| D3 | 3840 | 29406 | -0.016 +/- 0.020 | +0.444 +/- 0.096 |
| D3f | 3840 | 29406 | -0.016 +/- 0.021 | n/a |
| F1d | 1280 | 29406 | +0.754 +/- 0.025 | +0.754 +/- 0.023 |
| F4_K2 | 5120 | 29406 | +0.742 +/- 0.026 | +0.712 +/- 0.021 |
| D3_K2 | 3840 | 29406 | +0.306 +/- 0.040 | +0.495 +/- 0.040 |
| F4_K3 | 5120 | 29406 | +0.775 +/- 0.032 | +0.769 +/- 0.021 |
| D3_K3 | 3840 | 29406 | -0.017 +/- 0.021 | +0.209 +/- 0.034 |

## R2, target dv = v(t)-v(t-3), band 3-8 m/s (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 7772 | +0.058 +/- 0.012 | +0.042 +/- 0.030 |
| F1 | 1280 | 7772 | +0.080 +/- 0.030 | +0.066 +/- 0.040 |
| F4 | 5120 | 7772 | +0.123 +/- 0.040 | +0.100 +/- 0.047 |
| D3 | 3840 | 7772 | +0.115 +/- 0.012 | +0.083 +/- 0.020 |
| D3f | 3840 | 7772 | +0.134 +/- 0.012 | n/a |
| F1d | 1280 | 7772 | +0.075 +/- 0.034 | +0.051 +/- 0.053 |
| F4_K2 | 5120 | 7772 | +0.015 +/- 0.010 | -0.003 +/- 0.042 |
| D3_K2 | 3840 | 7772 | +0.025 +/- 0.020 | -0.003 +/- 0.026 |
| F4_K3 | 5120 | 7772 | +0.087 +/- 0.029 | +0.089 +/- 0.052 |
| D3_K3 | 3840 | 7772 | -0.002 +/- 0.002 | -0.019 +/- 0.010 |

## R2, target dv = v(t)-v(t-3), all windows (mean +/- SD over folds)

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 29406 | +0.029 +/- 0.014 | -0.035 +/- 0.025 |
| F1 | 1280 | 29406 | +0.042 +/- 0.026 | -0.021 +/- 0.024 |
| F4 | 5120 | 29406 | +0.065 +/- 0.025 | +0.002 +/- 0.045 |
| D3 | 3840 | 29406 | +0.066 +/- 0.014 | +0.007 +/- 0.019 |
| D3f | 3840 | 29406 | +0.082 +/- 0.013 | n/a |
| F1d | 1280 | 29406 | +0.037 +/- 0.027 | -0.051 +/- 0.037 |
| F4_K2 | 5120 | 29406 | +0.006 +/- 0.004 | -0.053 +/- 0.036 |
| D3_K2 | 3840 | 29406 | +0.012 +/- 0.012 | -0.053 +/- 0.035 |
| F4_K3 | 5120 | 29406 | +0.045 +/- 0.027 | -0.015 +/- 0.047 |
| D3_K3 | 3840 | 29406 | -0.001 +/- 0.001 | -0.041 +/- 0.019 |

## RMSE (m/s), band 3-8 m/s, target v

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 7772 | +2.77 +/- 0.16 | +2.71 +/- 0.21 |
| F1 | 1280 | 7772 | +3.03 +/- 0.22 | +2.85 +/- 0.06 |
| F4 | 5120 | 7772 | +2.97 +/- 0.20 | +2.73 +/- 0.15 |
| D3 | 3840 | 7772 | +5.01 +/- 0.29 | +3.79 +/- 0.24 |
| D3f | 3840 | 7772 | +5.01 +/- 0.29 | n/a |
| F1d | 1280 | 7772 | +3.12 +/- 0.25 | +2.99 +/- 0.31 |
| F4_K2 | 5120 | 7772 | +3.13 +/- 0.09 | +3.01 +/- 0.09 |
| D3_K2 | 3840 | 7772 | +4.57 +/- 0.31 | +3.97 +/- 0.33 |
| F4_K3 | 5120 | 7772 | +2.98 +/- 0.21 | +2.73 +/- 0.14 |
| D3_K3 | 3840 | 7772 | +5.01 +/- 0.29 | +4.73 +/- 0.33 |

## RMSE (m/s), band 3-8 m/s, target dv

| feature set | dim | n windows (all folds) | ridge | mlp |
|---|---|---|---|---|
| F1mean | 1024 | 7772 | +0.56 +/- 0.02 | +0.57 +/- 0.02 |
| F1 | 1280 | 7772 | +0.56 +/- 0.02 | +0.56 +/- 0.02 |
| F4 | 5120 | 7772 | +0.54 +/- 0.02 | +0.55 +/- 0.02 |
| D3 | 3840 | 7772 | +0.55 +/- 0.02 | +0.56 +/- 0.03 |
| D3f | 3840 | 7772 | +0.54 +/- 0.02 | n/a |
| F1d | 1280 | 7772 | +0.56 +/- 0.02 | +0.56 +/- 0.02 |
| F4_K2 | 5120 | 7772 | +0.58 +/- 0.03 | +0.58 +/- 0.02 |
| D3_K2 | 3840 | 7772 | +0.57 +/- 0.02 | +0.58 +/- 0.02 |
| F4_K3 | 5120 | 7772 | +0.55 +/- 0.02 | +0.55 +/- 0.02 |
| D3_K3 | 3840 | 7772 | +0.58 +/- 0.03 | +0.59 +/- 0.02 |

## Controls

K1 (amendment 1: (v, dv) permuted once across ALL windows before the split, standard pipeline incl. its own alpha choice, scored on the permuted test targets; known value: all-windows R2 in [-0.05, 0.03]):

| readout | feature set | target | mean all-windows R2 |
|---|---|---|---|
| ridge | F1mean | k1_v | -0.000 |
| ridge | F1mean | k1_dv | -0.001 |
| ridge | F1 | k1_v | -0.000 |
| ridge | F1 | k1_dv | -0.001 |
| ridge | F4 | k1_v | -0.000 |
| ridge | F4 | k1_dv | -0.001 |
| ridge | D3 | k1_v | -0.000 |
| ridge | D3 | k1_dv | -0.000 |
| ridge | D3f | k1_v | -0.000 |
| ridge | D3f | k1_dv | -0.000 |
| ridge | F1d | k1_v | -0.000 |
| ridge | F1d | k1_dv | -0.001 |
| ridge | F4_K2 | k1_v | -0.000 |
| ridge | F4_K2 | k1_dv | -0.001 |
| ridge | D3_K2 | k1_v | -0.000 |
| ridge | D3_K2 | k1_dv | -0.000 |
| ridge | F4_K3 | k1_v | -0.000 |
| ridge | F4_K3 | k1_dv | -0.001 |
| ridge | D3_K3 | k1_v | -0.000 |
| ridge | D3_K3 | k1_dv | -0.000 |
| mlp | F4 | k1_v | -0.009 |
| mlp | F4 | k1_dv | -0.008 |

K2 (frames t-3..t-1 from another clip of the same fold) and K3 (frame order shuffled per window) rows are the `_K2` and `_K3` rows of the R2 tables above; their gains over F1 are in the verdict table.

## Second grid pool8x20 (supplementary, ridge, 300-clip subset; cannot produce PRESENT)

F1_8 / D3_8 use a 160-cell pooled PCA basis, 8 dims per cell; F1_4s / D3f_4s are the matched 4x10 references on the same clips.


Target v, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 3858 | -5.482 +/- 1.664 |
| D3_8 | 3840 | 3858 | -12.560 +/- 1.437 |
| F1_4s | 1280 | 3858 | -3.571 +/- 1.016 |
| D3f_4s | 3840 | 3858 | -12.477 +/- 1.379 |

Target v, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 14700 | +0.650 +/- 0.013 |
| D3_8 | 3840 | 14700 | -0.013 +/- 0.020 |
| F1_4s | 1280 | 14700 | +0.750 +/- 0.021 |
| D3f_4s | 3840 | 14700 | -0.013 +/- 0.020 |

Target dv, band R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 3858 | +0.037 +/- 0.025 |
| D3_8 | 3840 | 3858 | +0.081 +/- 0.014 |
| F1_4s | 1280 | 3858 | +0.049 +/- 0.024 |
| D3f_4s | 3840 | 3858 | +0.102 +/- 0.008 |

Target dv, all-windows R2:

| feature set | dim | n windows (all folds) | ridge |
|---|---|---|---|
| F1_8 | 1280 | 14700 | +0.016 +/- 0.029 |
| D3_8 | 3840 | 14700 | +0.046 +/- 0.016 |
| F1_4s | 1280 | 14700 | +0.023 +/- 0.014 |
| D3f_4s | 3840 | 14700 | +0.064 +/- 0.017 |

v: 8x20 gain D3_8 - F1_8 = -7.079; 4x10 matched gain D3f_4s - F1_4s = -8.905 (band R2).

dv: 8x20 gain D3_8 - F1_8 = +0.044; 4x10 matched gain D3f_4s - F1_4s = +0.054 (band R2).

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
