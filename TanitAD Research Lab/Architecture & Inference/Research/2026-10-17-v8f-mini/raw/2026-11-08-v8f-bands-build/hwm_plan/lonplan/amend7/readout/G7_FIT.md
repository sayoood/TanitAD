# RESULT_AMEND7_FIT (generated; the MM writes the reading)

Tier T0 (true-future latents and R35 rollouts under the logged actions, teacher-forced): a world-model diagnostic, never driving performance. Evidence MEASURED. Estimator: paired clip-cluster percentile bootstrap, B 2000, seed 0, over the 1,000 clips (answers another draw of CLIPS only). F_T answers another READOUT-TRAINING run; neither answers another training run of R35.

**Outcome: R7-FIT-FAIL.** Controls: {'ok': True, 'red': []}. Next lever: RY-shape readout ELIMINATED as a lever at N = 1,000 at the FIT stage (logged); next lever: ACTRSP Option B (card AMEND_OPTB.md), handed section 5.2

## G7-FIT (card section 4), per chassis

| chassis | (a) R2 part | (b) pass-contrast part | label | M_k | F_T(R2) | F_T(r_wm) |
|---|---|---|---|---|---|---|
| s0 | False | False | R7-FIT-FAIL | 0.0238 | 0.0090 | 0.0017 |
| s1 | False | False | R7-FIT-FAIL | 0.0246 | 0.0123 | 0.0050 |

## FIT rows, chassis R35_s0 (58,934 pairs, 1,000 clips; held-out, cross-fit; T0, MEASURED)

| quantity | point [95% CI] |
|---|---|
| r2_RA | +0.2808 [+0.2551, +0.3040] |
| r2_MLPA | +0.2813 [+0.2433, +0.3148] |
| r2_MLPB | +0.2903 [+0.2533, +0.3236] |
| dR2_A | +0.0005 [-0.0266, +0.0270] |
| dR2_B | +0.0095 [-0.0163, +0.0348] |
| r_wm_RA | +0.8549 [+0.8322, +0.8728] |
| r_wm_MLPA | +0.7890 [+0.7643, +0.8096] |
| r_wm_MLPB | +0.7907 [+0.7658, +0.8109] |
| dr_A | -0.0659 [-0.0773, -0.0550] |
| dr_B | -0.0643 [-0.0761, -0.0527] |
| r2_ZERO (STATE7) | +0.0472 [+0.0076, +0.0840] |
| r2_KCTL | +0.2808 [+0.2551, +0.3040] |
| r2_CONST | -0.0011 [-0.0028, -0.0005] |
| s_swap_RA | +0.2762 [+0.2548, +0.2972] |
| r_swap_RA | +0.4672 [+0.4379, +0.4954] |
| s_swap_MLPA | +0.2097 [+0.1889, +0.2295] |
| r_swap_MLPA | +0.4285 [+0.3977, +0.4572] |

## FIT rows, chassis R35_s1 (58,934 pairs, 1,000 clips; held-out, cross-fit; T0, MEASURED)

| quantity | point [95% CI] |
|---|---|
| r2_RA | +0.2788 [+0.2535, +0.3029] |
| r2_MLPA | +0.2684 [+0.2300, +0.3038] |
| r2_MLPB | +0.2807 [+0.2454, +0.3125] |
| dR2_A | -0.0104 [-0.0389, +0.0176] |
| dR2_B | +0.0019 [-0.0221, +0.0261] |
| r_wm_RA | +0.8687 [+0.8484, +0.8854] |
| r_wm_MLPA | +0.7985 [+0.7746, +0.8181] |
| r_wm_MLPB | +0.8035 [+0.7816, +0.8214] |
| dr_A | -0.0703 [-0.0806, -0.0602] |
| dr_B | -0.0653 [-0.0752, -0.0558] |
| r2_ZERO (STATE7) | +0.0610 [+0.0207, +0.0981] |
| r2_KCTL | +0.2788 [+0.2535, +0.3029] |
| r2_CONST | -0.0011 [-0.0028, -0.0005] |
| s_swap_RA | +0.2741 [+0.2516, +0.2955] |
| r_swap_RA | +0.4644 [+0.4351, +0.4929] |
| s_swap_MLPA | +0.2055 [+0.1857, +0.2258] |
| r_swap_MLPA | +0.4277 [+0.3974, +0.4568] |

## GAIN7, chassis s0 (beside, no bar; T0, MEASURED; paired clip-cluster percentile bootstrap, B 2000, seed 0, over the 1,000 clips (answers another draw of CLIPS only))

| readout | mean G_aa [CI] | q10 | q50 | q90 | share G < 0.25 | n anchors |
|---|---|---|---|---|---|---|
| RA5 | +1.3099 [+1.2625, +1.3550] | 0.169 | 1.196 | 2.617 | 0.121 | 5922 |
| RA6 | +1.1466 [+1.1057, +1.1857] | 0.179 | 1.041 | 2.269 | 0.123 | 5922 |
| RA7A | +1.3959 [+1.3416, +1.4467] | 0.257 | 1.090 | 3.051 | 0.097 | 5922 |
| RA5, 859 new clips (held out) | +1.3208 [+1.2703, +1.3752] | 0.170 | 1.211 | 2.622 | 0.119 | 5085 |

r_wm of RA5 on the 859 new clips: +0.8698 [+0.8563, +0.8824].

## GAIN7, chassis s1 (beside, no bar; T0, MEASURED; paired clip-cluster percentile bootstrap, B 2000, seed 0, over the 1,000 clips (answers another draw of CLIPS only))

| readout | mean G_aa [CI] | q10 | q50 | q90 | share G < 0.25 | n anchors |
|---|---|---|---|---|---|---|
| RA5 | +1.2914 [+1.2389, +1.3387] | 0.085 | 1.189 | 2.627 | 0.143 | 5922 |
| RA6 | +1.0537 [+1.0127, +1.0916] | 0.117 | 0.974 | 2.065 | 0.139 | 5922 |
| RA7A | +1.2833 [+1.2290, +1.3335] | 0.169 | 1.034 | 2.792 | 0.144 | 5922 |
| RA5, 859 new clips (held out) | +1.3061 [+1.2558, +1.3585] | 0.096 | 1.202 | 2.632 | 0.138 | 5085 |

r_wm of RA5 on the 859 new clips: +0.8713 [+0.8579, +0.8832].

## Four families (all four are named; the FIT stage defines only the longitudinal readout diagnostics above)

* LONGITUDINAL: FIT rows below are readout diagnostics of the longitudinal channel (T0, MEASURED); plan cells: NOT COMPUTABLE (n = 0; a readout fit produces no trajectory)
* LATERAL: NOT COMPUTABLE (n = 0; a readout fit produces no trajectory)
* TACTICAL: NOT COMPUTABLE (n = 0; a readout fit produces no trajectory)
* STRATEGIC: NOT COMPUTABLE (n = 0; a readout fit produces no trajectory); the item 25 pause ended by PI ruling 2026-11-16, so it is reported in full on the SCORE read
