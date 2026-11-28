# BOXFUSE-T0 verdict (card PREREG_V8F_PROOF_ADD130)

Tier: T0 (recorded frames, no rollout; a diagnostic of a perception read and not driving performance)

## Families

* LONGITUDINAL (distance keeping): the family this read serves; at T0 it is measured as the gap error only.
* LATERAL: not measured.
* TACTICAL: not measured.
* STRATEGIC: not computed, n = 0: a gap read on recorded frames reads no strategic quantity.

## Names


## Measure: mean absolute gap error on 3841 evaluation frames of 141 clips (m); gate factor 1.50

* arm D: D_0 3.960; D_1 3.913; D_2 4.112; mean 3.9954
* arm BW (box read, locator's pick (row 197)): 5.417; 5.287; 5.343; mean 5.3490; R = 1.3388, interval [1.1710, 1.5281]
* arm FG (gated geometric mean, locator's pick, vision only): 3.665; 3.521; 3.682; mean 3.6228; R = 0.9068, interval [0.8545, 0.9537]
* arm FM (geometric mean with no gate (beside)): 4.044; 3.955; 4.055; mean 4.0181; R = 1.0057, interval [0.9274, 1.0917]
* arm FL (gated geometric mean, label's pick (bound, not a candidate)): 3.438; 3.306; 3.444; mean 3.3957; R = 0.8499, interval [0.7950, 0.8971]
* FG divided by BW: 0.6773, interval [0.5944, 0.7673]
* interval: 2000 draws, seed 129; two-sided 95 % percentile, cluster bootstrap over the evaluation clips, fits held fixed

## Far leads (label gap 40 to 60 m): n 785 in 39 clips

* mean signed error (read minus label), mean of three fits: D -6.251, BW -1.882, FG -5.261, FM -4.514, FL -5.345
* paired difference FG minus D, interval [0.2204, 1.6164]

## Beside, naming nothing

* frames with both picks 3393; the locator's pick is the label's pick on 2991 (share 0.8815); error of the box gap on these 2.125 m (arm D 2.427 m), on the others 22.018 m (arm D 7.482 m)
* the gate passes on 0.7866 of the rows and 0.8785 of the picked frames; on 0.9399 of the frames whose pick is the label's and on 0.4652 of the others; on the gated frames FG 2.264 m, arm D 2.738 m
* by label gap, arm D: 0-10 m n 596 E 1.148 signed 0.443 gate 0.7612; 10-20 m n 1111 E 3.442 signed 2.139 gate 0.7873; 20-40 m n 1349 E 3.778 signed -1.174 gate 0.8026; 40-60 m n 785 E 7.314 signed -6.251 gate 0.7775
* by label gap, arm BW: 0-10 m n 596 E 1.295 signed 0.451 gate 0.7612; 10-20 m n 1111 E 4.501 signed 3.676 gate 0.7873; 20-40 m n 1349 E 6.251 signed 1.697 gate 0.8026; 40-60 m n 785 E 8.077 signed -1.882 gate 0.7775
* by label gap, arm FG: 0-10 m n 596 E 1.007 signed 0.544 gate 0.7612; 10-20 m n 1111 E 3.166 signed 2.205 gate 0.7873; 20-40 m n 1349 E 3.496 signed -0.919 gate 0.8026; 40-60 m n 785 E 6.473 signed -5.261 gate 0.7775
* by label gap, arm FL: 0-10 m n 596 E 0.992 signed 0.531 gate 0.7612; 10-20 m n 1111 E 3.087 signed 2.139 gate 0.7873; 20-40 m n 1349 E 3.145 signed -1.132 gate 0.8026; 40-60 m n 785 E 6.090 signed -5.345 gate 0.7775

## Limits of the card

Three fits of one locator on the same train clips, one detector, the same 141 evaluation clips read a third time; the interval per fit answers what another draw of clips would say with the fits held fixed.
