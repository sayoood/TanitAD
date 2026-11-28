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
* arm BW (box read, locator's pick (row 197)): 5.394; 5.292; 5.360; mean 5.3486; R = 1.3387, interval [1.1713, 1.5276]
* arm FG (gated geometric mean, locator's pick, vision only): 3.626; 3.508; 3.662; mean 3.5987; R = 0.9007, interval [0.8509, 0.9456]
* arm FM (geometric mean with no gate (beside)): 3.975; 3.889; 4.014; mean 3.9593; R = 0.9910, interval [0.9106, 1.0837]
* arm FL (gated geometric mean, label's pick (bound, not a candidate)): 3.438; 3.306; 3.444; mean 3.3957; R = 0.8499, interval [0.7950, 0.8971]
* FG divided by BW: 0.6728, interval [0.5951, 0.7615]
* interval: 2000 draws, seed 129; two-sided 95 % percentile, cluster bootstrap over the evaluation clips, fits held fixed

## Far leads (label gap 40 to 60 m): n 785 in 39 clips

* mean signed error (read minus label), mean of three fits: D -6.251, BW -1.654, FG -5.129, FM -4.434, FL -5.345
* paired difference FG minus D, interval [0.3466, 1.7090]

## Beside, naming nothing

* frames with both picks 3412; the locator's pick is the label's pick on 2966 (share 0.8693); error of the box gap on these 2.109 m (arm D 2.483 m), on the others 20.649 m (arm D 6.967 m)
* the gate passes on 0.7823 of the rows and 0.8669 of the picked frames; on 0.9341 of the frames whose pick is the label's and on 0.4611 of the others; on the gated frames FG 2.207 m, arm D 2.714 m
* by label gap, arm D: 0-10 m n 596 E 1.148 signed 0.443 gate 0.7662; 10-20 m n 1111 E 3.442 signed 2.139 gate 0.7891; 20-40 m n 1349 E 3.778 signed -1.174 gate 0.7964; 40-60 m n 785 E 7.314 signed -6.251 gate 0.7605
* by label gap, arm BW: 0-10 m n 596 E 1.335 signed 0.491 gate 0.7662; 10-20 m n 1111 E 4.455 signed 3.612 gate 0.7891; 20-40 m n 1349 E 5.846 signed 1.251 gate 0.7964; 40-60 m n 785 E 8.805 signed -1.654 gate 0.7605
* by label gap, arm FG: 0-10 m n 596 E 0.996 signed 0.534 gate 0.7662; 10-20 m n 1111 E 3.138 signed 2.180 gate 0.7891; 20-40 m n 1349 E 3.443 signed -0.926 gate 0.7964; 40-60 m n 785 E 6.494 signed -5.129 gate 0.7605
* by label gap, arm FL: 0-10 m n 596 E 0.992 signed 0.531 gate 0.7662; 10-20 m n 1111 E 3.087 signed 2.139 gate 0.7891; 20-40 m n 1349 E 3.145 signed -1.132 gate 0.7964; 40-60 m n 785 E 6.090 signed -5.345 gate 0.7605

## Limits of the card

Three fits of one locator on the same train clips, one detector, the same 141 evaluation clips read a third time; the interval per fit answers what another draw of clips would say with the fits held fixed.
