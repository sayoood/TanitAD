# BOXFUSE-T0 verdict (card PREREG_V8F_PROOF_ADD129)

Tier: T0 (recorded frames, no rollout; a diagnostic of a perception read and not driving performance)

## Families

* LONGITUDINAL (distance keeping): the family this read serves; at T0 it is measured as the gap error only.
* LATERAL: not measured.
* TACTICAL: not measured.
* STRATEGIC: not computed, n = 0: a gap read on recorded frames reads no strategic quantity.

## Names

* `BOXFUSE-T0-BETTER`
* `BOXFUSE-T0-FAR-LEADS-STAY-SHORT`

## Measure: mean absolute gap error on 3841 evaluation frames of 141 clips (m); gate factor 1.50

* arm D: D_0 3.960; D_1 3.913; D_2 4.112; mean 3.9954
* arm BW (box read, locator's pick (row 197)): 4.891; 4.800; 4.845; mean 4.8453; R = 1.2127, interval [1.0874, 1.3580]
* arm FG (gated geometric mean, locator's pick, vision only): 3.605; 3.490; 3.646; mean 3.5803; R = 0.8961, interval [0.8449, 0.9407]
* arm FM (geometric mean with no gate (beside)): 3.831; 3.762; 3.869; mean 3.8210; R = 0.9563, interval [0.8994, 1.0165]
* arm FL (gated geometric mean, label's pick (bound, not a candidate)): 3.438; 3.306; 3.444; mean 3.3957; R = 0.8499, interval [0.7950, 0.8971]
* FG divided by BW: 0.7389, interval [0.6650, 0.8183]
* interval: 2000 draws, seed 129; two-sided 95 % percentile, cluster bootstrap over the evaluation clips, fits held fixed

## Far leads (label gap 40 to 60 m): n 785 in 39 clips

* mean signed error (read minus label), mean of three fits: D -6.251, BW -2.803, FG -5.384, FM -4.937, FL -5.345
* paired difference FG minus D, interval [0.0909, 1.5198]

## Beside, naming nothing

* frames with both picks 3407; the locator's pick is the label's pick on 3092 (share 0.9075); error of the box gap on these 2.172 m (arm D 2.523 m), on the others 21.444 m (arm D 8.196 m)
* the gate passes on 0.8025 of the rows and 0.8921 of the picked frames; on 0.9382 of the frames whose pick is the label's and on 0.4952 of the others; on the gated frames FG 2.229 m, arm D 2.746 m
* by label gap, arm D: 0-10 m n 596 E 1.148 signed 0.443 gate 0.7696; 10-20 m n 1111 E 3.442 signed 2.139 gate 0.7966; 20-40 m n 1349 E 3.778 signed -1.174 gate 0.8290; 40-60 m n 785 E 7.314 signed -6.251 gate 0.7902
* by label gap, arm BW: 0-10 m n 596 E 1.345 signed 0.504 gate 0.7696; 10-20 m n 1111 E 4.196 signed 3.354 gate 0.7966; 20-40 m n 1349 E 5.102 signed 0.423 gate 0.8290; 40-60 m n 785 E 7.981 signed -2.803 gate 0.7902
* by label gap, arm FG: 0-10 m n 596 E 0.996 signed 0.536 gate 0.7696; 10-20 m n 1111 E 3.134 signed 2.179 gate 0.7966; 20-40 m n 1349 E 3.392 signed -1.021 gate 0.8290; 40-60 m n 785 E 6.498 signed -5.384 gate 0.7902
* by label gap, arm FL: 0-10 m n 596 E 0.992 signed 0.531 gate 0.7696; 10-20 m n 1111 E 3.087 signed 2.139 gate 0.7966; 20-40 m n 1349 E 3.145 signed -1.132 gate 0.8290; 40-60 m n 785 E 6.090 signed -5.345 gate 0.7902

## Limits of the card

One detector, one set of weights, one fit of the locator, 141 evaluation clips used a second time (row 197 read the same clips); two parameters fixed without a fit; the interval answers what another draw of clips would say, not what another fit, another detector or another choice of the two parameters would say.
