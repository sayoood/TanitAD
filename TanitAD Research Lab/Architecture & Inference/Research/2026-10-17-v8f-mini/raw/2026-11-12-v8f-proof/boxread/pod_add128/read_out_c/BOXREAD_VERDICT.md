# BOXREAD-T0 verdict (card PREREG_V8F_PROOF_ADD128)

Tier: T0 (recorded frames, no rollout; a diagnostic of a perception read and not driving performance)

## Families

* LONGITUDINAL (distance keeping): the family this read serves; at T0 it is measured as the gap error only.
* LATERAL: not measured.
* TACTICAL: not measured.
* STRATEGIC: not computed, n = 0: a gap read on recorded frames reads no strategic quantity.

## Names

* `BOXREAD-T0-WIDE-WORSE`
* `BOXREAD-T0-TELE-WORSE`
* `BOXREAD-T0-TELE-DOES-NOT-ADD`
* `BOXREAD-T0-FAR-LEADS-LESS-SHORT`

## Measure: mean absolute gap error on 3841 evaluation frames of 141 clips (m)

* arm D, each fit singly: D_0 3.960; D_1 3.913; D_2 4.112; mean 3.9954
* arm BW, formed against each fit: BW_0 4.891; BW_1 4.800; BW_2 4.845; mean 4.8453; R = 1.2127, interval [1.0941, 1.3590]
* arm BT, formed against each fit: BT_0 5.314; BT_1 5.230; BT_2 5.274; mean 5.2728; R = 1.3197, interval [1.1643, 1.5051]
* arm BL, formed against each fit: BL_0 3.455; BL_1 3.394; BL_2 3.422; mean 3.4238; R = 0.8569, interval [0.7486, 0.9674]
* BT divided by BW: 1.0882, interval [1.0182, 1.1662]
* interval: 2000 draws, RandomState(127); two-sided 95 % percentile, cluster bootstrap over the evaluation clips, fits held fixed (it answers what another draw of clips would say, not what another fit would say)

## Far leads (label gap 40 to 60 m): n 785 in 39 clips

* mean signed error (read minus label), mean of three fits: D -6.251, BW -2.803, BT -2.362, BL -2.228
* paired difference BW minus D, interval [1.4941, 5.4242]

## Beside, naming nothing

* bound arm BL: R 0.8569, interval [0.7486, 0.9674]
* share of evaluation frames with a pick, BW: all 0.8995 (3455 of 3841); by label gap: 0-10 m 0.9513 (n 596); 10-20 m 0.8659 (n 1111); 20-40 m 0.9029 (n 1349); 40-60 m 0.9019 (n 785)
* share of evaluation frames with a pick, BT: all 0.8539 (3280 of 3841); by label gap: 0-10 m 0.8540 (n 596); 10-20 m 0.8200 (n 1111); 20-40 m 0.8807 (n 1349); 40-60 m 0.8561 (n 785)
* share of evaluation frames with a pick, BL: all 0.9302 (3573 of 3841); by label gap: 0-10 m 0.9782 (n 596); 10-20 m 0.9064 (n 1111); 20-40 m 0.9214 (n 1349); 40-60 m 0.9427 (n 785)
* error on the BW-picked frames alone (n 3455): arm BW 4.168, arm D on the same frames 3.223
* error on the BT-picked frames alone (n 3280): arm BT 4.500, arm D on the same frames 3.153
* error on the BL-picked frames alone (n 3573): arm BL 2.683, arm D on the same frames 3.297
* detector size against label size, BL-picked frames, mean absolute error of ln width and ln height in per cent: 0-10 m n 583 width 19.6 height 20.1; 10-20 m n 1007 width 15.8 height 9.6; 20-40 m n 1243 width 14.0 height 9.6; 40-60 m n 740 width 14.4 height 7.5; all n 3573 width 15.5 height 10.9
* detector size against label size, BW-picked frames, mean absolute error of ln width and ln height in per cent: 0-10 m n 567 width 21.0 height 21.6; 10-20 m n 962 width 21.0 height 15.1; 20-40 m n 1218 width 19.8 height 15.4; 40-60 m n 708 width 23.7 height 18.2; all n 3455 width 21.1 height 16.9
* detector size against label size, BT-picked frames, mean absolute error of ln width and ln height in per cent: 0-10 m n 509 width 42.9 height 42.5; 10-20 m n 911 width 28.5 height 16.7; 20-40 m n 1188 width 24.6 height 14.4; 40-60 m n 672 width 24.8 height 17.3; all n 3280 width 28.6 height 20.0
* by label gap, arm D (mean of three fits): 0-10 m n 596 E 1.148 signed 0.443; 10-20 m n 1111 E 3.442 signed 2.139; 20-40 m n 1349 E 3.778 signed -1.174; 40-60 m n 785 E 7.314 signed -6.251
* by label gap, arm BW (mean of three fits): 0-10 m n 596 E 1.345 signed 0.504; 10-20 m n 1111 E 4.196 signed 3.354; 20-40 m n 1349 E 5.102 signed 0.423; 40-60 m n 785 E 7.981 signed -2.803
* by label gap, arm BT (mean of three fits): 0-10 m n 596 E 2.292 signed 0.911; 10-20 m n 1111 E 5.121 signed 3.953; 20-40 m n 1349 E 5.421 signed 1.084; 40-60 m n 785 E 7.496 signed -2.362
* by label gap, arm BL (mean of three fits): 0-10 m n 596 E 1.038 signed 0.140; 10-20 m n 1111 E 3.135 signed 2.093; 20-40 m n 1349 E 3.564 signed -0.500; 40-60 m n 785 E 5.404 signed -2.228

## Counts

* {"n_clips": 141, "n_eval_clips_with_tele_file": 141, "n_eval_rows_with_label_pick": 3573, "n_eval_rows_with_tele_pick": 3280, "n_eval_rows_with_wide_pick": 3455, "n_map_frames_tele": 34057, "n_map_frames_wide": 36086, "n_rows": 3841, "n_train_clips": 500}

## Limits of the card

One detector, one set of weights, no training on our data; 141 evaluation clips; the map is fitted on 500 train clips; the tele read exists only where the lead is inside the tele window; frames with no pick take arm D's gap, so the arms are mixtures and the share of picked frames is part of every reading. Sheets with camera frames and boxes are private and stay off the repository. The interval answers what another draw of evaluation clips would say with the fits held fixed, not what another fit would say. The set of controls is that of card ADD128: it was changed after the controls run of card ADD127 had failed and its control values were known.
