# BOXFUSE-T0-LOCFIT verdict (card PREREG_V8F_PROOF_ADD130)

Tier: T0 (recorded frames, no rollout; a diagnostic of a perception read and not driving performance)

## Families

* LONGITUDINAL (distance keeping): the family this read serves; at T0 it is measured as the gap error only.
* LATERAL: not measured.
* TACTICAL: not measured.
* STRATEGIC: not computed, n = 0: a gap read on recorded frames reads no strategic quantity.

## Names

* `BOXFUSE-T0-LOCFIT-REPLICATED`
* `BOXFUSE-T0-LOCFIT-BAR-NOT-3-OF-3`

## The joined read FG against arm D, per fit of the locator (3,841 frames of 141 clips)

* LOC_s0 (row 198): R(FG) 0.8961, interval [0.8449, 0.9407]; box read alone R 1.2127; the locator's pick is the label's pick on 0.9075 of the frames with both picks
* LOC_s1: R(FG) 0.9007, interval [0.8509, 0.9456]; box read alone R 1.3387; the locator's pick is the label's pick on 0.8693 of the frames with both picks
* LOC_s2: R(FG) 0.9068, interval [0.8545, 0.9537]; box read alone R 1.3388; the locator's pick is the label's pick on 0.8815 of the frames with both picks
* LOC_s1 beside: FG 3.5987 m against arm D 3.9954 m; the gate passes on 0.7823 of the rows, on 0.9341 of the frames with the label's pick and 0.4611 of the others; far leads signed -5.129 m
* LOC_s2 beside: FG 3.6228 m against arm D 3.9954 m; the gate passes on 0.7866 of the rows, on 0.9399 of the frames with the label's pick and 0.4652 of the others; far leads signed -5.261 m
* R(FG) over the three fits: 0.8961 to 0.9068; bar of the first fit 0.90

## Limits of the card

Three fits of one locator on the same train clips, one detector, the same 141 evaluation clips read a third time; the interval per fit answers what another draw of clips would say with the fits held fixed.
