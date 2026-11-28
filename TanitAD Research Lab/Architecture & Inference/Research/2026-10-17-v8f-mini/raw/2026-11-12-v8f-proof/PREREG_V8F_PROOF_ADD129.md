# PREREG_V8F_PROOF_ADD129 — BOXFUSE-T0: the box read joined with the token head under an agreement gate

Tier T0: recorded frames, no rollout; a diagnostic of a perception read and not driving performance. Written after the names of card ADD128 (row 197), whose committed reading for `WORSE` asks for the next arm from the beside values: the pick is the cause, the box with the right pick is better than arm D. Frozen before any tool of this card. CPU only, on the banked detector files of card ADD127; no GPU, no download, no fit.

## 0. What is known when this card is written

Every value of row 197: the errors of arms D, `BW`, `BT`, `BL` in all and by gap bin, the pick shares, the size errors. Not known: any value of the two reads on a single frame, how often they agree, how often the locator's pick is the label's pick. The two parameters of this card (the geometric mean, the factor 1.5) are fixed here with no look at such values. The 141 evaluation clips are those of row 197: a result of this card is a second use of them, and the pilot set holds no further clips to confirm it on. The row says so.

## 1. Question

A wrong pick gives a box gap far from the token head's gap; a right pick gives two reads with different sources of error. Does a read that takes both where they agree and the token head where they do not beat arm D?

## 2. Material

That of card ADD128 unchanged: the detector files, the wide read, the pick by the locator `LOC_s0` (vision only), the size-to-gap map fitted on the 500 train clips with the label pick, the three banked fits of arm D, the 3,841 evaluation frames of 141 clips. The tele read is not used.

## 3. Arms

* `FG` (candidate form, vision only): on an evaluation frame with a pick of `BW`, when the box gap and arm D's gap differ by at most a factor of 1.5, the geometric mean of the two; else arm D's gap. A frame with no pick takes arm D's gap.
* `FM` (beside): the geometric mean on every picked frame, with no gate.
* `FL` (bound, not a candidate): as `FG` with the label's pick.
* Each arm is formed against each of the three fits of arm D; the three values are printed singly and as their mean.

## 4. Measures and names

All controls of section 5 have to pass before any name or value is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

Measure and estimator as in card ADD128: mean absolute gap error on the 3,841 frames, R(X) the mean error of arm X over the three fits divided by that of arm D, cluster bootstrap over the 141 clips, 2,000 draws, seed 129, fits held fixed.

* `BOXFUSE-T0-BETTER` when the interval of R(FG) lies below 1 and R(FG) is at most 0.90; `BOXFUSE-T0-SMALL-GAIN` when the interval lies below 1 and R(FG) is above 0.90; `BOXFUSE-T0-WORSE` when the interval lies above 1; `BOXFUSE-T0-NOT-SEPARATED` else.
* Far leads (label gap 40 to 60 m): `BOXFUSE-T0-FAR-LEADS-LESS-SHORT` when the paired difference of the signed error, `FG` minus D, has an interval above 0 and the signed error of `FG` is at most half of that of D in size; `BOXFUSE-T0-FAR-LEADS-STAY-SHORT` else.
* Beside, naming nothing: `FM` and `FL`; the share of evaluation frames with both picks on which the locator's pick is the label's pick, and the error of `BW` on the frames where it is and where it is not; the share of picked frames that pass the gate, in all, by gap bin, and split by whether the pick is the label's; the error of `FG` by gap bin.

## 5. Controls

* f0 inputs and identity: the controls file of card ADD128 says all pass and its input hashes are the present ones (detector listings, the three fits of arm D, the locator file, the join, the lists); the errors of arm D and of `BW`, computed by this tool, equal the banked values of card ADD128's result within 1e-6 m for each of the three fits.
* f1 the gate is live: with arm D's gaps permuted across the evaluation frames (seed 129) the share of picked frames that pass the gate is at most 0.75 of the share with the gaps in place.
* f2 vision only: `FG` and `FM` read no label at inference (the pick function takes the locator's file and refuses a label file; tool test).
* f3 instrument: the gate passes on at least 0.30 of the 3,841 evaluation frames; below that `FG` is arm D under another name.
* f4 tests: the tests of this card's tool pass before the read, with the test output older than the read; they hold a planted defect (a gate that lets every frame pass) that has to make f1 fail.
* f5 one evaluation set: every arm is read on the same 3,841 rows of the same 141 clips.

The first controls run on the pod is the run of record. The controls stage prints pass or fail, counts and the two gate shares of f1 and f3; it prints no error of an arm. A clause that fails for a reason of construction goes into a note before any value is opened; a missed bar of f1 or f3 fails the card as written.

## 6. Committed readings

* `BETTER`: the box read joined with the token head is a lever of the camera gap at T0. Next, each a decision of the PI: gap files of the candidate's rollouts from this read (a detector pass over the rollout windows) so that the lead rule can be run with it at T1-kin, and whether a detector enters the perception path.
* `SMALL-GAIN` or `NOT-SEPARATED`: the gate and the mean do not carry the box read past the token head by the bar; the beside values say whether wrong picks pass the gate or right picks are too few, and the next arm is a pick learned on the train clips among the detector's boxes, or, as a decision of the PI, a detector trained on our labels.
* `WORSE`: as written; the same beside values name the cause.

## 7. Families

LONGITUDINAL (distance keeping) is the family this read serves; at T0 it is measured as the gap error only. LATERAL and TACTICAL are not measured. STRATEGIC is not computed, n = 0: a gap read on recorded frames reads no strategic quantity.

## 8. Limits

One detector, one set of weights, one fit of the locator, 141 evaluation clips used a second time; two parameters fixed without a fit; the interval answers what another draw of clips would say, not what another fit, another detector or another choice of the two parameters would say.
