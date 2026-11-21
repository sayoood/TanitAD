# PREREG_V8F_PROOF_ADD100 (STEERREAD): the crop steered from the image

Fork v8proof, 2026-11-21. Tier T0 (a probe on logged frames; a diagnostic of a reader, never driving performance). Evidence class of every value read under this card: MEASURED. Designed after the read of card ADD99 (row 161: the pilot fails on the fixed windows, r +0.564 [+0.419, +0.675]; with the windows on the lead from the label the gap error is 2.91 m against 8.64 m of the present reader). By the reading committed in card ADD99 the missing piece is finding the lead in the image. This card is a third crop form of the same pilot on the same 641 clips and the same staged video (no new download), inside the PI's go of 2026-11-21. Frozen with its hash file before any tool, steering file, token or fit of it exists.

## 1. Question

When the position of the windows comes from the image alone (a localiser on the fixed windows), how much of the bound arm's cut of the gap error is kept?

## 2. Localiser (image only at inference)

* Input: the tokens of arm D of one frame (card ADD99 sections 3 and 4: the two fixed windows, 130 tokens). Model: the token reader of card ADD99 section 5 with note N1 (layer norm, linear to 256, position vectors, one query with one cross-attention layer of 4 heads, token dropout 0.1), then an MLP 256 to 256 to 2. Output: azimuth = 16 times tanh of the first output, elevation = 16 times tanh of the second, in degrees, in the angles of note N1.
* Target (training only): the direction of the centre of the lead's box of card ADD99's box files, on the frames where the join's rule finds a lead. Loss: smooth L1 with beta 0.5 degrees on both angles. AdamW, learning rate 1e-3, weight decay 0.05, batch 256, 20 epochs, one warm-up epoch then cosine to 0; the last epoch is kept (no selection). Seed 0.
* Cross-fit, so that no clip is steered by a localiser that read it: the 450 fit clips of card ADD99, in list order, form five folds by position modulo 5. Localiser L0 to L4: fit on the fit clips outside the fold, steers the fold. Localiser LA: fit on all 450 fit clips, steers the 50 selection clips and the 141 evaluation clips.
* Steering direction of a frame: the localiser's output on that frame, on every frame (with or without a lead).

## 3. Arm S and heads

* Arm S: W32 and W16 of card ADD99, both centred on the steering direction of the frame. Tokens as in card ADD99 section 4. Nothing of the label enters at inference.
* Head S-H5: head H5 of card ADD99, tokens only (the same head as the bound arm B-H5). Head S-H5d: the same, with the steering direction of each of the five frames (azimuth and elevation, each divided by 16) appended to the frame's vector, MLP 1290 to 256 to 1. Fit as in card ADD99 section 5 with note N1, fit seeds 0 and 1, the same fit and selection clips.

## 4. Read

The reader of card ADD99, unchanged in its estimator (tool `kinclose/kinclose4.py` through the wrapper), on the dumps of the two forms; the dumps of card ADD99 (D-H5, B-H5) and of the present reader are the references.

* Card name, on S-H5d: the ratio of the standard deviation of the gap error to that of the present reader, on the frames both measure, pooled over the two seeds, paired bootstrap over evaluation episodes with 2,000 draws (the estimator of card ADD99): `STEERREAD-CUT-BY-2-REACHED` when the upper end is at or below 0.5, `STEERREAD-CUT-BY-2-NOT-REACHED` when the lower end is above 0.5, else `STEERREAD-CUT-BY-2-NOT-DECIDED`.
* Beside it: the same for S-H5 (`STEERREAD-TOKENS-ONLY-CUT-BY-2-...`); per form the name of the tool's rule on r of RG10 (`STEERREAD-<form>-REACHES`, `-BELOW`, `-NOT-DECIDED`); for S-H5d the same ratio against D-H5 (`STEERREAD-BEATS-FIXED` when the upper end is below 1, else `STEERREAD-NOT-SEPARATED-FROM-FIXED`, or `STEERREAD-WORSE-THAN-FIXED` when the lower end is above 1) and against B-H5 (`STEERREAD-BELOW-BOUND` when the lower end is above 1, else `STEERREAD-NOT-SEPARATED-FROM-BOUND`).
* Printed with no name: the localiser's error on the evaluation frames with a lead box (median and 90th percentile of the absolute error in azimuth and in elevation), the share of those frames on which the box centre lies inside the W16 window of arm S, the gap error by range, the split of row 154, the three parts, the values per fit seed.

Reading committed in advance: REACHED on S-H5d opens a card for the rule read in T1-kin/hb0.2 (kinematic rollout on the model's own actions) of a form with no label at inference (presence from the present reader, gap and slope from arm S). NOT-REACHED with a localiser error that leaves the lead outside the W16 window on a large share of frames names the localiser as the limit (finer tokens for it, or more train clips, the latter the PI's call); NOT-REACHED with the lead inside the window names the label-exact centring of the bound arm as the source of its cut. NOT-DECIDED goes to the PI with the number.

## 5. Controls

All have to pass before any name or any value of this card is opened; they are logged in `TIMES_NOTE.md` before the names. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* s1 (cross-fit): every localiser's record lists the clips it read; no train clip is steered by a localiser that read it; no evaluation clip is read by a localiser or by a head fit.
* s2 (steering is the localiser's output): for 6 evaluation clips (first by sha256 hex of the clip id) the steering file is reproduced from the arm D tokens and localiser LA within 1e-3 degrees; the token manifest of arm S carries the sha256 of every steering file used.
* s3 (the route is the bound arm's route): with the box directions of card ADD99 given as steering, the arm S route gives arm B's tokens for the same 6 clips on the frames with a lead (cosine at or above 0.9999 on every such frame).
* s4 (the wrapper is the tool): control c2 of card ADD99, read again.
* s5 (the read can say both): the selftest of the reader with planted errors (scaled to 0.4 gives REACHED, to 0.9 gives NOT-REACHED), and control c3 of card ADD99.
* s6 (the fit makes no result of its own): S-H5d fit with permuted label gaps (one seed) does not get the name REACHES and its gap error has a standard deviation of at least 0.8 times that of the label gap.
* s7 (kit tests): the tests of the kit pass in the lead's run on the dev box before the read.
* s8 (complete): all 641 clips have steering files and arm S tokens with the frame count of the join.

## 6. Four metric families

LONGITUDINAL: gap error and r against the label closing speed at T0, a diagnostic of the reader. LATERAL, TACTICAL, STRATEGIC: not read at T0 (no rollout, no planner output); they are read with the rule form in T1-kin/hb0.2 under the next card, STRATEGIC included.

## 7. Cost and limits

ESTIMATED: one more decode and encode pass for one arm (about 25 minutes), six localiser fits and five head fits, about 1 GPU-h; with card ADD99 about 2 GPU-h in total for the pilot, inside the PI's go. Limits: T0 only; 500 train clips; two fit seeds, one localiser seed; the localiser reads pooled tokens of the fixed windows; arm S is steered frame by frame with no smoothing over time (a choice made before data).
