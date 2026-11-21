# PREREG_V8F_PROOF_ADD91 (GAPRANGE): the reader's gap error over range, and a slope of the reader's own gap with the label's run rule

Tier T0 (reads on logged frames; no rollout; a diagnostic, never a statement on driving performance). Evidence class of every value it will give: MEASURED. Inputs: the per-frame dumps of the reader (two reader seeds) and the label gap. No ego speed, no token, no fit, no new encode. Written after the read of card ADD90 and after the other session's read of rows 149 to 151, before any tool of this card exists.

## 1. Why, and what the lead knows

Row 150 (card ADD89) carries the reader's gap error as a post-hoc value with no name, because two controls of that card failed and the card has no map of arms to controls. The cost card to the PI needs that value under its own controls. Row 151 (card ADD90) gives what a slope tolerates (about 1 m) on the label's own runs; it does not say where over range the reader misses, and no slope of the reader's own gap with the label's run rule has been read.

Known to the lead from the opened part of card ADD89, and stated here because this card is designed after it: over all scored windows the reader's gap error has a standard deviation of 8.77 and 8.73 m, a mean of -2.11 and -2.59 m and a lag-1 autocorrelation of 0.939 and 0.942; the plain window slopes of the reader's gap read r +0.07 to +0.26. Not known to the lead: any value by range, and any slope of the reader's gap with a run rule.

## 2. Frames, bins, arms

* Frames: the scored rows of card ADD81 at 25 frames (2,572 per reader seed), each at its own frame, where the reader's gap is valid and finite at that frame. Target of the closing arms: the label closing speed of the row.
* Bins by the label gap at the row's frame: NEAR below 20 m, MID from 20 to below 40 m, FAR from 40 m, and ALL.
* Gap error: reader's gap minus label gap. Per bin: standard deviation about the bin mean, mean, mean absolute error, frames, episodes.
* Closing arms, W in 5, 10, 25 (frame period 0.2 s):
  * RG_W: least-squares slope of the reader's gap over the run found from the reader's own gap with the label's rule (going back from the row's frame while the reader's gap is valid and its step is at most 3.0 m, at most W frames); a run below 3 frames gives 0.0, as the rule's source does. This is a form that needs no label.
  * RL_W: the same slope of the reader's gap over the run found from the label gap (a step of the label gap of more than 3.0 m, a missing label or a missing reader value ends the run). A bound on run finding, not deployable.
  * RP_W: the plain slope of the reader's gap over the last W frames with a valid value, no run rule (the form of card ADD89), at least 2 frames.
* Intervals: two-level bootstrap, reader seeds then evaluation episodes (one episode draw shared by the seeds), 2,000 draws, percentile 2.5 and 97.5, seed 91.

## 3. Names (frozen)

* For each bin B in ALL, NEAR, MID, FAR: `GAPRANGE-GAP-ERROR-<B>-WITHIN-1M` when the upper end of the interval of the standard deviation is at or below 1.0 m; `-ABOVE-2M` when the lower end is above 2.0 m; `-NOT-DECIDED` otherwise; `-NO-DATA` when the bin has fewer than 100 frames or fewer than 5 episodes over the two seeds.
* For each W in 5, 10, 25, on ALL: `GAPRANGE-READER-RUN-<W>-REACHES-THE-REQUIREMENT` when the lower end of the interval of r of RG_W is at or above 0.9; `-BELOW-THE-REQUIREMENT` when the upper end is below 0.9; `-NOT-DECIDED` otherwise.
* On NEAR, for W 10: `GAPRANGE-READER-RUN-10-NEAR-REACHES`, `-NEAR-BELOW`, `-NEAR-NOT-DECIDED` by the same rule on the rows of the bin; `-NEAR-NO-DATA` as above.
* Run finding, on ALL at W 10, paired difference of the absolute error of RG_10 minus RL_10 on the rows where both have a value: `GAPRANGE-RUN-FINDING-COSTS` when the lower end is above 0; `GAPRANGE-RUN-FINDING-NOT-SEPARATED` otherwise.
* Reported with no name: every arm in every bin with r, interval, mean absolute error, standard deviation of its error, the standard deviation of the label closing speed in the bin, the share of rows whose run is below 3 frames; mean and mean absolute gap error per bin and per reader seed.

## 4. Controls

All five have to pass before any name or any value of this card is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* q0: the scored rows per reader seed equal those of card ADD81 (2,572 each, read from its control file).
* q1 (identity): with the label gap put in the place of the reader's gap, the gap error is 0 on every frame (largest absolute value at most 1e-9) and RG_5 equals the label closing speed on every scored row (largest absolute difference at most 1e-6).
* q2 (sign and spread, planted exactly): with the reader's gap replaced on the measured frames by the label gap plus 2.0 m plus s times a noise that is shifted and scaled to mean 0 and standard deviation 1 on exactly those frames, the tool's statistic returns a mean of +2.0 m and a standard deviation of s within 1e-9, for s 1 and s 8.
* q3 (bins): the three bins hold every measured frame exactly once per reader seed; a planted error with a spread of 1 m on NEAR and 8 m on MID and FAR (each scaled exactly within its bin) returns 1.0, 8.0 and 8.0 within 1e-9.
* q4: the self-test of the tool on the pod before the read, on a planted world: the run rule on the reader path equals a line-by-line port of the rule's source; a flipped sign of the error, a moved bin edge, a noise that is not scaled, a dropped step gate and a wrong frame count each turn their control red; no name is written and every value is held back when a control fails.

## 5. What follows, fixed now

* `GAPRANGE-READER-RUN-10-NEAR-REACHES` or `-NEAR-NOT-DECIDED`: the next card of the fork reads the lead rule with RG_10 as its closing input in T1-kin/hb0.2 at the pre-roll of 30 steps (6 s), through the closing table of the study wrapper, against the form without a closing input. No new encode.
* `-NEAR-BELOW` together with `GAPRANGE-GAP-ERROR-NEAR-ABOVE-2M`: the gap read is the limit also at the range where contact happens. The cost card goes to the PI with the per-range gap error of this card as its measured number in the place of the post-hoc value of row 150.
* `-NEAR-BELOW` with the gap error of NEAR not above 2 m: the reader's gap is near enough at short range and the slope still does not read; the next question is the time course of the error at short range, a new card.
* `GAPRANGE-RUN-FINDING-COSTS`: finding the changes of the lead from the noisy gap has a price of its own, which the ladder of row 151 does not hold; the cost card says so.

## 6. Limits

T0 only; a pass here is not a pass in the rule. The bins use the label gap, which a deployed system does not have. r inside a bin is read on a narrower range of the label than r over all rows, so the bin name is not the requirement of row 147 itself. The target is the label closing speed, a slope over at most five frames of the label gap. Two reader seeds of one reader. Designed after the opened part of card ADD89 and after card ADD90 (section 1).
