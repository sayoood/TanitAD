# PREREG_V8F_PROOF_ADD89 (KINCLOSE2): the arms of card ADD88 with the ego control in a form that is a unit check, and the gap error a window slope can carry

Tier T0 (reads on logged frames; no rollout; a diagnostic, never a statement on driving performance). Evidence class of every value it will give: MEASURED. No new encode, no camera frame, no fit, no parameter chosen on the scored frames. Written after card ADD88 was read with one control failed and before any tool of this card exists.

## 1. What card ADD88 left, and what the lead knows

Card ADD88 ran once to a read (2026-11-20, root `kinclose_b`): controls k0, k2, k3, k4 passed, k1 did not. By the card it has no name; its tool wrote none and holds every arm value under a withheld key. The lead has opened the control record only. Known from that record, and nothing else: in k1 the three planted faults of the ego path (sign flipped, path shifted by +10 and by -10 frames) each read worse than the path as it is, and the clause "the mean absolute error of B_25 is lower than that of BS_25 with a separated paired interval" is not met (whether by its point value or by its interval is not in the record and is not known to the lead).

The reading of the lead, which is why the form of the control is in question (rule of row 136): the clause that failed is not a check of unit and clock. It states the premise of card ADD88 on a perfect gap, that the ego path improves a window slope, and so it is an outcome. The clauses that do check sign and clock passed. A changed form belongs to a new card and a new run: this card. Its row is read by the other session before it is landed. This card is designed after the look at that control record; the pair B_25 against BS_25 is therefore reported as an outcome whose direction the lead partly knows, and no name that says the ego term helps on the label gap could rest on this card alone.

## 2. What is unchanged and what changes

Unchanged from card ADD88 with its note N1: every definition of section 2 (v, s, slope_W, the arms K_W, S_W, E_W, B_W, BS_W for W in 5, 10, 25, 50, the scored frames, the read, the intervals, the cell L25), the controls k0, k2, k3, k4, and the inputs.

Changed:

* Control k1 in the form of this card (unit, sign and clock of the ego speed, on the label gap): B_25 formed with each of five planted faults of the ego path has to read worse, in the point value of the mean absolute error on the same frames, than B_25 as it is: the path shifted by +10 frames, by -10 frames, its sign flipped, the ego speed multiplied by 3.6, the ego speed divided by 3.6. The two scale faults are new: they are the check that the speed is in metres per second, which the first form did not have.
* The pair B_25 against BS_25 leaves the control and is an outcome (section 3).

Added:

* Planted gap-noise arms on the label gap, on the frames where BS_W has a value, for W in 10 and 25 and a standard deviation sigma in 0.25, 0.5, 1 and 2 m: PW_(W, sigma) = slope_W(label gap + white Gaussian noise of standard deviation sigma); PS_(W, sigma) = slope_W(label gap + Gaussian noise of standard deviation sigma that follows a stationary first-order autoregression of 0.9 per frame, started anew in every clip). One draw each, fixed generator seed, the same noise for both reader seeds' rows of a clip frame.
* Reported with no name: the error of the reader's raw gap read against the label gap on the frames of the scored windows where both exist, per reader seed: mean, standard deviation, lag-1 autocorrelation.

## 3. Names

* `KINCLOSE2-EGO-TERM-ON-THE-LABEL-GAP-HELPS` when the paired difference of the absolute error B_25 minus BS_25 has its upper end below zero; `-HURTS` when its lower end is above zero; `-NOT-SEPARATED` otherwise.
* `KINCLOSE2-EGO-TERM-HELPS` / `-HURTS` / `-NOT-SEPARATED`: the same rule for K_25 minus S_25 (the reader's gap).
* `KINCLOSE2-K25-BETTER-THAN-OUTPUTS` when the paired difference K_25 minus L25 has its upper end below zero and K_25 has r at or above 0.30 in each reader seed; else `KINCLOSE2-K25-NOT-BETTER`.
* The reader's gap against the requirement of row 147: `KINCLOSE2-READER-GAP-REACHES-THE-REQUIREMENT` when the lower end of the interval of r of K_25 or of S_25 is at or above 0.9; `KINCLOSE2-READER-GAP-BELOW-THE-REQUIREMENT` when the upper ends of all K_W and all S_W (W in 5, 10, 25, 50) are below 0.9; else `KINCLOSE2-READER-GAP-NOT-DECIDED`.
* A perfect gap over a window, for W in 10 and 25: `KINCLOSE2-WINDOW-<W>-REACHES-THE-REQUIREMENT` when the lower end of r of BS_W is at or above 0.9; `KINCLOSE2-WINDOW-<W>-BELOW-THE-REQUIREMENT` when its upper end is below 0.9; else `KINCLOSE2-WINDOW-<W>-NOT-DECIDED`.
* The gap error a slope can carry, for each W in 10 and 25 whose window reaches the requirement, read on the smooth planted noise: `KINCLOSE2-GAP-ERROR-W<W>-UP-TO-<code>` with the code of the largest sigma (S025, S050, S100, S200) such that PS_(W, sigma) and every smaller sigma of the ladder have the lower end of r at or above 0.9; `KINCLOSE2-GAP-ERROR-W<W>-BELOW-S025` when the smallest does not. The white ladder is reported beside it with no name.
* Reported with no name: every arm at every W with r, interval, error, standard deviation and lag-1 autocorrelation of the error; E_25; the error standard deviation of each arm beside the 1.24 m/s of row 147.

## 4. Controls (all have to pass before a name or a value is opened)

* k0, k2, k3, k4 of card ADD88, unchanged.
* k1 in the form of section 2.
* k5 (planted arms): with sigma = 0 the planted arm equals BS_W exactly (largest absolute difference at most 1e-9); the realised standard deviation of every planted noise over the frames used is within 10 percent of sigma; the lag-1 autocorrelation of the smooth noise is between 0.85 and 0.95 and that of the white noise is within 0.05 of zero.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card.

## 5. What follows (committed before the data)

* `WINDOW-25-BELOW-THE-REQUIREMENT`: a slope over 5 s cannot reach the requirement even on a perfect gap; with `WINDOW-10-REACHES` the cost card says that a closing read from the gap history needs a window of about 2 s and a gap error no larger than the GAP-ERROR name, and sets the reader's measured gap error beside it. With both windows below the requirement the cost card says that the closing read has to come from the scene itself and not from a gap history.
* `READER-GAP-BELOW-THE-REQUIREMENT` with a window that reaches: the gap read is the limit, and the cost card carries the gap error required and the gap error measured.
* `K25-BETTER-THAN-OUTPUTS`: K_25 goes into the rule as a table at T1-kin/hb0.2 with the matched surrogate of card ADD85 and the arm E_25 beside it, in a card of its own; no adoption by this fork.
* The ego-term names are reported as they fall; `EGO-TERM-ON-THE-LABEL-GAP-NOT-SEPARATED` or `-HURTS` says that the premise of card ADD88 does not hold on these clips.

## 6. Limits known in advance

T0 only; designed after the control record of card ADD88 (section 1); the target is the label closing speed, a slope over at most five frames of a label gap, so every bound here is a statement about that label; a window estimate lags when the lead changes its speed, and a pass at T0 is not a pass in the rule; the planted noise is Gaussian with one coefficient and one draw per level, a model of a gap error and not the reader's error; the scale faults of k1 test the two factors 3.6 and 1/3.6 only.
