# PREREG_V8F_PROOF_ADD90 (KINCLOSE3): what a gap history can carry: the lag of a longer run and the gap error a slope tolerates

Tier T0 (reads on logged frames; no rollout; a diagnostic, never a statement on driving performance). Evidence class of every value it will give: MEASURED. Label gap only: no reader output, no ego speed, no token, no fit, no new encode. Written after the read of card ADD89 and before any tool of this card exists.

## 1. What cards ADD88 and ADD89 left, and what the lead knows

Card ADD88 has no name (control k1 failed). Card ADD89 failed controls k1 and k5; of it the lead has opened what neither guards (filter `kinclose/open_unguarded2.py`): the plain slopes of the label gap BS_W, the plain slopes of the reader's gap S_W, the cell L25, and the error of the reader's raw gap against the label gap. Known from that look:

* BS_5 reads the label closing speed at r +0.555 although control k0 shows it equal to the label on the 4,960 frames whose label run is five frames: the whole error sits on the 114 of 5,074 frames where the label's own rule ends the run at a step of more than 3.0 m (a change of the lead), and a plain slope runs across the step. The two WINDOW names of card ADD89 therefore measure the cost of such steps and not the lag of a window on one lead.
* The reader's gap error: standard deviation 8.77 and 8.73 m in the two reader seeds, mean -2.11 and -2.59 m, lag-1 autocorrelation 0.939 and 0.942.
* The planted arms of card ADD89 are unopened (k5: the realised spread of one level was 10.6 percent under its sigma where the band was 10 percent). The form of that control is in question: a realised spread of one draw has a sampling error that the band did not allow for. The changed form here plants the spread exactly. By the rule of row 136 the row of this card is read by the other session before it is landed.

## 2. Definitions (frame period 0.2 s; scored frames = the rows of `closeread_fit.build_rows(dump, 25, episode map)`; target = the label closing speed at the frame)

* Run of a frame j with cap W: the frames j, j-1, ... of the clip, at most W, as long as the label has a gap on the frame and the step of the label gap between neighbours is at most 3.0 m (the rule of `v8f_t1_eval_rule.py` lines 92 to 112 with W in place of 5). The run is always taken from the label gap itself, also for the planted arms.
* BG_W(j) for W in 5, 10, 25: the least-squares slope of the label gap over the run with cap W; a value when the run has at least 3 frames. BG_5 is the label closing speed.
* Planted noise, one draw per clip and kind, the same for every W and both reader seeds: white Gaussian; smooth = stationary first-order autoregression of 0.90 per frame; reader-like = stationary first-order autoregression of 0.94 per frame (the lag-1 value of the reader's gap error, section 1). Each noise is shifted and scaled so that over all frames with a label gap its mean is exactly 0 and its standard deviation exactly 1, then multiplied by sigma in 0.25, 0.5, 1, 2, 4, 8 m.
* Planted arms P<kind>_(W, sigma)(j): the slope over the same run of (label gap + noise), kinds W (white), S (smooth), R (reader-like), W in 5, 10, 25.
* Read per arm: pooled r with its interval, mean absolute error, standard deviation and lag-1 autocorrelation of the error. Intervals: two-level bootstrap (reader seeds, then evaluation episodes, the episode draw shared), 2,000 draws, percentile 2.5 / 97.5, fixed generator seed.

## 3. Names

* The lag of a longer run on one lead, for W in 10 and 25: `KINCLOSE3-RUN-<W>-REACHES-THE-REQUIREMENT` when the lower end of r of BG_W is at or above 0.9; `KINCLOSE3-RUN-<W>-BELOW-THE-REQUIREMENT` when its upper end is below 0.9; else `KINCLOSE3-RUN-<W>-NOT-DECIDED`.
* The gap error a slope tolerates, for W = 5 and for each W in 10 and 25 whose run reaches the requirement, and for each kind K in WHITE, SMOOTH, READERLIKE: `KINCLOSE3-GAP-ERROR-W<W>-<K>-UP-TO-<code>` with the code (S025, S050, S100, S200, S400, S800) of the largest sigma such that the planted arm at that sigma and at every smaller sigma of the ladder has the lower end of r at or above 0.9; `KINCLOSE3-GAP-ERROR-W<W>-<K>-BELOW-S025` when the smallest does not.
* `KINCLOSE3-READER-GAP-ERROR-ABOVE-EVERY-TOLERATED-LEVEL` when, for every W named above, the READERLIKE code is below S800 (the reader's measured gap error of 8.7 m lies above what a slope tolerates at every window); else `KINCLOSE3-READER-GAP-ERROR-WITHIN-A-TOLERATED-LEVEL`.
* Reported with no name: every arm with its r, interval, error and error spread; the error standard deviation of each arm beside the 1.24 m/s of row 147.

## 4. Controls (all have to pass before a name or a value is opened)

* g0 (definition): BG_5 equals the label closing speed on every scored frame, largest absolute difference at most 1e-6 m/s, and every scored frame has a value.
* g1 (planted path): with sigma = 0 every planted arm equals BG_W exactly (largest absolute difference at most 1e-9).
* g2 (planting): over the frames with a label gap each unit noise has mean 0 and standard deviation 1 within 1e-9; its lag-1 autocorrelation over neighbouring frames with a label gap is within 0.05 of zero (white), between 0.85 and 0.95 (smooth), between 0.89 and 0.97 (reader-like).
* g3 (frames): the number of scored frames per reader seed equals the count of the control file of card ADD81 (2,572).
* g4 (self-test of the tool on a synthetic world, run by the chain before the read): the run rule equals a line-by-line port of the rule's source on a world with planted steps of the gap; a slope that ignores the step fails g0; a noise that is not normalised fails g2; names are refused when a control fails.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card.

## 5. What follows (committed before the data)

* The cost card to the PI sets three numbers side by side: the gap error that a slope tolerates at each window (the READERLIKE and SMOOTH codes), the reader's measured gap error (8.7 m, section 1), and the r of the closing reads on record (0.38 to 0.49). `READER-GAP-ERROR-ABOVE-EVERY-TOLERATED-LEVEL`: the target handed to the PI for a new encode or a new reader is a gap error at or below the tolerated level, stated in metres, and the closing read follows from the gap history with no closing head; `-WITHIN-A-TOLERATED-LEVEL`: the reader's gap is not the limit in this model of its error, and the question goes to the form of its error (bias with range, changes of the lead).
* `RUN-25-BELOW` with `RUN-10-REACHES`: a gap history gives the closing read over about 2 s and not over 5 s.
* The ego term of cards ADD88 and ADD89 stays without a name and is not taken up here.

## 6. Limits known in advance

T0 only; designed after the look of section 1; the target is the label closing speed, itself a slope of the label gap, so every statement is about that label; the planted noise is Gaussian with one coefficient per kind and one draw, a model of a gap error: the reader's error also has a mean of -2.1 to -2.6 m and may depend on range, which the model does not have; the run is taken from the clean label gap, so the planted arms do not include the cost of finding the run from a noisy gap; a pass at T0 is not a pass in the rule.
