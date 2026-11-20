# PREREG_V8F_PROOF_ADD93 (GAPSPLIT): the reader's gap error split into the part that stays with a clip and the part that changes from frame to frame

Tier T0 (reads on logged frames; no rollout; a diagnostic, never a statement on driving performance). Evidence class of every value it will give: MEASURED. Inputs: the per-frame dumps of the reader (two reader seeds) and the label gap. No ego speed, no token, no fit, no new encode. Written after the read of card ADD92 and before any tool of this card exists.

## 1. Why, and what the lead knows

Row 153 (card ADD92): a closing read from a gap carries a scale error that stays with one clip up to the largest level planted (20 percent) and does not carry a frame-to-frame error above about 0.5 to 1 m. Row 152 (card ADD91) gives the reader's gap error as one standard deviation per range (8.64 m over all scored frames), which mixes the two parts. The cost card needs the reader's error in the terms of the bar, as the baseline of the pilot.

Known to the lead: rows 150 to 153, among them the lag-1 autocorrelation of the reader's gap error (0.94, post-hoc in row 150) and the r +0.26 of the slope of the reader's own gap (row 152), from which a fast part far above the bar is expected. Not known: any split of the reader's error by clip, and any frame-to-frame statistic of it.

## 2. Frames and statistics

* Frames: per reader seed, every frame of the 141 evaluation clips that has a valid reader gap and a label gap; a clip with fewer than 5 such frames is left out. Absolute error e: reader's gap minus label gap. Proportional error q: reader's gap over label gap, minus 1.
* CONST (absolute and proportional): the standard deviation over frames of the mean error of the frame's clip (each clip weighted by its frames).
* FAST (absolute and proportional): the root mean square over frames of the error minus the mean error of the frame's clip.
* STEP (absolute and proportional): the root mean square of the change of the error between two neighbouring frames of a clip that are both in the frame set.
* Reported beside them: the lag-1 autocorrelation of the error minus its clip mean; the share of CONST squared in CONST squared plus FAST squared; FAST and STEP (absolute) on the frames below 20 m of label gap.
* Bars, the same statistics on planted gaps of card ADD92 on the same frames: FAST absolute of the reader-like absolute level of 1 m (R-ABS S100); FAST proportional of the reader-like proportional level of 2.5 percent (R-PROP P025); STEP absolute of the white absolute level of 0.5 m (W-ABS S050); CONST proportional of the constant proportional level of 20 percent (C-PROP P200). The first three are the largest levels of their kind that hold r 0.9 through the read's own runs over 10 frames; the fourth is the largest level planted, which is carried.
* Intervals of the reader's statistics: two-level bootstrap, reader seeds then evaluation episodes (one episode draw shared by the seeds), 2,000 draws, percentile 2.5 and 97.5, seed 93. The bars are point values.

## 3. Names (frozen)

* `GAPSPLIT-FAST-ABS-ABOVE-THE-BAR` when the lower end of the reader's FAST absolute is above its bar; `-WITHIN-THE-BAR` when the upper end is at or below it; `-NOT-DECIDED` otherwise.
* `GAPSPLIT-FAST-PROP-ABOVE-THE-BAR`, `-WITHIN-THE-BAR`, `-NOT-DECIDED` by the same rule on FAST proportional.
* `GAPSPLIT-STEP-ABS-ABOVE-THE-BAR`, `-WITHIN-THE-BAR`, `-NOT-DECIDED` by the same rule on STEP absolute.
* `GAPSPLIT-CONSTANT-PROP-ABOVE-THE-CARRIED-LEVEL`, `-WITHIN-THE-CARRIED-LEVEL`, `-NOT-DECIDED` by the same rule on CONST proportional.
* Reported with no name: every statistic with its interval and per reader seed, each bar, and the ratio of the reader's value to its bar.

## 4. Controls

All five have to pass before any name or any value of this card is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* u0: the scored rows per reader seed equal those of card ADD81 (2,572 each).
* u1 (identity): with the label gap in the place of the reader's gap every statistic is 0 (at most 1e-9).
* u2 (planted split, exact): with the reader's gap replaced on the frame set by the label gap plus a part that is constant per clip with a standard deviation of exactly 3.0 m plus a part with a mean of 0 in every clip and a root mean square of exactly 1.0 m, the tool returns CONST 3.0 and FAST 1.0 within 1e-9; with the proportional form (constant part 0.2, fast part 0.05) it returns CONST 0.2 and FAST 0.05 within 1e-9.
* u3 (the planted gaps are those of card ADD92): r of the arms OWN-R-ABS-S100-W10 and OWN-C-PROP-P200-W10, recomputed by this tool, equal the values in the result file of card ADD92 within 1e-9.
* u4: the self-test of the tool on the pod before the read, on a planted world: a fast part taken about the mean of all frames, a proportional error computed as an absolute one, a planted noise of another seed and a wrong frame count each turn their control red; no name is written and every value is held back when a control fails.

## 5. What follows, fixed now

* `GAPSPLIT-FAST-ABS-ABOVE-THE-BAR` or `GAPSPLIT-STEP-ABS-ABOVE-THE-BAR`: the cost card carries the reader's fast part and its ratio to the bar as the baseline of the pilot.
* FAST and STEP both within their bars while the slope of the reader's own gap reads r +0.26 (row 152): the Gaussian form of the planted error does not describe the reader's error, and a bar in standard deviations is not enough; the cost card then states the pass bar of the pilot only as the r of the slope of the pilot's own gap.
* In every case the cost card states the first pass criterion of the pilot directly, as in row 152: r of the slope of the pilot's own gap with the run rule against the label closing speed, lower end at or above 0.9; the statistics of this card are the diagnostics beside it.
* `GAPSPLIT-CONSTANT-PROP-ABOVE-THE-CARRIED-LEVEL`: the reader's scale error per clip is above the largest level shown to be carried, and the cost card adds a bar on it for the pilot.

## 6. Limits

T0 only. The split is by clip, and a lead that changes inside a clip puts a part of a constant error into the fast part. The bars come from Gaussian planted error with one coefficient per kind. The label gap is the reference and has its own error. Two seeds of one reader. Designed after cards ADD89 to ADD92.
