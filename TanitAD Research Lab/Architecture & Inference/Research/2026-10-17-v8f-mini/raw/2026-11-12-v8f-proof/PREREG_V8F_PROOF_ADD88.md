# PREREG_V8F_PROOF_ADD88 (KINCLOSE): a closing read from the reader's gap over a window together with the measured ego speed, with no fit

Tier T0 (reads on logged frames; no rollout; a diagnostic, never a statement on driving performance). Evidence class of every value it will give: MEASURED. Written while the fits of card ADD86 run and before any value of card ADD86 is read; before any tool of this card exists. No new encode, no camera frame, no fit, no parameter chosen on the scored frames.

## 1. What is asked

The closing estimates on record take the reader's outputs over at most 25 frames (5 s) and nothing else (cards ADD77 to ADD79, rows 139 and 141: ridge L25 r 0.38, network N10 r 0.39); the token head takes tokens (row 143: r 0.47). None of them takes the ego speed, which the model has as a measured input (PI ruling: measured ego motion is an input) and which the rule itself already uses. The closing speed is the lead's speed minus the ego's speed. Over a window of some seconds the ego's speed changes and the lead's speed changes less; a slope of the gap over the window therefore mixes a lag of the ego's own speed into the estimate, which the measured ego path removes exactly. Asked: (a) does the reader's gap with the ego path read the label closing speed better than the estimates on record; (b) with a perfect gap in place of the reader's gap, how far does each window get, so that the cost card can say whether the gap read or the window is the limit.

## 2. Definitions (frame period 0.2 s; per reader seed; evaluation clips only)

* v(j): ego speed of the clip file of the evaluation cache (key `v`, 101 frames). s(j): ego path, the cumulative sum of 0.2 * (v(j-1) + v(j)) / 2 from frame 0.
* slope_W(x)(j): least-squares slope against time of x over the frames j-W+1 .. j that carry a value (for W up to 25 all of them do on a scored frame; for W = 50 the frames of the clip in j-49 .. j that carry a read, of which the last 25 are present by construction).
* Arms on the reader's gap g (raw gap read of the per-frame dump of card ADD77, `closeread/dump/both`), W in 5, 10, 25, 50:
  K_W(j) = slope_W(g + s)(j) - slope_5(s)(j)   (the lead's speed over the window minus the ego's speed over the label's own five frames);
  S_W(j) = slope_W(g)(j)   (the same window with no ego term).
* Arm with no knowledge of the lead: E_W(j) = slope_W(s)(j) - slope_5(s)(j).
* Bound arms with the label gap in place of the reader's gap, where the label gap has a value on every frame of the window: B_W and BS_W, formed as K_W and S_W.
* Scored frames: the rows of `closeread_fit.build_rows(dump, 25, episode map)` (the scored frames of H 25 of card ADD78, the frames of L25 and T25); target the label closing speed at the frame. Bound arms are read on the scored frames on which they have a value, and K_W is reported on those frames too.
* Read per arm: pooled r with its interval, per seed r, mean absolute error, standard deviation and lag-1 autocorrelation of the error, and the paired difference of the absolute error against another arm. Intervals: two-level bootstrap (reader seeds, then evaluation episodes, the episode draw shared by the seeds), 2,000 draws, percentile 2.5 / 97.5, fixed generator seed. The cell L25 of card ADD78 is applied through the banked estimator files as in the scorer of card ADD81.
* Primary arm, fixed here: K_25 (the window of L25 and T25). K_5, K_10, K_50 are reported and carry no name.

## 3. Names

* `KINCLOSE-K25-BETTER-THAN-OUTPUTS` when the paired difference of the absolute error K_25 minus L25 has its upper end below zero and K_25 has r at or above 0.30 in each reader seed; else `KINCLOSE-K25-NOT-BETTER`.
* `KINCLOSE-EGO-TERM-HELPS` when the paired difference K_25 minus S_25 has its upper end below zero; `KINCLOSE-EGO-TERM-HURTS` when its lower end is above zero; else `KINCLOSE-EGO-TERM-NOT-SEPARATED`.
* Against the requirement of row 147: `KINCLOSE-REACHES-THE-REQUIREMENT` when the lower end of the interval of r of K_25 is at or above 0.9; `KINCLOSE-BELOW-THE-REQUIREMENT` when the upper ends of K_5, K_10, K_25 and K_50 are all below 0.9; else `KINCLOSE-REQUIREMENT-NOT-DECIDED`.
* The limit: `KINCLOSE-GAP-READ-IS-THE-LIMIT` when the lower end of r of B_25 is at or above 0.9 and the upper end of r of K_25 on the same frames is below 0.9; `KINCLOSE-WINDOW-IS-THE-LIMIT` when the upper end of r of B_25 is below 0.9; else no name on the limit.
* Reported with no name: every arm of section 2 at every W; E_25 beside K_25 (what the ego path alone gives); the error standard deviation of each arm beside the 1.24 m/s of row 147.

## 4. Controls (all have to pass before a name or a value is opened)

* k0 (definition): B_5 equals the label closing speed on the scored frames whose label run is five frames, largest absolute difference at most 1e-6 m/s (the ego terms cancel exactly; this checks the slope, the frame period and the join of dump and label).
* k1 (unit and clock of the ego speed, on the label gap): the mean absolute error of B_25 is lower than that of BS_25 with a separated paired interval, and lower than that of B_25 formed with the ego path shifted by +10 frames, by -10 frames, and with the sign of the ego path flipped (three planted faults, each has to read worse in the point value).
* k2 (range): v is finite on every frame used, at or above zero, with a mean between 1 and 40 m/s over the scored frames.
* k3 (frames): the scored frames, their count per reader seed and the pooled r and mean absolute error of the cell L25 equal the values of the result file of card ADD81 (2,572 frames for H 25 in each reader seed; cell r and error as numbers).
* k4 (planted): on a synthetic world of the tool's self-test with a lead of piecewise constant speed, an ego speed that changes and a gap read with noise, K_25 reads the planted closing speed better than S_25, and with a gap read that is noise alone K_25 does not read it (r inside a band around the value of E_25).

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136).

## 5. What follows (committed before the data)

* `K25-BETTER-THAN-OUTPUTS`: K_25 goes into the rule as a table (wrapper of card ADD83, form of card ADD84), at T1-kin/hb0.2, with the matched surrogate of card ADD85 and the arm E_25 beside it as the two inputs that do not know the lead, in a card of its own; no adoption by this fork.
* `GAP-READ-IS-THE-LIMIT`: the cost card says that the target is the gap read (a slope over 5 s of a perfect gap reaches the requirement) and gives the error of the reader's gap that the slope can carry.
* `WINDOW-IS-THE-LIMIT`: a slope over 5 s cannot reach the requirement even with a perfect gap; the cost card says that a closing read has to come from a shorter span, that is from the scene itself (tokens of the lead region), and not from the gap history.
* `K25-NOT-BETTER` and `EGO-TERM-NOT-SEPARATED`: the ego path is not the missing piece; reported as such.

## 6. Limits known in advance

T0 only; the target is the label closing speed, a slope over at most five frames of a label gap; a window estimate lags when the lead changes its speed, and those are the moments in which the rule needs the read, so a pass at T0 is not a pass in the rule (the rule test is the card that follows); the ego speed is the logged one, as the reader sees logged frames; windows and the primary arm are fixed here and nothing is selected on the scored frames; chosen after rows 139 to 147 on the same 141 clips.
