# PREREG_V8F_PROOF_ADD92 (GAPREQ): what kind and size of gap error a closing read tolerates, with the runs found from the noisy gap itself

Tier T0 (reads on logged frames; no rollout; a diagnostic, never a statement on driving performance). Evidence class of every value it will give: MEASURED. Label gap only, with planted error: no reader output, no ego speed, no token, no fit, no new encode. Written after the read of card ADD91 and before any tool of this card exists.

## 1. Why, and what the lead knows

The cost card to the PI on a new read of the lead needs a pass bar for its pilot. Row 151 (card ADD90) gives a bar of about 1 m of absolute gap error, with two limits that it states: the runs come from the clean label gap, and the error is absolute. A camera read of range has an error that grows with range, and a part of it is a scale error that stays with one lead; a slope does not see a constant offset and sees a constant scale error only as the same share of the slope. So the bar may be far from 1 m in the terms a camera read is built in. This card plants proportional and absolute error in three time courses and finds the runs from the planted gap itself, the way a system with no label has to.

Known to the lead: rows 151 and 152 (the ladder on exact runs; the reader's gap error by range; a slope of the reader's own gap with the run rule reads r +0.26, and the reader's own runs read with a smaller error than the label's runs on the reader's gap). Not known: any value of a planted proportional error, of a per-clip constant error, or of a planted error with its own runs.

## 2. Frames, planted gaps, arms

* Frames: the scored rows of card ADD81 at 25 frames (2,572 per reader seed; the rows of the two seeds are equal here since no reader output enters). Target: the label closing speed of the row.
* Unit noise N, one draw per clip and kind, shifted and scaled to mean 0 and standard deviation 1 exactly over all frames with a label gap: W white; R first-order autoregression of 0.94 per frame (the time course of the reader's gap error); C constant over the clip (one value per clip).
* Planted gap: ABS, label gap + s * N with s in 0.5, 1, 2, 4 m (codes S050, S100, S200, S400), kinds W and R; PROP, label gap * (1 + q * N) with q in 0.025, 0.05, 0.10, 0.20 (codes P025, P050, P100, P200), kinds W, R and C. Twenty planted gaps.
* Arms per planted gap and W in 5, 10 (frame period 0.2 s): OWN, the least-squares slope of the planted gap over the run found from the planted gap itself with the label's rule (going back while a value exists and the step is at most 3.0 m, at most W frames; a run below 3 frames gives 0.0, as the rule's source does); EXACT, the slope of the planted gap over the run found from the label gap.
* Bins as in card ADD91 (NEAR: label gap below 20 m), used for reported values only.
* Intervals: two-level bootstrap as in card ADD91, 2,000 draws, seed 92.

## 3. Names (frozen)

* Ladder names, one per run form (OWN, EXACT), kind and type (W-ABS, R-ABS, W-PROP, R-PROP, C-PROP) and W (5, 10), on all rows: `GAPREQ-<run>-<kind>-<type>-W<W>-UP-TO-<code>` with the code of the largest level such that this level and every smaller one have the lower end of the interval of r at or above 0.9; `GAPREQ-<run>-<kind>-<type>-W<W>-BELOW-<code of the smallest level>` when the smallest level does not. Twenty names.
* Run finding at the level R-ABS S100, W 10, paired difference of the absolute error of OWN minus EXACT on the rows where both have a value: `GAPREQ-RUN-FINDING-COSTS` when the lower end is above 0, `GAPREQ-RUN-FINDING-HELPS` when the upper end is below 0, `GAPREQ-RUN-FINDING-NOT-SEPARATED` otherwise.
* Reported with no name: every arm with r, interval, mean absolute error, standard deviation of its error and the share of rows with a run below 3 frames, on all rows and on NEAR; the realised absolute gap error in metres of each PROP level, over all frames and over the frames below 20 m.

## 4. Controls

All five have to pass before any name or any value of this card is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* h0: the scored rows per reader seed equal those of card ADD81 (2,572 each).
* h1 (identities): at level 0 (the label gap itself) OWN and EXACT are equal on every row for both W, the run of 5 equals the label closing speed on every scored row (largest absolute difference at most 1e-6), and an absolute error that is constant over a clip (kind C, 4 m) leaves the OWN and the EXACT slope of every row unchanged (largest absolute difference at most 1e-9).
* h2 (noise): each unit noise has mean 0 and standard deviation 1 within 1e-9 over the frames with a label gap; lag-1 autocorrelation within 0.05 of zero for W, between 0.89 and 0.97 for R, and at least 0.999 for C.
* h3 (planted spread, exact): for every ABS level the standard deviation of planted minus label gap is s within 1e-9; for every PROP level the standard deviation of planted over label gap minus 1 is q within 1e-9; every planted gap is above 0.
* h4: the self-test of the tool on the pod before the read, on a planted world: a noise that is not scaled, a proportional level planted as an absolute one, a dropped step gate and a wrong frame count each turn their control red; no name is written and every value is held back when a control fails.

## 5. What follows, fixed now

* The pass bar of the pilot in the cost card is stated with the OWN names of kind R at W 10: the largest absolute level and the largest proportional level that hold r 0.9, each with the realised metres of the proportional level beside it.
* The OWN name of kind C-PROP at W 10 says how much scale error that stays with one lead a closing read carries; if it reaches P100 or P200 while kind W-PROP stops at a lower level, the cost card says that the pilot has to hold the frame-to-frame jitter of the gap, and not its absolute scale, under the bar.
* OWN names below EXACT names at the same kind and type: finding the changes of the lead from the noisy gap is part of the cost, and the cost card asks the pilot for a lead identity over time beside the gap.
* Every ladder BELOW its smallest level for OWN at kind R: no gap read of the sizes planted here carries the closing read through its own runs; the cost card then names a direct read of the closing speed from the image of the lead as the alternative to a better gap, without proposing it.

## 6. Limits

T0 only; a pass here is not a pass in the rule, and the requirement of r 0.9 comes from the plane of row 147 with the reader's gap as the rule's distance input. The planted error is Gaussian with one draw per clip and kind and one coefficient per kind; a real read mixes the three kinds. The gate of 3.0 m is the rule's own and is not tuned to the planted noise. The target is the label closing speed, a slope over at most five frames of the label gap, so level 0 holds by construction (h1). Designed after cards ADD89 to ADD91.
