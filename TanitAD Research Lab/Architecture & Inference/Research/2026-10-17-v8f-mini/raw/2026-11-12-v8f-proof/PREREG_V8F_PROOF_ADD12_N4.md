# Note N4 to PREREG_V8F_PROOF_ADD12 (SPATIAL): the first stage-0 read missed control s3; stage 0b, one instrument change, written before its run (2026-11-13)

## 1. The first read, as the card requires it to be reported

Stage 0 ran on 2026-11-13 13:20 to 13:27 (`spatial/s0/RESULT_S0_first.json`, `VERDICT_S0_first.md`; 9,321 bank frames, 1,809 windows, 441 windows with a lead at both times, 139 episodes).
Controls: s1 PASS (largest R2 under permuted labels -0.017, bar 0.02), s2 PASS (presence: 1.000 from the grid state, 0.518 from the mean state), s2p PASS (position: 1.000 and 0.495; the solver lemma of group P predicted 0.50), s4 PASS (0.0),
**s3 MISSED**: its first half holds (the ego-only gap R2 reads 0.1327 against the FULLRES value 0.133), its second half does not: the R2 of the true future state is below the R2 of the predicted state as point values (grid state 0.0204 against 0.0255; mean state 0.1776 against 0.1795).
By the card the stage is not quoted: **no outcome name of stage 0 is printed from the first read.**

## 2. Diagnosis (MEASURED on the first read; these numbers are diagnostics of the instrument, not outcomes)

* The r-b probe was fitted on the training-fold windows with a lead at both times only (N3b): about 350 windows for a state of 640 numbers. On the same kind of state and target the r-a probe, fitted on about 2,000 bank frames, reads the lead gap at R2 0.398; the r-b probe reads it at 0.02 from the TRUE future state. The r-b probe is starved, not the state.
* With a starved probe the order of (i) and (iii) is noise: the paired interval of (iii) minus (i) is [-0.102, +0.088] for the grid state and [-0.078, +0.070] for the mean state. All r-b intervals of the first read span 0 widely.
* r-a does not depend on the r-b probe. The lead decision N3b (which rows fit the r-b probe) is the cause; it was mine.

## 3. Stage 0b: what changes, fixed now, before any 0b number

I have seen every number of the first read. The change below is therefore a post-hoc repair of the instrument and is labelled as such wherever a 0b number is quoted.
* **One change, the rows that fit the r-b probe.** In each fold the lead-gap probe of r-b is the r-a probe of that fold and state: fitted on the true states of all training-fold bank frames with a finite lead gap (the card: "the probe is fitted on true future states of the training folds"; a state at a later time is a state), then applied unchanged to (i) the predicted state, (ii) the present state, (iii) the true future state of the test windows.
* **A second, stricter floor is added; none is removed.** (iv) stays as built (fitted on the windows). (iv-b) is the ego-only probe fitted on the same training-fold bank frames as the state probe and applied to the TRUE ego vector at t+2.0 s, which is more than an ego-only predictor could know. The outcome rule uses the HIGHER of the two floors (pooled R2).
* **No control is weakened.** s1, s2, s2p, s3 (both halves, as point values, as the card writes them) and s4 stay. A new consistency control s5 is added: under the change, the (iii) values of the test windows equal the r-a out-of-fold predictions at the same bank frames (largest absolute difference 0.0); it proves the window rows are aligned.
* r-a, the states, the predictor, the folds, the bootstrap and every outcome rule are unchanged; the r-a numbers of 0b equal those of the first read by construction.
* If s3 misses again, stage 0 stays unquoted, the diagnostic numbers are reported as such, and stage 1 runs as the card says (a negative or unquoted stage 0 does not close it).
