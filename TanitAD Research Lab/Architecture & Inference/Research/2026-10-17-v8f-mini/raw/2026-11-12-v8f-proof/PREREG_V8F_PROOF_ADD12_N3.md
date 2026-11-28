# Note N3 to PREREG_V8F_PROOF_ADD12 (SPATIAL), stage 0: reading choices fixed before the full read (2026-11-13)

Written before the full stage-0 read on true labels was launched; no outcome number of the card existed. What existed: unit tests on synthetic data and a 12-episode smoke with the lead labels permuted across rows, which prints shapes, timings and the label-free controls only (`spatial/s0/S0_DEFS.md` R17).
No question, state, predictor, outcome name or bar of the card moves. Definitions: `spatial/s0/S0_DEFS.md` R1 to R18.

* N3a, control s2 in two forms, BOTH must pass. The build found, label free, that a planted vector of the f5 size (the median token norm, about 100 natural standard deviations along its direction) is read from the mean state as well (AUROC 0.9996 for its PRESENCE on the smoke). That agrees with the solver lemma of group P: a mean over cells keeps presence and count (VC-P7) and loses position (VC-P3, VC-P4).
  * s2 (presence, as built): the vector at 12 natural standard deviations in one cell, in half of the rows; target = planted or not. The size was fixed by a label-free calibration.
  * s2p (position, lead rule R18): the vector at the f5 size in every row, in a cell or in its mirror cell of the same grid row; target = which of the two. No size is tuned; the lemma predicts 0.50 for the mean state at any size.
  * Bars for both, as the card writes them: AUROC above 0.95 from S0g, at or below 0.60 from S0m. Smoke values, label free: s2 1.000 and 0.521; s2p 1.000 and 0.455.
* N3b, rows. r-a is read on the bank frames of FULLRES (so that the second half of control s3, the ego-only gap R2 of 0.133 within 0.03, compares on the same rows); the predictor, r-b and the horizon use the eval-141 windows; r-b uses the windows with a lead at both times, and its probe is fitted on those windows of the training folds.
* N3c, actions. The ridge predictor takes the logged actions of the 10 steps standardised on the training windows (the action scaling of the model needs a checkpoint, and stage 0 uses none).
* N3d, r-c (pixels) is not run in stage 0: the frame targets and decoders of LATVIS were deleted as re-creatable and need about 32 minutes of CPU and the GPU. It stays a read of stage 1.
* N3e, "beats" in the outcome rules means: the difference is above 0 and its paired episode-cluster bootstrap interval (B 2,000) excludes 0.
* N3f, if a control misses, the numbers are written, no outcome name is printed and the job ends with code 2, as the card requires.
