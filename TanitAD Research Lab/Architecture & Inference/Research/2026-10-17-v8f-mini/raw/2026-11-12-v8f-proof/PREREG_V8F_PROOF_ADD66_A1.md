# Amendment 1 to card PREREG_V8F_PROOF_ADD66 (STRATROLL10K): the route label is the programme's own label of the anchor step

Written by the coordinator fork after the tools of the card were built and its controls-only run ended on the pod, and before any outcome cell of any dump has been computed (the controls-only mode computes none; time of the freeze: the line of this amendment in `TIMES_NOTE.md`; hash in the sidecar file). The card stays as frozen; this amendment replaces the label rule of its section 3, the definition of the cell `along`, and controls s3 and s4.

## 1. What the controls-only run found (MEASURED, no outcome cell; `integ10k/read_stratroll/CONTROLS_ONLY_STRATROLL.txt`)

Controls s1, s2, s4 and s6 pass on all 18 banked tick files. Control s3 as frozen FAILS and stays on record as failed: the label rule the card cites (the loop in `v8f_mini_data.py`) belongs to the synthetic-drive builder. The route label the models were trained on and the other session's evaluator scores against is the label of the v9 release (a route smoothed in arc length, with its own validity check), stored per logged step in the evaluation join (`join_eval141.npz`, fields `rc`, `rc_ok`, `off`, `n`). The card-rule label differs from it by up to about 14 m with 1,277 validity disagreements over the first-tick rows. The card would have scored the strategic head against a label it was not trained on.

## 2. The label (replaces the third bullet of section 3 of the card)

The route label of a tick row is the programme's own route label of its anchor step: `rc` and `rc_ok` of the evaluation join at that logged step of that clip, given in the frame of the logged pose at the anchor step and put into the world frame with that logged pose. A row is valid at an arc if `rc_ok` is set there and the anchor step lies inside the clip's label rows. No label is recomputed by the reader. The anchor rule of the card is unchanged.

Consequences inside the card, nothing else changes:

* `pos` and `head` at the first tick are then exactly the checkpoint position and heading errors of the other session's evaluator (`checkpoint_errors` in `v8f_consistency.py`) for an output of age 0; at the later ticks they are the same errors against the label of the logged step nearest to where the car is.
* `along` (reported beside, not a named cell): the arc along the logged route of the foot of the checkpoint minus that of the foot of the label point, so a checkpoint on its label reads 0 under any label. Valid if both feet are valid.
* `lat` (beside): unchanged, the distance to the polyline of the logged positions. The label lies on the smoothed route, so a checkpoint on its label can read a small `lat` in a curve; this is said where `lat` is quoted.

## 3. Controls (replace s3 and s4 of the card; s1, s2, s5 to s9 unchanged; all must pass)

* s3 (label path, known value from the programme's own code): on the first-tick rows, `pos` and `head` as the reader computes them (world frame, label from the join through the logged pose) equal `pos_err_m` and `head_err_deg` of the other session's function `checkpoint_errors` called on the raw `s_ckpt` and the join's `rc`, `rc_ok` of step t0 (largest difference at most 1e-3 m and 1e-2 deg; the same rows invalid). Only differences are printed. The result of s3 as first frozen is kept in the result file as failed.
* s4 (synthetic known values): the geometry checks of the card on a straight track and on a circle with a synthetic label; in addition for the new label path: a label given in the frame of a logged pose with a known heading lands at the analytic world position and heading; a step outside the clip's label rows reads invalid; a checkpoint on its label reads `pos` 0, `head` 0 and `along` 0; a checkpoint 4 m farther along a straight track reads `along` 4 m. Planted faults that must fail: the label taken one arc too far, and the label put into the world frame with the heading of the wrong sign.

## 4. Limits added

The label is the programme's smoothed route label, valid where its builder set `rc_ok`; rows whose anchor step has no valid label are not scored and are counted. The first reading of the dump structure and of the label code by the lead found the wrong builder; the control the card carried for exactly this caught it before any number was read.
