# Note N1 to card PREREG_V8F_PROOF_ADD66 (STRATROLL10K): a descriptive split of the rollout read into the head's own output and the pose it is placed in

Written by the coordinator fork after the read of the card with Amendment 1 (nine banked runs; controls s1 to s9 pass; the names stand as printed and are reported as written) and before any quantity of this note has been computed (time: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). The note names nothing and changes no name. Tier as the card.

## 1. Why

`STRATROLL10K-STRATEGIC-DEGRADES-ON-ROLLOUT` reads the route goal as the rollout places it in the world: the head's checkpoints, given in the frame of the executed pose, against the route label of the nearest logged step. In this tier the camera frames are the logged ones, so two things can move that number: (a) the head's own output changes because its ego history and route countdown come from the rollout, and (b) the executed pose (position and heading) differs from the pose the camera stood at, so an unchanged output lands elsewhere in the world. The card's committed sentence for this name ("the strategic head too responds to the model's own rolled history") is a statement about (a) alone. The sizes in the read (the degradation grows about in proportion to the arc, at 1.2 s already with the car 0.18 m from the logged track on average) could come from a heading offset of the executed pose as well as from the head. The split below says which, with rules fixed here.

## 2. Quantities (all from the same tick files and the same join; no model, no rerun)

* `pos_cam`: the other session's T0 formula at the head's own tick step: the distance between the raw checkpoint `s_ckpt` and the route label `rc` of the logged step j minus `age_s` (valid where `rc_ok`), both in the head's frame, as if the head stood where the camera stood. At the first tick it equals `pos` (control d1: largest difference at most 1e-9 m).
* The pose offset at the head's tick step: the executed pose `s_pose` minus the logged pose of step j minus `age_s`, in the logged frame: along, sideways, heading; mean and median of the absolute values, and the shares of rows with a heading offset above 5 and above 20 degrees. At the first tick all are 0 (control d2).
* ROLLCAM1, ROLLCAM2: as ROLL1, ROLL2 of the card with `pos_cam` in place of `pos` (same reference windows, same estimator), per run; beside them ROLL on exactly the rows valid for both.
* The pairs of the card at the third tick on `pos_cam` (K, causal, the rule forms, the backward and no-acceleration readouts against `R0`).
* The raw output distance between two runs at the same tick: the mean distance between their `s_ckpt` per arc (the camera frames are the same, so this is the head's response to the difference in ego history alone).
* Mean and median of `pos` and `pos_cam` per run, tick and arc; ROLL2 of `R0` by bins of the heading offset (up to 2, 2 to 5, 5 to 20, above 20 degrees) and the mean heading offset by bins of the speed at the tick (below 2, 2 to 8, above 8 m/s), with counts.
* `lat` is not quoted as a level from the read of the card: its validity rule lets rows through in which the logged car stands still short of the arc (the nearest point of the track is then its end), which inflates its mean; this note reports `lat` on the rows with a valid label only.

## 3. Reading rules (fixed before computing)

* HEAD: if ROLLCAM2 of `R0` reads WORSE at two or three arcs, the head's own output degrades on the rollout and the card's sentence about the strategic head holds, with ROLLCAM2 as its size. If ROLLCAM2 reads WORSE at no arc, the name DEGRADES is carried by the pose alone and the row says so in these words: the head's output keeps its T0 quality in the camera frame; what moved is the pose the rollout places it in. One arc: reported as mixed, with the table.
* SHARE: per arc, ROLLCAM2 divided by ROLL2 on the same rows is quoted as the share of the degradation that is in the head's own output; the rest is the pose offset meeting logged camera frames, a property of this tier (no re-perception) and of how the kinematic executor derives its heading.
* READOUT on the head: the K and causal pairs on `pos_cam` say whether the readout changes the head's output or only the pose. Whatever they read, the names of the card stand; the row quotes both.
* A control of this note fails (d1, d2): the split is not reported and the tool is repaired.

## 4. Limits

`pos_cam` scores the output against the label of the camera's pose even where the route countdown, dead-reckoned from the rollout's own motion, tells the head correctly that the car is elsewhere; a head that used that input well would be marked down by it. The split is descriptive: it was written knowing the read of the card, and it does not test the cause of a heading offset (the executor's heading comes from the tangent of the executed path, with no vehicle model).
