# Note N3 to card PREREG_V8F_PROOF_ADD73 (HANDSRC10K): the first read stopped on control b5 for the two runs that hand the logged speed; cause and form of record

Written by the coordinator fork after the first read of the card ended with part 1 not passed and before any value of part 1 exists (the reader writes none when a control fails; time: the line of this note in `TIMES_NOTE.md`, after the line that logs the failure; hash in the sidecar file).

## 1. What the read reported

`integ10k/read_handsrc_decomp/pod/READ_HANDSRC_DECOMP.json` (sha256 prefix 0008d75cea492dde): part 1, 20 of 22 controls passed. Not passed: b5 for the run LOGGED (share of windows within the bound 0.817 and 0.811, model seed 0 and 1) and for the run SPEED-LOGGED (0.818 and 0.812); bar 0.99. b5 passed for the own run and for PATH-LOGGED. c0, c1, c2, q1, h1, the bank control and the other controls of the split passed. No name and no value of part 1 was written. Part 2 (the family read) passed its three controls.

## 2. Cause

Control b5 of the split tool says: the first executed point of the aged track lies within 0.5 m plus one step of travel of the offset position, and it takes one step of travel as the own speed at the tick times 0.2 s. That is right for a run whose planner is handed the own speed. In the modes `logged` and `speed_logged` the planner is handed the logged speed at t0 (that is the arm), so the first step of its plan is as long as the handed speed makes it, not as long as the own speed would. The control asked of a run with a handed speed what holds for a run with the own speed. (The same kind of fault as sub-check 2b of note N1: a check written for one arm and applied to another.)

What supports this before any new read: the two runs that fail are exactly the two that hand the logged speed, and the two that keep the own speed pass; c0 passed, which says that the first replan segment of the LOGGED run equals the fresh track moved to the offset position to 1e-3 m on every window, so its first point is the first step of the fresh track from the offset position.

## 3. Form of record

b5: the first executed point of the aged track lies within 0.5 m plus one step of travel of the offset position for at least 99 percent of the windows, per model seed, where one step of travel is the speed handed to the planner at the tick (the field `v0` of the tick-0 row of the tick file of the run, which records the speed in the batch the model read) times 0.2 s. This form is used for all four runs. For the own run and PATH-LOGGED the handed speed is the own speed; the largest difference between the two is written beside. The slack of 0.5 m, the bar of 0.99 and the cross-check with the offsets file are not changed. The result of b5 in the form of the split tool (own speed) is written beside for every run, not judged.

If b5 in this form does not pass for a run, no name and no value of part 1 is written and that is logged.

## 4. The re-read

Reader `integ10k/read_handsrc_decomp.py` with this one control changed (hash logged in `TIMES_NOTE.md` before the run), new output folder `out/read_handsrc_decomp_n3` on the pod, pod script `integ10k/podread_handsrc_decomp_n3.sh` (the script of the first read with the folder name changed). Part 2 is computed again by the same code; a control p2 is added: every family name, headway share and family cell of part 2 of the re-read equals that of the first read (1e-9). The first read stays in its folder as read.

The queue of card ADD74 takes the re-read as its start condition (its queue file is written again before it has ever been started, and its hash logged). Card ADD74 uses b5 in the form of record of this note.
