# Note N1 to card PREREG_V8F_PROOF_ADD73 (HANDSRC10K): the pod check of the wrapper ended with two sub-checks not passed; what they showed and their form of record

Written by the coordinator fork after the pod check of the wrapper ended (`integ10k/CHECK_HANDOVER_SRC_chk1.txt`, last line FAIL) and before any rollout of the card exists (time: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). The check ran on a tiny grid (3 clips, 27 windows, model seed 0); its runs are tool controls, not runs of the card.

## 1. What the check reported

Passed: with `own` the new wrapper returns every dump array of the parent wrapper (largest difference 0, 3 files) and the same pre-roll offsets; under `logged` the tick-0 plan, tau, P, v0 and the head outputs equal those of the fresh run (largest difference 0); under `logged` the first replan segment of the model track, in the own frame at t0, equals the fresh track to 3.38e-14 m over the 27 windows, while the same measurement on the own run differs by 62.4 m; `speed_logged` hands the logged speed and the own positions and headings (differences exactly 0 in the group that is not replaced), `path_logged` the reverse; the pre-roll offsets of all modes equal those of `own` exactly; `--t0-state logged` without a pre-roll is refused.

Not passed:
* 1b, "the tick files: every array identical": the only array that differs is `tag`, which holds the run name. Two runs have two names; the sub-check compared a name. (The same kind of fault as control l12 of the reader of card ADD70, note N3.)
* 2b for the array `fl`: the first replan segment of the floor arm of the pinned tool differs from the fresh run by 19.6 m under `logged`. The parameter replaces the state in the batch the model is handed; the floor arm is another planner of the pinned tool, its track is not handed that state by this wrapper, and no reader of this study reads it (the readers use the model track `cl`). The sub-check asked of the floor arm what the card asks of the model arm. Why the floor arm keeps the own state was not examined further.

## 2. Form of record

* 1b: every array of the tick files but the run name is identical (largest difference 0).
* 2b: the bar of 1e-3 m is for the model arm; the floor arm is reported without a bar.

The check file is not edited: `integ10k/make_check_handover_n1.py` writes `integ10k/check_handover_src_n1.py` from it with these two sub-checks changed and nothing else. It is run on the pod with `--analyze-only` on the run folders the first check left (no new rollout), and the queue of the card starts only on its last line PASS. Control c0 of the card (the same identity on all 1,271 windows and both seeds, from the dumps of the card's own runs) stays as written.

## 3. A statement added before any rollout

The card reads the model track only. Nothing in this arm says anything about the floor arm of the pinned tool under the three modes.
