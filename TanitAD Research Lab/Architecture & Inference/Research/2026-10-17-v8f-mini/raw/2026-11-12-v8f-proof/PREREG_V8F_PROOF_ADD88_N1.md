# Note N1 of card PREREG_V8F_PROOF_ADD88 (KINCLOSE): the clips do not all have 101 frames

Written after the first chain of the card stopped on the pod in its ego step (2026-11-20 22:37:20Z, stop marker `EGO-FAILED`, fail closed, the self-test before it green) and before any read of the card: no control file and no result file of the card exists. The stop is logged in `TIMES_NOTE.md` before this note.

## What the card says and what the cache holds

Section 2 of the card writes "v(j): ego speed of the clip file of the evaluation cache (key `v`, 101 frames)". Read from the manifest of the evaluation cache after the stop: 129 clips have 101 frames, 10 have 100 and 2 have 104; the sum, 14,237, is the row count of the per-frame dump. The count 101 was the lead's, taken from the first clip of the manifest, and went into the helper's brief as a fact.

## Reading fixed here

* v(j) is the ego speed at frame j of the clip for every frame the clip has (100, 101 or 104). Nothing else of section 2 changes: the path s(j), the slopes, the arms, the scored frames, the read.
* No name, no rule of a name, no control and no bar changes.
* The tool: the ego file carries the frames per clip; the join refuses a clip whose ego series does not have exactly the frames the dump has for it (a join check that the first form of the tool did not have); the planted fault "path shifted by -10 frames" of control k1 holds the last frame of each clip (edge of the clip, not of the longest clip); the control record of k2 lists the frames per clip. The self-test world and the test file have clips of three lengths.
* The first pod root `/workspace/v8proof/kinclose` stays as it stopped; the chain runs again in the new root `/workspace/v8proof/kinclose_b` with the repaired tool.
