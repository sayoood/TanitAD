# PREREG_V8F_PROOF_ADD99 (CROPREAD): pilot of a new read of the lead's gap from camera-frame crops

Fork v8proof, 2026-11-21. Tier T0 (a probe on logged frames; a diagnostic of a reader, never driving performance). Evidence class of every value read under this card: MEASURED (ours, with the artifact path). The PI's go: "Follow your recommendation" (2026-11-21), on the cost card `COSTCARD_CLOSING_READ_2026-11-20.md` sections 0, 6 and 8: the smallest pilot first, the 141 evaluation clips plus 500 train clips, two arms, PASS and FAIL as committed there. Host by the fork's choice inside that go: the fork's own pod, which is idle. This card is frozen with its hash file before any tool, staged file, token or fit of it exists.

## 1. Question

Does a gap read from crops of the camera frame at a finer scale than the present reader's pooled tokens give a gap whose own slope reads the label closing speed at the requirement of r 0.9 (row 147), as row 152 reads the present reader (r +0.261 [+0.152, +0.362])?

## 2. Data

* Evaluation: the 141 clips of `join_eval141.npz`; the scored frames are the rows of the present reader's dumps `closeread/dump/both/reader_s0_score.npz` and `reader_s1_score.npz` (the frames of card ADD81), unchanged.
* Train: from `data/inputs/join_train.npz` (4,572 clips) the clips that the present reader's rule does not hold out (`int(sha256(clip id)[:8], 16) % 10 != 0`) and that have at least 10 frames with a valid lead gap (`lead_valid == 1`, finite, 0 to 60 m); MEASURED before this card on the pod: 1,679 clips. Ordered by the sha256 hex of the clip id, the first 500 are the train list. The first 450 of them are fit clips, the last 50 are selection clips (they choose the epoch and nothing else). A clip that cannot be staged is replaced by the next of the order, up to position 520.
* No evaluation clip is in the train join (MEASURED before this card: overlap 0). The selection uses the label's presence of a lead; that is a choice of training targets and is admissible.
* Source of the frames: the camera videos of the NVIDIA PhysicalAI-AV source at the revision the caches were built from, fetched by the pod's own stage step (the route of stage A of this fork). Videos and frames stay on the pod; none goes to the repo.

## 3. Windows (crops)

A window is a square part of the camera frame given by its centre direction (azimuth, elevation) and its angular size, sampled to 256 by 256 pixels with the projection routine of the cache's own frame build (the rectified frame; azimuth 0 is its centre column, elevation 0 its horizon row). Two sizes: W32 (32 degrees) and W16 (16 degrees).

* Arm D (deliverable; the position comes from nothing but the fixed geometry): W32 centred at azimuth 0, elevation -6 degrees; W16 centred at azimuth 0, elevation -2 degrees. The same windows on every frame of every clip.
* Arm B (bound; the label says where to look): W32 and W16 both centred on the direction of the centre of the lead's box in the label tracks. On frames with no lead in the label the windows of arm D are used. The window sizes are fixed, so the apparent size of the lead stays in the crop; no number of the label enters the reader's input.
* Arm B is a bound and a source of training targets only (vision-only rule at inference). PASS and FAIL are read on arm D.

## 4. Encoder and tokens

The frozen DINOv3-L of the caches (the weight file on the pod, sha256 dcb2e45127cc..., checked in `hierrep/dinov3_sha_check.json`), with the cache builder's own normalisation, on each 256 by 256 window: 16 by 16 patch tokens. Kept per window and frame: the class token and the patch tokens mean-pooled 2 by 2 to 8 by 8 (65 tokens of width 1024, float16). All frames of the 641 clips, both arms.

## 5. The gap head (fixed before data)

* Input per frame: the 130 tokens of the two windows. Layer norm, linear map to width 256, a learned position vector per token, one learned query that reads the tokens through one cross-attention layer (4 heads), giving one vector z of width 256 per frame; token dropout 0.1 in training.
* Head H1 (one frame): MLP 256 to 256 to 1 on z of the frame.
* Head H5 (five frames, causal, the history length of the present reader): MLP 1280 to 256 to 1 on z of the frames f-4 to f; a frame is used when the five frames exist in the clip.
* Output: gap = 60 times the sigmoid of the output, in metres. Loss: smooth L1 (beta 1 m) against the label gap on the frames with a valid lead gap.
* Fit: AdamW, learning rate 1e-3, weight decay 0.05, batch 256, 40 epochs, one warm-up epoch then cosine to 0; the epoch with the lowest loss on the 50 selection clips is kept. Fit seeds 0 and 1. One fit per arm, head and seed: eight fits, plus the fits of control c5.
* The fit sees train clips only. Nothing of the evaluation clips (tokens, labels, counts) enters a fit or a choice.

## 6. Read

For each arm, head and fit seed a dump in the form of the present reader's dumps on the same rows: the gap is the head's output, the fields the pilot has no head for carry a constant (presence 1, closing 0), every other field is copied from the present reader's dump of the same seed index. The estimator is the one of row 152, tool `kinclose/kinclose4.py` 9a734658232bc327, unchanged, called through a wrapper that swaps the dumps: the two fit seeds take the place of the two reader seeds; r of the run-rule slope of the read's own gap over at most 10 frames (arm name RG10 of the tool, all scored frames) against the label closing speed, interval by the tool's bootstrap over evaluation episodes.

Names of the card, read on arm D with head H5 (the deliverable form), as committed in the cost card:

* `CROPREAD-PILOT-PASS`: the lower end of r is at or above 0.9. Then a next card reads the rule form with that slope in T1-kin/hb0.2 (kinematic rollout on the model's own actions).
* `CROPREAD-PILOT-FAIL`: the upper end of r is below 0.9.
* `CROPREAD-PILOT-NOT-DECIDED`: anything between. The PI gets the number before any rollout.

Names beside it (they do not change the card's name):

* Per form of the four (D-H5, D-H1, B-H5, B-H1): `CROPREAD-<arm>-<head>-REACHES`, `-BELOW`, `-NOT-DECIDED`, by the same rule on RG10.
* The cut, on D-H5: the ratio of the standard deviation of the gap error of the pilot to that of the present reader on the frames both measure, pooled over the two seeds, interval by a paired bootstrap over evaluation episodes (2,000 draws): `CROPREAD-CUT-BY-2-REACHED` when the upper end of the ratio is at or below 0.5, `CROPREAD-CUT-BY-2-NOT-REACHED` when the lower end is above 0.5, else `CROPREAD-CUT-BY-2-NOT-DECIDED`. The same ratio for B-H5 is printed with the name `CROPREAD-BOUND-CUT-BY-2-...`.
* Printed with no name: the gap error by range of the label gap (the bins of row 152); the split of row 154 (tool `kinclose/kinclose6.py` f5be6f8077f6f563 through a wrapper: the part about the clip mean, the change from frame to frame, the scale error per clip); the three parts of card ADD98 computed on the pilot's raw gap (clip offset, slow drift, fast part), beside 5.68, 6.31 and 1.65 m of the present reader as the rule reads it (a different object: no map, no minimum; stated with the values); the values per fit seed; the loss on the selection clips.

Reading committed in advance: PASS on D-H5 opens the rule read. FAIL is reported as FAIL of the pilot with the cost card's consequence (the candidate keeps the form without a closing input) stated beside it. Whatever the name, the diagnostics name the next lever and the fork prepares its card: if B-H5 reaches the requirement or the cut while D-H5 does not, the missing piece is finding the lead in the image (a steered window); if D-H5 reaches the cut by 2 without the requirement on r, the rule read of the pilot's gap in the form of row 158 is the next card (the rule's own plane asks for the cut, row 158); if neither arm reaches the cut, the read is limited by the 500 clips or by the head, and a larger train subset is the PI's call.

## 7. Controls

All of c1 to c7 have to pass before any name or any value of this card is opened; they are logged in `TIMES_NOTE.md` before the names. With one of c1 to c7 failed nothing of the card is opened, the card is reported as failed with no name, and a changed form is a new card. Control p1 guards arm B only (the map is fixed here): with p1 failed and c1 to c7 passed, arm D is opened, arm B stays unopened and is reported as failed with no value.

* c1 (same frames, same encoder): for the 6 evaluation clips that come first in the order of the sha256 hex of the clip id, the full frame built from the staged video by the cache's own build route and encoded by the pilot's encoder path, pooled as the cache pools (8 by 20), is compared with the cached tokens of the clip. Pass: the median over all frames of the cosine between the tokens of a frame and the cached tokens of the same index is at or above 0.98, and on the frames with an ego speed of at least 2 m/s the cached frame of the clip that matches best is the frame of the same index on at least 95 percent.
* c2 (the wrapper is the tool): through the wrapper, the present reader's dumps give the values of `GAPRANGE_RESULT.json` 9965504cfde0dae4 for RG10 over all scored frames (r and both ends) and for the gap error over all frames (sd and both ends) within 1e-6.
* c3 (the read can say both): through the wrapper, a dump with the label gap plus a white error of 0.25 m gets the name REACHES, and one with the label gap plus a white error of 8 m gets the name BELOW.
* c4 (no evaluation data in a fit): the train list and the evaluation list have no clip in common; every fit's record lists the clip ids it read, all inside the train list; the sha256 of the train list is written before the first fit.
* c5 (the fit makes no result of its own): head H5 on arm D fit with the label gaps permuted across the train frames (one seed) does not get the name REACHES, and its gap error on the evaluation frames has a standard deviation of at least 0.8 times that of the label gap on the same frames.
* c6 (kit tests): the tests of the kit pass in the lead's run on the dev box before the read, with planted faults that the controls catch (a frame shift of one in the window builder, an evaluation clip in the train list, a mirrored azimuth).
* c7 (complete): all 141 evaluation clips and at least 490 train clips are staged and encoded for both arms with the frame count of the join; the clips left out are listed.
* p1 (the windows of arm B are on the lead): (i) the gap recomputed with the join build's own rule from the track the pilot takes the box centre from equals the join's lead gap within 0.5 m on at least 98 percent of the frames with a valid lead of the 641 clips; (ii) on the frames with a gap of at least 15 m, an absolute azimuth of the box centre of at least 3 degrees and an absolute path curvature of the ego of at least 1/300 per metre, the sign of the azimuth agrees with the sign of the curvature on at least 75 percent, with at least 200 such frames (with fewer the clause counts as failed); (iii) the elevation of the bottom of the box is below 0 on at least 98 percent of the frames with a valid lead and a gap below 20 m. The box comes from the obstacle labels of the source, fetched by the stage step for the 641 clips (the pod holds the join's gap and no box).

## 8. Four metric families

LONGITUDINAL: the read is an input of distance-keeping; at T0 it is scored as a gap error and as r against the label closing speed, a diagnostic of the reader. LATERAL, TACTICAL, STRATEGIC: not read under this card, since no rollout and no planner output exists at T0 for a gap reader; they are read with the rule form in T1-kin/hb0.2 under the next card if this one passes, STRATEGIC included as in cards ADD95 to ADD98.

## 9. Cost and limits

* ESTIMATED: about 8 GB of video download, four windows per frame for 641 clips (about 260,000 crops), under 1.5 GPU-h in total with the fits, about 35 GB of token files on the pod's disk (395 GB free, MEASURED). The PI's go covers 0.35 to 1.46 GPU-h per crop form and 13.4 GB at the full grid; the two sizes per arm are the fork's choice inside it and are stated here.
* Limits: T0 only; 500 train clips of one selection; one head form with two fit seeds (the interval answers for other evaluation episodes, the two seeds show the spread of the fit, nothing here answers for another train subset); the label closing speed is itself a slope of the label gap; arm B uses the label at inference and is no candidate form; no re-perception.
