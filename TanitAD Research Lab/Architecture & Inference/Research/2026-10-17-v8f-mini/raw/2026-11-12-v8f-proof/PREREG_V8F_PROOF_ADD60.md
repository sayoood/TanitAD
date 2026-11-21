# Card PREREG_V8F_PROOF_ADD60 (HALFSCALE-LEAD): the lead probe on half the corpus against the whole corpus, with the kit of card ADD45

Card of the v8F proof study, written by the coordinator fork. Frozen before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Probe only, no model is trained, no spend beyond the pod of this fork (the probe trains beside the rollout queue; at most two GPU jobs at a time). Tier: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance.

## 1. Why

* Card ADD45 asks whether more data lifts the read of the lead gap from the pooled tokens: the corpus (4,512 clips) against the union with the 10k clips (14,512). Its run waits for a slot on the other session's pod, and its kit and reader have not run end to end.
* The same question has a second point that needs no other pod: half of the corpus against the whole corpus, at the same number of steps. Run with the exact kit of card ADD45, it is also the first end-to-end run of that kit and of its reader.
* LEADHEAD stage 0b (MEASURED, compact paper): the learning curve of a lead head over training clips had not flattened. This card reads the same thing for the probe of card ADD43 at matched compute.

## 2. Arms

The kit of card ADD45 with its notes N1 and N2 (`dscale_lead/`, files as landed with commit 06f205e6; the kit script, the tool, the checks and the reader unchanged), run on the pod of this fork with these inputs:

* "union" of the kit = the study's corpus cache (4,572 training clips); "corpus list" of the kit = the first 2,286 clip ids of its training manifest in manifest order (file `half.txt`, its sha256 in `TIMES_NOTE.md` with the freeze line); the lead file = the corpus lead file of the dev box (sha256 f0440deb3884ee35), given for both lead inputs of the kit; join file, tracks and P10 prior = those of the study; tree = the shipped tree of card ADD45 (pins unchanged).
* Arms by the kit's folders: `runC` = HALF (2,286 clips, seeds 0 to 3), `runU` = WHOLE (4,572 clips, seeds 0 to 3), `runU2` = WHOLE with twice the steps (seed 0, beside), `runCSH` and `runUSH` = permuted gap labels (seed 0). Presence and gap in the loss, closing weight 0, 19,164 steps (38,328 for `runU2`), as card ADD45.
* Evaluation: the 141 evaluation clips of the study, the 429 scored windows of the 48 evaluation episodes of card ADD43, through the reader of card ADD45 with the corpus count set to 2,286.
* Budget: ESTIMATED 1 to 3 GPU-h on the pod of this fork beside the rollout queue. The kit's smoke measures it; a projection above 6 h stops the kit. If the container's free memory falls under the 28 GB the rollout queue asks for, the probe is stopped and restarted after the queue.

## 3. Reads and names (the rules of card ADD45 with note N1, on WHOLE minus HALF)

* Per arm: M2 (gap R2 and gap error), M4 (presence AUROC). Per seed, paired on the same windows: S = WHOLE minus HALF on M2 and on the gap error; two-level bootstrap (seeds, then episodes), B 2,000.
* `HALFSCALE-LEAD-READ-SCALES-GAP`: the two-level interval of S on M2 lies above 0, at least three of the four seeds are above 0, and the mean gap error of WHOLE over the seeds is at most 0.90 times that of HALF. `HALFSCALE-LEAD-READ-WORSE-AT-SCALE`: the interval lies below 0. Else `HALFSCALE-LEAD-READ-FLAT-AT-SCALE`. The reader prints the names of card ADD45 (`LEAD-READ-...`); under this card each stands for the name with the prefix `HALFSCALE-`.
* LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL, TACTICAL: not measured by this read. STRATEGIC: NOT COMPUTED (n = 0; a token probe reads no strategic quantity).

## 4. Controls (all must pass; else no name, and the differences stay withheld)

Those of card ADD45 with notes N1 and N2, as the kit and the reader compute them: d1 (with its planted id), d2j, d2n, d2g, d2c (with its planted shift), d2e (with its planted swap), d3, d4, d5, d7, the record and pack controls. Two of them say less here than on the union and are stated so in the row: d2 compares the lead file with itself (it passes by construction), and the token report d3t reads equal hashes by construction. d6 is a limit, as in card ADD45.

## 5. What follows (committed before the data)

* `SCALES-GAP`: between 2,286 and 4,572 clips of the corpus the gap read still rises at matched compute. Data scale is then a measured lever of the lead read inside the corpus, before the union read; the read of card ADD45 says whether it continues with 10k clips of another collection. The two reads are reported side by side as a three-point curve; no model arm is changed by this card.
* `FLAT-AT-SCALE`: more clips of the same collection do not lift the read at this probe and this number of steps. Card ADD45 then tests the remaining case (clips of another collection); the arm with twice the steps says whether the WHOLE arm is short of steps.
* `WORSE-AT-SCALE`: reported as written; the arm with twice the steps is read first, since at matched steps the WHOLE arm sees each clip half as often.
* A control fails, or the kit or the reader stops with a fault: nothing is named; the fault is logged in `TIMES_NOTE.md` before its repair, the repair goes into the kit of card ADD45 before that card runs on the other pod, and this card is repeated in a new output folder.

## 6. Limits

One frozen encoder; pooled tokens; the half is the first half of the manifest (one split, no repeat over splits); four seeds bound the run-to-run variance of the probe; matched steps mean the WHOLE arm sees each clip half as often (the arm with twice the steps is reported beside); a probe is not a model; the evaluation windows come from the same collection as the training clips. The tests of the kit ran under a stand-in test runner on this pod. This card does not replace card ADD45: it has no clips of the 10k collection.
