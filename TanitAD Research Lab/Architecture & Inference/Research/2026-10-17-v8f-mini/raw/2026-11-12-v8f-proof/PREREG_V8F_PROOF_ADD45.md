# Card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD): does more data lift the read of the lead from the pooled tokens? A probe-only read on the union of the corpus and the 10k clips

Card of the v8F proof study, written by the coordinator fork. Frozen before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). **It needs a go of the PI before any run: it uses the pod of the other v8F session, where the union tokens are.** Tier: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance.

## 1. Why

* Row H-V8F-PROOF-94 (MEASURED): the read of the lead is the first limit of the distance line, the gap before the closing speed.
* Row H-V8F-PROOF-97 (MEASURED): a finer pooled grid does not lift the gap read at this data size (gap R2 about 0.70 on either grid, closing-speed R2 about 0); card ADD43 names data scale as one of the two levers that remain.
* LEADHEAD stage 0b (MEASURED, compact paper): the learning curve of a lead head over training clips rises at every step and has not flattened.
* INHERITED from the tree owner session (its numpy pass on its pod, 2026-11-19; not re-verified here): the union of the corpus and the 10k clips carries a finite lead gap on 8,763 clips against 2,580 in the corpus file, with cached tokens of both grids for 14,512 clips. The lead head has never been trained at 10k.

## 2. Arms (probes on frozen tokens; no model is trained)

* Tool: the token probe of card ADD43 (`tok_probe_gr.py`, grid 4 by 10, per-token width 16, five-frame history, 19,164 steps, the loss and schedule of the lead-supervision arms), ported to read the union cache and the union lead file. The port changes input paths and the clip filter only.
* `P16C_sK`: trained on the corpus clips of the union only. `P16U_sK`: trained on the whole union. Seeds K = 0, 1, 2, 3. Eight arms, the same number of steps for both sets (matched compute).
* Beside, naming nothing: `P16U2_s0`, the union arm with twice the steps (says whether the union arm is under-trained at matched steps).
* Control arms: `P16CSH_s0` and `P16USH_s0`, permuted closing labels, at FULL length (19,164 steps). The 2,000-step form of card ADD43 is not used: row 97 showed that its train-log check cannot pass.
* Budget: ESTIMATED 3 to 5 pod-h on the pod of the other session (on the pod of this fork eight arms of this grid train together in about one hour; the other GPU is slower by a factor that is not measured). A 200-step smoke measures the step time before the run; an estimate above 6 pod-h goes back to the PI.

## 3. Reads

* Evaluation set: the 429 scored windows of the 48 evaluation episodes of card ADD43, unchanged, read from the same cache as the training clips.
* Per arm: M2 (gap R2 and gap error), M1 (closing-speed R2), M4 (presence AUROC).
* Per seed, paired on the same windows: S = `P16U_sK` minus `P16C_sK` on M2, on the gap error and on M1; two-level bootstrap (seeds, then episodes), B 2,000.
* LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL, TACTICAL: not measured by this read. STRATEGIC: NOT COMPUTED (n = 0; a token probe reads no strategic quantity).

## 4. Names

* `LEAD-READ-SCALES-GAP`: the two-level interval of S on M2 lies above 0, at least three of the four seeds are above 0, and the mean gap error of `P16U` over the seeds is at most 0.90 times that of `P16C`.
* `LEAD-READ-SCALES-CLOSING`: the same rule on M1, without the error clause.
* `LEAD-READ-WORSE-AT-SCALE`: the two-level interval of S on M2 lies below 0.
* Else `LEAD-READ-FLAT-AT-SCALE`.

## 5. Controls (all must pass; else no name, and the outcome differences stay withheld)

* d1 (no leak): no clip of the 48 evaluation episodes is in either training set (by clip id); the count of shared ids is exactly 0, and a planted shared id makes the check fail.
* d2 (labels on one clock): for 200 corpus clips drawn at random the gap and the closing speed of the union lead file equal those of the corpus lead file at the same frame index (tolerance 1e-4); the field that carries the closing speed is named with the file and line of the label builder before the port is written.
* d3 (the corpus arm is the known probe): `P16C_s0` and `P16C_s1` read M2 inside the range of the eight `TK16` arms of row 97 (0.513 to 0.791). Outside that range the port is not the same probe and nothing is named.
* d4 (training is live): the lead term of the last 1,000 steps is below that of the first 1,000 steps on every arm of full length, the control arms on their intact parts.
* d5 (negative control): the two arms with permuted closing labels read a closing-speed R2 whose interval includes 0.
* d6 (same order): within a seed the corpus clips are drawn in the same order in both arms where the sampler allows it; if it does not, this is stated as a limit and not as a pass.
* d7: the reader controls of card ADD43 (planted difference read back in the two-level form, a pair with itself reads exactly 0, a swap reads the negative).
* Tool checks before any run: unit tests of the changed loader, a counted mutation list (clip filter dropped, evaluation clips let into training, the wrong field as closing speed) that goes red, and a dry run.

## 6. What follows (committed before the data)

* `LEAD-READ-SCALES-GAP` (with or without `-CLOSING`): data scale is the lever of the lead read. The lead head enters the integrated candidate at 10k as its first lever, and no camera-frame step is asked of the PI now.
* `LEAD-READ-FLAT-AT-SCALE`: more clips of this kind do not lift the read through pooled tokens. The lever that remains is a second encoder pass over a crop of the lead region, which needs camera frames: a data step that goes to the PI as its own card with its cost. The integrated candidate still carries the lead head, with the expectation of the read of the corpus.
* `LEAD-READ-WORSE-AT-SCALE`: reported as written; the domain difference between the corpus and the 10k clips is then read first (the corpus arm on 10k evaluation clips), before any model arm with a lead head at 10k.

## 7. Limits

One frozen encoder; pooled tokens; the evaluation windows come from the corpus only, so a gain on 10k-like scenes is not seen; four seeds bound the run-to-run variance of the probe; matched steps mean the union arm sees each clip less often (the arm with twice the steps is reported beside); a probe is not a model.
