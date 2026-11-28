# PREREG_V8F_PROOF_ADD77 (CLOSEREAD): can a closing speed be read from the token reader's own per-frame outputs over a longer history? A probe on cached tokens, no model is trained

Written by the coordinator fork after the read of card ADD75 (row 134) and before any tool of this card exists. Time of freezing: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file. Tier: T0 (a probe on cached tokens of logged frames; it says what the reader's outputs allow, not what a model does with them); never driving performance. It runs on the pod of this fork only.

## 1. Why

* Row 134 (MEASURED, T1-kin/hb0.2): the contact that the reader-fed lead rule keeps at age is carried by the closing-speed input (share 0.76 of the gap to the label-fed form); the reader's own closing output removes 0.03 of it. By the fixed consequence of card ADD75 a closing-reader probe gets its own card: this one.
* Row 106 (MEASURED): a two-point change of the reader's gap read over 5 s is no closing speed at the tick (Pearson r against the label closing speed 0.275 and 0.198, bar 0.30). Rows 94 and 97: the closing-speed output of the reader, trained on a five-frame history of pooled tokens, reads a closing-speed R2 of about 0.
* Not tried: an estimator fitted on the whole sequence of the reader's per-frame reads (gap, presence, closing output) over a history of up to 25 frames, on clips the reader has not seen. A two-point difference uses two noisy reads; a fitted filter uses all of them and learns the error structure of the reader.

## 2. Data (all on the pod of this fork, cached tokens of the pooled 4 by 10 grid)

* Readers: `lr16/TK16_s0` and `lr16/TK16_s1` (the two reader seeds of the 10k line; trained on 4,080 clips, with 432 clips of the train split held out by the rule `int(sha256(clip id)[:8], 16) % 10 == 0`, as their `train_meta.json` says).
* Fit set: the 432 held-out clips (the reader has not seen them, and they are not evaluation clips). Score set: the evaluation clips (141 clips of the evaluation cache). Labels: presence, gap and closing speed of the corpus lead file (the label source of the study wrapper).
* Per clip and frame the tool writes the reader's read on the five-frame window that ends at the frame: presence p, gap read (before the map), closing output (times 10 m/s as the wrapper feeds it).

## 3. Arms

For a history of H frames (H = 5, 10, 25; the frame period is the one of the cache and is written into the result) the input is the stacked per-frame reads of the last H frames (gap read, presence, closing output), the target is the label closing speed at the last frame. A frame is used when the label has a lead with a closing speed at the frame and the H frames exist in the clip.

* Arm L (linear): ridge regression, the ridge weight chosen by a split of the fit set by clip (never on the score set).
* Arm N (small network): two hidden layers of 64 units on the same input, Huber loss, early stopping on the same split of the fit set by clip; seeds 0, 1, 2.
* Beside, naming nothing: the reader's own closing output at the frame; the two-point change of the gap read over H frames (the form of row 106).

## 4. Read and names

On the score frames, per reader seed: Pearson r between the estimate and the label closing speed, and the mean absolute error against the mean absolute error of the constant 0. Interval: two-level bootstrap over reader seeds and evaluation episodes (B 2,000); the paired difference of the two absolute errors on the same frames.

* `CLOSEREAD-<ARM>-H<H>-READS-CLOSING`: pooled r at least 0.30 with the lower end of its interval above 0, r at least 0.30 in both reader seeds, and the paired difference of the absolute errors (estimate minus constant 0) with its interval below 0.
* Else `CLOSEREAD-<ARM>-H<H>-NO-CLOSING`.
* `CLOSEREAD-A-CLOSING-IS-READABLE` when at least one arm and history reads READS-CLOSING; else `CLOSEREAD-NOT-READABLE-FROM-READER-OUTPUTS`.
* Six arm-history cells are read; with six cells one may pass by chance, so a single passing cell is named and marked as one of six, and the wrapper arm that follows (section 6) is its test on other data (the rollout windows).

Four families: this probe measures none of them (it reads a label of one LONGITUDINAL input). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. STRATEGIC: NOT COMPUTED (n = 0; a token probe reads no strategic quantity).

## 5. Controls (all have to pass; else no name and no outcome value is opened)

* c1 (no leak): no fit clip is an evaluation clip and no fit clip is in the reader's training list; the held-out rule recomputed from the clip ids gives the 432 clips of the reader's `train_meta.json` (count and list hash); a planted shared clip id makes the check fail.
* c2 (definition and clock, known value): the two-point change of the LABEL gap over 5 frames, on score frames where the label has a lead on all of them, correlates with the label closing speed at r at least 0.70 and with positive sign; shifted by plus or minus 10 frames against the labels the correlation is lower. This fixes sign, unit and frame clock of the target before any estimate is read.
* c3 (the dump is the reader the rule gets): on the fresh run `I10RQ50KHb20` of both seeds, presence and gap read in the rule log at the first replan tick of each window equal the per-frame dump at the same clip and frame (largest difference at most 1e-3; float32 on two paths), on every row with the window in the clip.
* c4 (negative control): arm L and arm N fitted on closing labels permuted over the fit frames read a pooled r whose interval includes 0.
* c5 (the tool reads a planted signal): on synthetic sequences (a gap that changes at a known rate plus noise) arm L with H = 25 reads the rate at r at least 0.90 and the name rule gives READS-CLOSING; on noise alone it gives NO-CLOSING.
* c6 (estimator): a pair with itself reads exactly 0; a swap reads the negative.
* c7 (fit and score sets are large enough): at least 2,000 fit frames and at least 500 score frames in at least 20 evaluation episodes per reader seed and history; the counts are written.
* Tool checks before any run: unit tests, a counted mutation list (the held-out rule dropped, evaluation clips let into the fit, the ridge weight chosen on the score set, the label of another frame as the target, the sign of the target turned) that goes red, and a dry run. The outcome values of the score set are not printed by any stage before the controls pass.

A control that does not pass is logged in `TIMES_NOTE.md` before any repair; the repair is a note to this card and the re-read goes into a new folder.

## 6. What follows (committed before the data)

* `CLOSEREAD-A-CLOSING-IS-READABLE`: the estimator of the passing cell with the largest pooled r becomes a closing source of the study wrapper (a new generation with its pod check) and is read in T1-kin at age and fresh with the second look against the forms of card ADD76 (own card). Only that rollout read can say that it lowers contact.
* `CLOSEREAD-NOT-READABLE-FROM-READER-OUTPUTS`: the closing speed is not in the reader's per-frame outputs at up to 25 frames of history. The next probe reads it from the reader's token features over the same histories with a trained temporal head (own card, a probe fit on the pod of this fork). If that fails as well, the levers that remain need a decision of the PI (a second encoder pass over a crop of the lead region or the token grid before pooling: both need camera frames or a new encode) and go to the PI as a card with their cost; beside it, without a PI decision, a rule form that needs no closing speed (a larger time gap or a look at every step) is read against its falling-behind cost.

## 7. Limits

One frozen encoder and pooled tokens; two reader seeds; the fit set is 432 clips; the labels come from the corpus lead file (its closing speed is the track's relative speed, which is not the same quantity as the change of the gap; c2 measures how close they are); logged frames only, so nothing is said about a rollout; six cells are read.
