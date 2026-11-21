# PREREG_V8F_PROOF_ADD81 (TOKCLOSE): does a temporal head on the token features of several frames read the closing speed better than the estimate from the reader's outputs?

Tier T0 (reads on logged frames; no rollout; a diagnostic of what the features carry, never a statement on driving performance). Evidence class of every value it will give: MEASURED. Written while the eight rollouts of card ADD79 run and before any value of card ADD79 is opened; no tool of this card exists.

## 1. Question and what is known

* Card ADD78 (MEASURED, T0): from the reader's three outputs per frame over a history, the label closing speed is read at a pooled r of +0.391 [+0.288, +0.478] (small network, 10 frames; cell N_H10) and +0.384 [+0.283, +0.475] (ridge, 25 frames; cell L_H25); the mean absolute error falls from 2.038 to 1.874 m/s and from 2.080 to 1.876 m/s. The reader's own closing output reads r +0.18.
* The reader's outputs are three numbers per frame. The pooled token features (4 x 10 tokens of 1,024 values per frame) may carry more of the lead's motion than those three numbers keep. Card ADD79 section 6 names this probe.
* Question: does a head that sees the token features of the last 10 or 25 frames read the closing speed better than the cells of card ADD78, on the same frames?

## 2. Arms (fixed before any fit)

Label: the closing speed of card ADD78, as the label source of the rule computes it (least-squares slope of the label gap over the trailing run of at most five frames, at least three; the function whose full-run frames passed control c2 of that card), the same function for fitting and for scoring. History H in frames: 10 and 25 (2 s and 5 s).

* E10, E25 (the probe as card ADD79 names it): per reader seed k the frozen frame embedding of the lr16 reader of seed k; a temporal head (one recurrent layer of 128 units over the last H embeddings, then two layers to one output) fitted on the 432 clips held out of the reader's training, early stopping on a split of those clips by clip (one fifth).
* T10, T25 (beyond that sentence, stated here before the data): head seed k; a token projection of the reader's form trained from scratch with the same temporal head, on the 4,080 clips of the reader's training; the 432 held-out clips give the stopping point; no clip of the evaluation set enters a fit or a choice.
* Optimiser, batch, step count, learning rate and loss are fixed in the tool's definition file before the first fit on data and are the same for all arms of a family; nothing is tuned on the evaluation clips.
* Seeds 0 and 1 per arm: eight fits. One more fit is the planted arm of control t1.

## 3. Read and names

Scored once on the 141 evaluation clips, on the scored frames of card ADD78 for the same H (3,248 frames for H 10 and 2,572 for H 25 per its result), per seed and pooled, episode unit the clip, intervals by the episode-cluster bootstrap. For each arm X: r against the label closing speed with its interval; mean absolute error; and against the ADD78 cell of the same H and seed (N_H10 for H 10, L_H25 for H 25, the saved estimators of card ADD79 applied to the same frames) the paired difference of the absolute error, X minus cell, with the paired episode-cluster bootstrap.

* `TOKCLOSE-<X>-READABLE` when the interval of the pooled r lies above 0, else `-NOT-READABLE`.
* `TOKCLOSE-<X>-BETTER-THAN-OUTPUTS` when the pooled paired difference of the absolute error has its interval below 0 and r of X is above r of the cell in both seeds; `-NOT-BETTER` otherwise.
* Verdict `TOKCLOSE-A-TOKEN-HEAD-READS-MORE` when at least one arm is BETTER-THAN-OUTPUTS (the count of four is written beside it); else `TOKCLOSE-NO-GAIN-FROM-TOKENS`.
* Reported beside, naming nothing: r per seed, the pooled r of all arms in one table with the cells of card ADD78, and the mean absolute error of a constant read.

## 4. Controls (all have to pass before a name or a value is opened)

* t0 (the frames and the label are those of card ADD78): on the evaluation clips the tool's label closing speed and its scored-frame sets equal the score dump of card ADD78 row by row (label to within 1e-6 m/s; the frame sets equal), and the saved estimators N10 and L25 applied by the tool reproduce the per-seed r of card ADD78 to within 0.005.
* t1 (a head cannot read what is not there): arm T10 fitted with the closing labels permuted among the labelled rows reads a pooled r inside [-0.10, +0.10] on the evaluation clips.
* t2 (the fits ran): for every fit the held-out loss at the stopping point is below the held-out loss before the first step; the count of labelled training frames and of clips is logged per fit and equals the split (4,080 and 432 clips, none of the 141).
* t3 (no leak of the future): the head output at frame t of a clip is unchanged when all frames after t are replaced by noise (largest difference at most 1e-5 m/s on a set of clips).
* t4 (planted name rule): on planted estimates with a known gain and with none the name rule gives BETTER-THAN-OUTPUTS and NOT-BETTER as planted.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136). A float tolerance of a tool's own sub-check is admissible with a note written after the failure is logged.

## 5. What follows (committed before the data)

* `TOKCLOSE-A-TOKEN-HEAD-READS-MORE`: the best arm becomes the closing input of the rule in a T1-kin/hb0.2 card (form RRe with the head's read per clip and frame; bar as cards ADD79 and ADD80 leave it: the estimate of card ADD79 or the best constant).
* `TOKCLOSE-NO-GAIN-FROM-TOKENS` with all arms READABLE: the pooled 4 x 10 token features carry the closing speed no better than the reader's outputs do; next is the cost card to the PI for a lead-region crop or a new encode at a finer grid (a PI decision: camera frames back on the pod), and the rule proposal rests on cards ADD79 and ADD80.
* Arms NOT-READABLE: reported as such; the same cost card.

## 6. Limits known in advance

T0 only: a better read is not a lower contact until a rollout card says so. Two seeds; 432 clips for the E arms; one architecture per family, chosen before the data and not searched; the label is a one-second slope of a label gap and carries its own noise, which bounds every r from above by an amount this card does not measure.
