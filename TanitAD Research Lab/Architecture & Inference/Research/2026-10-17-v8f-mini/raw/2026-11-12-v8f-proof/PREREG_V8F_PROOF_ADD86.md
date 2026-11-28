# PREREG_V8F_PROOF_ADD86 (TOKCLOSE2): how far the pooled token features go with a full fit, against the requirement of r 0.9

Tier T0 (reads on logged frames; no rollout; a diagnostic of what the features carry, never a statement on driving performance). Evidence class of every value it will give: MEASURED. Written after card ADD85 was read (row 147 written, with the other session for its read, not landed) and before any fit or tool of this card exists. No new encode, no camera frame, no new data.

## 1. What is known and what is asked

* Row 143 (card ADD81): the token head over 25 frames reads the label closing speed at r +0.468 [+0.331, +0.611]; its fits were capped at eight evaluations with a patience of two, and one of the two T25 fits ended at the cap while its stop loss still fell. Row 146: the head of that seed has a mean bias of +0.545 m/s at the looks with a lead.
* Row 147 (card ADD85): a closing read with a smooth error has to reach r 0.9 at T0 to remove 0.736 of the contact gap in the rule; the reads on record (r 0.38 to 0.47) are not distinguishable from an input that does not know the scene.
* Asked: does a full fit of the same head on the same pooled 4 x 10 token features move the read, and how far is it from 0.9? The answer sizes the cost card to the PI (lead-region crop or finer encode): if the present features stay far below the requirement with a full fit, no head on them will carry the rule.

## 2. Arms and fits

* T10L, T25L: the arms T10 and T25 of card ADD81 with one change of the fit: at most 24 evaluations (card ADD81: 8) and a patience of 4 evaluations (card ADD81: 2), set through the environment variable `TOKCLOSE_HP_OVERRIDE` of the unchanged tool `tokclose/tokclose_fit.py` 84e82ee9184289c4; seeds 0 and 1; the same training clips (4,080), stop clips (432), labels, architecture, optimiser, batch and loss. Four fits.
* The four E fits and the planted fit of card ADD81 are taken over as files, unchanged, so that the unchanged scorer finds its nine heads.
* New root on the pod `/workspace/v8proof/tokclose2`; chain script `tokclose2/run_tokclose2.sh`, written from `tokclose/run_tokclose.sh` 4def827fd897ce17 by a generator with counted edits.

## 3. Read and names

The scorer of card ADD81, unchanged, on the 141 evaluation clips and the scored frames of card ADD78 (2,572 frames for H 25, 3,248 for H 10), with `--allow-hp-override`. Per arm X in T10L, T25L: r against the label closing speed pooled over the seeds with its interval (episode-cluster bootstrap, the clip as the unit), per seed r, mean absolute error, and the paired difference of the absolute error against the ADD78 cell of the same H and seed.

* `TOKCLOSE2-<X>-READABLE` / `-NOT-READABLE` and `TOKCLOSE2-<X>-BETTER-THAN-OUTPUTS` / `-NOT-BETTER`: the rules of card ADD81 section 3, as the scorer writes them.
* `TOKCLOSE2-FULL-FIT-READS-MORE` when T25L is BETTER-THAN-OUTPUTS, else `TOKCLOSE2-FULL-FIT-NO-GAIN`.
* Against the requirement of row 147: `TOKCLOSE2-POOLED-TOKENS-BELOW-THE-REQUIREMENT` when the upper end of the interval of the pooled r is below 0.9 for both arms; `TOKCLOSE2-POOLED-TOKENS-REACH-THE-REQUIREMENT` when the lower end of the interval of T25L is at or above 0.9; `TOKCLOSE2-REQUIREMENT-NOT-DECIDED` otherwise.
* Reported with no name: for every T fit the stop loss before the fit and at the stop, the step of the best evaluation, the number of steps run and whether the fit ended at the cap; the r and error of card ADD81 beside the new ones.

## 4. Controls (all have to pass before a name or a value is opened)

* t0, t1, t3, t4 of card ADD81 as the scorer computes them; t2 as note N1 of card ADD81 reads it (the stop loss falls on each of the eight arm fits), with the override admitted by the flag.
* c0: the fifteen files taken over (four E fits and the planted fit, each with head, log and table) are equal in sha256 to the files of card ADD81.
* c1: the logs of the four T fits name the override (24 evaluations, patience 4) and the logs of the E fits name none.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136).

## 5. What follows (committed before the data)

* `POOLED-TOKENS-BELOW-THE-REQUIREMENT`: the cost card to the PI says that the pooled 4 x 10 token features do not carry the closing read the rule needs, with the r reached by a full fit beside the 0.9 required; a better read needs a new encode (a PI decision).
* `FULL-FIT-READS-MORE`: the full-fit head is the best read on record; it is tested in the rule only with the matched surrogate of card ADD85 beside it, in a card of its own.
* `FULL-FIT-NO-GAIN`: the cap of card ADD81 was not what held the read back.

## 6. Limits known in advance

T0 only; two seeds; one change of the fit and no search; the arm and the change were chosen after rows 143 to 147 on the same 141 clips; the label is a one-second slope of a label gap and bounds every r from above by an amount this card does not measure, so "below 0.9" is a statement about this label.
