# PREREG_V8F_PROOF_ADD83 (SHUFCLOSE10K): is the gain of the estimated closing speed from the scene, or from the spread of the input alone?

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react; heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`). Never a statement on driving performance. Evidence class of every value it will give: MEASURED. Written after cards ADD79 and ADD80 were read (rows 141, 142) and before the wrapper generation of this card exists.

## 1. Question and what is known

* Row 141: the ridge estimate of the closing speed lowers the contact of the reader-fed rule by -0.0185 [-0.0408, -0.0042] at a pre-roll of 30 steps (6 s); the network estimate by -0.0138 [-0.0333, +0.0000]. Both raise falling behind.
* Row 142: over the looks with a lead the estimates have a mean within 0.03 m/s of zero and a standard deviation of 1.25 to 1.30 m/s. Their gain is not a shift of the mean.
* The rule only ever lowers speed. An input with a mean of zero and a spread makes it brake on its excursions below zero and gives nothing back on those above; such an input could lower contact and raise falling behind with no knowledge of the scene. The estimate follows the label at r 0.39 only, so most of its spread is not the scene.
* Question: does the estimate of the clip lower contact more than the same estimate taken from another clip?

## 2. The surrogate (fixed before any rollout)

Per reader seed k and estimator E (N10, L25), from the saved table of the estimate on the 141 evaluation clips (`est_score_<E>_s<k>.npz` of card ADD79, one value per clip and frame, the values the rollouts of card ADD79 used to within 1.5e-05 m/s by its control e2): the surrogate table gives clip c at frame j the estimate of clip pi(c) at frame j. The map pi is fixed by a rule that uses no outcome: the clips are sorted by the share of their frames whose reader presence output is at least 0.5 in the score dump of reader seed k (ties by clip index) and neighbours in that order are exchanged (first with second, third with fourth, and so on; the last three form a cycle when the count is odd). No clip maps to itself. A frame that the other clip does not have, or where its estimate has no full history, has no surrogate value: the look counts as short and the closing input is 0.0, as in form RRe.

The surrogate has the values of the estimate, its time course inside a clip and, by the pairing, a like share of frames with a lead; it has no relation to the scene of the clip it is fed to.

## 3. Runs

A wrapper generation after rzgfhqe with one more closing source: the closing input of form RRe read from a table by clip and frame (`--closing-table`), everything else unchanged. Eight rollouts on the pod of this fork (10k line models `UE_s0`, `UE_s1`; reader seeds 0, 1; readout K; `--rule-every 5`; heading bound 0.2; q50 map): `I10RQ50eSN10KF5P30Hb20`, `I10RQ50eSL25KF5P30Hb20` (AGE: pre-roll of 30 steps, 6 s) and `I10RQ50eSN10KF5Hb20`, `I10RQ50eSL25KF5Hb20` (FRESH), seeds 0 and 1; order AGE SL25, AGE SN10, FRESH SL25, FRESH SN10. Banked beside them: the runs of the estimates themselves (row 141), forms RR, RRc, RLc and no rule (row 137).

## 4. Read and names

Reader: the engine of rows 137, 141, 142. Pairs per estimator E and setting, A minus B:

* `EST-VS-SHUF-E`: RRe(E) minus RRe(surrogate of E). The pair of the card.
* `SHUF-VS-RR-E`: RRe(surrogate of E) minus RR. What the spread alone does.
* `SHUF-VS-NO-RULE-E`: RRe(surrogate of E) minus no rule.

Names `SHUFCLOSE10K-<PAIR>-<SETTING>-C-<LOWER|HIGHER|NOT-SEPARATED>`, the same for U and for F, and one name per family.

Verdicts, per estimator E, on the AGE setting (where row 141 has its gain); FRESH is reported:

* `SHUFCLOSE10K-E-GAIN-FROM-THE-SCENE`: pair EST-VS-SHUF-E at AGE reads C LOWER (the estimate of the clip has less contact than its surrogate, separated) and U not HIGHER.
* `SHUFCLOSE10K-E-GAIN-NOT-SHOWN-FROM-THE-SCENE`: otherwise. Beside it the name of SHUF-VS-RR-E says whether the spread alone lowers contact (C LOWER), which would be the other reading shown, or whether neither is shown.
* L25 is the estimator whose gain row 141 has separated; N10 is read beside it and its verdict is marked as resting on a gain that was not separated.

## 5. Controls (all have to pass before a name or a value is opened)

* u0 (the table source is the estimate, pod check of the wrapper generation before the queue, three clips): with the table of the estimate itself and the identity map, the new form reproduces the banked run of card ADD79 of the same estimator on the subset: the closing column to within 1e-3 m/s on every row, the same rows marked full, and the plans of the first look to within 1e-3; with a table of zeros it equals form RR with largest difference 0; a planted table of +2 m/s shows as 2.0.
* u1 (the surrogate is what section 2 says): the map is a bijection of the 141 clips with no fixed point, it is the map that the rule of section 2 gives when recomputed from the dump, and every surrogate value equals the estimate table at the mapped clip and the same frame exactly.
* u2 (in the runs): the estimate column of the rule log equals the surrogate table at the clip and frame to within 1e-9 on the rows marked full, in all eight runs and both logs (the first look of the mid-look log is its smallest tick).
* u3 (the surrogate has the spread of the estimate): over the looks with a lead read present at AGE, per estimator and pooled over the seeds, the standard deviation of the surrogate input lies within 0.80 to 1.25 times that of the estimate input in the runs of card ADD79, its mean lies within 0.15 m/s of zero, and the share of looks marked full is at least 0.90. Else the AGE read of that estimator is not named. The same numbers are reported fresh and not judged.
* u4 (the surrogate does not know the scene): on the scored frames of card ADD78 the surrogate values follow the label closing speed of the clip they are fed to at an r inside [-0.15, +0.15], pooled over the seeds, per estimator (the estimate itself reads +0.39 and +0.38 there; the band is wide because about sixty clips carry these frames).
* g, s, r1, x1, m1, a1, i1, h1: as cards ADD76, ADD79, ADD80; the levels of the runs of card ADD79 recomputed here equal its result file.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136).

## 6. What follows (committed before the data)

* `L25-GAIN-FROM-THE-SCENE`: the estimate carries scene knowledge that the rule uses, small as it is; the proposal of row 141 goes to the PI and the other session with this control beside it, and a better read (card ADD81) is worth a rollout card.
* `L25-GAIN-NOT-SHOWN-FROM-THE-SCENE` with SHUF-VS-RR C LOWER: the gain of row 141 is what any input of that spread gives; the estimate is not proposed; what stays is a rule form with more caution, read by card ADD82, and the closing read has to beat it.
* `L25-GAIN-NOT-SHOWN-FROM-THE-SCENE` with SHUF-VS-RR not separated: neither reading is shown at two model seeds; the estimate is not proposed on this evidence, and the next step is the stronger read (card ADD81) with this control built into its rollout card.

## 7. Limits known in advance

Two model seeds, two reader seeds; one surrogate map per reader seed, not a distribution of maps; the surrogate keeps the time course inside a clip but not its relation to the gap read of the clip, so a look can pair a near gap with a closing input from a far lead; logged frames and a lead that does not react. A surrogate that passes u3 and u4 is a control of the spread reading, not a proof that no other statistic of the estimate matters.
