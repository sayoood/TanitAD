# PREREG_V8F_PROOF_ADD79 (ESTCLOSE10K): does a closing speed estimated from the reader's own outputs over a history lower the contact of the reader-fed lead rule?

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react; heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`). Never a statement on driving performance. Evidence class of every value it will give: MEASURED. Written before the wrapper generation exists and before any rollout of the card.

## 1. Question and what is known

* Row 137 (MEASURED): with the second look every five ticks the label closing speed in the reader-fed form lowers contact against the form without a closing speed by -0.0815 [-0.1366, -0.0327] at a pre-roll of 30 steps (6 s) and by -0.0234 [-0.0485, -0.0067] fresh; the reader's own closing output changes nothing.
* Card ADD78 (MEASURED, T0; its row is with the other session to read): a closing speed fitted on the reader's per-frame outputs over a history follows the label closing speed at a pooled r of +0.391 [+0.288, +0.478] (small network, 10 frames) and +0.384 [+0.283, +0.475] (ridge, 25 frames) on the evaluation clips.
* Question: fed to the rule as its closing input, does such an estimate lower contact? A read at r 0.39 follows a third of the standard deviation of the target; the expected effect is a fraction of the effect of the label closing speed and may be none. Both outcomes are committed below.

## 2. Estimators (fitted before any rollout, on the 432 clips held out of the reader's training; the evaluation clips enter no fit)

Per reader seed k (0, 1), fitted on the fit dump of that seed (`closeread/dump/both` on the pod, the files of card ADD78):

* N10 (primary; the cell with the largest pooled r of card ADD78): the mean of three small networks (two hidden layers of 64 units, Huber loss, early stopping on the split of the fit set by clip, seeds 0, 1, 2) on the stacked reads of the last 10 frames (gap read before the map, presence, closing output).
* L25 (beside; the cell with the largest lowering of the error): ridge regression on the stacked reads of the last 25 frames, ridge weight chosen on the same split.

The estimators are saved as weight files and applied by a numpy forward. The save tool has to reproduce the read of card ADD78 on the score frames (control e0).

## 3. Runs

Form RRe: the reader-fed rule of the candidate (reader presence, gap read with the q50 map) with closing = the estimate at the look, at every look (replan ticks and the second look). At a look whose history does not lie in the clip (fewer frames than the history plus the reader's five-frame window) the closing input is 0.0, as in form RR, and the look is counted. Reader seed k and estimator seed k go with model seed k.

Eight rollouts on the pod of this fork (10k line models `UE_s0`, `UE_s1`; readout K; `--rule-every 5`; heading bound 0.2; q50 map): `I10RQ50eN10KF5P30Hb20`, `I10RQ50eL25KF5P30Hb20` (AGE: pre-roll of 30 steps, 6 s; grid of 1,271 windows) and `I10RQ50eN10KF5Hb20`, `I10RQ50eL25KF5Hb20` (FRESH: pinned grid, 2,399 windows), seeds 0 and 1. Banked beside them (row 137): forms RR, RRc, RRh, RLc and no rule with the same second look and readout.

## 4. Read and names

Reader: the engine of row 137 (two-level bootstrap over model seeds and episodes; contact rate C and unsafe rate U on the windows with a lead; the four families, STRATEGIC by the instrument of card ADD66 read at T1-kin). Pairs per estimator E (N10, L25) and setting (AGE, FRESH), A minus B:

* `EST-CLOSING-E`: RRe(E) minus RR.
* `EST-VS-LABEL-CLOSING-E`: RRe(E) minus RRc.
* `EST-VS-LABEL-FED-E`: RRe(E) minus RLc.
* `EST-VS-NO-RULE-E`: RRe(E) minus no rule.

Names `ESTCLOSE10K-<PAIR>-<SETTING>-C-<LOWER|HIGHER|NOT-SEPARATED>`, the same for U, and one name per family (BETTER, WORSE, MIXED, NOT-SEPARATED by the reading rule of the family engine). The falling-behind rate is reported for every pair.

Verdicts:

* `ESTCLOSE10K-N10-SUPPORTED`: pair EST-CLOSING-N10 at AGE reads C LOWER, U not HIGHER and no family WORSE; and FRESH reads C not HIGHER, U not HIGHER and no family WORSE. Otherwise `ESTCLOSE10K-N10-NOT-SUPPORTED`.
* `ESTCLOSE10K-L25-SUPPORTED` / `-NOT-SUPPORTED`: the same rule for L25. Two estimators are read: a single SUPPORTED of the two is named and marked as one of two.
* Reported beside, naming nothing: the share of the contact gap between RR and RRc at AGE that each estimate removes (arithmetic on the pooled pair values).

## 5. Controls (all have to pass before a name or a value is opened)

* e0 (the saved estimator is the estimator of card ADD78): applied by the numpy forward to the score dump of its reader seed, each saved estimator reproduces the per-seed r of its cell in the result of card ADD78 to within 0.005, and for L25 the estimate of the fit tool row by row to within 1e-6 m/s.
* e1 (the estimate reaches the rule, pod check of the wrapper generation before the queue): with an estimator file of zero weights the form RRe reproduces the banked form RR on a subset of clips row by row (rule log and plan), and with an estimator file that returns the reader's closing output of the last frame it reproduces the banked form RRh on the looks with a full history; a planted constant estimator of +2 m/s shows as 2.0 in the closing column of the rule log on those looks.
* e2 (the estimate in the rollout is the estimate of the dump): on the FRESH runs, at the looks with a full history, the closing column of the rule log equals the saved estimator applied to the score dump at the same clip and frame to within 1e-3 m/s.
* e3 (coverage): the share of looks with a full history is reported per setting and estimator, for the looks of the scored window and separately for the looks of the pre-roll (the wrapper counts both). At AGE the share over the scored window has to be at least 0.90 for N10 and for L25, else the AGE read of that estimator is not named. The pre-roll share is reported and not judged: during a pre-roll that starts early in the clip the estimate is 0.0 for the first looks (history of 10 or 25 frames plus the reader window), so the AGE read of L25 in particular is a read of an estimate that was active for part of the pre-roll only.
* g, s, r1, x1, m1, a1 (the levels of the banked forms equal the result of row 137), i1, h1: as card ADD76.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136).

## 6. What follows (committed before the data)

* `ESTCLOSE10K-N10-SUPPORTED` (or L25 alone, marked one of two): proposal to the PI and the other session to specify the candidate's reader-fed rule with the estimated closing input; not adopted by this fork. Next on the same line: the estimator fitted on more held-out clips, or on a longer history.
* Both NOT-SUPPORTED with C NOT-SEPARATED: the reader's outputs do not carry enough of the closing speed for the rule at this r. Next probe: a temporal head on the token features of several frames (own card, T0, fitted on the held-out clips); after it the cost card to the PI for a lead-region crop or a new encode; beside it a rule form that needs no closing speed (its own card).
* A family WORSE or U HIGHER for an estimator: that estimator is not proposed, whatever its contact reads.

## 7. Limits known in advance

Two model seeds, two reader seeds; the estimators are fitted on 432 clips; logged frames and a lead that does not react; the estimate is a read of a one-second slope of the gap at r about 0.39; the label-fed forms are privileged study arms. FRESH windows that start early in the clip have no full history at their first looks, more so for L25 (25 frames plus the reader window): the fallback to 0.0 makes RRe equal RR there, which biases the FRESH pairs toward NOT-SEPARATED and is why coverage is a control at AGE and a reported number FRESH.
