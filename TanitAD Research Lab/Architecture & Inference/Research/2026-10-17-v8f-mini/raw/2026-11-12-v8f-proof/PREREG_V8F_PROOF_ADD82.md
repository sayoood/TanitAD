# PREREG_V8F_PROOF_ADD82 (NEGCONST10K): a constant closing input of the cautious sign in the reader-fed lead rule

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react; heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`). Never a statement on driving performance. Evidence class of every value it will give: MEASURED. Written after card ADD80 was read (row 142) and before any rollout of this card.

## 1. Question and what is known

* The closing input c of the rule is the rate of the gap: the rule imagines the lead at v_L = max(0, v_l + c) (`distrule/dist_rule.py`, line 13), so c below zero says the gap closes and the rule brakes more. Card ADD80 had this sign wrong in its premise and measured +1, +2, +4 m/s: contact rises in a dose (+0.0154, +0.0200, +0.0308 at a pre-roll of 30 steps, 6 s) and falling behind falls (row 142).
* The estimates of card ADD79 lower contact by -0.0138 (network, not separated) and -0.0185 [-0.0408, -0.0042] (ridge) with means within 0.03 m/s of zero over the looks with a lead (rows 141, 142).
* Two questions, the ones card ADD80 meant to ask: (a) is there a constant of the cautious sign that does as well as an estimate, in contact and in falling behind? (b) is such a constant a rule form worth proposing, a form that needs no closing read?

## 2. Runs

Form RRe of card ADD79 (wrapper generation rzgfhqe, a747abeb51ef041f) with a constant estimator file of history one frame: closing = c at every look where the reader has its five-frame window, 0.0 at a look without it.

Values c: -1.0, -2.0 and -4.0 m/s (arms M10, M20, M40). Twelve rollouts on the pod of this fork (10k line models `UE_s0`, `UE_s1`; reader seeds 0, 1; readout K; `--rule-every 5`; heading bound 0.2; q50 map): `I10RQ50eM10KF5P30Hb20`, `I10RQ50eM20KF5P30Hb20`, `I10RQ50eM40KF5P30Hb20` (AGE: pre-roll of 30 steps, 6 s; 1,271 windows) and `I10RQ50eM10KF5Hb20`, `I10RQ50eM20KF5Hb20`, `I10RQ50eM40KF5Hb20` (FRESH: pinned grid, 2,399 windows), seeds 0 and 1; order AGE M10, M20, M40, then FRESH M10, M20, M40. Banked beside them: forms RR, RRc, RLc and no rule (row 137), the runs of form RRe with N10 and L25 (row 141, its controls passed) and the runs with +1, +2, +4 m/s (row 142) for the dose.

## 3. Read and names

Reader: the engine of rows 137, 141 and 142 (two-level bootstrap over model seeds and episodes; contact rate C, unsafe rate U and falling-behind rate F on the windows with a lead; the four families, STRATEGIC by the instrument of card ADD66 read at T1-kin). Pairs per value (M10, M20, M40) and setting (AGE, FRESH), A minus B:

* `CONST-VS-RR-c`: RRe(c) minus RR.
* `CONST-VS-LABEL-CLOSING-c`: RRe(c) minus RRc.
* `CONST-VS-NO-RULE-c`: RRe(c) minus no rule.
* `CONST-VS-EST-E-c` for E in N10, L25: RRe(c) minus RRe(E).

Names `NEGCONST10K-<PAIR>-<SETTING>-C-<LOWER|HIGHER|NOT-SEPARATED>`, the same for U and for F, and one name per family. The levels of C, U and F per value and seed are reported at both settings, and beside them the dose over the six constants -4, -2, -1, +1, +2, +4 with the form RR at 0 (reported, naming nothing).

Verdicts:

* The form. `NEGCONST10K-<c>-SUPPORTED`: pair CONST-VS-RR-c at AGE reads C LOWER, U not HIGHER and no family WORSE; and FRESH reads C not HIGHER, U not HIGHER and no family WORSE. Otherwise `-NOT-SUPPORTED`. Three values are read: a single SUPPORTED is marked one of three. The F name of the pair is written beside every verdict: it is the price of the form, a LONGITUDINAL cost in distance-keeping where it is HIGHER.
* The control, per estimator E. A value c matches E when pair CONST-VS-EST-E-c at AGE reads C not HIGHER and F not HIGHER. `NEGCONST10K-E-NOT-BEYOND-A-CAUTIOUS-CONSTANT` when at least one of the three values matches E; `NEGCONST10K-E-BEYOND-A-CAUTIOUS-CONSTANT` when none does. NOT-SEPARATED on both measures counts as a match: the estimate has to show its worth.

## 4. Controls (all have to pass before a name or a value is opened)

* k0 (the files): each constant file returns its value c for any input of shape [1, 3]; hashes logged before the queue.
* k1 (the constant reaches the rule with its sign, pod check before the queue, three clips, the file of -2.0 m/s): the closing column of the rule log reads -2.0 at every look with the reader window and a lead read present; the imagined lead speed of those looks is max(0, v_l - 2.0) and is below v_l wherever v_l is above zero; the estimate column reads -2.0 on every row marked full; no short look from the fifth frame of a clip on.
* k2 (in the runs): the estimate column equals c to within 1e-6 on the rows marked full in all twelve runs, both logs (the first look of the mid-look log is its smallest tick); share of looks of the scored window marked full at least 0.99 per setting, else that setting is not named.
* k3: the levels of the runs of card ADD79 and of card ADD80 recomputed by the reader equal their result files.
* k4 (the sign, read from the data and not from a name): at AGE the pooled contact level is not higher at -1 m/s than at +1 m/s and not higher at -4 m/s than at +4 m/s. A failure means the rule does not respond to the sign as the code says, and no name is written.
* g, s, r1, x1, m1, a1 (the levels of the banked forms equal the result of row 137), i1, h1: as cards ADD76, ADD79, ADD80.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136).

## 5. What follows (committed before the data)

* `E-NOT-BEYOND-A-CAUTIOUS-CONSTANT` for both estimators: a constant does what the estimates do; the proposal for the candidate is the constant form when it is SUPPORTED, with its F price, and the closing read has to beat that constant (cards ADD81, ADD83).
* `E-BEYOND-A-CAUTIOUS-CONSTANT`: no constant shift reproduces the estimate's pair of contact and falling behind; with card ADD83 this decides whether the estimate is proposed.
* A value SUPPORTED with F HIGHER: reported as a price (contact against falling behind); the choice is the PI's. A value SUPPORTED with F not HIGHER: a rule form without a closing read at no measured price; proposal to the PI and the other session; not adopted by this fork.
* No value SUPPORTED: constants are not a rule form; the open scalar stays the closing read.

## 6. Limits known in advance

Two model seeds, two reader seeds; logged frames and a lead that does not react, so falling behind is measured against a lead that never opens the gap on purpose; three values chosen before the data, mirror images of card ADD80; a constant closing input is a changed rule parameter, measured here as a study arm, its adoption a design decision of the PI and the other session.
