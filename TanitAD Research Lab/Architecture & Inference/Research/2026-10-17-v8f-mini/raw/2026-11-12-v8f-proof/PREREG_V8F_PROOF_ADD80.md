# PREREG_V8F_PROOF_ADD80 (CONSTCLOSE10K): what does a constant closing input do to the reader-fed lead rule, and is an estimate worth more than a constant?

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react; heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`). Never a statement on driving performance. Evidence class of every value it will give: MEASURED. Written while the eight rollouts of card ADD79 run and before any value of card ADD79 is opened; no rollout of this card exists.

## 1. Question and what is known

* Row 137 (MEASURED): in the reader-fed rule with the second look, the label closing speed lowers contact against the form without a closing speed (RRc minus RR at a pre-roll of 30 steps, 6 s: -0.0815 [-0.1366, -0.0327]); the reader's own closing output does not.
* Card ADD79 (running) feeds an estimate of the closing speed (r about 0.39 against the label, card ADD78) into the same rule.
* A closing input larger than zero makes the rule keep more distance whatever it knows. An estimate that lowers contact may do so through what it knows of the scene or only through its mean. This card measures the second: the same form with a constant closing input, three values. It serves two ends: (a) the control for the reading of card ADD79, (b) the rule form that needs no closing speed, named in card ADD79 section 6.

## 2. Runs

Form RRe of card ADD79 (wrapper generation rzgfhqe, a747abeb51ef041f), the estimator file being a constant: closing = c at every look where the reader has its five-frame window (estimator files with a history of one frame, written by `closest.save_planted(kind "const", H 1, value c)`), 0.0 at a look without that window, as in form RR.

Values c: 1.0, 2.0 and 4.0 m/s. Twelve rollouts on the pod of this fork (10k line models `UE_s0`, `UE_s1`; reader seeds 0, 1; readout K; `--rule-every 5`; heading bound 0.2; q50 map): `I10RQ50eC10KF5P30Hb20`, `I10RQ50eC20KF5P30Hb20`, `I10RQ50eC40KF5P30Hb20` (AGE: pre-roll of 30 steps, 6 s; grid of 1,271 windows) and `I10RQ50eC10KF5Hb20`, `I10RQ50eC20KF5Hb20`, `I10RQ50eC40KF5Hb20` (FRESH: pinned grid, 2,399 windows), seeds 0 and 1. Banked beside them: forms RR, RRc, RLc and no rule (row 137) and, when the read of card ADD79 has all its controls passed, its runs of form RRe with N10 and L25.

## 3. Read and names

Reader: the engine of row 137 and card ADD79 (two-level bootstrap over model seeds and episodes; contact rate C, unsafe rate U and falling-behind rate F on the windows with a lead; the four families, STRATEGIC by the instrument of card ADD66 read at T1-kin). Pairs per value c (C10, C20, C40) and setting (AGE, FRESH), A minus B:

* `CONST-VS-RR-c`: RRe(c) minus RR.
* `CONST-VS-LABEL-CLOSING-c`: RRe(c) minus RRc.
* `CONST-VS-NO-RULE-c`: RRe(c) minus no rule.
* `CONST-VS-EST-E-c` for E in N10, L25: RRe(c) minus RRe(E). Written only when the read of card ADD79 has all its controls passed.

Names `CONSTCLOSE10K-<PAIR>-<SETTING>-C-<LOWER|HIGHER|NOT-SEPARATED>`, the same for U and for F (HIGHER when the interval of the pair lies above 0, LOWER when below), and one name per family (BETTER, WORSE, MIXED, NOT-SEPARATED by the reading rule of the family engine). The levels of C, U and F per value c and seed are reported at both settings.

Verdicts:

* The form (end b). `CONSTCLOSE10K-C<cc>-SUPPORTED`: pair CONST-VS-RR-c at AGE reads C LOWER, U not HIGHER and no family WORSE; and FRESH reads C not HIGHER, U not HIGHER and no family WORSE. Otherwise `-NOT-SUPPORTED`. Three values are read: a single SUPPORTED is marked one of three. The F name of the pair is written beside every verdict: it is the price of the form.
* The control (end a), per estimator E, only when the pairs with E are written. A value c matches E when pair CONST-VS-EST-E-c at AGE reads C not HIGHER and F not HIGHER. `CONSTCLOSE10K-E-NOT-BEYOND-A-CONSTANT` when at least one of the three values matches E; `CONSTCLOSE10K-E-BEYOND-A-CONSTANT` when none does (every value has more contact or more falling behind than the estimate, separated). NOT-SEPARATED on both measures counts as a match: the estimate has to show its worth.

## 4. Controls (all have to pass before a name or a value is opened)

* k0 (the files): each constant file loaded by `closest.py` returns its value c for any input of shape [1, 3]; the three files and their hashes are logged before the queue starts.
* k1 (the constant reaches the rule, pod check before the queue, three clips): with the file of 2.0 m/s the closing column of the rule log reads 2.0 at every look with the reader window and a lead read present, the estimate column reads 2.0 on every row marked full, and the count of looks without a full history is zero from the fifth frame of a clip on.
* k2 (in the runs): in all twelve runs the estimate column of the rule log equals c to within 1e-6 on the rows marked full, at replan ticks and at the second look; the share of looks of the scored window marked full is at least 0.99 at both settings, else the read of that setting is not named.
* k3 (the runs of card ADD79 are admitted): the pairs with N10 and L25 are computed only when `READ_ESTCLOSE10K.json` says that all its controls passed, and their levels equal that file.
* g, s, r1, x1, m1, a1 (the levels of the banked forms equal the result of row 137), i1, h1: as cards ADD76 and ADD79.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136).

## 5. What follows (committed before the data)

* A value SUPPORTED and its F name not HIGHER: a rule form without a closing read exists at no measured price; proposal to the PI and the other session beside the forms of card ADD79; not adopted by this fork.
* A value SUPPORTED with F HIGHER: the form buys contact with falling behind; the pair of numbers is reported as a price, the choice is the PI's.
* `E-BEYOND-A-CONSTANT`: the estimate carries scene knowledge that the rule uses; the estimator line continues (card ADD81, a temporal head on token features) and its bar is the estimate, not a constant.
* `E-NOT-BEYOND-A-CONSTANT` for both estimators: a closing read at r about 0.39 is worth no more than its mean in this rule; card ADD81 has to beat the best constant, and the proposal for the candidate is the constant form when it is SUPPORTED.
* No value SUPPORTED: the reader-fed rule gains nothing from a constant; the open scalar stays the closing read.

## 6. Limits known in advance

Two model seeds, two reader seeds; logged frames and a lead that does not react, so falling behind is measured against a lead that never opens the gap on purpose; three values of c chosen before the data (a constant read has a mean absolute error of about 2.0 m/s against the label closing speed on the evaluation frames, card ADD78); the label-fed forms are privileged study arms; a constant closing input is a changed rule parameter: as a study arm it is measured here, its adoption is a design decision of the PI and the other session.
