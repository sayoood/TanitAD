# PREREG_V8F_PROOF_ADD28: GAPFILTER. The closing speed of the lead from a causal filter over the head's own gap reads. Zero training, existing files.

Frozen before any number of this card is computed. Addendum to `PREREG_V8F_PROOF.md`, to card ADD24 with its notes N1 and N2, and to card ADD20 (the object slot). Tier T0 (a diagnostic on logged frames, never driving performance). Evidence class of every result: MEASURED once run. LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL, TACTICAL: not measured; STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it (all MEASURED, card ADD24, six arms, read under notes N1 and N2)

* The supervised head reads the gap of the lead on held-out windows at R2 0.54 to 0.74 (RMSE 5.3 to 6.9 m) and its presence at AUROC 0.90 to 0.92.
* Its own closing-speed output is flat: R2 +0.155 / +0.014 (weight 1), +0.092 / +0.026 (weight 10), +0.178 / +0.089 (detached), every interval includes 0; +0.18 to +0.21 on clips it was trained on. The slot increment DELTA of note N1 is flat in all three arm types. The training log shows the closing-speed part of the lead term not falling over 19,164 steps.
* Control p2b of card ADD27 (synthetic): a head of this form trained by gradient steps does not take a rate from five frames of a planted gap (closing R2 at most +0.09 where a slope on the known directions reads +0.94).
* The head sees five frames (0.8 s at the 0.2 s frame step). A rate needs either a gap read that is precise from frame to frame or a longer baseline.

## 2. The lever

The closing speed is not asked of the network. It is computed outside it by a causal filter over the per-frame gap output of the same head, over a longer baseline. This is the slot of card ADD20 in its simplest form: a learned measurement (gap, presence) and a fixed kinematic rule (the gap changes at the closing speed).

Inputs, per arm (LS_s0, LS_s1, LSH_s0, LSH_s1, LSD_s0, LSD_s1): the per-row head outputs already pulled (`spatial/leadsup/read/in/<ARM>_{eval,train}_lead.npz`): gap, presence logit, closing speed, for every row of the evaluation episodes and of the train probe clips. Nothing else of the model. Label: the closing speed at the scoring instant S0, by the label path of the reader of card ADD24 (control k1 reproduces 0.7499068).

Only rows at or before S0 enter (causal). Rows whose presence probability is below 0.5 are treated as missing.

Estimators (fixed here):

* E1 slope: least-squares slope of the head gap over the last K rows, K in {3, 5, 8, 10, 15}; at least 3 present rows, else the estimate is missing.
* E2 alpha-beta filter: state (gap, rate); predict gap += rate * dt; on a present row with residual r: gap += alpha * r, rate += (beta / dt) * r; alpha in {0.2, 0.35, 0.5, 0.7}, beta in {0.02, 0.05, 0.1, 0.2}; restart (gap = measurement, rate = 0) at the start of a clip and after more than 5 missing rows in a row; the estimate is missing until 3 present rows since the restart.
* Calibration: closing = a * estimate + b, least squares.
* Where an estimate is missing, the closing-speed output of the head is used (stated with its count).
* Selection: K, (alpha, beta), the calibration, and the choice between E1 and E2 (the lower clip-grouped five-fold mean squared error) are made on the train probe clips only. The chosen estimator is F.
* Reported beside, names nothing: E3 = a linear blend of the E1 estimate, the E2 estimate and the closing-speed output of the head, fitted on the train probe clips; and the same F with every choice cross-fitted by episode inside the evaluation bank.

## 3. Read (413 scored windows with a closing-speed label, 48 evaluation episodes; episode-cluster bootstrap B 2,000, rng 0)

* M1F: R2 of F against the label closing speed, per arm.
* Paired difference M1F minus M1 (the head's own closing-speed output of the same arm), same drawn episodes.
* DELTA-F: the slot increment of note ADD24_N1 with the closing speed of F in rule R3c (gap 2.0 s ahead from the present head gap, the rate and the logged ego term, against the rule without a rate).
* Reported, names nothing: M1F on the train probe clips; the count of windows where F is missing; M1F by K and by gains.

## 4. Controls (all must pass before any outcome is named)

* f1 label path: control k1 of the reader reproduces 0.7499068.
* f2 sign, units, alignment: with the label gap in place of the head gap, through the same code path, E1 at K = 5 reads the label closing speed at R2 of at least 0.8 on the scored windows.
* f3 causal: changing the head outputs of all rows after S0 changes no estimate (exactly 0).
* f4 permuted: with the gap sequences permuted among the scored windows, M1F is at most +0.02.
* f5 no evaluation data in any choice: with the evaluation labels replaced by noise, the selected K, gains, calibration and estimator are identical.
* f6 the records of the pulled files (control a1 of the reader) pass for every arm.

Reported with the controls, naming nothing:

* f7 the error of the head gap on the evaluation rows with a gap label: RMSE, RMSE of its frame-to-frame change, lag-1 autocorrelation (what a filter can remove and what it cannot).
* f8 the precision a rate needs: the label gap plus white noise of standard deviation 0.25, 0.5, 1, 2 and 4 m through E1 at K = 5, 10, 15: closing-speed R2 on the scored windows.

## 5. Outcomes (both committed), per arm type, both seeds needed

* `FILTER-READS-CLOSING` if M1F is above 0 with an interval that excludes 0 at both seeds, AND the paired difference to the head's own output is above 0 with an interval that excludes 0 at both seeds, AND its mean is larger than the floor 0.089 (the absolute difference of M1 between LSD_s0 and LSD_s1, card ADD24). Else `FILTER-FLAT`.
* `FILTER-SLOT-IMAGINES` if DELTA-F is above 0 with an interval that excludes 0 at both seeds and its permuted control (the rate of F permuted among the scored windows) is at most +0.02. Else `FILTER-SLOT-FLAT`.

## 6. Committed next steps

* `FILTER-READS-CLOSING` or `FILTER-SLOT-IMAGINES` in an arm type: the slot (head gap, rate of F, the fixed transition) becomes a source of the distance rule of card ADD26 beside `head` and `label` on the arms of that type; zero training, pod rollouts at T1-kin with the four families, own card.
* `FILTER-FLAT` in all three: the frame-to-frame precision of the gap read is the limit; f7 and f8 give the gap between what the head reads and what a rate needs. Then (1) the same read on the four probe arms of card ADD27 once pulled (what the pooled tokens allow when trained for the lead alone), named by the same rule per probe arm type against the detached arms; (2) if that is flat too, the cue branch: a finer grid or a crop of the lead region needs the camera frames re-encoded and stored; that is a data and compute step the PI decides and it is named as a blocker with its cost, not run.

## 7. Cost and limits

* Dev-box CPU, minutes; no GPU, no pod, no training.
* The head saw the train probe clips during training, so its gap reads are cleaner there than on the evaluation episodes; choices made on them can be off for the evaluation bank (the cross-fitted variant is reported beside).
* Two seeds per arm type; the bootstrap interval answers the question about episodes, not about training runs; the floor is from the two detached arms.
* 413 scored windows of 48 episodes; one corpus. A rate from a longer baseline lags a changing closing speed; the label is the closing speed at S0.
