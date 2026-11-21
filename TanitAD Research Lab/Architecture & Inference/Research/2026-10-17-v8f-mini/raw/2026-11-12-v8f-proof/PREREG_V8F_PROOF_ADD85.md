# PREREG_V8F_PROOF_ADD85 (SMOOTHREQ10K): the requirement for a closing read whose error is smooth in time, a smooth spread alone, and the token head against a surrogate of matched mean

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react; heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`). Never a statement on driving performance. Evidence class of every value it will give: MEASURED. Written after card ADD84 was read (row 146, not landed yet) and before any tool, table or rollout of this card exists. All arms at age only (a pre-roll of 30 steps, 6 s), form RRe with `--closing-table`, the wrapper generation and settings of card ADD84.

## 1. What is known and what is asked

* Row 146: the label closing speed with white calibrated noise removes 0.547 of the contact gap at r 0.7 and 0.736 at r 0.9; the same r 0.7 with an error that is smooth in time (first-order autoregression, coefficient 0.9 per frame) removes 0.151 and is separated from the white arm (+0.0323 [+0.0084, +0.0604]). One noise draw per arm and seed.
* Row 146: a white spread with no relation to the label does not lower contact. Rows 144 and 146: three surrogates (the estimate of another clip, smooth in time) read -0.0046, -0.0123, -0.0169 against the form without a closing input, none separated.
* Row 146: the token-head table is SUPPORTED, and the pair against its surrogate has no name because the spread control failed on its mean clause. Input statistics read from the rule logs of the banked runs after that read (inputs of the rule, not outcomes; looks with a lead read present and a full history at age): the token-head input has a mean of -0.052 m/s in seed 0 (3,057 looks) and +0.545 m/s in seed 1 (3,272 looks), its surrogate +0.022 and +0.697; standard deviations 1.220, 1.330 and 1.206, 1.405. The head of seed 1 is itself biased toward an opening gap; the clause "within 0.15 m/s of zero" was written for estimates with a mean near zero. The lead has seen the values of the two guarded pairs of card ADD84 (row 146 says how); this card is written with that knowledge.
* Asked: (a) what does a read with a smooth error need to reach; (b) does the white-against-smooth separation at r 0.7 repeat with second draws; (c) does a smooth spread with no relation to the scene lower contact; (d) does the token head lower contact more than a surrogate whose mean is matched; (e) what does the bias of the head cost.

## 2. Arms and tables (all fixed before any rollout)

On the frames where the label table L100 of card ADD84 has a value, with m, s, the formula and the exact planting of card ADD84 and its note N1:

* S50, S90: smooth error (autoregression 0.9 per frame along the frames of a clip) at r 0.50 and 0.90.
* S7B: a second draw of the smooth r 0.70. W7B: a second draw of the white r 0.70.

On every row of the table (all frames of the 141 clips):

* Z00: value = z e with e the same autoregression over all frames of a clip, made exactly mean 0 and unit standard deviation over all rows, z = the standard deviation of the finite values of the token-head table of the same seed. No relation to the label by construction of the draw.

From the token-head table TK and its surrogate STK of card ADD84 (same files, same map):

* STKM: STK minus a constant d_k per seed, d_k = (mean of the STK input) minus (mean of the TK input) over the looks with a lead read present and a full history in the banked age runs of card ADD84 of seed k (from their rule logs; by the statistics above about +0.073 and +0.152 m/s). Its mean at those looks then equals that of TK.
* TKD: TK minus its own mean over the same looks of seed k (about -0.052 and +0.545 m/s): the head without its mean bias. The constant comes from the evaluation frames; a deployed form would take it from clips held out of training, so TKD measures what the bias costs and is not a rule form to propose.

Random numbers: `numpy.random.default_rng(85000 + 10 a + k)`, a = 0..4 for S50, S90, S7B, W7B, Z00, k the seed; one standard normal per row in row order.

Runs `I10RQ50e<ARM>KF5P30Hb20`, seeds 0 and 1, fourteen rollouts, order STKM, TKD, Z00, S90, S50, S7B, W7B.

## 3. Read and names

Reader: the engine of rows 137 to 146. Pairs at age, A minus B (banked runs of card ADD84 where named): `X-VS-RR` for every arm X; `S90-VS-W90`, `S50-VS-W50`, `S7B-VS-W7B`, `S7B-VS-S70`, `W7B-VS-W70` (the last two: what a second draw changes); `TOK-VS-SHUFM` (banked TK minus STKM), `SHUFM-VS-RR`; `TKD-VS-TOK`, `TKD-VS-SHUFM`. Names `SMOOTHREQ10K-<PAIR>-AGE-C-<LOWER|HIGHER|NOT-SEPARATED>`, the same for U and F, one name per family. Reported without a name: per arm the levels C, U, F per seed and pooled, and the share of the contact gap between the forms without a closing input and with the label closing speed.

* Per arm X: `SMOOTHREQ10K-X-REMOVES-CONTACT` when X-VS-RR reads C LOWER and U not HIGHER, else `-NOT-SHOWN`.
* (a) `SMOOTHREQ10K-SMOOTH-REQUIREMENT-R<nn>`: nn the lowest of 50, 70, 90 whose smooth level has REMOVES-CONTACT and a pooled share of at least 0.5; for 70 both draws (the banked S70 and S7B) have to have REMOVES-CONTACT and the mean of their two shares has to be at least 0.5. Else `SMOOTHREQ10K-NO-TESTED-SMOOTH-LEVEL-REACHES-HALF`.
* (b) `SMOOTHREQ10K-TIME-COURSE-CONFIRMED` when S7B-VS-W7B reads C HIGHER, else `-TIME-COURSE-NOT-CONFIRMED`.
* (c) `SMOOTHREQ10K-SMOOTH-SPREAD-ALONE-REMOVES-CONTACT` when Z00-VS-RR reads C LOWER, else `-SMOOTH-SPREAD-ALONE-NOT-SHOWN`.
* (d) `SMOOTHREQ10K-TK-GAIN-FROM-THE-SCENE` when TOK-VS-SHUFM reads C LOWER and U not HIGHER, else `-TK-GAIN-NOT-SHOWN-FROM-THE-SCENE` with the name of SHUFM-VS-RR beside it.
* (e) `SMOOTHREQ10K-TKD-BIAS-COSTS-CONTACT` when TKD-VS-TOK reads C LOWER, else `-BIAS-COST-NOT-SHOWN`.

Also reported as a T0 statistic with no name, from the tables alone: the lag-1 autocorrelation inside a clip of the error (table minus label) of the token-head table and of the two estimate tables of card ADD79 on consecutive frames where the label has a value, beside those of the planted noises, so that the error model is tied to a measurement.

## 4. Controls (all have to pass before a name or a value is opened; the ones marked "before" are judged before the rollouts)

* b0 (before; planted quality): for S50, S90, S7B, W7B as control a1 of card ADD84 (r within 0.02 of the target, standard deviation within 0.95 to 1.05 times r s, mean within 0.05 of m, lag-1 autocorrelation of the planted noise in [0.85, 0.95] for the smooth arms and in [-0.05, +0.05] for W7B, the frames with a value those of L100 exactly); the planted noise of S7B against that of the banked S70, and of W7B against W70, has an r inside [-0.10, +0.10] (a second draw, not a copy).
* b1 (before; Z00): a value on every row; mean within 0.01 m/s of 0; standard deviation within 0.95 to 1.05 times that of the token-head table; lag-1 autocorrelation in [0.85, 0.95]; r against the label on the frames where the label has a value inside [-0.15, +0.15].
* b2 (before; STKM and TKD): STKM equals STK minus d_k and TKD equals TK minus its constant exactly, with the constants recomputed from the rule logs of the banked runs; the STK table is the file of the record of card ADD84; r of STKM against the label on the scored frames of card ADD78 for H 25 inside [-0.15, +0.15].
* b3 (in the runs): the estimate column of the rule log equals the table of the run at the clip and frame to within 1e-9 on the rows marked full, all runs, both logs.
* b4 (in the runs; the spread control in a form that fits a read with a mean): over the looks with a lead read present at age, pooled over the seeds, the standard deviation of the STKM input lies within 0.80 to 1.25 times that of the TK input of the banked runs, the mean of the STKM input lies within 0.15 m/s of the mean of the TK input, and the share of looks marked full is at least 0.90. Else the pairs of (d) have no name. This differs from control u3 of cards ADD83 and ADD84 in one place: the mean is compared with the mean of the read and not with zero.
* g, s, r1, x1, m1, a1 of the reader, i1, h1: as cards ADD76 to ADD84; the levels of the banked runs recomputed here equal the result file of card ADD84.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136). Because (d) is read under a control whose form differs from the one that failed in card ADD84, the row of this card is read by the other session before it is landed.

## 5. What follows (committed before the data)

* A smooth requirement at 50, 70 or 90: that r, for a read with a smooth error, is the target written into the cost card to the PI (lead-region crop or finer encode), with the share of contact it buys.
* `NO-TESTED-SMOOTH-LEVEL-REACHES-HALF`: a camera read of the closing speed with a smooth error is a weak lever even at r 0.9; the cost card says so, and the lever moves to how the rule uses its looks and to a planner trained on its own rolled history (the other session's line).
* `TIME-COURSE-NOT-CONFIRMED`: the separation of row 146 was one draw; the requirement is read from the pooled draws and marked so.
* `SMOOTH-SPREAD-ALONE-REMOVES-CONTACT`: the gains of rows 141 and 146 are read against arm Z00 and not against the form without a closing input; a read has to beat Z00.
* `TK-GAIN-FROM-THE-SCENE`: the token head goes into the proposal to the PI and the other session, and the T0 follow-up (a longer fit, more head seeds, the mean of the head calibrated on held-out clips) is pre-registered next. `TK-GAIN-NOT-SHOWN-FROM-THE-SCENE`: the token head is not proposed; the requirement of (a) says what a read has to reach.

## 6. Limits known in advance

Two model seeds, two reader seeds; two draws at r 0.7 and one at the other levels; a calibrated Gaussian error with one autoregression coefficient is a model of a read, tied to a measurement only through the reported autocorrelations; the planted arms have values only where the label has one, Z00 on every frame; the constants of STKM and TKD come from the evaluation looks; the lead has seen the values of the surrogate pairs of card ADD84 before writing (d) and b4; logged frames and a lead that does not react.
