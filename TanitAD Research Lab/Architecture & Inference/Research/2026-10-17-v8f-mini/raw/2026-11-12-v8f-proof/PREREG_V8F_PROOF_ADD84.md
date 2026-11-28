# PREREG_V8F_PROOF_ADD84 (TABCLOSE10K): the token-head read in the rule, and how good a closing read has to be

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react; heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`). Never a statement on driving performance. Evidence class of every value it will give: MEASURED. Written after card ADD81 was read (row 143) and before any rollout or tool of this card exists; cards ADD82 and ADD83 are not read yet. The closing source is the table path of the wrapper generation of card ADD83 (`--closing-table`); no rollout of this card starts before control u0 of card ADD83 has passed on the pod.

## 1. Question and what is known

* Row 141: an estimate of the closing speed from the reader's outputs (r 0.38 to 0.39 against the label) removes a fifth to a sixth of the contact that the label closing speed removes in the reader-fed rule (0.0815 at a pre-roll of 30 steps, 6 s).
* Row 143 (T0): the token head over 25 frames reads the label closing speed at r +0.468 [+0.331, +0.611] and a mean absolute error of 1.785 against 1.876 m/s for cell L25; the error difference is not separated ([-0.225, +0.029]), the name is `TOKCLOSE-NO-GAIN-FROM-TOKENS`.
* Not known: what a read at r 0.47 does in the rule, and how good a closing read has to be before it removes a given share of that contact. The second sizes the cost card of card ADD81 (a lead-region crop or a finer encode: what r would it have to reach?).
* The rule's closing input is the rate of the gap: the rule imagines the lead at max(0, v_l + c) (read from the code, `dist_rule.py` line 13). A value below zero says the gap shrinks and the rule brakes more.

## 2. Part A: the token-head table in the rule (arms TK, STK)

* Arm TK: the saved table of the T25 head of card ADD81, one value per clip and frame of the 141 evaluation clips (`tokclose_T25_s0_table.npz` dc85f3eb4cb558d1 with model and reader seed 0, `tokclose_T25_s1_table.npz` 6961a9e33f4e55dd with seed 1), fed as the closing input of form RRe by `--closing-table`. No new fit.
* Arm STK: the surrogate of the TK table by the map of card ADD83 (the estimate of the paired other clip at the same frame, the map built from the reader's presence share of the same seed).
* Runs (10k line models `UE_s0`, `UE_s1`; reader seeds 0, 1; readout K; `--rule-every 5`; heading bound 0.2; q50 map): `I10RQ50eTKKF5P30Hb20` (AGE), `I10RQ50eTKKF5Hb20` (FRESH), `I10RQ50eSTKKF5P30Hb20` (AGE), seeds 0 and 1: six rollouts.
* Pairs, A minus B: `TOK-VS-RR`, `TOK-VS-EST-L25`, `TOK-VS-EST-N10`, `TOK-VS-LABEL-CLOSING`, `TOK-VS-NO-RULE` (AGE and FRESH); `TOK-VS-SHUF`, `SHUF-VS-RR` (AGE). Names `TABCLOSE10K-<PAIR>-<SETTING>-C-<LOWER|HIGHER|NOT-SEPARATED>`, the same for U and F, one name per family.
* Verdicts: `TABCLOSE10K-TK-SUPPORTED` when at AGE TOK-VS-RR reads C LOWER, U not HIGHER and no family WORSE, and FRESH reads C not HIGHER, U not HIGHER and no family WORSE; else `-NOT-SUPPORTED`. The F name is written beside it. `TABCLOSE10K-TK-BEYOND-L25` when TOK-VS-EST-L25 at AGE reads C LOWER; else `-NOT-BEYOND-L25`. `TABCLOSE10K-TK-GAIN-FROM-THE-SCENE` when TOK-VS-SHUF at AGE reads C LOWER and U not HIGHER; else `-GAIN-NOT-SHOWN-FROM-THE-SCENE`, with the name of SHUF-VS-RR beside it.

## 3. Part B: the requirement curve (arms W00, W50, W70, W90, S70, L100)

A closing input of known quality is planted: the label closing speed with noise, as a table by clip and frame, AGE only.

* Arm L100: the closing input that form RRc feeds (the label closing speed where the label has one, no value elsewhere), as a table. The anchor of the curve.
* Arms W50, W70, W90: on the frames where L100 has a value, value = m + r^2 (L - m) + s sqrt(r^2 (1 - r^2)) e, with L the label value, m and s the mean and standard deviation of L over those frames, e independent standard normal per frame, r = 0.50, 0.70, 0.90. This is the form of a calibrated estimate: its r against the label is r and its standard deviation is r s. No value on the other frames.
* Arm S70: as W70 with a noise that is smooth in time: e follows a first-order autoregression along the frames of a clip with coefficient 0.9 per frame and unit variance (restarted at every clip). An estimate over a history has errors of this kind.
* Arm W00: value = m + 0.5 s e on the same frames: the spread of W50 with no relation to the label.
* Random numbers: `numpy.random.default_rng(84000 + 10 a + k)` with a = 0..4 for W00, W50, W70, W90, S70 and k the seed, drawn in the row order of the table.
* Runs `I10RQ50e<ARM>KF5P30Hb20`, seeds 0 and 1: twelve rollouts. Order of the card: TK AGE, STK AGE, L100, W70, W50, W90, W00, S70, TK FRESH.
* Pairs per arm X at AGE: `X-VS-RR`, `X-VS-LABEL-CLOSING` (against the banked run of form RRc), `X-VS-NO-RULE`; and `W50-VS-TOK`, `W50-VS-EST-L25`, `S70-VS-W70`, `W00-VS-RR`. Reported per arm: C, U, F levels per seed, and the share of the contact gap between forms RR and RRc that the arm removes (point value per seed and pooled; a share has no name).
* Names per arm X: `TABCLOSE10K-X-REMOVES-CONTACT` when X-VS-RR reads C LOWER and U not HIGHER, else `-NOT-SHOWN`; `TABCLOSE10K-X-AT-LABEL-LEVEL` when X-VS-LABEL-CLOSING reads C not HIGHER, else `-ABOVE-LABEL-LEVEL`.
* Verdict: `TABCLOSE10K-REQUIREMENT-R<nn>` with nn the lowest of 50, 70, 90 whose W arm has REMOVES-CONTACT and a pooled share of at least 0.5; `TABCLOSE10K-NO-TESTED-LEVEL-REACHES-HALF` when none has. `TABCLOSE10K-SPREAD-ALONE-REMOVES-CONTACT` when W00-VS-RR reads C LOWER, else `-SPREAD-ALONE-NOT-SHOWN`. The names of S70 are written beside those of W70; when they differ, the requirement is marked as depending on the time course of the error.

## 4. Controls (all have to pass before a name or a value is opened; the ones marked "before" are judged before the rollouts they guard)

* a0 (before; the anchor is the label input): at every look of the banked runs of form RRc under the same settings (replan log and mid-look log, both seeds), the L100 table at the clip and frame of the look equals the logged closing input to within 1e-6 where the log has a value other than 0, and has no value or 0 where the log has 0; the share of looks that agree is at least 0.999. No rollout of part B starts unless a0 passes.
* a1 (before; the planted quality is what section 3 says): per arm and seed, on the frames with a value, r against L is within 0.02 of the target (W00: inside [-0.03, +0.03]), the standard deviation within 0.95 to 1.05 times r s (W00: 0.5 s), the mean within 0.05 m/s of m; the lag-1 autocorrelation of the planted noise of S70 inside a clip lies in [0.85, 0.95] and that of W70 in [-0.05, +0.05]; the frames with a value are the frames of L100 exactly.
* a2 (in the runs): the estimate column of the rule log equals the table of the run at the clip and frame to within 1e-9 on the rows marked full, in all runs and both logs (the first look of the mid-look log is its smallest tick).
* a3 (the anchor reproduces form RRc): the levels C, U, F of the L100 runs equal those of the banked RRc runs per seed to within 1 window of 325.
* a4 (part A, the surrogate): u1, u3 and u4 of card ADD83 applied to the TK table (bijection with no fixed point and exact values; spread within 0.80 to 1.25 times, mean within 0.15 m/s, share of full looks at least 0.90 at AGE; r of the surrogate against the label inside [-0.15, +0.15]).
* g, s, r1, x1, m1, a1 of the reader, i1, h1: as cards ADD76, ADD79, ADD80; the levels of the banked runs (forms RR, RRc, no rule, estimates N10 and L25) recomputed here equal their result files.

A control that does not pass: the card is reported with that control failed and has no name for what the control guards; no form of a control is changed inside the card (rule of row 136).

## 5. What follows (committed before the data)

* `TK-SUPPORTED` with `TK-GAIN-FROM-THE-SCENE`: a head on the tokens is a closing read the rule can use; it goes into the proposal to the PI and the other session beside cards ADD82 and ADD83, and the T0 follow-up (a longer fit, more head seeds) is worth its minutes.
* `TK-NOT-SUPPORTED`, or the gain not shown from the scene: a read at r 0.47 is not enough in the rule; the requirement of part B says what would be.
* `REQUIREMENT-R50` or `-R70`: the target for a better closing read is written into the cost card to the PI (lead-region crop or finer encode) as an r to reach at T0, with the share of contact it buys.
* `REQUIREMENT-R90` or `NO-TESTED-LEVEL-REACHES-HALF`: the closing read is a weak lever at any quality a camera read is likely to reach; the cost card says so, and the next lever is the rule itself (a cautious form, card ADD82) or a planner trained on its own rolled history (the other session's line).
* `SPREAD-ALONE-REMOVES-CONTACT`: every gain of a noisy closing input in this card and in row 141 is read against arm W00 and not against form RR.

## 6. Limits known in advance

Two model seeds, two reader seeds; one noise draw per arm and seed, not a distribution of draws; a calibrated Gaussian error is a model of an estimate, not an estimate, and the pairs W50-VS-TOK and W50-VS-EST-L25 are the check of how far the model holds; the label closing speed is zero on a large share of looks and is itself a one-second slope of a label gap; logged frames and a lead that does not react; the choice of the T25 head for part A was made after row 143 on the same 141 clips. Part B says what a read of a given quality would buy in this rule and this tier; it does not say that such a read can be had from the camera.
