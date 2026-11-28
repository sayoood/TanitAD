# PREREG_V8F_PROOF_ADD36_N2 : LOGASSOC, does the logged data itself associate the ego's acceleration with the gap in the reversed direction?

Note to card `PREREG_V8F_PROOF_ADD36.md`. Frozen after row H-V8F-PROOF-77 (arm RL: the imagined gap responds to the ego's acceleration with the kinematic size and the reversed sign, slope -0.89; LS -0.45) and before the statistic of this note is computed: no number of it exists. The row states a hypothesis for the cause (the predictor learned the association of the logs, in which the ego's action follows the lead) and names this statistic as its test. No arm is trained, no model is read. Tier: a statistic of the logged corpus (T0 data, no model). STRATEGIC: PAUSED (PI item 25).

## 1. The statistic (eval pack of the lead reads, `eval_pack.npz`: label gap `G15` and ego speed `V` at frames 0..14 of each window at 0.2 s; frame 4 = t, frame 14 = t + 2.0 s)

* Windows: label lead present with a finite gap at frames 0..4 and at frame 14, finite ego speeds. Counts of windows and episodes are printed.
* a = (V[14] - V[4]) / 2.0 s, the mean ego acceleration over the 2.0 s. x = -2.0 * a (m): the change of the gap that this acceleration causes by kinematics over 2.0 s (the quantity on which the response slope of card ADD36 is read; a model that follows the kinematics has slope +1 on it).
* y = G15[14] - G15[4] (m), the logged change of the gap.
* b_raw = slope of y on x (with intercept). b_cond = the coefficient of x in a linear fit of y on x and on what is known at t: gap at t, ego speed at t, and the closing rate over the last 0.8 s ((G15[4] - G15[0]) / 0.8).
* Estimator: episode-cluster bootstrap over the episodes, B 2,000, rng 0.

## 2. Names

* `LOG-ASSOCIATION-REVERSED`: b_cond below 0 and its interval excludes 0. In the logs, more ego acceleration goes with a gap that opens: a model that learns the association answers an action question with the reversed sign.
* `LOG-ASSOCIATION-KINEMATIC`: b_cond at least 0.5 and its interval above 0.
* `LOG-ASSOCIATION-WEAK`: neither.
* Beside (names nothing): b_raw; the slopes of the models (LS -0.45, RL -0.89, LA +0.86) next to b_cond; the same statistic at 1.0 s (frame 9, x = -0.5 * a with a over 1.0 s); the share of the variance of y that x explains.

## 3. What follows

* `LOG-ASSOCIATION-REVERSED`: the hypothesis of row H-V8F-PROOF-77 is supported by the data the models were trained on (measured on the eval episodes of the same corpus): the reversed response is what an associative predictor must learn from driving logs, and a rule for the effect of the ego's own action has to be injected (the consistency term of card ADD37), whatever the predictor. This is stated in the paper as the reason for rule injection through imagination.
* `LOG-ASSOCIATION-KINEMATIC` or `-WEAK`: the hypothesis is not supported by this statistic; the cause of the reversed response is open and the row keeps its wording as a hypothesis with this result beside it. Next test then: an arm trained with offset actions without the term.

## 4. Controls (all must pass; else nothing is named)

* e1 (known kinematic): with y replaced by x plus noise of the spread of y drawn independently, b_raw and b_cond lie within 0.9 to 1.1.
* e2 (known zero): with x permuted across windows, the interval of b_cond includes 0.
* e3 (counts): at least 200 windows of at least 20 episodes.
* e4 (sign convention): on a planted lead that keeps a constant speed (y = (v_L - V[4]) * 2.0 + x with a constant v_L per window), b_cond is +1 to within 1e-6.

## 5. Limits

Eval episodes of the corpus (141 clips), not the training clips; a linear fit; the label lead of one tracker; the statistic shows an association in the data and not what a model has learned from it.
