# HOLD_DEFS (HOLDREAD, PREREG_V8F_PROOF_ADD15 sha256 168ad1a7ddc9...; MEASURED, T1-kin on logged frames, zero training; the hold rule is an EMULATION on banked rollouts)
Code: F:/TanitAD-scratch/v8proof/hold/holdread.py (line numbers below). Tests: test_holdread.py (7 tests, OK, tests.log).

## 1 Data
* Pair 1 and pair 2 dump folders: those of domprem.py PAIRS (DOMPREM_DEFS.md): DAG / FLATDAG _s{0,1}_m10_causal_heads and DAG0 / FLATDAG0 _s{0,1}_m10_causal_heads under E:/Projects/TanitAD-artifacts/v8f-dagger-t1 and v8f-flatdag-t1. 141 ep*.npz each, sorted order (holdread.py:50). Keys g, cl, ol, ws, eid, clip_index, v0.
* holdread.py:62 loads every key itself (to get `ol`) and asserts g, cl, ws, eid, clip_index, v0 equal hierwhere.load_arm bit for bit.
* Meaning of `ol`, confirmed from the writer: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/stack/scripts/v8f_t1_eval.py. `rollout()` (line 399-424) docstring: "Returns plan0 [B, 30, 2] (the tick-0 plans)"; lines 415-416 `if plan0 is None: plan0 = plan.copy()` on n = 0; line 545 `ol.append(plan0)`; line 552 `"ol": np.concatenate(ol)`. So `ol` is the plan at tick 0 of the rollout, i.e. the plan at the window start, held open loop. The read proceeded.

## 2 Definitions as implemented
* REST v0 < 2.0 (holdread.py:202); STAY REST and ||g[29]|| <= 2.0 (:203); GO = REST and not STAY. T_cl, T_ol = norm of cl[29], ol[29] (:208-209).
* Hit (:80), ADE6 (:75), standing track ADE6 = mean over knots of ||g|| (crossfit, :111).
* Folds (:85): REST episodes sorted, np.random.default_rng(0).permutation, dealt round-robin, 5 folds. Theta (:92): grid ascending, lowest mean ADE6 of the rule over the fit-fold REST windows, strict improvement needed to replace, so ties go to the smaller theta. Rule (:102): hold when T_ol < theta on REST (inverted: >=), track set to 0 for 30 knots. Cross-fit (:111).
* AUROC (:132): GO positive, ties half. Cross-track (:143), families (:163): LON speed MAE = hierwhere.comps speed_mae_mps; TAC = joint cell (hierwhere modes/joint) equals the logged cell; STRATEGIC PAUSED.
* Estimator: hierwhere.boot (taniteval.ci draws, B 2,000, seed 0). Labels: :334 Q1, :338 Q2, :347 Q3. Controls: :283.

## 3 Reading choices (none moves a threshold, bar, grid, fold rule, tolerance or outcome rule)
1. Fold dealing: "shuffled with seed 0, dealt round-robin" read as np.random.default_rng(0).permutation of the sorted REST episode ids, element i of the permutation to fold i mod 5. The same folds are used for every arm, pair and seed (they depend only on the REST episodes).
2. Theta ties: "the smaller theta" implemented as strict less-than over the ascending grid.
3. Hold test: strict T_ol < theta (card wording).
4. Bootstrap for Q3 holds the cross-fitted tracks fixed (theta is not re-fitted inside each draw); the interval is paired on windows, resampling episodes.
5. All-windows differences: sum over REST windows of the per-window difference divided by the number of all windows in the draw (outside REST the difference is 0 by construction).
6. Intervals for the Q1 median and 90th percentile are episode-cluster bootstrap on the STAY windows of each draw (the card prints the statistics; the interval is an addition).
7. Q2 AUROC interval: AUROC recomputed on the REST windows of each draw.
8. Control c6 is decided on the hierarchy arm (the card-outcome arm) at both seeds of pair 1; the flat arm is printed as c6info and cannot fail the read. Both read inverted > rule.
9. c2 and c3 and c1, c4 run for every pair, seed and arm; c5 for every pair and seed.
10. Cross-track: "last valid logged step" uses the step-validity gate of hierwhere.kinematics (step length > FF.MIN_DS_MPS x 0.2); fewer than 2 valid steps, including exactly 1, uses |y| of the endpoint error; the counts are in the verdict.
11. Labels for pair 2 are computed by the same rules and printed beside; the card outcome is pair 1 only.
12. Q1 median label threshold: "above 2.0 m" strict, on the point value.
13. Outcome HOLD-HELPS / HOLD-COSTS: point and interval both on the same side of 0 at both seeds.
