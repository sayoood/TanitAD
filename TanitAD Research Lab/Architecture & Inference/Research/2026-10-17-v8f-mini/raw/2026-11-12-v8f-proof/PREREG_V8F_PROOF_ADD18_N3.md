# PREREG_V8F_PROOF_ADD18_N3 : MOTION, calibrated forms of the pooled lead read

Note to card `PREREG_V8F_PROOF_ADD18.md` and note `PREREG_V8F_PROOF_ADD18_N2.md`. Frozen while RR_s2 trains and before RR_s3 exists: no checkpoint, no state and no read of either new seed exists. The card, note N2 and their names stay as written. Tier of every read: T0 (a world-model diagnostic on logged frames, never driving performance).

## 1. What was seen when the pooled tool of note N2 was tried on the two known seeds (MEASURED, `spatial/q2/pool4/smoke_RR/`)

* The tool reproduces the per-seed values of `rb_trained.py` (p1: difference 0.0), reads exactly 0 with the copy given as the rollout (p2) and uses the same 441 windows of 139 episodes in each seed (p4).
* Control p3 of note N2 (labels permuted across windows inside each seed, the reads left as they are) FAILS for RR: statistic A reads +0.062 [+0.027, +0.102] with labels that carry nothing. Cause: A is a difference of two R2, and with an uninformative label that difference is the difference of the spreads of the two reads over the spread of the label; the rolled-out read of RR has the smaller spread (6.44 and 6.62 m against 6.95 and 6.89 m for the present-state read; label 10.02 m). On the real labels A is +0.046 [+0.015, +0.082] over the two seeds. So a part of A of unknown size is spread and not information, and by note N2 (all controls must pass) its names cannot be given while p3 fails. p3 stays as written and is read again on four seeds.
* The same permutation leaves a calibrated form at -0.008 [-0.018, +0.006]: when every read is first mapped to the label by a linear fit made out of fold, a smaller spread gains nothing.
* Calibrated values on the two known seeds (declared as known; they chose the statistic and confirm nothing): the rolled-out read added to the present-state read +0.046 / +0.040, pooled +0.043 [+0.003, +0.086]; added to the present-state read and the ego-only floor +0.032 / +0.025, pooled +0.029 [-0.006, +0.066]; the most the probe can show (the true future state in place of the rollout) +0.084 [+0.037, +0.142] and +0.078 [+0.036, +0.129]. Base arm G the same way: +0.020 [-0.014, +0.066] and +0.011 [-0.020, +0.050]. Arm ROLL: +0.023 [-0.016, +0.062] and +0.009 [-0.026, +0.045].

## 2. The calibrated statistics (tool `spatial/q2/pool4/pool_rb.py`; same windows, labels, probes and two-level bootstrap as note N2)

Per seed, on the windows with a lead at both times, five folds by episode, a linear fit of the label on the named reads in four folds, applied to the fifth:

* A3 = R2 of (present-state read, rolled-out read) minus R2 of (present-state read): what the rollout adds to the present state.
* A3e = R2 of (present-state read, ego-only floor read, rolled-out read) minus R2 of (present-state read, ego-only floor read): what the rollout adds beyond the present state and the ego motion (the floor read knows the future ego vector).
* Headroom, A3c and A3ec: the same with the read of the true future state in place of the rollout.
* Pooled over seeds as in note N2 (seeds with replacement, then episodes; B 2,000; seed 0).

## 3. Names (added by this note; the two new seeds carry the confirmation)

* `MOTION-ADDS-4`: A3 is above 0 in both new seeds (2, 3); the interval of the two new seeds alone lies above 0; the four-seed interval lies above 0. Else `MOTION-ADDS-NOTHING-4`.
* `MOTION-ADDS-LEAD-4`: the same three conditions for A3e. Else, with `MOTION-ADDS-4`, `MOTION-ADDS-EGO-ONLY-4`.
* Reported beside, naming nothing: the share of the headroom reached (A3 over A3c, A3e over A3ec); RR minus the base arm G of the same seed on A3 and A3e (two-level, the same draws for both arms) for the seeds whose base read exists; the same statistics for ROLL and RES (seeds 0, 1).

## 4. Controls of the calibrated read (all must pass; else none of the names of section 3 is given)

* n1: p1, p2 and p4 of note N2 pass (same tool run).
* n2: with the labels permuted as in p3, A3 and A3e do not lie above 0 (read on the two known seeds: -0.008 [-0.018, +0.006] and -0.007 [-0.016, +0.006]).
* n3 (known positive): with the read of the true future state given as the rollout, the four-seed interval of A3 lies above 0 (the read can show an effect of the size of its headroom).
* n4: with the copy given as the rollout, A3 and A3e are 0 to within 1e-9.

## 5. What follows (committed before the data of the new seeds)

* `MOTION-ADDS-LEAD-4`: the rollout of RR holds lead information 2 s ahead that neither the present state nor the ego motion gives. Next, own card: the rolled-out states are read with a supervised lead head (an arm with the residual form, the rollout loss and the lead head), and candidate plans are scored by the imagined gap (planning by imagination), as section 6 of the card commits.
* `MOTION-ADDS-EGO-ONLY-4`: the rollout carries how the ego's own motion changes the gap and not the lead's motion. That is the part a planner needs to compare its own candidate plans against a lead that keeps its motion; the same next card, with the claim limited to the ego part, and the explicit object-motion target (arm PF) stays owed for the lead part.
* `MOTION-ADDS-NOTHING-4`: the values of the two known seeds did not hold; RR is not promoted on this read; the levers of the card for `MOTION-FLAT` run.
* In every case the names of note N2 are reported as that note rules (with p3 failing: not given, with the cause of section 1).

## 6. Limits

The calibrated statistics were chosen after the two known seeds were read with them; the confirmation therefore rests on the two new seeds, and the four-seed interval includes the two seeds that chose the statistic. One training run per seed; linear probes and a linear calibration; one corpus; T0 reads on logged frames with the logged ego actions given to the rollout. The headroom of the probe is small (about 0.08 of the label variance), so an absent effect is weak evidence.
