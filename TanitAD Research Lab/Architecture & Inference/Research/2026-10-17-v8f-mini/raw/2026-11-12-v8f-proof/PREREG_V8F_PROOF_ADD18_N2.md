# PREREG_V8F_PROOF_ADD18_N2 : MOTION, arm RR with two more seeds and a pooled read over four seeds

Note to card `PREREG_V8F_PROOF_ADD18.md` (with note `ADD18_N1`). Frozen before the arms RR_s2 and RR_s3 exist (no gate, no training step, no read). The card and its names stay as written; this note adds arms and one added read with its own names.

## 1. What was seen (MEASURED, T0, world-model diagnostics on logged frames, never driving performance)

Arm RR (residual form plus a rollout loss over 2.0 s) with both seeds of the card (row H-V8F-PROOF-60):

* One step ahead, error ratio predictor over copy: 0.870 [0.858, 0.882] / 0.870 [0.859, 0.881] (`STEP-BELOW-COPY`).
* Lead gap read 2.0 s ahead, rollout minus copy: +0.050 [+0.003, +0.098] (seed 0), +0.042 [-0.005, +0.091] (seed 1); rollout minus the ego-only floor +0.306 [+0.087, +0.619] / +0.315 [+0.076, +0.668]. By the rule of the card (both seeds, the weaker name) the name is `MOTION-FLAT`: seed 1 misses the copy bar by 0.005.
* Change of the lead gap, skill over persistence: +0.091 [-0.008, +0.212] / +0.070 [-0.023, +0.171] (`MOTION-STILL`).
* `KEEPS-POSITION`, `NO-COST`.

The episode-cluster bootstrap answers whether another draw of episodes would say the same; it is blind to training variance. Two seeds with the same size and sign, one of them just inside 0, are an indication. The cheapest experiment that can settle a miss of this size is more seeds, not more windows.

## 2. Added arms

* RR_s2 and RR_s3: the arm line of RR_s1 (chain motion_d, tree `tree_motion`, flags `--wm-state grid --wm-cell-dim 16 --wm-l1-residual on --wm-l1-roll-k 10 --w-wm-roll 1.0 --wm-l1-diag on`, 19,164 steps, from scratch) with the seed changed to 2 and to 3. Nothing else changes.
* Chain `motion_e` on the pod, started behind chain motion_d (one trainer with a rollout loss at a time); each arm passes the gate of the chain (selfcheck token) before it trains.
* Cost (ESTIMATED from RR_s1, which took about 2.2 h beside two other queues): about 4.5 L40-h.
* Base arms of the same seeds (G_s2, G_s3, trained for card ADD22) are read with the same lead read when their states are on the dev box; the comparison against the base is reported and names nothing in this note.

## 3. Reads

Per arm, unchanged tools and controls of the card: S (`ctx_step.py`), I and V (`ctx_roll.py`, `rb_trained.py`, `rb_change.py`), R (`s1_read.py`), F (`q3_read.py`, T1-kin). The names of the card are per seed pair and are not re-read by this note.

## 4. The pooled read (added; names of this note)

Windows and probes exactly as the lead read of the card (441 windows with a lead at both times, 139 episodes, out-of-fold probes of `rb_trained.py`).

* Statistic A (imagination): for each seed k the difference d_k = R2 of the rolled-out state minus R2 of the copy; the pooled value is the mean of d_k over the seeds. Interval: two-level bootstrap (seeds with replacement, then episodes with replacement inside each drawn seed), B 2,000, seed 0, percentiles 2.5 and 97.5.
* Statistic A2: the same with the ego-only floor in place of the copy.
* Statistic V (change): the change skill of the rollout over persistence per seed, pooled the same way.
* Two scopes are reported: all four seeds (0, 1, 2, 3; seeds 0 and 1 are already known and are declared as such), and the two new seeds alone (2, 3).

Names (all conditions needed):

* `MOTION-IMAGINES-4`: the four-seed interval of A lies above 0; the four-seed interval of A2 lies above 0; d_k is above 0 in at least three of the four seeds; and d_k is above 0 in both new seeds (the replication bar, which the known seeds cannot meet for the new ones). Else `MOTION-FLAT-4`.
* `MOTION-MOVES-4`: the four-seed interval of V lies above 0; the change skill is above 0 in at least three of the four seeds and in both new seeds. Else `MOTION-STILL-4`.
* Reported beside, names nothing: the intervals of the new seeds alone; RR minus G of the same seed on the rollout read and on the change skill for the seeds whose base read exists; the one-step ratio, position and family reads of the new arms.

## 5. Controls (all must pass; else nothing is named)

* The per-arm controls of the card for RR_s2 and RR_s3 (lead read t1 to t6, change k1 to k3, step s1 and s2, family q1 to q5) and the gate token of each arm.
* p1: the pooled tool run on one seed alone gives the point value of `rb_trained.py` for that seed (absolute difference at most 1e-9).
* p2: with the copy given as the rollout, statistic A is exactly 0.
* p3 (deliberate regression): with the lead labels permuted across windows inside each seed, statistic A does not lie above 0.
* p4: the four-seed read uses each seed once before resampling and the same 441 windows in every seed (counts printed).

## 6. What follows (committed before the data)

* `MOTION-IMAGINES-4`: RR is the proposed level-1 predictor form of the next flagship arm; next lever, own card: planning by imagination on it (candidate plans scored by the imagined lead gap), as section 6 of the card commits for `MOTION-IMAGINES`. With `MOTION-STILL-4` beside it, the change read stays the open cell and the explicit object-motion target (arm PF) stays owed.
* `MOTION-FLAT-4`: the result of two seeds did not hold over four; RR is not promoted; the levers of the card for `MOTION-FLAT` with `STEP-BELOW-COPY` run (explicit object-motion target, second seed of PF), and the attribution arms ROLL and RES say which part of RR to keep.
* In both cases the attribution of RR to the residual form or the rollout loss is read from ROLL_s0, ROLL_s1 and RES_s1 (chain motion_d) by the rule of the card.

## 7. Limits

Four seeds; one corpus; probes show what is linearly readable; T0 reads on logged frames; the pooled names are added names of this note and do not replace the names of the card; the base arms of seeds 2 and 3 may lack the lead read, in which case the comparison against the base is reported for seeds 0 and 1 only.
