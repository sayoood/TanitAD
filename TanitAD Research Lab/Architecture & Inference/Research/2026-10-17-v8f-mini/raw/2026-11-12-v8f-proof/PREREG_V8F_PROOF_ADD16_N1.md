# PREREG_V8F_PROOF_ADD16_N1: a secondary read for C1, the change form of the lead read (2026-11-13)

Written after the r-b read of the arms G_s0, G_s1 and W_s0 (`spatial/q2/out/VERDICT_RB_*.md`) and BEFORE any data of the arms P exist
(P_s0 passed its gate at 23:56Z and is in its first steps; no checkpoint of P exists, none is read). It changes no outcome name, bar or rule of the card.

## 1. Why

MEASURED on G (T0, controls t1 to t6 pass, 441 windows with a lead at both times, 139 episodes): lead gap R2 at +2.0 s read from the rolled-out state
0.442 / 0.437 (seeds 0 / 1), from the present state (copy) 0.496 / 0.473, from the true future state 0.552 / 0.543.
The rollout of the flat predictor does not beat the copy: -0.054 [-0.154, +0.033] / -0.036 [-0.153, +0.064].
The true future state itself adds only +0.056 [-0.004, +0.115] / +0.070 [+0.001, +0.142] to the copy: on this LEVEL read the gap at +2.0 s is mostly the gap now.
So `CELL-IMAGINES`, which asks the rollout to beat the copy with the interval excluding 0, sits at the edge of what a perfect predictor would reach at seed 0.
The level read stays as written and decides the names of the card. This note adds a read that isolates the motion.

## 2. The change form (names nothing)

* Windows, probes, folds and predictions: those of the r-b read, unchanged; no new fit. Inputs: the out-of-fold predictions `OOF_RB_<ARM>.npz` and the bank.
* Target: the change of the lead gap, gap(t + 2.0 s) minus gap(t), on the windows with a lead at both times.
* Predictions of the change: rollout = probe(rolled-out state) minus probe(present state); ceiling = probe(true future state) minus probe(present state). The copy predicts 0 by construction.
* Skill = 1 minus the sum of squared errors over the sum of squares of the target (skill over persistence); episode-cluster bootstrap, B 2,000; differences between arms paired on the same windows.
* Reported for P, G and W: the change skill of the rollout and of the ceiling, and P minus G (rollout) paired. For G and W this read is run after their level numbers were seen; for P it is registered here before its data.
* Controls: k1 the copy put in the place of the rollout reads exactly 0; k2 with the target shuffled across windows (fixed seed) every skill is at or below 0.02; k3 the identity: target equal to the ceiling prediction reads exactly 1.

## 3. Reading committed in advance

* The outcome name of C1 is the one the card defines on the level read; its committed next step runs whatever this note reads.
* If the change skill of the rollout of P is above 0 with the interval excluding 0 at both seeds AND above that of G (paired, interval excluding 0 at both seeds), the report adds one sentence: the cell predictor moves the lead in the state, and the level read at 2.0 s has little room to show it. A read at a longer horizon (4.0 s) is then proposed in its own card.
* If not, nothing is added: the level name stands alone.
