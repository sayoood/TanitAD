# PREREG_V8F_PROOF addendum 7: RESCURVE, is the resolution probe limited by rank, by data or by information, and does selection help at equal budget (frozen 2026-11-13 01:38 dev-box clock, before its data)

RESPROBE (addendum 1, read): outcome RES-REFUTED. The 8x20 grid added nothing over 4x10 (lead gap +0.0111 R2 [-0.0233, +0.0479]) and absolute levels are low (lead gap R2 0.31, lead present AUROC 0.79), with a ridge on 256 principal components and 139 episodes (lead gap: 56). Three explanations are open: the rank cut hides what the finer grid adds; the probe is short of data; the content is not in the pooled tokens. This card separates them with zero training, on the cached Gram matrices (res/cache/K40.npy, K160.npy, frames.npz). It also measures theorem T8 at probe level. Tier T0, a perception diagnostic, never driving performance. STRATEGIC: PAUSED (PI item 25), not applicable.

## Reads (targets GAP, NEAR, COUNT, LEADP; sets V40+E and V160+E, plus E as the floor; 5 folds by episode with the RESPROBE fold assignment; episode-cluster bootstrap B 2,000; paired differences)
* A. RANK: probe P1 at rank 16, 64, 256, 1,024 and with the full kernel (kernel ridge, no rank cut; ridge strength by the same inner split). For each rank: the metric and the paired difference V160+E minus V40+E.
* B. DATA: training-episode fractions 1/8, 1/4, 1/2, 1 of each training fold (nested subsets, 5 random draws, the held-out fold unchanged), at rank 256 and at the rank the inner split prefers in A. The metric against the number of training episodes, and the gain of the last doubling (1/2 to 1), paired, averaged over the draws.
* C. SELECT (theorem T8), at budgets 1/4 and 1/2 of the training episodes, rank 256: two label-free rules computed on the training folds only, (i) k-centre greedy on episode-mean kernel features, (ii) episodes ranked by summed ridge leverage of their frames, against 20 random draws at the same budget.

## Outcomes (fixed now)
* A: RANK-HIDES if, at rank 1,024 or with the full kernel, V160+E minus V40+E on GAP is >= +0.03 R2 with the interval excluding 0; RANK-FLAT otherwise. Reported per target as well.
* B, per target and set: DATA-LIMITED if the last doubling gains >= 0.03 (R2, or AUROC for LEADP) with the interval excluding 0; SATURATED if the upper end of its interval is below 0.03; UNDECIDED otherwise.
* C, per target: SELECT-HELPS if a rule is above the best of the 20 random draws at both budgets; SELECT-HURTS if below the worst at both; SELECT-FLAT otherwise.

## What each outcome commits
* RANK-HIDES: the grid comparison of RESPROBE was rank-limited; the finer grid stays a live lever and the 8x20 reader arm is sized. RANK-FLAT with DATA-LIMITED: more labelled clips come before a finer grid. RANK-FLAT with SATURATED: the content is not linearly present at either pooled grid; un-pooled tokens and the native-resolution crop on the pod are the remaining test.
* SELECT-HELPS: the selection rule is carried to the choice of training clips (its own card, with a trained arm). SELECT-FLAT: no selection claim is made from T8 for this representation.

## Controls (each must read its known value)
* c1: at fraction 1 and rank 256 the banked RESPROBE P1 values are reproduced to 1e-6 (RESULT_RESPROBE.json).
* c2: targets shuffled within fold: every R2 <= 0.02, and AUROC computed per fold and averaged within 0.03 of 0.5 (the pooled form is biased below 0.5 by the fold base rates: measured, 0.467).
* c3: a random draw passed through the SELECT code path lies inside the range of the 20 random draws.

## Limits stated in advance
Probe-level statements about a frozen representation, 139 episodes. A null from these probes is not a null about what a trained reader can learn. The interval answers another draw of episodes; closed-form probes have no training seed. Deviations go in the verdict, never in this card.
