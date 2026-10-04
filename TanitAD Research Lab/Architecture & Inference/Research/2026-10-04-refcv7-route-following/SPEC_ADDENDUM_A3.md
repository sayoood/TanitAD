# SPEC addendum A3 — the NONLINEAR re-scorer (pre-registered BEFORE it is computed)

**Written 2026-10-04 after A2's linear re-scorer was fitted on TRAIN (5-fold episode-grouped CV ADE 1.469 m vs the
shipped pick's 1.484 m — a 0.015 m gain) and before A2 or A3 was scored on EVAL.** sha256 + time in
`raw/SPEC_SHA256.txt`. POST-HOC; reported in its own section.

**Why.** A linear re-scorer failing says only *"not recoverable by a LINEAR map of these terms"* (CLAUDE.md: a
negative from a linear probe is not a negative about learnability). The cheapest discriminating experiment is the
same features through a nonlinear function class.

**Model (fixed now, no tuning on EVAL):** `sklearn.ensemble.HistGradientBoostingRegressor(max_iter=300,
learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False, random_state=0)`, POINTWISE
target `ADE_c − min_c' ADE_c'` per `reach_keep` candidate (TRAIN windows with a defined oracle), features = A2's 16
plus `v0` and the within-window ranks of `s_e9` and of the sampler confidence (19). Emitted pick = argmin predicted
target within `reach_keep`. Reported: 5-fold episode-grouped CV ADE on TRAIN, in-sample TRAIN ADE, EVAL seed 0 and
seed 1 with SPEC §5's metrics, paired episode-cluster bootstrap vs V0, and SPEC §5's four-part bar (unchanged).
Arm Y1 = all 19 features (the reported arm); Y2 = without the nav features (attribution).
