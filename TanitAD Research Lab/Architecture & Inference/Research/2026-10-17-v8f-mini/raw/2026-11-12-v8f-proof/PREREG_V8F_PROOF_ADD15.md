# PREREG_V8F_PROOF_ADD15: HOLDREAD, does the hierarchy hold a standstill, and would a hold rule help? (2026-11-13)

Frozen before any number of this card is computed. Zero training, zero rollouts: a read of banked T1-kin rollouts (kinematic rollout on the model's own actions, logged frames) on the dev-box CPU. Never driving performance. STRATEGIC family: PAUSED (PI item 25).
Theorem served: T3 (a rule in a shield holds by construction; what a rule needs is a cue).
Why this read, and what is already known and not blind (DOMPREM, addendum 14, `domprem/VERDICT_DOMPREM.md`): in the context v0 below 2 m/s (211 windows, 29 episodes) one fixed plan hits 31.3 % of the logged 6 s endpoints at 2.0 m, the hierarchy 7.1 % / 5.2 % and the flat arm 1.9 % / 2.8 %; lateral-mode accuracy there is 0.22 / 0.24. Not computed on any dump: how far the arms travel on windows where the log stays, any separation of stay from go by the plan, any effect of a hold rule.

## 1. Questions
Q1 (creep). On windows that start at rest and where the logged ego stays, how far do the arms travel in 6 s?
Q2 (cue). At rest, does the arm's own plan at the window start tell logged stay from logged go?
Q3 (rule, emulated). Does a hold rule keyed on that plan lower the error at rest?

## 2. Data
The dumps of addendum 14: pair 1 (DAG_s0, DAG_s1, FLATDAG_s0, FLATDAG_s1) and pair 2 (the dumps `hier/pair2/hierwhere2.py` reads). Arrays, 30 knots at 0.2 s, ego frame with the origin at the window start: `g` logged future, `cl` executed T1-kin track, `ol` the plan held from the window start (open loop). The agent confirms the meaning of `ol` from the writer of the dumps and states the file and line; if `ol` is not the plan at the window start, the read stops before any number.

## 3. Definitions (fixed here)
* REST: windows with dump `v0` < 2.0 m/s. STAY: REST windows with ||g[29]|| <= 2.0 m. GO: the other REST windows.
* Travel: T_cl = ||cl[29]||, T_ol = ||ol[29]||.
* Hit: ||x[29] - g[29]|| <= 2.0 m for a track x. ADE6: mean over the 30 knots of ||x - g||.
* Hold rule H(theta): on REST windows with T_ol < theta the track is replaced by standing at the origin for all 30 knots; every other window keeps `cl`. This is an emulation on banked rollouts, not a rollout.
* Cross-fitting of theta: 5 folds by episode (episodes of the REST context, shuffled with seed 0, dealt round-robin); on the four fit folds theta is the value of the grid {0.5, 1, 2, 3, 4, 6, 8, 12} m with the lowest mean ADE6 over their REST windows (ties: the smaller theta); it is applied to the held-out fold. The held-out predictions of the five folds form the ruled arm.
* Privileged ceiling: the rule with the logged STAY label in place of the cue.
* Estimator: `taniteval.ci` episode-cluster bootstrap, B 2,000, seed 0, paired on the same windows (ruled arm minus the arm as banked), through `hier/hierwhere.py` `boot`.
* Four families, each as ruled arm minus the arm as banked on REST windows: LONGITUDINAL = mean absolute speed error over the 30 knots (speed from step displacements as HIER_DEFS section 2); LATERAL = mean absolute cross-track error at 6 s (the component of x[29] - g[29] orthogonal to the logged heading of the last valid logged step; windows with fewer than 2 valid logged steps use the ego-frame lateral axis and are counted); TACTICAL = share of windows whose joint cell (HIER_DEFS) equals the logged cell; STRATEGIC = PAUSED.

## 4. Outcome rules (both outcomes committed)
* Q1, printed per arm: median and 90th percentile of T_cl on STAY windows, and the hit rate there. Label CREEP if the median T_cl of the hierarchy on STAY windows is above 2.0 m at both seeds of pair 1, else HOLDS.
* Q2, per arm: AUROC of T_ol for GO against STAY on REST windows, with interval. Label CUE-PRESENT if the lower bound is above 0.80 at both seeds of pair 1 for the hierarchy, CUE-ABSENT if the upper bound is below 0.65 at both, else CUE-WEAK. Printed beside: the AUROC of `v0` alone (the label comes from the logged motion, so `v0` partly reads its source).
* Q3, card outcome, hierarchy arm of pair 1, REST windows, ADE6 of the cross-fitted ruled arm minus the arm as banked: HOLD-HELPS if below 0 with an interval that excludes 0 at both seeds; HOLD-COSTS if above 0 in the same sense at both; else HOLD-UNDECIDED. Printed beside: the same on all 2,399 windows (windows outside REST are unchanged by construction), the hit-rate change, the four family rows, the flat arms, pair 2, the chosen theta per fold, the privileged ceiling.
* One training run per arm per seed: a separated interval is necessary and not sufficient.
* What follows, committed now. HOLD-HELPS: the next step is the rule inside a real T1-kin rollout (the tooling of RULESHIELD), theta fitted on training clips, with the four families and the six seeds of HIERREP as the replicate. HOLD-UNDECIDED or HOLD-COSTS with CUE-ABSENT or CUE-WEAK: the rule lacks a cue; the ceiling row says what a cue would be worth, and the lever is a go / stay cue at rest from perception, not the rule. HOLD-COSTS with CUE-PRESENT: the emulation is the suspect; the rollout version decides.

## 5. Controls (each must read its known value, else the read is not valid)
* c1: H(0) changes no window (maximum absolute difference 0.0).
* c2: H(infinity) on REST windows has a hit rate equal to the share of STAY windows among REST windows (exact), and outside REST it changes nothing.
* c3: the privileged rule has hit rate 1.0 on STAY windows (exact) and leaves GO windows unchanged.
* c4: no episode is in both the fit folds and the held-out fold of a split (overlap 0).
* c5: the counts of REST windows and episodes equal those of DOMPREM (211 windows, 29 episodes).
* c6 (deliberate regression): the rule with the cue inverted (hold when T_ol >= theta, same theta per fold) has a higher mean ADE6 on REST windows than the rule as defined, at both seeds of pair 1.

## 6. Limits declared
* An emulation: a held car in a T1-kin rollout decides again at 2 s and 4 s; here it stands for the whole window. The rollout version is the committed next step and is the number that would count.
* theta is fitted on evaluation episodes by cross-fitting, not on training clips (no training rollouts are banked on this box).
* 211 windows of 29 episodes: wide intervals are expected. STAY is defined with the logged future, so it is a label, not an input.
