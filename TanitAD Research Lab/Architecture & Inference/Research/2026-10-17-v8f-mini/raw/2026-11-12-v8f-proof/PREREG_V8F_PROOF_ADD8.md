# PREREG_V8F_PROOF addendum 8: GATEMIX, can the confidence of the tactical reader decide when the hierarchy should drive (frozen 2026-11-13 01:38 dev-box clock, before its data)

HIERWHERE and HIERWHERE-2 (addenda 6, 6b, read): the hierarchy is ahead of its flat control on selection-critical windows and, without own-state supervision, behind it on bulk windows; critical minus bulk is negative and separated in all four pair-seed cells. If the tactical output of the hierarchy can tell the two window groups apart, a commit rule that lets the hierarchy drive only then should be at least as good as both arms. This card tests that on the banked rollouts, zero training. Tier T1-kin on logged frames, a diagnostic, never driving performance. STRATEGIC: PAUSED (PI item 25).

## Data (banked, read-only)
The two pairs of addenda 6 and 6b: (DAG, FLAT-DAG) and (DAG0, FLAT-DAG-0), seeds 0 and 1; per window the ADE6 of each arm as in hier/hierwhere.py, and the tactical head outputs of the hierarchy arm at tick 0 from its heads dump (lateral and longitudinal class probabilities). Each T1-kin window starts from the logged state, so choosing per window which arm drives the window is a valid composite.

## The commit rule (fixed now; one primary score)
* Common-case joint class: the most frequent pair (argmax lateral class, argmax longitudinal class) of the hierarchy arm at tick 0 on the training folds.
* Score s(w): the largest product p_lat(i) * p_lon(j) over joint classes other than the common-case one, at tick 0 of window w. A high score means the hierarchy is confident of a decision that is not the common case.
* Composite: the hierarchy drives window w when s(w) >= tau, else the flat arm drives. tau is chosen on the training folds as the value that minimises the composite mean ADE6 there (candidate set: the 101 quantiles of s on the training folds, plus the two values that give all-hierarchy and all-flat). 5 folds by episode, fold seed 0; every held-out window is decided by a tau that never saw its episode.
* ORACLE gate (ceiling, privileged): the hierarchy drives exactly the logged selection-critical windows.

## Reads (paired episode-cluster bootstrap, B 2,000, seed 0; per pair and seed)
Composite minus hierarchy and composite minus flat on all windows (ADE6 and 2-6 s ADE); the same for the ORACLE gate; the share of windows driven by the hierarchy; the same differences on selection-critical and on bulk windows; AUROC of s(w) for the logged selection-critical label. Four families for the composite against each arm: LONGITUDINAL (speed error, along-track), LATERAL (cross-track, heading, curvature, yaw rate), TACTICAL (the AUROC above and the share), STRATEGIC PAUSED.

## Outcomes (fixed now)
* Per pair and seed: GATE-BEATS-BOTH if composite minus hierarchy and composite minus flat are both negative with intervals excluding 0; GATE-WORSE if either is positive with the interval excluding 0; GATE-TIES otherwise.
* Overall: GATE-DOMINATES if GATE-BEATS-BOTH holds at both seeds of at least one pair and no cell is GATE-WORSE; GATE-COSTS if any cell is GATE-WORSE; GATE-TIES otherwise.
No outcome is a claim that the hierarchy beats flat as an architecture: one training run per arm per seed, no replicate arm, a threshold cross-fitted on the evaluation clips, and a composite that needs both models to run. It answers one question: does the tactical output carry the information to commit only where committing pays.

## What each outcome commits
* GATE-DOMINATES: the commit rule becomes a trained arm (the operative reader falls back to a common-case plan below the threshold), with replicate arms, behind the launch gate. GATE-TIES or GATE-COSTS: the tactical confidence does not carry it; the lever is the decision accuracy itself (the stop cue of addendum 5 and the tactical labels), not a gate.

## Controls (each must read its known value)
* g1: tau below every score reproduces the hierarchy arm exactly, tau above every score reproduces the flat arm exactly (0.0 m).
* g2: the banked ADE6 differences are reproduced as in addenda 6 and 6b (-0.3451 / -0.0544 for the first pair; the h1a route for the second).
* g3: with s shuffled within fold, composite minus the better arm is not negative with an interval excluding 0, for at least 18 of 20 shuffle seeds per pair and seed.
* g4: the ORACLE composite equals the value implied by the banked group means (share-weighted) to 1e-9 m.

## Limits stated in advance
Post-hoc on banked runs whose primary outcome is NOT ESTABLISHED. Floors beside every number: F = 0.12075 m, tactical replicate floor 0.287 m. Deviations go in the verdict, never in this card.
