# PREREG_V8F_PROOF_ADD14: DOMPREM, does v8F-mini meet the premise of the dominance theorem? (2026-11-13)

Frozen before any number of this card is computed. Zero training, zero rollouts: a read of banked T1-kin rollouts (kinematic rollout on the model's own actions, logged frames) on the dev-box CPU. Never driving performance. STRATEGIC family: PAUSED (PI item 25).
Theorem served: T2, channel (b), in the Lean form `hierarchy_dominates_flat` (`lean/src/Hier.lean`; Proposition 5 of paper v5): in one context with two modes further apart than 2 tol + 2 r, a decided plan hits strictly more outcomes than every single plan if its selector is right on more outcomes than the larger mode holds.
What is already known and not blind: the ADE reads of the same rollouts (HIERWHERE, HIERWHERE2: both arms miss the logged 12-cell mode on 53 to 71 % of windows; the logged cell (KEEP, STRAIGHT) holds 1,094 of 2,399 windows). No hit rate, no per-context accuracy and no mode separation has been computed on any dump.

## 1. Questions
Q1 (premise, selection). Per context: is the share of windows on which the hierarchy's rollout lands in the logged mode above the share of the largest logged mode?
Q2 (premise, geometry). Per context: are the two largest logged modes further apart at 6 s than 2 tol + 2 r, and does a rollout in the right mode land within tol (the r <= tol premise)?
Q3 (conclusion, read directly). Under the hit-or-miss criterion, does the hierarchy hit more windows than its flat control, and more than the best single plan of the context?

## 2. Data
* Pair 1 (on-policy data in both arms): the dumps of HIER_DEFS section 1 (DAG_s0, DAG_s1, FLATDAG_s0, FLATDAG_s1; 2,399 windows, 141 episodes per seed).
* Pair 2 (no on-policy data): the dumps that `hier/pair2/hierwhere2.py` reads, same seeds.
* Secondary, when all 24 arms of HIERREP (addendum 11) are pulled: the same script on the six seeds of both protocols, seeds as the unit, with the rule of addendum 11 (all six seed differences agree in sign and the two-level interval excludes 0). Not run now.
* Arrays: `g` (logged future) and `cl` (executed T1-kin track), 30 knots; the 6 s endpoint is knot index 29. `v0` is the dump's measured ego speed at the window start.

## 3. Definitions (fixed here)
* Mode function: HIER_DEFS section 2, unchanged, imported from `hier/hierwhere.py` (`modes`, `joint`): 12 joint cells = 4 longitudinal x 3 lateral. An arm is RIGHT on a window when the cell of its `cl` equals the cell of `g`.
* Contexts, defined without the future: POOLED (all windows) and four bins of `v0` with fixed edges 2, 8 and 15 m/s: [0, 2), [2, 8), [8, 15), [15, inf). A context with fewer than 100 windows or fewer than 20 episodes is printed as TOO-SMALL and takes no part in an outcome.
* p_max(c): share of the largest logged joint cell among the windows of c. acc_a(c): share of windows of c on which arm a is RIGHT. Margin m_a(c) = acc_a(c) - p_max(c), one statistic, resampled with p_max recomputed in every draw.
* Hit: FDE6 = ||cl[29] - g[29]|| <= tol. tol = 2.0 m is the primary value; 1.0 m and 4.0 m are printed as sensitivity and decide nothing.
* Execution inside the mode: e_a(c) = share of hits among the windows of c on which arm a is RIGHT; r90_a(c) = 90th percentile of FDE6 on those windows.
* Best single plan of a context: the largest share of the logged 6 s endpoints of c that one disc of radius tol can hold (exact maximum, angular sweep). It is fitted on the same windows, so it is an upper bound for every context-constant plan.
* Mode separation: for the two largest logged cells of c, D(c) = distance between the means of their logged 6 s endpoints; rho(c) = the larger of their two 90th-percentile distances of a logged endpoint from its cell mean; ratio S(c) = D / (2 tol + 2 rho). The theorem takes r as a bound on every outcome; rho is a 90 % radius, a declared weakening.
* Estimator: `taniteval.ci` episode-cluster bootstrap, B 2,000, seed 0, the same draws for every statistic of a call, as `hier/hierwhere.py` `boot`. Differences between two arms are paired on the same windows.
* Four families: LONGITUDINAL = accuracy of the longitudinal mode (4 classes) and its margin over the largest longitudinal class; LATERAL = the same for the lateral mode (3 classes); TACTICAL = the joint cell (the outcome rows below); STRATEGIC = PAUSED.

## 4. Outcome rules (both outcomes committed)
* Per pair, seed and context: PREMISE-HOLDS if m_HIER(c) > 0 and its interval excludes 0; PREMISE-FAILS if m_HIER(c) < 0 and its interval excludes 0; else UNDECIDED.
* Q1, card outcome on pair 1: DOM-PREMISE-MET if PREMISE-HOLDS in POOLED at both seeds; DOM-PREMISE-PARTIAL if not, and PREMISE-HOLDS at both seeds in at least one v0 context; DOM-PREMISE-NOT-MET otherwise. Pair 2 is printed with the same labels and decides nothing.
* Q2: per context SEPARATED if S(c) > 1, else NOT-SEPARATED; EXEC-OK if r90_HIER(c) <= tol at both seeds, else EXEC-WIDE. Printed, no card outcome.
* Q3, card outcome on pair 1, SELECTION-CRITICAL windows (HIER_DEFS), tol 2.0 m: HIT-HIER if the hit-rate difference hierarchy minus flat is above 0 with an interval that excludes 0 at both seeds; HIT-FLAT if below 0 at both seeds in the same sense; else HIT-UNDECIDED. Printed beside: all windows and bulk windows; every context; hierarchy minus the best single plan.
* One training run per arm per seed: a separated interval is necessary and not sufficient. The replicate is the secondary read on HIERREP.
* What follows, committed now. DOM-PREMISE-NOT-MET: the theorem does not apply to v8F-mini today; the verdict prints, per context, the accuracy the hierarchy lacks (p_max - acc_HIER) and which of the two terms carries it (longitudinal or lateral mode), and the next lever is the accuracy of the tactical decision on that term, not a gate and not the operative level. DOM-PREMISE-MET or PARTIAL: the theorem applies in those contexts; its bound (the decided plan hits at least the RIGHT windows that execute within tol) is compared with the measured hit rates of Q3, and the next step is the same read on the six seeds.

## 5. Controls (each must read its known value, else the read is not valid)
* d1: the logged class counts and the per-arm RIGHT shares equal those of `hier/RESULT_HIERWHERE.json` (1,305 and 1,094 windows; joint counts; exact).
* d2: the logged future as an arm (`cl` := `g`) reads acc 1.0 and hit rate 1.0 in every context.
* d3: a constant arm that always lands in (KEEP, STRAIGHT) reads acc equal to the share of that cell exactly; an arm whose cells are permuted across the windows of a context (seed 0) reads acc inside the interval of the sum of squared cell shares.
* d4: the best-single-plan routine returns exactly 60 on a synthetic set of two clusters of 60 and 40 points separated beyond 2 tol + 2 r, and all points on one cluster of radius below tol.
* d5 (deliberate regression): the logged future shifted by 3.0 m across the track at every knot reads hit rate 0.0 at tol 2.0 m in every context.

## 6. Limits declared
* Contexts are speed bins, a coarse stand-in for what a flat plan conditions on; the theorem compares with plans constant on a context, which the flat arm is not. Q3 therefore reads the flat arm directly as well.
* RIGHT is defined on the executed rollout, so it folds the decision and its execution into one indicator; the tactical head's own top-1 is not read here.
* Hit-or-miss at one endpoint is one acceptance region among many; ADE stays the registered primary metric of the programme's comparison (addendum 11).
