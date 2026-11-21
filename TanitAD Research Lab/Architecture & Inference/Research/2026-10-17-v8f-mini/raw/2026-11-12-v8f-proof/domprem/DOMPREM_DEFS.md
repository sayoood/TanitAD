# DOMPREM_DEFS (PREREG_V8F_PROOF_ADD14 sha256 70b48463a537...; MEASURED, T1-kin on logged frames, CPU, zero training, zero rollouts)
Code: F:/TanitAD-scratch/v8proof/domprem/domprem.py (line numbers below). Tests: test_domprem.py (9 tests, tests.log). Tier T1-kin, never driving performance. STRATEGIC: PAUSED (PI item 25).

## 1 Data paths (read-only)
* Pair 1 (on-policy data in both arms), domprem.py:34-36: HIER = E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_DAG_s{0,1}_m10_causal_heads; FLAT = E:/Projects/TanitAD-artifacts/v8f-flatdag-t1/dump_FLATDAG_s{0,1}_m10_causal_heads. 141 ep*.npz each, sorted-file order (`hierwhere.load_arm`).
* Pair 2 (no on-policy data), the folders of hier/pair2/hierwhere2.py pass 2: HIER = .../v8f-dagger-t1/dump_DAG0_s{0,1}_m10_causal_heads; FLAT = .../v8f-flatdag-t1/dump_FLATDAG0_s{0,1}_m10_causal_heads.
* Known values for d1 (domprem.py:38): hier/RESULT_HIERWHERE.json (pair 1) and hier/pair2/read/RESULT_HIERWHERE.json (pair 2).
* Tool tree: the pinned tree imported by hier/hierwhere.py (HIER_DEFS deviation R1). `hierwhere` is imported (domprem.py:27), not copied: load_arm, modes, joint, boot, KEEP_STRAIGHT, LON, LAT.
* Arrays: `g` logged future, `cl` executed T1-kin track, 30 knots, 6 s endpoint = knot index 29 (`[:, 29]`, domprem.py:183, 197); `v0` the dump's measured ego speed.

## 2 Definitions as implemented
* Mode function: `hierwhere.modes` / `joint` (12 cells = 4 longitudinal x 3 lateral), applied to g and to each arm's cl (`arm_feats` :180, `world` :193). RIGHT (joint) = cell of cl equals cell of g (`rights` :205); lon-RIGHT and lat-RIGHT the same per mode.
* Contexts (`ctx_index` :103, `ctx_masks` :108): POOLED and `np.digitize(v0, (2, 8, 15))`, so a value equal to an edge goes to the upper bin. TOO-SMALL if < 100 windows or < 20 episodes (:42, :228), then no statistics and no part in an outcome.
* p_max, acc, margin (`maxshare` :114, `margin` :122): m = acc - p_max with p_max recomputed on the same (resampled) selection inside every bootstrap draw; also for the longitudinal (K = 4) and lateral (K = 3) modes.
* Hit: FDE6 = ||cl[29] - g[29]|| <= tol (`arm_feats` :183; `hits` in `read_world`). tol 2.0 primary; 1.0 and 4.0 sensitivity (TOLS :44).
* e and r90 (`share_hit_among_right` :127; r90 in `read_world`): e = hits among windows where the arm is joint-RIGHT (bootstrap interval); r90 = `np.percentile(FDE6 on RIGHT windows of the context, 90)`, linear interpolation, point value.
* Best single plan (`best_disc` :52): exact maximum number of logged 6 s endpoints in one closed disc of radius tol; per point i the arcs of centre angles that keep each neighbour inside (half-width acos(d / 2 tol)), maximum overlap by a sorted sweep, starts before ends on ties, wrap-around handled, coincident points counted always, tolerance 1e-9 (distance gate and an angular widening of 1e-9 / (4 tol) rad, which moves a centre by at most 5e-10 m). O(n^2 log n).
* Separation (`separation` :132): D = distance between the means of the logged 6 s endpoints of the two largest logged cells (ties: lower cell index); rho = larger of the two 90th-percentile distances of an endpoint from its cell mean; S = D / (2 tol + 2 rho), tol 2.0.
* Estimator: `hierwhere.boot` (taniteval.ci `_draws`, B 2,000, seed 0): the same resampled episodes for every statistic of one call (one call per pair and seed). Paired differences are computed on the same windows inside each draw.
* Outcomes (`label_premise` :391, `outcomes` :399): code, from the numbers, per the card's section 4. What follows: `follows` :426.
* Verdict: `write_verdict` :463, from the json.
* Controls: `run_controls` :313. d1 compares the 12 logged joint counts, the 1,305 / 1,094 class counts, the logged longitudinal and lateral counts, and the per-arm RIGHT count (known count = round((1 - banked error rate) x 2,399), required to lie within 0.01 of the unrounded product) of each pair and seed with the banked json; d2 cl := g; d3a constant straight 5 m/s arm; d3b permutation inside each non-small context (numpy default_rng(0)) against the bootstrap interval of the sum of squared cell shares; d4 synthetic sets (`synth_d4` :301); d5 g with +3.0 m added to the ego-frame lateral coordinate at every knot.

## 3 Reading choices and deviations (each moves no threshold, bar, context edge, tolerance or outcome rule)
1. **Best-plan disc for class rows.** The card defines the best single plan per context. For the rows on all / SELECTION-CRITICAL / BULK windows, the disc is fitted on all windows of the context and its share is read on the windows of the class (a context-constant plan). The share on the class's own windows is not computed. Moves no threshold, bar or outcome rule.
2. **Interval of "hierarchy minus best single plan".** The exact sweep is run on the full sample for the point; the bootstrap holds the fitted disc fixed and resamples the windows (the sweep is O(n^2 log n) and cannot run in every one of 2,000 draws for every cell). The interval therefore does not carry the fitting variance of the plan; the plan is fitted on the same windows, so the comparison is conservative for the hierarchy. Printed only, decides nothing. Moves no threshold, bar or outcome rule.
3. **Ties in the largest-cell choice** (p_max needs no choice; D, rho, S need the two largest cells): lower cell index first (stable sort). Moves no threshold, bar or outcome rule.
4. **v0 below 0** would fall in the first bin (digitize); the dumps hold no such value (checked by the context counts, which sum to the pooled count). Moves no threshold, bar or outcome rule.
5. **Q3 card outcome context.** The card names SELECTION-CRITICAL windows, tol 2.0 m, without a context: read as the pooled windows (POOLED context). Per-context rows are printed beside. Moves no threshold, bar or outcome rule.
6. **d3b scope.** "(seed 0)" is read as the permutation random seed 0 (numpy default_rng(0)), applied to every pair and seed, and to every non-small context separately, each context permuted on its own windows (POOLED permuted across all windows). Because the logged future is identical across seeds the two readings coincide. Moves no threshold, bar or outcome rule.
7. **d5 "across the track".** Read as the ego-frame lateral axis (+3.0 m on the second coordinate at every knot), the only direction defined for stationary windows; FDE6 is 3.0 m on every window. Moves no threshold, bar or outcome rule.
8. **d1 pair-2 values.** The card names RESULT_HIERWHERE.json; the pair-2 known values come from the pair-2 result json of the same reader. d1 is run for both pairs. Moves no threshold, bar or outcome rule.
9. **Tolerances in the best-plan routine** (the card fixes 1e-9): realised as a distance gate of 1e-9 and an angular widening of 1e-9 / (4 tol) rad, so that the recount of points inside the returned disc agrees with the sweep (asserted at run time). Moves no threshold, bar or outcome rule.
10. **Carrier of the lack (what follows, NOT-MET).** The card asks which of the two terms carries it: read as the larger of (p_max of the longitudinal mode minus the hierarchy's longitudinal accuracy) and (p_max of the lateral mode minus the hierarchy's lateral accuracy), both printed, point values. Moves no threshold, bar or outcome rule.
11. **Statistics for contexts marked TOO-SMALL** are not computed (only the window and episode counts are printed). Moves no threshold, bar or outcome rule.
12. **Intervals** are printed for acc, m, margins, e, hit rates, differences, D, rho and S; r90 and the best-plan count are point values. EXEC-OK uses the point r90, as the card states it. Moves no threshold, bar or outcome rule.
13. **Scratch file.** `dbg.py` in this folder is a debugging script written during the tolerance fix (a failed rm of it was refused by the box; it was not deleted). It is not part of the read. Moves nothing.

## 4 Controls
Values in the controls block of VERDICT_DOMPREM.md and in RESULT_DOMPREM.json (`controls`).
