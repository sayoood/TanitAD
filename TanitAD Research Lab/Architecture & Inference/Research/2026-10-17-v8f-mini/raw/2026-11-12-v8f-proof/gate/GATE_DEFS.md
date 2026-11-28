# GATE_DEFS (GATEMIX, PREREG_V8F_PROOF_ADD8 sha256 807bb2115013; MEASURED, tier T1-kin on logged frames, CPU, zero training and zero rollouts)
Code: F:/TanitAD-scratch/v8proof/gate/gatemix.py (reuses hier/hierwhere.py by import: `load_arm`, `comps`, `modes`, `boot`, `mean_over`; pinned tool tree E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a). Tests: test_gatemix.py (6 tests, tests.log).

## 1 Where the numbers come from
* Trajectories and ADE6: dumps dump_{DAG,DAG0}_s{0,1}_m10_causal_heads (v8f-dagger-t1) and dump_{FLATDAG,FLATDAG0}_s{0,1}_m10_causal_heads (v8f-flatdag-t1), keys `cl`, `g`, `ws`, `eid`, `clip_index`, `v0`; ADE6 = `H.comps(cl, g)["ade6_m"]` (hierwhere.py:109), 2-6 s ADE = `ade26_m` (knots 10..29). Same 2,399 windows, 141 episodes, per seed, per arm.
* Class probabilities: NOT in the dump keys (g, ws, eid, clip_index, v0, cl, ol, fl, ha0). They are in E:/Projects/TanitAD-artifacts/v8f-dagger-t1/ticks/ticks_{DAG,DAG0}_s{0,1}_m10_causal_heads.npz, rows with `tick == 0` (2,399 of 7,197): `t_lat_logits` [n,5] and `t_lon_logits` [n,7]. Only logits are stored: softmax per head (gatemix.py:softmax). Lateral 5 classes, longitudinal 7 classes, 35 joint classes, joint column = lat*7 + lon, joint probability = p_lat(i) * p_lon(j).
* Alignment (asserted, per arm and seed): tick-0 rows equal the dump windows in `ci` == `clip_index` and `t0` == `ws` over all 2,399, `v0` max abs diff 0.0, `age_t` all 0, `gt_ok` all true, `gt` vs dump `g` max abs diff <= 7.6e-6 m (R3). Values in RESULT_GATEMIX.json cells.*.alignment.
* Pairs: P1 = (DAG, FLATDAG), P2 = (DAG0, FLATDAG0); the hierarchy arm of each pair supplies the tick-0 probabilities. Seeds 0 and 1.

## 2 Rule
* Common-case joint class: argmax joint count of (argmax lat, argmax lon) over the training-fold windows (ties: lowest index), recomputed per outer fold.
* Score s(w) = max over joint classes other than the common-case class of p_lat * p_lon, from the fold's common class (gatemix.py:score).
* tau per outer fold: candidates = [-inf] + 101 quantiles (np.quantile, linear, 0..1 step 0.01) of s on the training folds + [+inf]; tau minimises the composite mean ADE6 on the training folds (hierarchy drives when s >= tau); tie -> first candidate in ascending order (gatemix.py:pick_tau).
* Folds: 5 by episode; unique episode ids sorted, permuted by `np.random.default_rng(0)`, fold = position mod 5 (gatemix.py:fold_assign; R2). Every held-out window is decided by the tau and common class of its fold's training windows only (test: perturbing the held-out fold changes neither).
* Composite = per-window pick of the hierarchy's or the flat arm's value of every metric. ORACLE: hierarchy drives exactly the logged selection-critical windows (`H.modes(g)` joint != KEEP-STRAIGHT, HIER_DEFS section 2).
* Estimator: H.boot = taniteval.ci `_draws` (episodes with replacement, B 2,000, seed 0); the same draws for every statistic of a cell; tau and the common class are fixed in the draws (R9).
* n per read: 2,399 windows / 141 episodes (all); selection-critical 1,305 / 110, bulk 1,094 / 122 (logged, identical across arms and seeds); family rows list their finite count (R10).

## 3 Controls (order: g2, tick load, g1, g4, g3, reads)
g2 bar 1e-4 m on point and both bounds (as h1). g1 exact 0.0. g4 1e-9 m. g3 >= 18 of 20 per pair and seed.

## 4 Deviations (none moves a score, a bar or an outcome rule)
* R1 g2 for pair 2: the card says "the h1a route". Done as: (DAG, FLATDAG0) against the banked h1a values (-0.4973/-0.2450), and (DAG0, FLATDAG0) against the value hier/pair2/read/RESULT_HIERWHERE.json holds (self-reference of the reader, +0.0462/+0.1597; not an independent bank).
* R2 The card fixes 5 folds by episode and fold seed 0, not the assignment mechanics; fixed above before any composite was formed.
* R3 The tick file stores `gt` at float32 precision: the alignment bar for `gt` is max abs diff < 1e-4 m (measured 7.6e-6), not exact equality; the first run stopped on the exact test before any composite, and the bar was relaxed only for this storage precision.
* R4 The outcome rules do not name the metric: ADE6 (the score's metric) decides; 2-6 s ADE is printed beside it.
* R5 g3 shuffle: the score of each fold group (training groups and the held-out group alike) is permuted within its group, per shuffle seed k (rng 1000+k), then the same cross-fit runs; better arm = lower mean ADE6 over all windows; the threshold 18 of 20 is scaled if fewer shuffles are asked.
* R6 Stop rules: g2, g1 or g4 failing stops before any read (exit 2); g3 failing leaves the reads printed but marked, exit 2 (the card names only g2 as a hard stop; the reads are not valid until g3 reads its known value).
* R7 Group and family differences are means over the windows of the group in each resample (ratio form, hierwhere mean_over); NaN family rows average over the windows where both composite and arm have a value.
* R8 Common-case and score are computed on the full 35-class joint of the softmaxed heads; no thresholding of the heads is used.
