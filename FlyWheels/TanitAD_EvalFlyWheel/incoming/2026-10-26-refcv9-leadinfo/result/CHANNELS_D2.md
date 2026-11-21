# CHANNELS_D2 -- CURVE ABSENT, XWALK-OWN ABSENT, XWALK-GT ABSENT

PREREG_CHANNELS_D2.md (sha256 `35fb8d023eb4a280508fddc4cb85cd582b697bc8e6d777f263c46d1d3b810ce2`); DIAGNOSTIC on logged eval139 labels and the planner's own logged outputs (S6, compliant, seed-0 battery draw `step48852ema_pf`); GT channels are UPPER BOUNDS, never inference inputs.  Rule (s3): PRESENT iff dAUC(channel - A) >= +0.03 with CI lo > 0 AND the shuffled control's dAUC point estimate <= +0.01.

## Words

| channel | word | n windows / episodes | dAUC(channel - A) [CI] | shuffled dAUC(C - A) [CI] | clause 1 | clause 2 |
|---|---|---|---|---|---|---|
| CURVE | **ABSENT** | 2495 / 136 | +0.0052 [-0.0158, +0.0255] | -0.0011 [-0.0048, +0.0027] | False | True |
| XWALK-OWN | **ABSENT** | 2495 / 136 | -0.0107 [-0.0206, -0.0028] | -0.0007 [-0.0037, +0.0021] | False | True |
| XWALK-GT | **ABSENT** | 2442 / 134 | -0.0116 [-0.0246, +0.0024] | -0.0009 [-0.0041, +0.0021] | False | True |

Next per prereg s3: All ABSENT -> the window-level decel decision is not recoverable from these explicit channels: the refcv9 lever is TRAINING the lon decision (a clean-split lon head with richer supervision), named in PLAN_REFCV9 with this evidence.

## Estimator
IDENTICAL to LEADINFO_D (imported from leadinfo_d.py): 5-fold episode-grouped cross-fit (fold seed 0) of L2 logistic regression (C=1, standardised on the fit fold); score = AUC of the out-of-fold P(decel); CIs = episode-RESAMPLING bootstrap (B=2000, seed 0, OOF predictions fixed, 2.5/97.5). It answers 'another draw of EPISODES'; it is blind to training-run variance (convex models: nil) and to INFERENCE-run variance of the stochastic planner (CURVE reads a sampled plan; see the plan-draw variant).

## AUCs (out-of-fold P(decel); A re-fit on each channel's own windows)

| channel | n (decel / cruise), base rate | A | A + block | A + shuffled block | B - C [CI] |
|---|---|---|---|---|---|
| CURVE | 2495 (639 / 1856), 0.2561 | 0.7160 [0.6481, 0.7811] | 0.7211 [0.6511, 0.7892] | 0.7148 [0.6488, 0.7798] | +0.0063 [-0.0148, +0.0267] |
| XWALK-OWN | 2495 (639 / 1856), 0.2561 | 0.7160 [0.6481, 0.7811] | 0.7052 [0.6354, 0.7715] | 0.7153 [0.6484, 0.7794] | -0.0101 [-0.0206, -0.0010] |
| XWALK-GT | 2442 (634 / 1808), 0.2596 | 0.7262 [0.6577, 0.7873] | 0.7145 [0.6445, 0.7784] | 0.7253 [0.6562, 0.7864] | -0.0108 [-0.0238, +0.0034] |

## Coverage (union population 2495 windows = 639 GT-decel + 1856 GT-cruise, compliant S6)

* **CURVE**: 2495/2495 (1.0); decel 639, cruise 1856; rule: every window has a finite native plan8 (asserted); uncovered windows by episode index: {}
* **XWALK-OWN**: 2495/2495 (1.0); decel 639, cruise 1856; rule: a MAPSEL raster file (e, ws) exists in F:/refcv9_mapsel/rast_s0 (the `code` plane is always present); uncovered windows by episode index: {}
* **XWALK-GT**: 2442/2495 (0.9788); decel 634, cruise 1808; rule: the raster file carries a `gt` plane (clips with no SAM3 GT file have none; unseen cells inside the corridor count as not-crosswalk / not-edge); uncovered windows by episode index: {92: 27, 136: 26}
* GT corridor seen-share quantiles (q0, q5, q25, q50) over windows with a gt plane: [0.911, 1.0, 1.0, 1.0]

## Convention controls

* class ids FROM SOURCE: crosswalk = 3, drivable edge = 5 (CLASS_KEYS ['nocls', 'drivable', 'lane', 'crosswalk', 'arrow', 'edge', 'hatched', 'sidewalk']); stack/tanitad/models/map_head_hires.py:998-999 (CLASS_KEYS, code order 0..7); stack/tanitad/data/semantic_map_gt_fine.py:116-117 (N_FINE_CLASSES = 8, NOT_SEEN_CODE = 255); refcv9_mapsel/code/mapsel_lib.py:19-21 (CELL_M, H, W, Y_HALF_M; DRV_CODES, EDGE_CODE = 5); refcv9_mapsel/code/mapsel_roll.py:33-35 (DRV_CODES, EDGE_CODE, SIDEWALK_CODE).
* raster convention (mapsel_lib.py:3-4, :19, :70): ego frame at t0, x forward, y left; [H=1000 rows along x 0..100 m, W=600 cols along y -30 (right) .. +30 (left)], 0.1 m cells; cell (i, j) = x in [0.1 i, 0.1 (i+1)), y in [-30 + 0.1 j, ...); corridor = rows 40..399 x cols 282..317 (360 x 36 cells).
* synthetic crosswalk (patch built with the SOURCE's `mapsel_lib.to_cells`): **a patch at x = 20 m, y = 0 is found at distance 20.0 m** (share 0.01543 = 200 / 12960 cells); 17 cases all pass = True (x < 4 m excluded, x > 40 m and |y| > 1.8 m not found, +/-1.7 m inside, class mismatch not found, nearest wins, shift +5 m -> +5 m).
* synthetic curve (analytic literals): 7 cases all pass = True (circle R = 50 m -> kappa 0.02, v_curve 10.0, alat 4.5, required decel 5.0 at v0 15; quarter-circle heading change 1.37445; right-angle corner kappa 0.141421 at 20.0 m; short-segment merge).
* real, human path on drivable cells (gt plane; seen cells; drivable share, all / |y|>=2 m): declared 0.9962 / 0.9996 (n 13426 / 2666); y-mirrored 0.8405 / 0.3048; transposed 0.2532 / 0.3652; rows-reversed 0.7582 / 0.443 -> asserts {'declared_minus_rows_reversed_all_ge_0.10': True, 'declared_all_ge_0.80': True, 'declared_minus_mirror_lat2_ge_0.10': True, 'declared_minus_transposed_all_ge_0.10': True}
* real, human path on drivable cells (own plane; seen cells; drivable share, all / |y|>=2 m): declared 0.937 / 0.8702 (n 13780 / 2681); y-mirrored 0.8124 / 0.304; transposed 0.2279 / 0.3702; rows-reversed 0.6866 / 0.4144 -> asserts {'declared_minus_rows_reversed_all_ge_0.10': True, 'declared_all_ge_0.80': True, 'declared_minus_mirror_lat2_ge_0.10': True, 'declared_minus_transposed_all_ge_0.10': True}
* own vs GT map, crosswalk in the corridor (n 2442): own any 0.1077, GT any 0.2199, agree 0.864, own-any given GT-any 0.4358, GT-any given own-any 0.8897; drivable edge in corridor: own 0.0213, GT 0.3624.
* plan join: E['plan8'] = DEPLOY8_POS S6_plan8 of the banked 48,852 per-frame battery (seed 0); its slots [1,3,4,5,6,7] equal the battery 'os' arm exactly (asserted, max diff 0.0); the SAME roll that produced the lon logits of feature set A; join by (clip, ws) asserted through l2_common.load_eval_pf's row checks.  MAPSEL trace plan8 vs the battery plan8: max |diff| 35.308 m, median over windows of the per-window max 0.575 m, 0.9831 of windows differ -- a second DRAW of the stochastic planner.
* identity with LEADINFO_D: A's out-of-fold predictions on LEADINFO_D's 2348 windows reproduced, max |diff| 0.0e+00.

## Feature census (descriptive)

* CURVE (q05/q50/q95): kappa_max [0.001, 0.002, 0.048]; alat_demand [0.017, 0.299, 2.519]; v_curve [6.426, 29.865, 44.721]; req_decel [0.0, 0.0, 1.207]; heading_change_abs [0.001, 0.039, 0.902]; dist_to_kmax [2.127, 24.816, 163.444]; windows with kappa above the floor 0.6725, required decel > 0: 0.0766, > 1 m/s: 0.0553.
* XWALK-OWN_crosswalk_in_corridor_decel_vs_cruise: {'decel': 0.133, 'cruise': 0.0959, 'overall': 0.1054}
* XWALK-OWN_edge_in_corridor_decel_vs_cruise: {'decel': 0.0156, 'cruise': 0.0226, 'overall': 0.0208}
* XWALK-GT_crosswalk_in_corridor_decel_vs_cruise: {'decel': 0.3186, 'cruise': 0.1853, 'overall': 0.2199}
* XWALK-GT_edge_in_corridor_decel_vs_cruise: {'decel': 0.4054, 'cruise': 0.3473, 'overall': 0.3624}
* CURVE_req_decel_gt_0_decel_vs_cruise: {'decel': 0.1424, 'cruise': 0.0539}

## Descriptive extras (cannot change a word)

* CURVE_second_planner_draw_MAPSEL_trace_plan8 (n 2495): AUC A 0.716, A+block 0.7205, A+shuffled 0.7143; dAUC(block - A) +0.0046 [-0.0166, +0.0255]; shuffled -0.0016 [-0.0056, +0.0024] (would-be word ABSENT).
* CURVE_signed_heading_change (n 2495): AUC A 0.716, A+block 0.7147, A+shuffled 0.7206; dAUC(block - A) -0.0012 [-0.0205, +0.0187]; shuffled +0.0046 [-0.0015, +0.0107] (would-be word ABSENT).
* XWALK-OWN_crosswalk_columns_only (n 2495): AUC A 0.716, A+block 0.712, A+shuffled 0.7155; dAUC(block - A) -0.0040 [-0.0083, +0.0001]; shuffled -0.0004 [-0.0019, +0.0009] (would-be word ABSENT).
* XWALK-GT_crosswalk_columns_only (n 2442): AUC A 0.7262, A+block 0.7228, A+shuffled 0.726; dAUC(block - A) -0.0034 [-0.0074, +0.0006]; shuffled -0.0002 [-0.0021, +0.0016] (would-be word ABSENT).
* XWALK-GT_edge_columns_only (n 2442): AUC A 0.7262, A+block 0.718, A+shuffled 0.7256; dAUC(block - A) -0.0081 [-0.0208, +0.0056]; shuffled -0.0006 [-0.0031, +0.0019] (would-be word ABSENT).
* XWALK-OWN_on_the_XWALK-GT_windows (n 2442): AUC A 0.7262, A+block 0.7159, A+shuffled 0.7248; dAUC(block - A) -0.0103 [-0.0200, -0.0029]; shuffled -0.0014 [-0.0053, +0.0020] (would-be word ABSENT).
* CURVE, nonlinear probe (one fixed untuned HistGradientBoostingClassifier (leadinfo_d.cross_fit_hgb), same folds; cannot change the word): AUC A 0.7332, B 0.7312, C 0.73; dAUC(B - A) -0.0021 [-0.0199, +0.0155]; dAUC(C - A) -0.0032 [-0.0081, +0.0014].
* XWALK-OWN, nonlinear probe (one fixed untuned HistGradientBoostingClassifier (leadinfo_d.cross_fit_hgb), same folds; cannot change the word): AUC A 0.7332, B 0.7313, C 0.734; dAUC(B - A) -0.0020 [-0.0059, +0.0018]; dAUC(C - A) +0.0008 [-0.0011, +0.0025].
* XWALK-GT, nonlinear probe (one fixed untuned HistGradientBoostingClassifier (leadinfo_d.cross_fit_hgb), same folds; cannot change the word): AUC A 0.7297, B 0.7244, C 0.7299; dAUC(B - A) -0.0053 [-0.0185, +0.0066]; dAUC(C - A) +0.0002 [-0.0026, +0.0029].
* block alone (does the block carry label information beyond v0 by itself, whether or not the lon head already has it?): CURVE: AUC v0 0.6022, v0 + block 0.6545 [0.5886, 0.7252], delta +0.0523 [+0.0033, +0.1004]; XWALK-OWN: AUC v0 0.6022, v0 + block 0.5845 [0.5151, 0.657], delta -0.0178 [-0.0321, -0.0066]; XWALK-GT: AUC v0 0.6022, v0 + block 0.5714 [0.5004, 0.6456], delta -0.0308 [-0.0529, -0.0098].

## Deviations
* D1 plan source: the prereg says 'the planner's own native S6 plan P_os8'. Used E['plan8'] (DEPLOY8_POS S6_plan8 of the banked battery roll -- the roll that produced feature set A's lon logits; its S6 slots equal the battery os arm exactly). The MAPSEL roll's plan8 is a DIFFERENT DRAW of the stochastic planner (median max-diff 0.57 m), so it is reported as an inference-variance variant, not as the channel.
* D2 details the prereg leaves open (fixed in code before any AUC): polyline = origin + 8 rear-axle slots; short segments merged by keeping a point only if >= 0.5 m from the last kept point; kappa_max floored at 1e-3 (R = 1 km) so v_curve <= 44.7 m/s and the derived columns stay finite; the 'curve-limited speed sqrt(2.0 / kappa_max)' uses a_lat = 2.0 m/s^2 as written; heading change = |last merged segment heading - first| (absolute value -- the sign is left/right, not decel-relevant; the SIGNED form is a descriptive variant); dist_to_kmax = total path length when no vertex exceeds the floor.
* D3 corridor on CELL CENTRES: rows 40..399 (x in [4, 40) m, near edge 0.1*row), cols 282..317 (centre y in [-1.75, +1.75]) = 360 x 36 cells; share = class cells / ALL corridor cells (unseen counts as not-class); nearest distance = near edge of the nearest class cell, none -> 40.0 m; XWALK blocks = [xwalk share, xwalk nearest dist, edge share, edge nearest dist] (4 columns).
* D4 coverage rules: CURVE all windows; XWALK-OWN = windows with a MAPSEL raster file; XWALK-GT = windows whose raster carries a `gt` plane. Each channel re-fits A on its own windows (prereg s1).
* D5 estimator note (same as LEADINFO_D D5): episode-RESAMPLING bootstrap of the AUC with OOF predictions fixed; fold assignment is re-drawn (seed 0) on each channel's own episodes, so channel folds differ from LEADINFO_D's.

Tier: DIAGNOSTIC (logged labels and the planner's own logged outputs, no closed loop) -- never driving performance. The CI answers 'another draw of EPISODES'.
