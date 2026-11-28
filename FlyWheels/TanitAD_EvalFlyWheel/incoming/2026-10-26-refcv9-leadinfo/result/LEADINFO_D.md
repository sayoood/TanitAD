# LEADINFO_D -- LEAD-INFO ABSENT

PREREG_LEADINFO_D.md (sha256 `8f02a6ddd3c6cf72bc748fdc5aebccee51f27fc54a02c9586259d5cbe3c1670e`), DIAGNOSTIC on logged eval139 labels (S6, compliant, seed-0 battery draw `step48852ema_pf`); no planner; GT lead features are an UPPER BOUND, never an inference input.

**Word:** LEAD-INFO ABSENT.  Clause 1 (dAUC(B-A) >= +0.03 with CI lo > 0): **False** (dAUC +0.0293, CI [-0.0027, +0.0618]).  Clause 2 (dAUC(C-A) CI contains 0): **False** (dAUC -0.0060, CI [-0.0110, -0.0014]).

Next per prereg s4: the traffic-light state (the v7.2 GT TL labels as the upper bound), same diagnostic.

## Estimator
5-fold episode-grouped cross-fit (fold seed 0) of L2 logistic regression (C=1, standardised on the fit fold); score = AUC of the out-of-fold P(decel). Delta-AUC CI: EPISODE-RESAMPLING bootstrap (B=2000, seed 0) -- draw the episodes with replacement, recompute AUC(B)-AUC(A) on the resampled windows with the out-of-fold predictions held FIXED, 2.5/97.5 percentiles. NOT taniteval.ci (a paired bootstrap of per-episode deltas is invalid for an AUC). Answers: would another draw of EPISODES say this? -- it is blind to training-run and to inference-run variance (the models here are convex, so the former is nil).

## n and coverage

* analysis set: **2348 windows, 135 episodes** (617 GT-decel, 1731 GT-cruise), decel base rate **0.2628**; folds [27, 27, 27, 27, 27] episodes / [406, 499, 400, 536, 507] windows.
* union population (compliant GT-decel + GT-cruise, S6): 2495 windows; **join record at ws: 2348 (0.9411)**, without: 147 (decel 617/639 covered, cruise 1731/1856); uncovered windows by episode index: {4: 12, 38: 10, 46: 25, 51: 10, 58: 19, 85: 15, 89: 3, 90: 9, 99: 22, 105: 7, 135: 15}.
* join: `E:\refcv6_eval_kit\data\joins\b1_train_plus_eval_agents.jsonl.xz` sha256 `a705a632cef07508d435cb152feac43b0d15137a09b08a527b5235da23d3d053` (26394 records for the 139 eval clips).

## AUC (out-of-fold P(decel); CI = episode-resampling bootstrap, B 2000, seed 0)

| model | AUC | 95% CI |
|---|---|---|
| lon head alone, P(BRAKE_TO+CREEP+HOLD) from the 6-logit softmax | 0.6794 | [0.6084, 0.7482] |
| A = softmax(6 LON6 logits) + v0 | 0.7143 | [0.6429, 0.7800] |
| B = A + GT lead block | 0.7437 | [0.6803, 0.8037] |
| C = A + lead block permuted across episodes (control) | 0.7083 | [0.6378, 0.7751] |
| extra (not prereg): A + lead block shuffled WITHIN episode | 0.7033 | [0.6293, 0.7723] |
| extra: v0 + lead block, no lon head | 0.6357 | [0.5689, 0.7048] |
| extra: v0 only | 0.5689 | [0.4937, 0.6452] |

lon head alone, 8-logit softmax variant: 0.6794;  lon head alone on the FULL union population (n=2495, incl. uncovered): 0.6745 (ref. 0.685).

| delta AUC | point | 95% CI |
|---|---|---|
| B - A | +0.0293 | [-0.0027, +0.0618] |
| C - A | -0.0060 | [-0.0110, -0.0014] |
| B - C | +0.0353 | [+0.0031, +0.0677] |
| (within-episode shuffle) - A [extra] | -0.0111 | [-0.0239, +0.0029] |
| A - lon head alone [extra] | +0.0349 | [-0.0175, +0.0906] |

**Known-value control (passes):** the 0.685 of the brief is reproduced by the renormalised head score decel/(decel+cruise) on the full union: **0.6851** (n=2495); on the analysis set 0.6884. The prereg's literal un-renormalised P(decel classes) is the 'lon head alone' row above (primary).

## Descriptive extras (cannot change the word; not part of the prereg)

* lead-present windows only (n=920, 247 decel, 84 episodes), same out-of-fold logistic predictions: AUC A 0.6339, B 0.6759, C 0.6227; dAUC(B-A) +0.0420 [-0.0278, +0.1138], dAUC(C-A) -0.0112 [-0.0233, +0.0001].
* nonlinear probe (one fixed untuned gradient-boosted-tree config, same folds): AUC A 0.7391, B 0.7559, C 0.7402; dAUC(B-A) +0.0168 [-0.0062, +0.0440], dAUC(C-A) +0.0011 [-0.0039, +0.0068]. (HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=0); NOT tuned)


## Lead presence

* lead present in **0.4003** of GT-decel windows vs **0.3888** of GT-cruise windows; difference +0.0115 CI [-0.1010, +0.1191] (episode-resampling bootstrap).
* leads 920, usable 1-s history 918, occluded share 0.0, range quantiles q05/25/50/75/95 [9.69, 20.96, 31.67, 47.86, 72.36] m, leads closer than 3 m: 2.
* convention: join agents are in the EGO frame of their own frame (+x forward, +y left); the lead is the nearest centre (smallest cx) with 0 < cx <= 80 m and |cy| <= 1.8 m at the NOW frame ws (= the window origin t0; refc_v3_train.py:4635 lookup(eid, t + w - 1)); all join classes count (automobile, truck, bus, person, rider, ...), occluded or not (GT upper bound)

## Alignment controls

* synthetic (production code path): **18 cases, all pass = True** -- an agent at (20, 0) m is found at range 20, lateral 0 (and: behind/outside/over-80 m not found, corridor edges inclusive, nearest wins, NO_LABEL = None, moving/stationary/turning-ego rates and world speed, history fallback, causality).
* real, pose index vs join frame_idx: stationary-track world-position std (p10 over tracks) by shift {-3: 0.296 m, -2: 0.229 m, -1: 0.163 m, 0: 0.109 m, 1: 0.161 m, 2: 0.222 m, 3: 0.289 m} -> best shift 0 (aligned at 0: True).
* real, ep_poses[ws,3] vs battery v0: max |diff| 0.00e+00 m/s over 3668 windows.
* real, lead range vs the ego's GT 1 s travel (GT cruising at > 10 m/s; 1022 windows, 320 with a lead): **21** leads with centre range < 1 s travel, 39 with rear edge < 1 s travel (min range - travel = -4.8 m); all covered windows with travel > 10 m: {'n': 1290, 'n_with_lead': 427, 'n_closer': 29}; contrast, GT-decel windows with travel > 10 m and a lead closer than the travel: 8.
* reading of those 21 hits (5 episodes): time gap range [0.745, 1.009] s, lead speed / v0 min-median-max [0.987, 1.071, 1.163], hits with a lead SLOWER than 0.9 v0 (a collision course): **0**. Zero collision-course cases: the hits are short-gap followers moving with the ego, so the alignment is not contradicted; reported as written, nothing was filtered.

## Deviations
* D1 path: the prereg names C:/Users/Admin/a40-rescue/b1_train_plus_eval_agents.jsonl.xz, which does not exist (the project drive D: is E: on this box). Used E:\refcv6_eval_kit\data\joins\b1_train_plus_eval_agents.jsonl.xz (sha256 a705a632cef07508...), BYTE-IDENTICAL to E:\Projects\TanitAD-artifacts\a40-rescue\b1_train_plus_eval_agents.jsonl.xz (sha256 equal: True).
* D2 coverage rule: windows with no join record at ws are counted/reported and EXCLUDED from the B/C comparison (NO_LABEL is not 'road clear'); A is fitted/scored on the SAME covered windows so A, B, C are paired. AUC(A) and AUC(lon head) on the full union population are reported beside it.
* D3 details the prereg leaves open (fixed in code BEFORE any AUC was computed, none tuned): range_x = the agent CENTRE cx (not the rear edge); the corridor is tested on the centre (|cy| <= 1.8, 0 < cx <= 80, both ends inclusive at 1.8 / 80); all join classes count (persons, riders, animals included) and occluded flags are ignored (GT upper bound; the selected leads are 0 % occluded anyway); range_rate uses the Euclidean range r = hypot(cx, cy) of the SAME track_id, baseline 1.0 s (10 frames) or, if that frame/track is absent, the oldest of 0.9..0.3 s; lead_speed = world displacement of the track over that baseline using the dump's ep_poses (pose index == join frame_idx, asserted by the shift sweep); no usable history (2 of 920 leads) -> rate 0 and lead_speed = v0; no lead -> lateral 0, lead_speed 0, range 80, rate 0, flag 0, time gap 80 / max(v0, 1). 'Past 1.0 s' is 10 provider frames (join t_s step 0.1007 s, used for dt).
* D4 head reference: the prereg's literal 'P(decel classes) from the softmax' (BRAKE_TO+CREEP+HOLD of the 6-logit softmax) reads 0.6745 on the full union; the brief's 0.685 is reproduced (0.6851) by the renormalised score decel/(decel+cruise). Both are reported; the literal one is the primary 'lon head alone' row.
* D5 (implementation note, not a departure): prereg s3's 'paired episode-cluster bootstrap with the out-of-fold predictions held fixed' is implemented as the brief says -- resample EPISODES (B 2000, seed 0), recompute AUC on the resampled windows for A, B, C on the SAME draw -- NOT taniteval.ci's per-episode-delta form, which is invalid for an AUC; the bootstrap universe is the 135 episodes with >= 1 covered union window (136 episodes have a union window, 139 exist).

Tier: DIAGNOSTIC (T0-style logged labels, no planner) -- never driving performance. The CI answers 'another draw of EPISODES'; the logistic models are convex so training-seed variance is nil, and no stochastic planner is involved.
