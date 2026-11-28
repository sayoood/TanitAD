# VERDICT_LATVIS_GW: LATVIS on the trained states of SPATIAL stage 1 (PREREG_V8F_PROOF_ADD12 section 4)

MEASURED (latvis_gw.py; RESULT_LATVIS_GW.json), tier T0: a world-model and perception diagnostic on logged frames, never driving performance.
Metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25), not a pass, not omitted.
Data: 1948 windows of 139 episodes, 9460 pool frames, 5 folds by episode, episode-cluster bootstrap B 2,000 seed 0 (the baseline's draws).

## Controls first

**Gate (v1 permuted-input skill <= 0.02, v3 train >= held-out, per arm): ALL OK.**
* P_s0: v1 permuted -0.0406 (ok); v3 train +0.7675 vs held-out +0.5751 (ok); v2 headroom 2.0s +0.00410 [+0.0035, +0.0047] (positive); c4 participation ratio 41.6 (ok)
* P_s1: v1 permuted -0.0487 (ok); v3 train +0.7952 vs held-out +0.5870 (ok); v2 headroom 2.0s +0.00415 [+0.0035, +0.0048] (positive); c4 participation ratio 43.2 (ok)
* frame identity (rebuilt frames of the 12 panel windows against the baseline sheet): 24 of 24 tiles equal.
* pairing with the baseline stage files: paired (max relative diff of the per-frame mean-image error 0.00e+00).

## V1 what is retained (skill = 1 - MSE / MSE of the training-fold mean image)

Token grid (baseline decoder on the 4x10 grid of the frozen tokens, 128 channels): skill +0.7770 [+0.7516, +0.8009] (baseline stage files latvis/stages/G_main_f*.npz).

| arm | held-out skill [95% CI] | ratio to the token grid (bar 0.90) | minus baseline of the same seed | per fold |
|---|---|---|---|---|
| P_s0 (grid, 640) | +0.5751 [+0.5342, +0.6156] | 0.740 | +0.2142 [+0.1790, +0.2472] vs Zs0 (HIER of seed 0, 128 wide mean state), paired (same frames, folds, episodes; episode-cluster bootstrap, the same draws) | +0.648, +0.595, +0.530, +0.601, +0.504 |
| P_s1 (grid, 640) | +0.5870 [+0.5401, +0.6291] | 0.755 | +0.2339 [+0.1952, +0.2726] vs Zs1 (HIER of seed 1, 128 wide mean state), paired (same frames, folds, episodes; episode-cluster bootstrap, the same draws) | +0.622, +0.616, +0.482, +0.610, +0.601 |

## V2 what is imagined (pixel MSE against the TRUE future frame; a = decoded predicted state, b = decoded start state, c = decoded true future state)

* P_s0 0.8s: b 0.01308, c 0.01090, headroom +0.00218 [+0.0019, +0.0025]; learned: a 0.01250, a - b -0.00058 [-0.0009, -0.0003] (IMAGINE-PIXELS); stack: a 0.01259, a - b -0.00049 [-0.0006, -0.0004] (IMAGINE-PIXELS)
* P_s0 2.0s: b 0.01492, c 0.01082, headroom +0.00410 [+0.0035, +0.0047]; learned: a 0.01535, a - b +0.00044 [-0.0003, +0.0012] (IMAGINE-FLAT); stack: a 0.01428, a - b -0.00064 [-0.0009, -0.0004] (IMAGINE-PIXELS)
* P_s1 0.8s: b 0.01279, c 0.01060, headroom +0.00219 [+0.0019, +0.0025]; learned: a 0.01206, a - b -0.00072 [-0.0009, -0.0005] (IMAGINE-PIXELS); stack: a 0.01222, a - b -0.00057 [-0.0007, -0.0005] (IMAGINE-PIXELS)
* P_s1 2.0s: b 0.01470, c 0.01054, headroom +0.00415 [+0.0035, +0.0048]; learned: a 0.01466, a - b -0.00004 [-0.0005, +0.0004] (IMAGINE-FLAT); stack: a 0.01390, a - b -0.00080 [-0.0010, -0.0006] (IMAGINE-PIXELS)

## Outcome names (card section 4, stage 1; each needs BOTH seeds)

* **Arm P: Q1 Z-ABOVE-MEAN** (per seed {'P_s0': 'Z-ABOVE-MEAN', 'P_s1': 'Z-ABOVE-MEAN'}); **Q2 pixels, learned rollout against the copy at 2.0 s: IMAGINE-FLAT** (per seed {'P_s0': 'IMAGINE-FLAT', 'P_s1': 'IMAGINE-FLAT'}); stacked predictor IMAGINE-PIXELS (per seed {'P_s0': 'IMAGINE-PIXELS', 'P_s1': 'IMAGINE-PIXELS'}).

Not part of this read: the r-a / r-b lead-gap reads of the stage-1 card (lead gap R2, imagined lead), Q2 IMAGINE-LEAD / IMAGINE-EGO, Q3 and the four families at T1-kin; they have their own tools.
Pictures: panel/panel_<arm>.png (12 windows of PANEL_WINDOWS.json; frame t, frame t + 2.0 s, decoded state(t), decoded predicted state(t + 2.0 s), decoded true state(t + 2.0 s)). Camera content: private, never landed.
Limits (card): a decoder shows what is shallowly recoverable from the state; 139 episodes; one checkpoint per arm and seed; L1 pixel error rewards the mean image. Reading choices RG1.. in LATVIS_GW_DEFS.md.
