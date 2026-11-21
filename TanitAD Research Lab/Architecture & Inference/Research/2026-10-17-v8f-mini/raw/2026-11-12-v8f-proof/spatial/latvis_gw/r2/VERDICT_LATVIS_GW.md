# VERDICT_LATVIS_GW: LATVIS on the trained states of SPATIAL stage 1 (PREREG_V8F_PROOF_ADD12 section 4)

MEASURED (latvis_gw.py; RESULT_LATVIS_GW.json), tier T0: a world-model and perception diagnostic on logged frames, never driving performance.
Metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25), not a pass, not omitted.
Data: 1948 windows of 139 episodes, 9460 pool frames, 5 folds by episode, episode-cluster bootstrap B 2,000 seed 0 (the baseline's draws).

## Controls first

**Gate (v1 permuted-input skill <= 0.02, v3 train >= held-out, per arm): ALL OK.**
* G_s0: v1 permuted -0.0471 (ok); v3 train +0.7830 vs held-out +0.6050 (ok); v2 headroom 2.0s +0.00430 [+0.0037, +0.0050] (positive); c4 participation ratio 40.5 (ok)
* G_s1: v1 permuted -0.0491 (ok); v3 train +0.7709 vs held-out +0.5553 (ok); v2 headroom 2.0s +0.00387 [+0.0033, +0.0045] (positive); c4 participation ratio 39.7 (ok)
* W_s0: v1 permuted -0.0599 (ok); v3 train +0.7187 vs held-out +0.3380 (ok); v2 headroom 2.0s +0.00054 [+0.0003, +0.0008] (positive); c4 participation ratio 13.0 (reported only, flag applies to the grid arm)
* W_s1: v1 permuted -0.0524 (ok); v3 train +0.6963 vs held-out +0.3453 (ok); v2 headroom 2.0s +0.00056 [+0.0004, +0.0008] (positive); c4 participation ratio 13.2 (reported only, flag applies to the grid arm)
* frame identity (rebuilt frames of the 12 panel windows against the baseline sheet): 24 of 24 tiles equal.
* pairing with the baseline stage files: paired (max relative diff of the per-frame mean-image error 0.00e+00).

## V1 what is retained (skill = 1 - MSE / MSE of the training-fold mean image)

Token grid (baseline decoder on the 4x10 grid of the frozen tokens, 128 channels): skill +0.7770 [+0.7516, +0.8009] (baseline stage files latvis/stages/G_main_f*.npz).

| arm | held-out skill [95% CI] | ratio to the token grid (bar 0.90) | minus baseline of the same seed | per fold |
|---|---|---|---|---|
| G_s0 (grid, 640) | +0.6050 [+0.5669, +0.6414] | 0.779 | +0.2441 [+0.2098, +0.2757] vs Zs0 (HIER of seed 0, 128 wide mean state), paired (same frames, folds, episodes; episode-cluster bootstrap, the same draws) | +0.616, +0.609, +0.553, +0.655, +0.593 |
| G_s1 (grid, 640) | +0.5553 [+0.5104, +0.5968] | 0.715 | +0.2023 [+0.1666, +0.2372] vs Zs1 (HIER of seed 1, 128 wide mean state), paired (same frames, folds, episodes; episode-cluster bootstrap, the same draws) | +0.573, +0.581, +0.497, +0.574, +0.548 |
| W_s0 (mean, 640) | +0.3380 [+0.2722, +0.4053] | 0.435 | -0.0229 [-0.0430, -0.0021] vs Zs0 (HIER of seed 0, 128 wide mean state), paired (same frames, folds, episodes; episode-cluster bootstrap, the same draws) | +0.418, +0.353, +0.237, +0.417, +0.269 |
| W_s1 (mean, 640) | +0.3453 [+0.2833, +0.4095] | 0.444 | -0.0077 [-0.0299, +0.0143] vs Zs1 (HIER of seed 1, 128 wide mean state), paired (same frames, folds, episodes; episode-cluster bootstrap, the same draws) | +0.400, +0.341, +0.262, +0.423, +0.305 |

## V2 what is imagined (pixel MSE against the TRUE future frame; a = decoded predicted state, b = decoded start state, c = decoded true future state)

* G_s0 0.8s: b 0.01242, c 0.01012, headroom +0.00230 [+0.0020, +0.0027]; learned: a 0.01181, a - b -0.00061 [-0.0009, -0.0003] (IMAGINE-PIXELS); stack: a 0.01184, a - b -0.00057 [-0.0007, -0.0004] (IMAGINE-PIXELS)
* G_s0 2.0s: b 0.01442, c 0.01012, headroom +0.00430 [+0.0037, +0.0050]; learned: a 0.01435, a - b -0.00006 [-0.0005, +0.0004] (IMAGINE-FLAT); stack: a 0.01364, a - b -0.00078 [-0.0010, -0.0006] (IMAGINE-PIXELS)
* G_s1 0.8s: b 0.01345, c 0.01138, headroom +0.00207 [+0.0018, +0.0024]; learned: a 0.01294, a - b -0.00051 [-0.0008, -0.0002] (IMAGINE-PIXELS); stack: a 0.01292, a - b -0.00053 [-0.0007, -0.0004] (IMAGINE-PIXELS)
* G_s1 2.0s: b 0.01526, c 0.01139, headroom +0.00387 [+0.0033, +0.0045]; learned: a 0.01591, a - b +0.00065 [-0.0001, +0.0014] (IMAGINE-FLAT); stack: a 0.01450, a - b -0.00075 [-0.0010, -0.0005] (IMAGINE-PIXELS)
* W_s0 0.8s: b 0.01724, c 0.01703, headroom +0.00022 [+0.0001, +0.0003]; learned: a 0.01708, a - b -0.00016 [-0.0003, +0.0000] (IMAGINE-FLAT); stack: a 0.01704, a - b -0.00021 [-0.0003, -0.0001] (IMAGINE-PIXELS)
* W_s0 2.0s: b 0.01754, c 0.01701, headroom +0.00054 [+0.0003, +0.0008]; learned: a 0.01742, a - b -0.00013 [-0.0005, +0.0002] (IMAGINE-FLAT); stack: a 0.01719, a - b -0.00035 [-0.0005, -0.0002] (IMAGINE-PIXELS)
* W_s1 0.8s: b 0.01696, c 0.01673, headroom +0.00023 [+0.0001, +0.0003]; learned: a 0.01683, a - b -0.00013 [-0.0003, +0.0000] (IMAGINE-FLAT); stack: a 0.01678, a - b -0.00018 [-0.0002, -0.0001] (IMAGINE-PIXELS)
* W_s1 2.0s: b 0.01732, c 0.01676, headroom +0.00056 [+0.0004, +0.0008]; learned: a 0.01729, a - b -0.00003 [-0.0004, +0.0003] (IMAGINE-FLAT); stack: a 0.01700, a - b -0.00032 [-0.0004, -0.0002] (IMAGINE-PIXELS)

## Outcome names (card section 4, stage 1; each needs BOTH seeds)

* **Arm G: Q1 Z-ABOVE-MEAN** (per seed {'G_s0': 'Z-ABOVE-MEAN', 'G_s1': 'Z-ABOVE-MEAN'}); **Q2 pixels, learned rollout against the copy at 2.0 s: IMAGINE-FLAT** (per seed {'G_s0': 'IMAGINE-FLAT', 'G_s1': 'IMAGINE-FLAT'}); stacked predictor IMAGINE-PIXELS (per seed {'G_s0': 'IMAGINE-PIXELS', 'G_s1': 'IMAGINE-PIXELS'}).
* **Arm W: Q1 Z-FLAT** (per seed {'W_s0': 'Z-FLAT', 'W_s1': 'Z-FLAT'}); **Q2 pixels, learned rollout against the copy at 2.0 s: IMAGINE-FLAT** (per seed {'W_s0': 'IMAGINE-FLAT', 'W_s1': 'IMAGINE-FLAT'}); stacked predictor IMAGINE-PIXELS (per seed {'W_s0': 'IMAGINE-PIXELS', 'W_s1': 'IMAGINE-PIXELS'}).
* **Attribution G against W (location against width): Q1 LOCATION; Q2 pixels NO-EFFECT-TO-ATTRIBUTE.**
  * seed0: G minus W skill +0.2670 [+0.2292, +0.3035]; G minus W in (a - b) at 2.0 s +0.00006 [-0.0005, +0.0006]
  * seed1: G minus W skill +0.2100 [+0.1730, +0.2455]; G minus W in (a - b) at 2.0 s +0.00068 [-0.0002, +0.0016]

Not part of this read: the r-a / r-b lead-gap reads of the stage-1 card (lead gap R2, imagined lead), Q2 IMAGINE-LEAD / IMAGINE-EGO, Q3 and the four families at T1-kin; they have their own tools.
Pictures: panel/panel_<arm>.png (12 windows of PANEL_WINDOWS.json; frame t, frame t + 2.0 s, decoded state(t), decoded predicted state(t + 2.0 s), decoded true state(t + 2.0 s)). Camera content: private, never landed.
Limits (card): a decoder shows what is shallowly recoverable from the state; 139 episodes; one checkpoint per arm and seed; L1 pixel error rewards the mean image. Reading choices RG1.. in LATVIS_GW_DEFS.md.
