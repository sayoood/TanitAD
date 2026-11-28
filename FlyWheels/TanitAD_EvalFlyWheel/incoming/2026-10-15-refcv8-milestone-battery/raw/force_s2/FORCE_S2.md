# FORCE_S2 -- is refcv8's near-term over-run in slow / slowing windows a TIMING, IDENTITY or DECODER defect?  (tag full)

Stamp for every number: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**.  Model: refcv8-r101-s0 EMA step 30,000 on CPU (fp32 trunk), per-window sampler seed identical across arms.  Estimator: paired episode-cluster bootstrap over the episodes present in the window set, models FIXED, B 2000, seed 0 (draws identical to taniteval.ci._draws): answers 'another draw of EPISODES' only; NOT another training run (single training seed), NOT another inference run (the sampler seed is fixed per window; arm N2 reports the draw-to-draw floor).

Coverage: rolled 440 of 440 planned (S2A 220/220, S2B 220/220).  Content check (recorded forced6 == plan): **PASS**.

## Verdict (committed table: TIMING-CONTROLLABLE | IDENTITY | DECODER | INCONCLUSIVE)

**INCONCLUSIVE**

* closure_F_slow on S2A u S2B: -0.907 [-1.893, -0.238] (n 440 / 80 ep)
* closure_F1 on the windows that have an F1 arm: +0.172 [-0.165, +0.448] (n 298 / 70 ep)
* like-for-like (F_slow on the F1 windows): -1.074 [-2.549, -0.263] (n 298 / 70 ep)
* n needed (ESTIMATED): `{"F1": {"assumption": "CI half-width ~ 1/sqrt(n_windows); the point estimate stays; ESTIMATED, not measured", "to_reach_closure_ge_0.50_with_lo_gt_0": "unreachable at this point estimate (point < 0.50)", "to_reach_hi_lt_0.25": 3712.8327395523847}, "F_slow": {"assumption": "CI half-width ~ 1/sqrt(n_windows); the point estimate stays; ESTIMATED, not measured", "to_reach_closure_ge_0.50_with_lo_gt_0": "unreachable at this point estimate (point < 0.50)", "to_reach_hi_lt_0.25": null}}`

## Per-set closures
* S2A: F1 +0.072 [-0.415, +0.410] (n 103 / 25 ep); F_slow -1.599 [-2.973, -0.680] (n 220 / 35 ep)
* S2B: F1 +0.256 [-0.357, +0.735] (n 195 / 64 ep); F_slow +0.606 [+0.148, +1.733] (n 220 / 71 ep)

## Mean d = ADE_arm - ADE_echo (S2 0-2 s, m; episode-cluster CI)
* S2A: N +0.1420 [+0.0991, +0.2005] n 220; F1 +0.1161 [+0.0663, +0.1783] n 103; F2 +0.3690 [+0.2483, +0.4962] n 220; N2 +0.1488 [+0.1001, +0.2072] n 100; Fown +0.1497 [+0.0991, +0.2004] n 100; GPU_N_banked +0.1358 [+0.0948, +0.1926] n 220
* S2B: N +0.0650 [+0.0173, +0.1131] n 220; F1 +0.0584 [+0.0129, +0.1043] n 195; F2 +0.0256 [-0.0190, +0.0732] n 220; N2 +0.0582 [+0.0025, +0.1169] n 100; Fown +0.0620 [+0.0023, +0.1234] n 100; GPU_N_banked +0.0535 [+0.0040, +0.1025] n 220
* S2A_u_S2B: N +0.1035 [+0.0681, +0.1411] n 440; F1 +0.0784 [+0.0431, +0.1154] n 298; F2 +0.1973 [+0.1273, +0.2686] n 440; N2 +0.1035 [+0.0648, +0.1448] n 200; Fown +0.1058 [+0.0615, +0.1521] n 200; GPU_N_banked +0.0946 [+0.0599, +0.1301] n 440

## Controls
* **KA** (CPU N vs banked GPU `os`): mean ADE diff all +0.0088 m (iid SE +0.0044); 0.05 m bar exceeded: {'all': False, 'S2A': False, 'S2B': False}.  Banked floor (GPU seed1 - seed0, same windows): mean +0.0058 rms +0.0982.  Median L2 waypoint diff 0.0277 m, max 1.171 m.  Head argmax agreement 0.989.
* **KB** (Fown moves the plan less than F_slow): displacement diff -0.2841 [-0.4125, -0.1598] -> PASS True.
* **R** (saved arrays reproduce the brief's E1 cell): {'delta': -0.0085, 'lo': -0.0229, 'hi': 0.0065, 'n_windows': 4159, 'n_episodes': 138} vs {'delta': -0.0085, 'lo': -0.0229, 'hi': 0.0065, 'n_windows': 4159, 'n_episodes': 138} -> PASS.

## E1 if S2A u S2B took the F_slow plans -- ARITHMETIC, NOT A RESULT
ARITHMETIC, NOT A RESULT.  SHIFT form: each full set's banked d is moved by the rolled subsample's mean (ADE_F_slow - ADE_N), CPU vs CPU, and the E1 cell is re-read on the SAME draws.  The CI is conditional on that shift being exact for the unrolled windows.
* shift per set (m): `{"S2A": 0.22700265503089248, "S2B": -0.03936796564445224}`; E1 native -0.0085 [-0.0229, +0.0065] -> if F_slow +0.0111 [-0.0084, +0.0310] (separated < 0: False).

## What this does NOT answer
One training seed (H-ESTIM-SEED-1); one fixed sampler seed per window (N2 reports the draw-to-draw floor); CPU fp32 trunk vs the banked GPU bf16 trunk (KA); the sets are a stratified SUBSAMPLE of S2A / S2B; closure_F1 and closure_F_slow are on different populations (like-for-like beside); open loop on logged frames -- not driving performance.  A timing verdict identifies WHERE the defect lives; the fix itself (time-aware tactical conditioning) is a retraining decision for the PI.
