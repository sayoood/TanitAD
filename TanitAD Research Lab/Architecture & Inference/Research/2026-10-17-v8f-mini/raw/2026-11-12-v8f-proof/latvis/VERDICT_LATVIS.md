# VERDICT_LATVIS: PREREG_V8F_PROOF_ADD10 (pictures decoded from the world-model latent and the token grid)

MEASURED (latvis.py; RESULT_LATVIS.json), tier T0: a world-model and perception diagnostic on logged frames, never driving performance.
Metric families: this read measures none of LONGITUDINAL, LATERAL or TACTICAL directly. STRATEGIC: PAUSED (PI decision queue item 25), not a pass, not omitted.
Data: 1948 windows of 139 episodes (28 windows dropped for lack of frames), 9460 frames in the pool, 5 folds by episode, episode-cluster bootstrap B 2,000 seed 0.

## Controls first

**Gate (v1 permuted-input skill <= 0.02 and v3 train >= held-out): ALL OK.**
* Zs0: v1 permuted held-out skill -0.0401 (bar <= 0.02, ok); v3 train-frame skill +0.7282 vs held-out +0.3609 (ok).
* Zs1: v1 permuted held-out skill -0.0356 (bar <= 0.02, ok); v3 train-frame skill +0.6819 vs held-out +0.3530 (ok).
* G: v1 permuted held-out skill -0.0474 (bar <= 0.02, ok); v3 train-frame skill +0.8676 vs held-out +0.7770 (ok).
* v2 (ceiling real, 2.0 s): Zs0 headroom +0.00051 [+0.0003, +0.0007] (positive); Zs1 headroom +0.00045 [+0.0003, +0.0006] (positive). A missing headroom gives NO-HEADROOM below (not a control miss).

## V1 what is retained (skill = 1 - MSE / MSE of the training-fold mean image)

| source | held-out skill [95% CI] | per fold |
|---|---|---|
| Zs0 | +0.3609 [+0.2989, +0.4229] | +0.382, +0.362, +0.307, +0.435, +0.320 |
| Zs1 | +0.3530 [+0.2929, +0.4152] | +0.378, +0.336, +0.309, +0.437, +0.310 |
| G | +0.7770 [+0.7516, +0.8009] | +0.793, +0.771, +0.754, +0.810, +0.760 |

G minus Z (paired, episode-cluster bootstrap): Zs0 +0.4161 [+0.3730, +0.4561]; Zs1 +0.4240 [+0.3814, +0.4654]
**V1 outcome: Z-BELOW-G**

## V2 what is imagined (pixel MSE against the TRUE future frame; a = decoded predicted latent, b = decoded start latent, c = decoded true future latent)

* Zs0 0.8s: b 0.01660, c 0.01635, headroom b - c +0.00024 [+0.0002, +0.0003]; learned: a 0.01642, a - b -0.00018 [-0.0003, -0.0000] (IMAGINE-PIXELS); stack: a 0.01645, a - b -0.00015 [-0.0002, -0.0001] (IMAGINE-PIXELS)
* Zs0 2.0s: b 0.01691, c 0.01640, headroom b - c +0.00051 [+0.0003, +0.0007]; learned: a 0.01682, a - b -0.00009 [-0.0004, +0.0002] (IMAGINE-FLAT); stack: a 0.01661, a - b -0.00030 [-0.0004, -0.0002] (IMAGINE-PIXELS)
* Zs1 0.8s: b 0.01676, c 0.01656, headroom b - c +0.00020 [+0.0001, +0.0003]; learned: a 0.01671, a - b -0.00005 [-0.0002, +0.0001] (IMAGINE-FLAT); stack: a 0.01660, a - b -0.00016 [-0.0002, -0.0001] (IMAGINE-PIXELS)
* Zs1 2.0s: b 0.01703, c 0.01658, headroom b - c +0.00045 [+0.0003, +0.0006]; learned: a 0.01749, a - b +0.00046 [+0.0001, +0.0009] (IMAGINE-BELOW-COPY); stack: a 0.01671, a - b -0.00032 [-0.0004, -0.0002] (IMAGINE-PIXELS)

**V2 outcome at 2.0 s (both seeds): learned rollout IMAGINE-FLAT; stacked predictor IMAGINE-PIXELS**

Pictures: panel/panel_s0.png and panel/panel_s1.png (12 windows fixed in PANEL_WINDOWS.json before any decoder; columns frame t, frame t + 2.0 s, decoded G(t), decoded Z(t), decoded predicted Z(t + 2.0 s), decoded true Z(t + 2.0 s)). Camera content: private, never landed.

Next lever committed by the card: Z-BELOW-G with IMAGINE-FLAT or NO-HEADROOM means the lever is a spatial world-model state (token grid, not its mean), with this read as its acceptance test (pod, launch gate); IMAGINE-PIXELS means decoded rollouts become the visual debug channel and decoders are fitted for levels 2 and 3.
Limits (card): a decoder shows what is shallowly recoverable from 128 numbers; blur is expected from a mean-pooled state; 139 episodes; one checkpoint per seed; L1 pixel error rewards the mean image, so skill is read against that floor. Implementation choices R1.. are in LATVIS_DEFS.md.
