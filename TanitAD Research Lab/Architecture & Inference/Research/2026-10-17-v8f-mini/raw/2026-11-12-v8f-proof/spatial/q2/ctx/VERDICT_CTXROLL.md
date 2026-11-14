# VERDICT CTXROLL (PREREG_V8F_PROOF_ADD17, note ADD17_N1)

MEASURED, tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
No training: the banked checkpoints, start frame 4 of each window, target frame 14 (+2.0 s). R1 = rollout from one state with context 1 (the banked rule); R3h = seeded with the true states of frames 2, 3, 4; R3g = from one state, context grown to 3 by its own predictions.

## Controls (first)

* G_s0: x1 banked rollout reproduced, max abs diff 0.0020 (tol 0.004); x2 1948 windows matched, 0 unmatched, frame 10 against banked true state 0.0020; x3 seeded loop at context 1 against the stack 6.7e-06 (tol 1e-05); x4 1809 overlap pairs, state 0.0020, rollout 0.0020; lead read t1 to t6 and change form k1 to k3: PASS
* G_s1: x1 banked rollout reproduced, max abs diff 0.0035 (tol 0.004); x2 1948 windows matched, 0 unmatched, frame 10 against banked true state 0.0020; x3 seeded loop at context 1 against the stack 6.1e-06 (tol 1e-05); x4 1809 overlap pairs, state 0.0020, rollout 0.0035; lead read t1 to t6 and change form k1 to k3: PASS
* W_s0: x1 banked rollout reproduced, max abs diff 0.0019 (tol 0.004); x2 1948 windows matched, 0 unmatched, frame 10 against banked true state 0.0019; x3 seeded loop at context 1 against the stack 2.1e-06 (tol 1e-05); x4 1809 overlap pairs, state 0.0018, rollout 0.0019; lead read t1 to t6 and change form k1 to k3: PASS
* ALL_PASS: True

## Lead gap R2 at +2.0 s (r-b read, unchanged tool; episode-cluster bootstrap B 2,000)

| arm | rollout | R2 of the rolled-out state | present state (copy) | true future state | rollout minus copy | minus R1 (paired) |
|---|---|---|---|---|---|---|
| G_s0 | R1 | +0.447 [+0.259, +0.579] | +0.491 | +0.542 | -0.044 [-0.145, +0.041] | - |
| G_s0 | R3h | +0.457 [+0.277, +0.585] | +0.491 | +0.542 | -0.034 [-0.121, +0.042] | +0.010 [-0.005, +0.030] |
| G_s0 | R3g | +0.457 [+0.276, +0.585] | +0.491 | +0.542 | -0.034 [-0.122, +0.042] | +0.010 [-0.005, +0.028] |
| G_s1 | R1 | +0.437 [+0.246, +0.578] | +0.470 | +0.541 | -0.032 [-0.147, +0.067] | - |
| G_s1 | R3h | +0.466 [+0.290, +0.593] | +0.470 | +0.541 | -0.004 [-0.096, +0.082] | +0.028 [+0.008, +0.055] |
| G_s1 | R3g | +0.464 [+0.288, +0.592] | +0.470 | +0.541 | -0.005 [-0.099, +0.082] | +0.027 [+0.007, +0.053] |
| W_s0 | R1 | +0.239 [-0.061, +0.450] | +0.316 | +0.340 | -0.078 [-0.222, +0.049] | - |
| W_s0 | R3h | +0.274 [-0.018, +0.477] | +0.316 | +0.340 | -0.042 [-0.148, +0.060] | +0.035 [+0.003, +0.079] |
| W_s0 | R3g | +0.272 [-0.020, +0.473] | +0.316 | +0.340 | -0.045 [-0.152, +0.059] | +0.033 [+0.002, +0.074] |

R3h minus R3g (paired): G_s0 +0.000 [-0.002, +0.003]; G_s1 +0.001 [-0.001, +0.004]; W_s0 +0.002 [-0.002, +0.007].

## Outcome (card section 4)

* G_s0: CTX-FLAT
* G_s1: CTX-HELPS
* **Name (both seeds, a mixed pair reads the weaker name): CTX-FLAT**
* beside, W_s0: CTX-HELPS (one seed, names nothing)

## Beside the name (names nothing)

| arm | one-step error over the copy error, context 1 | context 3 | latent MSE at +2.0 s: copy | R1 | R3h | R3g | change skill R1 | change skill R3h | true future state |
|---|---|---|---|---|---|---|---|---|---|
| G_s0 | 1.233 | 1.216 | 0.440 | 0.406 | 0.390 | 0.391 | -0.046 [-0.247, +0.100] | -0.038 [-0.234, +0.103] | +0.136 [+0.009, +0.258] |
| G_s1 | 1.254 | 1.238 | 0.431 | 0.408 | 0.388 | 0.390 | -0.139 [-0.299, -0.006] | -0.084 [-0.220, +0.043] | +0.179 [+0.033, +0.327] |
| W_s0 | 1.026 | 0.974 | 0.127 | 0.149 | 0.131 | 0.133 | -0.219 [-0.498, -0.007] | -0.126 [-0.334, +0.048] | -0.033 [-0.195, +0.144] |

A one-step ratio above 1 means the predictor is worse than the copy of the state one step (0.2 s) ahead.
Limits: one checkpoint per arm and seed; the probe is fitted on true states and applied to the rolled-out ones; the start frame is 4, so the windows differ from the frame-0 read; 139 episodes.
