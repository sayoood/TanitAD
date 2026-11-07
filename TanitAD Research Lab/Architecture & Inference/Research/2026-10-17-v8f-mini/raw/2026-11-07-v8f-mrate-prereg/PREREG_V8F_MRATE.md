# PREREG_V8F_MRATE: each world-model level reads its own history of past frames at its own rate (Master Mind, 2026-11-07, written before any MRATE code or run exists)

## Why
* The PI, 2026-11-07, verbatim: *"Let also focus on improving the prediction quality and the imagination capability of each abstraction, i propose to feed past image frames at different history and frequencies"*.
* LATSEM (`PREREG_V8F_LATSEM.md`, `0b0c2f73`; result `raw/2026-11-07-v8f-latsem-result/`) measured why this is the lever:
  * Every level encodes speed. No level encodes acceleration, yaw rate or the 2 s change, because each level's state is a pointwise function of ONE frame (window w = 1).
  * The split H-JEPA reports is absent (TD-INCONSISTENT).
  * Steering is barely imagined, with the wrong sign at L1.
  * WMACT adds that L1 overshoots by about 2x (`AMENDMENT_WMACT_1`, c 0.50).
* Today the readers see 4 frames at 0.2 s. The world model sees 16 frames, but its levels differ only in their predictor's step, not in what their state sees.

## Arms (WM-only: the image adapter, the readers and the planner are frozen at M2b s{k}; the planner reads no WM output, so driving cannot change)
* **W0 (control):** M2b's WM architecture (w = 1 at every level, strides 2 and 2, dim 128, beta 0.08, gamma 100), re-initialised and trained WM-only.
* **W1 (the lever):** the same, except that each level's state reads K = 3 states of the level below, spaced at that level's own step:
  * L1: the adapter tokens of frames t, t-1, t-2 at 0.2 s (0.4 s of history);
  * L2: L1 states 0.4 s apart (1.2 s in total);
  * L3: L2 states 0.8 s apart (2.8 s in total).
* Both arms use the same windows (hmax 28 frames, every frame real), data (M2b's training windows), seed per run, optimizer and step count. The step count and argv are fixed in `IMPL_CHOICES_MRATE.md`, frozen with its sha256 before the first run.
* Seeds 0 and 1 for each arm: 4 runs on box 2. Each needs a launch-gate PASS token; every new flag gets its G-DVB entry.

## Reads (T0, a WM diagnostic; eval-141 windows with 28 real frames, the same windows for every arm; M2b's own WM beside)
* E1: WMACT's statistics per level (SKILL and ACTION-USE, L1 k 1/5/10, L2 and L3 k 1/3), with WMACT's controls.
* E2: LATSEM's R1 probes (v, a, w, dv2, dpsi2), A2 SPEED-SKILL and R3 counterfactuals, with LATSEM's controls.
* Arm differences W1 - W0 per training seed, paired over the same windows. Episode-cluster bootstrap B 2,000, seed 0. This answers "another draw of episodes". Each outcome needs BOTH training seeds, which is this card's answer to "another training run".

## Controls (each must read its known value, or the read is INVALID)
1. **Planner identity:** M2b s{k}'s plans on 64 fixed eval windows are bit-equal before and after WM-only training.
2. **K = 1 identity:** W1's code with K = 1 builds W0, with the same parameter shapes and the same forward on a fixed batch.
3. **G-LIVE for the WM-only mode:** every WM loss term is finite, every WM parameter group receives gradient, and the adapter's gradient is exactly 0.0. A deliberate-regression arm with the WM loss weight 0 must fail G-LIVE.
4. WMACT's and LATSEM's own controls.

## Outcomes, committed now
1. **MRATE-MOTION:** W1 ENCODES a or w (R1 R² > 0, separated) at L2 or L3 at both seeds, W0 does not at that level and target, and W1 - W0 L3 k3 SKILL is not separated-worse at either seed. Multi-rate history puts motion into the state. W1 becomes the WM of record, and the next arm is the reliance arm (O reads W1's L2 / L3 rollouts), under its own card.
2. **MRATE-SKILL:** without MOTION, W1 - W0 SKILL is separated-better at L2 k3 or L3 k3 at both seeds. The same next arm, on W1.
3. **MRATE-NULL:** neither. Pooled 4x10 tokens may not carry frame-to-frame motion even over 2.8 s (v43 §7.31, §7.38). The next lever moves to the token: per-level differences of the unpooled patch tokens, under its own card.
4. **INVALID:** a control fails.

## Beside (decide nothing)
* L1's TRAIN-fitted shrink factor c for W0 and W1: does a multi-frame L1 still overshoot?
* R3 LAT; retrieval pictures as in LATSEM; LATSEM-2's scene targets when that read exists.

## Log (dated, append-only after the freeze)
* 2026-11-07 12:57:27 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 4213 bytes above this log, sha256 `50667a7997f0082420cc4d1a6655e2e27beaa56a6bacf8ce7954c6de800c5470`, before any code, run or statistic of this card exists.
