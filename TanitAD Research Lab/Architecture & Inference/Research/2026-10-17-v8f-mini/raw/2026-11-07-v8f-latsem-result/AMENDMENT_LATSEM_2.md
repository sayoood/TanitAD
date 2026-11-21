# AMENDMENT_LATSEM_2: score each level's predicted future speed in m/s through the probe (Master Mind, 2026-11-07, written after run 2 and before this statistic exists)

## Why
* Run 2 of `PREREG_V8F_LATSEM.md` (controls pass) reads R2 NOT at every level and horizon, and the ORACLE query (the true future latent) is not separated above 0 either: s0 L1 k10 +0.066 [-0.017, +0.128], L3 k3 -0.023 [-0.073, +0.030]. Retrieval from other episodes cannot resolve a 0.2 to 2.4 s speed change even with a perfect prediction, so R2 says nothing about the world model. The instrument has no headroom.
* R1 shows that speed is linearly decodable at every level (R2 0.42 to 0.66). Decoding through that probe is an instrument with headroom.

## The statistic
* For the R2 windows, level l, horizon k: v_hat(z) = the R1 speed probe of the other fold (cross-fitted, as R3 uses it). v* = the true speed at the future frame.
* SPEED-SKILL = 1 - sum (v_hat(query) - v*)^2 / sum (v_hat(start state) - v*)^2. Level 1 decides with the shrunk rollout; raw L1 and the oracle (the true future latent) are beside.
* **DECODES-FUTURE-SPEED at (l, k):** SPEED-SKILL > 0 separated at both seeds. Same estimator (episode-cluster bootstrap, B 2,000, seed 0).
* Control: the start-state query reads exactly 0.0.

## Beside, deciding nothing, post hoc
* R3's speed means suggest the imagined future answers braking much more than accelerating. CF-LON split into accel - hold and hold - brake, each by start speed (v0 < 3 m/s, 3 to 8, >= 8), at both seeds. It is labelled exploratory and promotes nothing without its own card.

## Log (dated, append-only after the freeze)
* 2026-11-07 10:06:21 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 1642 bytes above this log, sha256 `83446cc661ce44f638dceb152d2ed1703b1e77604515967b5541d49b4a3d3393`, before the SPEED-SKILL statistic exists.
