# AMENDMENT_WMACT_1: is level 1's prediction aimed right but too large? (Master Mind, 2026-11-07, written after the TRAIN read and before any shrink statistic exists)

## Why
* `PREREG_V8F_WMACT.md` (frozen `d21b14cc`) read L1 as ACTION-REACTIVE with PREDICTS not met on eval-141. The committed TRAIN read gives **L1 DOES NOT FIT**: TRAIN L1 k1 SKILL is −0.0010 / +0.0063, neither separated, on 1,833 windows from 141 clips (`theory/wmact/RESULT_WMACT_TRAIN.md`). Yet L1 ACTION-USE is separated and larger on TRAIN (+0.0159 / +0.0155).
* The card commits no step for "does not fit". A predictor whose change uses the action but does not beat "nothing changes" may point the right way with too large a step. That is testable without training.

## The statistic (level 1, k = 1, 5, 10; M2b s0 / s1)
* Δẑ = ẑ_k − z_start (the predicted change); Δz = z_k − z_start (the true change).
* **Shrink factor, fitted on TRAIN only:** c_k = Σ⟨Δẑ, Δz⟩ / Σ‖Δẑ‖², over the 1,833 TRAIN windows, per seed and k.
* **SHRUNK SKILL on eval** = 1 − Σ‖z_start + c_k Δẑ − z_k‖² / Σ‖z_start − z_k‖², over the 1,976 eval windows. c_k is fixed from TRAIN.
* Beside, deciding nothing: the oracle ceiling on eval, cos² = (Σ⟨Δẑ, Δz⟩)² / (Σ‖Δẑ‖² Σ‖Δz‖²), and c_k itself.
* Estimator: the card's episode-cluster bootstrap over eval's 141 episodes (B 2,000, seed 0), with c_k held fixed.

## Controls
1. **c = 1 reproduces the read of record:** SHRUNK SKILL with c_k := 1 equals the eval SKILL of `RESULT_WMACT.json` to 1e-9.
2. **c = 0 is the copy baseline:** SHRUNK SKILL reads exactly 0.0.

## Outcomes, committed now
* **L1-OVERSHOOTS:** SHRUNK SKILL at k1 > 0 and separated at both seeds. The level-1 predictor aims right and steps too far. A TRAIN-fitted scalar is a zero-training fix, and any reliance arm uses the shrunk L1 rollout.
* **L1-MISAIMED:** SHRUNK SKILL at k1 not separated at either seed. The predicted change does not line up with the real change. Any reliance arm reads L2 and L3 only, and L1's predictor is a training item (capacity or loss balance against SIGReg and IDM), under its own card.
* **MIXED:** anything else, as written.

## Log (dated, append-only after the freeze)
* 2026-11-07 09:31:20 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 2249 bytes above this log, sha256 `503ba1b881ce48715329bb6017d23f68477031c31d99e09b027e7cb8582afcc3`, before any shrink statistic exists.
* 2026-11-07 09:42 (dev-box clock, from `date`), Master Mind. **Result: L1-OVERSHOOTS** (`theory/wmact/RESULT_WMACT_SHRINK.md`, `.json`, `run_shrink.py`). Both controls hold (c = 1 reproduces the record, difference 0.0; c = 0 reads 0.0). c fitted on TRAIN is 0.496 / 0.526 at k1. SHRUNK SKILL on eval at k1 is +0.0313 [+0.0267, +0.0357] / +0.0310 [+0.0255, +0.0359], separated at both seeds, and equal to the eval ceiling cos² (0.0313 / 0.0311). At k10 it is +0.1310 / +0.1249, where the unshrunk rollout was −0.1500 / −0.1724.
