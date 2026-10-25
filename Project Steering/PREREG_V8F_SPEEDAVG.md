# PREREG_V8F_SPEEDAVG: is T's speed error variance? Average the two seeds' speed cells at inference (written 2026-10-25, before any SPEEDAVG data; CONDITIONAL)

**Status: pre-registered, CONDITIONAL.** Runs only if `PREREG_V8F_KINDX`'s seed-1 confirmation makes SPEED a carrier: gain_speed(X_s1) ≥ 2F = 0.2415 m and separated. If speed is not a carrier, this arm is not run. Zero training; dev-box CPU or box 2, under `PI-V8F-2026-10-25d`. It is the T-side arm `PREREG_V8F_KIND` routes a SPEED carrier to: "T's speed variance (two-seed code average / EMA)".

## Why (MEASURED, `RESULT_V8F_KINDX.md`, T0)

On X = `M2b-mixo_s0`, the true speed cells alone (T's code elsewhere) cut ADE6 by +0.716 [+0.623, +0.818] m, six times any other kind. The oracle says WHERE the headroom is, not WHY T's speed cells are wrong. They can be wrong by variance (noisy, seed-dependent) or by bias (systematically off). Averaging two independently trained T's removes part of the variance and none of the bias, so it separates the two at zero training cost.

## The arm

For each seed s ∈ {0, 1}: X_s's own forward, with T's five speed cells (`lon_v_target_ms`, `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms`; KIND's SPEED kind) replaced by the mean of X_0's and X_1's T predictions of those cells on the same window. Every other cell, T's decisions, S and O are X_s's own. Inputs at inference are both models' vision-only inputs: no ego channel, no situation-classifier output in the goal path.

## The read (T1-class open loop, eval-141, paired episode-cluster bootstrap, B = 2,000, 141 episodes; F = 0.1207 m)

| read | definition | bar |
|---|---|---|
| **PRIMARY** | own-code ADE6, SPEEDAVG_s − X_s | **≤ −F and separated, at BOTH seeds** |
| **CAPTURE** | (ADE6 X_s − SPEEDAVG_s) / gain_speed(X_s) | beside: the fraction of the speed oracle's headroom the average recovers |
| **SECONDARY** | speed MAE, accel MAE, distance keeping, LEAD-VIOL: SPEEDAVG − X | not separated-worse at either seed |
| beside | the four families; the speed cells' RMSE to the GT before and after averaging; the two seeds' speed-cell correlation | descriptive |

| outcome | condition | next (committed) |
|---|---|---|
| **VARIANCE PAYS** | PRIMARY and SECONDARY at both seeds | the two-seed speed average joins `forward_infer`; next, distil it into one T (a train-time arm under its own PREREG) so the deployed model needs one T |
| **PARTIAL** | PRIMARY < 0 and separated at both seeds but above −F | report CAPTURE; next: a temporal average of T's own speed cells over its last ticks (single model, zero training), under its own PREREG |
| **BIAS, NOT VARIANCE** | PRIMARY not separated at either seed | T's speed error is systematic; next: a T-side speed-head arm that targets bias (a re-weighted or class-balanced speed loss), under its own PREREG |
| **MIXED** | anything else | reported per seed; seeds before windows |

## Controls

1. Averaging a model with itself (X_s with X_s) reproduces X_s's own ADE6 exactly (Δ = 0.0 per window).
2. Replacing the speed cells with the GT reproduces KINDX's speed-only oracle per window (≤ 1e-4 m), which ties this harness to the KINDX read.
3. X_s's own forward reproduces its banked eval per window (≤ 1e-4 m).

⛔ No cell set, averaging rule, bar or routing changes after any SPEEDAVG number is seen.
