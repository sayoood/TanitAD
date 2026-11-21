# PREREG_V8F_SPEEDCAL: is T's speed error a head bias? Recalibrate T's speed cells with a map fitted on TRAINING windows (written 2026-10-25, before any SPEEDCAL data)

**Status: pre-registered.**
* Zero training; dev-box CPU. Under `PI-V8F-2026-10-25b` / `-25c` / `-25d` (box 2 and the dev-box CPU only).
* It is the Rule-Zero next lever after `PREREG_V8F_SPEEDAVG`, whose outcome was MIXED. That outcome routes no lever by itself, so this arm carries its own pre-registration.

## Why (MEASURED)

* **The headroom is real and sits in SPEED** (`RESULT_V8F_KINDX.md`, T0, both seeds). With the true speed cells alone, O cuts own-code ADE6 by +0.716 / +0.668 m. No other kind clears 2F at either seed.
* **The error is systematic, not seed variance** (`RESULT_V8F_SPEEDAVG.md`, descriptives). The two independently trained T's make the same speed errors: their errors correlate at 0.71–0.85 over the five cells. Averaging them cuts RMSE by only 3–12 %, and PRIMARY moved −0.041 / +0.019 m.
* A systematic error is either a **head bias** (for example, shrinkage of the speed head toward the mean speed) or **information the inputs do not carry**. Both are shared across training runs. This arm separates them at zero cost: a bias of the head is a function of T's own output, so a map fitted on training data can undo it. A missing input cannot be undone that way.

## The arm (one lever: an affine map per speed cell, fitted on training windows)

For each seed s, with X_s = `M2b-mixo_s` (s0 sha c6a62876…, s1 sha 68a0f335…), and each of the five SPEED cells c ∈ {`lon_v_target_ms`, `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms`} (KIND's SPEED kind):

$$\hat c' = a_{s,c}\,\hat c + b_{s,c}, \qquad (a_{s,c}, b_{s,c}) = \arg\min_{a,b} \sum_{w \in \mathcal{W}_{fit}} m_{w,c}\,\big(a\,\hat c_{s,w} + b - c^{GT}_{w}\big)^2$$

* $\hat c_{s,w}$ is X_s's own T prediction of cell c (in m/s) on training window w.
* $c^{GT}_w$ is the cell's training target, the one T is trained against (the v9 train labels).
* $m_{w,c}$ masks windows without a GT for that cell.
* $\mathcal{W}_{fit}$ is the training windows of HBLEND's fit set (`raw/2026-10-25-hblend/`: 141 training clips), the same train-only set HBLEND and OACC's weight rule use. **No eval window enters the fit.**
* The map is closed-form least squares: 10 numbers per seed, fixed before eval.
* **At inference** on eval-141, T's five speed cells are replaced by $\hat c'$. Every other cell, T's decisions, S and O are X_s's own, through the same hook as SPEEDAVG. The map is a fixed function of T's own output, so inference stays vision-only, with no ego channel and no situation-classifier output in the goal path.

## The read (T1-class open loop, eval-141, paired episode-cluster bootstrap, B = 2,000, 141 episodes; F = 0.1207 m)

| read | definition | bar |
|---|---|---|
| **PRIMARY** | own-code ADE6, SPEEDCAL_s − X_s | **≤ −F and separated, at BOTH seeds** |
| **CEILING** (T0 diagnostic, never a result) | ADE6 of the same affine form fitted on the EVAL GT itself, minus X_s | beside. The best ANY affine map can do. It decides between the two "no" rows below |
| CAPTURE | (ADE6 X_s − SPEEDCAL_s) / gain_speed(X_s) | beside |
| **SECONDARY** | speed MAE, accel MAE, distance keeping, LEAD-VIOL: SPEEDCAL − X | not separated-worse at either seed |
| beside | the four families, with the paired tactical / strategic tool `v8f_paired_families.py`; per cell $a$, $b$, and RMSE before/after on train and eval; the same map applied to B (`M2b_s0/s1`) | descriptive |

| outcome | condition | next (committed) |
|---|---|---|
| **HEAD BIAS PAYS** | PRIMARY and SECONDARY at both seeds | The calibration joins `forward_infer`: 10 buffers per model, deployable at zero training. The HSPEC dominance row is re-read on the calibrated model against P10. Then a T-side debiasing arm, so one model needs no post-map, under its own PREREG |
| **PARTIAL BIAS** | PRIMARY < 0 and separated at both seeds, but not ≤ −F at both | Report CAPTURE. Next: a non-linear (isotonic, per-regime) calibration under its own PREREG |
| **MAP DOES NOT TRANSFER** | PRIMARY not separated at either seed, AND CEILING ≤ −F at both seeds | The error is affine-recoverable but the 141-clip fit does not carry to eval. Next: the same map fitted on all 4,512 training clips' windows |
| **INFORMATION-LIMITED** | PRIMARY not separated at either seed, AND CEILING > −F at either seed | T's speed output carries no affine-recoverable information about the right speed. The speed lever moves to perception (what T sees: temporal context, encoder adaptation). That needs compute beyond box 2, so a PI go |
| **MIXED** | anything else | reported per seed; seeds before windows |

## Controls (each must read its known value before any number is read)

1. **The identity map** ($a = 1$, $b = 0$) reproduces X_s's own eval per window: Δ = 0.0.
2. **The fitted map** equals numpy `lstsq` on the same masked training arrays, to 1e-9.
3. **GT speed cells** through the same hook reproduce KINDX's speed-only oracle per window, ≤ 1e-4 m. This ties this harness to the KINDX read.
4. **A deliberate regression:** a map fitted on training targets PERMUTED across windows (which collapses toward the mean speed) must NOT be separated-better than X_s at either seed.

## Compute and order

Dev-box CPU, 8 threads. It needs X_s's T forward on the fit set's training windows (both seeds), plus 4 evals per seed. ESTIMATED under 1 h. It runs beside FLAT on box 2 and touches nothing there except read-only checkpoint copies already on the dev box.

⛔ No cell set, fit set, map form, bar or routing changes after any SPEEDCAL number is seen.
