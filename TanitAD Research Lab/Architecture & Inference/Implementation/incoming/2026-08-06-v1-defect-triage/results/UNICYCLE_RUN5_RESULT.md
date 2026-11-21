# Run 5 (latents-only unicycle readout) — PASS on every pre-registered clause, ADE included

**MEASURED 2026-08-06** · 3,000 steps, 58 min, frozen `flagship-v1arch-v2bal-30k` trunk
(md5 `c11575…` identical before/after) · 2.11 M trainable params · val = the fixed 128-window
OOD-val q90 batch · `results/unicycle_run5_train_log.jsonl.xz` ·
checkpoint `pod4:/workspace/experiments/unicycle-readout-v2-latentsonly/`.

## Final table

| val | baseline (displacement readout) | **run 5** | human | pre-reg clause |
|---|---|---|---|---|
| ADE (m) | 0.3612 | **0.3571** | 0 | (not required) **−1.1 %** |
| speed bias (m/s) | +0.4094 | **−0.0962** | 0 | \|·\| < 0.15 ✅ |
| accel RMS (m/s²) | 3.5298 | **0.7273** | 0.91 | ≤ 2× human ✅ |
| jerk RMS (m/s³) | 44.0103 | **1.2891** | 1.71 | ≤ 3× human ✅ |
| **net-yaw err (rad)** | 0.0328 | **0.0130 (−60 %)** | 0 | ≤ baseline ✅ |
| **wm_reliance** | (8.72) | **0.6233** | — | ≥ 0.5 ✅ **GATE PASS** |

Run 4 (shortcut head), for contrast: ADE 0.4078, net-yaw 0.0298, reliance **0.089 FAIL**.

## ⭐ THE MAIN TAKEAWAY

**The head that cannot cheat beats both the head that could and the original.** Removing the
shortcut inputs (v, control feedback) moved reliance 0.089 → 0.623 **and** ADE 0.408 → 0.357.
The crutch was not even buying accuracy — it was pure gradient-descent path-of-least-resistance.
⇒ *Assurance by construction beats assurance by regularisation: remove the bypass path, do not
penalise it.* (`shortcut_dropout=0.1` demonstrably failed at exactly this in run 4.)

## The canary trajectory, including the dip that recovered on its own

| step | wm_reliance (ADE) | real net-yaw | mean net-yaw |
|---|---|---|---|
| 500 | 0.6879 | 0.0213 | 0.0262 |
| 1000 | 0.1799 | 0.0190 | 0.0256 |
| 1500 | **−0.3853** | 0.0133 | 0.0206 |
| 2000 | 0.2468 | 0.0158 | 0.0286 |
| 2500 | 0.5834 | 0.0142 | 0.0259 |
| 3000 | **0.6233** | **0.0130** | 0.0249 |

Two readings:

1. **The mid-run dip self-corrected** — with no shortcut surface, the only way to keep improving
   *was* the WM, and reliance climbed back as ADE fell. In run 4 the same dip was terminal.
2. **Per-arm net-yaw shows the heading reliance never wavered**: at every canary, real latents
   roughly **halve** the remaining heading error vs mean latents (0.0130 vs 0.0249 at the end).
   The ADE-reliance dip at 1500 was longitudinal noise-sensitivity, not heading shortcut — the
   decomposition the per-family logging exists for.

## What this does and does not establish

* ✅ A 2.11 M readout on the frozen trunk fixes **every defect Sayed observed** (speed bias, accel,
  jerk, heading), matches the baseline's ADE, and demonstrably relies on the WM's rolled
  prediction (with the run-4 probe: mean-latent baseline ADE 1.62 m — the trunk's latents are
  load-bearing).
* ⚠️ **Canary-grade evidence**: one fixed 128-window val batch, point estimates, no interval.
  Registry entry requires the full four-family harness on the 40-episode corpus with the
  episode-cluster bootstrap, plus the temporal-stability instrument (replan jump / toggle).
* ⚠️ Val batch draws windows on the rollout grid with seed 123 — same batch for every number
  above, so all comparisons are paired; but it is not the t0-aligned 39-clip Alpamayo set.
