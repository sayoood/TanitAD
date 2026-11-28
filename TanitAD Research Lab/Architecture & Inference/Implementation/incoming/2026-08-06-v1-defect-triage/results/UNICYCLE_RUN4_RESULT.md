# Unicycle readout run 4 — kinematics FIXED, reliance gate FAILED, and the probe that assigns the blame

**MEASURED 2026-08-06** · 3,000 steps on the frozen `flagship-v1arch-v2bal-30k` trunk (md5-proved
unchanged) · val = 128 fixed windows, OOD-val q90 · artifacts:
`results/unicycle_run4_train_log.jsonl.xz`, `results/base_readout_reliance.json` ·
checkpoint `pod4:/workspace/experiments/unicycle-readout-v1-SHORTCUT/`.

## The two-sided result

| val | baseline (displacement readout) | **trained unicycle head** | human |
|---|---|---|---|
| speed bias (m/s) | +0.4094 | **−0.0964** | 0 |
| accel RMS (m/s²) | 3.5298 | **0.8252** | 0.91 |
| jerk RMS (m/s³) | 44.0103 | **1.0490** | 1.71 |
| **net-yaw err (rad)** | 0.0328 | **0.0298 (−9 %)** | 0 |
| ADE (m) | 0.3612 | 0.4078 (+13 %) | 0 |

⭐ Every kinematic defect fixed and the **heading improved** — against re-timing's +62 % regression
on the same quantity. The cost is +13 % ADE.

⛔ **BUT THE RELIANCE GATE FAILED — 0.0891 against the pre-registered 0.5** — so per the
pre-registration this head is a driving-dynamics predictor and **must not be presented as a WM
decoder**. The canary measured the drift live: **0.588 @500 → 0.402 @1500 → 0.164 @2000 →
0.089 @3000**. As ADE improved, the head migrated onto the `v0`/feedback shortcut. Without the
canary this would have shipped looking like a success.

## ⭐ The discriminating probe — the trunk is exonerated, the head is convicted

Same instrument, same 128 windows, on the **original** displacement readout:

| arm | ADE (m) |
|---|---|
| real latents | **0.3612** |
| batch-**mean** latents | **1.6158** — 3× worse than CV |
| shuffled | 0.5651 |
| **frozen** (temporal evolution removed) | 0.5018 ≈ CV |
| CV floor | 0.5051 |

`wm_reliance = 8.72` — the original readout **cannot function at all** without per-window latent
content. Two conclusions, both load-bearing:

1. **The latents carry the information; run 4's head chose not to read it.** The shortcut was the
   head's own input surface (`v` + `a_prev/yr_prev` with lag-1 +0.983), not a deficient trunk.
   `shortcut_dropout = 0.1` was far too weak against a +0.98-autocorrelated crutch.
2. ⭐ **The `frozen` arm is a programme-level positive:** freezing the latents' temporal evolution
   collapses performance to the CV floor — i.e. what the readout consumes is the **predictor's
   rolled prediction**, not a static scene summary. The WM's dynamics are genuinely load-bearing
   for trajectory decoding.

## Run 5 — remove the surface, not the temptation

Launched: identical run except `speed_input=False, predict_delta=False` — the head reads **only**
`(z_prev, z_hat)`, so it *cannot* bypass the WM by construction (and `in_dim` returns to 4096, so
the warm-start copies the full trained trunk). The integrator still carries the true `v0` and the
barriers still enforce feasibility; the question run 5 answers is whether the kinematic wins
survive without the shortcut inputs. Pre-registered: PASS = reliance ≥ 0.5 **and** jerk ≤ 3× human
**and** net-yaw ≤ baseline; the interesting failure mode is kinematics regressing, which would mean
the smoothness was coming from the feedback loop rather than the loss.

⚠️ Estimator: single fixed 128-window batch, point estimates, no interval — canary-grade, not
registry-grade. A registry entry needs the full four-family harness + episode-cluster bootstrap.
