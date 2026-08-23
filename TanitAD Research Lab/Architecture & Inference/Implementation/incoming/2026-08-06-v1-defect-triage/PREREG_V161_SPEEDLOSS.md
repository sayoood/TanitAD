# PRE-REGISTRATION — run 6: v1.6 + speed-profile loss (candidate `flagship-v161-speedloss`)

**Registered 2026-08-06, BEFORE launch.** Committed with both outcomes bound in advance
(operating standard rule 5). PI-approved lever: *"pre-register the arm and launch the retrain"*
(Sayed, 2026-08-06).

## The defect this attacks (MEASURED, `results/lag_response.json`)

v1.6 delays longitudinal response: in GT braking events it commands **16.2 %** of GT's
near-term deceleration (accel events: 15.1 %), trailing GT by **+0.28 s**; the decel-onset
horizon profile is a **ramp** (−0.15 m/s² at step 1 → GT's −1.1 only at 1.3 s). Curvature
magnitude is fine (ratio 0.993). The predicted latents **do** encode the imminent decel
(`z_hat@0.5 s → accel R² 0.60`, `results/vision_vs_ego.json`) — the information is present;
the head does not act on it early.

## Hypothesis (stated as HYPOTHESIS)

Position-L1 under-prices first-step acceleration (its position impact is quadratic in time),
making the ramp loss-optimal. A **speed-profile L1** term prices near-term speed error
uniformly across the horizon, so the loss-optimal plan brakes when GT brakes.

## The arm — ONE change from run 5

`train_unicycle_readout.py` gains `--w-speed` (default 0.0 = run 5 exactly):
`speed_l1 = mean |v_prof(wp) − v_prof(gt)|`, per-step speeds from waypoint differences
(eps-inside-sqrt; the stopped-path NaN-gradient trap is already pinned by test).
**Run 6 = run 5's exact args + `--w-speed 0.5`.** Everything else identical: frozen trunk
(md5 checked before/after), latents-only head (`speed_input=False, predict_delta=False`),
3,000 steps, batch 32, lr 3e-4 cosine, 600-episode local corpus, reliance canary every 500.
Out: `pod4:/workspace/experiments/unicycle-readout-v3-speedloss`.

## Pre-registered gates (instruments: `eval_v16.py` grid + `lag_response.py` + reliance canary)

**Primary (the claim under test):**
- P1: decel_response_ratio ≥ **0.40** (v1.6: 0.162)
- P2: lag_accel_s_mean ≤ **+0.15 s** (v1.6: +0.28)

**Non-regression (all must hold):**
- N1: ADE Δ vs v1.6 not CI-separated worse (paired episode-cluster bootstrap)
- N2: jerk RMS ≤ **3.42** (2× human 1.71; v1.6: 1.13 — headroom is expected and priced)
- N3: net-yaw error not CI-separated worse than v1.6
- N4: reliance canary final ≥ **0.5** (PASS) — the head must stay latent-reading
- N5: replan accel jump ≤ **0.30 m/s²** (v1.6: 0.102; v1arch: 1.13)

## Outcomes — bound now

- **A (P1∧P2∧N1–N5):** register `flagship-v161-speedloss` in `MODEL_REGISTRY.md` as the
  new candidate; full four-family eval before any promotion claim.
- **B (primary fails):** the position-loss hypothesis for the ramp is **REFUTED for this
  lever**; log the root-cause class; the next pre-registered lever is the event-weighted
  near-term accel-matching term (arm B) — NOT weight-tuning of this arm.
- **C (primary passes, any N fails):** trade-off documented per family, **no promotion**,
  PI decision with the paired CIs on the table.

*Evidence classes: all v1.6/v1arch numbers above MEASURED (artifact paths inline).*
