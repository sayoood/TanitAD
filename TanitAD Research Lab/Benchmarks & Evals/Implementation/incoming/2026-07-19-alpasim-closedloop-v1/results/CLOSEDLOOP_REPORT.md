# Closed-Loop Evaluation — TanitAD Flagship (imagination-in-the-loop)

**Date:** 2026-07-19 · **Pod:** tanitad-eval (A40) · **Corpus:** PhysicalAI-AV val (in-distribution) · **Renderer:** NONE.

NO-RENDERER closed loop: AlpaSim's NuRec renderer is unrunnable on this pod (unprivileged container, seccomp blocks user namespaces). With only a front camera + ego poses (no HD map / agent boxes) the honest closed-loop test is **imagination-in-the-loop drift + log-relative stability**, NOT a collision / drivable-area PDM. The flagship world model is used as its OWN neural simulator.

**Loop (per 0.1 s tick, 2 s horizon):** encode a real window -> latent z0; then (a) PLAN 2 s ego waypoints via the trained strategic->tactical hierarchy on the CURRENT (imagined) latent; (b) derive control a_k=(steer,accel) from the 0.5 s waypoint (pure-pursuit + speed P-controller); (c) IMAGINE z_{k+1}=operative_predictor(z_k, a_k) — the model consumes its OWN prediction (intent-free, the deployed regime); (d) DRIVE the ego pose by a kinematic bicycle under a_k. Re-plan every tick (receding horizon).

Two closed-loop paths are reported: **closed_bike** (kinematic integration of the executed controls — the HEADLINE) and **closed_grnd** (the operative predictor's own metric step-readout on the imagined roll — apples-to-apples with the teacher-forced open-loop). Baselines: **open_grnd** = the gate grounded rollout under TRUE actions; **open_bike** = bicycle under TRUE actions (the kinematic-fidelity FLOOR); **cv** = constant velocity.


## Headline — flagship-30k (v1 FINAL, step 29999, n=881 windows / 40 episodes)

- **Closed-loop ADE@2s = 1.685 m** (±0.098), **FDE@2s = 3.530 m** (closed_bike, headline).
- Teacher-forced open-loop grounded ADE@2s = 0.452 m; bicycle kinematic floor (true actions) = 0.513 m; CV baseline = 0.825 m.
- **Compounding delta (closed − open) @2s: grounded = +4.272 m, bicycle = +2.477 m** (point error).
- **Divergence rate (closed_bike drift > 5 m @2s) = 22.2%.**
- Speed-stratified closed_bike ADE@2s: low=1.645 m, high=1.821 m.

### Per-horizon (flagship-30k) — ADE (cumulative) / point-error, metres

| horizon | closed_bike ADE | closed_bike pt | closed_grnd ADE | open_grnd ADE | open_bike floor | CV |
|--|--|--|--|--|--|--|
| 0.5s | 0.227 | 0.227 | 0.449 | 0.076 | 0.090 | 0.129 |
| 1s | 0.584 | 0.941 | 1.056 | 0.158 | 0.192 | 0.297 |
| 1.5s | 1.070 | 2.043 | 1.826 | 0.288 | 0.333 | 0.530 |
| 2s | 1.685 | 3.530 | 2.674 | 0.452 | 0.513 | 0.825 |

## Compounding error (closed − open, per-horizon point error ± CI95)

| horizon | grounded Δ (m) | bicycle Δ (m) |
|--|--|--|
| 0.5s | +0.373 ±0.015 | +0.136 ±0.009 |
| 1s | +1.423 ±0.075 | +0.647 ±0.042 |
| 1.5s | +2.818 ±0.164 | +1.429 ±0.102 |
| 2s | +4.272 ±0.293 | +2.477 ±0.247 |

_Grounded Δ caveat: the step-readout is calibrated on TRUE-action rolled latents; under self-generated actions the imagined latents drift off that manifold, so this delta bundles genuine trajectory drift WITH step-readout off-manifold decode degradation. The bicycle delta below is controller-clean (no learned decode)._

## Stability / comfort (closed-loop executed controls)

- Divergence (>5 m @2s): 22.2% (±4.3 pts).
- Lateral-deviation growth vs GT: 0.5s=0.046 m, 1s=0.227 m, 1.5s=0.574 m, 2s=1.346 m (drift is longitudinal-dominated — the known high-speed weakness).
- Comfort: mean|accel|=2.069 m/s², mean|jerk|=3.631 m/s³, mean|lat_accel|=0.999 m/s²; 59% of steps exceed the 2.0 m/s² longitudinal comfort bound, 35% exceed jerk (noisy longitudinal command from the tactical head; lateral is smooth).

## Speed-stratified closed-loop drift (tertiles at [7.067, 13.325] m/s)

| stratum | mean speed | closed_bike ADE@2s | closed−open grnd Δ@2s | divergence | n |
|--|--|--|--|--|--|
| low | 3.867 | 1.645 | +2.339 | 21.8% | 294 |
| med | 10.390 | 1.729 | +2.485 | 18.8% | 293 |
| high | 23.645 | 1.821 | +2.071 | 29.9% | 294 |

## Arm comparison — does the speed channel help closed-loop stability?

| arm | ckpt | speed-in | closed_bike ADE@2s | FDE@2s | closed−open grnd Δ@2s | divergence | open_grnd (ref) |
|--|--|--|--|--|--|--|--|
| flagship-30k | 29999 | yes | 1.685 ±0.098 | 3.530 | +4.272 | 22.2% | 0.452 |
| flagship-speed | 19000 | yes | 1.656 ±0.119 | 3.462 | +3.369 | 21.7% | 0.628 |
| flagship-nospeed | 22000 | NO | 1.575 ±0.083 | 3.270 | +1.428 | 23.5% | 2.918 |

_REF-B is a direct planner (no operative latent predictor + metric step-readout), so it is architecturally incompatible with this imagination-in-the-loop harness and is not tabulated._

## Honest caveats

- NO collision / drivable-area / PDM — no HD map or agent boxes in our data; this is a drift/stability closed loop, not a safety one.
- SELF-REFERENTIAL: the world model is both state estimator and simulator; failures it cannot imagine are invisible (needs an external photoreal sim to cure).
- waypoint->control + bicycle are a HARNESS controller (not the model); open_bike is its kinematic fidelity floor so drift stays attributable.
- dataset actions correlate ~0.87 with bicycle-consistent controls (CAN signals) — residual shows up in open_bike.
- open-loop L2 is a weak claim (arXiv:2605.00066); the closed-loop delta is the point, not the absolute open-loop number.

**Re-run:** `python3 -m taniteval.closedloop --arm flagship-30k [--episodes 40]` (or `--all-flagships`); then `python3 closedloop_report.py`.