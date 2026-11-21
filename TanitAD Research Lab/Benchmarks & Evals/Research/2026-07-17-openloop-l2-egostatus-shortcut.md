# Community-standard open-loop L2 + the ego-status shortcut ceiling — 2026-07-17

**Agent:** Benchmarks & Eval (Thursday). **Quality:** full. **Compute:** dev-box CPU
(RTX 4060 idle, not needed), **$0, no pod touched.** **Goal advanced:** G1 (honest,
standardized denominator for the single-camera driving-capability gap — the top program risk).

## TL;DR
The Phase-0 driving diagnostic reports our D1 in a **camera-frame** space against a **best-of-3
kinematic floor** — neither is comparable to the number the field ranks driving on
(**nuScenes-style open-loop L2**, metric-BEV, ego frame). I built that protocol and measured, on our
real val corpora, the **no-vision "ego-status shortcut" ceiling** every open-loop L2 must be read
against (AD-MLP, arXiv 2312.03031). Two decision-grade results:

1. **A no-vision learned ego-status regressor scores avg L2 = 0.658 m (0.144 / 0.552 / 1.256 m @1/2/3s)
   on comma highway, held out by clip** — statistically tied with the CTRV kinematic baseline (0.656 m)
   and in the ballpark of AD-MLP's ~0.29 m on nuScenes. **This is the denominator a TanitAD open-loop L2
   must beat to mean anything.** No pixels, ~20 parameters.
2. **comma highway is 73.9 % straight-cruising** — *coincidentally identical* to the 73.9 % figure the
   ego-status critique reports for nuScenes. Our open-loop val therefore inherits the **exact
   shortcut pathology** the community levels at nuScenes: the aggregate L2 is dominated by trivial
   straight driving, and a model that reproduces ego-status extrapolation scores well **without driving**.

Implication for the top risk: the "is the model driving?" question **cannot** be settled by an open-loop
L2/ADE number on comma (or nuScenes) at all — the shortcut ceiling and the 74 %-straight population make
open-loop L2 a weak capability signal, exactly as arXiv 2605.00066 warned. The verdict must come from
(a) `skill_score = model_L2 ÷ shortcut_L2` reported **per curvature stratum**, and (b) closed-loop.

## What I built (intake `Implementation/incoming/2026-07-17-openloop-l2-egostatus-shortcut/`)
- `openloop_l2.py` — nuScenes-style L2 in **metric-BEV ego frame (metres)** under **both** reported
  averaging conventions (they disagree ~2×, papers rarely disclose which): `pointwise` (UniAD: L2 at
  exactly t) and `cumulative` (ST-P3/VAD: mean of per-step L2 up to t), + `avg` over {1,2,3}s. Plus a
  `collision_rate` proxy (axis-aligned ego footprint vs agent boxes) and the `RidgeTrajectoryHead`
  no-vision shortcut. Reuses the 07-15 floor predictors (vendored for a standalone `pytest`).
- `test_openloop_l2.py` — **8 analytic-ground-truth tests (G-B2)**, all green: convention arithmetic
  (constant vs growing per-step error), kinematic baselines perfect on exact-CV motion (`stop` = the
  displacement null), ridge recovers an exact linear map, the shortcut nails synthetic CV motion,
  collision hit/miss, skill_score.
- `run_openloop_l2.py` + `results_openloop_l2.json` — the measured run below. Clip-level 2/3–1/3 split
  (I3): shortcut **fit on train clips, every predictor scored on held-out val clips**.

## Measured results (median L2, pointwise=UniAD convention, metric-BEV metres)

### comma highway — 90 seq, 7 920 val anchors, v≈24.8 m/s
| predictor | L2@1s | L2@2s | L2@3s | avg | note |
|---|---:|---:|---:|---:|---|
| stop (null) | 24.88 | 49.85 | 74.89 | 49.87 | predict staying put = distance travelled; sanity ✓ (v·t) |
| go_straight | 0.219 | 0.841 | 1.850 | 0.973 | const heading+speed |
| cv | 0.225 | 0.855 | 1.872 | 0.981 | const world-velocity (weakest null on curves) |
| ctrv | 0.130 | 0.545 | 1.277 | 0.656 | const turn-rate+velocity |
| **best-of-3 floor** | **0.122** | **0.479** | **1.102** | **0.571** | per-anchor min — the kinematic denominator |
| **ego_status_mlp (no vision, learned, held-out)** | **0.144** | **0.552** | **1.256** | **0.658** | the AD-MLP shortcut ceiling on our data |

Curvature population: **straight 73.9 %**, gentle 12.0 %, sharp 0.3 %, standstill 13.8 %.

### cosmos_urban — 13 seq, 318 val anchors, v≈10.6 m/s (all straight)
| predictor | L2@1s | L2@2s | L2@3s | avg |
|---|---:|---:|---:|---:|
| stop | 10.69 | 21.42 | 32.08 | 21.37 |
| best-of-3 floor | 0.259 | 1.029 | 2.626 | 1.335 |
| **ego_status_mlp (held-out)** | 0.287 | **0.975** | **2.141** | **1.191** |

On urban (lower speed, more decel structure) the **learned** shortcut *beats* the hand kinematic floor
(1.19 vs 1.34) — it picks up systematic deceleration a fixed kinematic model misses. On highway the two
are tied (near-constant velocity leaves nothing to learn). Either way, **the shortcut is the ceiling**.

## Analysis — what this changes for TanitAD
- **Three communities, one denominator.** NAVSIM v2's constant-velocity *triviality filter*
  (arXiv 2506.04218), AD-MLP's *ego-status shortcut* (2312.03031), and our *best-of-3 kinematic floor*
  (07-15) are the same object. This run unifies them as `shortcut_L2 ≈ 0.66 m` (comma-hwy avg) and makes
  it the single normalizer. **`skill_score` is now defined in leaderboard-comparable units.**
- **Open-loop L2 on comma is a weak capability test — quantified.** 73.9 % straight ⇒ a no-vision
  ~20-param model is within noise of the best kinematic null. Any TanitAD highway L2 that is not a large
  factor *below* 0.66 m is not evidence of driving; it is evidence of ego-status extrapolation. This
  is the community-unit restatement of the framework's "10–15× worse than CV everywhere" verdict.
- **Unit reconciliation (resolves the framework caveat for the denominators).** The floor here
  (0.122 m@1s, metric-BEV) is consistent with the 07-15 floor (0.056 m@1s CTRV) — both real metres; the
  gap is anchor-set + median-of-averages vs per-stratum. The model's 6.44 m is **camera-frame** and
  still not directly divisible by these (G-B1) until a checkpoint is decoded in this metric-BEV ego
  frame — at which point `skill_score = model_L2 / 0.66`. For scale: a model at even 2 m highway L2
  would be **skill_score ≈ 3×** (3× worse than seeing no pixels).
- **Honest caveat (P8, same as the floor).** The shortcut uses **privileged GT ego-state history** a
  vision-only model never sees; it is the *denominator/ceiling*, not a model competitor. The fair
  vision target remains the oracle in-distribution ceiling (1.65 m@1s, framework §B).

## Blocker honestly recorded
G1's remaining half — a **model-relative** skill_score — still needs a post-reset checkpoint decoded in
this metric-BEV ego frame. The only locally-available ckpt is **pre-reset step-6500** (`ckpt_full.pt`,
9-ch encoder, no speed-input fix) and its D1 targets are camera-frame → **not comparable** (I did NOT
run it, to avoid an apples-to-oranges skill_score, G-B1). Post-reset ckpts are on pods (training,
off-limits) / gated HF. **Queued follow-up:** when a post-reset ckpt is pullable, add a metric-BEV
trajectory decode to `driving_diagnostic.py` and emit `skill_score` against the 0.66 m shortcut.

## Literature sweep (benchmark/dataset seam, D-028)
3 searches, no plan-changing new release. Confirmed the ego-status critique lineage (Zhai et al.
2312.03031 "Is Ego Status All You Need", CVPR'24 — 73.9 % straight, ego-MLP ~0.29 m nuScenes); NAVSIM v2
navhard leaderboard unchanged since the 07-09 refresh (DrivoR 56.3 / DriveFuture 55.5 / PDM-Closed 51.3
EPDMS). Noted for other seams: DeepSight (2605.10564, long-horizon latent-state E2E WM → Arch),
GPUDrive (2408.01584, 1 M-FPS multi-agent sim → Tools/closed-loop). No untriaged benchmark for us.

## Sources
- Zhai et al., "Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?", CVPR 2024 —
  https://arxiv.org/abs/2312.03031 (73.9 % straight; ego-status shortcut; ego-MLP ~0.29 m nuScenes)
- AD-MLP reference impl — https://github.com/E2E-AD/AD-MLP
- "Do Open-Loop Metrics Predict Closed-Loop Driving?" arXiv 2605.00066 (open-loop ⊥ closed-loop)
- NAVSIM v2 pseudo-sim / CV triviality filter — https://arxiv.org/abs/2506.04218
- Repo: `Benchmarks & Eval/DRIVING_DIAGNOSTIC_FRAMEWORK.md`, floor pkg
  `Implementation/incoming/2026-07-15-baseline-floor/`
