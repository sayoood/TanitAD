# Benchmark portfolio for a camera-only world-model + planner — ranked, GO/SKIP

**Package** `Benchmarks & Evals/Research/2026-08-29-benchmark-portfolio` (serves Opponent
Analysis too) · **Author** Research Lab agent (daily run 003, 2026-08-29) · literature only,
0 GPU. Builds on `Opponent Analysis/Research/2026-08-28-navsim-v2-camera-lane/RESULT.md`
(navhard protocol split, camera-only field, ego-status blocker) — nothing there is re-decided
here. **Purpose: the PI wants every benchmark PREPARED before scaled training finishes; this
is the approve-wholesale table.**

## The ranked portfolio

| # | benchmark | what it measures | loop type | camera-only? | cost (data / compute) | transfer to our claims | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **NavSim v2 navhard-two-stage** (EPDMS) | 9 sub-metrics aggregated post-simulation; stage 2 = 3DGS counterfactual starts | pseudo-closed-loop | YES — the active camera-only lane (55.5 learned / 56.3 +TTO / 56.6 privileged ceiling) | LOW-MED: offline scoring, released renders; shares the 450 GB NAVSIM data | closest community proxy to **T1** (counterfactual starts break the action-echo); 9 sub-metrics map onto the four families | **GO — already pinned 2026-08-28; unchanged** |
| 2 | **NAVSIM v1 navtest** (PDMS) | PDMS over 12k scenes (navtrain 103k) | pseudo-closed-loop | YES | MARGINAL once #1's infra exists (same 450 GB standalone download) | the ONLY placement of our declared reference **Drive-JEPA (89.0 perception-free / 93.7 full)** — required for reference comparability | **GO (piggyback on #1)** |
| 3 | **Bench2Drive** (CARLA v2) | Driving Score + success + **multi-ability** (merge/overtake/emergency-brake/give-way/traffic-sign) + smoothness (jerk, yaw) + efficiency | **TRUE closed-loop** | YES — UniAD/VAD/TCP-family baselines are camera-based; best DS 64.22 (DriveAdapter) | HIGH: 2 M-frame Think2Drive train set (Apache-2.0) ⇒ a separate CARLA-domain arm; reference eval cost 8×H800×2 days (VAD-scale; ours sub-300M ⇒ less, ESTIMATED pod-days) | the only community benchmark that scores **T1 closed-loop directly**; multi-ability ↔ TACTICAL family; smoothness ↔ lateral/comfort | **GO-conditional: PREP NOW, execute after scaled training** |
| 4 | **WOD-E2E** (Waymo vision E2E) | Rater Feedback Score (human preference labels) on 4,021 long-tail segments (<0.03 % frequency), 8-cam 360°, routing+ego given | open-loop | YES — by design | MED: TB-scale download (ESTIMATED); RFS eval itself cheap; **val labels public ⇒ local eval possible** | NOT T1 — but long-tail tactical judgment vs HUMAN preference complements the tactical family; the camera-only community focal point (yearly challenge) | **GO-second-wave (local val-RFS; challenge entry optional)** |
| 5 | **nuScenes open-loop** (L2/collision) | 3 s trajectory L2 + collision rate | open-loop | YES (6-cam pinhole rig ≠ ours) | LOW | **claim-inadmissible as driving evidence** — see criticisms below; fails four-family by construction | **SKIP as claim-bearing** (produce only if a reviewer demands, then only alongside the criticisms, stamped T0-open-loop) |
| 6 | **CARLA Leaderboard 2.1** (official) | DS on hidden long routes | true closed-loop | YES (SENSORS track) | VERY HIGH: month-scale submission engineering; official 2.0 sensor SOTA was ~6.8 DS; **2.0→2.1 scoring break (2025-03) voids longitudinal comparability** [PUBLISHED-SECONDARY, deliberate for a SKIP row] | #3 already covers CARLA closed-loop at tractable cost | **SKIP** (revisit only for public-visibility goals) |
| 7 | *nuPlan* (context row) | closed-loop planning w/ privileged perception | closed-loop | NO (not a sensor benchmark) | — | superseded for sensor agents by NAVSIM; its own indictment is banked (PDM: rule-based wins closed-loop) | **SKIP** |

## The published criticisms of nuScenes open-loop (all banked — cite whenever row 5 appears)

- **AD-MLP** [lib `2305.10430`]: an MLP with NO perception input (past trajectory, velocity)
  *reduces L2 by ~20 %* vs perception-based methods — open-loop L2 rewards ego-state
  extrapolation, not driving.
- **BEV-Planner** [lib `2312.03031`]: nuScenes' *"relatively simple driving scenarios"* lead
  models to *"rely predominantly on the ego vehicle's status"*; *"current metrics do not
  comprehensively assess the planning quality, leading to potentially biased conclusions."*
- **PDM** [lib `2306.07962`]: short-term planning and long-horizon ego-forecasting are
  *"fundamentally misaligned"*; the best open-loop result uses ONLY the centerline (ignoring
  map and agents); rule-based PDM-Closed wins closed-loop nuPlan.
- These are the community's version of our own MEASURED action-echo fact (EVAL_DOCTRINE §1.12:
  open-loop lateral skill 97.9 % → closed-loop ~5 %) — the doctrine and the field agree.

## Four-family coverage across the GO set

| family | NavSim v2/v1 | Bench2Drive | WOD-E2E |
|---|---|---|---|
| LONGITUDINAL | EP, TTC, NC | efficiency, emergency-brake | (inside RFS, not separable) |
| LATERAL | LK, DDC | smoothness (jerk/yaw) | (inside RFS) |
| TACTICAL | DAC, TLC | **multi-ability 5-way** | long-tail preference RFS |
| STRATEGIC | route-following in EP | route completion | routing given → conditioned |

No single benchmark covers all four; the GO set jointly does. Every quoted number carries
split + harness version (the ~30-pt two-protocol trap, predecessor F2).

## What this changes for TanitAD (≤3)

1. **Approve wholesale:** GO = navhard-two-stage (pinned) + navtest piggyback; GO-conditional
   = Bench2Drive (prep now, run post-scaled-training as the T1 community validation);
   GO-second-wave = WOD-E2E val-RFS; SKIP = nuScenes-OL claim-bearing, CARLA LB 2.1, nuPlan.
2. **Prep work items to schedule NOW** (all 0-training-blocking): (a) NAVSIM harness + 450 GB
   navtrain/navtest/navhard pull onto a pod (dd-verify quota, never df); (b) Bench2Drive repo
   + CARLA v2 on the Vulkan-fixed pod + loader for the Apache-2.0 2 M-frame set (separate
   arm — NEVER mixed with the parity corpus); (c) WOD-E2E val download + RFS scorer;
   (d) the register-token rig adapter (predecessor rec #2) is shared prep for #1/#2/#4 — one
   mechanism, three benchmarks.
3. **The ego-status two-arm design (PI decision, restated not re-decided)** now applies
   portfolio-wide: rows 4–5 hand ego status to entrants, and the banked criticisms show
   exactly why that inflates apparent skill — our vision-pure arm is publishable
   differentiation on EVERY row where leaders consume ego status.

## Named empty searches

- A **2026 edition** of the Waymo vision-E2E challenge: not confirmed this pass (2025 edition
  + CVPR-2026 WOD-E2E poster confirmed; cadence suggests yes — UNVERIFIED).
- Official **CARLA LB 2.1 camera-only standings post-rescore**: not resolved (leaderboard
  listing not fetched; the ~6.8 DS figure is 2.0-era, PUBLISHED-SECONDARY).
- **Bench2Drive eval wall-clock on a single consumer GPU**: no published figure; the
  8×H800×2-day reference is VAD-scale — our cost row is ESTIMATED and needs a measured pilot.

## Banked primaries

NEW: 2406.03877 (Bench2Drive) · 2510.26125 (WOD-E2E/RFS). CITED-BY UPDATED: 2406.15349
(NAVSIM v1) · 2305.10430 (AD-MLP) · 2312.03031 (BEV-Planner) · 2306.07962 (PDM). NavSim-v2
field numbers inherit from the predecessor package's banked libs (2601.22032, 2601.05083,
2606.07170). CARLA LB 2.1 facts deliberately left PUBLISHED-SECONDARY (SKIP row, not
registry-bound).
