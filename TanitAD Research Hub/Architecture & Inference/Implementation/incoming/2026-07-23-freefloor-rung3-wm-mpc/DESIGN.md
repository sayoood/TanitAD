# Free inference-time floor — RUNG 3: WM-planning (MPPI / CEM) over the world model

**Date:** 2026-07-23 (Berlin) · **Evidence class:** MEASURED (`wm_mpc_result.json`) ·
**Model:** flagship v1 (`flagship4b-speedjerk-30k`, ckpt step 29999, HF
`Sayood/tanitad-flagship-4b-speedjerk`) · **Compute:** RTX 4060 (local, no pod touched) ·
**Harness:** `taniteval/taniteval/closedloop.py` (task #21, no-renderer kinematic) + `wm_mpc.py` (this rung).

## Where this sits

The 2026-07-23 closed-loop WM verdict (`Architecture & Inference/Research/2026-07-23-closed-loop-wm-training-verdict.md`)
defines a **free inference-time floor** (ladder rows 1–3) to build regardless of any training lever,
because rows 1–3 add no training and *cannot learn to game the WM*:

1. road-boundary safety clamp / shield (sibling agent, Gate 0)
2. cost-guided diffusion at inference (sibling agent, Gate 0)
3. **WM-planning / MPC — upgrade the −0.213 re-planner to MPPI/CEM + (optional) learned terminal value** ← THIS

**Rung 3 baseline (MEASURED, `incoming/2026-07-22-imagination-closedloop-proof/`).** On the flagship v1
WM, two arms paired on identical held-out windows:

| arm | what it does | closed-loop ADE@2s | div>5 m@2s |
|---|---|---|---|
| **A** `open_plan_bike` | plan ONCE on the real latent, track the frozen plan open-loop (REF-C-style single-shot; predictor never stepped) | 1.933 [1.632, 2.275] | 39.2% |
| **B** `closed_bike` | **single-step re-plan**: re-plan the head every 0.1 s on the model's OWN imagined latent, follow the 0.5 s waypoint with a pure-pursuit inverse | 1.720 [1.444, 2.040] | 22.3% |

**paired Δ(B − A) ADE@2s = −0.213 m [−0.341, −0.053], separated (IMAGINATION_HELPS).**

Rung 3 asks: does **sampling-based receding-horizon planning over the WM** beat the single-step re-plan (B)?

## What changes — ONLY the action selection

Arm **C** `mpc_bike` keeps everything in arm B — the same strategic→tactical head re-planned every tick,
the same kinematic-bicycle drive, the same latent-imagination bookkeeping, the same held-out windows,
byte-identical at tick 0 — and replaces **only** B's deterministic pure-pursuit inverse with
**receding-horizon MPPI / CEM over the world model**. So the paired delta **(C − B)** isolates exactly
"sample K action sequences, roll each through the WM, select by imagined cost" versus "single-step re-plan".

Per 0.1 s control tick:

1. **Re-plan the head anchor** on the current (imagined) latent window → 2 s ego-frame waypoints (== arm B).
2. **Nominal action sequence** over horizon `H`: pure-pursuit tracking the densified anchor, open-loop
   through the bicycle. Its **first action == arm B's action** ⇒ MPPI is a *strict superset* of B
   (candidate-0 is always B's choice, so C can only differ if the WM prefers a sampled candidate).
3. **Sample `K` action sequences** = nominal + Gaussian noise on (steer, accel), clamped to the observed
   action envelope (`|steer|≤0.05`, `|accel|≤3.0`).
4. **Roll every candidate through the WM** operative predictor (`predictor(z, a)[1]`, the 1-step head),
   decoding the imagined metric path with the grounded step-readout (`accumulate_se2`). This is the
   *imagination* — the WM as its own simulator, broadcast to `b·K` in one batched pass (encode once).
5. **Score each imagined trajectory** with a **ground-truth-free** cost (the planner cannot see GT at deploy):
   - `track` = ‖imagined path − head anchor‖² over `H` — **the stay-on-intended-path / lateral-deviation
     "off-road" proxy** the floor targets (no map in our data → deviation from the intended path is the proxy);
   - `progress` = −forward reach at `H` (small weight; the 2-D anchor already encodes the speed intent);
   - `comfort` = jerk + |steer| + |accel| (the floor's comfort/jerk signal).
   Each component is **standardized across the K candidates per window** (scale-free, no magic units),
   then combined with pre-registered weights.
6. **Select the receding-horizon action.** MPPI: `w_i = softmax(−J_i / temp)`, take the **weighted first
   action**. CEM: take the **mean first action of the top-M elite** (lowest-cost) candidates.
7. **Execute** that first action exactly as arm B (imagine one WM step, integrate the bicycle, slide the
   window) → receding horizon.

**Learned terminal value (TD-MPC2-style):** deferred by design. The brief is "get plain MPPI/CEM working +
measured FIRST"; a learned value needs a trained `V`, which is a *training* task (out of scope: rung 3 is
inference-only, no training, no reward). The head's anchor **beyond H** already acts as an implicit terminal
reference for the tracking cost. A learned terminal value is the documented next step *iff* plain MPC clears the bar.

## Pre-registration (config LOCKED before the measured run; both outcomes committed)

Headline config (`wm_mpc.MPC_DEFAULT`, chosen a priori from the action envelope + comfort bounds,
**NOT tuned against the closed-loop ADE** — tuning K/σ/λ/H/weights on the same val episodes would leak the
test set and is refused):

```
method=mppi  K=8  H=8 (0.8 s)  sigma_steer=0.008 rad  sigma_accel=0.40 m/s^2
temp=0.7  w_track=1.0  w_progress=0.05  w_comfort=0.20  cost_path=grnd (WM-imagined)
```

Reading, on the SAME 12 held-out val episodes / windows as the −0.213 proof, **paired episode-cluster
bootstrap** (`taniteval/ci.py`), decision predicate = CI excludes 0:

- **C beats B** — paired Δ(C − B) ADE@2s **< 0 and CI-separated** (and/or divergence>5 m / lateral-off-road
  proxy separated in C's favour) ⇒ **rung 3 is a real lever beyond the single-step re-plan → PROMOTE to
  AlpaSim** for the decision-grade (non-self-referential) test.
- **C ties B** — Δ(C − B) CI **not** separated ⇒ **the single-step re-plan already captures the imagination
  benefit** at this fidelity; MPPI adds no measurable closed-loop gain ⇒ report plainly, do **not** promote
  on this evidence (AlpaSim could still separate them).
- **C worse than B** — Δ(C − B) **> 0 and CI-separated** ⇒ sampling+selecting on **this self-referential
  WM's** imagined latents degrades vs simply re-planning the head (a WM-exploitation signature in miniature)
  ⇒ keep the single-step re-plan; the lever needs a faithful sim + pessimism first.

**Compute budget:** report `ms/decision`; the registry measured CEM **~20.8 ms at K=8** (encode once,
broadcast — NOT re-encode per candidate; RETRACTION_LOG 07-21 "CEM infeasible 723 ms" was that arithmetic
error). Confirm it stays within a real-time tick (100 ms @ 10 Hz).

**Secondary grid** (reported as *sensitivity*, explicitly optimistic-if-selected — the headline is the
pre-registered config, never the grid-best): `K∈{4,16}`, `CEM(K=8)`, `cost_path=bike` (WM removed from the
cost — isolates whether the WM *consequence* matters vs pure kinematic MPC), `H=12`; plus a batch=1 single-ego
timing sweep `K∈{8,16,32}`.

**Built-in sanity check:** arms A and B are produced by the **unmodified** `closedloop.open_loop_plan_rollout`
/ `closed_loop_rollout`, so on the full 12 episodes they must reproduce the proof (A≈1.933, B≈1.720,
Δ(B−A)≈−0.213) — exactly as the DAgger probe's A0 reproduced 1.720. If they don't, the harness wiring is wrong
and the C reading is void.

## Caveats (binding — this is a MECHANISM proof, not a safety rate)

- **Self-referential.** The WM is BOTH the planner's state estimator and the simulator it plans against.
  MPPI selecting by imagined cost is the *cheap* form of the v4 imagine-and-select thesis; failures the WM
  cannot imagine are invisible. The candidate-0 = nominal anchoring + small σ + short H keep candidates near
  the already-validated arm-B action, bounding the off-manifold exposure the DAgger finding warned of —
  **but only AlpaSim (faithful, non-self-referential) is decision-grade.** This harness is VALIDATED for
  inference-mechanism proofs (it produced −0.213) and REFUTED for training (DAgger); rung 3 is inference-only.
- **Kinematic no-renderer harness**; off-road = large **lateral-deviation proxy**, not a map/collision rate.
- **n = 12 val episodes** (~265 windows), the cheap subset — the arms are paired on identical windows so the
  **Δ is the robust part**; absolute numbers carry wide CIs.
- The pure-pursuit inverse + bicycle are a **harness controller** shared by A/B/C; their fidelity floor
  cancels in the paired delta.

## Deliverable manifest — see VERDICT.md (results appended there + in `wm_mpc_result.json`).
