# VERDICT — free inference-time floor RUNG 3: WM-planning (MPPI / CEM) over the world model

**2026-07-23 (Berlin) · MEASURED (`wm_mpc_result.json`) · flagship v1 (`flagship4b-speedjerk-30k`,
ckpt 29999) · RTX 4060, no pod touched · n = 265 windows / 12 held-out val eps · paired
episode-cluster bootstrap (`taniteval/ci.py`).**

## Verdict: **TIE — WM-MPPI/CEM planning does NOT beat the single-step re-plan. Do NOT promote rung 3 on this evidence.**

Upgrading arm B's single-step re-plan (the MEASURED **Δ ADE@2s −0.213** imagination win) to
**receding-horizon MPPI/CEM over the world model** — sample K action sequences, roll each through the WM,
select by a GT-free stay-on-path + progress + comfort cost — **adds no measurable closed-loop gain** at this
fidelity. Every WM-cost config **ties** the single-step re-plan (paired CI includes 0); all deltas are
slightly **positive** (MPPI marginally *worse*, never better); and on the floor's own **lateral off-road
proxy** MPPI is **CI-separated worse**. The single-step re-plan already captures the imagination benefit.

This is the pre-registered **TIE branch** (`DESIGN.md`): report plainly, do not promote — AlpaSim (faithful,
non-self-referential) could still separate them, but the cheap harness gives no reason to.

## The number (paired Δ, negative = MPC beats the single-step re-plan B)

Baseline (reproduced EXACTLY in every config → harness wiring valid): **A** single-shot = **1.933**,
**B** single-step re-plan = **1.720**, **Δ(B−A) = −0.213** (the proof).

| config | K | H | method | cost | C ADE@2s | **Δ(C−B) ADE@2s** | 95% CI | separated | ms/dec (b≤8) |
|---|---|---|---|---|---|---|---|---|---|
| **headline** | 8 | 8 | MPPI | WM-grnd | 1.731 | **+0.011** | [−0.029, 0.071] | No → **TIE** | 14.1 |
| K16 | 16 | 8 | MPPI | WM-grnd | 1.725 | +0.005 | [−0.038, 0.066] | No → TIE | 27.5 |
| K4 | 4 | 8 | MPPI | WM-grnd | 1.757 | +0.038 | [−0.031, 0.133] | No → TIE | 8.0 |
| CEM | 8 | 8 | CEM | WM-grnd | 1.726 | +0.007 | [−0.027, 0.054] | No → TIE | 14.1 |
| bike-cost | 8 | 8 | MPPI | kinematic | 1.807 | +0.087 | [0.033, 0.164] | **Yes → WORSE** | 14.3 |
| H12 | 8 | 12 | MPPI | WM-grnd | 1.764 | +0.044 | [−0.025, 0.133] | No → TIE | 21.2 |

**Best config is K16 (Δ = +0.005) — still a tie.** No WM-cost config separates in C's favour; the only
separated result is *worse*.

Headline secondary metrics (K8, WM-grnd), paired Δ(C−B):

- divergence>5 m@2s = **+0.004** [−0.023, 0.038] — TIE (B 22.3% → C 22.6%)
- FDE@2s = **+0.040** [−0.070, 0.212] — TIE
- **lateral off-road proxy@2s = +0.136 [0.006, 0.327] — CI-separated WORSE** (B 1.30 m → C 1.44 m)
- Δ(C−A) ADE@2s = **−0.202** [−0.362, 0.015] — C recovers ~all of B's −0.213 win over single-shot, but
  marginally noisier so its CI just grazes 0.

## Why (HYPOTHESIS, not isolated)

1. **The pure-pursuit inverse is already near-optimal for tracking the head anchor**, and under near-nominal
   actions the WM's imagined dynamics are ~bicycle-consistent — so sampling around the nominal mostly
   recovers it (candidate-0 = arm B by construction; MPPI is a strict superset that finds nothing better).
2. **Stochastic action selection adds lateral jitter** → the separated-worse off-road proxy + high jerk
   (C mean |jerk| 3.78 m/s³): the sampled first-action variance costs smoothness the deterministic inverse keeps.
3. **The WM-grounded cost is the *better* of the two costs, not the problem.** The one separated result is
   `cost_path=bike` (**+0.087 worse**) — planning against the kinematic bicycle instead of the WM's imagined
   consequence hurts *more*. So the WM consequence in the cost helps relative to kinematic-only MPC; it just
   doesn't help relative to the single-step re-plan.

## Compute cost (well within a real-time tick)

- **Batched (b ≤ 8), RTX 4060:** K4 = 8.0, K8 = 14.1, K16 = 27.5 ms/decision (≈ linear in K); H12 = 21.2.
- **Single-ego deploy tick (batch=1, 40-tick mean):** MPPI K8 = **49.7**, K16 = 48.2, K32 = 62.1, CEM K8 =
  49.5 ms/decision — **all comfortably under the 100 ms / 10 Hz tick budget.** Same ballpark as the registry's
  ~20.8 ms @ K=8 (mine ~2× on a 4060 with an H=8 rollout; still real-time). **Feasibility is NOT the blocker
  — efficacy is.** (Confirms RETRACTION_LOG 07-21: encode once + broadcast; the "CEM 723 ms" claim was wrong.)

## Decision

- **Do NOT promote WM-MPPI/CEM as the rung-3 lever on this evidence.** The single-step re-plan (already in the
  deployed anchored-diffusion planner path) captures the imagination benefit; sampling-based WM-MPC adds cost
  and slightly worse road-keeping, not gain.
- **Rung 3 is deferred, not refuted** — like DAgger, it re-enters only as an **AlpaSim-validated** test
  (faithful, non-self-referential rollouts). The self-referential caveat cuts BOTH ways: this harness cannot
  credit a planner for consequences the WM cannot imagine, so a real off-road/collision cost (which needs a
  map/renderer) may still favour MPC where the lateral-deviation proxy cannot see it.
- **A learned terminal value (TD-MPC2) is not worth building yet** — plain MPC has not cleared plain re-plan,
  so a value head (a *training* task, out of scope here) has nothing to extend.
- **Keep the floor budget on rungs 1–2** (drivable-area guidance + road-boundary clamp, Gate 0) — those attack
  the off-road failure directly and cannot be gamed; rung 3 does not add to them here.

*Caveats (binding): self-referential + kinematic no-renderer harness (off-road = large lateral-deviation
proxy, NOT a real safety rate); n=12 val eps; within-harness MECHANISM proof; arms paired on identical
windows so the Δ is the robust part. AlpaSim is the decision-grade confirmer.*

## Deliverable manifest

| artifact | location | staged |
|---|---|---|
| MPPI/CEM planner + collect/analyze (arm C) | `repo: .../incoming/2026-07-23-freefloor-rung3-wm-mpc/wm_mpc.py` | git add |
| local driver (headline + grid + timing) | `repo: .../incoming/2026-07-23-freefloor-rung3-wm-mpc/run_mpc_local.py` | git add |
| MEASURED result (headline + grid + timing) | `repo: .../incoming/2026-07-23-freefloor-rung3-wm-mpc/wm_mpc_result.json` | git add |
| design + pre-registration | `repo: .../incoming/2026-07-23-freefloor-rung3-wm-mpc/DESIGN.md` | git add |
| this verdict | `repo: .../incoming/2026-07-23-freefloor-rung3-wm-mpc/VERDICT.md` | git add |
| provenance | `repo: .../incoming/2026-07-23-freefloor-rung3-wm-mpc/provenance.json` | git add |
| smoke result (2-ep calibration) | `repo: .../incoming/2026-07-23-freefloor-rung3-wm-mpc/wm_mpc_result_smoke.json` | git add |

**Reused read-only (NOT modified):** `taniteval/taniteval/closedloop.py` (task #21 harness — arms A/B via its
unmodified `open_loop_plan_rollout` / `closed_loop_rollout`, which reproduce the −0.213 proof), `taniteval/ci.py`
(paired episode-cluster bootstrap), `taniteval/registry.py`, `tanitad.models` (flagship v1 WM). No pod, no
training, no reward. `pytest` unaffected — no `stack/` or `taniteval/` package file was changed.

_All staged (git add), NOT committed, NOT pushed (Agent Operating Standard)._
