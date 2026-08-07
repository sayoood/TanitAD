# TanitAD-MPC — a sampling-MPC planner over the world model's predictive architecture

**Draft v1, 2026-08-06 (PI task: "draft and design the variant of the model using an MPC
planner using the predictive architecture of the wm").** Design only — no numbers claimed.

## 1. Why now — the measured motivation

Registry §1.12 (MEASURED 2026-08-06): with the recorded future actions removed, the current
decode stack loses its lateral skill almost entirely — hold-action reproduces **0 %** of GT
yaw-reversals (open-loop: 97.9 %), closed-loop ADE degrades 0.34→0.47 (v1.6) / 0.28→0.46
(v1.7), net-yaw ×6.7. **The open-loop lateral signal was an action echo.** Any deployable
configuration must therefore SUPPLY future actions rather than consume them — which is
exactly what MPC does by construction: it searches over candidate action sequences, using the
predictor as the rollout engine.

Second motivation: v5f measures a good fan and a weak selector (`oracle_ade` 0.53 improving,
`sel_gap` ~0.4 not closing). MPC replaces "learn to select" with "score by explicit +
learned costs" — a different attack on the same gap.

## 2. Architecture

```
  frames(8) ──► ENCODER (frozen, 87 M) ──► z_ctx
                                             │
     ┌── candidate action sequences ─────────┤
     │   a^(i) = (steer, accel)_{1..K},      ▼
     │   i = 1..N            PREDICTOR (frozen, 91 M) rolls each: ẑ^(i)_{1..K}
     │                                       │
     │                       READOUT (v1.6/v1.7 head or direct kinematic
     │                       integration of a^(i)) → trajectory τ^(i)
     │                                       ▼
     └──────────────  COST  J(τ^(i), ẑ^(i)) = Σ_j w_j C_j  ◄─── goal/route input
                             │                               (admissible per the
                     argmin / MPPI-weighted update            goal-input rule)
                             ▼
              a* — first action executed; REPLAN at 10 Hz
```

- **Sampler: MPPI / CEM over the UNICYCLE action space** (steer, accel at 0.1 s, K = 20).
  The v1.6 lesson transfers: parameterise candidates as controls, never free XY — every
  candidate dynamically feasible, |κ| ≤ 0.33, a ∈ vehicle envelope (same bounds Alpamayo
  publishes). Warm-start each cycle from the previous solution shifted one step (this also
  attacks the replan-jump metric structurally).
- **Rollout engine: the frozen predictor**, batched: N candidates roll in ONE forward pass
  per step ([N·B, …] batching — the fan is embarrassingly parallel, ~N× the cost of one
  roll; N = 64 with 2 CEM iterations ≈ 128 rolls per replan; A40 feasibility to be MEASURED,
  Thor deployment budget is the binding constraint and batch-8 SM saturation applies).
- **Action-channel contract:** candidates enter the predictor exactly as `signals_at`
  actions (steer = atan(2.9·κ)), the distribution it was trained on — the §1.12 pipeline
  already validates this interface end-to-end.

## 3. The cost — where the programme's instruments become the objective

| term | source | notes |
|---|---|---|
| imitation prior | distance to the readout head's own decode (v1.7) | keeps the search near learned behaviour; zero-weight ablation mandatory |
| progress / goal | distance-to-goal-point along candidate | goal = **predicted** goal point (goal-input rule: never the situation classifier's output) |
| comfort | accel/jerk/yaw-rate penalties (train-corpus p99 barriers — `kinematic.py` terms reused verbatim) | the SAME losses that trained v1.6/v1.7 become runtime costs |
| safety / distance-keeping | headway/TTC against predicted lead track | needs a lead predictor in latent space — probe first: ridge from ẑ to lead gap (the §1.10 probes' method; R² decides feasibility) |
| latent plausibility | −log-likelihood proxy: distance of ẑ^(i) to the predictor's unconditional roll | penalises action sequences the WM finds implausible; guards against adversarial-to-the-WM candidates |

**⛔ Attribution discipline:** every term has a weight flag and a pre-registered zero-weight
ablation; a planner that improved with ten live terms is unattributable (the `--v2` failure).

## 4. Relationship to the existing arms

- **vs v1.6/v1.7 (regress-and-ship):** MPC keeps them as the imitation prior AND the
  warm-start — they are the proposal distribution's mean, not competitors.
- **vs v5f (learn-to-select):** same fan-then-choose topology; v5f amortises the search into
  a selector network, MPC performs it online. The two share the oracle/selection instruments
  (`plan_ade`, `oracle_ade`, `sel_gap`) — directly comparable, same eval grid.
- **vs Alpamayo:** its diffusion expert samples 6 trajectories with NO explicit cost; minADE₆
  quietly assumes an oracle selector. MPC makes the selector explicit and auditable.

## 5. Evaluation plan (instruments exist)

Closed-loop grid of §1.12 (`closed_loop_dump.py` pattern, MPC supplies the actions), four
binding families + S-curve reproduction + lag/response + distance-keeping vs the lead block.
**The S-curve rate under MPC is the headline discriminator**: candidates that counter-steer
exist in the sampler by construction; whether the cost selects them tests whether the
LATENTS (via the cost's read of ẑ) carry the road geometry — the question §1.12 opened.

## 6. Staged experiments (each pre-registered, cheapest-first)

- **E0 (0 GPU-days):** cost-term feasibility probes on the banked dump — ridge R² from ẑ to
  lead gap / road curvature. Kills or funds the safety and geometry terms.
- **E1:** MPPI with kinematic integration only (no readout decode), imitation+comfort costs,
  40-episode closed-loop eval. The minimal falsifiable planner.
- **E2:** + goal term (predicted goal), + latent-plausibility term; ablation grid.
- **E3:** Thor latency budget: candidates × batch schedule vs the 10 Hz replan deadline
  (batch-8 saturation rule; MEASURED before any deployment claim).

## 7. Open questions carried honestly

1. Does the predictor stay faithful under SELF-proposed actions far from the data manifold?
   (§1.12 says quality degrades under self-actions near the manifold; MPPI explores wider.)
   Mitigation: the latent-plausibility cost + trust-region on candidate spread.
2. Perception loop stays open (context frames fixed at t0) — this is imagination-MPC, one
   replan deep; re-perception closed loop remains the AlpaSim/NuRec work item.
3. The lead-track predictor (safety term) may need its own small head — scope after E0.
