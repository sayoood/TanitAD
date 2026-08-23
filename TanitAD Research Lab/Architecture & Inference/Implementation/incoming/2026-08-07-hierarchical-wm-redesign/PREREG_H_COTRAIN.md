# PRE-REGISTRATION — H-COTRAIN: does joint WM+planner training overwrite physical-world
# state with ego-trajectory features? (PI hypothesis, 2026-08-11)

**The PI's hypothesis, faithfully stated:** the common training of WM + planner forces the
whole architecture to mimic trajectory information related to ego actions — so the trunk
learns an ego-action/trajectory feature extractor rather than the physical world.

**Why it is testable TONIGHT at ~0 marginal cost:** v5f's λ_plan curriculum is measured
and staged — **0 until step 2000, linear ramp to 8000, then 1.0 to 29999**
(`v4_curriculum.lambda_plan_at`, defaults `--phase-a-steps 2000 --phase-b-steps 8000`) —
and the restored milestone checkpoints (5k / 10k / 15k / 20k, plus the fully-measured 30k)
sample the trunk at λ≈0.5 and after 2k/7k/12k/22k steps of full planner gradient. If the
planner gradient progressively overwrites scene state, the milestone CURVE shows it.

## Existing evidence, both directions (all MEASURED, banked)

FOR the hypothesis:
- P1: ego state decodes from the PREDICTED latent BETTER than the encoded one (speed R²
  0.99 vs 0.74) — the rollout behaves like an ego-action integrator.
- P6/W3: action-induced latent change concentrates 0.91–0.94 of its energy in a 3-dim
  subspace — the ego interface dominates latent dynamics.
- §1.12: the action echo — open-loop "skill" that was action recitation.
- P1 lead follow-up: no readable lead-distance variable at any probe capacity.

AGAINST sufficiency (it may not be the cause):
- PUBLISHED: pure-SSL video models WITHOUT any planner also fail to form dense spatial
  features reliably (V-JEPA 2.1, arXiv 2603.14482) — absence of lead geometry does not
  require a planner-gradient explanation.
- The trunk retains plenty of non-action scene structure (P1 curvature 0.86; P7's
  calibrated uncertainty) — total mimicry is already excluded.

## The discriminating measurement (runs on pod5 after the current queue drains)

Probe battery (`probe_latent_state.py`, unchanged instrument, same corpus/join/881 grid)
on `ckpt_step{5000,10000,15000,20000}.pt`; the 30k row is already banked. Per milestone,
ENCODED-arm primary: speed / yaw-rate / curvature R², lead census + (dev-side, from the
banked arrays) the transform+MLP lead probes; P2 nuisance decodability; plus a latent
SPECTRUM summary (participation ratio + top-k eigenvalue share of z_enc) — the SIGReg
validation the PI asked to keep: SIGReg (full_relaxed, in the v4 loss at
train_flagship_v4.py:101) is doing its anti-collapse job iff effective rank does NOT
shrink as λ_plan ramps.

## Bound outcomes (before any number is seen)

- **H-COTRAIN CONFIRMED** if a scene variable that reads > 0.15 R²(enc) at step 5000
  DECLINES by ≥ 0.15 absolute by 30k while ego-dynamics R² does not decline (the planner
  gradient is eating scene state), or the spectrum's effective rank shrinks > 20 % across
  the ramp while ego probes hold. Consequence (pre-bound): λ_plan ceiling / gradient
  isolation (stop-grad from planner into encoder, planner reads spatial tokens through a
  bottleneck the WM loss does not share) becomes a NAMED v6 lever, and the
  `--lambda-plan 0` attributability arm (mode "0", byte-identical v1.5 regime) is the
  confirmatory training experiment for PI sign-off.
- **H-COTRAIN REJECTED (for the lead variable)** if lead-gap is unreadable at EVERY
  milestone including 5k — it never formed, so joint training cannot have destroyed it;
  the cause reverts to the survey's RC2/RC3 (objective/data pressure), and the label-free
  levers LF1–LF3 keep priority.
- **MIXED** per-variable outcomes are reported per-variable — no pooling.
- **SIGReg verdict** (independent): participation ratio at 30k ≥ 0.8× its 5k value ⇒
  VALIDATED (no progressive collapse under joint training); below ⇒ a real finding that
  the planner gradient defeats the regulariser, which itself would SUPPORT H-COTRAIN.

Artifacts: `p1-cotrain-{step}/p12_gate.json` + spectrum JSONs + a banked curve summary.
Estimator discipline: pooled OOF R² per milestone; episode-cluster bootstrap before any
registry claim; tier T0-diagnostic (representation interrogation).
