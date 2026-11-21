# LAB RUN 001 — Benchmarks & Evals — literature pass + ideation

`TanitAD Research Lab, daily run 001. Run label 2026-08-22 (Master Mind trigger);
executed 2026-08-23 wall-clock. Literature only, no GPU. Agenda targets:
RESEARCH_AGENDA.md Field 5, item 1 (⭐ MANDATORY NavSim adoption) and item 2
(eval science: open-loop ↔ closed-loop divergence).`

## Findings

### F5.1 — Pseudo-simulation: closed-loop correlation without sequential simulation ⭐ banked
- **Pseudo-Simulation for Autonomous Driving** — Cao, Hallgarten, Li, Dauner
  et al., arXiv **2506.04218** (Jun 2025; v3 Mar 2026; CoRL 2025; the
  `autonomousvision/navsim` protocol). **PUBLISHED, BANKED (library key
  `2506.04218`, tag `benchmark-navsim`)**. Code + public leaderboard released.
- **Finding:** evaluate on REAL logs (like open-loop) but augment them with
  **3D-Gaussian-Splatting-synthesised observations** varying in position,
  heading and speed around the planner's own first-stage endpoint, then
  **proximity-weight** the synthetic observations toward the AV's likely
  behaviour. Correlates with closed-loop simulation at **R² = 0.8** vs **0.7**
  for the best open-loop metric, while testing error recovery and causal
  confusion "without requiring sequential interactive simulation".
- **Impact:** this is the missing tier between our **T1** (action-closed loop)
  and the unprovisioned **T2** (re-perception sim) — and we already hold every
  ingredient: NuRec scenes (open msgpack), gsplat at 492 FPS on Thor, the T1
  harness, 40 val episodes. Our own measured open↔closed divergence (4.05×,
  agenda item 2) is exactly the gap this protocol was built to close.
- **What it would change:** TanitEval gains a NAVSIM-v2-style two-stage tier on
  OUR data (name to be fixed by the EvalFlyWheel — "T1.5" is a placeholder, the
  glossary owns the term), which is simultaneously the path to running NAVSIM
  v2 verbatim for community comparability.

### F5.2 — NAVSIM v2 / EPDMS: the mandatory target's metric set, precisely
- **NAVSIM v2 (Extended PDM Score)** — protocol as published in the
  `autonomousvision/navsim` repository and the pseudo-simulation paper.
  **PUBLISHED (protocol, secondary until the repo/tech report is banked).**
- **Finding:** EPDMS extends PDMS with Traffic-Light Compliance (TLC),
  Driving-Direction Compliance (DDC), Lane Keeping (LK) and Extended Comfort
  (EC), replacing Comfort with History Comfort (HC); two stages — EPDMS₁ on the
  real observation, EPDMS₂ on 3DGS-synthesised follow-ups around the stage-1
  endpoint, fused by Gaussian-weighted aggregation. Inputs: 2 s ego history +
  multi-view cameras; output: 4 s trajectory at 2 Hz.
- **Impact:** agenda item 1's "exact protocol, metrics, submission format" is
  now specified to the term level. Our four-family doctrine maps onto EPDMS
  sub-terms (LK/DDC ≈ lateral, TTC/EP ≈ longitudinal, DAC/TLC ≈ tactical) —
  EPDMS pools them; we keep families UNPOOLED internally and report EPDMS
  alongside for comparability, never instead.
- **What it would change:** P7 work items: EPDMS term calculators + the
  NAVSIM submission adapter; a registry column for EPDMS (tier-stamped).

### F5.3 — Reduced-fidelity simulation at massive throughput is a valid substrate
- **Scaling Self-Play for End-to-End Driving** — Rowe, Girgis, …, Heide,
  Vinitsky, Pal, Paull, arXiv **2606.19641** (Jun 2026). **PUBLISHED (abstract
  verified); NOT banked — secondary.**
- **Finding:** *Gigapixel*, a perspective renderer of a simplified bounding-box
  world at **50 000 agent-steps/s**; pixel policies trained by **self-play
  DAgger** (on-policy distillation from a privileged RL teacher) are
  competitive on **HUGSIM and NAVSIM-v2 without any human trajectory
  supervision**, and performance scales with self-play compute.
- **Impact:** a published existence proof for the Lab's "prove at small scale"
  mandate — eval and training substrates need not be photoreal to transfer;
  also a data point for the RL-post-training trend (TrainingFlyWheel mandate).
- **What it would change:** TanitSim's MVP may be a box-world renderer over our
  REAL scenario logs (world-on-rails others) rather than a neural renderer; the
  neural tier (F5.1) sits above it.

## Ideation — our own hypothesis

**H-EVAL-1 (OPEN, proposed).** *Pseudo-simulation survives reduced fidelity on
our own assets.* The pseudo-simulation protocol, re-implemented on our
NuRec/gsplat scenes at our corpus resolution (256×640, cylindrical projection —
stated because the pinhole formula is wrong there), preserves the closed-loop
correlation property: its score correlates with our T1 harness outcomes on
matched val episodes at **R² ≥ 0.7**, at **< 1/10** the compute of sequential
closed-loop rollout.

- **Cheapest discriminating experiment:** 10 val episodes with NuRec
  reconstructions; synthesise stage-2 observations on a small
  position/heading/speed grid around each arm's stage-1 endpoint (gsplat on
  Thor *after* the production run, or on the dev-box 4060 at reduced grid);
  score two existing arms; correlate against banked T1 outcomes per episode
  (paired episode-cluster bootstrap; n printed). Both arms' T1 numbers already
  exist — no new closed-loop rollouts needed.
- **Outcome A (R² ≥ 0.7):** the tier is adopted; NAVSIM-v2 verbatim becomes a
  configuration of the same code path; TanitSim gets its first measured
  fidelity rung.
- **Outcome B (R² < 0.7):** reduced fidelity breaks the correlation — report
  *which* perturbation axis fails (position vs heading vs speed) and escalate
  the renderer-fidelity requirement to the TanitSpear groundwork item.

## Transfer note (charter §7) — ⭐ immediately actionable

→ **TanitAD_EvalFlyWheel:** F5.1/F5.2 are the concrete implementation path for
the MANDATORY NavSim adoption (charter §5): (1) EPDMS term calculators,
(2) the two-stage protocol on our NuRec/gsplat assets (H-EVAL-1 as the
acceptance test), (3) leaderboard rows for NAVSIM-v2 baselines as
claimed-not-reproduced until rerun. This is the programme's largest evaluation
gap and the Lab has no instrument to close it alone — transfer requested with
accept/reject and reason.

## Evidence discipline

Paper numbers are PUBLISHED from verified abstracts; the banked key is the only
registry-admissible citation; programme figures (4.05× divergence, 492 FPS,
40 val episodes) are INHERITED from CLAUDE.md/memory and not re-measured here.
