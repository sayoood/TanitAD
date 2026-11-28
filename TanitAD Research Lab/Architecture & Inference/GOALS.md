# GOALS — Architecture & Inference (D-029 standing objectives)

> 1–3 concrete, measurable objectives with a target number + deadline. Each run: advance one with a
> measured step; a goal with no movement for two runs is escalated in STATE, not silently carried.
> Created 2026-07-17.

## G1 — Prove (or falsify) that the trained-encoder 4B world model beats constant-velocity on D1 [TOP PROGRAM RISK]
- **Target:** flagship @30k open-loop **D1 fwd-ADE@1s < the honest kinematic floor is impossible**
  (floor 0.056 m is near-unbeatable on highway); realistic architecture-side target = **D1@1s within
  2× of the best learned reference arm (REF-A 2.14 m) AND D2 action-discrimination not regressed**, by
  **W34 (≈2026-07-31)**. The capability proof is closed-loop D4–D6 (open-loop ADE ⊥ DS); my job is to
  ensure the *architecture* is not the bottleneck (readout sizing, conditioning, imagination) when the
  verdict lands.
- **Instrument:** gate runner D1/D2 + spectral + orthogonality on the flagship ckpt.
- **This-run step (2026-07-18):** E2 re-run on the **operative flagship-speed @19k** (not the pre-reset
  ckpt): active_k≈19, cov_eff_rank≈30 ≪ 2048 → readout capacity is NOT the D1 bottleneck, **reaffirmed on
  the shipping model**; and iso_ratio_active **0.254→0.546** (SIGReg converging toward the LeJEPA condition
  as predicted). Rules out "too small / mis-shaped a latent" as the D1-failure cause. Movement: **yes.**
- **Status:** on-track, gated on flagship @30k (~Jul-19–23) — E1+E2 re-run is turnkey the day it lands.

## G2 — Make the H15 imagination edge safe + honest for operative K-step self-monitoring
- **Target:** an H15 σ signal that **grows monotonically with blind-rollout horizon** (calibrated
  uncertainty) at k∈{1,2,4}, validated by D8 AUROC > 0.85 on degraded-visibility episodes, by **W35
  (≈2026-08-07)**.
- **This-run step (2026-07-18):** E1 re-run on the **operative flagship-speed @19k** — σ-dissipation +
  attractor collapse **REPRODUCE** (falsifier "speed recipe fixed it" NOT met); attractor sharper
  (→0.805), absolute σ *lower* (worse). Refinement: σ is *spatially* calibrated (hidden>visible +0.37;
  err↔var corr +0.29–0.43) but *temporally* flat → the target narrows to a **horizon-aware** σ (not a
  spatial rebuild). freeze-1 flat-safe on the shipping model too. Movement: **yes** — target sharpened,
  operative baseline set.
- **Status:** on-track; next = prototype 0b-A multi-step-rollout training (horizon-aware σ + anti-attractor)
  OR adopt 0b-B parallel-horizon operative mode (D-018 escalate before trained-config change).

## G3 — Efficiency moat: every architecture lever carries a measured quality-per-FLOP number
- **Target:** a live FLOPs/latency ledger (backlog #5) auto-appended to every bake-off arm, with batch-1
  4060 + Orin-projected numbers, by **W34**.
- **This-run step:** none (H15 per-tick latency was 2026-07-15). Movement: **no** — carry; not yet 2 runs stale.
- **Status:** pending; behind G1/G2 in priority while the flagship verdict is the critical path.
