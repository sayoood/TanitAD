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
- **This-run step:** E2 verified (readout active-rank ≈23–26/2048, iso_ratio_active 0.254 NOT-YET-admissible)
  → readout capacity is NOT the binding constraint (over-provisioned *and* under-isotropic); rules out
  "too small a latent" as the D1 failure cause. Movement: **yes.**
- **Status:** on-track, gated on flagship @30k (~Jul-19–23).

## G2 — Make the H15 imagination edge safe + honest for operative K-step self-monitoring
- **Target:** an H15 σ signal that **grows monotonically with blind-rollout horizon** (calibrated
  uncertainty) at k∈{1,2,4}, validated by D8 AUROC > 0.85 on degraded-visibility episodes, by **W35
  (≈2026-08-07)**.
- **This-run step:** E1 measured that the current 1-step field **dissipates σ** (−7.79→−8.55) and
  **collapses to an attractor** under autoregressive rollout → target quantified, cause localized to the
  recursion (freeze-1 is flat-safe). Movement: **yes** — baseline established, two design routes framed.
- **Status:** on-track; next = prototype 0b multi-step-rollout training OR adopt parallel-horizon
  operative mode (D-018 escalate before trained-config change).

## G3 — Efficiency moat: every architecture lever carries a measured quality-per-FLOP number
- **Target:** a live FLOPs/latency ledger (backlog #5) auto-appended to every bake-off arm, with batch-1
  4060 + Orin-projected numbers, by **W34**.
- **This-run step:** none (H15 per-tick latency was 2026-07-15). Movement: **no** — carry; not yet 2 runs stale.
- **Status:** pending; behind G1/G2 in priority while the flagship verdict is the critical path.
