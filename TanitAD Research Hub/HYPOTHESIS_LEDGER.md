# TanitAD — Hypothesis Ledger

> Living status tracker for H0–H15. Updated by the weekly agents and after every gate result.
> Statuses: `confirmed` | `validated-toy` (proven on toy/in-house scale) | `supported` (external evidence)
> | `open` | `at-risk` | `refuted`. Full analysis: `INITIAL_RESEARCH_SYNTHESIS.md`.

| H | Short name | Status (2026-07-05) | Phase-0 gate(s) | Owner agent |
|---|---|---|---|---|
| H0 | E2E superiority settled | confirmed | — | Opponent Analyzer |
| H1 | 4B architecture | validated-toy (5× hierarchy lift, ALPS-4B) | D5, D6 | Architecture & Inference |
| H2 | Attention-based Modality Steering | supported (DriveMoE/GEMINUS) — race is on | Phase-0 exit demo | Architecture & Inference |
| H3 | Latent world model core (LeJEPA/SigReg) | validated-toy (A1–A10) | D1, D2, D3 | Architecture & Inference |
| H4 | Frozen encoders comparison | open (cheap to answer) — real corpus (comma2k19) ready to make it meaningful | arm B at D1–D3 | Data Engineering |
| H5 | Efficient inference transfer | supported | CNCE tracked from day 1 | Architecture & Inference |
| H6 | Opponent weak-spot corpus | actionable | 3 scenarios in eval set | Opponent Analyzer |
| H7 | 1000× data via IDM + focal canonicalization | supported (VLM3, LAPA; +LAOF flow-consistency, Sensorimotor-WM 2026) — real (steer,accel) pairs from comma2k19 in hand | IDM calibration logged (steering-ratio residual = named artifact) | Data Engineering |
| H8 | MoE beyond sensors | prio-2, interface ready | — | Architecture & Inference |
| H9 | Inherent rule compliance (RMFM/barriers) | supported, concrete math | violation-rate metric | Benchmarks & Eval |
| H10 | Latent RAG continual learning | validated-toy w/ known failure mode (−24 % interference → surprise-gating) | D7 (end of P0/P1) | Architecture & Inference |
| H11 | Self-monitoring w/ guarantees | validated-toy (3 mechanisms) | D8 (AUROC > 0.85) | Benchmarks & Eval |
| H12 | Text as part, not core | supported, cheap path (1B LLM bridge) | command-conditioning only | Architecture & Inference |
| H13 | Extraction heads | settled pattern (probes = minimal form) | trajectory + BEV probes ship | Architecture & Inference |
| H14 | Physical grounding | Track 1 adopted (kinematic decoder + Kamm circle) | friction-violation metric | Architecture & Inference |
| H15 | Unobserved-area imagination | **in Phase 0 per D-008** — ImaginationField implemented (advection + refine + epistemic σ), trains with the world model | **D9** (hidden-sector cosine, calibration gap, LOPS) | Architecture & Inference |

## Gate definitions (Phase 0)

See `Project Steering/Phase 0 Plan.md` §4 for the full D1–D8 table with thresholds, instrument rows and ablations.

## Change log

- 2026-07-08: Architecture & Inference (Wed) — decoding-lever evidence (no status change, P8:
  corroboration is not confirmation; the D-gates are unpassed until the trained checkpoint). **H4/A4**
  external support: **Delta-JEPA** (arXiv 2606.31232) reconstructs the executed action from the *latent
  displacement between consecutive observations* — independently our residual (A4) + inverse-dynamics (A5)
  design. **H5** the multistep-rollout benefit gains a second data point (Pareto ≈ K=4, alongside
  2512.24497's 6-step real). **H1/H2** conditioning upgrades (AdaLN>FiLM, +RoPE) now triangulated across
  three sources (2512.24497, Delta-JEPA, OmniDreams 2606.03159). **H3** theory-watch live (Balestriero &
  LeCun spectral-SSL, IEEE SPMag 43(3) 2026). All turned into *falsifiable, gated* levers in the new
  **bake-off harness** (WP3, backlog #2) — AdaLN/RoPE/K-step enter as `planned` levers (gate + hypothesis +
  WP pointer), runnable only after the model code lands AND the checkpoint makes D1/D3 admissible (D-004).
  See `Architecture & Inference/Research/2026-07-08-bakeoff-harness-and-conditioning-levers.md`.
- 2026-07-07: **ALPS-4B v1.1 adopted (D-017).** H3 strengthened: A11 validates consequence-dominance
  in the egocentric observation model our pipeline uses (control 0.19→0.69/0.76 from the observation
  change alone). A13 demotes imag-rel to a diagnostic — D2 redefined (calibrated OR P4
  forward-dynamics acc > 0.7). H11 gains P4 as a cheap redundancy control channel. H15/H13
  object-centric branch scheduled to Phase 1 by the A12 binding laws. New doctrine: I7 task-identity
  fingerprints (implemented; corpora proven identical), I8 batch-1 memory profiling. Delta analysis:
  `Architecture & Inference/Research/2026-07-07-alps4b-v11-findings.md`.
- 2026-07-17: Opponent Analyzer (Fri, run #1) — competitive-evidence deltas (no status *upgrade*;
  nothing measured on our stack, P8). **H0** reinforced (external): mid-2026 competitor failures land
  squarely on the E2E weaknesses we target — Waymo's **3,871-vehicle construction-zone recall** (W-01),
  NTSB school-zone VRU-anticipation + school-bus stop-arm probes (W-02/W-03), Tesla's open NHTSA
  **degraded-visibility** case (W-04); the field's convergence on latent WMs (arXiv 2603.09086 cluster;
  Wayve GAIA-3 15 B; NVIDIA Alpamayo-2 32 B) means our moat is now hierarchy+efficiency+imagination+
  self-monitoring, not "world model" alone. **H6** stays *actionable* and now has a dated evidence corpus
  + a shipped scenario: **Work-Zone Phantom** intake pkg (W-01 → H15/H9/H1; 9/9 offline tests). Evidence
  strengthens the *need* for **H15** (LOPS/OKRI/LAL, D9), **H11** (degraded-visibility D8 stressor
  recommended), **H9** (closure/stop-arm compliance), and the **H1/H3/H5** efficiency wedge (CNCE vs 32 B/
  15 B competitors). See `Opponent Analyzer/Research/2026-07-17-opponent-sweep-w2.md` +
  `WEAKNESS_CATALOG.md` / `OPPONENT_PROFILES.md`.
- 2026-07-16: Benchmarks & Eval (Thu) — evidence-of-need deltas (no status *upgrade*; nothing measured
  on our stack yet, P8). **Custom metric suite implemented** (LAL/TMS/OKRI/CNCE/LOPS + trajectory seam,
  22 analytic-ground-truth tests) → the empirical instruments behind **H15** (LOPS/OKRI, D9 hidden-sector),
  **H5** (CNCE efficiency moat), **H11** (D8 monitoring context), **H9** (violation-rate home). Motivating
  external evidence: arXiv 2605.00066 shows **open-loop ADE/FDE ⊥ closed-loop DS** (ranking inversions) →
  reinforces that D1–D3 are necessary-not-sufficient and the custom, closed-loop-native metrics are what
  actually prove the edge; NAVSIM-v2 EPDMS criticized as thin on safety-critical occlusion (= our
  OKRI/LOPS niche). Validation rule adopted: closed-loop gate claims report mean±CI over ≥3 seeds
  (CARLA ~5 DS seed variance). See
  `Benchmarks & Eval/Research/2026-07-16-benchmark-ecosystem-and-metric-suite.md`.
- 2026-07-14: Architecture & Inference (Wed) — external evidence deltas (no status *upgrade*; none
  measured on our stack yet, P8). **H3** +LeWM (2-loss stable action-conditioned JEPA, no EMA/stop-grad)
  reinforces SIGReg-only anti-collapse; **L2 `p0-spectral-sizing` tool built** (fits the transition
  operator, reports the spectral knee vs the 2048 readout) — the empirical instrument for the H3
  latent-dim skeleton, awaiting a trained checkpoint. **H1** +V-JEPA-2-AC (300 M block-causal AC WM =
  our 261 M envelope); arXiv 2512.24497 validates ViT-L enc + depth-12 pred and AdaLN (=our FiLM)
  conditioning. **H5** +MTP draft-heads / K-step rollout-loss bake-off lever; revisable-diffusion
  planners = Phase-1 comparison only. **H4** DINO > V-JEPA encoders for planning (2512.24497) — supporting
  data point for frozen arm B. **H2** differentiator sharpened: route the tactical/sensor MoE on
  ImaginationField epistemic σ (vs DriveMoE/GEMINUS learned scene routers). **Cross-cutting (D1–D3):**
  2512.24497 shows decode ≠ planning → D1–D3 are instrument gates (necessary-not-sufficient), D4–D6
  arbitrate; encoded in the new gate runner. See
  `Architecture & Inference/Research/2026-07-14-gate-runner-and-jepa-wm-deltas.md`.
- 2026-07-06: **H1 strengthened** — (a) T-linear planning-regret bound (arXiv 2606.27014) gives the
  formal argument for horizon factorization via hierarchy; (b) HiT-JEPA (2507.00028) provides
  independent zero-shot-generalization evidence for hierarchical JEPA from urban computing.
  **H3 strengthened** — generalization theory formalizes the latent-vs-pixel trade-off (spectral
  tail vs sample complexity in latent dim k); data-efficiency claim gains a theoretical skeleton;
  `p0-spectral-sizing` experiment queued. **D-010 upgraded** — the max_a term in the regret bound
  is a formal argument for off-expert sim action coverage. Full analysis:
  `Architecture & Inference/Research/2026-07-06-jepa-generalization-theory-and-hit-jepa.md`.

- 2026-07-14: Data Eng — H7 external-support delta (no status change, P8): LAWM (latent actions from
  unlabeled video via world modeling), Drive-JEPA (V-JEPA latent WM for E2E driving), HiLAM (hierarchy×
  latent-action), CLAW/DeFI (label-free forward/inverse dynamics). H4 arm-B: **Cosmos-Drive-Dreams**
  (CC-BY-4.0) loader shipped → first publicly-claimable *rich* corpus for the frozen-vs-trained encoder
  comparison; Zenseact ZOD flagged as real-CAN #2. Binding H7 evidence (IDM steering-ratio residual)
  still pending on real Chunk_1. See `Data Engineering/Research/2026-07-14-cosmos-drive-dreams-loader-and-landscape.md`.
- 2026-07-07: Data Eng — H7 gains LAOF (optical-flow-consistent latent actions) + Sensorimotor-WM
  (IDM-as-perception) support; comma2k19 real (steer,accel,pose) pairs validated on real bytes;
  steering-ratio calibration residual named as the H7 artifact. H4 unblocked by real corpus.
- 2026-07-05: Ledger created at kickoff from initial research synthesis.
