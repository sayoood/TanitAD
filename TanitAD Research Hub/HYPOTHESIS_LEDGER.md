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

- 2026-07-06: **H1 strengthened** — (a) T-linear planning-regret bound (arXiv 2606.27014) gives the
  formal argument for horizon factorization via hierarchy; (b) HiT-JEPA (2507.00028) provides
  independent zero-shot-generalization evidence for hierarchical JEPA from urban computing.
  **H3 strengthened** — generalization theory formalizes the latent-vs-pixel trade-off (spectral
  tail vs sample complexity in latent dim k); data-efficiency claim gains a theoretical skeleton;
  `p0-spectral-sizing` experiment queued. **D-010 upgraded** — the max_a term in the regret bound
  is a formal argument for off-expert sim action coverage. Full analysis:
  `Architecture & Inference/Research/2026-07-06-jepa-generalization-theory-and-hit-jepa.md`.

- 2026-07-07: Data Eng — H7 gains LAOF (optical-flow-consistent latent actions) + Sensorimotor-WM
  (IDM-as-perception) support; comma2k19 real (steer,accel,pose) pairs validated on real bytes;
  steering-ratio calibration residual named as the H7 artifact. H4 unblocked by real corpus.
- 2026-07-05: Ledger created at kickoff from initial research synthesis.
