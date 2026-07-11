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
| H16 | Active depth interrogation (tactical-commanded, σ-triggered ROI depth queries) | open — Sayed's idea 2026-07-11; composes H15 σ-trigger + H2 scheduler + ZipDepth-class specialist; Phase-1 H2 window (~Sep); dossier: `Architecture & Inference/Research/H16_ACTIVE_DEPTH_INTERROGATION.md` | F1–F3 pre-registered (trigger selectivity, energy vs always-on, metric anchor) | Architecture & Inference |

## Gate definitions (Phase 0)

See `Project Steering/Phase 0 Plan.md` §4 for the full D1–D8 table with thresholds, instrument rows and ablations.

## Change log

- 2026-07-11: Data Engineering (Tue) — **H7** evidence (no status change, P8): **D-016 focal
  canonicalization validated end-to-end** on the trained encoder (measured, RTX 4060, $0) — using the
  correct per-camera intrinsics holds the SAME scene's encoding at cos ≥ 0.997 across a 25–100 % focal
  change, vs cos 0.60 / ~10–15× relative latent drift when intrinsics are ignored. First evidence the
  focal transform delivers focal-invariance (not just arithmetic); grounds the H7 heterogeneous-video
  flywheel (VLM3 principle) and sets the Y-pilot-50 Y1 focal-recovery acceptance bar (±25 %→cos≥.92).
  See `Data Engineering/Research/2026-07-11-focal-invariance-validation-and-sc13-sourcing.md`.
- 2026-07-24: Opponent Analyzer (Fri, run #2) — competitive-evidence deltas (no status *upgrade*;
  design-oracle numbers only, nothing measured on our stack, P8). **H9** gains a shipped scenario:
  **Stop-Arm Gate** intake pkg (W-03 → H9/H15; **11/11 offline tests**) with the first
  **violation-rate** contrast for the hard-barrier thesis — rule_barrier **0.0** vs soft_prior **1.0**
  over the free-path temptation sweep, and the barrier is *invariant* to temptation while the soft
  prior's line-crossing speed grows 3.0→9.6 m/s (the mechanistic barrier-vs-soft-prior difference).
  **H0/H6** reinforced (external): a **new emerging player, Avride** (Uber partner) is under NHTSA ODI
  investigation for **16 crashes** tied to lane-changing / same-lane following / stationary-object
  response; Waymo hit a **second recall + a new Dallas federal probe** (red-light running); Tesla's FSD
  probe is now an **Engineering Analysis over 3.2 M vehicles / 9 crashes / 1 fatality** naming a failed
  "degradation-detection" feature (W-04). **H1/H3/H5 wedge** sharpened by the deep-read of **Metis**
  (arXiv 2606.15869): it buys inference efficiency by letting the action head *skip generative rollout*
  (like our latent path) but is a **flat MoT, no hierarchy, no in-loop imagination, no self-monitoring,
  and reports no param count / compute-normalized metric** → our CNCE + hierarchy + H15 + H11 remain
  uncontested; "world model / efficient WAM" still not differentiating alone. See
  `Opponent Analyzer/Research/2026-07-24-opponent-sweep-w3.md` + updated `WEAKNESS_CATALOG.md` /
  `OPPONENT_PROFILES.md` / `SCENARIO_DATABASE.md`.
- 2026-07-09: Benchmarks & Eval (Thu) — **H15 instrument-hardening** (no status change, P8: measured on
  scripted policies + a noise model, not our checkpoint). The first live CARLA SC-01 run exposed **LAL-v1
  as blind to smooth anticipation** (both policies −0.7; the −1.5 m/s³ jerk trigger never fires on a
  comfort-bounded ease-off). Shipped **LAL-v2** (deceleration-onset by speed drop; the pre-line-of-sight
  generalization of TTB) → +0.3…+3.1 s anticipation lead vs −0.3 s reactive (7 analytic tests). Independent
  recompute confirmed **SC-01 LOPS 0.834** matches the analytic E=0.8325 of the injected σ=0.3 noise model
  (inside 95% CI, N=5000) → reproducible, but reactive's 0.0 is *structural* → proves latent-track presence
  not quality (honest bound on the H15 edge). OKRI ≥3-seed CI-separation rule made numeric (gap 19.5; ~2
  seeds at SD≈5, SD to be measured on the pod re-run). Ecosystem: DriveFuture 55.5 / DrivoR 56.3 EPDMS
  (navhard, Apr-2026) — another latent-WM board leader → "world model" still not differentiating (H0/H6),
  wedge stays hierarchy+efficiency+imagination+self-monitoring. See
  `Benchmarks & Eval/Research/2026-07-09-sc01-live-metric-audit-and-lal-v2.md`.
- 2026-07-09: Architecture & Inference (Wed) — **H5 K-step rollout, first measured arm** (no status
  change, P8: reduced-scale directional probe, D1 FAIL + D3 BLOCKED ⇒ no gate passed, D-004). Matched-compute
  K=2 vs K=1 (2×2000 steps, real comma2k19, 4060, OFAT-verified): rollout is **nearly free (+0.5 % wall-clock,
  0 params)**; the backlog falsifier metric (D2 dir-acc) **saturated at 1.0** so it can't discriminate; the
  discriminative signal `imag_rel` shows K=2 cuts 1-step latent-pred error vs persistence **2.914→1.049 (−64 %)**
  but does **not** help the 4-step horizon (I4 1.451→1.645) → **K must match the decode horizon** (K≈4 for the
  2-s D3 claim). Decision-grade = operative-scale K∈{1,2,4} sweep from pod2 step-8k. Also **H3 strengthened
  (external theory, no status change):** *When Does LeJEPA Learn a World Model?* (arXiv 2605.26379, LeCun/Klindt)
  proves LeJEPA/SIGReg gives **linear+orthogonal latent identifiability under a UNIQUE Gaussian prior →
  optimal latent-space planning** — grounds our SIGReg-only anti-collapse AND the linear transition proxy in
  `p0-spectral-sizing` (D-021). See `Architecture & Inference/Research/2026-07-09-kstep-rollout-bakeoff-and-lejepa-identifiability.md`.
- 2026-07-08: Architecture & Inference (Wed) — **H3 first trained-checkpoint measurement** (no status
  change, P8: mid-training + feeds D-021, not a passed gate): spectral-sizing on the step-6500 ckpt (fit
  R²=0.99, operator effective rank ≈43, energy knee 31, k*=21) → the action-conditioned transition operator
  is **genuinely low-rank** (~tens ≪ 2048) → **OVER-PROVISIONED** readout. Strengthens the H3 data-efficiency
  story with a real number, but rank is still climbing (35→43 over 3k→6.5k) so no resize is motivated;
  D-021 default holds. Artifact: `Architecture & Inference/Research/2026-07-08-spectral_step6500.json`.
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
