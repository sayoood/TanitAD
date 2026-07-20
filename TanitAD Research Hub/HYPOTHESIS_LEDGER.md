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
| H17 | Unified-FOV masked-periphery training (pad comma to 120°, imagine the unobserved edges) | open — Sayed 2026-07-12; inverts FOV unification, feeds H15 imagination signal + urban peripheral vision + foveated patching; dossier `Architecture & Inference/Research/UNIFIED_FOV_FOVEATED_PATCHING.md` | urban-ADE lift + imagination calibration without comma regress | Architecture & Inference |
| H18 | Hierarchical action grounding (metric-consequence grounding at operative + tactical + strategic levels) | open — Sayed 2026-07-12; operative grounding SHIPS now (§3.7 metric-invdyn + forward-metric-consistency, proprioceptive/self-supervised); tactical/strategic extension next | per-level: grounded consequence beats ungrounded selection at that horizon | Architecture & Inference |

## Gate definitions (Phase 0)

See `Project Steering/Phase 0 Plan.md` §4 for the full D1–D8 table with thresholds, instrument rows and ablations.

## Change log

- 2026-07-18: Fleet-review follow-up — external-survey derivation proposes H19 (discrete tactical vocabulary/LAMP), H20 (plan-persistence bridging/BridgeAD), H21 (latent GRPO-RFT/WorldRFT), H22 (shortcut-trained imagination/DreamerAD — pairs with the measured sigma-dissipation), H23 (cost-map decode head/PLAN-S), H24 (oracle-gap curriculum/ACID+our CTRV floor) as PROPOSED (not adopted; falsifiers + gates in Project Steering/Research/2026-07-17-external-survey-derivation.md). All six flagship-v2 levers externally validated (ColaVLA=goal-decode, PLAN-S=ego-to-planners).
- 2026-07-18: Architecture & Inference (Wed) — **E1+E2 re-run on the OPERATIVE flagship-speed @19k
  (drops the pre-reset step-6500 caveat; eval pod A40, PhysicalAI val, 2 seeds, $0). No status changes.**
  (1) **H15/H11/D8:** the σ-dissipation + attractor-collapse pathology **REPRODUCES on the operative model**
  — backlog P0.1 falsifier ("speed+jerk fixed it") **NOT met**. cos_rollout→chance by k3; σ_hidden
  −9.461→−9.564 (*lower* absolute σ than pre-reset = worse temporal calibration); attractor 0.219→**0.805**
  (sharper). **freeze-1 holds 0.213–0.232 flat across 8 horizons (7× persistence)** → the *cap operative H15
  self-monitor at 1-step / parallel-horizon* constraint is confirmed on the shipping model. Refinement: σ is
  *spatially* calibrated (hidden>visible +0.37; per-cell err↔var corr +0.29–0.43) but *temporally*
  anti-calibrated → design target narrows to a **horizon-aware** σ (multi-step-rollout training, D-018
  escalate). (2) **H3/D-021:** readout isotropy **converging as predicted** — iso_ratio_active **0.254→0.546**
  (crossed 0.5), cond_active 218→61, but still **NOT-YET-ADMISSIBLE** (rms_offdiag 0.32>0.1 → LeJEPA
  optimal-planning corollary still withheld). active_k≈19, cov_eff_rank≈30 ≪ 2048 → **latent capacity is not
  the D1 bottleneck (G1), reaffirmed on the operative model.** Decision-grade re-run at flagship @30k is
  turnkey. The 2026-07-10 orthogonality instrument is **still unmerged** (re-flagged). See
  `Architecture & Inference/Research/2026-07-18-operative-flagship-blind-rollout-and-orthogonality.md`.
- 2026-07-31 (real wall-clock 2026-07-17): Opponent Analyzer (Fri, run #3) — competitive-evidence deltas
  (no status *upgrade*; design-oracle numbers only, nothing measured on our stack, P8). **H15/A9** gain a
  shipped scenario: **Stationary-Lead** intake pkg (W-08 → H15/A9; **14/14 offline tests**) — the first
  **collision-rate + braking-onset-lead-time** contrast for the consequence-forward-model thesis: over the
  classification-ambiguity sweep {0…1}, collision rate **imagination 0.0 vs detection-reactive 0.4**,
  lead time **+1.20 s vs −1.26 s**, and the forward model is **invariant to ambiguity** (min-TTC 2.88 s)
  while the reactive policy degrades to a collision (drops the lead ≥ 0.75). **H0/H6/H11** reinforced
  (external, decisive): NHTSA's **2026-07-08 first-responder directive** demands every operator fix
  emergency-scene interference by end-July, with Administrator Morrison stating **"emergency scenes are
  not rare or extreme edge cases"** and calling the failure a **"functional insufficiency"** — the
  federal regulator now voices the scenario-database thesis verbatim (→ new **W-09**, **SC-06** elevated,
  maps H15/H11/A9/H9). Field-scan: **hierarchy surfacing** in academic AD-WM work (SGDrive 2601.05640) —
  H1 differentiator being explored, not yet with our combination. See
  `Opponent Analyzer/Research/2026-07-31-opponent-sweep-w4.md` + updated `WEAKNESS_CATALOG.md` /
  `OPPONENT_PROFILES.md` / `SCENARIO_DATABASE.md`.

- 2026-07-17: Architecture & Inference (Wed) — **two measured constraints, no status changes (P8,
  pre-reset directional step-6500 ckpt).** (1) **H15/H11/D8:** the 1-step-trained ImaginationField
  **dissipates epistemic σ and collapses to an attractor under blind autoregressive K-step rollout**
  (hidden-cell fidelity 0.357→0.011=chance by k4; σ log-var −7.79→−8.55 = *more* confident as it
  degrades; inter-sample cosine 0.21→0.57; belief energy −11× by k4) — the exact H11/D8 σ-trigger risk
  flagged 2026-07-15, now measured on real comma2k19, matching "Biased Dreams" (2604.25416). Cause is
  the *recursion*: freezing the k=1 imagination holds ~0.25 cosine flat across 8 horizons. **Constraint:
  cap the operative H15 self-monitor at 1-step / parallel-horizon until a multi-step-trained σ is
  validated** (D-018 escalate). (2) **H3/D-021:** found + **verified the stranded 2026-07-10
  orthogonality instrument** (unmerged 3 wks; withdrew my duplicate draft) — reproduces exactly on the
  step-6500 ckpt (n=2600>S): **iso_ratio_active 0.254 < 0.5, NOT-YET-ADMISSIBLE** → **D-021 = "identifies a
  low-dim subspace," NOT LeJEPA "optimal planning."** (Global isotropy ~0 is over-provisioning by design,
  not failure — the active-subspace read is the correct one.) Rules out latent capacity as a D1 bottleneck
  (G1); flagged the 07-10 instrument for orchestrator merge. See `Architecture & Inference/Research/2026-07-17-blind-rollout-uncertainty-dissipation-and-readout-orthogonality.md`.
- 2026-07-15: Data Eng (Tue) — **H7 data-availability delta (no status change, P8).** WorldModel-Synthetic-
  Scenarios (264k clips) **confirmed POSE-LESS on real bytes** → its pixels need an H7 inverse-dynamics head
  (Phase-1); its per-clip Qwen captions + weather/tod/region metadata are a **usable-now semantic-label index**.
  IDM/latent-action literature now dense (2601.05230 in-the-wild, 2602.16229 factored, LatentVLA, FLAM) →
  frozen-encoder IDM+WM on unlabelled video (our frozen-DINO REF-A lineage) is the standard recipe, with the
  comma/ZOD **real CAN** as the labelled bridge. H4 arm-B: **PandaSet** (CC-BY-4.0) loader shipped (intake, 16✓)
  but geometry-BLOCKED (front fx=1970 → f_eff 467≠266, D-016 R1 pad-crop+undistort needed — a blocking
  prerequisite for the whole owned real-urban tier incl. ZOD). See
  `Data Engineering/Research/2026-07-15-worldmodel-pose-gate-and-pandaset-geometry.md`.
- 2026-07-15: Architecture & Inference (Wed) — **H15 imagination edge verified LIVE-ACTIVE in the
  flagship (no status change, P8)**, resolving the 2026-07-14 program-report §8 `h15=0.0` WATCH. Measured
  on the exact code path (GPU): imagination module **built** (22.06 M params), gradient reaches it (L1
  44.6) + the encoder (L1 36.7), fire rate 0.45 ≈ `mask_prob`; `h15=0.0` is a **logging artifact** (last-
  micro sample of a stochastic gate — 46.3 % of rows false-zero while training). Edge is healthy; **the
  log was lying** → observability fix shipped (accum-window meter, intake 6✓), `h15_fire_frac→0` now the
  real dark-edge alarm. **No trained-config change** (D-018 restraint on a phantom). Theory-watch: UWM-JEPA
  (2605.25313, belief-space imagination) surfaces two H15 design gaps (1-step-only training; possible σ
  dissipation over rollout → H11/D8 trigger risk) as a new falsifiable backlog experiment.
- 2026-07-17: Benchmarks & Eval (Thu) — **validation-methodology hardening, no status change (P8).**
  Put the D1 denominator in **leaderboard-comparable units**: a no-vision ego-status shortcut (AD-MLP
  repro, arXiv 2312.03031) scores **avg open-loop L2 0.66 m** on comma-hwy (metric-BEV, held-out) — tied
  with CTRV → `skill_score = model_L2 ÷ 0.66 m`. **comma is 73.9 % straight = nuScenes' 73.9 %** → aggregate
  open-loop L2 is a **weak capability test**, reinforcing that **H3** decodability gates (D1–D3) are
  necessary-not-sufficient and the driving verdict needs per-stratum skill_score + **closed-loop** (D4–D6,
  arXiv 2605.00066) — no open-loop L2 alone arbitrates the single-camera driving claim. Nothing measured on
  our model (denominator only; local ckpt pre-reset camera-frame, not comparable). Intake
  `Benchmarks & Eval/Implementation/incoming/2026-07-17-openloop-l2-egostatus-shortcut/`.
- 2026-07-15: Benchmarks & Eval (Thu) — **instrument/denominator hardening, no status change (P8).**
  The **D1** decode-gate denominator is corrected: the driving diagnostic's "10–15× worse than
  constant-velocity" rests on a single-CV floor (≈0.28 m@1s), but a tested best-of-3 kinematic floor
  (CV / go-straight / **CTRV**, 26 132 anchors, comma-val + Cosmos-DD) is **≈0.056 m@1s** (CTRV wins
  55–58 %). → flagship held-out 6.44 m = **~115× the honest floor** (direction of the D1-FAIL verdict
  *reinforced*, not overturned; **H3** decodability gates D1–D3, **H13** probes). D1 should divide by
  the per-stratum best-of-3 floor (`skill_score`), and curvature strata must be speed-gated (v≥2 m/s;
  κ=yaw_rate/v singular at v→0). External support for the diagnostic's #1 remedy: **IDOL** inverse-
  dynamics-guided prediction (arXiv 2605.31476) → ego-motion grounding (**H1/H18**). Nothing measured
  on our model this run (denominator only). Intake `Benchmarks & Eval/Implementation/incoming/2026-07-15-baseline-floor/`.
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
- 2026-07-18: REF-C REDESIGN (Sayed-directed) — the tiny-TCP first build was wrong (2022 recipe, unimodal
  GRU reproduces our 3.38m tactical weakness, not on NAVSIM/Bench2Drive). NEW REF-C = **DiffusionDrive**
  (anchored truncated-diffusion decoder, 88.1 NAVSIM-v1 PDMS, 60M, ResNet-34, camera-capable; Hydra-MDP++
  86.6 camera-only) on REF-C's existing ResNet-34 encoder — a DECODER SWAP. Proven urban (NAVSIM/nuPlan)
  AND highway (VADv2 unseen-Town05 closed-loop; DiffRefiner Bench2Drive 87.1 DS). Multimodal-by-construction
  = the direct fix for unimodal-regression 3.38m. Our ideas graft at conditioning seams: strategic ctx->FiLM,
  **tactical maneuver-logits -> anchor priors (VALIDATES H19: our maneuver vocab == the anchor set,
  propose-and-refine)**, LAW aux, imagination encoder-side, metric-grounding replaces top-1 selection (=
  Hydra-MDP scoring on LOGGED data, no simulator). DATA FIX: REF-C trains on pod3 /workspace/pai_epcache
  (physicalai-train-51f40f5ebc21, 320-ep, SAME as REF-A) not comma — the comma run was a path error (missed
  pai_epcache). vs 1B-VLA (LinkVLA/AnchorVLA): rejected on feasibility (language data + 1B + edge-hostile);
  even they converge on anchor decoders, so take the component at 60M. Full design: agent transcript +
  Architecture backlog. Anchor vocab MUST be FPS over OUR trajectories (74%-straight skew kills k-means).
- 2026-07-18: **H4 CLOSES NEGATIVE (decision-grade, distill pre-gate on pod3, RED).** Frozen DINOv2-B/14
  features LACK generalizable ego-dynamics — a frozen-encoder + small-adapter design is capped below the
  trained-encoder ceiling and cannot solve the task from vision. Method: distill the flagship's trained
  encoder latents into a frozen-DINO adapter, probe speed/yaw R2 held-out. **CRITICAL protocol finding:
  the verdict flips on the split.** FRAME-split (leaky, adjacent 100ms frames near-identical) gives raw-DINO
  speed R2 0.986 = GREEN artifact; the HONEST held-out-EPISODE split gives speed 0.56 / yaw 0.37 = RED.
  Distill converged on train (cos 0.9998) but FAILED to generalize (held cos 0.60) and never beat a plain
  linear read of raw DINO (0.559 <= 0.596) -> the ADAPTER is not the bottleneck, the FEATURES are. Real
  deficit = YAW/rotation: frozen 0.37 vs flagship-latent 0.89 (only 42% recovered; speed 0.60 vs 0.67 is
  ~89%, and v0 is injected as an action channel anyway). End-to-end training shaped the flagship encoder to
  expose yaw geometrically; frozen DINO entangles it with appearance. **RECOMMENDATION: do NOT spend
  refa-distaux-30k; LoRA-unfreeze the last-k DINO blocks instead** (low-rank grads reshape features to
  expose yaw — the mechanism behind the flagship's 0.89). This is the H4' reframe: minimal adaptation, not
  frozen-purity. Directly answers Sayed's "frozen-encoder solving the task from vision": it doesn't; the
  path is LoRA. Validates the leakage-guard theme (episode-split matters). Artifacts pod3
  /workspace/tmp/refa_distgate/. See [[refa-bottleneck-diagnosis]] [[speed-input-fix-validated]].
- 2026-07-18: **H4 REFRAMED — the RED verdict was the EXPECTED result, not a death knell (deep DINO-in-AD
  analysis, cited).** No proven frozen-DINO system expects the encoder to carry ego-DYNAMICS. The entire
  DINO-world-model lineage (DINO-WM ICML25, DINO-world/Back-to-the-Features Meta FAIR, DINO-Foresight, LAW
  ICLR25, World4Drive ICCV25, GAIA-1/2, Vista) uses frozen features for SCENE/GEOMETRY ONLY and feeds
  proprioception/actions (speed, yaw-rate, curvature, waypoints) SEPARATELY as first-class conditioning,
  with dynamics living in the ACTION-CONDITIONED TRANSITION model (predict future features). Our speed-vs-yaw
  asymmetry (0.56 vs 0.37) is CONFIRMING: yaw is a pure between-frame rotation with no single-frame
  appearance signature — DINO-VO (RA-L25) had to BOLT ON geometric-matcher+pose-head to get ego-motion from
  frozen DINO; a per-frame semantic encoder neither does nor should encode yaw. **=> LoRA-unfreeze is the
  WRONG axis** (it improves SCENE features +8-9 IoU on BEV-seg but CANNOT manufacture a cross-frame quantity
  from a single-frame encoder, and adds overfit risk on 2376 eps — misdiagnoses a temporal gap as a
  feature-adaptation gap). **CORRECTED REF-A PLAN (supersedes the LoRA rec of the prior entry):** #1 keep
  DINO FROZEN, feed YAW-RATE/steering as an input channel (the EXACT fix that took speed R2 0.61->0.965 via
  v0-as-action-channel — extend to rotation; we have yaw-rate in the log) — the yaw deficit becomes
  irrelevant, not something to fix; #2 condition the world-model transition on ego-action (our 4-brain
  already is this shape); #3 cost-volume/flow neck over consecutive frozen patch-grids ONLY if we want
  vision to DERIVE yaw (DINO-VO/DINO-Tracker pattern) — reserve for vision-only inference/aux; #4 DEMOTE LoRA
  (r=4-8, last-2-4 blocks) to a SCENE-polish applied after #1/#2, NOT the yaw fix. Honesty: DINOv2 is proven
  in AD PERCEPTION (BEV-seg, depth, VO) but NOT dominant on planning leaderboards (ResNet-34 wins NAVSIM).
  H4 is NOT negative — frozen-DINO was MIS-WIRED, not unfit. New arm = refa-dynin (frozen + yaw-rate input),
  on 2376. See [[refa-bottleneck-diagnosis]] [[speed-input-fix-validated]].
- 2026-07-18: **H25 PROPOSED (flagship-v3 vision-reliance) + H26 PROPOSED (hierarchical cross-alignment,
  the CORE-GOAL proof).** Both ADDITIVE to the preserved LeJEPA core (E2E ViT encoder + action-conditioned
  OperativePredictor + SIGReg rank-stabilization — verified intact through v2, fourbrain.py:195/208/242,
  flagship_losses total lines 287-299).
  **H25 — Vision-reliance can be raised by DECOUPLING dynamics from the trained encoder.** Measured problem:
  imagination panel vision_use only 12.9% — the flagship leans on fed dynamics (v0/yr0) and its trained
  encoder REDUNDANTLY re-encodes them (yaw R2 0.89 in-latent) instead of spending capacity on scene. The
  DINO-WM lineage keeps the encoder purely perceptual and puts dynamics in the transition; the trained-encoder
  analog = discourage encoder-dynamics, free capacity for scene. EVIDENCE-BASED MEASURES (ranked): (1)
  future-action dropout — ALREADY in v2, VALIDATED as the mechanism (imagination panel D vs E gap); v3 = tune
  p / schedule it; (2) encoder-ego DECORRELATION aux — penalize mutual-info/linear-predictability between the
  encoder latent and the fed [v0,yr0], so the encoder stops re-encoding dynamics ("Is Ego Status All You Need"
  CVPR24 shows decoupling ego-state helps; DINO-WM keeps encoder perceptual); (3) scene-reconstruction aux head
  (depth/seg from the encoder) — force scene content (DINOv2+Metric3Dv2 division-of-labour, +8.9 IoU); (4)
  vision-ablation-CONSISTENCY penalty — penalize when mean-replaced vision yields a similar rollout (turn our
  imagination-panel ablation into a training signal). Falsifier: vision_use fails to rise >15% with the measure
  at flat ADE. Gate: vision_use >20% AND imagination >12% AND ade@2s non-regress vs v2. Cheapest+strongest =
  (1)+(2). NOT a v2 change (v2 locked); v3 candidate, measurement-gated.
  **H26 — Hierarchical cross-layer alignment (strategic->tactical->operative) measurably beats ungrounded
  selection at each horizon — the CORE-GOAL proof that hierarchical E2E world-understanding+planning is
  efficient & dominant (Sayed 2026-07-18).** Extends H1/H18. Design a cross-alignment ABLATION study in
  TanitEval: (a) per-layer conditioning ablation — does strategic ctx improve tactical maneuver/waypoint acc?
  does tactical intent improve operative rollout ADE? (FiLM-off vs on at each seam); (b) cross-layer agreement
  metrics — does the strategic route match the tactical maneuver match the operative trajectory (consistency)?
  (c) grounded-vs-ungrounded selection at each level (H18). Falsifier: ablating a layer's conditioning does NOT
  hurt the layer below (=> hierarchy is decorative). Gate: each conditioned layer > its unconditioned control,
  CI-separated, on the canonical val. This becomes a standing TanitEval "hierarchy panel". See
  [[whole-program-briefing-format]].
- 2026-07-18: **H26 FIRST VERDICT (flagship-speed @19k, TanitEval hierarchy panel, 881 windows/40 eps) —
  MIXED, honest.** Part B (mutual CONSISTENCY) SUPPORTED: maneuver<->trajectory agree 0.872, kappa 0.612;
  route<->maneuver "disagreement" is expected cross-timescale (15-25s route vs 2s maneuver), not incoherence.
  Part C (GROUNDING dominant, H18) SUPPORTED: grounded rollout 0.615m beats the ungrounded tactical head 3.43m (5x).
  Part A (top-down CONDITIONING helps downstream) FALSIFIED @19k: nav->strategic only echoes the command (route
  follow-acc 0.671 == majority-straight => ZERO vision route inference); ctx->tactical goal-cos separates but is
  negligible (+0.0045); intent->operative is HARMFUL when engaged (latent-cos 0.731 real vs 0.975 none) because
  `intent_proj` norm 31.4 SWAMPS the action-emb 28.3 in the FiLM cond -> the deployed rollout is intent-free BY
  DESIGN (that IS the 0.615m). READ: at 19k the 0.6m comes from the OPERATIVE predictor + grounded step-readout,
  NOT the top-down cascade; the strategic/tactical brains COHERE but do not yet DRIVE the operative.
  ACTIONABLE (flagship-v3 lever): normalize/rescale the intent->operative FiLM cond (LayerNorm the cond, or match
  `intent_proj` output norm to the action-emb ~28) so the seam becomes LOAD-BEARING instead of corrupting — this is
  the concrete path to the "hierarchy is dominant" core-goal proof. Panel is data-driven + AUTO-re-runs at 30k
  (thesis_read flips if the seams converge). nospeed@22k replicates (1/3 seams, kappa 0.583; grounded barely wins
  b/c the no-speed operative is 3.0m not 0.6m — consistent with [[speed-input-fix-validated]]). Deployed:
  taniteval/hierarchy.py + runner `hierarchy` subcmd + report section "02b". This same B1-class mis-scaling lesson
  (conditioning term must not swamp the base signal) informs the REF-B refbpatch-v2 anchored-decoder wiring.
- 2026-07-18: **H26 REFINEMENT — the intent->operative "help" is TWO reads, and per-window CONTENT is inert on
  BOTH arms (panel preview: refa-dinov2 30k FROZEN enc vs flagship 19k TRAINED enc).** The panel now separates
  `helps_vs_none` (engage the intent term vs zero it) from `per_window_content_helps` (real intent vs a
  mean/constant intent). (a) helps_vs_none FLIPS by encoder: flagship (trained) intent HURTS (cos 0.731 vs 0.975 —
  swamps), but frozen refa-dinov2 intent HELPS (0.936 vs 0.852) because the frozen encoder can't decode ego-motion
  so the operative CO-ADAPTS to a large constant intent offset (intent_proj norm **1792** vs act_emb 14.5). (b)
  per_window_content_helps ~= 0 on BOTH — the tactical brain's actual per-window decision adds ~nothing over a
  constant intent. IMPLICATION: the hierarchy's top-down CONTENT is not steering the operative in either regime
  (at most a constant offset). This is DEEPER than magnitude-swamping: the gated-intent v2 lever correctly fixes
  the swamp (and lets a fresh run avoid co-adapting to a content-inert offset) but does NOT by itself make
  per-window content matter. v3 OPEN QUESTION: why is per-window intent inert, and how to make the operative
  attend to it (intent-content-usage loss? non-additive injection? tactical-output quality?) — this is the real
  "hierarchy is dominant" core-goal lever. Frozen confound CONFIRMED: refa's intent behaves OPPOSITELY to
  flagship's, so the refa H26 read must use per_window_content (NOT helps_vs_none) as the honest metric. Panel now
  effect-size-gated (CI-sep AND >=0.02 acc / 0.05 m / 0.01 cos). refa dynin 5k read armed, ~15-27h out (rate
  ~16-25 s/step, slower than est). See [[refa-bottleneck-diagnosis]] [[speed-input-fix-validated]].
- 2026-07-18: **FLAGSHIP v1 FINAL (30k) — FIRST SUB-FLOOR ARM + H26 30k re-run PARTIAL FLIP (the armed
  follow-up).** TanitEval full suite on the true step-29999 ckpt (n=881/40 eps): ade_0_2s **0.4522+-0.031**
  (plain-mean 0.4271) — BELOW every trivial bar for the first time (best-of-3 floor 0.5005, CTRV 0.523, ridge
  ego-status ceiling 0.5735). vs 19k: delta 0.188m, win 81.2%, better on EVERY curvature stratum; miss@2m
  0.180->0.060 (3x); skill-vs-floor straights 1.488->1.032 (at-floor), gentle 0.679, sharp 0.599 (turns now
  BEAT floor); HIGH-SPEED remains the open weakness (1.785, only stratum above floor). H26 @30k: seams
  0/3 -> **1/3 — ctx->tactical FLIPPED to LOAD-BEARING with content_matters=true** (vs-mean maneuver
  delta +0.044 CI-sep; per-window content NO LONGER inert on this seam — training duration alone moved it).
  intent->operative STILL harmful (cos vs-none -0.238 ~= 19k) => the v2 gated-intent lever remains exactly
  right. nav->strategic still pure command-echo (route_skill_vs_chance 0.0 — route-from-vision is an open
  v3 target). H18 grounding dominance GREW (delta 2.70m). Imagination: vision_use ~12% flat (H25 still open,
  v3). REF-B lineage note: pod1 ckpt_prepatch_step8500.pt is BYTE-IDENTICAL (md5) to refb-speed-30k/ckpt.pt
  — one ckpt at step 10000, misnamed; benchmarked as refb-10k: 0.8255 (improving turns +0.255m vs 6k,
  slightly regressing straights). v1 pushed to HF (Sayood/tanitad-flagship-4b-speedjerk, gated-manual).
- 2026-07-18: **GENUINE-PREDICTION PROOF (flagship-30k, TanitEval generalization panel, 881 windows/40 eps) —
  the model genuinely predicts scene physics where it's REQUIRED, proven CAUSALLY.** Test B (vision-ablation on
  CTRV-divergence strata) is the headline: on high-divergence windows (upcoming turns/brakes CTRV cannot
  extrapolate) the model beats the CTRV oracle by **+0.796m on 94%** of them, and that ENTIRE advantage is
  VISION — mean-replacing the scene INVERTS it to -0.529m (worse than CTRV); **vision-effect +1.325m, CI
  [+1.04,+1.64]** (separated). So the anticipation is READ FROM THE SCENE, not extrapolated from dynamics.
  Support: (A) advantage over CTRV is MONOTONE-increasing with divergence (Q1 -0.372 -> Q4 +0.796) — beats the
  oracle most exactly where dynamics fails; (D) upcoming road curvature linearly decodable from the pooled
  latent R2 0.254 vs ego-kinematics-only 0.031 (**+0.223** = latent encodes SCENE not just ego-state); (E)
  predicted paths physically-shaped 95.9% vs GT 97.1% (learned physics, not memorized); (F-proxy) occluding
  road-ahead pixels shifts the prediction 1.60x more than periphery, dynamics fixed (reads_road_ahead). HONEST
  LIMITS: dynamics-guesses on low-divergence/near-inertial windows (correct — CTRV IS optimal there); LOSES to
  CTRV on top-10% high-speed (-0.617m, the open weakness); lead-time test inconclusive (instrument limit under
  action-grounded rollout). **CROSS-CORPUS OOD IS DATA-BLOCKED**: comma2k19 + Cosmos pixels are NOT staged on
  the eval pod, so "unseen-corpus" generalization + the real Cosmos counterfactual are UNRUN (tooling exists;
  needs data staging or own-dataset). Panel deployed (taniteval/generalization.py + 'generalize'/'gen-all' +
  dashboard 02d); RE-RUNS on v2 -> the vision-levers + loss-rebalance should DEEPEN the vision-effect = the
  before/after world-class arc. See [[speed-input-fix-validated]].
- 2026-07-19: **OOD GENERALIZATION (flagship-30k on UNSEEN corpora — comma2k19 + Cosmos NOW STAGED) — the
  in-dist proof does NOT transfer to beating the floor OOD, but the model STILL reads the OOD scene.** Model
  ADE@2s: physicalai (in-dist, n=881) **0.427** (BEATS CTRV floor 0.523, 49.7%); comma2k19 (OOD real hwy,
  n=2176) **0.849** (LOSES to floor 0.372, beats 17.5%); cosmos (OOD synth, n=92) **0.583** (loses to 0.358,
  29.4%). WHY: comma/cosmos are highway-dominated so the CTRV floor is very strong (~0.37m); model error
  ~DOUBLES OOD, high-divergence anticipation advantage collapses 0.80m->0.15m. HONEST NUANCE: vision-ablation
  STILL hurts on comma (0.27m, CI-sep) -> the model reads the OOD scene, it just can't net-beat the strong
  highway floor. Path feasibility drops to **62.8%** on OOD sharp-curvature (vs 97.8% in-dist). READ: the
  in-dist genuine-prediction is real but the model is PARTLY distribution-fit — NOT yet a corpus-general world
  model. Exactly what v2's vision-reliance + rebalance target; the panel now RE-RUNS on comma/cosmos to measure
  v2's OOD lift (baseline set). Cosmos pixels DOWNLOADABLE (pre-rendered 43GB shards); 24 clear/degraded weather
  PAIRS staged -> a TRUE weather-counterfactual is now a modest panel addition. See [[flagship-longitudinal-lever]].
