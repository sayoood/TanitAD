# TanitAD — Initial Deep Research Synthesis (Kickoff, 2026-07-05)

**Scope.** Research baseline for all hypotheses H0–H15 of the Mission Plan, combining:
(a) evidence from our own four experiment repos (ALPS-4B, 4B-HRM, ACRE, RSRA-4B),
(b) the 14 Gemini Deep Think analyses in `Ressources/Deep Think Analysis/`,
(c) fresh literature/market research (July 2026),
(d) mathematically solid extensions and concrete recommendations per hypothesis.

**How to read.** Each hypothesis gets: *Status* (validated / supported / open / risky), *Evidence*,
*Extension* (the mathematically solid approach we adopt), *Recommendation for Phase 0*, *Open questions*.
The decision consequences are consolidated in `Project Steering/Phase 0 Plan.md`.

---

## 0. The one-paragraph strategic picture (July 2026)

The industry has converged on end-to-end + world models (H0 confirmed): Wayve (raised $2.8 B mid-2026,
London robotaxi trials with Uber, GAIA-2 for synthetic data), Waymo ($126 B valuation after a $16 B
round, the only Western robotaxi at scale), Momenta (HK IPO filing, ~$14 B target, Uber Munich pilot),
Pony.ai (~720 robotaxis in China/Middle East), NVIDIA's Alpamayo open ecosystem (10 B-param reasoning
VLA + AlpaSim + AlpaGym + 1 727 h PhysicalAI-AV open dataset) commoditizing the baseline stack. The June
2026 UN WP.29 ADS regulation (UNR + GTR) is adopted — safety management systems, credible testing,
in-service monitoring/reporting (ISMR), DSSAD data recording. **Nobody occupies the Pareto point we
target: hierarchical latent world model, ~10–100 M params, orders of magnitude less data, no perception
labels, real-time on Orin-class compute, with built-in self-monitoring designed for the new regulation.**
That point is reachable because its components are individually validated — by our own repos and by the
2024–26 literature (LAW, World4Drive, V-JEPA-2-AC, Drive-JEPA, LeJEPA, ResWorld, IDOL).

---

## H0 — E2E superiority is settled; we fix its weaknesses. **Status: confirmed (external).**

All serious competitors are E2E or E2E-hybrid. The remaining weaknesses of E2E — opacity, no
self-knowledge, data hunger, rule compliance — are exactly the hypotheses H1, H9, H10, H11, H13.
Nothing to prove here; the moat is in the fixes.

## H1 — The 4B architecture (strategic / tactical / operative / fallback). **Status: core edge, partially validated in-house.**

**Evidence (in-house).**
- ALPS-4B measured a **~5× cross-room success lift** (0.07 → 0.36, oracle 0.57) from latent-graph
  strategic routing over an operative-only controller; decode error 0.19 wu after the harness fixes;
  action sensitivity 18.6; directional consistency 1.0 (`results/two_rooms/validation/RESULTS_SUMMARY.md`).
- 4B-HRM measured **5.9× FLOP reduction and 55× KV-cache compression** from hierarchical abstraction
  with entropy-gated escalation (`docs/BENCHMARK_RESULTS.md`).
- RSRA-4B: checker networks + Banach-contraction refinement give **provable convergence** of iterative
  refinement and 15–18× accuracy lifts on path-tracing tasks; the "over-refinement" failure mode and its
  fix (dynamic refinement depth) are documented.

**Evidence (external).** Michon's strategic/tactical/operational hierarchy is the canonical model of the
human driving task; WorldRFT (AAAI 2026) shows hierarchical decomposition inside a latent world model
helps planning; ReAL-AD and HANSOME explore the same direction. None combines hierarchy with from-scratch
SSL and calibrated imagination readout — that combination is our claim.

**Extension (adopted math).**
1. Layer contracts as *goal-conditioned constraint interfaces*: tactical emits (sub-goal latent,
   constraint set); operative treats them as soft constraints with an **entropy-gated veto** (Deep Think 4):
   when operative predictive entropy H_op ≥ τ_safe, tactical constraints are down-weighted and the
   fallback path is armed. This resolves the layer-conflict critique in a principled, measurable way.
2. Cross-layer consistency as an energy: E_total = αE_str + βE_tac + γE_op with stop-gradient isolation
   between layers and phase-shifted training (validated in ALPS-4B).
3. Frequency separation (thinking fast/slow): operative 10–20 Hz, tactical 1–2 Hz, strategic 0.1–0.5 Hz,
   fallback monitor at highest frequency, cheap by design (RSRA checker, not a second big net).

**Phase 0 recommendation.** Implement operative fully + tactical as maneuver-vocabulary
imagine-and-select; strategic as latent transition graph (port from ALPS-4B); fallback as monitor +
deterministic MRC hook. Gate: hierarchy-vs-flat at matched params (gate D5/D6 below).

**Open questions.** Exact task split tactical/strategic at driving timescales; conflict statistics
(how often does operative veto fire); end2end joint training schedule vs phase-shifted.

## H2 — Attention-based Modality Steering (ABMS). **Status: strong direction, externally emerging — we must move fast.**

**Evidence.** DriveMoE (2025) already selects camera views dynamically; GEMINUS routes planning experts
scene-adaptively; sensor-subset expert decoders improve robustness under sensor failure. Deep Think 3
contributes the engineering design: always-on sentry (radar + low-res front cam) feeding a
Gumbel-Sigmoid router, omniscient-teacher distillation to prevent "blind sentry" collapse, Schmitt
hysteresis for temporal stability, TensorRT dynamic shapes.

**Extension.** Router objective L = L_task + λ Σᵢ Cᵢ·pᵢ + β L_distill(teacher=all-sensors) with
per-modality compute costs Cᵢ; hysteresis via two thresholds; **safety invariant:** the fallback channel
(radar + front camera) is never gated. H15 (unobserved-area imagination) is the enabler: the tactical
layer may only power down a modality when the world model's predicted information gain from it is low
AND imagination uncertainty in its field of view is below threshold — this gives ABMS a principled,
world-model-native trigger instead of a heuristic.

**Phase 0 recommendation.** Single front camera first (per Mission Plan); implement the router interface
and prove the *mechanism* on MetaDrive multi-view / PhysicalAI-AV clips as a Phase-0-exit experiment:
tactical picks {front-only, +side, +rear} and we measure quality-vs-FLOPs Pareto. Full ABMS is Phase 1.

**Open questions.** Distillation budget; router training stability at low data; interaction with I2
batch-consistency (dynamic graphs must stay deterministic per mode).

## H3 — Latent world model (LeCun/JEPA direction) as the core. **Status: validated in-house at toy scale; the field's strongest open direction.**

**Evidence (in-house, ALPS-4B, all measured).** Pure-SSL SIGReg-only training without EMA/stop-grad
works (no collapse, effective rank healthy); imagine-and-select control works (direction acc 0.97,
position error 0.19 wu); **calibrated-decode doctrine** (fit probes on the predictor's own imagined
latents, not real-frame encodings: 0.97 vs 0.66); residual/delta prediction + change-weighted loss beat
plain MSE (0.97 vs 0.71) and flow sampling (0.44); inverse dynamics grounds the controllable state;
spatial grid readout ≫ global pooling; imagination error is a free OOD monitor; consequence-dominance
law (A8) tells us *why* egocentric driving is the ideal regime for this method.

**Evidence (external).** LeJEPA (Balestriero & LeCun, arXiv 2511.08544) provides the theory: isotropic
Gaussian is the optimal embedding distribution; SIGReg (Epps–Pulley statistic over random 1-D slices,
bounded gradients, single hyperparameter) achieves it. LAW (ICLR 2025), World4Drive (ICCV 2025),
ResWorld, IDOL independently validate latent-prediction-for-driving, and V-JEPA-2-AC's 62 h
action-conditioning stage anchors the data-efficiency claim.

**Extension.** Driving-specific anti-shortcut regularization: Deep Think 2's ego-compensated objective
(SE(3) ego-warp so static background stops dominating the loss) is adopted as a *candidate* upgrade to
change-weighted loss — to be bake-off'd, not assumed (instrument doctrine). Object permanence via latent
advection + epistemic gating feeds H15.

**Phase 0 recommendation.** This is the Phase 0 backbone. Reproduce A1–A5/A7 on driving data (gates
D1–D3), then hierarchy gates (D4–D6). Keep SIGReg λ=0.1, slices 512, patch 16, window 6–8 as starting
points (all validated).

## H4 — Frozen pretrained encoders as comparison arm. **Status: open, cheap to answer.**

**Evidence.** Drive-JEPA (93.3 PDMS on NAVSIM v1 with V-JEPA-2 encoder) shows the frozen-foundation path
works; Deep Think 11 designs ST-Drive-LoRA (frozen DINOv2-L + LoRA r=16 on Q,V + 1-D temporal convs,
~19.7 GB A40 peak); the PKQT proposal (`Ressources/End2End Driving based on DinoVx and AI planner.md`)
gives a complete 404 M-param reference design with a differentiable kinematic bicycle head and
winner-takes-all multi-mode loss.

**Recommendation.** Run as **arm B** at every gate: same predictor/planner stack, encoder swapped
(from-scratch vs frozen DINOv3+LoRA). Report both; the data-efficiency story is arm A's, the
absolute-performance hedge is arm B's. Components of PKQT (kinematic decoder head, WTA loss) are
adopted for the trajectory head in both arms.

## H5 — Efficient inference transfer (speculative decoding, sparse attention, MTP, flow matching). **Status: promising, needs adaptation research (hub stream).**

**Evidence.** LLM-side techniques are mature (FastMTP, sparse speculative verification, Vegas 2026…).
Deep Think 6 adapts speculative decoding to trajectories: small kinematic-neural draft model + bounded-
divergence acceptance (accept if ‖x_true − x_draft‖ ≤ ε_k), DSPARK-style uncertainty-truncated lookahead;
estimates 2.4× speedup / 59 % bandwidth cut on Thor-class hardware. Deep Think 9's flow-matching
trajectory head with 3-step Heun solver is the natural fast decoder.

**Assessment (honest).** Our latent imagine-and-select loop is already milliseconds (K≈9–15 batched
predictor passes, no diffusion, no CEM) — speculative decoding matters most for the *strategic/text*
brain (H12) and for large tactical vocabularies. Priority: medium in Phase 0, high in Phase 1.
**Recommendation.** Track FLOPs/decision and latency from day one (CNCE metric); implement MTP-style
multi-step prediction heads (predict k ∈ {1,2,4}) which is both a training signal and an inference
accelerator — validated in ALPS-4B as multi-horizon prediction.

## H6 — Opponent weak-spot corpus as instant moat. **Status: actionable immediately (hub stream).**

**Evidence.** Deep Think 7 catalogs failure classes with mechanisms: unprotected lefts (freezing-robot),
construction zones (prior-posterior divergence), traffic-cop gestures (monotonic logic deadlock), ghost
braking (conservative OR-logic), occlusion amnesia, covariate shift, spurious correlations. Public
incident reporting (CA DMV disengagements, NHTSA SGO) provides a free, continuously updated corpus.
**Recommendation.** The Opponent Analyzer agent maintains `Opponent Analyzer/Research/WEAKNESS_CATALOG.md`
→ each entry gets (i) a MetaDrive/CARLA scenario spec, (ii) a metric hook (OKRI/LOPS from Deep Think 14),
(iii) a training-data recipe. Phase 0 includes 3 scenarios: Ghost Cut-Through, Blind Creep (unprotected
left), Choke Weave (dense actors on throttled compute).

## H7 — 1000× data leverage from action-free video via inverse dynamics + focal-length canonicalization. **Status: supported externally, Phase 1 core.**

**Evidence.** VLM3 (Meta, May 2026) shows resizing all inputs to a fictive focal length (f≈1000 px)
stabilizes pixel↔metric correspondence and makes standard models native 3-D learners. Latent-action
models (Genie, LAPA, AdaWorld) infer actions from passive video; Deep Think 8 designs the full pipeline:
auto-calibration → f_fict unification → IDM (frozen DINO + correlation-volume motion extractor + mixture
density head) with kinematic-consistency (SfM odometry) and epistemic filtering of pseudo-labels
(discard top-15 % uncertain).
**Recommendation.** Phase 0: keep the door open — our inverse-dynamics head IS the seed IDM; log its
calibration quality. Phase 1: comma2k19-trained IDM pseudo-labels BDD100K/OpenDV/own GoPro; measure the
data-efficiency slope (the headline claim needs exactly this experiment).

## H8 — MoE beyond sensors. **Status: prio 2, keep interface ready.**

GEMINUS/DriveMoE show scene-adaptive experts work. Our tactical layer already has a Sparse MoE router
(ALPS-4B `moe_router.py`, load-balanced top-k). Decision: the tactical maneuver vocabulary IS the first
expert set; sensor-modality experts (H2) second; skill experts later. No extra Phase 0 work beyond
keeping the router interface.

## H9 — Inherent traffic-rule compliance without losing E2E. **Status: differentiating, mathematically concrete.**

**Evidence.** Survey literature (legal-logical specification integration 2025), GradSTL (differentiable
signal temporal logic), trajectory repair via SMT+reachability, and Deep Think 9's decisive analysis:
hard projection layers (LSCP) destroy smoothness; preference-tuning (T-DPO) leaves probability mass on
violations; **Reward-Modulated Flow Matching (RMFM)** — rules as stiff log-barrier energies shaping the
probability-flow ODE via continuous adjoint — keeps E2E character, zero runtime cost, and supports
**incremental rule addition as post-training** (retrain the alignment, not the system). ACRE contributes
the constraint-orthogonality mask idea (provable constraint satisfaction in latent space) as a
longer-term formal upgrade.
**Recommendation.** Phase 0: encode 3 rules (stop line, speed limit, no-collision) as differentiable
barriers on the decoded trajectory (cheap, works with WTA/kinematic head); measure violation rates as a
first-class metric. RMFM alignment is the Phase 1 flagship experiment for H9.

## H10 — Latent RAG / continual learning from experience. **Status: validated mechanism with a known failure mode; regulation-aligned.**

**Evidence (in-house).** ALPS-4B latent-RAG: +18.8 % on unseen surprise contexts but **−24 % interference
on well-predicted contexts** — retrieval must be surprise-gated. Deep Think 5 supplies the production
design: FAISS HNSW-PQ (<10 ms), geospatial sharding, cross-attention fusion with a learnable NULL token,
gate MLP initialized to −3.0 bias (sensor supremacy by default), hard rule: operative/safety head never
consumes memory, only tactical.
**Extension.** Write policy = imagination-error trigger (A9): store (z, Δz) exactly when the self-monitor
fires — the same signal serves H11 monitoring, H10 memory writes, and ISMR event logging. One mechanism,
three regulation-relevant capabilities.
**Recommendation.** Phase 0: implement the memory interface + write-on-surprise logging (no retrieval in
the control path yet); D7 gate (repeat-exposure improvement) at end of Phase 0 / Phase 1.

## H11 — Self-monitoring with guarantees. **Status: core differentiator; three independent validated mechanisms.**

**Evidence.** (i) Imagination error `‖ẑ−z‖/scale` discriminates familiar/unfamiliar dynamics (ALPS-4B
A9, free); (ii) RSRA checker networks score latent states against consequence targets with Banach-
contraction refinement (232 tests, convergence proof); (iii) Mahalanobis OOD on latent distribution +
effective-rank/variance collapse watchdog (ALPS-4B fallback watchdog). Deep Think 1 maps all three to
the UN ISMR reporting chain: latent trigger → deterministic MRM + constrained-decoding LLM report.
**Extension.** Layered monitors, one per brain: operative (imagination error, 10 Hz), tactical (checker
value + routing entropy, 1 Hz), strategic (Mahalanobis + KL drift, 0.1 Hz), each with its own alarm ROC
measured against injected anomalies (gate D8: AUROC > 0.85 vs unseen-town/weather).
**Recommendation.** Phase 0 implements monitor #1 (free) and the D8 evaluation harness. This is also our
regulatory story (next point) — treat it as a first-class deliverable, not plumbing.

## H12 — Text as a part, not the core. **Status: right call; cheap path exists.**

**Evidence.** Alpamayo-1 (10 B VLA) shows the industry going reasoning-VLA — expensive and, per our
thesis, wrong-way-around (language should read/steer the latent, not drive). Deep Think 10's design:
frozen 1 B LLM (4-bit) + bidirectional latent bridge (SwiGLU projection for text→AdaLN conditioning;
Perceiver resampler compressing world-model tokens → 8 soft prompts for vision→text), trained with
dense/delta SigLIP objectives — runs on 4 GB VRAM. This gives navigation-command conditioning +
reasoning-trace extraction + ISMR report generation without a VLA.
**Recommendation.** Phase 0: command conditioning only as a discrete embedding (turn-left/right/straight
— free with FiLM). The LLM bridge is Phase 1; ISMR generation Phase 2 (needs H11 triggers mature).

## H13 — Extraction heads. **Status: settled design pattern.**

Deep Think 12: stop-gradient firewall (heads read Z, never write), shared explainability adapter,
homoscedastic-uncertainty loss weighting, DETR-style behavior head with N=8 intention queries emitting
(trajectory, viability, rejection rationale) — the "considered alternatives" HMI story. Our calibrated
frozen probes (A3) are the same doctrine in minimal form — probes ARE extraction heads.
**Recommendation.** Phase 0 ships: trajectory probe (must), BEV occupancy probe (visualization),
maneuver-choice head with alternatives (from tactical imagine-and-select — free). Full DETR head Phase 1.

## H14 — Physical/knowledge grounding. **Status: open research; two concrete tracks.**

Track 1 (adopted now): **kinematic bicycle model as differentiable decoder** — network predicts controls,
physics integrates the trajectory; 100 % physically-realizable outputs by construction (PKQT design +
Deep Think 13 PINN residual formulation + Kamm-circle friction penalty ‖a‖ ≤ μg as loss barrier).
Track 2 (hub research): cultural/ethics priors as low-dim conditioning (Ω_ethic matrix, SVO parameters)
— interesting, Phase 2. ACRE's structured concept tensors are the long-horizon formal option.
**Recommendation.** Phase 0 adopts Track 1 in the trajectory head; friction-circle violation rate becomes
a standing metric.

## H15 — Unobserved-area imagination / object permanence. **Status: the key enabler of H2; concrete design available.**

Deep Think 2: latent advection field (occluded agents keep moving in latent space under a learned
velocity field) + ray-cast epistemic gating (loss active only where the model *should* know) + LOPS
metric (track hidden agent through occlusion). External: object permanence in world models is a known
V-JEPA-2 weak spot — a real differentiator if we nail it.
**Recommendation.** Phase 0: measure LOPS on synthetic occlusion scenarios (Ghost Cut-Through) as a
baseline; the advection regularizer is a Phase 1 architecture experiment (gate: LOPS uplift at equal
params).

---

## Cross-cutting: regulation (UN WP.29 ADS, June 2026)

Adopted framework (UNR + GTR; `Ressources/ECE-TRANS-WP.29-2026-139e.pdf`): safety management system,
credible testing + safety case, **in-service monitoring and reporting (ISMR)**, DSSAD event recording.
Design consequences (all already in the architecture): (1) H11 monitors = ISMR triggers; (2) latent
event log (z, Δz, alarms) = DSSAD substrate; (3) gates + instrument doctrine = credible-testing evidence;
(4) fallback brain + MRM = minimal-risk-manoeuvre requirement; (5) H13 heads = transparency evidence.
Action: Benchmarks & Eval agent maintains a requirements-to-evidence traceability table
(`Benchmarks & Eval/REGULATION_TRACE.md`).

## Cross-cutting: competitive positioning table (updated 2026-07)

| Player | Approach | Strength | Exploitable weakness |
|---|---|---|---|
| Waymo | Modular+learned, HD maps, huge fleet | Scale, safety record, $16 B war chest | Cost/vehicle, geofence economics, construction/gesture edge cases, no data-efficiency story |
| Wayve | E2E + GAIA world models, mapless | Talent, capital ($2.8 B), UK/EU access | Pixel-generative world models are compute-hungry; L2+ detour; no hierarchy/self-monitoring story |
| Momenta | Two-leg (mass-produced L2++ funds L4), China scale | Data flywheel, OEM deals | Strategy split, geopolitics, opaque safety case |
| Pony.ai | Robotaxi China/ME | Fleet scale | Same compute-heavy stack, limited Western presence |
| Autobrains | Efficiency-focused L2+ | Low-compute story overlaps ours | Sub-L3 focus = not a direct L4 competitor; watch their efficiency claims |
| NVIDIA Alpamayo | Open 10 B reasoning VLA + sim + data | Commoditizes baselines — helps us (free data/sim) | 10 B params on-vehicle is the anti-thesis of our efficiency claim |

**Our wedge:** data efficiency (H3+H7), inference efficiency (H1+H2+H5), inherent safety/monitoring
aligned with the fresh regulation (H1 fallback + H11 + H9) — provable with gates, at a compute budget a
single person can afford. The Alpamayo ecosystem (AlpaSim, PhysicalAI-AV, NuRec) is *our supply chain*,
not our competitor.

## Research agenda seeds per discipline (for the weekly agents)

- **Tools&DevEnv:** AlpaSim adoption path for a non-NVIDIA-scale team; MetaDrive vs CARLA closed-loop
  cost; replay/visualization MVP; colab-CLI as burst compute.
- **Data Engineering:** PhysicalAI-AV license + streaming performance; comma2k19 ingestion; VLM3-style
  focal canonicalization spec; weak-spot scenario data generation (H6).
- **Architecture & Inference:** ego-compensated loss vs change-weighting bake-off design; advection
  regularizer (H15); flow-matching trajectory head; MTP heads; Orin/Thor deployment path (TensorRT,
  INT8, batch-free norms).
- **Benchmarks & Eval:** implement LAL/TMS/OKRI/CNCE/LOPS; NAVSIM v2 EPDMS harness; regulation
  traceability; leaderboard bootstrap with published competitor numbers.
- **Opponent Analyzer:** weakness catalog v1 from DMV/NHTSA reports + Deep Think 7; track Wayve GAIA,
  Waymo research output, Momenta IPO filings, Alpamayo releases.

## Sources (key)

- Own repos: ALPS-4B (`docs/AD_TRANSFER_RESEARCH.md`, `docs/PROJECT_HANDOFF.md`, validation results), 4B-HRM, ACRE, RSRA-4B.
- [LeJEPA (arXiv 2511.08544)](https://arxiv.org/abs/2511.08544) · [lejepa code](https://github.com/rbalestr-lab/lejepa)
- [LAW](https://arxiv.org/abs/2406.08481) · [World4Drive](https://arxiv.org/abs/2507.00603) · [WorldRFT](https://arxiv.org/abs/2512.19133) · [V-JEPA 2](https://arxiv.org/abs/2506.09985) · [Drive-JEPA](https://arxiv.org/html/2601.22032v1)
- [NVIDIA Alpamayo](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) · [Alpamayo dev blog](https://developer.nvidia.com/blog/building-autonomous-vehicles-that-reason-with-nvidia-alpamayo/) · [PhysicalAI-AV dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) · [TechCrunch](https://techcrunch.com/2026/01/05/nvidia-launches-alpamayo-open-ai-models-that-allow-autonomous-vehicles-to-think-like-a-human/)
- [VLM3](https://arxiv.org/html/2605.30561v1) · [VLM3 code](https://github.com/facebookresearch/VLM3) · [LAPA](https://arxiv.org/abs/2410.11758) · [AdaWorld](https://arxiv.org/html/2503.18938v1)
- [DriveMoE](https://arxiv.org/pdf/2505.16278) · [GEMINUS](https://arxiv.org/html/2507.14456v2)
- [GradSTL](https://arxiv.org/pdf/2508.04438) · [Legal-logical integration survey](https://arxiv.org/pdf/2510.25386) · [Lawful AD requirements](https://arxiv.org/html/2604.24562v1)
- [RealDrive (RAG diffusion)](https://arxiv.org/pdf/2505.24808) · [RAG-Driver line of work](https://arxiv.org/pdf/2410.04759)
- [UNECE adoption press](https://unece.org/sustainable-development/press/unece-adopts-first-ever-global-rules-allowing-fully-autonomous) · [State of play](https://www.globalpolicywatch.com/2026/05/un-regulation-and-gtr-on-automated-driving-systems-current-state-of-play/) · [Sidley analysis](https://environmentalhealthsafetybrief.sidley.com/2026/03/04/a-new-global-milestone-for-autonomous-vehicles-what-the-un-global-technical-regulation-on-automated-driving-systems-means-for-autonomy-in-the-u-s-and-around-the-world/)
- Market: [Wayve $2.8 B](https://www.technology.org/2026/07/01/wayve-ai-self-driving-2-8-billion/) · [Waymo $126 B via SecondWave note](https://note.com/startup_now0708/n/n708666dbb0e2?hl=en) · [Uber+Wayve London](https://www.automotiveworld.com/articles/uber-and-wayve-to-trial-london-robotaxis-in-spring-2026/)
- [NAVSIM v2 context: HAD](https://arxiv.org/html/2604.03581v1) · [RAP](https://arxiv.org/pdf/2510.04333) · [GTRS](https://arxiv.org/pdf/2506.06664)
- Deep Think analyses 1–14: `Ressources/Deep Think Analysis/` (designs referenced per hypothesis above).
