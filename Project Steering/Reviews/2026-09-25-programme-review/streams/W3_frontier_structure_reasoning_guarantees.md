# Stream W3 — Frontier research: structured physical understanding, symbolic & neuro-symbolic reasoning, knowledge & rules, tool use, physics-grounded neural operators, mathematical guarantees

**Status: IN PROGRESS — banking incrementally.** Orchestrator: PI Sayed, whole-programme review, 2026-09-25.
Method: web research (WebSearch/WebFetch), primary sources preferred, arXiv ids quoted where found.
Evidence-class legend: **PUBLISHED** (title, arXiv id/venue, year, key number) · **ESTIMATED** · **HYPOTHESIS** · **UNVERIFIED**.

## 0. Delta check against existing docs (read first, not re-derived)

Skimmed before searching, per brief:
- `Project Steering/Proposals/External Anaysis.md` — already covers (do NOT re-report as new): VERDI
  (VLM→latent distillation via Progressive Feature Projectors), LVLDrive (LiDAR-Vision-Language,
  Gradual Fusion Q-Former), **Alpamayo-R1** (Chain-of-Causation, 132.8% causal-understanding
  improvement, 37% consistency gain, 99 ms latency — cited there without an arXiv id), ColaVLA
  (hierarchical parallel latent planner), LAMP (VQ-VAE lane-aligned motion primitives + topology
  filter), PLAN-S (4-channel semantic cost maps, AdaFiLM, -42% collision rate/3s on nuScenes),
  Fast-dDrive (block-diffusion + scaffold speculative decoding, 12x throughput). This stream treats
  these as KNOWN and either (a) skips them, (b) adds the primary-source arXiv id/verifies the number
  if cheaply found, or (c) extends with what that doc did not cover (e.g. Alpamayo-R1's actual
  causal-chain formalism vs. STL/LTL rule monitors, which the doc does not touch at all).
- `TanitAD Research Hub/2026-07-11-sayed-papers-screening.md` — AUTOPILOT-VQA (incident VQA bench,
  avoidability reasoning) and ZipDepth (6.1M-param distilled depth, 77 FPS Orin NX) already
  screened; not re-covered here except where relevant to a guarantee/reasoning angle.
- `Benchmarks & Eval/REGULATION_TRACE.md` — UN ADS / WP.29 (June 2026) requirements already mapped:
  safety mgmt system, credible testing (validated virtual toolchains accepted), DDT fallback/MRM,
  ISMR, DSSAD recording, ODD monitoring, baseline-adequacy gap (CTRV floor). This stream's Section F
  is complementary: ISO 26262 / ISO 21448 SOTIF / UL 4600 / ISO/PAS 8800 + formal-methods content
  (RSS, CBF, HJ reachability, conformal prediction, NN verification) is NOT in that doc at all.

---

## A. Structured physical world understanding

### A1. RiskWorld — object-centric latent world model for risk identification
- **What:** Object-centric world model that builds relation-aware object states from observed
  visual+motion histories, rolls them forward with RSSM-style latent dynamics, and decodes the
  imagined ego-object evolution into per-object risk scores + future-relation evidence — risk as a
  *readout of imagined rollout*, not a separate classifier head.
- **Evidence:** PUBLISHED — "RiskWorld: Object-Centric Latent World Modeling for Autonomous Driving
  Risk Identification," arXiv:2608.21414 (Aug 2026). RiskBench: **63.0% overall F1, 2.1% false-alarm
  rate**. Maturity: PROMISING (single-benchmark, no closed-loop number found).
- **Pain points:** **P10** (open-loop ADE blind to dominant failure) — a risk score computed from
  imagined rollout is exactly the kind of instrument that would have caught v1's closed-loop
  off-road/longitudinal failures that ADE@2s missed. **P7** (imagination anti-calibration) — this is
  a worked example of turning imagination into a *scored, benchmarked* output instead of a vibe.
- **Admissibility:** Object-centric decomposition happens on the ego-centric latent trained
  vision-only; risk scores are read from rollout, not from privileged state — compatible with the
  vision-only inference rule as described (their "ego-object histories" language suggests they may
  use ego state as one context stream, so this needs a source-code check before adoption, not just
  the abstract).
- **Cost:** ~3-5 A40-days (train a risk-readout head on TanitAD's existing H15 imagination latents —
  no new encoder), ~4-6 eng-days (RiskBench-style labels do not exist for PhysicalAI; would need to
  derive risk labels from `obstacle.offline` proximity/TTC, which the corpus already carries at
  97.44% coverage per the operating-standard trap log).
- **Experiment:** Train a linear risk-readout on frozen H15 imagination rollouts (candidate vs.
  chosen trajectory), label with TTC-to-lead-agent from `obstacle.offline`. **Outcome A:** readout
  F1 tracks closed-loop AlpaSim pass/fail arm-for-arm (imagination already encodes risk, just
  unread) → cheap win, wire into the selection layer for P1. **Outcome B:** readout F1 is near
  chance → H15's imagination is not risk-relevant in its current form, redirect effort to the
  anti-calibration fix (F, DreamLedger) before trying to read risk off it.

### A2. Relationally Grounded Latent World Models for Autonomous Driving
- **What:** Learns explicit relational (scene-graph-like) structure *into* a latent world model's
  representation — the relational supervision is a training-time signal, not a runtime input.
- **Evidence:** PUBLISHED — arXiv:2609.24626 (Sept 2026). Search summaries describe "large and
  persistent" gains from relational grounding but I could not pull the exact benchmark deltas
  (arXiv/HF fetch blocked in this environment — see note below). Maturity: PROMISING, numbers
  UNVERIFIED beyond search-snippet level.
- **Pain points:** **P3** (hierarchy lacks route/goal signal; strategic level unmeasured) — relational
  grounding is a candidate mechanism for injecting *structure* (which agent matters, right-of-way,
  lane relations) without injecting a privileged route. **P1** (selection) — a relationally-structured
  latent gives the tactical selector something more legible than a flat 768-d vector to condition on.
- **Admissibility:** Clean in principle — GraphPilot (A3 below) is the existence proof that relational
  structure can be a *train-time-only* teacher signal with a vision-only deployed model. Must verify
  this specific paper doesn't leak future/privileged relations into the encoder that also feeds the
  situation classifier (the BINDING situation-classifier firewall applies to any shared trunk).
- **Cost:** ~8-12 A40-days (auxiliary relational loss on the existing ViT encoder, needs an object/
  relation extraction step from `obstacle.offline` first), ~5 eng-days for the relation-extraction
  pipeline.
- **Experiment:** Add a relation-prediction auxiliary head (pairwise agent relations from
  `obstacle.offline`, e.g. "closing," "yielding," "same-lane") on top of the *frozen* v1 encoder
  latent, train-only, discarded at inference. **Outcome A:** the probe R² is high (structure is
  already latent, unread — matches the operating standard's repeated pattern of "it's there, we're
  not reading it"). **Outcome B:** low R² → the encoder genuinely lacks relational structure, which
  would upgrade this from a probe to an architecture change (auxiliary loss during pretraining, not
  post hoc).

### A3. GraphPilot — scene-graph conditioning as a train-time-only teacher (deployed graph-free)
- **What:** Conditions a language-based driving model on serialized traffic scene graphs during
  training; the striking result is that gains **persist even when the scene graph is not given at
  test time** — the model internalizes relational priors into its weights.
- **Evidence:** PUBLISHED — "GraphPilot: Grounded Scene Graph Conditioning for Language-Based
  Autonomous Driving," arXiv:2511.11266 (Nov 2025/2026 cycle), authors Schmidt, Enzweiler, Valada
  (code: github.com/iis-esslingen/GraphPilot). Evaluated on LangAuto + Bench2Drive vs. LMDrive,
  BEVDriver, SimLingo baselines — "large and persistent" Driving Score gains reported; exact deltas
  UNVERIFIED (could not fetch the PDF body in this environment). Maturity: PROMISING.
- **Pain points:** **P3** (no route/goal signal at the architecture level) and **P4** (small data) —
  this is the general recipe for injecting privileged/structured knowledge (maps, relations, even a
  route) at *train time only*, then deploying a model that needs none of it at inference. Directly
  reusable pattern for TanitAD's own admissibility constraint.
- **Admissibility:** This is the textbook shape of an admissible design under the BINDING rules:
  privileged structure at train time, vision/graph-free at inference. **Directly cite-worthy as
  precedent** the next time anyone proposes "but we could just feed the graph/route at inference" —
  GraphPilot's own result argues you don't need to, because the training signal survives distillation
  into weights.
- **Cost:** ~10-15 A40-days, ~6-8 eng-days (needs a scene-graph builder over `obstacle.offline`, which
  is a new but well-scoped ETL job, not a research risk).
- **Experiment:** Build a minimal 1-hop scene graph (ego + k-nearest agents, relation = relative
  lane/closing-rate/yield) from `obstacle.offline`, use as an auxiliary train-time loss on the
  tactical brain, remove at test time. **Outcome A:** Bench2Drive-style closed-loop metric improves
  with the graph loss removed at test → validates the whole "structure now, vision-only later"
  strategy for TanitAD. **Outcome B:** no improvement → either the graph loss is too weak a signal
  at our data scale (13h) or PhysicalAI's single-camera FRONT view lacks the geometry to make
  relations legible even as a training target.

### A4. WA-JEPA — extending V-JEPA-style video JEPA to world-ACTION modeling for driving
- **What:** Extends the V-JEPA2 video-JEPA paradigm with action-conditioning: hybrid future-masked
  pretraining + flow-matching prediction of future latents + joint future-action modeling in a shared
  spatiotemporal latent space — i.e. almost exactly TanitAD's own JEPA-style latent-prediction bet,
  done at a different (larger, multi-domain) scale.
- **Evidence:** PUBLISHED — arXiv:2608.20974 (Aug 2026), "WA-JEPA: Rethinking the Video JEPA Paradigm
  for World-Action Modeling in Autonomous Driving." Exact head-to-head numbers vs. baselines
  UNVERIFIED (fetch blocked); maturity PROMISING given it's a direct architectural sibling of our own
  operative-predictor design, not merely an analogy.
- **Pain points:** **P5** (frozen image encoders failed) — if WA-JEPA reports what pretraining recipe
  and how much data made JEPA-style action-conditioning work at scale, that is directly diagnostic for
  why our from-scratch encoder's speed-probe R² regressed v1→v2 (0.861→0.300, `MODEL_REGISTRY.md`).
  **P6** (closed-loop compounding error) — flow-matching future-latent prediction is a candidate
  replacement for the 20-sequential-step autoregressive rollout that currently dominates planning-tick
  latency (83.7-96.7% of the tick per `MODEL_REGISTRY.md`).
- **Admissibility:** Needs a read of the actual action-conditioning inputs — "joint future-action
  modeling" could mean ego-action only (fine) or could smuggle other privileged channels; UNVERIFIED
  until source is read.
- **Cost:** Full replication is large (~40-60 A40-days at TanitAD's data scale is a guess, ESTIMATED,
  not the paper's own compute); a narrow ablation (flow-matching predictor head bolted onto the
  existing encoder, A/B vs. the current 10-step autoregressive predictor) is ~10 A40-days, ~7 eng-days.
- **Experiment:** Swap only the operative predictor's rollout mechanism (sequential residual steps →
  one flow-matching jump to the K-step-ahead latent) on the existing v1 encoder, hold everything else
  fixed. **Outcome A:** ADE@2s within noise AND planning-tick rollout stage drops materially below the
  current 28.7-95ms-depending-on-precision figure → adopt, this directly attacks the P6/latency
  problem. **Outcome B:** ADE@2s degrades — sequential rollout's error-feedback (each step conditions
  on the last) may be load-bearing for our short 2s horizon at our data scale; log as a refuted-family
  result per the retraction-log discipline.

### A5. "Is Forward Prediction Enough? Physical State Grounding for JEPA World Models"
- **What:** Directly interrogates whether JEPA's forward-latent-prediction objective (predict the
  future embedding) is sufficient to acquire physically grounded state, or whether an explicit
  physical-state auxiliary is needed.
- **Evidence:** PUBLISHED — arXiv:2608.06799 (Aug 2026). Exact quantitative finding UNVERIFIED (fetch
  blocked; search snippet only confirms the framing, not the answer). Maturity: PROMISING but
  **the paper's conclusion itself is the load-bearing fact here and I could not verify which way it
  comes out** — flag for a follow-up fetch attempt via a non-blocked mirror before this is used to
  justify any architecture change.
- **Pain points:** **P7** (imagination anti-calibration) and **P2** (longitudinal) — this is the
  single most relevant external paper to TanitAD's own anti-collapse (SIGReg) + inverse-dynamics
  design, because it asks exactly the question TanitAD's architecture rests on: does latent
  prediction alone (even with anti-collapse) yield a physically grounded state, or do you need an
  explicit auxiliary (TanitAD already has one candidate answer in its own inverse-dynamics term —
  this paper is external evidence either for or against that choice, TOP PRIORITY to actually read).
- **Admissibility:** N/A (methodology paper, not a data source).
- **Cost:** Reading cost only (~0.5 eng-days) to resolve the UNVERIFIED status; if it recommends a
  specific auxiliary loss not already in TanitAD's recipe, replication is ~5-10 A40-days.
- **Experiment:** N/A until read — but the discriminating question it should settle for us is
  pre-registrable now: **Outcome A:** if the paper shows forward-prediction-alone models fail a
  physical-grounding probe that an inverse-dynamics-augmented model passes, that is external
  confirmation of TanitAD's existing inverse-dynamics term — cite it, done, no new work. **Outcome
  B:** if it shows even inverse-dynamics-style auxiliaries are insufficient and something else
  (e.g. an explicit contrastive future/counterfactual term) is needed, that is a direct, actionable
  lead against P7.

### A6. Intuitive physics in video/JEPA models — the humility check (Physics-IQ, IntPhys2)
- **What:** Two independent benchmarks quantify how much genuine physical understanding
  self-supervised video/JEPA models actually have, as opposed to visual realism.
- **Evidence:** PUBLISHED —
  (i) "Do generative video models understand physical principles?" / **Physics-IQ**, arXiv:2501.09038
  (Google DeepMind, Jan 2025/2026 cycle): 396 real 4K30 videos across solid mechanics/fluids/
  thermodynamics/optics/magnetism; **best model scored 29.5% of the 100% physical-variance
  ceiling** (Sora, Runway Gen-3, Pika, Lumiere, SVD, VideoPoet tested). Successor **Physics-IQ
  Verified**, arXiv:2606.18943 (2026): best model on the public leaderboard (per the project's GitHub
  README, github.com/google-deepmind/physics-IQ-benchmark) reaches **58.2% ± 1.8** (video-to-video,
  Magi-1 24B + GeoPhys/BoN) — still far under ceiling.
  (ii) **IntPhys2** (Meta): **V-JEPA-2-H-VM22M scores 0.52 accuracy** on the plausible/implausible
  discrimination task — **barely above the 0.50 chance floor**, while humans are near-ceiling.
  Maturity: PROVEN as a *negative* result (the gap is real and repeatedly measured across
  benchmarks/labs).
- **Pain points:** This is a **cautionary** finding for TanitAD's own JEPA-style bet, not a solution.
  It bears on **P7** directly: if even Meta's V-JEPA2 (far larger, far more data than our sub-300M/13h
  budget) is near chance on intuitive-physics discrimination, TanitAD's own imagination module should
  be assumed anti-calibrated and *not* physically grounded by default, not merely "possibly" so — the
  prior should be pessimistic until measured.
- **Admissibility:** N/A (benchmark finding, informs risk assessment not architecture).
- **Cost:** Near-zero to internalize; ~3-5 A40-days to build a TanitAD-scale IntPhys2-style
  plausible/implausible probe (generate physically-violating synthetic continuations is the hard
  part — would need a cheap physics-violation generator, e.g. object teleportation/interpenetration
  injected into PhysicalAI clips).
- **Experiment:** Build ~200 plausible/implausible clip pairs from PhysicalAI (implausible = object
  position/velocity edited to violate continuity) and score TanitAD's H15 imagination module's
  likelihood/surprise signal on each pair. **Outcome A:** surprise signal separates the pairs above
  chance → some physical grounding exists, worth calibrating (ties to DreamLedger, section F).
  **Outcome B:** at-chance separation, matching V-JEPA2 → confirms the pessimistic prior, and argues
  against using raw imagination-surprise as a safety signal (P7) until an explicit physics auxiliary
  is added.

### A7. Occupancy / 4D world models (OccWorld family) — genuinely structured, but a scale mismatch
- **What:** Predict future 3D/4D occupancy grids as the world-model state (vs. TanitAD's flat latent
  vector), giving an explicitly spatial, geometrically interpretable structure.
- **Evidence:** PUBLISHED — Drive-OccWorld (AAAI 2025 oral, arXiv:2408.14197); its ICRA 2026 successor
  **IR-WM** ("Implicit Residual World Models," arXiv:2510.16729) — improves efficiency via residual
  implicit occupancy rather than dense voxel prediction (numbers UNVERIFIED, fetch blocked).
  Maturity: PROVEN as an architecture family (multiple independent groups, multiple venues,
  2024-2026), but **all of it assumes multi-camera or LiDAR-level 3D input**, which TanitAD does not
  have (FRONT camera only).
- **Pain points:** **P3/P10** in principle (occupancy is a structured, checkable intermediate that
  could carry a strategic-level read), but this is the section's clearest **scale/data mismatch**:
  building a metric 3D occupancy grid from a single front camera at 256×256 without depth
  supervision (TanitAD has no LiDAR, no stereo) is a much harder, underdetermined problem than every
  OccWorld variant assumes. ZipDepth (already screened, 6.1M params, 77 FPS Orin NX, **affine-invariant,
  no metric scale**) is the closest fit but explicitly cannot give metric occupancy without an
  external scale prior TanitAD does not have (no GNSS, clip-local coordinates only).
- **Admissibility:** Clean (vision-only, monocular).
- **Cost:** Full occupancy-world-model adoption: large, ESTIMATED 30-50 A40-days plus a new
  depth/occupancy pretraining pipeline — a program-level bet, not a probe.
- **Experiment (cheap, discriminating, before any large bet):** Fine-tune ZipDepth (open weights) on
  a small PhysicalAI subset with a flat-ground metric-scale heuristic (camera height is known from
  intrinsics/extrinsics, which the corpus does carry per D-016 R1), then check whether decoded
  depth improves the existing longitudinal (P2) distance-keeping metric when fused as an auxiliary
  input to the operative predictor. **Outcome A:** distance-keeping error drops → occupancy-style
  geometric grounding is worth a real investment. **Outcome B:** no change → the bottleneck is
  elsewhere (e.g. in the tactical selection or speed-setting policy, not perception), and full
  occupancy-world-model adoption should be deprioritized relative to cheaper P2 fixes.

**Note on this section's evidence quality:** `arxiv.org` and `huggingface.co` were both blocked by
this environment's egress proxy for direct fetch (`EGRESS_BLOCKED`, confirmed via
`/root/.ccr/__agentproxy/status` — this is a declared organization policy, not a transient failure,
so it was not retried or routed around). All arXiv ids above are as returned by WebSearch snippets
(which appear to be served through a different path than direct fetch) and were **not** independently
confirmed by opening the PDF/abstract page. Titles, ids and the headline numbers I could cross-confirm
across ≥2 independent search snippets are reported as PUBLISHED; numbers that appeared in only one
search summary and could not be fetched for confirmation are explicitly flagged UNVERIFIED inline
above rather than silently reported as PUBLISHED.

---

## B. Symbolic & neuro-symbolic reasoning for driving

*(filling in below)*

---

## C. Knowledge injection

*(filling in below)*

---

## D. Tool use for AD

*(filling in below)*

---

## E. Physics-grounded neural operators & dynamics

*(filling in below)*

---

## F. Mathematical guarantees & runtime assurance

*(filling in below)*

---

## Ranked TOP-10 for TanitAD

*(filling in below)*

---

## Proposed "guaranteed envelope" architecture

*(filling in below)*

---

## What is hype

*(filling in below)*

---

## Deliverable manifest

| Artifact | Location |
|---|---|
| This report | `repo:Project Steering/Reviews/2026-09-25-programme-review/streams/W3_frontier_structure_reasoning_guarantees.md` |
