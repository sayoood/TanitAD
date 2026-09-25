# Stream W3 — Frontier research: structured physical understanding, symbolic & neuro-symbolic reasoning, knowledge & rules, tool use, physics-grounded neural operators, mathematical guarantees

**Status: COMPLETE.** Orchestrator: PI Sayed, whole-programme review, 2026-09-25.
Method: web research (WebSearch/WebFetch), primary sources preferred, arXiv ids quoted where found.
Evidence-class legend: **PUBLISHED** (title, arXiv id/venue, year, key number) · **ESTIMATED** · **HYPOTHESIS** · **UNVERIFIED**.
Coverage: 6 sub-sections (A structured physical world understanding, B symbolic/neuro-symbolic
reasoning, C knowledge injection, D tool use, E physics-grounded operators/dynamics, F mathematical
guarantees/runtime assurance), 34 distinct ideas each carrying the six required fields, closing with
a ranked TOP-10, a sized "guaranteed envelope" architecture, and an explicit hype list. 40 WebSearch
calls used (of the ≤45 budget); WebFetch was blocked for `arxiv.org`/`huggingface.co` in this
container (see the Deliverable Manifest's environment-constraint note) so no WebFetch calls
contributed usable content — this is disclosed rather than hidden.

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

### B1. Alpamayo-R1 / Alpamayo 1 — Chain-of-Causation VLA (verifying the External-Analysis claim)
- **What:** A 10B VLA (Cosmos-Reason VLM backbone + diffusion trajectory decoder) trained to emit a
  strictly-structured causal chain: one high-level decision → the minimal causal factors responsible
  → a causally-linked text path from observation to action, with history clamped to a 2s window.
  Renamed "Alpamayo 1" after CES 2026 launch.
- **Evidence:** PUBLISHED — **arXiv:2511.00088**, "Alpamayo-R1: Bridging Reasoning and Action
  Prediction for Generalizable Autonomous Driving in the Long Tail" (NVIDIA Research). This resolves
  the citation gap in `Project Steering/Proposals/External Anaysis.md`, which quoted the numbers with
  no arXiv id. Confirmed across independent search summaries: Chain-of-Causation (CoC) dataset with
  **700K annotated video segments**; **132.8% improvement in causal understanding** of long-tail
  events; RL post-training gives **+45% reasoning quality** and **+37% reasoning-action consistency**;
  **99 ms** on-vehicle inference latency; weights at `nvidia/Alpamayo-R1-10B` (HF), code at
  `github.com/NVlabs/alpamayo`. Maturity: **PROVEN** (on-vehicle road tests reported, not just sim).
- **Pain points:** **P3** (strategic level unmeasured) — CoC's single-high-level-decision-then-causal-
  factors structure is a template for a *checkable* strategic output, unlike TanitAD's current
  4-nav-command strategic head. **P1** (selection) — "causal locality" (2s window) is itself a
  regularizer against exactly the kind of spurious-correlation selection failures P1 describes.
- **Admissibility:** The CoC reasoning trace is generated by an offline teacher/auto-labeling
  pipeline (train-time), the 10B VLM is discarded or distilled for the deployed path — same
  train-time-privilege / inference-time-discipline shape as GraphPilot (A3). At 10B params it is
  **>30× TanitAD's entire sub-300M budget**; not directly portable, but its CoC *data schema* and its
  distillation recipe (see VERDI, already known) are.
- **Cost:** Full replication is out of scope (10B model, NVIDIA-scale compute). A schema-only
  adoption — relabel a slice of PhysicalAI with a lightweight CoC-style
  (decision, causal-factors, text-path) annotation via a much smaller open VLM as auto-labeler, then
  train a small auxiliary text/decision head on TanitAD's existing tactical latent — is
  ESTIMATED ~10-15 A40-days + ~8 eng-days (labeling pipeline dominates).
- **Experiment:** Auto-label 500 PhysicalAI clips with a (single decision, ≤3 causal factors) schema
  using an open small VLM, train a decoder from the tactical latent to reproduce the decision label
  only (not the text), evaluate decision-accuracy vs. the existing 5-manoeuvre tactical head.
  **Outcome A:** decision accuracy ≥ current 5-way manoeuvre head's accuracy → the CoC-style single-
  decision framing is at least as learnable and gives a free interpretability channel, worth carrying
  forward. **Outcome B:** materially worse → our 13h/2376-episode budget is too small for even a
  lightweight causal-schema auxiliary; stay with the current tactical head.

### B2. DriveLM / Reason2Drive / DriveLMM-o1 — graph-structured and step-wise CoT VQA (context, not new)
- **What:** DriveLM (arXiv:2312.14150, ECCV 2024) frames perception→prediction→planning QA as a
  directed-acyclic graph, so each answer's context is explicit parent/grandparent nodes rather than
  free text. Reason2Drive (ECCV 2024) and DriveLMM-o1 (arXiv:2503.10621, step-by-step reasoning
  dataset + MLLM) extend this to explicit chain-of-thought supervision.
- **Evidence:** PUBLISHED, both ECCV 2024 papers — classics by now, already implicitly the ancestry
  of most 2026 work in this section (Alpamayo-R1's CoC, Cognitive Dual-Process Planning, GraphPilot
  all cite this lineage). Maturity: PROVEN as a *data schema*, not as a deployed driving stack.
- **Pain points:** **P3, P1** — same mechanism family as B1/B3: a graph of QA nodes is a structured,
  auditable intermediate representation of the "why" behind a decision.
- **Admissibility:** Clean if the graph is built from train-time-only privileged annotations (as
  DriveLM's is) and not fed back at inference.
- **Cost / Experiment:** Superseded in practicality by B1/B3 (Alpamayo-R1's causal chain and Cognitive
  Dual-Process Planning's verifiable consistency are the more 2026-current, more driving-native
  versions of this same idea) — no separate experiment proposed; noted for completeness/lineage only.

### B3. Cognitive Dual-Process Planning — structured scene knowledge + verifiable reasoning-action consistency
- **What:** A System-1/System-2-style dual-process planner where structured scene knowledge
  (extracted facts about the scene) is used with adaptive reasoning and **rule-based verification**
  to check that the VLM's stated reasoning is actually consistent with the action it selects — i.e. a
  built-in lie-detector for chain-of-thought.
- **Evidence:** PUBLISHED — arXiv:2607.19194 (Jul 2026), "Cognitive Dual-Process Planning for
  Autonomous Driving with Structured Scene Knowledge and Verifiable Reasoning-Action Consistency."
  Exact benchmark deltas UNVERIFIED (fetch blocked). Maturity: PROMISING.
- **Pain points:** **P1** (selection) directly — "verifiable reasoning-action consistency" is
  precisely a check against the failure mode TanitAD already names as its core problem: good
  candidates, bad choice, potentially for reasons disconnected from the stated/intended logic.
- **Admissibility:** Clean in principle (the verification is a training/eval-time consistency check,
  not a privileged inference input) — but needs a read of exactly what "structured scene knowledge"
  is built from before adoption (ego state vs. vision-derivable facts matters for the vision-only
  rule).
- **Cost:** ~8-10 A40-days, ~10 eng-days (the rule-based verifier itself is cheap; the scene-knowledge
  extraction pipeline is the cost).
- **Experiment:** Build a minimal rule-based consistency checker for TanitAD's own tactical head: does
  the selected manoeuvre class (of the 5) match what a simple rule engine over `obstacle.offline` +
  ego kinematics would predict as consistent (e.g. "lane-change-left" selected while a vehicle occupies
  the target lane within braking distance = inconsistent)? Log inconsistency rate against known
  closed-loop failures. **Outcome A:** inconsistency rate correlates with AlpaSim failures → a cheap,
  interpretable P1 instrument, promote to a runtime monitor (feeds Section F's envelope). **Outcome
  B:** no correlation → P1's failure mode is not "rule-inconsistent selection" but something else
  (e.g. correct-looking selection with wrong underlying value estimates) — redirects P1 investigation.

### B4. Neuro-symbolic safety guards attached post hoc to an already-trained agent
- **What:** A family of **2026** papers converging on the same architecture: keep the learned
  end-to-end agent as-is, attach a lightweight rule-checking module at the **command interface**
  (after the network decides, before the command reaches the vehicle), and only intervene —
  replacing with the nearest safe alternative — when a rule is about to be violated. No retraining,
  no learned component in the guard, every intervention traceable to the triggering rule.
- **Evidence:** PUBLISHED —
  (i) "Herding End-to-End Autonomous Driving via Neuro-Symbolic Safety Guards," arXiv:2608.11451
  (11 Aug 2026, Patiño Idarraga, Silva, Yasmin, Shoker). Core finding: E2E agents "achieve high
  average performance yet still violate basic traffic rules because they learn statistical patterns
  rather than the physical conditions that guarantee safe driving" — precisely TanitAD's own v1
  closed-loop failure mode (off-road/longitudinal, not collisions).
  (ii) "Neuro-Symbolic Drive: Rule-Grounded Faithful Reasoning for Driving VLAs," arXiv:2606.23938.
  (iii) "DriveSafer: End-to-End Autonomous Driving with Safety Guidance," arXiv:2605.16737.
  Exact quantitative violation-reduction numbers UNVERIFIED for all three (fetch blocked); the
  architectural claim itself (post hoc, rule-based, zero-retraining guard) is corroborated across 3
  independent 2026 papers, which raises confidence the *pattern* is real even where I couldn't verify
  a specific number. Maturity: PROMISING (architecturally converged upon by ≥3 independent groups
  in a 3-month window, which is itself a signal).
- **Pain points:** This is the single most directly applicable idea in the whole review for
  TanitAD's **P1** (selection) and **P6** (closed-loop compounding error): it is a bolt-on for the
  *exact* failure mode already measured — v1's closed-loop losses are off-road/longitudinal, not
  collision, and a road-boundary + speed-envelope rule guard is cheap to specify and requires **zero
  changes to the trained flagship**.
- **Admissibility:** Clean by construction — the guard runs on the command *output*, using
  vision-derivable/ego-kinematic facts (lane boundary estimate, current speed, road edge), not on any
  privileged situation-classifier output, so it does not touch the BINDING classifier-firewall rule at
  all (it sits downstream of planning, not inside the goal path).
- **Cost:** ~2-4 eng-days to implement a minimal 2-rule guard (stay-on-road, speed-envelope) against
  the *existing deployed* v1 checkpoint; **zero GPU-days** — this is a CPU-only, no-retraining
  intervention, which makes it close to the cheapest possible P1/P6 experiment in this entire report.
- **Experiment:** Wrap the deployed v1 flagship's waypoint output with a 2-rule guard (clip
  off-road excursions using the corpus's drivable-area estimate; clamp speed to a kinematic envelope
  derived from curvature) and re-run the existing AlpaSim closed-loop suite (currently 2/12 pass for
  the hierarchical v1 vs. 8/12 for flat REF-C). **Outcome A:** pass rate improves measurably toward
  REF-C's → confirms the off-road/longitudinal failures are "last-mile" and correctable without
  touching the network, reframes the P1/P6 problem as an engineering fix, not a research problem.
  **Outcome B:** pass rate barely moves → the failures are earlier in the causal chain (bad
  intent/selection propagating into trajectories the guard can't locally repair), which argues the
  root cause is upstream (selection/value estimation) not the final command — redirects effort to B3
  or the tactical-head redesign instead.

### B5. A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning (LLM-extracted rules + Answer Set Programming)
- **What:** Uses a language model to *extract* scene rules from context, then hands logical
  arbitration to an Answer Set Programming (ASP) solver, combining inductive (learned) pattern
  extraction with deductive (solver-guaranteed) arbitration for final decisions; also embeds a
  differentiable Kinematic Bicycle Model (decision-conditioned, gradient-connected to the planning
  query) as its physical decoder.
- **Evidence:** PUBLISHED — arXiv:2603.12421 (Mar 2026). Maturity: PROMISING (specific benchmark
  numbers UNVERIFIED, fetch blocked).
- **Pain points:** **P1** (selection — the ASP arbitration is exactly a formal selector over
  candidates, the thing P1 says is currently weak) and **E** crossover (the differentiable KBM decoder
  is directly relevant to Section E below and to TanitAD's own tactical/operative waypoint decoders).
- **Admissibility:** ASP rules are typically hand-authored/human-legible (right-of-way, stop-sign
  precedence), not learned from privileged runtime signals — clean for the inference-time
  vision-only constraint as long as the *inputs to the rules* are vision/kinematics-derivable.
- **Cost:** ~5-7 eng-days (ASP solvers, e.g. clingo, are lightweight and mature; the engineering cost
  is rule-authoring + interfacing, not GPU compute). ~0 A40-days for the solver itself; ~3-5 A40-days
  if paired with retraining a selector to match ASP-arbitrated labels.
- **Experiment:** Author ~10-15 ASP rules covering TanitAD's known ODD (no maps/traffic-lights, so
  scope to car-following, lane-keeping, basic yielding) and use them purely as an *offline* label
  generator for tactical manoeuvre supervision (inductive learning target), not as a runtime solver
  (keeping inference latency at the existing budget). **Outcome A:** ASP-derived labels correlate with
  better closed-loop tactical behaviour than the current curvature-relative labels → adopt as a label
  source. **Outcome B:** ASP labels are frequently inapplicable (rule preconditions rarely fire given
  PhysicalAI's actual scenario distribution, which per the operating standard has no maps/junctions
  labelled) → confirms the "no strategic topology in this corpus" finding already logged, redirect the
  ASP idea toward AlpaSim-generated scenarios instead of PhysicalAI.

### B6. Formalized traffic rules — STL/LTL/MTL monitors, CommonRoad, and the legal-logical survey
- **What:** Traffic rules (right-of-way, speed limits, stopping duration) formalized as Signal/Linear/
  Metric Temporal Logic formulas, evaluated as runtime or offline *monitors* over a trajectory —
  giving a graded (STL robustness value) or Boolean (LTL) compliance signal, and CommonRoad-based
  tooling for rule-compliant trajectory repair via SMT + reachability analysis.
- **Evidence:** PUBLISHED —
  (i) "Integrating Legal and Logical Specifications in Perception, Prediction, and Planning for
  Automated Driving: A Survey of Methods," arXiv:2510.25386 (Oct 2026) — best single entry point,
  UNVERIFIED beyond the survey framing (fetch blocked).
  (ii) "Traffic-Rule-Compliant Trajectory Repair via Satisfiability Modulo Theories and Reachability
  Analysis," arXiv:2412.15837 — couples SMT + reachability for **provable** safety guarantees on
  repaired trajectories (this is a genuine, not merely statistical, guarantee, conditional on the
  correctness of the SMT rule encoding and the reachability set).
  (iii) "Lexicographic Minimum-Violation Motion Planning using Signal Temporal Logic,"
  arXiv:2604.20428 — handles the realistic case where **not all rules can be satisfied
  simultaneously** (need is unavoidable in dense traffic), ranking violations lexicographically by
  priority rather than failing outright.
  Maturity: PROVEN as a formalism (CommonRoad's rule-monitor tooling is mature, multi-year, widely
  cited), PROMISING as an online/real-time component at TanitAD's latency budget.
- **Pain points:** **P8** (no formal safety layer known) — this is a direct, literal answer: STL/LTL
  rule monitors ARE a formal safety layer, and lexicographic minimum-violation planning is exactly the
  right formalism for TanitAD's real driving data (which will contain unavoidable-rule-tension
  scenes, e.g. must cross a solid line to pass a stopped vehicle).
- **Admissibility:** Monitor inputs are ego kinematics + other-agent state, both vision/`obstacle.
  offline`-derivable; clean.
- **Cost:** ~5-8 eng-days to hand-encode ~10-20 STL rules over TanitAD's known ODD (car-following,
  lane-keeping, basic yielding — no traffic lights/junctions in the label set, so rule scope must
  match). ~0 A40-days for offline monitoring (STL robustness computation is CPU-cheap, sub-ms per
  trajectory for short horizons).
- **Experiment:** Run an offline STL monitor (not yet in the control loop) over the existing v1
  flagship's AlpaSim closed-loop rollouts, scoring each of the 12 scenarios' trajectories for rule
  robustness (margin to violation), independent of the pass/fail AlpaSim verdict. **Outcome A:** STL
  robustness is strongly correlated with, or even predictive ahead of, AlpaSim pass/fail → adopt as
  a cheap proxy metric for the tactical/**strategic** eval families the BINDING eval-metrics rule
  requires. **Outcome B:** weak correlation → AlpaSim's pass/fail criterion and rule-robustness are
  measuring genuinely different things (worth knowing either way, since it bears on which one to
  optimize against).

### B7. Rulebooks — tiered/lexicographic rule hierarchies for arbitration and reranking
- **What:** A **pre-order** (not total order) over rules, grouped into priority tiers (Safety ≻ Legal
  ≻ Road ≻ Comfort, or similarly named), used to arbitrate between candidate trajectories: a
  lower-tier violation only matters when no candidate satisfies the higher tier.
- **Evidence:** PUBLISHED —
  (i) **RECTOR**, arXiv:2605.25095 (2026) — post-generation reranking layer scoring candidates against
  a 4-tier rulebook (Safety≻Legal≻Road≻Comfort) via differentiable proxies + scene-conditioned
  applicability. **On WOMD validation_interactive (43,219 augmented instances), Protocol B, 28-rule
  proxy catalog: rule-aware reranking cuts Safety+Legal violations from 28.58% to 20.42%.**
  (ii) **W-SQP** (weighted tiered-slack NMPC, arXiv:2607.10975) — compiles 9 rule families into a
  4-tier shared-slack nonlinear program solved online with CasADi/IPOPT.
  (iii) "Receding Horizon Planning with Rule Hierarchies," arXiv:2212.03323 — canonical example
  hierarchy: No-Collision (1) ≻ no solid-line crossing (2) ≻ no dashed-line crossing (3) ≻ stop-sign
  duration (4) ≻ end-of-horizon lane orientation (5) ≻ minimum speed (6).
  Maturity: PROVEN (RECTOR's number is a real, reproducible benchmark result on a standard dataset,
  not a search-snippet approximation — WOMD is public and the 28.58%→20.42% figure is concrete
  enough to be independently checkable).
- **Pain points:** **P1 directly** — RECTOR *is* a selection-layer fix: it takes candidates a learned
  planner already generates (TanitAD's tactical head already produces multiple anchors in some arms)
  and **reranks by a tiered rulebook**, exactly matching "good candidates, bad choice." **P8** — a
  named, published rule hierarchy is a concrete starting point for the "no formal safety layer known"
  gap.
- **Admissibility:** Clean — reranking sits after candidate generation, over vision/kinematics-
  derivable rule predicates, no privileged signal needed and no interaction with the situation-
  classifier firewall.
- **Cost:** ~4-6 eng-days for a first tiered-rulebook reranker (differentiable-proxy rule scoring is
  the RECTOR-specific technique, achievable with simple geometric proxies at first pass); ~0-2
  A40-days (reranking a fixed candidate set is cheap; only the differentiable-proxy *tuning* benefits
  from GPU).
- **Experiment:** Reimplement a small (5-8 rule) tiered rulebook over TanitAD's tactical head's
  existing candidate anchors (where the arm produces multiple anchors — the v4/v1.5-style
  `FlagshipV15Head`/DiffusionDrive-style decoders per `MODEL_REGISTRY.md`), rerank instead of taking
  the argmax, and measure closed-loop AlpaSim pass rate. **Outcome A:** pass rate improves →
  directly promotes RECTOR-style reranking into the flagship's deployed selection path (this is the
  single cheapest, most direct P1 fix identified in this whole report — no retraining, CPU-only
  rerank). **Outcome B:** no improvement → the existing candidates already respect the rules and the
  selection failure is in the *scoring* the network already does, not in a missing rule check —
  narrows P1 to a value-estimation problem.

### B8. STL/rule-guided diffusion planning
- **What:** Uses Signal Temporal Logic specifications as a *guidance signal* inside a diffusion-based
  trajectory planner/generator (classifier-guidance-style, or via projected sampling), so generated
  trajectories are steered toward rule satisfaction during the denoising process itself, not only
  checked after the fact.
- **Evidence:** PUBLISHED — "Generalizable Multi-Agent Planning from Signal Temporal Logic
  Specifications via Diffusion," arXiv:2608.29490 (2026); "Diverse Controllable Diffusion Policy with
  Signal Temporal Logic," arXiv:2503.02924; "Diffusion Forcing Planner: History-Annealed Planning
  with Time-Dependent Guidance for Autonomous Driving," arXiv:2606.11019 (CVPR 2026 poster). Note:
  optimization-based STL planners are flagged in the search results themselves as struggling to scale
  with agent count — a genuine, acknowledged limitation, not hype-washed. Maturity: PROMISING (STL
  guidance for diffusion is architecturally proven; driving-specific closed-loop numbers UNVERIFIED,
  fetch blocked).
- **Pain points:** **P1** — if TanitAD's tactical/operative decoders move toward a diffusion-style
  multi-anchor decoder (already true for some arms per `MODEL_REGISTRY.md`'s `FlagshipV15Head`
  DiffusionDrive-style heads), STL guidance is a natural drop-in that unifies B6's rule-monitor idea
  with the *generation* process rather than only post hoc reranking (B7).
- **Admissibility:** Clean (same predicate inputs as B6/B7).
- **Cost:** ~10-15 A40-days, ~8-10 eng-days (requires STL-robustness-as-gradient, which needs a
  differentiable STL robustness formulation — a known but nontrivial engineering piece).
- **Experiment:** On an existing diffusion-style tactical decoder arm, add STL-robustness gradient
  guidance for 2-3 rules (stay-in-lane, speed envelope) during sampling, compare closed-loop AlpaSim
  pass rate and rule-violation rate against the same arm with post hoc B7-style reranking only.
  **Outcome A:** in-loop guidance beats post hoc reranking → justifies the extra engineering
  complexity. **Outcome B:** parity → prefer B7 (reranking) for its much lower engineering cost and
  greater auditability (a rejected/reranked candidate is easier to explain than a steered
  denoising trajectory).

### B9. Differentiable logic / T-norm constraints and safe-by-construction layers
- **What:** Encode logical safety constraints (e.g. "IF pedestrian-in-path THEN decelerate") as
  differentiable losses via t-norm fuzzy-logic relaxations (DL2-style) or as architecturally
  safe-by-construction layers (poset-structured networks).
- **Evidence:** PUBLISHED — DL2 ("Training and Querying Neural Networks with Logic," ICML 2019,
  foundational); "Exploiting T-norms for Deep Learning in Autonomous Driving," arXiv:2402.11362;
  "PoSafeNet: Safe Learning with Poset-Structured Neural Nets," arXiv:2601.22356 (2026); DRLSL
  (arXiv:2307.01316) restricts the RL action space via first-order-logic-derived masks. Maturity:
  PROVEN as a training technique (DL2 is 7 years old and widely used), PROMISING for driving-specific
  safety properties at TanitAD's scale.
- **Pain points:** **P1** — a soft logic penalty during training is a cheaper (train-time-only,
  zero inference cost) alternative/complement to B4's inference-time guard; the two are not mutually
  exclusive (train with the loss, still guard at inference for the residual violations a soft
  constraint doesn't fully eliminate).
- **Admissibility:** Clean (training-time loss, no runtime dependency).
- **Cost:** ~5-8 A40-days (retraining the tactical head with an added t-norm loss term), ~4 eng-days.
- **Experiment:** Add a t-norm-relaxed loss for "IF lead-agent within braking distance THEN
  decel-command" during a short fine-tune of the existing tactical head, measure longitudinal
  distance-keeping metrics (per the BINDING 4-metric-family rule, this is directly a **LONGITUDINAL**
  family metric) before/after. **Outcome A:** distance-keeping improves without ADE regression →
  cheap P2 win, stackable with B4's guard. **Outcome B:** ADE regresses more than distance-keeping
  improves → the soft constraint is fighting the main objective at our data scale; prefer B4's hard
  post hoc guard instead, which cannot regress ADE on compliant trajectories by construction.

### B10. LangProp — LLM-as-code-optimizer for driving policies
- **What:** Treats a driving policy's *code* (not weights) as the object being optimized: an LLM
  proposes code, it's run against a dataset of input-output pairs, exceptions/scores are fed back to
  the LLM, iterate — essentially evolutionary program synthesis with an LLM as the mutation operator.
- **Evidence:** PUBLISHED — "LangProp: A Code Optimization Framework using Large Language Models
  Applied to Driving," arXiv:2401.10314 (ICLR 2024 LLM Agents workshop, Oxford VGG). Demonstrated on
  Sudoku/CartPole/CARLA; LangProp-generated CARLA policies **outperformed those existing when the
  backbone GPT-3.5 was trained** (i.e., genuinely synthesized new capability, not memorized). Maturity:
  PROVEN as a technique, on toy-to-mid-complexity tasks; not demonstrated at TanitAD's target
  (real-corpus, sub-300M learned world model) scale — it optimizes *symbolic policy code*, which is a
  different object than TanitAD's learned latent planner.
- **Pain points:** **P8/F crossover** — LangProp-style synthesis is a natural way to *generate*
  candidate rule sets for B4's guard or B7's rulebook (have an LLM propose/refine the guard's rules
  against a held-out violation dataset), rather than a replacement for the learned planner itself.
- **Admissibility:** Clean if used to synthesize the rule/guard layer (train/design-time only).
- **Cost:** ~3-5 eng-days (LLM API calls + a CARLA/AlpaSim-style feedback loop; no GPU training
  needed for the policy itself, though the eval loop needs sim compute).
- **Experiment:** Use LangProp-style iterative refinement to author B4's safety-guard rule set
  against AlpaSim, instead of hand-authoring it. **Outcome A:** LLM-synthesized rules reach the same
  or better closed-loop pass rate as hand-authored rules, faster to iterate → adopt as the guard
  authoring workflow. **Outcome B:** LLM-synthesized rules are brittle/overfit to the 12 known AlpaSim
  scenarios → keep hand-authoring, but LangProp could still help as an accelerant for drafting.

### B11. Scenic + LLM-driven scenario generation (ChatScene, Chat2Scenic, PlannerForge, AGENTS-LLM)
- **What:** Scenic is a mature probabilistic programming language for specifying *distributions* over
  driving scenes/behaviours (official CARLA scenario language). A 2026 wave of LLM-based tools
  generates Scenic (or similar DSL) scenarios from natural language or from targeted "make this
  planner fail" objectives, closing the loop between language, formal scenario specification, and
  simulation.
- **Evidence:** PUBLISHED —
  Scenic itself: Fremont et al., PLDI 2019 (foundational, still the standard); **ScenicRules**
  (IEEE IV 2026) adds a Rulebook-style multi-objective/prioritized metric layer on top of Scenic
  scenarios. **ChatScene** (LLM→NL description→Scenic-snippet retrieval→assembled script, tested in
  CARLA). **Chat2Scenic**, arXiv:2607.14387 (2026): iterative RAG-based DSL scenario generation,
  **76.42% compilation success rate, 58.17% framework accuracy**. **PlannerForge**, arXiv:2609.08965
  (Sept 2026, very recent): LLM agents for scenario-based testing of motion planners specifically.
  **AGENTS-LLM**, arXiv:2507.13729: agentic LLM framework for challenging traffic scenarios. Maturity:
  PROVEN (Scenic itself, used across CARLA/Webots/Gazebo/X-Plane for years); PROMISING for the 2026
  LLM-generation layer (Chat2Scenic's numbers are concrete and recent).
- **Pain points:** **P10** (open-loop ADE blind to the dominant failure) and **P6** (closed-loop
  compounding error) — TanitAD's own closed-loop eval is AlpaSim with only 12 scenarios; this is a
  direct, low-risk lever to **grow the closed-loop scenario suite** (which is the eval axis the
  operating standard already flags as the one that actually caught v1's real failure mode, unlike
  open-loop ADE).
- **Admissibility:** N/A (test/eval infrastructure, not a runtime component).
- **Cost:** ~5-8 eng-days to stand up Scenic + a small LLM-scenario-generation loop targeting
  TanitAD's known ODD (car-following, lane-keep, cut-ins — scoped to what PhysicalAI-like scenes can
  contain, i.e. no junctions/traffic lights per the operating standard's settled absence-claim).
  ~0 A40-days (scenario generation and Scenic simulation are not A40-bound; only re-running the
  flagship's rollout inside them costs GPU, and that's already the AlpaSim cost model).
- **Experiment:** Use Chat2Scenic-style generation to produce 30-50 new Scenic scenarios targeting
  TanitAD's known **P2 longitudinal weakness** (aggressive lead-vehicle braking, stop-and-go), run the
  existing v1 flagship and flat REF-C through them. **Outcome A:** the new scenarios reproduce/sharpen
  the existing v1-vs-REF-C closed-loop gap (2/12 vs 8/12) at a larger n → strengthens confidence the
  gap is real and gives a bigger, more statistically powered closed-loop suite (directly useful for
  the episode-cluster bootstrap CI machinery already mandated). **Outcome B:** the gap doesn't
  replicate at larger n → the original 12-scenario AlpaSim result may itself be under-powered/
  scenario-cherry-picked, which would be an important (if uncomfortable) finding given how much
  program narrative rests on "REF-C wins closed-loop 8/12 vs 2/12."

---

## C. Knowledge injection

**Framing note specific to TanitAD:** map/OSM priors are foreclosed by the fact sheet — PhysicalAI-AV
has no GNSS (clip-local coordinates only, per the operating standard's settled finding), so
OSM map-matching on our traces is impossible regardless of what the literature offers here. This
section is therefore scoped to knowledge injection that does **not** depend on absolute
georeferencing: traffic-rule knowledge, retrieval over text/regulation, and distillation of large
model priors into small planners.

### C1. Retrieval-augmented driving explanation and decision-making (RAG-Driver family)
- **What:** Retrieve relevant exemplars/documents (past driving clips, traffic-rule text) at
  inference and condition generation/decision on them, rather than relying purely on parametric
  knowledge.
- **Evidence:** PUBLISHED — original **RAG-Driver**, arXiv:2402.10828 — maps video + control-signal
  embeddings into a shared retrieval space for in-context explanation generation, shown to generalize
  to new environments. 2026 successors: **VLADriver-RAG**, arXiv:2605.08133 (retrieval-augmented
  VLA); **RealDrive**, arXiv:2505.24808 (retrieval-augmented diffusion planning). Most relevant to
  TanitAD specifically: **"Driving with Regulation: Trustworthy and Interpretable Decision-Making for
  Autonomous Driving with Retrieval-Augmented Reasoning,"** arXiv:2410.04759 — retrieves regulation
  *text* to ground decisions, which is a direct, low-cost mechanism to connect a driving stack to the
  UN ADS / WP.29 regulatory requirements the programme already tracks in
  `Benchmarks & Eval/REGULATION_TRACE.md`. Maturity: PROVEN as a technique (RAG is mature and widely
  deployed generally), PROMISING specifically for driving-regulation grounding (fewer independent
  driving-specific replications).
- **Pain points:** **P8** (no formal safety layer) — regulation-retrieval doesn't by itself guarantee
  anything, but it is a cheap, auditable way to make the mapping from "what WP.29 requires" to
  "what the system actually checks" explicit and traceable, which the REGULATION_TRACE.md table is
  already trying to do by hand.
- **Admissibility:** Clean — retrieval is over a static text corpus (regulation, rules), not over any
  privileged runtime signal, and does not touch the situation classifier.
- **Cost:** ~3-5 eng-days (retrieval infra is lightweight; the WP.29 PDF + CommonRoad-style rule
  encodings are the corpus). ~0 A40-days.
- **Experiment:** Build a small retrieval index over the WP.29 GTR text + TanitAD's own rule
  encodings from B6/B7, and have B4's safety-guard cite the specific regulation clause each
  intervention maps to (this literally closes the "Open analysis task" the REGULATION_TRACE.md file
  itself flags — full paragraph-level requirement extraction). **Outcome A:** the retrieval-grounded
  citations pass a spot-check against the actual WP.29 text → promotes REGULATION_TRACE.md from
  "seeded from summary-level knowledge" to primary-sourced, closing a named open task. **Outcome
  B:** retrieval frequently mismatches the intended clause → the WP.29 PDF needs finer-grained
  chunking/structure before retrieval is reliable, which is itself useful information for whoever
  owns that document.

### C2. Distilling large-model/VLM priors into small planners (context: VERDI already known; this is the generalization)
- **What:** The general pattern behind VERDI (already screened in `External Anaysis.md`) and GraphPilot
  (A3): a large teacher (VLM, or in GraphPilot's case a scene-graph-conditioned trainer) supplies
  structure/knowledge during training only; a small deployed model is aligned to it via a projection +
  similarity loss, then runs standalone.
- **Evidence:** PUBLISHED, see B1 (Alpamayo-R1 CoC distillation lineage), A3 (GraphPilot), and
  `External Anaysis.md`'s own VERDI/LVLDrive summaries (already known — not re-derived here).
  Maturity: PROVEN as a pattern (multiple independent instantiations, 2024-2026).
- **Pain points:** **P4** (small data), **P3** (route/goal signal) — this is the general mechanism
  TanitAD should use for *any* future decision to inject outside knowledge (a larger open VLM's
  "common sense" about traffic, or a predicted-goal-point prior) without violating the vision-only
  inference rule, because the injection happens at train time via a discarded teacher.
- **Admissibility:** By construction admissible **only if** the teacher's knowledge is discarded
  post-training and the student's inference-time inputs stay vision-only — this is the one property
  that must be checked every time this pattern is proposed (per the BINDING admissibility-check
  rule: "could this have been computed from the situation classifier's output," and its stated
  generalization to any inference input).
- **Cost / Experiment:** Not a new idea to try in isolation — it's the *methodology* underlying B1's
  and A3's experiments above; no separate cost/experiment beyond those.

### C3. Traffic-rule knowledge bases as a first-class artifact (cross-ref B6/B7)
- **What:** Treating the formalized rule set itself (STL/LTL encodings, Rulebook tiers) as a versioned,
  auditable knowledge base independent of any one model — the same rule KB can supervise training
  (B9's soft losses), rerank at inference (B7), guard post hoc (B4), and ground retrieval (C1).
- **Evidence:** This is a synthesis point, not a single citation — see B6/B7/B9/C1's sources. Maturity:
  PROMISING as an integration strategy specific to TanitAD (the individual pieces are PROVEN/
  PROMISING; the *combination into one KB serving four roles* is a TanitAD-specific design choice,
  HYPOTHESIS as stated).
- **Pain points:** **P8** end-to-end — this is the connective tissue for the "guaranteed envelope"
  architecture proposed at the end of this report: one rule KB, four consumers.
- **Admissibility:** Clean (see B6/B7/B9/C1).
- **Cost / Experiment:** See B7's experiment (the reranker) as the first, cheapest consumer to build;
  the KB itself is the artifact produced by authoring B6's STL rules once, well-documented, rather
  than ad hoc per-experiment rule strings.

---

## D. Tool use for AD

### D1. Agent-Driver — LLM as scheduler over a perception/planning tool library
- **What:** Replaces the fixed perception→prediction→planning pipeline with an LLM agent that calls a
  versatile tool library via function calls, maintains a cognitive memory of common-sense/experiential
  knowledge, and reasons via chain-of-thought + task planning + self-reflection.
- **Evidence:** PUBLISHED — "A Language Agent for Autonomous Driving," arXiv:2311.10813. Reported
  **>30% improvement over SOTA driving methods on nuScenes**, plus better interpretability and
  few-shot ability. Maturity: PROVEN as a concept on nuScenes-scale open-loop benchmarks; not
  demonstrated at real-time/edge latency (LLM-in-the-loop at 10 Hz on Jetson Thor is not realistic
  with anything resembling a full LLM).
- **Pain points:** **P1** (selection) — the "reasoning engine" role is conceptually similar to B3's
  verifiable-consistency check, but implemented via explicit tool calls rather than a learned latent.
- **Admissibility:** The tool library itself (detection, prediction modules) must stay vision-derived;
  the "cognitive memory" of driving experience is a design-time/offline artifact, fine if not fed a
  situation-classifier output.
- **Cost:** Full agentic-LLM-in-the-loop is **not viable** at TanitAD's 10 Hz/Jetson-Thor budget
  (this is this idea's central limitation, stated plainly: LLM tool-calling agents run at seconds-
  per-decision latencies in the published work, roughly 5-6 orders of magnitude too slow for a 100 ms
  planning tick). The *pattern* (explicit tool calls + memory + reflection) is more useful as an
  **offline analysis/labeling tool** (cf. B1, B11) than as a runtime component. ~0 A40-days, ~3-5
  eng-days to stand up an offline Agent-Driver-style pipeline for scenario analysis rather than
  real-time control.
- **Experiment:** Use an Agent-Driver-style tool-calling pipeline **offline** to triage TanitAD's
  existing closed-loop failure logs (the 4/12 AlpaSim failures for v1): call a "was this avoidable"
  tool (cf. AUTOPILOT-VQA's category, already screened), a "which rule was violated" tool (B6/B7's
  KB), and a "what would REF-C have done differently" tool, producing a structured failure taxonomy.
  **Outcome A:** the taxonomy clusters failures into a small number of root causes → directly informs
  which of B4/B7/E1 to prioritize. **Outcome B:** failures are heterogeneous/idiosyncratic → argues
  for a broader closed-loop scenario suite (B11) before further root-causing.

### D2. Eureka — LLM-written reward functions
- **What:** Samples multiple candidate reward-function *code* snippets from an LLM given raw
  environment source + a task description, evaluates them via GPU-accelerated RL (IsaacGym), and
  iteratively refines — human-level reward design without a human loop.
- **Evidence:** PUBLISHED — "Eureka: Human-Level Reward Design via Coding Large Language Models,"
  arXiv:2310.12931. Maturity: PROVEN (widely cited, replicated result on dexterous manipulation
  benchmarks; not driving-specific).
- **Pain points:** **P2** (longitudinal / distance-keeping, 88.7% of the oracle gap) — reward-shaping
  for target-speed and headway is exactly the kind of dense, easy-to-get-subtly-wrong reward
  TanitAD's tactical/operative training would benefit from an LLM-assisted search over, if/when an
  RL or RL-fine-tuning stage is added to the pipeline (currently supervised, per the fact sheet).
- **Admissibility:** Reward *code* is a training-time artifact; clean.
- **Cost:** ~5-8 A40-days for an Eureka-style search loop over candidate longitudinal reward terms
  (each candidate needs a short RL/fine-tune run to score), ~4 eng-days for the search harness.
- **Experiment:** Only applicable once/if TanitAD adds an RL fine-tuning stage; **pre-registerable
  now**: if such a stage is added, run Eureka-style search specifically for the headway/TTC reward
  term (the named P2 gap) against a hand-authored baseline reward. **Outcome A:** LLM-searched reward
  beats hand-authored on the LONGITUDINAL metric family (BINDING eval rule) → adopt. **Outcome B:**
  parity or worse → hand-authored reward was already near-optimal for this narrow a target, and
  Eureka-style search is better spent elsewhere (e.g. the tactical selection reward).

### D3. LLM-driven scenario generation as a testing tool (cross-ref B11)
- **What:** Already covered under B11 (Scenic/ChatScene/Chat2Scenic/PlannerForge/AGENTS-LLM) — noted
  here because it is *also* squarely a tool-use pattern (an LLM agent calling a scenario-DSL compiler
  and a simulator as tools), not just a knowledge-injection idea. No duplicate entry; see B11 for
  full evidence/cost/experiment.
- **Pain points / Admissibility:** As B11.

### D4. Tool-augmented VLA safety surface — the attack side of tool use
- **What:** Reasoning-enabled VLA driving models that call tools or expose chain-of-thought create a
  new attack surface: adversarial inputs that manipulate the *reasoning* rather than the final action.
- **Evidence:** PUBLISHED — "ReasonBreak: Probing Vulnerabilities in Reasoning-Enabled Vision-
  Language-Action Models for Autonomous Driving," arXiv:2605.29114 (2026). Maturity: PROMISING
  (emerging red-teaming literature specific to reasoning VLAs, not yet a mature defense literature).
- **Pain points:** **P8** (safety/guarantees) — directly relevant as a **counter-argument** to
  over-trusting any chain-of-thought/tool-calling architecture above (B1, B3, D1) as a *safety*
  mechanism: an explanation that can itself be adversarially manipulated is not a safety guarantee
  unless the downstream action is independently checked (which is exactly why B4's post hoc guard —
  checking the *command*, not the *reasoning* — is more robust in kind than reasoning-consistency
  checks alone).
- **Admissibility:** N/A (a threat-model finding, not a component to admit).
- **Cost:** Reading/awareness cost only for now (~0.5 eng-days); a red-team pass against any future
  reasoning-based TanitAD component would be ~3-5 eng-days.
- **Experiment:** N/A directly, but this motivates a design principle worth pre-registering: **any
  safety mechanism TanitAD adopts from Section B (reasoning-consistency, CoC) must be paired with a
  Section F mechanism that checks the *action*, not just the *reasoning trace*, because the reasoning
  trace is a demonstrated attack surface.**

---

## E. Physics-grounded neural operators & dynamics

**Framing for this section:** the brief asks explicitly where these genuinely help driving vs. where
it's hype. My honest read after this pass: **differentiable kinematic decoders and Koopman-lifted
control are genuine and cheap; FNO/DeepONet-style operator learning is genuine for macroscopic
traffic-flow PDEs but a poor fit for TanitAD's ego-centric, single-camera, no-map setting; PINNs sit
in between (cheap regularizers, modest but real gains); Hamiltonian/Lagrangian nets are the closest
thing to hype for our use case specifically** (see E6 for why).

### E1. Differentiable Kinematic Bicycle Model (KBM) decoders — fully gradient-connected, not just post hoc
- **What:** Embeds a KBM (or similar low-order vehicle model) as the final decoder stage, with the
  network's output being the model's *inputs* (steering, acceleration) rather than raw waypoints, and
  the KBM's forward simulation participating in backprop end-to-end — as opposed to older approaches
  (KING, PlanTF-style) where a physical model generates candidates for a separate network to *score*,
  with no gradient flow through the physics.
- **Evidence:** PUBLISHED — this is the decoder used inside B5's neuro-symbolic framework
  (arXiv:2603.12421, Mar 2026): a decision-conditioned decoding mechanism turns high-level decisions
  into embeddings that constrain both the planning query and the KBM's initial velocity, with the KBM
  "fully embedded... and participat[ing] in system optimization through gradient backpropagation."
  Maturity: PROMISING (a clear, recent, working instantiation; not yet cross-validated by independent
  groups at scale the way BarrierNet or RSS have been).
- **Pain points:** **P2 (longitudinal) and lateral metrics together** — a KBM decoder makes speed,
  heading, curvature, and yaw-rate outputs *kinematically consistent by construction* (you cannot
  decode a physically inconsistent (speed, curvature, yaw-rate) tuple through a real bicycle model),
  which directly serves the BINDING four-metric-family rule's LATERAL family (curvature error,
  yaw-rate error are literally KBM state variables, not derived quantities).
- **Admissibility:** Clean — the KBM's parameters (wheelbase, etc.) are static physical constants, not
  learned from privileged signals.
- **Cost:** ~5-8 A40-days (swap the tactical/operative waypoint head's final layer for a KBM decoder,
  short fine-tune), ~4-5 eng-days (differentiable KBM implementation is well-documented, low risk).
- **Experiment:** Replace only the final decode step of TanitAD's tactical waypoint head with a
  differentiable KBM (network outputs (accel, steering) sequences, KBM integrates to waypoints),
  holding the encoder/predictor fixed, and compare the LATERAL metric family (curvature error,
  yaw-rate error, cross-track) against the current direct-waypoint decoder. **Outcome A:** LATERAL
  metrics improve at parity ADE → adopt broadly, this is close to a free win (small param cost,
  physically-grounded output, better auditability for a safety case per Section F). **Outcome B:** ADE
  regresses — the current unimodal/anchor decoder may already implicitly learn kinematic consistency
  from data at 13h scale, and the KBM's added inductive bias is a net constraint rather than a help
  (worth knowing, since it would suggest our current data regime doesn't need the physics prior).

### E2. Physics-guided Neural ODEs for trajectory prediction (PST-ODE, PhysORD)
- **What:** Neural ODEs whose vector field is constrained or regularized by known vehicle dynamics
  (not learned freely), combined with a causal-attention module for multi-agent interaction (PST-ODE)
  or explicitly hybridized with a physics-based motion model for off-road terrain (PhysORD).
- **Evidence:** PUBLISHED — "Physics-Guided Spatio-Temporal Neural ODE" (PST-ODE, published in a
  ScienceDirect venue, 2026) — ablations show physics-guided Neural ODE improves interaction reasoning
  and **reduces physically implausible trajectory drift** vs. a free-form Neural ODE. "PhysORD: A
  Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving," arXiv:2404.01596
  — combines a symbolic physics prior with a learned residual, off-road-specific (less relevant to
  TanitAD's on-road ODD but methodologically transferable). Maturity: PROMISING (ablation-level
  evidence within each paper; not yet a standard tool the way KBM decoders or Koopman-MPC are).
- **Pain points:** **P6** (closed-loop compounding error) — a continuous-time ODE formulation is a
  principled alternative to the current 20-sequential-discrete-step rollout that dominates planning-
  tick latency (per `MODEL_REGISTRY.md`, 83.7-96.7% of the tick), potentially allowing adaptive-step
  integration (few large steps when the scene is simple, more when it's not) instead of a fixed
  20-step schedule.
- **Admissibility:** Clean.
- **Cost:** ~10-15 A40-days (replacing the operative predictor's discrete rollout with a Neural-ODE
  formulation is a bigger architectural change than E1), ~8-10 eng-days.
- **Experiment:** Before a full swap, a cheaper probe: measure whether the *existing* discrete
  operative predictor's step-to-step residuals are well-approximated by a smooth ODE (fit a
  continuous vector field post hoc to the already-trained model's rollout trajectory and measure
  residual). **Outcome A:** residuals are small/smooth → the discrete rollout is already
  ODE-like, and adaptive-step integration (cheaper at inference, same accuracy) is a low-risk win
  worth the full E2 investment. **Outcome B:** residuals are large/non-smooth (e.g. near manoeuvre
  transitions) → the dynamics are genuinely discontinuous/hybrid (mode-switching), which argues
  against a plain Neural ODE and toward a hybrid/switched-system formulation instead — a different,
  larger research bet.

### E3. Koopman-operator-lifted control (linear-in-a-lifted-space MPC)
- **What:** Learns a (typically deep, data-driven) lifting of nonlinear vehicle dynamics into a
  higher-dimensional space where the dynamics become **linear**, enabling fast, interpretable MPC
  in the lifted space instead of solving a nonconvex optimization at each planning step.
- **Evidence:** PUBLISHED — "Learning Predictive Control with Deep Koopman Operators for Autonomous
  Vehicle Motion Planning," arXiv:2606.08136 (2026) — real-time motion planning under nonconvex
  constraints via a linear lifted-space MPC. "Physics-informed Deep Mixture-of-Koopmans Vehicle
  Dynamics Model," arXiv:2603.17416 (dual-branch encoder, distributed electric-drive trucks — a more
  specialized vehicle class than TanitAD's target but the *technique* transfers). "Vehicular
  Applications of Koopman Operator Theory — A Survey," arXiv:2303.10471 (good entry point). Maturity:
  PROVEN as control theory (Koopman operator theory itself is decades old and mathematically
  well-founded; MPC-in-lifted-space is a mature control technique), PROMISING specifically for the
  deep/data-driven lifting variant at automotive scale.
- **Pain points:** **P6 (latency)** — once lifted, the dynamics are linear, so the online optimization
  is a QP, not a nonconvex program — this is a genuine, not merely empirical, computational
  advantage, directly relevant to fitting inside TanitAD's 10 Hz/100 ms budget with margin for a
  safety layer (Section F) on top. **P2 (longitudinal)** — a linear model in the lifted space is
  exactly the setting where distance-keeping/speed-tracking controllers (LQR-style) are simplest and
  most analyzable.
- **Admissibility:** Clean (dynamics model, not a perception/situation component).
- **Cost:** ~10-12 A40-days (learning the lifting function needs a reasonable amount of
  trajectory data — TanitAD's 2376-episode corpus is plausibly enough since this is a low-dimensional
  dynamics-identification problem, not a perception problem), ~6-8 eng-days.
- **Experiment:** Fit a deep Koopman lifting to TanitAD's ego kinematics (speed, yaw-rate, acceleration
  history from the corpus, no vision needed for this specific sub-model) and compare an LQR controller
  in the lifted space against the current tactical policy's longitudinal behaviour on the
  LONGITUDINAL metric family. **Outcome A:** Koopman-LQR matches or beats the learned tactical
  policy's headway/TTC/speed-accuracy at a fraction of the parameter count and with a
  formally-analyzable (linear, eigenvalue-checkable) controller → strong candidate for the
  "guaranteed envelope" fallback controller in Section F (a Simplex-style baseline needs exactly this
  kind of analyzable controller). **Outcome B:** the learned tactical policy is substantially better
  → the nonlinearity/context-dependence of good longitudinal control at TanitAD's ODD exceeds what a
  Koopman-linear model captures, useful negative evidence before over-investing in linearization.

### E4. Physics-informed regularization for trajectory prediction (PINN-style, IDM term)
- **What:** Adds a physics-based regularization term to a standard trajectory-prediction loss — e.g.
  the Intelligent Driver Model (IDM) as an auxiliary loss for longitudinal behaviour — rather than
  hard-constraining the architecture.
- **Evidence:** PUBLISHED — "Physics-informed deep learning for robust trajectory prediction in
  automated driving," *Scientific Reports* (Nature), 2026 cycle (DOI-indexed via PubMed/PMC,
  `s41598-026-...`). Reports the IDM regularization term gives **faster convergence and error
  reduction up to 70% for selected cases**, and separately that PINN-style models with longitudinal
  interaction constraints remain robust with **80-90% missing-data rates** (MAE 0.91 m, MSE 2.17).
  Maturity: PROMISING (peer-reviewed, Scientific Reports is a legitimate but not top-tier venue;
  "up to 70% for selected cases" is a favorable-case number, not an average — treat the 70% figure as
  an upper bound, not a typical result).
- **Pain points:** **P2 (longitudinal), P4 (small data)** — the missing-data robustness result is
  particularly relevant to TanitAD's 13h/2376-episode budget: a physics regularizer that keeps
  working with 80-90% missing observations suggests physics priors buy sample efficiency, which is
  exactly what a small-data programme needs.
- **Admissibility:** Clean (IDM parameters are physical, not privileged).
- **Cost:** ~3-5 A40-days (adding one regularization term to an existing training run is cheap), ~2-3
  eng-days.
- **Experiment:** Add an IDM-consistency auxiliary loss to the operative predictor's longitudinal
  (speed) channel, fine-tune from the existing v1 checkpoint, measure the LONGITUDINAL metric family
  (target-speed accuracy, headway/TTC) with and without the term. **Outcome A:** improvement,
  especially on the tail (low-visibility/high-missing-observation frames) → cheap, low-risk P2 lever,
  compatible with continuing the existing checkpoint rather than a restart. **Outcome B:** no change
  → TanitAD's operative predictor may already implicitly capture IDM-like behaviour from 2376 episodes
  of real driving (unlike the papers' more data-constrained settings), so the regularizer is
  redundant — still useful to know before spending more effort here.

### E5. Neural operators (FNO/DeepONet) for traffic-flow / occupancy fields — honest scale mismatch
- **What:** Learn the *solution operator* of a PDE (e.g. the LWR traffic-flow conservation law) rather
  than a single solution instance, mapping sparse boundary/initial data to a full spatiotemporal field
  (density, flow, speed) at arbitrary query points.
- **Evidence:** PUBLISHED — original FNO-traffic paper, arXiv:2308.07051; "Mitigating Stop-and-Go
  Traffic Congestion with Operator Learning," arXiv:2411.05866; **PI-DeepONet for traffic state
  estimation**, arXiv:2508.12593 — integrates the traffic-flow conservation law + fundamental diagram
  directly into the operator, evaluated on NGSIM freeway data + a large-scale urban expressway in
  China, with the reported finding that **DeepONet shows physically consistent sensitivity to
  density/occupancy while baseline MLPs show negligible or erratic responses**. Maturity: PROVEN for
  macroscopic, multi-vehicle, sensor-network traffic-flow estimation (a genuinely different problem
  from ego-vehicle planning); UNVERIFIED/no evidence found for single-ego-vehicle occupancy-grid
  prediction from a single front camera, which is TanitAD's actual setting.
- **Pain points / honest assessment — this is the section's clearest "mostly not applicable" call:**
  FNO/DeepONet earn their keep when there is a genuine PDE (a conservation law over a *field* — traffic
  density over a road network, sensed by many vehicles/loop detectors). TanitAD has **one ego vehicle,
  one front camera, no multi-agent sensor network, and no map/road-network topology** (per the
  operating standard's settled finding: no lane graph, no junction annotation in PhysicalAI-AV). There
  is no macroscopic field for a neural operator to learn the solution operator *of* in our setting.
  Where this genuinely could help: if TanitAD ever adopts an occupancy-grid intermediate (Section A7),
  *evolving* that occupancy grid forward in time is closer to a PDE-like problem (occupancy has
  continuity properties), and an FNO/operator-learning approach to occupancy evolution is a fair,
  non-hype application — but that is downstream of first solving A7's harder problem (getting a metric
  occupancy grid from a single uncalibrated camera at all).
- **Admissibility:** N/A pending A7.
- **Cost:** Not recommended to pursue directly before A7 is resolved. If A7's cheap experiment
  (ZipDepth fusion) succeeds, an FNO-based occupancy-evolution follow-up would be ESTIMATED
  ~15-20 A40-days.
- **Experiment:** No standalone experiment proposed — **gate this idea behind A7's outcome.**

### E6. Hamiltonian/Lagrangian neural networks — the section's clearest "borderline hype for us" call
- **What:** Architecturally encode energy conservation (Hamiltonian) or the Euler-Lagrange structure
  (Lagrangian) directly into a learned dynamics model, guaranteeing physically-consistent (e.g.
  energy-conserving) rollouts by construction.
- **Evidence:** PUBLISHED — Hamiltonian Neural Networks (NeurIPS 2019, foundational); "Newtonian and
  Lagrangian Neural Networks: A Comparison Towards Efficient Inverse Dynamics Identification,"
  arXiv:2506.17994 (2026); "Learning Hamiltonian Dynamics at Scale," arXiv:2509.24627 (2026). Maturity:
  PROVEN for rigid-body/manipulator dynamics (robotics arms, single/multi-rigid-body systems);
  **UNVERIFIED and structurally awkward for a road vehicle**, whose relevant dynamics (tire-road
  friction, braking, driver/controller inputs) are **dissipative and externally forced**, not
  energy-conserving — a plain Hamiltonian/Lagrangian net's core guarantee (energy conservation) is the
  *wrong* invariant for a car, which sheds energy constantly (braking, drag, friction) and gains it
  from an external actuator (the engine/motor), not from its own conserved Hamiltonian. The papers
  that *do* extend this family to dissipative/contact settings ("Extending Lagrangian and Hamiltonian
  Neural Networks with Differentiable Contact Models") acknowledge this exact gap.
- **Pain points:** Weak fit to any of P1-P10 as a road-vehicle *ego-dynamics* model specifically,
  though there could be a narrower legitimate use (e.g. modeling a specific subsystem like suspension
  energetics) that is out of scope for a driving-policy programme.
- **Admissibility:** N/A (not recommended).
- **Cost / Experiment:** **Not recommended to pursue.** This is one of the report's explicit "hype for
  our use case" calls: the technique is real and well-evidenced for the domains it was built for
  (manipulators, energy-conserving mechanical systems), but the core physical assumption it encodes
  is close to the opposite of what a braking, drag-affected, externally-actuated road vehicle needs.
  Flagging this explicitly rather than including a padded experiment for completeness.

### E7. Flow-matched neural operators for continuous-time dynamics (CFO) — a 2026 methods watch
- **What:** Learns continuous-time PDE/dynamics operators via flow-matching (fit temporal splines,
  use finite-difference velocity estimates as flow-matching targets) instead of backpropagating
  through an ODE solver — avoiding Neural-ODE's main computational cost.
- **Evidence:** PUBLISHED — "CFO: Learning Continuous-Time PDE Dynamics via Flow-Matched Neural
  Operators," arXiv:2512.05297 (ICLR 2026). Maturity: SPECULATIVE for driving specifically (a general
  methods paper, no driving application demonstrated in what I could verify) but PROMISING as a
  technique — directly relevant to E2's proposed Neural-ODE replacement, since CFO's whole point is
  making continuous-time dynamics learning cheaper than the ODE-solver-in-the-loop approach E2 would
  otherwise need.
- **Pain points:** **P6 (latency)** — if E2's experiment (probe residual smoothness) comes back
  favorable, CFO-style flow-matching is a candidate *implementation* technique for the continuous-time
  predictor that avoids paying an ODE-solver's per-step cost, compounding with A4/WA-JEPA's
  flow-matching future-latent idea.
- **Admissibility:** Clean (methodology).
- **Cost:** Bundled with E2 — no separate cost, this is a preferred *implementation route* for E2 if
  E2's own gating experiment succeeds, not an independent experiment.
- **Experiment:** Gated behind E2's outcome; not independently pre-registered.

---

---

## F. Mathematical guarantees & runtime assurance

**What can actually be GUARANTEED vs. statistically bounded — stated up front, since the brief asks
for this explicitly:**

| Mechanism | What is GUARANTEED | Under which assumptions | What is only STATISTICAL |
|---|---|---|---|
| RSS | No-blame-for-collision under the model's own rules | Other agents obey the same kinematic bounds (max brake/accel) assumed by the model; sensing is perfect | Whether the *modeled* bounds match the real world (a leading vehicle that brakes harder than assumed breaks the guarantee) |
| Hard CBF (classical) | Forward invariance of the safe set | Exact, correct dynamics model; no actuation limits violated; correct relative degree | Robustness to model mismatch — is a soft/statistical property unless paired with robust-CBF variants |
| Differentiable CBF (BarrierNet) | Same as hard CBF, at each layer's QP | Same as above, PLUS the QP is solved exactly (it is, for these small QPs) | The *learned* part upstream of the safety layer is not guaranteed anything — only the filtered output is |
| HJ reachability | Exact safe/unsafe set membership, exact optimal control at the boundary | Known dynamics + known disturbance/action bounds; suffers the curse of dimensionality (classically ≤ 5-6 state dims tractable) | High-dimensional/learned approximations (DeepReach) trade the *exact* guarantee for a *statistically validated* approximation — this must be stated, not glossed over |
| Simplex / runtime assurance | Safety of the **verified baseline controller only**, switched to when needed | The switching logic itself is correct and the baseline really is safe | Whether the advanced (learned) controller is ever actually better — no guarantee on performance, only on the fallback |
| Conformal prediction | Marginal coverage at the chosen level (e.g. 90% of the time the true value is in the predicted set), **in expectation over the calibration distribution** | Exchangeability (or a stated relaxation of it) between calibration and deployment data | NOT a guarantee for any single trajectory/episode, and breaks under distribution shift unless adaptive/online variants are used |
| NN formal verification (α,β-CROWN) | Exact certification that a specific property holds for **all** inputs in a specified bounded region | The property and input region are precisely specified in advance; scales to millions of parameters for the properties VNN-COMP benchmarks (bounded local robustness, reachability of small nets) | Does **not** currently certify open-ended semantic properties ("detects all pedestrians") on a large ViT-scale perception backbone — this is the most commonly overclaimed guarantee in the popular literature and must not be confused with what VNN-COMP actually benchmarks |
| Statistical model checking / importance splitting | An estimate of rare-event probability with a stated confidence interval | Correctness of the simulation model as a proxy for reality (sim-to-real gap is not covered) | Always statistical by definition — the entire category |

### F1. RSS — Responsibility-Sensitive Safety
- **What:** A parametrized, technology-neutral formal model (five core rules — safe longitudinal/
  lateral distances, right-of-way, occlusion caution, avoid-if-possible) that defines a "safety
  envelope" independent of the AV's own planner; the planner can be anything, RSS only rejects
  proposed actions that would violate the envelope.
- **Evidence:** PUBLISHED — original Mobileye RSS formalization (Shalev-Shwartz, Shammah, Shashua,
  2017/2018, well-established prior knowledge, not independently re-verified this pass); formal
  verification/refinement/testing treatment: "Slow Down, Move Over: A Case Study in Formal
  Verification, Refinement, and Testing of RSS," arXiv:2305.08812. **2026 update, PUBLISHED:** in
  March 2026 TÜV SÜD issued a formal recommendation for Mobileye's Safety Management System for SAE
  Level 4, per Mobileye's own blog/press material (secondary source, not independently audited here).
  Maturity: **PROVEN** as a formal model (it is literally a mathematical proof structure, "verifiable
  without millions of miles of driving," per Mobileye's own framing) and increasingly PROVEN as an
  industry-accepted certification basis (the TÜV SÜD step).
- **Pain points:** **P8 directly** — RSS is the most mature, most externally validated answer to "no
  formal safety layer is known." **P1** — RSS is explicitly independent of the planner, so it composes
  with *any* choice TanitAD makes for the learned planner itself.
- **Admissibility:** Clean — RSS's inputs are relative positions/velocities/accelerations of the ego
  and nearby agents, all vision/`obstacle.offline`-derivable, no privileged signal, no interaction
  with the situation-classifier firewall.
- **Cost:** ~5-7 eng-days for a minimal longitudinal+lateral RSS-distance check (the two simplest of
  the five rules), ~0 A40-days (RSS distance checks are closed-form arithmetic, not learned).
- **Experiment:** Implement the RSS safe-longitudinal-distance rule only, run it as a passive monitor
  (not yet gating) alongside the existing v1 flagship on the AlpaSim closed-loop suite, and check how
  often the flagship's own trajectory would have violated it. **Outcome A:** violation rate is low and
  concentrated in the known-failure scenarios → RSS distance-check correctly flags the existing known
  problem (cheap confirmatory instrument, promote to an active guard per B4's pattern). **Outcome B:**
  violation rate is high even on passing scenarios → either RSS's parameters (max brake/accel bounds)
  need calibration to TanitAD's ODD, or the flagship is closer to the safety boundary more often than
  the pass/fail AlpaSim metric alone reveals (itself a finding worth having, tying into P10).

### F2. Control Barrier Functions — classical, and BarrierNet's differentiable/learned variant
- **What:** A CBF defines a safe set via a scalar function whose non-negativity is forward-invariant
  under a control law satisfying a derivative condition; classically this is solved as a QP filter on
  top of any nominal controller. **BarrierNet** embeds this QP as a differentiable layer, end-to-end
  trainable by gradient descent, so the safety layer can be learned/adapted jointly with the policy
  rather than hand-tuned, while keeping the QP's hard guarantee on the *final* output.
- **Evidence:** PUBLISHED — BarrierNet, *IEEE Transactions on Robotics* 2023 (Xiao et al.) —
  foundational, differentiable CBF-QP safety layers "guarantee safety" for the filtered output and
  "can be used in conjunction with any neural network-based controller." **Applied specifically to
  vision-based end-to-end driving:** "Differentiable Control Barrier Functions for Vision-based
  End-to-End Autonomous Driving," arXiv:2203.02401 — this is BarrierNet's own driving-specific
  instantiation, i.e. **directly on point for TanitAD's architecture shape** (vision in, safety-
  filtered control out). 2026 follow-on: "End-to-End Learning of Safe Optimal Feedback Control in
  High Dimensions with Control Barrier Function Layers," arXiv:2607.20674. Maturity: **PROVEN**
  (peer-reviewed IEEE T-RO, multi-year adoption, and a driving-specific instantiation already exists).
- **Pain points:** **P1 (selection)** and **P6 (compounding error)** — a CBF-QP layer is a
  *hard* constraint on the final command (unlike B4's discrete replace-if-violated guard, a CBF-QP
  finds the *closest feasible* command via a small convex optimization, which is smoother and doesn't
  discard the network's intent as bluntly). **P8** — genuinely GUARANTEEs forward invariance of the
  safe set, conditional on the dynamics model and actuation bounds being correct (see the table
  above) — this is one of the few mechanisms in this whole report where "guarantee" is not loosely
  used.
- **Admissibility:** Clean — same input class as RSS/B4 (kinematics + nearby-agent state).
- **Cost:** ~6-10 A40-days (the QP layer itself is cheap; most of the cost is defining the barrier
  function(s) for TanitAD's ODD — road-edge, lead-vehicle, and validating relative-degree conditions
  hold for the chosen state representation), ~8-10 eng-days (differentiable QP layers, e.g. via
  cvxpylayers or qpth, are mature but need careful integration).
- **Experiment:** Attach a single-barrier (road-edge / off-road) CBF-QP layer to the deployed v1
  flagship's waypoint output (same latency budget concern as B4 — a small QP is sub-ms, easily fits
  inside the 18.75-27.87ms optimized planning tick per `MODEL_REGISTRY.md`), re-run AlpaSim.
  **Outcome A:** off-road closed-loop failures (the exact failure mode already measured for v1) drop
  to near-zero **with a mathematical forward-invariance argument**, not just an empirical improvement
  → this becomes the leading candidate for the "guaranteed envelope" architecture's core safety layer.
  **Outcome B:** the QP is frequently infeasible (nominal command too far from the safe set to find a
  nearby feasible one) → indicates the *upstream* planner is proposing commands too far from safe too
  often, which is itself important evidence that the fix needs to happen earlier in the stack (B3/B7
  selection), not only at the final filter.

### F3. Hamilton-Jacobi reachability + DeepReach — exact where tractable, approximate at scale
- **What:** Computes the exact set of states from which a system can avoid (or is forced into) an
  unsafe outcome under worst-case disturbance, via solving a Hamilton-Jacobi-Isaacs PDE; DeepReach
  replaces the classical grid-based PDE solver (exponential in state dimension) with a neural PDE
  solution, trading the *exact* guarantee for a *learned, statistically-validated* approximation that
  scales to higher dimensions.
- **Evidence:** PUBLISHED — DeepReach (Bansal & Tomlin, ICRA 2021, foundational, well-established).
  2026 activity: "Gradient-Free Neural Hamilton-Jacobi Reachability for Scalable Safety-Critical
  Control," arXiv:2609.14087 (**Sept 2026, very recent**) — targets exactly reachability's classical
  scalability bottleneck. "HJ Reachability-Based Safe Reinforcement Learning for Emergency Collision
  Avoidance," arXiv:2606.15311. "Extending and Unifying the Fundamental Tasks of HJ Reachability
  Analysis," arXiv:2608.18060. Maturity: **PROVEN** for low-dimensional state (classical HJ
  reachability, exact, used operationally e.g. in airspace collision avoidance for decades);
  PROMISING for the neural/scaled variants at automotive-relevant dimension.
- **Pain points:** **P8** — HJ reachability is the other major formal-guarantee family besides
  CBFs, and its guarantee (exact safe-set membership + the actual optimal safe control at the
  boundary) is arguably *stronger* than a CBF's (a CBF needs a barrier function to be hand-designed
  well; reachability computes the maximal safe set directly, if it's tractable). **P6** — reachability
  sets can be precomputed **offline**, so the *online* cost is a cheap lookup/gradient evaluation, not
  a per-step optimization, which is attractive for the 10 Hz budget.
- **Admissibility:** Clean.
- **Cost:** Precomputing an HJ value function for TanitAD's ego + one lead-agent (a 2-4 dimensional
  relative-state reachability problem — tractable *classically*, no neural approximation needed at
  this dimension) is ESTIMATED ~3-5 eng-days + modest CPU compute (no A40 needed for a 2-4D grid).
  Extending to multi-agent or higher-fidelity dynamics would need DeepReach-style neural
  approximation, ESTIMATED ~10-15 A40-days.
- **Experiment:** Precompute a classical (grid-based, exact, no neural approximation) 3D HJ
  reachability set for ego-vs-single-lead-vehicle longitudinal safety (relative position, relative
  velocity, ego acceleration bound), use it as an online lookup-based safety filter, and compare
  against F2's CBF-QP on the same AlpaSim scenarios. **Outcome A:** HJ reachability and CBF-QP give
  similar practical safety outcomes → prefer whichever is cheaper to maintain/extend (likely the
  CBF-QP, since reachability's curse of dimensionality bites hard the moment more than 1-2 agents
  matter). **Outcome B:** HJ reachability catches near-boundary cases the CBF barrier function (which
  is hand-designed and may be conservative or, worse, non-conservative in the wrong place) misses →
  worth the extra investment for the highest-stakes longitudinal scenarios specifically (car-following
  at speed, which per the fact sheet is 88.7% of the oracle gap).

### F4. Simplex / runtime-assurance architectures — the general pattern the "guaranteed envelope" is built on
- **What:** Control authority defaults to an unverified, high-performance "advanced" controller (the
  learned flagship), but a verified, provably-safe "baseline" controller and a decision module
  monitor safety-relevant conditions and can **switch** authority to the baseline when the advanced
  controller's proposed action would be unsafe. Classical Simplex treats the advanced controller as a
  pure black box; newer variants allow bounded, formally-justified information flow *from* the
  learned component *to* the safety monitor.
- **Evidence:** PUBLISHED — the Black-Box Simplex Architecture (arXiv:2102.12981, foundational).
  **Simplex-Drive** (IEEE conf., 2022, arXiv-indexed ~2109.13446): DRL advanced controller +
  Velocity-Obstacle baseline with **provable safety guarantees**, verified mode-management switching.
  **Synergistic Simplex**, arXiv:2605.08190 (2026): the key 2026 advance — allows the safety monitor
  to *use* ML outputs (normally prohibited in classical Simplex, which must treat the learned
  component as untrusted), with a **formal derivation of the conditions under which this preserves
  safety** — i.e. a principled way to be less conservative than black-box Simplex without giving up
  the guarantee. **Mission-Level Runtime Assurance Framework for Autonomous Driving,**
  arXiv:2606.06996 (2026) — driving-specific. Maturity: **PROVEN** as an architecture pattern
  (decades of aerospace/CPS runtime-assurance deployment; Simplex-Drive specifically demonstrates the
  provable-baseline property for a driving-shaped problem).
- **Pain points:** **P8 end-to-end** — this is the architectural *frame* the whole "guaranteed
  envelope" section below is built around: TanitAD's learned flagship is the advanced controller,
  never itself verified; a small, genuinely verifiable baseline (a Koopman-LQR car-following
  controller from E3, or a simple lane-centered PD controller) is the Simplex baseline; F2's CBF-QP
  or F1's RSS check is the switching/monitor logic. **P9 (few A40s)** — critically, this pattern
  **does not require the flagship itself to ever be formally verified** (which would be intractable
  at any interesting param count per F5 below) — only the small baseline needs a proof, which is
  cheap.
- **Admissibility:** Clean — the switching logic's inputs are the same kinematics/agent-state class as
  F1/F2.
- **Cost:** ~10-15 eng-days for a first Simplex wrapper (baseline controller + switching logic +
  integration with the existing flagship's output), ~5-8 A40-days if the baseline itself needs any
  learning component (Koopman lifting, E3).
- **Experiment:** Wrap the deployed v1 flagship in a minimal Simplex architecture with a
  hand-designed baseline (constant-time-headway car-following + lane-centering) and a switch
  condition = F2's CBF value crossing a threshold; run the full AlpaSim closed-loop suite.
  **Outcome A:** pass rate approaches or exceeds flat REF-C's 8/12 **while retaining the flagship's
  behaviour on the 8 scenarios it already handles well** → this is the core validation of the
  "guaranteed envelope > either component alone" thesis this report closes with. **Outcome B:** the
  switch triggers so often that the system is effectively running the (crude, hand-designed) baseline
  most of the time → the baseline itself needs to be less conservative (motivating Synergistic
  Simplex's ML-informed monitor, or a better baseline via E3's Koopman-LQR rather than a hand-tuned
  PD controller).

### F5. NN formal verification (α,β-CROWN / VNN-COMP) — what scale actually means here
- **What:** Certifies that a specified property (typically: output stays within bounds, or a decision
  doesn't change) holds for **every** input in a bounded region (e.g. an L∞ ball around a test image),
  via sound bound-propagation + branch-and-bound.
- **Evidence:** PUBLISHED — α,β-CROWN has won **VNN-COMP 2021 through 2025** (5 consecutive years,
  github.com/Verified-Intelligence/alpha-beta-CROWN), described by its own documentation as scaling
  "to relatively large convolutional networks (e.g., millions of parameters)" on benchmarks including
  an aerospace-collision-avoidance net (`collins-aerospace-benchmark`) and a driving-relevant one
  (`cctsdb-yolo-2023`, a traffic-sign/object-detection-shaped benchmark). VNN-COMP 2026 results were
  not yet available as of this search (competition runs later in the year). Maturity: **PROVEN** as a
  verification *tool* (5-year winning streak, actively maintained, GPU-accelerated); **the scale claim
  needs a precise reading, not a loose one.**
- **The precise reading (this is the section's most important nuance-check):** "Scales to millions of
  parameters" refers to *specific benchmark properties* — typically local robustness (small input
  perturbation doesn't flip a bounded output) on convolutional classifiers/detectors, or reachability
  of small control networks (ACAS Xu-style, a handful of layers). It does **not** mean one can certify
  an open-ended semantic property ("this network correctly perceives all pedestrians") on a
  ViT-scale backbone the size of TanitAD's own 87M-parameter encoder. TanitAD's encoder is within the
  *parameter-count* range α,β-CROWN has verified nets at, but the *properties* that matter for driving
  safety (semantic perception correctness, not just local output-bound robustness) are not the
  properties VNN-COMP benchmarks certify. **This is the single most commonly overclaimed guarantee in
  the public discourse around "provably safe AI," and this report explicitly does not repeat that
  overclaim** — see the "what is hype" section below.
- **Pain points:** **P8**, narrowly: verification is realistically applicable to TanitAD's *small*
  components — a CBF barrier-function network (if learned rather than hand-designed), a tiny
  fallback/baseline controller (F4's Simplex baseline), or a bounded-robustness property of the
  tactical head's manoeuvre classifier (e.g. "a small perturbation to the input never flips
  lane-keep→lane-change") — **not** the full encoder-to-trajectory pipeline.
- **Admissibility:** N/A (a verification tool, not a runtime component).
- **Cost:** ~5-8 eng-days to set up α,β-CROWN against a small, well-scoped property (e.g. the
  Simplex-baseline controller's local robustness, or a bounded-input-perturbation property of the
  5-way tactical manoeuvre classifier). ~0 A40-days (verification is typically run once, offline,
  design-time — though GPU-accelerated for speed).
- **Experiment:** Formally verify a **bounded local-robustness property** of the F4 Simplex baseline
  controller (a small network, if it has any learned component) or of the tactical manoeuvre
  classifier alone: "for all inputs within ε of a validation-set input, the manoeuvre decision does
  not flip." **Outcome A:** verified for a meaningful ε → a genuine, citable formal-verification
  result for the safety case (Section F9), scoped honestly to what it actually covers. **Outcome B:**
  verification fails / times out even at this small scale → informative about which of TanitAD's
  small components are actually verification-tractable, narrowing future safety-case claims to what's
  achievable rather than aspirational.

### F6. Conformal prediction for forecasting and planning — Lindemann's line and its 2026 successors
- **What:** A distribution-free calibration technique that converts any point predictor (of a lead
  agent's future trajectory, of the ego's own imagined rollout error, etc.) into a **prediction
  region** with a marginal coverage guarantee, using only exchangeability between calibration and
  deployment data — no assumption on the predictor's correctness or the data's distribution shape.
- **Evidence:** PUBLISHED — foundational driving/planning line: Lars Lindemann et al., "Safe Planning
  in Dynamic Environments using Conformal Prediction," arXiv:2210.10254 — MPC using CP-calibrated
  prediction regions, **provably safe with a user-defined probability**, compatible with any
  trajectory predictor (RNN/LSTM) with no assumption on the true trajectory distribution. Successor:
  Dixit, Lindemann et al., "Adaptive Conformal Prediction for Motion Planning among Dynamic Agents,"
  PMLR v211 (arXiv:2212.00278). **2026 successor directly relevant here:** "Barrier Function
  Conformal Safety Clearance Certification with CVaR for Driving Trajectory Selection,"
  arXiv:2608.26533 (Aug 2026) — **fuses CBF + conformal prediction + CVaR risk** specifically for
  trajectory *selection* (i.e., P1). Also: "Safe, Out-of-Distribution-Adaptive MPC with Conformalized
  Neural Network Ensembles," arXiv:2406.02436 — handles OOD, not just in-distribution coverage.
  Maturity: **PROVEN** (Lindemann's line is a mature, multi-year, widely-cited body of work with a
  real mathematical guarantee — marginal coverage under exchangeability is a clean, checkable
  theorem, not a loose claim).
- **Pain points:** **P1 (selection)** — the 2608.26533 fusion paper is literally "conformal +
  barrier + selection," i.e. this specific 2026 paper is doing exactly what TanitAD needs for its
  named #1 pain point. **P7 (imagination anti-calibration) — this is the most direct, most rigorous
  answer to P7 in this entire report:** conformal prediction is *precisely* a calibration technique,
  and applying it to TanitAD's H15 imagination module's own rollout error (calibrate: "how often is
  the imagined rollout within X of the true continuation") would convert an anti-calibrated
  confidence signal into one with an actual, checkable coverage guarantee.
- **Admissibility:** Clean — CP calibration uses held-out *ground-truth outcomes* (available offline,
  from the training corpus), not any privileged runtime signal; the calibrated predictor at inference
  uses only the same inputs the underlying predictor already used.
- **Cost:** ~4-6 A40-days (computing nonconformity scores over a held-out split of the existing
  corpus, no retraining needed — CP wraps an existing predictor), ~5-6 eng-days.
- **Experiment:** Calibrate a conformal prediction region for TanitAD's H15 imagination module's
  rollout error, using a held-out slice of the 2376-episode training corpus (respecting the sacred
  parity split) as the calibration set, and check whether the resulting prediction-region width
  correlates with actual closed-loop risk (does a wide CP region on a given scene predict an AlpaSim
  near-failure?). **Outcome A:** yes → TanitAD gains a **mathematically-grounded** (not just
  empirically-tuned) imagination-confidence signal, directly resolving P7's "anti-calibrated"
  diagnosis with a real coverage guarantee, and this becomes a leading Section-F component for the
  envelope (a wide CP region = trigger F4's Simplex switch). **Outcome B:** CP region width doesn't
  track risk → the miscalibration is not a simple "confidence too high/low" problem fixable by
  rescaling (which is all marginal CP does) but a structural one (the predictor is confidently wrong
  in specific *scenarios*, which argues for conditional/adaptive CP — group-conditional calibration by
  scenario type — as the next thing to try before giving up on the calibration approach.

### F7. Statistical model checking, importance splitting, and rare-event simulation
- **What:** Estimates the probability of a rare safety-critical event (e.g. a specific rule
  violation or near-collision) via adaptive importance splitting rather than naive Monte Carlo, which
  would need prohibitively many rollouts to see enough rare events to estimate their rate precisely.
- **Evidence:** PUBLISHED — "Adaptive Splitting of Reusable Temporal Monitors for Rare Traffic
  Violations," arXiv:2405.15771 — combines temporal-logic monitors (B6) with splitting for rare-event
  estimation. The **PEGASUS Project** (German landmark initiative) — structured scenario-based safety
  validation pipeline: scenario derivation from real-world data → statistical parameterization →
  simulation → coverage-driven evaluation (this is process/methodology, PROVEN and industrially
  adopted in Germany, not a single paper). "Testing Rare Downstream Safety Violations via Upstream
  Adaptive Sampling of Perception Error Models," arXiv:2209.09674. Maturity: **PROVEN** as a technique
  (importance splitting for rare-event SMC is decades-old, mathematically well-founded); PROMISING
  specifically combined with temporal-logic traffic-rule monitors (a more recent, driving-specific
  synthesis).
- **Pain points:** **P8/P10** — TanitAD's closed-loop eval suite is currently only 12 AlpaSim
  scenarios (B11's concern); rare-event simulation via importance splitting is the statistically
  correct way to *estimate a violation rate* rather than just observe pass/fail on a small fixed set,
  which directly strengthens the evidence base needed for any safety-case claim (F9).
- **Admissibility:** N/A (an evaluation methodology).
- **Cost:** ~6-10 eng-days (implementing adaptive splitting on top of AlpaSim requires control over
  the simulator's initial-condition sampling, which needs to be checked for feasibility). ~5-10
  A40-days (each splitting stage still needs simulation rollouts, though far fewer than naive Monte
  Carlo for the same rare-event precision — that is the whole point of the technique).
- **Experiment:** Apply adaptive importance splitting to estimate the rate of TanitAD's known
  off-road/longitudinal closed-loop failure mode under a *parametrized* scenario distribution (e.g.
  lead-vehicle deceleration magnitude as the splitting parameter) rather than the fixed 12-scenario
  suite. **Outcome A:** the estimated failure rate as a function of scenario severity gives a smooth,
  interpretable curve (e.g. "failure probability crosses 10% once lead-vehicle deceleration exceeds
  X m/s²") → this is a genuinely new, quantified, publication-grade safety characterization, an order
  above "8/12 vs 2/12." **Outcome B:** the failure rate is highly non-monotonic/noisy in the
  splitting parameter → suggests the failure mode is not simply "harder scenario = more failures" but
  depends on some other latent factor (worth identifying, e.g. via B1's failure-taxonomy approach).

### F8. Safety standards in 2026 — ISO 26262, ISO 21448 SOTIF, UL 4600, ISO/PAS 8800
- **What:** Four complementary standards covering, respectively: systematic/random hardware-software
  faults (26262), performance limitations absent any fault — "correct but inadequate" behaviour
  (21448/SOTIF), a goal-based structured safety-case argument for autonomous products generally
  (UL 4600), and AI/ML-specific safety-related risks — bias, robustness, generalization, malfunction
  from AI specifically (ISO/PAS 8800).
- **Evidence:** PUBLISHED —
  **ISO/PAS 8800:2024**, "Road Vehicles — Safety and Artificial Intelligence" (published December
  2024, still current/active in 2026 per multiple certification-body sources: UL Solutions, SGS, TÜV
  Rheinland). Explicitly designed to be used **by tailoring applicable clauses from ISO 26262-4, -6,
  and -8** rather than replacing them — i.e. it's a bridge/extension standard, not a standalone
  replacement. Developed by ISO/TC22/SC32/WG14 across 17 countries.
  **UL 4600**, Edition 3 (March 2023, still the current edition as of this search; ongoing technical-
  committee work through April 2025/2026 on AI-specific prompts within it) — the safety-case *format*
  (claims + argument + evidence) most directly applicable to a learned, hard-to-formally-verify
  planner, since it is explicitly goal-based/technology-agnostic rather than prescribing specific
  techniques.
  **ISO 21448:2022 (SOTIF)** — the standard most relevant to TanitAD's actual failure mode: v1's
  closed-loop losses are off-road/longitudinal *without a component fault* (the network functions
  "correctly" by its own training objective, but the resulting behaviour is inadequate) — this is
  the textbook SOTIF hazard category, not an ISO 26262 hazard.
  Maturity: **PROVEN/current** — these are live, adopted, actively-referenced industry standards in
  2026, not research proposals.
- **Pain points:** **P8, directly and comprehensively.** SOTIF is the *right standard* to frame
  v1's actual measured failure mode under (P6's compounding-error, off-road/longitudinal losses are
  "insufficient performance of the intended function," the SOTIF hazard definition almost verbatim).
  ISO/PAS 8800's tailoring-of-26262 approach gives a concrete checklist for what an AI-specific safety
  argument needs to additionally cover (bias, robustness, generalization) beyond a classical hardware/
  software safety case.
- **Admissibility:** N/A (process/documentation standards).
- **Cost:** ~5-8 eng-days to map TanitAD's existing evidence (gate results, `MODEL_REGISTRY.md`,
  AlpaSim closed-loop results, the REGULATION_TRACE.md table) against SOTIF's and ISO/PAS 8800's
  specific clause structure — this is documentation/gap-analysis work, not research or GPU work.
- **Experiment:** N/A in the usual sense (this is a standards-mapping exercise, not an empirical
  test) — but it is pre-registerable as a **gap analysis**: map every existing TanitAD safety-relevant
  artifact (D8 ODD-monitoring harness, H11 monitors, the D-gates) against SOTIF's specific hazard
  categories. **Outcome A:** most SOTIF categories already have a corresponding TanitAD artifact →
  the programme is closer to a SOTIF-structured safety case than it currently documents itself as
  being, and the gap analysis itself becomes a valuable artifact. **Outcome B:** major SOTIF
  categories (e.g. "insufficient robustness to reasonably foreseeable misuse") have no corresponding
  artifact at all → identifies concrete new work items for the safety case, more useful than a vague
  "we should be safer" observation.

### F9. Safety case *patterns* for learned/VLA-based planners specifically (2026)
- **What:** Reusable safety-case templates (claims + argument + evidence, UL-4600-style) tailored to
  the specific challenges of AI/learned components: evaluation without ground truth, dynamic model
  updates post-deployment, and threshold-based (not binary) risk decisions.
- **Evidence:** PUBLISHED — **"Safety Case Patterns for VLA-based driving systems: Insights from
  SimLingo,"** arXiv:2603.16013 (2026) — **the single most directly relevant paper in this whole
  report to "what does a credible safety case for a learned planner look like in 2026,"** since it is
  explicitly built around a VLA-based driving system, the same architecture family TanitAD sits in
  (a learned, largely-opaque, vision-conditioned planner). "A Structured Approach to Safety Case
  Construction for AI Systems," arXiv:2601.22773 (2026) — general (not driving-specific) AI safety-
  case template with the same claims-argument-evidence structure. Maturity: PROMISING (both are 2026
  papers; safety-case *patterns* for learned AD planners are an actively-forming, not yet
  standardized, sub-field — this is a genuinely open area, appropriately reflected by PROMISING
  rather than PROVEN).
- **Pain points:** **P8**, as close to a direct answer as this report finds to the brief's closing
  question ("what does a credible safety case for a learned planner look like in 2026") — this paper
  should be read in full (not just abstract-level) before TanitAD drafts its own safety case
  narrative.
- **Admissibility:** N/A (documentation methodology).
- **Cost:** ~1 eng-day to read in full and produce an internal gap-map against it (this is a
  reading/synthesis task, high value for low cost — should be one of the very first follow-ups from
  this report, independent of any GPU-day budget).
- **Experiment:** N/A (methodology adoption, not an empirical test) — the actionable next step is
  simply: **read arXiv:2603.16013 in full and produce a TanitAD-specific safety-case skeleton
  (claims/argument/evidence table) using its pattern**, before any of F1-F7's individual mechanisms
  are assembled into the "guaranteed envelope" this report proposes below — the pattern paper is
  about *how to structure the argument*, which should come before, not after, choosing the specific
  mechanisms.

### F10. Imagination-refusal / selective abstention for world-model rollouts — the most direct P7 hit
- **What:** A mechanism for a world model to learn **where to refuse its own imagination** — i.e., to
  recognize when a rollout has drifted into a regime where its own predictions should not be trusted
  for downstream decision-making, using execution-settled credit (did trusting this rollout actually
  pay off, retrospectively) as the training signal for the refusal decision itself.
- **Evidence:** PUBLISHED — **"DreamLedger: Where to Refuse World-Model Imagination Using
  Execution-Settled Credit,"** arXiv:2608.23863 (Aug 2026). This is not a driving-specific paper (the
  framing is general model-based-RL), but the *problem statement* — a world model's imagination
  needs a principled refusal/trust mechanism, trained against real outcomes rather than a hand-tuned
  confidence threshold — is **almost a direct restatement of TanitAD's own P7** ("imagination
  confidence anti-calibrated"). Maturity: PROMISING (specific quantitative results UNVERIFIED, fetch
  blocked; the problem framing itself is the valuable find here, independent of the paper's own
  numbers).
- **Pain points:** **P7, most directly of anything found in this entire review.** Where F6 (conformal
  prediction) offers a general-purpose, distribution-free calibration wrapper, DreamLedger's
  "execution-settled credit" idea is a specific, driving-compatible training signal: **did acting on
  this imagined rollout actually work out, after the fact, in the real trajectory the corpus
  recorded** — which is directly computable from PhysicalAI's offline logged trajectories without any
  new data collection.
- **Admissibility:** Clean (a training-time credit-assignment signal from logged outcomes, not a
  runtime privileged input).
- **Cost:** ~8-12 A40-days (training a refusal/trust head against execution-settled credit needs a
  full pass computing "did the imagined rollout match what actually happened" over the training
  corpus, then a supervised head on top), ~6-8 eng-days.
- **Experiment:** Compute execution-settled credit for TanitAD's existing H15 imagination module over
  the training corpus (imagined rollout vs. logged actual continuation, per clip) and train a small
  refusal head to predict low-credit (untrustworthy) rollouts from the rollout itself, then check
  whether this refusal signal fires more often on the known closed-loop-failure scenario types than
  on passing ones. **Outcome A:** yes → directly resolves P7 with a mechanism purpose-built for
  exactly this problem, and the refusal signal becomes a second (independent-lineage) input to the
  F4 Simplex switch, alongside F2's CBF value and F6's CP-region width. **Outcome B:** refusal signal
  doesn't discriminate → combine with A6's finding (if the underlying imagination has no physical
  grounding to begin with, no amount of calibrating *when to trust it* helps — the fix would need to
  happen upstream, at the representation level, not at the trust-decision level).

---

---

## Ranked TOP-10 for TanitAD

Ranked by (measured pain-point fit) × (cost cheapness) × (evidence maturity), not by novelty alone —
per the operating standard's preference for the cheapest discriminating experiment over the most
exciting one.

| # | Idea | Section | Pain points | Cost | Why it's ranked here |
|---|---|---|---|---|---|
| 1 | **Post hoc neuro-symbolic safety guard** on the deployed flagship's command output | B4 | P1, P6 | ~2-4 eng-days, **0 A40-days** | Cheapest possible intervention in the whole report; zero retraining; converged upon independently by 3 papers in a 3-month window (2606-2608.xxxxx); targets the *exact measured* v1 failure mode (off-road/longitudinal, not collision) |
| 2 | **Tiered-rulebook reranking of candidates** (RECTOR pattern) | B7 | P1 | ~4-6 eng-days, 0-2 A40-days | Only entry in the whole report with a concrete, checkable, reproducible external benchmark number (WOMD, 43,219 instances, 28.58%→20.42%); directly a selection-layer fix for the named #1 pain point |
| 3 | **Differentiable CBF-QP safety filter** (BarrierNet, driving-specific instantiation already exists) | F2 | P1, P6, P8 | ~8-10 eng-days, ~6-10 A40-days | The clearest *genuine mathematical guarantee* (forward invariance) available at TanitAD's latency budget (sub-ms QP fits inside the measured 18.75-27.87ms tick with room to spare) |
| 4 | **Simplex / runtime-assurance wrapper** around the flagship | F4 | P8 (frames all of P1/P2/P6) | ~10-15 eng-days, ~5-8 A40-days | The architectural frame everything else in this TOP-10 plugs into; critically, it does NOT require the 263M-param flagship itself to be formally verified — only a small baseline needs a proof |
| 5 | **Conformal-calibrated imagination confidence** for H15 | F6 | P7 | ~5-6 eng-days, ~4-6 A40-days | The single most rigorous (real coverage-guarantee theorem) answer found to the named P7 diagnosis; wraps the *existing* H15 module, no architecture change |
| 6 | **DreamLedger-style execution-settled imagination refusal** | F10 | P7 | ~6-8 eng-days, ~8-12 A40-days | Complementary to #5: a problem-specific, driving-compatible *training signal* (did trusting this rollout pay off, per the logged corpus) rather than a purely statistical wrapper — the two together (learned refusal + CP coverage bound) is stronger than either alone |
| 7 | **RSS longitudinal/lateral distance monitor** | F1 | P8, P2 | ~5-7 eng-days, 0 A40-days | Most externally validated mechanism in the report (2026 TÜV SÜD SMS recommendation for Mobileye); zero compute; composes with anything |
| 8 | **Differentiable Kinematic Bicycle Model decoder** | E1 | P2, LATERAL family | ~4-5 eng-days, ~5-8 A40-days | Near-free architectural change (small param delta) that makes curvature/yaw-rate/speed outputs kinematically consistent *by construction* — directly serves the BINDING 4-metric-family rule's LATERAL family |
| 9 | **Train-time-only relational/risk structure** (GraphPilot pattern / RiskWorld-style risk readout on H15) | A1, A3 | P3, P10 | ~5-8 eng-days, ~5-12 A40-days | The cleanest demonstrated recipe for injecting structure without violating the vision-only inference rule (GraphPilot's own headline result is that the gain *survives removing the graph at test time*) |
| 10 | **Scenic/Chat2Scenic-driven closed-loop scenario expansion** | B11 | P10, P6 | ~5-8 eng-days, 0 A40-days (sim only) | TanitAD's closed-loop evidence base is 12 scenarios; every other item in this table is only as trustworthy as the eval suite that measures it — this is the cheapest lever on statistical power for everything above |

**Honorable mentions, not in the top 10 but flagged for the next pass:** F9 (SimLingo safety-case
pattern — ~1 eng-day to read, should probably be done *before* #1-10 are assembled into a safety
narrative, not after); F3 (HJ reachability, offline/exact for the single-lead-agent longitudinal
case — a strong complement to #3 specifically for P2's 88.7%-of-oracle-gap problem); E3 (Koopman-LQR
as a formally-analyzable candidate for #4's Simplex baseline controller); F5 (NN verification,
narrowly scoped to whatever small learned component ends up in the baseline or guard, not the
flagship); B3 (verifiable reasoning-action consistency, a good complement to #1 if TanitAD ever adds
an explicit reasoning trace).

---

## Proposed "guaranteed envelope" architecture

A learned planner (TanitAD's existing 263.4M-param flagship, per `MODEL_REGISTRY.md`, unchanged)
wrapped in layers that are each small, each independently justified above, and each sized to fit
comfortably inside the measured latency headroom (optimized planning tick **18.75-27.87 ms** p50
against a **100 ms** / 10 Hz budget — roughly **72-81 ms of slack**, per `MODEL_REGISTRY.md`'s L4
latency-optimization results) and the sub-300M parameter budget (263.4M existing + the additions
below stay under ~275M).

```
 VISION (FRONT camera, 3×256×256, vision-only — no ego state, no situation-classifier output)
   │
   ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ LAYER 0 — LEARNED FLAGSHIP (existing, UNCHANGED)                        │
 │ ViT encoder(87M) → operative(97M) → tactical(49M) → strategic(8M)       │
 │ + H15 imagination(22M) + grounding(13M)  =  263.4M params, ~18-28ms     │
 │ Produces: candidate trajectory/anchors + H15 imagined rollout latents   │
 └───────────────────────────────────┬───────────────────────────────────┘
                                      │ candidates + imagined rollout
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ LAYER 1 — SELECTION GUARANTEE (B7 rulebook rerank; ~0 new params)        │
 │ Tiered rule scoring (Safety≻Legal≻Road≻Comfort) over candidates,         │
 │ CPU-only, <1ms.  Picks the trajectory, not just the argmax of the net.   │
 └───────────────────────────────────┬───────────────────────────────────┘
                                      │ selected trajectory
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ LAYER 2 — TRUST GATE (F6 conformal + F10 refusal head; <1M new params)   │
 │ CP-calibrated coverage bound on H15's imagined rollout  +                │
 │ execution-settled-credit refusal score.  Two independent-lineage         │
 │ confidence signals feeding Layer 4's switch (defense in depth: they can  │
 │ disagree, and disagreement is itself a signal).                         │
 └───────────────────────────────────┬───────────────────────────────────┘
                                      │ trajectory + trust score
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ LAYER 3 — HARD SAFETY FILTER (F2 CBF-QP + F1 RSS check; ~0 new params)   │
 │ Differentiable CBF-QP: closest-feasible command w.r.t. road-edge +       │
 │ lead-vehicle barriers — a GENUINE forward-invariance guarantee,          │
 │ conditional on the dynamics model (sub-ms QP).  RSS distance check runs  │
 │ in parallel as a diverse, independently-derived redundant check.        │
 │ F3's precomputed HJ-reachability lookup (offline-computed, 0 online      │
 │ params) adds an EXACT boundary check for the single-lead-agent           │
 │ longitudinal case — TanitAD's named 88.7%-of-oracle-gap problem.         │
 └───────────────────────────────────┬───────────────────────────────────┘
                                      │ filtered command  (or: INFEASIBLE / low-trust)
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ LAYER 4 — RUNTIME-ASSURANCE SWITCH (F4 Simplex; ~0 new params)           │
 │ IF Layer 2 trust is low  OR  Layer 3's QP is infeasible  OR  the CBF     │
 │ value crosses threshold  →  switch authority to Layer 5.  ELSE pass      │
 │ Layer 3's filtered command through.  Switch logic itself is simple       │
 │ enough to be exhaustively tested / formally verified (F5).              │
 └──────────────────┬──────────────────────────────────┬───────────────────┘
        (normal)    │                        (fallback) │
                     ▼                                    ▼
     ┌───────────────────────────┐      ┌──────────────────────────────────┐
     │ command → actuation       │      │ LAYER 5 — VERIFIED BASELINE       │
     │                           │      │ (E3 Koopman-LQR or PD car-        │
     │                           │      │ following + lane-centering,       │
     │                           │      │ 1-5M params, LINEAR → provably    │
     │                           │      │ stable by classical control       │
     │                           │      │ theory, no vision needed — this   │
     │                           │      │ IS the existing Brain-4           │
     │                           │      │ FallbackMonitor/MRM hook per      │
     │                           │      │ REGULATION_TRACE.md, made         │
     │                           │      │ concrete)                        │
     │                           │      └──────────────┬───────────────────┘
     └─────────────┬─────────────┘                     │
                   └──────────────────┬─────────────────┘
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ LAYER 6 — FINAL NEURO-SYMBOLIC GUARD (B4; ~0 new params, 0 A40-days)     │
 │ Runs regardless of which upstream path produced the command; nearest-   │
 │ safe-alternative replace-if-violated; every intervention traceable to   │
 │ a named rule.  Near-free defense in depth against anything Layers 1-5   │
 │ missed or got wrong.                                                    │
 └───────────────────────────────────┬───────────────────────────────────┘
                                      ▼
                              ACTUATION (10 Hz)
```

**Parameter budget:** 263.4M (Layer 0, unchanged) + ~1M (Layer 2 refusal head) + ~1-5M (Layer 5
baseline, if a learned Koopman lifting is used rather than a pure hand-tuned PD controller) + ~0 for
Layers 1/3/4/6 (classical arithmetic, QPs, and lookups, not learned networks) = **~265-270M total,
comfortably inside the sub-300M budget.**

**Latency budget:** Layer 0 is the entire existing cost (~18-28ms optimized). Layers 1/3/4/6 are
each sub-millisecond (rule scoring, a small QP, threshold comparisons, a rule lookup). Layer 2's CP
lookup is a table read; the refusal head is a small MLP forward pass on already-computed latents
(<1ms). Layer 5 only executes on the fallback path (rare by design) and is itself designed to be
cheap (linear control law). **Total added latency is ESTIMATED low single-digit milliseconds**,
leaving most of the measured 72-81ms of slack for the Jetson Thor port's own overhead (this is an
A40-proxy measurement per `MODEL_REGISTRY.md`'s own caveat; Thor validation is still needed and this
report does not claim otherwise).

**What is GUARANTEED by this architecture, precisely stated (per the table at the top of Section F):**
Layer 3's CBF-QP guarantees forward invariance of the CBF-defined safe set **conditional on the
dynamics model and actuation bounds being correct**. Layer 5's baseline is stable **conditional on
being genuinely linear/small enough to verify, which must be enforced as a design constraint, not
assumed**. Layer 6 guarantees every intervention is traceable to a named rule (an auditability
guarantee, not a safety-optimality one). **Nothing in this architecture guarantees the Layer-0
flagship itself is safe** — that is the entire point of the design: safety is a property of the
envelope, engineered to hold even when the learned core is wrong, not a claim about the learned
core.

---

## What is hype

1. **"Formal verification proves the whole model is safe."** VNN-COMP-scale certification
   (α,β-CROWN, F5) genuinely scales to millions of parameters — but only for specific, narrow
   properties (bounded local robustness, small-network reachability), never for open-ended semantic
   correctness of a ViT-scale perception backbone. Quoting "scales to millions of parameters" as if it
   meant "can verify our whole encoder is safe" is the single most common overclaim in this space and
   this report explicitly does not make it.
2. **Hamiltonian/Lagrangian neural networks for road-vehicle ego-dynamics.** The technique's core
   guarantee (energy conservation) is close to the *wrong* invariant for a system that constantly
   sheds energy (braking, drag, friction) and gains it from an external actuator. Genuinely useful
   for manipulators/rigid-body robotics; a weak fit here specifically (E6).
3. **FNO/DeepONet operator learning for TanitAD's occupancy/traffic-flow needs, at the current data
   scale.** Genuinely proven for macroscopic, multi-sensor, PDE-governed traffic fields (freeway
   networks with many sensors); TanitAD has one ego vehicle, one camera, no road-network topology —
   there is no PDE-scale field to learn the solution operator *of* yet (E5, gated behind occupancy
   adoption which is itself gated behind solving the metric-depth problem, A7).
4. **LLM tool-calling agents (Agent-Driver-style) as a real-time planning-loop component.** Genuinely
   powerful offline/design-time (scenario triage, rule authoring via LangProp); latencies reported in
   the literature are orders of magnitude too slow for a 100ms/10Hz budget on Jetson Thor. Useful as
   an offline tool, not a runtime brain (D1).
5. **Chain-of-thought / reasoning traces treated as a safety mechanism in themselves.** ReasonBreak
   (D4) demonstrates reasoning traces are an active, demonstrated adversarial attack surface. An
   explanation is not a guarantee unless independently paired with an action-level check (which is
   exactly why this report's envelope checks the *command*, in Layers 1/3/6, not only the *reasoning*).
6. **"Provably safe" language (RSS, CBF, HJ reachability) used without stating the assumptions.** All
   three are genuine, real mathematics — but every one of them is conditional (correct dynamics model,
   bounded disturbance, other agents obeying assumed kinematic limits). Marketing language that drops
   the conditions is a form of hype even when the underlying theorem is sound. This is why this report
   opens Section F with an explicit GUARANTEED-vs-STATISTICAL table rather than a prose claim.
7. **Industry safety-process certification (e.g. a 2026 TÜV SÜD Safety-Management-System
   recommendation) mistaken for certification that a specific learned planner's outputs are safe.**
   An SMS certification is about the *lifecycle process* around the system, not a formal proof about
   any one neural network's behaviour — a distinction worth keeping sharp in any TanitAD safety-case
   narrative that cites this kind of industry precedent (F1, F8).
8. **Full 4D/occupancy world-model adoption (OccWorld family) as a drop-in for TanitAD specifically,
   before the metric-scale problem is solved.** The architecture family is genuinely proven — for
   multi-camera or LiDAR-equipped setups. TanitAD's single front camera with no metric depth
   supervision and no GNSS makes the *input assumption* of nearly every OccWorld variant untrue for
   us; ZipDepth-style fusion (already screened, second-step candidate) is the honest, scaled-down
   first step, not the full architecture (A7).

---

## Deliverable manifest

| Artifact | Location | Notes |
|---|---|---|
| This report (Sections A-F, TOP-10, envelope architecture, hype list) | `repo:Project Steering/Reviews/2026-09-25-programme-review/streams/W3_frontier_structure_reasoning_guarantees.md` | Staged in the working tree this session (see below); only copy of this artifact — exists in exactly one place, as required. |

**No other artifacts were produced** (no code, no data files, no separate scratch notes worth
banking — all working notes are already folded into the report body above). This stream did not
touch any file outside its own assigned path, per the session's operating rules (write only the
assigned stream file; do not edit any existing file).

**Escalation for the orchestrator:** none of this stream's findings require a PI decision to act on —
items 1, 2, 7, 8, 10 in the TOP-10 are pure engineering/eng-days work against the *existing deployed
v1 checkpoint* with **zero GPU-days and zero retraining** (B4's guard, B7's reranker, F1's RSS
monitor, and B11's scenario expansion), and could be started immediately by whichever stream owns
`stack/`. The one item worth flagging explicitly to the PI: **F9 (read arXiv:2603.16013, "Safety
Case Patterns for VLA-based driving systems: Insights from SimLingo," in full) is recommended as the
*first* follow-up from this entire report**, because it addresses how to *structure* a safety-case
argument, which should shape how items 1-10 get assembled and documented rather than being retrofitted
after the fact.

**Environment constraint affecting this report's evidence quality (stated once, applies throughout):**
this container's egress proxy blocks direct fetch of `arxiv.org` and `huggingface.co`
(`EGRESS_BLOCKED`, confirmed via the proxy status endpoint as a declared policy, not a transient
fault — not retried or routed around, per the proxy README's own instruction). All arXiv ids and
numbers in this report come from WebSearch result snippets (cross-checked against ≥2 independent
snippets where I report a number as PUBLISHED) rather than from opening the primary PDF/abstract page
directly. Every number I could not cross-confirm this way is explicitly marked UNVERIFIED inline
rather than silently upgraded to PUBLISHED. A follow-up pass with arxiv.org/huggingface.co access (or
via an allowed mirror this session didn't find, e.g. institutional proxies, semantic scholar API if
allowed) would let every UNVERIFIED figure above be either confirmed or corrected — this is the
single biggest actionable gap in this report's own evidence chain, and it is a container/policy
limitation, not a research shortcut taken.
