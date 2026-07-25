# H2 — Learned Sensor/Camera Selection as a Tactical Tool: Literature, SOTA Position, and Architecture Verdict

**Date:** 2026-07-25 · **Workstream:** H2 (attention-based modality steering) · **Type:** literature & SOTA positioning
**Commissioned by:** `Implementation/incoming/2026-07-25-h2-sensor-attention/H2_DESIGN_FRAMING.md` §"two commissioned inputs"
**Sibling input (not yet landed at time of writing):** `H2_SUBSTRATE_AND_LABELING.md` (what the AV corpus permits)

**Evidence-class discipline.** Every claim below is tagged `PUBLISHED` (with citation + the concrete number or
mechanism relied on) · `MEASURED` (ours + artifact) · `INHERITED` (from another agent/doc, not re-verified here) ·
`ESTIMATED` · `HYPOTHESIS`. Where a paper's number was read from an abstract or an HTML render rather than the
camera-ready tables, it is marked `PUBLISHED (abstract-level)` so it is not mistaken for a checked table value.

---

## 1. Executive verdict

### 1.1 Novelty — the honest read

**The core mechanism is NOT novel. A published, peer-reviewed system already does almost exactly the thing.**

`PUBLISHED` **DriveMoE** (Yang et al., arXiv:2505.16278, May 2025; **CVPR 2026**) builds a "Scene-Specialized
Vision MoE" whose **router selects camera views from the driving context**, paired with a "Skill-Specialized
Action MoE". Concretely: the vision router takes the **front-view embedding plus the future goal waypoint**,
emits a probability distribution across all views, and **selects the top-1 view at inference**. It is trained as a
**supervised classifier** — "binary camera-view selection labels `y_t ∈ {0,1}`" with cross-entropy. On
Bench2Drive it reports **Driving Score 74.22 / Success Rate 48.64 %** against its Drive-π0 base, with a
**vision-router accuracy of 88.85 %**, and an ablation in which dynamic supervised view selection (74.22 DS)
beats fixed multi-view configurations.

⚠️ **Version discrepancy, flagged rather than papered over.** The v1 and v2 HTML renders give **different
Drive-π0 baseline Driving Scores — 60.45 (v1) vs 55.85 (v2)** — while both give DriveMoE 74.22 / 48.64 %. The
paper's own stated "+22.8 % driving score" is consistent with the 60.45 figure, not 55.85 (which would be
+32.9 %), so the two numbers are probably different baseline configurations rather than a correction.
**Quote the DriveMoE number, quote the baseline as a range (55.85–60.45), and read the camera-ready tables
before any of this appears in a paper of ours.** No decision in this document depends on the baseline value.

If our claim is *"a learned router picks which camera to run, conditioned on the driving situation, in an MoE"* —
**that claim is taken.** We must not write it as new. Saying otherwise would be the exact failure the program's
operating standard calls out (asserting from a summary rather than the primary source).

**Four things remain genuinely open, and they are worth more than the taken claim.**

| # | Open contribution | Why it is open (with the prior work it beats) |
|---|---|---|
| **N1** | **Non-circular sensor-need supervision.** | `PUBLISHED` DriveMoE's camera labels come from **"manually designed filters based on future trajectories, bounding box, and maps"** (v2, Appendix 9), instantiated as per-scenario rules — *"Intersection Turning: annotate the front-side camera view pointing toward the intended exit"*, *"Highway Merging: annotate the side camera facing the merging location"*. **This is a rule, i.e. our forbidden L3** — and it carries a subtler circularity than the naive kind: the label is a deterministic function of the **future trajectory**, which is precisely what the planner is trained to output. A router that has learned the rule and a router that has learned sensor need are **indistinguishable under their evaluation**, because 88.85 % router accuracy is accuracy **against that same rule**. *(To be fair to the paper: the label uses privileged offline information — future trajectory, 3D boxes, maps — that is not a router input at inference, so this is materially better than our `_NAV_TO_ROUTE` lookup. The defect is evaluative, not fatal.)* Nobody in the driving camera-selection line has built the label from **counterfactual information value**; that machinery exists for **objects**, not sensors (HOIST, below). |
| **N2** | **Sensor request as a *consequence* of a tactical option** (situation → option → sensor), yielding compositional generalization as a *testable* prediction. | `PUBLISHED` DriveMoE routes (front embedding + goal waypoint) → camera **directly**; GEMINUS routes scene → **behaviour expert**, not → sensor. Neither factors the decision through an explicit tactical-option representation, and neither tests novel-situation × known-manoeuvre transfer. |
| **N3** | **Active sensing driven by a self-supervised latent world model's own epistemic state**, rather than by a supervised scenario classifier inside a VLA. | `PUBLISHED` The nearest 2026 work — AW4RE (arXiv:2604.16733) and "Certified World Models as Sensing Clocks" (arXiv:2607.01537) — is about **view synthesis under a sensing budget** and **validity-horizon deadlines**, not tactical camera routing in driving. The bridge is unbuilt. |
| **N4** | **Counterfactual withholding as the primary metric.** | `PUBLISHED` DriveMoE evaluates the router by **accuracy against its own heuristic labels**. No driving camera-selection paper we found withholds the requested camera and measures behavioural degradation versus withholding a non-requested one. |

**Recommended publication framing:** *not* "we invented camera selection", but **"the field's camera-selection
supervision is circular; here is a counterfactual construction that is not; and it changes the conclusion."**
That is defensible, cheap to demonstrate, and it is a contribution the closest prior work structurally cannot make.
`HYPOTHESIS` — the "changes the conclusion" half is the thing the experiment in §5 exists to test, and it may fail.

**⚠️ One honest deflation of the efficiency (C-EFF) story.** `PUBLISHED` The efficiency argument for adaptive
perception is already well-established outside our framing: *Energy-Efficient adaptive perception for autonomous
driving* (Knowledge-Based Systems, 2025) reports **up to 70.20 % energy reduction against static deployment for
~2 % detection-accuracy loss** by context-conditioned model selection. Selective camera activation is a variant of
a known lever. **C-EFF is not a USP on its own.** C-CAP (knowing when you cannot see enough) is where the
defensible claim lives.

### 1.2 V1 vs V2 — build **V1** (tactical head on the frozen WM encoder + predictor) first

| Axis | **V1** — head on frozen WM latents | **V2** — own encoder |
|---|---|---|
| Expected accuracy | **High enough.** `PUBLISHED` DriveMoE's vision router *is* a V1: a head over a front-view embedding + goal, reaching 88.85 % router accuracy. `PUBLISHED` FROST-Drive (arXiv:2601.03460, Jan 2026) reports a **frozen** VLM encoder + transformer adapter + GRU decoder *"significantly outperforms models that employ full fine-tuning"* on the Waymo Open E2E dataset (abstract-level). | Higher ceiling **in principle**; unproven here. |
| Cost | **Lowest.** Frozen ⇒ embeddings can be **precomputed once** for the whole corpus; only a small head trains. Hours, local-class. | Encoder-scale training. Pod-days. |
| Attributability | **Decisive win.** One lever changes. If V1 works, the capability is proven on a substrate the program already trusts. | Confounded by construction: a V2 win cannot be attributed between *the capability* and *the new encoder*. |
| Risk | The real risk is **substrate information starvation** — a front-camera-trained latent may not carry out-of-FOV cues. This is *cheaply testable first* (§5). | `INHERITED` (from `H2_DESIGN_FRAMING.md` §4) Branch-B's from-scratch camera-conditioned encoder **failed** — cross-rig speed R² **−0.667** vs frozen v1's **+0.657**, refuted at power. V2 inherits that risk and must state how it differs. |

**Recommendation: V1, and specifically V1's *probe* before V1's *head*.** Build nothing trainable until the probe
in §5 shows the frozen latent carries sensor-need information above the majority-class baseline **and** that the
task is not solvable without the image. V2 is then a *measured upgrade against a clean V1 baseline*, which is the
only ordering that keeps the gain attributable.

**One caveat that must be stated, not buried.** `HYPOTHESIS` The frozen encoder was trained on a front-camera
objective on a corpus that is `INHERITED` 74 % straight. Peripheral/edge evidence — the cue that says *"there is
something to your left you cannot see"* — is exactly what such an objective is least pressured to retain, and our
own multi-cam plan proposes **static sky/hood token masking** (`ENCODER_MULTICAM_OPTIMIZATION.md`), which trims
image periphery further. If §5's probe comes back null, the correct reading is **"the frozen substrate is
starved"**, not "the capability does not exist" — and the branch is V2 or a light encoder-side unfreeze, not
abandonment.

### 1.3 Top-3 failure modes, one line each (detail in §3)

1. **Label circularity** — the label is a lookup of something the model already sees. *Guard: a no-image control arm.*
2. **Router collapse to a constant / rare-class collapse.** *Guard: per-class router recall + routing entropy, and a majority-class-skill read, never accuracy.*
3. **Hard-gate train/test gap, and the efficiency claim evaporating.** *Guard: measure the need-rate from the labels before training (free), and evaluate on the exact hard-gated inference path.*

---

## 2. The seven research areas

### 2.1 Active perception / active sensing / next-best-view

**Foundations.** `PUBLISHED` Bajcsy, *Active Perception*, Proc. IEEE 76(8):966–1005, 1988; Aloimonos, Weiss &
Bandyopadhyay, *Active Vision*, IJCV, 1988. The modern restatement is `PUBLISHED` Bajcsy, Aloimonos & Tsotsos,
*Revisiting Active Perception*, Autonomous Robots, 2018, whose definition is the one to quote in our write-up:
an agent is an active perceiver if it **knows why it wishes to sense**, then chooses **what**, and determines
**how, when and where**. The paper's central point — the **"why"** is what distinguishes an active perceiver from
a passive observer — is precisely the argument for conditioning the sensor request on the *tactical option*
(our N2), not on the raw scene.

**How "should I look there?" is supervised — four families, all present in the literature:**

| Family | Objective | Representative work | Transfers to driving? |
|---|---|---|---|
| **Information gain / mutual information** | maximize `I(state ; observation \| action)` | `PUBLISHED` Krause, Singh & Guestrin, JMLR 2008 (near-optimal sensor placement; greedy has the submodular **1−1/e** guarantee); Krause & Guestrin, AAAI 2007 | Partially — needs a probabilistic state we can score. **We have one:** H15's per-sector epistemic σ. |
| **Convex relaxation of a binary selection** | relax `z ∈ {0,1}^n` to `[0,1]^n`, solve, round | `PUBLISHED` Joshi & Boyd, *Sensor Selection via Convex Optimization*, IEEE TSP 57(2):451–462, 2009 | As a **baseline/oracle** only — needs a linear-Gaussian measurement model we do not have. |
| **Task reward (RL / POMDP)** | choose sensing actions that maximize downstream return | `PUBLISHED` Mnih et al., *Recurrent Models of Visual Attention*, NeurIPS 2014 (glimpse policy trained by **REINFORCE**, with an auxiliary **baseline network** added specifically because the policy gradient is high-variance) | Yes — but see the variance pathology in §2.2. |
| **Task-policy uncertainty as the reward** | the *task* policy's own uncertainty pays the *looking* policy | `PUBLISHED` **DISaM / "Learning to Look"**, arXiv:2410.18964, Oct 2024 — factorizes into an **information-seeking** and an **information-receiving** policy, and trains them separately, *"using the information-receiving one to provide reward to train the information-seeking one"*; at test time it balances exploration/exploitation on the manipulation policy's uncertainty about the next action | **This is the one to copy.** It is non-circular *by construction* — the signal comes from downstream task competence, never from a rule over the input. |

**The canonical modern objective, written out.** `PUBLISHED` AW4RE (*Active World-Model with 4D-informed
Retrieval*, arXiv:2604.16733, Apr 2026) formalizes physical awareness as a POMDP in which the agent selects
**camera configurations** to maximize

> `R_task(k) + I(s ; o(k) | a(k), D(k)) − λ · Cost(a(k))`

i.e. **task return + information gain about the latent state − sensing cost**. This is exactly the shape our
tactical sensor-request head should optimize, and it is citable as prior formulation rather than invented here.
(Evaluated on Waymo Open against GEN3C; the paper's contribution is generative observation estimation, not driving
control.)

**When to re-sense.** `PUBLISHED` *Certified World Models as Sensing Clocks: Drift-Aware Deadlines for Active
Perception* (arXiv:2607.01537, Jul 2026) proposes the world model's **validity horizon** — the window in which its
predictions stay inside certified error bounds — as the **operational deadline** for re-sensing: sense too late and
you forfeit the validity guarantee; sense too early and you waste budget. `HYPOTHESIS` This composes directly with
H15's blind-rollout uncertainty-dissipation work and is the principled version of our σ-threshold trigger.

**Active perception in driving specifically** is a real, mature sub-literature — but it is about **motion for
visibility**, not sensor selection: `PUBLISHED` *Safe Occlusion-aware Autonomous Driving via Game-Theoretic Active
Perception* (arXiv:2105.08169); *Improved Occlusion Scenario Coverage with a POMDP-based Behavior Planner*
(Zhang & Steinhauser, ITSC 2021); *Learning Occlusion-aware Decision-making from Agent Interaction via Active
Perception* (arXiv:2409.17618, 2024). The standard mechanism is to instantiate **phantom road users** in occluded
regions and plan against their existence probability. **Relevance to us:** this literature already establishes
*which situations create decision-relevant invisibility* (occluded T-intersections, blind corners, unprotected
turns) — which is precisely the situation set our labels must cover, and an independent justification for the
lane-change / roundabout / intersection triad.

**Benchmark to be aware of.** `PUBLISHED` **ActiView** (arXiv:2410.04659; **ACL 2025 Long**) evaluates active
perception in MLLMs by restricting the perceptual field and requiring the model to **zoom or shift** to answer;
across **30 models** it finds a large gap in active-perception ability. Useful as an evaluation-design precedent
(restrict the field, then measure whether the model asks correctly), not as a driving result.

### 2.2 Learned sensor selection, adaptive & conditional computation

**The efficiency lineage** (all `PUBLISHED`): BranchyNet (Teerapittayanon et al., ICPR 2016) — side branches exit
early on **softmax entropy**, reported **2×–6×** speedups on CPU/GPU; MSDNet (Huang et al., ICLR 2018);
SkipNet (Wang et al., ECCV 2018) — a gating RNN skips residual layers; BlockDrop (Wu et al., CVPR 2018). The
umbrella reference is `PUBLISHED` Han et al., *Dynamic Neural Networks: A Survey*, IEEE TPAMI, 2021.

**The closest analogue to our problem is modality gating, not layer gating.** `PUBLISHED` **DynMM** (*Dynamic
Multimodal Fusion*, arXiv:2204.00102) learns a gating function producing **modality-level or fusion-level**
decisions on the fly, with a **resource-aware loss** that prices computation. Reported: **46.5 % computation
reduction on CMU-MOSEI with negligible accuracy loss**, and **>21 % computation saving on NYU Depth V2 while
improving segmentation** (abstract-level). This is the template for "run the left camera or don't", and it
demonstrates the compute-vs-accuracy knob exists.

**Driving-side adaptive perception** (`PUBLISHED`): *Energy-Efficient adaptive perception for autonomous driving
via lightweight policy learning* (Knowledge-Based Systems, 2025) — context features (traffic density, weather,
road complexity) select among detector variants for **up to 70.20 % energy reduction at ~2 % accuracy cost**.
*FDSNet* (Scientific Reports, 2025) dynamically selects the **fusion stage** by cross-modal feature-disagreement
scoring. Neither selects **which sensor to run**.

**How the discrete decision is trained — and what goes wrong:**

| Estimator | Reference | Known pathology |
|---|---|---|
| **REINFORCE / policy gradient** | `PUBLISHED` Mnih et al., NeurIPS 2014 | **High variance**; the paper itself adds a learned baseline network to make it trainable. With one binary decision per frame and a sparse downstream reward, variance is the dominant cost. |
| **Straight-through estimator** | `PUBLISHED` Bengio, Léonard & Courville, arXiv:1308.3432, 2013 | Biased gradient; forward/backward mismatch. |
| **Gumbel-Softmax / Concrete** | `PUBLISHED` Jang, Gu & Poole, ICLR 2017; Maddison, Mnih & Teh, ICLR 2017 | **Temperature schedule is the trap:** overly aggressive annealing causes **premature collapse to a suboptimal discrete solution**, while too-smooth relaxation yields **weak, indecisive gating**; the estimator is also **statistically biased** in some regimes. Applied to channel selection in `PUBLISHED` arXiv:1812.04180. |
| **Q-learning over the selection action** | `PUBLISHED` MVSelect, arXiv:2303.06145 (§2.5) | Chosen explicitly *because* view selection is discrete and non-differentiable — avoids relaxation error at the price of RL sample cost. |
| **Supervised classification of the choice** | `PUBLISHED` DriveMoE; GEMINUS | Cheapest and most stable — **but it requires a label, which is where circularity enters.** This is the trade the whole workstream turns on. |

**The train/test gap, stated precisely.** Every relaxed estimator trains a *soft* mixture and deploys a *hard*
top-k. The efficiency claim is only realized in the hard regime, and accuracy is only measured honestly there.
**Guard:** always report the metric computed on the deployed hard path.

### 2.3 Mixture-of-Experts routing — and what actually makes a router specialize

**The machinery** (`PUBLISHED`): Shazeer et al., *Outrageously Large Neural Networks*, ICLR 2017 — **noisy top-k
gating** plus two regularizers (an **importance** loss and a **load** loss) introduced specifically because gating
is **self-reinforcing**: an expert that wins early gets more gradient and wins more. Switch Transformer (Fedus,
Zoph & Shazeer, JMLR 2022) — the standard auxiliary load-balance loss over (fraction of tokens routed to expert
i) × (router probability mass on expert i), summed and scaled by N. ST-MoE (Zoph et al., 2022) — the **router
z-loss** for stability. V-MoE (Riquelme et al., NeurIPS 2021) — **batch-prioritized routing** in vision.
Expert-Choice routing (Zhou et al., NeurIPS 2022) — **experts pick their top-k tokens**, which *guarantees* uniform
utilization and removes the need for an auxiliary loss altogether.

**Two 2024–2025 results that should shape our design:**

- `PUBLISHED` **Loss-Free Balancing** (arXiv:2408.15664, Aug 2024; the strategy used in DeepSeek-V3): add a
  **per-expert bias** to the routing scores *before* top-k, and update that bias from each expert's **recent
  load** — excluded from the gating weight itself. The stated motivation is that **auxiliary losses inject
  interference gradients that impair model performance**. Validated to 3 B params / 200 B tokens.
- `PUBLISHED` **"Demons in the Detail"** (arXiv:2501.11873, Jan 2025): load-balancing computed at the
  **micro-batch** level *"push[es] the token evenly within each sequence"* and thereby **prevents specialization**;
  computing it at the **global-batch** level *"greatly improves the domain specialization of MoE experts"*.
  Experiments to 42.8 B params / 400 B tokens.

**⇒ The direct answer to "what makes a router specialize rather than degenerate to a constant":**
1. **A routing signal that is genuinely predictive of a difference in optimal downstream computation.** If all
   experts are equally good on all inputs, no loss can create specialization — the router is only overhead. This
   is the *first* thing to check, and it is a property of the **labels/data**, not of the architecture.
2. **Balance pressure at the right scope** — corpus/global-batch, never per-micro-batch (2501.11873).
3. **Balance without gradient interference** — prefer the bias-update scheme (2408.15664) over an auxiliary loss
   that fights the task objective.
4. **Anti-self-reinforcement noise** at the routing scores (Shazeer 2017; DriveMoE explicitly *"introduce[s] noise
   into the action router following DeepSeekMoE"*).
5. **Where the label is trustworthy, supervise the router directly** — this is what both driving MoEs do, and it is
   the single biggest reason they work at all.

**Driving MoEs — and the one number that should scare us.** `PUBLISHED` **GEMINUS** (arXiv:2507.14456, Jul 2025)
combines a Global Expert (trained on everything) with a Scene-Adaptive Experts Group, arbitrated by a
**Dual-aware Router** that computes routing uncertainty as **normalized entropy** `U(x) = H(P(x)) / log N` and
falls back to the Global Expert when `U(x) ≥ τ = 0.5`. Its Bench2Drive ablation:

| Arm | Driving Score | Success Rate | MultiAbility-Mean |
|---|---|---|---|
| GEMINUS (scenario-supervised + uncertainty) | **65.39** | **37.73 %** | **37.77 %** |
| ScenarioMoE (no uncertainty) | 62.38 | 32.27 % | 34.46 % |
| **VanillaMoE (no scenario supervision)** | **59.23** | 29.09 % | 32.05 % |
| SingleExpert baseline | 60.73 | 30.91 % | 31.63 % |

**`VanillaMoE (59.23) is WORSE than the SingleExpert baseline (60.73).`** An MoE router without label supervision
*cost* performance relative to not routing at all. And GEMINUS's router accuracy is **68.06 % overall but 2.87 %
on "Give Way"** — near-total collapse on one class, invisible in the aggregate. Both facts are `PUBLISHED` and both
are direct evidence for our #1 architectural risk.

**Collapse diagnostics to instrument from day one** (`PUBLISHED`, standard practice): **normalized routing
entropy** per layer (low ⇒ the router is committing to few experts), **per-expert utilization**, and **expert
co-activation**. See also arXiv:2605.19378 for a routing-collapse → "selective deadlock" taxonomy in visual
diffusion transformers, and arXiv:2604.21330 (teacher-guided routing) for the stabilization-by-dense-teacher idea.

**⚠️ The Soft-MoE trap.** `PUBLISHED` Soft MoE (Puigcerver et al., ICLR 2024, arXiv:2308.00951) is **fully
differentiable** and reports **no token dropping, no expert collapse, no routing instability** even with hundreds
of experts — which makes it tempting. **It does not solve our problem.** Soft MoE forms weighted combinations of
**all input tokens** for every expert; it saves *parameter-scaling* cost, not *input-acquisition* cost (Soft MoE
H/14: 128 experts, >40× the parameters of ViT-H/14, **+2 % inference time**). If every camera must still be
captured and tokenized to compute the soft weights, **C-EFF is dead**. Our gate must be **hard and upstream of the
encoder**. This is the one place where the stable option is not the admissible one.

### 2.4 Semantic situation/scenario classification for AD

**The taxonomy the field actually uses.** `PUBLISHED` **Bench2Drive** (NeurIPS 2024 Datasets & Benchmarks;
arXiv:2406.03877): 2 M annotated frames over **44 interactive scenarios** (cut-in, overtaking, detour, …),
23 weathers, 12 towns, 220 routes; the 44 scenarios roll up into **five ability categories — Merging, Overtaking,
Emergency Brake, Give Way, Traffic Sign**. Both DriveMoE and GEMINUS take their scenario/skill vocabulary from
exactly this five-way split. **Implication for us:** a five-way skill taxonomy is the de facto standard and using
it costs us no novelty; our lane-change / roundabout / intersection triad is a *different, FOV-motivated* cut and
should be justified as such (occlusion literature, §2.1) rather than presented as a new taxonomy.

**Labelling at scale — the scalable mechanism that is not a VLM.** `PUBLISHED` **RefAV / AV2 Scenario Mining**
(CVPR 2025 workshop challenge; technical reports arXiv:2506.11124): **10,000 planning-centric natural-language
queries over 1,000 Argoverse-2 logs**; the baseline has an **LLM compose hand-crafted atomic functions** (e.g.
`turning`, `has_objects_in_front`) into a program that filters 3D track predictions. Best challenge score
**53.38 HOTA-Temporal**. **This is the pattern we should copy for the L1 labeller:** the LLM writes the *program*,
the *program* runs on 3D tracks + HD map + ego pose. The label is then a deterministic function of world geometry —
auditable, cheap to re-run, and **not** a VLM's opinion about an image.

**VLM labelling reliability — the finding that should stop us using a VLM as the label source.**
`PUBLISHED` **DriveBench** (*Are VLMs Ready for Autonomous Driving?*, arXiv:2501.04003; **ICCV 2025**):
19,200 frames, 20,498 QA pairs, 12 VLMs, 17 settings including **text-only (no image)**. Reported perception
scores (HTML render; treat as `PUBLISHED (abstract/HTML-level)`):

- **GPT-4o: 35.37 % clean → 36.48 % text-only.** *Removing the image did not hurt; it helped.*
- **Qwen2-VL 7B: 28.99 % clean → 35.16 % text-only**, with the authors noting the model can *"'guess' the MCQ
  answers without visual information by leveraging plain text cues."*
- DriveLM 7B degrades properly (16.85 % → 8.75 %), showing the effect is model-dependent, not universal.

The paper's conclusion is that VLMs *"generate plausible responses derived from general knowledge or textual cues
rather than true visual grounding"* — **and that the standard metrics do not detect it.** Two consequences for us:
(a) do not use a VLM to produce the sensor-need label; (b) **the text-only ablation is a reusable circularity
detector**, and we adopt it in §5.

**Public datasets with sensor-need annotation: none found.** Second-probe discipline applied (operating standard
rule 2): probed (i) "camera importance / view importance annotation dataset driving" and (ii) "sensor need /
which sensor is needed annotation". Both returned **object**-importance resources only — `PUBLISHED` **HOIST**
(arXiv:2312.02467), **IDD-X** (arXiv:2404.08561, 85 h dual-view with front+rear important-object annotations),
**Rank2Tell** (~2,600 intersection frames, importance ranking over a 3-camera stitch). **DriveMoE had to create
its own camera-importance annotations for Bench2Drive** — which is itself the strongest available evidence that no
public sensor-need annotation existed as of CVPR 2026. `ESTIMATED` confidence that no public camera-need
annotation exists: high, but this is an absence claim and should be re-probed before it appears in a paper.

### 2.5 Multi-camera / surround-view driving models — is camera *selection* established?

**The dominant paradigm fuses everything, every frame.** `PUBLISHED` BEVFormer (ECCV 2022) — spatiotemporal
transformer over all views with a recurrent BEV memory; BEVFusion (ICRA 2023, arXiv:2205.13542) — unified BEV
representation, **+1.3 % mAP/NDS at 1.9× lower computation**, achieved by optimizing BEV pooling (**>40× latency
reduction** in the view transformation), i.e. *the field's answer to cost is a faster kernel, not fewer cameras*;
StreamPETR — object-centric query propagation across frames; SurroundOcc (ICCV 2023) — multi-camera 3D occupancy.
**None of these choose which camera to attend to.**

**The adjacent-but-different line is *robustness to missing views*, not selection.** `PUBLISHED` **M-BEV**
(AAAI 2024, arXiv:2312.12144) randomly masks camera-view features during training and reconstructs them from the
remaining views (Masked View Reconstruction), reporting **+10.3 % mAP for PETRv2 when the back view is absent**.
This is important to us for two reasons: it establishes that **a model can operate usefully with a view withheld**
(so our counterfactual-withholding evaluation is not testing an ill-posed condition), and it gives us a
**ready-made harness** — mask a view, measure downstream degradation.

**Camera selection outside driving is established.** `PUBLISHED` **MVSelect** (arXiv:2303.06145) formulates view
selection as a sequential decision problem over (selected-camera one-hot, max-pooled features of selected views)
and trains it with **Q-learning**. Its results are the best available template for a baseline ladder:

| Setting | Random | Dataset-level oracle | MVSelect | Instance-level oracle |
|---|---|---|---|---|
| ModelNet40, 12-view, fixed task net (acc.) | 71.5 % | 85.2 % | **88.2 %** | 96.5 % |
| Wildtrack, fixed task net (MODA) | 74.9 | 82.5 | **80.0** | 87.4 |

with **2 of 20 views ⇒ 3.6 G vs 36.5 G FLOPs (≈10×)** and **3.03× throughput**. Note MVSelect **beats** the
dataset-level oracle on classification but **loses to it** on Wildtrack detection — a candid demonstration that a
learned selector can be worse than a well-chosen *fixed* camera set. **That fixed-set oracle is a mandatory
baseline for us.** 2026 continues the line outside driving, e.g. `PUBLISHED` SkillMoV (arXiv:2606.17615),
mixture-of-view routing with prototype-conditioned gating.

**Verdict for the novelty claim:** *camera selection* is established in multi-view vision (MVSelect, 2023) and
**established in driving as of CVPR 2026 (DriveMoE)**. It is **not** unexplored. What is unexplored is N1–N4 of §1.1.

### 2.6 Human/driver gaze as supervision

**The datasets** (`PUBLISHED`): **DR(eye)VE** (Palazzi et al., TPAMI 2019) — ~6 h of **in-car** eye tracking, the
only major in-car set, with distraction annotations on ~20 % of frames; **BDD-A** (Xia et al., ACCV 2018,
arXiv:1711.06406) — gaze on **braking-event** clips collected **in-lab** because critical moments are too rare
in-car; **DADA-2000** (Fang et al., ITSC 2019) — 2,000 accident videos with gaze; **CoCAtt** (arXiv:2111.10014) —
cognitive-conditioned, with per-frame intent and distraction state. Survey: `PUBLISHED` Kotseruba & Tsotsos,
*Behavioral Research and Practical Models of Drivers' Attention*, arXiv:2104.05677.

**Known biases, and they are severe for our use:**
- **Center bias.** `PUBLISHED` *Towards Robust Unsupervised Attention Prediction in Autonomous Driving*
  (arXiv:2501.15045) builds **DriverAttention-C** (BDD-A-C, DR(eye)VE-C, DADA-2000-C; **115,332 frames**, four
  corruption families) and explicitly mitigates *"the central distribution of pseudo-labels"* with random-crop
  Mixup. A center-biased label will teach a router "the front camera is enough" — the constant-output failure.
- **Protocol validity.** `PUBLISHED` BDD-A's own authors concede third-person in-lab attention *"is inevitably
  different from first-person driver attention in the car"*; the DR(eye)VE line notes in-lab gaze is *"less
  credible as the participants were not the actual drivers."*

**Gaze *does* help policies** (`PUBLISHED`): gaze-modulated dropout in imitation learning (arXiv:1904.08377);
the Periphery-Fovea Multi-Resolution Driving Model guided by human attention (arXiv:1903.09950); *A Gaze Model
Improves Autonomous Driving* (ETRA 2019).

**⚠️ The precision point that decides whether we use gaze at all.** Every one of these datasets records gaze
**within a front-facing video frame**. Gaze therefore labels *"which region of the front image mattered"* — it
**cannot** label *"which additional camera was needed"*, because the thing that needed looking at was, by
construction of our problem, **outside the front FOV**. Mirror and shoulder checks — the exact human behaviour our
capability imitates — are largely *absent* from these front-camera gaze maps.
**Ruling: gaze is not admissible as the H2 label.** It is admissible as (a) a *validation* signal in the special
case of wide-FOV front cameras where the target is still in-frame, and (b) prior art to cite for the claim that
attention allocation is task-driven and learnable.

### 2.7 Evaluation — how to measure "requested the right sensor at the right time"

Four axes, each with a citable precedent. **The first is primary; the rest are secondary and must not stand in for it.**

| Axis | Metric | Precedent |
|---|---|---|
| **A. Decision-relevance (PRIMARY)** | **Counterfactual withholding**: when the model requests camera X, withhold X and measure behavioural degradation; compare to withholding a **non-requested** camera and a **random** camera. A correct requester degrades *more* when its own request is denied. | `PUBLISHED` HOIST (arXiv:2312.02467) defines object importance by a **Removal Score** — remove the object from sensor observations, measure the **L2 distance between waypoints predicted from true vs counterfactual observations** — plus a **Velocity Perturbation Score** (hard stop / speed-up / lane change, scored by time-to-closest-approach). Their counterfactual estimator scores **AP 0.710** against human importance labels vs **0.630** (inverse distance) and **0.572** (PlanT). `PUBLISHED` M-BEV (AAAI 2024) supplies the mask-a-view harness. **Lifting the removal score from objects to cameras is, as far as our probing goes, unpublished — this is N1/N4.** |
| **B. Label agreement** | **PR-AUC on the sensor-need label** (rare-event framing), reported **per class** and against a **majority-class** baseline — never bare accuracy. | `PUBLISHED` GEMINUS: 68.06 % overall router accuracy hides **2.87 %** on Give Way. `MEASURED` (ours) the strategic head reports `route_acc = 1.0` with `route_skill = 0.0`. Aggregate accuracy is a known liar in this exact setting. |
| **C. Timing** | **Precision / recall at event level + Time-to-Maneuver (TTM)** — the interval between the request and the onset of the manoeuvre. A correct request one frame before the conflict is worthless. | `PUBLISHED` Brain4Cars (Jain et al., ICCV 2015; arXiv:1601.00740) anticipates manoeuvres at **precision 90.5 % / recall 87.4 %** with **TTM 3.5 s** over 1,180 miles — the standard formulation and a realistic bar. |
| **D. Efficiency (C-EFF)** | Fraction of frames requiring ≥2 cameras; energy/FLOPs vs an always-all-cameras baseline **on the hard-gated path**; Pareto quality-vs-FLOPs. | `PUBLISHED` MVSelect: 10× FLOPs, 3.03× throughput at 2/20 views. `PUBLISHED` adaptive perception: 70.20 % energy at ~2 % accuracy cost. `INHERITED` our own G0.7 gate is already specified as a Pareto plot beating a fixed-camera baseline at matched FLOPs (`ROADMAP.md` L2). |

**Mandatory baseline ladder** (from MVSelect, §2.5): **random selection · best fixed camera set (dataset-level
oracle) · learned router · instance-level oracle**. Reporting the router without the **fixed-set oracle** is the
easiest way to publish a result that a constant would have matched.

**Estimator discipline** (binding, per `CLAUDE.md`): paired **episode-cluster bootstrap** (`taniteval/ci.py`,
B=2000) over ≥40 episode-clusters; every interval names its estimator; **never `overlapping_holdout_se`**.

---

## 3. Failure modes and the cheapest guard for each

### FM-1 — Label circularity (the known one; highest severity)

**The precedent, ours.** `MEASURED` (2026-07-25, `V4_FLAGSHIP_DESIGN.md` / `H2_DESIGN_FRAMING.md` §3): the
strategic brain's route target was `route_target = _NAV_TO_ROUTE[nav_cmd]` — a **lookup of its own input**. Route
cross-entropy reached **exactly 0.0** by step ~14.5 k and **`route_skill = 0.0` by construction**. The failure is
indistinguishable from "the idea doesn't work" unless someone reads the label definition.

**The equivalent trap here, and the fact that the field has already brushed it.** `PUBLISHED` DriveMoE's camera
labels are *"manually designed filters based on future trajectories, bounding box, and maps"* realised as
per-scenario rules (*intersection turning ⇒ front-side camera toward the intended exit*). Two observations, kept
separate because they carry different weight:
- **The weaker, fair one:** those inputs are *privileged offline* information, not router inputs, so this is not
  the naive `_NAV_TO_ROUTE` self-lookup. It is a legitimate weak-supervision construction.
- **The one that matters:** the label is a deterministic function of the **future trajectory the planner is trained
  to output**, and the router is then **scored against that same rule**. "Learned sensor need" and "learned the
  rule" produce identical numbers. **We would reproduce this exactly, in a harsher form, if we labelled
  `roundabout ⇒ left camera` (our L3) — where the antecedent is something the model directly perceives.**

**How the literature constructs non-circular "what should I have looked at" supervision — three mechanisms, in
increasing order of what they demand from us:**

1. **Counterfactual ablation of the information itself.** `PUBLISHED` HOIST: importance ≡ the measured change in
   the *planner's own output* when the entity is removed from the observation. The label is a **function of a
   downstream behavioural difference**, so it cannot be a lookup of an input — you must actually run the
   counterfactual. **Lifted to sensors:** *camera X was needed at t iff masking X changes the model's/expert's
   behaviour by more than δ.* This is our **L2** and it is the ideal validation signal.
2. **Downstream task competence as the reward.** `PUBLISHED` DISaM (arXiv:2410.18964): the information-**receiving**
   policy supplies the reward for the information-**seeking** policy. Non-circular by construction, and it never
   requires anyone to write down what "needed" means.
3. **World geometry + realised behaviour.** Our **L1** (from `H2_DESIGN_FRAMING.md` §3): camera X was needed iff an
   agent (a) fell in X's frustum, (b) was **not visible** in the front camera, **and** (c) **constrained the ego's
   realised behaviour**. Condition (c) is the load-bearing one — (a)+(b) alone are near-always true in traffic and
   would produce the constant-"yes" router. The scalable way to compute it is the **RefAV pattern** (§2.4):
   an LLM composes atomic geometric predicates; the *program*, not the LLM, produces the label.

**Cheapest guard — the no-image control arm.** Transplant DriveBench's text-only ablation (§2.4): train an
identical head that receives **only the privileged symbolic context** (nav command, manoeuvre label, map class)
and **no image features**. If it matches the full model's PR-AUC, the task is solvable without perception and the
capability claim is vacuous — **regardless of how high the absolute score is**. Cost: one extra small head, hours.
This single control would have caught the strategic-brain failure on day one, and it would catch DriveMoE's.

### FM-2 — Router collapse to a constant (or to the majority class)

**Why it is the default outcome, not the tail risk.** `PUBLISHED` Gating is **self-reinforcing** (Shazeer et al.,
ICLR 2017): an expert that wins early receives more gradient and wins more. `PUBLISHED` GEMINUS's **VanillaMoE
(59.23 DS) underperformed its own SingleExpert baseline (60.73 DS)** — unsupervised routing was *worse than not
routing* — and its supervised router still collapsed to **2.87 %** accuracy on the rare "Give Way" class while
reporting **68.06 %** overall. `PUBLISHED` DriveMoE's stated limitation is verbatim that *"effectively achieving
load balancing among experts remains a significant challenge as the number of experts grows"*, and its own
ablation shows 13 or 44 experts **degrading** through load imbalance where 6 works.

**Guards, cheapest first:**
- **Report skill, not accuracy** — per-class recall and PR-AUC against a **majority-class** baseline. Free.
  (`MEASURED` we have already been fooled once by `route_acc = 1.0 / route_skill = 0.0`.)
- **Instrument normalized routing entropy and per-expert utilization from step 0** (§2.3). Free; catches collapse
  during training rather than in the post-mortem.
- **Balance at global-batch scope, never micro-batch** — `PUBLISHED` arXiv:2501.11873: micro-batch LBL forces
  within-sequence uniformity and *prevents* specialization. Config change only.
- **Prefer the bias-update balancing scheme** (`PUBLISHED` arXiv:2408.15664) over an auxiliary loss whose
  interference gradients degrade the task objective. ~20 lines.
- **Check the premise before the architecture**: if masking the left camera changes nothing on ≥99 % of frames,
  no router can specialize and the honest result is a negative. This is knowable from the labels alone, **before
  training** — see §5.

### FM-3 — Hard-gate train/test gap, with the efficiency claim evaporating

Three coupled traps: (a) **temperature schedules** — `PUBLISHED` overly aggressive Gumbel annealing causes
**premature collapse to a suboptimal discrete solution**, too-smooth annealing gives an indecisive gate;
(b) **soft-train/hard-deploy mismatch** — the reported accuracy comes from a soft mixture the deployment never
runs; (c) **the Soft-MoE trap** (§2.3) — the stable, fully-differentiable option processes **all** input tokens,
so adopting it silently deletes C-EFF while everything still "works".

**Guards:**
- **Report every metric on the exact hard-gated inference path.** Free, and non-negotiable.
- **Compute the need-rate from the labels before any training.** If ≥2 cameras are needed on most frames, C-EFF is
  dead on arrival and only C-CAP is left — better to know for the cost of a histogram than after a training run.
- **Log the gate's realized duty cycle in evaluation** alongside quality, and present the result as a Pareto point
  vs the fixed-camera baseline at matched FLOPs (already the `ROADMAP.md` L2 gate).
- **A Straight-Through Gumbel gate with a logged temperature schedule**, plus the discrete-supervised head as the
  primary (DriveMoE's choice), keeping RL/Q-learning (MVSelect's choice) in reserve for when the label is weak.

**Honourable mention — FM-4, the corpus rig split.** `MEASURED` (memory: *PhysicalAI two rigs by cy*) the AV
front-wide corpus contains **two rigs** with different principal points; a geometric-center crop is ~215 px wrong
for rig B. Any camera-frustum geometry used to build the L1 label **must use per-clip `cy`**, or the frustum test
in condition (a) is systematically wrong for one rig — which would look exactly like label noise.

---

## 4. Recommended architecture (implementable detail)

**Build order: probe → V1 head → counterfactual evaluation → (only then) V2.**

### 4.1 The decision structure

Follow `H2_DESIGN_FRAMING.md` §2 — **⟨SITUATION⟩ → ⟨TACTICAL OPTION⟩ → ⟨SENSOR REQUEST⟩**, with the sensor head
**conditioned on the tactical-option representation**, never emitted as an independent class. This is our N2, and
it is also what makes the compositional-generalization test (novel situation × known manoeuvre ⇒ correct sensor)
meaningful. It is the operational form of Bajcsy et al.'s **"knows *why* it wishes to sense"**.

### 4.2 V1 concretely

```
frozen WM encoder (front camera only, v1's 87.1 M, NO gradient)
        │  z_t  [+ optional: predictor's imagination latents / H15 per-sector σ]
        ▼
  situation head        →  s_t   (factored; see below)
        │
        ▼
  tactical-option head  →  g_t   (LATERAL-intent × LONGITUDINAL-intent × yield/merge obligation)
        │                        ⚠️ NOT the existing 5-way manoeuvre softmax
        ▼
  sensor-request head   →  logits over {front, left, right, rear, ...}
        │                  router input = [z_t , g_t]      ← cf. DriveMoE: [front-view embedding, goal waypoint]
        ▼
  hard gate (top-k, k≥1, front always on) → encode only the requested cameras
```

**Justified design choices, each with its source:**

| Choice | Setting | Source |
|---|---|---|
| Router input | `[frozen latent, tactical-option embedding]` | `PUBLISHED` DriveMoE uses `[front-view embedding, future goal waypoint]` and reaches 88.85 % router accuracy — an existence proof that a small head over an embedding suffices. Our `g_t` replaces their goal waypoint and adds the causal factorization (N2). |
| Selection | **top-1 additional view** at inference, front always on | `PUBLISHED` DriveMoE selects top-1; `PUBLISHED` MVSelect retains near-full accuracy at 2–3 of N views. |
| Primary training signal | **supervised CE / focal on the L1 counterfactual label** | `PUBLISHED` supervised routing is the only variant that beat the single-expert baseline in GEMINUS's ablation; **but** the label must be L1, not L3. |
| Balancing | **per-expert bias updated from recent load**, global-batch scope | `PUBLISHED` arXiv:2408.15664 (no interference gradients); arXiv:2501.11873 (global-batch preserves specialization). |
| Anti-collapse noise | Gaussian noise on routing scores | `PUBLISHED` Shazeer et al., ICLR 2017; DriveMoE follows DeepSeekMoE here. |
| Uncertainty fallback | if router entropy `U(x) = H(P)/log N ≥ τ`, **activate the safe superset** (all cameras) rather than guessing | `PUBLISHED` GEMINUS's Dual-aware Router with `τ = 0.5`, adapted: their fallback is a Global Expert, ours is conservative sensing — the safety-correct direction. |
| Trigger, phase 2 | H15 per-sector epistemic σ as an auxiliary router input; validity-horizon deadline | `INHERITED` `H16_ACTIVE_DEPTH_INTERROGATION.md`; `PUBLISHED` arXiv:2607.01537 (sensing clocks). |
| Optional reward refinement | task-policy uncertainty pays the requester | `PUBLISHED` DISaM (arXiv:2410.18964); objective shape `R_task + I(s;o|a,D) − λ·Cost(a)` from AW4RE (arXiv:2604.16733). |

**Do NOT** reuse the existing 5-way manoeuvre softmax as `g_t` — `INHERITED`/`MEASURED` it mixes lateral and
longitudinal in one head and is the identified root cause of the program's longitudinal blindness (0/881
accelerate). Rebuilding it into a new brain would be a known defect adopted deliberately.

**Do NOT** use Soft MoE for the camera gate (§2.3) — it is the stable option and it deletes the efficiency claim.

### 4.3 What V2 would have to prove

V2 (own encoder) is admissible **only** as a measured upgrade against a landed V1 baseline, and its brief must
state **how it differs from Branch-B's from-scratch camera-conditioned encoder**, which `INHERITED` failed at power
(cross-rig speed R² −0.667 vs frozen v1's +0.657). Absent that statement, V2 is a repeat of a refuted experiment.

---

## 5. The cheapest discriminating experiment, with a pre-registered falsifier

**Name:** *Frozen-latent sensor-need probe with a no-image control.*
**Cost:** `ESTIMATED` hours, local (4060-class), **no pod, no GPU-day, no training of any encoder, no arm disturbed.**
**What it decides:** whether V1's substrate carries sensor-need information at all — **and** whether the task is
real or circular. Either answer is worth having before a single training commitment.

**Setup**
1. Build **L1** labels offline on the reserved multi-view subset (`INHERITED` Phase 0 Plan B2: ~500 multi-view
   clips, front+L+R+rear) from `obstacle.offline` 3D tracks (`INHERITED` present on 96.90 % of the corpus) +
   per-camera frustums (**per-clip `cy`**, FM-4) + realised ego behaviour for condition (c). Compute the
   **need-rate histogram first** — it is free and it decides whether C-EFF is alive.
2. Precompute **frozen** WM latents from the **front camera only**. No gradients anywhere upstream.
3. Train four small heads on identical windows and splits:
   - **A — probe:** 2-layer MLP on `z_t` → `P(camera X needed at t+Δ)`.
   - **B — no-image control:** identical head on **privileged symbolic context only** (nav command, manoeuvre
     label, map class) — **no image features**. *(DriveBench text-only ablation, §2.4.)*
   - **C — majority-class baseline.**
   - **D — best fixed camera set** (dataset-level oracle) and **random selection**. *(MVSelect ladder, §2.5.)*

**Primary read:** PR-AUC (rare-event framing), **per class**, paired **episode-cluster bootstrap** B=2000 over
≥40 episode-clusters, stratified over lane-change / roundabout / intersection.

**Pre-registered falsifiers — both outcomes committed in advance:**

- **F-A (substrate).** If **A**'s PR-AUC CI **overlaps C** (majority baseline), the frozen front-camera latent does
  **not** carry sensor-need information. ⇒ **V1 is refuted at the substrate.** Branch to V2 or a light encoder
  unfreeze — do **not** proceed to build a V1 head.
- **F-B (circularity — the decisive one).** If **B** (no image) matches **A** within the paired CI, then sensor
  need is predictable from symbolic context alone and **the capability claim is vacuous no matter how high A
  scores.** ⇒ The label or the framing is circular; return to label design before any training.
- **F-C (premise).** If the need-rate shows ≥2 cameras required on **>50 %** of frames, **C-EFF is dead** and H2 is
  a pure C-CAP capability claim. If it is **<1 %**, this is a rare-event problem and must be designed as one
  (stratified sampling, PR-AUC, an explicit power calculation before training).
- **Publishable null, stated in advance:** *"sensor need is not predictable from the front camera above the
  majority-class baseline"* is a real, reportable negative — and given §2.4's finding that nobody has tested this
  non-circularly, it would be a **more** interesting result than a modest positive.

**Why this is the right experiment:** it is the only test that separates the three hypotheses that would otherwise
look identical at the end of a full training run — *the capability exists*, *the substrate is starved*, and *the
label was circular all along*. `MEASURED` We have already spent months on the third one once.

---

## 6. Annotated bibliography

**Closest prior work (read these first)**
- **DriveMoE** — Yang, Chai, Jia, Li, Shao, Zhu, Su, Yan. arXiv:2505.16278 (May 2025), **CVPR 2026**.
  *The system that already does camera routing in driving.* Vision router: `[front-view embedding, goal waypoint]`
  → distribution over views → **top-1**; supervised by **binary camera labels from "manually designed filters based
  on future trajectories, bounding box, and maps"** (v2 App. 9) (← the circularity). Action MoE: 1 shared + 6
  experts/decoder-layer, top-3, CE on the 5 Bench2Drive skills, DeepSeekMoE-style routing noise + `L_LB`.
  **DS 74.22 / SR 48.64 %** vs Drive-π0 **55.85–60.45 DS / 30.00 % SR** (v1/v2 renders disagree on the baseline —
  see §1.1); router accuracy **88.85 %** (vision) / 65.40 % (action); 13 or 44 experts degrade via load imbalance;
  stated limitation is load balancing as expert count grows. Code: github.com/Thinklab-SJTU/DriveMoE.
- **GEMINUS** — arXiv:2507.14456 (Jul 2025). Global Expert + Scene-Adaptive Experts + **Dual-aware Router**
  (`U(x)=H(P)/log N`, fallback at τ=0.5). **The ablation that should govern our risk model: VanillaMoE 59.23 DS <
  SingleExpert 60.73 DS**; router accuracy 68.06 % overall but **2.87 % on Give Way**.
- **MVSelect** — arXiv:2303.06145. Camera-view selection by **Q-learning**; the random / dataset-oracle / learned /
  instance-oracle ladder we should adopt; 2 of 20 views ⇒ ~10× FLOPs and 3.03× throughput.

**Non-circular supervision**
- **HOIST / counterfactual object importance** — arXiv:2312.02467 (Dec 2023). Removal Score (L2 waypoint shift on
  removing the object) + Velocity Perturbation Score; 409 CARLA videos, 642 objects, 188 annotators;
  AP 0.710 vs 0.630 (inverse distance) / 0.572 (PlanT). **The mechanism to lift from objects to cameras.**
- **DISaM / "Learning to Look"** — arXiv:2410.18964 (Oct 2024). Information-seeking and information-receiving
  policies trained separately, the receiver rewarding the seeker. Non-circular by construction.
- **DriveBench** — arXiv:2501.04003, **ICCV 2025**. 19,200 frames / 20,498 QA / 12 VLMs / 17 settings.
  **GPT-4o 35.37 % clean → 36.48 % text-only.** The reusable circularity detector, and the reason not to use a VLM
  as a label source.
- **RefAV / AV2 Scenario Mining** — CVPR 2025 workshop; arXiv:2506.11124. 10,000 NL queries over 1,000 AV2 logs;
  LLM composes atomic geometric predicates over 3D tracks. **The scalable, auditable labelling pattern.**

**MoE routing & collapse**
- Shazeer et al., ICLR 2017 (noisy top-k; importance + load losses; self-reinforcing imbalance) ·
  Fedus, Zoph & Shazeer, JMLR 2022 (Switch aux loss) · Zoph et al. 2022 (ST-MoE router z-loss) ·
  Riquelme et al., NeurIPS 2021 (V-MoE, batch-prioritized routing) · Zhou et al., NeurIPS 2022 (Expert Choice —
  balance by construction) · **arXiv:2408.15664** (Loss-Free Balancing: per-expert bias from recent load;
  auxiliary losses inject harmful interference gradients; used in DeepSeek-V3) ·
  **arXiv:2501.11873** "Demons in the Detail" (**micro-batch LBL prevents specialization; global-batch restores it**;
  to 42.8 B params / 400 B tokens) · **arXiv:2308.00951** Soft MoE, ICLR 2024 (fully differentiable, no collapse —
  **but processes all tokens, so it cannot deliver input-side savings**).

**Active perception**
- Bajcsy, Proc. IEEE 1988 · Aloimonos et al., IJCV 1988 · **Bajcsy, Aloimonos & Tsotsos, Auton. Robots 2018**
  (the "why/what/how/when/where" definition) · Mnih et al., NeurIPS 2014 (hard attention via REINFORCE + baseline
  net for variance) · Krause, Singh & Guestrin, JMLR 2008 (submodular, 1−1/e) · Joshi & Boyd, IEEE TSP 2009
  (convex relaxation) · **AW4RE arXiv:2604.16733** (POMDP objective `R_task + I(s;o|a,D) − λ·Cost(a)`) ·
  **arXiv:2607.01537** (validity horizon as re-sensing deadline, Jul 2026) · **ActiView arXiv:2410.04659,
  ACL 2025** (active-perception benchmark, 30 models, large gap) · occlusion-aware driving: arXiv:2105.08169,
  Zhang & Steinhauser ITSC 2021, arXiv:2409.17618.

**Conditional computation**
- BranchyNet, ICPR 2016 (entropy exits, 2–6×) · MSDNet, ICLR 2018 · SkipNet, ECCV 2018 · BlockDrop, CVPR 2018 ·
  Han et al., TPAMI 2021 (survey) · **DynMM arXiv:2204.00102** (modality/fusion gating + resource-aware loss;
  46.5 % compute cut on CMU-MOSEI, >21 % on NYU Depth V2 with better segmentation) ·
  Jang/Gu/Poole ICLR 2017 & Maddison et al. ICLR 2017 (Gumbel-Softmax/Concrete) · Bengio et al. arXiv:1308.3432
  (straight-through) · arXiv:1812.04180 (Gumbel channel selection) · *Energy-Efficient adaptive perception for AD*,
  Knowledge-Based Systems 2025 (**70.20 % energy at ~2 % accuracy**).

**Multi-camera driving**
- BEVFormer, ECCV 2022 · BEVFusion, ICRA 2023 (arXiv:2205.13542; +1.3 % mAP/NDS at 1.9× lower compute, BEV pooling
  >40× faster) · StreamPETR · SurroundOcc, ICCV 2023 · **M-BEV, AAAI 2024 (arXiv:2312.12144)** — masked view
  reconstruction; **PETRv2 +10.3 % mAP with the back view missing**; our withholding harness.

**Gaze**
- DR(eye)VE, TPAMI 2019 (in-car, ~6 h) · **BDD-A, ACCV 2018 (arXiv:1711.06406)** — in-lab protocol, authors' own
  validity caveat · DADA-2000, ITSC 2019 · CoCAtt, arXiv:2111.10014 · **arXiv:2501.15045 / DriverAttention-C**
  (115,332 frames; explicit **central-bias** mitigation) · Kotseruba & Tsotsos, arXiv:2104.05677 (survey) ·
  gaze-modulated dropout arXiv:1904.08377 · Periphery-Fovea model arXiv:1903.09950.

**Situation taxonomy & timing**
- **Bench2Drive**, NeurIPS 2024 D&B (arXiv:2406.03877) — 44 scenarios → 5 abilities (Merging / Overtaking /
  Emergency Brake / Give Way / Traffic Sign); the de facto vocabulary of both driving MoEs ·
  **Brain4Cars**, ICCV 2015 (arXiv:1601.00740) — **precision 90.5 % / recall 87.4 %, TTM 3.5 s** over 1,180 miles;
  the timing-metric template · Michon 1985, *A critical view of driver behavior models* — the
  **strategic / tactical / operational** hierarchy our layer naming descends from · IDD-X arXiv:2404.08561 ·
  Rank2Tell · blind-spot FOV estimation arXiv:2402.00467.

---

## 7. Integration escalations (per `AGENT_OPERATING_STANDARD.md` rule 3)

1. **`H2_DESIGN_FRAMING.md` §4 asks for the V1/V2 ranking marked ⟨PENDING research input⟩.** It is delivered in
   §1.2 above: **V1, and the probe before the head.** The framing document's own recommendation is **confirmed**,
   and now carries two published supports it did not have — DriveMoE's router is structurally a V1 (88.85 % from an
   embedding + goal), and FROST-Drive (arXiv:2601.03460) reports frozen encoders outperforming full fine-tuning on
   Waymo Open E2E. **Someone should update that section rather than leave the ⟨PENDING⟩ marker.**
2. **`H2_DESIGN_FRAMING.md` §5's counterfactual-withholding primary metric should cite HOIST (arXiv:2312.02467)**
   as the published precedent for the estimator — it is currently presented as our own design, and it is stronger
   with the citation.
3. **`HYPOTHESIS_LEDGER.md` H2 row** is to be un-parked (per the framing doc). The literature verdict that belongs
   in it: *mechanism NOT novel (DriveMoE, CVPR 2026); the open contribution is non-circular supervision + the
   causal situation→option→sensor factorization + counterfactual evaluation.*
4. **`ENCODER_MULTICAM_OPTIMIZATION.md` static sky/hood token masking is in tension with H2's substrate.** Masking
   image periphery removes exactly the weak out-of-FOV cues the V1 probe depends on. The two work-packages should
   be sequenced so the probe (§5) runs **before** any static mask is baked into the encoder path.
5. **PI decision still open (framing doc §8, unchanged by this review):** whether C-EFF is a claim we want. §1.1
   shows C-EFF is a crowded lever (70.20 % energy reduction is already published for adaptive perception), while
   C-CAP is where the defensible novelty sits.
