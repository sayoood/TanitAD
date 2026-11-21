<title>Frontier Scan 2026-09-02 — a published JEPA world model fails our exact action test, and the fix is two loss terms</title>

# FRONTIER SCAN — 2026-09-02

`TanitAD Research Lab · daily pass under DAILY_RESEARCH_CHARTER.md + v2 AMENDMENT.`
`Band D x1 (7-step, 2 primaries) · Band A x3 · Band B x3 + 1 scan · Band C sweep · 17 queries, 4 empties named.`
`⚠️ THIS PASS IS THE FRONTIER SCAN ONLY. Today's FIVE domain packages already existed at trigger time and were NOT redone.`
`⭐ Six of the eight primaries used today were ALREADY BANKED and unread — retrieval cost ~0.`

---

## THE HEADLINE

⭐⭐⭐ **`LeWorldModel` — a well-regarded 2026 end-to-end JEPA world model — was measured by an independent
group to be ACTION-INSENSITIVE on exactly the diagnostic we built for v7, and it fails the way our arms
fail.** Delta-JEPA perturbs the action and measures the predictor's displacement from its own zero-action
prediction; LeWM's *"action-wise means remain concentrated near the origin and substantially overlap"*.

⇒ **Our P-1/P-2 gate problem is a documented property of end-to-end latent world models, not a TanitAD
defect.** That is the third time this pass-series has converted "our bug" into "a field property" — after
the B13 readout geometry and Waymo's long-horizon-stability concession. ⛔ **And this one arrives with a
published fix that is two loss terms, an ablation table, and a control that reads the known
no-information value.**

⛔ **Two of our own instruments are implicated, not only our architecture** — F1's leak argument. That is
the finding with the shortest path to action, and it is 0-GPU.

⚠️ **Read the risk in F1 before acting on the fix:** our action channel is realised motion, which may make
the proposed objective tautological on our data. The experiment is staged so that costs nothing to find out.

---

## F1 ⭐⭐⭐ A2 — Delta-JEPA: action-insensitivity is a known failure, and endpoint-concatenation probes cannot detect it

`PUBLISHED lib 2606.31232 · FULL TEXT READ · Zhang et al., UCAS · arXiv 2026-06-30 · debt D-6 (A2) DISCHARGED`
`⚠️ Was ALREADY BANKED and unread — a second instance of FS-5 in eight days.`

**The diagnosis, verbatim:**
> *"when trained end-to-end with only latent prediction objectives, JEPA-based world models can easily
> collapse to trivial constant representations... the model achieves deceptively low prediction loss while
> destroying the representation structure needed for planning."*

⛔⛔ **The leak argument — this is the half that touches our instruments, not our model:**
> *"its inverse dynamics module decodes actions from concatenated adjacent latent states [z_t, z_t+1].
> Because the forward predictor is itself conditioned on the executed action, end-to-end optimization can
> make the next-state representation z_t+1 absorb action-correlated cues that are easy for the inverse
> decoder to exploit, **without requiring the model to represent the actual transition between the two
> states**."*

**The method (LDAD).** Decode the executed action from the latent **displacement** `Dz_t = z_t+1 - z_t`,
not from `[z_t, z_t+1]`. The whole objective is two terms: `L = L_pred + L_action`. No pixel
reconstruction, no distribution-matching regulariser, no frozen encoder, no stop-gradient.

**Numbers** (planning success %, 4 visual continuous-control envs, 3 seeds, 50 epochs, lr 5e-5):

| result | value |
|---|---|
| OGB-Cube vs strongest baseline | **+15.14 pp** |
| Two-Room vs PLDM | **+6.27 pp** |
| Push-T vs LeWM | **+4.54 pp** |
| ⭐ **Ablation `Dz` vs `concat[z_t, z_t+1]`** (their Table 2) | **`Dz` wins on all four**; Push-T **+12.60**, Two-Room **+4.07** |
| ⭐ **Control — lambda=0 removes LDAD entirely** | *"the resulting model nearly collapses, yielding only a negligible planning success rate"* |
| Decoding-target ablation (their Table 3) | raw action best; D-joint-position comparable; D-finger-position much worse |

⭐ **Note what that lambda=0 row is:** a control that must read the no-information value, and does. This
paper satisfies the probe-panel discipline `CLAUDE.md` imposes on us, which is why its ablation is
admissible rather than merely suggestive.

| dim | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ — lands on **P-1** (no v7 arm beats hold-action), **P-2** (the predictor does not use its actions), and **backlog row 13** (H-RANK-17, additive motion latent `z_t + m_t ~ z_t+1`) — the same latent-difference family, now with an external primary and ablations. |
| **CONSEQUENCE** | Two distinct changes. **(a) Architectural:** LDAD is a candidate P-2 fix cheaper than every arm currently queued — two loss terms on the existing ladder. **(b) ⛔ Instrumental, and more urgent:** any TanitAD action-decodability probe that decodes from concatenated endpoints is **confounded by construction**, and its passes are not evidence of action-sensitivity. That is an audit, not an experiment, and it is free. |
| **COMBINATION** | Their zero-action displacement diagnostic is our `actdiv`/hold-action control, built independently. **We are not behind on the diagnostic; we are behind on the objective that fixes what it finds.** Combined with F4 (VLWM naming FiLM-style conditioning as the entangling mechanism), P-2's open branch (c) *representation* now has **two independent published mechanisms** where a week ago it had none. |
| **CHANCES / RISKS** | ⭐ Upside: a pre-registerable, controls-included fix for the programme's gating defect at tiny-ladder cost. ⚠️⛔ **The risk that decides whether it transfers at all: our action channel IS realised motion (r 0.9988).** Decoding realised motion from `Dz` may be **near-tautological** in any competent encoder — LDAD could sit near zero loss from initialisation and buy nothing. That is P-2 branch (b) biting the *fix* rather than the model. ⚠️ Also: four continuous-control benchmarks, **no driving**, and their encoder trains from scratch while ours is pretrained. |
| **EXPERIMENT** | *(proposed, unranked)* **Two-stage, and stage 1 is free.** ⭐ **Stage 1 (0 GPU, banked latents):** fit an LDAD decoder on existing v7 latents and report its loss against **two controls — a constant-only predictor and a shuffled-action control**. ⛔ **Committed in advance: if LDAD loss is already near the floor at initialisation, the objective is tautological on our action channel, the line is REFUTED for us, and no arm is spent.** Only a non-trivial gap earns Stage 2 (an LDAD arm on the v7-tiny ladder). |

## F2 ⛔ B13 — SparseOcc++ debt D-5 DISCHARGED, and the pre-registered outcome CLOSES the line

`PUBLISHED lib 2607.04732 · FULL TEXT READ · debt D-5 DISCHARGED · backlog P-11 / guideline S-3`

Backlog **P-11** pre-committed the read: *"if it requires occupancy GT we cannot produce, the line is
CLOSED and we say so."* **It requires exactly that, on two independent supervision paths:**

1. **Dense per-voxel semantic occupancy GT.** *"To supervise SCF learning, we efficiently generate ground
   truth (GT) from **semantic occupancy labels**."* Training benchmarks are **SemanticKITTI** (19 semantic
   classes per voxel, LiDAR segmentation + scene-completion labels) and **nuScenes-Occupancy** (dense 3D
   semantic occupancy annotations, 16 classes + empty).
2. **LiDAR.** *"L_depth is calculated between the predicted depth map and the ground truth projected by
   **point clouds**"* — supervision for the LSS component.

⛔ **PhysicalAI-AV has neither.** No map, no lane graph, no occupancy labels (settled at five probes); the
only 3D source is `obstacle.offline` — **10 dynamic-agent classes, 87,481 cuboids** — an agent-box corpus,
not voxel occupancy. ⇒ **THE SPARSEOCC++ LINE IS CLOSED FOR TANITAD. Recorded because it was
pre-registered, not because it is convenient.**

⭐ **The track is NOT closed, and the paper names the escape hatch itself.** Its related work cites a live
**self-supervised / 2D-rendering-supervision** family needing no occupancy GT — SelfOcc (CVPR'24),
GaussianOcc (ICCV'25), GaussTR (CVPR'25), RenderOcc, OccNeRF. ⭐ **We hold the assets that family runs on:
NuRec msgpack is open and gsplat does 492 FPS on Thor.** That is where B13 goes next.

⭐ **One transferable ablation, independent of the supervision blocker.** Their Table V finds the *linear*
head achieves the **highest geometry IoU (36.8)** *"because the linear head is supervised by an explicit
geometry loss... Conversely, the transformer decoder formulates occupancy prediction as mask generation
for semantic classes only, **imposing no explicit supervision on occupied voxels**."*
⇒ **Explicit geometry supervision beats an implicit semantic formulation at geometry.** Our measured
readout-geometry ceiling (4 azimuth bins over 120 deg) is currently trained with no explicit geometry term.

| dim | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐ — closes a line the register held open and redirects guideline S-3. |
| **CONSEQUENCE** | P-11 retires as **served-and-closed**. The geometry differentiator survives but must be built on self-supervised occupancy, not SparseOcc++. |
| **COMBINATION** | The linear-head result says our geometry ceiling may be an **objective** gap as much as a **resolution** gap — a materially cheaper hypothesis than re-architecting the readout, and it has never been tested. |
| **CHANCES / RISKS** | ⭐ A closed line is a saved quarter, and it was closed by a rule we wrote in advance. ⚠️ The self-supervised family is younger and weaker; it does **not** inherit SparseOcc++'s +2.3 IoU / 3.9x speed, and assuming it does would be the scope error again. |
| **EXPERIMENT** | *(proposed, unranked)* Add an **explicit geometry loss** to the v7 readout; read azimuth-bin geometry decodability against the current implicit formulation. **Committed outcome: if explicit supervision does not move decodability, the ceiling is resolution-bound and the readout must be re-architected — the expensive branch, but we would then know it is the necessary one.** |

## F3 ⭐⭐ A5 — WorldRoamBench's drift metric is now fully specified: FS-1 is unblocked

`PUBLISHED lib 2606.31672 · FULL TEXT READ (abstract-only yesterday) · unblocks backlog FS-1`

**The formulation, exactly.** Partition the rollout into **N = 10** windows; take the **best-quality** and
**worst-quality** windows; report the **relative percentage change** between them.
> *"D = 0.05 means the worst 10% window of the video is 5% below the best 10% window in average score;
> unlike a fixed start/end comparison, this localizes the strongest dip wherever it occurs, so transient
> mid-rollout collapses that are later masked by recovery still contribute to D."*

Relative rather than absolute *"keeps drift comparable across models with different score magnitudes"*.

⭐ **The non-monotonicity FS-1 was designed to test is MEASURED, not hypothesised:**
- *"A subset of models additionally show transient mid-rollout collapse, dipping sharply mid-sequence
  before partially recovering; **a start-vs-end comparison would treat them as stable**."*
- `minWM`: *"selective collapse, with its aesthetic score crashing after frame 150 even as imaging stays flat"*.
- `Matrix-Game 3.0`: *"progressive, accelerating decline"*. `HY-World 1.5` is best (aesthetic drift
  **13.85**, imaging drift **15.91**); `Yume 1.5` *"starts strong yet accumulates 23.01 imaging drift"*.

⭐⭐ **A second independent confirmation of our action-echo finding, now with numbers:**
> *"Models with high trajectory alignment (trajectory score above 85) can exhibit **below 65% per-frame
> strict action accuracy**, revealing a hidden failure mode."*

⚠️⛔ **SCOPE LIMIT, AND IT CHANGES WHAT FS-1 MAY PORT.** Their drift is computed over **image-quality**
scores (aesthetic / imaging). **Ours is a trajectory/latent quantity.** ⇒ **Port the segment-based
best-vs-worst FORMULATION, not the metric.** Quoting their 13.85 / 23.01 against our drift numbers would
be the `df` / `step_s` scope error in a new costume.

| dim | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ — FS-1 is a 0-GPU row whose only blocker was this definition. |
| **CONSEQUENCE** | FS-1 is executable today. If our banked v7 drift is non-monotone, **every drift number the programme holds is a lower bound** — including the MM-E1 EMA read (0.4531 -> 0.36-0.40) currently deciding a recipe. |
| **COMBINATION** | Their per-frame-vs-trajectory gap and our 97.9% / 0.0% / ~5% action echo are the same phenomenon in two fields. Three independent lines now agree (ours, Alpamayo, WorldRoamBench) ⇒ guideline **S-1** (T1 is the burden of proof) strengthens. |
| **CHANCES / RISKS** | ⭐ An external, citable instrument instead of a bespoke one. ⚠️ Built for open-world interactive WMs; the physics and memory dimensions do not port, and only the drift formulation should be lifted. |
| **EXPERIMENT** | FS-1 as written, plus the scope correction: N=10 segments, best-vs-worst, **relative** change, over the banked v7 rollout dumps. **Committed outcome unchanged: monotone ⇒ the instrument stands; non-monotone ⇒ every drift number is a lower bound and the instrument must change.** |

## F4 ⭐⭐ B12 — VLWM names FiLM-style conditioning as the mechanism that hides action contributions

`PUBLISHED lib 2606.21775 · FULL TEXT READ (abstract-only yesterday) · lands on gate problem P-2`

> *"**Architecture-locked action conditioning.** Because actions are fused inside transformer layers, if we
> provide multiple actions, they will be compressed into shared hidden representations, **which makes it
> difficult to disentangle their individual contributions to the predicted dynamics**."*

VLWM replaces FiLM / cross-attention with **action-as-token**: *"treats each action embedding as a standard
transformer token and interleaves action and state tokens within a single sequence"*, which *"eliminates
the need for a predefined action-conditioning interface"*. Backbone-agnostic — *"the encoder f can be a
JEPA-style, DINO-style, or any other latent encoder"*. **13% average over LeWM.**

⛔ **This matters because our P-2 investigation ELIMINATED zero-init FiLM as a cause on the grounds that
"it trained" (MM-E18).** VLWM's claim is different and fully compatible with that observation: FiLM can
train perfectly well **and still entangle** the action's contribution. ⇒ **"It trained" was never evidence
that the conditioning interface is innocent.** P-2 branch (c) re-opens with a named mechanism.

⭐ **Independent corroboration of yesterday's L-1.** *"training with large prediction horizons from the
outset is highly unstable... often resulting in noisy gradients, optimization failure, and representation
collapse"*, fixed by a cumulative-uniform horizon **curriculum** they find *"essential"*. That is a
**second independent line** for L-1 (horizon curriculum, not a smaller clip) against our k=60 divergence.

⚠️ **An honest threat to our thesis, recorded as one.** VLWM claims variable-horizon prediction *"enables
hierarchical planning without requiring separate high-level and low-level policies"* — one predictor
subsuming the hierarchy. ⛔ It is an **argument**, not an ablation against a hierarchy; but it is the
clearest published challenge to H1 we have logged, and it must not be filed as support.

| dim | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ — P-2 branch (c), backlog row 5 (h>=2 heads), FS-2, L-1 and H1 positioning all move. |
| **CONSEQUENCE** | FS-2 ("keep the head slots") is **strengthened**: the slots are the cheap side of an argument that now has a mechanism behind it. Backlog row 5 should proceed but not delete. |
| **COMBINATION** | F1 + F4 give P-2 two independent mechanisms — the **decoder side** (endpoint leak) and the **conditioning side** (FiLM entanglement). Neither was on the Lab's seed list a week ago, and **both came from Band-B tracks** — the PI's transfer mandate paying out. |
| **CHANCES / RISKS** | ⭐ Action-as-token is contained and composes with our trunk. ⚠️ It rebuilds the predictor's input interface — not a tiny-ladder-cheap edit — and the gains are on control benchmarks, not driving. |
| **EXPERIMENT** | *(proposed, unranked)* A **conditioning-interface two-way at matched params** on the v7-tiny ladder: current FiLM vs action-as-token, read on the h=1 **action/scene ratio** (currently 0.004) and hold-action ADE. **Committed outcome: if action-as-token does not raise the ratio >=3x, the conditioning interface is exonerated and P-2 branch (c) closes — worth knowing, because it leaves (b), our action channel, as the sole survivor.** |

## F5 ⚠️ B2 — the I-2 conversion recipe is real, but its payoff regime is one we do not occupy

`PUBLISHED lib 2601.22156 · FULL TEXT READ (abstract-only yesterday) · sharpens injected row I-2`

**HALO** (*Hybrid Attention via Layer Optimization*) converts pre-trained Transformers into RNN-attention
hybrids. The load-bearing step is **which attention layers to leave unconverted**: *"our objective is to
identify which attention layers are most important for modeling recall abilities and leave them
unconverted."* Conversion costs **2.3B tokens, <0.01% of pre-training data**, against *">10B tokens"* for
prior methods.

⛔ **But the benefit is stated for EXTREME context, which is not our regime.** The efficiency case is made
at **128K and 1M** context (Qwen3 OOMs at 1M); hybrids *"enjoy significant inference speedups over
Transformer-based models"* precisely *"in long-context scenarios"*. **Our v7 predictor operates at
K in {8, 60, 300} steps.** The paper also concedes *"distilled hybrid models typically underperform those
trained from scratch"*.

⇒ **I-2's method is confirmed and its expected payoff is DOWNGRADED in the same read.** Yesterday's
re-scope ("convert, don't retrain") stands as the right *method*; today adds that the *motivation* may not
transfer at all. **This is the honest shape of a Band-B finding: the transfer question answered against us.**

| dim | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐ — injected row I-2 and backlog P-6. |
| **CONSEQUENCE** | P-6's pre-committed read (*"SSM wins only if ms/step is flat in K AND lower in absolute terms at K=60"*) is exactly the right gate and **should run before any conversion work** — it tests the regime question this paper leaves open for us. |
| **COMBINATION** | Cuts against P-7 (flat strategic rollout undeployable at K=300): an architecture with cost flat in K would matter enormously — but the evidence is at 1M context, not at K=300, and nothing here bridges that. |
| **CHANCES / RISKS** | ⭐ Layer-selection-by-recall-importance transfers as an idea independent of the SSM question. ⚠️ Adopting HALO on the strength of a long-context result would be the `INHERITED` failure the operating standard bans. |
| **EXPERIMENT** | Run **P-6's benchmark first** (0 GPU then 4060); gate conversion on it. **Committed outcome: flat-but-slower at K=60 refutes the line for TanitAD.** |

## F6 ⛔ CORRECTION — the "sub-50 ms driving-VLA latency requirement" is not in the primary

`PUBLISHED lib 2512.16760 · FULL TEXT READ · Hu et al., 2025-12-18 · debt D-6 (A4) DISCHARGED`

The A4 search summary asserted *"Achieving sub-50ms inference remains an unmet requirement for
safety-critical deployment"* and attributed it to this survey. **The primary read contradicts that: the
document specifies no latency requirement at all.** Latency appears only qualitatively — *"Sparse query
methods significantly reduce inference latency"* — with **no concrete numbers anywhere**, and no
parameter-count-vs-performance data in any table.

⛔ **Had it not been checked, a fabricated 50 ms budget would have entered our deployment reasoning beside
the MEASURED 100 ms budget and the MEASURED K=300 / ~1,015 ms figure (P-7).** ⇒ **A search-engine summary
is not a source, and it is not even `RELAYED` — it is unattributed.** Same corruption path as yesterday's
D-1 (85.1 % / 92.1 %) and B2 (25 %) corrections: **three days running, the load-bearing number that failed
was the one nobody had opened the primary for.**

**What the survey does give us**, and it is genuinely useful: a two-paradigm taxonomy — **End-to-End VLA**
vs **Dual-System VLA**, the latter separating *"slow deliberation (via VLMs) from fast, safety-critical
execution (via planners)"*. ⭐ That is a third opponent-adjacent architecture converging on a fast/slow
split (with Waymo's Sensor-Fusion-Encoder + Driving-VLM and our own hierarchy). It also names a limitation
directly on our side of the argument: *"the absence of a dense future-world representation can restrict
long-horizon reasoning and planning safety"* — an argument **for** a world model inside the loop.

| dim | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐ — A4 rotation, P-7, and H1 positioning. |
| **CONSEQUENCE** | No latency number enters any TanitAD doc from this source. The fast/slow convergence is recorded as *architectural context*, explicitly **unablated**. |
| **COMBINATION** | Dual-System VLA + Waymo's hybrid + our hierarchy = three independent stacks with a fast/slow split. ⚠️ Convergence of **designs** is not evidence of **necessity**; FS-3 remains the only way we learn whether the seam pays. |
| **CHANCES / RISKS** | ⭐ The *"absence of a dense future-world representation"* line is quotable support for the programme's premise. ⚠️ It is a survey — `PUBLISHED`, but nothing in it is an ablation. |
| **EXPERIMENT** | *(process, not compute)* Add a standing check: **any number entering a TanitAD document from a web search must be traced to the primary in the same turn, or carried as `UNATTRIBUTED` and barred from decisions.** Cost: minutes. This pass alone would have caught one. |

---

# BAND D — OPPONENT DOCTRINE (priority 1)

## F7 ⭐⭐⭐ Mobileye's "Compound AI" — an opponent argues OUR thesis, and backs it with a THEOREM

`Band D · adjudicated under amendment §7.1, all seven steps · 2 primaries`

### Step 1 — CLASSIFY THE DOCUMENT

| | |
|---|---|
| **Document** | *"Compound AI: The framework powering scalable autonomy"*, Mobileye blog, **2025-07-31**. Evidence class **`PUBLISHED-BLOG`**. |
| **Second primary** | Shalev-Shwartz & Shashua, *"On the Sample Complexity of End-to-end Training vs. Semantic Abstraction Training"*, **arXiv 1604.06915, 2016-04-23**. Evidence class **`PUBLISHED`** (preprint, no venue). ⭐ **Banked this pass.** |
| **Competitive context** | Mobileye is a Tier-1 supplier selling *modularity as a product*; the blog is advocacy for an architecture its business model depends on. Published while every visible competitor was marketing end-to-end. |

### Step 2 — ⛔ IS IT AN EXPERIMENT?

**The blog: NO.** No ablation, no A/B, no matched comparison anywhere. Its only external citation is
Berkeley AI Research's general observation that *"state-of-the-art AI results are increasingly obtained by
compound systems"*. Its only numbers are hardware specs (EyeQ6 High **34 TOPS INT8**; *"over 1,000 frames
per second on pixel-labeling neural networks"*) — **specifications, not evidence for the architecture claim.**

⭐⭐ **The 2016 paper: NEITHER an experiment NOR attribution-from-exposure — it is a PROOF.** This is a
category the adjudication protocol had not yet met. It claims a **sample-complexity separation**:
> *"We demonstrate cases in which the number of training examples required by the end-to-end approach is
> exponentially larger than the number of examples required by the semantic abstraction approach."*

⚠️ **And the qualifier is load-bearing: *"We demonstrate CASES in which..."*** — a **constructed
existence result**, not a general theorem about driving. It proves the separation *can* occur, not that it
*does* occur in any real driving stack. **A proof of possibility read as a proof of necessity would be the
scope error our own retraction log is built around.** The paper also declares its regime explicitly:
*"in the regime where an extremely high accuracy is necessary"*.

### Step 3 — CONFIRMING EVIDENCE (independent)

- **Waymo's production stack** (read 2026-09-01, register W-13): explicitly hybrid — Sensor Fusion Encoder
  + Driving VLM + World Decoder, *"learned embeddings as a rich interface between model components"*.
  Independent of Mobileye and commercially the most exposed operator in the field.
- **Driving-VLA survey `2512.16760`** (F6): **Dual-System VLA** is a named, populated architecture family
  separating slow reasoning from fast execution.

### Step 4 — ⭐ CONTRADICTING EVIDENCE (actively sought, mandatory)

**Sought explicitly, and found — this claim does not survive as stated.**
- `2603.16050` (*The Era of End-to-End Autonomy*): the field records a *"decisive architectural
  transition"* in which end-to-end networks *"began to outperform modular stacks in terms of performance
  and passenger comfort metrics"*, with modular approaches *"increasingly brittle when confronted with the
  combinatorial complexity of real-world urban traffic"*.
- `2412.09602` (*Hidden Biases of End-to-End Driving Datasets*) — an active, credible E2E research line.
- **Tesla and Wayve** ship end-to-end commercially; Wayve *"employs a single end-to-end neural network...
  rather than breaking driving into separate perception, planning, and control modules"*.

⚠️ **Both bodies of evidence are weak in the same way**: the E2E-superiority claims are also largely
unablated at matched data and parameters. **Neither side has run the experiment.**

### Step 5 — VERDICT

| claim | verdict |
|---|---|
| **M-1** *Compound/modular AI is the right architecture for scalable autonomy* | ⚠️ **CONTESTED.** Confirmed by two independent stacks (Waymo, Dual-System VLA); contradicted by a documented field-wide shift toward E2E and by two commercial E2E deployments. **No controlled comparison exists on either side.** |
| **M-2** *Semantic-abstraction training needs exponentially fewer samples than end-to-end* | ⚠️ **SUPPORTED-AS-A-POSSIBILITY, UNSUPPORTED-AS-A-GENERAL-CLAIM.** The proof is a constructed existence result in a high-accuracy regime; it does not establish that real driving decomposes this way. |
| **M-3** *Modularity/redundancy/abstraction yield resilience across edge cases* | ⛔ **UNSUPPORTED-AS-STATED.** Asserted with zero ablation, zero metric, and one acknowledged trade-off. |

### Step 6 — ⭐⭐ DOES IT BIND ON US? (separate from whether it is true)

⛔ **PARTIALLY — and the distinction is one we could easily get wrong in our own favour.**

**TanitAD's hierarchy is NOT Mobileye's compound AI**, and M-2 does not transfer to it for free:

| | Mobileye | TanitAD |
|---|---|---|
| components | **hand-specified**, semantically named, with formal interfaces (RSS, REM) | **learned latent** strategic / tactical / operative levels |
| decomposition | **given by design** | **must itself be learned** |
| interfaces | engineered + formally verifiable | differentiable embeddings |

⛔ **The 2016 separation assumes the semantic decomposition is GIVEN.** Its sample-complexity advantage
comes from supervising components directly. **A hierarchy whose decomposition must be discovered does not
automatically inherit it** — and quoting M-2 as support for H1 would be exactly the kind of borrowed
authority the operating standard forbids. ⇒ **M-2 raises our prior for H1b. It is not evidence for it, and
it decides no GPU-day.**

⭐ **What DOES bind:** Mobileye is a third independent stack whose *deployed* architecture has a fast/slow
or modular seam. Combined with Waymo (W-13) and the Dual-System VLA family, **"pure end-to-end" is not the
industry consensus our positioning documents have sometimes assumed it to be.**

### Step 7 — GUIDELINES

- **STRATEGIC (S-5, proposed):** State in the paper that **the modular-vs-E2E question is unresolved and
  unablated on both sides**, and that TanitAD's contribution is *to run the matched-params comparison
  nobody has run* (backlog row 18, H1b). ⭐ **This converts our biggest positioning weakness — an untested
  thesis hypothesis — into the differentiator: we are the ones proposing to settle it.**
- **TACTICAL (T-3, proposed):** ⛔ **Never cite `1604.06915` as support for TanitAD's hierarchy without
  the given-vs-learned decomposition caveat.** Recorded in the register so a future pass cannot quote it
  bare.

### ⚠️ Opponent strengths recorded (amendment §7.2 — a Band-D package listing none is INCOMPLETE)

1. **Mobileye has a real theoretical argument, which is rare in this literature.** Waymo's doctrine post
   had zero ablations and zero theory; Mobileye has a proof, however narrow. **On evidence quality for an
   architecture claim, Mobileye is ahead of Waymo and ahead of us.**
2. **Shipping silicon with published efficiency numbers** (EyeQ6 High, 34 TOPS INT8, >1,000 FPS on
   pixel-labeling nets). Our efficiency claim is still a thesis; theirs is a part number.
3. **Formal safety interfaces (RSS) give them something we lack entirely** — a verifiable envelope
   independent of the learned components. Our safety story is a safety *case*, not a formal guarantee.
4. **Consistency:** they have argued this position since 2016 and their product line reflects it. That is
   not evidence, but it is not marketing drift either.

---

# BAND C — sweep

## F8 — C3: the online-learning ban is unfound at a THIRD probe; the primary is still 403 (D-4 stands)

`Band C / C3 · third independent probe · primary UNREAD · injected row I-3`

A legal analysis of the UN GTR (Sidley, **2026-03-04**, Raviv & Wittenberg) covers the regulation's
obligations in detail and states **online learning, continuous/self-learning, post-deployment model
changes, and re-approval of model updates are NOT MENTIONED**. It does document:
- **ISMR:** *"manufacturers must have processes to monitor ADS operations, investigate and report
  safety-relevant occurrences to authorities, and use those learnings to refine hazards."*
- **DSSAD:** *"a data storage system for automated driving (DSSAD) capability to record and store
  safety-related ADS performance data, with protections against unauthorized access or manipulation."*
- ⭐ **SMS:** *"a safety management system (SMS) that governs safety across the vehicle's entire life
  cycle, spanning development, production, deployment, and **post-deployment**."*
- The GTR's *"center of gravity"* is **the safety case**.

⭐ **That last item is the actual hook for I-3, and it is more useful than a ban would have been.**
Post-deployment change falls inside the SMS lifecycle ⇒ **an online-adapting encoder is a safety-case
obligation, not an illegality.** I-3's design question is *"can we evidence an adapting model under ISMR /
DSSAD / SMS?"*

⚠️⛔ **BUT: a secondary's silence is not the primary's silence.** *"Not mentioned in this analysis"* is
weaker than *"not in the regulation"*, and the analysis is a summary written for a different purpose.
**The GRVA primary returned HTTP 403 at a third distinct route today.** ⇒ **Status remains
"unsupported at three probes", NOT "refuted", and debt D-4 stands.** *(Escalated: this needs a route we do
not have — see the run summary's PI-queue.)*

## F9 — C4: the first leaderboard numbers the programme holds, and a sharpened efficiency wedge

`Band C / C4 · PUBLISHED (paper claims) · ⛔ NOT read off a leaderboard — see empty E7`

| system | score | split |
|---|---|---|
| **DrivoR** | **56.3 EPDMS** | NAVSIM v2 |
| **PDM-Closed** (privileged, GT perception) | **56.6 EPDMS** | NAVSIM v2 |
| **CLOVER** | **48.3 EPDMS** | navhard-two-stage |
| **RAP-DINO** | **36.9 EPDMS** | NAVSIM v2 |

⭐⭐ **DrivoR is within 0.3 EPDMS of a PRIVILEGED planner that consumes ground-truth perception.** With
backlog row 32's ~40 M parameter figure, that sharpens the efficiency wedge considerably: **"sub-300M" is
not merely community-demonstrated, a ~40 M camera-only model is at parity with a GT-perception planner on
this benchmark.** ⛔ Our efficiency claim must be restated against **that**, not against 32 B models.

⚠️⛔ **V-5 GUARDS, BOTH BINDING:** (1) **these splits are different** — NAVSIM v2 overall and
navhard-two-stage must never be pooled into one ranking; (2) **EPDMS is comparable only within one
scoring-basis era**, and none of these has been cross-checked against our four-family definitions
(D-EPDMS-FAM is still open). **No number here may enter a comparability table until backlog row 3's
portfolio decision and D-EPDMS-FAM land.**

## F10 — B9 SCAN: curation as the free lever (scan only, primary unread)

`Band B / B9 · SCAN · MiniWorld 2608.01127 + Summer-22B 2603.00173 banked/identified`

Surfaced **MiniWorld** (*"Democratizing the Training of Video World Models from Scratch"*, banked) and
**Summer-22B** (metadata-driven curation, multi-stage filtering, ~50 M clips). ⚠️ **The headline curation
claim — Open-Sora 2.0 filtering 70 M raw samples to a 10 M core, with *"filtering for aesthetic scores and
motion coherence yield[ing] higher returns than simply scaling dataset size"* — is `RELAYED` via a survey
summary. Primary NOT read. It may not decide anything**, and it is recorded here only because it is the
first B9 signal the programme has and it lands on backlog row 22 (H-DATA-1: does frozen-feature leverage
predict a clip's value?). **B9 DEEP is owed.**

---

## ⭐ MEASURED process finding — banking is still substituting for reading

`MEASURED (this pass, kb_add.py output + library.json probe)`

**6 of the 8 primaries this pass needed were ALREADY BANKED**: `2607.04732`, `2601.22156`, `2606.31672`,
`2606.21775` (the four carried debts) and — the sharper one — **`2606.31232` Delta-JEPA, which the web
surfaced as a discovery and which `kb_add` answered with *"already banked"***.

⛔ **Delta-JEPA is the most consequential finding of the pass and we have been holding it, unread.**
Yesterday's F6 measured 5 of 6; today is 6 of 8, and the ratio is no longer a curiosity — **it is the
Lab's dominant inefficiency.** Library now **313 entries / 2,136.4 MB**.

> ⭐ **Restating yesterday's rule with today's evidence: a banked primary with no five-dimension analysis
> is not an asset, it is a debt with a hash — and the highest-value item in the programme can sit in it.**

⇒ **FS-5 (banked-but-unread count per track in `TRACKS.md`) is no longer a hygiene row. It is the row that
would have surfaced Delta-JEPA a week earlier.** Escalated accordingly.

---

## What this changes for TanitAD — 3 recommendations (cap respected)

1. ⛔⛔ **Audit every action-decodability probe for the endpoint-concatenation leak, this week, 0 GPU (F1).**
   If a probe decodes the action from `[z_t, z_t+1]`, its passes are **not evidence of action-sensitivity**
   and any P-2 branch eliminated on its say-so must be re-opened. **This is an instrument-validity question
   sitting underneath the programme's gating problem — it outranks every architectural arm.**
2. ⭐⭐ **Run the free half of the LDAD experiment before designing any arm (F1).** Fit the displacement
   decoder on banked v7 latents against a constant-only control and a shuffled-action control.
   **Committed: near-floor loss at init ⇒ tautological on our action channel ⇒ line refuted, no arm spent.**
   This is the cheapest test of the programme's most expensive open question.
3. ⛔ **Close the SparseOcc++ line and say so (F2); redirect S-3's geometry differentiator to the
   self-supervised occupancy family we already hold assets for.** The pre-registered outcome was met —
   honouring it is what makes the next pre-registration worth writing.

## Load-bearing items this pass (V-2)

**Full primaries read (6):** `2606.31232` (Delta-JEPA) · `2607.04732` (SparseOcc++) · `2606.31672`
(WorldRoamBench) · `2606.21775` (VLWM) · `2601.22156` (HALO) · `2512.16760` (driving-VLA survey).
**Band D primaries:** Mobileye Compound AI blog (full) · `1604.06915` (abstract + framing; **full text not
read — the construction behind *"cases in which"* is unexamined and is a debt**).
**RELAYED, decision-barred:** the Open-Sora 70M→10M curation figure (F10).
**Most decision-relevant:** F1, then F2 (a line closed), then F3 (a 0-GPU row unblocked).

## Completeness self-report (charter §6 + amendment §8.1, fail-loudly clause)

| requirement | status |
|---|---|
| Band D adjudicated (all 7 steps) | ✅ 2 primaries, 3 claims, **counter-search recorded and it changed the verdict to CONTESTED** |
| Opponent strengths recorded | ✅ 4, including one where the opponent is **ahead of us on evidence quality** |
| >=2 full-text deep reads | ✅ **6** |
| Band A | ✅ A2 (Delta-JEPA), A4 (VLA survey), A5 (WorldRoamBench) — **debt D-6 DISCHARGED** |
| >=3 Band B deep-reads | ✅ B2, B12, B13 (all three full-text upgrades) + B9 SCAN |
| Band C sweep | ✅ C3 (third probe), C4 (first numbers) |
| Ledger appends | ✅ A2 **(created)**, A4 **(created)**, A5, B2, B12, B13, C1, C3 — **FS-6's two broken links FIXED** |
| Every empty named | ✅ 4 (E1 carried to 3 probes; E7, E8, E9 new) + 1 tool failure + **1 self-inflicted false negative logged** |
| Debts discharged | ✅ **D-5** (SparseOcc++ ⇒ line CLOSED), **D-6** (A2+A4), FS-1 blocker, FS-6 |
| **Debts standing / new** | ⛔ **D-4** UNECE primary (**3 failed routes** — needs a PI-side route) · **D-7 (new)** `1604.06915` full text unread · **D-8 (new)** B9 primary unread |
| ⛔ **Band-C/B BREADTH** | ⛔ **FAILED — 12 of 22 tracks unscanned (A1, A3, B1, B3-B8, B10, B11, C2), for the SECOND CONSECUTIVE DAY.** Yesterday this was a deliberate one-off budget choice; **twice is a pattern and is escalated, not excused.** |
