# Wayve — full opponent deep-dive (Opponent Analyzer, 2026-08-02)

> Commissioned by Sayed as a standalone deep-dive, outside the weekly sweep cadence. Companion to
> `2026-08-02-opponent-sweep-run5.md` (run #5). Evidence labels per G-O1: **FACT** (verified at a primary
> or named source) · **CLAIM** (reported, not independently verified) · **INFER** (my reasoning) ·
> **MEASURED** (ours, with artifact path).
>
> **Why this is worth a dedicated note:** Wayve is the opponent whose *technical thesis* is closest to
> ours (end-to-end learned driving + world models + no HD maps, from a research-first lab) and whose
> *architecture* is furthest from ours (one flat policy, world model deliberately outside the loop).
> Their published record is unusually rich, so most of this is FACT rather than INFER — which is not
> true of any other profile in the catalog.

---

## 0. Executive summary — six things, three of which change what we should do

1. **Wayve is not one architecture, it is two stacks.** A comparatively small on-car end-to-end policy,
   and a very large *offline* world-model / simulation / evaluation factory. The famous 15 B model does
   not drive the car. **Wayve's world model is an EVALUATOR; ours is a DRIVER.** That distinction is our
   cleanest differentiation — and it is now **FACT from Wayve's own page**, not an inference (§1.3).
2. **…but the mirror of that is uncomfortable: the thing we lack is the thing they are best at.** Ghost
   Gym + PRISM-1 + GAIA-3 is a *working closed-loop, photorealistic, camera-only* evaluation stack. We
   are walled on closed-loop (AlpaSim NO-GO, CARLA pixels host-blocked). They solved our #1 blocker as
   a side-effect of their data strategy (§5.1).
3. **LA-Pose (arXiv 2604.27448, 2026-04-30) is our H7, published, at 10.2 million clips.** Wayve trains
   an inverse-dynamics model on 10.2 M unlabelled driving clips to learn latent actions, then reads
   camera pose — *including field-of-view and metric scale* — off them. Those are **precisely the two
   gaps our own IDM pilot recorded**, and our pilot ran on **80 clips**. This is the single most
   program-relevant finding in this note (§5.2).
4. **One of their own results is evidence FOR our latent thesis and against their flagship.** LA-Pose
   found a **50-dimensional** latent bottleneck beat a 1,536-dimensional one *despite worse video
   reconstruction*. A small non-reconstructive latent beating a large reconstructive one is the core
   LeJEPA/SigReg argument (H3) — demonstrated in-house by the company whose headline model is a 15 B
   *reconstructive* diffusion model (§4.2).
5. **They have made the opposite safety bet to ours, explicitly.** Wayve's "Safety 2.0" is framed
   *against* scenario enumeration; our `SCENARIO_DATABASE.md` is scenario enumeration. Both bets are
   currently unproven, and the W-09 regulatory evidence cuts against theirs (§4.4).
6. **Their public safety and generalization claims have no denominators.** "40X improvement", "3X
   better", "fivefold" — all relative multipliers off an undisclosed baseline; no miles, no intervention
   rate, no thresholds, no OOD methodology. This is **W-11 in its purest form**, from the most technical
   opponent we track (§4.3).

---

## 1. The architecture, reconstructed

### 1.1 On-car: AV2.0 — one flat policy

**FACT.** Wayve's "AV2.0" replaces the modular sense-plan-act stack with **a single neural network**
converting raw sensor input into driving output; **camera-first + radar, mapless** (no HD maps), trained
largely self-supervised on petabyte-scale fleet and partner data. They **license software** rather than
operate vehicles or build cars. — <https://wayve.ai/technology/av2-0/> , <https://wayve.ai/technology/>

**FACT.** **LINGO-2** (Apr 2024) is the on-policy face of this: a vision model coupled to an
auto-regressive language model that **outputs a driving path *and* a natural-language commentary on its
own motion-planning decisions**, and accepts natural-language *instruction* to modulate behaviour.
Described as the **first closed-loop vision-language-action driving model tested on public roads**;
developed against Ghost Gym and on road. — <https://wayve.ai/thinking/lingo-2-driving-with-language/>

**FACT.** **Gen 3** (the current vehicle platform) runs on **NVIDIA DRIVE AGX Thor**, Blackwell,
**up to 2,000 FP4 TFLOPS**, DriveOS; targets **eyes-off L3 and driverless L4 across urban and highway**.
Wayve states it runs end-to-end models, **generative AI models in real time**, and LINGO-style language
interaction on-vehicle. Wayve is separately **evaluating NVIDIA Cosmos** for edge-case scenario search
(development, not onboard). — <https://wayve.ai/thinking/wayve-gen-3/>

**⚠️ The one number they have never published: the on-car model's parameter count.** Checked four ways
(their technology pages, the GAIA posts, the Gen-3 post, targeted search). They disclose **GAIA-1 = 9 B**
and **GAIA-3 = 15 B** — both offline — and nothing for the deployed policy. A secondary source cites a
**75-watt** deployable model (**CLAIM**, not from Wayve). **INFER:** disclosing the offline model's size
and not the on-car one is a choice; a company with a favourable on-car efficiency number generally
publishes it.

### 1.2 The offline factory — five models, one pipeline

| model | date | what it is | role |
|---|---|---|---|
| **FIERY** | Oct 2021 | future instance prediction in BEV from surround cameras | research ancestor |
| **MILE** | Nov 2022 | model-based imitation learning, latent world model, urban driving | research ancestor — **closest to our design** |
| **GAIA-1** | Sep 2023 | **9 B** generative world model, video+text+action → driving video (4,700 h UK) | generation |
| **Ghost Gym** | Dec 2023 | **closed-loop neural simulator**: neural renderer + simulated robot car + vehicle-dynamics model; action feeds back | **closed-loop eval** |
| **LINGO-1 / -2** | Nov 2023 / Apr 2024 | VLA; commentary + language-conditioned driving | on-car + introspection |
| **PRISM-1** | Jun 2024 | 4D photorealistic scene reconstruction from **cameras only** (no LiDAR) | powers next-gen Ghost Gym |
| **GAIA-2** | Mar 2025 | controllable **multi-view** generative world model (arXiv 2503.20523) | generation |
| **Rig3R** | Oct 2025 | geometric foundation model: multi-camera 3D + ego-motion, **rig-metadata conditioned** (NeurIPS 2025 Spotlight) | geometry |
| **GAIA-3** | **2 Dec 2025** | **15 B latent-diffusion world model** for **offline evaluation** | **evaluation** |
| **LA-Pose** | **30 Apr 2026** | IDM latent-action pretraining on **10.2 M unlabelled clips** → camera pose (arXiv 2604.27448) | **data efficiency** |

Sources: <https://wayve.ai/thinking/category/research/> , <https://wayve.ai/science/> , and the
per-model pages cited below.

### 1.3 GAIA-3 in detail — and the sentence that matters most to us

**FACT** (Wayve's own page and press release, 2 December 2025, London):

- **15 B parameters**, latent-diffusion; **video tokenizer twice the size** of GAIA-2's; **5× the compute**
  and **~10× the data** of GAIA-2; data spans **9 countries / 3 continents**, weighted toward
  safety-critical elements (pedestrians, cyclists, signs, traffic control).
- Pipeline: a **video tokenizer** compresses pixels into a latent space; a **latent diffusion world
  model** predicts future latents conditioned on past observation, **ego action (speed, curvature)**,
  **3D bounding boxes of other agents**, **environment** (weather, time of day) and **road attributes**
  (lane count, speed limits, cycle/bus lanes, zebra crossings, intersections, traffic lights).
- Control surfaces: action conditioning, **"World-on-Rails" perturbations** (change ego trajectory,
  hold the rest of the scene fixed), **text**, **embodiment / camera-rig** transfer, appearance.
- Claims: **"reduced synthetic-test rejection rates fivefold"**; **"GAIA-3 simulated testing closely
  mirrors real-world driving results"**; LiDAR-alignment consistency checks; correlation studies that it
  **"reliably predict[s] relative policy performance."**
- **Explicitly positioned for offline evaluation and safety validation, NOT real-time in-vehicle
  deployment.**

— <https://wayve.ai/thinking/gaia-3/> , <https://wayve.ai/press/wayve-launches-gaia3/>

**That last bullet is the load-bearing fact of this whole note.** Our profile has carried "GAIA stays an
offline data/eval factory" as an *inference from absence* since run #1. It is now an **affirmative
statement by Wayve**. The strategic consequence is in §5.

**INFER — but flag the strength of their claim, do not dismiss it.** "Our generative model reliably
predicts relative policy performance" is a *ranking* claim, and it is exactly the claim our own
Benchmarks & Eval work found fragile in the open-loop case (open-loop ADE ⊥ closed-loop DS, arXiv
2605.00066, with documented ranking inversions). Wayve is asserting they have a surrogate that ranks
policies correctly. **If true it is a very large advantage.** It is also **unreviewable**: GAIA-3 has a
blog post and a press release and, as far as I can find, **no paper** (§3.3).

### 1.4 LA-Pose in detail — the one to read

**FACT.** *LA-Pose: Latent Action Pretraining Meets Pose Estimation*, **arXiv 2604.27448**, submitted
**2026-04-30**, seven authors (Zhengqing Wang, Saurabh Nair, Prajwal Chidananda, et al.), CC BY-NC-ND.

- **Inverse- and forward-dynamics models** learn a **latent action** — a compact representation of how
  the world shifted between consecutive frames — from **10.2 million unlabelled driving clips**,
  Genie-inspired. The model is **never told speed or heading**; it infers them self-supervised.
- The learned latent actions **cluster into straight / left / right / stopped without a single pose
  label** — "structure emerges from scale."
- A **lightweight pose head** then converts motion features into 3D camera pose — **translation,
  rotation, field-of-view and metric scale** — in a **single forward pass**; finetuned on limited 3D
  annotations from **Waymo, nuScenes, Argoverse**.
- **A 50-dimensional latent bottleneck was optimal, beating a 1,536-dimensional one despite worse video
  reconstruction** — compression forces the representation to prioritise motion.
- Results: **>10 % higher pose accuracy than recent feed-forward methods** on Waymo and **PandaSet**;
  on PandaSet (**unseen**) it **outperforms all baselines** — a zero-shot generalization result.
- Stated purpose: generalize "to a new city or scenario **without running another LiDAR rig or building
  another labelled dataset**."
- **Acknowledged limitation:** performance degrades in **reverse motion** (few backing-up examples).

— <https://arxiv.org/abs/2604.27448> , <https://wayve.ai/thinking/la-pose/>

### 1.5 Rig3R — the multi-rig answer we happen to need

**FACT.** Rig3R (15 Oct 2025, **NeurIPS 2025 Spotlight**): shared **ViT-Large** image encoder → patch
tokens; a **metadata embedding layer encoding camera ID, timestamps and raymap calibration**; a second
**ViT-Large multiview decoder** fusing spatial/temporal/geometric information across views; three heads
emitting **pointmaps, pose raymaps and rig raymaps** in one forward pass. **Beats baselines by 17–45 %**
on real driving benchmarks; SOTA on Waymo Open and WayveScenes101; **improvements are most pronounced on
unseen camera configurations** when rig metadata is available. — <https://wayve.ai/thinking/rig3r/>

---

## 2. Business, deployment and capital (FACT unless marked)

- **Series D: $1.2 B primary (Feb 2026), $8.6 B post-money**, led by **Eclipse, Balderton, SoftBank
  Vision Fund 2**; returning Microsoft / NVIDIA / Uber; new Ontario Teachers', Baillie Gifford, British
  Business Bank, Icehouse, Schroders; automakers **Mercedes-Benz, Nissan, Stellantis**. Plus **$300 M
  from Uber conditional on deploying robotaxis in London** → **$1.5 B total potential**.
  ⚠️ **Discrepancy noted, not adopted:** a TechCrunch headline reads **"$1.8 B"** while its own body
  states $1.2 B + $300 M = $1.5 B at $8.6 B post. **Our catalog figure ($1.2 B / $1.5 B / $8.6 B) is
  correct and stands**; do not quote the $1.8 B headline.
  — <https://techcrunch.com/2026/02/24/self-driving-tech-startup-wayve-raises-1-8b-from-nvidia-uber-and-three-automakers/>
- **Series-D extension +$60 M** from **AMD, Arm, Qualcomm** (multi-SoC breadth). **$85 M employee tender
  (2026-07-01)** — liquidity, not new capital. **NVIDIA LOI for a proposed $500 M** dates from
  **2025-09-18** and is *superseded* by the Series D — do not re-report it as new.
- **Deployment reality check (this is a weakness, see §4.5):** first commercial launch is **Ford Mustang
  Mach-E on the Uber app in London, with a safety driver on board**, waiting on regulator go-ahead;
  Uber has opened a **London waitlist**. UK targets **fully driverless in 2027**; AV Act regulations to
  be updated **H2 2026**. **Waymo plans a London passenger service by Q3 2026 and has signalled its cars
  may carry no driver from the start.** Nissan integrates Wayve into ADAS **from 2027**; a
  **Wayve + Nissan + Uber Tokyo robotaxi pilot** is planned for **late 2026** on NVIDIA DRIVE Hyperion.
  — <https://techcrunch.com/2026/06/08/uber-wayve-and-waymo-are-headed-towards-a-robotaxi-showdown-in-london/> ,
  <https://wayve.ai/press/wayve-nissan-robotaxi-gtc/>

---

## 3. Strengths — stated honestly, because underrating them is how we lose

### 3.1 A closed-loop, camera-only evaluation stack that actually exists
Ghost Gym (neural renderer + dynamics model + action feedback) → PRISM-1 (4D reconstruction **from
cameras only, no LiDAR rig**) → GAIA-3 (controllable counterfactual generation + policy ranking). **This
is a complete answer to "how do you evaluate an end-to-end driver without driving a billion miles."**
We have no closed loop at all. **(INFER: this, not the on-car model, is their deepest moat.)**

### 3.2 Genuine multi-country generalization, with a data cost attached
**FACT** (Wayve's own numbers): UK→US required **500 hours of incremental US data over 8 weeks** to reach
UK-equivalent performance (100 h → "fivefold", 500 h → "40X"); **Germany zero-shot was "3X better" than
the initial US deployment** without fine-tuning; a new vehicle platform took **100 h** for "8X".
— <https://wayve.ai/thinking/multi-country-generalization/>
**INFER:** the second transfer being cheaper than the first is the classic multi-domain generalization
signature, and it is the strongest published evidence that the AV2.0 bet works. **500 hours to cross a
country boundary is the number our data-efficiency story has to beat or reframe.**

### 3.3 Research depth, and a correction to my own first reading
Rig3R took a **NeurIPS 2025 Spotlight**; LA-Pose is on arXiv; GAIA-1/-2 have technical reports.
⚠️ **Self-correction worth recording:** my first three probes — Wayve's `/science/` page, an arXiv
`all:Wayve` query (**0 results**), and an arXiv author search for Kendall — all suggested Wayve had
published nothing after March 2025. **That was wrong.** The `/thinking/category/research/` archive shows
**Rig3R (Oct 2025)** and **LA-Pose (May 2026)**. *Cause: arXiv does not index the affiliation string, and
Wayve's recent first authors are not the founders.* Operating-standard #2 earned its keep here — three
agreeing probes were still an insufficient basis for an absence claim.

### 3.4 Distribution, capital and compute-platform optionality
Uber (+$300 M milestone capital, London/Tokyo), three automakers as *investors and customers*
(Mercedes, Nissan, Stellantis), and a deliberate spread across **NVIDIA, AMD, Arm, Qualcomm** silicon.
They are not hostage to one SoC roadmap. UK/EU regulatory access is native.

---

## 4. Weaknesses — mechanism first, then the evidence

### 4.1 The world model does not drive the car (W-05)
**FACT:** GAIA-3 is 15 B and explicitly **not for real-time in-vehicle deployment**. **INFER:** a
generative pixel-decoding diffusion model is the wrong object to put in a 10 Hz decision loop, and Wayve
has architected around that rather than solving it. So the on-car policy remains a **flat end-to-end
network with no hierarchical decomposition, no imagination-in-the-loop, and no separable strategic
layer**. Every capability their world model has — counterfactual rollout, "what if I did this instead" —
is available **at validation time, not at decision time**.
**This is the differentiation. It is also the thing we have not yet proven we can do** (our run-#5 SC-13
result says our open-loop probe cannot demonstrate in-loop imagination; see §5.3).

### 4.2 An unresolved internal tension: reconstruct or predict?
**FACT:** LA-Pose reports a **50-d** latent beating a **1,536-d** one for motion tasks *despite worse
video reconstruction*. **FACT:** GAIA-3 is a 15 B model whose entire objective is high-fidelity
reconstruction/generation. **INFER:** Wayve's own smallest, newest, most efficient result argues that
reconstruction fidelity is not what downstream driving needs — which is precisely the JEPA/SigReg premise
(H3). They currently run both bets. **We should cite their 50-d result as third-party support for ours**,
and we should expect them to notice the tension before we do.

### 4.3 No denominators anywhere (W-11)
**FACT:** the safety framework page contains **no quantitative safety metrics, no thresholds, no runtime
monitoring detail, no OOD detection methodology**. **FACT:** the generalization numbers are **relative
multipliers off an undisclosed baseline** ("40X", "3X", "fivefold") with **no miles, no intervention
rate, no absolute performance figure**. **FACT:** GAIA-3's "reliably predicts relative policy
performance" has **no paper** behind it.
**INFER:** this is not sloppiness, it is a general condition of the field (IIHS, 2026-07-31: most
operators do not report miles driven, so no crash rate is computable — W-11). **It is also the axis on
which a small, rigorous program can visibly out-argue a $8.6 B one.** Our numbers ship with n, the
estimator, the falsifier, and a retraction log.

### 4.4 They bet *against* scenario enumeration — and the regulator is testing that bet
**FACT:** Wayve's "Safety 2.0" is framed explicitly *against* relying on extensive scenario modelling;
their claim is that "true safety comes from an AI that interprets the driving environment naturally, like
a human driver," supported by scenario-intelligence tooling, data-quality filtering and **natural-language
model introspection**. **INFER:** this is a coherent, genuinely different bet from ours, and it is **the
direct opposite of `SCENARIO_DATABASE.md` (H6)**. Two observations cut against them: (a) NHTSA's W-09
finding is that emergency-scene handling is a **"functional insufficiency"** across operators —
a *specific competence* failure, which is easier to attack scenario-wise than "interpret naturally";
(b) **LINGO-style introspection is post-hoc explanation, not a guarantee** — a commentary track is not a
runtime monitor with a threshold. **Their introspection is narrative; H11 asks for a number.**

### 4.5 Behind in their home market, and split across two products
**FACT:** first commercial launch is **safety-driver** Mach-E rides on Uber London, pending regulators,
with UK driverless targeted **2027** — while **Waymo targets a London service in Q3 2026, possibly
driverless from the start**. **FACT:** Wayve simultaneously pursues consumer L2+/L3 ADAS (Nissan from
2027, Mercedes, Stellantis) and L4 robotaxi. **INFER:** the licensing model that makes them
capital-efficient also means they do not control the deployment timeline; and a lab optimizing one
foundation model for both an eyes-off consumer ADAS and a driverless robotaxi is carrying two very
different safety cases on one network.

### 4.6 Sensing posture
**FACT:** camera-first + radar, mapless; PRISM-1 explicitly removes the LiDAR dependency for
reconstruction. **INFER:** this is a cost/scalability strength and a **degraded-visibility exposure**
(W-04) — the failure class that produced the Tesla engineering analysis and the Zoox smoke recall. Wayve
publishes **no calibrated epistemic-uncertainty mechanism**, so the W-04 mechanism (confident when the
sensing channel degrades) is not visibly addressed. **Honest caveat:** absence of publication is not
absence of mechanism — this is INFER, and it is the one place where a second probe would be worth doing
if this becomes a load-bearing claim.

---

## 5. Consequences for TanitAD — what actually changes

### 5.1 The closed loop stops being a nice-to-have and becomes the program's critical path
Run #5 measured that our **open-loop SC-13 probe cannot demonstrate in-loop imagination** (≈64 % of the
anticipation signal survives with the scene destroyed; ≈5 % is motion). Independently, this note shows
Wayve **deliberately keeps its world model out of the loop** — so "imagination at decision time" is
genuinely unoccupied ground. **Both facts point the same way: the only remaining test of H15 is
closed-loop, and it is now also our clearest differentiator.**
**And Wayve shows a route around our block.** We are stuck because AlpaSim was a NO-GO on the eval pod
and CARLA pixels are host-blocked — both **renderer** problems. **Ghost Gym + PRISM-1 is a
neural-renderer route: 4D reconstruction from camera-only logs, then closed-loop rollout against the
reconstruction.** That needs GPUs, not a graphics-capable container.
**→ Action (Tools & DevEnv + Architecture, escalate to Sayed):** add a **third option** to the
closed-loop decision alongside "AlpaGym on the A40" and "buy a graphics host": **a neural-reconstruction
closed loop built on our own PhysicalAI val episodes.** We already have 40 cached episodes, camera
intrinsics and extrinsics (`physicalai.py:153-154`), and per-clip rig data. This is a research-grade
build, not a weekend — but it is the only one of the three routes that no external gate can block.

### 5.2 H7 must be rescoped this week — it is no longer "does IDM work"
**Our position, MEASURED:** H7 is `Partially`, DoA 35 %, evidence *directional*: pseudo-labels ≈96 % of
real-label value (8 seeds) and an **80-clip** YouTube pilot at ≈92 % of ceiling, flagged
"⚠️ 80 clips / 3 seeds / **unknown intrinsics**, yaw ≈ 0"; **the data-efficiency slope itself is
untested.**
**Wayve's position, FACT:** the same mechanism (IDM → latent actions → downstream geometry) at
**10.2 M clips**, with a head that outputs **field-of-view and metric scale** — *the exact two gaps our
pilot recorded* — beating feed-forward SOTA by >10 % and generalizing zero-shot to an unseen dataset.

**Three consequences, in order:**
1. **The premise of H7 is now externally validated.** "Latent actions from unlabelled driving video
   organize into meaningful structure without pose labels" is no longer our conjecture. That *helps* us —
   cite it.
2. **H7 is dead as a differentiator stated as "IDM + focal canonicalization gives 1000× data."** They
   published it first, at five orders of magnitude more data. **Restate H7 as the *slope*** — the
   data-efficiency curve at matched parameters — **which Wayve has NOT published.** Their numbers are
   absolute-scale ("10.2 M clips"), not efficiency curves. The slope is the claim we can still own, and
   it is already what our ledger says is untested.
3. **Take the engineering.** LA-Pose is on arXiv and directly solves our pilot's intrinsics/metric-scale
   problem, and it benchmarks on **PandaSet — for which we already have an (unverdicted) loader intake
   package**. → **Data Engineering + Architecture: read 2604.27448 properly; the 50-d bottleneck, the
   FoV+scale head, and the reverse-motion failure mode are all directly transferable.**

### 5.3 H1 positioning — the squeeze is now three-sided
Run #5 already found **Orbis 2** (hierarchy on driving) and **HWM** (planning-time hierarchy off
driving). Wayve adds the third side: **a flat policy with an enormous offline world model**, i.e. the
"you don't need hierarchy, you need scale + a good evaluator" position, backed by $8.6 B.
**→ The H1 claim must be stated as all four qualifiers at once — planning-time hierarchy × in-loop
imagination × self-monitoring with a threshold × a published compute-normalized number, on driving.**
Drop any one and a well-funded published competitor answers it.

### 5.4 H11 is our best-defended ground — and we are not yet standing on it
Wayve's safety framework has **no OOD methodology, no runtime monitor, no thresholds** (FACT, their
page). LINGO introspection is **narrative, not numeric**. Nobody in the profile set publishes a
self-monitoring guarantee. **This is the widest open gap in the competitive field.**
**Honesty check (P8):** our **D8 AUROC bar (>0.85) is unmet** (preview p≈0.047, σ dissipating to chance
by k=4), and SC-06 is *blocked* on that same detector. **So H11 is an opportunity, not an advantage, and
it should be stated that way in any deck.** It is, however, the highest-value thing we could convert
from opportunity to advantage, because no competitor is contesting it.

### 5.5 H6 deserves an explicit re-examination, not a reflex defence
Wayve bets **against** scenario enumeration; we bet on it; **three of our four scenario packages are
unintegrated** (run #5 §5.1, 41/41 tests green). That combination should prompt a real question rather
than a defence: *is the scenario database a capability instrument or a narrative instrument?*
**My position (INFER):** it is a capability instrument **only if** each entry ends in a measured
model-side number. SC-13 is the first that got there and it returned a negative. That is the process
working. **But an enumeration strategy with a stalled integration path is strictly worse than Wayve's
bet**, because it has the cost of enumeration and none of the measurement. **→ The intake unblock (run
#5 §5.1) is the load-bearing action for H6, and this note raises its priority.**

### 5.6 Two direct engineering pickups
- **Rig3R answers a problem we have already measured.** Our PhysicalAI corpus has **two camera rigs**
  (cy≈543 / cy≈755) and a geometric-centre crop is **~215 px wrong for rig B** — a known cross-rig
  inconsistency in our training frames. Rig3R's answer is to **condition on rig metadata (camera ID,
  timestamps, raymap calibration)** and it reports its **largest gains on unseen rig configurations**.
  → **Data Engineering: rig-conditioning is a cheaper fix than filtering one rig out**, and there is now
  a NeurIPS-Spotlight design to copy.
- **GAIA-3's "World-on-Rails perturbation"** — perturb the ego trajectory, hold the rest of the scene
  fixed — **is the counterfactual our vision-effect ablation approximates by mean-replacing the scene.**
  Theirs is the better instrument. Worth reading before Benchmarks & Eval extends the ablation panel.

### 5.7 What to stop saying
- ❌ "Wayve's world model is compute-hungry **and they haven't figured out how to use it in the loop**."
  → They have *decided* not to, and said so. Say **"their world model validates; ours drives"** — it is
  both accurate and stronger.
- ❌ "GAIA is just a data factory." → GAIA-3 is positioned as an **evaluator that ranks policies**. If
  that claim holds it is a serious capability. Attack it on **reviewability** (no paper) and on
  **surrogate-vs-reality risk**, not on relevance.
- ❌ Quoting the **$1.8 B** funding headline (§2), or the **$500 M NVIDIA LOI** as current (Sept 2025,
  superseded).

---

## 6. What I could not determine (open, and honestly so)

1. **On-car parameter count / FLOPs / latency.** Never disclosed; four probes. This is the single number
   that would let us compute a real CNCE contrast against Wayve. **A 75 W figure exists as CLAIM only.**
2. **Whether GAIA-3's policy-ranking correlation holds.** No paper, no published correlation
   coefficients, no benchmark protocol.
3. **Whether Wayve runs any runtime uncertainty/OOD mechanism that they simply do not publish.** §4.6 is
   INFER from absence at one class of source; a second probe (patents, job ads, ISO 26262/SOTIF
   submissions) would be needed before this becomes load-bearing.
4. **Fleet size and total real-world miles.** Not found; consistent with W-11.
5. **How LINGO-2 and the current production driver relate.** LINGO-2 is Apr 2024; whether the shipped
   Gen-3 driver *is* a LINGO-descendant or a separate network is not stated.

---

## 7. Recommended actions (routed, priority order)

| # | Action | Owner | Why now |
|---|---|---|---|
| 1 | **Read LA-Pose 2604.27448 properly**; extract the 50-d bottleneck result, the FoV + metric-scale head, and the reverse-motion failure mode | Architecture & Inference + Data Eng | Directly closes our IDM pilot's two recorded gaps; rescopes H7 (§5.2) |
| 2 | **Add "neural-reconstruction closed loop" as a third option** in the closed-loop decision, beside AlpaGym-on-A40 and a graphics host | Tools & DevEnv → Sayed | The only route no external gate can block (§5.1) |
| 3 | **Restate H1 with all four qualifiers**; restate H7 as the **slope**, not the mechanism | Orchestrator / paper | Three published competitors now answer the short forms (§5.3, §5.2) |
| 4 | **Rig-metadata conditioning** for the two-rig PhysicalAI problem, following Rig3R | Data Engineering | Cheaper than dropping a rig; design published (§5.6) |
| 5 | **Unblock the scenario intake queue** | Orchestrator | H6's bet is strictly worse than Wayve's while integration is stalled (§5.5) |
| 6 | Add Wayve's **500 h / country** and **100 h / platform** figures to the CNCE-and-data-efficiency comparison table | Benchmarks & Eval | The number our H7 story must beat or reframe (§3.2) |
| 7 | State **H11 as an opportunity, not an advantage**, until D8 clears | Orchestrator / deck | P8 honesty; the gap is real but so is our unmet bar (§5.4) |

---

## 8. Catalog updates made alongside this note

- `OPPONENT_PROFILES.md` → Wayve section rewritten (architecture split, the offline factory table, the
  deployment reality check, the corrected "what would beat them").
- `WEAKNESS_CATALOG.md` → **W-05** gains Wayve's explicit not-for-in-vehicle statement; **W-11** gains
  Wayve as its most technical exemplar.
- `KNOWLEDGE_BASE.md` → LA-Pose, GAIA-3 detail, Rig3R, Ghost Gym/PRISM-1, the multi-country numbers.
