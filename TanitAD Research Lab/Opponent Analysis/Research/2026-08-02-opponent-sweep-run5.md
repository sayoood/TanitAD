# Opponent Analyzer — run #5 sweep (2026-08-02)

> **Dating change, read once:** this discipline had been dating its notes on a **narrative clock** that
> ran ~2.5 weeks ahead of wall-clock (a loop artefact). **That convention is retired here.** This note
> carries the **real** date. Consequence to watch for: run #4's note is named `2026-08-07` and is
> **older** than this one. **Order runs by run number, not by filename.**
>
> | run | note filename date | real wall-clock |
> |---|---|---|
> | #2 | 2026-07-24 | 2026-07-09 |
> | #3 | 2026-07-31 | 2026-07-17 |
> | #4 | 2026-08-07 | 2026-07-20 |
> | **#5 (this)** | **2026-08-02** | **2026-08-02** |
>
> Evidence labels per G-O1: **FACT** (verified at a primary or named source) · **CLAIM** (reported,
> not independently verified) · **INFER** (my own reasoning) · **MEASURED** (ours, with artifact path).

**Window covered:** 2026-07-20 → 2026-08-02 (13 real days — the first genuine news window since run #3;
run #4's was 3 days). Searches used: 15 of 25. Iterations: 1.

---

## 0. What this run changes, in five lines

1. **SC-13 is settled, and the answer is split.** The anticipation signal is **real and reproducible**
   (survived 2× the anchors; run #4's numbers re-derived to three decimals) — but it is **~64 %
   scene-independent and ~95 % motion-independent**, and the pre-registered survival condition is **NOT
   met** on the decision-grade interval. **The open-loop SC-13 probe is retired: it cannot demonstrate
   imagination, and two runs with better controls each time have now shown that** (§1).
2. **The regulator stopped being our story.** Zoox got a 2,500-vehicle commercial exemption on the
   **same day** the emergency-scene deadline lapsed unfixed. The pressure is real but it is **not** a
   barrier to scale (§2.1).
3. **"Hierarchical world model for driving" is now a published paper title** — Orbis 2, Freiburg,
   2026-07-17 (§3.1). Our differentiator must be restated, today.
4. **One of my own inferences is retracted**: the "EU blocks Chinese key-tech" wedge is dead — Momenta
   holds a Germany-wide L4 permit (§2.2).
5. **Our scenario pipeline is stalled at intake, and I have proof**: 3 of 4 packages (41 passing tests)
   never reached `stack/` (§5.1). This is the third run I have written it in a doc. It is now escalated
   as a blocker, not a note.

---

## 1. MEASURED — SC-13 resolved: the signal is real, and it is *not* the world model reading the scene

**Verdict in one line: the pre-registered SURVIVAL condition is NOT met.** The anticipation effect
survived doubling the anchors (falsifier F-A did not fire), but on the decision-grade interval it is
**not established as vision-driven** — and a control run #4 did not have shows why.

### 1.1 Setup

`sc13_probe_v5.py` on the **eval-pod A40**, **flagship v1** (`/workspace/v1_modelonly.pt`, step 30000),
canonical **40-episode** held-out PhysicalAI val, window 8, **stride 1** → **6,444 anchors**, wall-clock
**1,097 s**. Signal is unchanged: `D = CV_forward(2 s) − pred_forward(2 s)` (positive ⇒ the imagined
future is shorter than constant-velocity). Label `BRAKE_FAR` = a ≥1.5 m/s speed drop in the **2–3 s**
window with **<0.75 m/s** movement inside 0–2 s — braking that has **not started** and lies **outside**
the 2 s rollout. **n = 44 events across 15 episodes** vs 2,550 cruise anchors (run #4 had n=23).
All AUROCs below are **speed-matched** (per-event ±1 m/s), because events sit at median **7.33 m/s** and
cruise at **17.34 m/s**.

**Two things run #4 did not have.** (a) Two extra vision controls with *correct* input statistics —
`shuffled` (a real 8-frame window from a **different episode**) and `frozen` (this anchor's **own last
real frame repeated 8×**, real scene, no motion). Run #4's only control was a constant mean frame, which
is far off-manifold and may break the encoder rather than remove the hazard. (b) **Episode-cluster
bootstrap** intervals — resample the 40 episodes, not the anchors. Anchors 0.1 s apart are near
duplicates, so anchor-level resampling treats correlated copies as independent facts.

### 1.2 The replication check passed exactly

Because the probe records each anchor's start index, run #4's stride-2 anchor set is recoverable as a
**subset** of this run. Re-deriving it (not recalling it) reproduces run #4 **to three decimals**:

| matched AUROC, BRAKE_FAR | run #4 (reported) | run #5, stride-2 subset |
|---|---|---|
| held | 0.723 | **0.723** |
| blind | 0.654 | **0.653** |
| informed (leaks) | 0.680 | **0.680** |
| gt_oracle | 0.633 | **0.633** |
| reactive | 0.434 | **0.434** |
| n events / n cruise / n anchors | 23 / 1,283 / 3,241 | 23 / 1,283 / 3,241 |

This simultaneously confirms that `/workspace/v1_modelonly.pt` on the **re-provisioned** pod is the same
checkpoint run #4 measured (the registry path `/root/models/flagship-30k/ckpt.pt` no longer exists), and
that run #4's numbers were not an artifact.

### 1.3 The result

**Speed-matched AUROC on BRAKE_FAR, stride 1, n=44 events over 15 episodes:**

| arm | what it sees | matched | stratified | episode-cluster CI95 |
|---|---|---:|---:|---|
| `informed` ⚠ leaks | true future actions | 0.696 | 0.688 | [0.535, 0.875] |
| **`held`** | **real scene + motion** | **0.736** | 0.758 | [0.546, 0.900] |
| `frozen` | real scene, **no motion** | 0.723 | 0.741 | [0.546, 0.877] |
| `blind` | mean frame (off-manifold) | 0.672 | 0.719 | [0.492, 0.851] |
| `shuffled` | **a different episode's real scene** | 0.634 | 0.700 | [0.466, 0.789] |
| `gt_oracle` | the true 2 s trajectory | 0.620 | 0.607 | [0.388, 0.878] |
| `reactive` | −Δv/0.5 s, no model | 0.455 | 0.435 | [0.244, 0.725] |

**Paired differences (same resample; episode-cluster is the decision-grade one):**

| difference | point | **episode-cluster CI95** | anchor-level CI95 |
|---|---:|---|---|
| held − reactive | **+0.281** | **[+0.009, +0.562]** ✅ excludes 0 | [+0.156, +0.419] |
| held − shuffled | +0.102 | **[−0.011, +0.245]** ❌ includes 0 | [+0.012, +0.194] |
| held − blind | +0.064 | **[−0.019, +0.162]** ❌ includes 0 | [+0.017, +0.109] |
| held − frozen | +0.013 | **[−0.024, +0.050]** ❌ includes 0 | [−0.007, +0.034] |
| shuffled − reactive | +0.179 | [−0.104, +0.485] | [+0.026, +0.341] |

### 1.4 Verdict against the pre-registration

The falsifiers and the survival condition were committed in the script docstring **before** the run:

- **F-A (volume) — DID NOT FIRE.** margin `held − reactive` = **+0.281** (threshold ≤ +0.10), essentially
  unchanged from stride 2 (+0.289) with **twice** the events. **Run #4's in-domain positive was not
  small-n noise.** That question is closed.
- **F-B (vision) — the point margin cleared (+0.064 > +0.02) but the DECISION-GRADE INTERVAL DID NOT:**
  `held − blind` CI **[−0.019, +0.162]** and `held − shuffled` CI **[−0.011, +0.245]** both include 0.
- **SURVIVAL CONDITION: NOT MET** (it required both margins **and** both episode-cluster CIs to exclude
  zero). **Therefore the vision-driven-anticipation claim is NOT established.** Pre-registration means
  taking this reading, not the one the point estimates flatter.

⚠️ **Note precisely where the estimator changes the answer.** On the **anchor-level** bootstrap — run
#4's estimator — `held − shuffled` [+0.012, +0.194] and `held − blind` [+0.017, +0.109] **both exclude
zero**, and this section would have read "vision-driven anticipation confirmed." The **episode-cluster**
interval says otherwise. n=44 events live in only **15 episodes**; they are not 44 independent facts.
This is the same class of instrument error as the `overlapping_holdout_se` retraction in `CLAUDE.md`,
caught **before** publication this time rather than after.

### 1.5 What the new controls actually revealed (the useful part)

The controls decompose the +0.281 gap over the reactive floor:

| step | arm | matched | gained |
|---|---|---:|---:|
| kinematic floor | reactive | 0.455 | — |
| **+ model & ego channels, scene DESTROYED** | shuffled | 0.634 | **+0.179 (≈64 %)** |
| + the correct static scene | frozen | 0.723 | +0.089 (≈32 %) |
| + scene motion | held | 0.736 | +0.013 (≈5 %) |

Three readings, all new this run:

1. **Roughly two-thirds of the "anticipation" needs no correct scene at all.** `shuffled` is fed a real
   window **from a different episode** — zero information about this anchor's hazard — and still beats
   the reactive floor by +0.179. The `D = CV_fwd − pred_fwd` construction plus the ego-state channels
   carries most of the discrimination on its own.
2. **The motion in the observation is worth ~nothing here** (`held − frozen` = **+0.013**, CI
   [−0.024, +0.050]). A single frame repeated eight times scores 0.723 against held's 0.736. Whatever
   the model uses is a **static-frame property**, not a rolled-forward dynamic — which is precisely the
   opposite of the "consequence forward model" story SC-13 was built to argue.
3. **Vision does matter for trajectory accuracy, just not for this signal.** On the same anchors, 2 s
   ADE: held **1.186 m**, frozen 1.255, blind 1.250, **shuffled 1.321**, CV 1.743. Destroying the scene
   costs **+0.135 m** of ADE while costing only +0.102 AUROC (CI-inclusive of 0) on BRAKE_FAR. The model
   reads the scene; the scene is not what makes this particular detector work.

Also reproduced from run #4 and still odd: `held` (0.736) **exceeds `gt_oracle` (0.620)** — the score
computed from the *true* future trajectory. A signal beating the ground truth it is supposed to
anticipate is a property of the CV-deficit construction, not of foresight, and is further reason not to
read `held` as anticipation.

### 1.6 Consequences

- **SC-13 status → `live-measured — anticipation signal CONFIRMED over the reactive floor; VISION
  ATTRIBUTION NOT ESTABLISHED`.** The collision-rate/lead-time **design-oracle** contrast remains
  **unsupported** and stays out of every external narrative (unchanged from run #4).
- **H15 (imagination):** no status change, and the evidence moves **against** the open-loop form of the
  claim more sharply than run #4 put it — not because the effect vanished, but because the effect
  turns out to be **~64 % scene-independent and ~95 % motion-independent**. The open-loop probe has now
  been run twice with better controls each time; **it is answered, and the answer is that this
  instrument cannot demonstrate imagination.** Further open-loop probing of SC-13 is retired.
- **The `D = CV_fwd − pred_fwd` monitor feature (run #4's conditional recommendation to Benchmarks &
  Eval) — the recommendation SURVIVES, with its rationale rewritten.** It is a real, CI-separated
  improvement over a naive deceleration floor in-domain (+0.281, ep-cluster CI excludes 0), so it has
  value as a cheap label-free monitor. But it must **not** be described as vision- or
  imagination-driven, and its advantage over a **simple ego-kinematic feature is unproven** — the
  `shuffled` arm shows most of it survives with the scene destroyed. Keep run #4's competence guard
  (it was unreliable on comma2k19, where the model loses to CV).
- **What would actually settle the H15 claim:** not more open-loop anchors. The closed loop, where an
  imagined consequence changes an action and the action changes the outcome. That is now the *only*
  remaining test, which raises the priority of the closed-loop harness question in §2.3.

**Artifacts (banked in-repo, not left on the pod):**
`Implementation/sc13-real-probe/sc13_probe_v5.py`, `sc13_analyse_v5.py`,
`results/sc13_v1_stride1_analysis_{all,stride2}.json`, `results/sc13_v1_stride1.json`, and — the lesson
of run #4 — **the raw substrate `results/sc13_v1_stride1_windows.pt` (1.3 MB)**, so every future
re-analysis is free and survives the next pod re-provision.

---

## 2. Opponent deltas (2026-07-20 → 2026-08-02)

### 2.1 The strategic fact of this window: authorization decoupled from capability

**FACT.** On **2026-07-30** NHTSA **granted Zoox a commercial exemption** — the **first US authorization
for a purpose-built AV with no steering wheel, no pedals and no driver's seat** — permitting **paid
rides** in **up to 2,500 vehicles over two years** (Federal Register **2026-07-31**).
On **that same day**, NHTSA's end-of-July deadline for AV developers to present emergency-scene fixes
**expired with no public resolution**. Zoox's own smoke recall (105 vehicles) was **six weeks** earlier.
— <https://fortune.com/2026/07/31/zoox-robotaxi-steering-wheel-safety-data-gap/>

**FACT.** Meanwhile the pressure escalated in the *other* branch of government: Rep. **Kevin Mullin
(D-Calif.)** introduced the **"AV Emergency Response Coordination Act"** (week of **2026-07-28**) —
first-responder protocols, a **24 h hotline** for public officials, **NHTSA minimum standards**, and
**authority for cities to geofence AVs during emergencies**. SF Fire Chief **Dean Crispen** cites
robotaxis blocking fire stations and ambulance facilities.
— <https://techcrunch.com/2026/07/28/waymo-robotaxi-operators-face-fresh-scrutiny-over-emergency-response-failures/>

**INFER — and this is a correction to how I have been framing W-09 for two runs.** I have been
narrating the NHTSA directive as closing pressure on the incumbents. The 07-30 pairing says otherwise:
**the capability finding and the market authorization run on independent tracks.** Worse for the
"regulator will force it" reading, the legislative remedy on offer is **geofencing** — i.e. *route the
AV around emergency scenes*, which is an admission that nobody expects the in-vehicle capability soon.

**What this does to our story.** It removes a crutch and sharpens the actual claim:

- ❌ Not: *"their emergency-scene failures will be regulated away and we will be compliant."*
- ✅ But: *"the failure is documented, unfixed, not a barrier to their scale — and it is a capability
  we can demonstrate."* A vehicle that detects and clears an emergency corridor is worth more than an
  exemption, because the exemption is evidently obtainable without it.

That is exactly what **SC-06** measures, and it raises SC-06's value. It also raises the cost of SC-06
being **blocked** on an OOD detector we have not yet made work (run #4's finding, still true).

### 2.2 ⛔ RETRACTION — Momenta and the "EU market-access weakness"

**FACT.** On **2026-07-29** Momenta received a **Germany-wide Level-4 testing approval from the KBA**,
cleared for urban autonomous operation **nationwide**, and states it is the **first Chinese firm** to
hold one. It underpins the **Munich** robotaxi plan. **Uber increased its stake** in Momenta the same
week. Momenta separately confirmed **robovans in Suzhou** (07-27), extending its platform to
robotaxi + robovan + robotruck.
— <https://cnevpost.com/2026/07/29/momenta-cleared-test-robotaxis-across-germany/> ,
<https://cleantechnica.com/2026/07/29/momenta-to-test-robotaxis-across-germany-uber-invests-more/> ,
<https://cnevpost.com/2026/07/27/momenta-confirms-robovan-entry/>

**This falsifies run #3's INFER** that Uber's Munich switch from Momenta to Autobrains was caused by
"EU political resistance to sensitive Chinese key-tech", and that this constituted an EU-market-access
weakness for Momenta and Pony which our Western/EU-clean posture could turn into a wedge. **Both legs
fail:** the EU did not lock Momenta out (it granted a national permit), and Uber did not walk away (it
invested more). The framing is deleted from `OPPONENT_PROFILES.md` and must not appear in any deck.

**Root-cause class (for `RETRACTION_LOG.md`): a single-source geopolitical INFER promoted to a
market-structure conclusion.** One vendor-switch datum was read as a policy regime; the switch was real,
the *reason* was never sourced to a regulator or to either company, and it was then reused as a standing
strategic asset. **Rule (Operating Standard #2, absence/inference needs a second probe): an inference
about *why* a competitor lost a deal is not admissible as strategy until it is confirmed at a second,
independent source — and a competitor's regulatory posture is checked at the regulator, not the press.**

### 2.3 NVIDIA — W-05 wedge re-verified open, at the primary source

**FACT**, read from NVIDIA's own technical post rather than press coverage: **Alpamayo 2 Super** =
a **32 B VLM backbone**, *"3× the number of parameters as prior Alpamayo models"*, adding **360°
surround** perception and **Meta-Action** outputs, claiming *"state-of-the-art performance in multiple
aspects including reasoning quality, trajectory accuracy, alignment"* — **with no benchmark table, no
latency figure, no compute number, and no Nano tier reported at all.** Launched **GTC Taipei
2026-06-01**; weights and inference code **"coming summer 2026"**.
Two new companion assets: **AlpaGym**, an open-source, high-throughput **closed-loop RL** framework
(GRPO, default reward functions, single-GPU → multi-node), and **quantization scripts "coming soon."**
— <https://huggingface.co/blog/nvidia/nvidia-alpamayo-2>

- **Third consecutive run** the Nano-tier compute-normalized number has come back absent — and this
  time checked at the source, so it is no longer an INHERITED reading.
- **INFER:** shipping *quantization tooling* alongside a 3× parameter jump concedes the deployment-cost
  problem **in engineering** while declining to concede it **in the benchmark table**. That asymmetry is
  the CNCE wedge stated in NVIDIA's own release plan.
- **Actionable — and a correction to my own first draft of this section.** I nearly filed AlpaGym as a
  *new* asset. It is not: **Tools & DevEnv already logged "AlpaSim/AlpaGym = Phase-1 cloud (40–60 GB
  VRAM)" on 2026-07-06** (`PROJECT_STATE.md` §5). Checking the program's own record before asserting
  novelty is what makes the finding useful rather than noise. **What actually changed:** it is now
  released open-source, GRPO-based, with NVIDIA stating it scales **from a single GPU** to multi-node —
  which **contradicts the 40–60 GB Phase-1-cloud read on record** and makes it re-testable on the
  **A40 48 GB we already have**. That matters because AlpaSim was a NO-GO on the eval pod (unprivileged
  container, image-only NuRec) and CARLA pixels are host-blocked, so the closed loop is still walled.
  → **Tools & DevEnv: re-check the 40–60 GB figure against the released AlpaGym before we spend again
  on a graphics-capable host.**

### 2.4 Shorter deltas

| Opponent | Δ | Label |
|---|---|---|
| **Waymo** | Driverless rides opening in **San Diego, Las Vegas, Tampa, Denver** (2026-07-08), employees first — expansion continues *despite* the freeway-work-zone freeze | FACT |
| **Waymo** | Economics for the W-06 contrast: ~**3,700** I-PACE (Feb'26), ~**500 k** paid rides/week (May'26, 2× Apr'25), ~**$355 M** annualized revenue (Feb'26), ~**$15–17**/ride, ~**20** trips/vehicle/day → ~**$96 k**/vehicle/yr gross | FACT / **CLAIM** (revenue = Sacra third-party estimate, not a filing) |
| **Waymo** | A **second** fleet-scale stall surfaces: a **December power outage stranded dozens of vehicles** — a different trigger (infrastructure) from the July 4 event (crowd egress) → W-10 is a class, not a night | FACT |
| **Pony.ai** | **No Q2 delta exists to quote.** Q2/interim results report **2026-08-18**. W-06 stands on Q1. | FACT |
| **Wayve** | No material in-window delta. (Checked: the "NVIDIA $500 M" item that surfaces in search is **2025-09-18**, superseded by the Feb'26 Series D — *not* new.) | FACT |
| **Autobrains** | No in-window delta; prior **VinFast L4 for SE Asia on NVIDIA DRIVE Hyperion** (COMPUTEX 2026) remains the escalation to watch | FACT |
| **Zoox** | See §2.1 — exemption granted; the smoke recall (W-04/W-09) is unremedied in public | FACT |
| **Waabi** | **Not currently profiled.** Raised **~$1 B** (incl. a $750 M Series C, Khosla/G2) for self-driving trucks + robotaxis. Simulation-first, "Waabi World" generative sim — architecturally adjacent to us. **Add a profile next run.** | FACT |

---

## 3. Field scan (D-028 recency-first listing scan + citation-graph walk)

### 3.1 ★ Orbis 2 — "A Hierarchical World Model for Driving" (2607.15898, 2026-07-17)

**FACT.** Mittal, Mousakhan, Galesso, Farid, Dienert, Sahay, **Brox** (LMB Freiburg). Two-level
architecture: a **high-level predictor forecasting coarse scene structure over extended temporal
horizons**, and a **low-level generator producing detailed predictions conditioned on the high-level
output**. Evaluated on a standard driving-world-model suite, including **steering responsiveness on
counterfactual scenarios**. — <https://arxiv.org/abs/2607.15898>

**This displaces HWM as our sharpest differentiation risk.** Run #4 found HWM (2604.03208) doing
planning-time hierarchy on **manipulation and mazes**. Orbis 2 does hierarchy on **driving**, from a
top-tier vision group. The phrase *"a hierarchical world model for driving"* is, verbatim, how our H1
story opens.

**What is still ours (INFER, abstract-level — the deep-read must confirm):**

| pillar | HWM (2604.03208) | Orbis 2 (2607.15898) | TanitAD H1 |
|---|---|---|---|
| hierarchy | **planning-time** | representation / temporal | planning-time |
| domain | manipulation, maze | **driving** | driving |
| planner selects over imagined futures | partly (subgoal matching) | **not claimed** | yes (in-loop) |
| self-monitoring / OOD guarantee | no | no | H11 (claimed, **not yet met** — D8 AUROC unmet) |
| parameter count published | **no** | **no** | 286.34 M, published |
| compute-normalized number | no | no | CNCE 210,551 median |

⇒ **The differentiator is no longer "hierarchy". It is "hierarchy that a planner USES, with a number
attached."** Any deck line reading "hierarchical world model" without those qualifiers is now answerable
by two published papers. **Architecture & Inference: deep-read Orbis 2 first, ahead of SGDrive** —
resolve planning-time vs representation-only, and hunt for a parameter count.

### 3.2 The second pillar is moving too — self-monitoring

**FACT/INFER.** **CheckVLA** (2607.26789, 2026-07-29) uses an **action-conditioned world model to verify
policy execution at run time and replan when deviation exceeds a threshold** — the mechanism of **H11
self-monitoring + A9 fallback**, published. It is on VLA/robotics, not driving, and claims no
*guarantee*. So the "with guarantees, on driving" half of H11 is still open ground — but two of our four
moat pillars acquired published relatives inside one fortnight.

### 3.3 The literature is now working on our own instruments

All late-July 2026, from the raw listing scan (these were **not** reachable by our fixed query set —
which is the point of D-028):

- **Temporally Centered SIGReg** (2607.26924) — SIGReg on temporally-centred residuals to stop
  representation aliasing across tasks. **SigReg is our anti-collapse method**; this is a direct read.
- **What Can Latent World Models Know? Physical Parameter Identifiability** (2607.27017) — a
  certificate-gated protocol for *which physical quantities enter the latent*. A principled version of
  our speed-decodability / curvature probes.
- **ODEWorld** (2607.27924) — continuous latent velocity fields, explicitly targeting *"representation
  collapse in latent world models."*
- **Temporal-Distance JEPA** (2607.25337) — directed temporal cost for plan-aware JEPA representations
  without expensive test-time search.
- Efficiency axis: **DriftWorld** (2607.15065) — single-forward-pass rollouts at **30+ fps, 17× faster
  than diffusion**; **GigaWorld-Policy-0.5** (2607.13960) — action-only inference at **85 ms**.
- Driving-specific: **GeoWorldAD** (2607.17521), **M⁴World** (2607.14005, multi-view + LiDAR,
  minute-long streaming).

**Seam routing (D-028):** SIGReg / identifiability / ODEWorld → **Architecture & Inference**;
DriftWorld + GigaWorld latency → **Production & Optimization**; **no benchmark or dataset release in
this batch**, so Benchmarks & Eval takes none of it.

---

## 4. Catalog and database changes

- **W-11 (new)** — *No exposure denominator: safety claims that cannot be falsified.* IIHS
  (2026-07-31): most AV operators **don't report miles driven, so no crash rate is computable**, and
  there is **no standard for which incidents must be reported**. Pairs with W-07 (CA DMV retiring
  disengagements). **This is the weakness where we are already strongest** — published denominators,
  pre-registered gates with both outcomes committed, the episode-cluster bootstrap as the decision-grade
  interval, an honest kinematic floor, and an append-only retraction log. The counter is the operating
  standard itself, which also makes it the easiest to lose by one loose sentence.
- **W-09** — two deltas: the deadline **lapsed unfixed while the field scaled** (§2.1), and pressure
  **escalated to a draft statute** with a geofencing remedy.
- **W-10** — second instance, different trigger (December power outage).
- **W-05** — wedge re-verified **open** at NVIDIA's own text; AlpaGym + quantization tooling logged.
- **W-08 / SC-13** — see §1.
- **Watch-list** — Orbis 2 added at the top (ahead of HWM); CheckVLA added; the instrument-adjacent
  arXiv batch recorded with seam routing.

---

## 5. Escalations and handoffs

### 5.1 ⛔ ESCALATION — the scenario pipeline is stalled at intake (evidence, not a complaint)

Three of my four intake packages have **never been integrated**, and the `ORCHESTRATOR VERDICT` blocks
are still the unfilled template text:

| package | run | tests | verdict block | in `stack/tanitad/eval/scenarios/`? |
|---|---|---|---|---|
| `2026-07-17-work-zone-phantom-scenario` | #1 | 9 | **integrate** (2026-07-08) | ✅ `work_zone_phantom.py` |
| `2026-07-24-stop-arm-gate-scenario` (SC-04) | #2 | 11 | *unfilled template* | ❌ |
| `2026-07-31-stationary-lead-scenario` (SC-13) | #3 | 14 | *unfilled template* | ❌ |
| `2026-08-07-emergency-scene-scenario` (SC-06) | #4 | 16 | *unfilled template* | ❌ |

Verified directly: `ls stack/tanitad/eval/scenarios/` returns only `__init__.py`, `registry.py`,
`traffic_light.py`, `work_zone_phantom.py`. **41 passing tests across three packages are sitting in
`incoming/`, the oldest for three runs.**

**MEASURED 2026-08-02 — all three still pass, today, at this HEAD** (`venvs/tanitad`, Python 3.13.5,
numpy 2.5.1, offline/CPU): `stop-arm-gate` **11 passed in 0.13 s**, `stationary-lead` **14 passed in
0.10 s**, `emergency-scene` **16 passed in 0.10 s** = **41/41**. I re-ran them rather than quote the
authoring runs, so the ask carries no inherited numbers: **these are green now, not green once.**
Integration risk is one new self-contained file per package under `stack/tanitad/eval/scenarios/`,
importing nothing from the suite.

This is the failure the operating standard names: *an artifact on one disk is not done*, and *escalate
integration, don't write "please merge" into a doc.* I have now written it into a doc three times, which
is the wrong instrument. **Request to the Orchestrator: a single triage pass over all three, or an
explicit `defer`/`reject` with a reason.** A `defer` is a fine answer; **silence is the one answer that
costs us the H6 row** — the ledger currently reads *"4 scenarios shipped"*, which is true of authorship
and false of the stack.

**And it changes what I should build next.** Authoring a fifth scenario package into the same queue
would be adding to a stalled buffer. Run #6 priorities are therefore weighted toward **measurement on
our own checkpoints** (which needs no intake) over new scenario authoring — see `BACKLOG.md`.

### 5.2 Handoffs to other disciplines (no cross-boundary writes)

- **Architecture & Inference (Wed) — top priority:** **deep-read Orbis 2 (2607.15898)** ahead of SGDrive
  and HWM. The single question that matters: **is the hierarchy used by a planner at decision time, or
  is it a two-scale predictor?** Our whole H1 positioning depends on the answer. Second: **Temporally
  Centered SIGReg (2607.26924)** — it is our own anti-collapse method, and our SIGReg readout is
  currently `NOT-YET-ADMISSIBLE` (rms_offdiag 0.32 > 0.1), so this is a candidate fix, not just news.
  Third: latent-WM **identifiability** (2607.27017) as a principled frame for our decodability probes.
- **Benchmarks & Eval (Thu):** (a) **the `D = CV_fwd − pred_fwd` monitor feature — run #4's conditional
  recommendation STANDS, but its rationale is rewritten (§1.6).** Adopt it as a cheap label-free monitor
  that beats a naive deceleration floor in-domain (+0.281, episode-cluster CI excludes 0), **with**
  run #4's competence guard. **Do not label it vision- or imagination-driven**, and note that its
  advantage over a plain ego-kinematic feature is **unproven** (a window from a *different episode*
  retains ~64 % of the gap); (b) standing: blockage-duration + incursion-rate reducers over SC-06 `_extra`; **unify SC-06's
  `non_nominal_detected` with the SC-05 OOD head — one detector**; (c) the **SC-05 D8 bar remains
  GATING for SC-06 scoring**; (d) if you adopt anything from this run, adopt the **episode-cluster
  bootstrap over anchor-level bootstrap for any AUROC on windowed anchors** — §1 shows the two differ
  by enough to change a verdict.
- **Data Eng (Tue):** the stopped-lead ask is **withdrawn as a priority**. §1 shows the limit was never
  event count. If you have spare capacity, the higher-value item is **screening for smoke / flare /
  flashing-light events** (W-09/SC-06) — the Zoox recall makes smoke the highest-value visual cue in the
  corpus, and SC-06 is the scenario whose regulatory value went **up** this window.
- **Tools & DevEnv (Mon):** **evaluate AlpaGym** (NVIDIA's open closed-loop RL framework, §2.3) as a
  candidate for the closed-loop gap — it scales down to a single GPU, which AlpaSim did not. AlpaSim
  evaluation still open from run #2. CARLA emergency-vehicle / flare / cone assets + a smoke overlay for
  SC-06 remain queued behind that.
- **Production & Optimization (Sat):** **DriftWorld** (2607.15065, 30+ fps, 17× faster than diffusion)
  and **GigaWorld-Policy-0.5** (2607.13960, 85 ms) are the two efficiency-axis reads from this window.
- **Orchestrator:** (a) **§5.1 — triage the three stalled packages or say `defer`;** (b) log the
  **Momenta retraction** (§2.2) in `RETRACTION_LOG.md` under *single-source geopolitical INFER promoted
  to a market-structure conclusion*; (c) **narrative correction**: stop using "the regulator is closing
  in on them" — §2.1 falsifies it; the correct line is *the failure is unfixed and scaling anyway*;
  (d) **H1 positioning is now time-critical** (§3.1); (e) **W-10 is still `no-counter-yet` for us** —
  the run-#4 request for a Phase-0-scope-or-defer decision is unanswered.

---

## 6. Resource declaration (G-I)

| item | value |
|---|---|
| Resources | **Eval pod A40 48 GB** (`tanitad-eval`) — the SC-13 v5 probe (5 rollout arms × 6,436 anchors) + a 2,000-draw episode-cluster bootstrap analysis. Local dev box for authoring only. |
| Wall-clock | probe **~22 min** on the A40 (~33 s/episode × 40); analysis a few minutes; ~3 h including the sweep, authoring and the pod-reprovisioning detour. |
| Cost | **$0** (standing pod, no new spend) |
| Why not bigger | The eval pod **was** the resource. Nothing here needs training compute; the binding constraint is in-domain event count, not FLOPs — and §1 shows that even doubling anchors was not the limiting factor. |
| Coordination | The pod had been **re-provisioned** for the v1-vs-v2corpus work — `/root/models/` and the comma2k19 val are **gone**, and a **v1 checkpoint relay was in flight** when I arrived. I waited for the relay to finish (driver polls the transfer PID), verified the checkpoint loads, touched `results/LOCK.opponent-analyzer`, ran with `OMP_NUM_THREADS=6`, and released the lock. A concurrent CPU-only job (`run_ctrv_readjudication.py`) belonging to another workstream was left untouched. |

**Reproducibility note (and a near-miss).** Run #4's probe scripts and its `*_windows.pt` substrate were
on the *old* eval pod and are **gone**. The experiment was re-runnable only because the scripts had been
**banked into the repo** at run #4 (`Implementation/sc13-real-probe/`). That is the "finish before you
start" rule paying for itself; had run #4 left them on the pod, this run would have had nothing to
extend. The comma2k19 val cache was **not** banked and is genuinely lost from this pod — which is why §1
is in-domain only.
