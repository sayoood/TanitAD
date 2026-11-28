<title>RESULT — adjudicating Waymo's 10 lessons: two confirm our thesis, one is self-undercutting, and none is an experiment</title>

# RESULT — WAYMO'S "10 AI LESSONS" ADJUDICATED

`TanitAD Research Lab · Opponent Analysis · 2026-08-31`
`Target: Srikanth Thirumalai, "10 AI Lessons from Driving 200+ Million Fully Autonomous Miles", waymo.com/blog, 2026-08-26.`
`Method: per-claim adversarial adjudication — confirming evidence AND contradicting evidence sought separately for every load-bearing claim, then a verdict, then what it binds on US.`
`0 GPU · 0 spend · Thor untouched · no pod touched. Verbatim passages: raw/QUOTES.md.`

---

## 0. ⭐⭐⭐ THE ANSWER, IN THREE LINES

1. **Two of the ten lessons are strong independent confirmation of TanitAD's two most contested design bets** — closed-loop-over-open-loop (L5) and a fast/slow hierarchy (L7). Neither was written with us in mind, which is what makes them worth having.
2. **One lesson contains a concession that undercuts its own headline and supports our sub-300M thesis** (L3: they train large and **distil to a small onboard model**).
3. ⛔ **None of the ten is an experiment.** The article contains **zero controlled ablations**. Its evidence class is `PUBLISHED-BLOG` throughout — an *experience* claim over 200 M miles, published inside a coordinated competitive-comms push. That does not make it wrong; it makes it **inadmissible as a decision input on its own**, and the programme should treat every lesson as a hypothesis to be tested rather than a finding to be inherited.

---

## 1. ⚠️ FRAMING FIRST: what class of document is this?

| observation | evidence |
|---|---|
| Published **2026-08-26**, author Srikanth Thirumalai | the article itself |
| An **Axios exclusive the same day** — *"Waymo says there's no AI shortcut to self-driving"* | axios.com, 2026-08-26 |
| Trade press read it **the next day** as a competitive attack — *"Waymo takes a shot at Tesla's self-driving: it's a 'false summit'"* | electrek.co, 2026-08-27 |
| Waymo's co-CEO had made the camera-only argument **three weeks earlier** | electrek.co, 2026-08-04 |

⇒ **This is a positioning document released through a coordinated comms push, not a research report.**
Every one of the ten lessons happens to validate a Waymo architecture choice and, in four cases, to
criticise its principal competitor's. **Ten for ten is not what an honest audit of one's own
programme usually looks like** — our own retraction log is the counter-example.

⛔ **The methodological core of the critique:** *"200+ million fully autonomous miles"* is an
**exposure** number, not an **experiment**. Not one lesson is supported by an A/B, an ablation, a
matched-arm comparison, or a stated interval. There is no counterfactual Waymo — no camera-only
Waymo, no modular Waymo, no map-free Waymo — so the article cannot separate *"this is what we
built"* from *"this is what is necessary"*. **That distinction is the entire question**, and the
document does not address it anywhere.

⚠️ **Fairness, stated because it constrains our conclusions too:** exposure evidence is not
worthless. 200 M driverless miles is the largest such record in existence, and an operator's failure
catalogue contains real information no benchmark holds. The correct reading is **"strong prior,
untested attribution"** — not "marketing".

---

## 2. Per-lesson adjudication

**Verdict scale:** `CONFIRMS-US` · `SUPPORTED` (independent evidence agrees) · `CONTESTED` (credible
evidence both ways) · `UNSUPPORTED-AS-STATED` (claim outruns its evidence) · `BINDS-ON-US` /
`DOES-NOT-BIND` (does the lesson transfer given our different constraints?).

---

### L1 — *"Multimodal sensors are indispensable"* · ⚖️ **CONTESTED · DOES NOT BIND AS STATED**

**Their claim.** Camera-only is rejected after 200 M miles; lidar gives *"3D geometry with millimeter
precision"*, radar *"sees through"* weather. Their public illustration is a **Google I/O 2026 demo of
lidar detecting a pedestrian in a Phoenix dust storm invisible to camera**.

**Confirming evidence (independent).** Camera-only BEV reaches ~**85.1 % of lidar segmentation** and
~**92.1 % of lidar detection** performance *(⚠️ RELAYED — attributed to `2505.06113`, banked, **not yet
read**; the figures must be verified in the primary before reuse)*. The residual gap is
**specifically depth at range and adverse illumination/weather** — which is exactly where their
dust-storm anecdote sits.

**Contradicting evidence (independent).**
- `2507.17596` **PRIX** plans from **raw pixels** end-to-end and is reported to *"rival state-of-the-art multimodal systems"*.
- ⭐ The sharpest counter is a **convergence** observation, not a performance one: *"Waymo vehicles are controlled by a foundation model trained in an end-to-end fashion — just like Tesla and Wayve vehicles"* (understandingai.org). The public disagreement is about **sensors**, while the **architectures converged**.
- Tesla's counter-position is explicit — *"the self-driving problem is not a sensor problem, it's an AI problem"* (Elluswamy, ScaledML 2026-01-29) — and Wayve raised **$1.2 B** on a camera-first thesis with NVIDIA and Uber participating.

**Verdict.** ⚖️ **CONTESTED.** A single dust-storm demonstration is an **existence proof of a failure
mode**, not a distribution-level measurement, and no controlled camera-only-vs-multimodal ablation
is offered by either side. The honest state is: **lidar's advantage is real, localised to depth-at-
range and adverse conditions, and unquantified at the system level.**

⛔ **Why it does not bind on us — and this is the part that matters most.** **Our vision-only rule is
NOT a sensor-cost bet.** It is a **leak-avoidance** rule: *labels may use ego and privileged
channels; inference is vision-only*, because our situation labels are derived from ego dynamics and
an ego-fed classifier would be reading its own label's source. **Waymo's lesson argues about
capability; our rule is about admissibility.** They do not meet. ⇒ **L1 is not evidence against the
vision-only rule and must never be cited as such.**

⚠️ **But it does bind on one thing:** if we ever claim TanitAD is a *deployable* driving stack rather
than a research programme on a camera corpus, L1 becomes a live objection we have no answer to.
**Scope discipline: our claims are about a vision-only latent world model, not about L4 readiness.**

---

### L2 — *"HD maps are a powerful prior"* · ✅ **SUPPORTED · ⛔ BINDS ON US, AND NAMES A REAL WEAKNESS**

**Their claim.** Maps are *"another input—like sensors, but acting as a mental memory"*, valuable in
poor visibility, and they free real-time compute for dynamic elements.

**Contradicting evidence.** An active mapless/online-mapping literature exists (online lane-graph
extraction, CGNet, LGmap) whose entire premise is that **manual HD-map annotation does not scale** —
so "powerful" is not the same as "necessary", and the scaling objection is real.

**Verdict.** ✅ **SUPPORTED as stated** (a prior that helps), **not** supported as *necessary*.

⛔ **This is the lesson that identifies a genuine TanitAD structural weakness, and I am not going to
soften it.** Settled at five independent probes: PhysicalAI-AV contains **no map, no lane graph, no
junction annotation, no traffic-light feature and no route/goal signal** — the card says verbatim
*"we do not include open maps data"* — and `egomotion` carries **no lat/lon/GNSS**, so **OSM
map-matching on our traces is impossible**. Waymo's L2 says the prior we structurally lack is
load-bearing for exactly the strategic layer our thesis rests on.

⭐ **The tactical answer already exists in-programme and is under-exploited:** **NuRec scenes ship
`map.xodr`** (measured: `volume.nurec` is gzip+msgpack, not an opaque blob; gsplat runs at 492 FPS
on Thor). **That is our only route to a map prior, and L2 raises its priority.**

---

### L3 — *"Fewer, larger models are better"* · ⚠️ **UNSUPPORTED-AS-STATED · ⭐ SELF-UNDERCUTTING — IT SUPPORTS OUR THESIS**

**Their claim.** Specialised modules become *"unmaintainable at scale"*; consolidate into *"fewer,
high-capacity, specialised foundation models"*, riding *"scaling laws"* from LLMs, so *"data, rather
than brittle human priors, determine what is relevant."*

⭐⭐ **The concession inside the lesson.** The same paragraph says they use **"teacher-student models
to optimize onboard compute"**. ⇒ **The large model is a TRAINING artifact. The DEPLOYED model is
distilled and small.** The lesson's headline is about *maintainability and training*, and the
article silently concedes the deployment target is compressed — **which is TanitAD's architecture bet,
stated by our largest opponent.**

**Contradicting evidence (ours, MEASURED-by-vendor, today).** Alpamayo's own two model cards:
**10.5 B → 34.3 B moves open-loop minADE₆@6.4 s from 0.916 → 0.911 m (0.5 %) for 3.3× the
parameters**, and the stated closed-loop AlpaSim intervals **overlap** (1.37 ± 0.10 vs 1.50 ± 0.13).
The **language** score is what scales (LingoQA 74.2 → 79.2). ⇒ On the one public same-family pair we
can check, **"larger is better" holds for reasoning and does not hold for the trajectory.**

**Verdict.** ⚠️ **UNSUPPORTED-AS-STATED.** The maintainability half is credible and unfalsifiable-by-
us; the *capability* half ("scaling laws ⇒ better driving") is contradicted on trajectory metrics by
the only public evidence available and conceded by their own distillation step.

⇒ **GUIDELINE: stop treating "they are bigger" as a threat.** The programme's sub-300M bet is now
supported by (a) Alpamayo's flat trajectory scaling, (b) Waymo's own teacher-student concession, and
(c) our measured 100 ms budget. **Argue this in the paper.**

---

### L4 — *"You can't build trust with a black box"* · ✅ **SUPPORTED · CONFIRMS OUR HIERARCHY**

An *"independent onboard validation layer"* monitors every trajectory against *"hard physics-based
constraints and traffic laws"* as a *"hard backstop"*. **Verdict: SUPPORTED and uncontroversial** —
it is the standard safety-architecture position, and it argues **against** pure end-to-end, i.e. **for
a structured hierarchy with inspectable interfaces**, which is the TanitAD thesis. ⚠️ It also implies
their end-to-end foundation model is **not** trusted alone — worth remembering whenever their
end-to-end capability is quoted.

---

### L5 — *"Closed-loop simulation reveals more edge cases"* · ⭐⭐⭐ **CONFIRMS-US — THE STRONGEST ITEM IN THE ARTICLE**

**Their claim, verbatim on the mechanism:** open-loop replay is insufficient because surrounding
traffic *"moves, but it's completely indifferent to your actions"*; closed-loop *"mimic[s] real-world
cause and effect"*.

⭐⭐⭐ **This is `EVAL_DOCTRINE`'s T0/T1 split, stated by the operator with the most driverless miles
on earth, and arrived at independently.** Our doctrine came from our own measurement — open-loop
S-curve reproduction **97.9 %**, hold-action **0.0 %**, closed-loop **~5 %** — i.e. an **action echo**.
Waymo names the same defect from the simulation side.

**Now three independent lines agree**, which is a materially stronger position than we held this
morning:

| line | evidence |
|---|---|
| ours, MEASURED | action echo: 97.9 % open-loop vs 0.0 % hold-action |
| vendor, PUBLISHED | Alpamayo open-loop cannot separate a **3.3×** parameter gap (0.916 → 0.911 m) |
| operator, PUBLISHED-BLOG | *"completely indifferent to your actions"* |

⇒ ⛔ **T1 is not a TanitAD idiosyncrasy to be defended; it is the consensus position, and the burden
is now on any T0-only claim.** Every arm evaluated at T0 only is, by this standard, unevaluated.

---

### L6 — *"Every great driver needs a great Critic"* · ⚠️ **CONTESTED — THEIR INDEPENDENCE CLAIM IS ASSERTED, NOT DEMONSTRATED**

**Their claim.** A learned *"AI critic"* over millions of road miles and *"tens of billions"* of
simulated miles prevents the system from *"grading its own homework."*

⛔ **Contradicting evidence, and it is directly on point.** A **learned** critic trained on the same
data distribution as the policy is precisely the configuration in which evaluator failure correlates
with policy failure. `2606.03238` (*When RLHF Fails: A Mechanistic Taxonomy of Reward Hacking,
Collapse, and Evaluator Gaming*) and `2604.13602` (*Reward Hacking in the Era of Large Models*)
document that optimisation *"can raise the learned reward while external quality falls"*, and that a
critic *"may optimize toward evaluator artifacts... and the policy may inherit the same
misalignment."*

**Verdict.** ⚠️ **CONTESTED.** The *goal* (don't grade your own homework) is right and we should adopt
it. The *claim* that a learned critic achieves it is **asserted without an independence argument**.
A critic is independent when it can fail in ways the policy cannot — a property Waymo neither
defines nor evidences.

⭐ **This is our own rule in someone else's costume**, and we paid for it: *a probe that tunes on the
data it scores will manufacture a result — four distinct failures in one afternoon, each caught only
by a control that had to read a known value*. ⇒ **GUIDELINE: adopt the Critic pattern, but every
critic ships with a constant-only control and a raw-input floor, or it is not a critic — it is a
second head with the same blind spots.**

---

### L7 — *"VLMs improve scene reasoning"* but are *"too slow for real-time control"* and *"lack sufficient spatial awareness"* · ⭐⭐⭐ **CONFIRMS-US, THREE TIMES OVER**

This single lesson independently corroborates **three separate TanitAD positions**:

| their statement | our corresponding measured position |
|---|---|
| *"thinking fast and slow"* — fast sensor fusion for control, VLM for deliberation | **the strategic/tactical/operative hierarchy**, the programme's central thesis |
| VLMs *"too slow for real-time control"* | our own **MEASURED** rollout latency: linear in K, **203 ms at 37.8 M params for K=60** — a 10 B VLM is orders outside a 100 ms loop |
| VLMs *"lack sufficient spatial awareness"* | our **MEASURED v6 readout-geometry ceiling**: 16×40 tokens pooled to 4×4 = **4 azimuth bins over 120° (30°/bin), 2.1–7.8× too coarse** for BEV localisation |

⭐⭐ **The third row is the one to notice.** We diagnosed a geometry ceiling in our own encoder and
treated it as a TanitAD defect. Waymo reports the **same class of defect in frontier VLMs**. ⇒ It is
**not our bug — it is a property of semantic-pretrained vision stacks**, and it makes the geometry
fix a *differentiator* rather than a *catch-up*. This directly motivates the B13 track (below).

---

### L8 / L9 / L10 — governance, data flywheel, "no substitute for full autonomy" · ✅ **SUPPORTED, LOW INFORMATION FOR US**

- **L8 (governance)** and **L9 (data flywheel)** are organisational, and L9 is a description of the
  FlyWheel structure the programme already runs. Adopt-as-confirmation, nothing to test.
- **L10 (*"false summit"*)** is the competitive thesis against L2-to-L4 progression. **It does not
  engage TanitAD at all** — we are neither an L2 product nor a robotaxi. ⚠️ Its one transferable
  content: **capability measured under supervision systematically overstates capability without it**,
  which is L5 again in a deployment costume, and is the same reason our T0 numbers overstate.

---

## 3. Scientific strengths and weaknesses of this opponent

### Strengths (real, and we should not discount them)

1. **Exposure no one else has.** 200 M driverless miles produces a failure catalogue no benchmark contains.
2. **A closed-loop evaluation culture** — L5 + L6 + L9 describe a coherent, self-correcting measurement stack. This is genuinely ahead of the academic norm and ahead of us in scope (not in rigour).
3. **They deploy small.** The teacher-student concession means they have solved a compression problem we have only measured.
4. **Architectural convergence with their critics** (end-to-end foundation model) means their disagreement with Tesla/Wayve is narrower than the rhetoric — they are not betting on a different paradigm.

### Weaknesses (each is an opening)

1. ⛔ **Zero controlled ablations across ten lessons.** Every causal claim is attribution-from-exposure. **A programme that runs pre-registered discriminating experiments can say things they cannot.**
2. ⛔ **The Critic's independence is asserted, not demonstrated** (L6) — and the RLHF literature says that is precisely where learned evaluators fail.
3. ⛔ **Map dependence is a scaling liability by their own framing** — L2's value is highest exactly where coverage is hardest.
4. ⚠️ **Every lesson validates a prior Waymo decision.** Ten for ten, in a document released through a competitive comms push. Not falsification-seeking.
5. ⚠️ **No intervals, no `n`, no metric definitions anywhere.** Not one number in the article could enter our registry.

---

## 4. STRATEGIC guidelines for the frontier programme

| # | guideline | derived from |
|---|---|---|
| **S-1** | ⭐ **Stop defending T1; make it the burden of proof.** Three independent lines now agree that open-loop is uninformative. Any claim — ours or an opponent's — presented at T0 only is **unevaluated**, and the paper should say so in those words. | L5 + Alpamayo scaling + our action echo |
| **S-2** | ⭐ **Make "small at deployment" an argued position, not a constraint we apologise for.** Waymo distils to onboard; Alpamayo's trajectory metric is flat across 3.3×. The sub-300M bet is now externally supported. | L3 concession + Alpamayo |
| **S-3** | ⭐ **Claim the geometry gap as our differentiator.** Waymo names *"lack sufficient spatial awareness"* as a frontier-VLM limitation. We have it **measured** (4 azimuth bins / 120°). Fixing it is a contribution, not a repair. | L7 + our v6 ceiling |
| **S-4** | ⛔ **Fix our scope statement so L1 cannot land.** TanitAD claims a vision-only *latent world model on a camera corpus*, with a *leak-avoidance* inference rule. It does **not** claim L4 sensor sufficiency. Stating this precisely removes the strongest opponent argument against us at zero cost. | L1 |
| **S-5** | ⭐ **Our comparative advantage is experimental rigour, not scale.** They cannot run counterfactuals; we can. Every adopted opponent lesson enters as a **hypothesis with a pre-registered discriminating test**, never as inherited doctrine. | §3 weakness 1 |

## 5. TACTICAL guidelines

| # | action | why now |
|---|---|---|
| **T-1** | **Raise the NuRec `map.xodr` route to the top of the strategic-layer queue.** It is our only map prior, and L2 says the prior is load-bearing for exactly the layer our thesis rests on. | L2 names our one structural gap |
| **T-2** | **Every critic/scorer we build ships with a constant-only control and a raw-input floor** — and the v7 gate's learned components are audited for policy-critic correlation. | L6 contested + our own four-failure ridge-probe post-mortem |
| **T-3** | **Verify `2505.06113`'s 85.1 % / 92.1 % camera-vs-lidar parity figures in the primary.** They are currently RELAYED and are the most decision-relevant numbers in the L1 debate. | L1 evidence quality |
| **T-4** | **Add `2606.21775` (variable-length latent world models) to the horizon-ladder design review** — one predictor evaluating plans across horizons is exactly the tactical-6 s / strategic-30 s problem, and today's latency measurement says a flat K=300 rollout does not fit. | B12 scan, below |

---

## 6. ⛔ What this adjudication does NOT establish

1. **It does not refute L1.** Camera-only sufficiency for L4 is **open**; I found no controlled ablation on either side and am not claiming one exists.
2. **The 85.1 % / 92.1 % parity figures are RELAYED** — banked (`2505.06113`) but unread. They may not decide anything until verified.
3. **The Alpamayo counter-evidence to L3 is vendor self-report** on their own protocol, with `minADE₆` (best-of-6), and is **not** a controlled scaling study.
4. **I did not read the linked Waymo sources** — *"Demonstrably Safe AI for Autonomous Driving"* (2025-12), the Waymo Foundation Model post (2025-12), *"The Waymo World Model"* (2026-02). Those are the technical substrate under these lessons and are **queued, not covered**. Any claim about *how* Waymo does these things remains unverified.
5. **The comms-push framing is an inference from timing** (Axios 2026-08-26, Electrek 2026-08-27, co-CEO 2026-08-04), not from a stated intent. It calibrates the evidence class; it does not impugn the content.
