# H2 — Attention-based additional-sensor usage: design framing

**Status:** workstream OPENED 2026-07-25 by PI direction. This document is the **architectural framing
and design skeleton**; two commissioned inputs fill it in — `H2_SUBSTRATE_AND_LABELING.md` (what the AV
corpus actually permits) and `Research/2026-07-25-h2-sensor-attention/H2_RESEARCH_AND_SOTA.md` (prior art,
novelty, architecture ranking). **Sections marked ⟨PENDING⟩ are deliberately unfilled until those land** —
writing them now would be inference presented as design.

**Hypothesis home:** this is **H2** ("attention-based modality steering"), previously the portfolio's most
under-developed constitutional hypothesis (DoA 15 %, PARKED, *"nothing MEASURED on our stack"*). It is
hereby **UN-PARKED and promoted to a major workstream.** Update `HYPOTHESIS_LEDGER.md` accordingly.

---

## 1. The capability, stated precisely

The model runs primarily on the **front camera**. A **tactical** module continuously classifies the
driving situation and, when the situation warrants it, **requests an additional camera/sensor as a tool**.
This is the tactical brain's **MoE routing** decision: the router reads the situation and dispatches to
the expert/sensor the situation requires.

**Two claims are bundled here and must be separated, because they are measured differently:**

| | Claim | Measured by |
|---|---|---|
| **C-CAP** | *Capability.* The model knows **when it cannot see enough** and asks for the right sensor. | Does withholding the requested sensor degrade behaviour more than withholding a non-requested one? |
| **C-EFF** | *Efficiency.* Most frames need only the front camera, so selective activation buys embedded headroom. | What fraction of frames genuinely require ≥2 cameras, and what compute does that save? |

They are complementary but independent: a model could be right about *when* (C-CAP) and the answer could
be "almost always" (killing C-EFF), or the need could be rare (C-EFF strong) while the model's timing is
poor. **Report both; never let one stand in for the other.**

---

## 2. The output is a STRUCTURED, CAUSALLY-ORDERED triple — not a flat class

The PI's examples all decompose the same way:

> *"approaching intersection with unprotected left turn"* → *"yielding for left turn necessary"* → *"activating left camera"*
> *"driving behind slower vehicle"* → *"check overtaking to the left"* → *"activating left camera"*
> *"entering the roundabout"* → *(merge/yield)* → *"activating left camera"*

**⟨SITUATION⟩ → ⟨TACTICAL OPTION⟩ → ⟨SENSOR REQUEST⟩**

This ordering is the design's core, and it is a **causal** claim, not a formatting convention:

> **You need the left camera *because* you are considering a manoeuvre whose risk lies to the left.**

Three consequences follow, and each is a design requirement:

1. **The sensor request is predicted as a CONSEQUENCE of the tactical option, never as an independent
   class.** Architecturally: the sensor head is conditioned on the tactical-option representation.
2. **Interpretability comes free** — the model states *why* it wants the camera, in the form the PI asked
   for. That is the deployable artifact, and it is also the debugging surface.
3. **Compositional generalization becomes testable**: a *novel* situation paired with a *known* manoeuvre
   should yield the correct sensor. A flat situation→sensor classifier cannot do this; a
   situation→option→sensor model can. **This is a discriminating prediction, and we pre-register it.**

⚠️ **Do NOT reuse the existing 5-way manoeuvre softmax as the tactical-option head.** It is documented to
**mix lateral and longitudinal into one head**, which is the identified root cause of the program's
longitudinal blindness (0/881 accelerate, the speed-fan, no longitudinal signal in selection). The
tactical-option space here must be **factored** — at minimum lateral-intent × longitudinal-intent ×
yield/merge-obligation — or we rebuild a known defect into a new brain.

---

## 3. ⚠️ THE LABEL IS THE WHOLE EXPERIMENT — and the obvious label is circular

This is the single highest-risk decision in the workstream, and the program has already paid for getting
it wrong once.

**The precedent (MEASURED, 2026-07-25):** the strategic brain's route target was a *lookup of its own
input* — `route_target = _NAV_TO_ROUTE[nav_cmd]`. The model was trained to predict something it was
already being told. Route cross-entropy reached **exactly 0.0** by step ~14.5 k and **`route_skill = 0.0`
by construction.** Months of a brain that could not, even in principle, learn anything. That failure is
indistinguishable from "the idea doesn't work" unless you look at the label definition.

**The equivalent trap here:** labelling *"activate left camera"* by a rule over something the model also
sees — e.g. `roundabout ⇒ left camera`, or deriving it from the route/nav command. The model would learn
the rule, score ~100 %, and have demonstrated **nothing about sensor need**. It would look like a success.

### Label candidates, ranked

| | Definition | Circular? | Verdict |
|---|---|---|---|
| **L1 — counterfactual information (PREFERRED)** | Camera *X* was **needed** at time *t* iff there existed an agent/obstacle that (a) fell inside *X*'s frustum, (b) was **NOT visible** in the front camera (out-of-FOV or occluded), **and** (c) **constrained the ego's realised behaviour** (yield, wait, abort/defer a lane change). | **No** — derived from world geometry + realised behaviour, not from any model input | **Build this.** Requires `obstacle.offline` 3D tracks + per-camera extrinsics/intrinsics. Feasibility ⟨PENDING substrate⟩ |
| **L2 — oracle ablation** | Camera *X* was needed iff a model *with* *X* outperforms one *without* it on that window. | No | A true oracle and the ideal *validation* signal, but expensive (n models × n cameras). **Use as the evaluation ceiling, not the training target.** |
| **L3 — rule from manoeuvre/route labels** | `roundabout ⇒ left`, `left-turn ⇒ left`, … | **YES — circular** | **Forbidden as the primary target.** Admissible ONLY as (i) a weak prior for label bootstrapping and (ii) a sanity baseline the real model must beat. Any result on L3 alone is inadmissible. |

**Condition (c) is what makes L1 scientifically load-bearing.** Mere *presence* of an agent in the left
frustum is cheap and nearly always true in traffic; **decision-relevance** is the rare, meaningful event.
A label built only on (a)+(b) would produce a near-constant "yes" and a useless router — which is
precisely the **router-collapse** failure mode. ⟨Exact operationalization of (c) PENDING substrate: it
must be derivable from ego pose / kinematic labels without reference to any model input.⟩

**Coverage and class balance are gating facts, not details.** ⟨PENDING⟩ If the positive class is <1 % of
windows, the experiment is a rare-event detection problem and must be designed as one (stratified
sampling, PR-AUC not accuracy, and a power calculation *before* training).

---

## 4. Architecture — two variants, to be ranked before building

**V1 — tactical head on the FROZEN world-model encoder + predictor.** Reuses flagship-v1's frozen encoder
(and optionally the predictor's imagination) as substrate; a new tactical head emits the
situation→option→sensor triple. Precedent exists in-program: the D1 frozen-WM planner reached 0.599 m and
the frozen encoder is a proven substrate (it carries the IDM's speed R² 0.885, and E2a showed lateral
offset is linearly recoverable at R² 0.72). **Cheap, attributable, does not disturb any trained arm.**

**V2 — tactical layer with its OWN encoder and image processing.** Higher ceiling, far higher cost, and it
re-opens the encoder question the program has already spent heavily on. ⚠️ Relevant prior: **Branch-B's
from-scratch camera-conditioned encoder FAILED** (cross-rig speed R² −0.667 vs frozen v1's +0.657, refuted
at power) — an own-encoder route here inherits that risk and must state how it differs.

**Recommendation: build V1 first** — not because it will win, but because it is the **attributable** one.
If V1 works, the capability is proven on a substrate we already trust, and V2 becomes a measured upgrade
with a clean baseline. If we start with V2 and it works, we cannot say whether the win came from the
capability or from the new encoder. *(Final ranking ⟨PENDING research input⟩.)*

**Known risk to design against from the start — router collapse.** Discrete-routing modules degenerate to
a constant output with distressing reliability. Guards to specify: load-balancing auxiliary loss,
temperature/Gumbel schedule, and — most importantly — **a router that is measured against a
majority-class baseline, not against accuracy.** *(The strategic head reports `route_acc = 1.0` while
`route_skill = 0.0`: perfect accuracy, zero skill. We will not be fooled by that number twice.)*

---

## 5. Evaluation — the counterfactual withholding test is the primary

**Primary (C-CAP): counterfactual sensor withholding.** When the model requests camera *X*, withhold *X*
and measure the behavioural degradation; compare against withholding a **non-requested** camera and
against withholding a **random** camera. *A correct requester degrades significantly more when its own
request is denied.* This measures **decision-relevance**, not label agreement, and it is robust to label
noise — the property that matters, since our label is derived.

**Secondary:** PR-AUC on the L1 label (rare-event framing); situation-classification accuracy **vs a
majority-class baseline** (never bare accuracy); per-stratum breakdown over **lane-change / roundabout /
intersection**; and the **timing** of the request (is the camera asked for *early enough to act on*? A
correct request one frame before the conflict is useless).

**C-EFF:** the fraction of frames requiring ≥2 cameras, and the implied compute saving against an
always-all-cameras baseline.

**Estimator discipline (binding):** paired **episode-cluster bootstrap** (`taniteval/ci.py`, B=2000); every
interval names its estimator; ≥40 episode-clusters for any decision-grade claim; stratified reporting.
**Never `overlapping_holdout_se`** — measured on 2026-07-25 to bias intervals *and* point estimates.

**Pre-registration is required before any training run,** with both outcomes committed in advance —
including the honest null: *"sensor need is predictable from the front camera alone at ≤ chance above the
majority-class baseline"*, which would be a real and publishable negative.

---

## 6. Why this also strengthens the hierarchy proof

This workstream is **not** a detour from the Hierarchy Proof Program — it feeds it. HP-3 (route
counterfactual) asks whether the model's behaviour changes appropriately when the commanded intent
changes. **The sensor request is a second, independent channel of exactly that test**: swap the tactical
option and the requested sensor should change accordingly, *by the causal structure of §2*. A flat model
has no mechanism to do this. So H2 supplies a **discriminating prediction for the hierarchy** that is
cheaper to measure than trajectory-level behaviour and less confounded by the corpus's 74 %-straight
composition. ⟨To be added to the HPP battery as **HP-7** once the substrate verdict is in.⟩

---

## 7. Phasing (gated, each phase cheap before the next)

| Phase | Work | Gate to proceed |
|---|---|---|
| **H2-0** | Substrate + label feasibility ⟨RUNNING⟩; research/novelty ⟨RUNNING⟩ | L1 computable at usable coverage **and** the target situations are powered (n ≥ ~40 episode-clusters) |
| **H2-1** | Build the L1 labeller; publish coverage, class balance, and a **hand-audited sample** (the VLM-labelling lesson: an agreement number on an imbalanced corpus is not a recall number) | Label passes audit; positives are not degenerate |
| **H2-2** | Pre-register; V1 head on frozen WM; train the situation→option→sensor triple | Beats the majority-class baseline **with skill**, not accuracy |
| **H2-3** | The counterfactual withholding evaluation (the real test) | CI-separated degradation on requested-vs-non-requested withholding |
| **H2-4** | V2 (own encoder) as a measured upgrade against the V1 baseline; and HP-7 into the hierarchy battery | — |

**Corpus:** PhysicalAI-AV only, per PI direction. **Parity discipline applies** — any episode selection must
not break cross-arm comparability; the newly-landed content-hash guard covers this.

---

## 8. PI decisions — ANSWERED 2026-07-25 (these are now binding constraints, not open questions)

The three questions below were put to the PI and answered the same session. They are recorded here as
**design constraints**; §2–§7 above are to be read subject to them.

### D1 — Cameras only, in step 1
No radar/lidar in the tool vocabulary for the first cut. Other modalities may be noted if the corpus
carries them, but they are out of scope. *(Simplifies the tool vocabulary to a small discrete set of
camera IDs — which also makes the MoE router tractable and the efficiency arithmetic clean.)*

### D2 — ⭐ THE INPUT IS FRONT-CAMERA-ONLY. This is the experiment.
> *"I would like to validate the semantic classifier and action taker based only on the front camera."*

**Binding architectural constraint: at decision time the model sees ONLY the front camera.** The other
cameras exist exclusively as (a) the source of the ground-truth label and (b) the thing that gets
activated. **They are never model input.**

This sharpens the scientific question considerably, and it is worth stating the sharpened form plainly:

> **Is the NEED for an off-front camera predictable from the front camera alone?**

The hypothesis is that it is — because need is **announced by front-visible scene structure**: a junction
geometry approaching, a lead vehicle decelerating, a merge taper, a roundabout entry. The model does not
need to *see* the hazard; it needs to recognise **a situation in which an unseen hazard becomes
decision-relevant.** That is exactly a *semantic* competence rather than a perceptual one, which is why
the capability belongs in the **tactical** layer and not in perception.

⚠️ **The honest null must be pre-registered alongside it:** *if the need is genuinely unpredictable from
the front camera, H2 fails — and that is a real, publishable negative* ("off-front hazard need is not
anticipable from forward view alone"). We do not get to work around it by quietly leaking the other
cameras into the input.

### D3 — ⭐ The EFFICIENCY claim is the point (C-EFF promoted to PRIMARY)
> *"The point is not to process all cameras the whole time, only if the situation is requesting this —
> efficiency claim."*

C-EFF is **not optional and not secondary**. The headline number is therefore:

> **What fraction of frames genuinely require ≥2 cameras?**

and the deployment claim is the complement of it. This reframes the whole evaluation:
- If the fraction is small (say ~5 %), the claim is strong: **~95 % of multi-camera compute is avoidable
  without losing the capability** — a real embedded result for a sub-300 M model.
- If the fraction is large (say >50 %), the efficiency claim is **weak, and we need to know that before
  building**, not after. It is a gating measurement in H2-0, not a post-hoc report.
- The trade curve matters more than any single point: **recall of genuine need vs cameras activated.**
  A model that activates everything achieves perfect recall and zero efficiency; the baseline to beat is
  the always-on rig, and the metric is the Pareto frontier between missed need and compute spent.
- **Asymmetric costs must be stated:** a missed activation is a potential safety failure; a spurious
  activation costs only compute. The operating point should be chosen on that asymmetry, and reported —
  never left implicit in an F1.

### D4 — The corpus is NOT only highways; and additional data is authorized
> *"use the AV dataset from NVIDIA, it's not only highways, search for intersections or roundabouts, look
> for additional accessible data."*

⚠️ **My earlier framing that the corpus is US-highway-weighted was an UNVERIFIED assumption and is
withdrawn.** The substrate probe has been redirected to *actively search* PhysicalAI-AV for intersections
and roundabouts using the metadata that owns the fact (map/lane annotations, the full 36-feature schema,
junction annotations, and ego-trajectory geometry such as sustained yaw through a closed curve for
roundabouts) — rather than sampling and inferring. Counts are to be reported separately for
**intersection**, **roundabout**, **lane-change**, at episode and window granularity.

In parallel, an **external multi-camera data survey** is running (`Data Engineering/Research/
2026-07-25-h2-multicam-data-survey/`). Its binding constraints: **multi-camera + published extrinsics +
3D agent tracks + ego pose** (anything single-camera cannot serve this workstream — comma2k19 included),
our license taxonomy respected (**waymo/waymax are `refuse`**; PhysicalAI-AV is gated and must never enter
a published tier), and **drone/BEV datasets (rounD, inD, exiD, highD) explicitly flagged as
non-qualifying as a training source** — they have no front camera, so a rich roundabout count there must
not masquerade as a fit.

**Decision rule (pre-committed):** stay in-corpus if PhysicalAI-AV yields enough powered situations
(the standing bar is ≥ ~40 episode-clusters per stratum); acquire external data only below that, and only
for the specific stratum that is short.

---

## 9. Remaining open question for the PI

**Only one is left, and it is deferred until the counts are in:** if roundabouts specifically turn out to
be scarce in PhysicalAI-AV *and* the best external source is `nc-research`-licensed (nuScenes, A2D2, ZOD
and similar all are), then adding them means the H2 result is **research-tier only and cannot enter the
commercial dataset tier**. That is a strategic trade — scientific coverage vs commercial reach — and it is
the PI's call, not an agent's. It will be put once the two surveys report actual numbers.
