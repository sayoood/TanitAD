# D-DATA-EFFICIENCY — a reframe, a position, and five experiments

**Date** 2026-08-30 · **Owner** DataFlyWheel · **Status** POSITION + PROPOSALS.
Nothing here is a result. Every number carries its evidence class.

**The PI's question:** *"How can we achieve the best possible results with the
LOWEST AMOUNT OF DATA, and how does this correlate with the AI ARCHITECTURE and
the TRAINING PROCESS? Is it a combination of smart semantic distribution, rarity,
the right architecture, the right training recipe, other aspects?"*
**The motivation:** *"disrupt the best frontier players … by dramatically
reducing the dev, training and inference effort — data is probably one of the
largest levers."*

---

## 1. The reframe — three moves

### Move 1 — we are not short of hours. We are short of DECISIONS.

MEASURED today on our own corpus, 20 s trainable window, n = 4,719:

| | share | of our 26.2 h |
|---|---|---|
| clips containing a longitudinal **decision** (stop→launch or stop-and-go) | **13.3 %** | **≈ 3.5 h** |
| clips containing **none** — 20 s of cruise | **86.7 %** | ≈ 22.7 h |
| urban **and** decision-free | 62.8 % | ≈ 16.5 h |

⇒ **We do not hold 26 hours of driving data. We hold ~3.5 hours of driving
decisions and ~23 hours of cruise.** "Less data" is the wrong axis; the axis is
**decisions per hour**. Every lever below is judged by whether it raises that.

### Move 2 — three different efficiencies are being conflated

The motivation names *dev, training and inference* effort. These are separate
problems with separate levers, and conflating them is how a programme optimises
the easy one and reports it as the hard one:

| efficiency | the question | our lever |
|---|---|---|
| **DATA** | fewer hours for the same competence | curation, decision density |
| **TRAINING** | fewer GPU-hours for the same competence | within-clip sampling, curriculum, staging |
| **INFERENCE** | fewer FLOPs deployed | architecture, distillation — *unrelated to data* |

⭐ Only the first two are data questions. **Inference effort is won or lost in the
architecture, and no amount of curation touches it.** Worth saying plainly so the
data programme is not credited with — or blamed for — a number it cannot move.

### Move 3 — the target is CONTROL competence, and prediction is not a proxy for it

**PUBLISHED (DINO-WM, banked):** over **92× data**, prediction fidelity moves
**+4 %** while control competence moves **11.5×**. If that transfers, then:
* data that improves *prediction* is nearly worthless past a small volume;
* data that exercises *decisions* is what moves control;
* **any data score validated against reconstruction loss is measuring the
  saturated axis.**

Our own §15 result is the same lesson from the other side: an anti-drift
objective cut drift **25 %** and destroyed prediction, because the regulariser's
argument was the read's own residual. The programme already adopted *"drift alone
is never a valid objective"*. The generalisation: **optimise the axis you are
short of, and verify you have not bought it by destroying another.**

---

## 2. The measured ground (ours, today)

| finding | evidence |
|---|---|
| corpus is scene-**diverse**: 148 of 150 strata cells occupied, 140 with ≥5 clips | MEASURED |
| corpus is decision-**sparse**: 86.7 % of clips carry no longitudinal decision | MEASURED |
| 76.1 % of clips never stop; 11.8 % have a full stop→launch; 2.4 % stop-and-go | MEASURED |
| Codevilla's inertia failure does **not** apply to us — too few stopped frames, not too many | MEASURED |
| our **designed** corpus deleted highway entirely (hard gate ≤ 14 m/s) while the accidental one kept 26 % | MEASURED, source-verified |
| 88.7 % of the oracle gap is **longitudinal** — the axis our selection thinned | INHERITED (registry) |
| we use **1.54 % of clips and 0.22 % of frames** (1 of 7 cameras) of a 1,701 h parent | MEASURED |
| curation beat volume **~3,300×** (WOD-E2E: 12 h curated ≈ 40,000 h accumulated) | PUBLISHED |
| **no published work ablates diversity at FIXED hours** | probed, Research Lab |

---

## 3. My position — falsifiable, and not yet tested

> **Scene diversity and decision density are different quantities. We have been
> optimising the first. Control competence feeds on the second.**

Supporting reasoning, stated as reasoning and not as result:

* A 20 s clip of uninterrupted cruise is scenically rich and contains almost no
  control decision. Our corpus is 86.7 % such clips.
* If DINO-WM's prediction/control split transfers, our 88.7 %-longitudinal gap is
  a **data-composition** symptom before it is a model-capacity one — and scaling
  parameters against this corpus would spend on the saturated axis.
* ⇒ the cheapest large lever is not more hours. It is **raising decisions per
  hour**, by selection *across* clips and by weighting *within* them.

⛔ **This is a position, not a finding.** DDS has passing controls and **zero**
validation against trained outcomes. If a decision-dense arm does not beat a
scene-diverse one at equal hours and equal compute, the position is wrong and
should be dropped rather than defended.

---

## 4. Five proposals, ranked by information per unit cost

### P1 — The decision-density ablation *(the core experiment; also the literature gap)*
Equal **hours**, equal **GPU-time**, arms differing **only** in composition:
`decision-dense` · `scene-diverse` · `random` · `r0-gated` (free known-bad).
Score on **T1 control**, four metric families, never reconstruction.
**Why first:** it is simultaneously the test of the position, the validation of
DDS, and the unclaimed publishable result. **Cost:** 4 arms × one training run.

### P2 — Within-clip decision weighting *(cheapest lever in the programme)*
We select clips but train **uniformly over the 20 s window**, and decisions are
sparse *inside* a clip. Weight windows by decision density (we now ship
`stop_launch`, `lat_peak_m`, `stopped_frame_frac` per clip).
**Costs no new data at all** — it is a sampler change. **Why second:** if the
position is right, this is free competence; if it is wrong, it costs one run.

### P3 — Measure OUR saturation curve
Train at 25 / 50 / 100 % of the corpus, equal compute, and plot **prediction vs
control** separately. DINO-WM says they diverge; nobody has checked on our data.
**This answers "how much data do we actually need" with a number instead of an
argument** — and it tells us whether we are on the flat part of the curve, where
more hours are worthless.

### P4 — Rarity mining with no labels and no VLM
WOD-E2E curated for **<0.03 %** scenarios. We hold **1,701 h** and use 1.54 %.
Rare events — hard braking, large yaw, stop-launch, cut-ins — are detectable from
**egomotion alone**: no labels, no VLM, no GPU. This converts *"we use 1.5 %"*
into *"we use the RIGHT 1.5 %"*, and it is the only proposal that adds data
without adding hours-of-cruise.
⚠️ It needs the epcache built for whatever it selects — that is the real cost,
not the mining.

### P5 — The 7-camera question, asked rather than assumed
We read **1 of 7** cameras: a 7× frame multiplier at **zero acquisition cost**.
⚠️ But **more views is not more decisions** — the other six cameras see the same
decisions from other angles. My expectation is that it helps *perception* and
barely moves *control*, which by Move 3 is the saturated axis. **Cheap to test,
and worth testing precisely because it looks like free data and probably is not.**

---

## 5. Research strategy — how to run these so the answers are admissible

1. **Pre-register both outcomes** before each arm, per the operating standard. An
   experiment whose failure has no stated meaning cannot fail.
2. **Equal size AND equal GPU-time.** A score or a recipe that merely rewards
   more data or more compute has measured nothing.
3. **Score CONTROL, at tier T1.** Reconstruction and open-loop numbers are
   inadmissible for a competence claim (EVAL_DOCTRINE; and the action-echo
   result: 97.9 % open-loop vs 0.0 % hold-action).
4. **Every arm carries the four metric families**, per family with its `n` — a
   composite hides exactly the trade-off we are looking for.
5. **Controls that must read a known value.** The r0-gated arm is a free
   known-bad on the longitudinal axis; a random-subset arm is the floor.
6. **The paper follows the experiment, not the other way round.** The diversity-
   at-fixed-hours gap is real, but it is only ours if P1 produces a result.

---

## 6. What would refute me

* A **scene-diverse** arm matching or beating a **decision-dense** arm at equal
  hours and equal compute ⇒ the position is wrong; decision density is not the
  axis.
* **Prediction and control saturating together** on our data (P3) ⇒ the DINO-WM
  transfer does not hold here, and the "prediction is the saturated axis"
  argument collapses.
* **Within-clip weighting (P2) producing no gain** ⇒ decisions are not
  sufficiently sparse *within* clips for the sampler to matter, and the lever is
  across-clip only.

⭐ I would rather these be run than argued. The instruments now exist: strata
cells, decision flags per clip, DDS with passing controls, a fixed eval split,
and a T1 harness. **What is missing is a single equal-hours experiment, and
until it exists this document is a hypothesis with good bookkeeping.**
