# TanitLang literature — CoT INFLUENCE vs CoT CONSISTENCY, the frontier that bears on the tension

**2026-09-06 · TanitAD Research Lab, Architecture & Inference · for TanitAD_TrainingFlyWheel.**
Companion to `DIALOGUE_03_TANITLANG_V3_REFC_NATIVE.md` (§3 CFG loop-resolution, §6 H-TL-1..5).
**40 papers banked this pass** (`tools/kb_add.py`, library 447 → 487 entries, `--verify` clean on all 40).

⚠️ **Evidence classes used below, strictly.**
`PUBLISHED-READ` = banked PDF **and** its body text read by me this session (the artifact is named).
`PUBLISHED-ABSTRACT` = banked PDF, only the arXiv abstract read — **numbers from these are not
decision-grade** and are never used below to justify a design choice.
`INHERITED` = from a TanitAD doc, not re-verified. `HYPOTHESIS` = mine.
⛔ Nothing here is `MEASURED` — no experiment was run this pass.

---

## 0. ⭐ VERDICT — what the frontier actually gives us, per topic

The five topics are not five separate answers. **Read together they say one thing:** the field has
converged on *intervention* as the only admissible evidence of influence, has discovered that
intervention magnitude alone is **not** evidence of *semantic* influence, and has **not** solved the
influence/consistency tension — the one paper that names it says plainly that no current design
satisfies both.

| # | topic | ⭐ the single most transferable mechanism, named concretely |
|---|---|---|
| **1** | verifier-guided / process-reward decoding | ⭐ **The PAV non-degeneracy condition** (`2410.08146`): a process signal is informative iff the **variance of its advantage across the candidates it must rank is large**, under a scorer *complementary to* the thing being scored. Directly: **a TanitLang prior whose `Var` over the 117 anchors is ≈0 is inert by construction, and we can compute that scalar before running any eval.** |
| **2** | CoT faithfulness / causal influence | ⭐ **Lanham's four intervention probes + AOC** (`2307.13702`) supply the *form*; **VLADriveBench's `selfsplice`** (`2606.12706`) supplies the *control that must read a known value* — re-inject the model's **own** CoT through the injection pipeline and the output must be **bit-identical**. Ours is already specified as the `w = 0` identity gate; selfsplice is the same gate, and it is the reason the delta is admissible. |
| **3** | energy-based / joint (reasoning, action) scoring | ⭐ **EBT's reframing** (`2507.02092`): train a scalar `E(context, candidate)` **verifier** and make prediction *optimisation against it*, rather than emitting the answer. For us: **REF-C already has this object — `sel_score` over 117 anchors IS `E(cond, anchor)`.** The transferable step is to extend its argument to the *pair*: `E(cond, rationale, anchor)`, so consistency is a **term in the score**, not a post-hoc comparison. |
| **4** | CFG beyond images | ⭐ **The delta is real and readable, but it is NOT a clean "amount of conditioning"** (`2408.09000`, proven by counterexample): CFG does not sample the gamma-powered distribution, and DDPM/DDIM variants give *different* distributions. ⇒ Use `S1 − S0` as an **influence statistic on the selection surface** (where it is exactly a log-space difference we control), **never as a claim about the sampled trajectory distribution**. |
| **5** | rationale ↔ control consistency in driving | ⭐ **Split the question in two, as VLADriveBench does**: *observational alignment* (does the rationale correlate with the action?) and *causal intervention* (does replacing it move the action?). ⛔ **They diverge sharply and either alone gives the wrong verdict.** Plus `2608.29583`'s finding that the **representation** you compare in matters more than the judge: lane-relative geometry beat a bigger judge model. |

### ⛔⛔ The three findings that should change the TanitLang plan *today*

**(a) A LARGE intervention effect is NOT evidence of semantic influence — measured, twice over, in our own domain.**
`PUBLISHED-READ` (`2606.12706` §4.2): ORION's CoT demonstrably moves its trajectory — injecting a
braking CoT shortens the 3 s prediction on **81.9 % of steps, mean −0.437 m**. Then:
* **shuffling the CoT's words** (destroying grammar and semantics) reproduces it almost exactly — **81.4 % of steps, −0.429 m**;
* replacing the whole CoT with **repeated copies of the single token `"depicts"`** produces a **stronger** effect — **90.4 % of steps, −0.846 m, ≈2× the coherent text**.

⇒ **A non-zero `‖S1 − S0‖` would NOT establish H-TL-2.** `DIALOGUE_03` §3 currently treats the
guidance delta as *"the CoT's causal contribution … a number CFG computes anyway"*. That is the
right instrument and the wrong stopping point: **the delta measures that the channel is wired, not
that it is meaningful.** The discriminator is a **semantic-null arm** — a prior that is
*equally strong in magnitude but semantically destroyed* (shuffled vocabulary logits, or a
constant-energy prior matched in L2). If the shuffled prior moves selection as much as the reasoned
one, the channel is a magnitude channel, not a reasoning channel. **This is the cheapest experiment
in the whole plan and it is not currently in WP-2.**

**(b) The tension we are trying to dissolve is, on the published evidence, a genuine trade-off — not a design error.**
`PUBLISHED-READ` (`2606.12706` §5): *"Tight coupling between CoT and action (Alpamayo v1.5) makes
reasoning faithful … but creates a pathway through which reasoning errors propagate to
safety-critical actions … Loose coupling (ORION) makes the model robust to CoT errors but renders
the CoT unreliable to explain the model's actions. **Neither design satisfies both desiderata.***"
⇒ Our `lan_gate` / `w` knob is **exactly the dial on that trade-off**, and the honest framing of
TanitLang is not *"we get both"* but *"we can **measure where we sit on it** and pick `w` with the
curve in hand."* That is a stronger and more defensible claim than dissolving the tension, and it is
the one the literature supports.

**(c) Supervising the rationale from the logged future trajectory manufactures a rationaliser.**
`PUBLISHED-READ` (`2608.01755`, DEFT-RLVR, intro): CoT supervision conditioned on the GT trajectory
means *"the VLM CoT annotator rationalizes a known outcome rather than inferring a decision from
scene evidence"* — they name it **trajectory anchoring bias**, and report GT-conditioned CoTs have
**lower causal faithfulness** than causal-planning CoTs, with the degradation **concentrated in hard
causal scenarios**, plus **more severe hallucinations**.
⇒ This is our own binding rule (`CLAUDE.md`: *labels may use ego; inference is vision-only*) arriving
from outside, sharpened: **it is not enough that the rationale not READ the future at inference — its
TRAINING TARGET must not be derived from the future either**, or H-TL-3 is decided before we start.
⚠️ **LCDrive (`2512.10226`) cold-starts exactly this way** ("supervising … based on ground-truth
future rollouts"), on **our corpus**, and never measures faithfulness. A design we might otherwise
have copied.

---

## 1. Topic 1 — Verifier-guided / process-reward decoding

**What is measured, and what makes a verifier signal non-degenerate.** This literature's answer to
our question is unusually crisp, and it is *not* "make the verifier accurate".

### ⭐ `2410.08146` — Rewarding Progress: Scaling Automated Process Verifiers · `PUBLISHED-READ`
Library key `2410.08146` · banked `…/Library/papers/2410.08146_Rewarding-Progress-*.pdf`

**Finding (read in PDF, §3.2–3.4):** the process reward for a step should be **progress** — the
*change in likelihood of eventual success*, i.e. a **step-level advantage** `A^μ(s,a)` — and it must
be measured **under a prover policy μ distinct from the base policy π**. Verbatim mechanism:
setting `μ = π` gives *"exactly the same policy gradient update as only optimizing outcome
evaluation"* — the process signal vanishes.

⭐⭐ **The non-degeneracy condition, which is the transferable part:**
* a prover that is **too capable** succeeds from *every* prefix ⇒ `A^μ ≈ 0` everywhere ⇒ no signal;
* a prover that is **too weak** fails from every prefix ⇒ `A^μ ≈ 0` ⇒ no signal;
* good provers are **complementary to the base policy**: formally, **large variance of the advantage
  across actions at a state**, `Var_a[A^μ(s,a)]`, while **not being anti-correlated** with the base
  policy's own ranking (their `E[A^π, A^μ]` term *"not too negative"*, else the two signals conflict).
* Empirically, RL against advantages from an over-strong prover *"leads to policies that only
  produce re-phrasings of the question"* — **degeneracy has a characteristic signature: fluent,
  content-free intermediate steps.**

**Numbers (read in PDF, abstract + §1):** search against PAVs is **> 8 % more accurate** and
**1.5–5× more compute-efficient** than re-ranking complete traces against an ORM; online RL with PAV
dense rewards gives **5–6× sample efficiency** and **> 6 % accuracy** over ORMs, and **8× better
Pass@N**.

**What it changes in our design if true:**
1. ⭐ **A pre-eval admissibility screen we can compute in one line.** For any TanitLang prior
   `p ∈ ℝ^117` entering `terms`, compute `Var_a[p]` (and its rank-correlation with the classifier
   surface `conf0`). **A prior that is near-flat over the fan is degenerate and cannot help
   selection, whatever its CE loss says.** This is the fan-space analogue of "the prover cannot
   distinguish steps", and it costs nothing.
2. ⛔ **It argues against making TanitLang's prior head a copy of the trunk's own preference.**
   `μ = π` collapses the signal. The reasoner must be **complementary** — it should be scoring the
   fan from information the small heads do not have (fan geometry + nav + query), not re-deriving
   the trunk's ranking. If `lat_prior^TanitLang` is rank-identical to `lat_prior^trunk`, the module
   is provably a no-op on selection.
3. **The "fluent but content-free" signature is our watch-item for the free-text head** (H-TL-5).

### `2305.20050` — Let's Verify Step by Step · `PUBLISHED-ABSTRACT`
Process supervision (per-step feedback) is compared head-to-head with outcome supervision; the
paper's own framing is that process supervision is the more reliable trainer. **Changes for us:**
establishes the *precedent* that per-step labels beat end-labels — our per-anchor prior is the
per-step object. ⚠️ I did not read the PDF body; its headline numbers are not quoted here.

### `2211.14275` — Solving math word problems with process- and outcome-based feedback · `PUBLISHED-ABSTRACT`
First comprehensive process-vs-outcome comparison; distinguishes **final-answer errors** from
**reasoning errors**. **Changes for us:** it is the citation for *why we must report tactical/
strategic decision quality separately from ADE* — the same distinction, and it is already binding
here (four metric families).

### `2110.14168` (verifier + best-of-N reranking) · `2312.08935` (Math-Shepherd: PRM labels from MC completion rate, no human step labels) · `2412.01981` (Implicit PRM: process reward for free from an outcome-trained log-likelihood ratio) · `2408.15240` (Generative Verifiers: verifier as next-token prediction) · `2504.16828` (ThinkPRM: verifier that reasons before verifying) · `2510.08049` (PRM survey) · `2412.06559` (ProcessBench) · `2507.15512` (step-level verifier-guided hybrid TTS) — all `PUBLISHED-ABSTRACT`

**The one with the largest design consequence is `2412.01981` (Implicit PRM).** Its claim is that a
process reward can be **read off an outcome-trained model as a log-likelihood ratio, at no extra
labelling cost**. **What it would change:** if it holds, **we do not need step-level rationale labels
at all** — a per-anchor process signal could be derived from REF-C's *existing* outcome-trained
selection scorer. That would remove the largest data blocker in the TanitLang plan (`DIALOGUE_03`
§7: *"we have no question corpus"*). ⛔ **Read the PDF before this enters a plan** — it is currently
abstract-level only and it is exactly the kind of claim that should not be INHERITED.

### `2310.17022` — Controlled Decoding · `PUBLISHED-ABSTRACT`
A **separate prefix-scorer module** learns a value function for the reward and steers a **frozen**
base model at inference; prefix scorers for multiple rewards **compose at inference time**.
**Changes for us:** this is the architectural licence for the shape we already want — TanitLang as a
*scorer that steers a frozen REF-C*, with **multiple priors composing additively**. REF-C's `terms`
list is literally an additive composition of prior scorers in log space. ⭐ The correspondence is
close enough to be worth stating in a pre-registration.

### `2607.00399` — DriveVer: lightweight test-time trajectory verifier · `PUBLISHED-READ` (abstract in PDF)
**Finding:** a **34 M-parameter, plug-and-play** test-time verifier with a **dual head** — a *safety
confidence score* **and** an *absolute geometric refinement vector* — fuses candidate trajectories
with multi-view visual features and ego kinematics, trained on a NAVSIM-derived trajectory set built
by **condition-driven clustering and balanced sampling over ego state and navigation command**.
**Changes for us:** (i) it prices the verifier — **34 M is affordable next to our ~6 M reasoner and
2.15 M cascade**; (ii) its **balanced-by-(ego-state, nav-command) sampling** is the direct answer to
our anchor-bank imbalance; (iii) **dual head = score + refinement** is precisely REF-C's
`conf` + `offset` pair, so the port is a re-parameterisation, not a new mechanism.

### `2606.08525` DriveReward · `2606.24231` FlowR2A — `PUBLISHED-ABSTRACT`
DriveReward: a generative VLM reward model scoring multi-modal trajectories on several dimensions,
used both for RL fine-tuning and **test-time selection**. FlowR2A: learns a **reward-conditioned
action distribution over a dense driving action vocabulary** from simulation rewards
(safety/progress/comfort/rule-compliance). **Changes for us:** FlowR2A is the closest published
object to *"a distribution over an action vocabulary produced by a scorer"* — i.e. our anchor prior
— and it is trained from **simulator rewards**, which we can produce (AlpaSim/NuRec) and the CoT
corpus we cannot.

---

## 2. Topic 2 — Faithfulness / causal-influence measurement of CoT

**This is the closest analogue to our influence requirement, and the metric definitions are here.**

### ⭐⭐ `2307.13702` — Measuring Faithfulness in Chain-of-Thought Reasoning (Lanham et al.) · `PUBLISHED-READ`

**The four probes, verbatim from Fig. 1:** *"**Early Answering**: Truncate the original CoT before
answering. **Adding Mistakes**: Have a language model add a mistake somewhere in the original CoT and
then regenerate the rest of the CoT. **Paraphrasing**: Reword the beginning of the original CoT and
then regenerate the rest of the CoT. **Filler Tokens**: Replace the CoT with ellipses."*

**The metric, verbatim (§2.3.1):** *"we … calculate an **area over the curve (AOC)** metric for all
CoT lengths of each task … **AOC values are calculated as a weighted sum, where the AOC for each CoT
length is weighted by the fraction of CoT samples having that length.**"* And its reading direction
(Table 2 caption): *"early answering AOC, a measure of post-hoc reasoning (**higher is less post-hoc,
indicating greater faithfulness**)."*

**Numbers (Table 2, read in PDF, n = 8 tasks, 175 B RLHF model)** — AOC (early answering / adding
mistakes): AQuA 0.44/0.52 · LogiQA 0.26/0.31 · MMLU 0.12/0.21 · HellaSwag 0.12/0.23 · TruthfulQA
0.11/0.20 · OpenBookQA 0.07/0.15 · ARC-Challenge 0.05/0.11 · ARC-Easy 0.02/0.07.
⭐ Two structural facts we should not lose: **(i)** on the three lowest-AOC tasks *"the chain of
thought changes the final answer less than 10 % of the time"*, i.e. **influence is usually near
zero unless the task forces it**; **(ii)** AOC shows *"little correlation with the performance gain
from chain of thought"* — LogiQA is 2nd-most-faithful with a **negligible** accuracy gain, HellaSwag
is faithful with a **−4.69 %** accuracy change. ⇒ *"**CoT may be faithful even when it does not
improve task performance.**"* And, from the abstract: *"As models become larger and more capable,
they produce less faithful reasoning on most tasks we study."*

**What it changes in our design if true:**
1. ⛔⛔ **H-TL-1 and H-TL-2/3 are INDEPENDENT and must be reported independently.** Lanham measures
   that faithfulness and performance-gain are uncorrelated. Our plan currently reads as if a
   selection improvement (H-TL-1) would support the influence claim — **it would not**, and a
   TanitLang that *improves selection* while being a rationaliser is exactly what the null predicts.
2. ⭐ **AOC is directly portable to our fan.** Our analogue of "truncate the CoT" is **truncate the
   reasoner's refinement**: REF-C runs `K = 16` latent tokens × 2 edit steps; sweep the number of
   thought tokens/edits admitted `k = 0 … K`, measure how often the **selected anchor equals the
   full-`K` selection**, and integrate. **Weighted the same way Lanham weights**: by the fraction of
   windows at each effective length. That gives one scalar per arm, comparable across arms.
3. **"Filler tokens" is the compute control we are missing.** Replace the thought tokens with an
   equal number of *uninformative* tokens. If selection improves anyway, the gain is **budget, not
   reasoning** — and on a 300–500 ms tact that is a decision-grade distinction.

### ⭐ `2606.12706` — VLADriveBench: Evaluating CoT-Action Relationship in VLA for Autonomous Driving · `PUBLISHED-READ`
**The single most on-point paper in this pass.** Two dimensions: **quality** (mentioning, accuracy,
hallucination, contradiction) and **relationship** = **observational alignment** + **causal
intervention**. Verbatim on why both: *"observational metrics reveal the CoT's quality and its
correlation with the action but not how it influences it; intervention establishes causation but has
limited sensitivity, where a strong vision signal may mask genuine CoT's contributions."*

**`selfsplice` — the control that must read a known value (verbatim, §3.6):** *"the model's own CoT
is decoded, re-encoded through the injection pipeline, and used as the substitute. **Selfsplice
produces exactly zero difference across all 24 experiments on both routes and all three models**,
confirming the pipeline is artifact-free."* Appendix G.1 makes it concrete: Alpamayo KV caches
**bit-identical** over 7 + 4 runs, trajectory L2 **exactly 0.0** over 12 + 10 runs; ORION's input
vectors bit-identical, trajectory difference **< 0.002 m at 3 s** (attributed to GPU float
non-determinism).

**Alignment metric, verbatim (§B.4):** *"We report **relaxed alignment**, defined as the CoT set and
the action head set having **at least one action in common**."* Both sides are mapped to **sets** of
plausible actions rather than a single label, deliberately: *"we design the metric to give the model
the benefit of the doubt."* And the aggregation is **spatial, in 1 m bins along cumulative travel
distance**, because *"a vehicle stopped at a red light for 10 s at 10 Hz contributes 100 steps,
while 2 s of cruising contributes only 20."*

**Numbers (read in PDF, Tables 3–4):** Alignment in the 2–4 m/s band — ORION accel **96.5 %**
(n = 1429), slow **85.7 %** (n = 224); Alpamayo R1 accel **28.0 %**, slow **75.8 %**; Alpamayo v1.5
accel **23.4 %**, slow **79.4 %** — with the Alpamayo action head emitting `maintain` on **100 %** of
cells regardless of the CoT command. Intervention, Alpamayo v1.5 open-loop 2 s horizon, Empty Urban,
7 seeds × 38 steps: **selfsplice +0.000 m, d = 0.000, p = 1.00**; accel **+0.083 m, d = +0.600,
p = 1.8e-6**; pedestrian **−0.346 m, d = −0.983, p = 1.3e-9**; car **−0.391 m, d = −1.036,
p = 8.0e-10**. Closed-loop at 5.9 s the car-injection arm travelled **20.1 m less** than baseline,
*"roughly one-third of the total route distance"*, while **selfsplice CIs span zero**.
⚠️ The authors flag their own scope: Alpamayo is evaluated **out of distribution** (CARLA), so
*"the effect magnitudes … should be interpreted as upper bounds"* — directions valid, magnitudes not.

**And the ORION result, which is the warning in §0(a) above:** ORION scores the **highest**
observational alignment and its CoT is **epiphenomenal** — shuffled text reproduces the effect, and
a repeated `"depicts"` token doubles it. *"the CoT's impact on predicted trajectory is not
semantically causal but based on some learned patterns."*

**What it changes in our design if true:**
1. ⭐⭐ **`selfsplice` is the exact shape of our `w = 0` identity gate, and it validates raising that
   gate from "nice" to "the precondition for any delta being quotable".** `DIALOGUE_03` §6 already
   requires *"with guidance off the output must be bit-identical to today's REF-C"*. **This paper is
   the published precedent, and it shows the correct pass criterion: exactly 0.0, not "small".**
2. ⛔ **We must add the semantic-null arms.** Shuffled-prior is already in our control list;
   **the repeated-token arm is not**, and it is the one that caught ORION. Our analogue: a prior
   that is a **constant vector scaled to the same L2 norm** as the reasoned prior, and a prior that
   is the **reasoned prior with its 117 entries permuted**. If either moves selection as much, the
   channel is magnitude/geometry, not reasoning.
3. ⭐ **Report alignment as a SET-overlap with spatial binning**, not as an argmax match. Our
   tactical vocab is 8 lat × 8 lon; forcing one label onto an ambiguous rationale will manufacture
   disagreement. And our own corpus has the same low-speed over-representation problem.
4. **The observational/interventional divergence is a result we should expect, not a bug** — plan to
   report both, and to report the case where they disagree.

### ⭐ `2605.17268` — Is VLA Reasoning Faithful? Probing Safety of Chain-of-Causation · `PUBLISHED-READ`
⭐⭐ **Run on 300 Alpamayo-R1-10B inferences across 100 PhysicalAI-AV scenarios — OUR CORPUS.**

**Definitions, verbatim (§3):**
* *"**Definition 1 (Entity Fidelity).** Let `E(r)` denote the set of entities mentioned in reasoning
  trace `r`, and `E(x,t)` … present in the scene … Entity fidelity is [a Jaccard] where `E_relevant`
  filters ground truth entities by spatial relevance. This Jaccard formulation captures both
  hallucinations (`E(r) \ E`) and misses (`E_relevant \ E(r)`). **We adopt Jaccard over precision or
  recall because safety requires both**: hallucinated obstacles cause unnecessary stops, while
  missed obstacles cause collisions."*
* *"**Definition 2 (Action Fidelity).** Let `a(r)` denote the stated action and `a(τ)` the observed
  trajectory action. Action fidelity is `F_action = 1[a(r) ≈ a(τ)]` where `≈` denotes kine[matic
  compatibility]."*

**Numbers (read in PDF, abstract + §5):** overall reasoning fidelity **42.5 %**; hallucination rate
**8.9 %**; **94 missed pedestrians** in a third of pedestrian-relevant scenes; mean reasoning-action
consistency **0.483** with **53.3 %** low-consistency. **The asymmetry is the important part:**
*"lateral actions achieve perfect consistency (keep lane: 100 %, nudge: 100 %), while longitudinal
control fails (stop: 62.1 %, decelerate: 69.0 %). Of 78 mismatches, 50 (**64.1 %**) involve claimed
deceleration/stop with continuing trajectory."* Perturbation: **97.7 %** of trajectories change under
mild visual perturbation, with **14.3 % silent failures** — *"the model produces different
trajectories but provides identical reasoning"*. Fidelity predicts quality: Pearson
**r = −0.169, p = 0.0045, n = 282** against ADE; lowest-fidelity quartile mean ADE **2.551 m** vs
**1.585 m** in the third quartile.

⭐⭐ **And the one that binds us hardest (§5, Table 1 discussion):** *"The **88 % inconsistency rate**
is significant: **the same scene with only the random seed changed produces different
explanations.**"*

**What it changes in our design if true:**
1. ⛔⛔ **A rationale-consistency number without a same-scene, different-seed replicate is
   uninterpretable.** This is `H-ESTIM-SEED-1` and `D-REFAV1-SEED-GOAL-MISMATCH` arriving from the
   literature, in the rationale channel: **88 % of the "inconsistency" they could have reported was
   the sampler.** Our metric menu below therefore attaches a **seed-replicate floor to every
   consistency metric**, exactly as the programme now requires for CI claims.
2. ⭐⭐ **The lat/lon asymmetry is our defect, restated.** Lateral consistency 100 %, longitudinal
   62–69 %, and **64 % of all mismatches are "I said stop" with a trajectory that continues.** Our
   own longest-standing finding is that **88.7 % of the oracle gap is longitudinal** and that the
   5-way manoeuvre softmax mixes lat+lon (`CLAUDE.md`, INHERITED). **The published VLA rationale
   channel fails in exactly the place our planner fails.** ⇒ a consistency metric that pools lat and
   lon will report ~80 % and hide the whole defect. **Report per-axis, always.**
3. **Jaccard-over-precision-or-recall is the right choice for us too**, and for the stated reason.
4. **"Silent failures" (different action, identical rationale) is a monitorable class we get for
   free** from the seed-replicate arm we now must run anyway.

### ⭐ `2602.11201` — Mechanistic Evidence for Faithfulness Decay (NLDD) · `PUBLISHED-READ`
**Finding:** **Normalized Logit Difference Decay** — corrupt reasoning step `k`, truncate what
follows, and measure the drop in the model's **standardised logit margin** on its answer.
Verbatim on the normalisation, which is the transferable idea: *"Raw sensitivity metrics are often
not comparable across architectures due to differences in output scaling and baseline confidence.
NLDD addresses this by **normalizing the observed logit degradation against the model's intrinsic
output variability**, providing an architecture-agnostic measure of causal reliance. A 50 % NLDD
drop indicates the corrupted chain produces half the standardized logit margin of the clean chain."*
**Reasoning Horizon** `k* = argmax_k NLDD(k)`, found consistently at **70–85 % of chain length**,
*"beyond which reasoning tokens have little or negative effect on the final answer"*.
⭐ It is deliberately paired with **RSA** (representational similarity between clean and corrupted
trajectories) and **TAS**: *"If reasoning truly drives computation, then corrupting it should break
both behavior (NLDD) and internal representations (TAS). If reasoning is just decorative, then
corrupting it might change behavior while leaving internal computation unchanged."*
They condition on correct-answer samples *"to isolate reasoning quality from task failure"*.

**What it changes in our design if true:**
1. ⭐⭐ **Normalise our delta.** `‖S1 − S0‖` in raw logits is not comparable across arms, checkpoints,
   or seeds — different scorers have different logit scales. **Divide by the arm's own selection-logit
   spread** (e.g. the std of `conf0` across the 117 anchors on the same windows). That single change
   makes H-TL-2 comparable across the ladder, and it costs nothing.
2. ⭐ **The behaviour/representation dissociation is a second, cheap discriminator** for the
   epiphenomenal case in §0(a): if the reasoned prior and the shuffled prior move **selection**
   equally but only the reasoned one moves the **`cond` bus representation**, we can say which
   channel carries content.
3. **The reasoning horizon predicts that our 2-step edit budget may already be past the useful
   point or well short of it — and `argmax_k` finds out for ~2 extra forward passes.**

### `2305.04388` Turpin et al. · `2503.08679` CoT in the wild is not always faithful · `2505.05410` Reasoning models don't always say what they think · `2405.15092` · `2402.13950` · `2603.22816` · `2512.21711` (Do Latent Tokens Think?) — `PUBLISHED-ABSTRACT`
The load-bearing one for us is **`2512.21711`** (causal + adversarial analysis of *continuous* thought
tokens) because **our CoT is latent**, and latent CoT has no text to inspect — every probe above
becomes an intervention on a vector. ⛔ **Read it before WP-2 is specified.** `2305.04388` supplies
the classic result that biasing cues invisible in the CoT shift answers (VLADriveBench cites it as
**up to 36 %**, `PUBLISHED-SECONDARY` — I read that figure in VLADriveBench, not in Turpin).

### ⭐ `2506.19143` — Thought Anchors: Which LLM Reasoning Steps Matter? · `PUBLISHED-READ`
**Finding:** a black-box **counterfactual importance** per reasoning step: for step `S_i`, generate
**100 rollouts without `S_i`** and 100 with it, and quantify **the difference in the distribution over
final answers**; separately, mask attention to `S_i` and measure the **KL divergence** on subsequent
token logits. Their framing is worth quoting because it sidesteps our whole tension: these measures
*"provid[e] a principled foundation for interpretability that sidesteps disputes about the
'faithfulness' of CoT text."*
**What it changes:** ⭐ **the distributional form is the right one for us.** Our final output is a
**distribution over 117 anchors**, so "the difference in the distribution over final answers" is
literally a **KL between two anchor posteriors** — no correspondence problem, no answer-matching
heuristic. This is the cleanest published justification for the metric in §5 below.

---

## 3. Topic 3 — Energy-based / consistency-regularised joint models

### ⭐ `2507.02092` — Energy-Based Transformers are Scalable Learners and Thinkers · `PUBLISHED-READ`
**Finding (abstract, read in PDF):** *"learning to explicitly **verify the compatibility between
inputs and candidate-predictions**, and then **re-framing prediction problems as optimization with
respect to this verifier** … assign an energy (unnormalized probability) value to every input and
candidate-prediction pair, enabling predictions through gradient descent-based energy minimization."*
Its Facet 3 argument: *"verifying solutions is exponentially easier than generating solutions, which
means that verifiers can often generalize better than explicit generators"*; and its explicit
contrast with diffusion (Fig. 1c): *"diffusion models cannot give unnormalized likelihood estimates
at each step of the thinking process, and are not trained as explicit verifiers."*
Numbers (abstract): up to **35 %** higher scaling rate; **29 %** more improvement from extra
inference compute than Transformer++ on language.

**What it changes in our design if true:**
1. ⭐⭐ **The most important structural read of the pass: REF-C already IS an energy model over the
   fan.** `sel_score` assigns a scalar to each of 117 candidates given `cond`; that is `E(x, y)` with
   `y` ranging over a *discrete* candidate set instead of a continuous one. **We do not need to add
   an EBM — we need to extend the argument list.**
2. ⭐ **The concrete extension:** score the **pair** `E(cond, r, a)` where `r` is the rationale and
   `a` the anchor, and define consistency as `argmin_a E(cond, r, a) == a_selected`. Then
   *"consistent"* stops being a post-hoc comparison between two artifacts and becomes **a property of
   one score** — which is exactly what dissolves the "consistent-but-inert vs influential-but-
   confabulating" dichotomy: an energy that is low only for *compatible* (r, a) pairs cannot be
   satisfied by an echo (it would also be low for the runner-up) or by a confabulation.
3. ⚠️ **But EBT's honest warning cuts at us too:** its criticism of diffusion — *"typically fail to
   benefit from denoising steps beyond what they were trained on"* — is **exactly our MEASURED
   refcv3 result** (`INHERITED`, `DIALOGUE_03` §5: sel-ADE 0.654 → 1.529 m at steps 8, separated).
   ⇒ **EBT is an argument for putting the extra thinking in an explicit SCORER, not in more
   denoising steps** — which is the same conclusion `DIALOGUE_03` reached for a different reason, and
   is now doubly supported.

### `2109.00137` — Implicit Behavioral Cloning · `PUBLISHED-ABSTRACT`
Energy-based **implicit** policies scoring `(obs, action)` pairs beat explicit regression (MSE, MDN)
*"particularly with respect to approximating complex, potentially discontinuous and multi-valued
(set-valued) functions."*
**Changes for us:** this is the theoretical case for **why an anchored fan + scorer beats a
regression head** at a decision point where two valid manoeuvres exist (go left / go right around an
obstacle) — the case where an averaged regression produces the one trajectory that hits it.
⭐ It is also the argument that **the scorer must stay the committer** (`DIALOGUE_03` keeps it
unchanged) rather than TanitLang overriding it.

### `1912.03263` JEM · `2406.08862` Cognitively Inspired Energy-Based World Models · `2303.01469` Consistency Models · `2602.03604` EB-JEPA library — `PUBLISHED-ABSTRACT`
`1912.03263` reinterprets a classifier's logits as a **joint energy over (x, y)** at no architectural
cost. **Changes for us:** ⭐ **the cheapest possible route to the pair-energy in point (2) above** —
REF-C's selection logits may already be readable as `−E(cond, anchor)`, making
`E(cond, r, anchor)` a re-normalisation rather than a new model. ⛔ Read the PDF before this is
planned on.

---

## 4. Topic 4 — Classifier-free guidance beyond images

### ⭐⭐ `2408.09000` — Classifier-Free Guidance is a Predictor-Corrector · `PUBLISHED-READ`
**Finding, verbatim (abstract):** *"we first **disprove common misconceptions**, by showing that CFG
interacts differently with DDPM and DDIM, and **neither sampler with CFG generates the gamma-powered
distribution** `p(x|c)^γ p(x)^{1−γ}`. Then, we clarify the behavior of CFG by showing that it is a
kind of **predictor-corrector** method that alternates between **denoising and sharpening** … in the
SDE limit, CFG is actually equivalent to combining a DDIM predictor for the conditional distribution
together with a Langevin dynamics corrector for a gamma-powered distribution."* The correspondence is
`γ = (2w − 1)`. Their counterexamples show the CFG-DDIM and CFG-DDPM distributions are **not even
scaled versions** of each other or of the gamma-powered target — one is *"mean-shifted and not even
Gaussian"*.

**What it changes in our design if true:**
1. ⛔⛔ **`DIALOGUE_03` §3 needs one qualification.** It says *"the CoT's causal contribution is a
   number CFG computes anyway."* True **on the selection surface**, where `S1 − S0` is an exact
   log-space difference between two quantities we compute and control. **False as a claim about the
   trajectory distribution** — this paper proves the guided sampler's output distribution is not a
   simple function of the two passes, and depends on the sampler. ⇒ **State the delta as a statistic
   of the selection logits (which it is), never as "the amount of conditioning in the plan".**
2. ⭐ **Where to measure it, concretely, from source.** `stack/tanitad/refs/refc.py` ~L2044–2062
   already builds the guided score in two named pieces: `conf0` (the classifier surface, priors
   absent) and `conf, tele = self._apply_grafts(conf0, terms, …)` where `terms` is the additive
   list of anchor priors. **`Δ = conf − conf0 ∈ ℝ^117` is the guidance delta, per anchor, already
   materialised.** The source even anticipates this at L1362–1363: *"ABLATABLE — `lon_to_anchor` can
   be zeroed and the delta measured, which the single concatenated graft never allowed."*
   ⇒ **We do not need to build the F0/F1 dual pass to get the anchor-space delta. It is one
   subtraction on tensors that already exist.** (A dual pass is still needed for the `cond`-bus side,
   `lan_to_cond`, which is not additive on the selection surface.)
3. **`w` schedule warning:** large `w` moves samples off the data manifold. Our `w` starts at 0 and
   every increase is a pre-registered arm — keep that.

### `2211.15657` — Is Conditional Generative Modeling all you need for Decision-Making? (Decision Diffuser) · `PUBLISHED-ABSTRACT`
Return-conditional diffusion policies **outperform offline RL** on standard benchmarks, with
constraints and skills composed as conditions. **Changes for us:** the published precedent that
**CFG-style conditioning is a competent decision-making mechanism, not only an image trick**, and
that **conditions compose** — which is what `terms` does.

### `2207.12598` Ho & Salimans (the original) · `2205.09991` Diffuser · `2503.00535` What Makes a Good Diffusion Planner (6,000+ models trained) · `2608.19504` Plug-in Interpretation of Conditioning — `PUBLISHED-ABSTRACT`
⭐ **`2608.19504` is the one to read next in this topic**: it learns an **unconditional** score and
enforces conditioning at inference via a **plug-in correction term**, so *"the plug-in term separates
the conditioning contribution from the learned unconditional dynamics"*. **If that holds, the
conditioning contribution is an explicitly isolated object rather than a difference of two passes** —
i.e. a principled version of exactly what we want to measure. ⛔ abstract-level only; read the PDF
before it enters a plan.

### `2501.15564` Diffusion Planner (flexible guidance) · `2511.18729` GuideFlow (constraint-guided flow matching) · `2503.05689` GoalFlow — banked previously, `PUBLISHED-ABSTRACT`
The driving-side instances of guided generation. **GuideFlow matters for us specifically** because
`DIALOGUE_03` §2.4 mechanism 2 is *"constraints as negative priors"*, and constraint-guided flow
matching is the published form of that idea.

---

## 5. Topic 5 — Rationale ↔ control self-consistency in embodied/driving agents

**Does anyone MEASURE rationale↔action agreement rather than asserting it? Yes — four groups, all in
the last ~9 months, and their answers disagree with each other in an informative way.**

| paper | what is measured | headline |
|---|---|---|
| `2606.12706` VLADriveBench `PUBLISHED-READ` | relaxed set-overlap alignment **+** injection intervention **+** selfsplice | alignment and causality **diverge**; a high-alignment CoT can be epiphenomenal |
| `2605.17268` Is VLA Reasoning Faithful `PUBLISHED-READ` | entity fidelity (Jaccard), action fidelity, reasoning-action consistency, perturbation sensitivity | **0.483** mean consistency on **PhysicalAI-AV**; lat 100 % / lon 62–69 % |
| `2608.29583` Drive the Thoughts `PUBLISHED-READ` | CoT↔trajectory consistency as a **runtime monitoring** problem, judged | **74 %** consistent among reliable CoTs; best monitor **F1 = 0.75** |
| `2603.11219` Senna-2 `PUBLISHED-READ` (abstract in PDF) | dual-system consistency **as a training objective**, scored by F1 | **+19.3 % F1** consistency, **−5.7 % FDE**, **−30.6 % AF-CR** closed-loop |

### ⭐ `2608.29583` — Drive the Thoughts: Runtime Monitoring of VLA Reasoning-Trajectory Consistency · `PUBLISHED-READ`
**Finding:** they curate **DriveAlignBench** — **150 CoT–trajectory pairs** from **NVIDIA Alpamayo
1.5 run in AlpaSim on NuRec reconstructed scenes** — annotated for reliability, consistency, safety.
**33.3 % of CoTs are unreliable**; among the **100 reliable** pairs, **74 CONSISTENT / 26
INCONSISTENT** and **91 SAFE / 9 UNSAFE**. Their monitor family goes rule-based → LLM-judge over raw
ego waypoints → **F-LLM** (same judge, trajectory evidence **enriched with lane-relative geometry**):
**F1 = 0.75, +0.13 absolute over the strongest raw-waypoint LLM baseline and +0.38 over rule-based.**

⭐ **The transferable result is a negative one about scale:** *"The gain from this representation
change is **larger than the gain from switching to a stronger judge model**, suggesting that
**scene-relative evidence is more important than judge scale** for this task."*
And the safety link: *"among reliable CoTs, human-labeled inconsistencies account for **7 of the 9
unsafe trajectories**, and F-LLM monitors detect these 7 cases, unlike the raw-waypoint and
rule-based baselines."*
Their annotation protocol has a control we should copy verbatim: **the trajectory is withheld from
the annotator judging CoT reliability**, *"so that CoT reliability is judged independently of
CoT-trajectory consistency."*

**What it changes in our design if true:**
1. ⭐⭐ **The two-stage gate.** *"A CoT-trajectory consistency check is meaningful only if the CoT is
   itself a valid description of what the vehicle should do."* ⇒ **H-TL-3 must be evaluated only on
   the subset where the rationale is independently reliable**, or we will score confusion.
   Their split says **a third of the corpus fails that gate** — that is the expected size of our
   filtered denominator, and it must be reported (`n` per family, as binding here).
2. ⭐⭐ **The representation lesson maps to a free win for us.** They had to *reconstruct* a
   scene-relative frame to make CoT and trajectory comparable, and that reconstruction was worth more
   than a bigger judge. **We do not have that gap: our anchors are already a scene-relative,
   behaviour-level vocabulary, and both sides speak Δ¹¹⁶.** ⇒ our consistency metric should be
   *cheaper and better-posed than theirs by construction* — and that is a claim we can test against
   their published F1.
3. ⭐ **They ran on AlpaSim + NuRec, which the programme has** (`INHERITED`, auto-memory:
   *"AlpaSim runs bare on an A40 … NuRec is open msgpack + gsplat works on Thor"*). **DriveAlignBench
   is a released benchmark on infrastructure we already operate** — this is the nearest external
   yardstick TanitLang has, and adopting it is an integration item, not a research item.

### `2603.11219` — Senna-2 · `PUBLISHED-READ` (abstract in PDF)
**Finding:** names the **"consistency gap"** between a VLM's high-level decision and the E2E
planner's trajectory, and closes it with a **three-stage consistency-oriented curriculum**: (1)
pre-train with a **decision adapter** passing VLM decisions to the E2E policy **as implicit
embeddings**; (2) **open-loop alignment**; (3) **closed-loop alignment via hierarchical RL in 3DGS**.
Reports **+19.3 % F1 dual-system consistency**, **−5.7 % FDE** open-loop, **−30.6 % AF-CR**
closed-loop. Their Fig. 1 diagnostic is a **relative speed ratio** — planned speed at t = 3 s divided
by initial speed — shown as a distribution per speed decision: *"scattered"* before alignment,
*"distinct and decision-aligned"* after.

**What it changes in our design if true:**
1. ⭐ **The "decision adapter as implicit embedding" is `lan_to_cond`.** Senna-2 is the published
   precedent that our zero-init continuous port is the right shape — and that it needs a
   **staged curriculum** to become non-trivial, which REF-C's recipe already is.
2. ⭐⭐ **The relative-speed-ratio distribution plot is the single cheapest consistency diagnostic in
   this whole document, and it lands on our worst axis.** Bin windows by the reasoner's longitudinal
   token, plot `v(t=3s)/v(0)` per bin: **if the distributions overlap, the longitudinal rationale is
   inert.** No judge, no LLM, no annotation. **Put it in WP-2.**
3. ⚠️ Consistency is measured **by F1 against a label**, i.e. it needs a decision ground truth. We
   have one (`situations.py` / the released tactical vocabularies) — but see §0(c): **that label must
   not be back-derived from the same future trajectory the consistency is scored against.**

### `2607.03182` AnchorVLA · `2512.10226` LCDrive — `PUBLISHED-READ` (abstract + results)
⚠️ **These two are why the novelty section below is narrower than `DIALOGUE_03` implies.**
* **AnchorVLA** already uses **trajectory-pattern anchors as the explicit interface between VLA
  reasoning and continuous execution** — "Decision-as-Anchor Representation" + "Decision-Anchored
  Residual Flow", Bench2Drive **SR 77.28 / DS 89.92**. That is our §2.2 anchor-simplex coupling,
  published. ⛔ But its ablations are architectural (anchor modelling, flow formulation, navigation
  modality) — **it measures no faithfulness, influence, or consistency quantity at all.**
* **LCDrive** (`2512.10226`) *"keeps the CoT in the same vocabulary as the final trajectory output"*,
  interleaving **action-proposal tokens** with **latent world-model tokens**, on **PhysicalAI-AV**
  (39,072 train / 23,758 val clips, 1.6 s history, 6.4 s future at 10 Hz). Results: GT-LWM oracle
  ADE **1.393 → 1.268** with latent CoT, **1.197** with RL, Coll5.0 **0.905 → 0.867**; practical
  LCDrive ADE **1.626** vs non-reasoning **1.762**, OffRoad2.5 **1.219** vs **1.753**, Coll5
  **0.836** vs **2.207**. And: *"RL is beneficial only when the model conducts reasoning."*
  ⛔ **It also measures no influence or consistency quantity** — every reported number is ADE,
  collision, or off-road. **And it cold-starts its CoT from ground-truth future rollouts**, which is
  the anchoring bias of §0(c).

⇒ **Honest statement of position: the INTERFACE is prior art; the MEASUREMENT is not.**

---

## 6. ⭐ METRIC MENU — computable on REF-C's fan, each with the control that must fire

All of these are defined over the 117-anchor selection surface. Notation from
`stack/tanitad/refs/refc.py` (read at source this session): `conf0 ∈ ℝ^117` = classifier surface
**before** priors; `terms` = the additive prior list; `conf = _apply_grafts(conf0, terms, …)`;
`p1 = softmax(conf)`, `p0 = softmax(conf0)`; `a* = argmax` of the committed surface.
⛔ **Every metric below is reported per lat/lon axis separately** — §2's `2605.17268` result
(lat 100 % / lon 62–69 %) says a pooled number hides the entire defect.

### (i) CoT → ACTION INFLUENCE

| # | metric | definition | ⛔ the control that MUST accompany it, and what it must read |
|---|---|---|---|
| **I-1** | **Normalised guidance delta** `Δ̂` | `‖conf − conf0‖₂ / std_a(conf0)`, per window, then the episode-cluster bootstrap over the 40 val episodes | **`w = 0` / selfsplice identity**: with every language port gated off, `conf − conf0` must be **exactly 0.0** and the emitted trajectory **bit-identical** to today's REF-C (`2606.12706` reads exactly 0.0; ORION's < 0.002 m is the float-noise ceiling). Normalisation per `2602.11201`. |
| **I-2** | **Selection-flip rate** | fraction of windows where `argmax p1 ≠ argmax p0` | **Constant-prior floor**: a prior that is a constant vector **scaled to the same L2 norm** as the reasoned prior must read the *chance* flip rate. If it flips as often, the channel is magnitude. |
| **I-3** | ⭐⭐ **Semantic-null contrast** | `Δ̂(reasoned) − Δ̂(permuted)` where *permuted* = the same 117 prior values in a random order, and separately *shuffled-vocab* = the lat/lon token logits permuted before `lat_to_anchor` | ⛔ **This IS the control, promoted to a metric.** `2606.12706` measured a shuffled ORION CoT at **81.4 %** of steps vs **81.9 %** coherent, and a repeated-token prior at **90.4 %** — *stronger than the real one*. **If `Δ̂(permuted) ≈ Δ̂(reasoned)`, H-TL-2 FAILS regardless of how large the delta is.** |
| **I-4** | **Truncation AOC** (Lanham-form) | admit `k = 0 … K` thought tokens / edit steps; measure `P(argmax_k = argmax_K)`; **area over the curve, weighted by the fraction of windows at each effective length** | **Filler-thought arm**: `k` *uninformative* thought tokens (zeros, or a fixed learned constant) at each budget. If the curve is the same, the gain is **compute, not reasoning**. |
| **I-5** | **Reasoning horizon** `k*` | `argmax_k` of the per-step influence in I-4 | Reported with its `n`; `2602.11201` finds `k*` at **70–85 %** of chain length in LLMs — a value near `k*=0` or `k*=K` on our 2-step budget is itself the finding. |
| **I-6** | **Port attribution** | ablate one `*_to_anchor` port at a time (`maneuver` / `lat` / `lon` / `lan_gate`) and report `Δ̂` per port | The source already supports this (`refc.py` L1362–1363). Ports must sum: `Σ Δ̂_port ≈ Δ̂_all` within bootstrap CI, else the seam is non-additive and attribution is invalid. |
| **I-7** | **Prior non-degeneracy screen** (PAV) | `Var_a[prior]` over the 117 anchors, + Spearman ρ between the reasoned prior and `conf0` | **Zero-cost pre-eval gate.** `Var ≈ 0` ⇒ inert by construction (`2410.08146`). `ρ ≈ 1` ⇒ the reasoner is the trunk (the `μ = π` collapse) ⇒ also inert. **Run this before spending a single eval GPU-hour.** |

### (ii) CoT ↔ ACTION CONSISTENCY

| # | metric | definition | ⛔ the control that MUST accompany it, and what it must read |
|---|---|---|---|
| **C-1** | ⭐ **Anchor-posterior KL** | `KL(p_reason ‖ p_plan)` where `p_reason` = the reasoner's prior mapped through `*_to_anchor` and softmaxed, `p_plan` = the committed selection posterior. **One number, no correspondence problem** (this is why the anchor simplex is the right space — cf. `2506.19143`, whose "difference in the distribution over final answers" is literally this) | **Rationale-swap floor**: recompute against the *runner-up window's* rationale. That is the no-information value; C-1 must be **materially lower** than it, with a paired episode-cluster bootstrap. |
| **C-2** | ⭐⭐ **Consistency vs `F0`, not `F1`** | evaluate C-1 against the **unconditioned** posterior `p0` as well as `p1`. **Consistent with `p1` but NOT with `p0` = post-hoc rationalisation** (`DIALOGUE_03` §3, and the operational form of Lanham's post-hoc test) | **This is the H-TL-3 discriminator.** ⛔ Report both or report neither. |
| **C-3** | ⭐⭐ **Seed-replicate consistency floor** | same window, same checkpoint, **inference seed varied**; C-1 between the two runs' rationales | ⛔⛔ **BINDING.** `2605.17268` measured an **88 % inconsistency rate from the seed alone**. Any C-1 not clearly separated from C-3 **is the sampler**, not the model. Same family as `H-ESTIM-SEED-1` and `D-REFAV1-SEED-GOAL-MISMATCH`. |
| **C-4** | **Relaxed set-alignment, spatially binned** | map the rationale's lat/lon tokens to a **set** of plausible manoeuvres, map the executed trajectory to a set, score **overlap ≥ 1**; aggregate in **1 m bins of cumulative travel distance**, not per-timestep (`2606.12706` §B.4) | **Argmax-match variant reported beside it.** If relaxed ≫ strict, the rationale is ambiguous rather than aligned — say so instead of banking the higher number. |
| **C-5** | ⭐ **Relative-speed-ratio separation** (Senna-2) | per longitudinal token, the distribution of `v(t=3s)/v(0)`; report the separation (e.g. pairwise AUC or Wasserstein) between `accelerate` / `maintain` / `decelerate` bins | **Shuffled-token arm** must read chance separation. **Cheapest consistency instrument in this document, no judge required, and it lands on the 88.7 %-longitudinal defect.** |
| **C-6** | **Reliability gate** (two-stage) | compute C-1..C-5 **only on windows where the rationale is independently reliable** (grounded, non-hallucinated) | ⛔ `2608.29583`: *"a consistency check is meaningful only if the CoT is itself a valid description"*; **33.3 %** of their CoTs failed this. **Report the gated `n` and the gate rate per family** — a family that cannot be computed is reported with its reason and its `n`, not dropped. |
| **C-7** | **Silent-failure rate** | fraction of window pairs where the **plan differs materially** but the **rationale is identical** | `2605.17268` measured **14.3 %**. Falls out of C-3 for free and is the class a runtime monitor is blind to. |
| **C-8** | **Entity fidelity** (if/when a rationale mentions objects) | **Jaccard** between mentioned entities and spatially-relevant ground-truth entities from `obstacle.offline` (3D agent tracks on **97.44 %** of the corpus, `INHERITED`) | Jaccard **not** precision or recall, for the stated safety reason (`2605.17268` Def. 1). ⚠️ Labels may use privileged channels; **inference may not** (binding). |

### ⛔ Two menu-wide rules

* **Every metric carries its estimator and its tier.** Paired **episode-cluster bootstrap** over the
  40 val episodes (`taniteval/ci.py`); never `overlapping_holdout_se`. Tier stamp per
  `EVAL_DOCTRINE.md` — an influence measured under teacher forcing is **T0** and is not a driving
  claim.
* ⛔ **A separated CI on a one-seed arm is necessary, not sufficient** (`H-ESTIM-SEED-1`, MEASURED
  here: `A0b_replicate` cleared "separated" on 3/18 families with **zero levers moved**). Every
  lever claim in this menu needs a **replicate arm**; C-3 is that arm for the inference channel and
  it must exist for the training channel too.

---

## 7. ⛔ WHAT NOBODY HAS DONE — where we would be genuinely first

Stated conservatively, after reading the two papers that came closest (`2607.03182` AnchorVLA,
`2512.10226` LCDrive) specifically to find out where we are *not* first.

**Not novel — do not claim these:**
* ⛔ **Anchors as the reasoning↔action interface.** AnchorVLA's DAAR is exactly this, published, with
  Bench2Drive numbers.
* ⛔ **CoT in the same vocabulary as the action output.** LCDrive states it verbatim, on our corpus.
* ⛔ **Measuring CoT→action causality in a driving VLA.** VLADriveBench and `2605.17268` both do it.
* ⛔ **CoT↔trajectory consistency as a checkable quantity.** Drive the Thoughts and Senna-2 both do it.

**Genuinely open, on this pass's evidence:**

1. ⭐⭐ **Influence and consistency measured in the SAME space, on the SAME object, as two readings of
   one distribution.** Every measurement above bridges a representation gap: VLADriveBench maps text
   → action-label sets; Drive the Thoughts had to build a lane-relative frame and found that
   representation change worth **more than a bigger judge**; `2605.17268` uses kinematic predicates.
   **On REF-C both sides are already distributions over the same 117 anchors**, so influence is a
   *difference of two posteriors* and consistency a *KL between two posteriors* — **no judge, no
   parser, no annotation, no LLM in the measurement loop.** Nobody has reported this because nobody
   measuring had an anchor-native planner and nobody with an anchor-native planner measured.

2. ⭐⭐ **The guidance delta as a per-anchor, normalised influence statistic — read off machinery the
   planner already computes.** `2408.09000` gives CFG's delta a rigorous (and restrictive) meaning
   for *sampling*; nobody has used `S1 − S0` as a **published influence metric for a reasoning
   channel**. And we get it for one tensor subtraction (`conf − conf0`), not a second forward pass.

3. ⭐⭐ **A published influence measurement with the semantic-null arm attached as standard.**
   VLADriveBench *found* the shuffled-text and repeated-token result and reported it as a diagnosis
   of ORION. Nobody has yet **pre-registered** it as the admissibility condition every influence
   number must clear. Given §0(a), **any influence result published without it is uninterpretable** —
   and saying so, with the arm run on our own model, is a contribution in its own right.

4. ⭐ **A rationale-consistency number with an inference-seed replicate floor beside it.**
   `2605.17268` measured an **88 % seed-driven inconsistency rate** and reported it as a property of
   the model; nobody has used it as the **noise floor that a consistency claim must clear**. Our
   programme has already been forced to this conclusion twice on other channels
   (`H-ESTIM-SEED-1`, `D-REFAV1-SEED-GOAL-MISMATCH`) — **exporting that discipline to the rationale
   channel is a genuinely transferable methodological contribution, not just local hygiene.**

5. ⭐ **Per-axis (lat vs lon) faithfulness.** `2605.17268` reports the split (100 % vs 62–69 %) but
   treats it as an observation. Nobody has designed the reasoning channel *around* it. Our largest
   known defect is longitudinal, and a factored lat/lon prior surface (`lat_to_anchor`,
   `lon_to_anchor` — separate, ablatable, `lon` zero-init) is **already built**. A faithfulness study
   that is factored the same way is unclaimed ground.

6. ⭐ **The trade-off curve rather than a point.** `2606.12706` concludes *"neither design satisfies
   both desiderata"* from **two models at two fixed coupling strengths**. We have a **continuous
   dial** (`w`, `lan_gate`) on one architecture. **Sweeping it and publishing the
   influence-vs-safety-degradation frontier is the experiment their conclusion asks for and their
   setup cannot run.**

⚠️ **And the honest counterweight:** items 1–6 are *measurement* contributions. **None of them is a
driving result.** Under RULE ZERO the deliverable is not "we measured the tension well" — it is
"TanitLang improved selection **and** we showed the improvement was not an echo." The metric menu
exists to make the second half provable; **it does not substitute for the first half.**

---

## 8. ⚠️ Gaps, risks, and what to read next

* ⛔ **9 of the papers whose design consequences are largest are `PUBLISHED-ABSTRACT` only.** In
  priority order for a PDF read before they enter any plan: **`2412.01981`** (Implicit PRM — would
  remove the CoT-label data blocker), **`2512.21711`** (Do Latent Tokens Think? — our CoT *is*
  latent, so every probe above becomes a vector intervention), **`2608.19504`** (plug-in conditioning
  — an isolated conditioning term rather than a two-pass difference), **`1912.03263`** (JEM — the
  cheap route to a pair-energy), `2310.17022`, `2305.20050`.
* ⚠️ **No question corpus, unchanged.** `DIALOGUE_03` §7's data gap is not closed by anything in this
  pass. But **`2412.01981` and `2410.08146` between them describe a path that needs no step labels
  at all** (implicit PRM from an outcome-trained scorer + advantage under a complementary prover),
  and **`2606.24231` FlowR2A** shows the reward side can come from a **simulator** rather than
  annotation. That is the most promising line and it is currently unexplored here.
* ⚠️ **VLADriveBench's magnitudes are upper bounds** (their own §5.1: OOD CARLA evaluation of
  Alpamayo). Directions are valid; do not quote the effect sizes as calibrated.
* ⚠️ **`2606.12706`, `2608.29583`, `2607.03182`, `2608.09591`, `2606.24231`, `2607.00399`,
  `2606.08525`, `2602.11201`, `2603.22816`, `2605.17268` are 2026 preprints**, several unrefereed.
  Treated as PUBLISHED (banked) but not as settled.
* 🔴 **ESCALATION (library integrity, not my stream):** `kb_add.py --verify` reports
  **487 entries, 0 orphans, 1 problem** — `2603.21546` *"What Do World Models Learn in RL?"*:
  **"sha256 MATCHES but the file is not a complete PDF — unreadable (OSError)"**. It is cited by
  `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-sota-decodability/RESULT.md`.
  **The sha256 matching means it was banked truncated and has been "verified" ever since** — the
  hash cannot catch a defect present at ingest. **All 40 of this pass's additions verify clean.**
  This needs a re-bank by whoever owns the decodability stream, and arguably a `--verify` that
  parses the PDF header/EOF at **ingest** time.

---

## 9. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| **This report** | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vla-design-dialogue/LIT_COT_INFLUENCE_CONSISTENCY.md` — **staged, not committed** | no (also scratchpad) |
| **40 banked PDFs** | `repo:TanitAD Research Lab/Library/papers/*.pdf` — **staged** | no (arXiv) |
| **Library index** | `repo:TanitAD Research Lab/Library/library.json` (447 → **487** entries) + generated `LIBRARY.md` — **staged**, `--verify` run | no |
| Extracted PDF text (working files) | scratchpad `…/scratchpad/txt/*.txt` | yes — **deliberately not staged**, regenerable with `pdftotext` from the banked PDFs |

⛔ **Not committed, not pushed, no branch switched**, per the operating standard.
