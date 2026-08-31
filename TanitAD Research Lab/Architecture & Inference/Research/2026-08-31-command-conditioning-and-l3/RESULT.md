<title>RESULT — P2(b) is aimed at the channel; the literature says the binding constraint is the target</title>

# RESULT — E-LAB-ARCH-0831

`TanitAD Research Lab · Architecture & Inference · 2026-08-31`
`Seeds: V7_LAUNCH_GATE.md P2(b) · P5/L3 · P4. Literature pass, 0 GPU, 0 spend, Thor untouched.`
`Every verbatim passage lives in raw/QUOTES.md with its retrieval method. Nothing is quoted here that is not there.`

---

## 0. ⭐⭐⭐ THE ANSWER

**P2(b) asks about the CHANNEL. The literature's answer is that the binding constraint is
the TARGET — and that constraint would still bind with a perfect CAN command.**

`UWM-JEPA` (`2605.25313`, §4.2, full text) states the mechanism in one sentence:

> *"Under a teacher-forced JEPA objective the target encoder observes the trajectory that
> already contains the action's effect, which admits an action-invariant solution."*

and, in the same section, *"the target encoder has already consumed the observed future and
the predictor can match it without using the action."* They **measure** the consequence:
the action term's magnitude collapses to **‖H₁‖/‖H₀‖ ≈ 0.03** under teacher forcing, and
**recovers to 1.00 ± 0.15** once the target is built from a counterfactual action sequence
instead of the realised one.

Three facts make that transfer to us rather than merely rhyme with us:

1. ⭐ **Our objective is teacher-forced in exactly their sense — MEASURED in our source, not
   assumed.** `train_v6_staged.py:48` describes the family as *"prediction, each against its
   layer's stop-grad/EMA target"*; `:1765` speaks of *"O5's teacher side"*; `:4214`/`:4221`
   give `--o5-target {live, ema, frozen}` — **all three build the target by encoding the real
   observed future.** (raw/QUOTES.md Q8.)
2. ⭐ **Our measured number is the same defect, 5–7× deeper.** MM-E10, three 30k arms:
   h=1 action/scene ratio **0.00408 / 0.00416 / 0.00595** against their 0.03.
3. ⛔ **The mechanism is INDEPENDENT of where the action came from.** Whatever `a_t` is —
   curvature proxy, CAN steering-wheel angle, pedal position — the teacher-forced target
   already contains its effect, so the action-invariant solution remains available. **A
   command channel does not remove it.**

⇒ ⛔ **THE GATE'S CANDIDATE TABLE FOR P2 IS INCOMPLETE.** It lists (a) horizon, (b) channel
provenance, (c) action representation / latent geometry. **(d) TARGET CONSTRUCTION is a
fourth candidate, it is not a restatement of (c)** — (c) says *change how the action is
encoded*, (d) says *change what the prediction is scored against* — **and it is the only one
of the four with a published mechanism, a published magnitude, and a published fix.**

⭐ **It also retro-explains MM-E11**, which is currently an unexplained negative: restoring
O1 made the ratio **fall 0.40×** against a criterion demanding a 10× rise. Under (d) that is
the expected sign — O1 is another term scored against the realised future, so it adds
gradient to a loss the action-invariant solution already minimises.

⚠️ **And the honest limit, stated in their own words:** the fix as published needs
*"simulator-state access during training"* (§4.3). **We have no in-loop simulator.** The
transfer is therefore not free, and §9 gives the simulator-free variant and its cost.

⚠️ **Read §8 before acting on this.** One frontier driving model — `2606.12987` — *does*
condition on a CAN command and *is* action-controllable (ρ = 0.81 vs −0.18). It changes four
other things at once and is not a teacher-forced latent-regression model, so it does not
refute the mechanism above; but it is a real existence proof and the headline must be read
with it, not without it.

⛔ **CLOSURE RULE OBSERVED.** Per `V7_LAUNCH_GATE.md`: *"no item below may be marked closed
by me."* Nothing here closes P2. This package reports that **a criterion is not yet met and
that the candidate list is missing a member** — the ranking and the spend are the PI's call.

---

## 1. F1 ⭐⭐⭐ — Teacher-forced targets admit an action-invariant solution `[PUBLISHED + MEASURED + INFERRED]`

| layer | claim | evidence class |
|---|---|---|
| theirs | teacher-forced JEPA target admits an action-invariant solution; ‖H₁‖/‖H₀‖ ≈ **0.03** | **PUBLISHED** — `2605.25313` §4.2, full text, verbatim in Q1 |
| theirs | counterfactual-action targets restore ‖H₁‖/‖H₀‖ = **1.00 ± 0.15** across seeds | **PUBLISHED** — §4.3, Q1 |
| theirs | the fix requires *"simulator-state access during training"* | **PUBLISHED** — §4.3, Q1 |
| ours | our O5 target is built by encoding the real observed future under a stop-grad/EMA teacher | **MEASURED** — `train_v6_staged.py:48/:1765/:4214/:4221`, Q8 |
| ours | h=1 action/scene ratio **0.00408 / 0.00416 / 0.00595** on `o14fut30k` / `emao14_30k` / `postrain30k` | **MEASURED** — MM-E10, `GOALS_AND_CLAIMS.md`; both controls read their known values (C0 exactly 0.0, C1 0.18–0.43) |
| the join | ⇒ our action-deafness is an instance of the published failure mode | ⛔ **INFERRED.** Their arms are not ours; the ratios are not the same statistic on the same quantity. What is shared is the **objective structure**, and that is what the argument rests on. |

⚠️ **What would falsify the transfer.** If an arm trained with a target that does NOT
contain the action's effect still reads ratio ≈ 0.005, (d) is refuted for us. That is the
one-variable experiment named in §5.

⛔ **What I am NOT claiming.** I am not claiming (b) is wrong about the corpus — E-DEC-57 is
confirmed at the source and our channel really is a kinematic restatement. I am claiming
that **fixing the channel is not predicted to fix the deafness**, because a second mechanism
is sufficient on its own to produce it. Two causes can both be real; only one is load-bearing.

## 2. F2 ⭐⭐ — "The network is not forced to take the commands into account" is a 2018 result with a named fix `[PUBLISHED]`

Codevilla et al., `1710.02410` (ICRA 2018), on the *command input* architecture — the
command concatenated with the image and measurement features:

> *"However, the network is not forced to take the commands into account, which can lead to
> suboptimal performance in practice."*

Their fix is **architectural, not representational**: the *branched* architecture, in which
*"the command acts as a switch that selects which branch is used at any given time."*
**Held-out test town: branched 64 % vs command input 52 % success.**

⭐ **Why this lands on us.** Our conditioning is FiLM/AdaLN modulation — the same family as
concatenation in the respect that matters here: the network *may* use the channel and is
never *forced* to. **MM-E18 measured that our FiLM gain CONVERGED rather than straining**,
which under Codevilla's framing is the predicted equilibrium of an unforced channel, not an
anomaly. ⛔ Note this is a 2018 result: it means the gate's P2 is a **re-discovery**, and the
paper should say so rather than presenting it as new.

⭐ **The transferable mechanism:** in a branched head the un-selected branch receives **no
gradient**, so ignoring the command is not merely penalised — it is unrepresentable. We
already possess a discrete tactical vocabulary that could key such a switch.

⚠️ **Three limits, stated so the transfer is not oversold.** (i) Their command is a **4-way
navigational one-hot**, not a control signal — branching needs a small discrete vocabulary
and does not apply to a continuous 2-D channel. (ii) CARLA 2018, simulation. (iii) 64 vs 52
is one table; I have not verified seed counts at full text.

## 3. F3 ⭐⭐ — Our L3 bar is precedented, it has a name, and the paper that names it also indicts our h=1 predictor `[PUBLISHED, abstract only]`

`2607.27017` (2026-07-29) probes which physical parameters actually enter a latent, on
POKEWORLD plus **RH20T (two robots, 4,258 episodes)**. Four passages matter to P5:

1. ⭐ **The bar has a published name.** *"only the full multimodal objective forecasts force
   **beyond a persistence baseline**"* — a persistence baseline is `z_t` carried forward.
   ⇒ **H-LAB-L3-1 SUPPORTED.** Our L3 (*does `zhat` beat `z_t`?*) is not an invented bar,
   and the paper should cite this rather than claim novelty for the comparison itself.
2. ⭐⭐ **Their protocol is our ladder, arrived at independently.** *"A certificate-gated
   protocol first certifies each parameter as recoverable from raw observations, then
   measures whether it enters the latent, so a null result can be attributed to the objective
   rather than to the environment."* That is **L1/raw-input floor → L2 → L3**, and the stated
   rationale is exactly ours: without the floor, a null is uninterpretable.
3. ⭐⭐ **A mechanism for L3 failure that fits our arms:** *"under single-step prediction a
   vision-only latent discards even perfectly visible object state."* **MM-E14 measured that
   our three-horizon predictor is a one-horizon predictor** — ‖W1‖ 8.8386 vs ‖W2‖ 0.0262 /
   ‖W4‖ 0.0261, the h≥2 heads still at initialisation. ⇒ we are in the single-step,
   vision-only cell their result is about. `[INFERRED transfer]`
4. ⭐ **What the objective decides vs what the input decides:** *"Inputs limit what can be
   known, while prediction targets decide what is retained."*
   Stiffness enters the latent at **R² = 0.50 when touch is forecast**, versus **−0.02 when
   the same signal is merely fused into the input.** ⇒ **a signal present in the input but
   absent from the target does not enter the latent.** That is (d) again, from a second,
   independent paper and a different domain.

⚠️ **ABSTRACT ONLY.** I did not open the full text. Protocol details, probe families and the
definition of "certificate" are **UNVERIFIED**; treat the four points as the paper's own
summary claims, not as a method I have audited.

## 4. F4 ⭐ — ATM is a published, planner-free instrument for the exact P2+P5 pair `[PUBLISHED, abstract only]`

`2606.09028` (2026-06-08): *"ATM compares action information in real encoded transitions and
model-predicted transitions through lightweight post-hoc probes … without simulator rollout"*,
collapsible to a screening score, **>100× faster than CEM-coupled evaluation**. `AITS` then
turns action-identifiability into a **training signal** *"without changing the planner."*

⭐ **This is the same instrument family we built by hand** for MM-E10/MM-E12 and for the L3
probe — which is a mild validation of the design and a strong argument for citing rather than
claiming it.

⛔ **Do not use it to close a gate.** Its own stated condition is *"When the true success gap
is non-trivial, ATM achieves highly reliable pairwise ranking"* — a **screening** instrument
for ranking checkpoints, explicitly not a verdict instrument. Our doctrine's word for that is
T0-DIAGNOSTIC.

## 5. F5 ⭐ — P4: temporal abstraction is licensed, and the failure mode is named in advance `[PUBLISHED]`

**H-LAB-P4-1 SUPPORTED, and the symmetric criterion fired** — both halves are reported.

| paper | what it measured | reads for us |
|---|---|---|
| `2602.19634` *Compositional Planning with Jumpy World Models* (2026-02-23, Farebrother et al.) | multi-step "jumpy" models across **multiple timescales** + *"a novel consistency objective that aligns predictions across timescales"*; **+200 % relative** over planning with primitive actions on long-horizon tasks | ⭐ the **cross-timescale consistency objective** is the piece MM-E16's 1 : ~3 : ~15 ladder does not yet have. A ladder without it is three independent predictors, not a hierarchy. |
| `2607.12547` *Mind the Gap* (2026-07-14) | Hi-LeWM on PushT/Cube: *"Hierarchy does not automatically improve performance"*; **at short horizons the best config is a one-step high-level horizon**; the bottleneck is **high-level subgoal generation**, not the low-level controller; constrained macro-action search recovers **+11.3 / +14.7 pts** | ⛔ predicts **no gain at the operative band** — so a ladder must be judged at tactical/strategic or it will read as a regression. ⭐ And: *"Unconstrained search can select latent macro-actions that appear favorable under the learned model but produce poor control targets"* ⇒ the strategic level must search a set **encoded from real trajectories**, not free latent space. |

⭐ **A collision worth naming.** *Mind the Gap* **freezes the low-level controller** and finds
it *"can execute well-aligned intermediate targets"* — the bottleneck is above it. P5 records
our own uncomfortable observation that `splitp30k` is simultaneously the best encoder (L2
winner), the frozen-encoder arm, and the **deadest predictor** (the L3 deliberate-regression
arm). Those two point the same way: **freeze low, fix high.** ⛔ `[INFERRED]`, and against a
known counterweight — REF-A's frozen-encoder ceiling — so it is a hypothesis, not a
recommendation.

⚠️ **Both papers are manipulation/navigation, not driving** (Q5, Q6). Neither prices a 6–30 s
driving horizon. The transfer is structural, not numeric.

## 6. F6 — SD-JEPA gives "restating instead of transporting" an architectural remedy `[PUBLISHED, abstract only]`

`2605.31111` (2026-05-29) carves the latent into *"a low-dimensional progression subspace
shaped by a cosine-margin triplet loss, and a high-dimensional content subspace regularised
by the existing SIGReg objective"*, on the premise that *"no single coordinate of the latent
is designated to encode task progression."*

⇒ P5's phrasing — *"the predictor is not transporting the scene, it is restating it"* — is
precisely a latent with no designated progression coordinate. ⚠️ Their result is *"improves …
on the majority of its control benchmarks at matched compute"*: **no magnitude, not driving**.
Admissible as a candidate mechanism, inadmissible as an expected effect size.

## 7. HYPOTHESIS VERDICTS

| ID | verdict | on what |
|---|---|---|
| **H-LAB-CMD-1** | ⭐ **SUPPORTED** | `1710.02410` states the failure in its own words and reports 64 % vs 52 % between two conditioning mechanisms |
| **H-LAB-CMD-2** | ⛔ **REFUTED** — the outcome pre-committed in SPEC §2 as the spending-relevant one | `2605.25313` §4.2 identifies a constraint that binds regardless of channel provenance; `2607.27017` reproduces the input-vs-target asymmetry in a second domain |
| **H-LAB-L3-1** | ⭐ **SUPPORTED** | *"beyond a persistence baseline"*, `2607.27017` |
| **H-LAB-P4-1** | **SUPPORTED, with the symmetric criterion fired** | `2602.19634` +200 %; `2607.12547` *"Hierarchy does not automatically improve performance"* |

## 8. ⭐⭐ THE COUNTER-EXAMPLE — found by a parallel frontier sweep, and it makes F1 narrower

⚠️ **An earlier draft of this section claimed a two-probe absence: "no published mechanism by
which a genuine command channel restores action-sensitivity." A frontier sweep run in
parallel overturned the framing, and the corrected version is below. Recording the
supersession rather than quietly editing it, because a weak absence that gets overtaken is
exactly the pattern our retraction discipline exists to surface.**

**`2606.12987` — *Diffusion Transformer World-Action Model for AV Scene Prediction*
(2026-06-11) — is the one frontier driving world model in the sweep that conditions on a
genuinely independent command, and it is action-controllable.** I verified both at source:

> *"Ego-vehicle actions are extracted from CAN-bus data as 2D vectors aₜ=(steerₜ, accelₜ),
> z-score normalized using training-set statistics only."* — §3, full text

> *"The model is genuinely action-controllable (steering drives scene displacement,
> Spearman ρ = 0.81, vs −0.18 for regression)."* — abstract

⭐⭐ **Its action channel is the SAME 2-D `(steer, accel)` parameterisation as ours — read off
the bus instead of off the curvature column.** The corpus is **nuScenes** (150 held-out
scenes), which means a CAN action channel exists on a corpus this programme already handles.
⚠️ I did **not** verify which nuScenes CAN field they used; that is UNVERIFIED.

⛔ **But it does not isolate the channel, so it does not refute F1.** The same paper changes
four other things at once — a V-JEPA2 temporal-context encoder (*"reduces steering RMSE by
40 % over the best single-frame encoder"*), an **x₀ diffusion objective**, **residual
anchoring**, and matched sampling. **It is not a teacher-forced latent-regression model at
all**, so the (d) mechanism is not the one it escaped from. ⇒ **existence proof that a CAN
channel co-occurs with controllability; not evidence that the channel caused it.**

⭐⭐⭐ **AND ITS DIAGNOSIS IS OURS, IN A DRIVING MODEL, WITH A FIX THAT IS P4's MECHANISM:**

> *"We trace limited single-pass motion to a shared-present anchor and engineer a compact
> 1.7M-parameter 'jump' model that recovers full ground-truth motion magnitude (1.02× GT),
> where single-pass models capture less than half."* — abstract

> *"a compact (1.7 M-parameter, n_blocks=2, dim 192) network predicts a single Δt=4
> transition z_t → z_{t+4}, conditioned on the four intervening actions."* — full text

⇒ **A "shared-present anchor" limiting predicted motion IS P5's *"restating, not
transporting"*, named in a driving world model — and the fix was not a better channel and not
a bigger model, it was a TEMPORALLY ABSTRACT STEP.** ⭐ **P4 and P5 are the same lever here**,
and the jump model that fixed it is **1.7 M parameters**. ⚠️ Their deficit is *motion
magnitude*; **88.7 % of our oracle gap is longitudinal** — the same shape of defect, which is
an argument for attention, not a measurement.

### The pathology also has two published names now `[relayed, existence verified by banking]`

- `2607.22535` *Robot-Factored World Models via Robot Rendering* (2026-07-24) — **I verified
  this sentence myself in the abstract:** *"Conditioning directly on action commands asks the
  world model to learn the realization process itself, while conditioning on logged future
  states leaks the interaction outcomes it is meant to predict."* ⭐⭐ **Both horns of P2(b)
  stated as a dilemma** — and their answer is a *middle* signal (the command rolled through
  the vehicle's own controller into a nominal trajectory), neither raw command nor logged future.
- `2608.06706` *Dueling World Models* and `2607.22430` *On the Identifiability of Controlled
  World Models* were surfaced by the sweep as naming action-blindness and its identifiability
  conditions. ⚠️ **Existence, title and date are verified by banking; their internal quotes
  are RELAYED and not independently verified by me** — do not quote them until read.

### What I still did not find
- No driving-domain replication of the **teacher-forcing** result specifically. Both papers
  supporting F1 remain outside driving.

## 9. ⛔ ESCALATION — three decisions, none of them mine

1. ⭐⭐⭐ **ADD CAUSE (d) TARGET CONSTRUCTION TO P2's TABLE, AND RE-RANK.** (d) has a published
   mechanism, a published magnitude and a published fix; (b) and (c) have none of the three.
   ⇒ **the recommended ordering is (d) before (b).**
2. ⭐⭐ **DO NOT COMMIT A comma2k19 ARM TO P2(b) UNTIL (d) IS MEASURED.** The gate already
   notes (b) needs a **PI geometry decision** (65.203° vs our 120°) and the data is not on
   Thor. Under F1 that arm's predicted outcome is *no change*, which would cost a geometry
   concession and an arm to learn nothing. ⚠️ This is a recommendation to **re-order**, not to
   close — closing is the PI's.
3. ⭐ **THE SIMULATOR-FREE VARIANT OF THE (d) TEST — one arm, one variable, no new corpus.**
   Their counterfactual target needs a simulator we do not have. The property that matters is
   only that *"two different action sequences map to two different targets."* A
   **negative-action contrastive term** supplies that without a simulator: score the
   prediction against the realised future **and** against the same future under a perturbed
   action drawn from the corpus, requiring the two to separate. ⚠️ **Pre-registration
   required before it is built** — with the deliberate-regression arm the validation standard
   demands, since a term that merely inflates the ratio without improving prediction would
   look like success. ⛔ **This is a design sketch, not a validated recipe**; it has not been
   costed and no line of it exists.

## 10. Cross-field handoffs

- ⭐⭐ **→ Data Engineering.** `2607.27017`: *"Every arm missing information or prediction
  pressure stays flat over a fivefold data range"* and *"additional data improves only the
  parameters it already acquires."* ⇒ **D-DATA-EFFICIENCY has an objective-side precondition:
  a data lever cannot be scored honestly on an arm whose objective does not already acquire
  the quantity.** Measured over a 5× data range on 4,258 real robot episodes — the cleanest
  external warrant we have for sequencing objective work before data work.
- **→ Benchmarks & Evals.** ATM (`2606.09028`) is a candidate cheap screening instrument;
  its *"when the true success gap is non-trivial"* condition must travel with it.
- **→ Deployment & Optimization.** Jumpy/temporally-abstract rollout is not only an accuracy
  mechanism — fewer, larger steps is also the inference-cost mechanism for a 6 s horizon.

## 11. Deliverable manifest

| artifact | where | only-one-place? |
|---|---|---|
| `SPEC.md` (hypotheses + pre-committed criteria) | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-08-31-command-conditioning-and-l3/SPEC.md` | no — staged |
| `RESULT.md` (this) | same dir | no — staged |
| `raw/QUOTES.md` (verbatim primaries + retrieval method) | same dir | no — staged |
| `COMMS.md` | same dir | no — staged |
| banked PDFs | `repo:TanitAD Research Lab/Library/papers/` + `library.json` | no — staged |
| KB line | `repo:TanitAD Research Lab/Architecture & Inference/Research/KNOWLEDGE_BASE.md` | no — staged |

**No code changed. No test suite touched. No commit, no push, no branch switch.**
