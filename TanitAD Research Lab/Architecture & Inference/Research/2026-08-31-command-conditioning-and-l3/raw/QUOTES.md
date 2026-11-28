<title>raw/QUOTES — verbatim primary passages, 2026-08-31</title>

# raw/QUOTES.md — the quotable layer

`Every passage below was read from the primary source named, on 2026-08-31 (Europe/Berlin).`
`Nothing in RESULT.md may quote a number or a phrase that is not here.`
`Retrieval method is stated per entry, because "I read the abstract" and "I read §4.2" are`
`different evidence and collapsing them is how a PUBLISHED-SECONDARY claim gets laundered.`

---

## Q1 · UWM-JEPA — `2605.25313`

**Title:** *UWM-JEPA: Predictive World Models That Imagine in Belief Space*
**Authors:** Santosh Kumar Radha, Oktay Goktas · **Submitted:** 2026-05-25
**Retrieval:** `arxiv.org/abs/2605.25313` (metadata + abstract) **and** `arxiv.org/html/2605.25313v1`
(full text, §4.2 / §4.3). ⭐ The §4 passages are from the FULL TEXT, not the abstract.

> "Under a teacher-forced JEPA objective the target encoder observes the trajectory that
> already contains the action's effect, which admits an action-invariant solution."
> — §4.2

> "the target encoder has already consumed the observed future and the predictor can match
> it without using the action"
> — §4.2

> "‖H₁‖/‖H₀‖≈0.03"
> — §4.2, the measured magnitude of the action term under teacher forcing.

> "a simulator rollout under a freshly sampled counterfactual action sequence"
> — §4.3, the construction of the counterfactual target.

> "two different action sequences map to two different targets, so any H₁ that is too small
> incurs loss"
> — §4.3, why the counterfactual target binds the action.

> "‖H₁‖/‖H₀‖=1.00±0.15"
> — §4.3, after counterfactual training, across seeds.

> "simulator-state access during training"
> — §4.3, the stated requirement of the fix. ⛔ This is the sentence that limits the transfer.

**Abstract (verbatim, for scope):** *"World models for partially observed environments must
imagine multiple compatible hidden futures and steer between them under counterfactual
actions. Joint Embedding Predictive Architectures (JEPAs) do this in latent space, but a
vector-valued latent has no internal structure for carrying the belief over hidden
continuations through blind rollout. We introduce the Unitary World Model JEPA (UWM-JEPA), a
JEPA world model with a density-matrix latent on a joint system-environment space and a
learned unitary predictor. The construction preserves the joint-state spectrum exactly during
rollout, so the predictor itself cannot dissipate the represented uncertainty."*

⚠️ **Scope caveat I am recording against myself:** the paper's headline contribution is the
density-matrix latent, **not** the teacher-forcing analysis. §4.2/§4.3 are a controlled
ablation inside that paper. Citing them is legitimate; describing the paper as "about
teacher forcing" would not be.

---

## Q2 · Conditional Imitation Learning — `1710.02410`

**Title:** *End-to-end Driving via Conditional Imitation Learning*
**Authors:** Codevilla, Müller, López, Koltun, Dosovitskiy · ICRA 2018
**Retrieval:** `ar5iv.labs.arxiv.org/html/1710.02410` (LaTeX-derived full text of the arXiv
source). ⛔ An `alphaxiv` summary page was opened first and is **discarded as secondary**;
every quote below is from the ar5iv full text.

> "The command is processed as input by the network, together with the image and the
> measurements. The outputs of these modules are concatenated into a joint representation."
> — the *command input* architecture.

> "However, the network is not forced to take the commands into account, which can lead to
> suboptimal performance in practice."
> — the stated failure of the command-input architecture. ⭐ This is the sentence.

> "The command acts as a switch that selects which branch is used at any given time … one
> module might specialize in lane following, another in right turns, and a third in left turns."
> — the *branched* architecture.

> "continue (follow the road), left (turn left at the next intersection), straight (go
> straight at the next intersection), and right (turn right at the next intersection)"
> — the command vocabulary; represented as one-hot vectors.

**Numbers (Table I, simulation, Town 2 = held-out test town):**
branched **64 %** success · command input **52 %** success.

---

## Q3 · What Can Latent World Models Know? — `2607.27017`

**Title:** *What Can Latent World Models Know? Physical Parameter Identifiability in
Multimodal Predictive Representations*
**Authors:** Kaizhen Tan, Xin Xu, Siru Tao, Yixiao Li, Hanzhe Hong, Yang Feng, Heqing Du
**Submitted:** 2026-07-29 (v1) · **Retrieval:** `arxiv.org/abs/2607.27017` — ⚠️ **ABSTRACT
ONLY.** The protocol details are UNVERIFIED at full text; every claim I draw from this
paper is therefore bounded by what the abstract states.

> "A certificate-gated protocol first certifies each parameter as recoverable from raw
> observations, then measures whether it enters the latent, so a null result can be
> attributed to the objective rather than to the environment."

> "Inputs limit what can be known, while prediction targets decide what is retained."

> "Stiffness enters the latent only when touch is forecast (R²=0.50, compared with −0.02 when
> the same signal is merely fused into the input)"

> "under single-step prediction a vision-only latent discards even perfectly visible object state"

> "It carries a recoverability certificate of 0.89 yet plateaus near 0.13 under every
> deterministic prediction objective we test, while a supervised head on the same trunk
> reaches 0.45."

> "only the full multimodal objective forecasts force beyond a persistence baseline, with
> held-out gains that grow with scale"

> "Every arm missing information or prediction pressure stays flat over a fivefold data range"

> "Objective structure determines which physical parameters a latent acquires, and additional
> data improves only the parameters it already acquires."

Scale stated: POKEWORLD (controlled) + **RH20T, two robots, 4,258 episodes**.

---

## Q4 · ATM — `2606.09028`

**Title:** *ATM: Action-Consistency Transfer Matrix for Diagnosing and Improving Latent
World Models* · **Author:** Jiaheng Chen · **Submitted:** 2026-06-08
**Retrieval:** `arxiv.org/abs/2606.09028` — ⚠️ **ABSTRACT ONLY.**

> "ATM compares action information in real encoded transitions and model-predicted
> transitions through lightweight post-hoc probes, producing an interpretable matrix that
> reveals representation quality, transition-domain inconsistency, and failure modes without
> simulator rollout."

> "When the true success gap is non-trivial, ATM achieves highly reliable pairwise ranking,
> while reducing minutes-to-hours CEM evaluation to seconds-level transition analysis,
> yielding more than 100x speedup in our setup."

> "We further introduce AITS, showing that action-identifiability is not only diagnostic but
> also a useful training signal for improving downstream planning without changing the planner."

⚠️ Note the conditional: *"When the true success gap is non-trivial"*. That is a stated
limit of the instrument, and it is the half a careless citation drops.

---

## Q5 · Compositional Planning with Jumpy World Models — `2602.19634`

**Authors:** Jesse Farebrother, Matteo Pirotta, Andrea Tirinzoni, Marc G. Bellemare,
Alessandro Lazaric, Ahmed Touati · **Submitted:** 2026-02-23
**Retrieval:** `arxiv.org/abs/2602.19634` — abstract.

> "we address these challenges by learning predictive models of multi-step dynamics -- so-called
> jumpy world models -- that capture state occupancies induced by pre-trained policies across
> multiple timescales in an off-policy manner"

> "we enhance these models with a novel consistency objective that aligns predictions across
> timescales, improving long-horizon predictive accuracy"

> "yielding, on average, a 200% relative improvement over planning with primitive actions on
> long-horizon tasks"

Domain stated: *"challenging manipulation and navigation tasks"* — ⛔ **not driving.**

---

## Q6 · Mind the Gap — `2607.12547`

**Title:** *Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel*
**Authors:** Niccolò Caselli, Francesco Massafra, Samuele Punzo, Salvatore Lo Sardo,
Ippokratis Pantelidis, Sathya Kamesh Bhethanabhotla · **Submitted:** 2026-07-14 (v1)
**Retrieval:** `arxiv.org/abs/2607.12547` — abstract.

> "Hierarchy does not automatically improve performance: at short horizons, the best
> configuration uses a one-step high-level horizon, while longer horizons reveal a mismatch
> between the learned high-level action space and the inference-time search distribution."

> "Experiments with true future latent subgoals show that the frozen low-level controller can
> execute well-aligned intermediate targets, indicating that high-level subgoal generation is
> the main bottleneck."

> "Unconstrained search can select latent macro-actions that appear favorable under the
> learned model but produce poor control targets."

> "Constraining search around macro-actions encoded from training trajectories, with
> appropriate subgoal execution timing, recovers useful hierarchical regimes, improving over
> flat LeWM by +11.3 percentage points at medium-range horizons and +14.7 percentage points
> at the longest PushT horizon."

Benchmarks stated: **PushT and Cube** — ⛔ **not driving.**

---

## Q7 · Subspace-Decomposed JEPAs — `2605.31111`

**Title:** *Subspace-Decomposed JEPAs: Disentangling Progression and Content in Latent World Models*
**Authors:** Lucas Thil, Jesse Read, Rim Kaddah, Guillaume Doquet · **Submitted:** 2026-05-29
**Retrieval:** `arxiv.org/abs/2605.31111` — abstract.

> "Joint-Embedding Predictive Architectures (JEPAs) learn compact latent world models by
> predicting future embeddings, but no single coordinate of the latent is designated to encode
> task progression."

> "We carve the JEPA latent into two orthogonal subspaces with disjoint roles: a
> low-dimensional progression subspace shaped by a cosine-margin triplet loss, and a
> high-dimensional content subspace regularised by the existing SIGReg objective of LeWM"

> "SD-JEPA improves over the LeWM baseline on the majority of its control benchmarks at
> matched compute."

⚠️ *"on the majority of"* is not a number. Do not quote this as a magnitude.

---

## Q8 · OUR OWN SOURCE — the target construction in the live trainer

**Retrieval:** `stack/scripts/train_v6_staged.py`, read 2026-08-31. Evidence class **MEASURED**
(source read, line numbers given).

- `:48` — the module docstring describes the objective family as *"prediction, each against
  its layer's stop-grad/EMA target."*
- `:1765` — *"the view change lives entirely on O5's teacher side"* — O5 has a teacher side.
- `:1812` — `_ema_tau_at`: *"The O5-EMA teacher's tau AT THIS STEP"*.
- `:4214` / `:4221` — `o5_target` takes `live` / `ema` / `frozen`; all three build the target
  by encoding the **real observed future**.

⇒ **Our O5 is teacher-forced in exactly the sense Q1 §4.2 defines.** This is the load-bearing
link between Q1 and our arms, and it is measured in our source rather than assumed.
