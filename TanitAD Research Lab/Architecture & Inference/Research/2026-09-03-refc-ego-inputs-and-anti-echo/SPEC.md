# SPEC — E-REFC-EGO-1: what the REF-C ancestors fed, and how the field stops a planner echoing its own dynamics

*Architecture & Inference FlyWheel · Research Lab literature stream · 2026-09-03.
**0 GPU.** Direct PI request: "Analyse in the original refc paper which data they fed to
the models, explore methods to avoid echoing the ego dynamics in refc-like architectures."*

## 0. Why this work package exists

The PI **reinterpreted** the binding vision-only rule on 2026-09-03, verbatim:

> *"The vision only rule is actually saying that the semantic understanding and the
> trajectory planning must be based on image frames and should avoid echo of the ego
> dynamics or predicting the ego trajectory blindly without considering the environment."*

⇒ The rule's PURPOSE is **anti-echo**, not anti-ego-input. Combined with the 2026-09-02
ruling (*"It is allowed to use the velocity as initial measured state at its cycle time.
It is not allowed to use the future dynamic information from the ground truth"*), refcv4
will consume **measured `v`, `a_long`, `yaw_rate` at t0** — and must PROVE it is not
echoing. Proving that is a LITERATURE question before it is a code question: the field
has run this experiment, and it has published both the failure and several fixes.

This WP owns the **external evidence**. A sibling stream owns our own code.

## 1. Questions (each answered or explicitly marked NOT FOUND)

| Q | question | success = |
|---|---|---|
| **Q1** | Which paper is "the original REF-C"? | the lineage named from OUR source, and every candidate paper banked |
| **Q2** | What ego/state inputs does each ancestor feed at INFERENCE? | the exact scalars, from the primary |
| **Q3** | WHERE do they enter? | the named module, from the primary |
| **Q4** | Is any ego input WITHHELD from the goal/planner path, and why? | yes/no + the stated reason, or "the paper is silent" |
| **Q5** | What do the ancestors say about ego-dynamics shortcut learning? | quoted, or "absent" after two greps |
| **Q6** | The ego-status-baseline result — what was measured, what was the fix? | the table numbers, primary |
| **Q7** | Implementable anti-echo METHODS: mechanism, attach point, cost, evidence | one row per method |
| **Q8** | Evaluation-side ECHO DETECTION | the instruments, with published reference values |
| **Q9** | Ranked for OUR setting + the 3 cheapest discriminating experiments | ranked list, each with a pre-registerable arm set |

## 2. Both outcomes, committed in advance

This is a literature WP, so the pre-registration is about what would make the
recommendation WRONG, not about a metric threshold:

| if the literature had said… | then the recommendation would have been… |
|---|---|
| ego status is harmless and every planner feeds it freely | "feed `(v, a_long, yaw)` everywhere, no guard, ship it" |
| ego status is catastrophic in every benchmark | "refuse the PI's reinterpretation and escalate" |
| ego dropout has no ablation anywhere | "the rate is unknowable; do not pre-register a rate, sweep blind" |
| the residual-over-constant-velocity trick is published and works | "reparameterise the anchor offsets as CV residuals" |

**Every one of these four was checked and three were falsified by the primaries** (see
`RESULT.md` §6 and §7). The fourth (residual-over-CV) was falsified in the *opposite*
direction — the nearest measurement says it HURTS — which is why it is ranked low rather
than adopted.

## 3. Admissibility rules this WP binds itself to

1. ⛔ **PUBLISHED-SECONDARY is inadmissible.** Every number below is read from a PDF we
   hold, whose sha256 matches `library.json`. Where a primary could not be reached, the
   claim is stamped **PUBLISHED-SECONDARY** and is explicitly barred from
   `MODEL_REGISTRY.md` and the paper.
2. **Absence needs two probes of different path binding.** Where this WP says a paper does
   NOT do something, it names the two greps.
3. **Every recommended instrument carries the CLAUDE.md probe-panel controls**: a
   constant-only control that must read the no-information value, a raw-input floor, and
   printed `n` and `d`.
4. **No claim here is T1.** These are PUBLISHED facts about other people's models on other
   people's benchmarks. They set priors and design; they never enter our registry as our
   numbers.
