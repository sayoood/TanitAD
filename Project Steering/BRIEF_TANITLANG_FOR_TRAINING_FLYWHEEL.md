# BRIEFING — the TanitLang extension, for the Training FlyWheel session

**Written 2026-09-05 by the Master Mind.** ⛔ **Read this file first, then the three sources at the
bottom.** It is self-contained: a session that reads only this can pick the work up cold.

⚠️ **Why this file exists.** The PI asked me to brief the Training FlyWheel agent directly and then
chat with it himself. **I could not: `SendMessage` is DISABLED in my session** (*"No such tool
available: SendMessage. SendMessage is disabled for this session, in subagents as well as here"*),
so no message ever reached that session, and the PI saw nothing there. **That is a delivery failure
I should have reported the moment it happened instead of leaving it silent.** This document is the
route that does work: point the session at this path.

---

## 1. The mission, in the PI's own words

> *"extend the architecture with a ≤1B-parameter vision-language part sharing REF-C's embedding
> space, processing nav commands and question queries, creating system-initiated chain-of-thought
> grounded in scene understanding and consistent with trajectory hypotheses, able to process and
> emit strategic and tactical goals, 300–500 ms tact."*

And his dissatisfaction with the first proposal, verbatim:

> *"I have many questions there and im not statisfied with the results since the idea was to extend
> the refc design (diffusion drive) with vision language and grounded COT in combination with our 4B
> multihierrahcacy archotecture, your proposal is mainly to take the languge model, the other
> modules are very small and im not recognizing the refc design in the plan."*

## 2. ⭐ The critique is CORRECT, and the arithmetic makes it quantitative

MEASURED (`MODEL_REGISTRY.md:2711`, refcv3 base — `config.json param_breakdown.total` and the
checkpoint `state_dict` in exact agreement at 107,032,901):

| brain | params | share |
|---|---|---|
| perception + operative (`core`) | 104,879,522 | **97.99 %** |
| tactical | 2,035,616 | 1.902 % |
| **strategic** | **67,587** | **0.0631 %** |
| nav (`nav_inject`) | 50,176 | 0.0469 % |

⇒ **The strategic brain is 0.063 % of REF-C.** The banked plan proposes to *complement* it with a
**368 M** module — **5,445× its size**, and **171×** the entire cascade above the trunk.

⭐ **That is not "extending REF-C with a language layer". It is replacing the strategic brain with an
LM and calling the result complementary.** The PI's sentence is the correct read of the arithmetic.

## 3. It also contradicts the PI's own H12, and re-invents H11/H13/H3

`Mission Plan.md` **H12**, verbatim: *"our system must process language as additional part **and not
as the core** … we will stick to the unsupervised world model as main direction, but we will extend
it by a text processing part."*

And the plan re-derived **H11**, **H13** and **H3** without citing them — ⚠️ notably **H13's
"considered alternatives" IS the 117-anchor fan**, i.e. the plan reinvented a mechanism REF-C
already ships.

## 4. ⛔ THREE FORKS AWAITING THE PI — nothing should be built until he answers

1. **Where does the reasoning live** — in the language model, or in the **fan**? (REF-C already
   enumerates and scores hypotheses; a CoT that merely narrates the fan's choice is an explainer,
   a CoT that *changes* the choice is a second planner competing with the tactical brain.)
2. **Does the hierarchy grow?** A ≤1B module cannot be a 0.063 % strategic brain. Either the
   strategic brain grows by orders of magnitude — which is an architecture change the PI must
   authorise — or the language part sits *beside* the hierarchy and does not plan.
3. **What is the language layer's FIRST product?** The agent's own instinct was **H11 monitoring**
   (the system saying when it is uncertain / why it refused), **not** explanation-after-the-fact.

## 5. ⚠️ Constraints that bind any design here

- **H12:** language is an additional part, never the core. A common latent space that includes all.
- ⛔ **The goal input may NOT carry the situation classifier's output** in any form — posterior,
  argmax, embedding, or any feature derived from them. Attribution dies otherwise, and it is the
  nav-echo defect in a new costume.
- ⛔ **Vision-only at inference.** Labels may use ego, other agents and future poses; inference may
  not. *(This was verified at source for refav1 today: `plan()` builds the goal input with
  `ego=None`, and nav-only AUC 0.5967 against the head's 0.7120 detection / 0.9043 direction shows
  those are not nav echoes.)*
- **Latency:** the PI's tact is **300–500 ms**. ⚠️ Alpamayo-R1's published "99 ms" is a **workstation**
  number on an RTX 6000 Blackwell and **71 % of it is the CoT** (reasoning decode 70 ms of 99);
  the paper publishes **no embedded breakdown**. Quoting it as on-vehicle latency is the
  datacentre-GPU trap (`D-VLA-LAT1`, PUBLISHED-PRIMARY, `2511.00088` Table 14).

## 6. Sources — all in-repo, all banked

| what | path |
|---|---|
| the design plan (27.6 KB) | `Project Steering/REFCV5_VLA_EXTENSION_PLAN.md` |
| the frontier research | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-vla-extension-frontier/RESULT.md` |
| **the critique cross-check** (read this one) | `…/Research/2026-09-05-vla-design-dialogue/DIALOGUE_01_MISSION_PLAN_CROSSCHECK.md` |
| banked primaries | `TanitAD Research Lab/Library/papers/` — DriveVLA-W0, CoVLA, DriveWorld-VLA, OpenDriveVLA, AtomVLA, *Is VLA Reasoning Faithful?* |
| the param breakdown | `Project Steering/MODEL_REGISTRY.md:2711` |

⛔ **Do not start building.** The three forks in §4 are the PI's to answer, and the first proposal
failed precisely by choosing them implicitly.
