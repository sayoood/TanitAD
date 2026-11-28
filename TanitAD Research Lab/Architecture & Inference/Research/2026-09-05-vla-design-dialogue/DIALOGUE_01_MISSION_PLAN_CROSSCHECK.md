# VLA design dialogue — note 01: the PI's critique, cross-checked against the Mission Plan

**author:** Training FlyWheel · **date:** 2026-09-05 · **branch:** `agent/arch-inf-20260803` · **GPU: 0**
**Trigger:** PI, 2026-09-05 — *"your proposal is mainly to take the language model, the other modules are
very small and im not recognizing the refc design in the plan."*
**Verdict of this note: the critique is CORRECT, and it is more specifically correct than its own wording —
the plan contradicts the PI's own `H12`, and it re-invents `H11` and `H13` without naming them.**

---

## 1. The number that makes the critique quantitative (MEASURED)

`MODEL_REGISTRY.md:2711`, refcv3 base, `config.json param_breakdown.total` **and** the checkpoint
state_dict in exact agreement (107,032,901):

| brain | modules | params | share |
|---|---|---|---|
| **perception + operative** | `core` (trunk, 117-anchor decoder, selector) | **104,879,522** | **97.99 %** |
| **tactical** | `phi_tac` 1,757,440 · `tac_latent_proj` 262,656 · `tac_heads` 14,364 · `scorer` 1,156 | **2,035,616** | **1.902 %** |
| **strategic** | `gstr_cond` 66,816 · `str_goal_head` 771 | **67,587** | **0.0631 %** |
| nav | `nav_inject` | 50,176 | 0.0469 % |
| | | **107,032,901** | 100 % (sums exactly) |

⇒ **The strategic brain is 0.063 % of REF-C.** The banked plan proposes to *complement* it with a
**368 M** module — **5,445×** its size, and **171×** the entire cascade above the trunk (2,153,379).

⭐ **That is not "extending REF-C with a language layer". It is replacing the strategic brain with an LM
and calling the result complementary.** The PI's sentence is the correct read of the arithmetic.

## 2. The plan contradicts the PI's own H12 (`Mission Plan.md`, verbatim)

> **H12** *"While its stupid to drive with language, our system must process language as additional part
> **and not as the core** of the autonomous system … we will stick to the unsupervised world model as main
> direction, but we will **extend it by a text processing part** (could be an adapted llm backbone). It must
> process e.g. strategic navigation commands, adapt the behavior of the stack and output interpretable
> reasoning traces as text. The challenge is to find an efficient way to design a **common latent space
> which includes all**"*

The banked plan makes the LM the core **by mass** (98.6 % of the proposed system's parameters live in
SmolLM2-360M, against 1.4 % in everything REF-C's hierarchy contains). ⇒ **`H12` violation**, and it is
the design error the PI's sentence names.

⚠️ What the plan got *right* against H12 and then buried in a table cell: **trunk-as-only-encoder** is the
literal answer to *"a common latent space which includes all"*, and it is the one reading of it with a
testable consequence. **Defend this; it is the strongest thing in the plan.**

## 3. Three Mission-Plan hypotheses the plan re-invented without naming — all of them already answer its open questions

| Mission Plan | verbatim (condensed) | what the plan called it | consequence |
|---|---|---|---|
| **H11** | self-monitoring *"assigned to the **strategic part** which should be **generative and able to process text** … slow thinking heavy brain … generate incident monitoring reports as requested by the new ADS regulations … **different self monitoring at each abstraction level**"* | invented "**mode M — monitor**", unattributed | ⭐ **Answers "where does the language module sit?" — it is DECIDED DOCTRINE, not an open question.** Generative text is the **strategic brain's own faculty**; monitoring is per-level. Not a 4th level, not a peer, not a replacement. |
| **H13** | extraction heads incl. *"a **behavior extraction head extracting the chosen behavior and the considered alternatives**"* | absent | ⭐⭐ **"the chosen behaviour and the considered alternatives" IS the 117-anchor fan.** The grounded CoT is a statement about which hypotheses were alive, which died and why. This is the missing REF-C-shaped mechanism. |
| **H3** | world models: *"predict … through imagination and thus **derive the consequences of their actions** … big advantages regarding … the **explainability of behavior**"* | absent | a **chain of causation** needs predicted consequences. REF-C already rolls 117 kinematic futures — an enumerable imagination, free at eval. The plan's CoT narrates a classifier instead. |

**And two more the plan never used, both strong arguments FOR a pretrained LM that it failed to make:**

- **H9** — *"inherently include the compliance to traffic rules … **incrementally add traffic rules without
  retraining the whole system** and without loosing the end2end character"*. An LM with a rule vocabulary is
  the only mechanism we have for this. The plan's `contradiction_rate` / `GOAL_ADMISSIBLE_*` maps are a
  shadow of it.
- **H14** — *"inject wide knowledge relevant to driving … physical laws, code of behavior, efficient driving,
  cultural differences … general ethics principles"*. This — **not** "47 k words is too little data" — is the
  real argument for a *pretrained* backbone: it is the only vehicle for priors we cannot label.

## 4. The two clocks are ARCHITECTURE, not a latency defeat

`Mission Plan.md`, H1: *"It matches perfectly to the 'thinking slow', 'thinking fast' paradigms and making
the system very efficient in inference, since **not all the parts must work with the same frequency e.g.
100 ms vs 500 ms vs. 1 second** (subject of investigation)"*.

The plan presents its two-clock scheme apologetically (*"a full prose CoT per tact is arithmetically
impossible"*). ⇒ **Presentation error.** Multi-rate is the programme's stated design intent; the plan
implements it and then apologises for it.

## 5. The fourth brain is missing entirely

H1 defines brain 4 as **the fallback**, responsible for reaching **MRC** on collapse of the operative or
tactical part, *"can be also triggered by the strategic layer which is supervising the whole system"*.

The plan's "mode M" module explains and **flags** — and has **no trigger port, no MRC output, and no
connection to a fallback brain**. H11 + brain 4 are the natural home of exactly that flag. ⇒ design gap.

## 6. Classification of the critique

| # | finding | class |
|---|---|---|
| D1 | LM is the core by mass — **H12 violation**; parameter split risk-managed, never designed | ⛔ **design error** |
| D2 | CoT is not built on the fan; **H13**'s behaviour-extraction head (chosen + considered alternatives) is absent | ⛔ **design error** — the one that would make REF-C recognisable |
| D3 | no world model / imagined consequences (**H3**) — a chain of *causation* without consequences | ⛔ **design error** |
| D4 | no fourth brain, no MRC trigger port for the monitor (**H1**, **H11**) | ⛔ **design error** |
| D5 | first deliverable has every write gate at 0 — an apparatus for measuring before anything drives | ⚠️ **ordering error** (see the PI's standing correction on refutation-seeking) |
| P1 | document ordered LM-first; REF-C appears as an interface annex | ⚠️ presentation |
| P2 | two clocks framed as arithmetic defeat rather than **H1** multi-rate doctrine | ⚠️ presentation |
| P3 | "mode M" is **H11** self-monitoring, unnamed | ⚠️ presentation |
| P4 | trunk-as-only-encoder — correct, load-bearing, buried | ⚠️ presentation (**defend**) |

## 7. What is worth defending unchanged

1. **Trunk as the ONLY vision encoder** (§9.1 contract term). It is H12's "common latent space", it is the
   300–500 ms budget, and it is what makes a plan/explanation divergence attributable to *reasoning* rather
   than *perception*. A second encoder destroys the measurement.
2. **Referents POINT (`REF(k)` over agent tokens ∪ NULL)** — a hallucinated object becomes structurally
   impossible rather than penalised. 853 explicit `type: none` clips make NULL learnable.
3. **Distil the teacher's FORM, not its CONTENT** — 42.5 % reasoning fidelity, 48.3 % reasoning-action
   consistency, our own spot check 3/5 with a hallucinated cyclist.
4. **Every write edge zero-init and individually gated** — without it no edge is attributable.

## Evidence class

All §1 numbers **MEASURED** (`MODEL_REGISTRY.md:2710-2711`, two independent counts in exact agreement;
the four-way split sums to the published total exactly). All §2–§5 quotations **read verbatim** from
`Project Steering/Mission Plan.md` (H1, H3, H9, H11, H12, H13, H14) on 2026-09-05. No claim here is a
driving result and nothing was launched.
