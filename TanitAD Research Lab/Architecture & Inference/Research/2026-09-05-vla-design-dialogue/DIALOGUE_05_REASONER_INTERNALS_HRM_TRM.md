# TanitLang v5 — the reasoner internals, and what HRM/TRM actually give us

**2026-09-05 · TanitAD_TrainingFlyWheel.** Answers four things the PI asked: the reasoner's
internals; whether TanitLang can emit Alpamayo-style CoT (**yes — §2**); a deep analysis of
HRM/TRM for driving (**§3, and the honest finding is not the one I expected**); and the three
directions elaborated (**§4**). Status: DESIGN.

---

## 1. ⛔ The HRM/TRM finding that reframes everything — analysed before adopted

I went to HRM/TRM expecting to import a hierarchy. **The independent replication says the
hierarchy is not what works.**

**PUBLISHED — ARC Prize's own ablation of HRM** ([arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis)),
which isolated five candidate explanations (architecture, hierarchical inner loop, outer refinement
loop, augmentation, puzzle embeddings):

| finding | consequence for us |
|---|---|
| ⛔ **the "hierarchical" architecture had MINIMAL impact** vs a similarly-sized plain transformer | do **not** import HRM's two-module hierarchy as a hierarchy — we already have one, and theirs is not what earns their numbers |
| ⭐⭐ **the OUTER REFINEMENT LOOP drove the performance** — at inference, **one** refinement loop nearly **doubled** accuracy (18.6 % → 35.5 %) | ⭐ **REF-C ALREADY HAS THIS LOOP.** Truncated diffusion *is* an outer refinement loop |
| ⭐ the loop performs **"embedding-space reasoning: evolving latent vectors toward consistency"** | this is a description of what diffusion refinement does |
| ⛔ **`puzzle_id` is a strong, LIMITING dependency**; TRM ablation: a blank/random ID yields **ZERO** accuracy. ARC Prize recommends replacing it with *"context-derived task conditioning that generalises to unseen tasks"* | ⚠️ **the single biggest transfer risk** — see §3.3 |
| ⚠️ TRM: **94.4 % of accuracy is reached at recursion step 1** | deep recursion is not where the value is. REF-C's `steps=2` is already in the right regime |
| ⚠️ much of the headline number comes from **1000-sample augmentation + majority vote** (+10.75 pp) | ⭐ REF-C's **128-anchor fan + scorer is natively a sample-many-and-vote mechanism** |

**TRM's recursion, for the record** (`2510.04871`, 7 M params, **two layers**): a single tiny
network keeps a current answer `y` and a latent `z`, and alternates
`z ← f(x, y, z)` (think) then `y ← g(y, z)` (answer), under deep supervision.

⇒ ⭐⭐⭐ **THE INSIGHT: REF-C already runs TRM's `y`-update and is missing TRM's `z`.**
The truncated diffusion loop refines the *answer* (`x ← x + offset`) but carries **no reasoning
state between steps**. TanitLang's reasoner is precisely that missing `z`.

**We are not bolting HRM/TRM onto REF-C. We are completing REF-C's own loop with the latent it
never had** — and the component the independent ablation says is the one that matters is the
component we already own.

---

## 2. The reasoner internals

### 2.1 The recursion, concretely

Per tact, for each of the K = 5 shortlisted candidates:

```
  x  the question   scene tokens (pv attn, WP-3) + agent tokens (WP-6) + nav + v0 + situation
  y  the answer     the candidate trajectory  [S, 2]   (REF-C already refines this)
  z  the scratchpad  d_z = 128 reasoning latent        (NEW — this is TanitLang)

  repeat for r = 1..R  (R = 2, matching REF-C's steps and TRM's shallow-recursion finding):
      z ← z + Block_think(x, y, z)          # think: 2 layers, shared across r and across candidates
      y ← y + Block_answer(y, z)            # answer: REF-C's EXISTING offset head, now z-conditioned
```

* **`Block_think` is 2 layers** — TRM's own depth, and TRM is 7 M total. Ours is smaller because
  `x` is already encoded by the trunk.
* ⭐ **The two blocks share weights across the R steps and across the K candidates.** One reasoner
  runs 5×2 = 10 tiny passes per tact, not 5 separate networks.
* **`y`'s update is REF-C's existing offset head**, now additionally conditioned on `z` through a
  zero-init projection — so `z = 0` reproduces today's REF-C bit-for-bit.

### 2.2 What `z` decodes to — three heads, all from the same latent

| head | output | supervision | consumer |
|---|---|---|---|
| **cot** | the **11 Alpamayo `cot_tokens` fields** (`yield_`, `merge`, `overtake`, `gap`, `lane_change`, `evade_obj`, `oncoming`, `traffic_light`, `exit_side`, `speed_limit`, `evidence`) | Alpamayo, **4,719/4,719 clips** | the CoT trace (§2.3) |
| **referent** | a pointer into the agent tokens | `semantics.referents` | grounding — "*which* car" |
| **score** | a scalar delta for this candidate | the four families | WP-7's selector (5a) |
| **goal** | `lat`/`lon`/strategic distributions | v7 vocabularies | next tact's priors (5b) |

⭐ **All four are decodes of the same `z`.** That is what makes the explanation and the decision
the *same* object rather than two things we hope agree — the consistency property is structural.

### 2.3 ⛔ YES — it generates Alpamayo-style CoT traces. That was never dropped.

**The requirement is met, and the mechanism is the `cot` head above.** To be explicit about what
I mean by "structured", because that is where I was unclear:

* Alpamayo gives us both: `cot_tokens` (the 11 **structured** fields) **and** `cot_source.cot` /
  `chain_of_causation` (the **free-text** trace, e.g. *"Nudge left to increase clearance to the
  parked car on the right."*).
* **TanitLang predicts the structured fields recurrently in `z`, and renders the text from them.**
  The trace it emits is an Alpamayo-style causal sentence — *generated*, not retrieved.
* ⭐ **Why via structure rather than free generation:** the text is then a *function of a
  supervised state*, so "does the CoT match the plan" is a comparison between two decodes of one
  latent — not a semantic-similarity guess. And a hallucinated sentence cannot reach the planner
  because the planner consumes the fields, not the string.

⚠️ **The trade I am making, stated plainly:** free-form generation could say things the 11 fields
cannot express. That is a real expressivity cost. It is bounded by the fields' coverage, which is
**measurable on the 4,719 traces** — and measuring it is WP-A below, before committing.

### 2.4 Text INPUT — designed for, not built first (as you asked)

`x` accepts an optional text-token slot, zero-init and absent by default. When a query arrives it
enters as extra tokens to `Block_think`. ⛔ Not in step 1: we have **no question corpus**, so there
is nothing to train or evaluate it on. The port exists so adding it later is a data problem, not an
architecture change.

---

## 3. Applying HRM/TRM to driving — the three real problems

### 3.1 ⭐ The reasoning problem is *intrinsically formulated*, as you asked

ARC gives the model a puzzle. **Driving must formulate its own.** In this design the problem is
posed by the fan itself: *"five plausible futures are live; which one, and why?"* — a question that
exists only because REF-C enumerates hypotheses. **The reasoning problem is a property of the
situation**, and its difficulty is observable: the **score margin** between candidates 1 and 2.

### 3.2 Adaptive computation — what is honest here

⚠️ **TRM has NO halting mechanism** — fixed recursion depth (4 in the checkpoint they verify), and
HRM's ACT was one of the ablated components. **I will not claim adaptive computation as an
inherited property.**

⭐ But driving offers a natural difficulty signal ARC does not have: **the top-2 score margin.** A
wide margin means the fan already agrees; a narrow one means the decision is contested. **R can be
1 when the margin is wide and 2–3 when it is narrow** — a *situation-dependent* compute budget.
⚠️ This is **my proposal, not a published result**, and it must be pre-registered with a fixed-R
control, because a variable-compute arm that beats a fixed-R arm may simply be spending more.

### 3.3 ⛔⛔ THE TRANSFER RISK: `puzzle_id` has no driving analogue

**This is the finding most likely to sink a naive transfer, so it goes at the top of the risk
list.** TRM's ablation: replacing the puzzle ID with a blank or random token yields **zero
accuracy**. ARC Prize names it a *"strong, limiting dependency"* and recommends context-derived
conditioning.

⚠️ **In ARC, a puzzle's identity appears in both train and test. In driving, every window is a new
puzzle with no identity.** If the recursion's power depends on knowing *which* problem it is, and
driving cannot say, the mechanism does not transfer.

**The candidate analogue is the SITUATION** (intersection / highway / urban × day/night ×
stop-launch), which refcv5 already strata-codes. ⛔ **But this collides with a binding PI ruling:**
the goal input may not carry the situation classifier's output in any form. ⇒ the conditioning must
be **derived from the scene inside `Block_think`**, never read from the classifier's posterior —
otherwise attribution dies and it is the nav-echo defect again.

⭐ **This is the cheapest and most decisive first experiment** (WP-B): does a context-derived
situation code recover what `puzzle_id` provides? If not, the recursive-reasoner direction is
weaker than it looks and we should know before building it.

---

## 4. The three directions, elaborated

### 4.a LANGUAGE-FREE — `z` alone, no LM anywhere

The recursion runs, the four heads decode, WP-7's selector consumes the score delta, the goal
priors condition the next tact. **No text is produced at all.**

* ⭐ **Cheapest, fastest, and the strongest scientific claim**: if driving improves, reasoning
  improved — with no language model to confound it. ~**6 M** params, comfortably on the tact.
* ⛔ No explanation, no queries. Fails the PI's CoT requirement on its own.
* **Its role: the CONTROL for the other two.** Any gain the language-driven arm shows must be
  measured against this, or we cannot tell language from capacity.

### 4.b LANGUAGE-DRIVEN — an LM in the reasoning path

The LM consumes scene + fan and emits the CoT and goals directly; `z` is its hidden state.

* ⭐ Maximum expressivity, native free-form CoT and QA.
* ⛔ **Latency**: Alpamayo-R1 is 99 ms on an RTX 6000 Blackwell workstation with **71 % of it CoT
  decode** — text generation on a 300–500 ms embedded tact is not a budget, it is a hope.
* ⛔ **Attribution**: a 360 M module in the loop above a 13.7 M planning stack dissolves the
  hierarchy H12 protects.
* ⚠️ Honest reading: **not viable as the primary path on this tact**, and I would not propose it.

### 4.c COMBINED — ⭐ the recommended one

`z` reasons (4.a) and **decodes to text off the tact** (§2.2). The LM renders and, later, answers
queries. **Reasoning is on the tact; language is not.**

| | 4.a free | 4.b LM-driven | **4.c combined** |
|---|---|---|---|
| on-tact params | ~6 M | ~370 M | **~6 M** |
| CoT trace | ✗ | ✓ native | **✓ rendered from `z`** |
| text queries | ✗ | ✓ | **✓ later, port exists** |
| fits 300–500 ms | ✓ | ✗ | **✓** |
| hierarchy preserved | ✓ | ✗ | **✓** |
| confound risk | — | high | **controlled by 4.a as the arm's own baseline** |

---

## 5. Experiments — both of yours, plus the two that gate them

| # | question | why first |
|---|---|---|
| **WP-A** | **do the 11 fields COVER the free-text traces?** Fit text→fields→text on the 4,719 and measure what is lost | ⛔ this bounds §2.3's expressivity trade. **Zero GPU, hours.** If coverage is poor the whole structured route weakens and we should know now |
| **WP-B** | **is there a driving analogue of `puzzle_id`?** Context-derived situation code vs blank vs random | ⛔ §3.3 — the transfer risk. Cheap, and it decides whether the recursion is worth building |
| **WP-C** *(your 5a)* | `z`'s score delta re-ranks the top-5 | ceiling MEASURED: headroom 0.404 m, top-1 32.5 % → 75 % |
| **WP-D** *(your 5b)* | `z`'s goal heads condition the next tact's generator | the second of your two directions |
| **WP-E** | language-free (4.a) vs combined (4.c), same `z` | ⭐ isolates what language adds from what capacity adds |

**Controls, non-negotiable and each SHOWN TO FIRE before any number is quotable:**
`z = 0` reproduces REF-C bit-for-bit · shuffled-fan · shuffled-agent-tokens (refcv5 WP-6 already
specifies it) · constant-`z` floor · fixed-R control against any adaptive-R arm · **four families,
never ADE alone** (refcv5 WP-7 refuses ADE-only in advance).

⛔ Unbanked and required before any number enters the registry: `2510.04871` (TRM),
`2512.11847` (TRM inductive biases), the ARC Prize HRM analysis, `2602.12078` (TRM+Mamba-2),
`2511.02886` (TRM test-time adaptation).
