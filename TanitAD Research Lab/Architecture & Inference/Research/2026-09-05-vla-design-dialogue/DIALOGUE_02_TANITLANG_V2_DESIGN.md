# TanitLang — the language extension of REF-C's *mechanism*, not a language model beside it

**Written 2026-09-05, TanitAD_TrainingFlyWheel · v2, supersedes `REFCV5_VLA_EXTENSION_PLAN.md`
as the design; that document's interfaces (§2), budgets (§3) and work-package skeleton (§6)
are inherited unchanged.**
**Status:** DESIGN — nothing built. **Evidence classes marked per claim.**

---

## 0. The critique, and what was actually wrong

> *"your proposal is mainly to take the language model, the other modules are very small and
> im not recognizing the refc design in the plan."* — PI

⭐ **The critique is not about parameter counts. It is about MECHANISM.** The banked plan gives
the LM the *reasoning* and REF-C the *features*: trunk → projector → frozen LM + LoRA → write-back
heads. Read that diagram and REF-C is an encoder. **REF-C's actual design — enumerate a fan of
hypotheses, refine them by truncated diffusion, score and select — appears nowhere in the language
path.** A reader is right not to recognise it.

⇒ **This document inverts that.** The language layer is built *out of REF-C's own mechanism*. The
fan reasons; the language model renders. That is also the only reading of the PI's H12 —
*"language as additional part and **not as the core**"* — that has a testable consequence.

⚠️ **The PI's brief already answers the Master Mind's three open forks**, and this design follows
those answers rather than re-opening them: reasoning is *"grounded in… and consistent to the
trajectory hypotheses"* (fork 1 → the fan), the module *"complements our current strategic and
tactical layer"* (fork 2 → the hierarchy does **not** grow), and the first product is
*system-initiated CoT explaining behaviour, plus goal emission* (fork 3).

---

## 1. The thesis, in one paragraph

**REF-C already enumerates 117 trajectory hypotheses and scores them. TanitLang reasons over the
SAME 117 hypotheses in a shared latent space, using a short latent chain-of-thought; the EXISTING
scorer still commits; and a small frozen LM verbalises the committed choice on demand.** The
language part therefore *cannot* hallucinate a manoeuvre the plan does not contain — not because a
loss discourages it, but because **the only things it can talk about are fan members that exist**.
That is the same structural guarantee the anchor fan gives the planner, extended to language.

---

## 2. ⭐ The evidence that makes this the right split, not merely a tidy one

**PUBLISHED — the Exploration–Execution trade-off** (`2602.01148`, *Capabilities and Fundamental
Limits of Latent Chain-of-Thought*). Latent CoT is **strong at exploration** (97.0 % ProsQA) and
**weak at computation/execution** (34.1 % GSM8K). The authors prove the trade-off is governed by
*decisional certainty*: high certainty gives precise execution but inhibits exploration; low
certainty enables search but accumulates error. **They also prove curriculum learning is
NECESSARY — direct training provably fails on distributional mismatch.**

⭐ **Map that onto our stack and the division of labour is forced, not chosen:**

| phase | who does it | why |
|---|---|---|
| **EXPLORATION** — which hypotheses are live, what could happen | **latent CoT over the fan** | exactly where latent reasoning is measured strong |
| **EXECUTION / COMMITMENT** — pick one | **the existing REF-C scorer (unchanged)** | exactly where latent reasoning is measured weak |
| **VERBALISATION** — say it in words | **frozen small LM, on demand** | off the control path entirely |

⚠️ And the curriculum result is not a warning for us but a fit: **the v6/v7 staged ladder
(S-W → S-T → S-S → S-J) already IS a curriculum.** A from-scratch joint training of this module
would hit precisely the distributional mismatch the paper proves fatal.

**Converging frontier work, all PUBLISHED:**

| work | what it establishes | what we take |
|---|---|---|
| **LaST-VLA** `2603.01928` | latent spatio-temporal CoT in continuous hidden space; distils *"dynamic foresight from world models"* into the latent space. 91.3 PDMS NAVSIM v1 / 87.1 EPDMS v2 | ⭐ they must **import** world-model foresight; **we have a world model natively.** This is our structural advantage, not a gap |
| **CoVT** `2511.19418` | continuous visual thought within **~20 tokens**, distilled from lightweight experts | the latent-CoT token budget is ~20, not hundreds — this is what makes 300–500 ms feasible |
| **SOLVE** `2505.16805` | a **trajectory bank of 36 candidates**, k-means per nav command, selection by similarity | precedent that bank-and-select is competitive; ⭐ our fan is 117 and already exists |
| **Neuro-Symbolic Drive** `2606.23938` | supervises VLA reasoning with **rule-grounded traces from classical planners** | our "classical planner" is the unicycle-feasible fan + the four metric families |
| **Do Latent Tokens Think?** `2512.21711` | causal + adversarial analysis of continuous thought | ⛔ the control literature: latent CoT is **not** a free pass on faithfulness |
| **Alpamayo-R1** `2511.00088` Tbl 14 | 99 ms on an **RTX 6000 Blackwell workstation**, **71 % of it CoT decode** (70/99 ms) | ⚠️ the number our latency plan must not quote as embedded (`D-VLA-LAT1`) |

⛔ **The Alpamayo figure is the reason text decode cannot sit on the tact.** If 71 % of a
workstation's 99 ms is CoT decode, an embedded text CoT at 300–500 ms is not a budget, it is a hope.
**Latent CoT on the tact + text on demand is what makes the PI's tact reachable.**

---

## 3. Architecture

```
                    REF-C v5 trunk  (frozen; already running, ~20 M)
                              │
              ┌───────────────┼────────────────────────────┐
              ▼               ▼                            ▼
        pooled [160×d]   ctx / z_tac / z_str          anchor fan  [117 × 8 × 2]
              │               │                            │
              └───────┬───────┘                            │
                      ▼                                    │
         ┌────────────────────────┐                        │
         │  FanReasoner  (~8 M)   │◄───── cross-attends ───┘   ⭐ THE FAN IS THE INPUT
         │  latent CoT, K=16 toks │
         │  2 truncated steps     │        (REF-C's own idiom: enumerate → refine)
         └───────────┬────────────┘
                     │  per-anchor rationale latents  [117 × d_r]
        ┌────────────┼─────────────────────────┐
        ▼            ▼                         ▼
  goal heads    ⭐ EXISTING REF-C SCORER   consistency head
  (v7 tokens)      — UNCHANGED, COMMITS —   (agreement signal)
        │                   │                         │
        └───────────────────┴──────────┬──────────────┘
                                       ▼
                        ┌──────────────────────────┐
                        │ frozen small LM + LoRA   │   ≤1 Hz / on demand
                        │ SmolLM2-360M             │   ⛔ OFF THE TACT
                        └──────────────────────────┘
                                       ▼
                          text CoT · answers to queries
```

### 3.1 `FanReasoner` — the new module, and the only one on the tact

* **Input:** the trunk's `pooled` tokens, the layer latents, the **fan itself**, nav, `v0`.
* **Mechanism, deliberately REF-C's:** K = 16 latent "thought" tokens are initialised from the
  fan's own summary, then **refined for 2 truncated steps** — the same `decoder_steps=2` idiom, and
  the same reason (⚠️ MEASURED this session: beyond the trained operating point the fan *diverges* —
  sel-ADE 0.654 → 1.529 m at steps 8, SEPARATED. The refinement budget is a trained property, not a
  free knob).
* **Output:** a rationale latent **per anchor**, so every hypothesis carries its own "why".
* **Params (ESTIMATED, arithmetic in §5):** ~8 M. ⚠️ 4× the whole tactical brain and 118× the
  strategic brain — deliberately, and §6 states the test that decides whether that is warranted.

### 3.2 What is NOT changed

⛔ **The scorer still selects.** ⛔ **The fan is still the planner.** ⛔ **The hierarchy does not
grow.** The 2.15 M cascade keeps its authority; TanitLang adds a *view* of the fan, not a competing
one. That is fork 2 answered as the PI answered it.

### 3.3 The LM, and why it is small and late

Frozen **SmolLM2-360M** + LoRA r=16 (~3.5 M trainable). It receives the **committed** anchor, its
rationale latent, and the grounding tokens — and renders text. ⛔ It never sees the fan before
selection and cannot change the choice. Running ≤1 Hz keeps 71 %-of-latency-is-decode off the tact.

⚠️ **Qwen3-0.6B is the alternative** if 360 M proves too weak for QA. Both stay ≤1 B; SmolLM2 is
the default because `2504.05299` Finding 1 (PUBLISHED) reports small LMs pair better with small
encoders, and our trunk is ~20 M.

---

## 4. ⭐ The USP: hallucination is structurally impossible, and that is testable

The rationale is not free-generated text scored for plausibility. It is a **latent attached to a
fan member that exists**, and the text is a rendering of that latent.

| failure the field has | why it cannot occur here |
|---|---|
| CoT names a manoeuvre the plan does not execute | the rationale is *indexed by anchor*; naming a manoeuvre means naming an anchor, and every anchor is a real, unicycle-feasible trajectory |
| CoT is a post-hoc narration with no causal link | the rationale latents are **inputs to the consistency head**, and the anchor they index is the one the scorer ranked |
| CoT justifies a wrong decision confidently | the rationale for the *selected* anchor can be compared against the rationale for the *runner-up* — a disagreement signal the monitor emits |

⛔ **AND THE CONTROLS SHIP WITH IT — this is the half the field skips.** From `2512.21711`
(latent tokens are not automatically faithful) and from our own measured echo defects:

1. **shuffled-fan** — serve the rationale head another window's fan. If the rationale does not
   degrade, it is not reading the fan and the grounding is decorative. *(Already built for nav this
   session; same shape, and the roll-not-permutation detail carries over — a random permutation
   leaves ~1/B items holding their own and silently weakens the control.)*
2. **shuffled-nav** — already implemented and mandatory.
3. **rationale-swap** — render text from the *runner-up's* rationale. If the text is unchanged, the
   text is not reading the rationale.
4. ⭐ **constant-rationale floor** — a fixed rationale latent for every window. Any metric on which
   the real rationale does not beat this is a metric measuring the *scene*, not the reasoning.

⚠️ **Every one of these must be SHOWN TO FIRE on a deliberate-regression arm before any number from
the module is admissible.** A control that cannot detect the failure it names proves nothing —
measured three times in this programme in a single day.

---

## 5. Budgets

**Parameters (ESTIMATED, arithmetic shown):**

| part | count |
|---|---|
| `FanReasoner`: 4 layers, d=256, cross-attn to 160 pooled + 117 anchors | 4·(4·256² + 2·256·1024) ≈ **3.1 M** |
| fan/anchor encoder + rationale projection | ≈ **1.2 M** |
| K=16 latent tokens + 2-step refiner | ≈ **2.6 M** |
| goal + consistency heads (v7 vocab: 8+8+7+8, geometric goal 3) | ≈ **1.1 M** |
| **FanReasoner total (on the tact)** | **≈ 8 M** |
| frozen SmolLM2-360M (off the tact) | 362 M |
| LoRA r=16 + projector | ≈ **5 M** trainable |
| **resident total** | **≈ 375 M** — ⭐ well under 1 B |
| **trainable total** | **≈ 13 M** — a dev-box fine-tune, not a cluster job |

**Latency (ESTIMATED — ⛔ every figure replaced by WP-0 on Thor; `max_memory_allocated` is the only
admissible memory probe there):**

| on the tact (300–500 ms) | est. |
|---|---|
| trunk — already paid by REF-C | 0 (shared) |
| FanReasoner, 16 tokens × 2 steps, no text decode | **~25–45 ms** |
| **fits with large margin** | |
| off the tact (≤1 Hz) | |
| LM text render, ~60 tokens | ~200–400 ms — ⚠️ **budgeted OUTSIDE the tact, by design** |

---

## 6. Hypotheses, with both outcomes committed BEFORE any arm runs

| id | hypothesis | ⭐ if TRUE | ⛔ if FALSE |
|---|---|---|---|
| **H-LANG-1** | A rationale latent indexed by anchor predicts the scorer's choice better than a constant-rationale floor | the fan carries reasoning structure; proceed | the "rationale" is scene-reading; the module is an explainer only — say so and stop |
| **H-LANG-2** | The fan is where the reasoning is: shuffled-fan degrades the rationale materially | grounding is real | ⛔ grounding is decorative; the whole USP fails and must be reported as failed |
| **H-LANG-3** | Latent CoT reaches the tact where text CoT cannot (the 71 % decode result transfers) | the split is validated on our hardware | re-price; possibly no CoT on the tact at all |
| **H-LANG-4** | A 8 M FanReasoner beats a 10× larger strategic brain with no language part | capacity was not the strategic brain's problem — *structure* was | ⚠️ the hierarchy may genuinely need to grow; that is a PI decision, and this is the experiment that earns the right to ask |
| **H-LANG-5** | Distilled Alpamayo CoT improves the rationale over our own geometric labels | distillation is worth its cost | our labels suffice; skip the distillation entirely |

⭐ **H-LANG-4 is the one that answers the Master Mind's fork 2 empirically instead of by argument** —
and it is cheap: two v7-tiny arms, no full-scale run. **I would run it first**, because if a small
structured reasoner beats a big unstructured strategic brain, the 0.063 % figure stops being an
embarrassment and becomes a finding.

⚠️ **A T0 regression does not condemn any of these arms.** PUBLISHED (DINO-WM App. A.4.1): over 92×
data, prediction fidelity moved **+4 %** while control competence moved **11.5×**. Our corpus sits in
their n = 1,000–5,000 band, where prediction has saturated and control has not. **Both directions of
that asymmetry are committed here before numbers exist.**

---

## 7. Work packages, in dependency order

| # | package | gate |
|---|---|---|
| **WP-0** | Thor latency + `max_memory_allocated` for a 16-token, 2-step reasoner | every ESTIMATE in §5 becomes MEASURED or the design is re-priced |
| **WP-1** | `FanReasoner` + the four controls of §4, on the v7-tiny rig | ⛔ **all four controls SHOWN TO FIRE** on a deliberate-regression arm, or nothing proceeds |
| **WP-2** | H-LANG-4 — 8 M structured reasoner vs 10× strategic brain | answers fork 2 with a number |
| **WP-3** | Goal write-back through v7 vocabulary, monitor mode only (heads zero-init, gated shut) | the consistency signal exists and is non-inert |
| **WP-4** | LM attach, frozen + LoRA, text off the tact | text renders the *committed* choice; rationale-swap control fires |
| **WP-5** | Alpamayo CoT distillation (H-LANG-5) | only if WP-1–4 hold |

⛔ **Influence mode — the LM's goals actually steering the planner — is not in this plan.** It is a
separate PI decision, taken only after the monitor-mode consistency signal has been shown to be
real. That ordering is the whole difference between "an explanation channel" and "a second planner
competing with the tactical brain".

---

## 8. Honest limits

* ⚠️ **The nav channel is 1.313 bits per clip** (MEASURED today: `NAV_FOLLOW_ROAD` 2,993 /
  `NAV_TURN_R` 902 / `NAV_TURN_L` 824 over 4,719 — **0.8 KiB across the whole corpus**), and it is
  an **oracle** (`ego-future`, 4,719/4,719). Nav cannot be what justifies the parameters, and any
  nav-conditioned claim needs shuffled-nav beside it.
* ⚠️ **We have no question corpus.** "Question queries" is a data gap, not a modelling gap. The
  system-initiated CoT is buildable today; free-form QA is not.
* ⚠️ **The trunk is currently action-deaf** (`H-ARCH-ACTINS`: closed-loop vs hold-action gap ~1 %).
  Reasoning layered above a predictor that ignores its own action channel would look fluent regardless.
  ⭐ **I would want `action_sensitivity` moving before WP-3**, and the in-training val round built
  this session emits exactly that number every 100 steps.
* ⚠️ Alpamayo CoT alignment, corpus coverage and licence are inherited from the banked research
  (§0.2 there) and are **not re-verified in this document**.

---

## 9. Sources

Banked primaries in `TanitAD Research Lab/Library/papers/`. New to this document and **not yet
banked** — ⛔ they must be added by `tools/kb_add.py` before any of their numbers enters the
registry: `2602.01148` (latent-CoT limits — the load-bearing one), `2603.01928` (LaST-VLA),
`2511.19418` (CoVT), `2505.16805` (SOLVE), `2606.23938` (Neuro-Symbolic Drive), `2512.21711`
(latent-token faithfulness), `2511.12405` (VLA-R).
