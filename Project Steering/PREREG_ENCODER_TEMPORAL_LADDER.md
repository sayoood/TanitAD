# PRE-REGISTRATION — MM-E8: how should the ENCODER handle time?

**Written** 2026-08-30, **before any arm runs** · **Author** Master Mind ·
**Trigger** PI: *"Let's check a, b and c. Is it worth to investigate the other ideas
you mentioned?"* · **Tier** T0 (the reads are label-free diagnostics).

```yaml
hypothesis: MM-E8
question: our encoder receives 3 frames CHANNEL-STACKED (9ch) into one patch-embed
          Conv2d, while the PREDICTOR already models time over a 6-timestep window.
          Does the encoder's local motion help, hurt, or do nothing?
one_variable: n_stack (3 -> 1), which forces in_channels (9 -> 3)
held_constant: [corpus, steps 2000, batch 8, window 6, lr, cond_param,
                tac_vocab_version v6.0, init, all loss weights]
```

## 0. ⭐ WHY (a) IS RUN FIRST — it is not "cheapest", it is the GATE

The three options the PI listed are **not independent**; (a)'s outcome decides
whether (b) and (c) are worth building at all:

| if arm A1 (single-frame) reads… | then | consequence for (b)/(c) |
|---|---|---|
| **WORSE than 3-frame, beyond the seed band** | the encoder's local motion IS load-bearing | ⇒ **(b) is the right question** — same information, better representation (tubelet + temporal position embedding). Build it. |
| **BETTER, beyond the band** | folding motion into an appearance latent HURTS | ⇒ ⛔ **(b) and (c) are aimed the wrong way** — they add MORE encoder temporality. The direction is *less*. Do not build them. |
| **WITHIN the seed band** | the stack buys nothing measurable at this scale | ⇒ (b)/(c) chase a benefit not shown to exist; spend on the predictor instead. |

⇒ **Building (b) or (c) before (a) reads would be spending implementation on a
question whose answer might make them pointless.** That is the whole argument for
the order, and it is stronger than cost.

## 1. Arms

| arm | change | cost |
|---|---|---|
| **A0** `enc3f_2k` | incumbent: `n_stack 3`, `--in-channels 9` | ~17 min |
| **A1** `enc1f_2k` | `--newest-frame-only --in-channels 3` | ~17 min |
| **A2** `enc1f_2k_s1` | A1 + `--seed 1` | ~17 min |
| **A0b** `enc3f_2k_s1` | A0 + `--seed 1` | ~17 min |

⭐ **(a) needs NO implementation** — `--newest-frame-only` already exists and
`train_v6_staged.py:4109` REFUSES it unless `--in-channels 3`, so the inconsistent
combination cannot be run by accident.

⚠️ **BOTH SEEDS ARE MANDATORY, not optional.** The 2k seed spread on this recipe is
**0.0358 (drift) / 0.0357 (cos)** — larger than several effects we have chased. A
single-seed pair cannot resolve anything smaller and would produce a number I would
be tempted to believe. *(This is the D-SAFE-CAL and MM-E6 lesson: an unresolvable
comparison must be declared unresolved, not reported.)*

## 2. Reads and outcomes, committed in advance

Instruments: `latentmotion` drift · `meanpred` nrmse + cos_ctr · participation.

| outcome | criterion | consequence |
|---|---|---|
| **STACK-HELPS** | A1 worse than A0 on prediction beyond the seed band | build **(b)**: tubelet + temporal position embedding, as the better encoding of information now shown useful |
| **STACK-HURTS** | A1 **better** beyond the band | ⛔ (b)/(c) dropped. Adopt single-frame; the content/dynamics entanglement hypothesis gains its first direct support |
| **UNRESOLVED** | all deltas inside the seed band | ⛔ **NOT "equivalent".** Report as unresolved-at-this-scale; (b)/(c) stay unbuilt pending a powered test |
| 🔶 **MIXED** | drift and prediction disagree | numbers, no verdict (C160) |

⛔ **What this CANNOT show:** anything about driving (T0), anything at scale (2k),
and — critically — it **cannot distinguish "the encoder ignores the extra frames"
from "the extra frames carry nothing"**. That needs the control in §3.

## 3. The control that makes a null interpretable — REQUIRED before any STACK-HURTS
or UNRESOLVED verdict is quoted

**Shuffled-frame arm**: `n_stack 3`, but the 3 frames permuted in **random temporal
order** per sample. Small loader change.

* If shuffling **degrades** A0 ⇒ the encoder IS reading temporal order, and an
  A1≈A0 result means the *information* is redundant with the predictor.
* If shuffling changes **nothing** ⇒ the encoder never used temporal order at all,
  and A1≈A0 is trivially explained. **A null without this control is uninterpretable**
  — exactly the shape that produced MM-E6's VOID and D-SAFE-CAL's inert arm D.

⚠️ Not run in the first pass; **required before the null is published**.

## 4. ⛔ Answering the PI's second question: which OTHER ideas are worth it

| idea | verdict | reason |
|---|---|---|
| **Factorised space-time attention** (TimeSformer) | ⛔ **not a separate arm** | it is an *efficiency technique for* (b), not an alternative to it. It would be how (b) is implemented affordably, decided only if (b) is built. |
| **Two-pathway / SlowFast** (c) | ⚠️ **lowest priority** | it targets **multi-rate** motion. Our encoder window is **3 frames = 0.2 s**; there is no rate separation to exploit at that span. It would first need a longer encoder window, making it a *second* change stacked on an unvalidated first one. |
| **Single-frame + stronger predictor** (a) | ⭐ **run now** | free, and it is the gate above |
| **Tubelet + temporal position embedding** (b) | **conditional** | build **only** on STACK-HELPS |

⇒ **My answer to the PI: (a) now; (b) only if (a) says the stack matters; (c) not
worth it at a 0.2 s encoder window; factorised attention is not a separate question.**

## 5. Gates

`pytest -q` green. Controls that must read known values: the constant control in
`latentmotion` must read **exactly 0.0000**, and the drift positive control must land
in its known band — a panel where those drift is a broken rig, not a result.
Report **n** and the seed spread beside every delta.

---

## ⛔ OUTCOME — MM-E8 CANNOT BE READ AT 2k. THREE INDEPENDENT REASONS, ALL MEASURED AFTER LAUNCH.

**Read 2026-08-30 from the two completed 3-frame arms.** The ladder is stopped, not
because an arm failed but because **the design was unfit and I did not check before
launching.**

### 1. The seed spread is 1.6–2.6× larger than the number I put in the prereg

§1 committed *"the 2k seed spread on this recipe is 0.0358 (drift) / 0.0357 (cos)"*.
⛔ **That number was INHERITED from the P0 EMA arms — a different recipe — and I
built the resolution criterion on it.** Measured on `enc3f_2k` vs `enc3f_2k_s1`,
which differ only in seed:

| read | seed 0 | seed 1 | **measured spread** | assumed |
|---|---|---|---|---|
| drift r | 0.2690 | 0.3254 | **0.0564** | 0.0358 |
| cos_ctr | 0.1469 | 0.0523 | **0.0946** | 0.0357 |
| nrmse | 0.9957 | 0.9989 | 0.0032 | — |

⚠️ **The cos spread (0.0946) is LARGER THAN EITHER ARM'S VALUE** (0.147, 0.052). No
encoder effect of plausible size survives that.

### 2. Both arms **ARE the mean predictor**

`nrmse` 0.9957 / 0.9989, verdict *"IS the mean predictor"* on both. This is the
documented 2k scale-fact — every 2k arm sits at the mean-predictor floor — so the
prediction axis has no headroom in which an encoder change could show.

### 3. The k=1 shortcut is ACTIVE at 2k and ABSENT at 30k (MM-E9)

`o5_step1/o5_stepK` = **0.31–0.36 at 2k**, **1.00–1.11 at 30k**. The single-frame arm
removes the encoder's motion **and** the overlap, and the overlap is a 2k-only
phenomenon. Even a resolvable 2k result would answer a question that does not exist
at production scale.

*(A fourth, separate defect: `--newest-frame-only` is INERT in `train_v6_staged.py`
— MM-C11 — so the single-frame arms could not run regardless.)*

## Verdict, per the committed table

**UNRESOLVED at this scale — and NOT "equivalent".** §2 committed that deltas inside
the seed band are unresolved rather than null, and that commitment is what makes this
reportable instead of embarrassing: had I quoted a 2k encoder result it would have
been noise dressed as a finding.

⇒ **(b) and (c) stay unbuilt.** The gate in §0 never opened — it cannot open at 2k.
The question needs **30k arms**, where the shortcut is absent, the mean-predictor
floor is cleared, and the spread is (unknown and must be measured, not inherited).

## ⭐ The lesson, which is mine

**MEASURE THE SEED SPREAD BEFORE DESIGNING THE RESOLUTION CRITERION, NOT AFTER.** I
wrote a prereg whose entire power analysis rested on a number carried over from a
different recipe, marked it as a fact, and launched four arms on it. The
`INHERITED`-vs-`MEASURED` rule exists for exactly this and I applied it to other
people's numbers all day while quoting my own from memory.

⚠️ **And the cheap check existed**: two seeds of the incumbent is 34 minutes and
would have said *"this rig cannot resolve your question"* before any arm ran. **A
two-seed power probe belongs in front of every ladder**, not inside its outcome
section.
