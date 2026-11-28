# The box head's class output has collapsed — and the alignment spectrum was the wrong head

**Evidence class:** MEASURED (ours). **Compute:** CPU only, no GPU, no training. **Checkpoint:**
A8 `ckpt_5000`. **Date:** 2026-09-21.

Three findings, in the order they forced each other out. The third is a **correction to landed
work** and is logged as `RETR-2026-09-21-ALIGNMENT-MEASURED-ON-THE-WRONG-HEAD`.

---

## 1. SIZE is indistinguishable from a single global constant

`c920f15` established that the size **target** is 75–78 % predictable from target-side geometry and
concluded the head was not using available signal. That bounds what *could* be known; it never asked
what the head's own `l`/`w` are worth. A SPEC cannot commit a success criterion without that number.

Matched pairs from `match_slots`, whose Hungarian cost uses centre/cls/presence but **NOT size**
(`agent_slots.py:475-477`) — so scoring size on these pairs is not circular, the same argument that
made velocity the clean place to test alignment in `align_vs_skill.py`. 1,482 pairs / 35 episodes;
848 pairs / 18 episodes scored; every median computed fit-side only and applied to the disjoint half.

| arm | `l` MAE (m) | `w` MAE (m) |
|---|---|---|
| **O — the head** | **1.02306** | **0.31953** |
| `F_GLOBAL` — one median for everything | 1.03962 | 0.32174 |
| `F_CLSHAT` — median per the head's **own predicted** class | 1.03962 | 0.32174 |
| **`F_CLSGT` — median per the **true** class** | **0.39653** | **0.10100** |

* The head **does not beat a single constant** on either component — `l` gain 0.01656 CI
  [−0.00335, 0.04041]; `w` gain 0.00221 CI [−0.00206, 0.00760]. Both straddle zero.
* A **zero-parameter** median table on the true class beats the head **2.6× on `l`**
  (CI [−0.9695, −0.3521]) and **3.2× on `w`** (CI [−0.3448, −0.1193]), separated.
* ⭐ **`F_CLSHAT` reads identical to `F_GLOBAL` to five decimals.** That is only possible if the
  head's own class prediction carries no discrimination whatever — which forced §2.

Per-component, never pooled: `l` and `w` differ 3× in scale, and a pooled size error would be
dominated by `l` and hide `w` entirely — the defect that hid a completely dead velocity component
until `4a7166a` split it.

---

## 2. ⛔⛔ Total class collapse — one class of ten, everywhere

| | |
|---|---|
| matched pairs emitting `automobile` | **1,482 / 1,482** |
| **all** slots, 20 windows, matched or not | **2,000 / 2,000** |
| distinct classes predicted | **1 of 10** |
| top-1 accuracy | 0.77935 |
| majority-class baseline | **0.77935** — equal to 5 dp |
| mean top1 − top2 softmax margin | **0.59198** |
| GT `person` → predicted `person` | **188 → 0** |
| GT `rider` → predicted `rider` | **50 → 0** |
| imbalance, majority : rarest | **577.5 : 1** |

Accuracy equalling the majority baseline **exactly** is the signature of a constant predictor, not
an independent coincidence — and the 0.59 margin says the head is **confidently** wrong rather than
sitting on a knife edge. Widening from matched pairs to **every slot** matters: "collapsed on the
targets it was scored on" and "collapsed everywhere" are different claims, and only the second
indicts the head.

⚠️ **238 vulnerable road users are emitted as cars.** Stated plainly because it is a safety
property, not a metric artifact.

### The mechanism is named by the code's own comment — and it is a hypothesis, not a conviction

`cls` carries weight **1.0** in `SLOT_LOSS_W` and A8 ran `--w-agent 1.0`, so this is **not** a
down-weighting artifact. But the `cls` term is a plain `cross_entropy` with **no `weight=`**
(`agent_slots.py:591`), while presence two hundred lines earlier receives `NO_OBJECT_W = 0.1`,
introduced with the reason (`:230-231`):

> unmatched slots vastly outnumber matched ones, and an unweighted BCE simply learns "always empty"

**The same argument was never applied to `cls`**, on a 577.5:1 target — and the head learned
"always `automobile`", exactly as that sentence predicts for its own field.

⛔ **NOT PROVEN.** This is `ckpt_5000`. A collapse at 5,000 steps can be a training-**duration**
artifact rather than a loss-**design** one, and nothing here separates them. The separating
experiment is a class-weighted arm or a longer run — **GPU, and therefore the PI's call**.

---

## 3. ⛔⛔ The alignment spectrum was measured on a different head

`148ceb9`'s probe selected the layer to hook as *"the first `Linear` whose `out_features` equals the
total slot width"*. `SLOT_SLICES` sums to **21**, and the first match is **`core.agent_head`** — the
2-D `AgentSlotDecoder`, **16 queries**. But every skill, floor and defect number in this whole
investigation is scored on **`perception.box_dec`**, a `Box3DSlotDecoder` (`box3d_head.py:227`)
emitting **100 slots**, read as `box_slots` at `s1_pass.py:307` and produced at
`refcv6_perception_branch.py:426`.

`Box3DSlotDecoder` **subclasses** `AgentSlotDecoder` and replaces `self.head` with
`nn.Linear(d_model, SLOT3D_WIDTH)` (`:247`) — so the parent's width still matches a **different live
module**, and the loop `break`s on it.

| field | CORRECTED (`box_dec`, 100 slots) | quoted (`core.agent_head`, 16) | ratio |
|---|---|---|---|
| `occluded` | **92.21126** | 0.284 | **×324.7** |
| `cy` | 74.25487 | 56.7131 | ×1.31 |
| `cx` | 48.74788 | 24.77866 | ×1.97 |
| `yaw_cos` | 26.55927 | — | — |
| `yaw_sin` | 2.12932 | — | — |
| `presence` | 1.90992 | — | — |
| `cls` | 1.61041 | — | — |
| `v_rel_x` | 1.52587 | 7.258 | ×0.21 |
| `cz` | 0.35016 | *(absent — 2-D head)* | — |
| `v_rel_y` | 0.20805 | 0.08807 | ×2.36 |
| `l` | 0.19640 | 1.38458 | ×0.14 |
| `w` | 0.04704 | 0.2462 | ×0.19 |
| `h` | 0.03523 | *(absent — 2-D head)* | — |
| `yaw_rate_rel` | 0.02481 | 0.02459 | ×1.01 |

⛔ **A RE-ORDERING, not a rescaling.** `occluded` moves from second-**lowest** to **highest**;
`l` and `w` fall to the bottom. Per-claim exposure is in the retraction — in short: `3e3dac9` and
the predictability half of `c920f15` **survive** (target-side probes, no alignment term);
`148ceb9`'s presence conclusion **survives and is now measured correctly**
(`ratio_presence_to_median` **1.2517**, well above the 0.3 that would have indicted it);
`4a7166a`'s validation is **void as performed** because it joined alignment from one module to
skill from another, and its "82× apart" is really **7.3×**; and `c920f15`'s *"alignment does not
identify defects"* is **reversed in its evidence**.

⚠️ **Alignment is NOT thereby re-promoted to a diagnostic.** On the corrected numbers `cls` sits
mid-spectrum at **1.610** while being **totally collapsed** — a counter-example from §2 of this same
document. Honest status: **UNKNOWN and re-openable**, not "validated".

### The durable fix is an identity, not a better width

A width match is what failed, so a stricter width is not the repair. `align_fix.py` selects
`box_dec.head` **by name** and then asserts, per window, that the hooked tensor's slot count
**equals the slot count of the `box_slots` that same forward emitted** — a property no other module
can satisfy at any width. It fired immediately: `[1, 16, 256]` hooked against `box_slots [100, 4]`.

⭐ **What caught it was a POWER check, not a correctness check.** The class probe reported
`n_pairs = 248` where every sibling instrument reports **1,482** on the same 60 windows. The 6×
shortfall was the visible symptom; `max(row) = 99 > 16` was the proof. ⇒ **a count that disagrees
with a sibling instrument on the same input is a defect report**, and it is cheaper than any audit.

---

## 4. Consequence for `D-S1-DEP-BOX`

The blocker is **one defect with several faces**, not a list of independent items:

| face | evidence | what it needs |
|---|---|---|
| `cls` collapsed to one class | 2,000/2,000 slots; accuracy == majority baseline | class-weighted / focal `cls` ⇒ **GPU, PI's call** |
| `l`/`w` = global constant | loses to a true-class median lookup 2.6× / 3.2× | **downstream of `cls`** — likely free once class works |
| `presence` degenerate | no per-slot signal (`286e0d3`); NOT misalignment (ratio 1.25) | **GPU, PI's call** |
| `occluded` worse than free | loses to a 2-parameter read of its own azimuth (`6c5fb62`) | a **DECODE** change — zero training |

⇒ the SPEC's first arm is **not** "add size capacity". It is **weight the `cls` term the way
`NO_OBJECT_W` already weights presence** — with the deliberate-regression arm being the unweighted
configuration measured here, and the success criterion stated in **balanced accuracy against the
`1/K` constant-predictor value**, never raw accuracy, which a fully collapsed head already scores
**0.779** on.

## Manifest

| file | what |
|---|---|
| `code/size_floor.py` → `raw/size_floor.json` | §1 — the zero-parameter size floors |
| `code/cls_collapse.py` → `raw/cls_collapse.json` | §2 — the collapse, matched pairs and all slots |
| `code/align_fix.py` → `raw/align_fix.json` | §3 — the corrected spectrum and the identity guard |

## 5. ⚠️ Open, and running: LOSS defect or REPRESENTATION defect?

§2 names a suspect; it does not convict one. Two different defects produce an identical collapse and
demand **opposite** fixes — a class-weighted `cls` term (loss), or work upstream on queries/capacity
(representation). The separating test is a linear probe from `box_dec`'s own per-slot features to the
GT class: **succeeds ⇒ the signal is there and the objective is at fault; fails ⇒ re-weighting alone
would not fix it.**

⛔ **The first two attempts are both inadmissible, for two different reasons, and both are recorded
rather than discarded:**

1. **Wrong head** — it used the same width selector §3 retracts, binding to `core.agent_head`
   (16 queries) and collecting **248** rows where siblings collect 1,482. Void.
2. **The probe shared the defect it was checking for.** Re-run on the correct head and properly
   powered (n_fit 1,240, d 41 after a FIT-only PCA, so n ≥ 5d), an **unweighted** one-vs-rest ridge
   read balanced accuracy **0.11111** against a constant-predictor control of **exactly 1/9 =
   0.11111** — i.e. **the probe collapsed to a single class too**. On a 78 %-majority target that is
   what an unweighted estimator does, so *"the features carry no class"* and *"my estimator
   collapsed"* are indistinguishable from that number. ⛔ **A check that shares the defect it checks
   for is green forever** (CLAUDE.md, 2026-09-07) — here it would have been *red* forever, which is
   the same error with the sign flipped, and it would have blamed the representation on the strength
   of an estimator artifact.

⇒ the admissible form carries a **class-balanced (inverse-frequency) arm beside the unweighted
one**, with weights counted on the FIT split only. If the balanced arm clears the `1/K` control
while the unweighted one does not, the probe has **reproduced the head's failure and then removed
it with the exact remedy the SPEC proposes** — evidence for the loss reading from two directions at
once. That arm is running; its result lands separately rather than being previewed here.
