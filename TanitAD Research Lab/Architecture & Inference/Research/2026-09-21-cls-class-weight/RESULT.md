
<!-- CLS-CLASS-WEIGHT-CAPABILITY-2026-09-21 -->

### ⭐ 2026-09-21 — `H-BOXCLS-1` could not have launched whatever the PI decided: the class-weighted `cls` term DID NOT EXIST. It does now — mutation-proven 4/4, default bit-identical

MEASURED by me, CPU only, no GPU, no training
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-21-cls-class-weight/`).

`d7fa093` pre-registered the cls re-weighting arm and its gate passed; `731ecd7` widened it to both
heads. ⛔ **But two probes show the option it specifies does not exist:** no CLI flag mentions class
weighting, and `slot_set_loss` calls a plain `cross_entropy(..., reduction="sum")` with **no
`weight=`**. The arm was blocked on ME, not on the PI — *"every implementation, measurement,
instrument fix and banked artifact around them is mine, and none of them wait."*

⛔ **AND THE FOCAL LOSS WE ALREADY HAVE IS THE WRONG TOOL, for a stated reason.**
`refcv6_diffusion.focal_cls_loss` exists and the "already exists, never re-implement" rule is
tempting. Its own docstring refuses: it is a **SIGMOID** loss on `[B, N]` mode logits ported from
DD, and *"swapping in `cross_entropy` reproduces neither the gradient nor the scale"*. The slot
`cls` term is a 10-way **SOFTMAX**. Using it would change the loss **FAMILY**, not the weighting —
a second variable inside a one-variable arm. The prereg's `cross_entropy(weight=…)` was right.

**What landed:** `slot_set_loss(..., cls_class_weight=None)`, forwarded by **both** callers —
`box3d_set_loss` (the 3-D head `s1_pass` scores) and `agent_losses` (the 2-D seam feeding the
tactical decoder). Default `None` ⇒ **bit-identical** to today for every existing arm.

⭐ **THE SUBTLE PART IS THE DENOMINATOR, AND IT IS WHAT MOST OF THE GUARD PINS.**
`cross_entropy(reduction="sum", weight=w)` returns `Σ wᵢ·Lᵢ`, so dividing by the item COUNT would
make the term's **scale** move with the weights as well as its per-class emphasis — the `cls` term
would compete differently against `centre`, `size` and the rest for reasons nobody asked for, and
any arm using it would be confounded between *"the classes were re-balanced"* and *"the cls term
got louder"*. The normaliser is therefore `Σ wᵢ`. ⭐ That buys the identity control for free: at
`w = ones` the weight sum IS the count, so a uniform weight is bit-identical to passing nothing —
MEASURED at `rtol=0, atol=0`.

**Guard: 9 tests, mutation-proven 4/4** (`raw/mutation_proof_cls_weight.json`) — the denominator
ceasing to follow the weights; `weight=` dropped from the call; and **each** caller ceasing to
forward. Baseline green, every arm RED with named catchers, all three files restored
byte-identical.

⚠️ **A fixture defect caught before it shipped, and it is the reusable part.** A first hand-check
used `randint` over 12 slots and reported that a non-uniform weight **changed nothing** — which
reads exactly like a dead weight path. The cause was that class 0 never appeared in that draw.
⇒ the fixture is now DETERMINISTIC and a **same-breath control asserts the class counts**
(`7 / 4 / 1`), so "a non-uniform weight moves the loss" can never pass by reweighting an absent
class. *I nearly "fixed" working code on the strength of a badly built check.*

⚠️ **NOT YET DONE, and the arm still cannot launch:** the CLI flag, the **train-split** inverse
frequencies, and stamping the resulting 10-number vector into `config.json`. The vector must be
**stamped, not recomputed**, or two runs at the same flag are not the same arm. The eval-side
counts already measured (`automobile` 1155 … `stroller` 2, ratio 577.5 : 1) are a 1,482-pair
**eval** sample and are **not** admissible as the train weights. That derivation is the next
increment.
