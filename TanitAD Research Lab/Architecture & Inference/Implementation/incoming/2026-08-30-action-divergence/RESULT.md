# MM-E10 — ⭐⭐ ACTION-INSENSITIVITY CONFIRMED. H-ARCH-ACTINS resolved; the competing account REFUTED.

**Measured** 2026-08-30 on Thor · **T0-DIAGNOSTIC**, label-free, no training ·
raw `actdiv.json` md5 `c80398b49ca3065f37e4ad53bbc92cf4` · 24 val clips, 144
windows/arm, 8 action variants.

## The question this settles

`H-ARCH-ACTINS` proposed that the v7 T1 floor is **action-insensitivity** — the
model barely uses its action input — rather than bad driving. ⚠️ It was registered
with a **competing account that the banked numbers could not separate**: at heading
MAE ≈ 95° and ADE ≈ 14 m the rollout might be *degenerate*, so that no action choice
could move any metric. Both accounts predict the ~1 % closed-loop-vs-hold-action gap.

**The discriminator: hold the SCENE fixed, vary the ACTION, and measure against a
scale.** A raw "the output barely moved" is meaningless; the probe reports a ratio
against how much the output moves when the SCENE changes.

## Result

| arm | h | action_spread | scene_spread | **ratio** | C0 |
|---|---|---|---|---|---|
| `emao14_30k` | 1 | 0.00075 | 0.17983 | **0.00416** | PASS |
| `emao14_30k` | 2 | 0.00000 | 0.37167 | **0.00001** | PASS |
| `emao14_30k` | 4 | 0.00000 | 0.37167 | **0.00001** | PASS |
| `o14fut30k` | 1 | 0.00111 | 0.27194 | **0.00408** | PASS |
| `o14fut30k` | 2 | 0.00001 | 0.42592 | **0.00001** | PASS |
| `o14fut30k` | 4 | 0.00001 | 0.42592 | **0.00001** | PASS |
| `postrain30k` | 1 | 0.00139 | 0.23434 | **0.00595** | PASS |
| `postrain30k` | 2 | 0.00001 | 0.39906 | **0.00002** | PASS |
| `postrain30k` | 4 | 0.00001 | 0.39906 | **0.00002** | PASS |

⭐⭐ **VERDICT: ACTION-INSENSITIVITY, CONFIRMED AND EXTREME, ON ALL THREE 30k ARMS.**
At **h=1 the action moves the prediction 0.4–0.6 % as much as the scene does.** At
**h=2 and h=4 it is ~0.001–0.002 %** — the action is, to five decimals, ignored.

## Why the verdict is admissible — both controls read their known values

* **C0 IDENTITY passes on every row**: feeding the *same* actions twice gives spread
  **exactly 0.0**. The forward is deterministic, so the tiny action numbers are real
  signal-absence and not numerical noise.
* **C1 SCENE REFERENCE is LARGE**: 0.18–0.43. ⇒ ⛔ **The competing "degenerate
  rollout" account is REFUTED.** The model is emphatically *not* insensitive to
  everything — change the scene and the prediction moves a great deal. It is
  specifically **deaf to the action channel.**

That asymmetry is the whole result: a degenerate model would have shown a small
denominator too, and the ratio would have been 0/0 — which the prereg committed to
report as VOID rather than as collapse.

⚠️ **Variants were drawn by ROLL, never random permutation** — a permutation leaves
~1/B items holding their OWN actions, which biases the numerator DOWN, *toward* the
collapse conclusion. Same defect class caught in the nav control earlier today.

## ⭐ A SECOND, INCIDENTAL FINDING — the predictor collapses horizons ≥ 2

`h=2` and `h=4` produced identical `scene_spread` to five decimals, which I checked
rather than reported. Direct comparison on the predictor:

```
max |h1 − h2| = 1.211
max |h2 − h4| = 0.00179      (~680× smaller)
```

⇒ The predictor **separates one tick from two, and then essentially stops**. It emits
one long-horizon guess for everything beyond h=1. Registered as a distinct defect; it
is not implied by action-insensitivity and needs its own explanation.

## What this changes

⛔ **The drift lever is aimed at the wrong defect.** The pre-committed next
anti-drift step was the frozen-teacher feature target. But a world model whose
action channel contributes 0.4 % of its scene channel is not primarily suffering
from drift — **drift is downstream of a predictor that is barely conditioned at
all**, which is consistent with MM-E4 (drift is a symptom, trivially reducible) and
MM-E6 (drift is self-referential, living in directions the scene does not explain).

⇒ **The indicated lever is O1** — the term the code itself documents as *"the
ANTI-ACTION-ECHO measure"*, `default=1.0`, **which every one of these arms ran with
at weight 0**. That is the cheapest possible test of the diagnosis: turn on the
objective whose job is exactly this, one variable, tiny rig.

⚠️ **Committed in advance, before that arm runs:** per the banked precedent (an arm
**10.7× worse** on open-loop next-action MSE was **2.3× better** closed-loop),
**a de-confounding intervention is EXPECTED to make T0 worse.** If our own T0 gate
rejects the O1 arm on prediction, the gate is wrong, not the arm — and that must be
written into the pre-registration or the result will be discarded by reflex.

## Provenance (MM-C12)

Run **on Thor**, importing `/home/nvidia/TanitAD/stack/tanitad/__init__.py` —
**stamped into the output JSON**. Deliberate: the dev-box probe helper hardcodes a
third mirror whose `v6.py` matches neither the repo nor G:, and `load_trunk_auto`
*rebuilds* the model from the checkpoint's config, so the code version is
load-bearing. **These checkpoints were trained on Thor with this stack**, which makes
it the authoritative code for rebuilding them.
