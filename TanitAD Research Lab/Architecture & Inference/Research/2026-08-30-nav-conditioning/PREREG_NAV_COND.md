# PREREG — NAV-COND: nav-token conditioning on all three layers

`PRE-REGISTRATION, 2026-08-30, TanitAD_TrainingFlyWheel. Committed BEFORE any
nav-conditioned arm exists. PI DIRECTIVE (binding): the nav command token is a
MANDATORY input to the operative, tactical and strategic layers — "it's
conditioning the WM like the actions". Spec:
Project Steering/SPEC_NAV_CONDITIONING_ALL_LAYERS.md.`

---

## 0. ⛔ THE COMMITMENT THAT MUST EXIST BEFORE ANY NUMBER: T0 MAY GET WORSE

**If the nav-conditioned arm is WORSE on T0 than the unconditioned one, that is
NOT by itself grounds to reject it.**

PUBLISHED precedent, banked by the Research Lab this morning: an arm **10.7×
worse** on open-loop next-action MSE was **2.3× better closed-loop**, because
history only helps once the shortcut is closed. Our own T0 gate would have
rejected exactly that arm.

⇒ **Committed in advance: a T0 regression on a nav arm is escalated for a T1 read,
not auto-failed.** ⛔ If the gate rejects it on T0 alone, **the gate is wrong, not
the arm.** Writing this down now is the only thing that stops the gate from
quietly discarding the result we are trying to obtain.

⚠️ This is not a licence to ignore T0. A T0 regression **plus** a failed
shuffled-nav control (§2) is a dead arm, not a de-confounded one.

## 1. What is built (architecture, not a flag)

MEASURED starting point: `v6.py` contains **two** occurrences of "nav" and
**neither is a wiring**. The v6/v7 world model has **no nav conditioning today**;
it exists only in the older flagship line and in eval.

* ONE shared embedding of `NAV_COMMAND_TOKENS` (3) + the two continuous args
  (`distance_m`, `time_s`), **shared across all three layers** so they cannot
  drift apart.
* Injected at each layer's conditioning point as an **INPUT** — not a head, not a
  loss.
* ⛔ A missing nav token **RAISES**. It never defaults. A silent default is the
  `_ensure_ego` size-threshold trap in a new costume and would make two arms
  silently incomparable.
* Args normalised on **fit-split statistics only**, never per-batch.

## 2. ⛔ THE CONTROLS SHIP WITH THE CHANNEL, NOT AFTER IT

Every `nav_command` in B1 is `provenance: "ego-future"` — an **oracle** computed
from the ego's own future path (4,719/4,719; and note the `oracle` BOOLEAN is
absent on 529 of them, so provenance is the load-bearing field — see the
`v7_labels` RESULT). Flagship v1's route head was an exact bijection of the nav we
fed it (369/369, 81/81) and **scored 1.0000** — an echo read as skill.

| control | what it does | what it proves |
|---|---|---|
| **hold-nav** | freeze the token at t0 for the rollout | the arm is not merely tracking a changing oracle |
| **shuffled-nav** | serve ANOTHER CLIP's nav | ⛔ **decisive.** No degradation ⇒ the channel is INERT. Degradation to chance ⇒ the arm may be reading the future, not following a route |
| **no-nav** | the same arm without the channel | the size of what nav actually buys |

⛔ **No capability claim from a nav arm is admissible without shuffled-nav
reported beside it.**

⭐ **AND THE CONTROL ITSELF MUST BE SHOWN TO FIRE** (spec §4.3): a
deliberate-regression arm whose nav is shuffled must be **detected**. A control
that cannot detect a shuffled nav proves nothing about a real one — this is
TRAIN-C8's rule (presence of a control is not validity of a control) applied
before the channel can mislead us rather than after.

## 3. Committed outcomes, before any arm runs

| # | if | then |
|---|---|---|
| **V** | the shuffled-nav control does not fire on the deliberate-regression arm | ⛔ **VOID** — the instrument cannot see the failure it exists to see. No nav claim admissible. |
| **1** | nav arm beats no-nav on T0 **and** shuffled-nav degrades it | ⭐ the channel is live and used. Proceed to T1 with all four families. |
| **2** | nav arm ≈ no-nav **and** shuffled-nav does NOT degrade | ⛔ **the channel is INERT** — the model ignores it. Report as such; do not tune to rescue it. |
| **3** | nav arm beats no-nav **but** shuffled-nav does NOT degrade | ⛔ **the gain is not from nav.** Something else changed (capacity, init). Attribute before claiming. |
| **4** | nav arm WORSE on T0 **and** shuffled-nav degrades sharply | ⚠️ **the de-confounding case of §0** — escalate to T1, do NOT auto-fail. |
| **5** | shuffled-nav degrades performance **to chance** | ⚠️ the arm may be READING THE FUTURE rather than following a route. Treat as a leak finding, not a capability. |

## 4. Provenance stamping

`provenance` travels into `config.json` and every eval record. An arm trained or
evaluated on `ego-future` nav is **labelled wherever its numbers appear**, and a
future `nav-system` arm is a **different measurement** that must never be pooled
with it.

## 5. Information disjointness (PI 2026-08-03, unchanged)

The nav token is a route signal and is admissible. ⛔ Inadmissible: any nav input
derived from the **situation classifier's output** — posterior, argmax, embedding,
or any feature of them. Every arm states what its nav token is computed from; for
B1 the answer is **the ego's own future path**, which is why §2 is mandatory.
