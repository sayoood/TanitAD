# SPEC — NAV-COMMAND CONDITIONING AT ALL THREE ABSTRACTION LAYERS

**PI DIRECTIVE 2026-08-30, verbatim intent:** *"refine the architecture of all our
models to be trained in the future to get the nav command (this is part of our
released vocab) input token as input for all three abstraction layers — operative,
tactical and strategic. Starting now this input is MANDATORY for all our model
inference and tests, even for the operative one. It's conditioning the WM like the
actions."*

**Status: BINDING.** Owner: TrainingFlyWheel (models + trainers). Applies to every
arm trained from now: v7f, refc_v3, refa_v1, refd.

## 0. Where we start from — MEASURED 2026-08-30

⛔ **The v6/v7 world model has NO nav conditioning today.** `stack/tanitad/models/v6.py`
contains **2** occurrences of the string `nav`, neither a wiring. Nav lives only in the
older flagship line and in eval: `config.py`, `models/fourbrain.py`,
`models/refc_rescorer.py`, `models/vocab_v7.py`, `data/lan.py`,
`train/flagship_losses.py`, `train/heldout_gate.py`, `eval/route_cf.py`.
⇒ This is a genuine architecture change, not a flag.

**The vocabulary already exists and is released** (`models/vocab_v7.py`):

```
NAV_COMMAND_TOKENS = ["NAV_FOLLOW_ROAD", "NAV_TURN_L", "NAV_TURN_R"]     # n=3
NAV_ARG_SLOTS      = ["distance_m", "time_s"]
NAV_PROVENANCE     = ["nav-system", "ego-future"]
```

Corpus distribution (B1 release blob `ee44875916…`, 4,719 records):
`NAV_FOLLOW_ROAD` 2,993 · `NAV_TURN_R` 902 · `NAV_TURN_L` 824.

## 1. What to build

The nav token conditions **all three layers** — operative, tactical, strategic —
**exactly as the action channel conditions the operative predictor**. Not a head, not
a loss: an **input**.

* One embedding of the 3-token vocabulary + its two continuous args
  (`distance_m`, `time_s`), shared across layers so the three cannot drift apart.
* Injected at each layer's conditioning point, alongside whatever that layer already
  receives. The operative layer is explicitly included per the directive.
* **Mandatory at inference and in every test.** An arm that cannot produce a nav
  token must fail loudly, not fall back to a default — a silent default is the
  `_ensure_ego` size-threshold trap in a new costume, and it would make two arms
  silently incomparable.
* Args are continuous and must be normalised on **corpus statistics computed on the
  fit split only**, never per-batch.

## 2. ⛔ THE ONE THING THAT DECIDES WHETHER THE RESULT MEANS ANYTHING

**Every `nav_command` in B1 has `provenance = "ego-future"` — 4,719 of 4,719. It is an
ORACLE: computed from the ego's own future path.** The vocab anticipates a real
`nav-system` provenance; **our corpus has none of it.**

This does **not** conflict with the directive — the PI's own 2026-08-03 binding ruling
says a goal/route input IS admissible, and a real vehicle genuinely has a navigation
system. But it means the *measurement* needs care that the *architecture* does not:

⚠️ **Flagship v1's route head was an exact bijection of the nav we fed it (369/369 and
81/81) and scored 1.0000 — an echo of its own input read as skill.** CLAUDE.md states
the general form: *"A supplied route is optimistic by construction on PhysicalAI —
our only route supplier there is the ego's own future path."*

⭐ **THE CONSEQUENCE, AND IT FOLLOWS DIRECTLY FROM THE PI'S OWN FRAMING.** The
directive says nav conditions the WM *"like the actions."* Actions already carry a
mandatory anti-echo control — the **hold-action** floor an arm must beat. ⇒ **Nav
must carry the twin**, or we will not be able to tell "uses the route" from "echoes
the route":

| control | what it does | what it proves |
|---|---|---|
| **hold-nav** | freeze the nav token at its t0 value for the whole rollout | the arm is not merely tracking a changing oracle |
| **shuffled-nav** | serve another clip's nav token | ⛔ **the decisive one.** If performance does not degrade, the arm is ignoring nav and the conditioning is inert. If it degrades to chance, the arm may be *reading the future* rather than following a route |
| **no-nav ablation** | the same arm without the channel | the size of what nav actually buys |

⛔ **No capability claim from a nav-conditioned arm is admissible without the
shuffled-nav control reported beside it.** This is the same rule that made the T1
floor trustworthy, applied to the new channel before it can mislead us rather than
after.

⚠️ **Provenance is stamped, not assumed.** The token's `provenance` travels into the
run's `config.json` and into every eval record. An arm trained or evaluated on
`ego-future` nav is labelled as such wherever its numbers appear. When a real
`nav-system` source exists, the same arm re-evaluated under it is a *different
measurement* and must not be pooled with the oracle one.

## 3. Interaction with the information-disjointness rule

The PI's 2026-08-03 ruling stands unchanged: the goal path and the situation path stay
**information-disjoint at inference**. The nav token is a route signal and is
admissible; ⛔ what remains inadmissible is any nav input **derived from the situation
classifier's output** — class posterior, argmax, embedding, or any feature of them.
State, for any arm, what the nav token is computed from.

## 4. Gates

1. `pytest -q` green; the shared embedding proven shared by a test that changes it
   once and asserts all three layers move.
2. A test that a **missing** nav token RAISES rather than defaulting.
3. The three controls above implemented and **shown to fire** — in particular a
   deliberate-regression arm whose nav is shuffled must be *detected*, or the control
   is worthless.
4. First read at **T0** (cheap, label-free) to confirm the channel is live and
   non-inert, then **T1** with the four metric families.

⚠️ **Pre-register that a de-confounding result may make T0 WORSE.** The banked
literature is explicit: an arm 10.7× worse on open-loop next-action MSE was 2.3×
better closed-loop, because history helps only once the shortcut is closed. If our own
gate rejects that arm on T0, the gate is wrong, not the arm.
