# VERDICT — the two RL adapters bind DIFFERENT MODEL CLASSES, so the repoint is wrong

**2026-09-07 · Architecture & Inference · `agent/arch-inf-20260803`**
**Decision: option 3 — DIFFERENT MODEL FAMILIES. Guard `refcv3_adapter` in place.
⛔ Do NOT repoint the pilot at `refc_adapter`; it would hard-fail on the first batch.**

---

## 0. The question I was asked

Three commits tonight hardened `tanitad/rl/refc_adapter.py` with a channel contract.
A sibling measured that the only production RL script calls the *other* adapter, and
proposed **repointing the pilot** at the guarded module. A second proposal was to
**guard `refcv3_adapter` in place**. The brief made the deciding question explicit:

> are these two adapters serving the same models, or different ones?

⭐ **They serve different model classes, in different modules, with different forward
signatures and differently-shaped configs.** Every branch of the decision follows from
that one fact, and it is checkable in four lines of source.

---

## 1. The evidence — two classes, not one

| | `refcv3_adapter` (what the pilot uses) | `refc_adapter` (what was hardened) |
|---|---|---|
| model class | `refc.RefCModel` | `refc_v3.RefCV3Model` |
| defined at | `stack/tanitad/refs/refc.py:2515` | `stack/tanitad/refs/refc_v3.py:763` |
| `forward` at | `refc.py:2870` | `refc_v3.py:1361` |
| config type | `refc.RefCConfig` (`refc.py:602`) | `refc_v3.RefCV3Config` (`refc_v3.py:334`) |
| config nesting | **flat** — `anchors`, `graft_lan`, … at top level | **nested** — `core: refc.RefCConfig` (`refc_v3.py:347`) |

**Evidence class: PUBLISHED-CODE, read from source 2026-09-07.**

`refc.py` does not define, import, subclass or alias `RefCV3Model` — its whole import
block (`refc.py:139-173`, plus the deferred `:2444`) names `kinematic`, `feasible_decode`,
`goal_point`, `refc_sampler`, `refc_select`, `refc_tactical`, `imagination`. There is no
inheritance relationship in either direction.

⚠️ `refc_adapter`'s own module docstring says refcv4b/refcv5 "are *not* new model classes
— they are config variants of the same `RefCV3Model`". **That sentence is true and is not
the sentence that matters.** It is a claim about refcv4b/refcv5 vs refcv3. It says nothing
about `refc.RefCModel`, which is the class the pilot builds — and the pilot's own header
says so plainly (`rl_pilot_refc21.py:10`: *"model `refc.RefCModel(refc.refc_config())`"*,
wired at `:169`). **This is precisely the "true but wrong for the reader" failure mode:
a correct claim that implies a wrong next action.**

### 1.1 The two forward signatures, side by side

`refc.RefCModel.forward` (`refc.py:2870-2881`), 13 optional parameters:

```
nav_cmd, v0, maneuver_logits, target_latent, steps, lan, nav_known,
hierarchy_hook, ego_keep, withheld_speed, gp_point, gp_valid, agent_gt
```

`refc_v3.RefCV3Model.forward` (`refc_v3.py:1361-1372`), 13 optional parameters:

```
nav_cmd, v0, steps, lan, nav_known, ego_state, withheld_speed,
gp_point, gp_valid, agent_gt, nav_args, v_max_ms, v_max_valid
```

Same arity, **different sets**. The symmetric difference is seven channels:

| only on `RefCModel` | only on `RefCV3Model` |
|---|---|
| `maneuver_logits`, `target_latent`, `hierarchy_hook`, `ego_keep` | `ego_state`, `nav_args`, `v_max_ms`, `v_max_valid` |

MEASURED token census over both files (whole-file `str.count`, paired non-zero control —
`refc.py` 197,062 chars / `refc_v3.py` 110,854 chars, both non-zero, so neither read was a
mount flap):

| token | `refc.py` | `refc_v3.py` |
|---|---|---|
| `ego_state_inject` | **0** | 11 |
| `ego_state` | **0** | 51 |
| `ego_keep` | 17 | 15 |
| `v0_conditioned` | 7 | **0** |
| `graft_lan` | 13 | **0** |
| `nav_known_channel` | 8 | **0** |
| `nav_args` | **0** | 20 |
| `v_max_ms` | **0** | 11 |

⭐ Read the `v0_conditioned` / `graft_lan` / `nav_known_channel` rows carefully: they are
**zero in `refc_v3.py`**. Those fields live on `RefCConfig` and `RefCV3Model` reaches them
only through `.core`. That is the whole nesting story in one measurement.

---

## 2. What the repoint would actually do — it fails twice, not once

**Failure 1 — every predicate raises, on the FIRST batch.**
`refc_adapter.CHANNEL_REQUIREMENTS` declares its predicates as dotted paths rooted at the
*outer* config: `core.anchors.v0_conditioned`, `core.sel_reach_clamp`, `core.graft_lan`.
`_read_predicate` (`refc_adapter.py:~400`) **raises `ConditioningError` on any path that
does not resolve** — deliberately, so a never-firing check cannot masquerade as a passing
one. The pilot's model carries a bare `RefCConfig` (`rl_pilot_refc21.py:169` →
`refc.RefCModel(refc.refc_config())`; `refc_config()` at `refc.py:882` returns `RefCConfig()`),
and `RefCConfig` **has no attribute `core`**. Its fields sit flat:

```
refc.py:607   anchors: AnchorConfig = field(default_factory=AnchorConfig)
refc.py:693   graft_lan: bool = False
refc.py:778   sel_reach_clamp: bool = False
refc.py:836   nav_known_channel: bool = False
refc.py:383   AnchorConfig.v0_conditioned: bool = False
```

⇒ **MEASURED by running exactly that repoint** (`code/evidence.py`, banked in
`evidence.json`) — `ra.conditioning_requirements(RefCModel)` raises:

```
ConditioningError: conditioning predicate 'ego_state_inject' does not resolve on this
build: cfg has no attribute 'ego_state_inject'. ⛔ REFUSING rather than reading False.
```

⚠️ **Corrected against my own first draft, which predicted the message would name
`core.anchors.v0_conditioned`.** It names `ego_state_inject` instead, because
`CHANNEL_REQUIREMENTS` is walked in declaration order and `ego_state` is declared first.
The substance is unchanged and is if anything stronger: **four** of `refc_adapter`'s
predicates are unresolvable on a `RefCConfig` (`ego_state_inject`,
`core.anchors.v0_conditioned`, `core.sel_reach_clamp`, `core.graft_lan`), and the walk
refuses at the first one it reaches. **The repointed pilot would not run one step.**

⭐ The guarded module's own reasoning is the confirmation. `ConditioningContract`'s
docstring cites the mechanism as `refc.py:3097` — `v_ms = v0 if (cfg.sel_reach_clamp and
v0 is not None) else None` — **flat `cfg.sel_reach_clamp`**, because inside `refc.py`
`self.cfg` *is* a `RefCConfig`. The declaration one screen away writes the same field as
`core.sel_reach_clamp`. Both are right, for different objects. That is exactly the
"contract asserted about the wrong object" the brief warned against.

**Failure 2 — a channel that does not exist.**
`refc_adapter.FORWARD_KEYS` contains `ego_state`, and `forward_kwargs` passes **every**
key unconditionally (`kw = {k: batch.get(k) for k in FORWARD_KEYS}`), so a `None` is
still passed by name. `RefCModel.forward` has no `ego_state` parameter (**0 occurrences
in all 197,062 chars of `refc.py`**; its ego channel is named `ego_keep`).
⇒ `TypeError: forward() got an unexpected keyword argument 'ego_state'`.

Conversely `refc_adapter` would silently stop plumbing `maneuver_logits` and
`target_latent` — the external tactical-brain seams that only `RefCModel` has.

⛔ **So the repoint is not "a decision to escalate". It is a defect.** Had it been taken
on the strength of the naming (`refc_adapter` reading as the general case of
`refcv3_adapter`), it would have converted a silent gap into a loud crash — which is the
better failure, but it is still not the fix, and the reason it crashes is that the
contract describes a different model.

---

## 3. …and the third option is not free either

⚠️ **Guarding `refcv3_adapter` is right, but copying `refc_adapter`'s channel set into it
would reproduce the same error in the opposite direction.** The values must be derived
from `RefCModel`'s own signature and declared against `RefCConfig`'s own (flat) paths.
That is the bar the brief set and it is the bar this package builds to.

One further complication that the sibling proposals did not surface:
**`refcv3_adapter` is duck-typed.** It imports no model at all — its only imports are
`math`, `torch`, and `.config.PostTrainConfig`. Its docstring says "a real `RefCV3Model`",
but its two callers pass different classes:

| caller | model passed | evidence |
|---|---|---|
| `stack/scripts/rl_pilot_refc21.py:438` | `refc.RefCModel` | `:169` |
| `stack/tests/test_rl_refcv3_integration.py:154` | `refc_v3.RefCV3Model` | `:31` |

⇒ the guard **must** resolve against whichever model it is handed. A hard-coded channel
tuple would be wrong for one of its two callers no matter which one it was written for.
⭐ That settles the design: derive the channel set from
`inspect.signature(type(model).forward)` at contract-construction time, and let each
channel declare *alternative* config paths so the same declaration resolves on a flat
`RefCConfig` and on a nested `RefCV3Config`.

---

## 4. Independently confirmed: the wiring measurement

Re-measured with a **Python `os.walk` census** rather than `rg`, because grep
under-reports ~5× on the G: mount while exiting 0. **Control: 22,892 `.py` files walked
(non-zero ⇒ the mount was up).**

| probe | result |
|---|---|
| `make_refcv3_sample_fn` under `stack/scripts/` | **`rl_pilot_refc21.py` only** |
| `make_refc_sample_fn` under `stack/scripts/` | **none** |
| `assert_conditioning` in `refcv3_adapter.py` | **0** (control: the file's own name reads 1 ⇒ file was read) |
| any guard token in `rl_pilot_refc21.py` | `channel_guard` 0 · `preflight` 0 · `assert_conditioning` 0 · `ConditioningContract` 0 · `conditioning` 0 · `requirement` 0 (control: file len 23,808 chars) |

⇒ the sibling's finding is **CONFIRMED and strengthened**: the pilot reaches the model
through an adapter with no contract, and touches **none** of the three guard surfaces
built tonight — not the adapter's, not `channel_guard`'s launch preflight, not
`rl_control_space_preflight`. `channel_guard.py` is itself scoped to `refc_adapter`'s
`FORWARD_KEYS` vs `RefCV3Model.forward`, so it could not have covered the pilot either.

---

## 5. ⚠️ The limit this guard does NOT remove — stated, not implied

**On the pilot's config as it stands today, the derived requirement set is EMPTY and the
guard will not fire.** `refc_config()` returns bare `RefCConfig()` defaults, and all four
predicate fields default `False` (`refc.py:383`, `:693`, `:778`, `:836`). The guard
asserts *what the build declares*, and this build declares nothing.

⭐ That makes the guard a **regression preventer**, not a live-defect fix, and it is worth
saying which it is. It becomes load-bearing the moment the pilot is pointed at a
`v0_conditioned` / `graft_lan` build — which is the whole D-REFCV3 direction.

⛔ **AND IT EXPOSES A SEPARATE, LARGER DEFECT I AM NOT FIXING HERE.** The pilot rebuilds
its config from `refc_config()` defaults and loads `ck["model"]` with `strict=True`; the
checkpoint's *config* is never consulted. `EXPECT_PARAMS = 104_191_577` cannot catch a
mismatch on these flags because **none of them changes the parameter count** —
`v0_conditioned` is stored as a plain bool (`refc.py:1391`, `self.anchor_v0_cond =
bool(v0_conditioned)`) and `strict=True` compares tensor names and shapes, not flags.
⇒ if the P-RC21 checkpoint was trained v0-conditioned, the pilot is *already* running the
10 m/s fixed-reference action space, and a config-derived guard **cannot see it**, because
the config it reads is the one that is wrong.

That is a checkpoint-provenance question, not a conditioning question, and it needs the
argv/registry record for the P-RC21 cold start. **Escalating it rather than guessing.**

---

## 6. Decision

| option | verdict |
|---|---|
| repoint the pilot at `refc_adapter` | ⛔ **WRONG — a defect, not a decision.** Raises on batch 1 (`core.` unresolvable) and would `TypeError` on `ego_state`. |
| guard a dead module | ✅ **not the case** — `refcv3_adapter` is live, is the *only* adapter with a production caller, and has two callers on two different model classes. |
| give `refcv3_adapter` its own contract | ⭐ **CORRECT, and it is option 3 of the brief: different families, each needs its own.** Additive; changes nothing that anything calls. |

⇒ **Build the guard, derive it from the handed model's own signature, declare it against
`RefCConfig`'s flat paths, and do not inherit a single value from `refc_adapter`.**
