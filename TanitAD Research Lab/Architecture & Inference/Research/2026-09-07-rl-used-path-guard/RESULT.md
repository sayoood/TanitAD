# The path a production RL arm actually executes is now guarded

**2026-09-07 · Architecture & Inference · `agent/arch-inf-20260803`**
**Read `VERDICT.md` first — it establishes WHICH fix was right, and why the obvious one
was a defect rather than a decision.**

---

## The one-line answer

⭐ **YES — the adapter `rl_pilot_refc21.py` actually calls now asserts a conditioning
contract on every batch.** It asserts *nothing* on the pilot's current config, because
that config declares nothing; the guard is a **regression preventer** on the used path,
not a live-defect fix, and §5 says exactly when it becomes load-bearing.

⛔ **NOT the repoint.** `refcv3_adapter` and `refc_adapter` bind different classes; the
repoint raises before the first optimiser step. That is measured, not argued — §2.

---

## 1. What changed

| file | change |
|---|---|
| `stack/tanitad/rl/refcv3_adapter.py` | +619 lines: a conditioning contract derived from the **handed model's own forward signature**, asserted on **every** batch. Nothing existing was removed; `sample_offsets` / `gt_context` (re-exported by `refc_adapter`) are byte-unchanged. |
| `stack/tests/test_rl_refcv3_used_path_guard.py` | new, 19 tests, all green |
| `stack/tanitad/rl/refc_adapter.py` | ⭐ **untouched** — see §6 |

The factory gained `strict_conditioning: bool = True` and now records the arm on the
returned callable (`.strict_conditioning`, `.conditioning_contract`), so a run record can
bank what was actually asserted instead of trusting an argv.

---

## 2. The channel set, derived from `refcv3_adapter`'s own model

⛔ **Not copied from `refc_adapter`.** The derivation is

```
channels = inspect.signature(model.forward)
           - {self, frames, steps}
           - tanitad.channel_admissibility.excluded_channels()
```

and **every surviving channel must be declared**, so a forward that grows a channel
refuses at the next rollout construction instead of running blind to it. That is the
drift which hit `FORWARD_KEYS` twice (it lost `agent_gt`; then the worktree and HEAD
disagreed about it) — deriving only helps if an *underived* channel is loud.

**MEASURED** (`code/evidence.py` → `evidence.json`, read from the real classes):

| | `refc.RefCModel` — the pilot's family | `refc_v3.RefCV3Model` — this adapter's test caller |
|---|---|---|
| derived channels | `nav_cmd, v0, maneuver_logits, target_latent, lan, nav_known, hierarchy_hook, ego_keep, withheld_speed, agent_gt` (**10**) | `nav_cmd, v0, lan, nav_known, ego_state, withheld_speed, agent_gt` (**7**) |
| asserted | `v0`, `lan` | `v0`, `lan`, `ego_state` |
| not asserted, each with a stated reason + unblock | the other 8 | the other 4 |

### How it differs from `refc_adapter`'s

⭐ **The v3 column is EXACTLY `refc_adapter.FORWARD_KEYS`** — two independently built
lists agreeing, element for element, on the one family they share. That is the strongest
available check that this derivation is right, and it was not arranged: `FORWARD_KEYS` is
hand-kept, this one is computed.

The differences are all and only the differences between the models:

| | `refc_adapter` | this adapter |
|---|---|---|
| scope | one hand-kept tuple, `RefCV3Model` only | **derived per model**, both families |
| `ego_state` | required when `ego_state_inject` | in scope **only** on `RefCV3Model`; `refc.RefCModel` has no such parameter (0 occurrences of `ego_state` in 197,062 chars of `refc.py`) |
| `ego_keep`, `maneuver_logits`, `target_latent`, `hierarchy_hook` | absent — not parameters of `RefCV3Model` | declared NOT-ASSERTED; `None` is the deployed value for each, and `hierarchy_hook` is a *callable*, declared rather than type-filtered so it cannot vanish from scope unremarked |
| `v0` predicate | `core.anchors.v0_conditioned`, `core.sel_reach_clamp` | `anchors.v0_conditioned`, `core.anchors.v0_conditioned`, `sel_reach_clamp`, `core.sel_reach_clamp` — **alternatives**, because `RefCV3Config.core` IS a `RefCConfig` (`refc_v3.py:347`) and the pilot's config is the flat one |
| unresolvable predicate | raises on the first | raises only when **none** resolve — with `test_predicates_resolve_on_BOTH_real_config_shapes` closing the hole that leaves, by pinning exactly which path resolves on which shape |

⛔ **And a check `refc_adapter` does not have: the WIRING half.** `make_refcv3_sample_fn`
plumbs only `nav_cmd`/`v0`/`lan`. A channel that is REQUIRED, PRESENT in the batch, and
NOT PLUMBED would have passed every check and still arrived as `None` — every guard green,
the policy differently conditioned. `ego_state` is exactly that case. So `PLUMBED_CHANNELS`
is now enforced against the requirement map *and* is the tuple `_forward_kwargs` iterates,
so the plumbing the contract checks and the plumbing the forward receives are one object.

---

## 3. The temporal mutation proof

**6/6 mutations RED**, each a complete self-consistent edit (the harness refuses a
mutation that breaks the import — a `NameError` proves nothing about timing). Baseline
19/19 green; the adapter is sha-verified restored after each. Full record in
`mutation_result.json`.

| mutation | property removed | RED |
|---|---|---|
| **M1** once-only latch | per-batch → assert on batch 1 only | `..._SECOND_batch_not_only_the_first` |
| **M2** check after sampling | the ordering: refuse only *after* the forward ran | `..._SECOND_batch_not_only_the_first` |
| **M3** over-assert `v0` | require `v0` unconditionally | `test_a_build_NOT_trained_with_v0_does_NOT_require_it` |
| **M4** inherit `refc_adapter`'s predicates | root `v0` at `core.` — i.e. the copied contract | **6 tests** |
| **M5** drop the plumbing check | a required-but-unplumbed channel passes | `..._never_PLUMBS_is_REFUSED` |
| **M6** ignore undeclared channels | signature drift narrows scope silently | `test_an_UNDECLARED_forward_channel_is_REFUSED` |

⭐⭐ **M1 and M2 kill the SAME test for DIFFERENT reasons** — which is the point, and is
why the mutation was written twice:

```
M1  ...:298: Failed: DID NOT RAISE ConditioningError
M2  ...:304: AssertionError: the model was called 2 times — the guard refused
             AFTER sampling, which is not a guard
```

⇒ both halves of the temporal assertion are independently load-bearing: *that* it refuses
on batch #2, and *that the forward count stays at 1* when it does. A test asserting only
the first would have survived M2 — a guard that refuses after already sampling the
mis-conditioned batch.

⛔ **The over-assertion side is pinned as hard as the refusal**, because refusing valid
launches is how a guard gets deleted rather than fixed. M3 is its mutation, and the
expectation is hard-coded (`required == ()` exactly; a batch with **no** `v0` at all must
return a fan) — never the shape *"if missing: expect refusal; else: expect pass"*.

⚠️ M4's blast radius (6 tests) is the verdict made executable: copying a sibling's channel
set across is caught by six independent assertions, one of which is the refusal message
`NONE of the conditioning predicates ['core.anchors.v0_conditioned', 'core.sel_reach_clamp']
resolve on this build's config`.

---

## 4. Regression status — MEASURED against a control, not assumed

⛔ The full suite is **not** green on this branch, and "those failures look unrelated" is
not evidence. So the whole suite was run **twice**, identically, with only my two files
swapped for their `HEAD` versions in between:

| full `stack/` suite | failed | passed | skipped | errors | wall |
|---|---|---|---|---|---|
| **WITHOUT** my change (`HEAD` adapter, test file removed) | **31** | 7240 | 118 | 7 | 637.85 s |
| **WITH** my change | **31** | **7259** | 118 | 7 | 640.16 s |

⭐ **Same 31 failures, same 7 errors; the passed count rises by exactly 19 — my 19 new
tests.** The 31 pre-existing failures sit in ten files
(`test_decision_check`, `test_eval_contamination`, `test_kingate_contract`,
`test_launch_closure_audit`, `test_runbook_commands`, `test_secret_scan`,
`test_tactical_goal_underpowered_matches_census`, `test_text_encoding_is_explicit`,
`test_v6_chain`, `test_v6_st_launch_fixes`), **none of which contains a single reference
to either adapter** (measured: 0 occurrences of `refcv3_adapter` / `refc_adapter` /
`make_refcv3` / `conditioning` in all ten). Running just those ten in isolation gives
**29 failed + 7 errors byte-for-byte with and without the change**.

⚠️ The control run was bracketed by an md5 of the reverted adapter — identical before and
after — so the mirror was not re-synced mid-run, which is the failure mode that has
silently reverted a long run on this box before.

| targeted suite | result |
|---|---|
| `tests/test_rl_refcv3_used_path_guard.py` | **19 passed** |
| `tests/ -k "rl_ or refcv3 or refc_adapter or channel"` | **557 passed, 1 skipped, 0 failed** |
| the four conditioning/integration files together | **67 passed** |

`test_rl_refcv3_integration.py` matters most here: it drives a **real `RefCV3Model`**
through this adapter, and `refc_v3_smoke_config()` sets `core.sel_reach_clamp = True`
(`refc_v3.py:751`) — so the new contract resolves `v0` as **REQUIRED** on it and the test
passes only because its batch genuinely carries `v0`. The guard is exercised by the
pre-existing suite, not merely by its own.

---

## 5. ⚠️ What this does NOT fix — stated, not implied

**On the pilot's config as it stands, the required set is EMPTY** (measured: all ten
channels `false`). `refc_config()` returns bare `RefCConfig()` defaults and every predicate
field defaults `False` (`refc.py:383`, `:693`, `:778`, `:836`). The guard asserts *what the
build declares*, and this build declares nothing. It becomes load-bearing the moment the
pilot is pointed at a `v0_conditioned` or `graft_lan` build — which is the whole D-REFCV3
direction — or the moment someone edits `collate()`.

⛔ **AND IT EXPOSES A LARGER DEFECT I AM ESCALATING RATHER THAN FIXING.**
`rl_pilot_refc21.py` rebuilds its config from `refc_config()` defaults and loads
`ck["model"]` with `strict=True`; **the checkpoint's own config is never consulted**, and
`EXPECT_PARAMS = 104_191_577` cannot catch a mismatch on these flags because **none of them
changes the parameter count** — `v0_conditioned` is stored as a plain bool
(`refc.py:1391`) and `strict=True` compares tensor names and shapes, not flags.

⇒ **if the P-RC21 checkpoint was trained v0-conditioned, the wrong action space is already
selected at BUILD time**, and no config-derived guard can see it, because the config it
reads is the one that is wrong. That needs the cold start's argv from the registry.
**It is a checkpoint-provenance question, not a conditioning one, and it is open.**

⚠️ Also unchanged: this adapter still plumbs only three channels. That is now *declared*
(`PLUMBED_CHANNELS`) and *enforced* (a required channel outside it refuses), so it can no
longer be silent — but widening it is a separate decision with its own evidence.

---

## 6. Why `refc_adapter.py` was not touched

The machinery is **reused, not forked**: `ChannelRequirement`, `RequirementDeclarationError`
and `ConditioningError` are imported from `refc_adapter` — **lazily**, inside
`_machinery()`, because `refc_adapter` imports *this* module at module scope and a
top-level import back is a cycle. Both import orders are pinned by a parametrised
subprocess test, since a regression to a top-level import is a hard `ImportError` in only
one of the two.

⇒ the declarations that landed tonight in `refc_adapter` keep working exactly as they were,
on the model they were established for, and this contract is purely additive.

---

## Deliverable manifest

| artifact | where | only one place? |
|---|---|---|
| the verdict + its evidence | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-rl-used-path-guard/VERDICT.md` | no — staged |
| this result | `repo:…/2026-09-07-rl-used-path-guard/RESULT.md` | no — staged |
| the guard | `repo:stack/tanitad/rl/refcv3_adapter.py` | no — staged |
| the suite | `repo:stack/tests/test_rl_refcv3_used_path_guard.py` | no — staged |
| mutation harness + record | `repo:…/code/mutate.py`, `…/mutation_result.json` | no — staged |
| evidence dump | `repo:…/code/evidence.py`, `…/evidence.json` | no — staged |
| splice script (provenance of the edit) | `repo:…/code/splice.py` | no — staged |

⛔ Nothing lives only on a mirror or only on a pod. No GPU was used; the A40 training was
not touched. `refc.py` / `refc_v3.py` / `refc_v3_train.py` were read only — `git status`
reports them unchanged, as does `refc_adapter.py`.

⚠️ **One process rule broken, self-caught and verified.** A late one-line docstring
correction was applied **in place on the G: path** rather than authored locally and copied
in one op. The file was immediately re-read (435 lines, `ast.parse` OK) and re-hashed
against a fresh local copy — no truncation occurred — but the rule exists precisely
because that write can be interrupted, and getting away with it is not the same as being
right. Recorded rather than quietly dropped.

## ⭐ ESCALATIONS

1. **The pilot's checkpoint provenance** (§5) — was the P-RC21 cold start trained
   `v0_conditioned` / `sel_reach_clamp`? If yes, the pilot has been post-training on the
   fixed-10 m/s action space since it was written, and no guard in this package can see it.
2. **The repoint must not be taken** by another agent reading only the sibling packages —
   `VERDICT.md` §2 and `test_the_repoint_would_have_FAILED_not_merely_been_redundant`
   exist so that conclusion is executable rather than a preference.
