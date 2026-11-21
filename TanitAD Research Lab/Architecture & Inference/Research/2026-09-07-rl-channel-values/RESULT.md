# The RL channel VALUE flow — four channels, four verdicts, and the one that was live

**Date** 2026-09-07 · **Stream** Architecture & Inference · **Branch** `agent/arch-inf-20260803`
**Closes** the gap the sibling package left open (`Research/2026-09-07-rl-channel-admissibility/`
§5: *"`lan`, `nav_known`, `withheld_speed` and `v0` can each be silently `None` in an RL rollout
today, and nothing anywhere would say so"*).

⛔ **NO EVAL TIER AND NO FOUR-FAMILY TABLE.** No model produces a trajectory anywhere in this
package. It is a **wiring-contract audit**; stamping an eval tier or a longitudinal/lateral/
tactical/strategic table on it would be a category error.
⛔ **ZERO GPU.** refcv5-v2 (A40, PID 2560646) was not touched, nothing was shipped to a pod, and
every run below was CPU-only in an off-Drive mirror I created for this work
(`C:\Users\Admin\tanitad-chanval`) rather than `tanitad-wt`, which re-syncs from the repo
mid-run and silently drops edits.

---

## 0. The inherited claims — VERIFIED from source, and one of them is wrong

| claim | verdict | source |
|---|---|---|
| `assert_conditioning` IS wired ⇒ COVERAGE problem, not wiring | ✅ **CONFIRMED** | `refc_adapter.py:206`, inside `make_refc_sample_fn` |
| every `None`-flag entry evaluates False and is never checked | ✅ **CONFIRMED** | `refc_adapter.py:155` — `req[key] = bool(getattr(cfg, flag, False)) if flag else False` |
| `nav_cmd` and `agent_gt` carry good reasons, leave them alone | ✅ **CONFIRMED** (reasons kept verbatim in substance) | `refc_adapter.py:111-115`, `:123-133` |
| four carry nothing | ✅ **CONFIRMED** | `refc_adapter.py:118,119,120,122` |
| *"no config flag exists"* for `nav_known` / `withheld_speed` | ⛔ **FALSE for `nav_known`** | `cfg.core.nav_known_channel` — `refc.py:836` |

⛔ **THE PREMISE UNDER THREE OF THE FOUR WAS FALSE.** The old note hunted a **naming
convention** (`nav_known_inject`, `withheld_speed_inject`), did not find it, and concluded no
field existed. The fields are real and **NESTED** — which is exactly the trap the `agent_gt`
entry documents one line below, applied to itself:

| channel | declaring field | depth | reachable by the old flat `getattr`? |
|---|---|---|---|
| `v0` | `cfg.core.anchors.v0_conditioned` (`refc.py:383`) | **2** | ⛔ no |
| `lan` | `cfg.core.graft_lan` (`refc.py:693`) | 1 | ⛔ no |
| `nav_known` | `cfg.core.nav_known_channel` (`refc.py:836`) | 1 | ⛔ no |
| `withheld_speed` | genuinely none — **and correctly so** | — | — |

⚠️ **Every line cite the old note carried was STALE**, and I only found that by checking:
`refc_v3.py:986-991`, `:1000`, `:1005`, `:1006`, `:1012`, `:380`, `:356`, plus
`test_rl_refc_adapter_robust.py`'s `:1000`. The live forward begins at **`refc_v3.py:1361`** and
the `agent_gt` refusals are at **`:1398-1403`** and **`:1404-1409`**, not `:1006`/`:1012`. The
reasoning was exemplary; the coordinates had rotted. All are corrected, and
`test_rl_channel_value_flow.py` now **resolves** every predicate against a real config rather
than trusting a comment.

---

## 1. ⭐ THE VERDICTS — `v0` first

### 1.1 ⭐⭐ `v0` — **ASSERTABLE (case 1)**, and it was the worst of the four

Predicate: `core.anchors.v0_conditioned` **or** `core.sel_reach_clamp`.

⛔ **A missing `v0` does not merely re-condition the policy — it SILENTLY REPLACES THE ACTION
SPACE.** The chain, every link re-read from source:

```
refc.py:3097   v_ms = v0 if (cfg.sel_reach_clamp and v0 is not None) else None
refc.py:2128   bank = self.roll_bank(v_ms, ego_keep, ...)
refc.py:1722   if v_ms is None or anchor_withheld_bank == "none":  v = full(ref_speed)
```

⇒ **every anchor is rolled at the 10 m/s reference speed instead of the window's measured
speed, and nothing raises.** MEASURED delta between those two vocabularies
(`AnchorConfig` docstring, `refc.py:363-370`): the fixed-path set reads **0.3773 m**
oracle-in-vocabulary, the v0-conditioned family **0.2610 m**.

Two further silent sites on the same omission:
* `refc.py:2360` — `if sel.reach_clamp and v_ms is not None:` — the S2 reachability band is
  **skipped entirely**.
* `refc.py:2970-2977` — `v = 0` and `keep = 0`. Under X15 (`ego_valid_channel`) the zero is at
  least *flagged* rather than a lie, but the rollout then runs `keep = 0` on **100 %** of rows
  where training saw `keep = 1` on `1 - ego_dropout` of them. ⭐ That is precisely the argument
  this module's own docstring already refuses to accept for `ego_state`: *"dropout … does not
  make 'always missing' the same distribution the checkpoint was fitted under."*

⭐⭐ **AND IT IS LIVE, NOT LATENT.** `MODEL_REGISTRY.md` shows **both** refcv4b (line 2902) and
refcv5 (line 2949) ship **`--anchor-v0-conditioned`** in `config.json['argv']`. Those are the
checkpoints an RL rollout post-trains. ⇒ on today's fleet the predicate reads **True**, and
before this change a rollout that dropped `v0` would have silently sampled a different
vocabulary.

⭐ Independent corroboration of the severity, already in the registry and not written by me:
`refc_v3_train.py:538-545` **explicitly forbids** reading `model.core.decoder.anchors` for a
v0-conditioned vocabulary (the trainer must use `out["anchor_bank"]`), and the registry records
that ignoring this produced a real instrument defect — refcv4b's `oracle_sel` / `anchor_acc` are
marked **NOT VALID ON THAT ROW** for exactly this reason. The stored bank and the rolled bank are
already known to be non-interchangeable; a `None` `v0` swaps one for the other silently.

⛔ **Asserting it refuses no valid launch.** Every RL batch builder in the repo supplies `v0` —
`rl_pilot_refc21.py:136`, `rl_refcv3_min.py:469`, `rl_progress_leadcap_ab.py:142`,
`rl_a0_coverage.py:183`. The refusal can only fire on a batch that forgot it.

### 1.2 `lan` — **ASSERTABLE (case 1)**, latent on today's arms

Predicate: `core.graft_lan`.

⛔ `refc.py:3072` — `if self.cfg.graft_lan and lan is not None:` — the route encoder (`lan_enc`,
`lan_direction`) is **skipped**, and `refc.py:2887` says so outright (*"None -> the seam is
skipped entirely"*).

⭐ **The obvious defence does not hold, and I checked it rather than assuming.** *"The route is
missing on ~75 % of windows anyway, and `route_dropout` exists so the planner never becomes
route-dependent"* — true, but `route_dropout` **multiplies a present tensor by zero**
(`refc.py:3073-3077`), it does not pass `None`. When `graft_lan` is on the trainer swaps in
`lan_dataset_class` so **every** window carries a `lan` tensor
(`refc_v3_train.py:3746-3754`, `refc_train.py:713-718`), and "no route here" is expressed by the
per-anchor **`valid` flag** (`LanConfig.feats = 4` = cos, sin, lat_norm, valid; `refc.py:509`).
`lan=None` is not that in-distribution state — it is the batch forgetting the key, and it
**bypasses the encoder** rather than feeding it an honest zero.

⚠️ **Honest scope:** refcv4b's registry row states **"⛔ NO `--echo-base`, NO `--graft-lan`"**
(line 2897), so on today's checkpoints this predicate reads **False** and the assertion is
**latent**. It arms itself the moment a `graft_lan` build is post-trained. That is the correct
place for it — but I am not claiming it caught anything today.

### 1.3 `nav_known` — **GUARDED ELSEWHERE (case 2)**, and asserting it would be WRONG

A declaring field **does** exist (`cfg.core.nav_known_channel`) — the old note's claim was
false — **and it still must not be asserted here.** The model already refuses both silent
directions:

* `refc.py:3006-3009` — supplied while the gate is off → raise
* `refc.py:3017-3022` — gate on **and a `nav_cmd` was supplied** but the bit is missing → raise
  (*"Defaulting it to 1.0 would assert a judgement the labeller never made"*)

⛔ **The third branch is the reason:** `refc.py:3012-3016` — gate on, `nav_cmd is None` — the bit
legitimately defaults to `0.0`, because *"the `follow` fallback IS the sentinel"*. REF-C's
published arm decodes with `nav_cmd=None` (C6). A flat *"gate on ⇒ require `nav_known`"* would
**refuse the programme's own standard eval arm** — the identical trap the `nav_cmd` entry already
documents, one channel over. There is no silent divergence left to catch, only a plumbing duty
`FORWARD_KEYS` discharges. ⇒ **no assertion; a positive reason, recorded.**

### 1.4 `withheld_speed` — **NOT AN EXTERNAL CHANNEL AT ALL.** `None` is the CORRECT value

It is the model's **own** predicted 2 s speed routed back in, not something a rollout supplies:

* `refc.py:1697` — *"``withheld_speed`` [B] (optional) is the model's OWN predicted speed"*
* ⭐ `refc.py:2944-2946` — `if withheld_speed is None: withheld_speed = hk.get("bank_speed_pred")`
  — the hierarchy hook **fills it in itself**
* `refc.py:1674-1679` — read only under `anchor_withheld_bank == "pred"`, and only on withheld rows
* its one external use is the **eval-time SHUFFLE control**, which passes a permuted copy
* on a non-hier build it falls back to the fixed roll, which `refc.py:1663-1666` documents as
  intended (*"a caller that has no goal head"*)

⇒ asserting it would **refuse the normal path**. The old *"plumbed, not asserted"* was right by
accident and for no stated reason; it now carries the reason.

---

## 2. What was built — the same discipline the sibling imposed, one level down

The sibling made an **exclusion** unconstructible without `reason`/`unblock`/`evidence`/`owner`.
⭐ An **unasserted required channel** is the same object one level down — a decision the next
reader cannot see — so `_REQUIRING_FLAG` becomes **`CHANNEL_REQUIREMENTS`**, a tuple of
`ChannelRequirement` records that **refuse construction** without them. ⛔ *"No config flag
exists"* is refused **by name** as a placeholder: it is a fact about the code, not a reason, and
for three channels it was also false.

⭐⭐ **AND THE TRAP ITSELF IS CLOSED, NOT DOCUMENTED.** Predicates are now **dotted paths** read by
`_read_predicate`, which **RAISES** on a path that does not resolve instead of returning `False`:

```python
nxt = getattr(obj, part, _MISSING)
if nxt is _MISSING:
    raise ConditioningError(...)        # never `return False`
```

That single line is the difference between a guard and a false green. A mistyped, moved or
nested field can no longer produce a check that is present in the source, passes every batch and
can never fire — which is what happened to `v0`, `lan` and `nav_known`.

⚠️ **The test surface had the same blind spot as the code.** `test_rl_refc_adapter_robust.py`'s
`_Cfg` stub carried exactly two **flat** attributes, so no test in the file could ever exercise a
nested predicate — and `test_no_requirement_names_a_config_field_that_does_not_exist` was
doubly inert: it iterated a map whose six `None` entries skipped the loop body, and used
`hasattr` on the **class**, which cannot see a nested field even when named. Both are fixed: the
stub is now the **real `RefCV3Config`**, and the test **resolves** each dotted path against an
instance.

---

## 3. ⛔ THE MUTATION PROOF — the guard can fail, and the controls can pass

`code/mutate.py`, CPU-only in the off-Drive mirror; full output in `raw/mutation_proof.txt`.
Each mutation reintroduces one real defect in the real file and requires a red; the controls
require a green, because a check that always fails proves nothing either.

⚠️ **Deliberately NOT the shape the sibling found** in `test_rl_channel_guard.py` — no
expectation here is *"whatever the code does"*. Every pair is two separate tests with hard-coded
expectations: one that **must refuse**, one that **must pass**.

| # | mutation | expected | got | pytest |
|---|---|---|---|---|
| — | CONTROL, unmutated | GREEN | **GREEN** | 71 passed |
| M1 | ⭐ **the flat `getattr` restored — THE ORIGINAL DEFECT**: a nested predicate reads False forever | RED | **RED** | 7 failed |
| M2 | `v0` loses its predicates → back to unasserted | RED | **RED** | 4 failed |
| M3 | `lan` loses its predicate → back to unasserted | RED | **RED** | 2 failed |
| M4 | `nav_known` **OVER**-asserted → would refuse the `nav_cmd=None` eval arm | RED | **RED** | 1 failed |
| M5 | an unasserted channel's `unblock` becomes `"tbd"` | RED | **RED** | 2 errors (import-time refusal) |
| M6 | `withheld_speed` drops out of the records → a plumbed channel nobody ruled on | RED | **RED** | 3 failed |
| M7 | `assert_conditioning` stops raising → the guard is inert | RED | **RED** | 5 failed |

**9/9 expectations met.** ⭐ M4 matters as much as M2: it proves the suite refuses
**over**-assertion too, which is the failure mode that gets a guard deleted rather than fixed.

**Regression sweep: 196 passed** across the 9 affected suites (the sibling's 8, plus the new
`test_rl_channel_value_flow.py`) — was 176 + 20 new. `rl_control_space_preflight.py` CHECK 0
still reads **`=> COMPLETE`**.

---

## 4. ⭐ THE ANSWER TO THE STANDING QUESTION

> **Can a required channel still arrive as `None` unnoticed?**

**No — for every channel that is plumbed, the case is now closed, and three of the four verdicts
differ.** `v0` and `lan` are **asserted** from their own nested declaring fields and refuse a
batch that drops them; `ego_state` was already asserted; `nav_known` and `agent_gt` are refused
**by the model itself**, in both directions, with file:line recorded; `withheld_speed` is not an
external channel and `None` is its correct value; `nav_cmd` is `None` on purpose.

⚠️ **The one residual, stated plainly:** the guard fires on the **first batch only**
(`refc_adapter.py:205`, `checked["done"] = True`). A rollout whose *later* batches drop a channel
would not be caught. That is a deliberate cost/benefit call inherited from the original design,
not something this package changed — but it is now the only remaining hole in the value flow, and
it is written down rather than left to be rediscovered.

---

*Evidence classes: **PUBLISHED-CODE** — every line cite above re-read from source this session,
never inherited (the inherited ones were stale and are corrected). **MEASURED** — the mutation
table; the 196-test sweep; the registry argv reads for refcv4b/refcv5. Every absence claim was
paired with a same-breath control that read non-zero. Contended files `refc_v3.py`,
`refc_v3_train.py` and `refc.py` were **READ ONLY** and are byte-unchanged —
`git diff HEAD --stat` empty at end of turn, `refc_v3.py` md5 `b5535b59bf91ce83f94b38a0cb932dc7`
equal to session start.*
