# RL channel admissibility — the ruling per channel, and the declaration home that carries it

**Date** 2026-09-07 · **Stream** Architecture & Inference · **Branch** `agent/arch-inf-20260803`
**Commits** `3fd8db0` (the seams' declarations) · `1360fd9` (the three guards) · `<this package>`

⛔ **NO EVAL TIER AND NO FOUR-FAMILY TABLE.** Nothing here runs a model or produces a
trajectory. This is a **wiring-contract audit**; stamping a tier on it would be a
category error. ⛔ **ZERO GPU** — refcv5-v2 on the A40 was not touched, nothing was
shipped to a pod, and every run below was CPU-only in the off-Drive mirror.

---

## 0. The state that started this

`stack/tests/test_rl_forward_keys_cover_signature.py` was **red**, correctly:

```
FORWARD_KEYS is missing 3 channel(s) ... NOT declared diagnostic-only:
  ['nav_args', 'v_max_ms', 'v_max_valid']
```

`tanitad.rl.refc_adapter.forward_kwargs` iterates `FORWARD_KEYS`, so a missing channel
is never passed at all. The instruction was **not** to add three keys — the first
version of that test advised adding `gp_point`/`gp_valid`, and that advice would have
manufactured a label leak.

---

## 1. ⭐ THE RULING, per channel

Every claim below was verified **from source**, not inherited.

### 1.1 `gp_point` / `gp_valid` — EXCLUDED, **PERMANENT**

| | |
|---|---|
| **Where a rollout would get it** | Nowhere but the ego's own future pose |
| **Ruling** | Never admissible |
| **Unblock** | ⛔ NONE — and this is a decision, not an omission |

`GoalPointConfig` puts the goal at `t_goal_s = 4.0 s` and **refuses** a value at or
inside the scored horizon `t_pred_s = 2.0 s` (`__post_init__`: *a goal inside the
horizon is the ANSWER, not a route signal*). So the only supplier a batch has is the
ego's future path.

⭐ It costs nothing: the deployable arm **already has the goal** — `gp_head(ctx)`
predicts it from vision inside the forward. Leaving the kwarg at `None` is the correct
call, not a gap. ⇒ **Agreed with the inherited reading, and the "no route back" is now
written down as a decision rather than left as an absence.**

*Evidence: PUBLISHED-CODE — `stack/tanitad/refs/goal_point.py` (`GoalPointConfig.__post_init__`);
`stack/tanitad/refs/refc_v3.py` forward docstring and its `core.graft_gp_point` refusal.*

### 1.2 `nav_args` — EXCLUDED, **TEMPORARY (as BUILT)**

| | |
|---|---|
| **Where a rollout would get it** | `distance_m` ⭐ from a nav system; `time_s` ⛔ nowhere |
| **Ruling** | Not admissible **as built** |
| **Unblock** | A **distance-only mode** — `NAV_ARG_SLOTS = ("distance_m",)`, `NAV_ARG_DIMS = 2` |

**MEASURED from source, and the inherited reading is CONFIRMED:**

```python
# stack/scripts/s2_geom_emit_v7.py:713-716
return {"token": f"NAV_TURN_{side}",
        "args": {"distance_m": _arc_to(poses, key, nxt[0], hz),
                 "time_s": nxt[0]},
        "provenance": "ego-future", "oracle": True}
```

`time_s` is `nxt[0]` — **the time offset at which the ego's own future path reaches the
turn.** No nav system can know when you will arrive, because that depends on how you
drive: it is the ego's future **speed profile, inverted**, on the axis that owns
**88.7 %** of the oracle gap.

`distance_m` is different in kind. It is measured along the ego's driven path
(`_arc_to`), but **arc length to a fixed road feature is a property of the route** — a
map knows it without knowing how the car will be driven. *"Turn left in 300 m"* is a
real nav output; *"turn left in 11.4 s"* is not.

⚠️ **And there is no way to ship one without the other today** — all four verified:
`vocab_v7.NAV_ARG_SLOTS == ("distance_m", "time_s")`; `refc_v3.NAV_ARG_DIMS == 3` and
sizes `nn.Linear(NAV_ARG_DIMS, d_nav)`; the loader always emits `[dn, tn, 1.0]`
(`refc_v3_train.py`); and the forward **refuses any other width**. `--nav-args` is a
boolean — there is no distance-only mode to select.

⇒ **Agreed with the inherited reading.** The exclusion is on the whole channel and it
is **TEMPORARY**: the fix is a build option, not a ruling.

### 1.3 `v_max_ms` / `v_max_valid` — EXCLUDED, **TEMPORARY (this CORPUS)**

| | |
|---|---|
| **Where a rollout would get it** | ⭐ In principle a map/nav speed-limit service; ⛔ here, `max(ego realised speed)` over `[+2 s, +6 s]` |
| **Ruling** | Not admissible **from this corpus** |
| **Unblock** | A supplier that is **not** the ego's realised future (map / `map.xodr` / OSM / sign recognition) |

The PI's ruling stands and is not retracted: a posted limit is an **INPUT**, the way
`nav_command` is. The ladder, the units contract and the zero-init conditioner are
correct and unchanged. **What is not admissible is the value we can supply.**

`v7_labels.oracle_max_speed` states the provenance itself — *"its provenance is
`ego-future`: the training value is max of the ego's OWN REALISED speed over
[anchor+2 s, +6 s]"* — and gates the read behind `allow_oracle_nav`. And
`refc_v3_train.enable_max_speed` ships that value **RAW** into the batch; the ladder is
applied on the *model* side. So an adapter plumbing `batch["v_max_ms"]` plumbs the raw
ego-future speed.

⭐⭐ **I DID NOT STOP AT PROVENANCE, AND THE MEASUREMENT IS THE POINT.** Provenance
alone is not a sufficient discriminator — `nav_cmd` carries `provenance: "ego-future"`
on 4,719/4,719 records too, and it **is** plumbed. What separates them is measured:

5-fold **out-of-fold**, clip-disjoint, n = 4,572
(`Research/2026-09-06-max-speed-input/raw/quantization_panel_train.txt`):

| recovering the RAW ego-future ceiling from | R² | rmse |
|---|---|---|
| `v0` alone (control) | 0.8789 | 2.586 m/s |
| shuffled bin + `v0` (control) | 0.8789 | 2.586 m/s |
| **PINNED 8-step bin alone** | **0.9513** | 1.640 m/s |
| **PINNED bin + `v0`, per-bin** | **0.9702** | 1.282 m/s |
| ΔR² the bin ADDS over `v0` | **+0.0913** | cut 1.304 m/s |
| **UNRECOVERED share of the ego's future** | **0.0298** (train) / 0.0448 (eval) | |

⇒ **Quantization does not launder it.** Even at the pinned 8 steps, **97.0 %** of the
ego's realised speed ceiling over the scored horizon is recoverable from the channel.
The E16 module rejects the 13-step ladder at R² = 0.9884 as *"the bin IS the raw value
in disguise"*; 0.9702 is the same objection with a smaller number in front of it.

For a **supervised** arm that is a declared train/deploy mismatch, stamped and
reported — the module is right about that and it is not blocked. For an **RL rollout**
on the axis owning 88.7 % of the gap, it is the answer's upper envelope, handed over on
the very horizon being scored.

⇒ **Agreed with the inherited call, with a stronger and different reason.** Not
"provenance is ego-future" (which would have to exclude `nav_cmd` too, and the
programme deliberately plumbs it under a mandatory `shuffled` control) but **the
measured recoverability of the scored answer, R² = 0.9702 out-of-fold**.

### 1.4 ⭐ The discriminator that came out of this

Two tests, applied slot by slot, not channel by channel:

1. **Is the QUANTITY one an external system could supply?** — `nav_cmd` yes,
   `distance_m` yes, a posted limit yes; `time_s` no, a goal point no.
2. **Does OUR realisation of it encode the scored answer?** — `nav_cmd` no (a 3-way
   token; measured content 2.3 %); `v_max_ms` **yes, at R² = 0.9702**.

A channel must pass both. `nav_cmd` passes both and is plumbed **with the `shuffled`
control that `nav_conditioning` already makes mandatory**. `v_max_ms` passes (1) and
fails (2) *on this corpus only* — hence a supplier unblock, not a ruling against the
channel.

---

## 2. ⛔ What I found that was NOT in the brief, and is worse

### 2.1 The launch preflight was refusing every RL launch — and recommending the leak

`tanitad.rl.channel_guard.assert_forward_channels_complete` had **no exclusion concept
at all**: it compared `FORWARD_KEYS` against the **raw** signature.
`stack/scripts/rl_control_space_preflight.py:55` calls it on every launch. MEASURED
before the change:

```
PREFLIGHT RAISED: ChannelDriftError
  ... the signature accepts 12 optional conditioning channels, the adapter plumbs 7,
  and forward_kwargs() would SILENTLY DROP gp_point, gp_valid, nav_args, v_max_ms,
  v_max_valid. An arm launched in this state runs blind to ...
```

Two of those five are the goal point. **The refusal text instructed the operator to
create the label leak the goal-point stream had already ruled against.**

### 2.2 Its test suite was GREEN on that

`test_rl_channel_guard.py`'s live-pair test read `if missing: expect refusal; else:
expect pass` — an expected value of *"whatever the code does"* **cannot fail**. Its
fixture forward also used `gp_point`/`gp_valid` as the channels a `COMPLETE` tuple must
plumb, so the suite modelled the leak as the desired state. Both are fixed; the fixture
channels are renamed to neutral names so no test in the file teaches it.

### 2.3 A third copy of the same wrong comparison, already red

`test_rl_refc_adapter_robust.py::test_forward_kwargs_plumbs_every_channel_the_forward_accepts`
asserted `set(FORWARD_KEYS) == sig` — red before this change, and demanding the leak.
Now `sig - excluded`, with a vacuity gate and an assertion that the excluded channels
are **absent** from the kwargs rather than merely `None`.

### 2.4 ⚠️ HEAD's `refc_adapter.py` blob was still agent-blind

Not introduced here, found here:

```
HEAD:      FORWARD_KEYS = ("nav_cmd","v0","lan","nav_known","ego_state","withheld_speed")
worktree:  FORWARD_KEYS = (... "withheld_speed", "agent_gt")
```

The 2026-09-06 *"landing the guard silently reverted the fix"* incident was **still live
at HEAD** — the worktree's `agent_gt` fix had never been committed. Commit `1360fd9`
lands it.

---

## 3. What was built

**A declaration home that carries the reason per channel, owned by the seam.**

`tanitad.channel_admissibility` defines **only** the record shape and the union — never
a reason. `ChannelExclusion` **refuses construction** without `reason`, `unblock`,
`evidence` and `owner` (placeholder strings and sub-floor lengths are refused by name).
`permanent` defaults to `False`: temporary is the humbler default and a permanent call
must be typed. `SEAM_MODULES` lists the participants — central by necessity, because a
registry populated as an import side effect would hold *nothing* if the seam were not
imported, and the adapter would then plumb a label with no guard firing.

The three declarations live where the channels do:

| seam | declares |
|---|---|
| `stack/tanitad/refs/goal_point.py` | `gp_point`, `gp_valid` — PERMANENT |
| `stack/tanitad/models/nav_conditioning.py` | `nav_args` — TEMPORARY (distance-only mode) |
| `stack/tanitad/refs/max_speed_input.py` | `v_max_ms`, `v_max_valid` — TEMPORARY (non-ego-future supplier) |

`goal_point.DIAGNOSTIC_ONLY_FORWARD_KWARGS` survives as a **derived alias**, computed
from the records so the names cannot desync from the reasons.

**The guard now refuses three defects instead of one**, and checks the leak direction
**first** (a blind arm under-performs visibly; an arm fed the answer reads as a win):

| | |
|---|---|
| `MISSING` | a required channel absent → the arm runs blind (`ChannelDriftError`) |
| `LEAKED` | a declared channel present → the arm is fed a label (`ChannelLeakError`) |
| `STALE` | a declared channel the forward no longer has → covers nothing |

Refusals **quote the owning seam's reason and unblock condition**, and
`forward_channel_report` banks the full declarations into the preflight artifact, so
the run record carries *why* a channel was withheld.

**New test guards** (`test_rl_forward_keys_cover_signature.py`), on top of the four
that were kept verbatim in substance:

* every exclusion carries a non-empty **reason**, **unblock condition**, owner and
  evidence class;
* the permanent and temporary exclusions are **distinguishable** — if everything were
  marked permanent the reason guard would still pass and the two liftable channels
  would quietly become forever-channels.

---

## 4. ⛔ The mutation proof — the guards can fail

`code/mutate.py`, run in the off-Drive mirror; full output in `raw/mutation_proof.txt`.
Each mutation reintroduces one real defect in the real files and requires a red; the
controls require a green, because a check that always fails proves nothing either.

| # | mutation | expected | got | preflight refusal type |
|---|---|---|---|---|
| — | CONTROL, unmutated | GREEN | **GREEN** (51 passed) | `PASS` |
| M1 | an exclusion loses its **reason** | RED | **RED** (18 failed) | `ExclusionDeclarationError` |
| M2 | declared-excluded `gp_point` **moved into `FORWARD_KEYS`** | RED | **RED** (4 failed) | `ChannelLeakError` |
| M3 | a new forward channel in **neither** place | RED | **RED** (4 failed) | `ChannelDriftError` |
| — | CONTROL, after restore | GREEN | **GREEN** (51 passed) | `PASS` |

**5/5 expectations met**, and the three refusal *types are distinct* — the guard
discriminates between "cannot say why", "fed the answer" and "running blind" rather
than emitting one undifferentiated red.

Regression sweep: **176 passed** across the 8 affected suites
(`test_rl_forward_keys_cover_signature`, `test_rl_channel_guard`,
`test_rl_refc_adapter_robust`, `test_goal_point`, `test_max_speed_input`,
`test_max_speed_wiring`, `test_refc_v3_nav_args`, `test_nav_conditioning`) — was
1 failed + 1 blocked preflight.

---

## 5. ⭐ What an RL rollout is still blind to — DECISION or ACCIDENT?

**Five of the twelve forward channels are now withheld by DECISION** — each with a named
owner, a source-verified reason, an evidence class and, for three of the five, a written
route back. **The remaining blindness is one level down and is still an ACCIDENT:**
`forward_kwargs` does `batch.get(k)`, so a *required* channel whose batch key is simply
absent reaches the forward as `None` with nothing raising. MEASURED on
`batch = {"frames": ...}`:

```
kwargs silently carrying None: agent_gt, ego_state, lan, nav_cmd, nav_known, v0, withheld_speed
config-flag guard in _REQUIRING_FLAG:  ego_state -> 'ego_state_inject';  ALL SIX OTHERS -> None
```

`ego_state` is caught by `assert_conditioning`; `agent_gt` is caught by the model's own
two-direction refusal; `nav_cmd` is `None` **on purpose** (the C6 confound / `os_navzero`
arm). ⇒ **`lan`, `nav_known`, `withheld_speed` and `v0` can each be silently `None` in an
RL rollout today, and nothing anywhere would say so.** The key set is now a decision;
**the value flow is not**, and that is the next guard — the same defect class this
package closed, one level below it.

---

*Evidence classes: PUBLISHED-CODE (every line cite above re-read from source this
session, not inherited) · MEASURED (the OOF quantization panel; the mutation table; the
silent-`None` probe). Contended files `refc_v3.py` and `refc_v3_train.py` were READ ONLY
and are byte-unchanged — verified by `git diff HEAD --stat` at end of turn.*
