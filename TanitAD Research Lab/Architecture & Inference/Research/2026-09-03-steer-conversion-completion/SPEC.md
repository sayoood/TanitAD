# SPEC — completing the κ→steer crossing, and the cost's time grid

**Architecture & Inference FlyWheel · 2026-09-03 · BACKLOG R26 (Task A) + R27 (Task B)**
**Written BEFORE any edit.** Every source claim below is a read of the tree at
`refa_v1.py` md5 `6e494ef8116031bf5943ec477c2b8f9c` (1,920 lines, CRLF), branch
`agent/arch-inf-20260803`.

⚠️ This is a **CORRECTNESS** repair. `D-REFAV1-BOUNDARY-NULL` measured that the completed
conversion buys the planner **nothing**: `share(ii) = +0.0000 [0.0000, 0.0000]`, a zero-width
paired episode-cluster interval on 140/140 windows, both checkpoints. Nothing here is
expected to move a driving number and nothing here is sold as if it would.

---

## 0. Why this exists — the two defects, confirmed at source

### R26 — the κ→steer conversion is HALF applied

`as_command` is called at **exactly one site in the whole file**:

```
refa_v1.py:1832        controls_m = as_command(controls, model_action_units)
```

(verified: one grep hit for `as_command(` in `refa_v1.py` outside the import at `:61`).
That site is inside `_cost_chunk`, which scores **candidates**. The **goal** is rolled
through a different, unconverted path:

```
refa_v1.py:1683        acts = self.augment_actions(ctrl, v0)[:, ::stride][:, :cfg.tac_steps]
refa_v1.py:1684        goal = self.tactical.rollout(self._tac_field(last), acts, ...)
```

`ctrl` at `refa_v1.py:1680` comes from `canonical_controls(...)`, whose channel 1 is
GEOMETRY by construction (`GOAL_KAPPA_TURN = 0.08` at `refa_v1.py:119`, asserted by
`test_steer_curvature_interface.py::test_D4_the_goal_constants_stay_in_CURVATURE`).
⇒ with `model_action_units="steer"` the candidate is imagined in COMMAND units and the
goal it is scored against is imagined in GEOMETRY units. **MEASURED consequence**
(`Research/2026-09-03-refav1-cost-surface/`, 25 `TURN_R` windows): the converted canonical
turn lands **one ULP further** from its own goal — T1 5.96e-08 → 1.19e-07, advantage over
`cv` 0 → **−5.96e-08**. Default-OFF, so nothing banked is affected; the flag is a trap.

### R27 — the cost consumes the 2.0 s plan on the 0.6 s clock

With `plan_level="tactical"` (`refa_v1.py:265`, the shipped default and both banked
checkpoints), `_cost_chunk` rolls `self.tactical` on the planner's `[n, H, 2]` controls:

```
refa_v1.py:1786        search_pred = self.tactical if coarse else self.operative
refa_v1.py:1840        zk = pred.rollout(z, acts, intent=intent, last_only=True)
```

`H = cfg.plan_steps = round(plan_horizon_s / op_dt) = round(2.0 / 0.2) = 10`
(`refa_v1.py:247, 548`). The tactical predictor advances `tac_dt = 0.6 s` per step
(`refa_v1.py:191`), so those 10 entries are consumed as **6.0 s** — each 0.2 s action
imagined as lasting 0.6 s, and the candidate's tactical action *j* is operative action *j*.

The GOAL is built on the tactical clock **correctly** — and so is the TRAINING path, which
is the decisive fact:

```
refa_v1.py:1270 (forward, TRAINING)   tac_a = acts_in[:, ::self._stride(self.cfg.tac_dt)]
refa_v1.py:1272                       self.tactical.rollout(..., tac_a[:, :self.cfg.tac_steps], ...)
refa_v1.py:1683 (_imagine_tactical_goal)  acts = self.augment_actions(ctrl, v0)[:, ::stride][:, :cfg.tac_steps]
```

⇒ every tactical action the predictor was ever **trained** on is operative action `3j`.
`_cost_chunk`'s dense feed is the only place in the programme that hands the tactical
predictor a dense operative sequence, and it is therefore out of distribution as well as
out of units.

### R27b — T4 (`target_speed`) is dead in every production read

```
refa_v1.py:1852        if target_speed is not None:
refa_v1.py:1854            c = c + 0.10 * (v_end - target_speed).pow(2)
```

The two production callers of `RefAV1.plan` do **not** pass it:

| caller | line | passes `target_speed`? |
|---|---|---|
| `taniteval/tools/refav1_arm.py` | 735–737 | **no** (`feats, v0, nav_cmd, plan_cfg, goal_field, model_action_units`) |
| `taniteval/tools/cost_surface_probe.py` | 515 | **no** (`feats, v0, nav_cmd, plan_cfg, goal_field`) |

⇒ the shipped cost has **THREE** live terms (goal, jerk, curvature), not four. Confirmed
independently by `stack/tests/test_cost_surface_probe.py::test_target_speed_term_is_dead_unless_supplied_and_is_the_shipped_form`.

### Preflight — nothing here can touch a live training run

⛔ Verified before editing, two independent probes, **no pod was contacted**:

1. `grep -rn "\.plan(" --include=*.py stack taniteval colab` → hits only in
   `stack/tests/**`, `stack/experiments/alpasim-gsplat/{closedloop,openloop}_drive.py`
   (a *different* `policy.plan(list(frames), intr, v, nav, ...)` signature — not
   `RefAV1.plan`), `taniteval/tools/refav1_arm.py`, `taniteval/tools/cost_surface_probe.py`.
2. PowerShell `Select-String '\.plan\(|_imagine_tactical_goal|imagined_goal'` over
   `stack/scripts/**` and `colab/**` → **zero hits**.

⇒ `plan()`, `_imagine_tactical_goal` and `imagined_goal` are called by **no trainer**.
`stack/scripts/refa_v1_train.py` (refav1 on Thor) and the refcv3 trainer reach only
`forward()` / `_cf_term`, neither of which this change touches.

---

## 1. THE CROSSING-POINT TABLE

The contract, restated so the table can be checked against it:

* the predictor's action channel is **`(a, steer)`** — COMMAND, road-wheel angle
  (`kinematic.py:20-24`; `physicalai.signals_at` wrote `steer = arctan(L_enc·κ)` with
  `L_enc = 2.9`, `kinematic.py:36-51`);
* anything **integrating a path** needs **`κ = tan(steer)/L`** — GEOMETRY;
* a planner **candidate** converts `κ → arctan(L·κ)` **at the model boundary** while the
  trajectory it scores stays in κ.

`M` = crosses into the **MODEL** (must be COMMAND). `G` = stays in / enters **GEOMETRY**
(must be κ). `–` = unit-invariant (channel 0 only).

| # | site (`file:line`) | tensor | class | required units | TODAY | AFTER |
|---|---|---|---|---|---|---|
| 1 | `refa_v1.py:1679-1681` `canonical_controls(...)` → `ctrl` | goal profile `[B,30,2]` | G (produced) | κ | κ ✓ | κ — unchanged |
| 2 | `refa_v1.py:1683` `augment_actions(ctrl, v0)` → `:1684` `tactical.rollout` | goal actions `[B,10,a_in]` | **M** | **steer** | **κ ⛔ DEFECT** | `as_command(ctrl, units)` first |
| 3 | `refa_v1.py:1686-1687` return `{"controls": ctrl}` | provenance | G | κ | κ ✓ | κ — unchanged |
| 4 | `refa_v1.py:1818-1820` `seed = goal_action["controls"][:plan_steps]` → `seed_pool` → `icem_plan` | search seed `[1,H,2]` | G (search) | κ | κ ✓ | κ — unchanged |
| 5 | `refa_v1.py:1832` `as_command(controls, model_action_units)` → `:1838` `augment_actions` → `:1840` `rollout` | candidate `[n,H,a_in]` | **M** | steer | steer ✓ | unchanged (routed through the named boundary) |
| 6 | `refa_v1_plan.py:162-166` `_clip` (`kappa_max = 0.2`) | candidate | G (search box) | κ | κ ✓ | unchanged |
| 7 | `refa_v1.py:1846` jerk `controls[:,:,0]` | cost T2 | – | invariant | ✓ | unchanged |
| 8 | `refa_v1.py:1847` curvature `0.05·controls[...,1]²` | cost T3 | G | κ | κ ✓ | unchanged |
| 9 | `refa_v1.py:1852-1854` `v_end` from `controls[...,0]` | cost T4 (**dead**) | – | invariant | ✓ | unchanged (see §4) |
| 10 | `refa_v1.py:1888` coarse→fine re-score `_cost_chunk(stack, pred=self.operative, z0=last)` | winner + baselines | **M** | steer | via #5 ✓ | unchanged |
| 11 | `res.controls` → `refav1_arm.paths_from_controls` / `refa_v1_plan.unicycle_paths` | deployed path | G | κ | κ ✓ | unchanged |
| 12 | `refa_v1.py:1264` `forward()` `augment_actions(actions, v0)` (**TRAINING**) | loader actions | M | steer | **already steer** ✓ | ⛔ MUST NOT convert |
| 13 | `refa_v1.py:1614-1616` `_cf_term` `augment_actions(a[:, :j+1], v0)` (**TRAINING**) | loader actions | M | steer | **already steer** ✓ | ⛔ MUST NOT convert |
| 14 | `refa_v1.py:1279` `str_a = actions[:, ::stride]` (**TRAINING**, strategic) | loader actions, 2-wide | M | steer | already steer ✓ | ⛔ MUST NOT convert |

**Rows 12–14 are why the conversion cannot live inside `augment_actions`.** The loader's
`actions` are the v2ep COMMAND channel already; converting there would double-convert the
training path and silently change every live run. The boundary is *planner-side*.

**Only row 2 is wrong.** Rows 5 and 10 already convert; rows 1, 3, 4, 6, 8, 11 are
correctly GEOMETRY; rows 7, 9 are unit-invariant; rows 12–14 are training and already
COMMAND.

---

## 2. THE CHANGE (Task A) — the boundary gets a name, and both paths go through it

Two designs were available. **Chosen: the conversion moves one level up into a single
named boundary helper, and both crossing sites call it.**

> *Justification, two sentences.* The conversion is a property of the **boundary**, not of
> either path, and the defect was exactly that one path carried it and the other did not —
> so a repair that adds a second `as_command` call leaves the same shape of bug available
> to the third crossing site somebody adds next. Naming the boundary makes "did this
> crossing convert?" a one-line grep, and lets a test assert that every `augment_actions`
> call inside `plan()`'s graph is reached through it.

```python
def _model_actions(self, controls, v0, units):          # THE PLANNER→MODEL BOUNDARY
    return self.augment_actions(as_command(controls, units), v0)
```

* `_imagine_tactical_goal(..., *, units="kappa")` — keyword-only, defaulted, so the
  3-positional-arg call in the read-only `taniteval/tools/cost_surface_probe.py:243-245`
  keeps working and keeps measuring the shipped behaviour.
* `imagined_goal(..., model_action_units="kappa")` forwards it.
* `plan()` passes `units=model_action_units` at `:1800`.
* `_cost_chunk` at `:1832-1838` is rewritten as one `self._model_actions(...)` call —
  same two operations, same order, same objects.

**Order is safe.** `as_command` touches channel 1 only; `augment_actions` derives channel 2
from channel 0. So conversion-then-augment is identical in channels 0 and 2 to
augment-then-convert, and the subsample `[:, ::stride][:, :tac_steps]` commutes with both
(it is an index op on the time axis). This is the same argument the shipped comment at
`refa_v1.py:1836-1837` already makes for the candidate.

**Default OFF is byte-identical by construction**: `as_command(x, "kappa")` returns the
input **object** unchanged (`kinematic.py:86-95`), so under the default `_model_actions`
*is* `augment_actions`.

---

## 3. THE CHANGE (Task B) — one flag, `cost_time_grid`

New call-site argument on `plan()`, exactly like `model_action_units` and for the same
reason (`refa_v1.py:1735-1737`: a config field would change every serialised config dict
while a training run is live).

| value | meaning | applies when |
|---|---|---|
| `"dense"` (**DEFAULT**) | today's behaviour, byte-for-byte: all `H` operative entries fed to whichever predictor is rolling. | always |
| `"tactical"` | tactical step *j* consumes operative action `min(j·stride, H−1)`, for `j < tac_steps`. | **only** when the rolling predictor **is** `self.tactical` |

For `j·stride < H` this is exactly `[:, ::stride]` — the map `forward()` (`:1270`) trains
with and `_imagine_tactical_goal` (`:1683`) builds the goal with. Beyond the plan's 2.0 s
window the plan's **final** action is held (zero-order hold), which is the standard
receding-horizon continuation and the only defined way to span the 6 s the cost is
documented to span (`refa_v1.py:247`: *"optimised window; cost spans 6 s"*).

At `H = 10`, `stride = 3`, `tac_steps = 10` the index map is
`[0, 3, 6, 9, 9, 9, 9, 9, 9, 9]` against today's `[0..9]`.

**Why hold rather than truncate to 4 tactical steps (2.4 s).** Truncation would put the
candidate's terminal field at 2.4 s and the goal's at 6.0 s, and `1 − cos` between two
fields at different horizons is dominated by *how far ahead* they are, not by *where they
go*. Holding keeps both terminal fields at 6.0 s, which is the only configuration in which
the cosine means what the docstring says it means (`refa_v1.py:1739-1741`, "the tactical
brain's own imagined 6 s field").

**Guaranteed no-op cases** (each becomes a test that must read a known value):
`pred is self.operative` (the coarse→fine re-score, row 10) — untouched by construction; a
**constant** candidate — the two index maps select the same values, so the cost is
identical under both grids; `stride == 1 and tac_steps == H` — the maps coincide.

**Residual mismatch, stated because the fix does NOT remove it.** Even with both repairs
the *injected seed* cannot reproduce the goal: the seed is `ctrl[:10]` (2.0 s of a 6.0 s
profile), so its regridded map is `[0,3,6,9,9,9,9,9,9,9]` while the goal's is
`[0,3,…,27]`. They agree for `j ≤ 3` and diverge after. **The plan horizon is shorter than
the goal horizon and no unit or grid repair can change that** — it is a `plan_horizon_s`
design question and is escalated, not patched here.

---

## 4. T4 — recommendation: **KEEP, and do NOT delete**

The brief permits deletion only if deletion is the recommendation **and** it is a pure
deletion of unreachable code with a test proving unreachability. Neither holds:

1. **It is not unreachable.** `stack/tests/test_refa_v1.py:413` and
   `stack/tests/test_refa_v1_speed_channel.py:213` both call
   `plan(..., target_speed=10.0)` and assert on the result. Deleting T4 breaks two test
   files I do not own. "Dead" here means *no production caller passes it*, which is a
   different and weaker claim than "unreachable".
2. **Making it live is a design change, not a bug fix.** The tactical `lon` token already
   implies a target speed (`canonical_controls`, `GOAL_LON_DV_MPS`, `refa_v1.py:126-128`),
   so wiring it would give the tactical brain's **longitudinal** decision a second channel
   into the cost besides T1 — exactly the lever `D-REFAV1-GOAL-DEGENERATE` says is
   missing. That is a PI/Master-Mind call about what the cost *is*, and it would change
   every banked number.

⇒ **Deliverable instead of a deletion**: a test that pins the deadness claim itself
(an `ast` walk over the production modules asserting no `.plan(...)` call passes
`target_speed`), so the claim cannot rot into a false "four live terms" the way the
`36-features` count did. **ESCALATED** for a decision: wire T4 to the decoded `lon`
token, or delete it together with its two tests.

---

## 5. Pre-registered predictions (both outcomes committed in advance)

| # | prediction | how it is decided |
|---|---|---|
| P1 | Default OFF is **byte-identical**: `plan()` default vs `model_action_units="kappa"` vs `cost_time_grid="dense"` give `torch.equal` controls and identical cost on a real tiny-model rollout; the goal field under the new default equals the goal field from the **legacy expression re-implemented in the test**. | `test_steer_conversion_complete.py` A1/A2/B1 |
| P2 | With `"steer"`, **both** crossings convert: the actions recorded at the goal site equal `arctan(2.9·κ)` of the κ-run's, and so do the candidate's. Two distinct crossing sites are observed in one `plan()` call. | A3 (spy on `augment_actions`) |
| P3 | With **consistent** units, the candidate that *built* the goal reproduces the goal **exactly** (`torch.equal`), for `units="kappa"` and `units="steer"` alike. Under the shipped half-conversion it does not. | A4 |
| P4 | The turn's advantage over the straight line **does not get worse**: `adv(steer, steer) ≥ adv(steer, kappa[trap])` and `adv ≥ −1 ULP` under both consistent spellings. Direction only — no magnitude is claimed, and `D-REFAV1-BOUNDARY-NULL` says the magnitude is zero. | A5 |
| P5 | `cost_time_grid="tactical"` selects `[0,3,6,9,9,9,9,9,9,9]`; it is a **no-op** for a constant candidate and for the operative re-score; and it is `[:, ::3]` on its overlapping prefix — the goal's own map. | B2/B3/B4 |
| P6 | T4 is passed by **no** production `.plan()` call site (ast-verified) and contributes exactly `0.0` when absent. | B5 |
| ⛔ P7 | **Nothing here improves a driving metric.** If any arm of this work is later reported as a planner improvement without a fresh paired episode-cluster interval, that report is wrong. | stated, not tested |

## 6. Scope — files touched

| file | ownership | change |
|---|---|---|
| `stack/tanitad/refs/refa_v1.py` | owned | `_model_actions` helper; `_imagine_tactical_goal(units=)`; `imagined_goal(model_action_units=)`; `plan(cost_time_grid=)`; `_cost_chunk` regrid |
| `stack/tanitad/models/kinematic.py` | owned (only if a helper is missing) | **NOT TOUCHED** — `as_command` / `steer_of_kappa` / `_check_units` already exist and are exact |
| `stack/tests/test_steer_conversion_complete.py` | new, owned | the tests above |
| this package | new, owned | SPEC / RESULT / PROPOSED_REGISTER_ROWS |
| `taniteval/tools/{cost_surface_probe,refav1_arm}.py`, `stack/tanitad/refs/refa_v1_plan.py` | **read-only** | not touched |

---

## 7. DEVIATIONS — appended after the work, so the pre-registration above stays intact

Everything from §0 to §6 is as written before the first edit. Two things the plan did not
anticipate, both recorded in `RESULT.md`:

1. ⛔ **One file outside §6 had to be edited:**
   `stack/tests/test_steer_curvature_interface.py::test_D2_the_flag_actually_REACHES_the_model`.
   It asserted that `cv` / `hold_v0` do **not** move between the two spellings *"because
   arctan(0) == 0"* — an invariant that held **only because the conversion was half applied**.
   Once the goal converts, the goal *field* moves and every candidate is scored against a
   different reference. The assertion was **relocated**, not deleted: it is now made with a
   SUPPLIED `goal_field`, which no conversion touches. Flagged in `RESULT.md` §6 and in the
   proposed register row, because retiring a test that pins a defect is a decision.
2. **P3's tolerance was wrong as pre-registered.** It said the self-comparison would be exact;
   `1 − cos(x, x)` measured **2.00 ULPs**, matching the cost-surface probe's own C3 control
   (2.00 / 4.00 ULPs on the banked checkpoints). The identity is asserted with `torch.equal`
   on the FIELDS (which is exact) and the cost comparisons carry a 4-ULP budget
   (`FLOOR_ULPS`). See `RESULT.md` §3 — this became a finding rather than a nuisance.
