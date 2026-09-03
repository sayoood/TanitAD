# RESULT — the κ→steer crossing is complete; the cost's time grid has a corrected mode

**Architecture & Inference FlyWheel · 2026-09-03 · BACKLOG R26 + R27**
Branch `agent/arch-inf-20260803`. SPEC (written before any edit): `SPEC.md`.
Raw: `raw/evidence.py` → `raw/evidence.json`.

> ⛔ **ESCALATION — needs a Master Mind / PI decision, not a merge:**
> **(1)** `cost_time_grid="tactical"` is implemented and tested but **default-OFF**; turning it
> on changes what every future banked number means, so the flip is a decision, not a bug fix.
> **(2)** **T4 (`target_speed`) was NOT deleted** — it is unused by production but *not*
> unreachable (two test files exercise it), and wiring it to the decoded `lon` token is a
> change to what the cost *is*. Pick: wire it, or delete it together with its two tests.
> **(3)** The **plan horizon (2.0 s) is shorter than the goal horizon (6.0 s)** and no unit or
> grid repair removes that; it is a `plan_horizon_s` design question.
> **(4)** One PRE-EXISTING test outside this agent's declared ownership had to be amended —
> `test_steer_curvature_interface.py::test_D2` asserted an invariant that only held *because*
> the conversion was half applied (§6). The amendment is minimal and relocates the invariant
> rather than deleting it, but retiring a test that pins a defect is a decision, not a cleanup.

> ⚠️ **WHAT THIS IS WORTH.** `D-REFAV1-BOUNDARY-NULL` measured that completing the unit
> conversion moves the cost's minimum on **0 of 140** windows on **both** banked checkpoints
> — `share = +0.0000 [0.0000, 0.0000]`, a zero-width paired episode-cluster interval, on every
> stratum including the 25 `TURN_R` windows. **This repair buys the planner nothing
> measurable.** It removes a trap. Nothing below may be quoted as a planner improvement.

---

## 1. What was wrong, confirmed at source before editing

| | claim | verified how |
|---|---|---|
| R26 | `as_command` was called at **exactly one functional site**, `refa_v1.py:1832` inside `_cost_chunk` | `git show HEAD:stack/tanitad/refs/refa_v1.py \| grep -n as_command` → `:61` (import), `:1831` (comment), **`:1832`** (the only call) |
| R26 | the goal crossed into the model **unconverted** | HEAD's four `self.augment_actions(` sites: `:1264` (`forward`, TRAINING), `:1615` (`_cf_term`, TRAINING), **`:1683` (`_imagine_tactical_goal` — the defect)**, `:1838` (candidate, converted at `:1832`) |
| R27 | the 2.0 s plan is consumed on the 0.6 s clock | `H = plan_steps = round(2.0/0.2) = 10` (`refa_v1.py:247, 548`); `tac_dt = 0.6` (`:191`); `_cost_chunk` rolls `search_pred = self.tactical` (`:1791`) on all 10 entries (`:1840`) ⇒ 6.0 s |
| R27 | the goal — and the **training** path — use `[::stride]` | `refa_v1.py:1270-1272` (`forward`'s `tac_a`), `:1683` (goal). ⇒ the dense feed is also **out of distribution**, not merely out of clock |
| R27b | T4 is dead in production | ast walk over the two production callers: `taniteval/tools/refav1_arm.py:735` and `taniteval/tools/cost_surface_probe.py:515` pass no `target_speed` ⇒ **three** live cost terms |

**Preflight (no pod contacted).** Two independent probes agree that **no trainer** reaches
`plan()`, `_imagine_tactical_goal` or `imagined_goal`: (1) `grep -rn "\.plan("` over
`stack taniteval colab` hits only tests, the two taniteval tools, and
`stack/experiments/alpasim-gsplat/{closedloop,openloop}_drive.py` — a *different*
`policy.plan(list(frames), intr, v, nav, …)` signature; (2) PowerShell `Select-String` over
`stack/scripts/**` + `colab/**` returns **zero** hits. The live refav1 (Thor) and refcv3 (A40)
runs reach only `forward()` / `_cf_term`, and **neither is touched** by this change.

## 2. The change — one named boundary, two flags, both defaulting to the legacy path

The crossing-point table is in `SPEC.md` §1 (14 rows). Its verdict: **exactly one row was
wrong** — row 2, the goal's crossing. Rows 12–14 (`forward`, `_cf_term`, the strategic slice)
are TRAINING and already carry COMMAND units, which is why the conversion **cannot** live
inside `augment_actions`.

**Chosen design: the conversion moved one level up into a named boundary
(`RefAV1._model_actions`), and both planner-side crossings now go through it.** Justification
(as asked, two sentences): the conversion is a property of the *boundary*, not of either path,
and the defect was exactly that one path carried it and the other did not — so a repair that
merely adds a second `as_command` call leaves the same shape of bug available to the third
crossing somebody writes next. Naming the boundary makes "did this crossing convert?" a
one-line grep, and it is why `_model_actions`'s docstring, not a changelog, carries the
measurement that earned it.

| after | site | call |
|---|---|---|
| TRAINING | `refa_v1.py:1325` (`forward`), `:1676` (`_cf_term`) | `self.augment_actions(...)` — direct, **unconverted**, unchanged |
| PLANNER | `refa_v1.py:1761` (goal), `:1963` (candidate) | `self._model_actions(...)` — the only place `as_command` is applied |

New API, both call-site arguments (not config fields — a config field would change every
serialised config dict while a training run is live):

* `plan(..., model_action_units="kappa")` — now **forwarded to the goal**;
  `imagined_goal(..., model_action_units="kappa")` for the standalone path.
* `plan(..., cost_time_grid="dense")` — `"dense"` is today's behaviour; `"tactical"` puts the
  candidate on the goal's (and the training path's) grid. `COST_TIME_GRIDS` and
  `_check_cost_time_grid` refuse anything else.
* Provenance travels on the result: `res.model_action_units`, `res.cost_time_grid`.

## 3. MEASURED — the trap, and its repair, reproduced in miniature

Tiny random `RefAV1` (seed 0, CPU), tactical decode **pinned to `TURN_R` / `CRUISE`** so the
imagined goal carries curvature (`|κ| = 0.0800`, i.e. `GOAL_KAPPA_TURN`). The candidate is the
canonical profile that *built* the goal; the reference is the straight line (zero controls).
Goal term only (`1 − cosine_similarity`, float32). `raw/evidence.json` → `P1`.

| arm | `c_goal(turn)` | in ULPs | `c_goal(straight)` | **advantage of the turn** |
|---|---|---|---|---|
| **OFF** — κ on both sides (today's default) | 1.1921e-07 | 2.00 | 1.1921e-07 | **0.0** |
| **ON as shipped** — candidate steer, goal κ ⛔ | 3.5763e-07 | 6.00 | 1.1921e-07 | **−2.3842e-07** |
| **ON after the repair** — steer on both sides | 5.9605e-08 | 1.00 | 5.9605e-07 | **+5.3644e-07** |

Same shape as the register's read of the real checkpoints (advantage 0 → **−5.96e-08** under
the half-conversion, on 25 real `TURN_R` windows): the half-conversion makes the turn score
**worse than doing nothing**, and completing it restores the turn to the goal term's floor.
⚠️ **Every number here is a ULP multiple on a random tiny model.** It demonstrates the
mechanism; it measures nothing about driving, and the real effect is `share = 0.0000 [0, 0]`.

⭐ **A float32 fact worth pinning, and it cost a test iteration:** `1 − cos(x, x)` **is not
0.0**. The self-similarity of a real terminal field reads **2.00 ULPs** here, and the
cost-surface probe's control C3 failed its pre-registered *exactly-0.0* form at **2.00 / 4.00
ULPs** on the two banked checkpoints. Any assertion that a self-comparison is "zero" must
carry that budget (`FLOOR_ULPS = 4` in the test file) or it is asserting float32 does not exist.

## 4. The time grid — what the corrected horizon does

Derived from the **banked checkpoint's own** `config.json`
(`C:\Users\Admin\refav1_eval_slice\ckpt_ep2\config.json`), not from source defaults:
`op_dt 0.2`, `op_steps 30`, `tac_dt 0.6`, `tac_steps 10`, `a_dim 2`, **`speed_channel false`**,
`plan_horizon_s 2.0`, `plan_level "tactical"`, `verify_on_operative true`.

| | index map (operative step per tactical step) | imagined horizon |
|---|---|---|
| candidate, **`"dense"`** (today) | `[0,1,2,3,4,5,6,7,8,9]` | 6.0 s |
| candidate, **`"tactical"`** (repair) | `[0,3,6,9,9,9,9,9,9,9]` | 6.0 s |
| **goal** (`_imagine_tactical_goal`, and `forward`'s `tac_a`) | `[0,3,6,9,12,15,18,21,24,27]` | 6.0 s |

The repair puts the candidate on the goal's grid *while the plan lasts* (`j ≤ 3`) and then
holds the plan's final action — the standard receding-horizon continuation, and the only
defined way to span the 6 s the cost is documented to span (`refa_v1.py:247`, *"optimised
window; cost spans 6 s"*). **Truncating to 4 tactical steps (2.4 s) was rejected**: it would
put the candidate's terminal field at 2.4 s against a 6.0 s goal, and `1 − cos` between fields
at different horizons is dominated by *how far ahead* they are rather than *where they go*.

⛔ **The residual the repair does NOT remove.** `cand_fixed[4:] ≠ goal_map[4:]`: the plan
specifies 2.0 s of a 6.0 s goal, so the injected seed can never reproduce its own goal. That is
a `plan_horizon_s` question and is **escalated**, not patched (pinned by
`test_B2_the_corrected_index_map_is_the_GOALs_map_then_a_hold`).

### What it does to the cost-surface panel's headline numbers

**Cheaply derivable, model-free, from the banked dumps — and the answer is exact.**
`raw/evidence.json` → `P3`, over all 20 episodes / 140 windows / 3 planning arms of
`C:\Users\Admin\refav1_eval_slice\t1_dump\decisions\`:

| arm | windows | winner all-zero | **winner CONSTANT over H** | winning cost exactly 0.0 | cost range |
|---|---|---|---|---|---|
| `cl` | 140 | **140** | **140 (100 %)** | 88 | −2.384e-07 … +1.788e-07 |
| `cl_navshuf` | 140 | 122 | **140 (100 %)** | 92 | −2.384e-07 … +2.384e-07 |
| `cl_oraclegoal` | 140 | 96 | **140 (100 %)** | 0 | 2.503e-05 … 2.165e-02 |

With `speed_channel = false` in the banked config, `_model_actions` is the identity on the
controls, so the two index maps select **the same values** for any candidate that is constant
over the plan horizon. **Every banked winner is constant on 140/140 windows in all three arms**
⇒ **the banked winner's own cost is bit-identical under `cost_time_grid="tactical"`.**
Stronger, on the windows `D-REFAV1-GOAL-DEGENERATE` identified (goal = the `cv` rollout, so
`cv` scores the floor of a non-negative cost): the zero-control candidate's rollout is
*literally the same tensor* under both grids, so the argmin **cannot** move there either.

⚠️ **UNVERIFIED:** whether the corrected grid changes the *argmin* on the remaining windows,
and what it does to the panel's surface statistics (goal-term span in ULPs, `turn_frac`,
`|κ*|`). Those are non-constant candidates over the whole `(a, κ)` box, and re-deriving them
needs the model — `taniteval/tools/cost_surface_probe.py` is **read-only for this agent and
has no `cost_time_grid` argument**, so the panel was not re-run. The dumps contain
trajectories, controls, costs and labels only — **no features and no fields** — so no re-scoring
is possible from them alone. A re-run is one dev-box GPU job on the two local checkpoints
(`ckpt/`, `ckpt_ep2/`) once the probe is allowed to take the flag.

## 5. T4 (`target_speed`) — recommendation: **KEEP; deletion REJECTED**

Deletion was permitted only as a pure removal of unreachable code with a test proving
unreachability. **Neither holds.**

1. **Not unreachable.** `stack/tests/test_refa_v1.py:413` and
   `stack/tests/test_refa_v1_speed_channel.py:213` both call `plan(..., target_speed=10.0)`.
   "Dead" here means *no production caller passes it* — weaker than *unreachable*, and
   deleting it would break two test files this agent does not own.
2. **Making it live is a design change.** The tactical `lon` token already implies a target
   speed (`GOAL_LON_DV_MPS`, `refa_v1.py:126-128`), so wiring it would give the tactical
   brain's **longitudinal** decision a second channel into the cost besides T1 — the lever
   `D-REFAV1-GOAL-DEGENERATE` says is missing. That changes what the cost *is* and every
   banked number with it.

Shipped instead: the deadness claim is now **pinned by an ast walk**, not a grep (the naive
grep false-positives on `cost_surface_probe.py`, which contains the identifier in its own
re-implementation), and the pin asserts a `.plan(...)` call was **found**, so a rename cannot
make it vacuously true. Plus a behavioural pin: on the `cv` baseline `target_speed=v0` changes
the cost by exactly 0.0 and `target_speed=v0−1` adds exactly `0.10·1.0² = 0.1`.

## 6. Tests — baseline vs after

Selection `-k "refa or refav1 or kinematic or cost_surface or steer"`, run in the off-Drive
mirror `C:\Users\Admin\tanitad-wt` (venv `C:\Users\Admin\venvs\tanitad`).

| | passed | failed | skipped | deselected | items |
|---|---|---|---|---|---|
| **BASELINE** (HEAD, before any edit) | 389 | **1** | 1 | 5360 | 391 |
| **AFTER** | 404 | **1** | 0 | 5360 | 405 |

**405 − 391 = +14, exactly the 14 new tests in
`stack/tests/test_steer_conversion_complete.py`.** No test regressed.

⚠️ **The one failure is PRE-EXISTING and is a mirror artifact, not a regression:**
`stack/tests/test_decision_check.py::test_refa_surfaces_the_measured_arm` — it fails in the
mirror because `Project Steering/` is not mirrored, and the test's own stdout says so
(`DECISIONS_2026-07-20.md NOT FOUND`, `MODEL_REGISTRY.md NOT FOUND`, `RETRACTION_LOG.md NOT
FOUND`). Identical before and after.

⚠️ **The skipped→passed flip is ENVIRONMENTAL, not the change:**
`test_refa_v1_precision.py::test_cuda_bf16_autocast_tiny_step_has_finite_loss_and_grad_norm`
self-skips when the dev-box GPU is busy (`_cuda_busy_reason`, `test_refa_v1_precision.py:315-330`).
It skipped during the baseline run and again on re-check (*"GPU busy — util 59 %, 822/8188 MiB"*),
and happened to run — and pass — during the after run.

⛔ **ONE PRE-EXISTING TEST WAS AMENDED, and it is OUTSIDE this agent's declared file
ownership: `stack/tests/test_steer_curvature_interface.py::test_D2_the_flag_actually_REACHES_the_model`.**
It asserted that the zero-channel-1 baselines (`cv`, `hold_v0`) do **not** move between
`"kappa"` and `"steer"`, *"because arctan(0) == 0"*. **That assertion was only true while the
conversion was half applied**: with the goal now converted too, the goal *field* moves, so
every candidate — zero-curvature ones included — is scored against a different reference. The
amendment (a) keeps "the flag reaches the model", (b) records *why* the old invariant died,
and (c) **relocates the original invariant to the configuration where it still holds** — a
SUPPLIED `goal_field`, which no conversion touches, where `cv`/`hold_v0` must still be
identical and a curved baseline must still move. Both halves pass. **Flagged rather than
silently rewritten: a test that pins a defect is evidence, and retiring one is a decision.**

**Wider check (outside the `-k` filter):** the 7 other test files that import `refa_v1` —
`test_actdiv_anchored`, `test_hierarchy_rung`, `test_metric_dynamics_bptt`,
`test_refc_v3_scale_matrix`, `test_refd`, `test_refs_vocab_v7`, `test_residual_init_scale` —
**107 passed**, 0 failed.

New tests (all pass): `A1` OFF byte-identical on a real rollout · `A2` the default goal equals
the **legacy source expression** verbatim · `A3` ON converts **both** crossings in one `plan()`
call (spy on `augment_actions`) · `A4` a candidate in the same units reproduces its own goal
**exactly**, and the cross-units pairing does not · `A5` the turn's advantage does not get worse
(direction only) · `A6` bad units refused on both APIs · `B1` time flag OFF byte-identical and
refused loudly · `B2` the corrected index map is the goal's map then a hold · `B3` ON regrids
the tactical rollout and holds the last action · `B4` three no-op controls that must read known
values (constant candidate; the operative re-score untouched; ⚠️ the speed channel *does* move
and that is correct) · `B5` the two flags compose and the deployed plan stays curvature ·
`C1×2` no production `.plan()` passes `target_speed` (ast) · `C2` T4 is exactly 0 when absent
and hand-computable when supplied.

## 7. Deliverable manifest

| artifact | lives at | only one place? |
|---|---|---|
| `_model_actions`, `_imagine_tactical_goal(units=)`, `imagined_goal(model_action_units=)`, `plan(cost_time_grid=)`, `COST_TIME_GRIDS`, `_check_cost_time_grid`, result provenance | `repo:stack/tanitad/refs/refa_v1.py` (**staged**) | no — mirrored to `C:\Users\Admin\tanitad-wt` for running tests only |
| 14 new tests | `repo:stack/tests/test_steer_conversion_complete.py` (**staged**) | no — same mirror |
| ⛔ amended `test_D2` (outside declared ownership — see §6) | `repo:stack/tests/test_steer_curvature_interface.py` (**staged**) | no — same mirror |
| SPEC (crossing table, pre-registration) | `repo:…/Research/2026-09-03-steer-conversion-completion/SPEC.md` (**staged**) | yes |
| this file | `repo:…/2026-09-03-steer-conversion-completion/RESULT.md` (**staged**) | yes |
| proposed register rows | `repo:…/2026-09-03-steer-conversion-completion/PROPOSED_REGISTER_ROWS.md` (**staged**) | yes |
| evidence script + JSON | `repo:…/2026-09-03-steer-conversion-completion/raw/evidence.{py,json}` (**staged**) | yes |
| banked dumps + checkpoints read | `local:C:\Users\Admin\refav1_eval_slice\` (`t1_dump\`, `ckpt\`, `ckpt_ep2\`) | **⚠️ read-only inputs, NOT produced here, and they live ONLY on the dev box** |

`stack/tanitad/models/kinematic.py` was **not** touched — `as_command`, `steer_of_kappa` and
`_check_units` already exist and are exact; no helper was missing.
