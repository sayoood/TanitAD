# refav1 has TWO gates in series — and the shipped default closes the second one completely

**Rows:** `D-REFAV1-DRIVE-GATE` (decode), **`D-REFAV1-DRIVE-GATE2` (cost metric)**
**Class:** MEASURED, on the trained 21,109-step checkpoint · **Date:** 2026-09-05
**Instruments:** `tools/turn_execution.py`, `tools/assert_metric_gate.py`, `tools/assert_plan_gate.py`
**Raw:** `raw/turnexec_dump_*.json`, `raw/metric_gate.json`, `raw/plan_gate.json`

---

## The headline

> **Under `--cost-metric cos` — `refav1_arm.py`'s DEFAULT — the trained planner emits
> curvature EXACTLY 0.0 on all 38 windows where its goal head CORRECTLY decoded
> TURN_L / TURN_R.** The turn is proposed and rejected. Under `ccos` the same 38 windows
> turn on 63 %.
>
> ⇒ refav1's inability to turn is **not one problem, it is two in series**, and the second
> one is a **flag default**.

## The two gates, measured on the same banked dumps

| | **GATE 1 — the decode** | **GATE 2 — the cost metric** |
|---|---|---|
| what fails | the head decodes `LANE_KEEP` on **86.5 %** of windows | a correctly decoded `TURN` is **refused by the search** |
| mechanism | `LANE_KEEP` → `canonical_controls` writes zero curvature → the population's only curvature-carrying candidate is flat (`refa_v1.py:1933`, `:377-382`, `refa_v1_plan.py:178-186`, `:157`) | under `cos` the goal term realises ~1.2e-05 of its [0,2] range against a jerk charge of 0.094, so the whole population is excluded before the WM is consulted (`refa_v1.py:COST_METRICS`, already documented) |
| measured | planned curvature **exactly 0.0 on 244/244** LANE_KEEP windows | planned curvature **exactly 0.0 on 38/38** decoded-TURN windows under `cos` |
| opened by | a decision rule (`lat_logit_bias`, now implemented) | `--cost-metric ccos` |

### Turn execution, per banked dump (n = 282 windows, 141 episodes, step 21,109)

| dump | metric | decoded-TURN windows | **turn actually executed** | LANE_KEEP windows with curvature |
|---|---|---|---|---|
| `dump_cos_ext` | `cos` | 38 | **0.0 %** (κ absmax **0.0**) | 0.0 % |
| `dump_ccos_comp` | `ccos` | 38 | **63.2 %** (κ p50 **0.080**) | **0.0 %** |
| `dump_ccos_naive` | `ccos` | 38 | **92.1 %** (κ p50 0.080) | 11.9 % ⚠️ |

⛔ **CONTROL (same file, same read):** `ha0_ext`, a **non-planner** arm banked in the same
`decisions/*.npz`, carries non-zero curvature on **98.9 %** of windows, absmax **0.603**.
⇒ the curvature column is **live**, so every `0.0` above is a measurement and not a read
error. This control exists because "0 hits from a file that could not be read" is
indistinguishable from a genuine zero (CLAUDE.md).

⭐ **`dump_ccos_comp` is the clean configuration**: turns execute when decoded, and
LANE_KEEP stays exactly zero. `ccos_naive` turns more but leaks curvature onto 11.9 % of
hold windows — more steering is not automatically better steering.

### The tiny-model replication (search size is not the explanation)

`assert_metric_gate.py`, 12 windows × 4 search sizes, **OVERALL PASS**:

| metric (shipped weights) | 32×3 | 64×4 | 128×6 | **300×30** |
|---|---|---|---|---|
| `cos` | **0/12** | **0/12** | **0/12** | **0/12** |
| `ccos` (control) | 8/12 | 8/12 | 8/12 | 8/12 |

⇒ `cos` finds no turn at **300 samples × 30 iterations** — the production search size. This
is not an under-powered search; it is an excluded one. Gate 1 (`LANE_KEEP` → zero) holds
under **both** metrics at the shipped weights, so the two gates are not confounded.

---

## What the two gates do together

They **compound multiplicatively** on the road:

```
GT turns on 46.8 % of windows
  x  head decodes a turn on 20.5 % of those          [GATE 1]
  x  planner executes it on 63.2 % (ccos) / 0 % (cos) [GATE 2]
  x  direction correct 77.8 %
  = ~10 % of required turns executed correctly under ccos
  = ~ 0 % under cos, the shipped default
```

⇒ **Under the shipped default refav1 cannot turn at all, under ANY decision rule.** No
amount of goal-head work is visible until gate 2 is open. That ordering is the actionable
part of this finding: **gate 2 is a one-flag change; gate 1 is a modelling problem.**

---

## Consequences that need a decision above me (ESCALATED)

1. ⛔ **`refa_v1.py:COST_METRICS` says `ccos` is *"an INSTRUMENTED OPTION, not a candidate
   default; `_check_cost_metric` accepts it and nothing selects it."* That sentence is now
   contradicted by measurement:** `ccos` is a **prerequisite for lateral control**. Making
   it the default is a PI / Master Mind call (the same doc requires any arm flipping it to
   declare its weight triple in the same breath), so it is **not taken here** — but every
   refav1 arm run with `cos` has been measuring a planner that cannot steer.
2. **The `ccos` hold-branch is now load-bearing, not cosmetic.** The same docstring flags
   that on a LANE_KEEP goal `||g − z_ref||` is float32 rounding and the centred direction
   is noise, and parks a hold-branch as a pre-registration item. Gate 1 puts **86.5 %** of
   the grid in exactly that state.
3. **The 59-setting cost-weight surface sweep is fully explained** and needs no rerun: two
   of its weights were zero, the third (`W_VEND`) is structurally inert because
   `refav1_arm.py` never passes `target_speed` (`GATE_ON_REAL_MODEL.md` §4), and the metric
   it ran under could not execute a turn regardless.

---

## The lever, implemented

`lat_logit_bias` — an additive 8-vector on the lateral logits at `refa_v1.py:1933`, the one
site that decides gate 1. It parameterises the entire decision-rule family (prior
correction = `−τ·log prior`; commit threshold = one negative entry; **argmax = the zero
vector**).

* **Parity is pinned, not promised:** `stack/tests/test_refa_v1_lat_bias.py::test_a_*`
  asserts absent-bias ≡ zero-bias bit-for-bit on controls, cost and decoded token, each
  with a same-breath control that must show a difference.
* **Provenance travels:** `res.lat_logit_bias` is stamped beside `cost_metric` /
  `cost_weights`, so two arms differing only in the rule are distinguishable in their dumps.
* **Gate 2 is pinned too:** `test_d_cos_refuses_the_turn_that_ccos_executes` fails loudly if
  the default ever starts executing turns, so this finding cannot rot.
* Suite: **278 passed, 1 skipped** across `refa_v1` / `cost_ccos` / `cost_chord` /
  `ego_plan`; the new file is 8/8.

## What is NOT claimed

No ADE or four-family number for a new decision rule. That needs re-planning
(`GATE_ON_REAL_MODEL.md` §3 shows the lookup shortcut does **not** hold on turn windows),
and the compute for it was held back rather than displacing Thor's live Stage B.
