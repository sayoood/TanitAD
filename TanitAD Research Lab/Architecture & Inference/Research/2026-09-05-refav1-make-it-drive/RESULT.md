# REF-A v1 — MAKE IT DRIVE

**STATUS: Step 1 COMPLETE for the DIAGNOSIS and the LEVER. Plan-level A/B RUNNING.**
**Agent:** TanitAD Architecture & Inference FlyWheel · **Date:** 2026-09-05
**Branch:** `agent/arch-inf-20260803`
**Register rows:** `D-REFAV1-DRIVE-GATE`, `D-REFAV1-DRIVE-GATE-REAL`,
`D-REFAV1-DRIVE-GATE2`, `D-REFAV1-DRIVE-WEIGHTS-INERT`, `D-REFAV1-DRIVE-LEVER`

---

## The one-paragraph answer

refav1 does not fail to drive for a subtle reason. **It cannot turn**, and there are
**two structural gates in series** that stop it. Gate 1: the goal head decodes `LANE_KEEP`
on 86.5 % of windows, and a `LANE_KEEP` decode makes a turn *unreachable by the entire iCEM
population* — MEASURED as planned curvature **exactly 0.0 on 244/244** windows of the
trained checkpoint. Gate 2: under `--cost-metric cos`, **the arm's own default**, a
*correctly* decoded turn is **proposed and then refused** — planned curvature **exactly 0.0
on 38/38** windows where the head said TURN, while `ccos` executes 63 % of them. Under the
shipped default, refav1 cannot turn **under any decision rule**. Every cost weight examined
is inert. The lever that opens gate 1 is now implemented, tested and wired
(`--lat-logit-bias`), and gate 2 is a one-flag change that needs a PI ruling.

⇒ **The PI's question "must we retrain?" has a cheap true answer: NO, not for this.** Both
gates are inference-time. Nothing here required touching a weight.

---

## What was MEASURED (instruments and raw in this package)

| # | finding | number | where |
|---|---|---|---|
| 1 | `LANE_KEEP` ⇒ zero curvature, on the trained ckpt | **244/244**, `kappa_diff_max` **0.0** | `GATE_ON_REAL_MODEL.md` |
| 2 | ⭐ **`cos` refuses a correctly decoded turn** | **0/38** executed (`ccos`: 63 %, `ccos_naive`: 92 %) | `TWO_GATES.md` |
| 3 | `cos` finds no turn at **300×30**, the production search size | **0/12** at 4 search sizes | `raw/metric_gate.json` |
| 4 | W_VEND is structurally inert (`target_speed` never passed) | bit-identical over **1e-6…1e12** | `GATE_ON_REAL_MODEL.md` §4 |
| 5 | the Stage-B cost **is the goal term alone** | bit-identical to an all-zero triple | same |
| 6 | the decision-rule lever, parity-pinned | zero bias ≡ no flag, **bit-identical** on the real ckpt | `raw/lat_bias_parity_realckpt.json` |
| 7 | the banked sweep's `tau≥1.0` collapse is an **eps-smoothing artifact** | 282/282 and 1974/1974 windows decode to never-labelled classes | `raw/bias_ladder_*.json` |

Every one carries a same-breath control that had to read the opposite value — the
`ha0_ext` curvature column (98.9 % non-zero, absmax 0.603) proving the zeros are
measurements and not read errors; `ccos` proving `cos`'s zeros are about the metric;
a shifted-token comparison proving the 244/244 is not both-sides-zero.

## The gates compound

```
GT turns on 46.8 % of windows
  x 20.5 %  the head decodes a turn          [GATE 1 — a decision rule]
  x 63.2 %  the planner executes it (ccos)   [GATE 2 — a flag; 0 % under cos]
  x 77.8 %  direction correct
  = ~10 % of required turns executed under ccos, ~0 % under the shipped default
```

## The lever, delivered

`lat_logit_bias` — an additive 8-vector on the lateral logits at `refa_v1.py:1933`, the one
site that decides gate 1. It parameterises the whole rule family: prior correction is
`−τ·log π`, a commit threshold is one negative entry, **plain argmax is the zero vector**.

* `stack/tanitad/refs/refa_v1.py` — the parameter, threaded through `plan()` and
  `imagined_goal()`, stamped on the result as `res.lat_logit_bias`.
* `stack/tests/test_refa_v1_lat_bias.py` — **8/8**; parity pinned bit-for-bit, gate 2 pinned
  by `test_d_*` so it cannot rot. Suite: **278 passed, 1 skipped**.
* `taniteval/tools/refav1_arm.py` — `--lat-logit-bias`, with a stale-stack verify-gate, a
  per-run "did it reach `plan()`" check, and `manifest["goal_rule"]`.

### The ladder (zero GPU — the decode is deterministic in the logits)

| setting | curv % | turn recall | false-turn on straight | direction |
|---|---|---|---|---|
| **argmax (CONTROL)** | 0.135 | **0.205** | 0.073 | 0.778 |
| `commit b=1.0` | 0.181 | 0.265 | 0.107 | 0.743 |
| **`prior τ=0.5` (masked)** | 0.199 | **0.303** | 0.107 | 0.675 |
| `prior τ=0.75` (masked) | 0.358 | 0.500 | 0.233 | 0.652 |
| `commit b=1.5` | 0.617 | 0.795 | **0.460** ⚠️ floods | 0.590 |

Control reproduces the banked decode exactly (`{LANE_KEEP 244, TURN_L 7, TURN_R 31}`,
0.2045 / 0.7778). Stable on the 7× wider stride-5 panel (n = 1974).
⚠️ `correct_turn_rate` is **not** admissible as the objective: the uniform-random control
scores **0.280** on it, above every real setting, because a rule that turns constantly wins
recall × direction. Read recall beside the false-turn rate, always.

## RUNNING: the plan-level A/B (dev-box 4060, freed at ~16:0x)

`argmax` (paired control) vs `prior τ=0.5`, 30 episodes / 60 windows, stride 40,
`ccos` + Stage-B weights, labels **on** so the tactical family is available.
Dumps persist ⇒ a killed run is recovered with `--analyze-only`, never re-paid.
Driver: `tools/run_ab.sh`. Outputs: `C:/Users/Admin/refav1_drive/ab/`.

⛔ **No ADE or four-family number is claimed until it lands.** The lookup shortcut that would
have made this free does **not** hold on turn windows (`GATE_ON_REAL_MODEL.md` §3), and that
was checked rather than assumed.

## ESCALATED to the PI / Master Mind

1. **`ccos` is a prerequisite for lateral control** — but ⛔ **this does NOT reopen
   `D-REFAV1-CCOS-ARMS`'s "`ccos` does not become the default"**, and no ADE gain from
   `ccos` is claimed. The two compose: `D-REFAV1-SURFACE-PAIRED` (2) already measured
   `cos − ha0` at **exactly 0.0000, zero-width, on all three lateral metrics** — this
   package supplies the **mechanism** for that identity. `ccos` is **necessary but not
   sufficient**: it is the difference between *cannot steer* and *can*, and on today's goal
   head the turns it enables are taken on the wrong windows, which is why every family
   degraded. ⇒ the binding constraint is the **goal head**, exactly where
   `D-REFAV1-SURFACE-SCREEN` (2) pointed. What needs re-reading is the flag's *description*
   ("nothing selects it" now means "no refav1 arm can steer"), not the default.
2. **The `ccos` hold-branch is load-bearing.** On a `LANE_KEEP` goal `‖g − z_ref‖` is float32
   rounding and the centred direction is noise — and gate 1 puts **86.5 %** of the grid in
   exactly that state. Already parked as a pre-registration item; it should be prioritised.
3. **The 59-setting cost-weight surface sweep needs no rerun** — two weights were zero, the
   third structurally inert, and its metric could not execute a turn regardless.

## Next, in order

1. Read the A/B when it lands → the four families + ADE vs `ha` / `ha0_ext`.
2. Read Stage B's **`cl_oraclegoal`** arm (Thor, ~44/71 at hand-off) — it **bounds the
   prize**: if a perfect goal clears the controls, gate 1 is worth fixing properly; if not,
   the ceiling is elsewhere.
3. Only if 1–2 are insufficient: Step 2, the goal-head fine-tune (class-balanced loss, trunk
   and WM frozen, freeze asserted by parameter count).

⚠️ `H-ESTIM-SEED-1`: a separated CI from a one-seed arm is necessary, not sufficient. The
A/B is paired on identical windows and differs in exactly one flag, but a replicate arm is
required before any "the lever moved it" claim is quotable.

---

## Deliverable manifest

| artifact | location | status |
|---|---|---|
| `GATE_MECHANISM.md` | this package | banked `b90ed1f` |
| `GATE_ON_REAL_MODEL.md` | this package | banked `ae34ee5` |
| `TWO_GATES.md` | this package | banked `6753674` |
| `RESULT.md` (this file) | this package | banked |
| `tools/assert_gate.py`, `assert_plan_gate.py`, `assert_metric_gate.py` | this package | banked |
| `tools/validate_shortcut.py`, `turn_execution.py`, `bias_ladder.py`, `run_ab.sh` | this package | banked |
| `raw/*.json` (11 files) | this package | banked |
| **`stack/tanitad/refs/refa_v1.py`** (the lever) | repo | banked `6753674` |
| **`stack/tests/test_refa_v1_lat_bias.py`** | repo | banked `6753674` |
| **`taniteval/tools/refav1_arm.py`** (`--lat-logit-bias`) | repo | banked `87a56c4` |
| A/B dumps + records | `C:/Users/Admin/refav1_drive/ab/` (dev box, **off-repo**) | running |

⚠️ The A/B outputs are on the dev box only. When they land they must be banked into
`raw/` — dumps are large, so bank the **records** and the analysis, and note the dump path.
