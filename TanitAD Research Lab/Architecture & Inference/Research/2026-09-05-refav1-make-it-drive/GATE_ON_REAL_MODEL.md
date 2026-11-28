# The gate holds EXACTLY on the trained checkpoint — and two more weights are inert

**Rows:** `D-REFAV1-DRIVE-GATE-REAL`, `D-REFAV1-DRIVE-WEIGHTS-INERT`
**Class:** MEASURED · **Date:** 2026-09-05 · **Agent:** Architecture & Inference FlyWheel
**Instruments:** `tools/assert_plan_gate.py`, `tools/validate_shortcut.py`
**Raw:** `raw/plan_gate.json`, `raw/shortcut_dump_{cos_ext,ccos_comp,ccos_naive}.json`

---

## 1. The gate, measured end-to-end through the real `plan()`

`assert_plan_gate.py` drives the **actual** `RefAV1.plan()` — the function
`refav1_arm.py` calls — on a tiny real `RefAV1`, forcing the decoded lateral token as the
one independent variable. **OVERALL: PASS.**

| forced token | plan's sustained curvature | `res.source` |
|---|---|---|
| **LANE_KEEP** | **+0.000000** (absmax 0.000000) | `baseline:hold_v0` |
| **ABORT_LC** | **+0.000000** (absmax 0.000000) | `baseline:hold_v0` |
| LANE_CHANGE_L / _R | +0.008750 / −0.008750 | `cem` |
| NUDGE_L / _R | 0 mean, absmax 0.010 (S-curve) | `cem` |
| **TURN_L** | **+0.080000** | `cem` |
| **TURN_R** | **−0.080000** | `cem` |

Controls: the forcing reached `res.goal_action` for all 8 tokens; TURN_L/TURN_R are the
same-breath positive controls proving the planner *can* steer and in the correct sign.

## 2. ⭐ And it holds EXACTLY on the trained 21,109-step checkpoint

`validate_shortcut.py` compares the **banked `cl_controls`** (what
`refav1_arm.run_dump` actually recorded from `res.controls`) against
`canonical_controls(decoded token)`, **split by channel**.

| dump | windows with a zero-curvature token | **curvature match** | `kappa_diff_max` |
|---|---|---|---|
| `dump_cos_ext` | 244 | **1.0000 (244/244)** | **0.0** |
| `dump_ccos_comp` | 244 | **1.0000 (244/244)** | **0.0** |
| `dump_ccos_naive` | 244 | 0.8811 | 0.139 |

⇒ **When the goal head decodes LANE_KEEP, the trained planner's curvature output is
EXACTLY ZERO — 244 of 244 windows, difference 0.0, in two of the three banked
configurations.** The gate is not a tendency; it is an identity on this checkpoint.

⇒ On the **86.5 %** of the eval grid whose decoded goal is LANE_KEEP, **refav1's lateral
output is bit-identical to a constant-velocity baseline.** That is the whole reason it does
not drive: for most of the road it *is* `cv`.

**Control:** comparing against a **shifted** token matched only 0.35–8.5 % of windows
(`diff_p50` 0.025–0.056), so the comparison discriminates and the 1.0000 is not an
artefact of both sides being zero.

## 3. ⛔ The shortcut is NOT general — stated plainly, because I nearly relied on it

The tiny-model result invited a much larger claim: *"the planner is a lookup on the token,
so ADE for any decision rule is computable on CPU without iCEM."* **That is FALSE on the
trained model and the measurement says so.**

| dump | curvature match on **turn** tokens (n=38) |
|---|---|
| `dump_cos_ext` | **0.000** |
| `dump_ccos_comp` | 0.632 |
| `dump_ccos_naive` | 0.921 |

When the token carries curvature the search is **live** and lands somewhere other than the
canonical profile. The longitudinal channel is live throughout (accel match 0.60–0.75;
`accel_diff_max` 1.5 = the `decel_1.5` baseline winning the column).

⇒ **The gate is exact in one direction only:** a zero-curvature token *forces* a
zero-curvature plan; a curvature-carrying token *permits* a turn the planner then
searches. ⇒ **ADE under a new decision rule cannot be computed without re-planning**, and
no such number is claimed here.

⚠️ This also **retracts the top-k-seeding fix proposed in `GATE_MECHANISM.md` §"the fix"**.
With the Stage-B cost (§4) the goal term is the *entire* cost, so a TURN seed injected
while the goal field is still LANE_KEEP would be scored against a LANE_KEEP goal — which
under `ccos` on a hold goal is documented float32 rounding noise (`refa_v1.py:270-278`).
It would win or lose at random. **The decision rule is the correct lever precisely because
it moves the goal field and the seed together.**

## 4. Two more inert weights: the Stage-B cost is the GOAL TERM ALONE

Stage B runs `--cost-weights 0.0,0.0,64.29715042415070` (W_JERK = 0, W_KAPPA = 0), and
`refav1_arm.py` never passes `target_speed`, while `_cost_chunk` guards the W_VEND term on
`if target_speed is not None`. Measured rather than read:

| check | result |
|---|---|
| W_VEND swept **1e-6 … 1e12**, `target_speed=None` | controls **bit-identical** at all 5 settings ⇒ **INERT** |
| CONTROL: same sweep with `target_speed=0.0` | `max_abs_diff` **1.5** ⇒ the probe can see a difference |
| Stage-B triple vs **all-zero** triple | **bit-identical**, `max_abs_diff` **0.0** |
| CONTROL: `W_KAPPA=1e3` | `max_abs_diff` **0.080** ⇒ a live weight does change the plan |

⇒ **Under the shipped arm all three cost weights are inert and the cost is the goal term
alone.** W_JERK's inertness was already MEASURED (`D-REFAV1-SURFACE-PAIRED`, 24/26
candidates); this adds **W_KAPPA** (set to 0) and **W_VEND** (structurally inert), and
identifies the reason the 59-setting cost-weight surface sweep could not move anything:
**there was no live weight in it to move.**

---

## What this settles, and what it does not

**Settles.** refav1's lateral policy *is* the goal head's 8-way argmax. Every downstream
knob examined — three cost weights, 59 weight settings, the choice of cost metric — is
either inert or cannot reach behaviour the token did not propose. The single lever is the
token.

**Does not settle.** Whether moving the token *improves driving*. That needs re-planning
and the four families, and it is the next measurement — not an inference from this one.

**Evidence class.** MEASURED throughout, on the real `plan()` and on the trained
checkpoint's banked dumps; every "no change" assertion carries a same-breath control that
had to show a change, and every "exactly zero" carries one that had to be non-zero.
