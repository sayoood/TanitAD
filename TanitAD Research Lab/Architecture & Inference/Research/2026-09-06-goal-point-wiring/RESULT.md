# RESULT — GP-1 / GP-2 wiring, and **R4b: the decoder parameterisation defect**

**Agent:** goal-point wiring + R4b. **Date:** 2026-09-06. **Branch:** `agent/arch-inf-20260803`.
**Tier:** T1 (self-action open loop) for every measured number below.
**Evidence class:** MEASURED (ours), artifacts in `raw/`.
**Compute:** ZERO GPU. Every number here comes off the banked refcv4b T1 dump.
⛔ The A40 was not touched: refcv5 was left training undisturbed.

---

## 0. THE ONE LINE THE BRIEF ASKED FOR

> ⭐⭐ **YES — the shape defect is fixable without giving back the ADE halving, and it is not
> even a trade.** On a **held-out episode split** with the smoothing strength chosen on a
> **disjoint fit half**, a curvature-penalised refinement cuts curvature MAE
> **−0.001903 [−0.003359, −0.000587] SEPARATED** (0.007200 → 0.005330, **−26 %**) while ADE
> moves **−0.000211 [−0.000690, +0.000266], NOT separated** — i.e. unchanged — and speed MAE
> improves **−0.008266 [−0.011285, −0.005525] SEPARATED**. The only cost is
> **+0.000763 m of cross-track**, separated but **0.76 mm**. It needs **no retraining**: it is a
> projection of the offset the decoder already emits.

---

## 1. PART 1 — GP-1 and GP-2 are WIRED

| item | where | state |
|---|---|---|
| **GP-1** the gated `r_terms` entry | `stack/tanitad/refs/refc.py` | LANDED — `graft_gp_point` on `SelectionConfig`/`RefCConfig`, `gp_point_gate` (**+1 parameter**, zero-init), the term in the ranked-score block, `gp_point`/`gp_valid` plumbed from the hierarchy hook and overridable per call |
| **GP-2** the four flags | `stack/scripts/refc_v3_train.py` | LANDED — `--goal-point-inject`, `--goal-point-geo-prior`, `--goal-point-t`, `--goal-point-w`, plus the loss, the head-gate readouts in METRES, the `config.json` stamp and four startup refusals |
| pass-through + refusals | `stack/tanitad/refs/refc_v3.py` | LANDED — `gp_point`/`gp_valid` on `RefCV3Model.forward`, slot/scale derived, two refusals |
| tests | `stack/tests/test_goal_point_selection_seam.py` (23) · `test_goal_point_trainer_flags.py` (22) | 78 passed with the two pre-existing goal-point files |

**No second forward pass is needed**, as PREREG §8 predicted: `refc_v3.py` already emitted
`hook_out["goal_point"]` and `refc.py` now reads it out of the hook it already calls.

### ⛔⛔ 1.1 A UNITS DEFECT IN THE PRE-REGISTERED DIFF, FOUND BY WIRING IT

PREREG §8's draft snippet passes the head's output straight into
`anchor_goal_prior_at_time(gp_point, …)`. **That is wrong, and it fails silently.**

* `GoalPointHead` emits — and `goal_point_loss` / `encode_goal_point` consume — the
  **NORMALISED** point `(x / 40, y / 40)`.
* `anchor_goal_prior_at_time` compares the goal against `bank`, which is in **METRES**;
  `scale_m` there only makes the resulting *score* scale-free.

**MEASURED:** a 24 m goal entered as a 2.5 m one. Shapes matched, the term stayed finite, the
seam trained — and a **MIRRORED goal then moved the ranked score by 7.7e-4 and flipped 0 of 4
picks** on a rig whose anchors span **±66–82 m** laterally. That is **the PREREG §4
value-sensitivity gate failing for a UNITS reason** and reading as *"the geometric prior ignores
the goal"* — the one outcome that must never be produced by accident.

⇒ Fixed by putting the conversion in ONE named place,
`goal_point.anchor_goal_prior_at_time_from_norm`, and stating the units in
`anchor_goal_prior_at_time`'s own docstring. Pinned by
`test_UNITS_the_prior_takes_METRES_and_the_head_emits_NORMALISED` (a KNOWN value, not a shape
check) and `test_UNITS_the_decoder_uses_the_CONVERTING_form` (the model's seam must agree with the
helper to 1e-4, so this file cannot pin a function the model does not call).

⚠️ **Class:** the `anchors.pt` `controls[:, 1]` case verbatim — lateral acceleration read as
curvature gave 396 g vs 0.31 g, both tables plausible, the wrong one produced first from the
shipped file. A true quantity quoted outside its scope. The existing
`test_the_hook_emits_the_point_for_the_selection_seam` asserted **shape and finiteness** and
therefore could not see it.

### 1.2 THE LEAK GUARD — PROVED BY MUTATION, NOT BY ASSERTION

`GoalPointConfig.__post_init__` RAISES on `t_goal_s <= t_pred_s`. It is preserved **exactly as
designed** — a constructor refusal, never a warning
(`test_the_guard_is_a_REFUSAL_and_not_a_WARNING` asserts zero warnings are emitted).

**The mutation (`test_MUTATION_the_leak_guard_is_LOAD_BEARING_not_decorative`):** neuter
`__post_init__`, and a leaking arm becomes not merely constructible but **trainable end to end** —
`GoalPointConfig(t_goal_s=2.0)` builds, `RefCV3Model` accepts it, `m.gp_slot == 3` (**the scored
2 s endpoint**), and a forward emits a goal point. Restore the guard and the same arm is
impossible. ⇒ **the guard is what stops the leak**, which is the half `pytest.raises` alone
cannot show. Six `t_goal_s` values are parameterised over the refusal.

Two further refusals were added because a silent drop manufactures a FAILED gate out of a wiring
gap: `core.graft_gp_point` without `goal_point_inject` refuses at build; a supplied `gp_point`
into a seam-less build refuses at forward.

### 1.3 THE VALUE-SENSITIVITY GATE — RE-PROVEN AFTER WIRING

⭐ **The committed numbers are untouched**: mirrored goal **≤ −0.1212** turn accuracy (2× E13's
measured −0.0606), range-halved goal **≥ +0.05 m/s**, content share **≥ 0.25** (E13's is 0.023).
The two mutation unit tests that reintroduce the E13 collapse still pass (`test_goal_point.py`,
33 passed).

Re-proven **through the wired path**:

* the term's **own argmax follows the goal** — mirroring `y → −y` moves the picked anchor's
  lateral offset in the negative direction (`test_a_MIRRORED_goal_moves_the_S7_RANKING_…`);
* **the final pick follows** once the seam dominates the base confidence
  (`test_a_MIRRORED_goal_moves_the_FINAL_PICK_once_the_seam_dominates`);
* `valid = 0` leaves the ranked score **BIT-IDENTICAL** to the goal-free one — the X15 presence
  control;
* the gate is **bit-inert at init**, **moves the ranked score when opened**, and **receives
  gradient**, so a `gate 0.0000` reading at 40 k is a fact about training and not a dead wire.

⚠️ **SCOPE, STATED.** On a randomly-initialised 20-anchor smoke model the base-confidence spread
is ~47 while the S7 term's is ~2.5, so a mirrored goal at a small gate correctly does **not** flip
the argmax. That is not evidence about the lever; the PREREG §4 bars are measured on the **live
117-anchor fan**, and the banked proof they can fail (mirrored goal **+0.3843 m** / **−0.7842**
turn accuracy) stands unchanged.

### 1.4 One record defect found and closed

`_seam_stamp` read `core.gp_slot`, which until now was set only as a **side effect of building the
model**. A consumer that stamped first recorded **`gp_slot: -1`** — an artifact claiming the goal
was compared against slot −1 while the weights used slot 5. Now derived in `_pin_trainer_cfg`, at
config time. Caught by `test_the_stamp_carries_the_ADMISSIBILITY_DECLARATION`.

⚠️ **One deviation from PREREG §8's draft, stated rather than silent:** the draft passes
`prior_bank`, which is `None` on a fixed vocabulary (it is the legacy bit-exactness path for the
two S6 terms) and would raise there. The seam uses `bank`, which **is** `prior_bank` in the
registered arm (`--anchor-v0-conditioned`) and is functional where the draft would have crashed.

⚠️ **`--goal-point-inject` turns `nav_inject` OFF.** That is PREREG §2 (the ONE variable is the
TYPE of the conditioning signal), not a side effect, and it is stamped as
`goal_point.replaces_nav_inject`.

---

## 2. PART 2 — R4b: THE MECHANISM IS **CONFIRMED**

**Surface:** the banked refcv4b T1 dump, **4,823 windows / 141 episodes**, the 2 s SCORED grid
(4 waypoints at dt = 0.5 s), **13,558 curvature steps**. Estimator: **paired episode-cluster
bootstrap over the 141 episodes, steps RE-POOLED inside each draw** (B = 2000, seed 0),
INTERSECTION mask. Geometry: `taniteval.four_families._seq_geometry`, imported.

⛔ **K-CONTROLS — all PASS to ≤ 5e-5**, so this is the banked surface and not a re-derivation:
`ha0` 0.6723 · `ha0_ext` 0.2874 · `ha` 0.2996 · `os` 0.2965 · anchor 0.4281 ADE; curvature
`os` 0.008150 · anchor 0.004019 · `ha0` 0.006802; heading 1.2964 / 1.7665; cross 0.0978 / 0.1754.
Plus three structural identities (`plan_full[:, :4]` **is** `os` to 0 bits; `shrink_0` **is** the
anchor; a zero offset projects to the anchor).

### 2.1 anchor vs emitted, PAIRED, with the straight-line floor beside them

| arm | ADE 2 s (m) | **curvature MAE (1/m)** | heading (°) | cross (m) | speed MAE (m/s) | turn acc |
|---|---|---|---|---|---|---|
| `ha0` (**the straight-line floor**) | 0.6723 | **0.006802** | 2.7799 | 0.3132 | 0.4880 | 0.0000 |
| `ha0_ext` | 0.2874 | **0.003712** | 1.4322 | 0.1070 | 0.2540 | 0.7635 |
| `ha` | 0.2996 | 0.004030 | 1.5489 | 0.1226 | 0.2540 | 0.7419 |
| **the PICKED ANCHOR** | 0.4281 | **0.004019** | 1.7665 | 0.1754 | 0.3478 | 0.7682 |
| **the EMITTED plan** (`os`) | **0.2965** | **0.008150** | 1.2964 | 0.0978 | 0.2900 | 0.8594 |

**`emitted − anchor`, paired:**

| | delta | CI95 | |
|---|---|---|---|
| ADE 2 s | **−0.131669** | [−0.145772, −0.118232] | **SEPARATED** |
| **curvature MAE** | **+0.004132** | [+0.002482, +0.006122] | **SEPARATED** |
| heading | −0.4355° | [−0.5576, −0.3171] | SEPARATED |
| cross-track | −0.077528 | [−0.092737, −0.062200] | SEPARATED |
| speed MAE | −0.057793 | [−0.067472, −0.048148] | SEPARATED |

⇒ ⭐⭐ **CONFIRMED, and now with a paired interval where the PREREG had only two point
estimates.** The decoder's free-waypoint refinement buys **0.13 m of ADE, 0.44° of heading and
7.8 cm of cross-track**, and **pays 0.0041 1/m of curvature** — every one separated.
**The selection is fine; the refinement wrecks the shape.**

⇒ And the **picked anchor is already at the model-free floor**: anchor 0.004019 vs `ha0_ext`
0.003712 and `ha` 0.004030, and it beats the straight line `ha0` by
**−0.002898 [−0.003798, −0.002116] SEPARATED**. So the `os − ha0_ext` curvature gap
(**+0.003563 [+0.002053, +0.005422] SEPARATED**, the intersection-mask form PREREG §7 quotes) is
generated **entirely inside the decoder's output parameterisation**, downstream of routing and
downstream of selection.

### 2.2 THE FIX — pre-registered before it was scored

The refinement is `off = emitted − anchor`. Three families, each a map on **`off` only** — the
anchor is never touched:

* `shrink_α` = `anchor + α·off` — the trivial trade-off
* `poly_k` = `anchor + Π_k(off)`, `Π_k` = least squares onto a degree-k polynomial in time
* **`ridge_λ`** = `anchor + argmin_u ‖u − off‖² + λ‖D²u‖²` — **"penalise curvature in the
  refinement", solved in closed form instead of trained**

⭐ **COMMITTED IN ADVANCE, BOTH OUTCOMES.** *If every parameterisation that fixes curvature gives
back ≥ 50 % of the ADE halving, that is a real Pareto finding and it is reported as one.* It is
not what happened.

| arm | ADE | curv | ADE vs emitted | curv vs emitted |
|---|---|---|---|---|
| `shrink_0.75` | 0.3101 | 0.006544 | **+0.013620 SEP worse** | −0.001832 SEP |
| `poly_1` | 0.2967 | 0.005449 | +0.000234 **ns** | −0.002664 SEP |
| `poly_2` | 0.2957 | 0.005473 | −0.000778 SEP better | −0.002685 SEP |
| `ridge_0.3` | 0.2951 | 0.005250 | −0.001355 SEP better | −0.003003 SEP |
| `ridge_1` | 0.2955 | 0.004994 | −0.000989 SEP better | −0.003188 SEP |
| `ridge_3` | 0.2961 | 0.005152 | −0.000386 SEP better | −0.002977 SEP |
| `ridge_10 / 30 / 100` | 0.2965–0.2967 | 0.005329–0.005435 | +0.000012 … +0.000210 **ns** | −0.002678 … −0.002784 SEP |

⇒ **`shrink_α` IS the Pareto trade-off** — buying curvature with ADE, one for one. **The
smoothing families are not.** Every member of `ridge_λ` for λ ∈ [0.3, 100] and every `poly_k`
for k ≥ 1 cuts curvature by 33–39 % **and none is separated-worse on ADE.**

**TACTICAL:** turn-window lateral accuracy vs emitted is **not separated** for any smoothing arm
(`poly_2` **+0.0015** ns, `ridge_0.3` −0.0031 ns, `ridge_3` −0.0139 ns) over **647 turn windows**,
while the anchor is **−0.0912 [−0.1412, −0.0493] SEPARATED worse**. `ridge_1` improves
`lane_keep` recall **+0.0036 [+0.0005, +0.0067] SEPARATED**.
**LONGITUDINAL:** along-track MAE 0.2544 → 0.2537; step-wise speed MAE −0.00747 SEPARATED better.
**STRATEGIC:** ⛔ **NOT COMPUTABLE on this dump** — no route/goal option set travels with it.
Reported with its reason and never dropped.

### 2.3 ⛔ THE λ WAS RE-SELECTED ON A HELD-OUT SPLIT — the table above tunes on what it scores

The sweep above is scored on the data it was swept over, which is probe-failure-mode (3)
verbatim. So: split the **141 episodes** into a **FIT half (71 eps / 2,431 windows)** and a
**SCORE half (70 eps / 2,392 windows)**, state the rule *before* scoring — *argmin curvature MAE on
FIT subject to ADE(FIT) ≤ ADE_emitted(FIT) + 0.005 m* — and report the SCORE half only.
Selected: **λ = 3**.

| SCORE half, `ridge_3 − emitted` | delta | CI95 | |
|---|---|---|---|
| **ADE 2 s** | **−0.000211** | [−0.000690, +0.000266] | **not separated — the ADE halving is kept** |
| **curvature MAE** | **−0.001903** | [−0.003359, −0.000587] | **SEPARATED BETTER (−26 %)** |
| heading | −0.000408 rad | [−0.000994, +0.000191] | not separated |
| cross-track | **+0.000763** | [+0.000235, +0.001340] | **SEPARATED WORSE (0.76 mm)** |
| speed MAE | **−0.008266** | [−0.011285, −0.005525] | **SEPARATED BETTER** |

SCORE-half points: emitted 0.292271 / 0.007200 · **ridge_3 0.292060 / 0.005330** · anchor
0.416303 / 0.004282 · `ha0_ext` 0.272581 / 0.004038.
⇒ the `os − ha0_ext` curvature gap on the held-out half goes **0.003162 → 0.001292: 59 % of it
removed at no ADE cost.**

⚠️ The held-out margins are **smaller** than the full-set ones (−0.0019 vs −0.0030 curvature; ADE
ns rather than separated-better). That is what the split is for, and the held-out numbers are the
decision-grade ones.
⭐ Selection barely mattered here — **every** λ in [0.3, 100] is separated-better on curvature and
none is separated-worse on ADE — but *"it did not matter"* is a result, not a licence to skip it.

### 2.4 The controls that make the panel readable

⛔ **DELIBERATE REGRESSION — `ROUGHEN_hf_x2`** puts the removed high-frequency component back,
doubled. It reads curvature **+0.005626 [+0.004205, +0.007240] SEPARATED WORSE** (0.008150 →
0.013562), ADE +0.006085 SEP worse, heading +0.2091° SEP worse. ⇒ **the instrument responds in
BOTH directions**; a gate a regression cannot fail is not a gate.

⚠️ **A comment I had to correct from the arithmetic.** `poly_0` (a pure translation of the
refinement) is **NOT** curvature-inert: it reads **0.012097**, *worse* than the free offset.
`_seq_geometry` **prepends the ego origin** before differencing, so translating the path changes
the first step and therefore the first curvature pair. *"Smoother offset"* and *"smoother path"*
are different objects, and the first draft of that comment said otherwise.

⛔ **ORACLE ceilings are INADMISSIBLE as a capability claim** and are stamped as such — they fit
the basis to the GROUND TRUTH. `ORACLE_poly_1` 0.0580 m / 0.006284; `ORACLE_poly_2` 0.0059 m /
0.001423; `ORACLE_poly_3` 0.0000 (a degree-3 polynomial has 4 DOF on a 4-point grid, so it is an
identity and carries no information). Their only job is to price what a decoder *trained* in that
basis could reach. Note `ORACLE_poly_1`'s curvature (0.006284) is **worse than `ridge_1`'s
0.004994** — a least-squares fit to POSITION is not a fit to SHAPE, which is the whole point.

### 2.5 ⭐ INTEGRATION — the shipped Stage-0 seam does NOT already do this

`feasible_decode.project_feasible` exists behind `--feasible-decode`, so the honest first question
is whether we already have this lever. **We do not**, and the distinction is a scope one worth
stating: it caps curvature **MAGNITUDE** (|κ| ≤ κ_max, plus the Kamm disc) while R4b is about
curvature **ERROR** against the GT — a plan can sit far inside the friction circle and still be
wrong.

| arm | ADE vs emitted | curvature vs emitted |
|---|---|---|
| `FEASIBLE_emitted` (the shipped seam alone) | **+0.001886 SEP WORSE** | −0.000972 SEP (only **12 %** of the excess) |
| `ridge_3` (R4b) | −0.000386 SEP better | **−0.002977 SEP (37 %)** |
| `FEASIBLE_ridge_3` (composed) | +0.001058 SEP worse | **−0.003287 SEP (40 %)** |

⇒ they are **complementary, not duplicated**: the R4b smoother is the better ADE/curvature point;
composing adds the feasibility guarantee at the projection's own ADE cost.

---

## 3. ⛔ WHICH VARIANCE THESE INTERVALS ANSWER

*"Would another draw of EPISODES say this?"* — **nothing else.** Every R4b arm is **one
checkpoint's own output re-parameterised**, so the two operands of every margin come from the
**same forward pass**: there is no training-run variance (`H-ESTIM-SEED-1`) and no
inference-sampling variance (refcv4b's decoder is deterministic at inference,
`D-REFCV4B-SEED-SCOPE`) to answer for. This carries **no licence** to compare refcv4b to another
trained model.

---

## 4. WHAT IS DONE, AND WHAT REMAINS BLOCKED

**Done (clears its bar):** GP-1 and GP-2 are wired, tested and staged; the leak guard is proved
load-bearing by mutation; the value-sensitivity gate is re-proven through the wired path with its
committed numbers unchanged; R4b's mechanism is confirmed with paired intervals; and the shape
defect is shown fixable at no ADE cost on a held-out split.

**Blocked, and on what:**

1. ⛔ **The `gp_geo` / `gp_cond` / `base` training panel** — blocked on **the A40**, which is
   running refcv5 to ≈2026-09-08 07:33 UTC. The launch command is PREREG §9 verbatim; nothing
   else stands in its way. **A replicate arm is budgeted with it** (`H-ESTIM-SEED-1`: a separated
   interval between two TRAINED models is necessary, not sufficient).
2. ⚠️ **R4b as a shipped seam** needs its own pre-registration and a confirmation on a **different
   arm** — this is a post-hoc finding on the same dump the mechanism was diagnosed on. That
   pre-registration is written and lands with this package (`PREREG_R4B.md`); it costs **zero
   GPU** because the seam is an inference-time projection.
3. **The 6 s grid** is unmeasured here: the dump's GT (`g`) is the 2 s scored grid. The 8-slot
   plans are banked, so extending it is a re-analysis and not a re-run — named, not done.

---

## 5. THE SUITE, AS A CONTROLLED COMPARISON

| | passed | failed | errors | skipped |
|---|---|---|---|---|
| predecessor's baseline | 6,573 | **22** | **7** | — |
| **this turn** (`pytest -q tests/`, 985 s) | **6,618** | **22** | **7** | 118 |

⇒ **the failure and error counts are IDENTICAL**, and passed rises by **exactly +45** — the 23 +
22 tests added here. ⛔ **Zero of the 22 failures and zero of the 7 errors are in a file this turn
owns or touched** (`grep -c` over the failure list for `goal_point|test_refc_v3|test_refc\.py`
reads **0**). A targeted re-run of the 12 affected files against the FINAL tree reads **231 passed
/ 1 failed**, and that one failure is the pre-existing RL adapter drift below.

⚠️ **The one failure that touches this turn, and it was already RED.**
`test_rl_refc_adapter_robust.py::test_forward_kwargs_plumbs_every_channel_the_forward_accepts`
pins `refc_adapter.FORWARD_KEYS` against `RefCV3Model.forward`'s signature. It was failing
**before this turn** on `agent_gt`; this turn's `gp_point` / `gp_valid` widened the diff.

⛔ **RULED, by the stream that owns the edge** — and `stack/tanitad/rl/` was **not edited**, it is
another stream's file. **`gp_point` / `gp_valid` must NOT be plumbed from a batch.** They are a
DIAGNOSTIC PORT for the §2 eval-time interventions; the only thing a batch could supply them from
is the ego's own future path, i.e. **the LABEL**, which is a supplied route — optimistic by
construction on PhysicalAI — and would turn the deployable arm's own prediction into an oracle
read at inference. The deployable arm predicts its goal from vision *inside* the forward
(`gp_head(ctx)`), so leaving both at `None` is the correct call and the goal still reaches the
model. ⇒ `goal_point.DIAGNOSTIC_ONLY_FORWARD_KWARGS = ("gp_point", "gp_valid")` is **exported** so
the guard can exclude them by name and by reason instead of hardcoding a list that rots. MEASURED
with that exclusion applied, the only remaining gap is **`agent_gt`**, which belongs to the
agent-seam stream.

---

## 6. ARTIFACTS

| file | what |
|---|---|
| `raw/r4b_param_probe.py` | the probe (zero GPU) |
| `raw/R4B_PARAM.json` | every number above, with its CI, its n and its controls |
| `PREREG_R4B.md` | the pre-registration for R4b as a shipped seam |
| `stack/tests/test_goal_point_selection_seam.py` | 23 tests: the seam, the units pin, the leak-guard mutation |
| `stack/tests/test_goal_point_trainer_flags.py` | 22 tests: the four flags, the refusals, the record |
