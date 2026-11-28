# The GOAL POINT — what it is, what it provably fixes that a route command cannot, and the bars it must clear

**Evidence class: MEASURED (ours).** **Tier: T1** (refcv4b's banked `navflip_dump`; this turn
re-ranks *within* it — no new forward pass). **Estimator: paired episode-cluster bootstrap,
`n_boot = 2000`, seed 0.** ⛔ `overlapping_holdout_se` appears nowhere.
**Compute: ZERO GPU.** ⛔ The A40 was never touched — refcv5's 44 h queue is intact. The dev-box
RTX 4060 was **in use by a sibling** (`bank_eval_field.py`, 2,750 MiB) and was not touched either.
**Pre-registration for the arm this produces: `PREREG.md`, same directory.**

| | |
|---|---|
| inputs | `…/2026-09-06-refcv4b-navpred/raw/navflip_dump.tgz` (8 arms, 4,823 windows / 141 episodes) · `…/2026-09-04-refcv4b-turn-coverage/raw/anchors_live_refcv4b.pt`, md5 **`297f6f1db52f6a56094846b0d7f71ed9`** — the `anchors.pt` md5 in the navpred SPEC's `held_constant` |
| instruments | `raw/pair_lateral_np2.py` · `raw/gp_lateral_axis_probe.py` · `raw/gp_long_axis_probe.py` · `raw/gp_selection_probe.py` |
| artifacts | `raw/NP2_PAIRED_LATERAL_TACTICAL.json` · `raw/GP_LATERAL_ADAPTIVE.json` · `raw/GP_LONG_AXIS_t4.json` · `raw/GP_LATERAL_GUARDED_s{20,40,60,80}.json` · `raw/GP_LATERAL_AXIS_s{15,20,30,40,60}.json` · `raw/GP_SELECTION_CEILING.json` |
| code | `stack/tanitad/refs/goal_point.py` (new) · `stack/tanitad/refs/refc_v3.py` (additive, `goal_point_inject` defaults **False**) · `stack/tests/test_goal_point.py` (26) · `stack/tests/test_goal_point_wiring.py` (7) |

---

## 0. ⭐⭐⭐ THE ONE-LINE ANSWER

**A working goal point fixes the LONGITUDINAL SELECTION — 88.7 % of the programme's oracle gap —
which a categorical route command has no vocabulary to express and a bearing is structurally
incapable of carrying. MEASURED on refcv4b's own fan: a metric goal point at t = 4 s recovers
`−0.0945 [−0.1164, −0.0757]` m, 57.1 % of the longitudinal selection ceiling, separated; the
SAME goal with its range stripped is separated WORSE by `+2.3632` m; and an ORACLE 3-way route
command's best possible decoding of even the LATERAL axis is "go straight" for all three classes
— bit-identical to the no-information goal.**

---

## 1. ⛔⛔ THE CATEGORICAL CEILING, MEASURED AS AN IDENTITY

Holding the LONGITUDINAL anchor choice at the model's own and varying only the lateral index
(9 candidates per window), leak-guarded, **n = 4,257 windows / 140 episodes**
(`raw/GP_LATERAL_ADAPTIVE.json`):

| surface | bank ADE Δ vs live | % of ceiling | lat acc Δ | **TURN acc Δ** | % of ceiling |
|---|---|---|---|---|---|
| `lat_ORACLE` — **the ceiling** | **−0.0528** [−0.0645, −0.0420] **S** | 100 % | +0.0155 **S** | **+0.0953** [+0.0594, +0.1378] **S** | 100 % |
| ⭐ `point_ORACLE` | **−0.0414** [−0.0508, −0.0328] **S** | **78.4 %** | +0.0235 **S** | **+0.1115** [+0.0713, +0.1548] **S** | **117 %** |
| `bearing_ORACLE` | −0.0414 [−0.0506, −0.0328] **S** | 78.3 % | +0.0230 **S** | +0.1115 [+0.0713, +0.1548] **S** | 117 % |
| ⛔ `categorical_ORACLE_3way` | **+0.1356** [+0.0880, +0.1946] **S WORSE** | −257 % | −0.0754 **S** | **−0.7842** [−0.8385, −0.7190] **S** | −823 % |
| ⛔ `categorical_MODAL_3way` | **+0.1356 — identical to the row above** | — | −0.0754 | −0.7842 | — |
| `point_STRAIGHT` (no-information) | **+0.1356 — identical again** | — | −0.0754 | −0.7842 | — |
| `point_MIRRORED` (deliberate regression) | **+0.3843** [+0.2727, +0.5168] **S** | −728 % | −0.0935 **S** | **−0.7842 S** | −823 % |

⭐⭐ **The three shaded rows are the SAME NUMBERS to four decimals, and that is the finding.**
The strongest deterministic 3→9 decoder of an ORACLE route label picks lateral level **4 (a_lat =
0, straight) for every class** — `route_left` 231/448 = 51.6 %, `route_straight` 2105/2377 =
88.6 %, `route_right` 385/699 = 55.1 %. ⇒ **a 3-way categorical route token carries exactly as
much lateral-selection information as assuming the road goes straight**, and used as a selection
rule it is separated WORSE than the model's own pick, emitting **zero** turns.

⇒ This is the *mechanism* behind the navpred RESULT's presence/content split. E13's content share
is 2.3 % not because the model failed to learn the token, but because **there is almost nothing in
the token to learn**: at a 2 s horizon the modal correct anchor on a turn window is still straight.

### ⚠️ A CORRECTION I MADE IN FLIGHT — the arc-origin defect, and what it cost

An earlier pass of the lateral probe measured the anchor's arc from **bank slot 0** (already
~5 m down the road at 0.5 s) while measuring the goal's arc **from the car** — a systematic
several-metre mismatch between the two quantities being matched. Under it the point appeared to
beat the bearing by **38 % relative**. With both polylines re-based on the ego origin (which is
what `lan.horizon_lead_m` does, in terms), **the point and the bearing TIE EXACTLY**
(−0.0414 / −0.0414; +0.1115 / +0.1115). ⛔ **The 38 % claim is retracted.**
⭐ And the tie is an **identity, not a coincidence**: anchors compared at the same ARC all sit at
~the same range from the car, so `−‖a − p‖` and `cos(a, p)` are monotonically related. **At a fixed
arc, the "metric" half of a metric goal point is degenerate.** That is what sent this turn to the
longitudinal axis, and it is why the registered goal is at a fixed TIME.
*(Root-cause class: the derived-constant / units family — a correct formula applied on two
different origins. Caught by a unit test that demanded a KNOWN value:
`test_anchor_point_at_arc_reads_a_KNOWN_value_on_a_straight_bank` asserts 20.0 and asserts
AGAINST 25.0.)*

### ⚠️ AND A LEAK THAT LOOKED EXACTLY LIKE A FINDING

Without the guard, the sweep read *"the shorter the arc, the better the goal"* — 70.2 % recovery
at 15 m falling to 11.2 % at 60 m. But the 2 s ground-truth path at the corpus mean speed covers
~22.7 m: **a 15–20 m goal is INSIDE the scored horizon and IS the answer.** Applying the
programme's own guard (`lan.horizon_lead_m` = `max(2 s GT arc, v0·2) + 5 m`, guard mean **28.23 m**,
p95 59.96 m) removes it. The registered design places the goal at a fixed TIME beyond the horizon,
and `GoalPointConfig` **refuses `t_goal_s <= t_pred_s` at construction** so no leaking arm can be
built at all.

---

## 2. ⭐⭐⭐ THE LONGITUDINAL AXIS — where a goal POINT is not a bearing

Holding the LATERAL index at the model's own and varying only `a_long` (12.9 candidates per
window), goal at **t = 4.0 s** (strictly beyond the scored 2 s grid), **n = 4,286 / 141**
(`raw/GP_LONG_AXIS_t4.json`):

| surface | bank ADE Δ vs live | % of ceiling | speed MAE Δ (m/s) |
|---|---|---|---|
| `long_ORACLE` — **the ceiling** | **−0.1655** [−0.1925, −0.1422] **S** | 100 % | **−0.1727 S** |
| ⭐⭐ `point_TIME_ORACLE` | **−0.0945** [−0.1164, −0.0757] **S** | **57.1 %** | **−0.1184 S** |
| ⛔ `bearing_TIME_ORACLE` (range stripped) | **+2.3632** [+2.2004, +2.5166] **S WORSE** | −1428 % | +2.5825 **S** |
| ⛔ `point_TIME_RANGE_x0.5` (regression arm) | +1.8809 [+1.7111, +2.0430] **S** | −1136 % | +2.1083 **S** |
| ⛔ `point_TIME_RANGE_x2.0` (regression arm) | +1.6844 **S** | −1018 % | +1.9369 **S** |
| `point_TIME_CONSTANT` (no-information) | +1.4978 **S** | −905 % | +1.6698 **S** |

⇒ **the longitudinal ceiling is 3.1× the lateral one (−0.1655 vs −0.0528 m), the goal point
reaches 57.1 % of it, and the two signals that a route command IS — a class and a direction —
reach none of it.** `nav_cmd` is `("follow", "left", "right", "straight")`: a purely lateral
vocabulary. There is no categorical arm to compare against on this axis **because a categorical
route command cannot express one.**

⚠️ **READ `+2.3632` CORRECTLY — a bearing does not "harm", it is DEGENERATE.** Inside one
`a_lat` row every candidate points in nearly the same direction, so the cosine is nearly constant
across the fan and its argmax is decided by numerical hair: `bearing_TIME_ORACLE` changes **95.0 %**
of picks over **95** distinct anchors and lands systematically on the extremes. The claim is
therefore **"a bearing carries NO longitudinal information"**, not "a bearing is actively harmful";
the magnitude is what a near-arbitrary pick costs on this fan (a uniformly random reachable pick
costs +1.8880 m on the whole fan for comparison).

⭐ **Third corroboration, from the WHOLE fan** (both axes free, `raw/GP_SELECTION_CEILING.json`,
n = 4,823 / 141, all four controls PASS): `point_ORACLE_s60 − bearing_ORACLE_s60` =
**−0.6734 [−0.8124, −0.5326] m, separated** — the point beats the bearing by a wide margin the
moment the longitudinal axis is allowed to move, and ties it exactly when it is not. Same file:
the whole-fan selection ceiling is `oracle_anchor_2s − live_selection` = **−0.2288 [−0.2621,
−0.2004] m separated** (bank 0.4281 → 0.1993), and a random reachable pick is **+1.8880 m**
worse — so the model's own selector is doing real work, and there is real headroom above it.

---

## 3. ⭐ THE HEAD-ACCURACY REQUIREMENT — a design number, not an opinion

Injecting noise into the ORACLE goal and re-reading the recovery. This converts *"how good must
the goal-point head be?"* into a bar that is checkable on the head alone.

| σ | LATERAL: bank-ADE recovery | LATERAL: turn recovery | RANGE: bank-ADE recovery |
|---|---|---|---|
| 0 (oracle) | **+78.4 %** | +100 %† | **+57.1 %** |
| 0.5 m | +44.8 % **S** | +118.9 % **S** | +54.0 % **S** |
| 1.0 m | **−80.7 % S (worse than the model's own pick)** | +71.7 % **S** | +47.2 % **S** |
| 2.0 m | −390.8 % **S** | +52.8 % **S** | **+17.2 % S** |
| 4.0 m | −1025 % **S** | −54.7 % (not separated) | **−57.9 % S WORSE** |

† the σ = 0 lateral turn row reads 117 % of the ceiling because the *ceiling* is defined by 2 s
error, not by turn recall; small lateral noise also breaks ties toward curved anchors, which turn
recall rewards. ⚠️ Reported as a **tie-breaking artifact, not a benefit of noise.**

⇒ **Registered head gate: ≤ 1.0 m RMS lateral AND ≤ 2.0 m RMS range** on the 4 s goal
(`goal_point.lateral_rmse_m` / `range_rmse_m`, in METRES, every eval). Below 0.5 m lateral the ADE
contribution is positive too; above 4 m on either axis the lever is measured **negative**.

---

## 4. ⭐⭐ NP-2 CLOSED — the navpred RESULT's LATERAL/TACTICAL rows are now PAIRED

The navpred RESULT reported these per arm with `_intervals_complete: false` and said in terms:
*"Do not quote any of these as separated until NP-2 lands."* This lands it, zero GPU
(`raw/NP2_PAIRED_LATERAL_TACTICAL.json`). Every per-arm point estimate reproduces that table
exactly (`os` curv 0.008150 / head 1.2964; `os_navflip` 0.009270 / 1.3773; …), and the pooling
control passes on all 8 arms.

| margin | curvature (1/m) | heading (deg) | `turn_right` recall |
|---|---|---|---|
| ⭐ **`os_navflip − os`** | **+0.001120** [+0.000090, +0.002539] **S** | **+0.0810** [+0.0303, +0.1660] **S** | **−0.0606** [−0.0969, −0.0297] **S** |
| `os_navpred − os` | +0.000018 [−0.000277, +0.000280] no | +0.0213 [+0.0030, +0.0409] **S** | −0.0051 no |
| `os_navshuf − os` | +0.000535 [+0.000004, +0.001201] **S** | +0.0201 no | −0.0227 no |
| `os_navzero − os` | +0.001057 [+0.000476, +0.001866] **S** | +0.0563 no | −0.0227 no |
| ⭐ `os_navflip − os_navpred` | +0.001102 [+0.000120, +0.002457] **S** | +0.0597 [+0.0079, +0.1442] **S** | — |
| ⛔ **`os − ha0_ext`** | **+0.004438** [+0.002692, +0.006464] **S WORSE** | −0.1358 **S BETTER** | +0.1187 **S BETTER** |
| `os − ha0` | +0.001349 [−0.000301, +0.003299] **no** | −1.4835 **S** | — |
| `ha0_ext − ha0` | −0.003089 [−0.004052, −0.002276] **S** | −1.3477 **S** | — |

⭐⭐ **THREE CONSEQUENCES:**

1. **The nav CONTENT reaches path shape and the lateral decision with SEPARATED margins while its
   ADE margin is not separated.** The navpred RESULT's §4 observation is now a measured, paired
   fact. **An ADE-only read of route work has been reading the wrong instrument.**
2. ⭐ **The gating control that FAILED on ADE PASSES on the lateral family.**
   `os_navpred − os_navshuf` on curvature: **−0.000565 [−0.001134, −0.000094] SEPARATED** on the
   strictly-paired **intersection mask**; **−0.000517 [−0.001154, +0.000006]** — not separated *by
   6e-6* — on the own-mask form `ff.lateral` publishes. ⚠️ **The estimator choice flips the
   verdict, so both are reported and neither is cherry-picked.** Independent corroboration from a
   different metric: `os_navpred − os_navshuf` on `turn_left` recall is **+0.0319 [+0.0049,
   +0.0692], separated.** ⇒ the model's own predicted route IS beating a distribution-matched
   random token — just not on ADE.
   *(The mask difference is real: `ff.lateral` masks with the ARM'S OWN path, so the two operands
   of a difference can rest on slightly different step sets — 13,545–13,591 here.)*
3. ⛔ **On curvature the planner is statistically indistinguishable from a straight line**
   (`os − ha0` not separated) and **separated WORSE than the kinematic echo control**
   (`os − ha0_ext` +0.004438). That is the lateral defect, now with intervals — and §5 localises it.

---

## 5. ⛔⛔ WHERE THE CURVATURE DEFECT ACTUALLY LIVES — and it is NOT routing

| | ADE (m) | curvature (1/m) | heading (deg) | cross-track (m) |
|---|---|---|---|---|
| the **PICKED ANCHOR** (`sel_bank_nav_true`, unrefined) | 0.4281 | **0.004019** | 1.7665 | 0.1754 |
| the **EMITTED PLAN** (`plan_full_nav_true` = `os`) | **0.2965** | **0.008150** | **1.2964** | **0.0978** |

⇒ **the decoder's refinement halves ADE, improves heading 27 % and cross-track 44 % — and DOUBLES
curvature error (2.03×).** The anchors are constant-curvature unicycle rolls, smooth **by
construction**; the decoder emits free waypoints under an ADE-shaped loss and **buys position with
smoothness**.

⇒ ⭐ **The `+0.004438` curvature gap to `ha0_ext` is generated in the DECODER OUTPUT
PARAMETERISATION, not in routing.** A goal point — which changes *where the model aims*, not *how
its output is parameterised* — is **not predicted to close it**, and `PREREG.md` §7 registers that
as an adverse prior **in advance**, with the next lever named: **`R4b`, a curvature-parameterised
or residual-over-anchor decoder output.**
⚠️ This does not make the curvature bar disappear: `PREREG.md` §3 keeps `gp_geo − base` separated
negative on curvature as a committed criterion, and the predicted outcome is that it **FAILS**.
Committing the prediction is what makes the failure informative instead of embarrassing.

---

## 6. THE CONTROLS — every one, PASS and MISS

| # | control | must read | MEASURED | verdict |
|---|---|---|---|---|
| **K1** | `gt_future_ext` transformed into the ego frame must equal the scored 2 s grid `g` | < 1e-3 m | **7.63e-06 m** | ✅ PASS |
| **K2** | the **rebuilt v0-conditioned bank** must equal the banked `sel_bank_nav_true` | < 1e-4 m | **1.53e-05 m** | ✅ PASS |
| **K2ⁿ** | ⭐ **same-breath NEGATIVE control**: the FIXED `anchors` table read at the same index must **NOT** match | must differ | **157.62 m off** | ✅ the probe discriminates |
| **K3** | `os` ADE reproduces the banked value | 0.2965 | **0.2965** | ✅ PASS |
| **K4** | model-free `ha` / `ha0` / `ha0_ext` | 0.2996 / 0.6723 / 0.2874 | **0.2996 / 0.6723 / 0.2874** | ✅ PASS |
| **K5** | `plan_full[:4]` must equal `os` | < 1e-5 m | **0.0** | ✅ PASS |
| **K6** | pooling control: the probe's re-pooled curvature must equal `ff.lateral`'s | equal to 1e-6 | equal, **8/8 arms** | ✅ PASS |
| **K6b** | the probe's per-window speed MAE must equal `ff.longitudinal`'s `speed_mae_mps` | equal | **equal to 4 dp on all 11 surfaces** | ✅ PASS — the paired speed margins rest on the SHIPPED definition, not a re-implementation |
| **K7** | candidates per window | 9 lateral / 12.9 longitudinal | 9.0 (min 9) / 12.907 (min 12) | ✅ PASS |
| ⚠️ **K8** | anchors SHORTER than the goal arc are clamped to their endpoint | reported, not hidden | **26.5 %** (adaptive arc) | ⚠️ STATED — it makes distant goals LESS discriminative, i.e. it **understates** the lever |

⭐ **K2 is also an independent re-validation of the `alat` control units** (`D-REFCV4B-ASTAR-…`
sibling class): rebuilding the bank under `control_units='alat'` reproduces the banked selection
to 1.53e-05 m, while the fixed-anchor reading is 157.62 m off. If the units were curvature the
reconstruction would not match.

⛔ **`a_star`, `anchor_acc` and `sel_agrees_oracle` are NOT read anywhere in this package** —
`D-REFCV4B-ASTAR-GEOMETRY` is unrepaired (NP-4). Every oracle here is computed in-file from `g`
with its geometry stated.

---

## 7. ⛔ SCOPE — what these numbers do NOT say

1. **Selection path only, bank scale.** A counterfactual pick is scored on its UNREFINED anchor;
   the decoder's per-candidate offset is only known for the candidate actually selected, so a
   re-ranked arm's refined path cannot be reconstructed offline. Bank-scale ADE (live 0.4109) is
   never mixed with emitted-plan ADE (0.2965).
2. **Every goal row is ORACLE-FED = an UPPER BOUND.** A trained arm predicts the point; §3 is the
   requirement curve that says how much accuracy it must reach for the bound to be worth chasing.
3. **This is a re-ranking of a banked T1 dump, not a trained arm.** No claim here is a claim about
   refcv6's performance. It ranks levers and sizes bars; that is all.
4. **Which variance:** the fan and the checkpoint are held fixed, so the interval answers *"would
   another draw of EPISODES say this?"* — not another training run (`H-ESTIM-SEED-1`) and not
   another inference sample.

---

## 8. THE FOUR FAMILIES

**LATERAL** — §4 and §5, with paired margins for the first time (NP-2). **TACTICAL** — turn-window
lateral decision accuracy is the pre-registered PRIMARY (§1, §4); goal/anchor selection is
reported as the *ceiling* structure rather than as `anchor_acc`, which is blocked by NP-4.
**LONGITUDINAL** — §2: the ceiling is `−0.1655 m` / `−0.1727 m/s` speed MAE, and this is the
family the goal point is registered to move. **STRATEGIC** — unchanged by this turn and quoted
from the navpred read: route acc 0.7791, κ 0.4864, identical under all five conditionings;
`H-REFCV4B-ROUTE-ECHO` stands. ⛔ ADE is reported beside all four and is explicitly not the
headline.

---

## 9. DELIVERABLE MANIFEST

| artifact | where |
|---|---|
| `goal_point.py` — E15: label, guard, head, conditioning, param-free prior, provenance, head-gate readouts | repo `stack/tanitad/refs/goal_point.py` |
| wiring — `goal_point_inject` (default **False**), zero-init, emits `goal_point` for the selection seam | repo `stack/tanitad/refs/refc_v3.py` |
| 26 module tests incl. **two mutation tests** (the E13 collapse; the bearing's longitudinal blindness) | repo `stack/tests/test_goal_point.py` |
| 7 wiring tests incl. default-OFF parity and edge-reachability | repo `stack/tests/test_goal_point_wiring.py` |
| pre-registration with committed bars, the value-sensitivity gate and the launch command | repo `…/2026-09-06-goal-point/PREREG.md` |
| this read | repo `…/2026-09-06-goal-point/RESULT.md` |
| 4 probes + 11 result JSONs | repo `…/2026-09-06-goal-point/raw/` |

**Suite, CONTROLLED comparison** (mirror `C:\Users\Admin\tanitad-wt\stack`, the same 5 pre-existing
files before and after, `PYTHONPATH` verified to import the mirror):
`tests/test_lan.py test_refc_select.py test_refc_v4.py test_refc_v3.py test_refc_tactical.py`
→ **BEFORE (both new files absent, `refc_v3.py` unpatched): 136 passed, 1 skipped.**
→ **AFTER (module + wiring + tests present): 136 passed, 1 skipped.** Plus **33 new tests, all green.**

**FULL SUITE**, `cd stack && pytest -q -p no:cacheprovider` (15 m 35 s):
**6,573 passed · 118 skipped · 2 xfailed · 22 failed · 7 errors.** ⛔ *"They look pre-existing"* is
not evidence, so the 9 files carrying every one of them were re-run **with the change reverted**
(`refc_v3.py` restored from its pre-patch copy, both new files removed):
**BEFORE 15 failed / 227 passed / 3 skipped / 7 errors — AFTER 15 failed / 227 passed / 3 skipped
/ 7 errors. IDENTICAL.** ⇒ every failure is pre-existing and unrelated (`test_secret_scan` hook
currency, `test_runbook_commands`, `test_v6_chain`, `test_eval_contamination` fixtures,
`test_launch_closure_audit`, `test_text_encoding_is_explicit`, `test_kingate_contract`,
`test_rl_refc_adapter_robust`, `test_v6_st_launch_fixes`). The string `goal_point` appears **0**
times in the full run's failure output.

## 10. WORK ITEMS AND BLOCKERS

| id | item | status |
|---|---|---|
| **GP-1** | ⛔ **BLOCKER** — the geometric selection seam needs ~6 lines in `refc.py`'s ranked-score block (`PREREG.md` §8). **`refc.py` is not this agent's file.** `refc_v3.py` already emits `hook_out["goal_point"]`, so the patch needs no second forward. | **ESCALATED** to the REF-C model owner |
| **GP-2** | ⛔ **BLOCKER** — trainer flags `--goal-point-inject / --goal-point-geo-prior / --goal-point-t / --goal-point-w` in `scripts/refc_v3_train.py`. **Not this agent's file.** | **ESCALATED** |
| **GP-3** | ⛔ **BLOCKER (compute)** — the A40 runs refcv5 until ≈2026-09-08 06:20 UTC; the dev-box GPU is held by a sibling. The arm is a next-slot launch, and §9 of `PREREG.md` carries the exact command. | named, not idle |
| **GP-4** | the goal-point LABEL must be minted into the training batch (ego future at 4 s + validity). `goal_point_label_at_time` is written and tested; the dataset call site is one line in the trainer. | with GP-2 |
| **NP-2** | ⭐ **CLOSED this turn** — paired LATERAL/TACTICAL margins (§4) | done |
| **GP-5** | a **replicate arm** for `gp_geo − base` (`H-ESTIM-SEED-1`: a separated CI across two trained models is necessary, not sufficient — measured ~17 % false-positive rate) | budgeted in `PREREG.md` §10 |
