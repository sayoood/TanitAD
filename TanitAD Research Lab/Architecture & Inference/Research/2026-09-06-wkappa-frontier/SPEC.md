# SPEC (PRE-REGISTERED) — the `W_KAPPA` frontier: can an arm track the road better than a straight line AND still turn left?

`TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-wkappa-frontier/SPEC.md`
Architecture & Inference FlyWheel · 2026-09-06 · dev-box RTX 4060 only.
⛔ The A40 (refcv5, ETA 2026-09-08 07:33 UTC) and Thor are NOT touched by this package.

**Written before any arm of this package produced output.** Batch 1 was launched at the
timestamp recorded in `raw/AS_LAUNCHED.txt`; the first `plan()` alone takes ~53 s and a full
arm ~37 min, so no result of this SPEC's own panel existed when the criteria below were fixed.
The one number this SPEC was designed against — `gkappa`, `w_turn = 0` — is a **PRE-EXISTING
rollout produced by a sibling at 05:20 UTC today** and is declared as such in §2: it places the
sweep, and every confirmatory claim needs the replicates that did not exist when this was written.

---

## 1. The question, in one sentence

> Does a `W_KAPPA` configuration exist that tracks the road **better than a perfectly straight
> line** (curvature MAE separated below the `ha0` floor `0.040083`) **and still turns left**
> (`turn_left` recall separated above `0.0000`), with zero friction-circle violations,
> non-vacuously, replicated across inference seeds?

## 2. ⛔ THE SPACING DERIVATION, STATED BEFORE LAUNCH — and why the SCALAR sweep is REFUSED

Binding floors (`D-REFAV1-CG-SEEDFLOOR`, `…/2026-09-05-refav1-close-the-gaps/raw/frontier.txt`),
all **inference**-seed floors on this rig, because iCEM **samples**:

| quantity | floor |
|---|---|
| curvature MAE (1/m) | **0.00200** |
| ADE (m) | 0.1035 |
| heading MAE (deg) | 1.1458 |
| `turn_left` recall | **0.00000** (exact) |
| `turn_left` recall RESOLUTION | 1 window of `n_true = 11` = **0.0909** |

### 2.1 ⛔ The SCALAR `W_KAPPA` knee CANNOT be resolved on curvature. Three arms not run.

Banked rungs (`A2_WKAPPA_SWEEP_RUNGS.md`): curvature MAE `wk7` **0.033522** → `wk15`
**0.030982** over `ΔW_KAPPA = 8.11245`.
⇒ local slope **3.131e-4 per unit**; minimum curvature-resolvable step = `0.00200 / 3.131e-4`
= **6.39 units**. The whole `7 → 15.11` interval is **8.11 units** wide, i.e. **1.27 minimum
steps**. ⇒ **no interior scalar rung is distinguishable from BOTH of its neighbours**, and a
`wk9 / wk11 / wk13` ladder would measure noise and print a curve. **Those three arms are
deliberately NOT run**, and this paragraph is the reason, fixed in advance.

### 2.2 ⭐ The axis that CAN resolve: the goal-conditioned TURN weight

`--w-kappa-by-goal <w_lane_keep>,<w_turn>` (`D-REFAV1-GOAL-KAPPA-COST`, already implemented and
unit-pinned) selects the curvature weight **per window from the DECODED tactical lateral goal
token**. The endpoints are MEASURED and they are far apart:

| arm | `w_lane_keep` | `w_turn` | curv MAE | `turn_left` recall |
|---|---|---|---|---|
| `gkappa` (banked, seed 0) | 15.11245 | **0.0** | 0.040167 | **0.3636** |
| `wk15` (banked, seed 0) | 15.11245 | **15.11245** | 0.030982 | **0.0000** |

⇒ span **0.009185** over `Δw_turn = 15.11245` ⇒ slope **6.078e-4 per unit** — **1.94× steeper
per unit than the scalar knee**, because the weight now moves only on the 22 of 40 windows whose
decoded goal is a turn.
* curvature-resolvable step = `0.00200 / 6.078e-4` = **3.29 units**
* recall-resolvable step (≥ 1 window of 11, measured drop 4 windows over 15.11 units
  = 0.265 windows/unit) = `1 / 0.265` = **3.77 units**
* binding minimum = **max(3.29, 3.77) = 3.77 units**

⭐ **RUNG SPACING ADOPTED: `Δw_turn = 5.0375` (= 15.11245 / 3) — 1.34× the binding minimum.**
Expected curvature step **0.00306 = 1.53× the floor**; expected recall step **1.3 windows**.

⭐ **RUNGS: `w_turn ∈ {0, 5.03748, 10.07497, 15.11245}`, `w_lane_keep = w_shift = 15.11245`
pinned on every rung.**

⚠️ **The `w_turn = 15.11245` endpoint IS the plain scalar `wk15` arm, and that is the tool's own
statement, not an assumption**: `refav1_arm.py` **raises** on an all-equal map — *"gives every goal
class the SAME weight, which is bit-identical to `--cost-weights` with that `W_KAPPA` … To run the
scalar control, OMIT this flag."* So that rung is run with the flag omitted.

### 2.3 Replicates

⛔ **3 inference-seed replicates per rung** (`--plan-seed 0, 1, 2`), because refav1's planner
samples and a separated episode-cluster CI answers *"would another draw of EPISODES say this?"*
— never *"would another INFERENCE RUN say this?"* (`D-REFAV1-SEED-GOAL-MISMATCH`, `I17`/V3).
⛔ **An effect below the floor is not an effect**, and every interval in the RESULT names which
of the three variances it has answered.
⛔ **Training variance (`H-ESTIM-SEED-1` / `I17`/V2) is NOT priced by this package** — one
checkpoint, step 21,109 — and is reported as a named blocker, not discharged.

## 3. The committed outcomes — BOTH written before the data

### ⭐ PASS
A rung `w_turn = X` exists with, **at ≥ 3 inference seeds**:
1. **LATERAL** — curvature MAE **separated below `ha0` = 0.040083** (margin > 0.00200, the
   curvature floor), and
2. **TACTICAL** — `turn_left` recall **separated above 0.0000** (≥ 1 of `n_true = 11` at every
   seed; the recall floor is exactly 0.00000), and
3. **SAFETY** — `kamm_over_rate` **0.0000** at `v0 >= 2 m/s` (`n = 27`), **non-vacuously**
   (`max|kappa| >= 0.02`, the `M58` gate) **at every seed**.

⇒ *the trade-off is a property of the cost's FORM, not of curvature penalisation as such, and the
frontier point is named.*

### ⛔ FAIL
**Every rung that beats the straight floor has `turn_left` recall indistinguishable from zero**
(0 of 11 at any seed, or a recall that does not survive replication).
⇒ **the trade-off is FUNDAMENTAL to this cost geometry**, it is stated so, and the next lever is
named — **the cost's *form*, not its weight**. The pre-committed successor in that branch is
`GOAL_KAPPA_TURN` / the goal **vocabulary**: `wk15`'s `max|kappa|` is **0.0800**, which is
`GOAL_KAPPA_TURN` **exactly**, so at high weight the plan never exceeds the single canonical turn
curvature the vocabulary offers — a **one-level** lateral vocabulary (`L1-0.08`) is then the
binding constraint and no weight can cross it.

### ⚠️ The third outcome, also committed in advance
A rung may **beat the floor and keep the turn but LOSE the safety zero** (`kamm_over > 0`, or a
zero that does not replicate as `combined`'s did not: 0.0000 → 0.0741). That is reported as a
**PARTIAL** — named, not folded into either branch — with the three axes side by side.

## 4. ⛔ Reporting rules this package binds itself to

1. **Four families per rung, never pooled, never ADE alone.** Families with no instrument on this
   panel are reported **UNAVAILABLE with their reason and their `n`** (clause 5): STRATEGIC
   (`n = 0`, the refav1 arm tool emits no route head) and **distance-keeping** (`n = 0`, no
   lead-agent track supplied). They are **not silently dropped**.
2. **Estimator: paired episode-cluster bootstrap** (`taniteval/ci.py`, `n_boot 2000`, seed 0).
   ⛔ `overlapping_holdout_se` is never used and no `heldout` split-mean is quoted.
3. **Tier stamp on every number**: `cl` / `ha` / `ha0` / `ha0_ext` = **T1** (self-action OPEN
   loop, PI ruling 2026-09-02); `ol` = **T0**. ⛔ Nothing here is closed loop.
4. ⛔ **`I22` — a motion assertion must be made IN THE REGIME the safety metric is about.**
   Every friction-circle zero in this package is printed **with `turn_left` recall beside it**,
   in the same table. A safety zero bought by declining the manoeuvre class is not a safety
   result and is labelled as such.
5. ⛔ **The `|dyaw| > 0.15` turn gate is NEVER used** — `D-TURNGATE` measured that the ground
   truth itself fails it 3 of 9 (it demands R 19 m at `v0` 1.40 m/s). The recall used is the one
   the banked rungs already use: **per-class lateral recall from the four-family TACTICAL block,
   `turn_left` against `n_true = 11`**, printed by
   `…/2026-09-05-refav1-cost-geometry/raw/four_family_table.py` run **unmodified**.
6. **Known-value controls that must read their known values**, on every record:
   * the four **model-free** arms (`ha` 0.8888 / `ha0` 0.9251 / `ha0_ext` 0.8772 / `ol` 0.8052
     ADE; `ha0` curvature **0.040083**) must be **bit-identical across every record** — one
     surface, not many. A panel whose model-free arms drift is comparing different window grids.
   * the feasibility audit's ground-truth path `g` must read `kamm_over` **0.0000** (a recorded
     vehicle's motion is feasible by construction), and `ha0_ext` / `ha` / `ol` must read
     **non-zero**. A zero that is not bracketed on both sides is not a measurement.
   * `ha0` must read `straight = 1.0000` on the trivial profile — the floor is a *perfectly*
     straight line, verified `max|y| = 0.000000 m`.
7. **Every number carries its evidence class**: `MEASURED (ours + artifact path)` ·
   `INHERITED` · `PUBLISHED`. Rows carried in from the two banked packages are marked
   **INHERITED** and are never re-derived silently.

## 5. Scope — what this package cannot establish

* **40 windows / 8 episodes, one checkpoint (21,109).** Not a corpus-scale or parity claim.
* `kamm_over_rate` is a **COUNT over 27 windows**, not a bootstrapped mean: quoted with its `n`,
  never with a CI it does not have.
* **Inference-seed replicates are not training replicates.** `I17`/V2 stays unpriced.
* The stack used is the **`C:\Users\Admin\tanitad-ctg` clone** — the same snapshot that produced
  `gkappa` and every banked p4 arm, chosen for comparability. It differs from repo HEAD by an
  **additive, default-off** distance-keeping cost term (`dk_spec=None` ⇒ bit-identical), a live
  sibling's `D-REFAV1-DK-COST`. Stated, not hidden.
