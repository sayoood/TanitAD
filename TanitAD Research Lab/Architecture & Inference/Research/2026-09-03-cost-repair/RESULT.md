# RESULT — L3 + L4: the CHORD metric and the weight it forces

**Date:** 2026-09-03 (Europe/Berlin) · **FlyWheel:** Architecture & Inference ·
**Branch:** `agent/arch-inf-20260803` · **Tier:** **T0** (a COST/WM diagnostic anchored at
`t0`; every win count is a property of the COST, never of the car — no driving claim without
the T1 adapter) · **Evidence class:** MEASURED
**Pre-registration:** `SPEC.md` in this package, written 10:19 Berlin **before any edit to
`_cost_chunk` and before any measurement**. Nothing below re-writes a criterion after the fact;
where a committed criterion could not be evaluated, §7 names it and says what is missing.
**Primary sources — the only things quoted here:** `raw/cost_repair_ep2.json`,
`raw/cost_repair_incumbent.json`, `raw/cost_repair_ep2_rows.json`,
`raw/cost_repair_incumbent_rows.json`. **No number in this file was taken from any agent's
report prose.** Three numbers (S3, the tipping weight, the win counts) were **recomputed
independently from the row-level f64 arrays** and are marked ✅RECOMPUTED where they appear.

---

## 0. HEADLINE

> ## ⛔ THE JOINT CHANGE DOES **NOT** MAKE THE PLANNER TURN WHERE THE HUMAN TURNS.
>
> On the population where the **car actually turns** (`gt_turn_deg ≥ 5°`, **27 windows /
> 10 episodes / 3 tokens**, the same 27/10 on **both** checkpoints), the best cell in the whole
> panel — `chord + L0 + steer units` at the shipped `w_κ = 0.05` — puts the κ-argmin on the
> **human's side** on **6 of 27** windows on `ep2` and **0 of 27** on the `incumbent`, and wins
> the total cost on **10 of 27** / **1 of 27**. A majority is 14. **S5 FAILS as written**, on
> every clause it could fail on: it is a **minority**, it is **convention-dependent**
> (steer 6/27 vs κ 5/27, and 10 vs 1 wins), and it is **checkpoint-dependent**
> (`ep2` 6/27 vs `incumbent` 0/27).
>
> The one number that looks like a success — **25/25** on the decoder's curved-goal stratum —
> is **25 windows = 4 episodes = ONE token (`TURN_R × ADAPT_SPEED_FOR_CURVE`), on one
> checkpoint, under one of two unit conventions.** ⛔ **It is a DIRECTION, NOT A VERDICT.**
> Its sibling convention reads **5/25** and the other checkpoint has **no curved-goal window at
> all** (**UNDECIDABLE**, not passed).
>
> **And the mechanism is not the one the hypothesis named.** The 25/25 is bought by two things
> that are *not* "the planner learned to see curvature": (a) L0 (`goal_time_grid="plan"`) makes
> the canonical turn's goal term **exactly 0.0 on 25/25 windows** — an identity, not a
> prediction; (b) the chord is **5,793× larger in scale** than `1−cos`, so it outruns an
> unchanged `0.05·κ²`. ✅RECOMPUTED. This is the implicit reweighting **the SPEC itself declared
> in advance** (§1, L4) — which is why `chord` alone was pre-registered as a **regression arm**.
>
> ⭐ **The finding that matters most is the opposite of the one sought.** Once the chord makes
> the goal term *readable* at all, what it says is **"go straight"**: the goal term's own
> κ-argmin sits at **κ = 0 on 110–130 of 140 windows** under the chord (vs 0–11/140 under
> `1−cos`, where it is quantisation noise). The world model does not want to turn. **A cost
> repair cannot fix that** — the next branch is the goal SPACE
> (`PREREG_TACTICAL_DECODER.md` §10), not another weight.

**ESCALATION / INTEGRATION.** Three items need a PI or Master-Mind decision and are **not**
taken here: (1) `w_κ = 0.05` is **not** changed — the derived weights are reported only;
(2) the `cost_metric` flag is **default-safe and inert** (`"cos"` is bit-identical to HEAD) and
can be integrated as a flag, but ⛔ **`chord` must not become the default** — on this evidence
it is a 5,793× reweighting wearing a metric's clothes; (3) the decoder asks for curvature on
only **9 of the 27** windows where the car turns on `ep2`, and on **0 of 27** on the
`incumbent` — a decoder-level defect this package cannot fix and does not own.

---

## 1. What was measured, and on what

| | |
|---|---|
| checkpoints | **2**, both step **1,000**: `refav1_eval_slice/ckpt_ep2/ckpt.pt` (+`config.json`) and `refav1_eval_slice/ckpt/ckpt.pt` (config from `ckpt['cfg']`) |
| windows | **140 per checkpoint**, `--episodes-n 20`, `--window-stride 10` |
| episodes | **20** — ⚠️ the eval slice contains **exactly 20** episodes (`refav1_eval_slice/eps`, 20 files; `fp8` cache, 20 entries). The population is **already at the slice's ceiling** (§6) |
| candidate box | κ ∈ [−0.2, +0.2] on a **21-point** axis; `accel_axis = [0.0]` — ⚠️ **a single accel value**, so **21 cells per window, not the 189 the SPEC names** (§7) |
| shipped weights | `W_KAPPA = 0.05`, `W_JERK = 0.02`, `W_VEND = 0.1`, wheelbase 2.9, `plan_steps` 10, `cost_time_grid` dense (held constant) |
| goal dim | **65,536** (`[64, 1024]`) — ⚠️ **n = 140 ≪ d = 65,536: underpowered BY CONSTRUCTION** |
| panel | 8 cells = {`cos`, `chord`} × {κ (A), steer (B)} × {`full`, `plan` (L0)} |
| provenance | `cost_surface_probe.py` md5 `4c5dc3d57bf2cb06376fb788ec0cab3e` (imported, **not modified**); `refa_v1.py` md5 `3a539ea2996f1337e627a1e64c5fddda` |
| wall-clock | ep2 231.8 s, incumbent 267.5 s; dev-box 4060, forward-only. ⛔ Thor and the A40 pod were **not** contacted |

### The three populations, never mixed

| stratum | `ep2` | `incumbent` | what it is |
|---|---|---|---|
| `all` | 140 win / 20 ep / **3** tokens | 140 / 20 / **2** | every scored window |
| `curved_goal` (the **DECODER** asked for curvature) | **25 win / 4 ep / 1 token** (`TURN_R × ADAPT_SPEED_FOR_CURVE`) | **0 / 0 / 0** | ⛔ **a direction, not a verdict** / **UNDECIDABLE** |
| `gt_turn_ge_5deg` (the **CAR** turns) | **27 win / 10 ep / 3 tokens** | **27 / 10 / 2** | the population the human turns on |

⚠️ **The two turn strata overlap on only 9 of 27 windows on `ep2` and on 0 of 27 on the
`incumbent`.** The decoder and the car disagree about when there is a turn. Signed ground-truth
heading change over the 27 spans **−49.7° … +44.8°**.

---

## 2. S6 — the controls (a failure VOIDS the panel). **ALL PASS, on both checkpoints.**

| control | required known value | `ep2` | `incumbent` | |
|---|---|---|---|---|
| **X0** zero-model | goal ptp along κ **exactly 0.0**; `total(κ)−total(0)` recovers `0.05κ²`; argmin at κ=0 on 140/140 | ptp **0.0** both metrics; recovery residue **0.0** (cos) / **0.5 f64 ULP** (chord); **140/140** at κ=0 | ptp **0.0**; **0.0** / **0.5 ULP**; **140/140** | ✅ |
| **X1** constant-cost | `c_total ≡ 0.0`; argmin = index 0 (deterministic tie) | `max|c_total| = 0.0`, argmin 0 | same | ✅ |
| **X2a** shipped-cost re-scoring | ≤ 1e-6 relative vs `PlanResult.baseline_costs` | **worst rel err 0.0**, n=5 | **0.0**, n=5 | ✅ |
| **X2a** bit-identity | `_goal_term(·,"cos")` bit-identical to the shipped expression | **all bit-identical**, `max_abs_diff 0.0` | same | ✅ |
| **X2b** banked winner | reproduce the banked `cl` trajectory to < 3.8e-6 m | **max abs 0.0 m**, n=5 (mixed `cem` / `baseline:hold_v0`) | **0.0 m**, n=5 | ✅ |
| **X3** deliberate regression `√(2(1−cos))` in f32 | must recover **NOTHING** | `n_distinct` median **3.0** = cos's **3.0**; **0/140** windows where sqrt beats cos | median **3.0** = **3.0**; **0/140** | ✅ |

⭐ **X3 is the load-bearing control and it holds.** The same algebra, evaluated in f32, recovers
**nothing** (mean is in fact *lower* than cos: 2.84 vs 3.06 on `ep2`). So the gate can see the
defect it exists to catch, and the chord's gain in §3 is real arithmetic, not a bug in the meter.

---

## 3. S2 — the resolution claim. **PASSES. And the inherited `~10⁷` needs a correction.**

`n_distinct_f32` = how many values the planner can actually tell apart along κ (**capped at 21
by the axis**). `steps_true` = `ptp_f64 / spacing32(median)` — whether f32 *could* see it.

| cell | `n_distinct_f32` median — `ep2` | — `incumbent` | `steps_true` median — `ep2` | — `incumbent` |
|---|---|---|---|---|
| `cos__A__full` | 3.0 | 3.0 | 9.76e-01 | 2.73e-03 |
| `cos__B__plan` | 9.0 | 3.0 | 9.36e+00 | 2.44e-02 |
| `chord__A__full` | **21.0** (saturated) | **21.0** | 2.33e+07 | 2.23e+07 |
| `chord__B__plan` | **21.0** (saturated) | **21.0** | 1.83e+07 | 2.67e+07 |

**GAIN (chord / cos), paired per window, 95 % CI clustered on the 20 EPISODES:**

| | `n_distinct_f32` ratio (median, CI95) | `steps_true` ratio (median, CI95) |
|---|---|---|
| `ep2` A·full | **7.00** [7.00, 7.00] | 2.37e+07 [2.34e7, 2.41e7] |
| `ep2` B·plan | **2.33** [2.33, 2.63] | 1.94e+06 [1.91e6, 1.97e6] |
| `incumbent` A·full | **7.00** [7.00, 7.00] | 7.25e+09 [6.56e9, 1.72e10] |
| `incumbent` B·plan | **7.00** [7.00, 7.00] | 1.21e+09 [1.11e9, 1.40e9] |

**VERDICT S2: PASS.** Every CI excludes 1, on both checkpoints and both conventions.
`H-COST-CHORD-1` is **SUPPORTED** on its stated terms.

⚠️ **RETRACTION-WORTHY.** The brief's inherited **`~2 → ~10⁷`** figure is **not reproduced for
the quantity that matters to the planner.** Measured `n_distinct_f32` gain is **2.3× – 7×**,
not 10⁷, and it is **hard-capped at 21** by the grid — the chord saturates the axis. The `~10⁷`
belongs to a *different* statistic, `steps_true` (1.9e6 – 1.7e10 measured), which describes
what f32 *could* resolve, not what the planner counted. **Both halves of the SPEC's registered
prediction §3.7.2 are therefore graded:** the `n_distinct` prediction (10⁰–10²) is **CORRECT**;
the `steps_true` prediction (10³–10⁶) is **WRONG — measured 10⁶–10¹⁰, above the predicted band
on 7 of 8 cells.** Both go to the RETRACTION_LOG.

---

## 4. S3 / X4 — monotone equivalence. **Ordering half PASSES. Arithmetic half CANNOT BE
CONFIRMED AT THE COMMITTED THRESHOLD — and the reason is itself the finding.**

✅RECOMPUTED by this agent from `*_rows.json` `kmarg_f64`, all 140 windows × 21 κ cells,
4 matched cos/chord cell pairs, both checkpoints:

* **Discordant pairs: 0 out of 235,200** (29,400 pairs per cell-pair × 4 × 2). **Spearman ρ = 1.0
  exactly, on every window, on both checkpoints.** The chord **never re-ranks**. ✅ **PASS** —
  and `stack/tests/test_cost_chord.py::test_e_chord_is_monotone_equivalent_in_float64` pins the
  same assertion (`assert disc == 0`) independently.
* **`chord²/2 − (1−cos)` ≤ 1e-12 relative in f64: NOT MET.** Measured relative error median
  **3.9e-09** (`ep2` A·full) / **2.4e-07** (`incumbent` B·plan), p99 **5.0e-02 / 6.2e-01**,
  max **1.2e-01 / 1.35e+00**; **2,893–2,935 of 2,940 cells exceed 1e-12**, in every cell-pair.

⭐ **Read this correctly: the chord is not the term that fails the identity — `1−cos` is.**
On individually healthy cells the identity is exact to displayed f64 precision (ratio
`1.000000`). The tail blows up precisely where `1−cos` cancels against 1.0, and on the
`incumbent` the banked **f64** `1−cos` reads **exactly 0.0 on 47–67 of 2,940 cells** while the
chord is finite and monotone there. So the committed 1e-12 threshold is **unreachable by
construction**: it asks `1−cos` to represent itself to 1e-12 relative, which it cannot do even
in f64 at these magnitudes. **This is the L3 defect appearing inside its own control.**
⚠️ The threshold is graded **NOT MET**, not silently rescoped; §7 records what would settle it.

---

## 5. S4 — the derived weights and the deletion test. **PASSES (S4), and confirms M5.**

Deletion rule (pre-registered): `w` is a **deletion in disguise** iff `w·κ_max² < q(m)`, the
metric's own realised f32 quantum. Curved-goal stratum, `ep2` (**n = 25 windows / 4 episodes /
1 token — a direction, not a verdict**). ⛔ The `incumbent` stratum is **empty → every weight is
`null`/`nan` → UNDECIDABLE on that checkpoint.**

| cell | goal advantage `A` (median, f64) | `w_κ^tip` | `w_κ^comm` | `w_κ^κ-only` | tips that are **deletions** |
|---|---|---|---|---|---|
| `cos__A__full` | **−3.97e-09** | −3.10e-08 | 1.43e-06 | **−6.36e-07** | **25/25** |
| `cos__A__plan` | 8.47e-09 | 6.62e-08 | 2.30e-06 | 1.28e-06 | **20/25** |
| `cos__B__full` | **−6.38e-08** | −4.98e-07 | 1.26e-05 | **−1.02e-05** | **25/25** |
| `cos__B__plan` | 7.77e-08 | 6.07e-07 | 1.57e-05 | 1.18e-05 | **20/25** |
| `chord__A__full` | **−5.03e-05** | −3.93e-04 | 7.34e-03 | **−7.90e-03** | 20/25 |
| `chord__A__plan` | 1.30e-04 | 1.02e-03 | 1.05e-02 | 2.03e-02 | **0/25** |
| `chord__B__full` | **−3.06e-04** | −2.39e-03 | 2.37e-02 | **−4.80e-02** | 22/25 |
| **`chord__B__plan`** | **3.94e-04** | **3.07e-03** | 2.77e-02 | 6.16e-02 | **0/25** |

**VERDICT S4: PASS** — `chord__A__plan` and `chord__B__plan` each yield a `w_κ^tip` that is
**not** a deletion on **0/25** windows. `H-COST-WEIGHTS-1` is **SUPPORTED on the stratum it
could be evaluated on**, and **UNDECIDABLE on the `incumbent`**.

Three things this table says that must travel with it:

1. **M5 is confirmed and generalised.** Under the shipped `1−cos` the tipping weight is
   **6.6e-08 / 6.1e-07** and is a **deletion on 20–25 of 25** windows — consistent with the
   registered **1.66e-08, "a deletion not a weight."** Under `1−cos` there is **no κ² weight
   that ranks the turn without deleting the penalty.**
2. **A negative `w_κ^κ-only` is the "no weight whatsoever" outcome the SPEC demanded be stated
   plainly (§3.5).** It fires on **4 of 8 cells** — every `full`-grid cell and both `cos__*__full`.
   On those cells the jerk charge alone already exceeds the goal advantage: **no κ² weight
   whatsoever ranks the turn.**
3. `w_κ^comm` (commensurability) lands at **2.77e-02** under `chord__B__plan` — within 2× of the
   shipped 0.05. Under `cos` it is **1.4e-06 – 1.6e-05**, i.e. **3,000–36,000× below shipped**.
   That is M1's "99.5 % penalty" restated as a weight.

---

## 6. S5 — does the joint change rank a turn? **FAILS.** (the panel's reason for existing)

### 6a. The decoder's curved-goal stratum — `ep2` only

✅RECOMPUTED from the rows (`named_f64`, `kappa2_named`, `jerk2_named`); reproduces the summary
JSON **exactly**.

| cell | wins at shipped `w_κ = 0.05` |
|---|---|
| `cos` — all four cells | **0 / 25** |
| `chord__A__full`, `chord__B__full` | **0 / 25** |
| `chord__A__plan` (κ units) | **5 / 25** |
| **`chord__B__plan` (steer units)** | **25 / 25** |

⛔ **25/25 on n = 25 windows / 4 episodes / ONE token (`TURN_R × ADAPT_SPEED_FOR_CURVE`) is a
DIRECTION, NOT A VERDICT.** Its sibling convention reads 5/25 and the `incumbent` has no such
window at all. Per SPEC §3.6, a majority under **only one convention** and on **only one
checkpoint** is the pre-registered **FAILURE** condition, reported in the headline. This
reproduces M6 (exactly one configuration ranks a turn at shipped weights: L0 identity + chord +
steer units) on an independent decoder and **does not widen it**.

**The mechanism, ✅RECOMPUTED — and it is not resolution:**

* L0 (`goal_time_grid="plan"`) drives the canonical turn's goal term to **exactly 0.0 on 25/25**
  windows (vs **0/25** under `full`). The goal *is* the canonical turn on the plan grid: an
  **identity**, not a prediction. This is the whole of L0's contribution.
* The advantage `A` then equals the *cv* candidate's goal term: **7.63e-03** on the leading
  window under the chord, vs **2.91e-05** for the same window under `1−cos` (= chord²/2).
* The charge is **unchanged**: median `0.05·κ̄² = 3.20e-04`, median `0.02·j̄² = 0.0` (the
  canonical turn's jerk is zero), total **3.20e-04**.
* So the chord wins by **scale**, exactly the 5,793× implicit reweighting the SPEC declared in
  §1/L4 — which is why `chord` alone is a **regression arm**, not a candidate.

⚠️ **A reconciliation the register must settle before either number is quoted alone:** M4 records
the κ² charge as **2.24e-04**; measured here on this stratum it is **3.20e-04** (1.43× higher).
Both are MEASURED, from different packages. I do **not** claim M4 is wrong — I record the
discrepancy as **UNRESOLVED**.

### 6b. The geometric turn stratum — where the human actually turns (27 win / 10 ep, both ckpts)

**This is the question in the brief, and this is the answer.**

| cell | total-cost wins @ 0.05 — `ep2` | — `incumbent` | **argmin sign matches the human** @ 0.05 — `ep2` | — `incumbent` |
|---|---|---|---|---|
| `cos` — all four cells | **0 / 27** | **0 / 27** | **0 / 27** | **0 / 27** |
| `chord__A__full` | 0 / 27 | 0 / 27 | **0 / 27** | **0 / 27** |
| `chord__A__plan` | 1 / 27 | 1 / 27 | **5 / 27** | **0 / 27** |
| `chord__B__full` | 0 / 27 | 0 / 27 | **0 / 27** | **0 / 27** |
| **`chord__B__plan`** | **10 / 27** | **1 / 27** | **6 / 27** | **0 / 27** |

Majority = 14. **The best cell reaches 10.** And the sign of the turn — the literal content of
*"does it turn where the human turns"* — matches on **6 of 27 on one checkpoint and 0 of 27 on
the other.** Median goal advantage on this stratum is **0.0** in **every one of the 8 cells, on
both checkpoints**: on the windows where the car turns, the world model offers the cost **no
advantage at all** for turning.

**And the κ-argmin of the total cost barely moves:** `n_argmin_off_zero_at_shipped_w` over all
140 windows is **24/140** (`ep2` `chord__B__plan`), **19/140** (`ep2` `chord__A__plan`),
**0/140** everywhere else — and **0/140 in all 8 cells on the `incumbent`.** On the incumbent
checkpoint the planner's κ-argmin **never leaves zero, in any cell of the panel.**

⭐ **Why — and this is the result with the longest reach.** Under the chord the goal term's
**own** κ-argmin (before any penalty) sits at **κ = 0 on 110–130 of 140 windows** (`ep2` 110–130,
`incumbent` 129). Under `1−cos` it sits at zero on **0–11/140** — because there it is
quantisation noise, not a preference. **So the chord's real payload is not "the planner can now
turn"; it is "the planner's preference became legible, and it prefers straight on ~79–93 % of
windows."** No κ² weight repairs that. The defect has moved from the *cost's arithmetic* to the
*world model's goal field*.

**VERDICT S5: FAIL** — minority, convention-dependent, and checkpoint-dependent.

### 6c. Could the population be widened, and to what?

* **It already was**, as the SPEC required: the `gt_turn_ge_5deg` stratum (**27 / 10 / 3**) is a
  genuinely different and wider denominator than the curved-goal 25/4/1, and it is present on
  **both** checkpoints. It is where the FAIL above is read.
* **Not further, without new compute.** The eval slice holds **exactly 20 episodes**
  (`refav1_eval_slice/eps` = 20 files; `fp8` cache = 20). All 20 were used, at stride 10 →
  140/140 windows. **The panel is already at the slice's ceiling.** Widening means building a
  larger fp8 cache slice from the 2,376-episode AV set — GPU work, and ⛔ **not while `refav1`
  holds Thor** (to ≈ 2026-09-04 01:00Z).
* **The curved-goal stratum cannot be widened by more windows at all.** It is bounded by the
  **decoder**, which emitted a curvature-bearing token on 4 of 20 episodes (`ep2`) and **0 of 20**
  (`incumbent`). More episodes buy more windows of the same one token unless the decoder changes.
  ⇒ **`n_distinct_tokens = 1` is a decoder property, not a sampling accident.** Any future 25/25
  carries the same caveat until the decoder emits more than one curved token.

---

## 7. Committed criteria that could NOT be evaluated from the raw, and what is missing

| criterion | status | what is missing |
|---|---|---|
| **S3 second half** — `chord²/2 − (1−cos)` ≤ **1e-12** relative in f64 | **NOT MET** (measured median 3.9e-09 … 2.4e-07, max 1.35e+00; 2,893–2,935 of 2,940 cells over threshold) | Nothing is missing from the raw — the threshold is **unreachable by construction** (§4). Settling it needs the identity evaluated with `1−cos` computed in **f128 / exact** arithmetic, or the criterion restated against `chord` as the reference. ⚠️ **Not rescoped here.** |
| **S3 scope** — "over the full **189-cell** grid" | **REDUCED, not met as written** | The banked run has `accel_axis = [0.0]` — **21 cells** (the κ-marginal at a = 0), not 21 × 9 = 189. The `a` axis was never swept. A re-run with the full accel axis is required; the ordering result (0/235,200 discordant) stands on the 21-cell axis only. |
| **S1 / X5** — `plan(...)` vs `plan(..., cost_metric="cos")` return equal `controls`, `cost`, `source`, `baseline_costs`, `fine_costs` | **PARTIAL** | The panel JSON evidences only the **`_goal_term` bit-identity** half (`X2a_bit_identity`: all bit-identical, `max_abs_diff 0.0`, n=5) and the trajectory half (`X2b`: 0.0 m). The **`plan()`-level** equality is pinned by `stack/tests/test_cost_chord.py::test_c_the_default_changes_nothing` / `::test_d_the_flag_is_live_and_is_stamped_on_the_result`, i.e. by the test suite, **not** by this panel. §8 reports that suite's state. |
| **S5 on the `incumbent`** | **UNDECIDABLE** | The curved-goal stratum is **empty (n = 0)**. Not a fail; not a pass. Exactly as the SPEC predicted (§3.7.3). |
| **S4 on the `incumbent`** | **UNDECIDABLE** | Same cause — all derived weights are `null`; `w_derived_commensurable_median` is `nan` and `n_argmin_off_zero_at_derived_w = 0` is therefore **not evaluable**, not a measurement. |

Grading the SPEC's own registered prediction (§3.7), since it committed to being falsifiable:
**(1) S1/S3/S4 hold — CORRECT** (S3's ordering half; its 1e-12 half is NOT MET, §4).
**(2) factor far below 10⁷ — CORRECT for `n_distinct_f32` (2.3–7×), WRONG for `steps_true`
(10⁶–10¹⁰, above the predicted 10³–10⁶ band).**
**(3) S5 fails, convention-dependent, `incumbent` UNDECIDABLE — CORRECT on all three.**
**(4) tip is a deletion under `cos`, not under the chord — CORRECT** (20–25/25 vs 0/25).

---

## 8. AFTER test pass

Command, run in the mirror (`C:/Users/Admin/tanitad-wt`, venv
`C:\Users\Admin\venvs\tanitad\Scripts\python.exe`, `PYTHONPATH=…/stack;…/colab;…/taniteval`):

```
python -m pytest stack/tests -q -k "refa or refav1 or cost"
```

**AFTER: 1 failed, 379 passed, 5,476 deselected — 108.13 s.**

⚠️ **No BEFORE pass was run in this session** (the stalled agent left none, and reverting
`refa_v1.py` to HEAD to manufacture one would have meant editing a file while four other agents
are live). The single failure is therefore established as pre-existing **by construction and by
direct inspection**, not by a before/after diff — stated plainly rather than implied:

**The one failure, named:**

* **`stack/tests/test_decision_check.py::test_refa_surfaces_the_measured_arm`** —
  `AssertionError: REF-A must be findable — it is the measured frozen arm`, `assert []`.
  * It is selected by `-k` only because its **name** contains `refa`. It imports **`decision_check`
    and nothing else** — `grep` for `refa_v1`, `cost_metric`, `_cost_chunk`, `_goal_term`, `chord`
    in that file returns **zero hits**. It cannot observe this package's change.
  * Its own captured stdout names the cause: `DECISIONS_2026-07-20.md NOT FOUND`,
    `MODEL_REGISTRY.md NOT FOUND`, `RETRACTION_LOG.md NOT FOUND`. Verified on disk:
    in the **mirror**, all three are **ABSENT**; in the **repo**, `MODEL_REGISTRY.md` and
    `RETRACTION_LOG.md` are **PRESENT** and `DECISIONS_2026-07-20.md` is **ABSENT**.
    ⇒ the failure is a **record/mirror-sync gap**, not a code regression.

**The tests that do exercise the change are green:**
`stack/tests/test_cost_chord.py` alone → **18 passed in 1.68 s** (0 failed), including
`test_a_cos_branch_is_bit_identical_to_the_shipped_expression`,
`test_c_the_default_changes_nothing`, `test_e_chord_is_monotone_equivalent_in_float64`
(`assert disc == 0`), `test_f_the_deliberate_regression_recovers_nothing`,
`test_h2_the_cost_charges_the_named_W_KAPPA_and_nothing_else`,
`test_i_the_chord_is_not_weight_neutral`.

⇒ **No regression is attributable to this package.** The pre-existing failure is reported, not
absorbed, and it is a **documentation-record** finding in its own right: the decision record
`DECISIONS_2026-07-20.md` is missing from the repo, and the mirror carries neither the model
registry nor the retraction log.

---

## 9. PROPOSED REGISTER ROWS

> Proposed, not written. `Project Steering/MODEL_REGISTRY.md` / the claims register is the
> authority; these rows are submitted for adoption.

### `D-COST-CHORD` — decision/finding · **T0** · evidence class **MEASURED**

> The `cost_metric` flag (`"cos"` default, `"chord"` alternative) is implemented in
> `stack/tanitad/refs/refa_v1.py` `_cost_chunk` and is **default-safe**: `"cos"` is
> **bit-identical** to the shipped expression (`max_abs_diff 0.0`, n = 5) and reproduces the
> banked winner trajectory to **0.0 m** against a 3.8e-6 m gate, on **both** step-1,000
> checkpoints. The chord raises the goal term's κ-resolution by a median factor of
> **2.3×–7.0×** in `n_distinct_f32` (episode-clustered CI95 excludes 1 on 4/4 cells × 2
> checkpoints; **hard-capped at 21 by the grid**) and **1.9e6–1.7e10** in `steps_true`; the
> deliberate f32 regression `√(2(1−cos))` recovers **nothing** (0/140 windows). The two metrics
> are **exactly rank-equivalent**: **0 discordant pairs in 235,200** on the 21-point κ axis.
> ⛔ **The chord is NOT weight-neutral and must not become the default**: it is a **5,793×
> implicit reweighting** against an unchanged `0.05·κ²`.
> ⚠️ **Corrects the inherited `~2 → ~10⁷` claim** — that factor belongs to `steps_true`, not to
> the planner-visible `n_distinct_f32`, which measures **2.3×–7×**. → RETRACTION_LOG.
> ⚠️ Scope: `accel_axis = [0.0]`; the 189-cell grid of the SPEC was **not** swept.
> **Source:** `…/2026-09-03-cost-repair/raw/cost_repair_{ep2,incumbent}{,_rows}.json`;
> `stack/tests/test_cost_chord.py`. **n = 140 windows / 20 episodes per checkpoint; d = 65,536
> — underpowered by construction. T0: no driving claim without the T1 adapter.**

### `H-COST-WEIGHTS-1` — hypothesis · **T0** · **PARTIALLY SUPPORTED / S5 REFUTED**

> **SUPPORTED (S4):** with the chord in place a commensurable, non-deleting κ² weight exists —
> `w_κ^tip = 3.07e-03` (`chord__B__plan`) and `1.02e-03` (`chord__A__plan`), a deletion on
> **0/25** windows; `w_κ^comm = 2.77e-02`, within 2× of the shipped 0.05. Under the shipped
> `1−cos` the tipping weight is **6.6e-08 / 6.1e-07** and is a **deletion on 20–25 of 25** —
> confirming **M5** (`1.66e-08`, "a deletion not a weight"). On **4 of 8 cells**
> `w_κ^κ-only < 0`: **no κ² weight whatsoever ranks the turn** there.
> **REFUTED (S5):** the joint change does **not** rank the turn on a majority under both
> conventions and both checkpoints. On the geometric turn stratum (**27 win / 10 ep**, both
> checkpoints) the best cell wins **10/27** (`ep2`) and **1/27** (`incumbent`), and its argmin
> matches the human's turn **sign** on **6/27** and **0/27**. The **25/25** on the decoder's
> curved-goal stratum is **25 windows / 4 episodes / ONE token, one checkpoint, one convention
> — a DIRECTION, NOT A VERDICT** (sibling convention 5/25; `incumbent` stratum empty ⇒
> **UNDECIDABLE**). Its mechanism is L0 forcing the canonical goal term to **exactly 0.0 on
> 25/25** plus the chord's scale — **not** new curvature sensitivity.
> ⭐ **Consequence:** under the chord the goal term's own κ-argmin is at **κ = 0 on 110–130 of
> 140** windows. **The world model prefers straight; a cost repair cannot fix that.** Per the
> SPEC's own refutation clause, the programme goes to the **goal SPACE** branch
> (`PREREG_TACTICAL_DECODER.md` §10).
> ⚠️ **UNRESOLVED:** the κ² charge reads **3.20e-04** here vs **2.24e-04** in M4 (1.43×).
> **Escalated, not silently reconciled.**
