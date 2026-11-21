# SPEC — L3 + L4: the CHORD metric and the weight it forces, pre-registered together

**Date:** 2026-09-03 (Europe/Berlin) · **FlyWheel:** Architecture & Inference ·
**Branch:** `agent/arch-inf-20260803` · **Status:** written **BEFORE any edit to `_cost_chunk`
and BEFORE any measurement**. Nothing in §4–§6 had been run when this file was written.
**Governing pre-registration:** `Project Steering/PREREG_TACTICAL_DECODER.md` §5 (L3 + L4,
*"jointly, never L3 alone"*), §7 S3, §8.2 Q2.
**Validation contract:** `.claude/skills/TanitAD_ValidateAIDesign` as CORRECTED 2026-09-03.
**Owner files:** `stack/tanitad/refs/refa_v1.py` (the COST only — `_cost_chunk` and its
weights), NEW `stack/tests/test_cost_chord.py`, this package.
**Read-only:** `taniteval/tools/cost_surface_probe.py`, `…/2026-09-03-refav1-cost-surface/`,
`…/2026-09-03-tacdec-L0-L1-E0/`, `stack/tanitad/refs/refa_v1_plan.py`.

> ⚠️ **THE RETIRED FLOOR.** No arm here is passed or failed on `participation >= 8.56`.
> That floor is retired; no live instrument reproduces it. This package changes a **cost**,
> not a representation, so participation is not a gate and is not quoted. Absent a matched
> reference the honest reading is **UNDECIDABLE — neither pass nor fail**.

> ⚠️ **BOTH OUTCOMES ARE COMMITTED IN ADVANCE.** §3 names, with thresholds, what counts as
> success and what refutes each claim, and §3.7 records my prediction so it is falsifiable.

---

## 0. What is already MEASURED and is NOT re-litigated here

Cited, not re-derived. The registry and the raw JSON win over any prose.

| # | fact | source (primary) |
|---|---|---|
| M1 | the `0.05·κ²` penalty is **99.5 %** of all cost variation over the candidate box; the world model contributes **1.63e-10** along κ | `D-REFAV1-COST-SURFACE`; `…/2026-09-03-refav1-cost-surface/raw/cost_surface_{ep2,incumbent}.json` |
| M2 | `1−cos` spans a median **2.00 ULPs along κ** vs **167.5 / 19,804** along `a`; the shipped argmin is `(a=0, κ=0)` on **139/140** and the origin is the global minimum on **140/140** | same, rows `*_goal_kappa_range_in_ulps` |
| M3 | against the ORACLE 6 s field the canonical turn is FURTHER than constant velocity on **22/25** (conv A) / **24/25** (conv B) ⇒ **the defect is the cost** | `D-TACDEC-L0-L1-E0`; `…/2026-09-03-tacdec-L0-L1-E0/raw/e0_goalspace_ep2.json` |
| M4 | the goal-term advantage's median is **1.49e-09** against a κ² charge of **2.24e-04** — short by **1.5e5**; the turn's TOTAL cost wins **0/25** under shipped `1−cos` on every arm and both conventions | same |
| M5 | the tipping κ² weight under the shipped metric is **1.66e-08** | same, `penalty_scale_that_would_tip_it__1mcos` |
| M6 | exactly one configuration ranks the turn at shipped weights: **L0 identity (`goal_time_grid="plan"`) + chord + convention B = 25/25**; convention A gives **5/25**; margins **1.19–1.23×** on three of four episodes; **n = 25 windows / 4 episodes / ONE token** | same |
| M7 | `model_action_units="steer"` wins nothing under the SHIPPED cost (`D-REFAV1-BOUNDARY-NULL`) and is NECESSARY under the only cost that ranks a turn | registry row + M6 |
| M8 | the candidate box is `κ ∈ [−0.2, +0.2]`, `a ∈ [−4, +4]` (`PlanConfig.kappa_max/a_max`); the goal space is `[64, 1024]` = **65,536** dims | `refa_v1_plan.py:87-88`; `cost_surface_ep2.json` `grid`/`model.cfg` |
| M9 | **no trainer calls `plan()`** — verified here with two differently-bound probes (POSIX `grep -c "\.plan("` on `stack/scripts/refa_v1_train.py` → **0**, exit 1; CPython absolute-path `read().count()` → `.plan(` **0**, `plan_cfg` **0**, `icem` **0**). ⇒ a `plan()` flag cannot perturb the live Thor run. | this package, §5 |

---

## 1. The change under test

**L3 — the metric.** `_cost_chunk`'s goal term is
`1 − cosine_similarity(ẑ, ĝ)`. The alternative is the **chord**
`‖ẑ/‖ẑ‖ − ĝ/‖ĝ‖‖₂`. On normalised vectors `chord = √(2(1−cos))`, so the two are
**monotone-equivalent** and cannot change the goal term's own ranking. What changes is the
**arithmetic**: `1 − cos` is a catastrophic cancellation against 1.0 and inherits the
absolute quantum `spacing(1f)/2 = 5.9604645e-08` whatever its value; the chord is a norm of
a difference computed at the difference's own scale.

**L4 — the weight.** `PREREG_TACTICAL_DECODER.md` §5 is explicit: *"Swapping the metric while
holding `0.05` fixed is therefore **not a one-variable arm** — it is a metric change **and** an
implicit 5,793× reweighting."* So the metric flag and a weight statement land together.

**Implementation shape (declared before writing it):** a module-level `COST_METRICS` tuple +
`_check_cost_metric` + a module-level `_goal_term(zt, g, metric)` called at the ONE site
inside `_cost_chunk`, plus a `plan()` keyword `cost_metric: str = "cos"` and provenance
`res.cost_metric` — the exact pattern `cost_time_grid` / `goal_time_grid` already use.
⛔ **No `RefAV1Config` field** (that would change every serialised config while a run is live)
and **no weight default is changed in this package**; the weights are *derived and reported*,
and any change to `0.05` is a PI / Master-Mind decision, escalated, not taken.

---

## 2. Hypotheses (IDs proposed with this package)

### `H-COST-CHORD-1`
> The chord is not a cosmetic reformulation of `1 − cos`: computed directly as a norm it
> makes the goal term's κ-response **resolvable in float32**, where `1 − cos` quantises it
> away. **Therefore:** over the planner's own candidate box, on the real latents of the two
> banked step-1,000 checkpoints, the number of **distinct representable float32 values** the
> goal term takes along κ rises by a factor whose episode-clustered 95 % interval **excludes
> 1**, while the two metrics' **f64 rankings agree exactly** on every window.

### `H-COST-WEIGHTS-1`
> With the chord in place there exists a κ² weight that is **commensurable** — the goal term
> and the penalty have comparable dynamic range over the candidate box — and that is **not a
> deletion in disguise** (§3.5). **Therefore:** at that weight the canonical turn's TOTAL
> cost beats constant velocity on a **majority** of curved-goal windows under **both** unit
> conventions and **both** `goal_time_grid` settings. **Refutation:** every weight that ranks
> the turn on a majority under both conventions satisfies the deletion test ⇒ the cost cannot
> be repaired by a weight and the goal SPACE branch (`PREREG_TACTICAL_DECODER.md` §10) is
> where the programme goes next.

---

## 3. The pre-registration block

```yaml
hypothesis:   [H-COST-CHORD-1, H-COST-WEIGHTS-1]
one_variable: cost_metric                    # C0 vs C2; the ONLY difference
              # ⚠️ and it is NOT one-variable against the shipped cost unless the
              # weight is declared with it -- which is why L4 is in the same SPEC.
held_constant: [checkpoint, windows, episodes, nav, labels, plan_cfg.seed,
                plan_cfg.n_samples, plan_cfg.n_iters, plan_cfg.n_elites,
                candidate_box, w_jerk, w_vend, cost_time_grid, wheelbase,
                predictor, goal_source, dtype_of_the_rollout]
success: "see S1-S6 below, each with its threshold"
failure:  "see the FAILURE column of S1-S6; any one of them firing is reported
           in the HEADLINE, not in a footnote"
controls: [X0_zero_model, X1_constant_cost, X2_harness_gate_banked_winners,
           X3_deliberate_regression_sqrt_of_1mcos_in_f32,
           X4_monotone_identity_in_f64, X5_default_is_byte_identical,
           X6_n_printed_every_time]
splits:   {fit: "none -- no parameter is fit on any scored quantity",
           val: "none",
           test: "the 140 banked windows / 20 episodes on BOTH banked
                  step-1,000 checkpoints, scored once"}
tier:     T0                # a COST/WM diagnostic anchored at t0.
                            # turn_frac is a property of the COST, never of the
                            # car. NO driving claim without the T1 adapter.
```

### 3.1 The arms

| id | goal metric | `goal_time_grid` | units | note |
|---|---|---|---|---|
| **C0** | `cos` (shipped) | `full` | A (`kappa`) | the baseline; must be byte-identical to HEAD |
| **C1** | `cos` | `plan` (L0) | A / B | L0 alone |
| **C2** | `chord` | `full` | A / B | **the metric alone — declared a REGRESSION arm**, because it silently reweights |
| **C3** | `chord` | `plan` (L0) | A / B | the joint candidate |
| **X3** | `√(2·(1−cos))` computed in **f32** | either | either | ⭐ the deliberate regression for `H-COST-CHORD-1` |
| **X0** | either, **zero-model predictor** | `full` | A | the penalty, isolated exactly |
| **X1** | goal term dropped, all weights 0 | — | A | the constant-cost control |

The full panel is the **2 × 2 of {cos, chord} × {κ, steer}, with and without L0** = 8 cells,
on **both** checkpoints, over **140** windows each.

### 3.2 S1 — the default does not move (`X5`)

**SUCCESS:** `plan(...)` and `plan(..., cost_metric="cos")` return **equal** `controls`,
`cost`, `source`, `baseline_costs` and `fine_costs`; and `_goal_term(zt, g, "cos")` is
**bit-identical** to `1.0 − F.cosine_similarity(zt.flatten(1), g.flatten(1), dim=-1)` on
random and on real fields.
**FAILURE:** any difference at all. Then the flag is not default-safe and nothing else in
this document may be reported.

### 3.3 S2 — the representable-step claim, PROVED HERE, not inherited

⛔ The `~2 → ~10⁷` figure in my brief is **INHERITED** and is treated as unproven until
measured. The measurement is defined **before** it is taken:

For each window, on the planner's own candidate box, `a = 0`, κ swept over the **21-point**
axis the banked surface used (`build_grid`, zero forced exactly on-grid), on the **real
latents**:

* `n_distinct_f32(m)` — the number of **distinct float32 values** the metric takes over the
  21 κ cells. This is the operational meaning of *"representable steps"*: how many values the
  planner can actually tell apart.
* `steps_f32(m)` = `ptp_f32(m) / q_f32(m)`, where `q_f32(m)` is the **realised quantum** —
  the smallest non-zero absolute difference between two f32 values on the κ axis (0 if none).
* `steps_true(m)` = `ptp_f64(m) / spacing32(median_f32(m))` — how many float32 steps the
  **true** κ-variation spans at that metric's own operating point. This is the number that
  says whether f32 *could* see the response.

**SUCCESS:** median over windows of `n_distinct_f32(chord) / n_distinct_f32(cos)` **> 1**
with an episode-clustered 95 % interval excluding 1, on **both** checkpoints and **both**
conventions.
**FAILURE:** the interval includes 1 ⇒ **`H-COST-CHORD-1` is REFUTED**; the chord buys no
resolution on the real latents and R29 / L3 must be withdrawn.
⚠️ **The measured factor is reported as measured.** If it is 10² and not 10⁷ I say so, in
the headline, and the brief's figure is sent to the RETRACTION_LOG.

### 3.4 S3 — monotone equivalence (`X4`)

**SUCCESS:** over the full 189-cell grid, in **f64**, `chord` and `1−cos` induce the
**identical** ordering (Spearman ρ = 1.0 exactly, 0 discordant pairs) on **every** window;
and `chord² / 2 − (1−cos)` is ≤ 1e-12 relative in f64.
**FAILURE:** any discordant pair ⇒ the implementation is not the chord and S2 is void.

### 3.5 S4 — the weights, DERIVED, with a deletion test that is not a matter of taste

Three weights, each defined before it is measured:

1. **`w_κ^tip`** — the *decision-level* weight: the multiplier `s` on **both** explicit
   weights at which the canonical turn's total cost exactly equals constant velocity's, i.e.
   `s* = A / (κ̄²_turn + j̄²_turn)` with `A` the goal-term advantage in f64;
   `w_κ^tip = 0.05 · median(s*)`. Reported **per cell of the 2 × 2 × L0 panel**.
2. **`w_κ^comm`** — the *commensurability* weight: `Δ_goal / κ_max²`, where `Δ_goal` is the
   goal term's peak-to-peak along κ over the candidate box **at the same metric**. At this
   weight the two terms have equal dynamic range over the box — the property the shipped
   cost lacks (M1: 99.5 % penalty).
3. **`w_κ^κ-only`** — holding `w_jerk = 0.02` fixed: `(A − 0.02·j̄²_turn) / κ̄²_turn`.
   ⚠️ This can be **negative**; if it is, the jerk charge alone already exceeds the goal
   advantage and **no κ² weight whatsoever ranks the turn**. That outcome must be stated
   plainly, not buried.

**THE DELETION TEST (pre-registered, measurable, not a matter of taste).** A weight `w` is a
**deletion in disguise** iff

> `w · κ_max²  <  q(m)` ,  where `q(m)` is the metric's own realised float32 quantum at the
> window's operating point (§3.3).

Rationale: below that, the entire penalty over the *whole* candidate box is smaller than one
representable step of the term it is supposed to trade against — the argmin is then decided
by rounding, not by the penalty, and the cost surface is numerically indistinguishable from
one with `w = 0`. A second, cruder ratio `w / 0.05` is reported beside it.
**SUCCESS:** at least one cell of the panel yields a `w_κ^tip` that is **NOT** a deletion.
**FAILURE:** every cell's tipping weight is a deletion ⇒ **`H-COST-WEIGHTS-1` is REFUTED**.

### 3.6 S5 — does the joint change rank a turn, and on how many episodes and tokens

**SUCCESS:** at the **shipped** `w_κ = 0.05`, the turn's TOTAL cost beats cv on a **majority**
of the curved-goal windows under **both** conventions **and** on **both** checkpoints, with
the count reported against its true n (windows / **episodes** / **distinct tokens**).
**FAILURE:** a majority under only one convention, or on only one checkpoint ⇒ the joint
change is **convention-dependent or checkpoint-dependent** and is reported as such in the
headline — exactly the failure mode M6 already shows for the metric alone.

⭐ **POPULATION WIDENING IS PART OF THE SPEC, NOT AN OPTIONAL EXTRA.** M6's 25/25 is
**4 episodes and ONE token**. This package must report, for every count:
`n_windows / n_episodes / n_distinct_(lat,lon)_tokens`, and must additionally score

* the **incumbent** checkpoint's curved-goal stratum (a second, independent decoder), and
* the **geometric turn stratum** — windows with `gt_turn_deg ≥ 5°` from the banked cost
  surface — which is the population the *human* turns on and is a **different denominator**;
  no count mixes the two.

⛔ **A 25/25 on 4 clusters and one token is a DIRECTION, not a verdict**, and every table
that carries it carries that sentence.

### 3.7 S6 — the controls must read their known values (a failure VOIDS the panel)

| id | control | the KNOWN value it must read |
|---|---|---|
| **X0** | **zero-model**: the predictor is fed the zero action for every candidate, so it cannot see κ | goal-term peak-to-peak along κ **EXACTLY 0.0** under **both** metrics; `total(κ) − total(0)` **exactly** `0.05·κ²`; argmin at κ = 0. **This isolates the penalty exactly.** |
| **X1** | **constant-cost**: goal term dropped, all weights 0 | `c_total ≡ 0.0` everywhere; argmin = the first grid index (deterministic tie) |
| **X2** | **harness gate — the banked winners**, and it runs BEFORE any conclusion | (a) my re-scoring of the named baselines equals the SHIPPED `PlanResult.baseline_costs` to ≤ 1e-6 relative, with `cost_metric="cos"`; (b) a real `plan()` re-run reproduces the banked `cl` trajectory to **< 3.8e-6 m** max abs — the precedent gate |
| **X3** | **deliberate regression**: the chord computed as `√(2·(1−cos))` **in f32** | must recover **NO** resolution — `n_distinct_f32` equal to the cos arm's. If X3 *does* recover resolution, the gate cannot see the defect it exists to catch and **no S2 pass may be reported.** |
| **X4** | monotone identity in f64 | §3.4 |
| **X5** | the default | §3.2 |
| **X6** | n and d printed beside every number | `n ≪ d` (65,536) named as underpowered BY CONSTRUCTION where it applies |

### 3.8 My registered prediction, so it is falsifiable

1. **S1, S3, S4 will hold.** The chord is monotone-equivalent by algebra and the flag is a
   two-line branch.
2. **S2 will hold, but the factor will be far below 10⁷** — I predict a median
   `n_distinct_f32` gain in the **10⁰–10²** range on the *shipped* (`goal_time_grid="full"`)
   arm, because `n_distinct` is capped at 21 by the grid itself, and a `steps_true` gain in
   the **10³–10⁶** range. **I predict the inherited `~10⁷` is NOT reproduced** and will need a
   correction rather than a citation.
3. **S5 will FAIL as written** — I expect the majority to hold under convention B and not
   under A, reproducing M6's convention-dependence on the ep2 checkpoint, and I expect the
   **incumbent** checkpoint to have **no** curved-goal stratum at all (M6's sibling package
   measured `LANE_KEEP` on 140/140 for the incumbent), which would make S5 **UNDECIDABLE on
   that checkpoint rather than failed** — and I will say `UNDECIDABLE`, not `pass`.
4. **The `w_κ^tip` under `cos` will be a deletion; under the chord it will not.**

If any of these is wrong I say so, in the headline, and the RETRACTION_LOG gets the row.

---

## 4. What this package does NOT claim, stated in advance

* **Not a driving claim.** Tier **T0**. `turn_frac` and every win-count here are properties of
  the COST evaluated on banked latents. No four-metric-family panel is reported because no arm
  reaches G-DRIVE (`PREREG_TACTICAL_DECODER.md` §6d).
* **Not a claim that the WM's turn is correct** — M3's oracle floor bounds *"can this cost
  rank a turn against the truth"*, not *"is the imagined turn right"*.
* **Not a weight change.** The weights are derived and reported. Changing `0.05` in the
  shipped default is escalated.
* **Not a horizon change.** `plan_horizon_s` stays 2.0; L0 buys identity, not horizon.

## 5. Refusals recorded in advance

* ⛔ **Thor is NOT contacted** (`refav1` speed epoch to ≈ 2026-09-04 02:00Z) and **no pod is
  contacted** (refcv3 on the A40 to ≈ 22:15Z). The dev-box 4060 is checked for another
  python compute app before every run and yields if one appears; anything over 55 min is
  DETACHED via `powershell.exe Start-Process`.
* ⛔ `taniteval/tools/cost_surface_probe.py` is **imported, never modified**; its md5 is
  banked as provenance and its `WindowContext` (C1-gated against the shipped cost) is what
  makes a re-scoring admissible at all.
* ⛔ The two source packages are **read-only**.
* ⛔ No commit, no push, no branch switch; every deliverable is **staged**.
* ⛔ The retired `participation >= 8.56` floor is not used, quoted, or reintroduced.
