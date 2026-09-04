# PROPOSED REGISTER ROWS — L3 (the chord) + L4 (the weight)

⛔ **PROPOSED, not applied.** This FlyWheel does not own `Project Steering/GOALS_AND_CLAIMS.md`,
`MODEL_REGISTRY.md` or `RETRACTION_LOG.md`. Owner: the **Master Mind**.
Every number carries its evidence class, its tier and its artifact path, and every one is
reproducible from `./raw/` at **0 GPU** via `tools/cost_repair_probe.py --from-rows`.

---

## 1. `D-COST-CHORD` — the goal metric is flag-gated, and the chord resolves what the cosine quantises away

| field | value |
|---|---|
| **id** | `D-COST-CHORD` |
| **status** | **LANDED, DEFAULT OFF** (staged, not committed; `agent/arch-inf-20260803`) |
| **class** | planner / cost |
| **claim** | `RefAV1.plan(cost_metric="chord")` computes the goal term as `‖ẑ − ĝ‖` directly instead of `1 − cos(ẑ, ĝ)`. The two are monotone-equivalent (`chord = √(2(1−cos))`) and provably cannot re-rank in exact arithmetic; in float32 on the real latents the chord separates **21 / 21** cells of the planner's κ axis on **140/140 windows of both banked checkpoints**, where `1 − cos` separates a median of **3**. `"cos"` remains the default and is byte-identical to the shipped code. |
| **the defect it repairs, MEASURED here** | `1 − cos` is `1.0 − c` with `c ∈ [0.5, 1)`, so it is an integer multiple of `2⁻²⁴`. Its realised float32 quantum is **exactly 5.9604645e-08 on every one of 280 window-reads** (140 × 2 checkpoints, both conventions) — it does **not** shrink with the term's value. The true κ-response over the whole candidate box is **1.627e-10** (incumbent) / **5.816e-08** (ep2), i.e. **1/366** of one representable step on the incumbent. |
| **the size of the repair** | `steps_true` (true f64 κ-response ÷ the metric's delivered f32 resolution): `1 − cos` **0.0027 – 9.36**; `chord` **1.83e+07 – 2.67e+07**. Paired, episode-clustered (20 clusters): **×2.37e+07** [2.34e+07, 2.41e+07] on ep2·A and **×7.25e+09** [6.56e+09, 1.72e+10] on incumbent·A. `n_distinct_f32` ratio **×7.00** [7.00, 7.00]. Every interval excludes 1. |
| **⚠️ the correction that must travel with it** | The brief's *"~2 → ~10⁷"* is **two different statistics** and **must not be quoted as a ratio**. The "~2" is the f32-*observed* spread in quanta (banked `*_goal_kappa_range_in_ulps`); the "10⁷" is the chord's `steps_true`. Their quotient (≈ 5e+06) is neither quantity. The honest single number is the **paired ratio** above. |
| **⭐ and it is NOT weight-neutral** | the chord multiplies the goal term by `1/chord` against an unchanged `W_KAPPA` (banked **5,792.6×**). Flipping it alone is a metric change **AND** an implicit re-weighting — `PREREG_TACTICAL_DECODER.md` §5 L3. **It may not ship without `H-COST-WEIGHTS-1`'s weight statement.** |
| **evidence** | MEASURED, **T0**. `raw/cost_repair_ep2.json`, `raw/cost_repair_incumbent.json` (→ `cells.*`), per-window rows in `raw/cost_repair_*_rows.json`. Controls in §3 of `RESULT.md`, all ✅. |
| **pinned by** | `stack/tests/test_cost_chord.py` — **18 tests**, including the deliberate regression `test_f_*` (`√(2(1−cos))` in f32 recovers **no** ordering), `test_e2_*` (over 8 draws at the real goal dimension 65,536: chord **120/120** concordant pairs on every draw, cosine **5–78** of 120 with chance at 60), `test_c_*` (the default is indistinguishable), and `test_h2_*` (`W_KAPPA` is what `_cost_chunk` charges, verified by a 10× perturbation that leaves every zero-curvature baseline EXACTLY unchanged). |
| **files** | `stack/tanitad/refs/refa_v1.py` — `COST_METRICS`, `_check_cost_metric`, **`_goal_term`** (the ONE site the goal distance is computed), `plan(..., cost_metric="cos")`, `res.cost_metric`. |
| **⛔ not shipped** | Thor's live `refav1-b1-v72-ep3-speed` keeps its launch line. `plan()` is called by **no trainer** — verified with two differently-bound probes on `refa_v1_train.py` (`.plan(` **0**, `plan_cfg` **0**, `icem` **0**). |

---

## 2. `H-COST-WEIGHTS-1` — the weight the chord forces, and the deletion test that names the alternative

| field | value |
|---|---|
| **id** | `H-COST-WEIGHTS-1` |
| **status** | **OPEN → SUPPORTED at T0** on one checkpoint, one token; **UNDECIDABLE** on the other checkpoint |
| **hypothesis** | with the chord in place there exists a κ² weight that is *commensurable* over the candidate box and is **not a deletion in disguise**, at which the canonical turn's total cost beats constant velocity on a majority of curved-goal windows under both conventions and both `goal_time_grid` settings. |
| **the deletion test (pre-registered, measurable)** | `w` is a **DELETION** iff `w · κ_max² < q(m)`, the metric's realised float32 quantum: the whole penalty over the whole box is then smaller than **one representable step** of the term it trades against, and the argmin is decided by rounding. Thresholds MEASURED: `cos` **1.49e-06**; `chord` **4.93e-05** (conservative — grid-limited). |
| **⛔ the verdict on the SHIPPED metric** | the best `1 − cos` cell's tipping weight is **6.07e-07** — **below the 1.49e-06 threshold**, a deletion on **20 of 25** windows. Under `goal_time_grid="full"` it is **NEGATIVE** on both conventions: **no κ² weight whatsoever ranks the turn**, because the `0.02·jerk²` charge alone exceeds the goal advantage. This independently confirms the banked **1.66e-08** (`D-TACDEC-L0-L1-E0`), whose `w·κ_max² = 6.6e-10` is **90×** below one representable step. **⇒ under the shipped metric the answer is a DELETION, not a weight.** |
| **⭐ the verdict under the chord** | `w_κ^comm = 0.0277` against the shipped **0.05** — a factor **1.8**. Under `1 − cos` the same quantity is **1.57e-05**, a factor **3,185**. `w_κ^κ-only = 0.0616 > 0.05`, so **the turn wins at the shipped weight and no weight change is required to land L3**. Deletions **0 / 25**. |
| **the weight statement L4 owes, in one line** | **`W_KAPPA` stays at 0.05; the commensurable value is 0.0277 and the admissible band is [0.0, 0.0616]. Any number inside it is a PI decision, escalated, not taken here.** |
| **⇒ the quantitative content of "99.5 % of all cost variation is the penalty"** | the **3,185× vs 1.8×** gap is a property of the **METRIC**, not of the weight. `D-REFAV1-COST-SURFACE`'s headline is a metric finding that was read as a weight finding. |
| **the outcome vs the pre-registration** | `SPEC.md` S5 requires a majority under **both** conventions. Convention B gives **25/25**, convention A gives **5/25** ⇒ **S5 FAILS AS WRITTEN, convention-dependently** — reproducing `D-TACDEC-L0-L1-E0` S3 exactly, but now through the **shipped `_cost_chunk`** rather than a post-hoc `√(2·cos_f64)`. That is a cross-validation of that package by a different instrument. |
| **⚠️ the n, which must be quoted with every count** | **25 windows = 4 EPISODES = ONE token** (`TURN_R × ADAPT_SPEED_FOR_CURVE`). A 25/25 on 4 clusters and one token is a **DIRECTION, NOT A VERDICT**. |
| **the incumbent** | **UNDECIDABLE, not failed**: its lateral head emits `LANE_KEEP` on **140/140**, so there are **0** curved-goal windows and the question has no population there. |
| **evidence** | MEASURED, **T0**. `raw/cost_repair_ep2.json` → `weights.*`; `RESULT.md` §4. |

---

## 3. `D-COST-ARGMIN-MOVES` — the first refav1 cost surface whose minimum is not "go straight", and the defect it hands on

| field | value |
|---|---|
| **id** | `D-COST-ARGMIN-MOVES` |
| **status** | **MEASURED, decided** |
| **claim** | at the **SHIPPED** `W_KAPPA = 0.05`, the κ-marginal argmin of the total cost leaves κ = 0 on **24 / 140** windows under `chord + goal_time_grid="plan" + units="steer"`, and on **0 / 140** under every other cell of the 2 × 2 × {L0 on/off} panel — including the chord **without** L0. Baseline: the shipped argmin is `(a = 0, κ = 0)` on **139/140** (`D-REFAV1-COST-SURFACE`). |
| **⛔ AND THE NEGATIVE THAT DECIDES WHAT IS NEXT** | on the **27 windows where the car actually turns ≥ 5°** (**10 episodes, 3 tokens** — a wider population than the 25 and a **different denominator**), the repaired cost points the human's way on **6 / 27 ≈ 22 %**, at or below the ~1/3 chance rate for a three-way sign. |
| **the decomposition, per window** | 17/27 the decoder emitted an all-zero canonical control (`goal ≡ cv` identically — **the cost has nothing to rank**); 1/27 `LANE_KEEP×ADAPT_SPEED_FOR_CURVE`, ranked but straight; **9/27 the decoder emitted `TURN_R` and the cost's argmin left zero on 9 / 9** (κ = −0.04 / −0.06); of those 9 the human turned **LEFT** on **3** — and the head emits `TURN_L` **0 times in 140 windows**. |
| **⇒ the reading** | **L3 + L4 removes the COST as the binding constraint and hands the problem to L2** (`H-REFAV1-TAC-DECODER-1`, the decoder decision rule, still unrun). Where the decoder asks for a turn, the repaired cost now grants it 9/9; it asks on 9 of 27 and asks wrongly on 3 of those 9. **The next arm is the decoder, not another cost arm — and this package's own evidence is what says so.** |
| **tier** | **T0.** `turn_frac`, win counts and argmins are properties of the COST. **No driving claim; none is possible without the T1 adapter.** |
| **evidence** | MEASURED. `raw/cost_repair_ep2.json` → `argmin.by_cell`, `strata.by_cell`; per-window detail in `raw/cost_repair_ep2_rows.json`. |

---

## 4. `D-COST-SURFACE-REPRODUCED` — an independent instrument reproduces `D-REFAV1-COST-SURFACE`

| field | value |
|---|---|
| **id** | `D-COST-SURFACE-REPRODUCED` |
| **status** | **MEASURED** (proposed as a confirmation row, not a new finding) |
| **claim** | a second tool, on the same 140 windows / 20 episodes of both banked checkpoints, reproduces the banked κ-marginal reads: `steps_f32` median **2 / 9 / 2 / 2** against the banked `*_goal_kappa_range_in_ulps` **2.0000 / 9.0000 / 2.0000 / 2.0000** (ep2·A, ep2·B, incumbent·A, incumbent·B), and the f64 κ-response to the quoted digits (**1.627e-10** incumbent·A, **5.816e-08** ep2·A, **5.411e-07** ep2·B, **1.455e-09** incumbent·B). |
| **why it counts** | `taniteval/tools/cost_surface_probe.py` was **imported, never modified**, and its md5 at run time (`4c5dc3d57bf2cb06376fb788ec0cab3e`) is **identical to the md5 recorded inside the banked `cost_surface_ep2.json`**. The harness that produced the banked panel and the harness that reproduced it are the same bytes; the *metric evaluation* is this package's and is independently gated bit-identical to it. |
| **the harness gates, run BEFORE any conclusion** | C1-style shipped-cost gate: worst relative error **0.0** (exact), 5/5 windows, both checkpoints. Bit-identity of the rollout+metric: max abs diff **0.0**, 5/5. **Banked-winner gate at the precedent's own 3.8e-6 m: max abs 0.0 m, 5/5, both checkpoints.** |
| **evidence** | `raw/cost_repair_*.json` → `controls.X2a_*`, `controls.X2b_banked_winner`, `meta.source_md5`. |

---

## 5. For the RETRACTION_LOG — a prediction I registered and got wrong

| field | value |
|---|---|
| **what** | `SPEC.md` §3.8 #2, registered **before** measuring: *"I predict the inherited `~10⁷` is NOT reproduced"*, with `steps_true` expected in **10³–10⁶**. |
| **MEASURED** | `steps_true` for the chord is **1.83e+07 – 2.67e+07**, and the paired gain on the incumbent is **7.25e+09** — the figure reproduced, and on one checkpoint exceeded by two orders. |
| **status** | **the PREDICTION is retracted, not a measurement.** The `n_distinct` half of the same prediction (10⁰–10², capped by the 21-point grid) held at ×7.00. Recorded here rather than quietly folded in, per `PREREG_TACTICAL_DECODER.md` §7's precedent. |

---

## 6. Qualifiers other rows now need (Master Mind to apply)

* **`D-REFAV1-BOUNDARY-NULL`** (*"`model_action_units="steer"` removes a trap, wins nothing"*) needs
  a **qualifier, not a retraction**: it wins nothing under the shipped `1 − cos`, and it is
  **NECESSARY** under the only metric that ranks a turn — the 25/25 and the 24/140 argmin movement
  exist **only** under convention B.
* **`D-REFAV1-COST-SURFACE`**'s *"99.5 % of all cost variation is the penalty"* should carry the
  measurement that it is a **METRIC** property: the commensurable κ² weight is **1.57e-05** under
  `1 − cos` and **0.0277** under the chord, against a shipped 0.05 — **3,185×** vs **1.8×**.
* **`D-SEED-GOAL-FIXED`** (L0) should record that it is **inside** the weight, not beside it: every
  `goal_time_grid="full"` cell has a **negative** `w_κ^κ-only`, i.e. the chord **without** L0 ranks
  the turn **0/25** and no κ² weight exists that would change that.
