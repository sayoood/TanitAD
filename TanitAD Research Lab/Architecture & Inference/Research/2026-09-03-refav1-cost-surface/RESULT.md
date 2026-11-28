# RESULT — R10 · the refav1 planner COST SURFACE, decomposed and attributed

**Tier T0** (a cost / world-model diagnostic anchored at t0). `turn_frac` below is a
property of the COST, never of the car; **no driving claim is made here** — that needs
the T1 adapter (`taniteval/tools/refav1_arm.py`).
**Device:** dev-box RTX 4060 (8,188 MiB), checked free of other python compute apps
before and during. **Thor and the A40 were not contacted.**
**Population:** the SAME 140 windows / 20 episodes / stride 10 as both banked T1 reads,
on BOTH banked step-1,000 checkpoints (incumbent `ckpt/ckpt.pt`; clean epoch
`ckpt_ep2/ckpt.pt` + its `config.json`).
**Grid:** 21 κ × 9 a = 189 constant-over-horizon candidates at the planner's own `H = 10`
(`kappa_max 0.2`, `a_max 4.0`), plus the 5 named candidates `icem_plan` injects.
Pre-registration: `SPEC.md` in this directory, staged before the probe ran.
Instrument: `taniteval/tools/cost_surface_probe.py` (new), pinned by
`stack/tests/test_cost_surface_probe.py` — **33 tests green**, and **393 passed / 1
skipped** across the whole planner-and-kinematics slice of `stack/tests`
(`-k "cost_surface or refa_v1 or steer or kinematic or planner or icem or refav1"`), with
the full tree collecting cleanly at 5,749 tests.

---

## 0. Headline

### 0.1 The single decisive measurement

On the clean epoch there are **25 windows where the tactical decoder asks for `TURN_R`**
— the only population in this corpus where the cost has a lateral question to answer.
`plan()` injects the canonical turn as a seed candidate (`refa_v1.py:1790-1793`). Scored
against `cv`:

| on the 25 `TURN_R` windows | `cv` (straight) | `goal_canonical` (the turn) |
|---|---|---|
| T1 — `1 − cos(z_terminal, goal)` | **0.0000e+00** | **5.9605e-08** (= 1 ULP) |
| T2 — jerk | 0.0000e+00 | 0.0000e+00 |
| T3 — `0.05·κ²` | 0.0000e+00 | **3.2000e-04** |
| **total** | **0.0000e+00** | **3.2006e-04** |
| beats `cv` | — | **0 / 25** |

⛔⛔ **Rolled 10 steps under the very turn profile that BUILT the goal, the tactical
predictor produces a terminal field that float32 cannot distinguish from the one it
produces under ZERO actions — the straight line's cosine to the turn goal is `1.0` to the
last bit, and the turn's is one ULP WORSE. What the correct turn BUYS in the model is
exactly 0.0000e+00; what it PAYS is 3.2000e-04.** No re-weighting rescues this, because
there is no signal to re-weight.

### 0.2 The three-factor reading is refuted as an ordering

| | incumbent | clean epoch |
|---|---|---|
| total variation of the shipped cost over the candidate box | **2.010e-03** | **2.011e-03** |
| … of which the `0.05·κ²` penalty at `kappa_max` | **2.000e-03 (99.5 %)** | **2.000e-03 (99.4 %)** |
| … the world model along the **acceleration** axis (f64) | 1.00e-05 (0.5 %) | 1.18e-03 |
| … the world model along the **curvature** axis (f64) | **1.63e-10 (8×10⁻⁶ %)** | **5.82e-08** |
| the κ² charge at `GOAL_KAPPA_TURN`, ÷ the model's whole curvature range | **1.97 × 10⁶ ×** | **5.50 × 10³ ×** |

The shipped grid minimum is (a = 0, κ = 0) on **139/140** windows and the origin is the
global minimum on **140/140**; `turn_frac(A_full) = 0.0 %` with a **zero-width** interval
on every stratum of both checkpoints.

⭐ **And the two checkpoints differ 117× in the cost's own dynamic range** (goal-term
range 171 → 19,961 float32 ULPs; 36 → 164 distinct values over 189 candidates) **while
their deployed trajectories are byte-identical** — an independent corroboration, on a
cost-side instrument, of `D-REFAV1-PAIRED-READ-VOID` point (5).

### 0.3 The unit conversion changes nothing — and it is INCOMPLETE

Applying `κ → arctan(L·κ)` at the model boundary, verified identical to
`kinematic.as_command` to **0.0** (control C7), moves the cost's minimum on **0 of 140**
windows on **both** checkpoints: `share(ii) = +0.0000 [0.0000, 0.0000]`, a zero-width
paired episode-cluster interval, on **every** stratum — the 25 `TURN_R` windows and the
27 human-turn windows included. The programme's own estimator agrees:
`paired_episode_cluster_bootstrap` delta 0.0, `p_delta_gt0` **0.0000** over 10,000
resamples (`raw/ci_crosscheck.json`). It does exactly what
`C-STEER-CURVATURE-INTERFACE` says it should — the imagined lateral displacement ×**2.935**
(incumbent 2.097e-05 → 6.153e-05; clean epoch's displacement ratio 0.70 % → 2.11 %,
×3.0) and the goal term's curvature range ×**8.9–9.3** — but that is a factor of 9 on a
quantity 5 × 10³ to 2 × 10⁶ times too small to compete.

⛔⛔ **AND THE SHIPPED REPAIR IS INCOMPLETE, WHICH THIS PROBE CAUGHT.**
`as_command` is called at exactly **one** site — `refa_v1.py:1832`, inside `_cost_chunk`
— so under `model_action_units="steer"` the CANDIDATE crosses to the model as
`arctan(L·κ)` while the GOAL is still rolled with the RAW κ
(`_imagine_tactical_goal` → `augment_actions`, `refa_v1.py:1680`, unconverted).
**Candidate and goal are then compared across two different action conventions**, and the
effect is measurable: on the 25 `TURN_R` windows the converted canonical turn moves from
T1 = 5.96e-08 to **1.19e-07** — one ULP *further* from its own goal — and the turn's
modelled advantage goes from exactly 0 to **−5.96e-08**. ⇒ **the conversion must be
applied on BOTH sides of the comparison, or not at all.** Escalated in §8.

### 0.4 Two factors the brief does not contain dominate the three it does

* **(0) THE IMAGINED GOAL IS ITSELF THE CONSTANT-VELOCITY ROLLOUT** on **129/140**
  windows (incumbent) and **110/140** (clean epoch). The incumbent's tactical heads decode
  `LANE_KEEP` on **140/140**, so the goal carries **zero curvature on every window**, and
  `CRUISE`/`ADAPT_SPEED_FOR_CURVE` make the canonical controls exactly zero. The goal
  rollout and the `cv` rollout are then the SAME forward pass, so T1 = `1 − cos(x, x)` = 0
  **exactly** (measured: `cv`, `hold_v0` and `goal_canonical` all read `c_total`
  0.0000e+00). With every term non-negative that is the **global minimum by
  construction**. Two independent probes agree window-for-window
  (`raw/goal_degeneracy_second_probe.json` re-derives `canonical_controls` from the banked
  tokens with NO model in the loop). ⭐ **On the incumbent the goal asks for a turn on
  ZERO of the 27 windows where the human turns ≥ 5°/2 s — recall 0.000.**
* **(iv) IN THE SHIPPED FLOAT32, THE CURVATURE AXIS IS AT THE INSTRUMENT'S NOISE FLOOR.**
  `1 − cosine_similarity` is float32; near `cos = 1` the representable steps are
  5.9604645e-08 apart. The goal term is an exact ULP multiple on **100 %** of windows and
  spans a median of **2.00 ULPs along the curvature axis** on both checkpoints — against
  167.5 (incumbent) and 19,804 (clean epoch) along the acceleration axis. Read straight
  off the banked decision dumps with no model at all, the *shipped planner's own winning
  cost* takes only **six** distinct values over 140 windows — `{−4, −2, 0, +1, +2, +3}`
  ULPs — and **some are NEGATIVE**, i.e. float32 `cosine_similarity` returned a value
  greater than 1 (`raw/banked_source_check.json`). A non-negative quantity that comes out
  negative from rounding is not a small measurement; it is no measurement.

**And the clearest single number about the search itself:** the imitation head's own
**proposal** — the only candidate carrying learned behaviour — is rejected at `c_total`
**1.564e-02** (incumbent) / **1.761e-02** (clean epoch) against `cv`'s 0.0, of which
**93–94 % is the jerk term** and 7 % the κ² term, while **its goal term is 0.0000e+00 /
5.96e-08, i.e. identical to `cv`'s**. The planner discards the proposal for being
non-smooth, and the world model has no opinion.

**A fifth defect**, structural, read from source and pinned by a test: **the planner's
2.0 s control sequence is consumed by the tactical predictor on the 0.6 s tactical
clock**, so a 2.0 s plan is imagined as a **6.0 s** manoeuvre, and the goal's j-th action
is operative step `3j` while the candidate's is operative step `j`.

**The wheelbase does not matter.** `turn_frac` of the repaired arm is 0.0 at
L ∈ {2.73, 2.9, 3.085, 3.216} on both checkpoints — every value the release's vehicles
and the encoding take.

---

## 1. The cost's exact decomposition — read from source before running

`RefAV1.plan._cost_chunk`, `stack/tanitad/refs/refa_v1.py:1795-1820` (repo blob at the
time of the source read; the running tree's md5s are recorded in both raw JSONs):

| # | line | term | weight | units of the squared quantity | sees κ … | live? |
|---|---|---|---|---|---|---|
| T1 | `:1807-1813` | `1 − cosine_similarity(z_terminal, goal)` | **1.0** | dimensionless ∈ [0, 2] | **through the model** (κ crosses at `:1803`) | yes |
| T2 | `:1814-1815` | `0.02 · mean_k((a_{k+1}−a_k)/dt)²` | 0.02 | (m/s³)² | no (channel 0) | yes; ≡ 0 on constant candidates, 93–94 % of the proposal's cost |
| T3 | `:1816` | `0.05 · mean_k(κ_k²)` | 0.05 | (1/m)² | **RAW** | yes — and it is 99.4–99.5 % of the surface |
| T4 | `:1817-1819` | `0.10 · (v_0 + Σa·dt − target_speed)²` | 0.10 | (m/s)² | no | ⛔ **DEAD** |

⚠️ **The brief's "four terms" is wrong in the way that matters: T4 never executes.**
`target_speed` is `None` on every call — `taniteval/tools/refav1_arm.py` calls
`model.plan(feats, v0=…, nav_cmd=…, plan_cfg=…, goal_field=…)` and the token
`target_speed` does not appear anywhere in that file (two probes, two mechanisms:
`grep -c` on the off-Drive mirror = 0, `Select-String` on the G: repo copy = 0).
⇒ **the tactical brain's LONGITUDINAL decision reaches the cost only through T1.**

**Where the units break (factor ii).** `PlanConfig.kappa_max = 0.2`, `_clip`
(`refa_v1_plan.py:161-164`), `GOAL_KAPPA_MAX / GOAL_KAPPA_TURN` (`refa_v1.py:115-116`)
and `unicycle_paths` (`refa_v1_plan.py:291-301`) are all GEOMETRY (1/m).
`augment_actions` (`refa_v1.py:1150-1193`) hands the same tensor to a predictor trained on
the COMMAND channel (road-wheel angle). Both banked checkpoints carry
`speed_channel: false` and `a_dim: 2`, so `augment_actions` returns the controls
**unchanged** (`:1179-1180`) and the crossing is a bare pass-through.

**Where the units break a second time, in TIME (factor v).** With
`plan_level: "tactical"` — both checkpoints — `_cost_chunk` rolls `self.tactical` on the
`[n, H=10, 2]` controls (`:1805`). `H` is 10 **operative** steps = 2.0 s at
`op_dt = 0.2`; the tactical predictor advances `tac_dt = 0.6` s per step, so the same 10
entries are consumed as **6.0 s**. The GOAL is built on the tactical clock *correctly* —
`_imagine_tactical_goal` subsamples the 30-step canonical profile by
`_stride(tac_dt) = 3` before rolling (`refa_v1.py:1679-1681`). ⇒ the goal's j-th action is
operative step `3j`, the candidate's is operative step `j`.
Pinned by `test_the_coarse_cost_consumes_the_plan_on_the_TACTICAL_clock`.

---

## 1b. ⛔ THE PANEL IS VOID AS PRE-REGISTERED — and here is exactly what failed

`PANEL_VOID = True` on **both** runs. **One control failed its pre-registered form: C3,
the zero-model.** SPEC.md committed it as *"`c_goal` must be EXACTLY constant across the
whole grid (peak-to-peak 0.0)"*. Measured over 140 windows: **1.19e-07 = 2.00 ULPs**
(incumbent) and **2.38e-07 = 4.00 ULPs** (clean epoch). On the 7-window smoke it did read
exactly 0.0, which is how it reached SPEC in that form.

**The diagnosis, and why it makes the result stronger rather than weaker.** The zero-model
feeds a bit-identical zero action to every candidate, so the only thing that can differ is
the rollout's dependence on **batch position** — the 189 candidates are scored in chunks
of 64/64/61. Two sub-assertions of C3 passed on both runs, and they are the ones every
curvature conclusion rests on:

* `total(κ) − total(0) == 0.05·κ²` **to 1.16e-10** along the a = 0 row — so *along the
  curvature axis*, the zero-model's goal term is flat to 1.16e-10, i.e. exactly; and
* the zero-model's argmin sits at κ = 0 on **140/140** windows, both checkpoints.

⇒ **This instrument has a MEASURED noise floor of 2–4 ULPs across the grid, and 1.16e-10
along the κ axis.** Every claim is stated against it:

| quantity | incumbent | clean epoch | vs the grid floor |
|---|---|---|---|
| goal term along the **acceleration** axis | 167.5 ULPs | 19,804 ULPs | **84× / 4,951× — resolved** |
| goal term along the **curvature** axis | 2.00 ULPs | 2.00 ULPs | **1.0× / 0.5× — NOT resolved in float32** |

The float64 arms (`*_f64`) exist precisely to see under that floor, and they do: the
curvature range is a real 1.63e-10 / 5.82e-08 — above the 1.16e-10 the zero-model bounds
it by — but three to six orders below the κ² charge. ⛔ The correct reading of the void is
therefore **not** "the panel is unusable"; it is *"the shipped float32 curvature signal is
at this instrument's noise floor, and the conclusions that depend on it are the f64
ones."* §0.1 does not depend on the argmin at all — it is the arithmetic of three named
candidates.

---

## 2-5. The measurement

### Controls — every one must read its known value

| control | reads | incumbent | clean epoch (ep2) |
|---|---|---|---|
| C0 decomposition identity | terms sum to the total, ≤ 1e-6 | 0.00e+00 PASS | 1.16e-10 PASS |
| C1 shipped-cost gate | our re-scoring == PlanResult.baseline_costs | rel err 0.00e+00 (oracle-goal arm scale 0.0367, n=10) PASS | rel err 0.00e+00 (oracle-goal arm scale 0.0301, n=10) PASS |
| C2 banked-winner gate | plan() reproduces the banked cl path < 3.8e-6 m | 0.00e+00 m (n=5) PASS | 0.00e+00 m (n=5) PASS |
| C3 zero-model | action-blind predictor ⇒ goal term flat; total(κ)−total(0) == 0.05κ² exactly | ptp 1.2e-07, penalty recovered to 1.2e-10, argmin κ=0 on all FAIL | ptp 2.4e-07, penalty recovered to 1.2e-10, argmin κ=0 on all FAIL |
| C4 constant-cost | all weights zeroed ⇒ total ≡ 0, argmin = index 0 | PASS | PASS |
| C5 channel-0 invariance | A and B agree EXACTLY on channel 0 | jerk 0.0e+00, a-marginal 0.0e+00 PASS | jerk 0.0e+00, a-marginal 0.0e+00 PASS |
| C6 n and d | printed | n=140 windows, d=189 candidates | n=140 windows, d=189 candidates |
| C7 conversion agreement | our arctan(L·κ) == kinematic.as_command | max |Δ| 0.0e+00, L=2.9 PASS | max |Δ| 0.0e+00, L=2.9 PASS |
| **PANEL_VOID** | any must-pass control failed | **True** | **True** |

### The goal term's resolution — the read that reframes the question

| quantity | incumbent | clean epoch (ep2) |
|---|---|---|
| goal-term range across the WHOLE grid, in float32 ULPs (median) | 171.0 | 19961.0 |
| … along the κ axis at a = 0, in ULPs (median) — convention A | **2.00** | **2.00** |
| … along the κ axis at a = 0, in ULPs (median) — convention B | **2.00** | **9.00** |
| … along the a axis at κ = 0, in ULPs (median) | 167.5 | 19804.5 |
| distinct float32 values the goal term takes over 189 cells (median) | 36 | 164 |
| goal-term values are exact ULP multiples (fraction of windows) | 100% | 100% |
| goal-term range in float64 (median) | 1.013e-05 | 1.190e-03 |
| the explicit κ² penalty at GOAL_KAPPA_TURN = 0.08 | 3.200e-04 | 3.200e-04 |

### Why the goal term loses its own signal: `1 - cos` is QUADRATIC near its optimum

`1 - cos(z, g) = d^2/2` where `d = ||z_hat - g_hat||` is the chord distance between the L2-normalised fields. Near `cos = 1` the cost therefore SQUARES the displacement it is trying to measure, and the square lands under float32's resolution. The chord distance is the SAME ordering (monotone), computed without a cancellation.

| quantity | incumbent | clean epoch (ep2) |
|---|---|---|
| goal term `1 - cos` at the (a=0, kappa=0) cell (median) | 4.996e-16 | 1.485e-09 |
| the same as a CHORD DISTANCE `d = sqrt(2(1-cos))` (median) | 3.156e-08 | 5.449e-05 |
| kappa-axis range of `d` (median) | 2.029e-05 | 2.954e-04 |
| float32 resolution AT that `d` | 1.819e-12 | 2.910e-11 |
| **representable float32 steps the kappa axis spans, as a chord distance** | 10,852,345 | 10,149,448 |
| _n windows with a banked full surface_ | 8 | 8 |

⇒ the same information, in the same float32, spans **~2 steps as `1 - cos`** and the number above **as a chord distance**. The loss is catastrophic cancellation in `1 - cos`, not a limit of the model.


### Factor (0) — the imagined goal itself

| quantity | incumbent | clean epoch (ep2) |
|---|---|---|
| windows whose canonical goal controls are EXACTLY zero | **129 / 140** (92.1%) | **110 / 140** (78.6%) |
| decoded tactical LATERAL token | {'LANE_KEEP': 140} | {'LANE_KEEP': 115, 'TURN_R': 25} |
| decoded tactical LONGITUDINAL token | {'CRUISE': 98, 'ADAPT_SPEED_FOR_CURVE': 42} | {'CRUISE': 98, 'ADAPT_SPEED_FOR_CURVE': 42} |

### Does the imagined goal ask for a turn where the human turns?

The 'human turns' stratum is GT heading change ≥ 5° over the 2.0 s plan horizon, read from the raw 10 Hz poses. Ground truth SELECTS windows here; it never enters a cost.

| quantity | incumbent | clean epoch (ep2) |
|---|---|---|
| GT heading change over 2.0 s — p95 / max (deg) | 28.6 | 28.6 |
| … max (deg) | 49.7 | 49.7 |
| windows where the HUMAN turns ≥ 5°/2 s | 27 | 27 |
| … of those, ≥ 20°/2 s (an unambiguous turn) | 9 | 9 |
| … of those, implied curvature ≥ 0.04 1/m — the scale `GOAL_KAPPA_TURN` addresses | 9 | 9 |
| windows where the imagined GOAL carries curvature | 0 | 25 |
| both | 0 | 9 |
| human turns but the goal is straight | **27** | **18** |
| goal turns but the human does not | 0 | 16 |
| **recall** — the goal turns GIVEN the human turns | **0.000** | **0.333** |
| precision — the human turns GIVEN the goal turns | 0.000 | 0.360 |

### The attribution panel — incumbent

`turn_frac` = fraction of windows whose grid argmin has |κ*| ≥ 0.04 (= ½ GOAL_KAPPA_TURN). Paired episode-cluster bootstrap, 10,000 resamples, [2.5, 97.5] pct.

| stratum | n (eps) | A_full (shipped) | A_nopen (−penalty) | B_full (+conversion) | B_nopen (both) | A_full_f64 (exact cosine) | B_nopen_f64 (all three) |
|---|---|---|---|---|---|---|---|
| `all` | 140 (20) | 0.0% [0.0, 0.0] | 89.3% [84.3, 94.3] | 0.0% [0.0, 0.0] | 89.3% [84.3, 94.3] | 0.0% [0.0, 0.0] | 7.1% [2.9, 12.9] |
| `live_goal` | 11 (6) | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 90.9% [72.7, 100.0] | 0.0% [0.0, 0.0] | 90.9% [72.7, 100.0] |
| `zero_goal` | 129 (20) | 0.0% [0.0, 0.0] | 88.4% [83.1, 93.7] | 0.0% [0.0, 0.0] | 89.1% [83.3, 94.6] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `goal_turn` | 0 | — | — | — | — | — | — |
| `human_turns` | 27 (10) | 0.0% [0.0, 0.0] | 85.2% [66.7, 100.0] | 0.0% [0.0, 0.0] | 85.2% [75.0, 96.0] | 0.0% [0.0, 0.0] | 3.7% [0.0, 11.8] |
| `human_turns_live_goal` | 2 (2) | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 50.0% [0.0, 100.0] | 0.0% [0.0, 0.0] | 50.0% [0.0, 100.0] |

**Factor shares** (Δ turn_frac from the shipped arm, same estimator):

| stratum | n | (iii) penalty | (ii) boundary | (ii)+(iii) | (iv) f32 saturation | all three |
|---|---|---|---|---|---|---|
| `all` | 140 | 89.3% [84.3, 94.3] | 0.0% [0.0, 0.0] | 89.3% [84.3, 94.3] | 0.0% [0.0, 0.0] | 7.1% [2.9, 12.9] |
| `live_goal` | 11 | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 90.9% [72.7, 100.0] | 0.0% [0.0, 0.0] | 90.9% [72.7, 100.0] |
| `zero_goal` | 129 | 88.4% [83.1, 93.7] | 0.0% [0.0, 0.0] | 89.1% [83.3, 94.6] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `human_turns` | 27 | 85.2% [66.7, 100.0] | 0.0% [0.0, 0.0] | 85.2% [75.0, 96.0] | 0.0% [0.0, 0.0] | 3.7% [0.0, 11.8] |
| `human_turns_live_goal` | 2 | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | 50.0% [0.0, 100.0] | 0.0% [0.0, 0.0] | 50.0% [0.0, 100.0] |

**The surface itself** (medians over the stratum):

| stratum | n | κ-range of T1, f64 | a-range of T1, f64 | lateral/longitudinal potency (f64) | B/A κ-range | T1 gain at κ_turn (f64) | κ² charge at κ_turn | windows where gain > charge (A / B) |
|---|---|---|---|---|---|---|---|---|
| `all` | 140 | 1.627e-10 | 1.001e-05 | 1.521e-05 | 1.000 | -2.235e-11 | 3.200e-04 | 0% / 0% |
| `live_goal` | 11 | 3.717e-09 | 8.237e-06 | 4.545e-04 | 1.000 | -7.102e-10 | 3.200e-04 | 0% / 0% |
| `zero_goal` | 129 | 1.576e-10 | 1.040e-05 | 1.481e-05 | 1.000 | -2.154e-11 | 3.200e-04 | 0% / 0% |
| `human_turns` | 27 | 1.612e-10 | 1.053e-05 | 1.452e-05 | 1.000 | -2.236e-11 | 3.200e-04 | 0% / 0% |
| `human_turns_live_goal` | 2 | 3.983e-09 | 8.064e-06 | 4.593e-04 | 1.250 | -7.730e-10 | 3.200e-04 | 0% / 0% |

**The named candidates** (the ones `icem_plan` injects, plus the canonical-goal seed `plan()` adds):

| candidate | A c_total (median) | of which T1 | T2 jerk | T3 κ² | beats `cv` (A) | beats `cv` (B) |
|---|---|---|---|---|---|---|
| `cv` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0% | 0% |
| `decel_1.5` | 1.192e-06 | 1.192e-06 | 0.000e+00 | 0.000e+00 | 0% | 0% |
| `goal_canonical` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0% | 0% |
| `hold_v0` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0% | 0% |
| `proposal` | 1.564e-02 | 0.000e+00 | 1.453e-02 | 1.161e-03 | 0% | 0% |

**Wheelbase sensitivity** (turn_frac of the repaired arm at other L):

| L | turn_frac |
|---|---|
| Lsweep_2.73_turn | 0.0% |
| Lsweep_3.085_turn | 0.0% |
| Lsweep_3.216_turn | 0.0% |
| B_full_at_2.9 | 0.0% |

_source: `raw/cost_surface_incumbent.json`; device cuda; 140 windows; 1256.3 s; ckpt step 1000_

**Source md5 of every file this run imported:**

* `stack/tanitad/refs/refa_v1.py` = `6e494ef8116031bf5943ec477c2b8f9c`
* `stack/tanitad/refs/refa_v1_plan.py` = `6e3c8f1ee5260dd350cda366bd22272a`
* `stack/tanitad/models/kinematic.py` = `9714850930b66f0f049c2692ceaca862`
* `taniteval/tools/refav1_arm.py` = `b6896dd7d82382c8b148738fca77ba22`
* `taniteval/tools/cost_surface_probe.py` = `4c5dc3d57bf2cb06376fb788ec0cab3e`
* `_tanitad_imported_from` = `C:\Users\Admin\tanitad-wt\stack\tanitad\__init__.py`

### The attribution panel — clean epoch (ep2)

`turn_frac` = fraction of windows whose grid argmin has |κ*| ≥ 0.04 (= ½ GOAL_KAPPA_TURN). Paired episode-cluster bootstrap, 10,000 resamples, [2.5, 97.5] pct.

| stratum | n (eps) | A_full (shipped) | A_nopen (−penalty) | B_full (+conversion) | B_nopen (both) | A_full_f64 (exact cosine) | B_nopen_f64 (all three) |
|---|---|---|---|---|---|---|---|
| `all` | 140 (20) | 0.0% [0.0, 0.0] | 85.7% [80.7, 90.7] | 0.0% [0.0, 0.0] | 43.6% [34.3, 52.9] | 0.0% [0.0, 0.0] | 5.0% [1.4, 8.6] |
| `live_goal` | 30 (6) | 0.0% [0.0, 0.0] | 80.0% [66.7, 96.0] | 0.0% [0.0, 0.0] | 60.0% [48.3, 72.0] | 0.0% [0.0, 0.0] | 23.3% [14.6, 45.5] |
| `zero_goal` | 110 (17) | 0.0% [0.0, 0.0] | 87.3% [82.2, 92.2] | 0.0% [0.0, 0.0] | 39.1% [28.4, 49.6] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `goal_turn` | 25 (4) | 0.0% [0.0, 0.0] | 80.0% [64.0, 100.0] | 0.0% [0.0, 0.0] | 60.0% [45.5, 71.4] | 0.0% [0.0, 0.0] | 12.0% [5.3, 14.3] |
| `human_turns` | 27 (10) | 0.0% [0.0, 0.0] | 85.2% [71.9, 100.0] | 0.0% [0.0, 0.0] | 33.3% [18.5, 47.6] | 0.0% [0.0, 0.0] | 3.7% [0.0, 11.8] |
| `human_turns_live_goal` | 10 (4) | 0.0% [0.0, 0.0] | 80.0% [57.1, 100.0] | 0.0% [0.0, 0.0] | 40.0% [10.0, 71.4] | 0.0% [0.0, 0.0] | 10.0% [0.0, 50.0] |

**Factor shares** (Δ turn_frac from the shipped arm, same estimator):

| stratum | n | (iii) penalty | (ii) boundary | (ii)+(iii) | (iv) f32 saturation | all three |
|---|---|---|---|---|---|---|
| `all` | 140 | 85.7% [80.7, 90.7] | 0.0% [0.0, 0.0] | 43.6% [34.3, 52.9] | 0.0% [0.0, 0.0] | 5.0% [1.4, 8.6] |
| `live_goal` | 30 | 80.0% [66.7, 96.0] | 0.0% [0.0, 0.0] | 60.0% [48.3, 72.0] | 0.0% [0.0, 0.0] | 23.3% [14.6, 45.5] |
| `zero_goal` | 110 | 87.3% [82.2, 92.2] | 0.0% [0.0, 0.0] | 39.1% [28.4, 49.6] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| `goal_turn` | 25 | 80.0% [64.0, 100.0] | 0.0% [0.0, 0.0] | 60.0% [45.5, 71.4] | 0.0% [0.0, 0.0] | 12.0% [5.3, 14.3] |
| `human_turns` | 27 | 85.2% [71.9, 100.0] | 0.0% [0.0, 0.0] | 33.3% [18.5, 47.6] | 0.0% [0.0, 0.0] | 3.7% [0.0, 11.8] |
| `human_turns_live_goal` | 10 | 80.0% [57.1, 100.0] | 0.0% [0.0, 0.0] | 40.0% [10.0, 71.4] | 0.0% [0.0, 0.0] | 10.0% [0.0, 50.0] |

**The surface itself** (medians over the stratum):

| stratum | n | κ-range of T1, f64 | a-range of T1, f64 | lateral/longitudinal potency (f64) | B/A κ-range | T1 gain at κ_turn (f64) | κ² charge at κ_turn | windows where gain > charge (A / B) |
|---|---|---|---|---|---|---|---|---|
| `all` | 140 | 5.816e-08 | 1.180e-03 | 4.936e-05 | 4.500 | -7.779e-09 | 3.200e-04 | 0% / 0% |
| `live_goal` | 30 | 5.735e-08 | 1.170e-03 | 4.757e-05 | 3.000 | -1.214e-08 | 3.200e-04 | 0% / 0% |
| `zero_goal` | 110 | 5.834e-08 | 1.187e-03 | 4.944e-05 | 4.500 | -7.627e-09 | 3.200e-04 | 0% / 0% |
| `goal_turn` | 25 | 5.712e-08 | 1.175e-03 | 4.752e-05 | 3.500 | -1.205e-08 | 3.200e-04 | 0% / 0% |
| `human_turns` | 27 | 5.770e-08 | 1.175e-03 | 4.920e-05 | 4.000 | -7.718e-09 | 3.200e-04 | 0% / 0% |
| `human_turns_live_goal` | 10 | 5.705e-08 | 1.184e-03 | 4.747e-05 | 3.000 | -1.203e-08 | 3.200e-04 | 0% / 0% |

**The named candidates** (the ones `icem_plan` injects, plus the canonical-goal seed `plan()` adds):

| candidate | A c_total (median) | of which T1 | T2 jerk | T3 κ² | beats `cv` (A) | beats `cv` (B) |
|---|---|---|---|---|---|---|
| `cv` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0% | 0% |
| `decel_1.5` | 1.333e-04 | 1.333e-04 | 0.000e+00 | 0.000e+00 | 0% | 0% |
| `goal_canonical` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0% | 0% |
| `hold_v0` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0% | 0% |
| `proposal` | 1.761e-02 | 5.960e-08 | 1.649e-02 | 1.225e-03 | 0% | 0% |

**Wheelbase sensitivity** (turn_frac of the repaired arm at other L):

| L | turn_frac |
|---|---|
| Lsweep_2.73_turn | 0.0% |
| Lsweep_3.085_turn | 0.0% |
| Lsweep_3.216_turn | 0.0% |
| B_full_at_2.9 | 0.0% |

_source: `raw/cost_surface_ep2.json`; device cuda; 140 windows; 1254.6 s; ckpt step 1000_

**Source md5 of every file this run imported:**

* `stack/tanitad/refs/refa_v1.py` = `6e494ef8116031bf5943ec477c2b8f9c`
* `stack/tanitad/refs/refa_v1_plan.py` = `6e3c8f1ee5260dd350cda366bd22272a`
* `stack/tanitad/models/kinematic.py` = `9714850930b66f0f049c2692ceaca862`
* `taniteval/tools/refav1_arm.py` = `b6896dd7d82382c8b148738fca77ba22`
* `taniteval/tools/cost_surface_probe.py` = `4c5dc3d57bf2cb06376fb788ec0cab3e`
* `_tanitad_imported_from` = `C:\Users\Admin\tanitad-wt\stack\tanitad\__init__.py`

---

## 5b. The pre-registration ledger - which committed outcome fired

SPEC.md section 5 committed BOTH outcomes for each factor before the probe ran. The
DISCRIMINATING stratum it pre-registered - windows whose imagined goal is not the
zero-control rollout - is n = 11 on the incumbent (longitudinal-only goals) and n = 30 on
the clean epoch, **of which 25 carry a TURN**. The incumbent has NO window with a curved
goal, which is itself finding (0).

| factor | outcome X (committed) | outcome Y (committed) | measured | which fired |
|---|---|---|---|---|
| **(0) degenerate goal** | >= 50 % of windows have an exactly-zero canonical goal ⇒ the flat plan is exactly optimal there and the repair target is the TACTICAL HEAD | < 50 % ⇒ the goal is live and (i)-(iii) carry the phenomenon | **92.1 %** (129/140, incumbent) and **78.6 %** (110/140, clean epoch); `LANE_KEEP` on 140/140 incumbent | **X**, on both checkpoints |
| **(ii) x2.9 boundary** | `share(ii) > 0`, CI excludes 0 ⇒ the one-line conversion is the cheapest repair | CI includes 0 ⇒ the conversion alone does not make the planner turn, **and must not be sold as the fix** | `share(ii) = +0.0000 [0.0000, 0.0000]` on ALL strata of BOTH checkpoints, incl. the 25 `TURN_R` windows; `p_delta_gt0 = 0.0000` over 10,000 resamples | **Y** |
| **(iii) kappa^2 penalty** | `share(iii) > 0`, CI excludes 0 ⇒ re-weight or drop T3 | CI includes 0 ⇒ T3 is not the binding constraint | shipped f32: `+0.8929 [0.8429, 0.9429]` / `+0.8571 [0.8071, 0.9071]`. Exact cosine: `+0.0786 [0.0286, 0.1357]` / `+0.0500 [0.0143, 0.0857]` - all exclude 0 | **X**, with the flat-surface caveat of section 6 R3 |
| **(i) lateral potency** | residual after `B_nopen` on the TURN stratum > 0.5 ⇒ the repair is a TRAINING lever | <= 0.5 ⇒ potency suffices once (ii)/(iii) are repaired | **on the clean epoch's 25 `TURN_R` windows: residual after `B_nopen_f64` = 1 - 0.120 = 0.880**; and the direct read of section 0.1 is stronger still - the correct turn's modelled advantage over `cv` is **exactly 0.0000e+00** | **X**, decisively |

⚠️ **What this ledger does not let me claim.** The (iii) row fired X for a reason that is
*not* the mechanism the SPEC had in mind: removing T3 does not reveal a modelled
preference, it reveals a flat box (section 6 R3). And the incumbent could not test (i) on
the pre-registered TURN stratum at all, because that stratum is empty there - the clean
epoch is what made the pre-registered test executable.

---

## 6. What is the CHEAPEST repair that makes the planner turn where the human turns?

⛔ **There isn't a cheap one, and the cheap-looking one is measured to do nothing.** The
repairs below are ordered by measured leverage, each stated with what it buys and what it
does not. Every diff is **PROPOSED, not applied** — `refa_v1.py`, `refa_v1_plan.py` and
`refav1_arm.py` are read-only to this stream.

⭐ **All of this is inference-side and CANNOT disturb the live training.** `RefAV1.plan`
is called from exactly two places in the tree — `taniteval/tools/refav1_arm.py:735` (the
T1 adapter) and this probe — and from no trainer; `_cost_chunk` is a closure inside
`plan()`, which is `@torch.no_grad()`. Verified by grep over `stack/`, `taniteval/`,
`colab/`.

### R1 — the unit conversion. NECESSARY, CORRECT, ALREADY IMPLEMENTED, and MEASURED TO CHANGE NOTHING HERE

`κ → steer = arctan(L_enc·κ)` at the model boundary and nowhere else.
**Not a proposal from this stream — it already exists in the repo**, as
`RefAV1.plan(..., model_action_units="steer")` on top of
`stack/tanitad/models/kinematic.py`'s `as_command` / `STEER_WHEELBASE_M = 2.9`, wired to
`taniteval/tools/refav1_arm.py --action-units` and defaulting to `"kappa"` so every banked
number stays reproducible. This probe's own conversion agrees with that helper to **0.0**
(control C7).

* **What it buys, measured:** the imagined lateral displacement of the terminal field
  ×**2.935** (2.097e-05 → 6.153e-05), matching the predicted ×2.9 to 1 %; the goal term's
  curvature range ×**8.9** (1.63e-10 → 1.45e-09); the lateral/longitudinal displacement
  ratio 0.42 % → 1.24 %.
* **What it does not buy:** `share(ii) = +0.0000`, interval `[0.0000, 0.0000]`, on
  **140/140** windows of **both** checkpoints and on **every** stratum — including the 25
  `TURN_R` windows and the 27 where the human turns. The programme's own paired estimator
  agrees: `paired_episode_cluster_bootstrap` delta 0.0, `p_delta_gt0` **0.0000** over
  10,000 resamples, `separated = False` (`raw/ci_crosscheck.json`).
* ⛔ **AND AS SHIPPED IT IS INCOMPLETE — CONVERT BOTH SIDES OR NEITHER.** `as_command` is
  called at exactly one site, `refa_v1.py:1832` inside `_cost_chunk`, so with
  `model_action_units="steer"` the CANDIDATE crosses to the model as `arctan(L·κ)` while
  the GOAL is still rolled with the RAW κ (`_imagine_tactical_goal` →
  `augment_actions`, `refa_v1.py:1680`). Measured on the 25 `TURN_R` windows: the
  converted canonical turn moves from T1 = 5.96e-08 to **1.19e-07** — one ULP *further*
  from its own goal — and the turn's modelled advantage over `cv` goes from exactly 0 to
  **−5.96e-08**. The one-line completion is to thread the same `model_action_units`
  through `_imagine_tactical_goal`'s `augment_actions` call, or to convert neither.
  **PROPOSED, not applied; escalated in §8.**
* ⇒ **Ship it because it is correct — after completing it — and never because it makes
  the planner turn.**

### R2 — the arithmetic: stop computing a near-1 similarity by cancellation

`1 − cos(z, g)` equals `d²/2` where `d = ‖ẑ − ĝ‖` is the chord distance between the
L2-normalised fields, so the shipped term **squares** the displacement it is trying to
measure and lands under float32's resolution. The chord distance is **monotone in the same
ordering** — it changes no decision in exact arithmetic — and it is computed as a norm of
differences instead of a cancellation:

```diff
--- a/stack/tanitad/refs/refa_v1.py
+++ b/stack/tanitad/refs/refa_v1.py
@@ -1807,8 +1807,20 @@ def _cost_chunk(controls, pred=None, z0=None) -> Tensor:
             if goal_t is not None:
                 # an OPERATIVE terminal field is pooled into the tactical query
                 # space by the model's own tac_pool (see the docstring)
                 zt = zk if pred is self.tactical else self._tac_field(zk)
                 g = goal_t.expand(n, -1, -1)
-                c = c + (1.0 - F.cosine_similarity(
-                    zt.flatten(1), g.flatten(1), dim=-1))
+                # ⛔ `1 - cos` IS d^2/2 AND FLOAT32 CANNOT HOLD IT. Near cos = 1
+                # the representable steps are 5.9604645e-08 apart; MEASURED
+                # 2026-09-03 over 140 windows the term is an exact ULP multiple
+                # on 100 % of them, takes ~36 distinct values over a 189-cell
+                # candidate grid, and spans 2.00 ULPs along the curvature axis.
+                # The banked planner's own winning cost took SIX distinct values
+                # over 140 windows and some were NEGATIVE. The chord distance is
+                # the SAME ordering (monotone in 1 - cos) with no cancellation:
+                c = c + (F.normalize(zt.flatten(1), dim=-1) -
+                         F.normalize(g.flatten(1), dim=-1)).norm(dim=-1)
```

* **What it buys, measured:** on the curvature axis the term goes from **2.00
  representable float32 steps** to the number in the chord table above (~10⁷). It also
  rescales T1: the longitudinal range becomes ≈ `sqrt(2·1.00e-05)` = **4.5e-03**, which is
  **larger** than the κ² charge at `kappa_max` (2.0e-03) — so the world model would, for
  the first time, outvote the penalty longitudinally, **without touching a weight**.
* **What it does not buy:** the lateral range becomes ≈ `sqrt(2·1.63e-10)` = **1.8e-05**,
  still **18× below** the κ² charge at `GOAL_KAPPA_TURN` (3.2e-04) and **111× below** it
  at `kappa_max`. R2 alone does not make the planner turn.
* ⚠️ **It is not free of consequence**: T1's magnitude moves by ~3 orders, so every
  weight that competes with it (0.02 jerk, 0.05 κ²) is implicitly re-tuned. Land it with
  R3, not on its own, and re-run this probe as the acceptance test.

### R3 — the κ² weight. A re-weighting alone CANNOT recover the lateral decision

`0.05` was never calibrated against the term it competes with. The measured requirement:

| to be commensurate with … | measured size | required weight on `mean(κ²)` at κ = 0.2 |
|---|---|---|
| the goal term's **longitudinal** range (shipped `1−cos`) | 1.00e-05 | ≤ 2.5e-04 (**200× smaller**) |
| the goal term's **curvature** range (shipped `1−cos`) | 1.63e-10 | ≤ 4.1e-09 (**12 million× smaller**) |
| the goal term's **curvature** range, **after R2** | 1.8e-05 | ≤ 4.5e-04 (**110× smaller**) |

⛔ **Do not simply delete T3.** The `A_nopen` arm measures exactly what that produces:
`turn_frac` jumps to **0.8929** — while the surface's peak-to-peak collapses from 2.01e-03
to **1.02e-05** and the median |κ*| runs to the grid edge (0.16–0.20). In float64 the same
ablation gives **0.0786**, and on the 129 zero-goal windows it gives **0.0000**. The 0.89
is a **coin flip over a numerically flat box**, not steering; the `*_surface_ptp` column
exists so nobody can quote it as a repair.
⇒ **T3's weight must fall by ~100× and R2 must land, or the change trades a deterministic
wrong answer for a random one.**

### R4 — the goal, which no cost repair can reach. THIS IS THE ACTUAL LEVER

On the incumbent the tactical heads decode `LANE_KEEP` on **140/140** windows, so the
imagined goal carries **zero curvature everywhere** and the straight plan is the correct
minimiser of a correctly-computed cost. **On this checkpoint the planner is not broken; it
is obeying a goal that never asks for a turn** — and it asks for none on **0 of the 27**
windows where the human turns (recall 0.000). 129/140 windows have canonical controls that
are exactly zero, which makes the flat plan the exact global minimum by construction.
The clean epoch is better and still not good: `TURN_R` on 25/140, 110/140 still exactly
zero.
⇒ **the lever is the tactical decoder, and it is a TRAINING lever, not a planner one.**

⚠️ **But R4 alone would not be enough either, and §0.1 is why.** On the 25 windows where
the decoder DOES ask for a turn, the correct turn's modelled advantage over the straight
line is **exactly 0.0000e+00** — the tactical predictor rolled under the turn profile that
BUILT the goal is float32-indistinguishable from the same predictor rolled under zero
actions. Fixing the decoder gives the cost a lateral question; it does not give the
imagination an answer. **Both levers are training levers, and they are different ones.**

### R5 — the time-units defect

`_cost_chunk` consumes the planner's 2.0 s control sequence on the 0.6 s tactical clock
(§1). Either roll the cost on the operative predictor (`plan_level="operative"`), or
subsample the candidate the way `_imagine_tactical_goal` subsamples the goal. Correctness,
independent of everything above; **not** quantified by this probe beyond the source read
and its pinning test.

### R6 — the honest alternative, if a lateral gradient is wanted NOW

The planner already holds a metric-space quantity on both sides: `unicycle_paths(controls,
v0, dt)` for the candidate, and the same integrator on `canonical_controls` for the goal.
A metre-space lateral term would separate a 12.5 m turn from a straight line by **metres**,
not by 1.63e-10. ⚠️ **It bypasses the world model**, so it is a way to restore a lateral
gradient, **not** a way to make the imagination decide — and it must never be reported as
the latter. Offered because the alternative (waiting for the imagination to become
laterally potent) is a training programme, not a patch.

---

## 6b. Limitations of this instrument, stated before anyone has to find them

1. **The grid is CONSTANT over the horizon** — 21 κ × 9 a = 189 candidates, each held
   fixed for all 10 steps. That is a 2-D slice of the planner's 20-D candidate space, so
   `turn_frac` is *"where does the cost's minimum sit over constant controls"*, not
   *"what would iCEM return"*. Two things bound the gap: the **named** candidates
   (`cv` / `hold_v0` / `decel_1.5` / `proposal` / `goal_canonical`) are the non-constant
   ones `icem_plan` actually injects and are scored alongside, and the harness gate C2
   shows the real `plan()` returns the zero control on the same windows to 0.0 m. On a
   constant grid T2 (jerk) is identically zero, so the comfort term cannot move the grid
   minimum — a structural fact, stated rather than hidden, and the named candidates are
   where T2 is exercised (it is 93 % of the proposal's cost).
2. **n is small on the strata that discriminate** (11 live-goal windows on the incumbent;
   2 on live-goal ∩ human-turns). Intervals are printed; no point estimate on those
   strata is quotable bare.
3. **An argmin taken on a numerically flat surface is a coin flip.** The `*_surface_ptp`
   column makes this visible; it is exactly what the `A_nopen` arm does.
4. **C3 failed its pre-registered form** (§1b) and the panel is `PANEL_VOID = True`. The
   measured noise floor is 2.00 ULPs across the grid and 1.16e-10 along the κ axis; every
   claim is stated against it.
5. **`turn_frac`'s threshold (|κ*| ≥ 0.04 = ½ `GOAL_KAPPA_TURN`) is a convention.** Median
   |κ*| per arm is reported beside it.
6. **The `*_gain_at_kturn` columns are evaluated at κ = +0.08 only**, so on a `TURN_R`
   window (canonical κ = −0.08) they read the wrong side. The verdict uses the
   sign-agnostic reads: the κ-axis RANGE and the argmin arms.

## 7. What this result does NOT claim

* **No driving claim.** Everything here is T0. A change in `turn_frac` is a necessary, not
  sufficient, condition for better driving; only the T1 adapter can close that.
* **No re-ranking of the two checkpoints.** They are reported side by side.
* **No claim that the predictor is lateral-deaf.** `D-ACTDIV-ANCHORED-REFAV1` measured
  that it is not. The finding here is that the response is **real, small, and both
  discarded by the cost's arithmetic and out-voted by its penalty**.
  ⚠️ **The 0.42 % lateral/longitudinal displacement ratio measured here is NOT a
  replication of that row's 0.329 %** and must not be quoted as one: three things differ —
  the horizon (this is the **tactical** terminal field at 10 tactical steps, that was the
  **h = 1** prediction), the normalisation (this is the ratio of MAX displacement across
  the planner's own ±`kappa_max` / ±`a_max` box, that was matched ≈ 2σ corpus-action
  levels), and the population (140 eval windows vs 144 diagnostic windows). That the two
  land within a factor of ~1.3 of each other across those three differences is
  **corroborative**, and that is all it is.
* **No edit to `refa_v1.py`, `refa_v1_plan.py` or `refav1_arm.py`.** All read-only to this
  stream; R2/R3/R5/R6 are PROPOSED.
* **`L_enc = 2.9` is INHERITED**, from the sibling Data-Engineering stream's staged
  artifact `TanitAD Research Lab/Data Engineering/Research/2026-09-03-steer-curvature-interface/raw/lenc_exact.json`
  (algebraic inversion against the producer's own `curvature` column, 20/20 clips,
  pointwise IQR ~1.2e-07). This stream re-verified only that
  `kinematic.STEER_WHEELBASE_M == 2.9` and that its own conversion matches
  `kinematic.as_command` exactly (C7). **The verdict is insensitive to it**: `turn_frac` of
  the repaired arm is 0.0 at every L ∈ {2.73, 2.9, 3.085, 3.216}.

## 7b. The concurrency hazard SPEC §7 recorded, and how it resolved

SPEC.md §7 recorded, before the run, that the off-Drive mirror was **ahead of the repo** on
`kinematic.py`, `refa_v1.py`, `refa_v1_plan.py` and `refav1_arm.py` — a sibling stream's
steer-conversion work living on one disk — and bound this probe to record the md5 of every
file it imports. Three things then happened and all three are recorded rather than assumed:

1. **The sibling stream synced its work into the repo at ~07:08–07:11**, before this run
   launched at 07:20. Verified by md5: repo and mirror are now **identical** on all four
   files, and the repo copies contain `STEER_WHEELBASE_M` and `model_action_units`
   (positive assertion via `Select-String`, since `diff` on the G: mount returned empty
   for files whose md5s differ — the documented `ls-tree` failure family). **The stranding
   hazard is resolved; no escalation is needed for it.**
2. **The files this run actually imported are pinned.** `raw/source_md5_at_run_time.txt`
   snapshots them before the run; both JSONs record the same md5s at the end of their own
   run. They match, so no file changed under either run.
3. **The behavioural guarantee does not rest on md5 anyway.** C1 (our re-scoring ==
   `PlanResult.baseline_costs`, rel err 0.0) and C2 (a real `plan()` reproduces the banked
   `cl` trajectory to 0.0 m) run **in-process** against artifacts banked before any of
   these edits. Whatever bytes were loaded, they reproduce the shipped and banked
   behaviour exactly. That is the property that matters, and it is measured per run.

## 8. ESCALATION — what needs a decision, and from whom

1. ⛔ **THE SHIPPED κ→steer CONVERSION IS INCOMPLETE AND THE SIBLING STREAM SHOULD BE
   TOLD TODAY.** `as_command` is applied to the candidate (`refa_v1.py:1832`) and NOT to
   the imagined goal (`refa_v1.py:1680`), so `model_action_units="steer"` compares the two
   across different action conventions — measured, on the 25 `TURN_R` windows, as the
   converted turn landing one ULP FURTHER from its own goal. It is default-OFF, so nothing
   banked is affected, but the flag is currently a trap for whoever turns it on. This is
   the one item with a same-day owner.
2. **R2 + R3 are a joint change to `_cost_chunk` and need an owner and a decision.** They
   are inference-only and cannot disturb Thor. The acceptance test is this probe re-run;
   the pass condition should be pre-registered before the change, not after.
3. **R4 is the real lever and it is not in this stream's file ownership.** `LANE_KEEP` on
   140/140 windows at step 1,000 is a tactical-decoder finding and belongs to whoever owns
   the tactical head and the live ep3 run.
4. **`D-REFAV1-PAIRED-READ-VOID` should be read together with this.** Its "the deployed
   trajectory cannot discriminate recipes" now has a mechanism that is arithmetic, not
   representational — and that changes which experiments are worth compute.

---

## 9. Package contents

Everything below is in the repo working tree and `git add`ed (blob-verified), and every
file also exists in a second location — nothing here lives in one place only.

| file | what it is |
|---|---|
| `SPEC.md` | the pre-registration, staged before the probe ran |
| `RESULT.md` | this document |
| `raw/cost_surface_incumbent.json` | the full run on `ckpt/ckpt.pt` — 140 rows, 8 banked surfaces, both gates, all controls |
| `raw/cost_surface_ep2.json` | the same on the clean epoch `ckpt_ep2/ckpt.pt` |
| `raw/goal_degeneracy_second_probe.{py,json}` | the model-free second probe of factor (0) |
| `raw/banked_source_check.{py,json}` | the third probe: the shipped winning cost read straight off the banked T1 dumps (six distinct ULP values, some negative) |
| `raw/turn_seed_check.{py,json}` | the decisive per-stratum read of §0.1: the canonical turn vs `cv` on the 25 `TURN_R` windows |
| `raw/ci_crosscheck.{py,json}` | our bootstrap against `taniteval/ci.py`'s own episode-cluster and paired estimators |
| `raw/make_tables.py`, `raw/assemble_result.py` | the analysis that produced §2–5 |
| `raw/run_cost_surface.ps1`, `raw/run.log` | the exact detached invocation and its log |
| `raw/source_md5_at_run_time.txt` | the md5 of every imported source file, snapshotted before the run |
| `raw/tables.md` | the generated tables, standalone |

The instrument itself is `taniteval/tools/cost_surface_probe.py`, pinned by
`stack/tests/test_cost_surface_probe.py`. The two checkpoints and the fp8/episode caches
are INPUTS, not deliverables of this stream: they live on the dev box under
`C:\Users\Admin\refav1_eval_slice\` and on Thor.
