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