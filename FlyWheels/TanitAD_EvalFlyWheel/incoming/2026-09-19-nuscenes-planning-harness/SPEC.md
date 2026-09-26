# SPEC — nuScenes open-loop planning harness (W6 / E5) — targets PRE-STATED

**Package** `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/`
**Written** 2026-09-19, BEFORE any test in `taniteval/tests/test_nuscenes_planning.py` was run.
Every expected value below is a LITERAL derived here from a closed form or copied from a banked
primary. The tests assert these literals; none is an expression over the code under test.

⛔ **Not a criterion.** nuScenes open-loop planning is inadmissible as a TanitAD criterion
(`H-EVAL-6` SUPPORTED; `D-BENCH-PORT` SKIP claim-bearing). Every artifact this harness writes carries
`claim_bearing: false`; the API has no way to set it true.

## 0. Reference implementations (PUBLISHED-CODE, pinned; extracts in `raw/reference_extracts/`)

| id | repo @ commit | what is read |
|---|---|---|
| ST-P3 | `OpenDriveLab/ST-P3@69aabefd2610951d9e34238142776ed2228673be` | `stp3/metrics.py:263-396` (PlanningMetric), `evaluate.py:57-73,121-137,162-166` (reduction), `stp3/datas/NuscenesData.py:96-148,303-348,505-532` |
| UniAD | `OpenDriveLab/UniAD@609ee083ea51c3521c323f1279dfc4cee0e60467` | `…/planning_head_plugin/planning_metrics.py:15-149`, `…/datasets/nuscenes_e2e_dataset.py:1030-1050` (strategy switch), `…/uniad/apis/test.py:92-102`, `…/data_utils/trajectory_api.py:226-282`, `…/pipelines/occflow_label.py:11-160`, `projects/configs/stage2_e2e/base_e2e.py:58-61,498-499,560-561` |
| UniAD @ first public commit | `OpenDriveLab/UniAD@4e91222da855fa83a8caa55f512536e82f9f9818` (2023-03-29 "init") | `…/nuscenes_e2e_dataset.py:1024-1036` — prints `value[i]` only (at-t); the `stp3` option arrives in `33cd8ecf7ad5ea4b7d779a2176841d3c1c4d4174` (2024-03-19, "Add option for legacy planning metric definition") |
| VAD | `hustvl/VAD@1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8` | `…/VAD/planner/metric_stp3.py:1-336`, `…/VAD/VAD.py:418-435,593-641`, `…/datasets/nuscenes_vad_dataset.py:1210-1230,1795-1813`, `tools/data_converter/vad_nuscenes_converter.py:231-271,349-400,430-459`, `projects/configs/VAD/VAD_base_e2e.py:11,330-355` |
| AD-MLP | `E2E-AD/AD-MLP@4b93ba085ee47474152f282177865796ea577fc0` | `README.md:40-70,99-100`, `deps/stp3/evaluate_for_mlp.py:53-120`, `deps/stp3/stp3/planning_metrics.py:16-148` |
| devkit | `nutonomy/nuscenes-devkit@b40adc467b919192899405d9b77871afee8efa07` | `python-sdk/nuscenes/utils/splits.py:102-150`, `can_bus/can_bus_api.py:50-53`, `map_expansion/map_api.py:94-101`, `docs/schema_nuscenes.md`, `README.md:94-164` |
| OpenCV | `opencv/opencv@4.5.4` | `modules/imgproc/src/drawing.cpp:46-58,80-145,159-297,1258-1471,1984-2014`, `include/opencv2/imgproc.hpp:4832-4912` — the `cv2.fillPoly` every occupancy builder calls (ported, see §6) |

## 1. The two conventions (the only thing the brief calls "the convention")

Per-timestep means `v[0..5]` at `{0.5,1.0,1.5,2.0,2.5,3.0} s` over the pipeline's scored samples:

* **`uniad-noavg` (AT_T)** — UniAD `nuscenes_e2e_dataset.py:1043-1044`: column `i` = `v[i]`.
  1 s / 2 s / 3 s = `v[1]`, `v[3]`, `v[5]`.
* **`stp3-temavg` (AVG_UP_TO_T)** — ST-P3 `evaluate.py:71-73,135-137,166` (three metric instances with
  `n_future = 2,4,6`, reported as `value.mean()`), identical to UniAD's own `stp3` branch
  (`:1041-1042`, `value[:i+1].mean()`): 1 s / 2 s / 3 s = `mean(v[0:2])`, `mean(v[0:4])`, `mean(v[0:6])`.
* **Avg.** (the papers' column) = mean of the 1 s / 2 s / 3 s values, inside one convention.
* The SAME reduction applies to L2 **and** collision (all three codebases reduce every key the same way).

## 2. Analytic targets — L2

Frame: nuScenes LiDAR frame at t0 (x RIGHT, y FORWARD), waypoints `k = 1..6` at `0.5 k` s.
GT: straight, 5 m/s: `gt_k = (0, 2.5 k)`.

| target | prediction | AT_T 1/2/3 s, Avg | AVG_UP_TO_T 1/2/3 s, Avg | half-second columns |
|---|---|---|---|---|
| **A. constant lateral offset** | `gt_k + (1.5, 0)` | **1.5, 1.5, 1.5, 1.5** | **1.5, 1.5, 1.5, 1.5** | all 1.5 (both) |
| **B. linearly growing error** | `gt_k + (0.5 k, 0)` | **1.0, 2.0, 3.0, 2.0** | **0.75, 1.25, 1.75, 1.25** | AT_T `0.5,1.0,1.5,2.0,2.5,3.0`; AVG `0.5,0.75,1.0,1.25,1.5,1.75` |

Closed forms for B (error `e(t) = t` m): AT_T `L2(t) = t`; AVG `L2(t) = (2t+1)/4`. All values are
exact binary fractions — asserted with `==`, not a tolerance.

⭐ **A cannot discriminate the conventions** (both give 1.5). It is kept as the control that proves the
kernel, and B is the discriminating fixture.

## 3. Mutation arm (must go RED)

The source text of `nuscenes_planning.py` is loaded, the reducer dispatch table is mutated so the two
conventions are swapped (the replacement must match exactly once, else the mutation arm itself fails),
and the mutated module is executed. **Target B must FAIL on the mutant; target A must still PASS**.

## 4. Independent real-data cross-check (PUBLISHED — PARA-Drive, banked `paradrive-cvpr2024`, Table 1)

PARA-Drive re-scores ONE VAD checkpoint with and without temporal averaging. Its at-t row (*"+ Remove
averaging over time"*) is the per-timestep vector; its VAD-protocol row is the averaged one.

| input (at-t, per 0.5 s) | AT_T 1/2/3 s, Ave | AVG_UP_TO_T 1/2/3 s, Ave | tolerance |
|---|---|---|---|
| L2 `0.2788 0.5380 0.8239 1.1513 1.5322 1.9821` | **0.5380, 1.1513, 1.9821, 1.2238** | **0.4084, 0.6980, 1.0511, 0.7192** (= the published VAD-protocol row) | 1e-4 |
| Col % `0.10 0.04 0.16 0.39 0.61 1.11` | **0.04, 0.39, 1.11, 0.51** | **0.07, 0.17, 0.40, 0.21** (= the published VAD-protocol row) | 5e-3 |

⭐ This is the independently-authored target the CLAUDE.md rule asks for: the paper's own two rows are
related by exactly the reduction the harness implements, and it also proves VAD's collision is
time-averaged.

## 5. Collision kernels — literal cells

Ego footprint (all three codebases): `W = 1.85`, `H = 4.084`, corners
`(-H/2+0.5, ±W/2)…(H/2+0.5, ±W/2)`, grid `[-50, 50, 0.5]` → 200 × 200, `bx = -49.75`, `dx = 0.5`.
**Rasterised ego box `rc` = rows 97..104 × cols 98..101 = 32 pixels** (the reference's own comment
reads `# (n_future, 32, 2)`, `planning_metrics.py:64`).

Waypoint `(x = 0, y = 10)` (LiDAR frame), obstacle cells per kernel (timestep index 3 only):

| kernel | box rows × cols at the waypoint | point-check cell | lateral sign: waypoint `(x=-2, y=10)` box cols |
|---|---|---|---|
| **UniAD** (grid rows = forward, cols = RIGHT) | 117..124 × 98..101 | (119, 99) | 94..97 |
| **ST-P3** (grid rows = forward, cols = LEFT) | 117..124 × 98..101 | (119, 99) | 102..105 |
| **VAD** (grid rows = −forward, cols = right) | 76..83 × 98..101 | **(29, 49)** ⚠️ | 94..97 |

* Known overlap: occupied cell inside the box at t = 3 → `obj_box_col[3] = 1`, all other t = 0.
* Known miss: occupied cell one column outside → 0 everywhere.
* GT exclusion: an obstacle hit by BOTH the GT box and the predicted box at t → `obj_box_col[t] = 0`
  (excluded) while the raw `gt_box_col[t] = 1` (the GT-control floor — PARA-Drive fn. 4).
* ⚠️ **VAD's point check reads cell (29, 49) for a waypoint at (0, 10)** — `metric_stp3.py:272-273`
  compute `(-bx/2 ∓ …)/dx` = `49.75 − 2y`, `49.75 + 2x` instead of `100 ± 2·`. Verbatim means the
  harness reproduces it; the published VAD column is `obj_box_col` (the box check), not `obj_col`.

## 6. `cv2.fillPoly` port (LINE_8, shift 0) — literal masks

| polygon (x, y) | grid | filled |
|---|---|---|
| rectangle (2,3),(5,3),(5,6),(2,6) | 10 × 10 | rows 3..6 × cols 2..5 (16 px) |
| diamond (2,0),(4,2),(2,4),(0,2) | 5 × 5 | row0 {2}; row1 {1,2,3}; row2 {0..4}; row3 {1,2,3}; row4 {2} (13 px) |
| clipped rectangle (-3,-3),(2,-3),(2,2),(-3,2) | 5 × 5 | rows 0..2 × cols 0..2 (9 px) |

Parity with a real `cv2` is **NOT RUN** on this box (no OpenCV in the venv; installing is out of
scope) — the test exists and skips with that reason; a skip is reported as NOT RUN, never as a pass.

## 7. Sample sets (which frames are scored)

| pipeline | rule (source) | 40-sample scene | 40 + 39 scenes | full val (6,019 samples / 150 scenes) |
|---|---|---|---|---|
| UniAD | every sample; missing future timesteps masked to 0 error, denominator counts the sample (`planning_metrics.py:125,142`; `trajectory_api.py:262-270`) | 40 | 79 | **6,019** |
| VAD | `fut_valid_flag`: 6 future samples exist (`vad_nuscenes_converter.py:248-254`; dataset mean over valid only `:1799-1812`) | 34 | 67 | **5,119** = 6,019 − 6·150 (BEV-Planner states 5,119 — PUBLISHED) |
| ST-P3 / AD-MLP | 3 past + 6 future samples in the same scene (`NuscenesData.py:124-148`, `TIME_RECEPTIVE_FIELD 3`, `N_FUTURE_FRAMES 6`); scenes 419 + CAN blacklist removed | 32 | 63 | **4,819** = 6,019 − 8·150 (AD-MLP §3.3 states 4,819 — PUBLISHED) |

No val scene is in the CAN blacklist or 419 (MEASURED from `splits.py` @ b40adc4: 150 val scenes).

## 8. GT trajectory (common to all three) — synthetic-metadata literals

GT = the LIDAR_TOP sensor ORIGIN at future keyframes, in the LiDAR frame at t0 (ST-P3
`get_gt_trajectory`, UniAD `get_sdc_planning_label`, VAD converter `:431-450`). Synthetic rig: LiDAR at
`(0.943, 0, 1.84)` in the ego frame, yaw −90° (x_lidar = right, y_lidar = forward).

* **Straight** ego at 4 m/s along global +x: GT = `(0, 2k)` for k = 1..6 (the lever arm cancels).
* **Left arc** R = 20 m, 5 m/s, ψ_k = 0.125 k:
  `GT_k = ( −(R(1−cos ψ_k) + a·sin ψ_k), R·sin ψ_k + a·cos ψ_k − a )`, a = 0.943.
  k = 6: `x = −6.009008` (20·(1−cos 0.75) = 5.366223, 0.943·sin 0.75 = 0.642785) → **command = LEFT**
  (x ≤ −2, the ±2 m GT-derived rule). *(Pre-registration erratum, made before any test ran: the first
  draft of this line read −6.009016 — an arithmetic slip; the closed form above is the target.)*

## 9. Input adapter geometry — nominal CAM_FRONT into the cylindrical training frame

`NUSCENES_CAM_FRONT_INTR_NOMINAL` (calib.py:1286-1289; ESTIMATED until per-sample intrinsics are read)
into `PHYSICALAI_WIDE120_256x640` (calib.py:1247-1248), closed form computed independently of calib.py:

| quantity | target |
|---|---|
| camera HFOV from the nominal matrix | **64.561°** (the nuScenes paper quotes a nominal **70°**, 1903.11027 p.4) |
| observed columns (centre row) | **145..488 = 344 of 640** |
| observed rows (column 319) | **9..225 = 217 of 256** |
| observed pixels | **70,737 of 163,840 → observed_frac 0.431744** |
| same, rig-clean slice `[40:216, 8:632]` (176 × 624) | **60,399 of 109,824 → 0.549962** |

The mapping is a RAY resample (`calib.pinhole_rectify(..., frame=…)`), never a resize; the stamp
(frame tag, observed_frac, observed column/row span) is mandatory on every model arm.

## 10. Plan → LiDAR frame (model output conversion)

Model knots in OUR ego frame (x forward, y left) at `(0.5,1,1.5,2,3,4,5,6) s` (E2 `KNOT_T_S`) →
C2 cubic spline in time through the origin (E2's declared method) → 6 waypoints → LiDAR origin via the
lever arm. Targets: straight plan → `(0, 2.5 k)` exactly; the §8 arc sampled at the knots → equals the
§8 GT within **0.01 m** (spline + tangent-heading error); a plan equal to GT scores **L2 = 0**.

## 11. What would falsify the harness

Any literal above failing; the mutation arm passing; the §4 cross-check missing by more than its
tolerance; on first real data, the sample counts differing from 6,019 / 5,119 / 4,819.
