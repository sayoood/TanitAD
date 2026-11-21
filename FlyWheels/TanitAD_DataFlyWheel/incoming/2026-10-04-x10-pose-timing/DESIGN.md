# X10 — pose-to-image timing: design (Data FlyWheel, 2026-10-04)

*Brief: Master Mind work item X10 (SPEC_REFCV8_DRAFT §8.10, Q10). Base for every file reference below: lander tip
`agent/arch-inf-20260803` @ `acc6180` (the D: working tree is OLDER for `v2_dataset.py` and `refc_v3_train.py`; the
tip blob was used). Line numbers are tip lines. Evidence classes: MEASURED (artifact path) · DERIVED (from
source) · UNVERIFIED. CPU only; no Thor, no GPU, no G:. Clip ids appear only as sha12.*

## 0. Verdict in five lines

1. **The defect is real and exactly as D3 stated** — MEASURED: image-minus-pose time is a sawtooth in [0, 35.4] ms with
   mean **16.50 ms** (`raw/x10_measure.json`: eval139 16.505 ms, n 27,664 rows; train 4,369 clips 16.502 ms, n 869,278).
2. **It enters in ONE place**: `v2_compressed.py:119-124` (twin `physicalai.py:717-735`) — poses are sampled at
   `t_query`, the image is the first camera frame **at or after** `t_query`.
3. **The cache cannot be corrected from itself** — it stores poses, never the camera timestamps
   (`v2_compressed.py:150-158`). The fix needs a 5.4 MB sidecar built from the 4,508 corpus clips' `timestamps.parquet`
   (all present locally; **built: 4,508/4,508**). No cache rebuild.
4. **The literal reading of the brief — "interpolate the pose track to each camera frame's timestamp", per row — is the
   WRONG correction for the targets** (§4): it converts a second-order bias into a first-order timing jitter that is
   **~10× larger than the defect** (MEASURED, §5). The right one is ONE shift per window, the NOW row's.
5. **The effect on the training targets is small** (0–2 s mean **0.009 m**, 6 s mean **0.060 m**, p95 0.21 m; v0 mean
   **0.008 m/s**) and zero-mean — label noise, not bias. It is implemented, opt-in, default OFF, bit-identical when off.

## 1. Where the ego trajectory targets and the per-frame ego state come from (trace)

| step | what happens | file:line (tip) |
|---|---|---|
| raw camera clock | one `timestamp` (µs, int64) per camera frame in `<clip>.timestamps.parquet`; ~30 Hz (33,333 µs), 605 frames per 20 s clip, first frame −133 ms…0 | `v2_compressed.py:111-113` (MEASURED: `raw/x10_measure.json` `corpus_delta`; first parquet inspected in-session) |
| raw pose clock | egomotion `timestamp` (µs), ~100 Hz (median step 10 ms; a few clips irregular), **same clock** as the camera — the pipeline's own assumption (`physicalai.py:598` "same clock") | `physicalai.py:596-640` `signals_at` = `np.interp(t_query, t, col)` (`:616`) |
| the 10 Hz grid | `n_target = int(span/unit*10)`; `t_query = linspace(t_frames[0], t_frames[-1], n_target)` → step `span/(n-1)` = **100.667 ms**, not 100.000 | `v2_compressed.py:119-120` (twin `physicalai.py:717-718`) |
| **the image** | `frame_idx = searchsorted(t_frames, t_query)` — side='left' → the first frame with `t_frame >= t_query` | `v2_compressed.py:121-123` (twin `physicalai.py:719`) |
| **the pose** | `actions, poses = signals_at(ego, t_query)` — the egomotion **at `t_query`**, not at `t_frames[frame_idx]` | `v2_compressed.py:124` (twin `physicalai.py:735`) |
| what the cache keeps | `poses`, `actions`, jpeg/png of the `frame_idx` frames. **Not** `t_frames`, not `frame_idx`, not the offset | `v2_compressed.py:150-158` |
| provider rows | `poses[n_stack-1:]` (the first 2 raw rows are dropped; `raw_offset = 2`) | `v2_dataset.py:381`, `v2_compressed.py:197`, `refc_v3_train.py:3985` |
| B1 cache provenance | the refcv6/7 cache is built by `pod_build_b1_epcache.py:134` → `build_compressed` → `_resampled` — the same function | `pod_build_b1_epcache.py:10,134` |
| window item | `pose_last = ep.poses[t+w-1]` (NOW), `future_poses = ep.poses[t+w : t+w+20]` | `_contract.py:136-137`; window 8 / max_horizon 20 → `t_max = T-28` (`:120`) = 171 windows/clip |
| refcv extras | `future_poses_ext = ep.poses[idx.clamp(T-1)]` (60 rows), `pose_hist = ep.poses[t:t+w]` (if `--ego-history`), `goal_tac` | `refc_v3_train.py:4084-4102` |
| **trajectory target** | `waypoint_targets(pose_last, fut_ext, horizons)` = `ego_frame(xy[t+k] - xy[t], yaw[t])` at ticks `(5,10,15,20,30,40,50,60)` | `refc_v3_train.py:4863` (r8 copy `:4849`), `refb_labels.py:210-223` |
| **v0** | `v0 = pose_last[:, 3]` → `model(..., v0=v0)` (a MODEL INPUT; PI ruling: velocity at cycle time is admissible) | `refc_v3_train.py:4725` |
| other readers of the batch poses | kinematic tactical labels `window_factored_labels(pose_last, fut_ext[:, :20])`; r8 teacher; ego-history encoder (`set_ego_window(pose_hist)`, reads speed / dv / yaw-rate, **not position**) | `refc_v3_train.py:4916, 4851, 4811`; `ego_history.py:74-101` |
| the v9 join key | `(sid, k = t + w - 1 + raw_offset)`, with `now_s` re-derived by `_now_s` (grid-start + dt clock) and **checked to 1e-6 s** | `refc_v3_train.py:4004-4025` |
| the box / agent join's clock | per-track nearest sample to `t_s`, where `t_s` = the cache poses **registered back onto the egomotion log** (`register_poses_to_time`) — i.e. the **pose clock `t_query`**, not the image's | `build_b1_agent_join_3d.py:231,253` |

**Where the offset enters.** At `v2_compressed.py:121` vs `:124`: the *image* index is a ceil onto the camera grid, the
*pose* time is not. Because the camera period (33.333 ms) and the grid step (100.667 ms = 3 × 33.333 + 0.667 ms) beat
against each other, `delta_k = t_image − t_pose` is a **deterministic sawtooth that falls 0.667 ms per row and wraps
every ~50 rows (5 s)** (MEASURED: first sidecar row `[0, 32667, 32000, 31332, 30665, …, 16, 32682]` µs). It carries
**no scene information**, so it is zero-mean label noise to the learner (signed longitudinal mean of the resulting target
shift: −0.0035 m eval / +0.0013 m train at 6 s).

## 2. What D3 measured, and the independent derivation

D3's `RESULT.md` carries one sentence for this (*"the image is the next camera frame 0–34 ms later (mean 16.5 ms) → a
longitudinal offset of 0.19 m mean, up to ~0.9 m at p95 clip speed"*); **its derivation script / artifact was not banked**
(`D3_raw_plausibility/scripts` has no timing script; `raw/` has no timing file). So the numbers were re-derived, not copied:

| quantity | D3 (INHERITED) | this package (MEASURED, `raw/x10_measure.json`) |
|---|---|---|
| offset range, mean | 0–34 ms, 16.5 ms | eval139: 0–34.02 ms, **16.505** ms (n 27,664); train: 0–35.43 ms, **16.502** ms (n 869,278) |
| mean position offset | 0.19 m | NOW-pose shift, trainer windows, eval139 (n 23,772): **0.193 m** mean (0.186 m against the 100 Hz log, n 27,664) |
| "~0.9 m at p95 clip speed" | a product (δ_max × p95 speed ≈ 0.034 s × 27 m/s) | the JOINT distribution: p95 **0.56 m**, p99 0.84 m, max 1.18 m |

Independent routes used (all agree; routes 1-3 numeric, route 4 pixels):
1. **Reconstruction of the cache from raw logs**: the cached `poses` equal `signals_at(ego, linspace-grid)` rebuilt from the
   raw camera timestamps + raw egomotion **bit-exactly** — max |Δ| xy 0.0 m, yaw 0.0 rad, v 0.0 m/s over **539 clips**
   (139 eval + 400 seeded train; `reconstruction_of_cached_poses_from_raw_logs`). So the `t_query` this package uses IS the cache's.
2. **The clip-clock sidecar** (`refcv6_clip_clock_sidecar.jsonl`), derived by inverting the *poses* onto the 100 Hz log (no
   camera timestamps involved): `grid_start_s` agrees with `(t_frame[0] − t_ego[0])` to **≤ 4.3e-4 ms**, `dt_s` to **≤ 3.8e-9 s**
   (n 535). Full corpus K4 in the builder: 4,483 clips, worst |Δdt| 6.2e-9 s.
3. **A closed form from four numbers** (`pose_sync.analytic_delta_s`: first/last timestamp, frame count, `n_target`;
   ideal constant-period camera): per-clip max deviation from the measured sawtooth, modulo one frame period, median
   **53 µs**, p95 1.9 ms, max 2.06 ms (539 clips; 656 of 108,315 rows sit on a ceil-vs-searchsorted boundary tie).

4. **Pixels** (`code/x10_pixel_check.py`, `raw/x10_pixel_check.json`): for 4 rows on 2 eval clips chosen with `delta` 24-31 ms (where a
   "nearest frame" cache would hold `frame_idx - 1`), the cached image was compared with the builder's own decode + cylindrical remap of
   raw frames `frame_idx - 2 ... + 2`. **All 4 rows best-match offset 0** (mean abs error 0.72-0.82 grey levels vs 1.54-10.4 for the best
   neighbour, 2.1x-14x). The match is not bit-exact (PNG cache; the cause - decoder threading / colour conversion in my re-decode - was not
   chased), and n = 4 rows is small; it rules out the nearest-frame alternative, which is the point.

UNVERIFIED (named, not assumed): (a) that the decoded pixels of row `k` are camera frame `frame_idx[k]` for ALL rows - MEASURED on 4 rows
only (route 4), the rest rests on the builder code; (b) what instant the provider's camera `timestamp` stamps
(start / mid / end of exposure; a rolling-shutter constant offset would be a *bias*, not a sawtooth, and this
correction cannot see it); (c) that the camera and egomotion clocks are synchronised below 1 ms in the *dataset*
(the pipeline assumes it; the exact reconstruction in (1) shows only that the pipeline uses it consistently).

## 3. What the correction changes — and what it deliberately does not

The correction re-samples a window's ego state at the instant the NOW image was captured:
`pose'(row j) = pose(t_query[j] + delta_NOW)`, one shift per window (`tanitad/data/pose_sync.py`).

| consumer | changes? | how much (MEASURED, eval139 n 23,772 windows, `raw/x10_dataset_e2e.json`; train-sample in `raw/x10_measure.json`) |
|---|---|---|
| **trajectory targets** `waypoint_targets` | yes | 0–2 s pooled (ticks 1–20, n 475,440): mean **0.0093 m**, p50 0.0036, p95 0.038, p99 0.074, max 0.28. 6 s (tick 60, n 18,351): mean **0.0597 m**, p95 0.210, p99 0.389, max 0.754. Train sample (400 clips): 0–2 s mean 0.0104, 6 s mean 0.0681 / p95 0.236 |
| **v0** `= pose_last[:, 3]` | yes | mean **0.0084 m/s**, p95 0.033, max 0.122 (= a·δ; the speed channel at the image instant) |
| `pose_hist` (ego-history input) | yes (shifted) | the encoder reads speed / dv / yaw-rate: |Δspeed| mean 0.0085 m/s (p95 0.034), |Δyaw-rate| mean 0.0006 rad/s (p95 0.0024); the 0.196 m absolute-position shift of the history is invisible to it |
| `goal_tac` targets | yes | xy shift mean 0.037 m (p95 0.139) |
| `actions` / `future_actions` | yes (so `a0`, curvature ride the same clock as `v0`) | max-abs per window mean 0.056 (accel-dominated) |
| kinematic factored tactical labels (`refc_v3_train.py:4916`) | flips at class boundaries | lat 28 / 23,772 (0.12 %), lon 66 / 23,772 (0.28 %) |
| **v9 label join** `(sid, k)` + `now_s` | **NO** | rows are not renumbered, `_now_s` / `_raw_offset` untouched (pinned by `test_dataset_known_value_waypoint_shift_and_row_indices_unmoved`); the v9 labels are keyed on the label clock `grid_start_s + (k + 2)·dt_s`, whose own 34 ms vs a ±2 s band edge is immaterial |
| `nav_cmd`, route labels, `lan`, agent-future transform, agent/3D/map joins | **NO** | read `ep.poses` through other layers or are keyed on the pose clock; coarse or oracle labels (see §7 for the box join) |
| frames, row count, masks (`future_valid_ext`) | NO | tail rows are linearly extrapolated by ≤ 0.34 row so validity is unchanged; the extrapolation is inside the measured interpolation error (pose at NOW: mean 0.5 mm, p95 1.8 mm, max 4.9 mm vs the 100 Hz log) |

**Why the target shift is small although the position offset is 0.19 m.** A trajectory target is a *displacement* from the
NOW pose in the NOW frame. Sliding the whole window by δ leaves a constant-velocity displacement unchanged exactly
(translation invariance; pinned by `test_constant_velocity_targets_do_not_move`) and changes an accelerating one by
`a·δ·H` (pinned: 0.033220 m for a = 2 m/s², 10 ticks). The 0.19 m is an *absolute registration* error — it matters to
anything that puts a pose on the image, **not** to ego-relative targets. That distinction is the main finding.

## 4. The least invasive correction, and why not the others

| option | what it is | verdict |
|---|---|---|
| **B. window shift (chosen)** | one shift per window = the NOW row's `delta`, applied to `pose_last`, `future_*`, `pose_hist`, `goal_tac`, `actions` by linear interpolation of the cached 10 Hz track; sidecar of `delta` per raw row | elapsed times stay exactly `h·dt`; anchor is the true image instant; ~110 lines in the dataset; no cache change |
| A. per-row shift ("interpolate the pose track to each camera frame's timestamp") | `pose'(j) = pose(t_query[j] + delta_j)` for every row, as `ep.poses` | **REJECTED, MEASURED:** waypoint `h` then spans `h·dt + delta_{r+h} − delta_r`, a ±33 ms jitter: target moves vs the uncorrected one by mean **0.109 m** (p95 0.360) at 0–2 s and 0.130 m pooled over 0–6 s — **~11× the 0.0093 m the defect is worth**; vs the correct B it is 0.108 m. Test `test_REGRESSION_per_row_shift_injects_first_order_timing_jitter` (literal 12 m/s × 20 ms = 0.24 m) |
| C. cache rebuild with poses sampled at `t_frames[frame_idx]` | the "obvious" builder fix | **is option A baked in** → same jitter, plus 4,508 clips × a 3.4–4.6 h job, and the cache is shared with the warm start |
| D. cache rebuild choosing the *nearest* frame | shrinks `|delta|` to ≤ 16.7 ms (mean 8.3 ms) | still a rebuild; leaves a residual; not preferred |
| E. fix at inference | — | not applicable and **forbidden**: this is a training-target / label-time correction; it adds no inference input |

The sidecar (`refcv6_pose_sync_sidecar.jsonl`, JSON-lines `{sid, dt_s, n_rows, delta_us[]}`, **no clip id**, keyed on
`stable_episode_id`) is built by `scripts/build_pose_sync_sidecar.py` from `timestamps.parquet` only (K1 synthetic-camera
self-test; K2 refuses a clip whose rebuilt grid is not the cache's row count; K3 sid matches `episode_uid`; K4 `dt_s`
equals the independent clock sidecar's to 1e-6 s). Reader refusals: negative or > 50 ms `delta` (sign / unit error),
length ≠ `n_rows`, `dt_s` outside the grid band, duplicate sid with different content, zero rows. The dataset method
`V3Dataset.enable_pose_sync` additionally refuses a sidecar that covers none of the split, one with > 1 % uncovered clips,
and any covered clip whose row count is not `len(poses) + raw_offset`.

Integration (all opt-in, default OFF): `--pose-sync-sidecar PATH` on `refc_v3_train.py`; with it absent the item's key set,
values and RNG consumption are byte-identical to the tip (tests pin this). `config.json` gets `pose_sync` (None = the
offset is UNCORRECTED, never a silent default). `declared_vs_built` registers the new dest.

## 5. Effect on refcv8's warm start from refcv7-50,400

* **Targets do not enter the forward pass.** A changed target is therefore harmless at step 0: the forward is identical for
  identical inputs, and only the loss values and gradients differ — by cm-scale amounts (table §3).
* **Inputs that do change at step 0** (and therefore the step-0 forward): `v0` (mean 0.0084 m/s = 0.03 km/h, p95 0.033 m/s,
  max 0.12 m/s = 0.44 km/h), and — only because refcv7 ran `--ego-history` — the ego-history encoder's speed / dv / yaw-rate
  channels (above). The warm start therefore sees a sub-0.05 km/h input perturbation, not a distribution shift (it was
  not forward-tested on the checkpoint: the checkpoint is not on this box's reach; UNVERIFIED beyond the input arithmetic). `actions` enter the forward only through
  `ego_state_inject` (OFF in refcv7) or `residual_prior ha0_ext`; they ride the same clock so `v0` and `a0` stay mutually consistent.
* The frozen/warm weights were trained on the **uncorrected** clock; the correction moves the *target* by far less than
  the model's own error (§6), so it is not a distribution shift for the heads either. It is, however, a second changed
  lever versus refcv7 — see the decision in `RESULT.md`.

## 6. Eval-side note (flagged, not decided)

`train()` now also calls `enable_pose_sync` on the in-training eval dataset (same flag), so train and eval ride one clock.
The **standalone** refcv7 eval kit builds its own `V3Dataset` (`tanitad/eval/refcv7_loader.py:787`); it is NOT changed. If
refcv8 is scored there on the uncorrected clock, the GT differs from the training target by the §3 amounts (≈ 1 cm mean at
2 s) — below the rig's ≈ 0.30 m seed floor — and refcv8 stays directly comparable to every banked refcv7 number. Adding the
same three lines there is the Master Mind's call.

## 7. What X10 does NOT fix (named so it is not assumed fixed)

* **Box / agent labels.** The 2-D/3-D box join takes per-track nearest samples at the *pose* clock (`build_b1_agent_join_3d.py:253`),
  so boxes are as stale vs the image as the poses were (a static object is mis-placed by `v_ego·δ`, mean 0.19 m); it also has its
  own nearest-sample rounding (up to half a label period). The same sidecar could re-time that join; it needs the join rebuilt —
  out of scope here (a cache-adjacent artifact shared with the warm start).
* **SAM3 map GT** time base: UNVERIFIED (not traced).
* **Any constant camera-timestamp offset** (§2 (b)).

## 8. Files (see `LANDING_READY_X10.txt`)

New: `stack/tanitad/data/pose_sync.py`, `stack/scripts/build_pose_sync_sidecar.py`, `stack/tests/test_pose_sync.py`.
Modified (built on the tip blob): `stack/scripts/refc_v3_train.py` (+109 / −0 lines, CRLF preserved),
`stack/tanitad/train/declared_vs_built.py` (+1 / −0, LF), `stack/tests/test_declared_vs_built.py` (+2 / −0, LF: the registry size
is PINNED at 256 and the new dest makes it 257 — the one regression the neighbour run found, now fixed; any other refcv8 stream
that adds a trainer flag edits the same line, so land the flags one at a time). Base blobs and md5s in the landing file.
