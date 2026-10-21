# NT-DECIDE E1 ⇄ trainer: the data interface for arm A1 (`H-NT-DECIDE-1`)

DataFlyWheel (E1 writer + download), 2026-10-21, Berlin. **This file is the contract.** The trainer-side agent owns labels (E2/E3), argv, the launch gate and the controls; E1 owns the token list, the frames, the poses and the `*.v2ep.pt` files described here. If this file and the code disagree, the code is wrong and is fixed to match this file.

Evidence classes: **MEASURED** (this session, artifact named) · **READ** (from source at `fb47a58`, `file:line`) · **DECLARED** (a choice made here).

---

## 0. Two things the trainer MUST do. Without them the files cannot be trained on. ⛔

The consumer's loader is the spec (`stack/tanitad/data/v2_dataset.py` + `stack/scripts/v2_compressed.py::build_compressed` at `fb47a58`, byte-identical on Thor: md5 `870066f2…`, MEASURED). Read literally, that loader **cannot hold an ST token-episode at any affordable size**:

* Under ST, all 10 window slots hold the t0 stitch, and the 20 future slots hold the LAW frame (§3).
* In the `fb47a58` format, raw frame `i` is `jpeg_buf[offs[i]:offs[i+1]]`, and `offs` is the prefix sum of `jpeg_len` (`v2_dataset.py:102-106, 136-149`). A frame cannot be referenced twice.
* One window (`t = 0`, `max_horizon = 20`) decodes raw rows 0…29. So every token would need **30 PNG copies**: 30 × 538,884 B (`PNG_PROBE.json`) ≈ **16.2 MB per token**, which is **≈ 1.61 TB** for 99,824 tokens. Thor has 181 GB free (MEASURED 20:57, after a real 4 GiB write test).

**R1 — the loader must accept an optional `frame_index`.** E1 stores each unique frame ONCE, plus `frame_index: int64 [30]`, which maps raw row → index into `jpeg_len`.
* The patch, its tests and their results are in `nt_build/data/loader_patch/` (§8). It changes about 20 lines of `v2_dataset.py`.
* **If the key is absent, the old lines run unchanged.** Every existing cache stays byte-identical, and the tests show it on real PhysicalAI `*.v2ep.pt` files.
* An **unpatched** loader opening an E1 file **fails loud** on the first decode, because `jpeg_len` has 1–2 entries and the decode asks for raw row ≥ 2. It never returns a wrong frame.

**R2 — exactly ONE window per token-episode: `t = 0`.** `EpisodeWindowDataset` indexes `t in range(T_out − window − max_horizon)` (`_contract.py:119-121`; `refb_train.py:205` via `build_window_index`). `FailLoudWindowDataset` refuses `len(frames) != len(poses)` (`refb_train.py:188-193`). And `future_valid_ext = idx <= T − 1` (`refc_v3_train.py:4519-4521`).
* So a token-episode that carries the full 6.0 s target needs `T_out` = 8 + 60 = **68 provider rows**.
* That gives 68 − 8 − 20 = **40 windows** per file. Only `t = 0` is the token: its NOW is t0, its frames are the ST feed, and its 60-tick future is the target.
* ⇒ **Right after `V3Dataset(...)` is constructed, and before any `enable_*` join walks `self.index`, set `ds.index = [(e, t) for (e, t) in ds.index if t == 0]` for the NavSim training dataset only.** Then assert `len(ds.index) == len(eps)`.
* The eval139 PhysicalAI forgetting monitor keeps its own unchanged dataset.
* ⛔ **This cannot be done on the E1 side.** Trimming `poses` to 31 raw rows would make `t = 0` the only window, but `future_valid_ext` would then be valid only to 2.1 s. That loses the 2.5–6.0 s targets.
* Fail-loud backstop: `frame_index` has exactly **30** rows, so any window `t ≥ 1` raises `IndexError` on its first decode. An unfiltered run dies on its first batch, since 39 of every 40 windows are `t ≥ 1`. It never trains on a wrong window. The launch-time assert above is still the right place to catch it.

---

## 1. The token list (E1 owns it) — DONE, MEASURED 2026-10-21 21:01 Berlin

| item | value |
|---|---|
| file (Thor) | `/home/nvidia/r8ns/nt/TOKENS_A1.json` (file sha256 `3e75e94a8dbc03f1bb496da6dcd87d63e0c1d0deb077dd0c7f25c7d7a0724453`) |
| tokens / logs | **99,824 / 1,154** (= DESIGN §3 expectation, `matches_expected: true`) |
| sorted-token sha256 | `7399461ac87cff4edc164dfa315556f442f26bcbe1cab58f63be2ba2cf1ff25e` (`navtrain_common.sha256_tokens`: newline-joined, sorted) |
| universe | official `navtrain.yaml` (sha256 `c37fea56…`, 103,288 tokens / 1,192 logs). Every token was located exactly once in its trainval log pickle, and each has a devkit scene window (4 history + 10 future, roadblock ids at t0). |
| held out BY LOG | all **38** selection-scan logs (`NAVTRAIN_SAMPLE_TOKENS.json` `per_log`, sample sha `9dd68698…`): 3,464 navtrain tokens.<br>Every navtest log (`navtest.yaml` sha256 `61284edf…`, 136 logs / 12,146 tokens). |
| **K3 positive assertion** | `train_logs ∩ selection_logs == set()` → **True** (1,154 vs 38)<br>`train_logs ∩ navtest_logs == set()` → **True** (1,154 vs 136)<br>Token level: pool ∩ sample tokens (2,089) = 0; pool ∩ navtest tokens (12,146) = 0. |
| deliberate regression | adding one selection log → **REFUSED**; adding one navtest log → **REFUSED** (`nt_tokens.py --` self-test, run inside every build) |
| schema | `status: FIXED`, `tokens`, `count`, `sha256_sorted_tokens`, `token_log`, `log_names`, `held_out`, `K3_exclusion`. These are the fields `export_navtrain_inputs.load_sample` needs, so the r8ns exporter runs on it unchanged. |
| per-token index | `/home/nvidia/r8ns/nt/TOKEN_INDEX_A1.json.gz`: `{token: {log, pos, n_log_frames, window_ok, next_is_navtrain_token, next_token}}` |
| devkit inputs | `/home/nvidia/r8ns/nt/inputs/a1_inputs.json.gz` (sha256 `66eaaeb80d9a31e5da64a80f0abf01fb9e559cb940ddf2cb3d29eddcf4bd98fa`, 99,824 records, missing 0, unexpected 0).<br>Made by the r8ns agent's `export_navtrain_inputs.py`, **run unchanged** (navsim v1.1 @ 3e8291b, devkit `SceneLoader.get_agent_input_from_token`): W3's per-token record (`ego_statuses` ×4, `driving_command`, `frame_tokens`, `timestamps_us`, `cams`, `human_future_poses` 8 × 0.5 s, `cv_poses`, `map_name`, `scene_token`). **E2 should read `driving_command` from here or from the payload's `nt` block (§4); both are the same export.** |

LAW-frame availability (MEASURED from the index): for **82,314 / 99,824 = 82.46 %** of tokens, the next 2 Hz frame is itself an A1 token, so its stitch is in `navtrain_current`. For the other 17.54 % the t0 + 0.5 s frame is not a `navtrain_current` frame (filter gaps and log ends), and the LAW slot falls back to t0 (§3; counted per file in `nt.law_src`). The history shards (+144 GB) would recover part of it; they are **not** downloaded (not authorised).

## 2. Files and where they live (Thor)

| what | path |
|---|---|
| training cache dir (flat; pass to `--v2-cache`) | `/home/nvidia/r8ns/nt/v2ep_a1/` |
| one file per token | `/home/nvidia/r8ns/nt/v2ep_a1/navtrain-<token>.v2ep.pt`.<br>`clip_id` = the filename stem = **`navtrain-<token>`** (DECLARED: a hyphen, not the DESIGN's `navtrain:`; a colon is illegal on Windows / exFAT copies, and the stem must equal `clip_id`). |
| join key | `sid = v2_dataset.stable_episode_id("navtrain-<token>")`. The manifest carries it, and collisions are checked to be zero over the whole set. |
| K1 control files (selection-sample tokens; **never** in the training dir) | `/home/nvidia/r8ns/nt/k1_sel/` |
| per-token manifest | `/home/nvidia/r8ns/nt/MANIFEST_A1.jsonl` (§5) |
| heartbeat | `/home/nvidia/r8ns/nt/PROGRESS.json`, updated per shard (§5) |
| clock sidecar (optional, DECLARED clock of these files) | `/home/nvidia/r8ns/nt/CLIP_CLOCK_A1.jsonl` (+ `.meta.json`, schema `tanitad.clip_clock/1`): `{"sid", "clip_id", "grid_start_s": 0.0, "dt_s": 0.1}` |
| rig table | `/home/nvidia/r8ns/nt/RIGS_A1.json`: `{rig_key: frames416.rig_record}` (CAM_F0 mount, virtual-camera rotation, F0 intrinsics), plus `rigs/<rig_key>.src.npy` (int8 `[416,1024]` camera index per pixel, −1 unobserved; for the vis1 sidecar) |
| completion marker | `/home/nvidia/r8ns/nt/DONE_A1.json`, written last, only when every token is written and verified or listed in `MISSING_A1.json` with its reason |

`_v2manifest.pt` is built at the end by the **patched** loader (`load_or_build_manifest`). An unpatched loader also builds it (it reads poses and meta only); the content is the same, version 3.

## 3. The raw timeline of one token-episode (DECLARED; every number below is a literal the controls check)

Raw row `r` ↔ time **τ_r = (r − 9) × 0.1 s** relative to the token's t0 frame. NOW is raw row **9** = provider row **7** = `t + window − 1` at `t = 0` (`_raw_offset = 2`, `refc_v3_train.py:4290-4294`).

| raw rows | τ | frame (via `frame_index`) | poses `(x, y, yaw, v)` |
|---|---|---|---|
| 0–1 | −0.9, −0.8 | t0 stitch | LINEAR interpolation of the devkit ego states (stack padding: the provider drops these two rows) |
| 2–9 | −0.7 … 0.0 | t0 stitch (ST) | **exactly** `refcv8_bridge.ego_history_poses(declare8(ego_statuses, "R8_LEGAL"), times_rel_t0(rec))`, IMPORTED and called, so `pose_hist` = the inference feed (K1, ≤ 1e-6) |
| 10–29 | +0.1 … +2.0 | **LAW frame** = the stitch of the t0 + 0.5 s frame (the next token's current frame); t0 stitch if unavailable | not-a-knot C2 cubic spline (§3.1) |
| 30 … `T_raw − 1` | +2.1 … ≤ +6.0 | no frame (`frame_index` has 30 rows) | spline |

* **Future validity.** `T_raw = 10 + n_future_ticks`. `n_future_ticks` is the number of 0.1 s ticks up to the last usable 2 Hz future frame (at most 60, i.e. 6.0 s). That makes `future_valid_ext` exactly right with no new key.
* **A devkit-kept token always has ≥ 10 future frames (5.0 s)**, so `T_raw ≥ 60` when the log has no dropped frame.
* A dropped 2 Hz frame (|Δt − 0.5 s| > 0.05 s; MEASURED cases are Δt = 0.999–1.000 s) ends the usable future at the frame before the gap (`nt.future_gap_truncated`).
  * A token with fewer than **21** usable future ticks cannot form its one window. It is **not written**: it goes to `MISSING_A1.json` with reason `poses: … usable future ticks`.
  * MEASURED on shard 1: 8 of 458 tokens (1.7 %).
  * ⚠️ The devkit's own GT is index-based and silently treats the gap as 0.5 s. E1 does not copy that error into the target.
* A dropped frame inside the 1.5 s history moves the first spline knot forward to the gap (`nt.history_knot_start` ∈ {−3…0}). The 10 Hz history rows are still the bridge's function at actual times.
* **Stacked LAW target.** The LAW target row is future row `LAW_AHEAD − 1 = 4` (`refc_train.py:114`; `refc_v3_train.py:5583`) = stacked raw `[12, 13, 14]` = **LAW, LAW, LAW**, which is the ST form of the t0 + 0.5 s frame.
* Future rows 0–1 (raw `[8,9,10]` and `[9,10,11]`) mix t0 and LAW. Rows 5–19 repeat LAW. ⚠️ **No consumer other than LAW may read `future_frames`.** The trainer-side controls must prove that (DESIGN §2).

### 3.1 Pose conventions

* **Frame:** the token's **t0 ego frame**, x forward, y left, yaw counter-clockwise, metres / rad / m s⁻¹. NOW = `(0, 0, 0, v0)`.
  * This is NavSim's own `EgoStatus.ego_pose` frame, so `pose_hist` equals the bridge feed.
  * ⚠️ This differs from PhysicalAI, where poses are clip-local with an arbitrary origin. Every consumer in the recipe reads poses relative to `pose_last` (READ); E3 must emit any pose-frame quantity in this frame.
* **History (τ ≤ 0):** the devkit's ego states (float32 `ego_pose`, `ego_velocity`), interpolated by the bridge at the export's **actual** frame times.
* **Future (τ > 0):** from the OpenScene log pickle, which is an independent implementation (K2 compares it with the devkit):
  * global pose = `ego2global_translation[:2]` and yaw from `ego2global_rotation` (w, x, y, z), taken SE(2)-relative to t0;
  * knots at the **nominal** 2 Hz times `0.5·k` (k = −3 … K);
  * `scipy.interpolate.CubicSpline(bc_type="not-a-knot")` on x, y and unwrapped yaw;
  * `v` = linear interpolation of `hypot(ego_dynamic_state[0:2])`, the bridge's own rule for `v`.
  * Nominal knot times make the 2 Hz target **equal the devkit's `get_future_trajectory` frame-for-frame** (K2 ≤ 1e-3 m), and they are what NavSim scores against.
  * ⚠️ Actual nuPlan frame spacing is 0.49933–0.50055 s (MEASURED on one log), so the history (actual times) and the future (nominal times) differ by ≤ 0.7 ms per frame. That is declared and not corrected.
* **actions `[T_raw, 2]` = (steer, accel), float32. NOT consumed by the refcv8 recipe** (`--residual-prior ha0_ext_pose` ∉ `kinematic_prior.ACTION_MODES = ("ha0_ext",)`, READ). They are filled for schema fidelity:
  * `steer = atan(2.9 · κ)` with `κ = ẏaw / v` from the spline when `v ≥ 1.0 m/s`, else 0 (the 2.9 m legacy wheelbase, `physicalai.signals_at`);
  * `accel` = linear interpolation of `ego_dynamic_state[2]` (longitudinal, ego frame).
  * Anything that wants to consume them must say so first.

## 4. Payload: exact keys, dtypes and shapes

The keys `build_compressed` writes are mirrored from a real refcv8 training file (`refcv8-b1-416x1024-train-4572/clip-sha12-8d2bef17c465.v2ep.pt`, MEASURED with `torch.load(mmap=True)`), plus three new keys that the `fb47a58` loader ignores.

| key | dtype / type | shape / value | note |
|---|---|---|---|
| `jpeg_buf` | `torch.uint8` | `[n_bytes]` | concatenated **PNG** files: `[t0]` or `[t0, LAW]` (`torchvision.io.encode_png` on the CHW u8 stitch, the trainer's codec, `v2_compressed.py:143`) |
| `jpeg_len` | `torch.int64` | `[n_unique]`, 1 or 2 | byte length of each PNG |
| `poses` | `torch.float32` | `[T_raw, 4]`, `60 ≤ T_raw ≤ 70` | §3 |
| `actions` | `torch.float32` | `[T_raw, 2]` | §3.1 |
| `n_stack` | `int` | `3` | |
| `image_size` | `int` | `256` | mirrors the real cache's legacy value; geometry lives in `frame` / `image_h` / `image_w` |
| `episode_id` | `int` | `int.from_bytes(b"navt", "big")` = 1851881076 for every file | `build_compressed`'s 4-byte formula, kept verbatim. ⛔ **Constant over this corpus. Never use it.** `build_v2_providers(stable_ids=True)` (the default; the trainer's call) uses `stable_episode_id(clip_id)` |
| `clip_id` | `str` | `"navtrain-<token>"` | |
| `quality` | `int` | `90` | mirrors the cache (unused for PNG) |
| `image_h`, `image_w` | `int` | `416`, `1024` | |
| `frame` | `dict` | `trunk_shapes.FRAME_416x1024.to_dict()` = `{height 416, width 1024, f_ref 488.92398517830253, projection "cylindrical"}` | |
| `projection_mode` | `str` | `"cylindrical"` | |
| `codec` | `str` | `"png"` | lossless, so the training pixels = the stitch, bit for bit |
| **`frame_index`** (NEW, R1) | `torch.int64` | `[30]` | raw row → PNG index. Rows 0–9 → 0; rows 10–29 → 1 (or 0 on fallback) |
| **`nt`** (NEW) | `dict` | — | provenance: `schema "nt-e1/1"`, `token`, `log_name`, `map_name`, `log_pos`, `t0_timestamp_us`, `shard`, `driving_command_t0` (int list [4], the devkit export), `law_src` ∈ {`next_token`, `fallback_t0_not_in_navtrain_current`, `fallback_t0_missing`}, `law_token`, `t0_frame_sha16` / `law_frame_sha16` (sha256[:16] of the HWC u8 `[416,1024,3]` stitch, the bank's convention), `rig_key`, `observed_frac`, `mean_px`, `n_future_ticks`, `future_gap_truncated`, `history_knot_start`, `v0`, `pose_frame "t0_ego"`, `raw_now_row 9`, `dt_s 0.1`, `hist_source`, `future_source`, `writer` (tool + code sha256) |
| **`nt_frame`** (NEW) | `str` | `"ST"` | the frame-slot construction; equal to R8_LEGAL's `slot_sources8(times, "ST")` |

**What E1 does NOT provide** (trainer side, E2/E3/E4):
* the nav token (`NAVSIM_LEGAL_NAV_CMD[driving_command_t0]`);
* v9 / v7 labels and their release md5;
* the unknown speed row;
* agent / box3d joins and the vis1 sidecar (from OpenScene `anns`; E1 supplies `rig_key`, `RIGS_A1.json` and the `src` maps);
* the map-off store;
* the held-out guard in the trainer (K3: the token list above already satisfies it);
* the window filter (R2).

## 5. Manifest and heartbeat

`MANIFEST_A1.jsonl`: one JSON line per token, appended **after** the file is written and re-verified:
* re-loaded with `torch.load`;
* every PNG decoded with `torchvision.io.decode_png` and its HWC sha256 re-derived equal to the stitch's;
* keys, dtypes and shapes asserted.

Fields: `token, clip_id, sid, log_name, shard, file, bytes, file_sha256, t0_frame_sha16, law_frame_sha16, law_src, law_token, T_raw, n_future_ticks, future_gap_truncated, rig_key, observed_frac, mean_px, written` (ISO time).

`PROGRESS.json` is rewritten after every shard, and on every heartbeat during a shard:
* `shards_done`, `shard_in_progress`;
* `tokens_written / tokens_total`, `tokens_pending_law`, `missing`;
* `bank_bytes`, `disk_free_bytes`, `disk_peak_used_by_nt_bytes`;
* `download {last_MBps, mean_MBps, eta}`, `updated`.

`MISSING_A1.json` lists any token whose t0 jpgs were in no shard, with the reason. It is listed, never silently dropped.

## 6. Controls E1 runs (results in `nt_build/data/SMOKE_REPORT.json` and `VERIFY_PROD.json`)

**Status: all PASS**, MEASURED 2026-10-21:
* smoke: 20 selection-sample tokens from shard 1 (~21:24 Berlin), then 200 tokens with the v1.2 fix (`SMOKE_REPORT_K1_200.json`; the 200 K1 files are in `k1_sel/` for the trainer's K1);
* production check (`nt_verify_prod.py`): 24 random production files, 21:37 Berlin.

| # | control | must read | read |
|---|---|---|---|
| E1-K1a | stitched t0 frame = bank row (sha16 + `array_equal`) | bit-exact | **20/20**, then **200/200** |
| E1-K1a′ | item `frames` = `refcv8_bridge.pack_rows(bank, slot_sources8(times,"ST"))` | `torch.equal` | **200/200** items |
| E1-K1b | `ep.poses[0:8]` vs `ego_history_poses` | ≤ 1e-6 | **0.0** (200/200 `array_equal`) |
| E1-K1c | mutation `wrong_token` | RED | **RED** (frame ≠ bank) |
| E1-PNG | every stored PNG decodes to the stitch | bit-exact | **365/365** in the 200-token smoke (200 t0 + 165 LAW); every production file is verified at write |
| E1-INT | straight constant-velocity track | ≤ 1e-9 m | **1.4e-14 m**; circle R 20 m / 8 m/s: max radius error **8.9e-4 m** (stated, not a bar) |
| E1-K2pre | future 0.5–4.0 s vs devkit `human_future_poses` | ≤ 1e-3 m | v1.1: 6.4e-5 m on 20 tokens but **FAIL 0.021 m on 200** (yaw convention, §7 v1.2). v1.2: **1.9e-6 m** (yaw 5.9e-8 rad) on 200 |
| E1-LOAD | patched loader + **`refc_v3_train.V3Dataset`** (the trainer's own class) + R2 filter | one window per file; shapes; LAW row; `pose_last` | 8,000 → **200** windows (40 per file before R2). `frames [8,9,416,1024]` u8, `future_frames [20,9,416,1024]`, `future_poses_ext [60,4]`, `future_valid_ext.sum()` 60. LAW row = ST of the next frame on 165/165. `pose_last` = (0,0,0,v0) on 200/200 |
| E1-UNPATCHED | `fb47a58` loader on the same files | raises | `IndexError: index 2 is out of bounds for dimension 0 with size 2` |
| E1-REG | patched vs `fb47a58` loader on 3 real refcv8 PhysicalAI files | byte-identical | **12/12** window slices + `decode_full_episode` equal |
| E1-PROD | 24 random PRODUCTION files through `V3Dataset` | file sha = manifest; all 24 frame slots = t0 sha; LAW row = LAW sha; `valid_ext` = min(60, n_future_ticks); `future_poses_ext[4]` = raw row 14 | **24/24 on every check** (955 → 24 windows) |

Original control definitions:

| # | control | must read |
|---|---|---|
| E1-K1a | stitched t0 frame of a selection-sample token = the r8ns frame bank's row for that frame (`frame_bank/frames_s00.npy`, `index.json` `rows[-1]`), **and** the item's `frames` = `refcv8_bridge.pack_rows(bank_frames, slot_sources8(times, "ST"))` | bit-exact (sha16 equal, `torch.equal`) |
| E1-K1b | the item's `pose_hist` (`ep.poses[0:8]`) vs `ego_history_poses` | max \|Δ\| ≤ 1e-6 (expected 0.0) |
| E1-K1c | mutation `wrong_token` (the neighbour token's frame written under this token) | RED |
| E1-PNG | PNG round trip of every frame written | bit-exact |
| E1-INT | constant-velocity straight segment through the same spline code | exact (≤ 1e-9 m); a constant-speed circle reproduces its radius to a stated tolerance |
| E1-K2pre | future knots at 0.5…4.0 s vs the devkit's `human_future_poses` | max \|Δ\| ≤ 1e-3 m (x, y) |
| E1-LOAD | the patched `v2_dataset.build_v2_providers` + `FailLoudWindowDataset` (R2 filter) open the file | `frames [8,9,416,1024]`, `future_frames [20,9,416,1024]`, `future_poses_ext [60,4]`, `pose_last == (0,0,0,v0)`, one window |
| E1-UNPATCHED | the `fb47a58` loader on the same file | fails loud (IndexError), never a wrong frame |
| E1-REG | the patched loader on real PhysicalAI `*.v2ep.pt` files | byte-identical to the unpatched loader |

## 7. Changes to this contract

The writer bumps `nt.schema`, and this file gets a dated entry below. Nothing changes silently.

* 2026-10-21 21:20 Berlin — v1 written (E1, DataFlyWheel).
* 2026-10-21 21:27 Berlin — v1.1, before any file was consumed:
  * the dropped-frame rules above were made explicit;
  * `nt.history_knot_start` was added.
  * The 450 files written under the first build were **deleted and rebuilt**, so the corpus has one rule.
* 2026-10-21 ~22:00 Berlin — **v1.2, `nt.schema = "nt-e1/1.2"`. A real defect, caught by E1-K2pre on 200 tokens** (20 had passed by luck).
  * **The defect.** The future poses used the textbook ZYX yaw `atan2(2(wz + xy), …)`. The devkit reads `ego2global_rotation` through `pyquaternion.Quaternion(*q).yaw_pitch_roll[0]` = `atan2(2(wz − xy), 1 − 2(y² + z²))` (`navsim/common/dataclasses.py:171-173, 389-391`; MEASURED identical on Thor).
  * **Its size.** The t0 heading differed by ~1e-4 rad. The future then sat **up to 2.1 cm** off the devkit GT at 4 s, and off the history rows, which come from the devkit.
  * **The fix.** `nt_e1.quat_yaw` now uses the devkit convention. K2pre reads **1.9e-6 m** over 200 tokens.
  * **Repair.** The 9,034 files already written (shards 1–5) had **only `poses`/`actions` recomputed** by `nt_repair_poses.py`. The PNG frames are byte-identical, the history rows were asserted unchanged, and every file was re-verified.
  * **Size of the repair.** Future xy shift: max **6.7 cm**, p50 0.5 mm, p99 2.4 cm.
  * The pre-repair manifest is kept as `MANIFEST_A1.pre_v1.2_yaw.jsonl`, and repaired rows carry `"repaired": "v1.2-yaw"`.
  * ⇒ **Every file in `v2ep_a1/` is `nt-e1/1.2`.**

## 8. Where the code lives

Dev box `C:/Users/Admin/refcv8_pf/nt_build/data/`, shipped md5-verified to Thor `/home/nvidia/r8ns/nt/code/`:
* `nt_tokens.py` — the token list;
* `nt_download.sh` → `nt_download_v2.sh` (live since 21:37: retries a shard up to 40 times and never skips one);
* `nt_e1.py` — the writer library;
* `nt_e1_run.py` + `nt_e1_supervise.sh` — the per-shard pipeline and its restart loop;
* `nt_smoke.py` — the controls;
* `nt_verify_prod.py` — the production-file check;
* `loader_patch/` — R1:
  * `v2_dataset_frame_index.diff` against `fb47a58`;
  * `v2_dataset_patched.py`, the result;
  * `v2_dataset_fb47a58.py`, the base.

## 9. Production run: measured facts so far, and how to check or resume it

**Measured (shards 1–3, 21:20–21:37 Berlin):**
* shard 2 holds 2,889 frames, and **every one is an A1 t0 frame**: `navtrain_current` carries exactly the token frames, which is why the LAW fallback rate is ~17 %;
* 2,889 tokens stitched; 2,836 written; 53 MISSING (dropped-frame futures, 1.8 %); 0 pending;
* bank **1,050,365 B per token** (2 PNGs in 83.2 % of files) ⇒ projected bank ≈ **103 GB** for ~98.2 k tokens;
* stitch + encode **0.107 s per frame** (3 torch threads + 1 encode thread), so a shard is processed in **~404 s against a ~668 s download** — the run is download-bound;
* download **12.86–13.19 MB/s** per shard (`logs/dl.log`), the same as S1's 12.8. Shard 4's logged 47 MB/s is an artefact: it was resumed after the downloader was swapped to v2, and only the resumed part was timed. Thor reaches HF over wifi, so the rate is link-bound (INHERITED, memory note).
* **At 22:12 Berlin:**
  * shards 1–7 done; **13,259 written**, **160 MISSING** (all dropped-frame futures, 1.2 %), 0 pending;
  * LAW = next token for 10,922 of 13,259 (82.4 %);
  * **1,047,870 B per token** ⇒ final bank ≈ **103 GB**;
  * disk free 158.2 GB; peak nt usage so far 24.2 GB.
* **Projections:**
  * peak nt usage ≈ **120 GB** (bank ~96 GB + shards 31/32 in flight + scratch), i.e. minimum free ≈ 60 GB if nothing else grows;
  * download finishes ≈ **03:30 Berlin** (256.6 GB left at 13.1 MB/s from 22:03);
  * **`DONE_A1.json` ≈ 03:55 Berlin**, after the last shard (~10 min) and the finalize (`_v2manifest.pt` scan, clock sidecar, ~10–15 min).

**Check:**
```
ssh -n tanitad-thor 'cat /home/nvidia/r8ns/nt/PROGRESS.json'      # status, shards_done, tokens_written, missing, disk, download ETA
ssh -n tanitad-thor 'tail -3 /home/nvidia/r8ns/nt/logs/dl.log; tail -3 /home/nvidia/r8ns/nt/logs/e1_sup.log'
ssh -n tanitad-thor 'ls /home/nvidia/r8ns/nt/DONE_A1.json'          # exists only when everything is written and verified
```

**Resume** after a reboot or a supervisor give-up. Both scripts resume from their state; nothing needs cleaning:
```
cd /home/nvidia/r8ns/nt && setsid -f bash code/nt_download_v2.sh > logs/dl2.out 2>&1 < /dev/null
cd /home/nvidia/r8ns/nt && setsid -f bash code/nt_e1_supervise.sh > logs/e1_sup.out 2>&1 < /dev/null
```
Before resuming, check that neither is already running (`ps -eo pid,args | grep "[n]t_e1_\|[n]t_download"`). Kill only by explicit PID: supervisor first, then runner.
