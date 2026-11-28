# Data Card — comma2k19 (TanitAD primary Phase-0 corpus, D-009)

**Agent:** Data Engineering (Tuesday). **Date:** 2026-07-07. **Status of loader:** shipped
(`stack/tanitad/data/comma2k19.py`, D-009 by Sayed); **real-data decode validated today** (this note).

> G-D1 required fields (license, size, actions availability, cost-to-first-batch) are all filled below,
> each backed by a source link or a measured command on this dev machine.

## 1. Identity & license

| Field | Value | Source |
|---|---|---|
| Name | comma2k19 (comma.ai) | [arXiv 1812.05752](https://arxiv.org/abs/1812.05752) |
| **License** | **MIT** — clean for public claims/derived pseudo-labels, no non-commercial clause | [commaai/comma2k19 GitHub](https://github.com/commaai/comma2k19) |
| Access | HF dataset `commaai/comma2k19`, **NOT gated** (verified `whoami=Sayood`, `repo_info.gated=False`, 2026-07-07) | measured |
| Content | 2019 × 1-min segments, ~33 h, 20 km of CA-280 highway (San Jose↔SF) | arXiv |

## 2. Size & layout (measured via `HfApi.get_paths_info`, 2026-07-07)

| Artifact | Size | Note |
|---|---|---|
| `raw_data/Chunk_1.zip` … `Chunk_10.zip` | **8.73 GB each ≈ 100 GB total** | official layout; Chunk_1 ≈ 3.3 h = first corpus (D-009) |
| `data/demo-0000{0,1,2}-of-00003.parquet` | 81 MB each (~243 MB) | HF-repackaged demo; needs `pyarrow` (not installed) — **not** the loader's input layout |
| `compression_challenge/<route>/10/video.hevc` | 37.5 MB | single addressable 1-min clip — used for today's decode validation |

On-disk segment layout the loader targets (inside each `Chunk_*.zip`):
```
Chunk_<n>/<dongle_id|YYYY-MM-DD--HH-MM-SS>/<segment>/
    video.hevc                              20 fps, 1164×874 (H.265)
    processed_log/CAN/speed/{t,value}       m/s
    processed_log/CAN/steering_angle/{t,value}   deg (STEERING WHEEL, not road wheel)
    global_pose/frame_times                 s, one per camera frame
    global_pose/frame_positions             ECEF meters [T,3]
    global_pose/frame_velocities            ECEF m/s   [T,3]
```

## 3. Actions & targets availability (H7-relevant)

- **Actions: YES, real.** Longitudinal = accel from CAN `speed` (finite diff); lateral = CAN
  `steering_angle` (steering-**wheel** deg) → road-wheel rad via a **steering ratio** (loader constant
  `STEER_RATIO = 15.3`, Civic/Corolla-class). The ratio is car-specific (~13–18) and is the H7 IDM
  **calibration knob** to log — see research note §3. Source for the steering-wheel-vs-road-wheel
  distinction: [openpilot steering-control](https://github.com/commaai/comma-steering-control).
- **Poses: YES, real.** GNSS/INS `frame_positions`+`frame_velocities` (ECEF), tightly-coupled
  INS/GNSS/Vision optimizer (Laika). Loader → segment-local ENU (x_east, y_north, yaw, v). No labels.
- **Perception labels: NONE** (matches our zero-perception-label thesis; this is a feature).

## 4. Cost to first (real) training batch — engineer-hours

Loader code = **0 h** (D-009 shipped; decode path validated today). Remaining:

| Step | Where | Est. |
|---|---|---|
| Supervised pull of `Chunk_1.zip` (8.7 GB) + unzip | **Linux A40 pod** (see `|` caveat §6) | ~0.5 h wall (download-bound) |
| Point `discover_segments()` at the unzipped root; smoke `build_episode` on 3 segments | pod | ~0.5 h |
| Wire into the Stage-A data loop / regenerate as the real Stage-A source | pod | ~1 h |
| **Total to first real batch** | | **~1–2 engineer-hours** (all download/wire, no new code) |

`av` (PyAV) is the only extra runtime dep for decode — **already installed** on this machine (py3.13);
no OpenCV needed. Add it to the `[real]` extra before the pod run.

## 5. Measured on REAL data today (single segment, `compression_challenge/.../10/video.hevc`)

- `_decode_video` (loader's own path) decodes real comma HEVC via `av` → `[200, 3, 256, 256]` uint8 in
  **1.9 s (~105 fps)** on py3.13/Windows/RTX-4060 box. `stack_two_frames` → `[199, 6, 256, 256]`. ✔ shapes
  match the `base250cam` contract end-to-end on real bytes.
- **A8 consequence-dominance (real highway): `frame_change_fraction` = 0.053 @ thr 0.05, 0.012 @ thr 0.10**
  — only marginally above the toy BEV floor (0.03). Raw-RGB pixel diff under-reads consequence on
  low-texture highway (sky/road); this is the empirical case for **change-weighted losses** and the
  2-frame stack. See research note §4.

## 6. Known subtleties / risks (P8 — record the sharp edges)

1. **Windows `|` path — handled by the D-009 extractor.** Route folders are `dongle|date`; `|` is illegal
   on Win32, so a plain `extractall()` or a raw-layout `hf_hub_download` fails (`WinError 123`, confirmed
   2026-07-07). D-009 already ships **`stack/scripts/extract_comma2k19.py`** which rewrites `|`→`_` in
   member paths on extract; `route_of`/`split_by_route` operate on the sanitized `dongle_date` name (still
   unique per route), so grouping is unaffected. **Use that extractor** — do not `unzip`/`extractall`
   directly on Windows, and do not `hf_hub_download` the raw `|` paths (fetch the zip, then extract).
2. **Steering ratio is a constant (15.3) v0.** Real ratio varies by car and is non-linear near center
   (openpilot issue #599). Log per-segment calibration residual (H7).
3. **ECEF→ENU uses a geocentric-latitude approximation** (loader docstring): sub-0.3 % tilt over a 1-min
   segment — fine for local frames; revisit if we ever chain segments into a route-global map.
4. **Demo parquet ≠ loader input.** `data/*.parquet` is a HF repackaging; the loader consumes the
   original `Chunk_*` layout. Don't wire the parquet by mistake.

## 7. Verdict

comma2k19 is **GO** as the D-009 primary corpus: MIT-licensed, ungated, real actions + poses, zero labels,
loader shipped and decode-validated on real bytes. First real batch is ~1–2 h of (Linux) download/wiring,
zero new code. The only correctness item to close before headline H7 claims is the steering-ratio
calibration log.
