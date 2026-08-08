# trajrecon in the TanitAD stack — integration notes

This file is **ours**. `README.md` and `AUDIT.md` in this directory are the upstream
project's own documents, copied verbatim; where they disagree with the code, the code
wins and the disagreement is recorded here.

## What this package is for

Ground-truth ego-trajectory reconstruction from smartphone dashcam recordings. It
produces exactly the quantities `tanitad/data/_contract.py` already requires —
`poses [T,4]` (x, y, yaw, v) and both action channels (steer rad, accel m/s²) — from a
Sensor Logger (Android) export. That makes it a real-data source for the episode
contract alongside `comma2k19.py`, with no annotation required.

## Running a recording

`pipeline.py` consumes Sensor Logger `.zip` archives **directly** — no manual
extraction — and is resumable via a registry keyed on an input fingerprint, so an
interrupted run costs at most one recording.

```bash
python -m tanitad.data.trajrecon.pipeline \
    --input-dir  <folder containing the .zip> \
    --output-dir <output folder> \
    --lane-width 3.50
```

Requires the `trajrecon` extra (`pip install -e 'stack[trajrecon]'`) **and ffmpeg AND
ffprobe on PATH**. Neither is a pip dependency. MEASURED 2026-08-08: `imageio-ffmpeg`
supplies ffmpeg **only** and is NOT sufficient — `timesync.probe_video` shells out to
ffprobe. `apt-get install ffmpeg` supplies both.

## Parameter decisions for the 2026-08-08 recording

| parameter | value | evidence class |
|---|---|---|
| `--lane-width` | **3.50** | operator-supplied: PI states the drive is in **France**. French autoroute/main-road standard is 3.50 m; the code default of 3.65 is the US 12-ft figure. |
| `--cam-height` | 1.17 (default) | code default. Also `--plane-calib` is ON by default and measures height from the road-plane homography, so the operator value is a fallback, not the operating assumption. **Report what plane-calib actually measured.** |
| `--lateral-offset` | −0.35 (default) | prior only. Unobservable from motion; `lane_calib` recovers it per recording from the ego-lane centre when markings are usable. |
| `--wheelbase`, `--steering-ratio` | Audi A6 e-tron | upstream defaults. Progressive (variable-ratio) steering, if fitted, makes the wheel angle right in *shape* and approximate in *scale*. |

⚠️ **`--lane-width` is not cosmetic.** `scale_calib` uses the true lane width as the
external metric that separates focal length from camera height (it can only observe the
product `f*h`). Leaving the US 3.65 default on a French recording biases that separation,
and therefore the entire ground projection scale, by roughly 4 %.

⚠️ **README vs code disagree on camera height.** The README's parameter table says
1.25 m; its own limitations section says 1.17 m; `pipeline.py --cam-height` defaults to
**1.17**. Height scales the whole ground projection linearly. The code is the source of
truth.

## Evidence class of the upstream accuracy figures

`README.md` quotes hold-out RMS **2.23 m** position, **1.27 m/s** speed, **0.84°**
heading on the 2025-08-11 session. These are **INHERITED** — they come from the upstream
project's documentation, have NOT been reproduced in this repo, and must not be cited as
a TanitAD result until re-measured here with `validate.py` (k-fold hold-out, not fit
residual).

## Standstill: do not re-introduce a known defect

`estimate_steering` returns a `valid` mask that is False where speed is too low for
steering to be observable (`v_min` 1.5 m/s), and `pipeline.py` carries
`--standstill-speed` (0.15 m/s) and a shortened `--t-future-standstill`. **Any adapter
must propagate that mask rather than emit steer values at standstill.**

This is the same defect class the registry already documents for comma2k19, where
heading derived from the velocity vector is undefined at rest: **26.27 %** of frames in
the `v < 0.5 m/s` bin carried physically impossible yaw rates, and it read as a model
failure (pooled `yaw_rate` R² 0.1046 against those labels vs 0.8108 against repaired
ones, with nothing retrained). Do not rediscover this.

## Transfer provenance

The source was copied from a Google Drive folder that this environment cannot reach
(`drive.google.com` is 403 at the agent proxy), via the Drive connector, which returns
bytes inline as base64.

- `read_file_content` is **unusable** for Python — it strips every level of indentation
  and adds markdown escapes.
- Re-emitting base64 through a model context corrupts silently. Every file is therefore
  verified against Drive's own `fileSize` **and** against the size of the blob git
  actually stored (`git cat-file -s`), plus `ast.parse`.
- **The guard caught real corruption**, twice in ways `ast.parse` alone would have
  missed: `vp_calib.py` arrived 3 bytes over and **parsed as valid Python**;
  `vehicle_frame.py` arrived 1 byte over; `trajectory.py` arrived truncated at 19495 of
  25338 B. None was committed.
- Line endings are **not normalised** (files are a mix of CRLF and LF, faithful to
  source). The repo has no `.gitattributes` and `core.autocrlf` is unset. **Adding a
  normalising rule would change the byte counts and silently invalidate every
  verification recorded here.**

## Not done

The **2026-08-08 recording has not been processed and no results video exists.** The zip
is 180 MB (~240 MB as base64) and no context window holds that; the sensor CSVs alone are
~1 MB each, also past the limit. Rendering requires the data reachable by `curl`, i.e.
the cloud environment's network access set to **Custom** with `drive.google.com`,
`drive.usercontent.google.com` and `*.googleusercontent.com` allowed (keeping the default
package-manager list), plus link-sharing on the file. Environment changes apply to **new
sessions** only.
