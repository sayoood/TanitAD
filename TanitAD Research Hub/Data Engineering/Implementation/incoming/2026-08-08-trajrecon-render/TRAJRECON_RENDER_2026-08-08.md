# 2026-08-08 trajectory reconstruction — RUN. Both recordings, one OK, one REJECTED.

**This closes `TRAJRECON_FETCH_BLOCKER_2026-08-08.md`.** That document recorded the render as NOT
RUN behind two independent blockers — Drive policy-denied at the agent proxy, and the files never
link-shared. Both are now cleared and the pipeline has been run end to end.

**Evidence class: MEASURED (ours).** Every number below comes from `report.json` / `process.log` in
`raw/`, not from a summary. The upstream README's hold-out figures (2.23 m / 1.27 m/s / 0.84°)
remain **INHERITED** and are *not* quoted here as ours — they are superseded by the measurements in
§3, which are the first hold-out numbers this programme has produced on its own recording.

---

## 1. Inputs — fetched and verified

| | recording A | recording B |
|---|---|---|
| file | `2026-08-08_14-05-42-android.zip` | `2026-08-08_14-19-54-android.zip` |
| Drive id | `1AiH-m9E5tt73Q8kyvwFqVGpfjGVvDyzK` | `1DitzfASALJhJ5GqE-sBEqa3t_oyOipHN` |
| bytes | 18,701,510 | **179,607,367** — byte-exact vs the brief |
| md5 | `71779c836615a7cf4006c8e2630373ac` | `1fe7849ce7deb394ad8814e02197bc2c` |
| device | Samsung SM-G990B, Sensor Logger 1.62.1, Europe/Berlin | same |
| camera | 1920×1080 @ 30.01 fps, 8.53 s | 1920×1080 @ 29.92 fps, 81.75 s |
| IMU | 105.2 Hz accel/gyro/gravity | same |
| GNSS | 10 fixes, hacc 3.81 m | 84 fixes, hacc 3.81 m |

Fetched with `tools/gdrive_fetch.py` (this branch). Both archives pass `zipfile.testzip()`; neither
is the interstitial HTML the brief warns about.

## 2. Recording A — **REJECTED**, and correctly so

`raw/report_14-05-42.json`, verbatim findings:

| level | code | message |
|---|---|---|
| **REJECT** | `NO_MOTION` | only 40 m travelled (< 50.0 m) |
| **REJECT** | `TOO_SHORT` | less than 15 s of overlapping video and sensor data (overlap 8.5 s) |
| DEGRADED | `ACCEL_UNUSABLE` | accelerometer carries no usable longitudinal signal (quality 0.03) |
| DEGRADED | `WEAK_VEHICLE_FRAME` | phone→vehicle lateral axis weakly determined (lateral_score 0.041) |
| DEGRADED | `EMPTY_CSV` | `Annotation.csv` is essentially empty (0 bytes) |

GNSS: 10 fixes, `frozen_fraction` 0.0, speed 3.46–5.32 m/s, `gps_distance_m` 39.8.

**This is the pipeline's "never process blind" gate working, not a failure of the run.** Per the
brief, a REJECT is a valid result and must be reported as the finding rather than tuned away. No
flags were changed to make it pass.

## 3. Recording B — **VERDICT: OK**, with our own hold-out numbers

### 3.1 Hold-out validation — GNSS never seen by the filter

5 folds, each holding out every 5th fix (≈ 5 s outage), n = 82:

| quantity | mean | **rms** | p95 | max |
|---|---|---|---|---|
| position (m) | 0.59 | **0.70** | 1.36 | 1.71 |
| speed (m/s) | 0.06 | **0.08** | 0.16 | 0.22 |
| heading (deg, v ≥ 2.5) | 0.43 | **0.53** | 0.97 | 1.00 |

In-sample fit residual: position rms 0.65 m, speed rms 0.01 m/s.

⚠️ **Do not read these as "3× better than upstream."** They are a different recording, a different
route and a different fold protocol from the INHERITED 2.23 m / 1.27 m/s / 0.84°. The honest
statement is narrower and stronger: *on our own 2026-08-08 recording the hold-out position rms is
0.70 m*, and that number is ours.

### 3.2 Internal consistency

```
|d(pos)/dt| vs speed state   rms 0.0017 m/s
course vs heading (v>2)      rms 0.007 deg
speed non-negative           min 19.661 m/s
pos 1-sigma                  median 1.27 m   max 1.92 m
longitudinal accel           p99 1.81 m/s^2  max 3.54
yaw rate                     p99 4.1 deg/s   max 4.5
```

### 3.3 Time sync

`t_video_start_s` **3.3043**, source **gyro-xcorr (bands agree to 2 ms)**, `sync_score` **0.47**.
Vehicle frame: gps-accel (r = +0.23), up = gravity with sign from GNSS (r = +0.37),
lateral_score 0.306.

### 3.4 Trajectory and steering

2216 of 2446 frames carry a complete [−3, +5] s window — **100 % of the emitted records**, spanning
3.50–77.49 s (74.0 s). `gps_distance_m` **1730.7** over the full 82.5 s session; integrating the
emitted speed over the 74.0 s window gives ≈ 1558 m. Both are correct for their spans — the report's
1730.7 m is the one to quote.

Accelerometer quality separates the two recordings cleanly: **`accel_tier` 1, `accel_quality`
0.394** here, against tier 4 / 0.031 on recording A (which is why A's speed would have been
GNSS-only). 83 GNSS fixes, `frozen_fraction` 0.0, `moving_fraction` 1.0, `max_gap_s` 1.0.

| | |
|---|---|
| speed | 19.67–23.32 m/s (**70.8–84.0 km/h**), mean 21.06 |
| steering wheel (Audi A6 e-tron, L = 2.946 m, 15.9:1) | −10.6 … +12.2°, p99 |·| 10.7° |
| wheel rate | p99 19.0 °/s, max 50.0 °/s |
| curvature | max |k| 0.0034 1/m (min radius **293.9 m**) |
| steering 1-sigma | median 0.7° |
| `steer_valid` | **100.0 %** of frames |

### 3.5 ⚠️ Camera calibration — **DEGRADED**, and one cause is a dependency break

```
VERDICT: DEGRADED
[DEGRADED] MOUNT_NOT_CALIBRATED  mount angles were not estimated from the data; nominal values in use
    hfov_deg 66.0 | mount_yaw/pitch/roll 0.0/0.0/0.0 | cam_height_m 1.17 | horizon_row_px 540.0
```

Four calibrators, four outcomes — and they do **not** share a cause:

| calibrator | outcome | cause |
|---|---|---|
| `lane_calib` | skipped | ⚠️ **cv2 5.0 API break — see §4.** Lateral offset fell back to the −0.35 **prior**; it was never measured. |
| `plane_calib` | too few usable homographies | genuine — uses `findHomography`, no changed API |
| `vp_calib` | too few vertical lines | genuine — uses `.reshape(-1,4)`, correct under both cv2 majors |
| `scale_calib` | 0 usable tracks, `f*h=nan` | genuine — `goodFeaturesToTrack`/`calcOpticalFlowPyrLK` unchanged in 5.x |

⇒ **The brief's Step-5 item 3 is only partly answerable from this run.** Camera height is the 1.17
default, *not* a `plane_calib` measurement; lateral offset is the −0.35 prior, *not* a `lane_calib`
measurement. Reporting either as "measured" would be false. This is visible in `validation.png`:
the projected corridor sits right of the true ego lane, which is what an uncalibrated mount yaw plus
a fallback lateral prior look like.

**Credit where due: the pipeline does not hide this.** `raw/calibration_14-19-54.json` labels every
parameter with its provenance, and the labels are the honest ones:

```json
"height_m":          {"value": 1.17, "source": "operator default (not measured)"},
"roll_deg":          {"value": 0.0,  "source": "assumed level (not measured)"},
"lateral_offset_m":  {"value": -0.35,"source": "operator prior"},
"intrinsics_note":   "gyro focal refinement rejected (r=0.24, n=57) - kept nominal",
"extrinsics_note":   "FOE fit produced no usable flow - kept nominal"
```

An instrument that writes "not measured" next to its own defaults is doing the thing this programme
keeps having to relearn. The failure here is the dependency cap, not the pipeline's honesty.

### 3.6 The render

`overlay.mp4`: **1898×720**, h264, 29.92 fps, **2216 frames, 74.06 s, 156 MB**, encoded in 852 s
(~2.6 frames/s — the cost of compositing a 1080p projection plus a BEV panel per frame). Total
pipeline wall time 1012 s. A 1280-wide, CRF-30 delivery copy is 14 MB.

## 4. The cv2 5.0 break — MEASURED, and it degrades instead of failing

`raw/cv2_hough_shape_probe_2026-08-08.txt`:

```
HoughLinesP          (10, 4)     <- CHANGED in 5.x (4.x returns (N, 1, 4))
HoughLines           (23, 1, 2)  unchanged
goodFeaturesToTrack  (22, 1, 2)  unchanged
calcOpticalFlowPyrLK (22, 1, 2)  unchanged, status (22, 1) uint8
```

`lane_calib.py:182` iterates `seg[:, 0]`, which under 5.x is a 1-D array of scalars →
`cannot unpack non-iterable numpy.int32 object`. The pipeline catches it and **warns**, so the run
still reports `VERDICT: OK` while silently losing a calibration. *A dependency break that costs a
measurement instead of raising is the dangerous kind* — same family as the probe-answers-a-different-
question traps in `CLAUDE.md`.

`stack/pyproject.toml` declared `opencv-python>=4.9` with **no upper bound**, which is what admitted
5.x. Now capped `>=4.9,<5`. Capping beats patching `lane_calib.py`: the 24 upstream modules are
landed byte-exact and that provenance is worth more than avoiding a pin.

## 5. The pipeline could not start at all before this branch

Seven byte-exact modules import each other as `from trajlib import …`; nothing in the repo provided
`trajlib`, so the documented `python -m tanitad.data.trajrecon.pipeline` died on its first import.
84 passing tests could not see it because **none of them imported anything** — they read sources as
bytes. Fixed with a `sys.modules.setdefault` alias in `trajrecon/__init__.py` (ours, so byte-exactness
holds) and guarded by `stack/tests/test_trajrecon_imports.py`. Full write-up in commit `346c7343`.

## 6. Deliverable manifest

| artifact | where |
|---|---|
| `overlay.mp4` — the results video | `/root/trajdata/out/2026-08-08_14-19-54-android/` (sent to the PI) |
| `validation.png` — composite: projection + BEV + HUD | same |
| `trajectory.jsonl` — 2216 records, 81-point [−3,+5] s window each | same |
| `report.json` / `report.md` / `process.log` / `sensors/` / `frames/` (2216) | same |
| `raw/report_14-05-42.json`, `raw/diagnosis_14-05-42.log` | this bundle, repo |
| `raw/pipeline_run.log` | this bundle, repo |
| `raw/cv2_hough_shape_probe_2026-08-08.txt` | this bundle, repo |

⚠️ **The bulk outputs live on this container's disk only** — 2216 JPEGs and a ~70 MB mp4 are not
committed. The container is ephemeral. Re-running from the two Drive ids reproduces them exactly;
the ids, sizes and md5s in §1 are the reproduction key.

## 7. ⭐ The opencv-4.x re-run — the fix did NOT recover the measurement, and that is the finding

The obvious follow-up was to re-run under `opencv-python-headless<5` so `lane_calib` would work.
Done (`--force --only 14-19-54 --no-video`, cv2 **4.14.0**, `HoughLinesP` back to `(N,1,4)`).

**First: the solve is deterministic.** Trajectory and steering came back bit-identical — hold-out
0.70 m / 0.08 m/s / 0.53°, min radius 293.9 m, `steer_valid` 100 %. ⇒ the cv2 major touches
**calibration only**, which was assumed before and is now measured.

**Second, and the point: `lane_calib` now runs — and declines anyway.**

```
LaneCalib(yaw=declined, lateral=declined, lane_width=3.50 m, frames=59, segments=976,
          yaw_spread=0.20 deg)
  WARN  yaw declined: -7.01 deg is 7.0 deg from the FOE, not credible
  WARN  lateral declined: only 7 usable frames
  INFO  pitch cross-check: lane VP row is 62.0 px from the FOE row
```

59 frames, 976 segments — real work, then refused on its own quality gates. `plane_calib` likewise
now produces a fit and rejects it:

```
PlaneCalib from 8 road homographies
  roll = +0.80 deg (spread +/-14.85) | pitch = +10.29 deg (spread +/-3.00) | height = 2.139 m (spread +/-0.222)
  pitch cross-check vs FOE +0.00 deg: 10.29 deg apart
  WARN  plane height rejected: only 8 homographies; pitch disagrees with the FOE by 10.3 deg;
        height spread +/-0.22 m; height 2.14 m is not a windscreen mount
```

`vp_calib` (too few vertical lines) and `scale_calib` (0 usable tracks) are unchanged.

**Camera verdict under cv2 4.14 is IDENTICAL to cv2 5.0:** `DEGRADED`, `MOUNT_NOT_CALIBRATED`,
yaw/pitch/roll 0.0, height 1.17, hfov 66.0.

`diff` of the two `calibration.json` files is exact and worth quoting, because it says precisely
what the pin bought — **every applied parameter is identical; only the `cross_checks` block gains
detail**:

```diff
-        "plane": "insufficient homographies",
+        "pitch_foe_vs_plane_deg": 10.294,
+        "plane_pairs": 8,
+        "plane_height_rejected": "only 8 homographies; pitch disagrees with the FOE by 10.3 deg;
+                                  height spread +/-0.22 m; height 2.14 m is not a windscreen mount",
+        "lane_frames": 59,
+        "lane_segments": 976,
+        "pitch_foe_vs_lane_vp_px": 62.0,
```

`K`, `extrinsics` and every entry under `parameters` are byte-identical. So the pin did not change
the *answer* — it changed a silent failure into a **recorded, numeric disagreement**, which is what
made §8.1 diagnosable at all.

### Two consequences

1. ⇒ **The delivered video is not materially wrong.** It was rendered with nominal calibration, and
   the correctly-pinned run yields that same nominal calibration. **No re-render is needed** — which
   is only knowable because the re-run was actually done rather than assumed either way.
2. ⇒ **The brief's Step-5 item 3 now has a real answer, and it is a negative one:** the lateral
   offset and camera height **cannot be measured from this recording**. Not "a bug stopped us" —
   every calibrator that could speak, spoke, and each declined for a stated numeric reason. That is
   a materially different and more useful claim than the cv2-5 run could support.

⚠️ **The opencv cap is still right and still worth having.** It did not change this recording's
outcome, but under cv2 5 `lane_calib` never got to *have* an opinion — the decline above was
indistinguishable from a crash. A gate that cannot run is not a gate that passed.

## 8. What is still open

1. **Four independent declines on a clean 74 s daylight highway clip.** The recurring theme is the
   **FOE**: `lane_calib`'s yaw is 7.0° from it, `plane_calib`'s pitch 10.3° from it, and the lane VP
   row sits 62 px off it — while the FOE fit itself "produced no usable flow". If the FOE reference
   is wrong, three cross-checks are being judged against a bad ruler. **This is the first thing to
   investigate before scaling this corpus**, and it is cheap: the disagreements are all recorded.
2. **`scale_calib` gets 0 usable tracks under both cv2 majors.** Independent of the FOE question,
   and unexplained.
3. **Recording A is unusable** (40 m, 8.5 s). If short clips are expected in the corpus, the 50 m /
   15 s gates are the thing to revisit — deliberately, not by loosening flags to make one file pass.
4. **A windscreen-mount height of 2.139 m was rejected as implausible.** Either the homographies are
   bad or the mount is not where the prior assumes; a tape measure settles it in one minute and
   would unblock `--cam-height`.
