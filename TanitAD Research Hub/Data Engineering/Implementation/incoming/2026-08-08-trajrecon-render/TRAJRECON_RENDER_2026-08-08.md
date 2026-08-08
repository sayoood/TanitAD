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

## 7b. ⭐⭐ THE YAW WAS MEASURED AND THROWN AWAY — and the root cause is an instrument misuse

Sayed flagged the overlay yaw as wrong. It is, and the pipeline had already computed the right
answer. The full chain, every step measured:

### The immediate defect

`lane_calib` measured **mount yaw −7.01°** (59 frames, 976 segments, half-split spread **0.20°** —
its own stability gate is 0.6°, comfortably passed). It was then discarded by the credibility gate
at `lane_calib.py:243`:

```
yaw declined: -7.01 deg is 7.0 deg from the FOE, not credible
```

**There was no FOE.** Every FOE failure path in `camera.py` (`:396,:401,:403,:408`) leaves `cam.yaw`
at the nominal **0.0** and records `extrinsics_note`; only the success path (`:414`) sets
`extrinsics`. Our `calibration.json` has **no `extrinsics` key**. So the gate
`abs(yaw_deg - degrees(cam.yaw)) > 4.0` degenerated into *"reject any mount yaw beyond 4° of dead
ahead"* — **rejecting hardest exactly when the mount is most crooked**, which is when the correction
matters most.

**Independently confirmed** (`raw/independent_vp_yaw_fit.txt`) with a from-scratch VP fit — not
trajrecon code — over 154 straight frames / 6689 segments: **−6.05°, 95 % CI [−6.28, −5.83]**.

**Cost:** lateral error `d·tan θ` — 1.2 m at 10 m, 2.5 m at 20 m, **4.9 m at 40 m**. Half a 3.50 m
lane is 1.75 m, so the corridor leaves its own lane **~15 m ahead**. Diagnostic: the error grows
with distance (a yaw signature); a bad lateral offset would be a constant shift.

### Three layers made it quiet, and the third is the worst

1. the gate rejected a good measurement against a placeholder;
2. the warning **called the placeholder "the FOE"** — `pipeline.py:482` even names the variable
   `foe_yaw` — so the log reads like a real disagreement between two independent estimates;
3. `lane_calib.py:295` returns `yaw_deg if yaw_ok else None`, so `pipeline.py:494`'s
   `if res.yaw_deg is not None` was False and the one line that spells out the damage **never
   printed**: `yaw: FOE +0.00 deg -> lane VP -7.01 deg (-7.01 deg, 4.89 m at 40 m)`.
   **The louder the error, the quieter the log.**

### ⭐ Root cause: a timing instrument used as a magnitude instrument

`raw/foe_root_cause_probe.txt`. `estimate_foe` drops any frame pair whose rotation rate exceeds
**2.01 °/s on all three axes**, reading those rates from `timesync.image_angular_rate` — whose own
docstring (`timesync.py:236-239`) says:

> *"Forward translation still leaks into `tx` ... so the yaw amplitude here is only approximate.
> That is fine — **only the timing of the signal is used**."*

MEASURED over t = 5–20 s, n = 449 pairs:

| | p50 | pass < 2.01 °/s |
|---|---|---|
| **camera** yaw | **5.42 °/s** | **4.5 %** |
| camera pitch | 2.80 °/s | 33.4 % |
| camera roll | 1.87 °/s | 53.2 % |
| **all three** | — | **0.4 % — 2 of 449 frames** |
| **gyro** x/y/z (true rotation, 105 Hz) | 1.46 / 1.64 / 1.81 °/s | 64.6 / 57.1 / 54.5 % |

The camera-derived yaw rate is inflated **~3.7×** over the true rate, and the leak **scales with
speed** — this clip is 70–84 km/h. At 0.4 %, the first 40 s (the `cam_flow` coverage, since
`--calib-seconds` defaults to 40 on an 81.75 s video) yields **~5 usable frames**, below
`estimate_foe`'s `if len(idxs) < 10: return None`.

**The data is fine; the instrument feeding the gate is not.** Gating on the *gyro* instead, averaged
per frame interval, gives **49.0 % quiet frames and 393 consecutive pairs** in the same 40 s — far
above the 10 needed. And the message `"FOE fit produced no usable flow"` points at the flow when the
actual cause is the **frame selection**.

### Fixed, and verified

At the **call site** (`pipeline.py`), not inside `lane_calib` — the caller is what knows whether a
FOE reference exists, so it supplies the bound: 4.0° when the FOE was measured, else **15.0°**, the
same plausibility constant `camera.py:405` uses on the FOE itself. The half-split stability gate is
untouched and remains the real quality check.

Result: `yaw=-7.01 deg` **accepted**, `mount_yaw_deg -7.01` applied, the suppressed diagnostic prints
`4.89 m at 40 m` (hand calculation: 4.92 m), and the **measured lane width moved 3.50 → 3.60 m** —
independent corroboration, since width is measured at the settled yaw.

⚠️ `pipeline.py` is therefore **no longer byte-exact** against upstream. `__init__.py` now carries an
explicit "deliberate divergences" section rather than letting the verbatim claim rot silently.

## 7c. ⭐⭐⭐ ROOT CAUSE FIXED: the gyro-gated FOE, and the cascade it unblocked

`estimate_foe`'s rotation gate now reads the **gyro** instead of `image_angular_rate`
(`camera._gyro_rot_on_frame_pairs`). It builds a `(t_mid, omega, comps)` triple of the same shape,
so `estimate_foe` consumes it unchanged. Two details carry the safety:

* `t_mid` is `0.5 * (pts[:-1] + pts[1:])` — the *identical* arithmetic `estimate_foe` uses to
  recompute midpoints, so its `< 1e-6` join matches bit-exactly.
* `comps` is the rotation **magnitude** replicated across three columns, making
  `np.all(|comps| < r, axis=1)` equal `|ω| < r`. Mount-frame independent (no phone-axes → camera-axes
  mapping needed) and strictly conservative. Intervals with no gyro sample are `inf`, never `0.0` —
  a zero-filled gap would read as *the quietest data available*.

### The FOE now succeeds, and three methods agree

```
FOE at (797,464) px, 425093 flow vectors, inliers=0.86        <- was: no usable frame pairs
yaw: FOE -6.31 deg -> lane VP -7.01 deg (-0.70 deg, 0.49 m at 40 m)
```

| method | physics | yaw |
|---|---|---|
| FOE, gyro-gated | translational optical flow | **−6.31°** |
| `lane_calib` VP | lane markings | **−7.01°** |
| independent fit (ours, §7b) | lane markings, own code | **−6.05°** [−6.28, −5.83] |

The FOE pixel **(797, 464)** against our independent VP **(803.3, 523.4)**: the *x* agree to **6 px
= 0.24°**, from entirely different cues. **And the credibility gate now passes on merit** — 0.70°
apart, well inside the original 4.0° — not because anything was loosened.

### The cascade from one fix

| | before | after |
|---|---|---|
| FOE | no usable frame pairs | **425,093 vectors, 86 % inliers** |
| lane yaw | declined | **−7.01° accepted** |
| lane **lateral** | declined (7 usable frames) | **+0.25 m MEASURED** |
| `plane_calib` | 8 homographies, h = 2.139 m, 10.3° off FOE | **14**, h = **1.691 m**, **4.8°** off |
| `scale_calib` | **0 tracks** | **1576 tracks**, f·h = 1608 |
| lane-VP vs FOE row | 62.0 px | **13.7 px** |
| horizon row | 540.0 (nominal) | **464.4** |
| **camera verdict** | **DEGRADED** | **✅ OK** |

Final provenance, every line from `raw/calibration_14-19-54_foe-gyro.json`:

```
yaw_deg          -7.007  <- lane vanishing point, 976 segments / 59 straight frames (spread 0.20 deg)
pitch_deg        -2.928  <- FOE at (797,464) px, 425093 flow vectors, inliers=0.86
lateral_offset_m  +0.25  <- ego-lane centre vs camera over 59 frames (lane width 3.60 m)
height_m           1.17  <- operator default (NOT MEASURED)   <-- the only one left
foe_rotation_gate        <- gyro |omega| per frame interval
```

⇒ **The brief's Step-5 item 3 is now fully answered** except camera height, and that one is honestly
labelled. `plane_calib` still declines its height (spread ±0.56 m over 14 homographies), so 1.17 m
remains an operator value — a tape measure settles it in a minute.

⚠️ **This supersedes §7b's conclusion that the pipeline.py yaw-gate fix was what recovered the
measurement.** With a working FOE the *original* 4.0° gate passes on its own. The `pipeline.py`
change is still correct — it is the safety net for when the FOE genuinely fails — but it is no
longer the load-bearing fix. The load-bearing fix is this one.

⚠️ `camera.py` is now also a **deliberate divergence** from upstream; recorded in `__init__.py`.

### 7d. ⚠️ Do NOT read "VERDICT: OK" as "the calibration is now right"

Yaw is settled. **Pitch and height are not**, and they are what govern the near-field ground
projection. Measured, on the same run that reports OK:

| quantity | estimates | status |
|---|---|---|
| **pitch** | FOE **−2.93°** vs `plane_calib` **+1.89°** | **4.82° apart** |
| **height** | `scale_calib` implies **1.088 m** (f·h 1608.1 / f 1478.3; declined, spread 62 %) · `plane_calib` **1.691 m** (declined, ±0.56) · **1.17 m operator default IN USE** | **1.09–1.69 m, a 55 % spread** |
| **lane width** | passed `--lane-width 3.50`, `lane_calib` **measured 3.60 m** | **+2.9 %**, consistent with the f·h uncertainty |

The consequence is visible. Applying pitch −2.93° widened the near-field corridor by **+15 %**
(514 → 591 px at row 0.96H) and it now **overhangs the right lane edge**, where the yaw-only render
sat inside the lane near the car but diverged with distance. Yaw error is a *wedge*; pitch/height
error is a *near-field scale* error. **The first is fixed; the second is now the dominant one.**

⚠️ **And the verdict itself is over-optimistic for exactly the reason the old message was
over-pessimistic.** `diagnose.py:301` keys on `"extrinsics" in cam.source` — a single boolean for
"did the FOE run". Before the fix that made it claim "nominal values in use" while yaw was measured;
now it reports **OK** while camera height is still an unmeasured operator default and two pitch
estimates disagree by 4.8°. Same single-key test, opposite failure. It should key on *which*
parameters were measured, not on whether one estimator ran.

⇒ **Cheapest next action, by a wide margin: put a tape measure on the phone mount.** `scale_calib`
can only observe the **product** f·h, so an external height pins the focal length, which pins the
entire ground projection — and would settle the 3.50-vs-3.60 lane-width discrepancy at the same
time. One minute of physical measurement replaces all three declined estimators.

## 8. What is still open

1. ⭐ **Fix `estimate_foe`'s rotation gate to read the gyro** (§7b). This is now the top item: it is
   the upstream cause of the yaw defect, it is measured, and recovering the FOE would also deliver
   **pitch**, which is still nominal 0.0. Not a one-line swap — `calibrate_camera` passes the same
   `cam_flow` to `estimate_focal_from_gyro`, which genuinely needs the *image* flow, so
   `estimate_foe` needs its own rotation source rather than a substitution.
   Related and cheap: `--calib-seconds` (default 40) silently caps `cam_flow` coverage to the first
   40 s of an 81.75 s clip, halving the FOE's frame budget for no stated reason.
2. **The `MOUNT_NOT_CALIBRATED` message is now stale.** `diagnose.py:301` keys it on
   `"extrinsics" not in cam.source`, which only the FOE sets — so the block prints *"nominal values
   in use"* two lines above `mount_yaw_deg -7.01`. The `DEGRADED` **verdict is still correct**
   (pitch, roll, height and lateral genuinely are nominal), so the verdict logic was deliberately
   left alone; only the wording is wrong.
2. **`scale_calib` gets 0 usable tracks under both cv2 majors.** Independent of the FOE question,
   and unexplained.
3. **Recording A is unusable** (40 m, 8.5 s). If short clips are expected in the corpus, the 50 m /
   15 s gates are the thing to revisit — deliberately, not by loosening flags to make one file pass.
4. **A windscreen-mount height of 2.139 m was rejected as implausible.** Either the homographies are
   bad or the mount is not where the prior assumes; a tape measure settles it in one minute and
   would unblock `--cam-height`.
