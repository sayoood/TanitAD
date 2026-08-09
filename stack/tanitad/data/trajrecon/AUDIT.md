# Audit — smartphone dashcam ground-truth trajectory pipeline

Review of `Sensor_Data_synchronization/SensorDataSynchronizationPipeline.ipynb` and
`Trajectory_Calucation_Deepthink/TrajectoryCalculationPipeline.ipynb`, validated against
the two recordings on disk.

Data used
* `Liebst_ckelweg-2025-08-14_15-15-02-android` — raw export, 78 s, real dashcam video, **GPS dead**
* `synchronized_output/` (2025-08-11) — 2247 exported frames, **GPS healthy**, reassembled into a
  continuous session (`work/session_2025-08-11`) so the estimator could be validated against it

---

## 1. Findings ranked by impact

### F1 — `creation_time` is the END of the recording, not the start (critical)

`extract_video_timestamps_and_frames()` timestamps every frame as
`creation_time + POS_MSEC`. Android's MP4 muxer writes `creation_time` when it **finalises**
the file. Proof on the 08-14 recording:

| quantity | value |
|---|---|
| `Metadata.csv:recording epoch time` | 2025-08-14 15:15:02.750 UTC |
| `format.tags.creation_time` | 2025-08-14 15:16:21.000 UTC |
| video duration | 77.713 s |
| `creation_time − duration` | 15:15:03.287 → **0.537 s after logging start** |

If `creation_time` were the start, the video's first frame would land 78.25 s into the session —
after the sensor log (78.4 s) had essentially ended. The recording would be empty. It is not.

### F2 — the sync anchors to the first GPS fix, so the error is whatever the GNSS warm-up was (critical)

`link_sensors_to_frames()` computes `time_offset = video_start − first_gps_local_time`, then adds
that offset to sensor timestamps that were **already absolute UTC**. Working the algebra through,
frame at PTS `p` ends up paired with sensor time

```
elapsed(sensor) = elapsed(first GPS fix) + p        # what the code does
elapsed(sensor) = elapsed(video start)   + p        # what is correct
```

The two `creation_time` errors partially cancel, leaving a residual equal to
`t(first GPS fix) − t(true video start)`. That is **uncontrolled and varies per recording**:

| recording | residual |
|---|---|
| 2025-08-14 | +0.07 s (lucky — GPS fix arrived 0.6 s in) |
| 2025-08-11 | **+0.30 s** (measured: true start 0.733 s, code used 1.037 s) |
| cold GNSS start | seconds |

0.30 s at 11 m/s is **3.3 m of vehicle travel and ~9 frames** of misalignment — the label is
attached to the wrong image.

### F3 — per-frame 5 s windows are the wrong architecture (critical)

`process_frame()` re-solves an independent filter inside `[t_frame, t_frame + 5 s]` for each of
2247 frames. Consequences:

1. **No past.** The window starts *at* the frame, so each solve begins from a cold zero-information
   state. The brief asks for backward *and* forward optimisation; only forward existed.
2. **Neighbouring frames disagree.** Frames 33 ms apart are solved independently, so differentiating
   the exported positions across frames does not reproduce the exported velocities.
3. **Wasted information.** A 5 s window sees ~5 GPS fixes; the session has 78. IMU biases are not
   observable from 5 fixes.
4. **Cost.** 2247 windows × 296 KB of duplicated sensor JSON = the 2 GB `synchronized_output.zip`
   for 78 s of driving; each sensor sample is stored ~150 times.

### F4 — no data-quality gate: bad GNSS produces confident, wrong ground truth (critical)

All 78 rows of `Liebst_ckelweg-2025-08-14/Location.csv` are byte-identical — a cached last-known-fix
that never updated during the whole drive (lat/lon/speed/bearing/accuracy constant; `speed` frozen at
2.716 m/s, `bearing` at 106°). The old code has no check for this, and would have emitted a smooth,
plausible-looking straight-line trajectory at constant speed for a drive that actually turned corners.
That is the worst possible failure mode for training data: silently wrong, not obviously broken.

### F5 — `bearingAccuracy == 0` is treated as "perfect", not "unknown" (high)

```python
valid_b_mask = (v_gps >= 1.0) & (df_gps[acc_col].values > 0.0)
```
On the 08-14 device `bearingAccuracy` is 0 for every fix. The mask goes all-False, the code falls
through to `clean_math_bearing = full(..., 90 − raw_bearing[0])` — a **constant heading** — and every
trajectory becomes a straight line. Device- and OS-dependent, and silent.

### F6 — single-axis IMU alignment, re-decided every window (high)

`align_axis()` picks the one raw IMU axis best correlated with a GPS derivative, per 5 s window.
A mount that is not near-axis-aligned is mis-modelled by up to 45°; the winning axis can flip between
adjacent windows, making the output discontinuous; and it needs GPS, so it silently falls back to a
hard-coded default exactly when GPS is bad.

### F7 — the Kalman model is not a valid smoother (high)

In `KalmanSmoother1D`:
* `a_adj = a_raw − mean(a_raw) + mean_gps_a` forces the IMU mean to the GPS mean *inside each
  window*, which destroys the bias state's job and injects a window-dependent offset.
* `u_a = a_eval*conf + gradient(v_ref)*(1−conf)` blends the control input with a derivative of the
  measurement — the same GPS data then enters again as `z`. Using a measurement as its own control
  input double-counts it and makes the covariance meaningless.
* `conf` is binary (0 or 1) from `abs(corr) < 0.25`, so the model switches discontinuously.
* `yaw_rate_smooth = u_w − out_w[:,1]` reports the *input minus bias*, not the smoothed state.

### F8 — frame timing from `CAP_PROP_POS_MSEC` is right, but the fps assumption elsewhere is not (medium)

Measured on the 08-14 video: nominal 30 fps, actual **29.9436** fps, per-frame jitter ±1.5 ms.
`POS_MSEC` (used here) is correct. Any downstream code using `index / 30` accumulates **1.7 s** of
drift over 78 s.

### F9 — timezone handling is fragile even though it happens to work (medium)

`datetime.fromisoformat(creation_time_str).replace(tzinfo=timezone.utc)` **discards** a real offset
if one is present. The fallback path `datetime.fromtimestamp(mtime, tz=timezone.utc)` mislabels local
time as UTC. Neither fires on this data, but both are latent.

Verified *not* a bug: `local_time` in the CSVs is consistent with the epoch `time` column — there is
no double-offset. (`Metadata.csv:recording time` is UTC-based while `recording timezone` says
Europe/Berlin; a Sensor Logger labelling quirk, harmless.)

### F10 — the accelerometer carries no usable vehicle acceleration on this device (critical for method choice)

This is a data finding, not a code bug, but it determines what the pipeline can achieve.

`Accelerometer.csv` is Android *linear* acceleration (gravity removed by the fusion). On the 08-11
session its forward component integrates to **−0.8 m/s** over 75 s while the vehicle actually gained
**+7.8 m/s**; correlation with GPS `dv/dt` is **0.06**. During a braking event from 8.2 m/s the mean
forward component is +0.008 m/s².

Attempts to recover it all fail:

| gravity removal | corr(a_lat, ω) — should be strongly positive | implied speed |
|---|---|---|
| Android linear accel | +0.165 | 1.4 m/s |
| `TotalAcceleration − Gravity` | −0.043 | −0.4 m/s |
| `TotalAcceleration − lowpass(0.5…0.01 Hz)` | −0.01 … −0.12 | ≈0 |

An assumption-free scan over **all** 3D directions finds a best correlation of only **0.32–0.36**
against the physically-required lateral acceleration `v·ω`, with amplitude attenuated ~3×
(measured std 0.29 vs expected 0.79 m/s²). Same on both sessions.

The **gyroscope is excellent** — heading RMS 0.6–0.8° against held-out GNSS.

Root cause is not isolated between Android's fusion high-pass, a compliant phone mount, or
app-side filtering. See recommendations.

---

## 2. What the corrected pipeline does

| stage | module | approach |
|---|---|---|
| load | `io_sensorlogger.py` | anchor on `recording epoch time`; recompute `seconds_elapsed` from integer ns |
| QC | `quality.py` | frozen-GNSS, accuracy, rate, gap and motion gates — **fail loudly** |
| sync | `timesync.py` | real container PTS + `creation_time − duration`, then refined by correlating image-derived angular rate against the gyro triad, in 3 frequency bands |
| mount | `vehicle_frame.py` | one rotation per session: up from gravity (or gyro PCA), left from `a_lat = v·ω`, forward closes the triad |
| accel | `accel_source.py` | tiered source selection with automatic quality scoring; disables itself rather than injecting noise |
| trajectory | `trajectory.py` | one EKF + RTS smoother over the **whole session**; per-frame ego windows are slices of it |
| validation | `validate.py` | k-fold **hold-out** over GNSS fixes + internal consistency |
| camera | `camera.py` | focal from gyro/flow regression; mount yaw+pitch from the focus of expansion |
| render | `viz.py` | BEV panel + image ribbon with correct behind-camera clipping |

### Measured performance (08-11 session, hold-out — every 5th GNSS fix removed, ≈5 s outages)

| metric | value |
|---|---|
| position | mean 1.61 m, **RMS 2.23 m**, p95 4.53 m, max 6.93 m |
| speed | mean 0.70, **RMS 1.27 m/s** |
| heading (v ≥ 2.5 m/s) | mean 0.66°, **RMS 0.84°**, p95 1.55° |
| path length | 784.2 m est. vs 774.7 m GNSS speed integral (1.2 %) |
| `d(pos)/dt` vs speed state | RMS 0.004 m/s |
| course vs heading | RMS 0.74° |
| longitudinal accel | p99 2.69 m/s², max 4.36 (physically plausible) |

Context: median GNSS `horizontalAccuracy` on this session is 3.79 m, so 2.23 m RMS means the
smoother is genuinely averaging down receiver noise rather than chasing it.

Two tuning decisions were made by minimising **hold-out** error rather than by eye:

* `hacc_inflate = 1.0` — inflating the receiver's reported accuracy consistently *hurt*
  hold-out position (2.23 → 2.87 m at 1.5×). Trust the receiver's own number.
* `sigma_accel_no_imu = 0.8` — larger values keep improving hold-out position (1.87 m at 1.6)
  but at the cost of unphysical acceleration (p99 6.7 m/s²). 0.8 is the best position error
  subject to p99 longitudinal acceleration staying under ~3 m/s². For training data, physical
  plausibility is worth more than the last half-metre.

### Low-speed, stop and turn behaviour

Aggregate hold-out numbers hide the regimes that actually break a unicycle-model filter, so these
were tested separately (`work/stress_lowspeed.py`, `work/stress_synthetic.py`).

**Turning is not a problem.** Hold-out error is *best* where the yaw rate is highest:

| \|yaw rate\| | n | RMS |
|---|---|---|
| < 3 °/s | 45 | 2.32 m |
| 3–10 °/s | 24 | 2.30 m |
| > 10 °/s | 7 | **1.23 m** |

The 90° junction, four heading wrap-around crossings, and yaw rates up to 30 °/s are all handled.
Covariance stays positive semi-definite throughout (min eigenvalue 6.0 × 10⁻⁶), stays symmetric to
4 × 10⁻¹⁴, and there are no NaNs.

**Low speed is worse, and the transition band is worst:**

| speed | n | RMS |
|---|---|---|
| < 1 m/s | 2 | 3.31 m |
| 1–4 m/s | 7 | 2.47 m |
| 4–9 m/s | 9 | **4.82 m** |
| > 9 m/s | 58 | 1.35 m |

The 4–9 m/s band is the accelerate/decelerate transition, which is exactly where a missing
longitudinal accelerometer (F10) hurts most. The small-n bands are indicative, not conclusive.

**The 08-11 session contains no true stop.** Minimum speed 0.67 m/s, only 2 of 77 GNSS fixes below
0.5 m/s, 0.0 % of samples under 0.5 m/s. The standstill code paths therefore could not be validated
from it, and a synthetic 10 s stop was built instead. It exposed three real defects:

1. **The ZUPT threshold was dead code.** A stationary receiver does not report 0.00 m/s — Doppler
   noise keeps it jittering a few tenths. At `zupt_speed = 0.25` the constraint fired on 0 of 8
   fixes in a stop with 0.4 m/s speed noise. Raised to 0.7 m/s.
2. **The constraint was only applied at GNSS instants**, so the state drifted for a second and got
   yanked back. It now holds across the whole stationary span.
3. **Heading random-walked through the stop** (~5° over 8 s). At v = 0 the position carries no
   heading information and the GNSS bearing is gated off, so nothing was holding it. Process noise
   is now scaled down 20× while stationary.

Plus two more found by inspection: **speed could go negative** (the unicycle model allows reverse;
a forward dashcam never does) — now constrained by a pseudo-measurement; and the **trajectory ribbon
collapsed to zero width at a standstill**, because it normalised a zero-length path tangent. It now
uses the vehicle heading, which is defined even when stopped.

After the fixes the synthetic stop holds speed below 0.001 m/s, position drift to 0.28 m and heading
wander to 1.37° over 5.5 s. Real-session metrics are unchanged (the new path never fires there).

**Still untested on real data:** a genuine multi-second stop. A recording with several junction
stops is the one piece of data needed to close this out.

### The "ripple" in the exported path: mostly a rendering artefact

A visible wobble in the bird's-eye trace turned out to be two separate things, and only one of
them was real.

**The apparent ripple was rasterisation.** Measured on straight stretches against a
constant-curvature reference, the lateral wobble is **0.042 m RMS / 0.24 m peak — and it is
identical in every filter configuration**, including `hacc_inflate`, `sigma_lat`, bearing-noise
inflation, gyro low-pass, and even switching the GNSS position measurement off entirely
(0.0418 m either way). The trajectory did not change between the "smooth" matplotlib stills and
the "rippled" video. What changed was the renderer: OpenCV rasterises at integer pixels unless
given a `shift`, and at ~18 px/m that quantisation is ±2.8 cm — enough to make a smooth line look
wavy, while matplotlib draws in float coordinates. Drawing with 1/64 px sub-pixel precision
removed it.

**A real component exists but is small.** The residual 0.042 m does come from 0.3–1 Hz gyro
content, which survives every measurement-side knob because the gyro is the propagation *input*.
Expressed as a steering rate it is 242 °/s at p99 — faster than a driver saws a wheel on a
straight road.

**Constraining the path to fix it was a bad trade, and is now off by default.** Low-passing
curvature to hit a 180 °/s limit needs a 0.366 Hz cutoff, which also smears genuine manoeuvres:

| | |
|---|---|
| heading rotation at the junction turn | up to **4.7°** |
| ego-window 5 s endpoint displacement | median 0.18 m, p95 0.94 m, **max 2.48 m at t = 52.0 s** |
| ripple actually removed | 0.042 → 0.037 m |

2.5 m of error at a turn is enough to throw the projected ribbon off the road entirely — which is
exactly what it did. Buying that to remove 5 mm of ripple is not worth it. `apply_drivability`
remains available and documented, but the default pipeline leaves the estimated path alone.
Rate-limiting is still applied to the **steering read-out**, where it costs nothing and stops the
gauge jittering.

### Correction: the road-feature calibration is ill-conditioned — do not use it

I initially applied the road-feature fit on top of the FOE calibration and shipped the result. That
was wrong, and it made the projection **worse**, visibly so in pitch. Refitting the same tracks with
different channel and free-parameter choices gives:

| configuration | roll | height |
|---|---|---|
| roll only, lateral channel | +0.40° | — |
| roll only, both channels | +0.79° | — |
| roll + height, lateral | +0.33° | 0.800 m *(hit bound)* |
| height only, long range | — | 1.109 m |
| all free (what was shipped) | +0.46° | 1.304 m |
| all free, repeat run | +0.30° | 1.337 m |

Roll spans 0.30–0.79°, height 1.11–1.34 m, two fits run into their bounds, and the best residual
improvement is **4 %** (1.379 → 1.328 m). Large, unstable parameter motion for no cost reduction is
overfitting, not calibration.

Two compounding causes:

1. **The longitudinal channel is biased.** The +2.96 m residual at 2–8 m is an LK tracking artefact —
   the road sweeps hundreds of pixels between the paired frames and the tracker under-reads it.
   Pitch and height are exactly the parameters that trade against longitudinal range, so that
   artefact flows straight into them.
2. **The lateral channel cannot see pitch or height at all.** Freed on that channel alone they run
   to their bounds while the residual collapses to 0.089 m — pure overfitting.

The FOE estimate this was overriding is far better conditioned: 318 000 flow vectors at 93 %
inliers, **direction**-based rather than magnitude-based, and independently cross-checked against a
Hough vanishing point. **The shipped configuration is FOE yaw/pitch + roll 0 + measured height.**
`--ground-calib` is retained but marked experimental and is off by default.

### Where the residual lateral offset comes from — and why no algorithm can remove it

The projected band sits consistently ~0.5–1 m right of the lane. That was diagnosed by fitting the
ground-plane extrinsics against tracked road features (`trajlib/ground_calib.py`): a patch of road
is a static world point, so between two frames its position in vehicle coordinates must change by
exactly the displacement the trajectory already knows, `P2 = R(-dpsi)(P1 - dp)`. Fitting
roll/pitch/yaw/height to minimise that residual over 5054 tracks is ordinary least squares — no
sliders.

**Result: the angular geometry is already right.** The lateral residual is unbiased at every range:

| range | n | longitudinal bias | lateral bias | lateral RMS |
|---|---|---|---|---|
| 2–8 m | 266 | +2.96 m | **−0.05 m** | 0.14 m |
| 8–15 m | 1169 | +1.58 m | **−0.03 m** | 0.48 m |
| 15–25 m | 1712 | +0.63 m | **−0.03 m** | 0.45 m |
| 25–45 m | 1901 | +0.65 m | **+0.02 m** | 0.35 m |

Fitted corrections were small — roll **+0.46°** (previously assumed 0 and never estimated, since the
FOE is roll-symmetric), pitch −0.21°, yaw +0.02°, height 1.25 → **1.30 m**. The large *longitudinal*
near-range bias is an LK tracking artefact: at 2–8 m the road sweeps hundreds of pixels between
frames and the tracker under-reads the displacement. It is not a calibration error.

**The offset is structurally unobservable from motion.** Translating the camera sideways displaces
every ground point identically at *both* ends of a track, so it cancels exactly out of `P2 - P1`.
No amount of feature tracking recovers it — that is a proof about the observability of the
parameter, not a shortcoming of the fit.

The only motion-free alternative is a lane-centring assumption. A first attempt
(`estimate_lateral_offset_by_lane_centring`: average a metric bird's-eye view over straight frames,
find the marking stripes) came back **inconclusive** on the Android footage — stripes at
1.68 / 0.63 / 0.03 / −0.78 m imply a 1.41 m lane, which the plausibility check rejects rather than
reporting a confident wrong number.

#### Correction (iOS recordings): lane geometry does recover it

The unobservability proof above is correct and still stands, but the conclusion drawn from it —
"measure it with a tape measure, no algorithm can help" — was too strong. It is unobservable *from
motion*. Lane markings are an external metric reference, and they supply exactly the information
that feature tracking cancels out. `trajlib/lane_calib.py` does this per recording, and on the
Pacific Coast Highway iPhone recording it succeeds where the earlier BEV-stripe attempt failed:
lane width **3.40 m**, ego-lane centre relative to the camera **+0.20 m**, giving
`lateral_offset_m = −0.20 m` against the operator prior of −0.35 m.

The earlier attempt failed not because the idea is unsound but because it worked from a stitched
bird's-eye average, which smears the stripes when the yaw is wrong. Detecting ridges per frame and
histogramming their back-projected lateral position is far better conditioned.

The operator prior remains the fallback: when the road is unmarked, too curved, or the scatter is
wide, the estimator declines and `--lateral-offset` is used. **The sign is still worth supplying** —
it is the one thing an operator knows for free.

### One more artefact found and fixed during tuning

The zero-velocity update originally fired on any single GNSS fix reading below 0.25 m/s. On the
08-11 session the receiver emits an isolated `speed = 0.0` at t≈48 s and again at t≈54 s while the
vehicle is actually crawling at 2–5 m/s between them. Each one punched a hard notch to exactly zero
into the smoothed speed — a fake brake-to-stop-and-launch pair. Requiring three consecutive
near-zero fixes before a ZUPT fires removed it, and improved everything at once: hold-out position
2.66 → 2.23 m, speed 1.54 → 1.27 m/s, and p99 longitudinal acceleration 6.36 → 2.69 m/s².

### Camera calibration, cross-validated

Both mount angles are recovered from the data, then checked against a completely independent
signal:

| quantity | value | how |
|---|---|---|
| focal length | 1677 px (HFOV 59.6°) | regression of image yaw rate on gyro yaw rate, r = 0.91, n = 89 |
| focus of expansion | (748, 737) px | 270 534 flow vectors over straight-driving windows, 93 % inliers |
| Hough vanishing point | ~(806, 684) px | straight edges in single frames — **uses no motion at all** |

Both put the mount well off nominal — **yaw −7.2°, pitch +6.7°** — i.e. the phone was cradled
noticeably left-of-straight and tilted up. Assuming a nominal mount would have put the horizon at
row 540 instead of 737; every projected trajectory would have been wrong by that much. That is the
substantive result of the cross-check, and it is unambiguous: both methods place the vanishing
point ~150 px left of and ~150 px below the principal point.

On the precision of the agreement, be careful. The Hough VP is **noisy**: individual frames scatter
by 150–290 px, and its median only settles with enough frames:

| frames in median | Hough VP |
|---|---|
| 5 | (794, 703) |
| 10 | (873, 665) |
| 20 | (836, 676) |
| 40 | (806, 684) |

So the honest FOE-vs-Hough disagreement is **~100 px ≈ 3.4°**, not the 0.7° that a 5-frame median
happened to produce in one run. The cross-check is a coarse sanity check on the FOE, not a
competing measurement of comparable precision — the FOE is built from ~10⁵ flow vectors at 93 %
inliers, the Hough VP from a few dozen line segments per frame. `calibrate_camera` now medians over
40 frames, reports the Hough estimate's own scatter alongside the disagreement, and only warns when
they differ by more than 12 % of the image width.

One correction was needed to get the FOE itself right. The flow-line construction is only valid for
*pure translation*, and a windshield mount pitches constantly over road bumps. Filtering on the
vehicle's yaw rate does not catch that — the bumps are vertical. Rejecting frame pairs whose own
measured yaw/pitch/roll rate exceeds 0.035 rad/s is what makes the FOE stable.

### Synchronisation results

| recording | old pipeline | corrected | agreement |
|---|---|---|---|
| 2025-08-14 | 0.606 s (first GPS fix) | **0.2134 s** | 3 bands within 1 ms, score 0.59 |
| 2025-08-11 | 1.037 s (first GPS fix) | **0.7327 s** | 3 bands within 96 ms, score 0.89 |
| PCH (iPhone 14) | — | **0.1068 s** | bands within 7 ms, score 0.93 |
| Rose Ave (iPhone 14) | — | **0.118 s** | bands within 17 ms, score 0.98 |

---

## 2b. Findings from the iOS recordings

Two iPhone 14 recordings exposed three defects that the Android data could not, because each one
needed a convention the Android exports happen to satisfy.

### F11 — the gravity vector's sign is platform-dependent (critical)

Android's `Gravity` stream points **up** (a phone lying face-up reads +9.81 on z); iOS CoreMotion
points it **down**. The vehicle-frame estimator assumed the Android convention, so on iOS the whole
frame was inverted about the vertical and every turn integrated backwards. Measured on Rose Ave:
`corr(ω_up, GNSS dψ/dt) = −0.43`.

The fix is not to special-case iOS but to *measure* the sign: `_sign_from_gnss` correlates rotation
about the candidate up-axis against the GNSS heading rate and flips when they disagree, reporting
the correlation in the provenance string. On PCH this took hold-out position from 1.30 m to
**0.62 m** and course-vs-heading from 0.144° to **0.014°** — that recording's sign had been wrong
too, just less visibly.

### F12 — "quality unknown" was treated as "trusted" (critical)

`accel_source` scored each candidate accelerometer against GNSS-derived acceleration and set
`usable = isnan(quality) or quality >= min_quality`. A NaN score therefore meant *use it*. The score
was NaN whenever a single GNSS speed sample was NaN, because the gate `np.ptp(speed) > 1.0` returns
NaN and tests False.

On Rose Ave this let a **reversed** forward axis through — `corr(a_fwd, GNSS dv/dt) = −0.77`, an
integrated +70.7 m/s against a true 0→0 speed change. Hold-out position was **76.82 m**; the
recording was rejected as bad data when the data was fine.

The same NaN also disabled the forward-sign check in `vehicle_frame`, which is what reversed the
axis in the first place — one missing sample silently switched off two independent safeguards. Both
are now NaN-safe (filter to finite samples, then test), unknown quality no longer implies usable,
and a strongly negative correlation is reported as `REVERSED … vehicle frame is wrong` rather than
being quietly negated, so the frame gets fixed instead of patched over.

| Rose Ave | before | after |
|---|---|---|
| hold-out position RMS | 76.82 m | **2.41 m** |
| hold-out speed RMS | 6.57 m/s | **0.54 m/s** |
| hold-out heading RMS | 6.23° | **4.81°** |
| verdict | REJECTED | DONE |

### F13 — the FOE's yaw is biased by curves; lane markings fix it (high)

The FOE is fitted over the whole session, curves included, so a route with a net turning bias walks
it sideways. On PCH the FOE put "straight ahead" at u = 1087 px while the lane vanishing point put
it at **u = 1120 px** — a **0.98° yaw error**, which is 0.70 m of lateral displacement at a 40 m
look-ahead and was plainly visible as a projected ribbon drifting off the lane toward the horizon.

The FOE's *pitch* is unaffected and excellent: the two agree on the horizon row to **0.7 px**.

`trajlib/lane_calib.py` estimates yaw from the lane vanishing point on straight, fast frames. It is
measured as a vanishing point in the image rather than by back-projecting to the road plane, and
that distinction matters — a VP is invariant to how high a feature sits above the road, so
guardrails and kerbs are valid constraints. The road-plane version of the same idea assumes z = 0
and is biased by exactly those features: it read 4.51° against the VP's 4.79°.

| estimator | yaw |
|---|---|
| FOE (previous) | +3.81° |
| road-plane parallelism | +4.51° (biased by elevated features) |
| **lane VP** | **+4.81°** |

The VP is stable across every split of the data — 4.66–4.84° over halves, stricter straightness
gates and road-surface-only segments, with a half-split spread of 0.02° on the production run.

**Camera height remains unmeasured.** An attempt to recover it from ground-feature flow against
GNSS speed (back-projected distances scale linearly with the assumed height, so apparent ground
speed does too) was too noisy to use: median 1.06 m with an IQR of 0.84–1.43 m, and it diverged
outright at longer baselines as LK lost the road texture. The measured lane width of 3.40 m at the
assumed 1.17 m is consistent with a real 11–12 ft lane, which bounds the height loosely to
≈1.05–1.35 m. It stays at the operator default and is labelled as such.

### Cross-platform support

Platform is detected from the metadata `platform` field, then the device name, then the file
inventory. Beyond the gravity sign, iOS differs in three ways that are now handled:

* **Accelerometer units** — iOS exports `AccelerometerUncalibrated.csv` in *g*, Android
  `TotalAcceleration.csv` in m/s². Detected from the data rather than the filename: a
  gravity-bearing stream has a mean vector norm of 1 g. Linear acceleration is then *derived*
  (`total_accel − gravity`), never inferred from a filename.
* **Sentinel values** — iOS writes `−1` for unavailable speed/bearing/accuracy. Left as −1 these
  poison the filter; one such value in the first fix made the entire Rose Ave state NaN.
* **`creation_time`** — the END of the recording on Android, the START on iOS. `pick_anchor` chooses
  by physical plausibility rather than by platform.

---

## 2c. Findings from the projection review

Four defects found by looking at rendered frames, not by any gate. They share a
pattern worth naming: **every check I had built compared a parameter against a
method that shared its assumptions**, so the errors cancelled and the checks
passed. Non-circular checks are the cheap ones and they were the missing ones.

### F14 -- focal length is not observable from this footage (critical)

Back-projection onto the road plane gives

    lateral       y = (u - cx) * h / (v - v_horizon)      -- no f
    longitudinal  x = f * h / (v - v_horizon)             -- only the product f*h

so the ground plane sees ``h`` laterally and ``f*h`` longitudinally and **can
never separate f from h**. Back-project with a wrong f, re-project with the same
wrong f, and the pixels return exactly where they started.

The Pacific Coast Highway recording shipped at f = 1901 px (53.6 deg HFOV)
against Rose Ave's 1427 px (67.9 deg) from the *same phone and camera* -- 33%
apart -- and every check passed: plausible lane width (tests only h), lane
markings re-projecting onto the paint (self-consistent), centring correct to
6 cm. Only the depth scale was wrong, rendering the 40 m curvature at the
visual position of 30 m. That is what "the curvature is not looking right"
looked like.

``scale_calib`` closes the missing direction by measuring ``f*h`` from tracked
road features against GNSS distance, using ``1/(v - v_horizon) = (x0 - s)/(f*h)``
-- image rows only, long baselines, no flow magnitude, so it survives the static
windscreen artefacts that defeat flow-based estimators here. PCH: 1709 from 495
tracks, 15% spread, giving f = 1361 px and h = 1.256 m.

**Five methods, five answers.** Gyro regression gave 1901 (PCH) and 1427 (Rose
Ave); ``f*h``/lane-width 1361; a per-band transfer function ~1050; rotation
homography autocalibration was degenerate at 1 deg/frame. Attempts to arbitrate
with the MUTCD dash cycle failed because PCH's markings are solid and the
detector locked onto ~5 m pavement joints, and a GNSS row-prediction test gave
41 px errors for every candidate. **The intrinsics cannot be recovered
reliably from driving footage of this kind.** A checkerboard calibration, once
per device, removes the whole problem; that is the recommendation.

The one cheap non-circular gate now in place: an iPhone cannot have a 53.6 deg
HFOV. ``check_focal_plausible`` rejects anything outside 55-85 deg and would
have caught this on the first run, for free.

### F15 -- the trajectory was the phone's path, not the vehicle's (critical in turns)

GNSS locates the phone. A rigid body's points trace *different* curves, and a
point mounted L metres ahead of the rear axle swings wide through every turn.
The renderer applied the lateral offset as a constant shift, ``y = y - W``, and
ignored L entirely. The residual is

    dy = -L.sin(psi) + W.(1 - cos psi)

-- zero on straight road, growing with heading change. Measured with L = 2.4 m:
PCH p90 0.39 m; **Rose Ave p99 2.08 m, max 2.57 m** at intersections. Straight-road
validation showed centimetres throughout, which is exactly why it survived.

``to_vehicle_reference`` now applies the proper rigid transform per point and
recomputes speed along the transformed path (phone and rear axle differ by about
``omega*L``, ~0.6 m/s at 0.3 rad/s). The exported trajectory is the **rear axle's**
path, and ``calibration.json`` records ``trajectory_reference``.

### F16 -- calibration.json contradicted itself (high)

``extrinsics.lateral_offset_m`` was hardcoded to ``0.0`` while
``parameters.lateral_offset_m`` carried the real value. Stale from before the
camera model held its own mount offsets. Rendered video was unaffected, but every
downstream consumer of ``extrinsics`` got the wrong mounting position -- and it
misled a diagnostic during this very review into reporting a metre of error that
did not exist.

### F17 -- one speed gate served two different measurements (medium)

``_straight_fast_times`` gates at 12 m/s. The lane vanishing point needs that;
the lateral offset does not. On the urban Rose Ave recording it kept only the few
fast stretches, whose lane structure differs, and returned -0.325 m where every
window of a lower-speed sample returns **-0.20 m** -- the same value the other
recording from that phone gives. The lateral step now uses its own 6 m/s floor.

### What is measured, and what is still assumed

| parameter | PCH | Rose Ave | 08-11 Galaxy |
|---|---|---|---|
| focal | 1361 px *measured* | 1427 px *gyro* | 1677 px *gyro* |
| height | 1.256 m *measured* | 1.17 m *assumed* | 1.17 m *assumed* |
| yaw | lane VP | lane VP | FOE (lane declined) |
| lateral | -0.20 m *measured* | -0.20 m *measured* | *unmeasurable, IQR 0.9 m* |
| mount L | 2.1 m *operator* | 2.1 m *operator* | 2.1 m *operator* |

Three physical constants -- focal, camera height, mount position -- account for
most of the remaining error and none of them is reliably recoverable from
driving footage. A checkerboard and a tape measure would fix all three in under
an hour, and no further algorithm work on this data will.

---

## 3. Recommendations

**Change the logging configuration.** Export `TotalAcceleration`, `Gravity` *and* `Orientation`
alongside the current streams. The 08-11 export carried only Accelerometer/Gyroscope/Location, which
removes any chance of recovering longitudinal dynamics from inertial data.

**Isolate the accelerometer problem (F10) with a controlled test.** Clamp the phone rigidly (no
suction arm), drive a straight road with several hard accelerate/brake cycles and a good GNSS fix,
then compare `TotalAcceleration` projected on the forward axis against GNSS Doppler `dv/dt`. That
separates "Android filtering" from "mount compliance" in one 5-minute experiment. Until it is
resolved, speed accuracy is capped at ~1.5 m/s RMS across GNSS outages.

**Log raw GNSS if the app allows it.** comma2k19 gets ~40 % better positions by reprocessing raw
GNSS observables with Laika instead of trusting the fused fix. Carrier-phase / Doppler processing is
the single biggest available accuracy gain.

**Stop storing per-frame sensor windows.** Store one session-level sensor file plus one trajectory
file, and slice at training time. This removes ~150× duplication.

**Add a stationary-start convention.** Ten seconds of stillness at the start of each recording gives
a clean gyro-bias estimate and an unambiguous gravity/mount reference.

**Camera height and lateral offset must be measured once per vehicle** and stored. They cannot be
recovered from this data, and the image projection scales directly with height.

---

## 4. Reference points from the literature

* **comma2k19** ([arXiv:1812.05752](https://arxiv.org/abs/1812.05752)) — the closest published analogue:
  road-facing camera + phone-grade GNSS + 9-axis IMU, ground-truth poses from a tightly coupled
  INS/GNSS/vision optimiser over raw GNSS observables.
* **Camera–IMU temporal calibration** — cross-correlating angular rates is the standard first step
  (Kalibr's `--imu-delay-by-correlation`); a continuous-time batch fit refines it
  ([arXiv:1808.00692](https://arxiv.org/pdf/1808.00692)).
* **RTS smoothing for batch GNSS/INS** is the accepted way to turn a causal filter into ground truth
  — used here at session level.

Next step beyond this work: replace the EKF+RTS with a factor graph (GTSAM) over IMU pre-integration
factors plus GNSS and visual-odometry factors. That buys full 6-DoF poses and consistent covariance,
and is the path comma2k19 took.
