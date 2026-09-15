# TrajectoryReconstruction

Ground-truth ego-trajectory generation from smartphone dashcam recordings, for training
end-to-end driving models.

Input: a Sensor Logger (Android) export — camera MP4 plus IMU / GNSS / orientation CSVs.
Output: for every video frame, the vehicle's past and future trajectory in the ego frame,
plus a bird's-eye view and an in-image projection.

See [AUDIT.md](AUDIT.md) for the review of the previous pipeline, the bugs found, and the
measured performance of this one.

---

## Quick start

```bash
python run_demo.py --session 0811 --frames 900 --render 12
```

Outputs land in `work/demo_out/`:

| file | contents |
|---|---|
| `frames/overlay_*.png` | side-by-side image projection + bird's-eye view |
| `ego_trajectories.json` | per-frame ego trajectory, camera model, sync result |
| `session_summary.png` | session path, speed profile, time-shift scan |

`--session 0814` runs the raw recording with the real MP4 instead (its GNSS is dead, so it
demonstrates the quality gate and the synchronisation, not the trajectory).

## Pipeline

```
load ──▶ QC gates ──▶ time sync ──▶ vehicle frame ──▶ trajectory ──▶ validation ──▶ render
```

```python
from trajlib.io_sensorlogger import load_session
from trajlib import timesync as TS, quality as Q
from trajlib.vehicle_frame import estimate_vehicle_frame
from trajlib.trajectory import estimate_trajectory, ego_trajectory
from trajlib.validate import validate
from trajlib.camera import calibrate_camera
from trajlib import viz

session = load_session("Liebst_ckelweg-2025-08-14_15-15-02-android")

qc = Q.merge(Q.check_gps(session["gps"]), Q.check_imu(session), Q.check_motion(session["gps"]))
if not qc.ok:
    raise SystemExit(qc)                     # never ship a session that fails here

video   = TS.probe_video(session.video_path)
flow    = TS.image_angular_rate(video)       # ~15 s for 78 s of 1080p
sync    = TS.synchronise(session, video, cam_flow=flow)

vframe  = estimate_vehicle_frame(session)    # one phone->vehicle rotation per session
traj    = estimate_trajectory(session, vframe)   # EKF + RTS over the WHOLE session
print(validate(session, vframe, traj, k=5))      # hold-out, not fit residual

cam = calibrate_camera(video, session, sync, traj=traj, cam_flow=flow, vframe=vframe,
                       height_m=1.25)        # measure height per vehicle!

for i, pts in enumerate(video.pts):
    ego = ego_trajectory(traj, sync.t_video_start + pts, t_past=3.0, t_future=5.0)
```

## Conventions

**Ego frame is FLU** (ROS / ISO 8855): `x` forward, `y` left, `z` up, origin at the vehicle
at the reference instant. `ego_trajectory` returns `t` running from `-t_past` to `+t_future`
with `x = y = 0` at `t = 0`.

**Session timebase** is `seconds_elapsed` = seconds since `Metadata.csv:recording epoch time`.
`session.elapsed_to_utc()` converts to absolute UTC epoch seconds.

**Camera frame is OpenCV** (`x` right, `y` down, `z` forward).

## Things you must supply

| parameter | why | default |
|---|---|---|
| `height_m` | camera height above the road — the image projection scales linearly with it | 1.25 m |
| `lateral_offset_m` | camera offset from the vehicle centreline (+left). Unobservable from *motion* — it cancels out of any feature-tracking residual — but `lane_calib` recovers it from lane geometry when markings are usable. Supply it as a fallback and to fix the sign; `--lateral-offset` | −0.35 m |
| `vehicle_width` | width of the drawn ribbon | 1.8 m |

`height_m` is genuinely not recoverable from the recording so far: an attempt to measure it from
ground-feature flow against GNSS speed came back at 1.06 m with an IQR of 0.84–1.43 m, too wide to
use (AUDIT.md F13). Measure it once per mount. The lateral offset *is* recovered per recording when
the road is marked — the operator value is the fallback and the sign prior.

## Module map

| module | role |
|---|---|
| `io_sensorlogger.py` | load a Sensor Logger export onto one timebase |
| `quality.py` | QC gates — frozen GNSS, rates, gaps, motion |
| `timesync.py` | container PTS + optical-flow/gyro cross-correlation |
| `vehicle_frame.py` | phone → vehicle rotation, once per session |
| `accel_source.py` | tiered longitudinal-acceleration selection with quality scoring |
| `trajectory.py` | session-level EKF + RTS smoother; per-frame ego slicing |
| `validate.py` | k-fold hold-out + internal consistency |
| `camera.py` | intrinsics from gyro, mount angles from the focus of expansion |
| `lane_calib.py` | mount yaw from the lane vanishing point; lateral offset from the ego-lane centre |
| `scale_calib.py` | separates focal from camera height via `f*h` and lane width |
| `frames.py` | uniform frame access for an MP4 or a folder of extracted frames |
| `plane_calib.py` | camera height from road-plane homography decomposition |
| `vp_calib.py` | roll from the vertical vanishing point |
| `diagnose.py` | OK / DEGRADED / REJECT verdicts with the evidence behind each |
| `viz.py` | BEV panel and in-image ribbon |
| `frame_folder.py` | treat a folder of extracted frames as a video source |
| `steering.py` | curvature, steering-wheel angle, drivability constraint |
| `ground_calib.py` | fit roll/pitch/yaw/height from tracked road features |

## Requirements

`numpy`, `scipy`, `pandas`, `opencv-python`, `matplotlib`, and **ffmpeg/ffprobe on PATH**.

## Measured accuracy

On the 2025-08-11 session, **hold-out** (every 5th GNSS fix removed and refitted, so the filter
bridges ~5 s blind):

| | |
|---|---|
| position | RMS **2.23 m** (median GNSS `horizontalAccuracy` on the same session: 3.79 m) |
| speed | RMS **1.27 m/s** |
| heading (v ≥ 2.5 m/s) | RMS **0.84°** |

Camera mount recovered from the data as yaw −7.2°, pitch +6.7° (far from nominal). An independent
Hough vanishing point agrees to ~3° — enough to confirm the mount is genuinely off-axis, but it is
a coarse sanity check, not a second precise measurement: the per-frame Hough VP scatters by
150–290 px, whereas the FOE uses ~10⁵ flow vectors at 93 % inliers.

## Steering angle

`trajlib/steering.py` estimates the steering-wheel angle from the reconstructed motion with a
kinematic bicycle model plus a linear understeer term:

    delta = atan(L * kappa) + K_us * a_lat        wheel = delta * steering_ratio

Defaults are for the **Audi A6 e-tron** (wheelbase 2.946 m, ratio ~15.9:1). Note the car can be
ordered with progressive (variable-ratio) steering, in which case a single ratio makes the estimate
accurate in *shape* and approximate in *scale*.

It is a derived quantity, not a measurement, so its error is inherited: with `kappa = omega / v`
the relative error is the yaw-rate error plus the speed error. The gyro is excellent here, so speed
dominates -- ~1.3 m/s RMS, i.e. 12 % at 11 m/s but 40 % at 3 m/s, which is exactly where the
sharpest steering happens. `SteeringResult.sigma_wheel_deg` carries this through (median 1.3 deg,
p90 5.6 deg on the 08-11 session). Below 1.5 m/s samples are flagged invalid rather than
extrapolated: steering is genuinely unobservable from a stationary vehicle's motion.

Rate limiting (default 180 deg/s) is applied to the **read-out only**. Letting it rewrite the path
is available via `apply_drivability` but off by default -- see AUDIT.md for why.

## Known limitations

* **Speed accuracy is GNSS-limited on the Galaxy recordings** (~1.3 m/s RMS across 5 s outages),
  whose accelerometer carries no usable vehicle acceleration — see F10 in AUDIT.md. The iPhone 14
  recordings do have a usable tier-1 accelerometer (quality 0.77) and reach 0.18–0.54 m/s.
* **Trajectory is 2D.** Height comes from neither GNSS altitude nor the barometer yet; the
  projection assumes a locally flat road.
* **Camera roll is assumed zero** unless `vp_calib` finds enough vertical structure. The focus of
  expansion is roll-symmetric and constrains yaw and pitch only.
* **Camera height is the operator default (1.17 m), not measured.** It scales the whole ground
  projection. The measured lane width bounds it loosely to ≈1.05–1.35 m; see AUDIT.md F13.
* GNSS is the fused platform fix, not raw observables.
* **The ribbon is centred on the camera's ground track, not the vehicle centreline.** The
  trajectory origin is effectively the phone/GNSS antenna. `lane_calib` measures the offset per
  recording from the ego-lane centre, assuming the driver is lane-centred *on average over the
  session*; on an unmarked or persistently off-centre drive it declines and the operator prior is
  used instead.
* **The lane estimators need marked, reasonably straight road at speed.** Below 12 m/s, on
  unmarked roads, or on a continuously curving route they decline and yaw falls back to the FOE —
  which carries the curve-induced bias described in AUDIT.md F13.
* The mount is assumed rigid for the whole session. A phone that is re-seated mid-recording
  will silently break the vehicle-frame estimate; `tilt_stability_deg` in `VehicleFrame` is the
  thing to watch.
