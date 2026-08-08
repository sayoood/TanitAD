"""Pre-flight diagnosis: decide whether a recording *can* be processed at all.

The governing rule of this pipeline is **never process blind**.  The original
code had no gate, and the consequence is on disk: the 2025-08-14 recording has
78 byte-identical GNSS rows -- a cached fix the receiver replayed for the whole
drive -- and it would have produced a smooth, confident, entirely fictional
straight-line trajectory.  Bad ground truth that *looks* fine is worse than no
ground truth, because nothing downstream will ever flag it.

So every recording is examined before any solving happens, and the outcome is
one of three:

``OK``       every check passed; process it.
``DEGRADED`` usable, but something is worse than it should be; process it and
             record the caveat so the session can be filtered later.
``REJECT``   a check failed that makes the output untrustworthy; write the
             analysis and stop.

Each finding carries the measurement that produced it, so a rejection is a
diagnosis rather than a verdict.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

OK, DEGRADED, REJECT = "OK", "DEGRADED", "REJECT"
_RANK = {OK: 0, DEGRADED: 1, REJECT: 2}


@dataclass
class Finding:
    level: str
    code: str
    message: str
    evidence: dict = field(default_factory=dict)

    def __str__(self):
        ev = "  ".join(f"{k}={v}" for k, v in self.evidence.items())
        return f"[{self.level:8s}] {self.code:22s} {self.message}" + (f"\n{'':35s}{ev}" if ev else "")


@dataclass
class Diagnosis:
    verdict: str = OK
    findings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, level, code, message, **evidence):
        self.findings.append(Finding(level, code, message, evidence))
        if _RANK[level] > _RANK[self.verdict]:
            self.verdict = level
        return self

    @property
    def ok(self):
        return self.verdict != REJECT

    def to_dict(self):
        return dict(verdict=self.verdict, stats=self.stats,
                    findings=[dict(level=f.level, code=f.code, message=f.message,
                                   evidence=f.evidence) for f in self.findings])

    def __str__(self):
        lines = [f"VERDICT: {self.verdict}"]
        for f in self.findings:
            lines.append(str(f))
        if self.stats:
            lines.append("measurements:")
            for k, v in self.stats.items():
                lines.append(f"    {k:28s} {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def diagnose_export(path: str) -> Diagnosis:
    """Structural check of an unpacked Sensor Logger export, before loading.

    Platform-aware: Android and iOS write partly disjoint stream sets, so a
    fixed list of "required" files would flag every iOS recording for missing
    TotalAcceleration and every Android one for missing Compass.
    """
    from .io_sensorlogger import detect_platform

    d = Diagnosis()
    present = set(os.listdir(path))
    plat = detect_platform(path)
    d.stats["platform_detected"] = plat
    for f in ("Metadata.csv", "Gyroscope.csv", "Location.csv"):
        if f not in present:
            d.add(REJECT, "MISSING_STREAM", f"{f} is absent", consequence="cannot process")
    for f in ("Accelerometer.csv", "Gravity.csv", "Orientation.csv"):
        if f not in present:
            d.add(DEGRADED, "MISSING_STREAM", f"{f} is absent", consequence="reduced capability")
    # the raw specific force has a different file name on each platform
    if not ({"TotalAcceleration.csv", "AccelerometerUncalibrated.csv"} & present):
        d.add(DEGRADED, "MISSING_STREAM", "no raw specific-force stream "
              "(TotalAcceleration / AccelerometerUncalibrated)",
              consequence="only the high-passed linear accel is available")
    cam = os.path.join(path, "Camera")
    vids = [f for f in os.listdir(cam) if f.lower().endswith((".mp4", ".mov"))] \
        if os.path.isdir(cam) else []
    if not vids:
        d.add(REJECT, "NO_VIDEO", "no video file under Camera/")
    elif len(vids) > 1:
        d.add(DEGRADED, "MULTIPLE_VIDEOS", f"{len(vids)} videos found; using the epoch-named one",
              files=vids)
    for f in present:
        if f.endswith(".csv"):
            p = os.path.join(path, f)
            if os.path.getsize(p) < 64:
                d.add(DEGRADED, "EMPTY_CSV", f"{f} is essentially empty",
                      bytes=os.path.getsize(p))
    return d


def diagnose_session(session, video=None) -> Diagnosis:
    """Content check: sensors, GNSS health, motion, and video/sensor consistency."""
    from . import quality as Q

    d = Diagnosis()
    gps = session.get("gps")
    d.stats["platform"] = session.platform
    d.stats["device"] = session.device

    # -- acceleration units -------------------------------------------------- #
    # Sensor Logger mixes g and m/s^2 within a single iOS export.  The loader
    # rescales from the data, but a stream whose magnitude matches neither is a
    # hard stop: silently treating g as m/s^2 is a 9.8x error that every
    # downstream number would absorb without complaint.
    for key in ("gravity", "total_accel"):
        if not session.has(key):
            continue
        v = session[key][["x", "y", "z"]].to_numpy(dtype=float)
        m = float(np.median(np.linalg.norm(v, axis=1)))
        d.stats[f"{key}_mag_ms2"] = round(m, 4)
        if not (8.5 < m < 11.0):
            d.add(REJECT, "BAD_ACCEL_UNITS",
                  f"{key} has median magnitude {m:.3f} m/s^2 after unit normalisation; "
                  f"it should be ~9.81. The unit could not be determined.",
                  stream=session.units.get(key, "?"))
    if session.units:
        d.stats["unit_notes"] = "; ".join(f"{k}: {v}" for k, v in session.units.items()
                                          if k in ("accel", "total_accel", "gravity"))

    # -- GNSS -------------------------------------------------------------- #
    qg = Q.check_gps(gps)
    d.stats.update(qg.stats)
    for f in qg.fatal:
        d.add(REJECT, "GNSS_UNUSABLE", f)
    for w in qg.warnings:
        d.add(DEGRADED, "GNSS_SUSPECT", w)

    # -- IMU ---------------------------------------------------------------- #
    qi = Q.check_imu(session)
    d.stats.update(qi.stats)
    for f in qi.fatal:
        d.add(REJECT, "IMU_UNUSABLE", f)
    for w in qi.warnings:
        d.add(DEGRADED, "IMU_SUSPECT", w)

    # -- actually drove somewhere ------------------------------------------- #
    qm = Q.check_motion(gps)
    d.stats.update(qm.stats)
    for f in qm.fatal:
        d.add(REJECT, "NO_MOTION", f)
    for w in qm.warnings:
        d.add(DEGRADED, "LITTLE_MOTION", w)

    # -- longitudinal accelerometer usefulness ------------------------------ #
    if session.has("accel") and gps is not None and len(gps) > 8:
        try:
            from .accel_source import longitudinal_acceleration
            from .vehicle_frame import estimate_vehicle_frame
            vf = estimate_vehicle_frame(session)
            t = np.arange(session["accel"]["seconds_elapsed"].iloc[0],
                          session["accel"]["seconds_elapsed"].iloc[-1], 0.02)
            a = longitudinal_acceleration(session, vf, t)
            d.stats["accel_tier"] = a.tier
            d.stats["accel_quality"] = None if not np.isfinite(a.quality) else round(a.quality, 3)
            if not a.usable:
                d.add(DEGRADED, "ACCEL_UNUSABLE",
                      "accelerometer carries no usable longitudinal signal; "
                      "speed will be GNSS-driven only", source=a.name)
            if vf.lateral_score < 0.25:
                d.add(DEGRADED, "WEAK_VEHICLE_FRAME",
                      "phone->vehicle lateral axis is weakly determined",
                      lateral_score=round(vf.lateral_score, 3))
        except Exception as e:                      # diagnosis must never crash the run
            d.add(DEGRADED, "ACCEL_CHECK_FAILED", f"could not assess accelerometer: {e}")

    # -- video vs sensors --------------------------------------------------- #
    if video is not None:
        d.stats["video_duration_s"] = round(video.duration, 2)
        d.stats["video_frames"] = video.nb_frames
        d.stats["video_fps"] = round(video.avg_fps, 3)
        if video.nb_frames < 60:
            d.add(REJECT, "VIDEO_TOO_SHORT", "fewer than 60 frames", frames=video.nb_frames)
        if len(video.pts) and video.nb_frames and abs(len(video.pts) - video.nb_frames) > 2:
            d.add(DEGRADED, "PTS_COUNT_MISMATCH",
                  "frame count disagrees with the number of PTS entries",
                  nb_frames=video.nb_frames, n_pts=len(video.pts))
        overlap = min(session.duration, video.duration)
        d.stats["sensor_duration_s"] = round(session.duration, 2)
        if overlap < 15:
            d.add(REJECT, "TOO_SHORT",
                  "less than 15 s of overlapping video and sensor data",
                  overlap_s=round(overlap, 1))
        if video.duration > session.duration + 2.0:
            d.add(DEGRADED, "VIDEO_OUTLASTS_SENSORS",
                  "video is longer than the sensor log", video_s=round(video.duration, 1),
                  sensors_s=round(session.duration, 1))
    return d


def diagnose_sync(sync, min_score: float = 0.25) -> Diagnosis:
    """Was the video/sensor alignment actually established, or merely assumed?"""
    d = Diagnosis()
    d.stats["t_video_start_s"] = round(sync.t_video_start, 4)
    d.stats["sync_source"] = sync.source
    d.stats["sync_score"] = None if sync.xcorr_peak is None else round(sync.xcorr_peak, 3)
    if sync.xcorr_shift is None:
        d.add(REJECT, "SYNC_UNVERIFIED",
              "optical-flow/gyro cross-correlation did not converge; the video-to-sensor "
              "offset is an unverified container guess and every label would be attached "
              "to the wrong frame", source=sync.source)
    elif sync.xcorr_peak is not None and sync.xcorr_peak < min_score:
        d.add(DEGRADED, "SYNC_WEAK", "cross-correlation agreed but weakly",
              score=round(sync.xcorr_peak, 3))
    return d


def diagnose_trajectory(traj, validation=None) -> Diagnosis:
    """Sanity of the solved trajectory: physics, numerics, coverage."""
    d = Diagnosis()
    dt = traj.t[1] - traj.t[0]
    v = traj.speed
    yr = np.rad2deg(traj.yaw_rate)
    a = np.gradient(v, dt)

    d.stats["speed_max_ms"] = round(float(v.max()), 2)
    d.stats["pos_sigma_median_m"] = round(float(np.median(traj.pos_std())), 2)
    d.stats["accel_p99_ms2"] = round(float(np.percentile(np.abs(a), 99)), 2)
    d.stats["yawrate_p99_degs"] = round(float(np.percentile(np.abs(yr), 99)), 1)

    if not np.isfinite(traj.x).all():
        d.add(REJECT, "TRAJECTORY_NAN", "non-finite values in the solved state")
    if v.min() < -0.5:
        d.add(REJECT, "NEGATIVE_SPEED", "speed went substantially negative",
              min_ms=round(float(v.min()), 2))
    if np.percentile(np.abs(a), 99) > 8.0:
        d.add(DEGRADED, "IMPLAUSIBLE_ACCEL", "99th-percentile longitudinal acceleration "
              "exceeds what a road car does", p99=round(float(np.percentile(np.abs(a), 99)), 2))
    if np.percentile(np.abs(yr), 99) > 90.0:
        d.add(DEGRADED, "IMPLAUSIBLE_YAWRATE", "99th-percentile yaw rate is implausible",
              p99=round(float(np.percentile(np.abs(yr), 99)), 1))
    if np.median(traj.pos_std()) > 8.0:
        d.add(DEGRADED, "HIGH_POSITION_UNCERTAINTY", "median position 1-sigma is large",
              sigma_m=round(float(np.median(traj.pos_std())), 2))

    if validation is not None:
        for k, s in validation.holdout.items():
            d.stats[f"holdout_{k}"] = s
        try:
            rms = float(validation.holdout["position (m)"].split("rms=")[1].split()[0])
            d.stats["holdout_pos_rms_m"] = rms
            if rms > 12.0:
                d.add(REJECT, "HOLDOUT_FAILED",
                      "hold-out position error is too large for ground truth", rms_m=rms)
            elif rms > 6.0:
                d.add(DEGRADED, "HOLDOUT_POOR", "hold-out position error is high", rms_m=rms)
        except Exception:
            pass
    return d


def diagnose_camera(cam) -> Diagnosis:
    """Is the camera calibration self-consistent and physically plausible?"""
    d = Diagnosis()
    d.stats["hfov_deg"] = round(cam.hfov_deg(), 1)
    d.stats["mount_yaw_deg"] = round(float(np.rad2deg(cam.yaw)), 2)
    d.stats["mount_pitch_deg"] = round(float(np.rad2deg(cam.pitch)), 2)
    d.stats["mount_roll_deg"] = round(float(np.rad2deg(cam.roll)), 2)
    d.stats["cam_height_m"] = round(cam.height_m, 3)
    d.stats["horizon_row_px"] = round(cam.horizon_v(), 1)

    if not (35 < cam.hfov_deg() < 100):
        d.add(DEGRADED, "ODD_FOV", "focal length implies an unusual field of view",
              hfov=round(cam.hfov_deg(), 1))
    if abs(np.rad2deg(cam.yaw)) > 20 or abs(np.rad2deg(cam.pitch)) > 20:
        d.add(DEGRADED, "EXTREME_MOUNT", "mount angles are large; check the cradle",
              yaw=round(float(np.rad2deg(cam.yaw)), 1),
              pitch=round(float(np.rad2deg(cam.pitch)), 1))
    hv = cam.horizon_v()
    if not np.isfinite(hv) or not (0 < hv < cam.height):
        d.add(REJECT, "HORIZON_OFF_IMAGE",
              "the road's vanishing line falls outside the image; the ground projection "
              "cannot be valid", horizon_row=None if not np.isfinite(hv) else round(hv, 1))
    if "extrinsics" not in cam.source:
        d.add(DEGRADED, "MOUNT_NOT_CALIBRATED",
              "mount angles were not estimated from the data; nominal values in use")
    return d


def merge(*diags) -> Diagnosis:
    out = Diagnosis()
    for x in diags:
        if x is None:
            continue
        out.findings += x.findings
        out.stats.update(x.stats)
        if _RANK[x.verdict] > _RANK[out.verdict]:
            out.verdict = x.verdict
    return out
