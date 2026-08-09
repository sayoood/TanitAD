"""Loader for Sensor Logger (android) recording folders.

Timebase facts established empirically on this data set (see docs/AUDIT.md):

*   ``Metadata.csv:recording epoch time`` is the UTC epoch in **milliseconds** at
    which logging started.  It is identical to the camera file stem
    (``Camera/<epoch_ms>.mp4``).
*   Every sensor CSV carries ``time`` = UTC epoch in **nanoseconds** and
    ``seconds_elapsed`` = ``(time - recording_epoch_ns) / 1e9`` exactly.
*   ``local_time`` is consistent with ``time``; it is redundant and is dropped.
*   ``Accelerometer.csv`` is **linear** acceleration (gravity already removed by
    the Android fusion).  ``TotalAcceleration.csv`` is the raw specific force and
    ``Gravity.csv`` the estimated gravity vector; Total ~= Accelerometer + Gravity.

``seconds_elapsed`` is the canonical session timebase used everywhere downstream.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Several logical streams have more than one possible file name.  iOS has no
# TotalAcceleration; its raw specific force arrives as AccelerometerUncalibrated.
_SENSOR_FILES = {
    "accel": ["Accelerometer.csv"],                 # linear / user acceleration
    "gyro": ["Gyroscope.csv"],                      # rad/s
    "gravity": ["Gravity.csv"],
    "total_accel": ["TotalAcceleration.csv",        # Android
                    "AccelerometerUncalibrated.csv"],  # iOS equivalent
    "orientation": ["Orientation.csv"],
    "magnetometer": ["Magnetometer.csv"],
    "gps": ["Location.csv"],
    "barometer": ["Barometer.csv"],
}

G0 = 9.80665

# Sensor Logger is not internally consistent about acceleration units.  Measured
# on real exports (median vector magnitude):
#
#   stream                       Android      iOS
#   Accelerometer.csv             0.553      1.011      <- iOS in g
#   TotalAcceleration.csv        10.006     absent
#   AccelerometerUncalibrated    10.006      0.994      <- iOS in g
#   Gravity.csv                   9.807      9.807      <- both m/s^2
#
# So a single iOS export mixes g and m/s^2, and reading its accelerometer as
# m/s^2 makes every acceleration 9.8x too small with nothing downstream to
# notice.  Units are therefore *detected from the data* rather than assumed:
# a stream that must have magnitude ~g is measured, and the scale follows.
_ACCEL_STREAMS = ("accel", "total_accel", "gravity")


@dataclass
class Session:
    """A single recording: sensors on a common ``seconds_elapsed`` timebase."""

    path: str
    epoch_ms: int
    device: str
    timezone_name: str
    platform: str = "?"
    sensors: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    video_path: str | None = None

    # -- convenience accessors -------------------------------------------------
    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.sensors[key]

    def get(self, key: str, default=None):
        return self.sensors.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.sensors and len(self.sensors[key]) > 0

    @property
    def t0_utc(self) -> datetime:
        return datetime.fromtimestamp(self.epoch_ms / 1000.0, tz=timezone.utc)

    def elapsed_to_utc(self, t_elapsed):
        return self.epoch_ms / 1000.0 + np.asarray(t_elapsed, dtype=float)

    @property
    def duration(self) -> float:
        ends = [df["seconds_elapsed"].iloc[-1] for df in self.sensors.values() if len(df)]
        return float(max(ends)) if ends else 0.0

    def summary(self) -> str:
        lines = [
            f"Session   : {os.path.basename(self.path)}",
            f"Device    : {self.device}   platform={self.platform}   tz={self.timezone_name}",
            f"Start UTC : {self.t0_utc.isoformat()}  (epoch_ms={self.epoch_ms})",
            f"Duration  : {self.duration:.2f} s",
            f"Video     : {os.path.basename(self.video_path) if self.video_path else '-'}",
        ]
        for name, df in self.sensors.items():
            if not len(df):
                continue
            t = df["seconds_elapsed"].to_numpy()
            rate = 1.0 / np.median(np.diff(t)) if len(t) > 2 else float("nan")
            lines.append(f"  {name:13s} n={len(df):6d}  {t[0]:7.2f}..{t[-1]:7.2f}s  {rate:7.2f} Hz")
        return "\n".join(lines)


def _read_metadata(path: str) -> dict:
    meta_path = os.path.join(path, "Metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata.csv not found in {path}")
    m = pd.read_csv(meta_path).iloc[0].to_dict()
    return {str(k).strip(): v for k, v in m.items()}


def _find_video(path: str) -> str | None:
    cam = os.path.join(path, "Camera")
    if not os.path.isdir(cam):
        return None
    vids = [f for f in os.listdir(cam) if f.lower().endswith((".mp4", ".mov"))]
    if not vids:
        return None
    # prefer the one whose stem is a plausible epoch-ms integer
    vids.sort(key=lambda f: (not re.fullmatch(r"\d{13}", os.path.splitext(f)[0]), f))
    return os.path.join(cam, vids[0])


# Streams that only one platform produces.  Used to infer the platform when the
# metadata does not say, and to stop the diagnosis complaining about a stream
# that the device was never going to write in the first place.
_ANDROID_ONLY = {"TotalAcceleration.csv", "Barometer.csv", "Magnetometer.csv",
                 "MagnetometerUncalibrated.csv"}
_IOS_ONLY = {"AccelerometerUncalibrated.csv", "Compass.csv"}


def detect_platform(path: str, meta: dict | None = None) -> str:
    """``'ios'`` / ``'android'`` / ``'unknown'``.

    ``Metadata.csv`` carries a ``platform`` field and that is used when present.
    It is not always present in older exports, so the file inventory is the
    fallback: the two platforms write partly disjoint sets of streams.
    """
    if meta is None:
        try:
            meta = _read_metadata(path)
        except Exception:
            meta = {}
    p = str(meta.get("platform", "")).strip().lower()
    if p in ("ios", "android"):
        return p
    dev = str(meta.get("device name", "")).lower()
    if any(k in dev for k in ("iphone", "ipad", "ipod")):
        return "ios"
    try:
        present = set(os.listdir(path))
    except OSError:
        return "unknown"
    a, i = len(present & _ANDROID_ONLY), len(present & _IOS_ONLY)
    if a > i:
        return "android"
    if i > a:
        return "ios"
    return "unknown"


def load_session(path: str, sensors: list[str] | None = None) -> Session:
    """Load a Sensor Logger export directory into a :class:`Session`."""
    meta = _read_metadata(path)
    epoch_ms = int(meta["recording epoch time"])
    platform = detect_platform(path, meta)
    want = sensors if sensors is not None else list(_SENSOR_FILES)

    data: dict[str, pd.DataFrame] = {}
    units: dict[str, str] = {}
    for key in want:
        candidates = _SENSOR_FILES.get(key)
        if candidates is None:
            continue
        fpath = next((os.path.join(path, c) for c in candidates
                      if os.path.exists(os.path.join(path, c))), None)
        if fpath is None:
            continue
        df = pd.read_csv(fpath, skipinitialspace=True)
        if not len(df):
            continue
        df = df.drop(columns=[c for c in ("local_time",) if c in df.columns])
        # Recompute seconds_elapsed from the integer epoch: the CSV column is a
        # float32-ish round-trip and drifts by ~1e-7 s; the integer ns is exact.
        if "time" in df.columns:
            df["seconds_elapsed"] = (df["time"].to_numpy(dtype="int64") - epoch_ms * 1_000_000) / 1e9
        df = df.drop_duplicates(subset=["time"]).sort_values("seconds_elapsed").reset_index(drop=True)
        data[key] = df
        units[key] = os.path.basename(fpath)

    _normalise_accel_units(data, units)
    _clean_location_sentinels(data)

    return Session(
        path=path,
        epoch_ms=epoch_ms,
        device=str(meta.get("device name", "?")),
        timezone_name=str(meta.get("recording timezone", "?")),
        platform=platform,
        sensors=data,
        units=units,
        video_path=_find_video(path),
    )


def _median_mag(df):
    c = [x for x in ("x", "y", "z") if x in df.columns]
    if len(c) != 3 or not len(df):
        return None
    return float(np.median(np.linalg.norm(df[c].to_numpy(dtype=float), axis=1)))


def _normalise_accel_units(data, units):
    """Put every acceleration stream in m/s^2 and make ``accel`` mean *linear*.

    Sensor Logger's units are not consistent across platforms, and on iOS not
    even within one export.  Measured on real files (raw, as written):

        stream                        Android            iOS
        Gravity                       9.783 m/s^2        9.799 m/s^2
        TotalAcceleration             9.958 m/s^2        absent
        AccelerometerUncalibrated     9.958 m/s^2        0.989 g     <- g!
        Accelerometer                 0.042 (linear)     0.138 (linear)

    iOS converts the *calibrated* streams to m/s^2 but leaves the *uncalibrated*
    raw stream in g, so a single export mixes both.  Guessing from a filename or
    a platform string would be fragile, so the unit is read off the data: any
    stream that contains gravity has a mean vector of one g, whatever the unit,
    and that fixes the scale.

    ``accel`` has no such anchor -- its mean is ~0 by construction -- so instead
    of trying to infer its unit it is **derived** as ``total_accel - gravity``
    whenever both are present.  That is unambiguous, identical in meaning on both
    platforms, and removes the failure mode where the raw and linear streams
    carry different units and the mismatch is invisible.
    """
    # 1. streams that contain gravity: their mean vector must be one g
    for key in ("gravity", "total_accel"):
        if key not in data:
            continue
        v = data[key][["x", "y", "z"]].to_numpy(dtype=float)
        dc = float(np.linalg.norm(v.mean(axis=0)))
        if 0.5 < dc < 2.0:
            k = G0
            note = f"[g -> m/s^2, x{G0:.5f}]"
        elif 5.0 < dc < 15.0:
            k, note = 1.0, "[m/s^2]"
        else:
            k, note = 1.0, f"[UNRECOGNISED |mean|={dc:.3f} - unit not established]"
        if k != 1.0:
            for c in ("x", "y", "z"):
                data[key][c] = data[key][c].to_numpy(dtype=float) * k
        units[key] = units.get(key, "") + " " + note

    # 2. linear acceleration: derive it rather than trust its unit
    if "total_accel" in data and "gravity" in data:
        raw, g = data["total_accel"], data["gravity"]
        t = raw["seconds_elapsed"].to_numpy(dtype=float)
        tg = g["seconds_elapsed"].to_numpy(dtype=float)
        lin = raw.copy()
        for c in ("x", "y", "z"):
            lin[c] = raw[c].to_numpy(dtype=float) - np.interp(t, tg, g[c].to_numpy(dtype=float))
        data["accel"] = lin
        units["accel"] = "derived: total_accel - gravity [m/s^2, linear]"
    elif "accel" in data:
        v = data["accel"][["x", "y", "z"]].to_numpy(dtype=float)
        units["accel"] = (units.get("accel", "") +
                          f" [linear, unit UNVERIFIED: no raw+gravity pair to derive from; "
                          f"std={np.std(np.linalg.norm(v, axis=1)):.3f}]")


def _clean_location_sentinels(data):
    """iOS CoreLocation reports -1 for "unavailable"; NaN says so honestly.

    Left as -1, ``speed`` becomes a negative velocity measurement and ``bearing``
    a heading of -1 deg -- both perfectly valid-looking numbers that a filter
    will dutifully fit.
    """
    g = data.get("gps")
    if g is None or not len(g):
        return
    for c in ("speed", "bearing", "speedAccuracy", "bearingAccuracy",
              "horizontalAccuracy", "verticalAccuracy"):
        if c in g.columns:
            v = g[c].to_numpy(dtype=float)
            g[c] = np.where(v < 0, np.nan, v)


def load_reassembled_session(path: str, epoch_ms: int | None = None) -> Session:
    """Load a session rebuilt from per-frame ``sensor_data.json`` windows.

    Expects ``imu.csv`` / ``gyro.csv`` / ``gps.csv`` with the original Sensor
    Logger columns.  Used to validate against the already-synchronised output.
    """
    mapping = {"accel": "imu.csv", "gyro": "gyro.csv", "gps": "gps.csv"}
    data = {}
    for key, fname in mapping.items():
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, skipinitialspace=True)
        df = df.drop(columns=[c for c in ("local_time",) if c in df.columns])
        df = df.drop_duplicates(subset=["time"]).sort_values("seconds_elapsed").reset_index(drop=True)
        data[key] = df
    if epoch_ms is None and data:
        any_df = next(iter(data.values()))
        epoch_ms = int(round((any_df["time"].iloc[0] / 1e6) - any_df["seconds_elapsed"].iloc[0] * 1000.0))
    return Session(path=path, epoch_ms=int(epoch_ms or 0), device="?", timezone_name="?", sensors=data)


def resample_uniform(df: pd.DataFrame, cols: list[str], t_grid: np.ndarray,
                     max_gap: float = 0.5) -> np.ndarray:
    """Linear-interpolate ``cols`` of ``df`` onto ``t_grid``.

    Samples further than ``max_gap`` from any source sample are returned as NaN
    so that data drop-outs stay visible instead of being silently bridged.
    """
    t = df["seconds_elapsed"].to_numpy(dtype=float)
    out = np.empty((len(t_grid), len(cols)), dtype=float)
    for j, c in enumerate(cols):
        out[:, j] = np.interp(t_grid, t, df[c].to_numpy(dtype=float))
    idx = np.searchsorted(t, t_grid).clip(1, len(t) - 1)
    gap = np.minimum(np.abs(t_grid - t[idx - 1]), np.abs(t[idx] - t_grid))
    out[gap > max_gap, :] = np.nan
    return out
