"""Data-quality gates.

These exist because the pipeline previously failed *silently*.  The sample
recording ``Liebst_ckelweg-2025-08-14`` contains 78 GPS rows that are byte-for-byte
identical -- a cached last-known-fix that the receiver never updated during the
entire 78 s drive.  Fed to a fusion filter this looks like "driving at a constant
2.716 m/s on bearing 106 forever" and yields a perfectly smooth, perfectly wrong
straight-line trajectory.  Nothing in the original code could tell that apart
from a good fix, so bad ground truth would flow straight into the training set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QCReport:
    ok: bool
    fatal: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        lines = [f"[QC {head}]"]
        for f in self.fatal:
            lines.append(f"  FATAL   {f}")
        for w in self.warnings:
            lines.append(f"  WARN    {w}")
        for k, v in self.stats.items():
            lines.append(f"  stat    {k} = {v}")
        return "\n".join(lines)


def check_gps(gps: pd.DataFrame, min_fixes: int = 10,
              max_frozen_fraction: float = 0.5,
              max_hacc: float = 25.0) -> QCReport:
    """Reject frozen / absent / hopelessly noisy GNSS."""
    rep = QCReport(ok=True)
    if gps is None or len(gps) < min_fixes:
        rep.ok = False
        rep.fatal.append(f"only {0 if gps is None else len(gps)} GPS fixes (need >= {min_fixes})")
        return rep

    cols = [c for c in ("latitude", "longitude", "speed", "bearing") if c in gps.columns]
    same = np.asarray((gps[cols].diff().abs().sum(axis=1) == 0).to_numpy(), dtype=bool).copy()
    same[0] = False
    frozen = float(same.mean())
    rep.stats["frozen_fraction"] = round(frozen, 3)
    rep.stats["n_fixes"] = len(gps)

    if frozen > max_frozen_fraction:
        rep.ok = False
        rep.fatal.append(
            f"GPS frozen: {frozen:.0%} of fixes are identical to the previous one "
            f"(receiver is replaying a cached fix, not tracking)")

    # A receiver that reports a *constant* accuracy is also reporting a stale fix.
    for c in ("horizontalAccuracy", "speed", "bearing"):
        if c in gps.columns and gps[c].nunique() == 1 and len(gps) > min_fixes:
            rep.warnings.append(f"'{c}' is constant across all {len(gps)} fixes")

    if "horizontalAccuracy" in gps.columns:
        hacc = gps["horizontalAccuracy"].to_numpy(dtype=float)
        rep.stats["hacc_median_m"] = round(float(np.median(hacc)), 2)
        if np.median(hacc) > max_hacc:
            rep.ok = False
            rep.fatal.append(f"median horizontal accuracy {np.median(hacc):.1f} m > {max_hacc} m")

    if "speed" in gps.columns:
        v = gps["speed"].to_numpy(dtype=float)
        fin = v[np.isfinite(v)]
        rep.stats["n_speed_missing"] = int((~np.isfinite(v)).sum())
        if len(fin):
            rep.stats["speed_range_ms"] = (round(float(fin.min()), 2), round(float(fin.max()), 2))
        else:
            rep.ok = False
            rep.fatal.append("every GPS fix has a missing speed")

    t = gps["seconds_elapsed"].to_numpy(dtype=float)
    if len(t) > 2:
        gaps = np.diff(t)
        rep.stats["max_gap_s"] = round(float(gaps.max()), 2)
        if gaps.max() > 10.0:
            rep.warnings.append(f"GPS outage of {gaps.max():.1f} s")
    return rep


def check_imu(session, expect_hz: float = 100.0) -> QCReport:
    """Sanity-check IMU presence, rate, gaps and gravity magnitude."""
    rep = QCReport(ok=True)
    for key in ("accel", "gyro"):
        if not session.has(key):
            rep.ok = False
            rep.fatal.append(f"missing {key}")
            return rep

    for key in ("accel", "gyro", "gravity"):
        if not session.has(key):
            continue
        t = session[key]["seconds_elapsed"].to_numpy(dtype=float)
        dt = np.diff(t)
        rate = 1.0 / np.median(dt)
        rep.stats[f"{key}_hz"] = round(float(rate), 2)
        if rate < 0.5 * expect_hz:
            rep.warnings.append(f"{key} sampled at {rate:.1f} Hz, expected ~{expect_hz:.0f} Hz")
        if dt.max() > 0.5:
            rep.warnings.append(f"{key} has a {dt.max():.2f} s gap")

    if session.has("gravity"):
        g = np.linalg.norm(session["gravity"][["x", "y", "z"]].to_numpy(dtype=float), axis=1)
        rep.stats["gravity_norm"] = round(float(np.median(g)), 3)
        if not (9.0 < np.median(g) < 10.5):
            rep.warnings.append(f"gravity magnitude {np.median(g):.2f} m/s^2 is out of range")

    if session.has("gyro"):
        w = session["gyro"][["x", "y", "z"]].to_numpy(dtype=float)
        if np.abs(w).max() > 15.0:
            rep.warnings.append("gyroscope saturating (>15 rad/s)")
        rep.stats["gyro_absmax"] = round(float(np.abs(w).max()), 3)
    return rep


def check_motion(gps: pd.DataFrame, min_distance_m: float = 50.0,
                 min_moving_fraction: float = 0.3) -> QCReport:
    """Reject sessions that never actually drove anywhere."""
    rep = QCReport(ok=True)
    if gps is None or len(gps) < 3 or "speed" not in gps.columns:
        rep.warnings.append("cannot assess motion without GPS speed")
        return rep
    t = gps["seconds_elapsed"].to_numpy(dtype=float)
    v = gps["speed"].to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(t)      # iOS writes -1 -> NaN for "unavailable"
    if ok.sum() < 3:
        rep.warnings.append("too few fixes with a usable speed to assess motion")
        return rep
    dist = float(np.trapezoid(v[ok], t[ok]))
    moving = float((v[ok] > 1.0).mean())
    rep.stats["gps_distance_m"] = round(dist, 1)
    rep.stats["moving_fraction"] = round(moving, 2)
    if dist < min_distance_m:
        rep.ok = False
        rep.fatal.append(f"only {dist:.0f} m travelled (< {min_distance_m} m)")
    if moving < min_moving_fraction:
        rep.warnings.append(f"vehicle moving for only {moving:.0%} of the session")
    return rep


def merge(*reports: QCReport) -> QCReport:
    out = QCReport(ok=all(r.ok for r in reports))
    for r in reports:
        out.fatal += r.fatal
        out.warnings += r.warnings
        out.stats.update(r.stats)
    return out
