"""Time-varying camera attitude for rendering: a yaw (and optionally horizon) track.

⛔ WHY A CONSTANT YAW CANNOT WORK ON AN EIS RECORDING. The 2026-08-08 phone had
electronic stabilisation ON (operator-confirmed, BEV_SEMANTIC_CALIB Part 6), and
EIS moves the crop window during the clip. A moved crop is, for drawing purposes,
a moved camera. MEASURED 2026-09-24 by fitting both lane lines as whole Hough
lines on gated straight frames: the lane's vanishing point walks from column
~751 px (t 0-10 s) to ~830 px (t 40-50 s) — about 3 deg of equivalent yaw, lag-1
autocorrelation +0.59 (a slow drift, not frame noise), correlated with TIME
(r -0.41) and NOT with steering (r +0.09). The car was not turning; the image
was.

That is why seven successive constant-yaw renders each fixed one stretch of the
clip and broke another: v6 (-7.75) matched the start, v8 (-6.40) sat between,
and the 30-50 s stretch the PI kept sending wants about -4.9.

WHAT THIS MODULE IS. A track of ``(t_session_s, yaw_deg[, horizon_row])`` measured
OUTSIDE the pipeline, interpolated per rendered frame. It changes only what the
overlay is DRAWN with. The trajectory is in the vehicle frame and does not depend
on the camera, so nothing in ``trajectory.jsonl`` moves.

⚠️ WHAT IT ASSUMES, AND WHAT THAT HIDES. A track fitted to the lane's vanishing
point assumes the car heads ALONG the lane on average over the smoothing window.
Real heading changes shorter than that window survive; anything slower is
absorbed as "camera". Choose the window longer than a lane change (> 5 s) and say
which window was used. The track is a MEASUREMENT of the effective camera
pointing, not a mount calibration, and must never be quoted as one.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np

__all__ = ["AttitudeTrack", "load_attitude_track", "camera_for_time"]


@dataclass(frozen=True)
class AttitudeTrack:
    t: np.ndarray                 # session seconds, strictly increasing
    yaw_deg: np.ndarray
    horizon_row: np.ndarray | None
    source: str = ""

    def yaw_at(self, t: float) -> float:
        return float(np.interp(t, self.t, self.yaw_deg))

    def horizon_at(self, t: float) -> float | None:
        if self.horizon_row is None:
            return None
        return float(np.interp(t, self.t, self.horizon_row))


def load_attitude_track(path: str) -> AttitudeTrack:
    """Read ``{"t_session_s": [...], "yaw_deg": [...], "horizon_row": [...]?}``."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return attitude_track_from(d)


def attitude_track_from(d: dict) -> AttitudeTrack:
    t = np.asarray(d["t_session_s"], dtype=float)
    yaw = np.asarray(d["yaw_deg"], dtype=float)
    hz = d.get("horizon_row")
    hz = None if hz is None else np.asarray(hz, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("an attitude track needs at least two samples")
    if yaw.shape != t.shape or (hz is not None and hz.shape != t.shape):
        raise ValueError("t_session_s, yaw_deg and horizon_row must be the same length")
    if not (np.isfinite(t).all() and np.isfinite(yaw).all()
            and (hz is None or np.isfinite(hz).all())):
        raise ValueError("attitude track contains non-finite values")
    o = np.argsort(t, kind="stable")
    t, yaw = t[o], yaw[o]
    hz = None if hz is None else hz[o]
    if np.any(np.diff(t) <= 0):
        raise ValueError("attitude track has repeated timestamps")
    return AttitudeTrack(t=t, yaw_deg=yaw, horizon_row=hz, source=str(d.get("source", "")))


def camera_for_time(cam, track: AttitudeTrack | None, t: float):
    """A COPY of ``cam`` with the track's yaw (and horizon, if given) at time ``t``.

    The horizon is converted to a pitch with exactly the formula the
    ``--horizon-row`` override uses (``pitch = atan2(row - cy, fy)``), so a track
    holding a constant row renders identically to that override. ``cam`` itself is
    never mutated — the pipeline writes it to the calibration record afterwards.
    """
    if track is None:
        return cam
    c = copy.copy(cam)
    c.yaw = float(np.deg2rad(track.yaw_at(t)))
    row = track.horizon_at(t)
    if row is not None:
        c.pitch = float(np.arctan2(row - c.cy, c.fy))
    return c
