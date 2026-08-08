"""Separate focal length from camera height using the road plane and GNSS.

The problem
-----------
Back-projecting a pixel onto the road plane gives

    lateral       y = (u - cx) * h / (v - v_horizon)      -- no f
    longitudinal  x = f * h / (v - v_horizon)             -- f and h only as f*h

so **the ground plane can never separate f from h on its own**: it sees ``h`` in
the lateral direction and the product ``f*h`` in the longitudinal one.  Every
self-consistency check built purely on ground back-projection is therefore blind
to a focal error -- back-project with the wrong f and re-project with the same
wrong f and the pixels come back exactly where they started.

That blindness is not hypothetical.  The Pacific Coast Highway recording shipped
with f = 1901 px (53.6 deg HFOV) against Rose Ave's 1427 px (67.9 deg) from the
*same phone and camera*, and every check passed: lane width looked plausible
(it only tests h), lane markings re-projected onto the paint (self-consistent),
centring was correct to 6 cm.  Only the depth scale was wrong -- by 33% -- which
renders the trajectory's 40 m curvature at the visual position of 30 m.

The closure
-----------
Two measurements that constrain *different* directions:

1.  ``f*h`` from tracked road features against GNSS distance.  A static ground
    point obeys ``v - v_horizon = f*h/x``, so as the vehicle advances by ``s``,

        1 / (v - v_horizon) = (x0 - s) / (f*h)

    which is linear in ``s`` with slope ``-1/(f*h)``.  Fitting that per track
    uses a long baseline, only image *rows*, and no flow magnitude -- so it is
    immune both to yaw/lateral error and to the static windscreen artefacts
    (permits, reflections) that defeat flow-based focal estimators on this
    footage.

2.  ``h`` from lane width, which is independent of f.  This needs the true lane
    width as external metric knowledge -- an honest operator input, not
    something the recording can supply.

Then ``f = (f*h) / h``, and the result is checked against the device's nominal
field of view, which no amount of self-consistency can substitute for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import FrameSource

# Nominal horizontal FOV bands for known capture devices, used only as a sanity
# bound.  A phone main camera in video mode is never as narrow as 55 deg, and a
# non-fisheye rear camera is never as wide as 90 deg.
_FOV_BAND_DEG = (55.0, 85.0)


@dataclass
class ScaleResult:
    fh: float                       # f * h, px*m
    fh_n: int
    fh_spread: float                # IQR / median
    focal_px: float | None
    height_m: float | None
    lane_width_used_m: float | None
    notes: list[str] = field(default_factory=list)

    def __str__(self):
        f = "declined" if self.focal_px is None else f"{self.focal_px:.0f} px"
        h = "declined" if self.height_m is None else f"{self.height_m:.3f} m"
        hf = ("" if self.focal_px is None
              else f", hfov {np.degrees(2 * np.arctan(960.0 / self.focal_px)):.1f} deg")
        return (f"ScaleCalib(f*h={self.fh:.0f} from {self.fh_n} tracks, "
                f"spread {self.fh_spread * 100:.0f}%, focal={f}{hf}, height={h})")


def estimate_fh(video, frames, speeds, times, v_horizon: float,
                width: int, height: int, max_starts: int = 110,
                life: int = 26, stride: int = 2, min_speed: float = 8.0):
    """Measure ``f*h`` from tracked road features.  Returns (values, note)."""
    import cv2

    frames = np.asarray(frames)
    speeds = np.asarray(speeds, dtype=float)
    times = np.asarray(times, dtype=float)
    order = np.argsort(frames)
    frames, speeds, times = frames[order], speeds[order], times[order]
    if len(frames) < 40:
        return np.array([]), "too few trajectory frames"
    s_cum = np.concatenate([[0.0], np.cumsum(0.5 * (speeds[1:] + speeds[:-1]) * np.diff(times))])
    s_of = {int(f): float(s) for f, s in zip(frames, s_cum)}

    src = FrameSource(video)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    # road surface, central columns: limits roll/pitch cross-coupling, and keeps
    # the sample on the plane rather than on kerbs and parked cars
    mask = np.zeros((height, width), np.uint8)
    mask[int(v_horizon + 0.020 * height):int(0.80 * height),
         int(0.30 * width):int(0.70 * width)] = 255

    cand = [int(f) for f, v in zip(frames, speeds) if v > min_speed]
    if len(cand) < 40:
        src.release()
        return np.array([]), "too little time above the speed floor"

    out = []
    for f0 in cand[::max(1, len(cand) // max_starts)]:
        src.seek(f0)
        im = src.read()
        if im is None:
            continue
        pg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        p = cv2.goodFeaturesToTrack(pg, 120, 0.01, 10, mask=mask, blockSize=7)
        if p is None or len(p) < 12:
            continue
        tracks = {i: [(f0, float(p[i, 0, 1]))] for i in range(len(p))}
        alive = np.ones(len(p), bool)
        prev = p
        for k in range(1, life):
            for _ in range(stride - 1):
                src.read()
            im = src.read()
            if im is None:
                break
            fk = f0 + k * stride
            if fk not in s_of:
                break
            g1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, g1, prev, None, **lk)
            bk, st2, _ = cv2.calcOpticalFlowPyrLK(g1, pg, nxt, None, **lk)
            good = ((st.ravel() == 1) & (st2.ravel() == 1) &
                    (np.linalg.norm(prev.reshape(-1, 2) - bk.reshape(-1, 2), axis=1) < 0.6))
            for i in range(len(p)):
                if alive[i] and good[i]:
                    tracks[i].append((fk, float(nxt[i, 0, 1])))
                elif alive[i]:
                    alive[i] = False
            prev, pg = nxt, g1
            if alive.sum() < 5:
                break
        for tr in tracks.values():
            if len(tr) < 10:
                continue
            fr = np.array([a for a, _ in tr])
            dv = np.array([b for _, b in tr]) - v_horizon
            ok2 = dv > 12.0                      # stay away from the singular horizon
            if ok2.sum() < 8:
                continue
            ss = np.array([s_of[int(a)] for a in fr])[ok2]
            ss = ss - ss[0]
            y = 1.0 / dv[ok2]
            if np.ptp(ss) < 6.0:
                continue
            b, a = np.polyfit(ss, y, 1)
            if b >= 0:                            # a static point must approach
                continue
            if np.std(y - (a + b * ss)) / max(np.std(y), 1e-9) > 0.12:
                continue                          # not a rigid point on the plane
            out.append(-1.0 / b)
    src.release()
    return np.array(out), ""


def solve(fh_vals, lane_width_measured_m: float | None, height_ref_m: float,
          lane_width_true_m: float | None, cx: float,
          fov_band=_FOV_BAND_DEG) -> ScaleResult:
    """Combine ``f*h`` with a lane-width-derived height to separate f and h."""
    notes: list[str] = []
    if len(fh_vals) < 40:
        return ScaleResult(float("nan"), len(fh_vals), float("nan"), None, None, None,
                           [f"only {len(fh_vals)} usable tracks"])
    med = float(np.median(fh_vals))
    iqr = float(np.percentile(fh_vals, 75) - np.percentile(fh_vals, 25))
    spread = iqr / max(med, 1e-9)
    res = ScaleResult(med, len(fh_vals), spread, None, None, None, notes)
    if spread > 0.20:
        notes.append(f"f*h declined: track spread {spread * 100:.0f}% is too wide")
        return res
    if not lane_width_measured_m or not lane_width_true_m:
        notes.append("no lane width available: f and h cannot be separated")
        return res

    h = height_ref_m * lane_width_true_m / lane_width_measured_m
    f = med / h
    hfov = float(np.degrees(2 * np.arctan(cx / f)))
    if not (fov_band[0] <= hfov <= fov_band[1]):
        notes.append(f"declined: implied HFOV {hfov:.1f} deg is outside "
                     f"{fov_band[0]:.0f}-{fov_band[1]:.0f} deg for a phone camera")
        return res
    if not (0.9 <= h <= 1.8):
        notes.append(f"declined: implied height {h:.2f} m is not a windscreen mount")
        return res
    res.focal_px, res.height_m, res.lane_width_used_m = f, h, lane_width_true_m
    return res


def check_focal_plausible(focal_px: float, cx: float, fov_band=_FOV_BAND_DEG):
    """Cheap non-circular gate: is this focal physically possible for a phone?

    Every other check in the pipeline compares the focal against something that
    was derived using it.  This one does not, which is why it is worth having:
    it alone would have rejected the 53.6 deg HFOV that shipped.
    """
    hfov = float(np.degrees(2 * np.arctan(cx / focal_px)))
    ok = fov_band[0] <= hfov <= fov_band[1]
    return ok, hfov
