"""Mount yaw and lateral offset from lane markings.

Why this exists
---------------
The focus of expansion gives yaw and pitch from optical flow, and its pitch is
excellent -- on the Pacific Coast Highway recording the FOE row and the lane
vanishing point agree to 0.7 px (0.02 deg).  Its *yaw*, though, is fitted over
the whole session, curves included, and a route with a net turning bias walks it
sideways.  Measured on that same recording the FOE put "straight ahead" at
u = 1087 px while the lane markings put it at u = 1120 px: a 0.98 deg error,
which is 0.70 m of lateral displacement at a 40 m look-ahead and is plainly
visible as a projected ribbon that drifts off the lane toward the horizon.

Lane markings fix both remaining degrees of freedom, and neither needs a target:

**Yaw** -- on a straight road the markings are parallel to the direction of
travel, so their vanishing point *is* "straight ahead".  This is measured as a
vanishing point in the image rather than by back-projecting to the road plane,
which matters: a vanishing point is invariant to how high a feature sits above
the road, so guardrails, kerbs and post lines are all valid constraints.  The
road-plane version of the same idea assumes z = 0 and is biased by exactly those
elevated features -- on the PCH recording it read 4.51 deg against the vanishing
point's 4.79 deg, and the vanishing point is the one that holds up across every
split of the data (4.66 .. 4.84 deg over halves, stricter straightness gates and
road-surface-only segments).

**Lateral offset** -- with yaw known, marking pixels back-project to metric
ground coordinates, and a histogram of their lateral position peaks at the lane
boundaries.  Averaged over a session the driver sits mid-lane, so the ego-lane
centre measured relative to the camera is the camera's offset from the vehicle
centreline, sign flipped: a camera mounted right of centre sees the centreline
to its left.  This is the parameter that ``ground_calib`` calls unobservable --
and it is, from *motion*, because a sideways camera shift moves every ground
point identically and cancels out of any feature-tracking residual.  It is not
unobservable from lane geometry, which supplies the missing external reference.

Both estimates are gated.  If the road is too curved, the markings too sparse or
the two halves of the session disagree, the estimator declines and the caller
keeps the FOE value rather than being handed a confident wrong number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import R_CV_NOMINAL, _rot
from .frames import FrameSource


@dataclass
class LaneCalibResult:
    yaw_deg: float | None                 # None when the gate rejected it
    lateral_offset_m: float | None
    pitch_check_px: float                 # |lane VP row - FOE row|, a pitch cross-check
    lane_width_m: float
    n_frames: int
    n_segments: int
    yaw_spread_deg: float                 # half-split disagreement
    notes: list[str] = field(default_factory=list)

    def __str__(self):
        y = "declined" if self.yaw_deg is None else f"{self.yaw_deg:+.2f} deg"
        l = "declined" if self.lateral_offset_m is None else f"{self.lateral_offset_m:+.2f} m"
        return (f"LaneCalib(yaw={y}, lateral={l}, lane_width={self.lane_width_m:.2f} m, "
                f"frames={self.n_frames}, segments={self.n_segments}, "
                f"yaw_spread={self.yaw_spread_deg:.2f} deg)")


def _ridge_mask(gray, r0, r1):
    """Bright-ridge mask for lane markings, ridge width scaled by perspective."""
    m = np.zeros(gray.shape, np.uint8)
    for y in range(r0, r1, 2):
        w = max(2, int(round(2 + 26 * (y - r0) / max(1, r1 - r0))))
        row = gray[y].astype(np.int16)
        c = 2 * row - np.roll(row, w) - np.roll(row, -w)
        c[:w] = 0
        c[-w:] = 0
        m[y, c > max(24, int(np.percentile(c, 99.0)))] = 255
    return m


def _ridge_points(gray, r0, r1, pct=99.2, thr_min=28):
    us, vs = [], []
    for y in range(r0, r1, 2):
        w = max(2, int(round(2 + 26 * (y - r0) / max(1, r1 - r0))))
        row = gray[y].astype(np.int16)
        c = 2 * row - np.roll(row, w) - np.roll(row, -w)
        c[:w] = 0
        c[-w:] = 0
        xs = np.nonzero(c > max(thr_min, int(np.percentile(c, pct))))[0]
        us.append(xs)
        vs.append(np.full(len(xs), y))
    if not us:
        return np.empty(0), np.empty(0)
    return np.concatenate(us).astype(float), np.concatenate(vs).astype(float)


def _fit_vp(lines):
    """Robust vanishing point: smallest singular vector, Cauchy-reweighted."""
    L = np.asarray(lines)
    w = np.ones(len(L))
    v = np.array([0.0, 0.0, 1.0])
    for _ in range(12):
        _, _, vt = np.linalg.svd(L * w[:, None])
        v = vt[-1]
        if abs(v[2]) < 1e-12:
            return None
        v = v / v[2]
        r = np.abs(L @ v)
        s = 1.4826 * np.median(r) + 1e-9
        w = 1.0 / np.sqrt(1.0 + (r / (2.5 * s)) ** 2)
    return v


def _straight_fast_times(session, vframe, max_yawrate_deg_s=1.0, min_speed=12.0,
                         min_gap_s=1.0, max_frames=80):
    gy = session["gyro"]
    tg = gy["seconds_elapsed"].to_numpy(float)
    W = np.stack([gy[c].to_numpy(float) for c in "xyz"], axis=1)
    wz = vframe.to_vehicle(W)[:, 2]
    dt = float(np.median(np.diff(tg)))
    k = max(1, int(round(2.0 / dt)))
    wz_s = np.convolve(wz, np.ones(k) / k, "same")
    if not session.has("gps"):
        return []
    gps = session["gps"]
    tv = gps["seconds_elapsed"].to_numpy(float)
    vv = gps["speed"].to_numpy(float)
    fin = np.isfinite(tv) & np.isfinite(vv)
    if fin.sum() < 5:
        return []
    ok = (np.abs(np.rad2deg(wz_s)) < max_yawrate_deg_s) & (np.interp(tg, tv[fin], vv[fin]) > min_speed)
    sel = []
    for t in tg[ok]:
        if not sel or t - sel[-1] > min_gap_s:
            sel.append(float(t))
    return sel[:max_frames]


def estimate(session, video, cam, vframe, t_video_start_s: float = 0.0,
             max_yaw_correction_deg: float = 4.0,
             lateral_min_speed: float = 6.0) -> LaneCalibResult:
    """Estimate mount yaw and lateral offset from lane markings.

    ``cam`` supplies the intrinsics, the current pitch and the camera height;
    only yaw and the lateral offset are estimated here.  ``video`` is a
    ``VideoInfo`` or a ``FrameFolderVideo`` -- the 08-11 session has no MP4.
    """
    import cv2

    notes: list[str] = []
    times = _straight_fast_times(session, vframe)
    if len(times) < 12:
        return LaneCalibResult(None, None, float("nan"), float("nan"), len(times), 0,
                               float("nan"), ["too few straight, fast frames"])

    src = FrameSource(video)
    fps = src.fps
    if not fps or not np.isfinite(fps) or fps <= 0:
        src.release()
        return LaneCalibResult(None, None, float("nan"), float("nan"), 0, 0,
                               float("nan"), ["unreadable video fps"])

    per_frame_lines: list[np.ndarray] = []
    frames_used = 0
    grays: list[tuple[float, np.ndarray]] = []
    for t in times:
        src.seek(int(round((t - t_video_start_s) * fps)))
        fr = src.read()
        if fr is None:
            continue
        H = fr.shape[0]
        r0, r1 = int(0.52 * H), int(0.88 * H)
        g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        grays.append((t, g))
        seg = cv2.HoughLinesP(_ridge_mask(g, r0, r1), 1, np.pi / 360,
                              threshold=40, minLineLength=55, maxLineGap=14)
        if seg is None:
            continue
        got = []
        for x1, y1, x2, y2 in seg[:, 0]:
            if y1 == y2:
                continue
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if not (18 < ang < 78):        # drop horizon clutter and vertical poles
                continue
            l = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
            got.append(l / np.linalg.norm(l[:2]))
        if got:
            per_frame_lines.append(np.array(got))
            frames_used += 1
    # The vanishing point needs fast, straight, highway-like geometry, but the
    # lateral offset does not -- and on an urban street the >12 m/s gate keeps
    # only a few fast stretches whose lane structure differs from the rest.  On
    # Rose Ave that read -0.325 m where every window of a lower-speed sample
    # reads -0.20 m, the same value the other recording from this phone gives.
    # Collect a second, slower set for the lateral step only.
    if lateral_min_speed < 12.0:
        extra = _straight_fast_times(session, vframe, max_yawrate_deg_s=1.0,
                                     min_speed=lateral_min_speed, min_gap_s=1.0,
                                     max_frames=80)
        have = {round(t, 2) for t, _ in grays}
        for t in extra:
            if round(t, 2) in have:
                continue
            src.seek(int(round((t - t_video_start_s) * fps)))
            fr = src.read()
            if fr is None:
                continue
            grays.append((t, cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0)))
    src.release()

    n_seg = int(sum(len(a) for a in per_frame_lines))
    if frames_used < 10 or n_seg < 200:
        return LaneCalibResult(None, None, float("nan"), float("nan"), frames_used, n_seg,
                               float("nan"), ["too few lane-marking segments"])

    L = np.concatenate(per_frame_lines)
    v = _fit_vp(L)
    if v is None:
        return LaneCalibResult(None, None, float("nan"), float("nan"), frames_used, n_seg,
                               float("nan"), ["vanishing point did not converge"])

    # half-split stability: a curved route or a drifting mount shows up here
    h = len(per_frame_lines) // 2
    yaws = []
    for part in (per_frame_lines[:h], per_frame_lines[h:]):
        if len(part) < 5:
            continue
        vp = _fit_vp(np.concatenate(part))
        if vp is not None:
            yaws.append(np.degrees(np.arctan2(vp[0] - cam.cx, cam.fx)))
    spread = float(max(yaws) - min(yaws)) if len(yaws) == 2 else float("nan")

    yaw_deg = float(np.degrees(np.arctan2(v[0] - cam.cx, cam.fx)))
    pitch_check = float(abs(v[1] - (cam.cy + cam.fy * np.tan(cam.pitch))))

    yaw_ok = True
    if np.isfinite(spread) and spread > 0.6:
        notes.append(f"yaw declined: halves disagree by {spread:.2f} deg")
        yaw_ok = False
    if abs(yaw_deg - np.degrees(cam.yaw)) > max_yaw_correction_deg:
        notes.append(f"yaw declined: {yaw_deg:+.2f} deg is "
                     f"{abs(yaw_deg - np.degrees(cam.yaw)):.1f} deg from the FOE, not credible")
        yaw_ok = False

    # ---- lateral offset, at the yaw we just settled on ---------------------- #
    yaw_use = np.deg2rad(yaw_deg) if yaw_ok else cam.yaw
    R = _rot(yaw_use, cam.pitch, cam.roll) @ R_CV_NOMINAL
    centres, widths = [], []
    for t, g in grays:
        H = g.shape[0]
        u, vv_ = _ridge_points(g, int(0.55 * H), int(0.86 * H))
        if len(u) < 150:
            continue
        d = np.stack([(u - cam.cx) / cam.fx, (vv_ - cam.cy) / cam.fy, np.ones(len(u))], 1) @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -cam.height_m / d[:, 2]
        m = np.isfinite(s) & (s > 0)
        X, Y = d[:, 0] * s, d[:, 1] * s
        m &= (X > 7) & (X < 40) & (np.abs(Y) < 7.5)
        if m.sum() < 150:
            continue
        hist, edges = np.histogram(Y[m], bins=np.arange(-7.5, 7.55, 0.10))
        ctr = 0.5 * (edges[1:] + edges[:-1])
        hs = np.convolve(hist, np.ones(3) / 3, "same")
        pk = [i for i in range(1, len(hs) - 1)
              if hs[i] > hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] > 0.25 * hs.max()]
        if len(pk) < 2:
            continue
        py = ctr[pk]
        left, right = py[py > 0.3], py[py < -0.3]
        if not len(left) or not len(right):
            continue
        yl, yr = left.min(), right.max()
        w = yl - yr
        if not (2.6 < w < 4.6):
            continue
        centres.append(0.5 * (yl + yr))
        widths.append(w)

    lat = None
    lane_w = float(np.median(widths)) if widths else float("nan")
    if len(centres) >= 10:
        c = np.array(centres)
        iqr = float(np.percentile(c, 75) - np.percentile(c, 25))
        if iqr < 0.5:
            lat = float(-np.median(c))
        else:
            notes.append(f"lateral declined: scatter too wide (IQR {iqr:.2f} m)")
    else:
        notes.append(f"lateral declined: only {len(centres)} usable frames")

    return LaneCalibResult(yaw_deg if yaw_ok else None, lat, pitch_check, lane_w,
                           frames_used, n_seg, spread, notes)


def apply(cam, res: LaneCalibResult, set_yaw: bool = True, set_lateral: bool = True):
    """Fold the accepted estimates into ``cam``.  Returns a provenance dict."""
    prov = {}
    if set_yaw and res.yaw_deg is not None:
        prov["yaw_deg"] = {
            "value": round(res.yaw_deg, 3), "unit": "deg",
            "source": (f"lane vanishing point, {res.n_segments} segments over "
                       f"{res.n_frames} straight frames (half-split spread "
                       f"{res.yaw_spread_deg:.2f} deg)"),
        }
        cam.yaw = np.deg2rad(res.yaw_deg)
    if set_lateral and res.lateral_offset_m is not None:
        prov["lateral_offset_m"] = {
            "value": round(res.lateral_offset_m, 3), "unit": "m",
            "source": (f"ego-lane centre vs camera over {res.n_frames} frames "
                       f"(lane width {res.lane_width_m:.2f} m)"),
            "note": "+left. Assumes lane-centred driving on average over the session.",
        }
        # cam.lateral_m is set by the caller from the final operator/measured
        # value; do not zero it here.
    return prov
