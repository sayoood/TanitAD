"""``f*h`` AND the horizon row from ground flow, measured against the odometer.

WHY THIS EXISTS ALONGSIDE ``scale_calib``
-----------------------------------------
``scale_calib.estimate_fh`` measures the same product, and on
``2026-08-08_14-19-54-android`` it DECLINED: ``f*h = 1608.1`` with a 62 % spread over
1576 tracks. The reason is structural, not statistical. It takes ``v_horizon`` as an
**input** -- it is passed ``cam.horizon_v()`` -- and the range of a road point is
``x = f*h / (v - v_horizon)``. Feed it a horizon that is 50 px wrong and every track
returns a biased ``f*h``, with the bias varying by how far down the frame the track
sat. That is exactly a 62 % spread, and no amount of averaging removes it.

This module solves for the horizon **jointly**, so it cannot be poisoned that way.

THE MODEL
---------
With ``q = v - v_h``, ``A = f*h`` and a metric odometer step ``D``, a point on the road
plane obeys ``q' = q A / (A - D q)``. That inverts per point, in closed form::

    A = D q q' / (q' - q)

Every tracked point is therefore one estimate of ``A`` **given** a horizon. The horizon
is then fixed by CONSISTENCY rather than by fitting a curve: at the true ``v_h`` the
per-point ``A`` values show no trend against image row; at a wrong one they do. Scan
``v_h``, take the zero-trend crossing.

⚠️ **The row band stops at 0.77 of image height, and that is measured.** Below ~row 830
on this rig the tracked flow is ~0 while the plane model predicts 59-135 px. Re-tracking
with a 51 px window and 7 pyramid levels DOUBLES the yield there (20 % -> 41 %) and the
measured motion stays at -0.2 px: it is the car's own bodywork, not a tracker failure.

⚠️ **What this does NOT give you is the height.** The ground plane sees ``h`` laterally
and ``f*h`` longitudinally, so motion can never separate them -- MEASURED here, not
assumed: forcing ``h`` from 0.90 to 1.60 m moved the fitted ``f*h`` by 0.4 % and the cost
by 0.2 %. Splitting ``f*h`` needs a lateral metre-stick (the lane width), which is what
``scale_calib.solve`` already does with this module's ``fh``.

⚠️ **Video stabilisation crops, and the crop lands in ``f``.** See
``camera.nominal_camera``: a quoted field of view describes the uncropped lens, so with
EIS on the recorded focal is larger. On this recording the lens is 26 mm-equivalent
(HFOV 67.3 deg = 1442 px at 1920 wide) and the focal in force was 1713 px -- a 1.19x
crop, confirmed by the operator. Measuring ``f*h`` here and dividing by a height from
the paint is what caught it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: image-height fractions bounding the road band used for tracking.
ROAD_BAND = (0.55, 0.77)
_LK = dict(winSize=(21, 21), maxLevel=4)


@dataclass
class FlowCalibResult:
    fh: float = float("nan")                 # px * m
    fh_ci: tuple = (float("nan"), float("nan"))
    horizon_row: float = float("nan")        # px
    horizon_ci: tuple = (float("nan"), float("nan"))
    n_points: int = 0
    n_pairs: int = 0
    rel_iqr: float = float("nan")
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return np.isfinite(self.fh) and np.isfinite(self.horizon_row) and self.n_points > 0

    def __str__(self) -> str:
        if not self.ok:
            return "f*h not measured from ground flow: " + "; ".join(self.notes)
        return (f"f*h = {self.fh:.1f} px*m  95% CI [{self.fh_ci[0]:.0f}, {self.fh_ci[1]:.0f}]\n"
                f"  horizon row = {self.horizon_row:.1f} px  "
                f"95% CI [{self.horizon_ci[0]:.1f}, {self.horizon_ci[1]:.1f}]\n"
                f"  {self.n_points} tracked points over {self.n_pairs} frame pairs, "
                f"relative IQR {self.rel_iqr:.3f}")


def _road_mask(gray, ridge_points=None):
    """Where to seek trackable points: on the road markings, nowhere else.

    A plain corner detector does not find the road, it AVOIDS it -- corners live where
    texture is and asphalt is smooth. MEASURED: unmasked, 6807 forward-backward-checked
    tracks gave 2.3 % inliers to any road-plane model at a 400 px median reprojection,
    because the features were guardrail, rock face, other vehicles and bonnet.
    """
    import cv2
    h, w = gray.shape[:2]
    r0, r1 = int(ROAD_BAND[0] * h), int(ROAD_BAND[1] * h)
    m = np.zeros((h, w), np.uint8)
    if ridge_points is not None and len(ridge_points[0]):
        u, v = ridge_points
        keep = (v >= r0) & (v < r1)
        if keep.sum() > 50:
            m[np.clip(v[keep], 0, h - 1).astype(int),
              np.clip(u[keep], 0, w - 1).astype(int)] = 255
            return cv2.dilate(m, np.ones((11, 11), np.uint8))
    m[r0:r1, :] = 255
    return m


def _track_pair(i0, i1, mask, max_pts=700, fb_tol=0.6):
    """Forward-backward-checked LK correspondences.

    The round-trip check is not optional on road texture: a lane line is locally
    one-dimensional, so LK slides along it for free and returns a confident, wrong
    correspondence.
    """
    import cv2
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
    p0 = cv2.goodFeaturesToTrack(i0, maxCorners=max_pts, qualityLevel=0.005,
                                 minDistance=6, mask=mask, blockSize=7)
    if p0 is None or len(p0) < 20:
        return None
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(i0, i1, p0, None, criteria=crit, **_LK)
    pb, s0, _ = cv2.calcOpticalFlowPyrLK(i1, i0, p1, None, criteria=crit, **_LK)
    ok = (s1.ravel() == 1) & (s0.ravel() == 1)
    ok &= np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < fb_tol
    if ok.sum() < 25:
        return None
    return p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]


def _solve(v, dv, D, grid, rng):
    """(horizon, f*h) from per-point votes, by the zero-trend criterion."""
    def at(vh):
        q = v - vh
        qp = q + dv
        m = (q > 25) & (dv > 0.5) & (qp > q)
        if m.sum() < 120:
            return None
        a = D[m] * q[m] * qp[m] / (qp[m] - q[m])
        qq = q[m]
        med = np.median(a)
        k = (a > 0.3 * med) & (a < 3.0 * med)
        if k.sum() < 80:
            return None
        aa, qq = a[k], qq[k]
        i, j = rng.integers(0, len(aa), 4000), rng.integers(0, len(aa), 4000)
        s = np.abs(qq[i] - qq[j]) > 20
        if s.sum() < 100:
            return None
        slope = float(np.median((aa[i][s] - aa[j][s]) / (qq[i][s] - qq[j][s])))
        iqr = float(np.percentile(aa, 75) - np.percentile(aa, 25))
        return slope, float(np.median(aa)), int(k.sum()), iqr / max(med, 1e-9)

    sl, vs = [], []
    for vh in grid:
        r = at(float(vh))
        if r:
            sl.append(r[0]); vs.append(float(vh))
    if len(sl) < 4:
        return None
    sl, vs = np.asarray(sl), np.asarray(vs)
    if not (sl.min() < 0 < sl.max()):
        return None                      # the zero crossing is not bracketed
    vh = float(np.interp(0.0, sl, vs))
    r = at(vh)
    return None if r is None else (vh, r[1], r[2], r[3])


def estimate(video, traj, sync, cam, *, max_pairs: int = 200, min_speed: float = 8.0,
             n_boot: int = 120, seed: int = 0, ridge_fn=None,
             horizon_grid=None) -> FlowCalibResult:
    """Measure ``f*h`` and the horizon row from consecutive-frame ground flow.

    ``ridge_fn(gray) -> (u, v)`` optionally supplies marking pixels to mask on; without
    it the whole road band is searched, which works but yields fewer usable points.
    """
    import cv2
    from trajlib.scale_calib import FrameSource

    res = FlowCalibResult()
    grid = np.arange(380.0, 581.0, 2.0) if horizon_grid is None else np.asarray(horizon_grid)
    try:
        ft = sync.frame_times(video.pts)
    except Exception as e:                                  # pragma: no cover
        res.notes.append(f"no frame times: {e}")
        return res
    if len(ft) < 60:
        res.notes.append("too few frames")
        return res
    spd = np.interp(ft, traj.t, traj.speed)
    cand = [i for i in range(len(ft) - 1) if spd[i] > min_speed]
    if len(cand) < 30:
        res.notes.append("too little time above the speed floor")
        return res
    step = max(1, len(cand) // max_pairs)
    cand = cand[::step]

    src = FrameSource(video)
    V, DV, D, G = [], [], [], []
    try:
        for gi, i0 in enumerate(cand):
            src.seek(i0)
            a = src.read()
            b = src.read()
            if a is None or b is None:
                continue
            g0 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            # METRIC, and it does not scale with the calibration -- the whole ruler
            d = float(np.interp(ft[i0 + 1], traj.t, traj.s)
                      - np.interp(ft[i0], traj.t, traj.s)) if hasattr(traj, "s") \
                else float(spd[i0] * (ft[i0 + 1] - ft[i0]))
            if not (0.15 < d < 3.0):
                continue
            rp = ridge_fn(g0) if ridge_fn else None
            got = _track_pair(g0, g1, _road_mask(g0, rp))
            if got is None:
                continue
            uv0, uv1 = got
            V.append(uv0[:, 1]); DV.append(uv1[:, 1] - uv0[:, 1])
            D.append(np.full(len(uv0), d)); G.append(np.full(len(uv0), gi))
    finally:
        try:
            src.release()
        except Exception:
            pass

    if not V:
        res.notes.append("no frame pair produced usable tracks")
        return res
    V = np.concatenate(V); DV = np.concatenate(DV)
    D = np.concatenate(D); G = np.concatenate(G)
    res.n_points, res.n_pairs = int(len(V)), int(len(np.unique(G)))
    rng = np.random.default_rng(seed)
    base = _solve(V, DV, D, grid, rng)
    if base is None:
        res.notes.append("the zero-trend horizon is not bracketed in "
                         f"{grid[0]:.0f}-{grid[-1]:.0f} px")
        return res
    res.horizon_row, res.fh, _, res.rel_iqr = base

    groups = np.unique(G)
    if len(groups) >= 6 and n_boot:
        boot = []
        for _ in range(n_boot):
            pick = rng.choice(len(groups), len(groups), replace=True)
            sel = np.concatenate([np.flatnonzero(G == groups[p]) for p in pick])
            got = _solve(V[sel], DV[sel], D[sel], grid, rng)
            if got:
                boot.append((got[0], got[1]))
        if len(boot) >= 20:
            bt = np.asarray(boot)
            res.horizon_ci = (float(np.percentile(bt[:, 0], 2.5)),
                              float(np.percentile(bt[:, 0], 97.5)))
            res.fh_ci = (float(np.percentile(bt[:, 1], 2.5)),
                         float(np.percentile(bt[:, 1], 97.5)))
        else:
            res.notes.append("bootstrap did not converge; intervals omitted")
    # ⚠️ THE LONGITUDINAL AND LATERAL FAMILIES DISAGREE ON THIS RIG, AND THE OPERATOR
    # MUST SEE IT RATHER THAN INHERIT WHICHEVER RAN LAST. MEASURED on
    # 2026-08-08_14-19-54-android: every flow-based estimate puts the horizon LOW
    # (431, 437, 448 px from three variants) and every paint-based one puts it HIGH
    # (465 adopted, ~485 from lane-width range-consistency) -- ~40 px apart, which is
    # ~12% of f*h. The paint-based set is the one directly validated (the projected
    # lane lands on the markings within a few cm over 8-20 m, and the lane measures
    # 3.55 m against a 3.50 m standard); this module's own synthetic self-test passes,
    # so the disagreement is a property of the real road -- most likely that the
    # surface over 7-17 m is not the single plane both models assume.
    if cam is not None:
        try:
            cur = float(cam.horizon_v())
            res.notes.append(f"horizon from flow {res.horizon_row:.1f} px vs the "
                             f"camera's current {cur:.1f} px "
                             f"({res.horizon_row - cur:+.1f} px)")
            if abs(res.horizon_row - cur) > 25.0:
                res.notes.append(
                    "⚠️ that is a LARGE disagreement. Flow and paint measure different "
                    "things when the road is not planar; do not adopt either silently. "
                    "Validate against the markings (project the lane edges and measure "
                    "the residual) before trusting f*h to set a height.")
        except Exception:
            pass
    if res.rel_iqr > 0.45:
        res.notes.append(f"relative IQR {res.rel_iqr:.2f} is high — treat f*h as "
                         f"indicative, not decisive")
    return res
