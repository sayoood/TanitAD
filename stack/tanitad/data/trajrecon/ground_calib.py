"""Fit the residual ground-projection extrinsics from tracked road features.

.. warning::
   **EXPERIMENTAL -- do not enable on data like this without re-checking.**
   On the 08-11 session this fit is ill-conditioned and its output must not be
   used to override the focus-of-expansion calibration.  Fitting the same tracks
   with different channel / free-parameter choices gives:

   ===========================  ========  =========
   configuration                roll      height
   ===========================  ========  =========
   roll only, lateral            +0.40      --
   roll only, both channels      +0.79      --
   roll + height, lateral        +0.33     0.800 (bound)
   height only, long range        --       1.109
   all free                      +0.46     1.304
   all free (repeat)             +0.30     1.337
   ===========================  ========  =========

   Roll spans 0.30-0.79 deg, height 1.11-1.34 m, two fits hit their bounds, and
   the best residual improvement any of them achieves is 4 % (1.379 -> 1.328 m).
   Large unstable parameter motion for no cost reduction is not calibration.

   The FOE estimate it would replace is far better conditioned: 318 000 flow
   vectors, 93 % inliers, *direction*-based (so immune to the flow-magnitude
   bias below), and independently cross-checked against a Hough vanishing point.
   Prefer it.

The focus of expansion pins the camera's **boresight** (yaw and pitch), but it
says nothing about **roll** -- a radial flow field is roll-symmetric -- and
nothing about the camera **height**, which sets the metric scale of the whole
ground plane.  Both were being assumed: roll = 0 and height = 1.25 m by decree.
Either one being wrong tilts or scales the projected ribbon, and no amount of
staring at the overlay tells you which.

There is a measurement available that needs no lane detection and no manual
tuning.  A patch of road surface is a static point in the world.  Between two
frames the vehicle moves by a displacement the trajectory already knows, so the
patch's position *in vehicle coordinates* must change by exactly that much:

    P2  =  R(-dpsi) (P1 - dp)

Back-project the tracked pixels onto the road plane with a candidate camera
model, predict ``P2`` from the trajectory, and the residual is a direct function
of the calibration error.  Fitting roll / pitch / yaw / height to minimise it
over thousands of tracks is then just least squares -- the geometry does the
work, not a slider.

What this can and cannot recover
--------------------------------
* **Recoverable**: roll, pitch, yaw, camera height.  Each leaves a distinct
  signature -- height scales the residual radially, pitch trades off against it
  but with a different range dependence, yaw rotates it, roll shears it.
* **Not recoverable**: the camera's *lateral offset from the vehicle centreline*.
  Translating the camera sideways moves every ground point by the same amount at
  both times, so it cancels out of the displacement entirely.  That offset is
  unobservable from motion and needs either a tape measure or an assumption
  about lane keeping -- see :func:`estimate_lateral_offset_by_lane_centring`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import wrap_pi


@dataclass
class GroundCalibResult:
    d_roll_deg: float
    d_pitch_deg: float
    d_yaw_deg: float
    height_m: float
    rms_before_m: float
    rms_after_m: float
    n_tracks: int

    def summary(self) -> str:
        return (f"GroundCalib on {self.n_tracks} road-feature tracks\n"
                f"  residual RMS : {self.rms_before_m:.3f} m -> {self.rms_after_m:.3f} m\n"
                f"  d_roll  = {self.d_roll_deg:+.2f} deg\n"
                f"  d_pitch = {self.d_pitch_deg:+.2f} deg\n"
                f"  d_yaw   = {self.d_yaw_deg:+.2f} deg\n"
                f"  height  = {self.height_m:.3f} m")


def unproject_ground(cam, uv, height_m: float | None = None):
    """Pixels -> (X, Y) on the road plane in the camera's vehicle frame.

    Returns ``(pts, valid)``; ``valid`` is False for rays at or above the
    horizon, which never meet the ground.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    h = cam.height_m if height_m is None else height_m
    d_c = np.stack([(uv[:, 0] - cam.cx) / cam.fx,
                    (uv[:, 1] - cam.cy) / cam.fy,
                    np.ones(len(uv))], axis=1)
    d_v = d_c @ cam.R_cv                      # R_cv^T applied on the right
    valid = d_v[:, 2] < -1e-6                 # must point downward to hit the road
    s = np.where(valid, h / np.where(valid, -d_v[:, 2], 1.0), np.nan)
    return np.stack([s * d_v[:, 0], s * d_v[:, 1]], axis=1), valid


def collect_road_tracks(video, cam, traj, sync, t_windows=None, gap_frames: int = 8,
                        max_pairs: int = 260, proc_width: int = 960,
                        roi=(0.62, 0.96), min_speed: float = 4.0,
                        corridor_half_w: float = 3.2, corridor_x=(5.0, 32.0)):
    """Track road-surface patches and pair them with the trajectory's motion.

    Features are taken from a band low in the image, which is road surface rather
    than sky, buildings or other traffic.  ``gap_frames`` sets the baseline: long
    enough that the vehicle has moved measurably, short enough that the patch is
    still in view and still trackable.
    """
    import cv2
    from .timesync import _iter_gray_frames

    scale = proc_width / video.width
    pw, ph = proc_width, int(round(video.height * scale / 2) * 2)

    # Mask the actual road *corridor*, projected from the camera model, rather
    # than a horizontal band.  A band below the horizon still contains kerb,
    # verge, parked cars and hedges -- none of which lie on the road plane, and
    # all of which break a planar homography: with a plain band, 169 of 209
    # frame pairs failed to reach 12 RANSAC inliers.  Projecting a corridor of
    # known width keeps the points that the planar assumption actually holds for.
    mask = np.zeros((ph, pw), dtype=np.uint8)
    xs = np.linspace(corridor_x[0], corridor_x[1], 60)
    left = np.stack([xs, np.full_like(xs, corridor_half_w), np.zeros_like(xs)], axis=1)
    right = np.stack([xs, np.full_like(xs, -corridor_half_w), np.zeros_like(xs)], axis=1)
    uvl, okl = cam.project(left)
    uvr, okr = cam.project(right)
    ok = okl & okr
    if ok.sum() >= 2:
        poly = np.concatenate([uvl[ok], uvr[ok][::-1]]) * scale
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    if mask.sum() < 500:                      # corridor off-screen -> fall back to a band
        hv = cam.horizon_v() * scale
        top = int(np.clip(hv + 0.055 * ph, 0.30 * ph, 0.92 * ph))
        mask[top:int(0.965 * ph), int(0.12 * pw):int(0.92 * pw)] = 255
    lk = dict(winSize=(25, 25), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    feat = dict(maxCorners=500, qualityLevel=0.008, minDistance=8, blockSize=7)

    ft = sync.frame_times(video.pts)
    ok_t = (ft >= traj.t[0]) & (ft <= traj.t[-1])
    speed = np.interp(ft, traj.t, traj.speed)
    usable = np.nonzero(ok_t & (speed > min_speed))[0]
    if t_windows:
        keep = np.zeros(len(ft), dtype=bool)
        for lo, hi in t_windows:
            keep |= (video.pts >= lo) & (video.pts <= hi)
        usable = usable[keep[usable]]
    if len(usable) < gap_frames + 4:
        return None

    starts = usable[usable + gap_frames < len(ft)]
    if len(starts) > max_pairs:
        starts = starts[np.linspace(0, len(starts) - 1, max_pairs).astype(int)]
    need = sorted(set(starts.tolist()) | set((starts + gap_frames).tolist()))

    src = (video.iter_gray(pw, ph, need[0], need[-1] - need[0] + 1) if hasattr(video, "iter_gray")
           else _iter_gray_frames(video.path, pw, ph, need[0], need[-1] - need[0] + 1))
    want = set(need)
    frames = {}
    for k, fr in enumerate(src):
        idx = need[0] + k
        if idx in want:
            frames[idx] = fr.copy()

    rows = []
    psi = np.unwrap(traj.heading)
    for i0 in starts:
        i1 = i0 + gap_frames
        if i0 not in frames or i1 not in frames:
            continue
        p0 = cv2.goodFeaturesToTrack(frames[i0], mask=mask, **feat)
        if p0 is None or len(p0) < 12:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(frames[i0], frames[i1], p0, None, **lk)
        if p1 is None or st is None or st.sum() < 12:
            continue
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(frames[i1], frames[i0], p1, None, **lk)
        good = (st.ravel() == 1) & (st2.ravel() == 1)
        good &= np.linalg.norm(p0.reshape(-1, 2) - pb.reshape(-1, 2), axis=1) < 1.0
        if good.sum() < 12:
            continue
        a = p0.reshape(-1, 2)[good] / scale         # back to full-resolution pixels
        b = p1.reshape(-1, 2)[good] / scale

        t0, t1 = ft[i0], ft[i1]
        dpsi = float(wrap_pi(np.interp(t1, traj.t, psi) - np.interp(t0, traj.t, psi)))
        dE = float(np.interp(t1, traj.t, traj.E) - np.interp(t0, traj.t, traj.E))
        dN = float(np.interp(t1, traj.t, traj.N) - np.interp(t0, traj.t, traj.N))
        p0_ = float(np.interp(t0, traj.t, psi))
        c, s = np.cos(-p0_), np.sin(-p0_)
        dp = np.array([c * dE - s * dN, s * dE + c * dN])   # displacement in frame-0 axes
        rows.append((a, b, dp, dpsi, float(t0), float(t1)))
    return rows


def prefilter(rows, cam, x_min: float = 2.0, x_max: float = 45.0):
    """Keep only tracks that land on a sensible stretch of road under ``cam``.

    Done once, up front.  Selecting inside the residual instead makes the
    residual vector's *length* depend on the parameters, which least-squares
    solvers reject outright -- and any solver that tolerated it would be free to
    lower the cost by discarding awkward points rather than by fitting them.
    """
    out = []
    for a, b, dp, dpsi, t0, t1 in rows:
        P1, v1 = unproject_ground(cam, a)
        P2, v2 = unproject_ground(cam, b)
        m = (v1 & v2 & np.isfinite(P1).all(1) & np.isfinite(P2).all(1)
             & (P1[:, 0] > x_min) & (P1[:, 0] < x_max)
             & (P2[:, 0] > 0.5) & (np.abs(P1[:, 1]) < 12.0))
        if m.sum() >= 6:
            out.append((a[m], b[m], dp, dpsi, t0, t1))
    return out


def _residual(params, rows, cam, cap: float = 25.0, att=None, channel: str = 'both'):
    import copy
    d_roll, d_pitch, d_yaw, h = params
    c2 = copy.copy(cam)
    c2.roll = cam.roll + d_roll
    c2.pitch = cam.pitch + d_pitch
    c2.yaw = cam.yaw + d_yaw
    c2.height_m = h
    res = []
    for a, b, dp, dpsi, t0, t1 in rows:
        # per-frame body attitude, if supplied
        if att is not None:
            ta, dpi, dro = att
            from .camera import camera_at
            ca = camera_at(c2, float(np.interp(t0, ta, dpi)), float(np.interp(t0, ta, dro)))
            cb = camera_at(c2, float(np.interp(t1, ta, dpi)), float(np.interp(t1, ta, dro)))
        else:
            ca = cb = c2
        P1, _ = unproject_ground(ca, a)
        P2, _ = unproject_ground(cb, b)
        cc, ss = np.cos(-dpsi), np.sin(-dpsi)
        q = P1 - dp
        pred = np.stack([cc * q[:, 0] - ss * q[:, 1], ss * q[:, 0] + cc * q[:, 1]], axis=1)
        res.append(P2 - pred)
    R = np.concatenate(res)
    # The longitudinal channel carries a large LK tracking bias at short range
    # (+2.96 m at 2-8 m, decaying with distance) because the road sweeps hundreds
    # of pixels between the paired frames and the tracker under-reads it.  Pitch
    # and height are exactly the parameters that trade against longitudinal range,
    # so fitting that channel pours a tracking artefact straight into them.  The
    # lateral channel has no such bias, which is why 'lateral' is the default.
    if channel == 'lateral':
        R = R[:, 1:2]
    elif channel == 'longitudinal':
        R = R[:, 0:1]
    r = R.ravel()
    # a ray that has swung above the horizon gives inf; bound it so the solver
    # sees a steep penalty rather than a NaN it cannot step away from
    return np.clip(np.nan_to_num(r, nan=cap, posinf=cap, neginf=-cap), -cap, cap)


def fit_ground_extrinsics(rows, cam, fit_height: bool = False, att=None,
                          channel: str = 'lateral',
                          free=('roll',)) -> GroundCalibResult:
    """Least-squares fit of roll/pitch/yaw (and optionally height) to the tracks."""
    from scipy.optimize import least_squares

    rows = prefilter(rows, cam)
    if not rows:
        raise ValueError("no tracks survived the road-plane prefilter")
    n = sum(len(r[0]) for r in rows)
    r0 = _residual([0.0, 0.0, 0.0, cam.height_m], rows, cam, att=att, channel=channel)
    rms0 = float(np.sqrt(np.mean(r0 ** 2)))

    # Only free the parameters this data can actually constrain.  Letting all
    # four float is ill-conditioned: on the lateral channel pitch, yaw and height
    # all run to their bounds while the residual collapses to 0.089 m, which is
    # overfitting, not calibration.  The lateral channel sees *roll*; pitch and
    # yaw are far better determined by the FOE, which is direction-based and so
    # immune to the flow-magnitude bias that contaminates the longitudinal channel.
    e = 1e-9
    span = dict(roll=np.deg2rad(12), pitch=np.deg2rad(8), yaw=np.deg2rad(8))
    lo = [-span["roll"] if "roll" in free else -e,
          -span["pitch"] if "pitch" in free else -e,
          -span["yaw"] if "yaw" in free else -e,
          0.8 if fit_height else cam.height_m - e]
    hi = [span["roll"] if "roll" in free else e,
          span["pitch"] if "pitch" in free else e,
          span["yaw"] if "yaw" in free else e,
          2.2 if fit_height else cam.height_m + e]
    sol = least_squares(_residual, [0.0, 0.0, 0.0, cam.height_m], bounds=(lo, hi),
                        args=(rows, cam, 25.0, att, channel), loss="soft_l1", f_scale=0.25, max_nfev=200)
    rms1 = float(np.sqrt(np.mean(sol.fun ** 2)))
    return GroundCalibResult(d_roll_deg=float(np.rad2deg(sol.x[0])),
                             d_pitch_deg=float(np.rad2deg(sol.x[1])),
                             d_yaw_deg=float(np.rad2deg(sol.x[2])),
                             height_m=float(sol.x[3]),
                             rms_before_m=rms0, rms_after_m=rms1, n_tracks=n)


def apply(cam, res: GroundCalibResult):
    """Return a copy of ``cam`` with the fitted corrections applied."""
    import copy
    c = copy.copy(cam)
    c.roll = cam.roll + np.deg2rad(res.d_roll_deg)
    c.pitch = cam.pitch + np.deg2rad(res.d_pitch_deg)
    c.yaw = cam.yaw + np.deg2rad(res.d_yaw_deg)
    c.height_m = res.height_m
    c.source = dict(cam.source)
    c.source["ground_calib"] = (f"road-feature fit on {res.n_tracks} tracks: "
                                f"roll{res.d_roll_deg:+.2f} pitch{res.d_pitch_deg:+.2f} "
                                f"yaw{res.d_yaw_deg:+.2f} deg, h={res.height_m:.2f} m, "
                                f"residual {res.rms_before_m:.3f}->{res.rms_after_m:.3f} m")
    return c


# --------------------------------------------------------------------------- #
# The one parameter motion cannot give you
# --------------------------------------------------------------------------- #
def road_ipm(video, cam, frame_indices, x_range=(6.0, 26.0), y_half: float = 6.0,
             px_per_m: float = 20.0):
    """Average bird's-eye view of the road surface over many frames.

    Individual frames are noisy and dashed markings come and go; averaging over
    a few hundred straight-driving frames turns them into continuous bright
    stripes at fixed lateral offsets, which is exactly what is needed to locate
    the lane.
    """
    import cv2

    nx = int((x_range[1] - x_range[0]) * px_per_m)
    ny = int(2 * y_half * px_per_m)
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_half, -y_half, ny)             # +y (left) on the left
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)], axis=1)
    uv, ok = cam.project(pts)
    mapx = np.where(ok, uv[:, 0], -1).reshape(nx, ny).astype(np.float32)
    mapy = np.where(ok, uv[:, 1], -1).reshape(nx, ny).astype(np.float32)

    acc = np.zeros((nx, ny), dtype=np.float64)
    n = 0
    for fi in frame_indices:
        img = video.read_bgr(int(fi)) if hasattr(video, "read_bgr") else None
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        acc += cv2.remap(g, mapy, mapx, cv2.INTER_LINEAR, borderValue=0)
        n += 1
    return (acc / max(n, 1)), ys, n


def estimate_lateral_offset_by_lane_centring(video, cam, traj, sync, max_frames: int = 260,
                                             max_yaw_rate_deg_s: float = 2.0,
                                             min_speed: float = 6.0, lane_width_hint: float = 3.0):
    """Camera offset from the lane centre, from averaged lane markings.

    **This is the one extrinsic that motion cannot supply.**  Sliding the camera
    sideways displaces every ground point identically at both ends of a track, so
    it cancels out of ``P2 - P1`` and :func:`fit_ground_extrinsics` is blind to
    it.  The only ways to pin it down are a tape measure in the car, or an
    assumption about where the vehicle sits in its lane.

    This takes the second route and says so: it averages a metric bird's-eye view
    over straight-driving frames, finds the two brightest lateral stripes either
    side of the camera (the lane markings), and reports how far the camera sits
    from the midpoint between them.  That is a *lane-centring assumption*, not a
    measurement -- a driver who habitually hugs one side will bias it.  Treat the
    output as a starting value to sanity-check against a tape measure, not as
    ground truth.
    """
    from scipy.signal import find_peaks

    yr = np.rad2deg(traj.yaw_rate)
    ft = sync.frame_times(video.pts)
    ok = (ft >= traj.t[0]) & (ft <= traj.t[-1])
    q = np.interp(ft, traj.t, np.abs(yr)) < max_yaw_rate_deg_s
    q &= np.interp(ft, traj.t, traj.speed) > min_speed
    idx = np.nonzero(ok & q)[0]
    if len(idx) < 30:
        return None
    if len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).astype(int)]

    bev, ys, n = road_ipm(video, cam, idx)
    prof = bev.mean(axis=0)
    prof = prof - np.convolve(prof, np.ones(41) / 41, mode="same")   # flatten illumination
    pk, props = find_peaks(prof, prominence=max(1.0, 0.30 * prof.std()), distance=int(0.35 * 20))
    if len(pk) < 2:
        return dict(offset_m=None, n_frames=n, note="fewer than two lane markings found")

    y_pk = ys[pk]
    left = y_pk[y_pk > 0.2]
    right = y_pk[y_pk < -0.2]
    if not len(left) or not len(right):
        return dict(offset_m=None, n_frames=n, peaks_m=np.round(y_pk, 2).tolist(),
                    note="markings found on only one side of the camera")
    yl = left[np.argmin(left)]                 # nearest marking to the left
    yr_ = right[np.argmax(right)]              # nearest to the right
    width = yl - yr_
    centre = 0.5 * (yl + yr_)
    return dict(offset_m=float(-centre), lane_width_m=float(width),
                marking_left_m=float(yl), marking_right_m=float(yr_),
                peaks_m=np.round(y_pk, 2).tolist(), n_frames=n,
                plausible=bool(2.2 < width < 4.2),
                note="lane-centring ASSUMPTION, not a measurement")
