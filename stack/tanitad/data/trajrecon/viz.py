"""Visualisation: bird's-eye view and trajectory projection into the image."""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def clip_polyline_to_camera(pts_v: np.ndarray, cam, min_z: float = 0.5):
    """Split a vehicle-frame polyline into pieces that lie in front of the camera.

    Projecting without this is the classic dashcam-overlay artefact: a point a
    few centimetres *behind* the image plane still yields a finite pixel, so the
    past part of the trajectory gets drawn as a stray line whipping across the
    sky.  Segments crossing the plane are cut exactly at the crossing.
    """
    pts_v = np.atleast_2d(np.asarray(pts_v, dtype=float))
    z = cam.to_camera(pts_v)[:, 2]
    good = z > min_z
    out, cur = [], []
    for i in range(len(pts_v)):
        if good[i]:
            cur.append(pts_v[i])
            if i + 1 < len(pts_v) and not good[i + 1]:
                a = (z[i] - min_z) / (z[i] - z[i + 1])
                cur.append(pts_v[i] + a * (pts_v[i + 1] - pts_v[i]))
        else:
            if i + 1 < len(pts_v) and good[i + 1]:
                a = (min_z - z[i]) / (z[i + 1] - z[i])
                cur = [pts_v[i] + a * (pts_v[i + 1] - pts_v[i])]
            elif cur:
                out.append(np.array(cur)); cur = []
    if cur:
        out.append(np.array(cur))
    return [s for s in out if len(s) >= 2]


def trajectory_ribbon(x, y, width: float = 1.8, z: float = 0.0, yaw=None):
    """Left and right edges of a constant-width band centred on the path.

    The band's normal comes from ``yaw`` when it is supplied, and only falls back
    to the path tangent otherwise.  That matters at a standstill: a stopped
    vehicle's successive trajectory points are identical, the tangent is
    undefined, and normalising a zero-length gradient collapses the ribbon to
    zero width -- it disappears from the overlay exactly when the car is stopped
    at a junction.  The vehicle still has a heading while stationary, so use it.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if yaw is not None:
        yaw = np.asarray(yaw, dtype=float)
        tx, ty = np.cos(yaw), np.sin(yaw)
    else:
        tx, ty = np.gradient(x), np.gradient(y)
        n = np.hypot(tx, ty)
        bad = n < 1e-6
        if bad.any() and not bad.all():
            # carry the nearest valid tangent across the degenerate stretch
            good = np.flatnonzero(~bad)
            nearest = good[np.argmin(np.abs(np.flatnonzero(bad)[:, None] - good[None, :]), axis=1)]
            tx[bad], ty[bad] = tx[nearest], ty[nearest]
            n = np.hypot(tx, ty)
        elif bad.all():
            tx, ty = np.ones_like(x), np.zeros_like(y)
            n = np.ones_like(x)
        tx, ty = tx / n, ty / n
    nx, ny = -ty, tx                    # unit left-normal
    h = width / 2.0
    left = np.stack([x + nx * h, y + ny * h, np.full_like(x, z)], axis=1)
    right = np.stack([x - nx * h, y - ny * h, np.full_like(x, z)], axis=1)
    return left, right


# --------------------------------------------------------------------------- #
# Image overlay
# --------------------------------------------------------------------------- #
def draw_trajectory_on_image(img, ego, cam, vehicle_width: float = 1.8,
                             draw_past: bool = True, tick_every: float = 1.0,
                             alpha: float = 0.35, draw_horizon: bool = True,
                             near_clip_m: float = 4.0, label_max_m: float = 60.0,
                             lateral_offset_m: float = 0.0):
    """Render the ego trajectory onto a BGR image (OpenCV array).

    Future path is drawn as a translucent ribbon with per-second tick marks,
    the past as a thin dashed line.  Returns a new image.

    ``near_clip_m`` drops the first few metres of the future path.  The ribbon
    is a fixed 1.8 m wide in the world, so right at the camera it subtends most
    of the frame and buries the road; starting it a car-length ahead is both
    more readable and more honest, since that near strip is under the bonnet and
    not actually visible.
    """
    import cv2

    out = img.copy()
    overlay = img.copy()
    H, W = out.shape[:2]
    SHIFT = 6                      # sub-pixel rasterisation, see draw_bev_cv
    K = 1 << SHIFT
    sub = lambda a: np.round(np.asarray(a, dtype=float) * K).astype(np.int32)

    t, x, y = ego["t"], ego["x"], ego["y"]
    # The trajectory is the *camera's* ground track (the phone is what GNSS
    # locates).  The vehicle body is offset from it by however far the mount sits
    # off the centreline, so the drawn band has to be shifted by that much or it
    # rides consistently to one side of the lane.  This offset is structurally
    # unobservable from motion -- see ground_calib -- so it has to be measured.
    if lateral_offset_m:
        y = y - lateral_offset_m
    s_fwd = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    s_fwd = s_fwd - np.interp(0.0, t, s_fwd)
    # Scale the near clip with speed.  A fixed 4 m is right at 50 km/h but at a
    # 0.7 m/s crawl it is most of the whole 5 s window, so the overlay went almost
    # empty exactly where a reviewer most wants to check it.  Clip roughly the
    # distance covered in the next 0.4 s instead, with a 1 m floor.
    near = float(np.clip(0.4 * abs(ego.get("speed_ref", 0.0)), 1.0, near_clip_m))
    fut = (t >= 0) & (s_fwd >= near)
    if fut.sum() < 2:                      # stopped: show the whole future window
        fut = t >= 0
    past = t <= 0

    # ---- future ribbon, drawn far-to-near so nearer quads paint on top ---- #
    if fut.sum() >= 2:
        left, right = trajectory_ribbon(x[fut], y[fut], vehicle_width,
                                        yaw=ego['yaw'][fut] if 'yaw' in ego else None)
        tf = t[fut]
        pc_l = cam.to_camera(left)[:, 2]
        pc_r = cam.to_camera(right)[:, 2]
        uv_l, ok_l = cam.project(left)
        uv_r, ok_r = cam.project(right)
        span = max(tf[-1] - tf[0], 1e-6)
        quads = []
        for i in range(len(tf) - 1):
            if not (ok_l[i] and ok_l[i + 1] and ok_r[i] and ok_r[i + 1]):
                continue
            quad = sub([uv_l[i], uv_l[i + 1], uv_r[i + 1], uv_r[i]])
            depth = 0.25 * (pc_l[i] + pc_l[i + 1] + pc_r[i] + pc_r[i + 1])
            f = (tf[i] - tf[0]) / span
            # green (near, imminent) -> amber (far, later)
            col = (int(60 + 40 * f), int(230 - 60 * f), int(40 + 200 * f))
            quads.append((depth, quad, col))
        for _, quad, col in sorted(quads, key=lambda q: -q[0]):
            cv2.fillConvexPoly(overlay, quad, col, lineType=cv2.LINE_AA, shift=SHIFT)

        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        # centre line + edges on top of the translucent fill
        centre = np.stack([x[fut], y[fut], np.zeros(fut.sum())], axis=1)
        for seg, colr, th in ((centre, (255, 255, 255), 2),
                              (left, (200, 255, 200), 1),
                              (right, (200, 255, 200), 1)):
            for piece in clip_polyline_to_camera(seg, cam):
                uv, ok = cam.project(piece)
                if ok.sum() >= 2:
                    cv2.polylines(out, [sub(uv[ok])], False, colr, th, cv2.LINE_AA, shift=SHIFT)

        # one tick per `tick_every` seconds, labelled with time and distance
        sf = s_fwd[fut]
        last_row = None
        for tv in np.arange(np.ceil(tf[0]), tf[-1] + 1e-9, tick_every):
            xi = np.interp(tv, tf, x[fut]); yi = np.interp(tv, tf, y[fut])
            si = np.interp(tv, tf, sf)
            yi_hdg = np.interp(tv, tf, ego['yaw'][fut]) if 'yaw' in ego else None
            if yi_hdg is None:
                dxi = np.interp(tv, tf, np.gradient(x[fut]))
                dyi = np.interp(tv, tf, np.gradient(y[fut]))
                n = float(np.hypot(dxi, dyi)) or 1.0
                nx, ny = -dyi / n, dxi / n
            else:
                nx, ny = -np.sin(yi_hdg), np.cos(yi_hdg)
            bar = np.array([[xi + nx * vehicle_width / 2, yi + ny * vehicle_width / 2, 0.0],
                            [xi - nx * vehicle_width / 2, yi - ny * vehicle_width / 2, 0.0]])
            uv, ok = cam.project(bar)
            if not ok.all():
                continue
            ps = sub(uv)
            cv2.line(out, tuple(ps[0]), tuple(ps[1]), (255, 255, 255), 2, cv2.LINE_AA, shift=SHIFT)
            p = uv.astype(np.int32)
            # Labels pile up on top of each other near the vanishing point, where
            # successive seconds are only a few pixels apart -- keep them legible.
            row = int(p[1][1])
            if si > label_max_m or (last_row is not None and abs(row - last_row) < 18):
                continue
            last_row = row
            lab = f"{tv:.0f}s / {si:.0f}m"
            org = (int(p[1][0]) + 6, row + 4)
            cv2.putText(out, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- past track ---- #
    if draw_past and past.sum() >= 2:
        seg = np.stack([x[past], y[past], np.zeros(past.sum())], axis=1)
        for piece in clip_polyline_to_camera(seg, cam):
            uv, ok = cam.project(piece)
            p = sub(uv[ok])
            for i in range(0, len(p) - 1, 2):      # dashed
                cv2.line(out, tuple(p[i]), tuple(p[i + 1]), (255, 160, 60), 2,
                         cv2.LINE_AA, shift=SHIFT)

    if draw_horizon:
        hv = cam.horizon_v()
        if np.isfinite(hv) and 0 <= hv < H:
            cv2.line(out, (0, int(hv)), (W, int(hv)), (120, 120, 255), 1, cv2.LINE_AA)

    # In a tight turn the path leaves the ~60 deg field of view within a second or
    # two, so the overlay simply stops -- which is correct but reads as "nothing
    # was predicted".  Mark the edge it exited through and how far out it goes.
    if fut.sum() >= 2:
        centre = np.stack([x[fut], y[fut], np.zeros(int(fut.sum()))], axis=1)
        uv, ok = cam.project(centre)
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if inside.any() and not inside.all():
            last = int(np.nonzero(inside)[0][-1])
            if last + 1 < len(uv):
                pu, pv = uv[last]
                side_right = uv[last + 1][0] > pu
                ex = W - 26 if side_right else 26
                ey = int(np.clip(pv, 30, H - 30))
                tip = (ex + (14 if side_right else -14), ey)
                cv2.arrowedLine(out, (int(np.clip(pu, 30, W - 30)), ey), tip,
                                (60, 230, 255), 3, cv2.LINE_AA, tipLength=0.4)
                lab = f"path exits frame, continues {np.hypot(x[fut][-1], y[fut][-1]):.0f} m"
                org = (ex - (230 if side_right else -10), ey - 12)
                cv2.putText(out, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 230, 255), 1,
                            cv2.LINE_AA)

    return out


def draw_hud(img, ego, extra: dict | None = None):
    """Small text panel: speed, yaw rate, uncertainty, completeness."""
    import cv2
    out = img.copy()
    v = ego["speed_ref"]
    i0 = int(np.argmin(np.abs(ego["t"])))
    yr = np.rad2deg(ego["yaw_rate"][i0]) if len(ego["yaw_rate"]) else 0.0
    lines = [f"speed   {v:5.2f} m/s  ({v * 3.6:5.1f} km/h)",
             f"yawrate {yr:+6.1f} deg/s",
             f"pos 1s  {np.median(ego['pos_std']):4.2f} m",
             f"t = {ego['t_ref']:.2f} s" + ("" if ego["complete"] else "  [TRUNCATED]")]
    for k, val in (extra or {}).items():
        lines.append(f"{k} {val}")
    y = 30
    for ln in lines:
        cv2.putText(out, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 255, 120), 1, cv2.LINE_AA)
        y += 26
    return out


# --------------------------------------------------------------------------- #
# Bird's-eye view
# --------------------------------------------------------------------------- #
def _nice_step(span: float, target_ticks: float = 5.0) -> float:
    """A round grid step (1/2/5 x 10^k) giving roughly ``target_ticks`` divisions."""
    if not np.isfinite(span) or span <= 0:
        return 10.0
    raw = span / max(target_ticks, 1.0)
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def bev_extent(ego, lateral_offset_m: float = 0.0, pad: float = 0.12,
               min_forward: float = 25.0, min_y_half: float = 6.0):
    """Axis limits that contain the whole ego window, however fast it is going.

    A fixed 46 m box was right at 50 km/h and wrong everywhere else: it clipped
    the 5 s horizon above ~33 km/h (at 100 km/h the horizon is 139 m, three times
    the box) and at a crawl it left the path as a dot at the origin.  The limits
    are taken from the data instead, with floors so a stopped vehicle does not
    zoom to an absurd scale.
    """
    x = np.asarray(ego["x"], dtype=float)
    y = np.asarray(ego["y"], dtype=float) - lateral_offset_m
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return (-10.0, 46.0), 15.0
    x, y = x[ok], y[ok]
    x_lo, x_hi = float(x.min()), float(x.max())
    span = max(x_hi - x_lo, 1.0)
    x_lo -= pad * span
    x_hi += pad * span
    # always keep a little road behind the car, and a sane minimum ahead
    x_lo = min(x_lo, -5.0)
    x_hi = max(x_hi, x_lo + min_forward)
    yh = max(float(np.abs(y).max()) * (1.0 + pad), min_y_half)
    # Deliberately not snapped to the tick step: rounding a 112 m window out to
    # the nearest 50 m turned it into 200 m and shrank the path to a third of the
    # panel.  Ticks are placed at round values *inside* the range instead.
    return (float(x_lo), float(x_hi)), float(yh)


def draw_bev_cv(ego, size=(620, 720), vehicle_width: float = 1.8,
                x_range=None, y_half: float | None = None, title: str | None = None,
                lateral_offset_m: float = 0.0):
    """Bird's-eye panel drawn straight with OpenCV primitives.

    Styled to match :func:`draw_bev` (light background, grey grid, labelled
    axes) -- the matplotlib version reads more cleanly than a dark panel, but a
    figure costs ~0.4 s to build and save, which is tens of minutes of pure
    plotting over a whole session.  This draws the same picture in a few
    milliseconds so a full-length video is practical.

    Same convention as the matplotlib panel: x forward is up, +y (left) is left.

    ``x_range`` and ``y_half`` default to whatever contains the ego window, so
    the full horizon stays on screen at any speed; pass them explicitly to pin
    the scale.
    """
    import cv2

    if x_range is None or y_half is None:
        auto_x, auto_y = bev_extent(ego, lateral_offset_m)
        x_range = x_range or auto_x
        y_half = y_half if y_half is not None else auto_y

    W, H = size
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    ML, MR, MT, MB = 62, 12, 30, 46                 # margins for the axis labels
    PW, PH = W - ML - MR, H - MT - MB
    s = min(PH / (x_range[1] - x_range[0]), PW / (2.0 * y_half))
    # Equal aspect means one axis ends up with slack.  Grow both to exactly the
    # visible area, keeping the data centred, so the grid fills the panel rather
    # than stopping short of the frame.
    _xc = 0.5 * (x_range[0] + x_range[1])
    x_range = (_xc - PH / (2.0 * s), _xc + PH / (2.0 * s))
    y_half = PW / (2.0 * s)
    cx_px = ML + PW / 2.0
    cy_px = MT + PH + x_range[0] * s

    # OpenCV rasterises at integer pixels unless told otherwise, and at ~18 px/m
    # that quantisation is +-2.8 cm -- it turns a perfectly smooth path into a
    # visibly wavy line.  Matplotlib draws in float coordinates, which is the only
    # reason its version of this panel looked smoother; the underlying trajectory
    # is identical.  SHIFT gives OpenCV 1/64 px sub-pixel precision.
    SHIFT = 6
    K = 1 << SHIFT

    def to_px(x, y):
        u = cx_px - np.asarray(y, dtype=float) * s
        v = cy_px - np.asarray(x, dtype=float) * s
        return np.stack([u, v], axis=-1).astype(np.int32)

    def to_sub(x, y):
        u = (cx_px - np.asarray(y, dtype=float) * s) * K
        v = (cy_px - np.asarray(x, dtype=float) * s) * K
        return np.round(np.stack([u, v], axis=-1)).astype(np.int32)

    GRID, RING, AXIS, TEXT = (232, 232, 232), (238, 238, 238), (150, 150, 150), (70, 70, 70)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # Everything inside the axes is drawn on its own layer and only the plot
    # rectangle is copied back.  Clipping each primitive by hand is fiddly and
    # easy to miss -- the past-track polyline and the range rings both used to
    # bleed out over the tick labels.
    layer = np.full_like(img, 255)

    # Grid and range rings follow the axis span, so the panel stays readable
    # whether the 5 s horizon is 12 m or 140 m.
    xstep = _nice_step(x_range[1] - x_range[0])
    ystep = _nice_step(2.0 * y_half, target_ticks=4.0)
    fmt = (lambda v: f"{v:g}")
    for r in np.arange(xstep, max(abs(x_range[0]), abs(x_range[1])) + 1e-9, xstep):
        cv2.circle(layer, tuple(to_px(0, 0)), int(r * s), RING, 1, cv2.LINE_AA)
    x_ticks = list(np.arange(np.ceil(x_range[0] / xstep) * xstep,
                             x_range[1] + 1e-9, xstep))
    y_ticks = list(np.arange(-np.floor(y_half / ystep) * ystep,
                             np.floor(y_half / ystep) * ystep + 1e-9, ystep))
    for xv in x_ticks:
        cv2.line(layer, tuple(to_px(xv, y_half)), tuple(to_px(xv, -y_half)), GRID, 1, cv2.LINE_AA)
    for yv in y_ticks:
        cv2.line(layer, tuple(to_px(x_range[0], yv)), tuple(to_px(x_range[1], yv)),
                 GRID, 1, cv2.LINE_AA)
    cv2.line(layer, tuple(to_px(x_range[0], 0)), tuple(to_px(x_range[1], 0)), AXIS, 1, cv2.LINE_AA)
    cv2.line(layer, tuple(to_px(0, y_half)), tuple(to_px(0, -y_half)), AXIS, 1, cv2.LINE_AA)

    t, x, y = ego["t"], ego["x"], ego["y"]
    if lateral_offset_m:
        y = y - lateral_offset_m           # camera track -> vehicle-body track
    fut, past = t >= 0, t <= 0

    if fut.sum() > 1:
        left, right = trajectory_ribbon(x[fut], y[fut], vehicle_width,
                                        yaw=ego['yaw'][fut] if 'yaw' in ego else None)
        poly = np.concatenate([to_sub(left[:, 0], left[:, 1]),
                               to_sub(right[::-1, 0], right[::-1, 1])])
        band = layer.copy()
        cv2.fillPoly(band, [poly], (132, 220, 61), lineType=cv2.LINE_AA, shift=SHIFT)
        cv2.addWeighted(band, 0.30, layer, 0.70, 0, layer)
        cv2.polylines(layer, [to_sub(x[fut], y[fut])], False, (75, 127, 27), 2,
                      cv2.LINE_AA, shift=SHIFT)
    if past.sum() > 1:
        cv2.polylines(layer, [to_sub(x[past], y[past])], False, (64, 159, 255), 2,
                      cv2.LINE_AA, shift=SHIFT)

    box = np.array([[2.2, vehicle_width / 2], [2.2, -vehicle_width / 2],
                    [-2.2, -vehicle_width / 2], [-2.2, vehicle_width / 2]])
    cv2.fillPoly(layer, [to_sub(box[:, 0], box[:, 1])], (176, 108, 43),
                 lineType=cv2.LINE_AA, shift=SHIFT)
    cv2.polylines(layer, [to_sub(box[:, 0], box[:, 1])], True, (60, 60, 60), 1,
                  cv2.LINE_AA, shift=SHIFT)

    if fut.sum() > 1:
        for tv in np.arange(1, np.floor(t[fut][-1]) + 1e-9):
            p = to_sub(np.interp(tv, t[fut], x[fut]), np.interp(tv, t[fut], y[fut]))
            cv2.circle(layer, tuple(p), 3 * K, (75, 127, 27), -1, cv2.LINE_AA, shift=SHIFT)
            p = to_px(np.interp(tv, t[fut], x[fut]), np.interp(tv, t[fut], y[fut]))
            cv2.putText(layer, f"{tv:.0f}s", (int(p[0]) + 7, int(p[1]) + 4),
                        FONT, 0.42, TEXT, 1, cv2.LINE_AA)

    img[MT:MT + PH, ML:ML + PW] = layer[MT:MT + PH, ML:ML + PW]
    cv2.rectangle(img, (ML, MT), (ML + PW, MT + PH), AXIS, 1)

    for xv in x_ticks:
        p = to_px(xv, y_half)
        cv2.putText(img, fmt(xv), (ML - 34, int(p[1]) + 4), FONT, 0.42, TEXT, 1, cv2.LINE_AA)
    for yv in y_ticks:
        p = to_px(x_range[0], yv)
        cv2.putText(img, fmt(yv), (int(p[0]) - 9, MT + PH + 17), FONT, 0.42, TEXT, 1, cv2.LINE_AA)

    cv2.putText(img, title or "bird's-eye    orange = past    green = future",
                (ML, 20), FONT, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "x [m]", (6, MT + 12), FONT, 0.40, TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, "lateral y [m]   (+ = left)", (ML + PW // 2 - 82, H - 12),
                FONT, 0.44, TEXT, 1, cv2.LINE_AA)
    return img


def compose_panels(img_cam, img_bev, height: int = 720):
    """Camera view on the left, bird's-eye on the right, matched heights."""
    import cv2
    h, w = img_cam.shape[:2]
    cam = cv2.resize(img_cam, (int(round(w * height / h / 2) * 2), height),
                     interpolation=cv2.INTER_AREA)
    if img_bev.shape[0] != height:
        img_bev = cv2.resize(img_bev, (img_bev.shape[1], height))
    out = np.hstack([cam, img_bev])
    if out.shape[1] % 2:                  # H.264 needs even dimensions
        out = out[:, :-1]
    return out


def draw_bev(ax, ego, vehicle_width: float = 1.8, xlim=(-8, 45), ylim=(-14, 14)):
    """Matplotlib bird's-eye panel, ISO 8855 / ROS style (x forward, y left)."""
    t, x, y = ego["t"], ego["x"], ego["y"]
    fut, past = t >= 0, t <= 0

    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    for r in (10, 20, 30, 40):
        ax.add_artist(__import__("matplotlib.patches", fromlist=["Circle"]).Circle(
            (0, 0), r, fill=False, ec="0.9", lw=0.7, zorder=0))

    if past.sum() > 1:
        ax.plot(y[past], x[past], "-", color="#ff9f40", lw=1.6, label="past", zorder=2)
    if fut.sum() > 1:
        left, right = trajectory_ribbon(x[fut], y[fut], vehicle_width,
                                        yaw=ego['yaw'][fut] if 'yaw' in ego else None)
        ax.fill(np.concatenate([left[:, 1], right[::-1, 1]]),
                np.concatenate([left[:, 0], right[::-1, 0]]),
                color="#3ddc84", alpha=0.30, lw=0, zorder=1)
        sc = ax.scatter(y[fut], x[fut], c=t[fut], cmap="viridis", s=12, zorder=3)
        ax.plot(y[fut], x[fut], "-", color="#1b7f4b", lw=1.2, zorder=3)
        for tv in np.arange(1, np.floor(t[fut][-1]) + 1e-9):
            ax.annotate(f"{tv:.0f}s", (np.interp(tv, t[fut], y[fut]), np.interp(tv, t[fut], x[fut])),
                        fontsize=7, color="0.25", xytext=(4, 2), textcoords="offset points", zorder=4)
    # ego vehicle box
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (-vehicle_width / 2, -2.2), vehicle_width, 4.4, fc="#2b6cb0", ec="k", lw=0.8, alpha=.85, zorder=5))

    ax.set_xlim(ylim[1], ylim[0])           # +y (left) on the left of the plot
    ax.set_ylim(*xlim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lateral y [m]   (+ = left)")
    ax.set_ylabel("longitudinal x [m]   (+ = forward)")
    ax.grid(True, ls=":", alpha=.4)
    return ax


# --------------------------------------------------------------------------- #
# Steering-wheel gauge
# --------------------------------------------------------------------------- #
def draw_steering_wheel(img, wheel_deg: float, valid: bool = True,
                        centre=None, radius: int = 84, alpha: float = 0.78,
                        max_deg: float = 520.0, rate_deg_s: float | None = None):
    """Draw a rotating steering-wheel gauge, by default at the top-right.

    The rim, hub and three spokes rotate with ``wheel_deg`` (positive = left/
    counter-clockwise, matching the +yaw-is-left convention used everywhere
    else).  A fixed 12 o'clock index mark stays put so the rotation is readable,
    and an arc from that mark shows the magnitude and direction at a glance.

    When ``valid`` is False -- the vehicle is too slow for steering to be
    observable from its motion -- the gauge greys out and says so rather than
    displaying a number that is not measurable.
    """
    import cv2

    out = img.copy()
    H, W = out.shape[:2]
    if centre is None:
        centre = (W - int(radius * 1.45) - 16, int(radius * 1.42) + 12)
    cx, cy = int(centre[0]), int(centre[1])
    SHIFT = 6
    K = 1 << SHIFT
    C = lambda p: (int(round(p[0] * K)), int(round(p[1] * K)))

    rim = (235, 235, 240) if valid else (150, 150, 150)
    accent = (90, 235, 120) if valid else (140, 140, 140)
    panel = out.copy()
    cv2.circle(panel, (cx, cy), int(radius * 1.30), (28, 28, 32), -1, cv2.LINE_AA)
    cv2.addWeighted(panel, alpha, out, 1 - alpha, 0, out)

    # travel arc from top, in the direction actually steered
    a = float(np.clip(wheel_deg, -max_deg, max_deg))
    if valid and abs(a) > 1.0:
        cv2.ellipse(out, (cx, cy), (int(radius * 1.16), int(radius * 1.16)),
                    -90.0, 0.0, -a, accent, 5, cv2.LINE_AA)
    cv2.ellipse(out, (cx, cy), (int(radius * 1.16), int(radius * 1.16)),
                -90.0, -12.0, 12.0, (120, 120, 130), 1, cv2.LINE_AA)

    th = np.deg2rad(-a)                       # image y is down -> negate
    ct, st = np.cos(th), np.sin(th)
    rot = lambda vx, vy: (cx + vx * ct - vy * st, cy + vx * st + vy * ct)

    cv2.circle(out, (cx, cy), radius, rim, 7, cv2.LINE_AA)
    for ang in (200.0, 340.0, 90.0):          # two lower spokes + one upper
        r = np.deg2rad(ang)
        p = rot(radius * 0.92 * np.cos(r), -radius * 0.92 * np.sin(r))
        cv2.line(out, C((cx, cy)), C(p), rim, 6, cv2.LINE_AA, shift=SHIFT)
    cv2.circle(out, (cx, cy), int(radius * 0.30), (55, 55, 62), -1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), int(radius * 0.30), rim, 2, cv2.LINE_AA)
    # index mark on the rim, rotating with the wheel
    cv2.line(out, C(rot(0, -radius * 0.72)), C(rot(0, -radius * 1.02)), accent, 5,
             cv2.LINE_AA, shift=SHIFT)
    # fixed straight-ahead reference
    cv2.line(out, (cx, cy - int(radius * 1.28)), (cx, cy - int(radius * 1.10)),
             (200, 200, 210), 2, cv2.LINE_AA)

    if valid:
        # OpenCV's Hershey fonts are ASCII-only -- a degree sign renders as "??"
        txt = f"{a:+.0f} deg"
        sub = "left" if a > 0 else ("right" if a < 0 else "centred")
    else:
        txt, sub = "--", "too slow"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    yb = cy + int(radius * 1.34) + 30
    cv2.putText(out, txt, (cx - tw // 2, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, txt, (cx - tw // 2, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                accent, 2, cv2.LINE_AA)
    lab = f"steering {sub}" if rate_deg_s is None else f"steering {sub}   {rate_deg_s:+.0f} deg/s"
    (lw, _), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(out, lab, (cx - lw // 2, yb + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, lab, (cx - lw // 2, yb + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (215, 215, 225), 1, cv2.LINE_AA)
    return out
