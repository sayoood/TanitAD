#!/usr/bin/env python3
"""Measure, ON THE DELIVERED VIDEO, how far the drawn corridor is from the lane
line -- IN METRES, AT RANGES OUT TO 50 m.

WHY THIS EXISTS: the previous on-video check was wrong in three ways at once, and
I used it to reject a calibration.

  1. ⛔ WRONG PANEL WIDTH. It assumed the camera panel was 1280*1178/1280 = 1178 px
     wide. MEASURED here: ``compose_panels`` resizes 1920x1080 to height 720, so the
     camera panel is exactly 1280 px (scale 2/3) and the BEV starts at column 1280
     (first near-white column = 1280, confirmed on both renders). The old probe cut
     153 source columns off the right, so every RIGHT-hand number it printed was
     measured against a truncated image.
  2. ⛔ IT NEVER SAW THE FAR FIELD. It sampled fixed composite rows 420-660. With
     f*h = 2703 and horizon 485 those rows are source rows 630-990, i.e. ranges
     **4.8 m to 14.2 m**. Sayed's complaint is *"cuts road markings at LARGE
     distances"*. The far field lives in composite rows ~360-440 and was never read.
  3. ⛔ ITS GREEN TEST DROPPED THE FAR RIBBON. The fill ramps green->amber with time
     (``col = (60+40f, 230-60f, 40+200f)`` BGR), so beyond a few seconds G < R and the
     fill is not green at all; only the two edge polylines, drawn opaque at
     (200,255,200), stay green. The probe required >= 25 green pixels in a row, which
     the far field never has -- it silently skipped exactly the rows that matter.

WHAT THIS MEASURES. For each frame: the corridor's left edge is read from the
composite (the light-green edge polyline), and the LEFT PAINTED LINE is read from the
ORIGINAL source frame -- no green tint, no video recompression -- as a RANSAC line
through ridge points across the whole far band. Fitting one line across rows, rather
than searching a +/-1 m window row by row, is what keeps the association coherent:
a per-row window jumps between the lane line, the shoulder and the barrier, which is
how ``lane_residual.py`` came to confirm whatever height it was handed.

The line's identity is anchored at 12 m (where every arm agrees) and the divergence
is then read out to 50 m. So the number reported is exactly the symptom: *does the
corridor stay the same distance from the SAME painted line as range grows?*
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

PANEL = 1280                 # camera panel width in the composite (MEASURED)
S = 2.0 / 3.0                # composite / source scale (720/1080 == 1280/1920)
ANCHOR_M = 12.0              # range at which the painted line's identity is fixed


def corridor_edges(frame, row, v_min=150, max_gap=None):
    """Left/right edge columns of the drawn ribbon at one composite row.

    The two edge polylines are drawn OPAQUE at (200,255,200) after the fill is
    blended, so they are green at every range even where the fill has gone amber.

    ⛔ THE HUE TEST ALONE MATCHES ROADSIDE VEGETATION, and that manufactured a
    metric. MEASURED 2026-09-19 on the v6 render, composite frame 165 row 340:
    ``g > r+20 & g > b+20`` selected columns 834-845 at BGR (28,99,77)/(0,64,42)
    -- **dark foliage on the bank**, nowhere near the ribbon, which at the next
    scanned row sits at columns 470-521 in pale (206,229,208). Two distinct
    failures followed, and BOTH inflate a margin in metres:

      * the foliage ALONE was returned as a 16 px "ribbon". Scaled by the drawn
        width it gives 0.112 m/px -- **20x the true scale at that row** -- so an
        ordinary pixel distance became a -6.72 / -8.08 / -10.68 m "margin";
      * where the ribbon and the foliage share a row, min/max over green spans
        BOTH: row 360 returned (470, 835) for a ribbon that ends at 521, i.e. a
        547 px "1.855 m", collapsing the scale to 0.003 m/px.

    These produced "19.8 % of frames leave the road" on a calibration whose every
    flagged frame is, on inspection, fully on-road at every range. Same class as
    ``R-2026-09-15-seam``: a detector locking onto continuous non-target structure
    that the hue test could not exclude.

    The drawn edge is PALE -- every channel above ~190 -- while foliage is dark;
    ``v_min`` is that separation and is the ONLY thing added to the hue test.
    ``max_gap`` additionally rejects a row whose green pixels are not contiguous
    enough to be one ribbon, which is what catches a merge the brightness gate
    misses (a sunlit leaf).
    """
    im = frame[row, :PANEL].astype(int)
    b, g, r = im[:, 0], im[:, 1], im[:, 2]
    green = (g > r + 20) & (g > b + 20) & (g > v_min)
    idx = np.flatnonzero(green)
    if len(idx) < 2 or idx.max() - idx.min() < 3:
        return None
    if max_gap is not None and len(idx) > 1:
        # keep the widest run whose internal gaps stay under max_gap; a genuine
        # ribbon is the fill plus two edges, so its gaps are small.
        brk = np.flatnonzero(np.diff(idx) > max_gap)
        starts = np.concatenate(([0], brk + 1))
        ends = np.concatenate((brk, [len(idx) - 1]))
        k = int(np.argmax(idx[ends] - idx[starts]))
        idx = idx[starts[k]:ends[k] + 1]
        if len(idx) < 2 or idx.max() - idx.min() < 3:
            return None
    return float(idx.min()), float(idx.max())


def ridge_width_px(x, fx, paint_m=0.18):
    """Half-width, in px, of the ridge operator that MATCHES the paint at range x.

    ⛔ THIS IS WHY THE FAR FIELD KEPT COMING BACK EMPTY. ``lane_calib._ridge_points``
    ramps its offset ``w`` linearly with the row's position in the band it is given,
    which is calibrated for the full road band (0.50-0.95 H). Handed a NARROW far band
    it puts w = 2 at the top -- and a 0.18 m line at 40 m is 7.5 px wide, so
    ``2*row - row(-2) - row(+2)`` samples entirely INSIDE the paint and responds with
    almost nothing. MEASURED: association succeeded at 25 m in 23/150 frames and at
    40 m in 2/150, while the line is plainly visible in the frame.

    The operator's offset must be set by the PAINT, not by the row's rank in whatever
    band happened to be passed: w ~ the line's half-width + 2.
    """
    return int(max(2, round(0.5 * paint_m * fx / max(x, 1.0) + 2)))


def ridge_cols(row_vals, w, lo=None, hi=None, thr_min=14):
    """Columns of bright narrow ridges in one image row.

    ``lo``/``hi`` restrict BOTH the search and the threshold statistics to a column
    window.  A global 99th-percentile threshold is set by the brightest structure
    anywhere in the row -- near the horizon that is sky, signs and distant vehicles --
    and it buries a far lane line that is genuinely there.
    """
    row = row_vals.astype(np.int16)
    c = 2 * row - np.roll(row, w) - np.roll(row, -w)
    c[:w] = 0
    c[-w:] = 0
    if lo is not None:
        m = np.zeros_like(c)
        a_, b_ = max(0, int(lo)), min(len(c), int(hi))
        if b_ - a_ < 8:
            return np.empty(0)
        m[a_:b_] = c[a_:b_]
        seg = c[a_:b_]
        thr = max(thr_min, float(np.percentile(seg, 88)))
        c = m
    else:
        thr = max(28, float(np.percentile(c, 99.0)))
    xs = np.flatnonzero(c > thr)
    if len(xs) == 0:
        return xs
    # collapse runs to their centre so a 6 px wide line is one sample, not six
    brk = np.flatnonzero(np.diff(xs) > 2)
    groups = np.split(xs, brk + 1)
    return np.array([g.mean() for g in groups])


def track_line_up(gray, seed_row, seed_col, fx, horizon, fh, top_row, step=6):
    """Follow ONE painted line upward from a seed. No calibration enters the tracking.

    ⛔ WHY THE WINDOW-AROUND-A-PREDICTION VERSION HAD TO GO. It chose, at every range,
    the ridge NEAREST the predicted line position -- and the prediction moves with the
    calibration being tested. MEASURED, and this is what exposed it: ``out_caddy``
    (yaw -6.80) and ``out_yaw78`` (yaw -7.80) differ by exactly 1.0 deg, so geometry
    REQUIRES their residual slopes to differ by 0.0175 m/m. The probe reported a
    difference of 0.0038 -- it had absorbed 78 % of a known, imposed 1 deg. That is
    ``lane_residual.py``'s circularity in a new costume, and it would have certified
    whatever yaw it was handed.

    The tracker takes the prediction ONCE, at the near seed (10 m, where the window is
    150 px wide and features are far apart), and from there follows the paint by local
    continuity alone: extrapolate with the current slope, search a few px, update.
    A calibration error can no longer walk the measurement, because after the seed the
    calibration is not consulted again.
    """
    H, W = gray.shape
    col = float(seed_col)
    slope = 0.0                            # d(col)/d(row), learned as we climb
    pts = [(float(seed_row), col)]
    misses = 0
    v = float(seed_row)
    while v - step > top_row:
        v -= step
        x = fh / max(v - horizon, 1e-3)
        w = ridge_width_px(x, fx)
        pred = col + slope * (-step)
        half = max(5.0, abs(slope) * step + 5.0)
        lo, hi = pred - half, pred + half
        cols = ridge_cols(gray[int(round(v))], w, lo=lo - 4, hi=hi + 4, thr_min=10)
        cand = cols[(cols >= lo) & (cols <= hi)] if len(cols) else np.empty(0)
        if len(cand) == 0:
            misses += 1
            if misses >= 4:
                break
            col = pred                     # coast through a dashed gap
            continue
        misses = 0
        new = float(cand[np.argmin(np.abs(cand - pred))])
        s_obs = (new - col) / (-step)
        slope = 0.6 * slope + 0.4 * s_obs if len(pts) > 2 else s_obs
        col = new
        pts.append((v, col))
    if len(pts) < 6:
        return None
    P = np.asarray(pts)
    return P


def line_col_at(P, v):
    """Column of the tracked line at image row ``v`` (linear interp, no extrapolation)."""
    rows_, cols_ = P[::-1, 0], P[::-1, 1]
    if v < rows_[0] - 2 or v > rows_[-1] + 2:
        return None
    return float(np.interp(v, rows_, cols_))


def associate(src_gray, rows, edges, fx, gap_m=0.85, win_m=0.90, jump_m=0.35):
    """Distance (m) from the corridor's left edge to the left painted line, per range.

    ⚠️ This is an association window, the construct that made ``lane_residual.py``
    circular — so it is built to fail loudly instead of quietly:

    * the window is METRIC (``win_m`` at every range, so ``win_m*fx/x`` pixels), not a
      fixed pixel count; at 40 m it is 37 px while the adjacent lane's line is 146 px
      away, so it cannot reach the wrong line;
    * it is centred on the PREDICTION ``corridor_edge - 0.85 m`` (half of 3.5 m lane
      minus half of the 1.8 m ribbon), which is what "the corridor sits correctly in
      the lane" means — so a frame where that is false shows up as a large residual or
      as a dropped frame, never as a confirmation;
    * a WITHIN-FRAME COHERENCE check rejects the frame outright if the residual jumps
      more than ``jump_m`` between adjacent ranges. That is what stops the window from
      walking off the lane line onto the shoulder as range grows, which is exactly how
      the earlier probe manufactured a smooth profile out of three different features.
    """
    raise NotImplementedError("superseded by track_line_up; see the docstring there")


def measure(src_gray, rows, edges, fx, horizon, fh, seed_x, gap_m=0.85, seed_win_m=0.75,
            tol_px=3.0, min_inliers=22, min_span=90.0, road=None):
    """Corridor-left-edge minus painted-line distance, in metres, at each range.

    ONE straight line is fitted to real ridge points across the whole 10-50 m band by
    RANSAC, with a GLOBAL detection threshold so that "there is no paint here" is an
    answer the operator can actually give. The calibration enters exactly once, as an
    identity gate: the winning line must pass near the corridor's left edge at the
    NEAR range. A gate on the level cannot manufacture a slope, which is the quantity
    being measured -- and the imposed-1-deg check in ``--validate`` is what proves it.

    ⛔ Two earlier designs failed here and both are kept in this file as warnings:
    the per-range nearest-to-prediction window (absorbed 78 % of a known 1 deg), and
    ``track_line_up`` (a relative threshold inside a 10 px window ALWAYS returns a
    candidate, so it never reports a miss and random-walks; MEASURED: IQR out to
    +9.4 m and per-frame slopes of 0.47 m/m = 27 deg).
    """
    e0 = edges.get(seed_x)
    if e0 is None:
        return None, "no corridor at the seed range", None
    v0 = horizon + fh / seed_x
    pred0 = e0[0] / S - gap_m * fx / seed_x
    half0 = seed_win_m * fx / seed_x

    # ⛔ ``road`` is the ROAD MASK (R-2026-09-14-foliage). Without it the ridge operator
    # returns the brightest narrow features in the row, which on this recording is the
    # sunlit bank, not the paint. The identity gate below gave this probe partial
    # protection -- a candidate line has to pass within 0.75 m of a predicted position
    # near the corridor -- but "partial" is not a property to rely on, and a foliage
    # line that happens to satisfy the gate would be indistinguishable. The mask removes
    # the possibility rather than making it unlikely.
    band = [(x, v) for x, v in rows]
    pts = []

    def _cols(v, x):
        vi = int(round(v))
        if road is not None and (vi < 0 or vi >= road.shape[0] or road[vi].max() == 0):
            return ()
        out = ridge_cols(src_gray[vi], ridge_width_px(x, fx))
        if road is None:
            return out
        return [c for c in out
                if 0 <= int(c) < road.shape[1] and road[vi, int(c)]]

    for x, v in band:
        for c in _cols(v, x):
            pts.append((v, c))
    # a line needs support between the sampled ranges too, not just at them
    vmin, vmax = min(v for _, v in band), max(v for _, v in band)
    for v in np.arange(vmin, vmax, 4.0):
        x = fh / max(v - horizon, 1e-3)
        for c in _cols(v, x):
            pts.append((float(v), float(c)))
    if len(pts) < min_inliers:
        return None, "no paint detected in the band", None
    P = np.asarray(pts, float)

    rng = np.random.default_rng(0)
    best, best_n = None, 0
    n = len(P)
    for _ in range(900):
        i, j = rng.integers(0, n, 2)
        if abs(P[i, 0] - P[j, 0]) < min_span:
            continue
        m = (P[j, 1] - P[i, 1]) / (P[j, 0] - P[i, 0])
        b = P[i, 1] - m * P[i, 0]
        if abs(m * v0 + b - pred0) > half0:
            continue                       # not the ego lane's left boundary
        if not (0.2 <= abs(m) <= 4.0):
            continue                       # not road-parallel (m = y0/h physically)
        d = np.abs(P[:, 1] - (m * P[:, 0] + b))
        k = int((d < tol_px).sum())
        if k > best_n:
            best_n, best = k, (m, b)
    if best is None or best_n < min_inliers:
        return None, "no line passed the identity gate", None
    m, b = best
    for _ in range(3):
        d = np.abs(P[:, 1] - (m * P[:, 0] + b))
        inl = P[d < tol_px]
        if len(inl) < min_inliers or np.ptp(inl[:, 0]) < min_span:
            return None, "line support too short", None
        m, b = np.polyfit(inl[:, 0], inl[:, 1], 1)

    out, ev = [], []
    for x, v in rows:
        e = edges.get(x)
        if e is None:
            continue
        cu = e[0] / S
        out.append((x, float((cu - (m * v + b)) * x / fx)))
        ev.append((v, cu))
    if len(out) < 4:
        return None, "corridor too short", None

    # ⭐ THE CALIBRATION-FREE NUMBER. Everything above is a distance in metres, and a
    # metre depends on the horizon you read the row through -- which is why ``caddy``
    # measured 0.21 deg on its own grid and 1.22 deg on the common one, from the SAME
    # video. Algebra: residual(x) = (P + Q*v_m)*x/f_m + Q*f*h_m/f_m, so the SLOPE
    # carries the metric horizon ``v_m`` inside it. Comparing arms by drift in m/m is
    # therefore comparing interpretations as much as overlays.
    #
    # The row where the corridor's LEFT EDGE crosses the painted line is not an
    # interpretation. It is two lines in one image meeting at a pixel, and it is
    # exactly Sayed's question: where does the corridor leave the lane?
    E = np.asarray(ev, float)
    if len(E) >= 3:
        me, be = np.polyfit(E[:, 0], E[:, 1], 1)
        if abs(me - m) > 1e-6:
            v_cross = float((b - be) / (me - m))
            # (lane line, corridor line) so a caller can re-ask the question for a
            # DIFFERENT yaw without re-rendering: a yaw change of d shifts the drawn
            # corridor by f*d px at every row, i.e. it moves ``be`` and nothing else.
            return out, "ok", (v_cross, float(m), float(b), float(me), float(be))
    return out, "ok", None


def fit_left_line(src_gray, rows_src, anchor_row_src, anchor_col_src, tol_px=4.0):
    """RANSAC a straight line through ridge points, anchored near the corridor edge.

    Returns ``(slope, intercept)`` in source pixels with ``col = slope*row + b``,
    or ``None``.  Only points LEFT of the corridor centre are considered, and the
    winning line must pass within 40 px of ``anchor_col_src`` at the anchor row --
    that is what makes it the ego lane's left boundary and not the shoulder.
    """
    pts = []
    for v in rows_src:
        w = max(2, int(round(2 + 26 * (v - rows_src[0]) / max(1, rows_src[-1] - rows_src[0]))))
        cols = ridge_cols(src_gray[int(v)], w)
        for c in cols:
            pts.append((v, c))
    if len(pts) < 12:
        return None
    P = np.asarray(pts, float)
    best, best_n = None, 0
    rng = np.random.default_rng(0)
    n = len(P)
    for _ in range(1500):
        i, j = rng.integers(0, n, 2)
        if abs(P[i, 0] - P[j, 0]) < 40:
            continue
        m = (P[j, 1] - P[i, 1]) / (P[j, 0] - P[i, 0])
        b = P[i, 1] - m * P[i, 0]
        if abs(m * anchor_row_src + b - anchor_col_src) > 60:
            continue                      # not the line the corridor is following
        d = np.abs(P[:, 1] - (m * P[:, 0] + b))
        k = int((d < tol_px).sum())
        if k > best_n:
            best_n, best = k, (m, b)
    if best is None or best_n < 12:
        return None
    m, b = best
    for _ in range(3):                    # refit on inliers
        d = np.abs(P[:, 1] - (m * P[:, 0] + b))
        inl = P[d < tol_px]
        if len(inl) < 8:
            break
        m, b = np.polyfit(inl[:, 0], inl[:, 1], 1)
    return float(m), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames-dir", default="/root/trajdata/out_bev/2026-08-08_14-19-54-android/frames")
    ap.add_argument("--fh", type=float, required=True, help="f*h in px*m for this arm")
    ap.add_argument("--horizon", type=float, required=True, help="horizon row, SOURCE px")
    ap.add_argument("--fx", type=float, required=True, help="focal length, SOURCE px")
    ap.add_argument("--ranges", type=float, nargs="+",
                    default=[10., 12., 15., 20., 25., 30., 40.])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--label", default="")
    ap.add_argument("--mask", dest="mask", action="store_true", default=True,
                    help="restrict ridge detection to the projected road (default ON)")
    ap.add_argument("--no-mask", dest="mask", action="store_false",
                    help="the pre-R-2026-09-14-foliage behaviour, for comparison only")
    ap.add_argument("--y-left", type=float, default=4.2)
    ap.add_argument("--y-right", type=float, default=2.6)
    ap.add_argument("--yaw-base", type=float, default=0.0,
                    help="the yaw this video was RENDERED with, so the scan can name "
                         "absolute yaws instead of offsets")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    src = sorted(pathlib.Path(a.frames_dir).glob("*.jpg"))
    cap = cv2.VideoCapture(a.video)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def row_src(x):
        return a.horizon + a.fh / x

    rows = {x: row_src(x) for x in a.ranges}
    band = np.arange(int(round(row_src(max(a.ranges)))) - 4,
                     int(round(row_src(min(a.ranges)))) + 4, 2, dtype=float)

    # ⚠️ SELECTION CONTROL. The coherence gate drops frames, and the frames it drops
    # are not a random sample -- a corridor that is badly off the lane is exactly what
    # makes the association jump. Reporting only the surviving frames would be the
    # C6 confound again. So every frame is classified by the trajectory's own yaw rate
    # and the drop reason is counted, and the straight/curved split is printed.
    import run_real as RR_
    yawrate = {}
    for r in RR_.load_records(RR_.RUN):
        t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
        m = np.abs(t) < 0.4
        if m.sum() >= 3:
            yawrate[int(r["frame"])] = abs(float(np.rad2deg(np.polyfit(t[m], yw[m], 1)[0])))

    acc = {x: [] for x in a.ranges}
    slopes = []
    slopes_by_class = {"straight": [], "curved": []}
    crossings = []                 # (row, range_m, road class) where corridor meets paint
    lines_for_scan = []            # (lane m, lane b, corridor m, corridor b, class)
    drops = {}
    nfit = 0
    road_m = None
    rows_l = [(x, rows[x]) for x in sorted(a.ranges)][::-1]      # far -> near rows
    rows_l = sorted(rows_l, key=lambda p: p[1])                  # by image row, top first
    want = set(np.linspace(30, min(nfr, len(src)) - 30, a.n).astype(int).tolist())
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            g = cv2.imread(str(src[i]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                edges = {x: corridor_edges(fr, int(round(rows[x] * S))) for x in a.ranges}
                if road_m is None and a.mask:
                    from road_mask import calib as _calib, road_mask as _rm
                    _P = _calib(fx=a.fx, height=a.fh / a.fx, horizon=a.horizon,
                                yaw=a.yaw_base)
                    road_m = _rm(_P, g.shape, y_left=a.y_left, y_right=a.y_right,
                                 x_range=(8.0, max(a.ranges) + 5.0))
                    print(f"road mask: {100*float(road_m.mean())/255:.1f} % of the frame")
                got, why, v_cross = measure(g, rows_l, edges, a.fx, a.horizon, a.fh,
                                            min(a.ranges), road=road_m)
                if why != "ok":
                    drops[why] = drops.get(why, 0) + 1
                if got and len(got) >= 4:
                    xs = np.array([x for x, _ in got], float)
                    ds = np.array([d for _, d in got], float)
                    sl = float(np.polyfit(xs, ds, 1)[0])
                    fno = int(src[i].stem)
                    yr = yawrate.get(fno)
                    cls = None if yr is None else ("straight" if yr <= 0.6 else "curved")
                    if cls:
                        slopes_by_class[cls].append(sl)
                    if why == "ok":
                        nfit += 1
                        if v_cross is not None:
                            vc = v_cross[0]
                            rng_ = (a.fh / (vc - a.horizon)) if vc > a.horizon + 8 else float("inf")
                            crossings.append((float(vc), float(rng_), cls))
                            lines_for_scan.append((v_cross[1], v_cross[2],
                                                   v_cross[3], v_cross[4], cls))
                        for x, d in got:
                            acc[x].append((d, rows[x]))
                        slopes.append(sl)
        i += 1
        if i > max(want):
            break
    cap.release()

    print(f"\n{a.label or a.video}")
    print(f"  f*h {a.fh:.0f}   horizon {a.horizon:.1f}   fx {a.fx:.0f}   "
          f"coherent association in {nfit}/{len(want)} frames")
    print(f"  {'range':>7}{'corridor left edge - painted line':>36}{'n':>7}")
    out = {}
    prof = []
    for x in a.ranges:
        v = [d for d, _ in acc[x]]
        if len(v) < 15:
            print(f"  {x:5.0f} m{'--':>36}{len(v):7d}")
            continue
        med = float(np.median(v))
        q1, q3 = np.percentile(v, [25, 75])
        print(f"  {x:5.0f} m{med:+27.2f} m   [{q1:+.2f},{q3:+.2f}]{len(v):7d}")
        out[str(x)] = dict(median_m=med, iqr=[float(q1), float(q3)], n=len(v))
        prof.append((x, med))
    if len(prof) >= 3:
        xs = np.array([p[0] for p in prof], float)
        ds = np.array([p[1] for p in prof], float)
        s = float(np.polyfit(xs, ds, 1)[0])
        print(f"\n  drift {s:+.4f} m/m = {np.rad2deg(np.arctan(abs(s))):.2f} deg"
              f"   ->{40*s:+.2f} m by 40 m")
        near = [d for x_, d in prof if x_ <= 15]
        far = [d for x_, d in prof if x_ >= 25]
        if near and far:
            print(f"  near (<=15 m) median {np.median(near):+.2f} m   "
                  f"far (>=25 m) median {np.median(far):+.2f} m   "
                  f"change {np.median(far)-np.median(near):+.2f} m")
        out["_profile_m"] = [[float(x), float(d)] for x, d in prof]
        out["_slope_m_per_m"] = s
    if len(slopes) >= 20:
        S_ = np.asarray(slopes)
        med = float(np.median(S_))
        lo, hi = np.percentile(S_, [2.5, 97.5])
        print(f"\n  PER-FRAME drift (the frame-local test, immune to where the car "
              f"sits in the lane)")
        print(f"    median {med:+.4f} m/m = {np.rad2deg(np.arctan(abs(med))):.2f} deg"
              f"   ->{40*med:+.2f} m by 40 m   2.5-97.5% [{lo:+.4f}, {hi:+.4f}]"
              f"   n {len(S_)}")
        out["per_frame_slope"] = dict(median=med, lo=float(lo), hi=float(hi), n=len(S_))

    if len(crossings) >= 20:
        C = np.asarray([c[1] for c in crossings], float)
        R = np.asarray([c[0] for c in crossings], float)
        finite = np.isfinite(C)
        print(f"\n  ⭐ WHERE THE CORRIDOR'S LEFT EDGE MEETS THE PAINTED LINE")
        print(f"     (two lines in one image crossing at a pixel — this one does NOT")
        print(f"      depend on which horizon you read the rows through)")
        print(f"     crossing row   median {np.median(R):6.1f}"
              f"   10-90% [{np.percentile(R,10):.0f}, {np.percentile(R,90):.0f}]"
              f"   horizon is {a.horizon:.0f}")
        above = int((R <= a.horizon + 8).sum())
        print(f"     crosses ABOVE the horizon (i.e. never, on the visible road): "
              f"{above}/{len(R)} frames = {100*above/len(R):.0f} %")
        if finite.sum() >= 10:
            cf = C[finite]
            print(f"     when it does cross: median {np.median(cf):6.1f} m"
                  f"   25-75% [{np.percentile(cf,25):.0f}, {np.percentile(cf,75):.0f}] m"
                  f"   10th pct {np.percentile(cf,10):.0f} m   n {len(cf)}")
        for cls in ("straight", "curved"):
            sel = [c for c in crossings if c[2] == cls]
            if len(sel) >= 10:
                rr = np.asarray([c[0] for c in sel]); ab = int((rr <= a.horizon + 8).sum())
                cc = np.asarray([c[1] for c in sel]); cc = cc[np.isfinite(cc)]
                print(f"     {cls:9s} n {len(sel):4d}   never-crosses {100*ab/len(sel):3.0f} %"
                      + (f"   else median {np.median(cc):.0f} m" if len(cc) else ""))
        out["crossing"] = dict(row_median=float(np.median(R)),
                               never_frac=float(above / len(R)),
                               range_median=float(np.median(C[finite])) if finite.sum() else None,
                               n=len(R))

    if len(lines_for_scan) >= 20:
        # ⭐ DECOMPOSITION. The lane line's column AT THE HORIZON ROW is cx + f*tan(theta),
        # where theta is the angle between the CAMERA AXIS and the lane. On straight road
        # the vehicle is parallel to the lane, so that angle IS the camera's mount yaw --
        # measured from the paint, with no trajectory anywhere in it.
        #
        # The yaw scan above instead finds the yaw that makes the DRAWN CORRIDOR parallel
        # to the lane, and the corridor comes from the trajectory. The difference between
        # the two is therefore the trajectory's systematic HEADING BIAS, which is a
        # different defect with a different fix.
        st = [(lm, lb) for lm, lb, _, _, cls in lines_for_scan if cls == "straight"]
        if len(st) >= 15:
            uh = np.array([lm * a.horizon + lb for lm, lb in st])
            yaw_cam = np.rad2deg(np.arctan((uh - 960.0) / a.fx))
            med = float(np.median(yaw_cam))
            sd = 1.4826 * float(np.median(np.abs(yaw_cam - med)))
            print(f"\n  ⭐ CAMERA YAW FROM THE PAINT ALONE (no trajectory, no ego motion)")
            print(f"     the left lane line's column at the horizon row is cx + f*tan(yaw)")
            print(f"     yaw {med:+.2f} deg   robust sd {sd:.2f} deg   n {len(st)} straight frames")
            print(f"     10-90% [{np.percentile(yaw_cam,10):+.2f}, "
                  f"{np.percentile(yaw_cam,90):+.2f}] deg")
            out["yaw_camera_from_paint"] = dict(median=med, sd=sd, n=len(st))

    if len(lines_for_scan) >= 20:
        print(f"\n  ⭐ YAW SCAN WITHOUT RE-RENDERING")
        print(f"     A yaw change d displaces the drawn corridor by f*d px at EVERY row,")
        print(f"     so its image line moves in intercept only. The lane lines are already")
        print(f"     measured, so every candidate yaw can be scored on the frames in hand.")
        print(f"     Target: the two lines should meet AT the horizon ({a.horizon:.0f}) —")
        print(f"     that is what 'the corridor is parallel to the lane' means.")
        print(f"\n     {'yaw':>8}{'median crossing row':>22}{'never crosses':>16}"
              f"{'straight: never':>18}")
        base = a.yaw_base
        best = None
        for d in np.arange(-1.5, 3.01, 0.25):
            vcs, vcs_st = [], []
            for lm, lb, cm, cb, cls in lines_for_scan:
                cb2 = cb + a.fx * np.deg2rad(d)
                if abs(cm - lm) < 1e-6:
                    continue
                vc = (lb - cb2) / (cm - lm)
                vcs.append(vc)
                if cls == "straight":
                    vcs_st.append(vc)
            if len(vcs) < 15:
                continue
            V_ = np.asarray(vcs); Vs = np.asarray(vcs_st) if vcs_st else V_
            never = float((V_ <= a.horizon + 8).mean())
            never_st = float((Vs <= a.horizon + 8).mean())
            score = abs(np.median(V_) - a.horizon)
            if best is None or score < best[0]:
                best = (score, d, float(np.median(V_)), never, never_st)
            if abs(d * 4 - round(d * 4)) < 1e-9 and abs(round(d * 2) - d * 2) < 1e-9:
                print(f"     {base + d:8.2f}{np.median(V_):22.1f}{100*never:15.0f}%"
                      f"{100*never_st:17.0f}%")
        if best:
            print(f"\n     BEST yaw {base + best[1]:+.2f} deg "
                  f"({best[1]:+.2f} from the rendered {base:+.2f}):"
                  f"  median crossing row {best[2]:.1f} vs horizon {a.horizon:.0f},"
                  f"  never-crosses {100*best[3]:.0f} % (straight {100*best[4]:.0f} %)")
            out["yaw_scan_best"] = dict(yaw=float(base + best[1]), delta=float(best[1]),
                                        row=best[2], never=best[3], never_straight=best[4])

    print(f"\n  SELECTION CONTROL — frames dropped: "
          + ", ".join(f"{k} {v}" for k, v in drops.items() if v))
    print(f"  drift by road class, gate OFF (so the drops cannot flatter the result):")
    for cls in ("straight", "curved"):
        v = np.asarray(slopes_by_class[cls])
        if len(v) >= 10:
            print(f"    {cls:9s} n {len(v):4d}   median {np.median(v):+.4f} m/m"
                  f" = {np.rad2deg(np.arctan(abs(np.median(v)))):.2f} deg"
                  f"   |slope| median {np.median(np.abs(v)):.4f}"
                  f"   90th pct |slope| {np.percentile(np.abs(v), 90):.4f}"
                  f" ({40*np.percentile(np.abs(v), 90):.2f} m at 40 m)")
        else:
            print(f"    {cls:9s} n {len(v):4d}   -- too few")
    out["drops"] = drops
    out["by_class"] = {k: dict(n=len(v), median=float(np.median(v)) if v else None,
                               abs_median=float(np.median(np.abs(v))) if v else None,
                               abs_p90=float(np.percentile(np.abs(v), 90)) if v else None)
                       for k, v in slopes_by_class.items()}
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
