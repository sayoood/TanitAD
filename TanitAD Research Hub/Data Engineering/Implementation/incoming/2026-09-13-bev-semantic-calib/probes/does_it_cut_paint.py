#!/usr/bin/env python3
"""Does the drawn corridor COVER painted markings — and is it where the car is?

Sayed: *"are you satisfied with results, i see the trajectory cutting road markings"*.
No. Every number I have quoted is a MEDIAN over frames, and a correct median is
perfectly consistent with a substantial minority of frames being visibly wrong. The
symptom has never been measured directly with a working instrument — the one that
tried (`/tmp/overlap.py`) was wrong three ways and is retracted in
`R-2026-09-14-onvideo`.

THIS MEASURES THE SYMPTOM. For each frame, at each range:

  * the corridor's two edges, read from the composite (the opaque (200,255,200)
    edge polylines, which stay green at every range even where the fill goes amber);
  * the two painted lane boundaries, fitted in the ORIGINAL source frame — no green
    tint, no video recompression;
  * then three things that are facts about the image, not interpretations:

        COVERAGE   is any paint column strictly inside the corridor's edges?
        PLACEMENT  (corridor centre − lane centre), in metres
        WIDTHS     the corridor's drawn width and the lane's, in the same units

PLACEMENT is the one that decides whether this is a defect or the truth. The corridor
is drawn around the CAR's path, so if the car sits off centre the corridor must too.
But the camera's own offset from lane centre is separately measurable from the paint
alone — `(w/2)·(m_L + m_R)/|m_L − m_R|`, in which the height cancels — so the two can
be compared, and any difference is the drawing being in the wrong place.

⚠️ Read COVERAGE against the geometry before calling it an error: a 1.855 m ribbon in
a 3.3 m lane, offset 0.4 m, has ~0.12 m of clearance, and a marking is 0.15-0.20 m
wide. Touching is then the CORRECT picture, and the question is only whether the
placement is right.

═══ THREE THINGS FIXED HERE, ALL FOUND BY READING THE CODE RATHER THAN ITS OUTPUT ═══

**1. THE ROAD MASK (`R-2026-09-14-foliage`).** This probe needs the RIGHT lane
boundary, and the right of this carriageway is a sunlit bank that `ridge_cols` prefers
to paint. Unmasked, this probe was measuring shrubbery on the very side its answer
depends on. `road_mask` now gates every ridge column. Nothing this file printed before
that mask is admissible.

**2. ⛔ THE PLACEMENT SIGN WAS WRONG — the two quantities it subtracted point in
OPPOSITE directions.** `place` was `(u_corridor − u_lane)·x/f`, an IMAGE-column
difference, positive to the RIGHT. `cam_off` is `h·(m_L + m_R)/2`, a VEHICLE-frame
lateral, and this calibration's ``u = cx − f(y − lat)/x`` makes y positive to the
LEFT. The old line ``pl - veh`` therefore added two errors instead of cancelling them
and would have reported ~2× the true defect (or ~0 when the defect was real). Both are
now in the vehicle frame, positive LEFT, and the check is a difference of like things.

**3. THE PARALLELISM GATE SILENTLY ASSUMED THE HORIZON.** It compared the implied
width at 12 m and 30 m, and those ranges come from the assumed ``v_h`` — so it was
really testing *"do these two lines meet at row 448.4?"* with a tolerance of about
11 px. The lane boundaries on this recording meet at **467.7** (§77), so the gate was
rejecting correct pairs for disagreeing with an assumption. It is replaced by the
horizon-FREE pair of facts that two road-parallel lines actually have:

    lane width   w   = h · |m_R − m_L|                     (h only, no v_h, no f)
    they meet at v_0 = (b_L − b_R) / (m_R − m_L)           (a MEASUREMENT of v_h)

and ``v_0`` is now reported rather than assumed — a second, free result, on the one
calibration quantity still unresolved (row-flow 438 / lens ≲450 / boundaries 467.7).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bev_calib as BC, lag_scale as LS, run_real as RR                # noqa: E402
from overlay_far import corridor_edges, ridge_cols, ridge_width_px, S  # noqa: E402
from noise_floor import ransac_line                                    # noqa: E402
from road_mask import road_mask                                        # noqa: E402


def fit_pair(cols_at, band, horizon, fh, v_ref, u_ref, height,
             w_lo=2.6, w_hi=4.6, v0_lo=380.0, v0_hi=540.0):
    """The two lane boundaries in one frame, or None.

    ⚠️ ONE implementation, used by both the measurement and the preview. The
    foliage retraction happened because two detectors in this directory had
    drifted apart and neither was ever drawn on an image; a preview that does not
    run the SAME code as the measurement is not a check of the measurement.
    """
    pts = []
    for v in band:
        x = fh / max(v - horizon, 1e-3)
        for c in cols_at(v, x):
            pts.append((float(v), float(c)))
    if len(pts) < 50:
        return None
    P = np.asarray(pts, float)
    lines, rem = [], P
    for k in range(5):
        ln, inl = ransac_line(rem, seed=k + 1)
        if ln is None:
            break
        lines.append((ln, int(inl.sum())))
        rem = rem[~inl]
        if len(rem) < 30:
            break
    L = [(ln, n) for ln, n in lines if ln[0] * v_ref + ln[1] < u_ref]
    R = [(ln, n) for ln, n in lines if ln[0] * v_ref + ln[1] >= u_ref]
    if not (L and R):
        return None
    (ml, bl), nl = max(L, key=lambda z: z[1])
    (mr, br), nr = max(R, key=lambda z: z[1])
    # ⛔ THE SLOPE GATE ALONE IS NOT ENOUGH. A pair can pass |m_L-m_R| in the
    # single-lane band and still not be two parallel boundaries. MEASURED without
    # a second check: the "lane" read 6.41 m at 10 m growing to 18.28 m at 40 m.
    # The right check is HORIZON-FREE (see the header): two road-parallel lines
    # have a constant separation in units of h -- w = h*dm -- and they MEET
    # somewhere, and that meeting row is the horizon rather than a thing to
    # assume. Gate on the width and on v_0 being anywhere plausible; report v_0.
    if abs(mr - ml) < 1e-6:
        return None
    v0 = (bl - br) / (mr - ml)
    wlane = abs(ml - mr) * height
    if not (w_lo < wlane < w_hi and v0_lo < v0 < v0_hi):
        return None
    return dict(ml=ml, bl=bl, mr=mr, br=br, nl=nl, nr=nr, v0=v0, w=wlane, pts=P)


def _cols_factory(g, road, fx, paint_k=3.0):
    """Ridge columns that are inside the road mask AND are actually PAINT.

    ⛔ THE MASK WAS NOT ENOUGH, AND THE PREVIEW IS WHAT SHOWED IT. With the road
    mask on, the right-hand "lane boundary" this probe fitted was a TAR SEAM
    running up the middle of the ego lane — drawn on the image it is unmistakable,
    and in 183 frames of statistics it was invisible. `R-2026-09-14-foliage` one
    layer down: mask out the bank and the detector moves to the next-brightest
    non-paint structure it can find.

    THE MECHANISM, and it is not bad luck. RANSAC scores a line by INLIER COUNT.
    The left boundary here is SOLID and the right boundary is DASHED, so a
    continuous seam offers more inliers over the same rows than the real right
    line does. The detector was not confused — it was answering the question it
    was asked. That is why `clear L` sat flat to 0.02 m while `clear R` drifted
    0.58 m: only the right-hand fit was wrong.

    THE DISCRIMINATOR IS BRIGHTNESS, NOT SHAPE. `ridge_cols` responds to any
    narrow feature brighter than its immediate neighbours, and a pale seam clears
    that bar easily. Paint does not merely beat its neighbours, it is far brighter
    than the ROAD: gate on the row's own robust statistics, measured inside the
    mask, so the threshold follows sun, shadow and exposure instead of being a
    constant that is wrong twice a minute.

    ⛔ AND THE THRESHOLD WAS BEING SET BY PIXELS THAT ARE THEN DISCARDED. The
    masked call was ``ridge_cols(row, w)`` with no window, so its threshold is the
    99th percentile of the WHOLE row — on this recording the sunlit bank and the
    concrete barrier, both OUTSIDE the mask. Structures we throw away were setting
    the bar the paint had to clear, and the mask was applied afterwards, so
    masking made the starvation worse rather than better. MEASURED: with the mask
    on, three of four preview frames found no lane pair at all while the left
    boundary is plainly bright paint. ``ridge_cols`` has taken ``lo``/``hi`` for
    exactly this since it was written — nothing was passing them. Same family as
    `df` on a pod, `free` on Thor and cgroup `usage_in_bytes`: a statistic
    aggregated over the wrong scope, read as an answer.
    """
    def cols_at(v, x):
        vi = int(round(v))
        if not (0 <= vi < g.shape[0]):
            return ()
        if road is None:
            return ridge_cols(g[vi], ridge_width_px(x, fx))
        on_cols = np.flatnonzero(road[vi])
        if on_cols.size < 24:
            return ()
        lo, hi = int(on_cols[0]), int(on_cols[-1]) + 1
        out = ridge_cols(g[vi], ridge_width_px(x, fx), lo=lo, hi=hi)
        keep = [c for c in out if lo <= int(c) < hi and road[vi, int(c)]]
        if not keep or paint_k <= 0:
            return keep
        row = g[vi]
        on = row[on_cols].astype(np.float32)
        lvl = float(np.median(on))
        sig = max(1.4826 * float(np.median(np.abs(on - lvl))), 3.0)
        return [c for c in keep if float(row[int(c)]) > lvl + paint_k * sig]
    return cols_at


def preview(a, out_path, n=4):
    """Draw the fitted PAIR, the mask and the corridor on real frames — and LOOK.

    `R-2026-09-14-foliage` is the rule that the detector's INPUT must be drawn
    before the first statistic. This extends it to the detector's OUTPUT: the
    masked run still produced a right-hand line whose clearance drifted 0.58 m
    while the left one sat flat to 0.02 m, and no histogram can tell you whether
    that line is on the paint. A picture can.
    """
    P0 = dict(RR.NOMINAL)
    P0.update(fx=a.fx, height=a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P0["pitch"] = LS.pitch_for_horizon(P0, a.horizon)
    fh = a.fx * a.height
    band = np.arange(a.horizon + fh / max(a.ranges), a.horizon + fh / min(a.ranges), 1.0)
    v_ref = a.horizon + fh / 12.0
    u_ref = float(BC.project_ground(np.array([[12.0, 0.0]]), P0)[0][0])

    src = sorted(pathlib.Path(a.frames_dir).glob("*.jpg"))
    cap = cv2.VideoCapture(a.video)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sorted(set(np.linspace(300, min(nfr, len(src)) - 300, n).astype(int).tolist()))
    road, out, i = None, [], 0
    while True:
        ok, fr = cap.read()
        if not ok or i > max(want):
            break
        if i in want:
            g = cv2.imread(str(src[i]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                if road is None and a.mask:
                    road = road_mask(P0, g.shape, y_left=a.y_left, y_right=a.y_right,
                                     x_range=(8.0, max(a.ranges) + 5.0))
                cols_at = _cols_factory(g, road, a.fx, a.paint_k)
                r = fit_pair(cols_at, band, a.horizon, fh, v_ref, u_ref, a.height)
                im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                if road is not None:
                    im[road > 0] = (0.75 * im[road > 0] + np.array([60, 0, 0])).astype(np.uint8)
                for v in band[::2]:
                    x = fh / max(v - a.horizon, 1e-3)
                    for c in cols_at(v, x):
                        cv2.circle(im, (int(c), int(round(v))), 2, (0, 230, 230), -1)
                txt = f"frame {i}: NO PAIR"
                if r is not None:
                    for m, b, col in ((r["ml"], r["bl"], (255, 80, 80)),
                                      (r["mr"], r["br"], (80, 80, 255))):
                        p0 = (int(m * band[0] + b), int(band[0]))
                        p1 = (int(m * band[-1] + b), int(band[-1]))
                        cv2.line(im, p0, p1, col, 2)
                    txt = (f"frame {i}: lane {r['w']:.2f} m  v0 {r['v0']:.0f}  "
                           f"support {r['nl']}L/{r['nr']}R")
                for x in (10.0, 20.0, 30.0):                 # corridor edges, green
                    v = a.horizon + fh / x
                    e = corridor_edges(fr, int(round(v * S)))
                    if e is not None:
                        cv2.line(im, (int(e[0] / S), int(round(v))),
                                 (int(e[1] / S), int(round(v))), (0, 255, 0), 2)
                # A LATERAL RULER. Guessing a mask width from arithmetic is how the
                # right edge ended up on the gravel apron: read the paint's offset
                # off the picture instead. Each line is a constant vehicle-frame y.
                for y_m in a.ruler:
                    xs = np.linspace(9.0, max(a.ranges), 60)
                    uv = BC.project_ground(np.stack([xs, np.full_like(xs, y_m)], 1), P0)
                    uv = uv[np.isfinite(uv).all(1)]
                    if len(uv) >= 2:
                        cv2.polylines(im, [np.round(uv).astype(np.int32)], False,
                                      (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(im, f"{y_m:+.1f}", (int(uv[-1, 0]) - 18,
                                    int(uv[-1, 1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 255, 255), 2)
                cv2.putText(im, txt + "   BLUE=left fit  RED=right fit  GREEN=corridor",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3)
                out.append(im[440:900])
        i += 1
    cap.release()
    if not out:
        print("no preview frames")
        return 1
    img = np.vstack(out)
    cv2.imwrite(str(out_path),
                cv2.resize(img, (1500, int(img.shape[0] * 1500 / img.shape[1]))))
    print(f"wrote {out_path}")
    return 0


def histogram(a):
    """Where is the paint, in metres, without fitting anything.

    ⛔ WHY THIS EXISTS. Three detectors in this directory have now landed on
    something that is not paint — the bank (`R-2026-09-14-foliage`), a tar seam,
    and the gravel apron at the foot of the barrier. Every one of them was a
    *selection* step choosing a winner: RANSAC by inlier count, a Hough peak, a
    seeded association. A selection step cannot report that it chose badly.

    A HISTOGRAM CAN. Convert each detection to a vehicle-frame lateral and count.
    The lane boundaries, the seam and the gravel all appear as separate peaks,
    labelled in metres, and the reader sees the structure instead of a winner. It
    is also free of every fragile part of the pipeline: no line fit, no pair
    constraint, no parallelism gate, no horizon assumption beyond the one already
    in the projection.

    The map from column to lateral is EXACT, not linearised: at a fixed range the
    ground projection is affine in y, so two probe points per row invert it.
    """
    P0 = dict(RR.NOMINAL)
    P0.update(fx=a.fx, height=a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P0["pitch"] = LS.pitch_for_horizon(P0, a.horizon)
    fh = a.fx * a.height
    band = np.arange(a.horizon + fh / max(a.ranges), a.horizon + fh / min(a.ranges), 1.0)

    # column -> lateral, per row, from the real projection (yaw included)
    # ⛔ THE +LON IS NOT COSMETIC. ``project_ground``'s x is measured from the
    # TRAJECTORY ORIGIN (the rear axle); the camera sits ``longitudinal`` = 2.1 m
    # forward of it. But ``x = f·h/(v - v_h)`` is the range from the CAMERA. Feeding
    # one into the other asks for the scale at ``x - 2.1`` and applies it at ``x``,
    # so every lateral comes out too small by ``(1 - 2.1/x)`` -- 21 % at 10 m, 7 % at
    # 30 m. RANGE-DEPENDENT, so it also manufactures a drift that looks exactly like
    # a horizon error. See `R-2026-09-15-longitudinal`.
    LON = float(P0.get("longitudinal", 0.0))
    y_of = {}
    for v in band:
        x = fh / max(v - a.horizon, 1e-3) + LON
        uv = BC.project_ground(np.array([[x, 0.0], [x, 1.0]]), P0)
        if np.isfinite(uv).all() and abs(uv[1, 0] - uv[0, 0]) > 1e-6:
            y_of[int(round(v))] = (float(uv[0, 0]), float(uv[1, 0] - uv[0, 0]))

    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = pathlib.Path(a.frames_dir)
    avail = [f for f in sorted(int(p.stem) for p in fdir.glob("*.jpg"))
             if (r := recs.get(f)) and r["complete"] and r["speed_ms"] > 8]
    pick = [avail[i] for i in np.linspace(0, len(avail) - 1,
                                          min(a.n, len(avail))).astype(int)]
    road, Y, per_frame, raw = None, [], [], []
    for f in pick:
        g = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if road is None:
            road = road_mask(P0, g.shape, y_left=a.y_left, y_right=a.y_right,
                             x_range=(8.0, max(a.ranges) + 5.0)) if a.mask else None
        cols_at = _cols_factory(g, road, a.fx, a.paint_k)
        ys = []
        for v in band:
            vi = int(round(v))
            if vi not in y_of:
                continue
            u0, du = y_of[vi]
            x = fh / max(v - a.horizon, 1e-3)
            for c in cols_at(v, x):
                ys.append((float(c) - u0) / du)
                raw.append((float(v), float(c)))
        Y.extend(ys)
        per_frame.append(ys)
    Y = np.asarray(Y, float)
    print(f"\n{a.label or 'paint histogram'}")
    print(f"{len(pick)} frames · mask {'ON' if a.mask else 'OFF'} "
          f"(+{a.y_left}/-{a.y_right} m) · paint-k {a.paint_k} · {len(Y)} detections\n")
    if len(Y) < 500:
        print("⛔ too few detections")
        return 1
    lo, hi, step = -a.y_right - 0.2, a.y_left + 0.2, 0.10
    edges = np.arange(lo, hi + step, step)
    hist, _ = np.histogram(Y, bins=edges)
    top = max(hist.max(), 1)
    print("   lateral (m, + = LEFT of the camera)      detections")
    for c, e0 in zip(hist, edges[:-1]):
        if c:
            print(f"   {e0:+5.2f} .. {e0+step:+5.2f}  {c:6d}  "
                  f"{'#' * int(round(56 * c / top))}")
    # peaks: local maxima at least 0.5 m apart carrying >= 4 % of the mode
    order = np.argsort(hist)[::-1]
    peaks = []
    for i in order:
        if hist[i] < 0.04 * top:
            break
        yc = edges[i] + step / 2
        if all(abs(yc - p) > 0.5 for p, _ in peaks):
            peaks.append((yc, int(hist[i])))
    peaks.sort(key=lambda z: -z[0])
    print(f"\n  peaks (>=0.5 m apart, >=4 % of the mode), left to right:")
    for yc, c in peaks:
        print(f"    y = {yc:+.2f} m   n {c}")
    out = dict(n_frames=len(pick), n_det=int(len(Y)), mask=a.mask,
               y_left=a.y_left, y_right=a.y_right, paint_k=a.paint_k,
               height=a.height, peaks=[[float(y), int(c)] for y, c in peaks],
               hist=[[float(e), int(c)] for e, c in zip(edges[:-1], hist)])
    if len(peaks) >= 2:
        yl, yr = peaks[0][0], peaks[-1][0]
        print(f"\n  ⭐ outermost pair {yl:+.2f} .. {yr:+.2f}  ->  separation "
              f"{yl - yr:.2f} m at h = {a.height} m")
        print(f"     camera sits {0.5*(yl+yr):+.2f} m from that pair's centre "
              f"(+ = camera LEFT of it)")
        print(f"  ⚠️  READ THE HISTOGRAM BEFORE THE PEAK LIST. A peak is only a lane")
        print(f"     boundary if the picture says so; the gravel apron and an asphalt")
        print(f"     seam both make peaks, and both have already fooled a fit here.")
        out["separation_m"] = float(yl - yr)
        out["camera_from_pair_centre_m"] = float(0.5 * (yl + yr))
    # ── the same detections, RE-REFERENCED TO THE LEFT LINE IN THEIR OWN FRAME ──
    #
    # ⭐ THE POOLED HISTOGRAM ABOVE CANNOT RESOLVE THE RIGHT-HAND LINE, and the
    # reason is not the detector. The car wanders inside its lane by several
    # tenths of a metre, so every feature is convolved with that wander and a
    # 0.15 m line becomes a 0.6 m hump. On the left that still resolves, because
    # the boundary is solid, bright and has nothing beside it. On the right the
    # smeared line lands on the shoulder of the gravel apron's much larger peak
    # and stops being a local maximum at all — so a peak-picker returns the
    # gravel, exactly as the RANSAC returned the seam.
    #
    # The wander is COMMON MODE: it moves the whole scene together. Measure every
    # detection against the left line IN THE SAME FRAME and it cancels. What
    # survives is rigid — the lane width — and what does not is the shoulder,
    # whose distance from the paint genuinely varies. The instrument therefore
    # SHARPENS what it is looking for and BLURS what has been fooling it.
    lane_ref = a.y_left - 0.6
    rel = []
    for ys in per_frame:
        ys = np.asarray(ys, float)
        if len(ys) < 25:
            continue
        cand = ys[ys > 0.6]
        if len(cand) < 12:
            continue
        hb = np.arange(0.6, a.y_left + 0.2, 0.10)
        hh, _ = np.histogram(cand, bins=hb)
        if hh.max() < 8:
            continue
        i = int(np.argmax(hh))
        sel = cand[np.abs(cand - (hb[i] + 0.05)) < 0.30]
        if len(sel) < 8:
            continue
        rel.extend((ys - float(np.median(sel))).tolist())
    rel = np.asarray(rel, float)
    print(f"\n  ═══ RE-REFERENCED TO THE LEFT LINE (per frame) — {len(rel)} detections ═══")
    if len(rel) < 500:
        print("  too few")
    else:
        e2 = np.arange(-5.0, 1.0001, 0.10)
        h2, _ = np.histogram(rel, bins=e2)
        t2 = max(h2.max(), 1)
        print("   metres RIGHT of the left line             detections")
        for c, e0 in zip(h2, e2[:-1]):
            if c >= 0.02 * t2:
                print(f"   {-e0-0.10:+5.2f} .. {-e0:+5.2f}  {c:6d}  "
                      f"{'#' * int(round(56 * c / t2))}")
        # strongest local maximum in the admissible lane band, right of the line
        band2 = (e2[:-1] + 0.05)
        ok = (band2 > -4.6) & (band2 < -2.4)
        if ok.sum():
            j = int(np.argmax(np.where(ok, h2, -1)))
            wbin = -band2[j]
            sel = rel[np.abs(rel - band2[j]) < 0.35]
            w = -float(np.median(sel)) if len(sel) >= 30 else wbin
            out["lane_width_rel_m"] = w
            out["lane_width_rel_n"] = int(len(sel))
            print(f"\n  ⭐ LANE WIDTH (left line -> the strongest peak 2.4-4.6 m right"
                  f" of it)\n     = {w:.3f} m at h = {a.height} m, n {len(sel)}"
                  f"   [scales linearly with h]")
            print(f"     h that makes this lane {a.lane_m} m wide:"
                  f"  {a.height * a.lane_m / w:.3f} m")
            if "peaks" in out and peaks:
                yl = peaks[0][0]
                print(f"     camera vs that lane's centre: "
                      f"{(1.747 - w / 2.0) * -1:+.2f} m using the pooled left-line "
                      f"centroid +1.75 (+ = camera LEFT of centre)")
    # ── IS IT PAINT? THE DASHED LINE BLINKS; THE GRAVEL DOES NOT ─────────────
    #
    # ⭐ THE RIGHT-HAND BOUNDARY IS DASHED AND THE GRAVEL APRON IS CONTINUOUS,
    # and that is a property no amount of brightness gating or masking can see.
    # In the pooled histogram the dashed line has no local maximum at all — it
    # rides the flank of the gravel's much larger peak — so every estimator that
    # picks a peak returns the gravel. But watch ONE lateral bin across frames:
    # a dashed line is BURSTY (a dash enters the range window, then a gap), while
    # a shoulder is STEADY. Occupancy and the Fano factor separate them, and they
    # are cheap.
    #
    #   occupancy  = fraction of frames in which this bin has any detection
    #   Fano       = var/mean of the per-frame count; ~1 for a steady source,
    #                >1 for a bursty one
    #
    # ⚠️ Read this WITH the width, not instead of it. Paint is narrow AND
    # (if dashed) bursty; the gravel is broad AND steady; a seam is narrow and
    # steady. It takes both axes to name a feature, which is exactly why one
    # number has been wrong here three times.
    if per_frame:
        eb = np.arange(-a.y_right - 0.2, a.y_left + 0.2 + 1e-9, 0.10)
        M = np.zeros((len(per_frame), len(eb) - 1), float)
        for i, ys in enumerate(per_frame):
            if ys:
                M[i], _ = np.histogram(np.asarray(ys, float), bins=eb)
        mean = M.mean(0)
        occ = (M > 0).mean(0)
        fano = np.divide(M.var(0), np.maximum(mean, 1e-9))
        print(f"\n  ═══ IS IT PAINT? occupancy and burstiness per lateral bin ═══")
        print(f"  {'lateral':>14}{'mean/frame':>12}{'occupancy':>11}{'Fano':>8}   verdict")
        for j, e0 in enumerate(eb[:-1]):
            if mean[j] < 0.30:
                continue
            yc = e0 + 0.05
            v = ("gravel/verge (steady, broad)" if occ[j] > 0.90 and fano[j] < 2.0
                 else "BURSTY -> dashed paint" if fano[j] >= 2.0 and occ[j] < 0.92
                 else "steady, narrow -> solid paint or a seam")
            print(f"  {yc:+8.2f} m   {mean[j]:11.2f}{100*occ[j]:10.0f}%{fano[j]:8.2f}   {v}")
        out["bin_profile"] = [[float(eb[j] + 0.05), float(mean[j]), float(occ[j]),
                               float(fano[j])] for j in range(len(eb) - 1)
                              if mean[j] >= 0.30]

    # ── THE HORIZON, FROM THE ONE FEATURE WE KNOW IS REAL ────────────────────
    #
    # ⭐ A LANE BOUNDARY IS AT THE SAME LATERAL AT 10 m AND AT 30 m. That is not
    # an assumption about this road, it is what "road-parallel" means. And the
    # lateral we recover from a column depends on the range, which depends on the
    # horizon: ``x = f·h/(v − v_h)``. So a wrong ``v_h`` makes one physical line
    # appear at DIFFERENT laterals at different ranges, and the ``v_h`` that
    # removes that drift is the horizon — measured, from the strong solid left
    # line alone, with no pair, no second line, no vanishing-point fit and no
    # lens argument.
    #
    # ⚠️ This is scale-free in ``f·h``: multiplying ``f·h`` scales every lateral
    # by the same factor at every range, so it cannot be recovered here and is
    # not claimed. What IS recovered is the horizon, which the row-flow (438),
    # the lens bound (<= ~450) and the boundary-convergence row (466) disagree on
    # by 28 px — a 6 % range error at 30 m and the reason a "3.5 m lane" and a
    # "2.75 m lane" can both be read off the same recording.
    if raw:
        RV = np.array([r[0] for r in raw], float)
        RU = np.array([r[1] for r in raw], float)
        print(f"\n  ═══ HORIZON SCAN — the v_h at which the left line stops moving ═══")
        print(f"  {'v_h':>7}{'y(9-13 m)':>12}{'y(13-20 m)':>12}{'y(20-30 m)':>12}"
              f"{'drift':>9}")
        best = None
        for vh in np.arange(a.scan[0], a.scan[1] + 1e-6, a.scan[2]):
            Pt = dict(P0)
            Pt["pitch"] = LS.pitch_for_horizon(Pt, float(vh))
            fht = a.fx * a.height
            rows_t = np.unique(np.round(RV).astype(int))
            ymap = {}
            for vi in rows_t:
                q = vi - vh
                if q <= 4.0:
                    continue
                xx = fht / q                       # range from the CAMERA
                uv = BC.project_ground(np.array([[xx + LON, 0.0], [xx + LON, 1.0]]), Pt)
                if np.isfinite(uv).all() and abs(uv[1, 0] - uv[0, 0]) > 1e-6:
                    ymap[vi] = (float(uv[0, 0]), float(uv[1, 0] - uv[0, 0]), xx)
            pk = []
            for r_lo, r_hi in ((9.0, 13.0), (13.0, 20.0), (20.0, 30.0)):
                yy = []
                for v, u in zip(RV, RU):
                    m = ymap.get(int(round(v)))
                    if m and r_lo <= m[2] < r_hi:
                        yy.append((u - m[0]) / m[1])
                yy = np.asarray(yy)
                yy = yy[(yy > 0.5) & (yy < 4.0)]
                if len(yy) < 150:
                    pk.append(float("nan"))
                    continue
                hb = np.arange(0.5, 4.0, 0.08)
                hh, _ = np.histogram(yy, bins=hb)
                i = int(np.argmax(hh))
                sel = yy[np.abs(yy - (hb[i] + 0.04)) < 0.28]
                pk.append(float(np.median(sel)) if len(sel) >= 40 else float("nan"))
            if np.isfinite(pk).all():
                drift = float(np.nanmax(pk) - np.nanmin(pk))
                flag = ""
                if best is None or drift < best[1]:
                    best, flag = (float(vh), drift, list(pk)), ""
                print(f"  {vh:7.1f}{pk[0]:12.3f}{pk[1]:12.3f}{pk[2]:12.3f}"
                      f"{drift:9.3f}{flag}")
        if best:
            print(f"\n  ⭐ HORIZON = {best[0]:.1f} px   (residual drift {best[1]:.3f} m "
                  f"across 9-30 m)")
            print(f"     rendered at {a.horizon:.1f}; row-flow said 438; the lens bound "
                  f"said <= ~450;\n     the boundary-convergence row said 466")
            print(f"     left line then sits at {np.mean(best[2]):+.3f} m from the camera "
                  f"at h = {a.height} m")
            out["horizon_from_drift"] = best[0]
            out["horizon_drift_m"] = best[1]
            out["left_line_y_at_best"] = float(np.mean(best[2]))
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


def dash(a):
    """Which lateral bin is PAINT — by periodicity, on CONSECUTIVE frames.

    ⛔ THE FIX FOR `R-2026-09-15-burstiness`. That test asked whether a lateral bin
    is bursty, which should separate a DASHED line from a CONTINUOUS gravel apron.
    It failed completely — Fano 15–75 in every bin, including the left SOLID line
    — because the frames were sampled **~9 s apart** for coverage, so section, sun,
    exposure and lateral position all changed between samples and swamped the dash
    period. **It measured the sampling scheme.** At stride 1 (29.94 fps) the dash
    period is the only thing changing.

    ⭐ AND AT STRIDE 1 THERE IS A FAR BETTER DISCRIMINATOR THAN BURSTINESS:
    **a dashed line is PERIODIC.** At ~21 m/s a 12 m dash pitch gives a 0.57 s
    cycle, about 17 frames; gravel has no period at all. Periodicity is a
    structural claim, not a variance claim, so it cannot be faked by the scene
    changing — and it survives a detector that misses half the dashes.

    ⚠️ CONTROLS ARE PART OF THE MEASUREMENT, because §107 is exactly the story of
    an instrument that produced regular-looking structure out of nothing (peaks
    spaced by precisely the peak-picker's own 0.5 m rule). Two are reported:

      * the LEFT line is solid, so it must show **no** periodicity. If it does,
        the instrument is manufacturing it and nothing here is admissible.
      * `dash_standard.py` records that an earlier autocorrelation returned a
        **frame comb** — it tracked frame spacing, not road. Here the series is
        one value per frame per lateral bin, so there is no stacking and no comb;
        the control is what confirms it rather than the argument.
    """
    P0 = dict(RR.NOMINAL)
    P0.update(fx=a.fx, height=a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P0["pitch"] = LS.pitch_for_horizon(P0, a.horizon)
    fh = a.fx * a.height
    LON = float(P0.get("longitudinal", 0.0))
    band = np.arange(a.horizon + fh / max(a.ranges), a.horizon + fh / min(a.ranges), 1.0)
    y_of = {}
    for v in band:
        x = fh / max(v - a.horizon, 1e-3) + LON
        uv = BC.project_ground(np.array([[x, 0.0], [x, 1.0]]), P0)
        if np.isfinite(uv).all() and abs(uv[1, 0] - uv[0, 0]) > 1e-6:
            y_of[int(round(v))] = (float(uv[0, 0]), float(uv[1, 0] - uv[0, 0]))

    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = pathlib.Path(a.frames_dir)
    have = sorted(int(p.stem) for p in fdir.glob("*.jpg"))
    start = a.start if a.start else have[len(have) // 3]
    seq = [f for f in range(start, start + a.n) if f in set(have) and f in recs]
    if len(seq) < 200:
        print("⛔ not enough consecutive frames")
        return 1
    v_ms = float(np.median([recs[f]["speed_ms"] for f in seq]))
    print(f"CONSECUTIVE frames {seq[0]}..{seq[-1]} ({len(seq)} at {a.fps} fps), "
          f"median speed {v_ms:.1f} m/s, band {min(a.ranges):.0f}-{max(a.ranges):.0f} m")
    print(f"a 12 m dash pitch would give a period of {12.0/v_ms*a.fps:.1f} frames; "
          f"a 17 m pitch {17.0/v_ms*a.fps:.1f}\n")

    edges = np.arange(-a.y_right, a.y_left + 1e-9, 0.10)
    road, M = None, np.zeros((len(seq), len(edges) - 1), float)
    for k, f in enumerate(seq):
        g = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if road is None:
            road = road_mask(P0, g.shape, y_left=a.y_left, y_right=a.y_right,
                             x_range=(8.0, max(a.ranges) + 5.0))
        cols_at = _cols_factory(g, road, a.fx, a.paint_k)
        ys = []
        for v in band:
            vi = int(round(v))
            if vi not in y_of:
                continue
            u0, du = y_of[vi]
            for c in cols_at(v, fh / max(v - a.horizon, 1e-3)):
                ys.append((float(c) - u0) / du)
        if ys:
            M[k], _ = np.histogram(np.asarray(ys, float), bins=edges)

    lags = np.arange(a.lag_lo, a.lag_hi + 1)
    print(f"  {'lateral':>9}{'mean':>8}{'occup':>8}{'peak acf':>10}{'at lag':>8}"
          f"{'= pitch':>10}   verdict")
    out = []
    for j in range(M.shape[1]):
        s = M[:, j]
        if s.mean() < 0.40:
            continue
        # ⛔ THE CONTROL CAUGHT THIS. Raw autocorrelation flagged EVERY bin as
        # dashed at lag 6 -- including the SOLID left line -- and lag 6 was the
        # scan's own lower bound. Consecutive frames 0.2 s apart are nearly
        # identical, so the detector's own temporal smoothness dominates the acf
        # at short lags and looks exactly like a period. HIGH-PASS it: subtract a
        # moving average of `--smooth` frames, which kills the smooth component
        # while leaving a genuine dash cycle (17-24 frames here) untouched.
        w = max(3, int(a.smooth) | 1)
        ker = np.ones(w) / w
        trend = np.convolve(s, ker, mode="same")
        z = s - trend
        den = float((z * z).sum())
        if den <= 0:
            continue
        acf = np.array([float((z[:-L] * z[L:]).sum()) / den for L in lags])
        i = int(np.argmax(acf))
        peak, lag = float(acf[i]), int(lags[i])
        pitch = v_ms * lag / a.fps
        yc = edges[j] + 0.05
        occ = float((s > 0).mean())
        v = ("⭐ DASHED PAINT" if peak >= a.acf_min
             else "solid paint or continuous" if occ > 0.85
             else "no period — verge/gravel or noise")
        print(f"  {yc:+8.2f}{s.mean():8.2f}{100*occ:7.0f}%{peak:10.2f}{lag:8d}"
              f"{pitch:9.1f} m   {v}")
        out.append(dict(y=float(yc), mean=float(s.mean()), occ=occ, acf=peak,
                        lag=lag, pitch=float(pitch)))
        if any(abs(yc - q) < 0.06 for q in a.profile):
            curve = " ".join(f"{L}:{v:+.2f}" for L, v in zip(lags, acf)
                             if L % 4 == 0)
            print(f"        acf({yc:+.2f}) = {curve}")
    print(f"\n  ⚠️ CONTROL: the LEFT line (around +2.0 m) is SOLID and MUST show a low")
    print(f"     peak acf. If it is flagged dashed, this instrument manufactures")
    print(f"     periodicity and nothing above is admissible.")
    if a.json:
        a.json.write_text(json.dumps(dict(frames=[seq[0], seq[-1]], v_ms=v_ms,
                                          fps=a.fps, bins=out), indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames-dir",
                    default="/root/trajdata/out_bev/2026-08-08_14-19-54-android/frames")
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--horizon", type=float, default=448.4)
    ap.add_argument("--yaw", type=float, default=-5.35)
    ap.add_argument("--lateral", type=float, default=-0.126,
                    help="the --lateral-offset the render was drawn with; y positive LEFT, "
                         "so -0.126 means the camera sits 0.126 m RIGHT of the centreline")
    ap.add_argument("--lane-m", type=float, default=3.5,
                    help="assumed lane width, used ONLY for the h-free placement column")
    ap.add_argument("--ranges", type=float, nargs="+",
                    default=[10., 15., 20., 25., 30., 40.])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--mask", dest="mask", action="store_true", default=True)
    ap.add_argument("--no-mask", dest="mask", action="store_false",
                    help="the pre-`R-2026-09-14-foliage` behaviour, for A/B only")
    ap.add_argument("--y-left", type=float, default=4.2)
    ap.add_argument("--y-right", type=float, default=2.6)
    ap.add_argument("--ruler", type=float, nargs="+",
                    default=[2.0, 1.0, -1.0, -1.5, -2.0, -2.5],
                    help="vehicle-frame lateral offsets to draw as guide lines "
                         "in --preview, so the mask width is read, not guessed")
    ap.add_argument("--paint-k", type=float, default=3.0,
                    help="a detection must exceed the row's masked median by this "
                         "many robust sd to count as paint; 0 disables (the old behaviour)")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--scan", type=float, nargs=3,
                    metavar=("LO", "HI", "STEP"), default=[380.0, 470.0, 2.0],
                    help="horizon scan range. ⚠️ CHECK THE OPTIMUM IS NOT AT AN "
                         "END OF IT — a boundary solution is not a measurement")
    ap.add_argument("--dash", action="store_true",
                    help="which lateral bin is paint, by PERIODICITY on consecutive "
                         "frames — the fix for R-2026-09-15-burstiness")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--fps", type=float, default=29.94)
    ap.add_argument("--lag-lo", type=int, default=12,
                    help="below ~12 frames no dash pitch is physical at this "
                         "speed; the raw acf there is detector smoothness")
    ap.add_argument("--smooth", type=int, default=9,
                    help="moving-average window removed before the acf")
    ap.add_argument("--profile", type=float, nargs="*", default=[],
                    help="print the full acf curve for these lateral bins")
    ap.add_argument("--lag-hi", type=int, default=70)
    ap.add_argument("--acf-min", type=float, default=0.25)
    ap.add_argument("--hist", action="store_true",
                    help="histogram every detection in metres and exit — no fit, "
                         "no winner, so a non-paint peak is visible instead of chosen")
    ap.add_argument("--preview", type=pathlib.Path, default=None,
                    help="write an annotated image of the FIT and exit — do this first")
    a = ap.parse_args()
    if a.preview:
        return preview(a, a.preview)
    if a.dash:
        return dash(a)
    if a.hist:
        return histogram(a)

    P0 = dict(RR.NOMINAL)
    P0.update(fx=a.fx, height=a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P0["pitch"] = LS.pitch_for_horizon(P0, a.horizon)
    fh = a.fx * a.height
    rows = {x: a.horizon + fh / x for x in a.ranges}
    band = np.arange(a.horizon + fh / max(a.ranges), a.horizon + fh / min(a.ranges), 1.0)
    v_ref = a.horizon + fh / 12.0
    u_ref = float(BC.project_ground(np.array([[12.0, 0.0]]), P0)[0][0])

    src = sorted(pathlib.Path(a.frames_dir).glob("*.jpg"))
    cap = cv2.VideoCapture(a.video)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(np.linspace(30, min(nfr, len(src)) - 30, a.n).astype(int).tolist())

    road = None
    cov = {x: [0, 0] for x in a.ranges}          # [covered, seen]
    place = {x: [] for x in a.ranges}            # metres, vehicle frame, +LEFT
    place_h = {x: [] for x in a.ranges}          # h-free: fraction of a lane, +LEFT
    lanew = {x: [] for x in a.ranges}
    corrw = {x: [] for x in a.ranges}
    clear_l = {x: [] for x in a.ranges}
    clear_r = {x: [] for x in a.ranges}
    cam_off, v0s, frames_ok, frames_pts = [], [], 0, 0

    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            g = cv2.imread(str(src[i]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                if road is None and a.mask:
                    road = road_mask(P0, g.shape, y_left=a.y_left, y_right=a.y_right,
                                     x_range=(8.0, max(a.ranges) + 5.0))
                    print(f"road mask: {100*float(road.mean())/255:.1f} % of the frame, "
                          f"rows {np.nonzero(road.any(1))[0].min()}-"
                          f"{np.nonzero(road.any(1))[0].max()}")

                cols_at = _cols_factory(g, road, a.fx, a.paint_k)
                r = fit_pair(cols_at, band, a.horizon, fh, v_ref, u_ref, a.height)
                frames_pts += 1
                if r is not None:
                    ml, bl, mr, br = r["ml"], r["bl"], r["mr"], r["br"]
                    frames_ok += 1
                    v0s.append(r["v0"])
                    # h*(m_L+m_R)/2 = camera y minus lane-centre y, +LEFT.
                    cam_off.append(a.height * 0.5 * (ml + mr))
                    for x in a.ranges:
                        v = rows[x]
                        e = corridor_edges(fr, int(round(v * S)))
                        if e is None:
                            continue
                        gl, gr = e[0] / S, e[1] / S
                        uL, uR = ml * v + bl, mr * v + br
                        mpp = x / a.fx
                        cov[x][1] += 1
                        if [c for c in cols_at(v, x) if gl + 2 < c < gr - 2]:
                            cov[x][0] += 1
                        # u is +RIGHT, vehicle y is +LEFT -> negate.
                        du = 0.5 * (gl + gr) - 0.5 * (uL + uR)
                        place[x].append(-du * mpp)
                        if abs(uR - uL) > 1.0:
                            place_h[x].append(-du / (uR - uL) * a.lane_m)
                        lanew[x].append((uR - uL) * mpp)
                        corrw[x].append((gr - gl) * mpp)
                        clear_l[x].append((gl - uL) * mpp)
                        clear_r[x].append((uR - gr) * mpp)
        i += 1
        if i > max(want):
            break
    cap.release()

    mad = lambda z: 1.4826 * np.median(np.abs(np.asarray(z) - np.median(z)))
    print(f"\n{a.label or a.video}")
    print(f"mask {'ON' if a.mask else 'OFF'} · {frames_pts} frames with enough ridge "
          f"points · {frames_ok} with a valid single-lane pair "
          f"({100*frames_ok/max(frames_pts,1):.0f} %)\n")
    if frames_ok < 20:
        print("⛔ too few valid lane pairs — nothing quotable")
        return 1

    print(f"  {'range':>6}{'covers paint':>14}{'corr - lane centre':>20}"
          f"{'(h-free)':>10}{'lane w':>9}{'ribbon w':>10}{'clear L':>9}{'clear R':>9}")
    out = {}
    for x in a.ranges:
        c, s = cov[x]
        if s < 20:
            print(f"  {x:4.0f} m{'-- too few --':>14}")
            continue
        med = lambda d: float(np.median(d[x])) if len(d[x]) >= 20 else float("nan")
        print(f"  {x:4.0f} m{100*c/s:>12.0f}%{med(place):>17.2f} m"
              f"{med(place_h):>9.2f}{med(lanew):>9.2f}{med(corrw):>10.2f}"
              f"{med(clear_l):>9.2f}{med(clear_r):>9.2f}")
        out[str(x)] = dict(covered_pct=100.0 * c / s, n=s, placement=med(place),
                           placement_hfree=med(place_h), lane_w=med(lanew),
                           corridor_w=med(corrw), clear_left=med(clear_l),
                           clear_right=med(clear_r))
    print("  (placement and clearances are +LEFT in the vehicle frame; 'h-free' uses "
          f"only\n   the ratio to the lane width and a {a.lane_m} m lane — no h, no horizon, no f)")

    print(f"\n  ⭐ WHERE THE TWO LANE BOUNDARIES MEET  v_0 = {np.median(v0s):.1f} px"
          f"   robust sd {mad(v0s):.1f}   n {len(v0s)}")
    print(f"     that row IS the horizon if both lines are road-parallel. Rendered at "
          f"{a.horizon:.1f}.")
    print(f"     (unmasked this read 467.7; row-flow says 438; the lens bound says "
          f"<= ~450)")

    co = float(np.median(cam_off))
    veh = co - a.lateral          # vehicle centreline = camera y minus the mount offset
    allp = [v for x in a.ranges for v in place[x]]
    allph = [v for x in a.ranges for v in place_h[x]]
    pl, plh = float(np.median(allp)), float(np.median(allph))
    print(f"\n  WHERE THE CAR ACTUALLY IS, from the paint alone (h cancels in the ratio):")
    print(f"    camera             {co:+.2f} m from lane centre   robust sd {mad(cam_off):.2f}"
          f"   (+ = left)")
    print(f"    vehicle centreline {veh:+.2f} m  (camera minus the {a.lateral:+.3f} m mount offset)")
    print(f"    corridor drawn at  {pl:+.2f} m   (h-free: {plh:+.2f} m)")
    print(f"    => the drawing is  {pl - veh:+.2f} m from where the car is"
          f"   (h-free: {plh - veh:+.2f} m)")
    out["placement_check"] = dict(camera=co, camera_sd=float(mad(cam_off)), vehicle=veh,
                                  corridor=pl, corridor_hfree=plh, error=pl - veh,
                                  error_hfree=plh - veh, lateral_flag=a.lateral,
                                  v0=float(np.median(v0s)), v0_sd=float(mad(v0s)),
                                  n_pairs=frames_ok)
    if abs(plh - veh) > 0.10:
        print(f"    ⛔ that is a PLACEMENT DEFECT worth {abs(plh-veh):.2f} m, and in a lane")
        print(f"       with ~0.15 m of clearance it is the difference between touching")
        print(f"       the paint and not.")
        print(f"    ⚠️  BUT it is only a defect in the RENDER if the mount offset is right.")
        print(f"       A drawing error of d is indistinguishable from a mount offset that")
        print(f"       is wrong by d: --lateral-offset {a.lateral:+.3f} would have to be "
              f"{a.lateral + (plh - veh):+.3f}")
        print(f"       to absorb it. Only a tape measure separates those two readings.")
    else:
        print(f"    ✅ within 0.10 m: the corridor is where the car is, and any contact")
        print(f"       with the paint is the car's position, not the drawing's.")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
