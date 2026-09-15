#!/usr/bin/env python3
"""Where is the corridor, against the ONE line we trust — end to end, one projection.

Sayed's complaint is lateral: the corridor cuts road markings. Three probes have
tried to answer it and each was contaminated on the RIGHT-hand boundary — the
sunlit bank (`R-2026-09-14-foliage`), a tar seam and the gravel apron
(`R-2026-09-15-seam`). This one **never uses the right boundary at all.**

WHAT IT COMPARES, AND WHY IT IS FREE OF THE USUAL TRAPS

  * the corridor's drawn edges, read out of the composite;
  * the LEFT painted line, which §92 established is a clean, isolated, narrow
    peak — the one feature on this recording that has survived every check;

both converted to VEHICLE-frame metres **through the same `project_ground` the
renderer itself used**. No `x = f·h/q` shortcut, no `mpp = x/f` approximation, no
lane width, no right-hand line, no pair, no vanishing point.

⛔ TWO MISTAKES THIS FILE EXISTS TO NOT REPEAT.

1. **Mixing two coordinate frames.** The lateral histogram probes `project_ground`
   at vehicle `y = 0` and `y = 1`, so what it returns is a **VEHICLE-frame**
   lateral — measured from the vehicle centreline, NOT from the camera. I read
   `+1.62 m` as camera-frame and got a placement error 0.13 m too large. The
   corridor is drawn about vehicle `y = 0`, so in this frame the comparison is
   direct and the mount offset does not enter at all.

2. **Comparing two things at different ROWS.** Reading the drawn ribbon at
   `v = v_h + f·h/x` while predicting at `project_ground(x)`'s own row compares
   different places if the two disagree by even a pixel near the horizon. Here
   BOTH are read at the row `project_ground` returns.

WHAT A RESULT MEANS. The corridor is drawn about the vehicle centreline, so its
measured centre should be `0.00 m` by construction — that is the renderer
self-consistency check, and it passes (centre error −0.038…+0.006 m over
10–30 m). The physical question is then the ONE number this prints:

    left line, in vehicle metres  —  and whether the car is where it should be.

⚠️ The ribbon follows the FUTURE PATH, so on a curve it correctly leaves the lane.
Only low-yaw-rate frames are used, and the profile is printed per range so a
residual angle shows as a slope rather than hiding in a mean.
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
from overlay_far import corridor_edges, S                              # noqa: E402
from road_mask import road_mask                                        # noqa: E402
from does_it_cut_paint import _cols_factory                            # noqa: E402


def per_frame(a, P, geo, hw):
    """The MINIMUM clearance per frame, and whether it is explained by turning.

    ⭐ WHY THE MEDIAN CANNOT ANSWER SAYED'S QUESTION. He reports the corridor
    cutting road markings; the medians say there is 1.02–1.12 m of clearance at
    every range. Both can be true: the left line's robust sd runs 0.23 m at 10 m
    to 0.54 m at 30 m, so a median with a metre of room is entirely consistent
    with a visible minority of frames at or past zero. **A complaint about what
    is visible is a complaint about the TAIL, and a median is the one statistic
    guaranteed not to show it.**

    AND THE TAIL HAS TWO CAUSES THAT DESERVE OPPOSITE RESPONSES:

      * the ribbon follows the FUTURE PATH, so in a turn it *correctly* leaves the
        lane — that is the overlay working, and "fixing" it would be a regression;
      * or the geometry is wrong, in which case it happens on straight road too.

    Yaw rate separates them, which is why every frame is kept here and the
    straight-only filter of :func:`main` is deliberately NOT applied.

    ⚠️ THE LEFT LINE IS TRACKED BY PROXIMITY TO A FIRST-PASS ESTIMATE, not by
    taking the nearest candidate to the vehicle. `R-2026-09-15-seam` is exactly
    what happens when a detector is free to pick whichever feature is closest to
    the answer it wants: a tar seam runs up the middle of the ego lane, and
    "smallest y" would select it in precisely the frames that would then be
    reported as near-zero clearance. That would MANUFACTURE the tail this
    function exists to measure.
    """
    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    src = sorted(pathlib.Path(a.frames_dir).glob("*.jpg"))
    cap = cv2.VideoCapture(a.video)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hi = min(nfr, len(src)) - 30
    cand = [f for f, r in sorted(recs.items())
            if r["complete"] and r["speed_ms"] > 8 and 30 <= f < hi]
    want = set(np.asarray(cand)[np.linspace(0, len(cand) - 1,
                                            min(a.n, len(cand))).astype(int)].tolist())
    print(f"{len(cand)} usable frames (v > 8 m/s, ALL yaw rates); sampling {len(want)}")

    road, raw, i = None, [], 0
    while True:
        ok, fr = cap.read()
        if not ok or i > max(want):
            break
        if i in want:
            g = cv2.imread(str(src[i]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                if road is None:
                    road = road_mask(P, g.shape, y_left=4.4, y_right=4.4,
                                     x_range=(8.0, max(a.ranges) + 5.0))
                cols_at = _cols_factory(g, road, a.fx, a.paint_k)
                row = {}
                for x, G in geo.items():
                    ys = [(c - G["u0"]) / G["du"] for c in cols_at(G["row"], x)]
                    row[x] = [y for y in ys if 0.2 < y < 4.2]
                # ⛔ THERE IS NO `yaw_rate_dps` IN THESE RECORDS. `.get(..., 0.0)`
                # returned the default for every frame, so the yaw-rate split put
                # 228/228 frames in one band and `main`'s "straight frames only"
                # filter passed 2156 of 2216 -- i.e. it never filtered at all.
                # What the records DO carry is the future path itself (`x`, `y`),
                # and `y` interpolated at a range IS the ribbon's predicted lateral
                # there. That is not a proxy for the cause, it is the cause.
                r = recs[i]
                px, py = np.asarray(r["x"], float), np.asarray(r["y"], float)
                fwd = px > 0.0
                path = {x: float(np.interp(x, px[fwd], py[fwd])) for x in geo}
                raw.append((i, float(r.get("steer_wheel_deg", 0.0)), row, path))
        i += 1
    cap.release()
    if len(raw) < 40:
        print("⛔ too few frames")
        return 1

    # pass 1: where the left line sits, from the well-populated frames
    pool = [y for _, _, row, _p in raw for x in row for y in row[x] if 1.4 < y < 3.2]
    if len(pool) < 200:
        print("⛔ no left-line pool")
        return 1
    ref = float(np.median(pool))
    print(f"\n  first-pass left line at {ref:+.2f} m; tracking within "
          f"±{a.track_win} m of it\n")

    # pass 2: per frame, the worst clearance over the range set
    worst, wyaw, wpath, per_x = [], [], [], {x: [] for x in geo}
    for f, steer, row, path in raw:
        cl = []
        for x in geo:
            near = [y for y in row[x] if abs(y - ref) < a.track_win]
            if near:
                y = min(near, key=lambda z: abs(z - ref))
                # clearance = left line -> the RIBBON's left edge, which sits at
                # path_y(x) + hw, not at hw: the ribbon follows the future path.
                per_x[x].append(y - (path[x] + hw))
                cl.append(y - (path[x] + hw))
        if len(cl) >= 3:
            worst.append(min(cl))
            wyaw.append(abs(steer))
            wpath.append(path[max(geo)])
    worst = np.asarray(worst)
    wyaw = np.asarray(wyaw)
    wpath = np.asarray(wpath)
    print(f"  {len(worst)} frames with >= 3 ranges tracked\n")
    print(f"  ⭐ MINIMUM clearance per frame, ribbon edge -> left line (metres)")
    for p in (0, 1, 5, 10, 25, 50, 75, 95, 100):
        print(f"       p{p:<3d} {np.percentile(worst, p):+7.2f} m"
              + ("   <- the corridor is ON the line" if np.percentile(worst, p) < 0.05
                 else ""))
    for thr in (0.0, 0.10, 0.25):
        n = int((worst < thr).sum())
        print(f"     frames with clearance < {thr:.2f} m:  {n} / {len(worst)} "
              f"({100*n/len(worst):.1f} %)")

    print(f"\n  IS IT TURNING?  |steering wheel| vs the frame's worst clearance")
    print(f"  {'|steer|':>14}{'n':>6}{'median clear':>14}{'p5':>9}{'frac < 0':>10}")
    edges = [0.0, 1.0, 2.0, 4.0, 7.0, 99.0]
    rows_out = []
    for lo, h_ in zip(edges[:-1], edges[1:]):
        m = (wyaw >= lo) & (wyaw < h_)
        if m.sum() < 10:
            continue
        w = worst[m]
        print(f"  {lo:5.1f}-{h_:<5.1f} deg  {m.sum():6d}{np.median(w):+13.2f}"
              f"{np.percentile(w,5):+9.2f}{100*(w<0).mean():9.1f} %")
        rows_out.append(dict(lo=lo, hi=h_, n=int(m.sum()), median=float(np.median(w)),
                             p5=float(np.percentile(w, 5)),
                             frac_neg=float((w < 0).mean())))
    if len(worst) > 30 and np.std(wpath) > 0:
        rp = float(np.corrcoef(wpath, worst)[0, 1])
        print(f"\n  the ribbon's OWN predicted lateral at {max(geo):.0f} m:"
              f"  median {np.median(wpath):+.2f} m,"
              f"  5-95 % [{np.percentile(wpath,5):+.2f}, {np.percentile(wpath,95):+.2f}]")
        print(f"  Pearson r(path lateral, worst clearance) = {rp:+.3f}"
              f"   (expect ~ -1 if the excursion IS the predicted turn)")
    if len(worst) > 30 and wyaw.std() > 0:
        r = float(np.corrcoef(wyaw, worst)[0, 1])
        print(f"  Pearson r(|steering wheel|, worst clearance) = {r:+.3f}")
        print(f"     r << 0 => the tail IS turning, and the overlay is correct:")
        print(f"     the ribbon follows the future path and a turn takes it across a line.")
        print(f"     r ~ 0  => the tail is NOT explained by turning, and the geometry is at fault.")
    out = dict(ref=ref, n=len(worst), track_win=a.track_win,
               pct={str(p): float(np.percentile(worst, p))
                    for p in (0, 1, 5, 10, 25, 50, 75, 95, 100)},
               frac_lt={f"{t:.2f}": float((worst < t).mean()) for t in (0.0, 0.10, 0.25)},
               yaw_bands=rows_out,
               path_lat_med=float(np.median(wpath)),
               r_path=float(np.corrcoef(wpath, worst)[0, 1]) if np.std(wpath) else None,
               r=float(np.corrcoef(wyaw, worst)[0, 1]) if wyaw.std() > 0 else None,
               per_range={str(x): dict(n=len(v), median=float(np.median(v)),
                                       p5=float(np.percentile(v, 5)))
                          for x, v in per_x.items() if len(v) >= 20})
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
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
    ap.add_argument("--lateral", type=float, default=-0.126)
    ap.add_argument("--vehicle-width", type=float, default=1.855)
    ap.add_argument("--ranges", type=float, nargs="+", default=[10., 15., 20., 25., 30.])
    ap.add_argument("--max-yaw-rate", type=float, default=1.0, help="deg/s")
    ap.add_argument("--paint-k", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--per-frame", action="store_true",
                    help="the MINIMUM clearance per frame and its correlation with "
                         "yaw rate — the tail, which no median can show")
    ap.add_argument("--track-win", type=float, default=0.80,
                    help="track the left line within this many metres of a first-pass "
                         "estimate; a free 'nearest to the vehicle' pick would select a "
                         "tar seam in exactly the frames reported as near-zero clearance")
    a = ap.parse_args()

    P = dict(RR.NOMINAL)
    P.update(fx=a.fx, height=a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)
    hw = a.vehicle_width / 2.0

    # Per range: the row to read, and the affine column->vehicle-y map at that row.
    geo = {}
    for x in a.ranges:
        uv = BC.project_ground(np.array([[x, 0.0], [x, 1.0], [x, hw], [x, -hw]]), P)
        if not np.isfinite(uv).all():
            continue
        u0, u1 = float(uv[0, 0]), float(uv[1, 0])
        if abs(u1 - u0) < 1e-6:
            continue
        geo[x] = dict(row=float(uv[0, 1]), u0=u0, du=u1 - u0,
                      pred_l=float(uv[2, 0]), pred_r=float(uv[3, 0]))

    if a.per_frame:
        return per_frame(a, P, geo, hw)

    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    ok_f = sorted(f for f, r in recs.items()
                  if r["complete"] and r["speed_ms"] > 10
                  and abs(r.get("yaw_rate_dps", 0.0)) < a.max_yaw_rate)
    src = sorted(pathlib.Path(a.frames_dir).glob("*.jpg"))
    cap = cv2.VideoCapture(a.video)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hi = min(nfr, len(src)) - 30
    cand = [f for f in ok_f if 30 <= f < hi]
    want = set(np.asarray(cand)[np.linspace(0, len(cand) - 1,
                                            min(a.n, len(cand))).astype(int)].tolist())
    print(f"{len(cand)} straight frames (|yaw rate| < {a.max_yaw_rate} deg/s, "
          f"v > 10 m/s); sampling {len(want)}")

    road = None
    CEN = {x: [] for x in geo}
    LEFT = {x: [] for x in geo}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok or i > max(want):
            break
        if i in want:
            g = cv2.imread(str(src[i]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                if road is None:
                    road = road_mask(P, g.shape, y_left=3.6, y_right=3.6,
                                     x_range=(8.0, max(a.ranges) + 5.0))
                cols_at = _cols_factory(g, road, a.fx, a.paint_k)
                for x, G in geo.items():
                    v = G["row"]
                    e = corridor_edges(fr, int(round(v * S)))
                    if e is not None:
                        gl, gr = e[0] / S, e[1] / S
                        CEN[x].append((0.5 * (gl + gr) - G["u0"]) / G["du"])
                    # the LEFT line: the brightest paint left of the vehicle axis
                    cand_y = [(c - G["u0"]) / G["du"] for c in cols_at(v, x)]
                    cand_y = [y for y in cand_y if 0.8 < y < 3.2]
                    if cand_y:
                        LEFT[x].append(min(cand_y))
        i += 1
    cap.release()

    mad = lambda z: 1.4826 * np.median(np.abs(np.asarray(z) - np.median(z)))
    print(f"\n  ALL NUMBERS ARE VEHICLE-FRAME METRES, + = LEFT of the vehicle centreline.")
    print(f"  The ribbon is drawn about y = 0, so 'corridor centre' is the renderer's")
    print(f"  own self-consistency and should read 0.00.\n")
    print(f"  {'range':>6}{'n':>5}{'corridor centre':>17}{'left line':>12}{'sd':>7}"
          f"{'gap to ribbon':>15}")
    out = {}
    for x in sorted(geo):
        if len(LEFT[x]) < 20 or len(CEN[x]) < 20:
            print(f"  {x:4.0f} m{'-- too few --':>22}")
            continue
        c = float(np.median(CEN[x]))
        l = float(np.median(LEFT[x]))
        print(f"  {x:4.0f} m{len(LEFT[x]):5d}{c:+15.3f} m{l:+11.2f}"
              f"{mad(LEFT[x]):7.2f}{l - hw:+13.2f} m")
        out[str(x)] = dict(n=len(LEFT[x]), corridor_centre=c, left_line=l,
                           left_sd=float(mad(LEFT[x])), gap=l - hw)
    if out:
        ls = [v["left_line"] for v in out.values()]
        print(f"\n  ⭐ LEFT LINE AT {np.median(ls):+.2f} m FROM THE VEHICLE CENTRELINE")
        print(f"     a {a.vehicle_width} m vehicle centred in a lane of width W sits with its")
        print(f"     left boundary at W/2, so this reading implies W = {2*np.median(ls):.2f} m")
        print(f"     if the car is centred — and §97 bounds W at <= 3.16 m by the optics.")
        print(f"     Excess over a centred car in a 3.00 m lane: "
              f"{np.median(ls) - 1.50:+.2f} m (+ = car sits RIGHT of centre)")
        out["summary"] = dict(left_line_median=float(np.median(ls)),
                              implied_W_if_centred=float(2 * np.median(ls)),
                              offset_vs_3m_lane=float(np.median(ls) - 1.50))
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
