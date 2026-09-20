#!/usr/bin/env python3
"""LANE CONTAINMENT — the ribbon must stay BETWEEN THE PAINTED LINES.

⛔ WHY THIS REPLACES `road_containment.py` AS THE DECIDING METRIC. Sayed, seven
renders in: *"the trajectory still [is] leaving the road, knowing that ego [is]
driving between the road markings."* The second clause is the specification and I
scored the first clause only.

`road_containment.py` asks **"is the ribbon on drivable asphalt?"** and its answer
for the current render is 96 % of frames clean. That answer is true and it is not
the question, because **this is a TWO-LANE carriageway**: the asphalt continues
across the left-hand lane line for another 3.5 m, so a ribbon that drifts half a
metre toward the neighbouring lane never leaves the drivable mask and never costs
a point. The metric was permissive in exactly the direction of the error.

MEASURED 2026-09-19 on the frame Sayed sent (source 908, t = 33.63 s), with the
scale `h/(v-v_h)` and no other calibration:

    range   clear to LEFT line   clear to RIGHT line   lane    centred clear
     10 m        +0.43 m              +1.29 m          3.59 m      0.87 m
     12 m        +0.31 m              +1.39 m          3.58 m      0.86 m
     15 m        +0.13 m               (dashed gap)

Both clearances are POSITIVE -- so `road_containment` sees nothing wrong, and so
would a "does it cross a line" test on that frame alone. But the ribbon is
**0.44 m left of the lane centre at 10 m and 0.55 m at 12 m**, and a bias that
GROWS WITH RANGE is a residual yaw, which is the defect that has survived every
round of this calibration.

⇒ **The quantity is the ribbon's offset from the LANE CENTRE, not its distance to
the nearest verge.** Its correct value is known and it is zero: the car was
driving between the markings, so the ribbon drawn around the car's own recorded
future path must be centred between them.

⚠️ WHAT IT REFUSES TO MEASURE, ON PURPOSE. A sample is dropped unless BOTH lines
are found and the implied lane width is physically plausible. The right-hand line
here is DASHED, so a nearest-ridge search that is allowed to run past it will
happily return the shoulder edge line or the gravel beyond -- that is
`R-2026-09-15-seam` exactly, and it is what produced a phantom right boundary at
-2.55 m once already. A dropped sample costs n; a wrong one costs a render.
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
from overlay_far import ridge_cols, ridge_width_px                     # noqa: E402

VEH_W = 1.855
LANE_LO, LANE_HI = 2.9, 4.3        # plausible carriageway lane width, metres


def load(n, speed_min=8.0):
    """Grey frames + future paths, spread over the whole recording."""
    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = RR.RUN / "frames"
    have = {int(p.stem) for p in fdir.glob("*.jpg")}
    cand = [f for f, r in sorted(recs.items())
            if r["complete"] and f in have and r["speed_ms"] > speed_min]
    pick = [cand[i] for i in np.linspace(0, len(cand) - 1,
                                         min(n, len(cand))).astype(int)]
    out = []
    for f in pick:
        g = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        r = recs[f]
        px, py = np.asarray(r["x"], float), np.asarray(r["y"], float)
        fwd = px > 0.0
        if fwd.sum() < 3:
            continue
        out.append((g, px[fwd], py[fwd], f))
    return out


def paint_near(row, gl, gr, mpp, x, fx, reach_m=3.0, k=2.5):
    """The ego lane's two painted lines, as columns either side of the RIBBON CENTRE.

    ⛔ THE SIDES ARE TAKEN ABOUT THE CENTRE, NOT ABOUT THE RIBBON EDGES, AND THAT
    IS THE WHOLE POINT. Searching strictly outside the edges cannot see a line the
    ribbon is sitting ON: the sample is dropped for want of a detection instead of
    being recorded as a crossing. MEASURED 2026-09-19: the edge-anchored version
    reported ``cross 0.0 %`` for every calibration in the panel -- including the
    one shipped first, whose ribbon is over a metre off the lane centre at 30 m.
    A metric that is structurally incapable of returning a failure is not
    measuring the failure, and "0.0 %" reads like a pass.

    The window is anchored on the ribbon and is only ``reach_m`` wide either side,
    and the threshold statistics come from inside that same window -- a global
    threshold is set by whatever is brightest anywhere in the row (sky, a sign, a
    vehicle) and buries the line that is actually there.
    """
    mid = 0.5 * (gl + gr)
    lo = max(0, int(mid - reach_m / mpp))
    hi = min(len(row), int(mid + reach_m / mpp))
    if hi - lo < 40:
        return None, None
    cols = ridge_cols(row, ridge_width_px(x, fx), lo=lo, hi=hi)
    seg = row[lo:hi].astype(np.float32)
    lvl = float(np.median(seg))
    sig = max(1.4826 * float(np.median(np.abs(seg - lvl))), 3.0)
    cols = [c for c in cols if float(row[int(c)]) > lvl + k * sig]
    left = [c for c in cols if c < mid]
    right = [c for c in cols if c > mid]
    return (max(left) if left else None), (min(right) if right else None)


def measure(data, fx, height, horizon, yaw, lateral, ranges, reach_m=3.0, k=2.5):
    """Per-sample ribbon offset from the LANE CENTRE (metres, + = too far LEFT).

    Returns (offsets, per_range_offsets, cross_flags, n_seen) where ``cross`` is
    True when either clearance is negative, i.e. the ribbon is over a line.
    """
    P = dict(RR.NOMINAL)
    P.update(fx=fx, height=height, lateral=lateral, yaw=np.deg2rad(yaw))
    P["pitch"] = LS.pitch_for_horizon(P, horizon)
    LON = float(P.get("longitudinal", 0.0))
    hw = VEH_W / 2.0
    off = []
    per_range = [[] for _ in ranges]
    cross = []
    seen = 0
    for g, px, py, _f in data:
        H, W = g.shape
        for kx, x in enumerate(ranges):
            xc = x + LON
            yc = float(np.interp(xc, px, py))
            uv = BC.project_ground(np.array([[xc, yc + hw], [xc, yc - hw]], float), P)
            if not np.isfinite(uv).all():
                continue
            gl, gr = float(uv[0, 0]), float(uv[1, 0])
            v = 0.5 * (uv[0, 1] + uv[1, 1])
            vi = int(round(v))
            if not (0 <= vi < H) or v - horizon <= 1.0 or gr <= gl:
                continue
            seen += 1
            mpp = height / (v - horizon)
            pl, pr = paint_near(g[vi], gl, gr, mpp, x, fx, reach_m, k)
            if pl is None or pr is None:
                continue
            cl = (gl - pl) * mpp
            cr = (pr - gr) * mpp
            lane = cl + cr + VEH_W
            if not (LANE_LO <= lane <= LANE_HI):
                continue                       # not the ego lane's two lines
            d = 0.5 * (cr - cl)                # + => ribbon sits too far LEFT
            off.append(d)
            per_range[kx].append(d)
            cross.append(cl < 0 or cr < 0)
    return (np.asarray(off, float), [np.asarray(v, float) for v in per_range],
            np.asarray(cross, bool), seen)


def per_frame(data, fx, height, horizon, yaw, lateral, ranges, min_n=3, **kw):
    """Per-FRAME median offset from the lane centre, plus its crossing count.

    ⚠️ A MEDIAN OVER SAMPLES AND A DISTRIBUTION OVER FRAMES ARE DIFFERENT CLAIMS,
    and reporting the first as if it were the second is the mistake this whole
    calibration has now made twice (`R-2026-09-19-greenhue`). A calibration can
    have a median offset of zero while a fifth of frames are half a metre out,
    and it is a frame that Sayed looks at.

    ⚠️ IT ALSO CANNOT BE READ PER FRAME AS A CALIBRATION ERROR. The metric's
    premise is that the car drives CENTRED between the markings -- true on
    average, false in any given frame, where the driver is a few centimetres off
    and correcting. The per-frame spread therefore contains real driving as well
    as calibration error, and only the CENTRAL VALUE over many frames is a
    calibration claim. The spread bounds how much a single frame can be trusted.
    """
    out = []
    for rec in data:
        off, _pr, cross, _seen = measure([rec], fx, height, horizon, yaw, lateral,
                                         ranges, **kw)
        if len(off) >= min_n:
            out.append((rec[3], float(np.median(off)), int(cross.sum()), len(off)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--horizon", type=float, default=472.0)
    ap.add_argument("--ranges", type=float, nargs="+",
                    default=[8., 10., 12., 15., 18., 22., 26., 30.])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--yaw-scan", type=float, nargs=3, default=[-11.0, -5.0, 0.25],
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--lateral-scan", type=float, nargs=3, default=[-0.70, 0.30, 0.05],
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--baseline", type=float, nargs=2, default=[-7.75, -0.15],
                    metavar=("YAW", "LATERAL"))
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    print(f"loading {a.n} frames ...", flush=True)
    data = load(a.n)
    print(f"{len(data)} frames x {len(a.ranges)} ranges\n", flush=True)

    def report(tag, yaw, lat):
        off, pr, cross, seen = measure(data, a.fx, a.height, a.horizon, yaw, lat,
                                       a.ranges)
        if not len(off):
            print(f"  {tag:<30} no usable samples"); return None
        med = float(np.median(off))
        rsd = 1.4826 * float(np.median(np.abs(off - med)))
        print(f"  {tag:<30} offset {med:+6.3f} +-{rsd:5.3f} m   "
              f"cross {100*cross.mean():5.1f}%   n {len(off):5d}/{seen}")
        return off, pr, cross

    print("RIBBON OFFSET FROM THE LANE CENTRE   (+ = drawn too far LEFT; target 0.000)")
    by, bl = a.baseline
    base = report(f"v6 yaw {by} lat {bl}", by, bl)
    for nm, y, l in (("v1 shipped", -5.35, -0.126),
                     ("v5", -7.01, -0.088),
                     ("pipeline LaneCalib", -6.86, -0.088)):
        report(nm, y, l)

    if base is not None:
        print("\n  PER RANGE (median offset, m) — a trend with range is a YAW error")
        print("   " + "  ".join(f"{r:>5.0f}m" for r in a.ranges))
        print("   " + "  ".join(f"{np.median(v):+6.3f}" if len(v) >= 8 else "   n/a"
                                for v in base[1]))
        ok = [(r, np.median(v)) for r, v in zip(a.ranges, base[1]) if len(v) >= 8]
        if len(ok) >= 3:
            rr = np.array([r for r, _ in ok]); dd = np.array([d for _, d in ok])
            A = np.stack([np.ones_like(rr), rr], 1)
            c, m = np.linalg.lstsq(A, dd, rcond=None)[0]
            res = dd - (c + m * rr)
            r2 = 1 - res.var() / max(dd.var(), 1e-12)
            print(f"\n  fit offset(x) = {c:+.3f} {m:+.4f}*x   (R^2 {r2:.3f}, n {len(rr)})")
            print(f"  ⇒ level term {c:+.3f} m is a LATERAL error; slope {m:+.4f} m/m "
                  f"= {np.rad2deg(np.arctan(m)):+.2f} deg is a YAW error")
            # ⛔ NO CORRECTED PARAMETERS ARE PRINTED FROM THIS FIT. The first
            # version of this block inverted the slope analytically and proposed
            # yaw -8.96; the scan below, which EVALUATES the metric instead of
            # linearising it, puts the optimum at -7.00 -- the opposite
            # direction. `R-2026-09-16-yawnotlateral` is precisely this mistake
            # made once already. The fit diagnoses WHICH parameter is wrong; only
            # the scan says what to set it to.
            print("    (magnitudes only — the correction comes from the scan below,"
                  " never from inverting this fit)")

    print("\n═══ JOINT SCAN yaw x lateral, minimising |median offset| ═══", flush=True)
    ys = np.arange(a.yaw_scan[0], a.yaw_scan[1] + 1e-9, a.yaw_scan[2])
    ls = np.arange(a.lateral_scan[0], a.lateral_scan[1] + 1e-9, a.lateral_scan[2])
    best = None
    grid = np.full((len(ys), len(ls)), np.nan)
    for i, y in enumerate(ys):
        for j, l in enumerate(ls):
            off, pr, cross, _ = measure(data, a.fx, a.height, a.horizon, y, l, a.ranges)
            if len(off) < 50:
                continue
            # the score is the WORST per-range median, so a calibration cannot win
            # by trading the near field against the far field -- which is how the
            # yaw error hid behind a good average for three renders.
            meds = [abs(np.median(v)) for v in pr if len(v) >= 8]
            if len(meds) < 3:
                continue
            s = float(max(meds))
            grid[i, j] = s
            if best is None or s < best[0]:
                best = (s, y, l, float(np.median(off)), float(cross.mean()))
        print(f"  yaw {y:+6.2f}  " + " ".join(
            f"{grid[i,j]:5.3f}" if np.isfinite(grid[i, j]) else "  .  "
            for j in range(len(ls))), flush=True)
    print("  lat:        " + " ".join(f"{l:+5.2f}" for l in ls))
    if best is None:
        print("⛔ no admissible cell"); return 1
    print(f"\n  ⭐ best: yaw {best[1]:+.2f}  lateral {best[2]:+.3f}  "
          f"-> worst-range |offset| {best[0]:.3f} m, median {best[3]:+.3f} m, "
          f"cross {100*best[4]:.1f}%")
    i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
    if i in (0, len(ys) - 1) or j in (0, len(ls) - 1):
        print("  ⛔ OPTIMUM ON THE EDGE OF THE SCAN — widen it; this is not a measurement")
    report(f"⇒ optimum {best[1]:+.2f}/{best[2]:+.3f}", best[1], best[2])
    if a.json:
        a.json.write_text(json.dumps(dict(
            yaw=float(best[1]), lateral=float(best[2]), horizon=a.horizon,
            worst_range_offset=float(best[0]), median_offset=float(best[3]),
            cross_rate=float(best[4]), ranges=a.ranges, n_frames=len(data),
            yaws=ys.tolist(), lats=ls.tolist(),
            grid=[[None if not np.isfinite(g) else float(g) for g in row]
                  for row in grid]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
