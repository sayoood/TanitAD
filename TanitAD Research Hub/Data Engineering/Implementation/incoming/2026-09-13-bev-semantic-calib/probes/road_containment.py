#!/usr/bin/env python3
"""ROAD CONTAINMENT — one metric for the whole calibration, with a known target.

⛔ WHY THIS REPLACES EVERYTHING BEFORE IT. Sayed, after five renders:
*"can you check on your own and also consider all frames, you need to introduce a
metric which represents non-leaving-the-road, which [is] a clear simple verified
goal, since we know the future."*

He is right, and the diagnosis of my own failure is in the record above: I tuned
one parameter at a time, each against a bespoke instrument, on a handful of
ranges, and each "fix" traded one error for another —

  * `R-2026-09-15-oneside`   — measured only the LEFT boundary, so a placement
                               error was invisible;
  * `R-2026-09-16-yawnotlateral` — measured only 7–12 m, so a YAW error was
                               fitted with the LATERAL parameter and made worse
                               at every longer range.

**Both failures are the same failure: no single quantity that the whole
calibration had to answer to.** A metric fixes that, and this one has three
properties the ad-hoc probes never had.

1. **IT IS THE GOAL, NOT A PROXY.** "The drawn corridor stays on the road."
   Not a clearance, not an angle, not a vanishing point — the thing being complained
   about.

2. **ITS CORRECT VALUE IS KNOWN — and that is what "since we know the future"
   buys.** The ribbon is drawn around the car's ACTUAL RECORDED future path. The
   car did not leave the road. Therefore **a correct calibration must score ~100 %**,
   and any shortfall is calibration error rather than something to be interpreted.
   Every earlier instrument produced a number with no known right answer, which is
   precisely why I could keep talking myself into whichever reading preserved the
   status quo.

3. **IT IS SCORED OVER EVERY FRAME AND EVERY RANGE**, and reported per range, so a
   calibration that wins near and loses far — which is what a yaw error does, and
   what I shipped three times — cannot hide in an average.

THE SURFACE. Drivable road is NEAR-GREY. Measured on this recording: asphalt
S = 8, lane paint S = 4, pale shoulder S = 79, bank S = 95, sky S = 139. A single
saturation threshold separates the carriageway from the shoulder that the corridor
has been spilling onto, and it puts the PAINT on the road side where it belongs.

⚠️ WHAT COULD GAME IT, AND WHY IT DOES NOT. Ranges are fixed in METRES and the
ribbon is a fixed 1.855 m, so the projection cannot win by shrinking or by
retreating to the near field. Dropping the horizon does move samples toward the
wide near field, so the horizon is scanned over a PHYSICALLY ADMISSIBLE BAND and
the surface is printed rather than blindly maximised — an optimum at the edge of
its own scan is not a measurement, a rule this document has had to learn five
separate times.
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

DS = 2                      # mask downsample; 0.5 px at source is far below the signal


def drivable_mask(img, a_max=8, b_min=-22, b_max=6, legacy=False,
                  s_max=35, v_min=60):
    """Carriageway (asphalt + paint) by CHROMATICITY, in CIELAB.

    ⛔ THE HSV VERSION CALLED SHADOWED ASPHALT "NOT ROAD", AND THAT INVENTED A
    CALIBRATION DEFECT. Shadow on this road is lit by sky, so it is BLUE, and
    HSV's saturation is ``(max-min)/max`` -- a dark blue-grey pixel is therefore
    highly "saturated" even though it is plainly grey road. MEASURED 2026-09-19,
    frame 1860: shadowed asphalt BGR (98,79,58) reads **S = 104**, against the
    ``S < 35`` written for the sunlit case (S = 8). The consequence was not a
    small bias: the six worst-scoring frames in the whole recording were frames
    whose ribbon is squarely BETWEEN THE PAINTED LINES, failed on shadow.

    Lab separates it cleanly because shadow changes L* and leaves chromaticity
    alone. MEASURED over hand-placed regions of frame 1860 (n = 4.4k-68k each):

        region              L*        a*        b*     kept by this test
        sunlit asphalt     119      -1.3      -3.0        99.8 %
        shadowed asphalt    76      -1.3     -12.1       100.0 %   <-- was LOST
        lane paint         155      -2.6      +0.6        96.5 %
        vegetation bank    107      -7.9     +23.6         0.7 %
        concrete barrier   193      +2.0     +21.3         0.0 %   <-- was KEPT
        sky                183      -6.5     -33.2         0.0 %

    a* is near zero for road whatever the illumination; the verge is warm
    (b* > +20) and the sky is deep blue (b* = -33), so a two-sided b* window
    takes both. The barrier going from accepted to rejected matters as much as
    the shadow: it was padding the margin on the right-hand side.

    ``legacy=True`` restores the HSV test so the change stays auditable.
    """
    if legacy:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[:, :, 1] < s_max) & (hsv[:, :, 2] > v_min)).astype(np.uint8)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
        a = lab[:, :, 1] - 128
        b = lab[:, :, 2] - 128
        m = ((np.abs(a) < a_max) & (b > b_min) & (b < b_max)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return m[::DS, ::DS]


def load(n, s_max=35, v_min=60, legacy=False):
    """Masks and future paths for n frames spread over the whole recording."""
    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = RR.RUN / "frames"
    have = {int(p.stem) for p in fdir.glob("*.jpg")}
    cand = [f for f, r in sorted(recs.items())
            if r["complete"] and f in have and r["speed_ms"] > 8]
    pick = [cand[i] for i in np.linspace(0, len(cand) - 1,
                                         min(n, len(cand))).astype(int)]
    out = []
    for f in pick:
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        r = recs[f]
        px, py = np.asarray(r["x"], float), np.asarray(r["y"], float)
        fwd = px > 0.0
        out.append((drivable_mask(img, legacy=legacy, s_max=s_max,
                                  v_min=v_min), px[fwd], py[fwd], f))
    return out


def score(data, fx, height, horizon, yaw, lateral, ranges, half_w,
          per_range=False):
    """Fraction of (frame, range) samples with BOTH ribbon edges on the road.

    The ribbon's centre at range x is the car's own recorded lateral there, so the
    only thing under test is the projection.
    """
    P = dict(RR.NOMINAL)
    P.update(fx=fx, height=height, lateral=lateral, yaw=np.deg2rad(yaw))
    P["pitch"] = LS.pitch_for_horizon(P, horizon)
    LON = float(P.get("longitudinal", 0.0))
    hit = np.zeros(len(ranges)); tot = np.zeros(len(ranges))
    for m, px, py, *_ in data:
        H, W = m.shape
        pts = []
        for x in ranges:
            xc = x + LON                      # project_ground's x is REAR-AXLE based
            yc = float(np.interp(xc, px, py))
            pts.append([xc, yc + half_w]); pts.append([xc, yc - half_w])
        uv = BC.project_ground(np.asarray(pts, float), P)
        for k in range(len(ranges)):
            a, b = uv[2 * k], uv[2 * k + 1]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            ca, ra = int(a[0] / DS), int(a[1] / DS)
            cb, rb = int(b[0] / DS), int(b[1] / DS)
            if not (0 <= ra < H and 0 <= rb < H and 0 <= ca < W and 0 <= cb < W):
                continue
            tot[k] += 1
            if m[ra, ca] and m[rb, cb]:
                hit[k] += 1
    ok = tot > 0
    if not ok.any():
        return (0.0, hit, tot) if per_range else 0.0
    overall = hit[ok].sum() / tot[ok].sum()
    return (overall, hit, tot) if per_range else overall


def row_margin_m(mask_row, col, height, v_src, horizon):
    """Signed LATERAL margin, in metres, from one ribbon edge to the road edge.

    Positive = this much road still to the outside of the edge; negative = the
    edge is this far INTO the verge, which is the thing being complained about.

    ⚠️ THE SCALE IS EXACT AND NEEDS NO EXTRA CALIBRATION. With
    ``u = cx - f(y-lat)/x`` and ``x = f*h/(v-v_h)``, the focal length cancels:

        du/dy = -f/x = -(v - v_h)/h    =>    |dy/du| = h / (v - v_h)

    so a pixel at image row ``v`` is worth ``h/(v-v_h)`` metres of lateral, and
    the metre value of a margin depends on the camera HEIGHT alone.

    ⛔ WHY THIS IS A ROW SCAN AND NOT A DISTANCE TRANSFORM. A 2-D transform
    returns the nearest non-road pixel in ANY direction, and near the horizon the
    road band is only a few rows tall, so the nearest exit is upward -- it would
    report a lateral margin that is really a longitudinal one, shrinking with
    range for a ribbon that is perfectly placed. The complaint is lateral, so the
    measurement is lateral.
    """
    W = len(mask_row)
    c = int(round(col))
    if not (0 <= c < W):
        return None
    mpp = height / max(v_src - horizon, 1e-6) * DS
    if mask_row[c]:                      # on road: distance out to the nearer edge
        li = c
        while li > 0 and mask_row[li - 1]:
            li -= 1
        ri = c
        while ri < W - 1 and mask_row[ri + 1]:
            ri += 1
        return float(min(c - li, ri - c) * mpp)
    d = 1                                 # off road: distance back to the road
    while d < W:
        if c - d >= 0 and mask_row[c - d]:
            break
        if c + d < W and mask_row[c + d]:
            break
        d += 1
    return float(-d * mpp)


def margins(data, fx, height, horizon, yaw, lateral, ranges, half_w):
    """Per-frame MINIMUM lateral margin over all ranges, and the per-sample values.

    ⛔ THIS EXISTS BECAUSE THE SAMPLE AVERAGE AND THE PER-FRAME MINIMUM ARE
    DIFFERENT QUESTIONS AND I REPORTED THE WRONG ONE. ``score`` above answers
    "what fraction of samples are on the road" -- 97.9 % on v6. Sayed is not
    looking at a sample average; he is looking at ONE FRAME, and a frame is bad
    if ANY part of the ribbon leaves the road. A per-frame minimum is the
    quantity his complaint is about, and it can be far worse than the mean.

    ⚠️ It reads NOTHING from the rendered video. The first attempt at this metric
    recovered the ribbon from the render with ``corridor_edges``, which locked
    onto roadside FOLIAGE and invented margins of -6 to -16 m on frames that are
    fully on-road (see that function's docstring). The projection already gives
    the edge position exactly, so the detector was never needed.
    """
    P = dict(RR.NOMINAL)
    P.update(fx=fx, height=height, lateral=lateral, yaw=np.deg2rad(yaw))
    P["pitch"] = LS.pitch_for_horizon(P, horizon)
    LON = float(P.get("longitudinal", 0.0))
    per_frame = []
    per_range = [[] for _ in ranges]
    for m, px, py, *rest in data:
        H, W = m.shape
        pts = []
        for x in ranges:
            xc = x + LON
            yc = float(np.interp(xc, px, py))
            pts.append([xc, yc + half_w]); pts.append([xc, yc - half_w])
        uv = BC.project_ground(np.asarray(pts, float), P)
        worst = None
        for k in range(len(ranges)):
            vals = []
            for e in (uv[2 * k], uv[2 * k + 1]):
                if not np.isfinite(e).all():
                    continue
                r = int(e[1] / DS)
                if not (0 <= r < H):
                    continue
                g = row_margin_m(m[r], e[0] / DS, height, float(e[1]), horizon)
                if g is not None:
                    vals.append(g)
            if not vals:
                continue
            mk = min(vals)
            per_range[k].append(mk)
            worst = mk if worst is None else min(worst, mk)
        if worst is not None:
            per_frame.append((rest[0] if rest else -1, worst))
    return np.asarray(per_frame, float), [np.asarray(v, float) for v in per_range]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--vehicle-width", type=float, default=1.855)
    ap.add_argument("--ranges", type=float, nargs="+",
                    default=[10., 15., 20., 25., 30., 35., 40.])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--s-max", type=float, default=35.0)
    ap.add_argument("--v-min", type=float, default=60.0)
    ap.add_argument("--yaw-scan", type=float, nargs=3, default=[-9.0, -4.0, 0.25],
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--horizon-scan", type=float, nargs=3, default=[436.0, 476.0, 4.0],
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--lateral-scan", type=float, nargs=3, default=[-0.5, 0.3, 0.05],
                    metavar=("LO", "HI", "STEP"))
    ap.add_argument("--baseline", type=float, nargs=3, default=[-7.01, 448.4, -0.088],
                    metavar=("YAW", "HORIZON", "LATERAL"))
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()
    hw = a.vehicle_width / 2.0

    print(f"loading {a.n} frames and building drivable masks "
          f"(S < {a.s_max}, V > {a.v_min})...")
    data = load(a.n, a.s_max, a.v_min)
    print(f"{len(data)} frames × {len(a.ranges)} ranges = "
          f"{len(data)*len(a.ranges)} samples per evaluation\n")

    def report(tag, yaw, hz, lat):
        s, hit, tot = score(data, a.fx, a.height, hz, yaw, lat, a.ranges, hw, True)
        cols = "  ".join(f"{r:.0f}m {100*h/max(t,1):5.1f}%"
                         for r, h, t in zip(a.ranges, hit, tot))
        print(f"  {tag:<34} {100*s:5.1f}%   {cols}")
        return s

    print("ON-ROAD RATE — both ribbon edges on drivable surface (target 100 %)")
    print(f"  {'calibration':<34} {'all':>6}   per range")
    by, bh, bl = a.baseline
    report(f"baseline yaw {by} hz {bh} lat {bl}", by, bh, bl)
    for nm, y, h_, l_ in (("shipped v1 (yaw -5.35, lat -0.126)", -5.35, 448.4, -0.126),
                          ("v3 (yaw -5.35, lat -0.41)", -5.35, 448.4, -0.41),
                          ("v4 (yaw -7.39, lat -0.088)", -7.39, 448.4, -0.088)):
        report(nm, y, h_, l_)

    print("\n═══ JOINT SCAN over yaw × horizon (lateral held at baseline) ═══")
    ys = np.arange(*[a.yaw_scan[0], a.yaw_scan[1] + 1e-9, a.yaw_scan[2]][:3]) \
        if False else np.arange(a.yaw_scan[0], a.yaw_scan[1] + 1e-9, a.yaw_scan[2])
    hs = np.arange(a.horizon_scan[0], a.horizon_scan[1] + 1e-9, a.horizon_scan[2])
    grid = np.zeros((len(hs), len(ys)))
    for i, hz in enumerate(hs):
        for j, yw in enumerate(ys):
            grid[i, j] = score(data, a.fx, a.height, hz, yw, bl, a.ranges, hw)
    print("      yaw ->  " + " ".join(f"{y:6.2f}" for y in ys))
    for i, hz in enumerate(hs):
        print(f"  hz {hz:6.1f}  " + " ".join(f"{100*g:6.1f}" for g in grid[i]))
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    print(f"\n  ⭐ best on this grid: horizon {hs[i]:.1f}, yaw {ys[j]:.2f}  "
          f"-> {100*grid[i,j]:.1f} %")
    edge = (i in (0, len(hs) - 1)) or (j in (0, len(ys) - 1))
    if edge:
        print("  ⛔ OPTIMUM IS ON THE EDGE OF THE SCAN — widen it; this is not a measurement")

    print(f"\n═══ LATERAL scan at horizon {hs[i]:.1f}, yaw {ys[j]:.2f} ═══")
    ls = np.arange(a.lateral_scan[0], a.lateral_scan[1] + 1e-9, a.lateral_scan[2])
    lv = [score(data, a.fx, a.height, hs[i], ys[j], l, a.ranges, hw) for l in ls]
    print("  " + " ".join(f"{l:+5.2f}" for l in ls))
    print("  " + " ".join(f"{100*v:5.1f}" for v in lv))
    k = int(np.argmax(lv))
    print(f"\n  ⭐ best lateral {ls[k]:+.3f} -> {100*lv[k]:.1f} %"
          + ("   ⛔ ON THE SCAN EDGE" if k in (0, len(ls) - 1) else ""))

    print(f"\n═══ FINAL ═══")
    report("optimum", ys[j], hs[i], ls[k])
    if a.json:
        a.json.write_text(json.dumps(dict(
            yaw=float(ys[j]), horizon=float(hs[i]), lateral=float(ls[k]),
            score=float(score(data, a.fx, a.height, hs[i], ys[j], ls[k], a.ranges, hw)),
            ranges=a.ranges, n_frames=len(data),
            grid=[[float(g) for g in row] for row in grid],
            yaws=[float(y) for y in ys], horizons=[float(h) for h in hs]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
