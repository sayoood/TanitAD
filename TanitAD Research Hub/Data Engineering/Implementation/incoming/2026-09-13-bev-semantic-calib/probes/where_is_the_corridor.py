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
