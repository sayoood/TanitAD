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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames-dir",
                    default="/root/trajdata/out_bev/2026-08-08_14-19-54-android/frames")
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--horizon", type=float, default=448.4)
    ap.add_argument("--yaw", type=float, default=-5.35)
    ap.add_argument("--ranges", type=float, nargs="+",
                    default=[10., 15., 20., 25., 30., 40.])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P0 = dict(RR.NOMINAL)
    P0.update(fx=a.fx, height=a.height, lateral=-0.126, yaw=np.deg2rad(a.yaw))
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

    cov = {x: [0, 0] for x in a.ranges}          # [covered, seen]
    place = {x: [] for x in a.ranges}
    lanew = {x: [] for x in a.ranges}
    corrw = {x: [] for x in a.ranges}
    clear_l = {x: [] for x in a.ranges}
    clear_r = {x: [] for x in a.ranges}
    cam_off, frames_ok = [], 0

    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            g = cv2.imread(str(src[i]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                pts = []
                for v in band:
                    x = fh / max(v - a.horizon, 1e-3)
                    for c in ridge_cols(g[int(round(v))], ridge_width_px(x, a.fx)):
                        pts.append((float(v), float(c)))
                if len(pts) >= 50:
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
                    if L and R:
                        (ml, bl), _ = max(L, key=lambda z: z[1])
                        (mr, br), _ = max(R, key=lambda z: z[1])
                        dm = abs(ml - mr)
                        # ⛔ THE SLOPE GATE IS NOT ENOUGH. A pair can pass |m_L-m_R| in
                        # the single-lane band and still not be two parallel boundaries:
                        # what matters is where the two lines MEET. Two road-parallel
                        # lines converge at the horizon, so their separation obeys
                        # Delta_u = dm*(v - v_h) and the implied width is constant with
                        # range. MEASURED without this check: the "lane" read 6.41 m at
                        # 10 m growing to 18.28 m at 40 m -- the right-hand line was not
                        # a lane boundary, and every placement number built on it was
                        # nonsense. Require range-constancy of the width instead.
                        wn = abs((mr - ml) * (a.horizon + fh / 12.0) + (br - bl)) \
                             * a.height / (fh / 12.0)
                        wf = abs((mr - ml) * (a.horizon + fh / 30.0) + (br - bl)) \
                             * a.height / (fh / 30.0)
                        parallel = wn > 0.5 and abs(np.log(wf / wn)) < 0.10
                        if 1.6 < dm < 3.2 and parallel and 2.6 < wn < 4.6:
                            frames_ok += 1
                            wlane = dm * a.height
                            cam_off.append(wlane / 2.0 * (ml + mr) / dm)
                            for x in a.ranges:
                                v = rows[x]
                                e = corridor_edges(fr, int(round(v * S)))
                                if e is None:
                                    continue
                                gl, gr = e[0] / S, e[1] / S
                                uL, uR = ml * v + bl, mr * v + br
                                mpp = x / a.fx
                                cov[x][1] += 1
                                # paint strictly inside the drawn band?
                                cols = ridge_cols(g[int(round(v))],
                                                  ridge_width_px(x, a.fx))
                                inside = [c for c in cols if gl + 2 < c < gr - 2]
                                if inside:
                                    cov[x][0] += 1
                                place[x].append((0.5 * (gl + gr) - 0.5 * (uL + uR)) * mpp)
                                lanew[x].append((uR - uL) * mpp)
                                corrw[x].append((gr - gl) * mpp)
                                clear_l[x].append((gl - uL) * mpp)
                                clear_r[x].append((uR - gr) * mpp)
        i += 1
        if i > max(want):
            break
    cap.release()

    print(f"\n{a.label or a.video}")
    print(f"{frames_ok} frames with a valid single-lane pair\n")
    print(f"  {'range':>6}{'covers paint':>14}{'corridor - lane centre':>24}"
          f"{'lane w':>9}{'ribbon w':>10}{'clear L':>9}{'clear R':>9}")
    out = {}
    for x in a.ranges:
        c, s = cov[x]
        if s < 20:
            print(f"  {x:4.0f} m{'-- too few --':>14}")
            continue
        med = lambda d: float(np.median(d[x])) if len(d[x]) >= 20 else float("nan")
        print(f"  {x:4.0f} m{100*c/s:>12.0f}%{med(place):>21.2f} m"
              f"{med(lanew):>8.2f}{med(corrw):>10.2f}{med(clear_l):>9.2f}{med(clear_r):>9.2f}")
        out[str(x)] = dict(covered_pct=100.0 * c / s, n=s, placement=med(place),
                           lane_w=med(lanew), corridor_w=med(corrw),
                           clear_left=med(clear_l), clear_right=med(clear_r))

    if cam_off:
        co = float(np.median(cam_off))
        veh = co + 0.126          # camera is 0.126 m right of the centreline (-0.126 flag)
        pl = float(np.median([v for x in a.ranges for v in place[x]])) if place else float("nan")
        print(f"\n  WHERE THE CAR ACTUALLY IS, from the paint alone (h cancels):")
        print(f"    camera   {co:+.2f} m from lane centre")
        print(f"    vehicle centreline {veh:+.2f} m  (camera + the 0.126 m mount offset)")
        print(f"    corridor drawn at  {pl:+.2f} m")
        print(f"    => the drawing is {pl - veh:+.2f} m from where the car is")
        out["placement_check"] = dict(camera=co, vehicle=veh, corridor=pl, error=pl - veh)
        if abs(pl - veh) > 0.10:
            print(f"    ⛔ that is a PLACEMENT DEFECT worth {abs(pl-veh):.2f} m, and in a lane")
            print(f"       with ~0.15 m of clearance it is the difference between touching")
            print(f"       the paint and not.")
        else:
            print(f"    ✅ within 0.10 m: the corridor is where the car is, and any contact")
            print(f"       with the paint is the car's position, not the drawing's.")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
