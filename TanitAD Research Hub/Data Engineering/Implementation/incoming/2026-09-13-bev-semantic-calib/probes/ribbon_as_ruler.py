#!/usr/bin/env python3
"""Where is the paint, measured with the DRAWN RIBBON as the ruler.

⛔ WHY THIS EXISTS. Sayed, looking at the delivered render: *"the trajectory still
[is] leaving the road, knowing that ego [is] driving between the road markings"*.
He is right, and my Part 22–23 analysis missed it because **it only ever measured
the LEFT line** — the clean one — and read the resulting asymmetry as the car
sitting right of centre rather than as the drawing being displaced.

The asymmetry was in my own numbers the whole time: left boundary **+2.05 m**,
right boundary **−1.10 m** from the vehicle centreline (§116/§117). If the car
drives centred between the markings those must be symmetric. They differ by
**0.95 m**.

THE INSTRUMENT. Every previous probe converted pixels to metres through
`project_ground`, so every one of them inherited whatever is wrong with `lateral`,
`h`, `f` and the horizon — including the very parameter under suspicion. This one
does not convert at all:

    the drawn ribbon is a known 1.855 m wide, so measure everything in units of
    its own width at the same image row, then multiply by 1.855.

**No focal length, no camera height, no horizon, no lateral offset.** The only
inputs are two image widths in the same row and the vehicle's width, and the
ribbon is exactly as wide as the renderer believes the car to be — which is the
thing we want the paint measured against.

WHAT IT ANSWERS.

    left line  and  right line, in metres from the RIBBON'S OWN CENTRE

If the car is centred between the markings, those two must be equal and opposite.
Any offset is the drawing's lateral error **in metres**, and it is exactly the
correction to apply to `--lateral-offset`.

⚠️ The ribbon follows the FUTURE PATH, so its centre is at `path_y(x)`, not at 0.
Near rows are used (the path lateral is smallest there) and `path_y` is subtracted
from the records, so what comes out is the mount error and not the driving.

⚠️ The lane width falls out too, as `(left − right)`, and it is likewise
calibration-free — a cross-check on §117's 3.15 m that shares none of its inputs.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--horizon", type=float, default=448.4)
    ap.add_argument("--yaw", type=float, default=-5.35)
    ap.add_argument("--lateral", type=float, default=-0.126)
    ap.add_argument("--vehicle-width", type=float, default=1.855)
    ap.add_argument("--ranges", type=float, nargs="+", default=[9., 11., 13., 16.])
    ap.add_argument("--paint-k", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    # P is used ONLY to pick which rows to read and to read path_y — never to
    # convert a column to a lateral. That conversion is what this probe avoids.
    P = dict(RR.NOMINAL)
    P.update(fx=a.fx, height=a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)
    LON = float(P.get("longitudinal", 0.0))
    rows = {}
    for x in a.ranges:
        uv = BC.project_ground(np.array([[x + LON, 0.0]]), P)
        if np.isfinite(uv).all():
            rows[x] = float(uv[0, 1])

    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    src = sorted(pathlib.Path(a.frames_dir).glob("*.jpg"))
    cap = cv2.VideoCapture(a.video)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hi = min(nfr, len(src)) - 30
    cand = [f for f, r in sorted(recs.items()) if r["complete"] and 30 <= f < hi]
    want = set(np.asarray(cand)[np.linspace(0, len(cand) - 1,
                                            min(a.n, len(cand))).astype(int)].tolist())
    print(f"{len(want)} frames; ribbon width = {a.vehicle_width} m is the ONLY "
          f"physical input\n")

    REL = []                       # paint offsets in metres from the ribbon centre
    widths = []                    # ribbon width in px, for the record
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok or i > max(want):
            break
        if i in want:
            g = cv2.imread(str(src[i]), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                r = recs[i]
                px, py = np.asarray(r["x"], float), np.asarray(r["y"], float)
                fwd = px > 0.0
                for x, v in rows.items():
                    e = corridor_edges(fr, int(round(v * S)))
                    if e is None:
                        continue
                    gl, gr = e[0] / S, e[1] / S
                    w_px = gr - gl
                    if w_px < 40:              # too narrow to be a reliable ruler
                        continue
                    widths.append(w_px)
                    mid = 0.5 * (gl + gr)
                    # metres per pixel FROM THE RIBBON, at this row, this frame
                    mpp = a.vehicle_width / w_px
                    path_y = float(np.interp(x + LON, px[fwd], py[fwd]))
                    vi = int(round(v))
                    if not (0 <= vi < g.shape[0]):
                        continue
                    lo = max(0, int(mid - 3.2 / mpp))
                    hiC = min(g.shape[1], int(mid + 3.2 / mpp))
                    cols = ridge_cols(g[vi], ridge_width_px(x, a.fx), lo=lo, hi=hiC)
                    # ⚠️ PAINT GATE, and it stays calibration-free: the threshold
                    # is the row's OWN robust statistics inside the same window.
                    # Without it this histogram is flat -- it counts road texture,
                    # and the peak-picker then returns whatever broad structure is
                    # biggest, which is how R-2026-09-15-seam happened.
                    seg = g[vi, lo:hiC].astype(np.float32)
                    if seg.size < 40:
                        continue
                    lvl = float(np.median(seg))
                    sig = max(1.4826 * float(np.median(np.abs(seg - lvl))), 3.0)
                    cols = [c for c in cols
                            if float(g[vi, int(c)]) > lvl + a.paint_k * sig]
                    for c in cols:
                        # +LEFT: image columns grow rightward, so negate.
                        # subtract path_y so a turn does not read as a mount error
                        REL.append(-(float(c) - mid) * mpp + path_y)
        i += 1
    cap.release()

    REL = np.asarray(REL, float)
    print(f"ribbon width {np.median(widths):.0f} px median; {len(REL)} paint detections\n")
    if len(REL) < 400:
        print("⛔ too few")
        return 1
    edges = np.arange(-3.2, 3.21, 0.10)
    hist, _ = np.histogram(REL, bins=edges)
    top = max(hist.max(), 1)
    print("  metres from the RIBBON'S CENTRE (+ = LEFT)        detections")
    for c, e0 in zip(hist, edges[:-1]):
        if c >= 0.02 * top:
            mark = "  <-- ribbon edge" if abs(abs(e0 + 0.05) - a.vehicle_width / 2) < 0.06 else ""
            print(f"   {e0:+5.2f} .. {e0+0.10:+5.2f}  {c:6d}  "
                  f"{'#' * int(round(46 * c / top))}{mark}")
    # the two lines: strongest peak each side, at least half a ribbon out
    out = {}
    hw = a.vehicle_width / 2.0
    # ⚠️ THE RIGHT SEARCH IS WINDOWED, AND THAT IS A DECLARED PRIOR, NOT A FIT.
    # Unwindowed, the right "line" comes back at -2.55 m, which is the SHOULDER
    # EDGE LINE AND GRAVEL -- the same mass that produced `R-2026-09-15-seam` and
    # §107's phantom 0.60 m peak spacing. The ego lane's right boundary is dashed,
    # so it is weak in a pooled histogram, and a peak-picker will always prefer
    # the continuous structure beyond it. The window says "inside 2.2 m", which
    # the §116 dash-periodicity result (-1.10 m, three stretches, controls
    # passing) independently supports. Both candidates are reported so the
    # reader sees what was excluded.
    for side, sel in (("left", (edges[:-1] + 0.05) > 0.6),
                      ("right", ((edges[:-1] + 0.05) < -0.95)
                                & ((edges[:-1] + 0.05) > -2.2))):
        h2 = np.where(sel, hist, 0)
        j = int(np.argmax(h2))
        yc = edges[j] + 0.05
        near = REL[np.abs(REL - yc) < 0.30]
        val = float(np.median(near)) if len(near) >= 30 else float(yc)
        out[side] = dict(y=val, n=int(len(near)))
        print(f"\n  {side:>5} line at {val:+.2f} m from the ribbon centre "
              f"(ribbon edge is at {hw if side=='left' else -hw:+.2f})")
    far = REL[(REL < -2.2) & (REL > -3.2)]
    if len(far) >= 50:
        print(f"\n  (excluded by the window: a larger mass at "
              f"{np.median(far):+.2f} m -- shoulder edge line + gravel)")
    L, R = out["left"]["y"], out["right"]["y"]
    print(f"\n  ⭐ lane width  = {L - R:.2f} m          (calibration-free)")
    print(f"  ⭐ ribbon centre sits {-(L + R) / 2:+.2f} m from the lane centre"
          f"   (+ = ribbon too far LEFT, − = too far RIGHT)")
    print(f"     clearance left  {L - hw:+.2f} m")
    print(f"     clearance right {-R - hw:+.2f} m")
    corr = (L + R) / 2.0
    print(f"\n  ⇒ if the car drives centred between the markings, --lateral-offset")
    print(f"     should change by {corr:+.2f} m:  {a.lateral:+.3f}  ->  "
          f"{a.lateral + corr:+.3f} m")
    out.update(lane_width=L - R, ribbon_offset=-(L + R) / 2.0,
               clear_left=L - hw, clear_right=-R - hw,
               lateral_now=a.lateral, lateral_suggested=a.lateral + corr,
               n=int(len(REL)), ribbon_px=float(np.median(widths)))
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
