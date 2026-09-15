#!/usr/bin/env python3
"""Settle the horizon by making TWO independent trends vanish at once.

THE SITUATION. Two instruments now measure the horizon row, by the same kind of
criterion (drive a range-trend to zero) but on completely different physics:

    row-flow      q' = qA/(A - Dq) on 62,928 tracked points over 218 pairs.
                  Uses EGO MOTION (the odometer) and no paint.   -> 440 px
    lane width    Delta_y = Delta_u * h/(v - v_h) held range-independent.
                  Uses PAINT and no ego motion, and NO FOCAL LENGTH AT ALL. -> 464 px

24 px apart, and neither is flat at the other's value (at 464 the flow still trends
1.149; at 440 the width still trends 1.097). R-2026-09-13-horizon is explicit that the
wrong move here is to adjudicate on plausibility -- that is how 523.4 px survived. The
right move is to note that BOTH are zero-trend criteria on the SAME parameter and fit
them TOGETHER, and to report the disagreement rather than hide it inside a winner.

WHAT THE JOINT SOLUTION ALSO BUYS. The two curves carry different combinations:

    flow  gives  f*h(v_h)                 (metric, from the odometer)
    width gives  h(v_h) = 3.5 / r(v_h)    (metric, from Sayed's 3.5 m lane)
    =>    f(v_h) = f*h(v_h) * r(v_h) / 3.5

so the horizon fixes the WHOLE calibration -- and ``f`` then faces an orthogonal
check that entered neither fit: the S21 FE's lens is ~1442 px uncropped at 1920 wide,
and EIS can only ever crop IN, so ``f >= 1442`` is a hard floor if stabilisation is
active and ``f ~ 1442`` if it is not.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

F_UNCROPPED = 1442.0     # 26 mm-equiv lens, 4:3 diagonal 79.5 deg -> HFOV 66.7 deg at 1920 px


def flow_curve(npz, grid, row_max=832.0):
    z = np.load(npz, allow_pickle=True)
    V, DV, DD, PID = (z["V"].astype(float), z["DV"].astype(float),
                      z["DD"].astype(float), z["PID"])
    m = (V < row_max) & (DV > 0.5) & (DD > 0.05)
    V, DV, DD, PID = V[m], DV[m], DD[m], PID[m]
    out = {}
    for vh in grid:
        q = V - vh
        ok = q > 30
        if ok.sum() < 500:
            continue
        qq, dv, dd = q[ok], DV[ok], DD[ok]
        A = dd * qq * (qq + dv) / dv
        g = np.isfinite(A) & (A > 300) & (A < 12000)
        if g.sum() < 500:
            continue
        A, qq = A[g], qq[g]
        lo = qq <= np.percentile(qq, 33)
        hi = qq >= np.percentile(qq, 67)
        # ⚠️ q LARGE means NEAR. Report far/near so both curves share one orientation.
        out[float(vh)] = (float(np.log(np.median(A[lo]) / np.median(A[hi]))),
                          float(np.median(A)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairdata", default="/tmp/claude-0/-home-user-TanitAD/"
                    "d367c501-690c-51f0-8672-b9ccb83dd3ca/scratchpad/pairdata.npz")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--lane-m", type=float, default=3.5)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    grid = np.arange(410.0, 510.0, 2.0)
    fl = flow_curve(a.pairdata, grid)
    print(f"flow curve over {len(fl)} horizon values (62,928 points, 218 pairs)")

    # --- the width curve, recomputed here so both share one grid -------------- #
    import lane_width_far as LW
    import run_real as RR
    import cv2
    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    keep = []
    for r in usable:
        t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
        m = np.abs(t) < 0.4
        if m.sum() >= 3 and abs(np.rad2deg(np.polyfit(t[m], yw[m], 1)[0])) <= 0.6:
            keep.append(r)
    pairs = []
    for pi in np.linspace(0, len(keep) - 1, min(a.frames, len(keep))).astype(int):
        f = int(keep[pi]["frame"])
        g = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H = g.shape[0]
        rows = np.arange(int(0.50 * H), int(0.82 * H), 3)
        P = LW.ridge_points(g, rows)
        if len(P) < 40:
            continue
        lines, rem = [], P
        for k in range(4):
            lk, ink = LW.ransac_line(rem, seed=k + 1)
            if lk is None:
                break
            lines.append(lk); rem = rem[~ink]
            if len(rem) < 40:
                break
        if len(lines) < 2:
            continue
        vref = float(rows[int(0.75 * len(rows))])
        lines.sort(key=lambda L: L[0] * vref + L[1])
        for (m1, b1), (m2, b2) in zip(lines, lines[1:]):
            if m1 == m2:
                continue
            vx = (b2 - b1) / (m1 - m2)
            if not (300 < vx < 640):
                continue
            obs = [(float(v), abs(float((m2 * v + b2) - (m1 * v + b1)))) for v in rows
                   if abs((m2 * v + b2) - (m1 * v + b1)) >= 25]
            if len(obs) >= 25:
                pairs.append(np.asarray(obs, float))
    print(f"width curve from {len(pairs)} adjacent line pairs "
          f"in {min(a.frames, len(keep))} straight frames")

    def width_at(vh):
        """(log far/near trend, median Delta_y/h over SINGLE-LANE pairs)."""
        logs, seps = [], []
        for O in pairs:
            q = O[:, 0] - vh
            ok = q > 25
            if ok.sum() < 18:
                continue
            r = O[ok, 1] / q[ok]; vv = O[ok, 0]
            lo = vv <= np.percentile(vv, 33); hi = vv >= np.percentile(vv, 67)
            if lo.sum() < 5 or hi.sum() < 5:
                continue
            A_, B_ = float(np.median(r[hi])), float(np.median(r[lo]))
            if A_ <= 0 or B_ <= 0:
                continue
            logs.append(np.log(A_ / B_))          # hi rows = NEAR; so this is near/far
            seps.append(float(np.median(r)))
        if len(logs) < 15:
            return None
        seps = np.asarray(seps)
        prim = seps[(seps > 1.6) & (seps < 3.2)]
        if len(prim) < 15:
            return None
        # orient as far/near, matching flow_curve
        return -float(np.median(logs)), float(np.median(prim))

    print(f"\n  {'horizon':>8}{'flow trend':>12}{'width trend':>13}{'joint':>9}"
          f"{'f*h':>8}{'h (m)':>8}{'f (px)':>9}{'crop':>7}")
    rows_out, best = [], None
    for vh in grid:
        if vh not in fl:
            continue
        lf, fh = fl[vh]
        w = width_at(vh)
        if w is None:
            continue
        lw, r = w
        h = a.lane_m / r
        f = fh / h
        joint = lf ** 2 + lw ** 2
        rows_out.append(dict(horizon=float(vh), flow_trend=lf, width_trend=lw,
                             joint=joint, fh=fh, h=h, f=f))
        if best is None or joint < best["joint"]:
            best = rows_out[-1]
        if int(vh) % 6 == 0:
            print(f"  {vh:8.0f}{lf:+12.3f}{lw:+13.3f}{joint:9.4f}"
                  f"{fh:8.0f}{h:8.3f}{f:9.0f}{f/F_UNCROPPED:7.2f}")
    if best is None:
        print("  no horizon evaluated on both curves")
        return 1
    edge = best["horizon"] <= grid[0] + 2 or best["horizon"] >= grid[-1] - 2
    print(f"\n  JOINT ZERO  horizon {best['horizon']:.0f} px"
          + ("   ⛔ AT THE SCAN BOUNDARY — not a minimum" if edge else ""))
    print(f"    flow trend {best['flow_trend']:+.3f}   width trend {best['width_trend']:+.3f}"
          f"   (each alone would sit elsewhere; this is the compromise, and the")
    print(f"     residual trends are the honest statement of how far apart they are)")
    print(f"    f*h {best['fh']:.0f} px·m    h {best['h']:.3f} m    f {best['f']:.0f} px"
          f"    HFOV {np.rad2deg(2*np.arctan(1920/(2*best['f']))):.1f} deg"
          f"    crop {best['f']/F_UNCROPPED:.2f}x")
    print(f"\n  ORTHOGONAL CHECK — the lens. 26 mm-equiv gives f = {F_UNCROPPED:.0f} px "
          f"uncropped; EIS can only crop IN.")
    for tag, key in (("flow-only  ", "flow_trend"), ("width-only ", "width_trend")):
        c = min(rows_out, key=lambda d: abs(d[key]))
        print(f"    {tag} horizon {c['horizon']:.0f}  ->  h {c['h']:.3f} m   "
              f"f {c['f']:.0f} px   crop {c['f']/F_UNCROPPED:.2f}x"
              + ("   ⛔ crop < 1 is IMPOSSIBLE" if c['f'] < F_UNCROPPED * 0.995 else ""))
    if a.json:
        a.json.write_text(json.dumps(dict(best=best, curve=rows_out,
                                          f_uncropped=F_UNCROPPED), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
