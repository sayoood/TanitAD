#!/usr/bin/env python3
"""Horizon row and CAMERA HEIGHT from lane width alone — no focal length involved.

THE ALGEBRA THAT MAKES THIS WORK. A ground point at range ``x``, lateral ``y`` images
at ``u = cx + f(y-lat)/x`` and ``v = v_h + f*h/x``. Eliminate ``x``::

    Delta_y  =  Delta_u * h / (v - v_h)

**f is gone.** The lateral metric scale on the ground plane is ``h/(v - v_h)`` and
nothing else — which is the same degeneracy that has blocked this recording all along,
here turned into a tool instead of an obstacle:

  * the LANE WIDTH MUST NOT DEPEND ON RANGE, and only the true ``v_h`` makes it
    range-independent  ->  **v_h is measured, with no focal length and no ego motion**;
  * Sayed states the lane is **3.5 m** (standard French motorway)  ->  ``h`` follows as
    ``3.5 / median(Delta_u/(v - v_h))``  ->  **the camera height, which NO instrument in
    this session has yet measured** (peak-pairing, projection-association and the
    row-gap probe all failed, and ``lane_residual.py`` turned out to confirm whatever
    height it was handed);
  * ``f = (f*h)/h`` then closes the loop against the flow measurement of ``f*h``.

⛔ WHY THIS COULD NOT BE DONE EARLIER, AND WHAT CHANGED. ``lane_width.py`` ran this
idea in the near field only and gave a soft "~485". The horizon is BARELY OBSERVABLE
near the camera: at 10 m ``v - v_h`` is ~270 px, so a 20 px horizon error is 7 %; at
50 m it is ~54 px and the same error is 37 %. The far field is where the signal is,
and it was unreachable because ``lane_calib._ridge_points`` ramps its operator width
by the row's RANK IN THE BAND, not by the paint's apparent width — see
``overlay_far.ridge_width_px``. With that fixed, lines track to 50 m and the far field
does the work.

⚠️ CAVEATS, stated because they bound the answer:
  * road CROWN puts the two lines at slightly different heights (2.5 % crossfall over
    3.5 m = 0.09 m, ~5 % of h). That is a real bias on ``h``, not on ``v_h``.
  * 3.5 m is INHERITED from Sayed, not measured here. ``v_h`` does not depend on it;
    ``h`` is exactly proportional to it.
  * this measures where LANE LINES vanish, which my own R-2026-09-13-horizon retraction
    warns is not identical to where the GROUND PLANE vanishes. Constancy-of-width is a
    different criterion from VP-convergence, and it is the one that governs whether the
    BEV panel is metrically right — but it is not orthogonal to camber.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_real as RR                                               # noqa: E402
from overlay_far import ridge_cols                                  # noqa: E402

W_SCHEDULE = (3, 5, 8, 13, 20)   # merged, so no horizon is assumed to size the operator


def ridge_points(gray, rows):
    """Ridge columns over a row band, detected at SEVERAL operator widths and merged.

    Merging is what keeps the horizon out of the front end: the right operator width
    depends on the paint's apparent width, which depends on range, which depends on the
    very ``v_h`` being measured. Running all widths and merging removes the dependence.
    """
    pts = []
    for v in rows:
        row = gray[int(v)]
        found = []
        for w in W_SCHEDULE:
            found.extend(ridge_cols(row, w).tolist())
        if not found:
            continue
        found = np.sort(np.asarray(found))
        brk = np.flatnonzero(np.diff(found) > 3)
        for g in np.split(found, brk + 1):
            pts.append((float(v), float(g.mean())))
    return np.asarray(pts, float) if pts else np.empty((0, 2))


def ransac_line(P, tol=3.0, min_inl=18, min_span=110.0, iters=700, seed=0,
                slope_band=(0.15, 6.0)):
    rng = np.random.default_rng(seed)
    n = len(P)
    best, best_n = None, 0
    if n < min_inl:
        return None, None
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if abs(P[i, 0] - P[j, 0]) < min_span:
            continue
        m = (P[j, 1] - P[i, 1]) / (P[j, 0] - P[i, 0])
        if not (slope_band[0] <= abs(m) <= slope_band[1]):
            continue
        b = P[i, 1] - m * P[i, 0]
        d = np.abs(P[:, 1] - (m * P[:, 0] + b))
        k = int((d < tol).sum())
        if k > best_n:
            best_n, best = k, (m, b)
    if best is None or best_n < min_inl:
        return None, None
    m, b = best
    for _ in range(3):
        d = np.abs(P[:, 1] - (m * P[:, 0] + b))
        inl = P[d < tol]
        if len(inl) < min_inl or np.ptp(inl[:, 0]) < min_span:
            return None, None
        m, b = np.polyfit(inl[:, 0], inl[:, 1], 1)
    d = np.abs(P[:, 1] - (m * P[:, 0] + b))
    return (float(m), float(b)), d < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--lane-m", type=float, default=3.5)
    ap.add_argument("--row-lo", type=float, default=0.50, help="fraction of image height")
    ap.add_argument("--row-hi", type=float, default=0.82)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    keep = []
    for r in usable:
        t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
        m = np.abs(t) < 0.4
        if m.sum() >= 3 and abs(np.rad2deg(np.polyfit(t[m], yw[m], 1)[0])) <= 0.6:
            keep.append(r)

    per_frame = []          # (frame, [(v, du), ...])
    for pi in np.linspace(0, len(keep) - 1, min(a.frames, len(keep))).astype(int):
        f = int(keep[pi]["frame"])
        g = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        H = g.shape[0]
        rows = np.arange(int(a.row_lo * H), int(a.row_hi * H), 3)
        P = ridge_points(g, rows)
        if len(P) < 40:
            continue
        # ⚠️ EXTRACT UP TO FOUR LINES, NOT TWO. The first pass took the two strongest
        # and called their separation "the lane". MEASURED with two: Delta_y/h = 4.29,
        # which at h = 1.622 m is 7.0 m — TWO lane widths, because the two strongest
        # long solid lines on a motorway are the median edge and the right-hand edge,
        # not the two sides of the ego lane. Calling that 3.5 m returned h = 0.82 m and
        # a 35 deg HFOV, both physically impossible; picking the interpretation that
        # rescues the number would be exactly the plausibility-over-measurement failure
        # in R-2026-09-13-horizon. So: find several lines and report the distribution of
        # ADJACENT separations. A lane structure shows up as a cluster, not an assumption.
        lines, rem = [], P
        for k in range(4):
            lk, ink = ransac_line(rem, seed=k + 1)
            if lk is None:
                break
            lines.append(lk)
            rem = rem[~ink]
            if len(rem) < 40:
                break
        if len(lines) < 2:
            continue
        vref = float(rows[int(0.75 * len(rows))])          # a near row, well below the VP
        lines.sort(key=lambda L: L[0] * vref + L[1])       # left to right at that row
        pairs = []
        for (m1, b1), (m2, b2) in zip(lines, lines[1:]):
            if (m1 - m2) == 0:
                continue
            vx = (b2 - b1) / (m1 - m2)
            if not (300 < vx < 640):
                continue                                   # not road-parallel
            obs = [(float(v), abs(float((m2 * v + b2) - (m1 * v + b1)))) for v in rows
                   if abs((m2 * v + b2) - (m1 * v + b1)) >= 25]
            if len(obs) >= 25:
                pairs.append((obs, float(vx)))
        for obs, vx in pairs:
            per_frame.append((f, obs, vx))

    print(f"{len(per_frame)} frames of {min(a.frames, len(keep))} gave a clean line PAIR")
    if len(per_frame) < 20:
        print("  not enough — the width criterion cannot be evaluated")
        return 1

    vps = np.array([p[2] for p in per_frame])
    print(f"  line-pair intersection row (the LANE VP): median {np.median(vps):.1f}"
          f"   robust sd {1.4826*np.median(np.abs(vps-np.median(vps))):.1f}"
          f"   10-90% [{np.percentile(vps,10):.0f}, {np.percentile(vps,90):.0f}]")

    allobs = [(v, du, k) for k, (_, obs, _) in enumerate(per_frame) for v, du in obs]
    V = np.array([o[0] for o in allobs]); DU = np.array([o[1] for o in allobs])
    K = np.array([o[2] for o in allobs])

    def flatness(vh, keys=None):
        """How much the implied lateral scale still trends with image row.

        ⚠️ PER PAIR, then aggregated. Pooling every observation and splitting the pool
        into near and far thirds is wrong here: the near third and the far third would
        be drawn from DIFFERENT line pairs (a two-lane span survives further up the
        image than a one-lane span), so the ratio would measure which pairs reach which
        rows, not whether the width is range-independent. Each pair is its own control.
        """
        logs = []
        ks = range(len(per_frame)) if keys is None else keys
        for k in ks:
            sel = (K == k) & (V - vh > 25.0)
            if sel.sum() < 18:
                continue
            rr = DU[sel] / (V[sel] - vh); vv = V[sel]
            lo = vv <= np.percentile(vv, 33); hi = vv >= np.percentile(vv, 67)
            if lo.sum() < 5 or hi.sum() < 5:
                continue
            a_, b_ = float(np.median(rr[hi])), float(np.median(rr[lo]))
            if a_ > 0 and b_ > 0:
                logs.append(np.log(a_ / b_))
        if len(logs) < 15:
            return np.inf, np.nan
        med = float(np.median(logs))
        return abs(med), float(np.exp(med))

    print(f"\n  {'horizon':>9}{'near/far width ratio':>24}{'|log|':>10}")
    grid = np.arange(400.0, 560.0, 2.0)
    scored = []
    for vh in grid:
        s, ratio = flatness(vh)
        scored.append((s, vh, ratio))
        if int(vh) % 10 == 0 and np.isfinite(s):
            flag = ""
            print(f"  {vh:9.0f}{ratio:24.3f}{s:10.3f}{flag}")
    scored = [t for t in scored if np.isfinite(t[0])]
    s_best, vh_best, ratio_best = min(scored)
    edge = vh_best <= grid[0] + 2 or vh_best >= grid[-1] - 2
    print(f"\n  BEST horizon {vh_best:.0f} px   near/far width ratio {ratio_best:.3f}"
          + ("   ⛔ AT THE SCAN BOUNDARY — not a minimum, do not quote" if edge else ""))
    if edge:
        return 2

    q = V - vh_best
    ok = q > 25.0
    r = DU[ok] / q[ok]

    # WHICH separations are these? One value per adjacent PAIR, then the histogram.
    per_pair = []
    for k in range(len(per_frame)):
        sel = (K == k) & ok
        if sel.sum() >= 10:
            per_pair.append(float(np.median(DU[sel] / q[sel])))
    per_pair = np.asarray(per_pair)
    print(f"\n  SEPARATION OF ADJACENT LINE PAIRS, in units of Delta_y/h "
          f"(n = {len(per_pair)} pairs)")
    hist, edges = np.histogram(per_pair, bins=np.arange(0.0, 9.01, 0.25))
    for c, lo_, hi_ in zip(hist, edges[:-1], edges[1:]):
        if c:
            bar = "#" * int(round(40 * c / max(hist.max(), 1)))
            print(f"    {lo_:4.2f}-{hi_:4.2f}  {c:4d}  {bar}")
    print(f"    a 3.5 m lane at h=1.62 m would sit at 2.16; at h=1.30 m, 2.69;"
          f"  TWO lanes (7.0 m) at h=1.62 m sit at 4.32")
    prim = per_pair[(per_pair > 1.6) & (per_pair < 3.2)]
    print(f"    single-lane cluster (1.6-3.2): n {len(prim)}"
          + (f"   median {np.median(prim):.3f}" if len(prim) else ""))

    if len(prim) >= 20:
        r_lane = float(np.median(prim))
        h_hat = a.lane_m / r_lane
        print(f"\n  using the SINGLE-LANE cluster: h = {a.lane_m} / {r_lane:.4f} = "
              f"{h_hat:.3f} m")
    else:
        h_hat = a.lane_m / float(np.median(r))
        print(f"\n  ⛔ no single-lane cluster; falling back to all pairs, which is "
              f"NOT a lane width: h = {h_hat:.3f} m — do not quote")

    # frame-cluster bootstrap over the two quantities together
    rng = np.random.default_rng(0)
    nf = len(per_frame)
    bh, bv = [], []
    for _ in range(200):
        pick = rng.integers(0, nf, nf).tolist()
        cand = [(flatness(vh, keys=pick)[0], vh) for vh in grid[::2]]
        cand = [c for c in cand if np.isfinite(c[0])]
        if not cand:
            continue
        _, vhb = min(cand)
        vals = []
        for k in pick:
            sel = (K == k) & (V - vhb > 25.0)
            if sel.sum() >= 10:
                vals.append(float(np.median(DU[sel] / (V[sel] - vhb))))
        vals = [x for x in vals if 1.6 < x < 3.2]
        if len(vals) < 15:
            continue
        bv.append(vhb); bh.append(a.lane_m / float(np.median(vals)))
    if bh:
        print(f"  frame-cluster bootstrap 95% CI   horizon "
              f"[{np.percentile(bv,2.5):.0f}, {np.percentile(bv,97.5):.0f}] px"
              f"   height [{np.percentile(bh,2.5):.3f}, {np.percentile(bh,97.5):.3f}] m"
              f"   (n={len(bh)})")

    print(f"\n  CLOSING THE LOOP with the flow measurement of f*h:")
    for fh in (2444., 2666., 2900.):
        print(f"    f*h {fh:.0f} px·m  ->  f = {fh/h_hat:7.1f} px   "
              f"(HFOV {np.rad2deg(2*np.arctan(1920/(2*fh/h_hat))):.1f} deg, "
              f"crop {fh/h_hat/1442:.2f}x of the 26 mm-equiv lens)")

    if a.json:
        a.json.write_text(json.dumps(dict(
            n_frames=len(per_frame), horizon=float(vh_best), ratio=float(ratio_best),
            height_m=float(h_hat), lane_m=a.lane_m,
            lane_vp_row_median=float(np.median(vps)),
            ci_horizon=[float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))] if bh else None,
            ci_height=[float(np.percentile(bh, 2.5)), float(np.percentile(bh, 97.5))] if bh else None,
        ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
