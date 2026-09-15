#!/usr/bin/env python3
"""Per-frame horizon from GROUND FLOW — the deployable version.

WHY THIS AND NOT THE LANE-BASED ONE
-----------------------------------
Part 5 measured that the camera's pitch relative to the road swings 1.75 deg
(p10-p90, 52 px of horizon) during the clip, and that giving each frame its own
horizon removes 59% of the residual. That measurement used the solid left lane line,
which is fine as a measurement and useless as a deployable estimator: it needs paint,
it needs the paint to be SOLID (the dashed side yields 8 samples a frame against 19),
and it produced only 39 usable frames out of the clip.

Ground flow needs no markings. With ``A = f*h`` already known from the odometer, the
row-flow relation inverts for the horizon in CLOSED FORM, once per tracked point::

    A*dv = D (v - v_h)(v' - v_h)
    =>  v_h = [ (v+v') - sqrt( (v+v')^2 - 4( v v' - A dv / D ) ) ] / 2

(the lower root: the horizon lies above the ground points, i.e. at a smaller row).
Every tracked point votes; the median is the frame's horizon. Hundreds of votes per
frame instead of five.

⚠️ This estimator ASSUMES ``f*h``. It cannot be used to check ``f*h`` -- that would be
circular. ``f*h`` comes from ``flow_scale.row_flow_fit`` over the whole clip, where the
horizon is a free parameter; here it is held and only the per-frame horizon moves.

⚠️ Validated against the INDEPENDENT lane-residual horizon of Part 5, which shares no
machinery with it: different features (paint vs texture), different model (lateral
residual vs row flow), different frames-to-parameters ratio.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import flow_scale as FS, run_real as RR


def horizon_one(i0, i1, D, fh, rows=(0.55, 0.77), max_pts=700):
    """Median per-point horizon vote for one consecutive frame pair."""
    m = FS.road_mask(i0, row_band=rows)
    if m.sum() == 0:
        return None
    uv0, uv1 = FS.track_pair(i0, i1, n_pts=max_pts, mask=m)
    if len(uv0) < 40:
        return None
    v, vp = uv0[:, 1], uv1[:, 1]
    dv = vp - v
    ok = dv > 0.5                                  # the ground must move DOWN
    if ok.sum() < 30:
        return None
    v, vp, dv = v[ok], vp[ok], dv[ok]
    s = v + vp
    disc = s * s - 4.0 * (v * vp - fh * dv / D)
    good = disc > 0
    if good.sum() < 25:
        return None
    vh = (s[good] - np.sqrt(disc[good])) / 2.0
    vh = vh[(vh > 250) & (vh < 620)]               # outside this it is not a horizon
    if len(vh) < 25:
        return None
    return float(np.median(vh)), int(len(vh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fh", type=float, default=2444.6)
    ap.add_argument("--frames", type=int, default=260)
    ap.add_argument("--smooth", type=float, default=0.5, help="seconds of smoothing")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    by = {int(r["frame"]): r for r in recs}
    usable = sorted(k for k, r in by.items()
                    if r["complete"] and r["speed_ms"] > 8.0
                    and (fdir / f"{k:06d}.jpg").exists())
    fps = 29.922
    picks = [usable[i] for i in np.linspace(0, len(usable) - 1, a.frames).astype(int)]

    fr, hz, nv = [], [], []
    for f in picks:
        p1 = fdir / f"{f+1:06d}.jpg"
        if not p1.exists():
            continue
        i0 = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        i1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
        if i0 is None or i1 is None:
            continue
        r = by[f]
        t = np.asarray(r["t"], float)
        D = float(np.interp(1.0 / fps, t, np.asarray(r["x"], float)))
        if D <= 0.2:
            continue
        got = horizon_one(i0, i1, D, a.fh)
        if got is None:
            continue
        fr.append(f); hz.append(got[0]); nv.append(got[1])
    if len(hz) < 30:
        print(f"only {len(hz)} frames produced a horizon — estimator unusable")
        return 1
    fr = np.asarray(fr); hz = np.asarray(hz); nv = np.asarray(nv)

    print(f"{len(hz)} frames, median {int(np.median(nv))} point-votes each "
          f"(the lane-based estimator managed 39 frames and 5 samples)\n")
    print(f"  per-frame horizon: median {np.median(hz):.1f} px   "
          f"p10 {np.percentile(hz,10):.1f}   p90 {np.percentile(hz,90):.1f}   "
          f"sd {np.std(hz):.1f}")
    print(f"  p10-p90 spread {np.percentile(hz,90)-np.percentile(hz,10):.1f} px = "
          f"{np.rad2deg(np.arctan((np.percentile(hz,90)-np.percentile(hz,10))/1713.)):.2f} deg of pitch")

    # cross-check against the INDEPENDENT lane-residual horizon from Part 5
    ref = pathlib.Path(__file__).resolve().parent.parent / "raw" / "per_frame_horizon.json"
    if ref.exists():
        R = json.loads(ref.read_text())
        rf = np.asarray(R["frames"]); rh = np.asarray(R["best_hz"])
        pairs = [(h, rh[np.argmin(np.abs(rf - f))]) for f, h in zip(fr, hz)
                 if np.min(np.abs(rf - f)) <= 8]
        if len(pairs) >= 12:
            A_ = np.array([p[0] for p in pairs]); B_ = np.array([p[1] for p in pairs])
            r = float(np.corrcoef(A_, B_)[0, 1])
            print(f"\n  cross-check vs the lane-residual horizon ({len(pairs)} matched frames,")
            print(f"  no shared features or model): r = {r:+.3f}, "
                  f"median difference {np.median(A_-B_):+.1f} px")
            if r > 0.4:
                print("  => two independent estimators agree the horizon MOVES, and agree on when.")
            else:
                print("  ⚠️ they do NOT agree frame by frame. One of them is tracking noise;")
                print("     do not deploy a per-frame horizon on this evidence alone.")
        else:
            print(f"\n  only {len(pairs)} frames matched for the cross-check — not enough")
    if a.json:
        a.json.write_text(json.dumps(dict(frames=fr.tolist(), horizon=hz.tolist(),
                                          votes=nv.tolist(), fh=a.fh), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
