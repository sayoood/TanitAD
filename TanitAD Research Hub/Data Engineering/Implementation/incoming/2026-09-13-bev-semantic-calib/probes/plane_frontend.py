#!/usr/bin/env python3
"""WHY does `plane_calib` say "too few usable homographies" on this recording?

Sayed asks whether SLAM / "optimise the parameters for the best match of static
features" would settle the calibration. The pipeline already contains the right
member of that family — `plane_calib.py` — and it is **failing silently**:

    WARN  plane calibration produced too few usable homographies

That module decomposes the inter-frame road-plane homography `H = R + (t/d)·nᵀ`.
The normal `n` IS the camera's roll and pitch **relative to the road, per frame
pair**, and with the metric baseline from the trajectory, `t/d` gives the camera
HEIGHT. That is precisely the per-frame quantity a fixed extrinsic cannot carry —
and on the 08-11 session it worked (93 usable pairs, height 1.167 m [1.100, 1.223]).

So the question is not "should we build SLAM", it is "why does the one we have
produce nothing here". This probe instruments every stage of the front end and
counts survivors, instead of guessing.

⚠️ The prior suspicion, from evidence already in this directory: `flow_scale`'s 2-D
fit failed the same way — `goodFeaturesToTrack` AVOIDS THE ROAD because asphalt is
smooth (2.3 % inliers at 400 px median reprojection), and the fix was to seed on the
pipeline's own ridge detector instead. If that is the failure here too, the same fix
applies. If it is the corridor MASK landing in the wrong place — it is projected with
`cam`, whose horizon was 37-75 px wrong for most of this session — that is a
different fix. The probe distinguishes them by counting.
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

FPS = 29.9219


def corridor_mask(P, shape, half_w=3.2, x_range=(5.0, 32.0)):
    h, w = shape
    xs = np.linspace(x_range[0], x_range[1], 60)
    left = BC.project_ground(np.stack([xs, np.full_like(xs, half_w)], 1), P)
    right = BC.project_ground(np.stack([xs, np.full_like(xs, -half_w)], 1), P)
    ok = np.isfinite(left).all(1) & np.isfinite(right).all(1)
    m = np.zeros((h, w), np.uint8)
    if ok.sum() >= 2:
        poly = np.concatenate([left[ok], right[ok][::-1]])
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    return m


def seed_ridges(gray, mask, P, fh, horizon, fx):
    """Marking-seeded features: paint lies EXACTLY on the road plane and is the one
    thing on a motorway that is both high-contrast and coplanar."""
    from overlay_far import ridge_cols, ridge_width_px
    pts = []
    ys, xs_ = np.nonzero(mask)
    if len(ys) == 0:
        return np.empty((0, 2), np.float32)
    for v in range(int(ys.min()), int(ys.max()), 3):
        x_m = fh / max(v - horizon, 1e-3)
        if not (3.0 < x_m < 60.0):
            continue
        cols = ridge_cols(gray[v], ridge_width_px(x_m, fx))
        for c in cols:
            if 0 <= int(c) < mask.shape[1] and mask[v, int(c)]:
                pts.append((float(c), float(v)))
    return np.asarray(pts, np.float32).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--fx", type=float, default=1533.0)
    ap.add_argument("--height", type=float, default=1.586)
    ap.add_argument("--horizon", type=float, default=448.4)
    ap.add_argument("--yaw", type=float, default=-5.35)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P = dict(RR.NOMINAL)
    P.update(fx=a.fx, height=a.height, lateral=-0.126, yaw=np.deg2rad(a.yaw))
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)
    fh = a.fx * a.height
    K = np.array([[a.fx, 0, 960.0], [0, a.fx, 540.0], [0, 0, 1.0]])

    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = RR.RUN / "frames"
    frames = sorted(fdir.glob("*.jpg"))
    idx = [int(p.stem) for p in frames]
    have = set(idx)

    lk = dict(winSize=(25, 25), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    feat = dict(maxCorners=500, qualityLevel=0.008, minDistance=8, blockSize=7)

    stages = {k: {"n_feat": [], "n_track": [], "n_inl": [], "ok": 0, "tried": 0,
                  "fail": {}} for k in ("goodFeatures", "goodFeatures+CLAHE", "ridges",
                                        "ridges+goodFeatures")}
    out_vals = {k: {"roll": [], "pitch": [], "height": []} for k in stages}

    cand = [f for f in idx if (f + a.gap) in have and f in recs
            and recs[f]["complete"] and recs[f]["speed_ms"] > 6.0]
    pick = [cand[i] for i in np.linspace(0, len(cand) - 1, min(a.pairs, len(cand))).astype(int)]
    print(f"{len(cand)} candidate pairs (gap {a.gap} frames = {a.gap/FPS:.3f} s), "
          f"probing {len(pick)}\n")

    mask_cache = None
    for f0 in pick:
        g0 = cv2.imread(str(fdir / f"{f0:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        g1 = cv2.imread(str(fdir / f"{f0+a.gap:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g0 is None or g1 is None:
            continue
        if mask_cache is None:
            mask_cache = corridor_mask(P, g0.shape)
            print(f"corridor mask covers {100*mask_cache.mean()/255:.2f} % of the frame, "
                  f"rows {np.nonzero(mask_cache.any(1))[0].min()}-"
                  f"{np.nonzero(mask_cache.any(1))[0].max()}")
        mask = mask_cache
        r = recs[f0]
        t = np.asarray(r["t"], float)
        dt = a.gap / FPS
        if t.min() > 0 or t.max() < dt:
            continue
        base = float(np.hypot(np.interp(dt, t, np.asarray(r["x"], float)),
                              np.interp(dt, t, np.asarray(r["y"], float))))
        if base < 0.8:
            continue

        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        variants = {
            "goodFeatures": cv2.goodFeaturesToTrack(g0, mask=mask, **feat),
            "goodFeatures+CLAHE": cv2.goodFeaturesToTrack(cl.apply(g0), mask=mask, **feat),
            "ridges": seed_ridges(g0, mask, P, fh, a.horizon, a.fx).reshape(-1, 1, 2),
        }
        gf = variants["goodFeatures+CLAHE"]
        rg = variants["ridges"]
        variants["ridges+goodFeatures"] = (
            np.concatenate([gf, rg]) if gf is not None and len(rg) else (gf if gf is not None else rg))

        for name, p0 in variants.items():
            S = stages[name]
            S["tried"] += 1
            if p0 is None or len(p0) < 14:
                S["n_feat"].append(0 if p0 is None else len(p0))
                S["fail"]["no features"] = S["fail"].get("no features", 0) + 1
                continue
            p0 = np.ascontiguousarray(p0.reshape(-1, 1, 2), np.float32)
            S["n_feat"].append(len(p0))
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & \
                   (np.linalg.norm(p0.reshape(-1, 2) - pb.reshape(-1, 2), axis=1) < 1.0)
            A, B = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
            S["n_track"].append(int(good.sum()))
            if len(A) < 14:
                S["fail"]["too few tracked"] = S["fail"].get("too few tracked", 0) + 1
                continue
            H, m = cv2.findHomography(A.reshape(-1, 1, 2), B.reshape(-1, 1, 2),
                                      cv2.RANSAC, 1.5, maxIters=4000, confidence=0.995)
            if H is None or m is None:
                S["fail"]["no homography"] = S["fail"].get("no homography", 0) + 1
                continue
            S["n_inl"].append(int(m.sum()))
            if int(m.sum()) < 12:
                S["fail"]["<12 inliers"] = S["fail"].get("<12 inliers", 0) + 1
                continue
            n_sol, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
            best = None
            for i in range(n_sol):
                nv = np.asarray(Ns[i], float).ravel()
                nx, ny, nz = float(nv[0]), float(nv[1]), float(nv[2])
                pitch = np.arcsin(np.clip(nz, -1, 1)); roll = np.arctan2(nx, -ny)
                if abs(np.rad2deg(pitch)) > 25 or abs(np.rad2deg(roll)) > 25:
                    continue
                tn = float(np.linalg.norm(Ts[i]))
                if tn < 1e-6:
                    continue
                d = base / tn
                if not (0.6 < d < 2.6):
                    continue
                ang = float(np.rad2deg(np.arccos(
                    np.clip((np.trace(Rs[i]) - 1) / 2, -1, 1))))
                if best is None or ang < best[0]:
                    best = (ang, np.rad2deg(roll), np.rad2deg(pitch), d)
            if best is None:
                S["fail"]["no admissible decomposition"] = \
                    S["fail"].get("no admissible decomposition", 0) + 1
                continue
            S["ok"] += 1
            out_vals[name]["roll"].append(best[1])
            out_vals[name]["pitch"].append(best[2])
            out_vals[name]["height"].append(best[3])

    print(f"\n{'front end':>20}{'feats':>8}{'tracked':>9}{'inliers':>9}"
          f"{'USABLE PAIRS':>14}")
    res = {}
    for name, S in stages.items():
        med = lambda v: (f"{np.median(v):.0f}" if v else "-")
        print(f"{name:>20}{med(S['n_feat']):>8}{med(S['n_track']):>9}"
              f"{med(S['n_inl']):>9}{S['ok']:>8}/{S['tried']:<5}")
        res[name] = dict(ok=S["ok"], tried=S["tried"],
                         n_feat=float(np.median(S["n_feat"])) if S["n_feat"] else 0,
                         n_track=float(np.median(S["n_track"])) if S["n_track"] else 0,
                         n_inl=float(np.median(S["n_inl"])) if S["n_inl"] else 0,
                         fail=S["fail"])
    print(f"\nwhere each one dies:")
    for name, S in stages.items():
        if S["fail"]:
            print(f"  {name:>20}  " + ", ".join(f"{k} {v}" for k, v in
                                                sorted(S["fail"].items(), key=lambda x: -x[1])))
    print(f"\n{'front end':>20}{'roll':>18}{'pitch':>18}{'HEIGHT':>20}")
    for name, V in out_vals.items():
        if len(V["height"]) >= 8:
            h = np.asarray(V["height"]); r_ = np.asarray(V["roll"]); p_ = np.asarray(V["pitch"])
            print(f"{name:>20}{np.median(r_):+10.2f}±{1.4826*np.median(np.abs(r_-np.median(r_))):.2f}"
                  f"{np.median(p_):+10.2f}±{1.4826*np.median(np.abs(p_-np.median(p_))):.2f}"
                  f"{np.median(h):12.3f}±{1.4826*np.median(np.abs(h-np.median(h))):.3f} m"
                  f"  (n {len(h)})")
            res[name]["roll_deg"] = float(np.median(r_))
            res[name]["pitch_deg"] = float(np.median(p_))
            res[name]["height_m"] = float(np.median(h))
            res[name]["height_mad"] = float(1.4826*np.median(np.abs(h-np.median(h))))
        else:
            print(f"{name:>20}{'-- too few --':>56}")
    print(f"\n  for reference, measured elsewhere in this directory:")
    print(f"    height 1.573 m [1.511, 1.628] (lane-line slope difference, no horizon/focal)")
    print(f"    pitch  from horizon 448.4 at f {a.fx:.0f} -> "
          f"{np.rad2deg(P['pitch']):+.2f} deg")
    if a.json:
        a.json.write_text(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
