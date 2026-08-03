#!/usr/bin/env python3
"""ADVERSARIAL, GPU-FREE: does the mp4 lead or trail the rig, and is the offset +6?

Independent of the renderer entirely. Two 1-D series:
  M[t] = mean |frame_t - frame_{t-1}| on the decoded mp4          (image motion)
  S[f] = |p(f) - p(f-1)| from rig cameras_frame_T_rig_worlds      (ego motion, metres)
If mp4 index = rig index + L, then M[f+L] should track S[f]. Cross-correlate for L.
Also: are the first/last frames black or duplicated (a leader/trailer)?
"""
import json, sys, time
import numpy as np, cv2

ROOT = "/home/nvidia/nurec_scenes/sample_set/26.04_release"
CAM = "camera_front_wide_120fov"
sys.path.insert(0, "/home/nvidia/nurec-gsplat")
from nurec_loader import RigTrajectories

def motion_series(mp4, small=(192, 108)):
    cap = cv2.VideoCapture(mp4)
    meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev = None; M = []; means = []; pts = []
    n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        g = cv2.cvtColor(cv2.resize(img, small, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        means.append(float(g.mean()))
        M.append(float(np.abs(g - prev).mean()) if prev is not None else np.nan)
        prev = g; n += 1
    cap.release()
    return n, meta, np.array(M), np.array(means), np.array(pts)

def ego_series(scene):
    rig = RigTrajectories(f"{ROOT}/{scene}/rig_trajectories.json")
    nf = rig.n_frames(CAM)
    P = np.array([rig.T_rig_world(CAM, f, 1)[:3, 3] for f in range(nf)])
    S = np.r_[np.nan, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    ts = np.array([rig.frame_timestamps_us(CAM, f) for f in range(nf)], np.float64)
    return nf, S, ts

def best_lag(M, S, lags=range(-15, 16)):
    out = {}
    for L in lags:
        # mp4 index = rig index + L
        f = np.arange(1, len(S))
        i = f + L
        ok = (i >= 1) & (i < len(M))
        a = S[f[ok]]; b = M[i[ok]]
        good = np.isfinite(a) & np.isfinite(b)
        a, b = a[good], b[good]
        if len(a) < 50:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        out[L] = (round(r, 5), len(a))
    return out

for scene in sys.argv[1:]:
    t0 = time.time()
    mp4 = f"{ROOT}/{scene}/{CAM}.mp4"
    n_dec, meta, M, means, pts = motion_series(mp4)
    nf, S, ts = ego_series(scene)
    print("=" * 78)
    print(f"SCENE {scene[:8]}   decode {time.time()-t0:.1f}s")
    print(f"  mp4 decodable frames = {n_dec}   CAP_PROP_FRAME_COUNT = {meta}")
    print(f"  rig n_frames({CAM}) = {nf}     mp4 - rig = {n_dec - nf}")
    dt = np.diff(pts)
    print(f"  mp4 PTS: first={pts[0]:.2f}ms last={pts[-1]:.2f}ms  dPTS median={np.median(dt):.4f}ms "
          f"min={dt.min():.4f} max={dt.max():.4f}")
    print(f"  rig ts : first_end={ts[0,1]:.0f}us last_end={ts[-1,1]:.0f}us  "
          f"period={(ts[1,1]-ts[0,1])/1000:.4f}ms  span={(ts[-1,1]-ts[0,1])/1000:.1f}ms")
    print(f"  mp4 span = {pts[-1]-pts[0]:.1f} ms over {n_dec-1} intervals")
    print(f"  FIRST 10 frame means: {np.round(means[:10],3)}")
    print(f"  LAST  10 frame means: {np.round(means[-10:],3)}")
    print(f"  FIRST 10 |dI| (frame-to-frame): {np.round(M[:10],4)}")
    print(f"  LAST  10 |dI|: {np.round(M[-10:],4)}")
    print(f"  n consecutive-duplicate frames (|dI|<1e-6): {int(np.nansum(M < 1e-6))}")
    lag = best_lag(M, S)
    b = max(lag, key=lambda L: lag[L][0])
    print("  cross-correlation r(ego-step, image-motion) by lag L  [mp4_idx = rig_idx + L]:")
    for L in sorted(lag):
        star = "  <== ARGMAX" if L == b else ""
        print(f"      L={L:+3d}  r={lag[L][0]:+.4f}  n={lag[L][1]}{star}")
    print(f"  ==> BEST LAG L = {b:+d}   r = {lag[b][0]:.4f}   (r at L=0 is {lag.get(0,(float('nan'),))[0]})")
