"""Front-camera-only SAM3-map sequences for clips whose side cameras and LiDAR are not on the programme's disks.

Per clip, 5 Hz:
  image   the virtual nuPlan CAM_F0 pinhole view (1920x1080, K_V, yaw 0, pitch 1 deg) reprojected EXACTLY from the
          front-wide f-theta camera through its optical centre -- the same view_map / remap as build_frames_v2.py
  pose    egomotion (per-clip Alpamayo parquet, same microsecond clock as the camera), nearest sample
  ground  NO LiDAR: the ego's own path (rear axle on the road, -3 s .. +6 s) in the current rig frame, spread +-6 m
          sideways, written as lidar.npy-shaped pseudo ground points so P.ground_grid consumes it unchanged
          (labels may use ego; road slope and body pitch along the path are captured, cross-slope is not)
Output: /home/nvidia/sam3map/front/seq_<c8>/<token>/{images/CAM_F0.jpg, calib.npz, frame.json, meta.json, lidar.npy}
        + poses.json
"""
import hashlib, io, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import av
from PIL import Image

sys.path.insert(0, "/home/nvidia/sam3map/eval")
from ftheta_pinhole import FTheta, quat_to_R  # noqa: E402

D = Path("/home/nvidia/sam3map/data")
OUT = Path("/home/nvidia/sam3map/front")
OUT_W, OUT_H = 1920, 1080
K_V = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
PITCH_DEG = 1.0
FEAT = "camera_front_wide_120fov"


def virtual_R(yaw_deg, pitch_deg):
    psi, phi = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(psi) * math.cos(phi), math.sin(psi) * math.cos(phi), -math.sin(phi)])
    right = np.array([math.sin(psi), -math.cos(psi), 0.0])
    return np.stack([right, np.cross(fwd, right), fwd], axis=1)


def theta_max(ft):
    r = math.hypot(max(ft.cx, ft.width - 1 - ft.cx), max(ft.cy, ft.height - 1 - ft.cy))
    return ft.theta_of_r(r, hi=2.2)


def view_map(R_v, R_s, ft, scale):
    u, v = np.meshgrid(np.arange(OUT_W, dtype=np.float64), np.arange(OUT_H, dtype=np.float64))
    rays = np.stack([(u - K_V[0, 2]) / K_V[0, 0], (v - K_V[1, 2]) / K_V[1, 1], np.ones_like(u)], axis=-1)
    rs = rays @ R_v.T @ R_s
    x, y, z = rs[..., 0], rs[..., 1], rs[..., 2]
    rho = np.hypot(x, y); th = np.arctan2(rho, z); r = ft.r_of_theta(th)
    k = np.where(rho > 1e-12, r / np.maximum(rho, 1e-12), 0.0)
    mx = (ft.cx + x * k) / scale; my = (ft.cy + y * k) / scale
    ws, hs = ft.width / scale, ft.height / scale
    valid = (th < theta_max(ft)) & (mx >= 0) & (mx <= ws - 1) & (my >= 0) & (my <= hs - 1)
    return mx.astype(np.float32), my.astype(np.float32), valid


def remap(img, mx, my, valid):
    h, w = img.shape[:2]
    xs, ys = np.clip(mx, 0, w - 1), np.clip(my, 0, h - 1)
    x0, y0 = np.floor(xs).astype(np.int32), np.floor(ys).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (xs - x0)[..., None], (ys - y0)[..., None]
    im = img.astype(np.float32)
    out = (im[y0, x0] * (1 - wx) + im[y0, x1] * wx) * (1 - wy) + (im[y1, x0] * (1 - wx) + im[y1, x1] * wx) * wy
    out[~valid] = 0.0
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def box_downscale(img, s):
    if s == 1:
        return img
    h, w = (img.shape[0] // s) * s, (img.shape[1] // s) * s
    return img[:h, :w].reshape(h // s, s, w // s, s, 3).mean(axis=(1, 3)).astype(np.uint8)


def pose_at(ego, t):
    i = int(np.argmin(np.abs(ego["timestamp"].to_numpy(np.float64) - t))); r = ego.iloc[i]
    T = np.eye(4); T[:3, :3] = quat_to_R(r.qx, r.qy, r.qz, r.qw); T[:3, 3] = [r.x, r.y, r.z]
    return T


def path_ground(ego, T, t):
    ts = ego["timestamp"].to_numpy(np.float64)
    sel = (ts >= t - 3e6) & (ts <= t + 6e6)
    P = ego.loc[sel, ["x", "y", "z"]].to_numpy(np.float64)[::5]
    if len(P) < 3:
        return np.zeros((0, 3), np.float32)
    q = (np.c_[P, np.ones(len(P))] @ np.linalg.inv(T).T)[:, :3]
    tang = np.gradient(q[:, :2], axis=0); tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    s = np.arange(-6.0, 6.01, 0.5)
    pts = (q[:, None, :2] + nrm[:, None, :] * s[None, :, None]).reshape(-1, 2)
    z = np.repeat(q[:, 2], len(s))
    return np.c_[pts, z].astype(np.float32)


def build(clip, chunk):
    c8 = clip[:8]; sha = hashlib.sha256(clip.encode()).hexdigest()[:12]
    idf = pd.read_parquet(D / "calib" / f"camera_intrinsics.chunk_{chunk:04d}.parquet").reset_index()
    edf = pd.read_parquet(D / "calib" / f"sensor_extrinsics.chunk_{chunk:04d}.parquet").reset_index()
    r = idf[(idf["clip_id"] == clip) & (idf["camera_name"] == FEAT)].iloc[0]
    ft = FTheta(poly=tuple(float(r[f"fw_poly_{i}"]) for i in range(5)), cx=float(r["cx"]), cy=float(r["cy"]), width=int(r["width"]), height=int(r["height"]))
    e = edf[(edf["clip_id"] == clip) & (edf["sensor_name"] == FEAT)].iloc[0]
    R_s = quat_to_R(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"])); t_s = np.array([float(e["x"]), float(e["y"]), float(e["z"])])
    scale = max(1, int(math.floor(ft.poly[1] / K_V[0, 0])))
    R_v = virtual_R(0.0, PITCH_DEG)
    mx, my, valid = view_map(R_v, R_s, ft, scale)
    ego = pd.read_parquet(D / "egomotion_alpamayo" / f"{clip}.parquet").sort_values("timestamp").reset_index(drop=True)
    tsd = pd.read_parquet(D / "frontwide" / f"{clip}.timestamps.parquet")
    times = tsd["timestamp"].to_numpy(np.float64)
    t_refs = np.arange(times[0] + 0.5e6, times[-1] - 0.5e6, 0.2e6)
    want = {}
    for t in t_refs:
        want.setdefault(int(np.argmin(np.abs(times - t))), []).append(float(t))
    sd = OUT / f"seq_{c8}"; sd.mkdir(parents=True, exist_ok=True)
    poses = {}
    with av.open(str(D / "frontwide" / f"{clip}.mp4")) as cont:
        st = cont.streams.video[0]; st.thread_type = "AUTO"
        for i, fr in enumerate(cont.decode(st)):
            if i not in want:
                continue
            res = remap(box_downscale(fr.to_ndarray(format="rgb24"), scale), mx, my, valid)
            for t in want[i]:
                tok = f"{c8}_t{int(round((t - times[0]) / 1e3)):05d}"
                d = sd / tok; (d / "images").mkdir(parents=True, exist_ok=True)
                Image.fromarray(res).save(d / "images" / "CAM_F0.jpg", quality=95)
                np.savez(d / "calib.npz", cam_intrinsic=K_V[None], sensor2lidar_rotation=R_v[None], sensor2lidar_translation=t_s[None])
                (d / "frame.json").write_text(json.dumps({"dataset_type": "nuplan", "cam_order": ["CAM_F0"]}), encoding="utf-8")
                T = pose_at(ego, t)
                np.save(d / "lidar.npy", path_ground(ego, T, t))
                (d / "meta.json").write_text(json.dumps({"token": tok, "clip_sha12": sha, "t_ref_us": t, "builder": "build_front_seq",
                                                         "ground": "ego path -3..+6 s spread +-6 m (no LiDAR)", "chunk": int(chunk)}), encoding="utf-8")
                poses[tok] = {"T_world_rig": T.tolist()}
            if i >= max(want):
                break
    (sd / "poses.json").write_text(json.dumps(poses), encoding="utf-8")
    print(f"{sha}: {len(poses)} frames, observed {valid.mean():.3f}, prefilter x{scale}", flush=True)


if __name__ == "__main__":
    for item in sys.argv[1:]:
        clip, chunk = item.split(":")
        build(clip, int(chunk))
    print("ZZFRONT-BUILD-DONEZZ")
