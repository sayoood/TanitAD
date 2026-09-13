"""Front-camera-only sequences on the NATIVE f-theta front-wide camera (120 deg), for the SAM3 map v5.

Same as build_front_seq.py (5 Hz, egomotion poses, ego-path ground) but the image is the decoded native frame -- no
reprojection to the 63.7 deg virtual pinhole view, no interpolation -- and calib.npz carries the f-theta model
(`ftheta_poly`, `ftheta_cx`, `ftheta_cy`) that camera_model.Camera reads.
Output: /home/nvidia/sam3map/front_native/seq_<c8>/<token>/{images/CAM_F0.jpg, calib.npz, frame.json, meta.json, lidar.npy}
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import av
from PIL import Image

sys.path.insert(0, "/home/nvidia/sam3map/eval")
from ftheta_pinhole import quat_to_R  # noqa: E402
from build_front_seq import pose_at, path_ground, D, FEAT  # noqa: E402

OUT = Path("/home/nvidia/sam3map/front_native")


def build(clip, chunk):
    c8 = clip[:8]; sha = hashlib.sha256(clip.encode()).hexdigest()[:12]
    idf = pd.read_parquet(D / "calib" / f"camera_intrinsics.chunk_{chunk:04d}.parquet").reset_index()
    edf = pd.read_parquet(D / "calib" / f"sensor_extrinsics.chunk_{chunk:04d}.parquet").reset_index()
    r = idf[(idf["clip_id"] == clip) & (idf["camera_name"] == FEAT)].iloc[0]
    poly = np.array([float(r[f"fw_poly_{i}"]) for i in range(5)])
    e = edf[(edf["clip_id"] == clip) & (edf["sensor_name"] == FEAT)].iloc[0]
    R_s = quat_to_R(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"])); t_s = np.array([float(e["x"]), float(e["y"]), float(e["z"])])
    ego = pd.read_parquet(D / "egomotion_alpamayo" / f"{clip}.parquet").sort_values("timestamp").reset_index(drop=True)
    times = pd.read_parquet(D / "frontwide" / f"{clip}.timestamps.parquet")["timestamp"].to_numpy(np.float64)
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
            img = fr.to_ndarray(format="rgb24")
            for t in want[i]:
                tok = f"{c8}_t{int(round((t - times[0]) / 1e3)):05d}"
                d = sd / tok; (d / "images").mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(d / "images" / "CAM_F0.jpg", quality=95)
                np.savez(d / "calib.npz", ftheta_poly=poly[None], ftheta_cx=np.array([float(r["cx"])]), ftheta_cy=np.array([float(r["cy"])]),
                         sensor2lidar_rotation=R_s[None], sensor2lidar_translation=t_s[None], image_wh=np.array([img.shape[1], img.shape[0]]))
                (d / "frame.json").write_text(json.dumps({"dataset_type": "native", "camera_model": "ftheta", "cam_order": ["CAM_F0"]}), encoding="utf-8")
                T = pose_at(ego, times[i])
                np.save(d / "lidar.npy", path_ground(ego, T, times[i]))
                (d / "meta.json").write_text(json.dumps({"token": tok, "clip_sha12": sha, "t_ref_us": t, "t_frame_us": float(times[i]),
                                                         "builder": "build_front_seq_native", "camera": "native f-theta front-wide 120 deg",
                                                         "ground": "ego path -3..+6 s spread +-6 m (no LiDAR)", "chunk": int(chunk)}), encoding="utf-8")
                poses[tok] = {"T_world_rig": T.tolist()}
            if i >= max(want):
                break
    (sd / "poses.json").write_text(json.dumps(poses), encoding="utf-8")
    print(f"{sha}: {len(poses)} native frames, image {img.shape[1]}x{img.shape[0]}, camera height {t_s[2]:.3f} m", flush=True)


if __name__ == "__main__":
    for item in sys.argv[1:]:
        clip, chunk = item.split(":")
        build(clip, int(chunk))
    print("ZZNATIVE-BUILD-DONEZZ")
