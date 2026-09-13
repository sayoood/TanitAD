"""All-camera NATIVE sequences for the SAM3 map ground truth (7 f-theta cameras, LiDAR ground, 5 Hz).

Per token (the SAME t_ref grid and token names as build_frames_v2.py, so the LiDAR sweep and pose already validated for
that token are reused): each native camera's frame nearest to t_ref, undistorted by nothing -- the camera model
(f-theta polynomial + extrinsics) travels in calib.npz and camera_model.Camera reads it.
Views: CAM_FW front wide 120, CAM_CL / CAM_CR cross 120, CAM_RL / CAM_RR rear 70, CAM_RT rear tele 30, CAM_FT front tele 30.
Output: /home/nvidia/sam3map/native7/seq_<c8>/<token>/{images/<CAM>.jpg, calib.npz, frame.json, meta.json, lidar.npy} + poses.json
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import av
from PIL import Image

sys.path.insert(0, "/home/nvidia/sam3map/eval")
from ftheta_pinhole import quat_to_R  # noqa: E402

D = Path("/home/nvidia/sam3map/data"); SRC = D / "native7"; V2 = Path("/home/nvidia/qwendrive/v2")
OUT = Path("/home/nvidia/sam3map/native7")
CAMS = {"CAM_FW": "camera_front_wide_120fov", "CAM_CL": "camera_cross_left_120fov", "CAM_CR": "camera_cross_right_120fov",
        "CAM_RL": "camera_rear_left_70fov", "CAM_RR": "camera_rear_right_70fov", "CAM_RT": "camera_rear_tele_30fov",
        "CAM_FT": "camera_front_tele_30fov"}


def build(clip, chunk=768):
    c8 = clip[:8]; sha = hashlib.sha256(clip.encode()).hexdigest()[:12]
    idf = pd.read_parquet(D / "calib" / f"camera_intrinsics.chunk_{chunk:04d}.parquet").reset_index()
    edf = pd.read_parquet(D / "calib" / f"sensor_extrinsics.chunk_{chunk:04d}.parquet").reset_index()
    cal = {}
    for cam, feat in CAMS.items():
        r = idf[(idf["clip_id"] == clip) & (idf["camera_name"] == feat)].iloc[0]
        e = edf[(edf["clip_id"] == clip) & (edf["sensor_name"] == feat)].iloc[0]
        cal[cam] = dict(poly=[float(r[f"fw_poly_{i}"]) for i in range(5)], cx=float(r["cx"]), cy=float(r["cy"]),
                        R=quat_to_R(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"])), t=np.array([float(e["x"]), float(e["y"]), float(e["z"])]))
    times = {cam: pd.read_parquet(SRC / f"{clip}.{feat}.timestamps.parquet")["timestamp"].to_numpy(np.float64) for cam, feat in CAMS.items()}
    ref = times["CAM_FW"]
    t_refs = list(np.arange(ref[0] + 0.5e6, ref[-1] - 0.5e6, 1e6 / 5.0))
    tokens = [f"{c8}_t{int(round((t - ref[0]) / 1e3)):05d}" for t in t_refs]
    v2seq = V2 / f"seq_{c8}"
    assert all((v2seq / tok / "lidar.npy").exists() for tok in tokens), "token grid differs from the v2 sequence"
    sd = OUT / f"seq_{c8}"
    for tok in tokens:
        (sd / tok / "images").mkdir(parents=True, exist_ok=True)
    dts = {tok: {} for tok in tokens}; sizes = {}
    for cam, feat in CAMS.items():
        want = {}
        for tok, t in zip(tokens, t_refs):
            i = int(np.argmin(np.abs(times[cam] - t))); want.setdefault(i, []).append(tok); dts[tok][cam] = round((times[cam][i] - t) / 1e3, 2)
        with av.open(str(SRC / f"{clip}.{feat}.mp4")) as cont:
            st = cont.streams.video[0]; st.thread_type = "AUTO"
            for i, fr in enumerate(cont.decode(st)):
                if i in want:
                    img = fr.to_ndarray(format="rgb24"); sizes[cam] = (img.shape[1], img.shape[0])
                    for tok in want[i]:
                        Image.fromarray(img).save(sd / tok / "images" / f"{cam}.jpg", quality=95)
                if i >= max(want):
                    break
    order = list(CAMS)
    for tok, t in zip(tokens, t_refs):
        d = sd / tok
        np.savez(d / "calib.npz", ftheta_poly=np.array([cal[c]["poly"] for c in order]), ftheta_cx=np.array([cal[c]["cx"] for c in order]),
                 ftheta_cy=np.array([cal[c]["cy"] for c in order]), sensor2lidar_rotation=np.stack([cal[c]["R"] for c in order]),
                 sensor2lidar_translation=np.stack([cal[c]["t"] for c in order]), image_wh=np.array([sizes[c] for c in order]))
        (d / "frame.json").write_text(json.dumps({"dataset_type": "native", "camera_model": "ftheta", "cam_order": order}), encoding="utf-8")
        shutil.copy(v2seq / tok / "lidar.npy", d / "lidar.npy")
        (d / "meta.json").write_text(json.dumps({"token": tok, "clip_sha12": sha, "t_ref_us": float(t), "builder": "build_native7_seq",
                                                 "cameras": {c: {"feature": CAMS[c], "wh": sizes[c], "dt_ms": dts[tok][c]} for c in order},
                                                 "ground": "LiDAR (the v2 sequence's sweep for this token)"}), encoding="utf-8")
    shutil.copy(v2seq / "poses.json", sd / "poses.json")
    print(f"{sha}: {len(tokens)} tokens x {len(order)} native cameras; sizes {sizes}; max |dt| {max(abs(v) for x in dts.values() for v in x.values()):.1f} ms", flush=True)


if __name__ == "__main__":
    for clip in sys.argv[1:]:
        build(clip)
    print("ZZNATIVE7-DONEZZ")
