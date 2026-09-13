"""⛔⛔ SUPERSEDED 2026-09-13 -- DO NOT REUSE THIS PACKING OR THIS RENDERER.

The views built here (focal 313.7 / 664.2 / 1732.3 px @896) are OUTSIDE the fixed camera geometry
Qwen-Drive's weights were trained on (721.0 px, 63.7 deg). MEASURED on the upstream demo: changing ONLY
the focal drops detection 54/57 -> 0/57. The renderer decoded occupancy/map from a guessed schema.
Replacement: TanitAD Research Lab/Data Engineering/Research/2026-09-13-qwen-drive-usage-review/
(code/build_frames_v2.py + the upstream scripts/visualize_perception.py). Kept for provenance only.
"""
"""Pack a SMALL PhysicalAI sample into Qwen-Drive perception frame directories.

⛔⛔ PERCEPTION TEACHER ONLY. Qwen-Drive-1.0 finishes at or near the BOTTOM of its
own 916-scenario AlpaSim closed-loop table (AlpaSim-all 0.16 vs Alpamayo-R1's
0.36, 2.25x worse) while winning every open-loop board. Its 3D boxes / occupancy
/ map segmentation are usable as a teacher; its TRAJECTORIES, PLANS AND ACTIONS
ARE NOT, ever. Distilling its planning would import precisely the weakness this
programme exists to fix. The planner weights are not even downloaded.
   -- PI directive 2026-09-11; teardown at
      `TanitAD Research Lab/Opponent Analysis/Research/2026-09-10-qwen-drive-teardown/RESULT.md`

WHAT THIS BUILDS
----------------
For each sampled frame, one directory in the layout
`qwen_drive_perception.dataset.PerceptionFrame` reads:

    <token>/frame.json    view-tag / image content list, cam_order, dataset_type
    <token>/images/*.jpg  RECTIFIED pinhole views, 896x512
    <token>/calib.npz     cam_intrinsic, sensor2lidar_rotation/translation, lidar2ego
    <token>/gt.npz        OUR obstacle.offline boxes (the teacher is scored against these)
    <token>/meta.json     provenance: clip, times, per-camera observed fraction

FRAMES AND UNITS -- stated because a formula quoted outside its frame reads
exactly like an answer:
  * PhysicalAI `obstacle.offline` cuboids are in the RIG frame at their OWN
    timestamp: +x forward, +y LEFT, +z UP, origin rear axle on the road plane.
  * We adopt RIG as BOTH the "lidar" and the "ego" frame, so `lidar2ego` is the
    IDENTITY -- which is also what Qwen-Drive uses for `dataset_type="nuplan"`.
  * `sensor2lidar_rotation` columns are the camera axes in the rig frame, camera
    convention +x right, +y down, +z boresight. PhysicalAI extrinsics quaternions
    are CAMERA->RIG, so R_cam_to_rig is used directly.
  * Box row is [x, y, z, w, l, h, yaw, vx, vy] with z at the box BOTTOM, w the
    extent along heading (local +x) and l the lateral extent -- mmdet3d order, per
    `qwen_drive_perception.geometry.box_corners`. Our GT gives CENTRE z and
    size_x = length along heading, so z_bottom = center_z - size_z/2.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

from ftheta_pinhole import (FTheta, build_K, choose_pinhole, quat_to_R,
                            rectify_map, remap_bilinear)

ROOT = Path(r"C:\Users\Admin\tanitad-data\physicalai")
OUT_W, OUT_H = 896, 512
CHUNK = 768

# ---------------------------------------------------------------------------
# The rig -> Qwen-Drive view-tag map.
#
# MEASURED boresight azimuths (799 clips / 8 calibration chunks, this session):
#   rear_tele  +179.83 | rear_left  +151.96+/-2.18 | cross_left  +66.81+/-0.76
#   front_wide   -0.40+/-0.46 | cross_right -66.47 | rear_right -151.25
# Qwen-Drive's canonical tags sit at 0, +/-45, +/-90, +/-135, 180. Each camera is
# assigned to its NEAREST canonical azimuth, uniformly. The result is exactly the
# nuScenes 6-camera arrangement, which is a supported Nv (the paper states
# Nv in {6, 8}) and is in the model's training distribution.
#
# ⚠️ `camera_front_tele_30fov` (az -0.23) is EXCLUDED BY DESIGN, not by oversight:
# its azimuth duplicates `camera_front_wide_120fov`, and there is exactly one
# FRONT tag. The rig has 7 cameras but only 6 DISTINCT azimuths.
# ⚠️ The cross cameras at +/-67 deg are nearly equidistant from the 45 and 90 slots
# (21.8 vs 23.2 deg); 45 wins on the uniform nearest rule and also matches
# nuScenes' FRONT_LEFT at +55 deg. The TAG is only a soft textual prior -- the
# geometry that drives the lift-splat comes from the real K and extrinsics.
# ---------------------------------------------------------------------------
CAMERAS = [
    # (feature name, view tag, rectified HFOV deg, short id used in images/)
    ("camera_front_wide_120fov", "<FRONT VIEW>", 110.0, "CAM_F0"),
    ("camera_cross_left_120fov", "<FRONT LEFT VIEW>", 110.0, "CAM_L0"),
    ("camera_cross_right_120fov", "<FRONT RIGHT VIEW>", 110.0, "CAM_R0"),
    ("camera_rear_left_70fov", "<BACK LEFT VIEW>", 68.0, "CAM_L2"),
    ("camera_rear_right_70fov", "<BACK RIGHT VIEW>", 68.0, "CAM_R2"),
    ("camera_rear_tele_30fov", "<BACK VIEW>", 29.0, "CAM_B0"),
]
# Content order follows the ring clockwise from the front, as in the demo frames.
CONTENT_ORDER = ["CAM_F0", "CAM_R0", "CAM_R2", "CAM_B0", "CAM_L2", "CAM_L0"]
TAG_OF = {sid: tag for _, tag, _, sid in CAMERAS}
FEATURE_OF = {sid: feat for feat, _, _, sid in CAMERAS}
HFOV_OF = {sid: hf for _, _, hf, sid in CAMERAS}

# Our 10 obstacle.offline classes -> Qwen-Drive's unified 7-class detection
# taxonomy (DET_CLASS_NAMES). Mapped at the coarsest mutually compatible
# granularity, the same discipline Qwen-Drive used across nuScenes/OpenScene.
CLASS_MAP = {
    "automobile": "vehicle",
    "heavy_truck": "vehicle",
    "bus": "vehicle",
    "trailer": "vehicle",
    "other_vehicle": "vehicle",
    "person": "pedestrian",
    "rider": "bicycle",
    "stroller": "generic_object",
    "protruding_object": "generic_object",
    "animal": "generic_object",
}
DET_CLASS_NAMES = ("vehicle", "czone_sign", "bicycle", "generic_object",
                   "pedestrian", "traffic_cone", "barrier")
DET_INDEX = {n: i for i, n in enumerate(DET_CLASS_NAMES)}
GT_TOL_S = 0.06          # per-track nearest-time tolerance, as in bev_raster.py


def yaw_from_quat(qx, qy, qz, qw) -> float:
    R = quat_to_R(qx, qy, qz, qw)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def load_calib(chunk: int, clip: str) -> tuple[dict, dict]:
    """Per-clip f-theta intrinsics and camera->rig extrinsics, for every camera."""
    ip = ROOT / "calibration" / "camera_intrinsics" / f"camera_intrinsics.chunk_{chunk:04d}.parquet"
    ep = ROOT / "calibration" / "sensor_extrinsics" / f"sensor_extrinsics.chunk_{chunk:04d}.parquet"
    idf = pd.read_parquet(ip).reset_index()
    edf = pd.read_parquet(ep).reset_index()
    idf = idf[idf["clip_id"] == clip]
    edf = edf[edf["clip_id"] == clip]
    intr, extr = {}, {}
    for feat, _, _, sid in CAMERAS:
        r = idf[idf["camera_name"] == feat]
        if len(r) != 1:
            raise RuntimeError(f"{clip}: {feat} intrinsics rows={len(r)}, expected 1")
        r = r.iloc[0]
        intr[sid] = FTheta(
            poly=tuple(float(r[f"fw_poly_{i}"]) for i in range(5)),
            cx=float(r["cx"]), cy=float(r["cy"]),
            width=int(r["width"]), height=int(r["height"]))
        e = edf[edf["sensor_name"] == feat]
        if len(e) != 1:
            raise RuntimeError(f"{clip}: {feat} extrinsics rows={len(e)}, expected 1")
        e = e.iloc[0]
        extr[sid] = {
            "R": quat_to_R(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"])),
            "t": np.array([float(e["x"]), float(e["y"]), float(e["z"])], dtype=np.float64),
        }
    return intr, extr


def camera_paths(sid: str, clip: str) -> tuple[Path, Path]:
    """(mp4, timestamps) for one camera of one clip, from disk or the chunk zip."""
    feat = FEATURE_OF[sid]
    direct = ROOT / "r0" / "camera_front_wide"
    mp4 = direct / f"{clip}.{feat}.mp4"
    if mp4.exists():
        return mp4, direct / f"{clip}.{feat}.timestamps.parquet"
    return (ROOT / "camera" / feat / f"{feat}.chunk_{CHUNK:04d}.zip"), None


def read_camera_stream(sid: str, clip: str, want_idx: list[int]) -> tuple[dict, np.ndarray]:
    """Decode only ``want_idx`` frames plus that camera's timestamp vector."""
    feat = FEATURE_OF[sid]
    mp4_path, ts_path = camera_paths(sid, clip)
    if ts_path is not None:                       # loose files on disk
        ts = pd.read_parquet(ts_path)
        tcol = next(c for c in ts.columns if "time" in c.lower())
        times = ts[tcol].to_numpy(np.float64)
        handle = open(mp4_path, "rb")
        close = [handle]
    else:                                          # inside the chunk zip
        z = zipfile.ZipFile(mp4_path)
        tname = f"{clip}.{feat}.timestamps.parquet"
        vname = f"{clip}.{feat}.mp4"
        names = set(z.namelist())
        if tname not in names or vname not in names:
            raise FileNotFoundError(f"{clip}: {feat} missing in {mp4_path.name}")
        ts = pd.read_parquet(io.BytesIO(z.read(tname)))
        tcol = next(c for c in ts.columns if "time" in c.lower())
        times = ts[tcol].to_numpy(np.float64)
        handle = io.BytesIO(z.read(vname))
        close = [z]
    want = set(int(i) for i in want_idx)
    out: dict[int, np.ndarray] = {}
    with av.open(handle) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i in want:
                out[i] = frame.to_ndarray(format="rgb24")
                if len(out) == len(want):
                    break
    for c in close:
        try:
            c.close()
        except Exception:
            pass
    return out, times


def gt_boxes_at(gt: pd.DataFrame, t_us: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-track nearest cuboid within GT_TOL_S of ``t_us``; rig frame, mmdet3d row."""
    tol = GT_TOL_S * 1e6
    rows, labels, names = [], [], []
    for _, g in gt.groupby("track_id", sort=False):
        dt = (g["timestamp_us"].to_numpy(np.float64) - t_us)
        k = int(np.argmin(np.abs(dt)))
        if abs(dt[k]) > tol:
            continue
        r = g.iloc[k]
        name = CLASS_MAP.get(str(r["label_class"]))
        if name is None:
            continue
        yaw = yaw_from_quat(r["orientation_x"], r["orientation_y"],
                            r["orientation_z"], r["orientation_w"])
        rows.append([
            float(r["center_x"]), float(r["center_y"]),
            float(r["center_z"]) - float(r["size_z"]) / 2.0,   # z at box BOTTOM
            float(r["size_x"]), float(r["size_y"]), float(r["size_z"]),
            yaw, 0.0, 0.0,
        ])
        labels.append(DET_INDEX[name])
        names.append(str(r["label_class"]))
    if not rows:
        return (np.zeros((0, 9), np.float32), np.zeros((0,), np.int64), [])
    return (np.asarray(rows, np.float32), np.asarray(labels, np.int64), names)


def build_frame(out_dir: Path, clip: str, t_ref_us: float, intr, extr,
                gt: pd.DataFrame, streams: dict) -> dict:
    """Write one <token>/ package and return its provenance record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    Ks, Rs, ts_, obs = [], [], [], {}
    for sid in CONTENT_ORDER:
        frames, times = streams[sid]
        idx = int(np.argmin(np.abs(times - t_ref_us)))
        img = frames[idx]
        it = intr[sid]
        pin = choose_pinhole(it, OUT_W, OUT_H, HFOV_OF[sid])
        mx, my, valid = rectify_map(it, OUT_W, OUT_H, pin["f"], pin["u0"], pin["v0"])
        rect = remap_bilinear(img, mx, my, valid)
        Image.fromarray(rect).save(out_dir / "images" / f"{sid}.jpg", quality=95)
        Ks.append(build_K(pin["f"], pin["u0"], pin["v0"]))
        Rs.append(extr[sid]["R"])
        ts_.append(extr[sid]["t"])
        obs[sid] = {
            "observed_frac": float(valid.mean()),
            "f_px": pin["f"], "u0": pin["u0"], "v0": pin["v0"],
            "hfov_deg": pin["hfov_deg"], "vfov_deg": pin["vfov_deg"],
            "native_cx": it.cx, "native_cy": it.cy,
            "dt_to_ref_ms": float((times[idx] - t_ref_us) / 1e3),
        }
    np.savez(out_dir / "calib.npz",
             cam_intrinsic=np.stack(Ks).astype(np.float64),
             sensor2lidar_rotation=np.stack(Rs).astype(np.float64),
             sensor2lidar_translation=np.stack(ts_).astype(np.float64),
             lidar2ego=np.eye(4, dtype=np.float64))
    boxes, labels, names = gt_boxes_at(gt, t_ref_us)
    # ⚠️ Fixed-width unicode, NOT dtype=object: `PerceptionFrame.__init__` loads
    # gt.npz with the numpy default allow_pickle=False, and an object array makes
    # it raise. (And `allow_pickle=` is not a savez option -- passing it silently
    # stores an ARRAY BY THAT NAME, which is how this was missed the first time.)
    np.savez(out_dir / "gt.npz", boxes=boxes, labels=labels,
             raw_class=np.asarray(names, dtype="<U24"))
    content = []
    for sid in CONTENT_ORDER:
        content.append({"text": TAG_OF[sid]})
        content.append({"image": sid})
    content.append({"text": "Analyze the current driving scene."})
    (out_dir / "frame.json").write_text(json.dumps({
        "dataset_type": "nuplan",
        "cam_order": CONTENT_ORDER,
        "content": content,
    }, indent=2), encoding="utf-8")
    rec = {
        "token": out_dir.name, "clip_id": clip, "t_ref_us": float(t_ref_us),
        "n_gt_boxes": int(len(boxes)), "cameras": obs,
        "feature_of": {s: FEATURE_OF[s] for s in CONTENT_ORDER},
        "view_tag": {s: TAG_OF[s] for s in CONTENT_ORDER},
    }
    (out_dir / "meta.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True, help="JSON list of {clip, tier, times_s}")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    zf = zipfile.ZipFile(ROOT / "labels" / "obstacle.offline" /
                         f"obstacle.offline.chunk_{CHUNK:04d}.zip")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for entry in plan:
        clip = entry["clip"]
        intr, extr = load_calib(CHUNK, clip)
        member = next(n for n in zf.namelist() if clip in n)
        gt = pd.read_parquet(io.BytesIO(zf.read(member)))
        bad = set(gt["reference_frame"].unique()) - {"rig"}
        if bad:
            raise RuntimeError(f"{clip}: non-rig reference_frame {bad}")

        # reference clock = the front camera's own timestamps
        _, ref_times = read_camera_stream("CAM_F0", clip, [0])
        t0, t1 = ref_times[0], ref_times[-1]
        t_refs = [t0 + (t1 - t0) * f for f in entry["fracs"]]

        streams = {}
        for sid in CONTENT_ORDER:
            _, times = read_camera_stream(sid, clip, [0]) if sid != "CAM_F0" else (None, ref_times)
            idxs = sorted({int(np.argmin(np.abs(times - t))) for t in t_refs})
            frames, times2 = read_camera_stream(sid, clip, idxs)
            streams[sid] = (frames, times2)
            print(f"  decoded {sid:8s} {len(frames)} frames", flush=True)

        for j, t in enumerate(t_refs):
            token = f"{clip[:8]}_f{j}"
            rec = build_frame(args.out / token, clip, t, intr, extr, gt, streams)
            rec["tier"] = entry["tier"]
            rec["frac"] = entry["fracs"][j]
            manifest.append(rec)
            print(f"  {token}: {rec['n_gt_boxes']} GT boxes  "
                  f"obs_frac=" + ",".join(f"{rec['cameras'][s]['observed_frac']:.2f}"
                                          for s in CONTENT_ORDER), flush=True)
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE {len(manifest)} frames -> {args.out}")


if __name__ == "__main__":
    main()
