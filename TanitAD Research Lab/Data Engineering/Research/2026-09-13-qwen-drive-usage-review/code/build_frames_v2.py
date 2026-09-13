"""PhysicalAI 7-camera f-theta rig -> a VIRTUAL nuPlan 8-camera pinhole rig (Qwen-Drive perception, v2).

⛔⛔ PERCEPTION TEACHER ONLY -- never Qwen-Drive's trajectories (PI directive 2026-09-11).

WHY v2 EXISTS -- the v1 packing (2026-09-11, `build_frames.py`) was OUT OF THE WEIGHTS'
DISTRIBUTION on every geometric axis the upstream docs pin, MEASURED against the demo's
own calib.npz on 2026-09-13:

  | property            | Qwen-Drive nuPlan rig (demo)     | v1 PhysicalAI packing           |
  |---------------------|----------------------------------|---------------------------------|
  | cameras             | 8                                | 6 (R1, L1 absent)               |
  | focal @896x512      | fx 721.0 / fy 732.4, ALL views   | 313.7 / 664.2 / 1732.3          |
  | HFOV                | 63.7 deg, ALL views              | 110 / 68 / 29 deg               |
  | principal point     | (448, 265.5) centred             | v0 up to 383.7                  |
  | yaw                 | 0, +-55, +-111, +-141, 180       | 0, +-67, +-152, 180             |
  | ego origin          | 0.325 m ABOVE the road           | ON the road                     |

  Upstream `docs/perception.md`: "trained ... at a fixed 896x512 input resolution with fixed
  camera configurations. A different camera layout or resolution is not covered by the
  released weights and will likely degrade results." v1 changed the image SCALE by 2.3x.

WHAT v2 DOES
  Every output view is a pinhole camera with nuPlan's EXACT intrinsics (1920x1080,
  f = 1545 px, principal point (960, 560)), a nuPlan-like yaw, 1 deg down-pitch and zero
  roll, synthesised from ONE PhysicalAI f-theta camera THROUGH THAT CAMERA'S OWN OPTICAL
  CENTRE. A rotation about the optical centre is an exact reprojection at every depth, so
  there is no parallax and no seam; the calibration written to calib.npz is the virtual
  camera's, so `lidar2img` is exact for every pixel.

  ⚠️ What v2 CANNOT fix, stated so it is not mistaken for fixed:
  * mounting HEIGHT: the cross cameras sit at ~0.81 m (nuPlan ~1.5 m above the ego
    origin). Changing it needs depth (novel-view synthesis); the geometry stays exact.
  * yaw: a view whose nuPlan yaw leaves its source camera's field is rotated toward that
    camera until >= MIN_OBSERVED of its pixels are real (L1/R1 and L2/R2). The yaw used is
    written per frame; it is geometry, not a guess.
  * CAM_B0: the only rear-facing source is the 30 deg tele, so B0 is ~43 % observed and
    the rest is BLACK -- recorded in meta, never filled from another camera (a fill would
    put a 2.0 m-baseline parallax seam straight behind the car).

FRAMES
  * "lidar" frame := PhysicalAI RIG frame (+x fwd, +y left, +z up, origin rear axle ON the
    road). obstacle.offline cuboids and LiDAR points live here.
  * "ego" frame := rig shifted DOWN by EGO_Z_SHIFT, i.e. lidar2ego = translate(0, 0, -0.325),
    so the road sits at z = -0.325 as in nuPlan and the occupancy z-bins line up.
  * Packed GT boxes are in the EGO frame (upstream convention); predictions come back in
    the lidar frame and the upstream visualizer converts them through lidar2ego.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

V1_CODE = Path(r"D:\Projects\TanitAD\TanitAD Research Lab\Data Engineering\Research"
               r"\2026-09-11-qwen-drive-perception-sample\code")
sys.path.insert(0, str(V1_CODE))
from ftheta_pinhole import FTheta, quat_to_R  # noqa: E402

ROOT = Path(r"C:\Users\Admin\tanitad-data\physicalai")
LIDAR_CACHE = Path(r"C:\Users\Admin\tanitad-caches\lidar-bev-20260911")
EGO_DIR = ROOT / "labels" / "egomotion_alpamayo"
CHUNK = 768

# nuPlan camera model -- MEASURED from qwen-drive/data/demo/perception/*/calib.npz: all eight
# cameras of all four nuPlan demo frames carry exactly this K at 1920x1080.
OUT_W, OUT_H = 1920, 1080
K_V = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
PITCH_DEG = 1.0          # demo cameras: 0.0 .. 2.4 deg down
EGO_Z_SHIFT = 0.325      # demo LiDAR ground mode z = -0.325 in 4/4 nuPlan frames
MIN_OBSERVED = 0.97

CAM_ORDER = ["CAM_F0", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0", "CAM_L2", "CAM_L1", "CAM_L0"]
VIEWS = {  # cam -> (view tag, PhysicalAI source, candidate yaws, nuPlan's own yaw first)
    "CAM_F0": ("<FRONT VIEW>", "camera_front_wide_120fov", [0.0]),
    "CAM_R0": ("<FRONT RIGHT VIEW>", "camera_cross_right_120fov", [-55.0, -60.0, -65.0]),
    "CAM_R1": ("<RIGHT VIEW>", "camera_cross_right_120fov", [-111.0, -105.0, -100.0, -95.0, -90.0, -85.0]),
    "CAM_R2": ("<BACK RIGHT VIEW>", "camera_rear_right_70fov", [-141.0, -145.0, -149.0, -152.0, -155.0]),
    "CAM_B0": ("<BACK VIEW>", "camera_rear_tele_30fov", [180.0]),
    "CAM_L2": ("<BACK LEFT VIEW>", "camera_rear_left_70fov", [141.0, 145.0, 149.0, 152.0, 155.0]),
    "CAM_L1": ("<LEFT VIEW>", "camera_cross_left_120fov", [111.0, 105.0, 100.0, 95.0, 90.0, 85.0]),
    "CAM_L0": ("<FRONT LEFT VIEW>", "camera_cross_left_120fov", [55.0, 60.0, 65.0]),
}
SOURCES = sorted({v[1] for v in VIEWS.values()})
INSTRUCTION = "Analyze the current driving scene."

CLASS_MAP = {"automobile": "vehicle", "heavy_truck": "vehicle", "bus": "vehicle", "trailer": "vehicle",
             "other_vehicle": "vehicle", "person": "pedestrian", "rider": "bicycle",
             "stroller": "generic_object", "protruding_object": "generic_object", "animal": "generic_object"}
DET_CLASS_NAMES = ("vehicle", "czone_sign", "bicycle", "generic_object", "pedestrian", "traffic_cone", "barrier")
DET_INDEX = {n: i for i, n in enumerate(DET_CLASS_NAMES)}
GT_TOL_US = 60_000


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- geometry
def virtual_R(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """camera->rig rotation; columns are the camera's +x right, +y down, +z forward in rig."""
    psi, phi = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(psi) * math.cos(phi), math.sin(psi) * math.cos(phi), -math.sin(phi)])
    right = np.array([math.sin(psi), -math.cos(psi), 0.0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=1)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    return R


def theta_max(ft: FTheta) -> float:
    """Largest incidence angle anywhere on the sensor (the far corner)."""
    r = math.hypot(max(ft.cx, ft.width - 1 - ft.cx), max(ft.cy, ft.height - 1 - ft.cy))
    return ft.theta_of_r(r, hi=2.2)


def view_map(R_v, R_s, ft: FTheta, scale: int):
    """(map_x, map_y, valid) into the (possibly pre-downscaled) SOURCE image for every output pixel."""
    u, v = np.meshgrid(np.arange(OUT_W, dtype=np.float64), np.arange(OUT_H, dtype=np.float64))
    rays = np.stack([(u - K_V[0, 2]) / K_V[0, 0], (v - K_V[1, 2]) / K_V[1, 1], np.ones_like(u)], axis=-1)
    rs = rays @ R_v.T @ R_s            # virtual cam -> rig -> source cam   (R_s^T applied row-wise)
    x, y, z = rs[..., 0], rs[..., 1], rs[..., 2]
    rho = np.hypot(x, y)
    th = np.arctan2(rho, z)
    r = ft.r_of_theta(th)
    k = np.where(rho > 1e-12, r / np.maximum(rho, 1e-12), 0.0)
    mx = (ft.cx + x * k) / scale
    my = (ft.cy + y * k) / scale
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


def box_downscale(img: np.ndarray, s: int) -> np.ndarray:
    if s == 1:
        return img
    h, w = (img.shape[0] // s) * s, (img.shape[1] // s) * s
    return img[:h, :w].reshape(h // s, s, w // s, s, 3).mean(axis=(1, 3)).astype(np.uint8)


# --------------------------------------------------------------------------- data access
def load_calib(clip: str):
    ip = ROOT / "calibration" / "camera_intrinsics" / f"camera_intrinsics.chunk_{CHUNK:04d}.parquet"
    ep = ROOT / "calibration" / "sensor_extrinsics" / f"sensor_extrinsics.chunk_{CHUNK:04d}.parquet"
    idf = pd.read_parquet(ip).reset_index()
    edf = pd.read_parquet(ep).reset_index()
    idf, edf = idf[idf["clip_id"] == clip], edf[edf["clip_id"] == clip]
    intr, extr = {}, {}
    for feat in SOURCES:
        r = idf[idf["camera_name"] == feat]
        assert len(r) == 1, f"{feat} intrinsics rows={len(r)}"
        r = r.iloc[0]
        intr[feat] = FTheta(poly=tuple(float(r[f"fw_poly_{i}"]) for i in range(5)),
                            cx=float(r["cx"]), cy=float(r["cy"]), width=int(r["width"]), height=int(r["height"]))
    for feat in SOURCES + ["lidar_top_360fov"]:
        e = edf[edf["sensor_name"] == feat]
        assert len(e) == 1, f"{feat} extrinsics rows={len(e)}"
        e = e.iloc[0]
        extr[feat] = {"R": quat_to_R(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"])),
                      "t": np.array([float(e["x"]), float(e["y"]), float(e["z"])])}
    return intr, extr


def open_stream(feat: str, clip: str):
    # loose files first (two layouts exist on this box), then the chunk zip
    for mp4, tsp in ((ROOT / "r0" / "camera_front_wide" / f"{clip}.{feat}.mp4",
                      ROOT / "r0" / "camera_front_wide" / f"{clip}.{feat}.timestamps.parquet"),
                     (ROOT / "camera" / feat / f"{clip}.mp4",
                      ROOT / "camera" / feat / f"{clip}.timestamps.parquet")):
        if mp4.exists() and tsp.exists():
            return open(mp4, "rb"), pd.read_parquet(tsp)
    z = zipfile.ZipFile(ROOT / "camera" / feat / f"{feat}.chunk_{CHUNK:04d}.zip")
    ts = pd.read_parquet(io.BytesIO(z.read(f"{clip}.{feat}.timestamps.parquet")))
    return io.BytesIO(z.read(f"{clip}.{feat}.mp4")), ts


def times_of(ts: pd.DataFrame) -> np.ndarray:
    col = next(c for c in ts.columns if "time" in c.lower())
    return ts[col].to_numpy(np.float64)


def gt_boxes_at(gt: pd.DataFrame, t_us: float):
    rows, labels, names = [], [], []
    for _, g in gt.groupby("track_id", sort=False):
        dt = g["timestamp_us"].to_numpy(np.float64) - t_us
        k = int(np.argmin(np.abs(dt)))
        if abs(dt[k]) > GT_TOL_US:
            continue
        r = g.iloc[k]
        name = CLASS_MAP.get(str(r["label_class"]))
        if name is None:
            continue
        R = quat_to_R(r["orientation_x"], r["orientation_y"], r["orientation_z"], r["orientation_w"])
        rows.append([float(r["center_x"]), float(r["center_y"]),
                     float(r["center_z"]) - float(r["size_z"]) / 2.0 - EGO_Z_SHIFT,   # EGO frame, z at BOTTOM
                     float(r["size_x"]), float(r["size_y"]), float(r["size_z"]),
                     float(math.atan2(R[1, 0], R[0, 0])), 0.0, 0.0])
        labels.append(DET_INDEX[name])
        names.append(str(r["label_class"]))
    if not rows:
        return np.zeros((0, 9), np.float32), np.zeros((0,), np.int64), np.zeros((0,), "<U24")
    return np.asarray(rows, np.float32), np.asarray(labels, np.int64), np.asarray(names, "<U24")


class Lidar:
    """Nearest-spin LiDAR in the RIG frame, deskewed to the camera instant when egomotion exists."""

    def __init__(self, clip: str, ext: dict):
        import pyarrow.parquet as pq
        self.path = LIDAR_CACHE / f"{clip}.lidar_top_360fov.parquet"
        self.ok = self.path.exists()
        if not self.ok:
            return
        self.pf = pq.ParquetFile(self.path)
        sp = pq.read_table(self.path, columns=["spin_start_timestamp", "spin_end_timestamp"]).to_pydict()
        self.mid = (np.asarray(sp["spin_start_timestamp"], np.int64) + np.asarray(sp["spin_end_timestamp"], np.int64)) // 2
        self.ext = ext
        ep = EGO_DIR / f"{clip}.parquet"
        self.ego = pq.read_table(ep).to_pydict() if ep.exists() else None

    def at(self, t_us: float):
        import DracoPy
        si = int(np.abs(self.mid - t_us).argmin())
        blob = self.pf.read_row_group(si, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py()
        pc = DracoPy.decode(blob)
        pts = np.asarray(pc.points, np.float64)
        assert pts.shape[0] > 0 and float(np.abs(pts).max()) > 0, "empty LiDAR decode"
        rig = pts @ self.ext["R"].T + self.ext["t"]
        dt_ms = float((self.mid[si] - t_us) / 1e3)
        deskew = False
        if self.ego is not None:
            attrs = {a["unique_id"]: a["data"] for a in pc.attributes}
            if 1 in attrs or 2 in attrs:
                pt_ts = attrs[2][:, 0].astype(np.int64) if 2 in attrs else None
                if pt_ts is not None:
                    ts = np.asarray(self.ego["timestamp"], np.float64)
                    i = int(np.clip(np.searchsorted(ts, t_us), 1, len(ts) - 1))
                    v = math.hypot(self.ego["vx"][i], self.ego["vy"][i]) if "vx" in self.ego else 0.0
                    yaw = 2.0 * np.arctan2(np.asarray(self.ego["qz"]), np.asarray(self.ego["qw"]))
                    dts = (ts[i] - ts[i - 1]) * 1e-6
                    wz = math.atan2(math.sin(yaw[i] - yaw[i - 1]), math.cos(yaw[i] - yaw[i - 1])) / dts if dts > 0 else 0.0
                    d = (pt_ts.astype(np.float64) - t_us) * 1e-6
                    c, s = np.cos(-wz * d), np.sin(-wz * d)
                    xs = rig[:, 0] + v * d
                    rig = np.stack([xs * c - rig[:, 1] * s, xs * s + rig[:, 1] * c, rig[:, 2]], axis=1)
                    deskew = True
        return rig.astype(np.float32), dt_ms, deskew


def pseudo_occ(points_rig: np.ndarray, boxes_ego: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """LiDAR-derived VISUAL reference on nuPlan's occupancy grid, EGO frame, [X, Y, Z].

    ⛔ NOT semantic ground truth: ground returns are drawn as `driveable` because nothing in
    PhysicalAI separates road from sidewalk; everything else that is not inside a labelled
    box is `background`.
    """
    occ = np.full((200, 200, 16), 9, np.int8)
    if points_rig is None or len(points_rig) == 0:
        return occ
    p = points_rig.astype(np.float64).copy()
    p[:, 2] -= EGO_Z_SHIFT
    ix = np.floor((p[:, 0] + 50.0) / 0.5).astype(int)
    iy = np.floor((p[:, 1] + 50.0) / 0.5).astype(int)
    iz = np.floor((p[:, 2] + 4.0) / 0.5).astype(int)
    m = (ix >= 0) & (ix < 200) & (iy >= 0) & (iy < 200) & (iz >= 0) & (iz < 16)
    p, ix, iy, iz = p[m], ix[m], iy[m], iz[m]
    # local ground: 4 m cell minimum, a point within 0.30 m of it is ground
    gx, gy = ix // 8, iy // 8
    key = gx * 25 + gy
    zmin = np.full(625, np.inf)
    np.minimum.at(zmin, key, p[:, 2])
    ground = p[:, 2] < zmin[key] + 0.30
    lab = np.where(ground, 7, 8).astype(np.int8)
    for b, l in zip(boxes_ego, labels):
        c, s = math.cos(-b[6]), math.sin(-b[6])
        dx, dy = p[:, 0] - b[0], p[:, 1] - b[1]
        lx, ly = dx * c - dy * s, dx * s + dy * c
        inside = (np.abs(lx) <= b[3] / 2 + 0.2) & (np.abs(ly) <= b[4] / 2 + 0.2) & \
                 (p[:, 2] >= b[2] - 0.1) & (p[:, 2] <= b[2] + b[5] + 0.2)
        lab[inside] = int(l)
    order = np.argsort(lab == 8)       # write background first so agents and ground win ties
    occ[ix[order[::-1]], iy[order[::-1]], iz[order[::-1]]] = lab[order[::-1]]
    return occ


# --------------------------------------------------------------------------- build
def build_clip(clip: str, t_refs: list[float], tokens: list[str], out: Path) -> list[dict]:
    intr, extr = load_calib(clip)
    zf = zipfile.ZipFile(ROOT / "labels" / "obstacle.offline" / f"obstacle.offline.chunk_{CHUNK:04d}.zip")
    gt = pd.read_parquet(io.BytesIO(zf.read(next(n for n in zf.namelist() if clip in n))))
    assert set(gt["reference_frame"].unique()) <= {"rig"}, "non-rig obstacle frame"

    # ---- per-view virtual camera, chosen once per clip from calibration only ----------
    views = {}
    for cam in CAM_ORDER:
        tag, feat, yaws = VIEWS[cam]
        ft = intr[feat]
        scale = max(1, int(math.floor(ft.poly[1] / K_V[0, 0])))
        best = None
        for yaw in yaws:
            R_v = virtual_R(yaw, PITCH_DEG)
            mx, my, valid = view_map(R_v, extr[feat]["R"], ft, scale)
            frac = float(valid.mean())
            if best is None or frac > best[4] + 1e-9:
                best = (yaw, R_v, (mx, my, valid), scale, frac)
            if frac >= MIN_OBSERVED:
                best = (yaw, R_v, (mx, my, valid), scale, frac)
                break
        views[cam] = {"tag": tag, "feat": feat, "yaw": best[0], "R": best[1], "map": best[2],
                      "scale": best[3], "observed_frac": best[4], "t": extr[feat]["t"]}
        print(f"  {cam}: {feat:28s} yaw {best[0]:7.1f} (nuPlan {yaws[0]:7.1f})  observed {best[4]:.3f}  prefilter x{best[3]}",
              flush=True)

    dirs = [out / tok for tok in tokens]
    for d in dirs:
        (d / "images").mkdir(parents=True, exist_ok=True)

    # ---- stream each SOURCE camera once, writing every view it feeds ------------------
    dt_ms = {tok: {} for tok in tokens}
    for feat in SOURCES:
        cams = [c for c in CAM_ORDER if views[c]["feat"] == feat]
        handle, ts = open_stream(feat, clip)
        times = times_of(ts)
        want = {}
        for tok, t in zip(tokens, t_refs):
            i = int(np.argmin(np.abs(times - t)))
            want.setdefault(i, []).append(tok)
            for c in cams:
                dt_ms[tok][c] = float((times[i] - t) / 1e3)
        todo = set(want)
        with av.open(handle) as cont:
            st = cont.streams.video[0]
            st.thread_type = "AUTO"
            for i, fr in enumerate(cont.decode(st)):
                if i in todo:
                    img = box_downscale(fr.to_ndarray(format="rgb24"), views[cams[0]]["scale"])
                    for c in cams:
                        mx, my, valid = views[c]["map"]
                        res = remap(img, mx, my, valid)
                        for tok in want[i]:
                            Image.fromarray(res).save(out / tok / "images" / f"{c}.jpg", quality=95)
                    todo.discard(i)
                    if not todo:
                        break
        assert not todo, f"{feat}: {len(todo)} wanted frames never decoded"
        print(f"  decoded {feat}: {len(want)} frames -> {len(cams)} view(s)", flush=True)

    lid = Lidar(clip, extr["lidar_top_360fov"])
    recs = []
    for tok, t in zip(tokens, t_refs):
        d = out / tok
        np.savez(d / "calib.npz",
                 cam_intrinsic=np.stack([K_V] * len(CAM_ORDER)).astype(np.float64),
                 sensor2lidar_rotation=np.stack([views[c]["R"] for c in CAM_ORDER]).astype(np.float64),
                 sensor2lidar_translation=np.stack([views[c]["t"] for c in CAM_ORDER]).astype(np.float64),
                 lidar2ego=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -EGO_Z_SHIFT], [0, 0, 0, 1]], np.float64))
        boxes, labels, names = gt_boxes_at(gt, t)
        lidar_rec = None
        pts = None
        if lid.ok:
            pts, ldt, desk = lid.at(t)
            # lidar.npy is for the upstream visualizer only (it draws <= 80k points); the
            # pseudo-occupancy below is built from the FULL sweep
            keep = np.random.default_rng(0).permutation(len(pts))[:100_000]
            np.save(d / "lidar.npy", pts[np.sort(keep)])
            r = np.hypot(pts[:, 0], pts[:, 1])
            h, e = np.histogram(pts[(r > 3) & (r < 20), 2], bins=np.arange(-2, 2.001, 0.05))
            lidar_rec = {"n_points": int(len(pts)), "spin_mid_dt_ms": ldt, "deskewed": desk,
                         "ground_z_mode_rig_m": float(e[int(np.argmax(h))] + 0.025)}
        occ = pseudo_occ(pts, boxes, labels)
        np.savez(d / "gt.npz", boxes=boxes, labels=labels, raw_class=names, occ=occ,
                 map=np.zeros((200, 400), np.int8))
        content = []
        for c in CAM_ORDER:
            content += [{"text": views[c]["tag"]}, {"image": c}]
        content.append({"text": INSTRUCTION})
        (d / "frame.json").write_text(json.dumps({"dataset_type": "nuplan", "cam_order": CAM_ORDER,
                                                  "content": content}, indent=2), encoding="utf-8")
        rec = {"token": tok, "clip_sha12": sha12(clip), "clip_id": clip, "t_ref_us": float(t),
               "builder": "build_frames_v2", "ego_z_shift_m": EGO_Z_SHIFT, "n_gt_boxes": int(len(boxes)),
               "gt_occ": "LiDAR pseudo-occupancy (ground drawn as driveable; NOT semantic GT)" if lid.ok else "none",
               "gt_map": "none -- PhysicalAI-AV ships no map",
               "lidar": lidar_rec,
               "views": {c: {"source": views[c]["feat"], "yaw_deg": views[c]["yaw"],
                             "nuplan_yaw_deg": VIEWS[c][2][0], "pitch_deg": PITCH_DEG,
                             "observed_frac": round(views[c]["observed_frac"], 4),
                             "prefilter": views[c]["scale"], "dt_to_ref_ms": dt_ms[tok].get(c)}
                         for c in CAM_ORDER}}
        (d / "meta.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        recs.append(rec)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fracs", help="comma list of clip fractions (the v1 sample instants)")
    g.add_argument("--hz", type=float, help="sample the whole clip at this rate")
    ap.add_argument("--margin-s", type=float, default=0.5)
    a = ap.parse_args()

    handle, ts = open_stream("camera_front_wide_120fov", a.clip)
    ref = times_of(ts)
    t0, t1 = float(ref[0]), float(ref[-1])
    if a.fracs:
        fr = [float(x) for x in a.fracs.split(",")]
        t_refs = [t0 + (t1 - t0) * f for f in fr]
        tokens = [f"{a.clip[:8]}_f{j}" for j in range(len(fr))]
    else:
        t_refs = list(np.arange(t0 + a.margin_s * 1e6, t1 - a.margin_s * 1e6, 1e6 / a.hz))
        tokens = [f"{a.clip[:8]}_t{int(round((t - t0) / 1e3)):05d}" for t in t_refs]
    print(f"{a.clip[:8]}: {len(t_refs)} instants", flush=True)
    recs = build_clip(a.clip, t_refs, tokens, a.out)
    man = a.out / "MANIFEST.jsonl"
    with open(man, "a", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"WROTE {len(recs)} frames -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
