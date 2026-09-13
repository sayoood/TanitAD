"""Qwen-Drive map / occupancy quality on OUR PhysicalAI frames, with the demo-calibrated metrics.

Arms (same frames, same evidence):
  v1         the rejected 2026-09-11 packing (legacy instants only)
  v2         single-frame v2 packing
  v2_fuse1s  v2 maps fused over +-5 frames (+-1.0 s) into the current frame through ego poses, majority vote
  v2_fuse2s  the same over +-10 frames (+-2.0 s)
  SHUFFLED   the v2 prediction of the same frame index from a DIFFERENT clip      (no-information control)
  MIRRORED   the v2 prediction flipped left-right                                  (geometry control)
  PRIOR      per-cell majority class of all v2 maps of the OTHER clips             (scene-agnostic control)
Evidence per frame: the full LiDAR sweep nearest the frame (rig frame, intensity kept), GT agent boxes,
the ego's own driven path from egomotion, and high-intensity ground returns as a paint proxy.
"""
import io, json, math, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import DracoPy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapq_core as mc  # noqa: E402

V2 = Path(r"C:/Users/Admin/qwenvis/v2")
PRED = V2 / "preds"
LIDAR = Path(r"C:/Users/Admin/tanitad-caches/lidar-bev-20260911")
EGO_ZIP = Path(r"C:/Users/Admin/tanitad-data/physicalai/labels/egomotion/egomotion.chunk_0768.zip")
EXT = Path(r"C:/Users/Admin/tanitad-data/physicalai/calibration/sensor_extrinsics/sensor_extrinsics.chunk_0768.parquet")
Z = 0.325
SEQS = ["4fbd97b6a4b7", "73495082f98b", "0d90d20036a3", "6924358fafe0"]
CLIP = {"4fbd97b6a4b7": "4fbd97b6a4b7", "73495082f98b": "73495082f98b",
        "0d90d20036a3": "0d90d20036a3", "6924358fafe0": "6924358fafe0"}


def quat_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw); qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


class Clip:
    def __init__(self, c8):
        cid = CLIP[c8]
        e = pd.read_parquet(EXT).reset_index(); e = e[(e["clip_id"] == cid) & (e["sensor_name"] == "lidar_top_360fov")].iloc[0]
        self.R, self.t = quat_R(e.qx, e.qy, e.qz, e.qw), np.array([e.x, e.y, e.z])
        self.pf = pq.ParquetFile(LIDAR / f"{cid}.lidar_top_360fov.parquet")
        sp = pq.read_table(LIDAR / f"{cid}.lidar_top_360fov.parquet", columns=["spin_start_timestamp", "spin_end_timestamp"]).to_pydict()
        self.mid = (np.asarray(sp["spin_start_timestamp"], np.int64) + np.asarray(sp["spin_end_timestamp"], np.int64)) // 2
        z = zipfile.ZipFile(EGO_ZIP)
        self.ego = pd.read_parquet(io.BytesIO(z.read(next(n for n in z.namelist() if cid in n))))
        self.ets = self.ego["timestamp"].to_numpy(np.float64)

    def pose(self, t):
        r = self.ego.iloc[int(np.argmin(np.abs(self.ets - t)))]
        T = np.eye(4); T[:3, :3] = quat_R(r.qx, r.qy, r.qz, r.qw); T[:3, 3] = [r.x, r.y, r.z]
        return T

    def sweep_ego(self, t):
        """Nearest sweep, rig frame -> v2 ego frame (z - 0.325); column 3 = intensity."""
        si = int(np.abs(self.mid - t).argmin())
        pc = DracoPy.decode(self.pf.read_row_group(si, columns=["draco_encoded_pointcloud"]).column(0)[0].as_py())
        pts = np.asarray(pc.points, np.float64); inten = {a["unique_id"]: a["data"] for a in pc.attributes}[1][:, 0].astype(np.float64)
        rig = pts @ self.R.T + self.t
        rig[:, 2] -= Z
        return np.c_[rig, inten], float((self.mid[si] - t) / 1e3)

    def path_xy(self, t):
        """The ego's own driven path (all egomotion samples of the clip) in the rig frame at t, inside the map window."""
        T = self.pose(t)
        P = np.c_[self.ego[["x", "y", "z"]].to_numpy(np.float64), np.ones(len(self.ego))]
        q = (P @ np.linalg.inv(T).T)[:, :2]
        return q[(np.abs(q[:, 0]) <= 30) & (np.abs(q[:, 1]) <= 15)]


def paint_points(ground):
    r = np.hypot(ground[:, 0], ground[:, 1]); g = ground[r <= 25.0]
    if not len(g):
        return g[:, :2]
    thr = max(20.0, 4.0 * float(np.median(g[:, 3])))
    return g[g[:, 3] >= thr, :2]


GRID = None


def grid_xy():
    global GRID
    if GRID is None:
        i, j = np.meshgrid(np.arange(200), np.arange(400), indexing="ij")
        GRID = (np.c_[(-30 + (j.ravel() + 0.5) * 0.15), (-15 + (i.ravel() + 0.5) * 0.15)], i.ravel(), j.ravel())
    return GRID


def fuse(maps, poses, j, half):
    """Majority vote over frames j-half..j+half, each warped into frame j's map raster."""
    xy, ii, jj = grid_xy()
    H = np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))]
    votes = np.zeros((len(xy), 6), np.int32)
    for k in range(max(0, j - half), min(len(maps), j + half + 1)):
        q = H @ (np.linalg.inv(poses[k]) @ poses[j]).T
        ci = np.floor((q[:, 1] + 15) / 0.15).astype(int); cj = np.floor((q[:, 0] + 30) / 0.15).astype(int)
        ok = (ci >= 0) & (ci < 200) & (cj >= 0) & (cj < 400)
        votes[np.flatnonzero(ok), maps[k][ci[ok], cj[ok]]] += 1
    out = votes.argmax(axis=1)
    none = votes.sum(axis=1) == 0
    out[none] = maps[j].ravel()[none]
    return out.reshape(200, 400)


def main():
    rows = {}
    per_clip = {}
    # ---- sequences ---------------------------------------------------------------------------
    maps_all, occ_all, toks_all = {}, {}, {}
    for c8 in SEQS:
        toks = sorted(p.name for p in (V2 / f"seq_{c8}").iterdir() if p.is_dir())
        toks_all[c8] = toks
        maps_all[c8] = [np.load(PRED / "v2" / f"out_seq_{c8}" / f"{t}.npz")["map"].astype(int) for t in toks]
        occ_all[c8] = [np.load(PRED / "v2" / f"out_seq_{c8}" / f"{t}.npz")["occ"] for t in toks]
    prior = {}
    for c8 in SEQS:
        others = np.stack([m for o in SEQS if o != c8 for m in maps_all[o]])
        cnt = np.stack([(others == k).sum(axis=0) for k in range(6)])
        prior[c8] = cnt.argmax(axis=0)
    for ci, c8 in enumerate(SEQS):
        clip = Clip(c8)
        toks = toks_all[c8]
        metas = [json.loads((V2 / f"seq_{c8}" / t / "meta.json").read_text(encoding="utf-8")) for t in toks]
        poses = [clip.pose(m["t_ref_us"]) for m in metas]
        other = SEQS[(ci + 1) % len(SEQS)]
        for j, (tok, meta) in enumerate(zip(toks, metas)):
            pts, dtl = clip.sweep_ego(meta["t_ref_us"])
            g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz")
            boxes, veh = g["boxes"], g["boxes"][g["labels"] == 0]
            split = mc.split_points(pts, boxes)
            edge = mc.edge_evidence(split[1], split[0])
            kw = dict(ego_path_xy=clip.path_xy(meta["t_ref_us"]), paint_xy=paint_points(split[1]), split=split, edge=edge)
            m, o = maps_all[c8][j], occ_all[c8][j]
            jo = min(j, len(maps_all[other]) - 1)
            arms = {"v2": (m, o), "v2_fuse1s": (fuse(maps_all[c8], poses, j, 5), o), "v2_fuse2s": (fuse(maps_all[c8], poses, j, 10), o),
                    "SHUFFLED": (maps_all[other][jo], occ_all[other][jo]), "MIRRORED": (m[::-1, :], o.reshape(200, 200, 16)[:, ::-1, :]),
                    "PRIOR": (prior[c8], None)}
            for arm, (mm, oo) in arms.items():
                r = mc.frame_metrics(mm, oo, pts, boxes, veh, **kw)
                rows.setdefault(arm, []).append(r)
                per_clip.setdefault(f"{c8}:{arm}", []).append(r)
        print(f"  {c8}: {len(toks)} frames scored", flush=True)
    # ---- legacy instants: v1 vs v2 (clips with LiDAR only) ------------------------------------
    leg = sorted(p.name for p in (V2 / "frames_legacy").iterdir() if p.is_dir() and p.name.split("_")[0] in CLIP)
    clips = {}
    for tok in leg:
        c8 = tok.split("_")[0]
        clip = clips.setdefault(c8, Clip(c8))
        meta = json.loads((V2 / "frames_legacy" / tok / "meta.json").read_text(encoding="utf-8"))
        pts, _ = clip.sweep_ego(meta["t_ref_us"])
        g = np.load(V2 / "frames_legacy" / tok / "gt.npz")
        split = mc.split_points(pts, g["boxes"]); edge = mc.edge_evidence(split[1], split[0])
        kw = dict(ego_path_xy=clip.path_xy(meta["t_ref_us"]), paint_xy=paint_points(split[1]), split=split, edge=edge)
        for arm, src in (("legacy_v1", PRED / "out"), ("legacy_v2", PRED / "v2" / "out_legacy")):
            z = np.load(src / f"{tok}.npz")
            rows.setdefault(arm, []).append(mc.frame_metrics(z["map"].astype(int), z["occ"], pts, g["boxes"], g["boxes"][g["labels"] == 0], **kw))
    res = {arm: mc.aggregate(r) for arm, r in rows.items()}
    res["per_clip"] = {k: mc.aggregate(v) for k, v in per_clip.items()}
    res["per_clip_median_of_frames"] = {k: {m: round(float(np.median([a / n for r in v for mm, (a, n) in r.items() if mm == m and n > 0])), 4)
                                            for m in sorted({mm for r in v for mm in r})
                                            if any(mm == m and n > 0 for r in v for mm, (a, n) in r.items())}
                                        for k, v in per_clip.items()}
    res["demo_calibration"] = json.loads((V2 / "demo_calibration.json").read_text(encoding="utf-8"))
    (V2 / "mapq_ours.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    keys = ["MAP_A", "MAP_A2", "MAP_B", "MAP_C", "MAP_D", "MAP_E", "OCC_A", "OCC_A2", "OCC_B", "OCC_C"]
    print(f"{'arm':10s} " + " ".join(f"{k:>8s}" for k in keys))
    cal = res["demo_calibration"]
    for arm in ("GT", "PRED", "SHUFFLED", "MIRRORED"):
        print(f"demo:{arm:5s} " + " ".join(f"{cal[arm][k]['value']:>8}" if k in cal[arm] else f"{'-':>8s}" for k in keys))
    for arm in ("legacy_v1", "legacy_v2", "v2", "v2_fuse1s", "v2_fuse2s", "SHUFFLED", "MIRRORED", "PRIOR"):
        print(f"{arm:10s} " + " ".join(f"{res[arm][k]['value']:>8}" if k in res[arm] else f"{'-':>8s}" for k in keys))
    print("ZZMAPQ-OURS-DONEZZ")


if __name__ == "__main__":
    main()
