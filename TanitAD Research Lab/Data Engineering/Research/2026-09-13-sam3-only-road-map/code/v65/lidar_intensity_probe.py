"""Does PhysicalAI's LiDAR intensity separate road paint from asphalt? Ground returns (mapq_core.split_points) within 25 m of
every 4th frame, looked up in a world map; intensity percentiles per map class, plus the share passing mapq_ours.paint_points'
rule (>= max(20, 4 x sweep median)) and a per-sweep percentile rule.
Usage: lidar_intensity_probe.py <c8> <npz dir> <render dir>"""
import json, sys
from pathlib import Path
HERE = Path("/home/nvidia/sam3map/eval")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "stub"))
import numpy as np
V2 = Path("/home/nvidia/qwendrive/v2"); D = Path("/home/nvidia/sam3map/data")
import mapq_ours as mo
import mapq_core as mc
mo.V2, mo.PRED, mo.LIDAR = V2, D / "preds", D
mo.EGO_ZIP, mo.EXT = D / "egomotion.chunk_0768.zip", D / "sensor_extrinsics.chunk_0768.parquet"
c8, npz_dir, rdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
clip = mo.Clip(c8)
wm = dict(np.load(rdir / "worldmap.npz", allow_pickle=True)); cls = wm["cls"]; (wx0, wy0), res = wm["origin"], float(wm["res"])
by = {k: [] for k in (0, 1, 2, 3, 4, 5, 6, 7)}
rule = {k: [0, 0, 0] for k in by}
for f in sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))[::4]:
    z = np.load(f, allow_pickle=True); tok = str(z["tok"]); T = z["T_world_rig"]
    meta = json.loads((V2 / f"seq_{c8}" / tok / "meta.json").read_text(encoding="utf-8"))
    pts = np.load(D / "sweeps" / c8 / f"{int(np.abs(clip.mid - meta['t_ref_us']).argmin()):04d}.npy")
    g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz")
    ground = mc.split_points(pts, g["boxes"])[1]
    ground = ground[np.hypot(ground[:, 0], ground[:, 1]) <= 25.0]
    if not len(ground):
        continue
    med = float(np.median(ground[:, 3])); thr = max(20.0, 4.0 * med)
    w = ground[:, :2] @ T[:2, :2].T + T[:2, 3]
    i = np.floor((w[:, 0] - wx0) / res).astype(int); j = np.floor((w[:, 1] - wy0) / res).astype(int)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    c = np.full(len(ground), 255); c[ok] = cls[i[ok], j[ok]]
    road_int = ground[c == 1, 3]
    p97 = float(np.percentile(road_int, 97)) if len(road_int) > 100 else np.inf
    for k in by:
        sel = c == k
        if sel.any():
            by[k].append(ground[sel, 3]); rule[k][0] += int(sel.sum()); rule[k][1] += int((ground[sel, 3] >= thr).sum()); rule[k][2] += int((ground[sel, 3] > p97).sum())
print("sweep intensity dtype/range example:", pts.dtype, float(pts[:, 3].min()), float(pts[:, 3].max()))
for k, v in by.items():
    if not v:
        continue
    a = np.concatenate(v)
    print(f"class {k}: n {len(a):7d}  p10 {np.percentile(a, 10):7.2f} p50 {np.percentile(a, 50):7.2f} p90 {np.percentile(a, 90):7.2f} p99 {np.percentile(a, 99):7.2f}"
          f"  pass max(20,4med) {rule[k][1] / max(rule[k][0], 1):.3f}  pass > road p97 {rule[k][2] / max(rule[k][0], 1):.3f}")
print("ZZINT-DONEZZ")
