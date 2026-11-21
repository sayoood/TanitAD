"""MAP_B per frame for two world maps, with the scorer's OWN ego-path evidence (mapq_ours.Clip.path_xy), and the codes
of the second map where the first is on-road. Thor paths as in thor_run.py."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "stub"))
import mapq_ours as mo  # noqa: E402
D = Path("/home/nvidia/sam3map/data")
mo.V2, mo.PRED, mo.LIDAR = Path("/home/nvidia/qwendrive/v2"), D / "preds", D
mo.EGO_ZIP, mo.EXT = D / "egomotion.chunk_0768.zip", D / "sensor_extrinsics.chunk_0768.parquet"
import mapq_core as mc  # noqa: E402
import sam3map_score as S  # noqa: E402

c8, a_p, b_p, npz_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
A = dict(np.load(a_p, allow_pickle=True)); B = dict(np.load(b_p, allow_pickle=True))
clip = mo.Clip(c8)
rows, codes = [], {}
for f in sorted(npz_dir.glob("[0-9][0-9][0-9].npz")):
    z = np.load(f, allow_pickle=True); T = np.array(z["T_world_rig"])
    meta = json.loads((mo.V2 / f"seq_{c8}" / str(z["tok"]) / "meta.json").read_text())
    path = clip.path_xy(meta["t_ref_us"])
    ra, rb = S.worldmap_raster(A, T), S.worldmap_raster(B, T)
    ca, ka = mc.map_cls(ra, path); cb, kb = mc.map_cls(rb, path)
    ona = np.isin(ca, mc.ON_ROAD); onb = np.isin(cb, mc.ON_ROAD)
    rows.append((f.name, len(ca), float(ona.mean()) if len(ca) else None, float(onb.mean()) if len(cb) else None))
    # raw world-map codes of B at path points A calls on-road but B does not
    pin = path[ka]
    w = pin @ T[:2, :2].T + T[:2, 3]
    cls, (x0, y0), res = B["cls"], B["origin"], float(B["res"])
    i = np.floor((w[:, 0] - x0) / res).astype(int); j = np.floor((w[:, 1] - y0) / res).astype(int)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    raw = np.full(len(w), 254, np.int64); raw[ok] = cls[i[ok], j[ok]]
    for k in raw[ona & ~onb]:
        codes[int(k)] = codes.get(int(k), 0) + 1
pa = [r[2] for r in rows if r[2] is not None]; pb = [r[3] for r in rows if r[3] is not None]
worst = sorted(rows, key=lambda r: (r[3] or 0) - (r[2] or 0))[:8]
print(json.dumps({"n_path_points_per_frame_mean": float(np.mean([r[1] for r in rows])), "mean_on_road_A": round(float(np.mean(pa)), 4),
                  "mean_on_road_B": round(float(np.mean(pb)), 4), "B_codes_where_A_on_road_B_not": codes, "worst_frames": worst}))
