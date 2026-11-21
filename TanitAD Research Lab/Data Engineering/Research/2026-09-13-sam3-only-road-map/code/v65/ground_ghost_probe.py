"""Is MAP_A2's local ground (mapq_core.split_points: MINIMUM z of a 2 m cell) pulled down by returns BELOW the road surface
(wet-road multipath / reflections)? Per frame within 30 m: 2 m cells with >= 20 returns; compare the minimum with the 10th
percentile; count cells where min < p10 - 0.5 m, and count 'tall' points (min-ground rule) that sit within 0.3 m of the
p10 ground (i.e. road-surface returns the metric calls tall obstacles). Night and day clip.
Usage: ground_ghost_probe.py <c8>=<npz dir> [...]"""
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
for arg in sys.argv[1:]:
    c8, nd = arg.split("=", 1)
    clip = mo.Clip(c8)
    tot = {"cells": 0, "ghost_cells": 0, "tall": 0, "tall_on_surface": 0, "pts_below_p10_by_0.5": 0, "pts": 0}
    for f in sorted(Path(nd).glob("[0-9][0-9][0-9].npz"))[::4]:
        z = np.load(f, allow_pickle=True); tok = str(z["tok"])
        meta = json.loads((V2 / f"seq_{c8}" / tok / "meta.json").read_text(encoding="utf-8"))
        pts = np.load(D / "sweeps" / c8 / f"{int(np.abs(clip.mid - meta['t_ref_us']).argmin()):04d}.npy")
        p = pts[np.hypot(pts[:, 0], pts[:, 1]) <= 32]
        key = (np.floor((p[:, 0] + 64) / 2).astype(int) * 64 + np.floor((p[:, 1] + 64) / 2).astype(int))
        order = np.argsort(key, kind="stable"); ks, zs = key[order], p[order, 2]
        starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]; ends = np.r_[starts[1:], len(ks)]
        zmin = {}; p10 = {}
        for a, b in zip(starts, ends):
            if b - a >= 20:
                zmin[ks[a]] = zs[a:b].min(); p10[ks[a]] = np.percentile(zs[a:b], 10)
                tot["cells"] += 1; tot["ghost_cells"] += int(zmin[ks[a]] < p10[ks[a]] - 0.5)
        g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz")
        tall = mc.split_points(pts, g["boxes"])[4]
        kt = (np.floor((tall[:, 0] + 64) / 2).astype(int) * 64 + np.floor((tall[:, 1] + 64) / 2).astype(int))
        pg = np.array([p10.get(k, np.nan) for k in kt])
        tot["tall"] += len(tall); tot["tall_on_surface"] += int((np.abs(tall[:, 2] - pg) <= 0.3).sum())
        pk = np.array([p10.get(k, np.nan) for k in key])
        tot["pts"] += len(p); tot["pts_below_p10_by_0.5"] += int((p[:, 2] < pk - 0.5).sum())
    print(c8, tot, " ghost-cell share", round(tot["ghost_cells"] / max(tot["cells"], 1), 3), " tall points on the road surface",
          round(tot["tall_on_surface"] / max(tot["tall"], 1), 3), flush=True)
print("ZZGHOST-DONEZZ")
