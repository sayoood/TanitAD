"""A2 hits in RIG coordinates: are the tall points on the night map's drivable surface a fixed offset from the ego (ego
body / roof rig / a following or lead car without a box) or scattered static things? Prints the 1 m histogram peaks, the
per-frame hit counts beside the number of tracked boxes, and the share of hits whose world cell is hit in >= 5 frames.
Usage: a2_rigpos.py <c8> <npz dir> <render dir>"""
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
import sam3map_score as S
c8, npz_dir, rdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
clip = mo.Clip(c8)
mo.Clip.sweep_ego = lambda self, t: (np.load(D / "sweeps" / c8 / f"{int(np.abs(self.mid - t).argmin()):04d}.npy"), 0.0)
wm = dict(np.load(rdir / "worldmap.npz", allow_pickle=True)); (wx0, wy0), res = wm["origin"], float(wm["res"])
H = np.zeros((70, 60), np.int64)          # x -30..40, y -30..30 in 1 m bins
wcells = {}
for f in sorted(npz_dir.glob("[0-9][0-9][0-9].npz")):
    z = np.load(f, allow_pickle=True); tok = str(z["tok"]); T = z["T_world_rig"]
    meta = json.loads((V2 / f"seq_{c8}" / tok / "meta.json").read_text(encoding="utf-8"))
    pts, _ = clip.sweep_ego(meta["t_ref_us"])
    g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz")
    tall = mc.split_points(pts, g["boxes"])[4]
    gtm = S.worldmap_raster(wm, T)
    c_, k_ = mc.map_cls(gtm, tall[:, :2]); tk = tall[k_][c_ == 1]
    hi = np.floor(tk[:, 0] + 30).astype(int); hj = np.floor(tk[:, 1] + 30).astype(int)
    ok = (hi >= 0) & (hi < 70) & (hj >= 0) & (hj < 60); np.add.at(H, (hi[ok], hj[ok]), 1)
    w = tk[:, :2] @ T[:2, :2].T + T[:2, 3]
    for cell in set(zip(np.floor((w[:, 0] - wx0) / 0.5).astype(int).tolist(), np.floor((w[:, 1] - wy0) / 0.5).astype(int).tolist())):
        wcells[cell] = wcells.get(cell, 0) + 1
    n = int(f.stem)
    if n % 8 == 0:
        print(f"frame {n}: hits {len(tk)}, boxes {len(g['boxes'])} (vehicles {int((g['labels'] == 0).sum())}), hit rig x range {np.percentile(tk[:, 0], [10, 50, 90]).round(1) if len(tk) else '-'} y {np.percentile(tk[:, 1], [10, 50, 90]).round(1) if len(tk) else '-'}", flush=True)
flat = np.argsort(H.ravel())[::-1][:12]
print("top rig bins (x, y, hits):", [(int(i // 60) - 30, int(i % 60) - 30, int(H.ravel()[i])) for i in flat])
v = np.array(list(wcells.values()))
print("world 0.5 m cells with hits:", len(v), " share hit in >= 5 frames:", round(float((v >= 5).mean()), 3), " in 1 frame:", round(float((v == 1).mean()), 3))
print("ZZRIGPOS-DONEZZ")
