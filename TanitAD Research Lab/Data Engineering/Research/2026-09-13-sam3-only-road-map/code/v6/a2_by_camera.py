"""Which camera puts LiDAR tall obstacles on 'drivable'? Per camera and range band: the share of that frame's tall static
obstacle points (mapq_core.split_points: 1.2-3.0 m above local ground, outside agent boxes, ego footprint excluded) whose
ground cell (0.15 m) the camera's lifted drivable points (classes 1,2,3,4,6) cover, with the camera range of those
points. Same-frame evidence only (no vote), so it attributes the bleed to a view."""
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
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import sam3_paint as P  # noqa: E402
import camera_model as CM  # noqa: E402
import ground_surface as GS  # noqa: E402

c8, npz_dir, root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
clip = mo.Clip(c8)
C8 = c8


def sweep_cached(self, t):
    si = int(np.abs(self.mid - t).argmin())
    return np.load(D / "sweeps" / C8 / f"{si:04d}.npy"), float((self.mid[si] - t) / 1e3)


mo.Clip.sweep_ego = sweep_cached
BANDS = ((0, 8), (8, 15), (15, 30))
acc = {}
for f in sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))[::2]:
    z = np.load(f, allow_pickle=True); tok = str(z["tok"])
    meta = json.loads((mo.V2 / f"seq_{c8}" / tok / "meta.json").read_text())
    pts, _ = clip.sweep_ego(meta["t_ref_us"])
    g = np.load(mo.V2 / f"seq_{c8}" / tok / "gt.npz")
    static, ground, agent, above, tall = mc.split_points(pts, g["boxes"])
    if not len(tall):
        continue
    fd = root / f"seq_{c8}" / tok
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    ti = np.floor((tall[:, 0] + 30) / 0.15).astype(int); tj = np.floor((tall[:, 1] + 15) / 0.15).astype(int)
    tin = (ti >= 0) & (ti < 400) & (tj >= 0) & (tj < 200)
    for cam in [k[4:] for k in z.files if k.startswith("cls_")]:
        C = CM.Camera.from_calib(c, fr["cam_order"].index(cam))
        full = np.repeat(np.repeat(z[f"cls_{cam}"], 2, axis=0), 2, axis=1)
        xy, rng = C.lift(np.isin(full, (1, 2, 3, 4, 6)), sg, stride=2)
        seen_xy, _ = C.lift(full >= 0, sg, stride=4)                          # every ground pixel: what this camera can see
        for a, b in BANDS:
            m = (rng >= a) & (rng < b)
            cov = np.zeros((400, 200), bool); vis = np.zeros((400, 200), bool)
            i = np.floor((xy[m, 0] + 30) / 0.15).astype(int); j = np.floor((xy[m, 1] + 15) / 0.15).astype(int)
            ok = (i >= 0) & (i < 400) & (j >= 0) & (j < 200); cov[i[ok], j[ok]] = True
            k_ = acc.setdefault(f"{cam} {a}-{b}m", [0, 0])
            k_[0] += int(cov[ti[tin], tj[tin]].sum()); k_[1] += int(tin.sum())
print(json.dumps({k: {"tall_on_drivable": v[0], "tall_total": v[1], "share": round(v[0] / max(v[1], 1), 4)} for k, v in sorted(acc.items())}, indent=0))
