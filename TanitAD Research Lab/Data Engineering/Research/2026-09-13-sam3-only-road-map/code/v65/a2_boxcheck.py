"""Frame-level check: the tall LiDAR clusters behind / ahead of the ego vs the tracked boxes of that frame and the
neighbouring label timestamps. Prints boxes within 15 m (x, y, z, l, w, h, yaw, label) and the tall-point clusters."""
import json, sys
from pathlib import Path
HERE = Path("/home/nvidia/sam3map/eval")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "stub"))
import numpy as np
from scipy import ndimage as ndi
V2 = Path("/home/nvidia/qwendrive/v2"); D = Path("/home/nvidia/sam3map/data")
import mapq_ours as mo
import mapq_core as mc
mo.V2, mo.PRED, mo.LIDAR = V2, D / "preds", D
mo.EGO_ZIP, mo.EXT = D / "egomotion.chunk_0768.zip", D / "sensor_extrinsics.chunk_0768.parquet"
c8, npz_dir = sys.argv[1], Path(sys.argv[2])
clip = mo.Clip(c8)
for n in [int(x) for x in sys.argv[3].split(",")]:
    z = np.load(npz_dir / f"{n:03d}.npz", allow_pickle=True); tok = str(z["tok"])
    meta = json.loads((V2 / f"seq_{c8}" / tok / "meta.json").read_text(encoding="utf-8"))
    si = int(np.abs(clip.mid - meta["t_ref_us"]).argmin())
    pts = np.load(D / "sweeps" / c8 / f"{si:04d}.npy")
    g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz")
    print(f"frame {n} tok {tok} sweep {si} dt_ms {(clip.mid[si] - meta['t_ref_us']) / 1e3:.1f}  gt keys {g.files}")
    b = g["boxes"]; lab = g["labels"]
    near = np.hypot(b[:, 0], b[:, 1]) <= 15
    for bb, ll in zip(b[near], lab[near]):
        print("   box", np.round(bb[:7], 2), "label", int(ll))
    tall = mc.split_points(pts, b)[4]
    occ = np.zeros((60, 60), bool)
    i = np.floor(tall[:, 0] + 30).astype(int); j = np.floor(tall[:, 1] + 30).astype(int); ok = (i >= 0) & (i < 60) & (j >= 0) & (j < 60)
    occ[i[ok], j[ok]] = True
    labm, nl = ndi.label(occ)
    for k in range(1, nl + 1):
        m = labm == k
        sel = ok.copy(); sel[ok] = m[i[ok], j[ok]]
        c = tall[sel]
        if len(c) < 30 or np.hypot(c[:, 0].mean(), c[:, 1].mean()) > 15:
            continue
        print(f"   tall cluster n {len(c)} x {c[:, 0].min():.1f}..{c[:, 0].max():.1f} y {c[:, 1].min():.1f}..{c[:, 1].max():.1f} z {c[:, 2].min():.2f}..{c[:, 2].max():.2f}")
