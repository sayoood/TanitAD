"""Where inside each native camera does the tall-obstacle bleed come from: pixels INSIDE the v2 virtual 63.7-deg views cut
from that camera (same centre; F0<-FW, L0/L1<-CL, R0/R1<-CR, L2<-RL, R2<-RR) or the native PERIPHERY outside them (and
the tele cameras FT / RT, which the v2 map never used)? Same-frame evidence as a2_by_camera.py: the share of the frame's
LiDAR tall-obstacle points whose 0.15 m cell the camera part's lifted drivable points cover, by camera range."""
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
clip = mo.Clip(c8); C8 = c8
USED_V2 = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2")


def sweep_cached(self, t):
    si = int(np.abs(self.mid - t).argmin())
    return np.load(D / "sweeps" / C8 / f"{si:04d}.npy"), float((self.mid[si] - t) / 1e3)


mo.Clip.sweep_ego = sweep_cached
BANDS = ((0, 8), (8, 15), (15, 30))
ii, jj = np.mgrid[0:540, 0:960]; u_n = (jj * 2 + 1.0).ravel(); v_n = (ii * 2 + 1.0).ravel()
acc, masks = {}, {}
for f in sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))[::2]:
    z = np.load(f, allow_pickle=True); tok = str(z["tok"])
    meta = json.loads((mo.V2 / f"seq_{c8}" / tok / "meta.json").read_text())
    pts, _ = clip.sweep_ego(meta["t_ref_us"])
    g = np.load(mo.V2 / f"seq_{c8}" / tok / "gt.npz")
    tall = mc.split_points(pts, g["boxes"])[4]
    if not len(tall):
        continue
    fd = root / f"seq_{c8}" / tok; vd = mo.V2 / f"seq_{c8}" / tok
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    cv = np.load(vd / "calib.npz"); fv = json.loads((vd / "frame.json").read_text())
    grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    ti = np.floor((tall[:, 0] + 30) / 0.15).astype(int); tj = np.floor((tall[:, 1] + 15) / 0.15).astype(int)
    tin = (ti >= 0) & (ti < 400) & (tj >= 0) & (tj < 200)
    for cam in [k[4:] for k in z.files if k.startswith("cls_")]:
        C = CM.Camera.from_calib(c, fr["cam_order"].index(cam))
        if cam not in masks:                                   # pixels inside any USED v2 virtual view of the same centre
            inside = np.zeros(len(u_n), bool)
            ray = C.rays_rig(u_n, v_n); good = np.isfinite(ray).all(axis=1)
            for vi, vname in enumerate(fv["cam_order"]):
                if vname in USED_V2 and np.allclose(cv["sensor2lidar_translation"][vi], C.t, atol=1e-4):
                    Cv = CM.Camera.from_calib(cv, vi)
                    ok = Cv.project_rig(C.t + np.where(good[:, None], ray, 0.0) * 10.0)[2] & good
                    inside |= ok
            masks[cam] = inside.reshape(540, 960)
        full_cls = np.repeat(np.repeat(z[f"cls_{cam}"], 2, axis=0), 2, axis=1)
        full_in = np.repeat(np.repeat(masks[cam], 2, axis=0), 2, axis=1)
        for part, pm in (("inside_v2_view", full_in), ("periphery", ~full_in)):
            xy, rng = C.lift(np.isin(full_cls, (1, 2, 3, 4, 6)) & pm, sg, stride=2)
            for a, b in BANDS:
                m = (rng >= a) & (rng < b)
                cov = np.zeros((400, 200), bool)
                i = np.floor((xy[m, 0] + 30) / 0.15).astype(int); j = np.floor((xy[m, 1] + 15) / 0.15).astype(int)
                ok = (i >= 0) & (i < 400) & (j >= 0) & (j < 200); cov[i[ok], j[ok]] = True
                k_ = acc.setdefault((cam, part, f"{a}-{b}m"), [0, 0])
                k_[0] += int(cov[ti[tin], tj[tin]].sum()); k_[1] += int(tin.sum())
rows = {}
for (cam, part, band), (h, n) in sorted(acc.items()):
    rows.setdefault(cam, {}).setdefault(part, {})[band] = round(h / max(n, 1), 4)
tot = {part: round(sum(h for (c_, p_, b_), (h, n) in acc.items() if p_ == part) / max(next(iter(acc.values()))[1], 1), 4) for part in ("inside_v2_view", "periphery")}
print(json.dumps({"share_of_tall_points_by_camera_part_band": rows, "sum_over_cameras_and_bands": tot,
                  "inside_share_of_camera_pixels": {k: round(float(v.mean()), 3) for k, v in masks.items()}}, indent=1))
