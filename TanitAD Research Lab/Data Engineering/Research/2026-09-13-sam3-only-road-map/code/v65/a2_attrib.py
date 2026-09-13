"""Where do the ground-truth world map's MAP_A2 failures lie? (tall static LiDAR obstacles, 1.2-3 m above the local ground,
agents and the ego removed -- mapq_core.split_points -- that fall on a DRIVABLE cell of the map raster at their frame).
For every such point: distance to the nearest NON-drivable cell of the same raster (0.15 m; boundary spill vs interior),
height above ground, range from the rig, and whether it lies in a tracked agent box grown by 1.5 m (an unlabelled part of
an agent). Hits are also accumulated in the world frame and drawn over the world map (magenta) for a visual check.
ROBUST=1: the A2r definition of a2_robust.py (ground = 10th percentile of the 2 m cell, agent boxes grown 1.0 m).
Usage: a2_attrib.py <c8> <npz dir> <render dir with worldmap.npz> <out prefix>"""
import json, os, sys
from pathlib import Path
HERE = Path("/home/nvidia/sam3map/eval")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "stub"))
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi

V2 = Path("/home/nvidia/qwendrive/v2"); D = Path("/home/nvidia/sam3map/data")
import mapq_ours as mo                                   # noqa: E402
import mapq_core as mc                                   # noqa: E402
mo.V2, mo.PRED, mo.LIDAR = V2, D / "preds", D
mo.EGO_ZIP, mo.EXT = D / "egomotion.chunk_0768.zip", D / "sensor_extrinsics.chunk_0768.parquet"
import sam3map_score as S                                # noqa: E402
import sam3map_render_v4 as R4                           # noqa: E402

c8, npz_dir, rdir, outp = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
clip = mo.Clip(c8)


def sweep_cached(self, t):
    si = int(np.abs(self.mid - t).argmin())
    return np.load(D / "sweeps" / c8 / f"{si:04d}.npy"), 0.0


mo.Clip.sweep_ego = sweep_cached
wm = dict(np.load(rdir / "worldmap.npz", allow_pickle=True))
cls = wm["cls"]; (wx0, wy0), res = wm["origin"], float(wm["res"])
hits_w = np.zeros(cls.shape, np.int32)
rows = []
for f in sorted(npz_dir.glob("[0-9][0-9][0-9].npz")):
    z = np.load(f, allow_pickle=True); tok = str(z["tok"]); T = z["T_world_rig"]
    meta = json.loads((V2 / f"seq_{c8}" / tok / "meta.json").read_text(encoding="utf-8"))
    pts, _ = clip.sweep_ego(meta["t_ref_us"])
    g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz"); boxes = g["boxes"]
    static, ground, agent, above, tall = mc.split_points(pts, boxes)
    if os.environ.get("ROBUST") == "1":
        p_ = pts[np.hypot(pts[:, 0], pts[:, 1]) <= mc.R_MAX + 2.0]
        k2 = (np.floor((p_[:, 0] + 64) / 2).astype(int) * 64 + np.floor((p_[:, 1] + 64) / 2).astype(int))
        k6 = (np.floor((p_[:, 0] + 66) / 6).astype(int) * 23 + np.floor((p_[:, 1] + 66) / 6).astype(int))
        def pct(keys, nb):
            o = np.full(nb, np.nan); cnt = np.bincount(keys, minlength=nb); order = np.argsort(keys, kind="stable"); ks, zs = keys[order], p_[order, 2]
            st = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]; en = np.r_[st[1:], len(ks)]
            for a_, b_ in zip(st, en):
                o[ks[a_]] = np.percentile(zs[a_:b_], 10)
            return o, cnt
        p10, cnt = pct(k2, 64 * 64 + 64); p106, _ = pct(k6, 23 * 23 + 23)
        gr_ = np.where(cnt[k2] >= 20, p10[k2], p106[k6])
        inb = np.zeros(len(p_), bool)
        for b in boxes:
            cc, ss = np.cos(-b[6]), np.sin(-b[6]); dx, dy = p_[:, 0] - b[0], p_[:, 1] - b[1]
            inb |= (np.abs(dx * cc - dy * ss) <= b[3] / 2 + 1.0) & (np.abs(dx * ss + dy * cc) <= b[4] / 2 + 1.0)
        ego_ = (p_[:, 0] > -2.0) & (p_[:, 0] < 5.0) & (np.abs(p_[:, 1]) < 1.5)
        tall = p_[(~inb) & (~ego_) & (p_[:, 2] > gr_ + 1.2) & (p_[:, 2] < gr_ + 3.0)]
    gtm = S.worldmap_raster(wm, T)
    c_, k_ = mc.map_cls(gtm, tall[:, :2])
    tk = tall[k_]
    on = c_ == 1
    if not on.any():
        continue
    nd = ~np.isin(gtm, (1, 2, 4))
    dist = ndi.distance_transform_edt(~nd) * 0.15                     # metres to the nearest non-drivable raster cell
    i = np.floor((tk[on, 1] + 15) / 0.15).astype(int); j = np.floor((tk[on, 0] + 30) / 0.15).astype(int)
    d_nd = dist[i, j]
    # local ground as split_points: 2 m cell minimum
    p = pts[np.hypot(pts[:, 0], pts[:, 1]) <= 32]
    key = (np.floor((p[:, 0] + 64) / 2).astype(int) * 64 + np.floor((p[:, 1] + 64) / 2).astype(int))
    zmin = np.full(64 * 64 + 64, np.inf); np.minimum.at(zmin, key, p[:, 2])
    kk = (np.floor((tk[on, 0] + 64) / 2).astype(int) * 64 + np.floor((tk[on, 1] + 64) / 2).astype(int))
    hgt = tk[on, 2] - zmin[kk]
    near_agent = np.zeros(on.sum(), bool)
    for b in boxes:
        cc, ss = np.cos(-b[6]), np.sin(-b[6]); dx, dy = tk[on, 0] - b[0], tk[on, 1] - b[1]
        lx, ly = dx * cc - dy * ss, dx * ss + dy * cc
        near_agent |= (np.abs(lx) <= b[3] / 2 + 1.5) & (np.abs(ly) <= b[4] / 2 + 1.5)
    rng = np.hypot(tk[on, 0], tk[on, 1])
    for a, b_, h_, r_, na in zip(d_nd, rng, hgt, rng, near_agent):
        rows.append((float(a), float(r_), float(h_), bool(na)))
    w = tk[on, :2] @ T[:2, :2].T + T[:2, 3]
    wi = np.floor((w[:, 0] - wx0) / res).astype(int); wj = np.floor((w[:, 1] - wy0) / res).astype(int)
    okw = (wi >= 0) & (wi < cls.shape[0]) & (wj >= 0) & (wj < cls.shape[1])
    np.add.at(hits_w, (wi[okw], wj[okw]), 1)
a = np.array(rows) if rows else np.zeros((0, 4))
rep = {"hits": len(a)}
if len(a):
    d_, r_, h_, na = a[:, 0], a[:, 1], a[:, 2], a[:, 3] > 0.5
    rep["distance_to_non_drivable_m"] = {"<=0.3": round(float((d_ <= 0.3).mean()), 3), "0.3-1": round(float(((d_ > 0.3) & (d_ <= 1)).mean()), 3),
                                         "1-3": round(float(((d_ > 1) & (d_ <= 3)).mean()), 3), ">3": round(float((d_ > 3).mean()), 3)}
    rep["range_m"] = {"<10": round(float((r_ < 10).mean()), 3), "10-20": round(float(((r_ >= 10) & (r_ < 20)).mean()), 3), ">=20": round(float((r_ >= 20).mean()), 3)}
    rep["height_m"] = {"1.2-1.8": round(float((h_ < 1.8).mean()), 3), "1.8-2.4": round(float(((h_ >= 1.8) & (h_ < 2.4)).mean()), 3), ">=2.4": round(float((h_ >= 2.4).mean()), 3)}
    rep["within_1.5m_of_an_agent_box"] = round(float(na.mean()), 3)
    rep["interior_gt3m_and_not_near_agent"] = round(float(((d_ > 3) & ~na).mean()), 3)
# world picture: map colours + A2 hits in magenta (dilated), cropped to the seen area
img = np.zeros(cls.shape + (3,), np.uint8); img[:] = (14, 16, 15); img[cls == 0] = (58, 62, 60)
for k in (7, 1, 2, 4, 6, 3, 5):
    img[cls == k] = R4.COL[k]
hm = cv2.dilate((hits_w > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
img[hm] = (255, 0, 255)
seen = np.argwhere(cls != 255); (i0, j0), (i1, j1) = seen.min(0), seen.max(0)
Image.fromarray(img[i0:i1 + 1, j0:j1 + 1].transpose(1, 0, 2)[::-1]).save(f"{outp}.png")
Path(f"{outp}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps(rep, indent=1)); print("ZZA2ATTRIB-DONEZZ")
