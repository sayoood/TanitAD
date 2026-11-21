"""Objective checks for the PI's three questions (night clip, t = 17.2 s), on every world-map variant of one clip.
The world map is the ground-truth candidate (0.1 m, world frame); every check runs inside ROI = seen cells within 30 m of the
driven path, and again inside the PI's window (the BEV of one frame, a box in that frame's rig coordinates).
  NOISE      fragments = 8-connected components of ONE class smaller than 0.5 m^2 (surface classes 0 / 1 / 7) or 0.1 m^2
             (paint 2 / 3 / 4 / 6 and edge 5), per 1,000 m^2 of ROI; and the surface boundary length (m per 1,000 m^2): 4-neighbour
             cell pairs whose surface differs (background / drivable incl. paint / sidewalk-verge; edge cells skipped), x 0.1 m.
  CROSSWALK  coloured share of the crossing area = crosswalk cells / cells of their closing with a 1.5 m square. A filled
             zebra area reads ~1.0, stripes only ~0.4-0.6 (half the crossing is gaps).
  CURB       LiDAR reference, independent of SAM3: per frame, ground returns (mapq_core.split_points: agents removed) on a 0.3 m
             grid within 20 m of the rig; a cell is a height step when a 4-neighbour differs by 0.08-0.35 m. Accumulated in the
             world: a curb cell is a step in >= 2 frames and in >= 20 % of the frames that covered it; kept only in components
             >= 2 m across. RECALL = share of curb cells within 1.0 m of the map's drivable surface that have a map edge within
             0.6 m (and within 1.0 m). PRECISION = share of map edge cells inside LiDAR coverage within 0.6 m of a curb cell or
             a static obstacle (>= 2 frames). Label-time LiDAR is NOT used by any map variant scored here.
  PAINT      LiDAR reference, independent of the camera: high-intensity ground returns (mapq_ours.paint_points: within 25 m,
             intensity >= max(20, 4 x the sweep's median)) accumulated in the world; a LiDAR paint cell is hit in >= 2 frames.
             RECALL = share of LiDAR paint cells on the map's drivable surface with a map paint cell (line / crosswalk / arrow /
             hatched) within 0.3 m; PRECISION = share of map paint cells (inside LiDAR ground coverage, >= 3 frames) within 0.3 m of
             a LiDAR paint cell, per class. For crosswalk stripes, precision is the colour-between-stripes check.
Usage: pi_checks.py <c8> <npz dir for poses> <out json> <frame J> <x0,x1,y0,y1> <label=render dir> [...]
"""
import json, sys
from pathlib import Path
HERE = Path("/home/nvidia/sam3map/eval")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "stub"))
import numpy as np
import cv2

V2 = Path("/home/nvidia/qwendrive/v2"); D = Path("/home/nvidia/sam3map/data")
import mapq_ours as mo                                   # noqa: E402  (stubbed DracoPy; sweeps decoded on the dev box)
import mapq_core as mc                                   # noqa: E402
mo.V2, mo.PRED, mo.LIDAR = V2, D / "preds", D
mo.EGO_ZIP, mo.EXT = D / "egomotion.chunk_0768.zip", D / "sensor_extrinsics.chunk_0768.parquet"

c8, npz_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
J = int(sys.argv[4]); BOX = [float(v) for v in sys.argv[5].split(",")]
variants = [(a.split("=", 1)[0], Path(a.split("=", 1)[1])) for a in sys.argv[6:]]
variants = [(l, d) for l, d in variants if (d / "worldmap.npz").exists()]
files = sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))
frames = [np.load(f, allow_pickle=True) for f in files]
toks = [str(f["tok"]) for f in frames]; Ts = [f["T_world_rig"] for f in frames]
W0 = dict(np.load(variants[0][1] / "worldmap.npz", allow_pickle=True))
(wx0, wy0), RES = W0["origin"], float(W0["res"]); WW, WH = W0["cls"].shape
for l, d in variants:
    w_ = np.load(d / "worldmap.npz", allow_pickle=True)
    assert w_["cls"].shape == (WW, WH) and np.allclose(w_["origin"], (wx0, wy0)), f"{l}: world grid differs"

# ---------------- regions
path = np.zeros((WW, WH), np.uint8)
for T in Ts:
    i, j = int((T[0, 3] - wx0) / RES), int((T[1, 3] - wy0) / RES); path[i, j] = 1
dpath = cv2.distanceTransform((1 - path).astype(np.uint8), cv2.DIST_L2, 5) * RES
ii, jj = np.meshgrid(np.arange(WW), np.arange(WH), indexing="ij")
TJ = Ts[J]; q = (np.stack([wx0 + (ii + 0.5) * RES, wy0 + (jj + 0.5) * RES], -1) - TJ[:2, 3]) @ TJ[:2, :2]
win = (q[..., 0] >= BOX[0]) & (q[..., 0] <= BOX[1]) & (q[..., 1] >= BOX[2]) & (q[..., 1] <= BOX[3])
del q, ii, jj

# ---------------- LiDAR curb and static-obstacle evidence in the world frame
clip = mo.Clip(c8)


def sweep_cached(self, t):
    si = int(np.abs(self.mid - t).argmin())
    return np.load(D / "sweeps" / c8 / f"{si:04d}.npy"), 0.0


mo.Clip.sweep_ego = sweep_cached
G, R = 0.3, 20.0
NG = int(2 * R / G)
step_hits = np.zeros((WW, WH), np.int16); cov_hits = np.zeros((WW, WH), np.int16); stat_hits = np.zeros((WW, WH), np.int16)
paint_hits = np.zeros((WW, WH), np.int16); gcov_hits = np.zeros((WW, WH), np.int16)


def mark(acc, xy_rig, T, half):
    """Add one vote per world cell covered by the given rig-frame points (each a square of +-half cells)."""
    if not len(xy_rig):
        return
    w = xy_rig @ T[:2, :2].T + T[:2, 3]
    ci = np.floor((w[:, 0] - wx0) / RES).astype(np.int64); cj = np.floor((w[:, 1] - wy0) / RES).astype(np.int64)
    m = np.zeros((WW, WH), np.uint8)
    ok = (ci >= half) & (ci < WW - half) & (cj >= half) & (cj < WH - half)
    m[ci[ok], cj[ok]] = 1
    if half:
        m = cv2.dilate(m, np.ones((2 * half + 1, 2 * half + 1), np.uint8))
    acc += m


for n, tok in enumerate(toks):
    meta = json.loads((V2 / f"seq_{c8}" / tok / "meta.json").read_text(encoding="utf-8"))
    pts, _ = clip.sweep_ego(meta["t_ref_us"])
    g = np.load(V2 / f"seq_{c8}" / tok / "gt.npz")
    static, ground, _, _, _ = mc.split_points(pts, g["boxes"])
    gr = ground[(np.abs(ground[:, 0]) < R) & (np.abs(ground[:, 1]) < R)]
    gi = np.floor((gr[:, 0] + R) / G).astype(int); gj = np.floor((gr[:, 1] + R) / G).astype(int)
    okg = (gi >= 0) & (gi < NG) & (gj >= 0) & (gj < NG)
    tmp = np.full(NG * NG, np.inf); np.minimum.at(tmp, gi[okg] * NG + gj[okg], gr[okg, 2]); tmp[np.isinf(tmp)] = np.nan
    zg = tmp.reshape(NG, NG)
    step = np.zeros((NG, NG), bool)
    with np.errstate(invalid="ignore"):
        dx = np.abs(zg[1:, :] - zg[:-1, :]); hx = (dx >= 0.08) & (dx <= 0.35); step[1:, :] |= hx; step[:-1, :] |= hx
        dy = np.abs(zg[:, 1:] - zg[:, :-1]); hy = (dy >= 0.08) & (dy <= 0.35); step[:, 1:] |= hy; step[:, :-1] |= hy
    cx = -R + (np.arange(NG) + 0.5) * G
    si_, sj_ = np.nonzero(step); ci_, cj_ = np.nonzero(~np.isnan(zg))
    near_s = np.hypot(cx[si_], cx[sj_]) <= R; near_c = np.hypot(cx[ci_], cx[cj_]) <= R
    mark(step_hits, np.c_[cx[si_][near_s], cx[sj_][near_s]], Ts[n], 1)
    mark(cov_hits, np.c_[cx[ci_][near_c], cx[cj_][near_c]], Ts[n], 1)
    st = static[np.hypot(static[:, 0], static[:, 1]) <= R]
    mark(stat_hits, st[:, :2], Ts[n], 0)
    mark(paint_hits, mo.paint_points(ground), Ts[n], 0)
    g25 = ground[np.hypot(ground[:, 0], ground[:, 1]) <= 25.0]
    mark(gcov_hits, g25[:, :2], Ts[n], 1)
    if n % 20 == 0:
        print(f"  lidar frame {n}", flush=True)

curb = (step_hits >= 2) & (step_hits >= 0.2 * np.maximum(cov_hits, 1))
nc, lab, stt, _ = cv2.connectedComponentsWithStats(cv2.dilate(curb.astype(np.uint8), np.ones((3, 3), np.uint8)), connectivity=8)
big = np.flatnonzero(np.hypot(stt[:, cv2.CC_STAT_WIDTH], stt[:, cv2.CC_STAT_HEIGHT]) * RES >= 2.0)
curb_long = curb & np.isin(lab, big[big > 0])
static_w = stat_hits >= 2
covered = cov_hits >= 1
d_curb_or_static = cv2.distanceTransform((~(curb | static_w)).astype(np.uint8), cv2.DIST_L2, 5) * RES
lidar_paint = paint_hits >= 2; gcov_ok = gcov_hits >= 3
d_lpaint = cv2.distanceTransform((~lidar_paint).astype(np.uint8), cv2.DIST_L2, 5) * RES


def frag_count(cls, roi):
    tot = 0; per = {}
    for k in range(0, 8):
        m = ((cls == k) & roi).astype(np.uint8)
        if not m.any():
            per[k] = 0; continue
        n_, _, s_, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        lim = 50 if k in (0, 1, 7) else 10
        per[k] = int((s_[1:, cv2.CC_STAT_AREA] < lim).sum()); tot += per[k]
    return tot, per


def surface_boundary_m(cls, roi):
    s = np.full(cls.shape, -1, np.int8)
    s[cls == 0] = 0; s[np.isin(cls, (1, 2, 3, 4, 6))] = 1; s[cls == 7] = 2
    s[~roi] = -1
    a, b = s[1:, :], s[:-1, :]; c, e = s[:, 1:], s[:, :-1]
    n = int(((a >= 0) & (b >= 0) & (a != b)).sum() + ((c >= 0) & (e >= 0) & (c != e)).sum())
    return n * RES


rep = {"clip": c8, "frame_J": J, "window_box_rig": BOX, "lidar": {"curb_cells": int(curb.sum()), "curb_cells_long": int(curb_long.sum()),
       "static_cells": int(static_w.sum()), "covered_cells": int(covered.sum()), "lidar_paint_cells": int(lidar_paint.sum()),
       "lidar_ground_covered_cells": int(gcov_ok.sum())}, "variants": {}}
for label, d in variants:
    cls = np.load(d / "worldmap.npz", allow_pickle=True)["cls"]
    seen = cls != 255
    roi = seen & (dpath <= 30.0)
    r = {}
    for name, reg in (("roi", roi), ("pi_window", roi & win)):
        area = float(reg.sum()) * RES ** 2
        fr_, per = frag_count(cls, reg)
        c3 = ((cls == 3) & reg).astype(np.uint8)
        cl3 = cv2.morphologyEx(c3, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
        drv = np.isin(cls, (1, 2, 3, 4, 6))
        d_drv = cv2.distanceTransform((~drv).astype(np.uint8), cv2.DIST_L2, 5) * RES
        d_edge = cv2.distanceTransform((cls != 5).astype(np.uint8), cv2.DIST_L2, 5) * RES
        ref = curb_long & reg & (d_drv <= 1.0)
        edges_cov = (cls == 5) & reg & covered
        r[name] = {"area_m2": round(area, 1),
                   "fragments_per_1000m2": round(fr_ / max(area, 1e-6) * 1000, 2), "fragments_by_class": per,
                   "surface_boundary_m_per_1000m2": round(surface_boundary_m(cls, reg) / max(area, 1e-6) * 1000, 1),
                   "crosswalk_m2": round(float(c3.sum()) * RES ** 2, 1),
                   "crosswalk_coloured_share_of_crossing_area": round(float(c3.sum()) / max(int(cl3.sum()), 1), 3) if c3.any() else None,
                   "curb_reference_cells": int(ref.sum()),
                   "curb_recall_edge_within_0.6m": round(float((d_edge[ref] <= 0.6).mean()), 3) if ref.any() else None,
                   "curb_recall_edge_within_1.0m": round(float((d_edge[ref] <= 1.0).mean()), 3) if ref.any() else None,
                   "edge_cells_in_lidar_coverage": int(edges_cov.sum()),
                   "edge_precision_within_0.6m": round(float((d_curb_or_static[edges_cov] <= 0.6).mean()), 3) if edges_cov.any() else None}
        pm = np.isin(cls, (2, 3, 4, 6))
        d_pm = cv2.distanceTransform((~pm).astype(np.uint8), cv2.DIST_L2, 5) * RES
        refp = lidar_paint & reg & np.isin(cls, (1, 2, 3, 4, 6))
        r[name]["lidar_paint_cells_on_drivable"] = int(refp.sum())
        r[name]["paint_recall_within_0.3m"] = round(float((d_pm[refp] <= 0.3).mean()), 3) if refp.any() else None
        for kname, kset in (("all_paint", (2, 3, 4, 6)), ("line", (2,)), ("crosswalk", (3,)), ("arrow_hatched", (4, 6))):
            cc = np.isin(cls, kset) & reg & gcov_ok
            r[name][f"{kname}_m2_in_lidar_coverage"] = round(float(cc.sum()) * RES ** 2, 1)
            r[name][f"{kname}_precision_within_0.3m"] = round(float((d_lpaint[cc] <= 0.3).mean()), 3) if cc.any() else None
    rep["variants"][label] = r
    print(f"{label:22s} ROI frag {r['roi']['fragments_per_1000m2']:7.2f} bnd {r['roi']['surface_boundary_m_per_1000m2']:7.1f} xw {r['roi']['crosswalk_coloured_share_of_crossing_area']} "
          f"curb R {r['roi']['curb_recall_edge_within_0.6m']} P {r['roi']['edge_precision_within_0.6m']} | WIN frag {r['pi_window']['fragments_per_1000m2']:7.2f} "
          f"xw {r['pi_window']['crosswalk_coloured_share_of_crossing_area']} curb R {r['pi_window']['curb_recall_edge_within_0.6m']} (n {r['pi_window']['curb_reference_cells']})", flush=True)
    q_ = r['roi']
    print(f"{'':22s} PAINT recall {q_['paint_recall_within_0.3m']} (n {q_['lidar_paint_cells_on_drivable']}) precision all {q_['all_paint_precision_within_0.3m']} "
          f"line {q_['line_precision_within_0.3m']} ({q_['line_m2_in_lidar_coverage']} m2) crosswalk {q_['crosswalk_precision_within_0.3m']} ({q_['crosswalk_m2_in_lidar_coverage']} m2)", flush=True)
out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
print("ZZPICHECKS-DONEZZ")
