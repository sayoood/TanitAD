"""Render the SAM3-only map video, v5 BEV: per-view METRIC ground layers and a non-causal ground-truth map at every frame.

Why (MEASURED 2026-09-13 on the day clip, token 41, tophat_probe2.py): the v4 renderer trimmed paint with a white
top-hat of a fixed 31 px in the IMAGE. Paint wider than 31 px in both image directions loses its interior: on the rear
tele camera 55-69 % of the SAM3 thin-paint pixels at 8-20 m are kept only by a 127 px top-hat, on the cross-right camera
33-39 % at 0-8 m -- hatched stripes and the yield triangle kept only their outlines. A pixel-sized element cannot be
right for 7 cameras with focal lengths from 930 to ~3500 px at every range; a metre-sized one can.
  layer      for every frame k and camera V, a 0.10 m grid in the rig frame of k (x -35..45, y -35..35, cells within
             R_MAX of the camera): each cell's ground point (smooth ground) projects through V's own camera model; the
             class is read from V's SAM3 raster (nearest), the luminance from the image MIP level whose pixel footprint
             matches the cell (bilinear) -- an antialiased inverse warp, no scatter, no holes
  paint      white top-hat of the layer luminance with a 1.5 m disc; a thin-class cell (line, arrow/text, hatched) stays
             paint only if brighter than max(10, 2.5 x MAD of the layer's road cells), else it becomes drivable
  GT map     all layers of the clip accumulate into ONE world map at 0.10 m: each cell keeps the class of the NEAREST
             labelled observation (camera range; a z-buffer, not a vote). Labels may use future frames; inference never
             sees this. The BEV at frame n is that map cut out in the rig frame of n.
Layer codes: 255 not seen (outside the image, behind the camera, beyond R_MAX, or the ego body), 0 seen but no map class,
1 drivable, 2 lane / road line, 3 crosswalk, 4 arrow / text, 5 non-drivable edge, 6 hatched area, 7 sidewalk / verge.
Outputs: f%04d.png per frame, worldmap.npz (cls, rng, origin, res), render_v5_stats.json.
"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
import sam3map_render_v4 as R4                                  # colours, names, fonts, overlay, BEV painter

V2 = Path("/home/nvidia/qwendrive/v2")
RES, LX, LY, R_MAX = 0.10, (-35.0, 45.0), (-35.0, 35.0), 35.0
TOPHAT_M, EDGE_MAX_M, MIP_LEVELS = 1.5, 12.0, 5
BX, BY = (-20.0, 40.0), (-15.0, 15.0)                            # the BEV panel (rig frame of the rendered frame)
THIN = (2, 3, 4, 6) if os.environ.get("CROSSWALK_STRIPES") == "1" else (2, 4, 6)   # stripes-only crosswalks: trimmed to bright paint like lines
COMPOSITE = os.environ.get("SAM3MAP_COMPOSITE", "nearest")      # nearest (v6.1) | surface_vote (v6.2, pre-registered below)
# surface_vote, PRE-REGISTERED 2026-09-13 before its first run (the v6.1 GT world map put 5.3 % of LiDAR tall obstacles on
# drivable: a far "road" label always beat a near "background" view of a pole or wall). The SURFACE of a cell (drivable
# incl. paint / sidewalk-verge / background) is the vote of every observation weighted 1 / (1 + (range / 4 m)^2)^2, so near
# views dominate but several agree; paint and edges stay the NEAREST labelled view (sharp lines) and paint is kept only
# where the surface is drivable. WORKS on the day clip if GT MAP_A2 <= 0.03 AND MAP_C >= 0.75 AND MAP_B >= 0.99 AND
# MAP_E >= 0.48, with its mirrored control below on B, C and E. Anything else is a FAIL, reported as such.
#   RESULT (MEASURED, day clip): FAIL -- A2 0.0317, B 0.9411, C 0.5877, E 0.4798 (v6.1: 0.0533 / 1.0 / 0.7727 / 0.4798).
#   Cause: "background" merges static non-road (poles, walls) with DYNAMIC occluders; cars seen near out-voted the road.
# surface_vote_occ, PRE-REGISTERED 2026-09-13 before its first run: the same vote, but every camera pixel covered by a
# tracked agent box of that frame (PhysicalAI obstacle.offline via gt.npz; label-time evidence, admissible for labels)
# is NOT SEEN instead of background, so only static things vote against the road. SAME bars as surface_vote.
#   RESULT (MEASURED, day clip): FAIL -- A2 0.0291, B 0.8271, C 0.6266, E 0.506. World-map crops at frame 33: black
#   background bands along road/sidewalk borders and along the ego path behind the clip start.
# surface_vote_occ2, PRE-REGISTERED 2026-09-13 before its first run: as surface_vote_occ, plus (a) pixels the refinement
# zeroed on purpose (raw SAM3 class != 0, refined == 0: ego patch, thin edges) are NOT SEEN, not background, and (b) a
# SAM3 "no class" observation votes with weight 0.5 (it says "none of my prompts", a positive label says what it is).
# SAME bars as surface_vote. Raw rasters from SAM3MAP_RAW_DIR.
#   RESULT (MEASURED, day clip): FAIL -- A2 0.0297, B 0.8866, C 0.6364, E 0.506 (mirror 0.4959 / 0.7283 / 0.5487 / 0.2634).
#   THREE pre-registered vote arms fail B and C: a vote removes poles and walls (A2 0.053 -> 0.03) but loses road wherever
#   the near views are blocked (the ego's own lane between a lead car and a tailgating van). The nearest-labelled rule
#   (v6.1) stays the delivered ground truth; its known cost is A2 0.053. Next lever (not run): LiDAR tall obstacles written
#   into the label -- admissible for labels, but MAP_A2 reads the same LiDAR, so it would need an independent check.
# front_priority, PRE-REGISTERED 2026-09-13 before its first run (front camera only vs all 7, day clip, same pipeline: front-only GT
# map A2 0.0191 / E 0.6512 but B 0.5092 from coverage; all 7 cameras A2 0.0533 / E 0.4798, B 1.0): a cell the front camera
# ever LABELLED takes the front camera's nearest label; a cell it saw only as background stays background; a cell it never
# saw takes the all-camera nearest label. WORKS on the day clip if the GT map has MAP_A2 <= 0.030 AND MAP_E >= 0.60 AND
# MAP_B >= 0.99 AND MAP_C >= 0.70, with the mirrored control below on B, C, E. Night reported beside.
# majority_v65 (2026-09-13, PI questions on the night exit at t = 17.2 s; exploratory, NOT pre-registered): MEASURED that 48 %
# (front only) / 63 % (all 7) of BEV cells receive two or more classes over the clip and the nearest view picks a minority
# label in 6-8 % of cells (speckle); the left island curb lacks edges where SAM3 found no grass / curb mask within 3 px of
# the road boundary (gap 23-50 px in tokens 82-91). Rule: SURFACE = weighted majority of LABELLED views (drivable incl. paint
# vs sidewalk-verge; background never votes, so vehicles do not erase road); paint k on a drivable cell if its weight is
# >= 0.3 of the drivable weight (0.15 for crosswalk stripes); EDGES = sidewalk-verge cells (opened 5x5) touching drivable (opened
# 5x5) in the consolidated map, components >= 10 cells, plus cells labelled edge in >= 25 % of their near (<= 12 m) observation weight.
# v5c options (2026-09-13, PI "continue and improve"; exploratory, reported against v5b on the same data):
#   AGENT_DRIVABLE=1  the ego's swept footprint and every tracked VEHICLE footprint of every frame (obstacle.offline via gt.npz)
#                     are drivable surface: sidewalk / background / never-seen / edge cells under them become drivable. Label-time
#                     evidence (admissible for labels); MAP_B and MAP_C then read the same evidence and are no longer independent.
#   SURFACE_MODE=k    k x k majority filter on the surface (background / drivable / sidewalk) before paint and edges.
#   PAINT_SHARE=x     share of the drivable vote a line / arrow / hatched class needs (v5b: 0.3); crosswalk stripes keep 0.15.
# v5d options (2026-09-13, MEASURED night_paint_probe.py: on the night clip the front camera's SAM3 line cells have a 1.5 m
# top-hat median of 2-4 grey levels and the fixed floor of 10 keeps 1-4 % of them -- the night map had 66 line cells from the
# front camera; road cells' top-hat 90th percentile is 2-3, road MAD 0.2-1.0, so the floor, not the noise, decided):
#   PAINT_RULE=p95   a thin-class cell is paint when its top-hat exceeds max(3, 95th percentile of the layer's ROAD cells' top-hat):
#                    5 % false paint on asphalt by construction, adapted per camera and frame (v5c: max(10, 2.5 x MAD)).
#   NO_FRAMES=1      write worldmap.npz and stats only (the comparison videos draw their own frames).
PAINT_RULE = os.environ.get("PAINT_RULE", "mad10")
NO_FRAMES = os.environ.get("NO_FRAMES") == "1"
AGENT_DRIVABLE = os.environ.get("AGENT_DRIVABLE") == "1"
SURFACE_MODE = int(os.environ.get("SURFACE_MODE", "0"))
PAINT_SHARE_ENV = float(os.environ.get("PAINT_SHARE", "0.3"))
FRONT_SET = ("CAM_FW", "CAM_F0")
BOX_ROOT = Path(os.environ.get("SAM3MAP_BOXES_ROOT", "/home/nvidia/qwendrive/v2"))
RAW_DIR = Path(os.environ["SAM3MAP_RAW_DIR"]) if os.environ.get("SAM3MAP_RAW_DIR") else None
BG_W = 0.5 if os.environ.get("SAM3MAP_COMPOSITE", "").endswith("_occ2") else 1.0
NX, NY = int(round((LX[1] - LX[0]) / RES)), int(round((LY[1] - LY[0]) / RES))
GXL = LX[0] + (np.arange(NX) + 0.5) * RES; GYL = LY[0] + (np.arange(NY) + 0.5) * RES


def bilinear(img, u, v):
    h, w = img.shape
    u = np.clip(u, 0, w - 1.001); v = np.clip(v, 0, h - 1.001)
    u0 = u.astype(np.int32); v0 = v.astype(np.int32); du = u - u0; dv = v - v0
    return ((img[v0, u0] * (1 - du) + img[v0, u0 + 1] * du) * (1 - dv) + (img[v0 + 1, u0] * (1 - du) + img[v0 + 1, u0 + 1] * du) * dv)


def agent_occluders(C, boxes, shape=(540, 960)):
    """Pixels (960x540) covered by tracked agent boxes: convex hull of each box's 8 corners projected through the camera.
    The box z is taken as either its bottom or its centre (the hull spans both), +-0.2 m."""
    m = np.zeros(shape, np.uint8)
    for b in boxes:
        x, y, z, l, w, h, yaw = [float(v) for v in b[:7]]
        if np.hypot(x - C.t[0], y - C.t[1]) > R_MAX + 10.0:
            continue
        c_, s_ = np.cos(yaw), np.sin(yaw)
        dx = np.array([1, 1, -1, -1]) * l / 2; dy = np.array([1, -1, -1, 1]) * w / 2
        cx = x + dx * c_ - dy * s_; cy = y + dx * s_ + dy * c_
        Pb = np.c_[np.r_[cx, cx], np.r_[cy, cy], np.r_[np.full(4, z - h / 2 - 0.2), np.full(4, z + h + 0.2)]]
        Pc = (Pb - C.t) @ C.R
        ok = Pc[:, 2] > 0.1
        if C.ft is not None:
            ok &= np.arctan2(np.hypot(Pc[:, 0], Pc[:, 1]), Pc[:, 2]) <= C._th_tab[-1]
        if ok.sum() < 4:
            continue
        u, v, _ = C.project_cam(Pc[ok])
        pts = np.c_[u / 2, v / 2]
        if not np.isfinite(pts).all() or pts[:, 0].max() < 0 or pts[:, 0].min() > shape[1] or pts[:, 1].max() < 0 or pts[:, 1].min() > shape[0]:
            continue
        cv2.fillConvexPoly(m, cv2.convexHull(np.clip(pts, -5000, 5000).astype(np.int32)), 1)
    return m.astype(bool)


def layer_fields(C, sg, img, cls_small, ego_small, ev_small=None, occ_small=None):
    """Dense metric layer of one camera at one frame, in that frame's rig coordinates: code [NX, NY] (255 not seen, else
    the SAM3 class read through the camera model), SAM3 evidence bits, and the bright-paint flag (1.5 m white top-hat of
    the MIP luminance > max(10, 2.5 x MAD of the layer's road cells))."""
    N = NX * NY
    code = np.full(N, 255, np.uint8); ev = np.zeros(N, np.uint8); paint = np.zeros(N, bool)
    gx, gy = np.meshgrid(GXL, GYL, indexing="ij")
    near = np.hypot(gx - C.t[0], gy - C.t[1]) <= R_MAX
    idx = np.flatnonzero(near.ravel())
    xy = np.c_[gx.ravel()[idx], gy.ravel()[idx]]
    Pk = np.c_[xy, GS.height(xy, sg)]
    u, v, ok = C.project_rig(Pk)
    keep = np.flatnonzero(ok)
    idx, Pk, u, v = idx[keep], Pk[keep], u[keep], v[keep]
    if not len(idx):
        return code.reshape(NX, NY), ev.reshape(NX, NY), paint.reshape(NX, NY), {"thr": None}
    su = np.clip((u / 2).astype(np.int32), 0, cls_small.shape[1] - 1); sv = np.clip((v / 2).astype(np.int32), 0, cls_small.shape[0] - 1)
    c = cls_small[sv, su].copy()
    if ego_small is not None:
        c[ego_small[sv, su]] = 255
    if occ_small is not None:
        c[occ_small[sv, su]] = 255
    dist = np.hypot(Pk[:, 0] - C.t[0], Pk[:, 1] - C.t[1])
    c[(c == 5) & (dist > EDGE_MAX_M)] = 0
    code[idx] = c
    if ev_small is not None:
        ev[idx] = ev_small[sv, su]
    # MIP level from the pixel footprint of one cell (the larger of the two ground directions)
    ux, vx, _ = C.project_rig(Pk + [RES, 0.0, 0.0]); uy, vy, _ = C.project_rig(Pk + [0.0, RES, 0.0])
    fp = np.maximum(np.hypot(ux - u, vx - v), np.hypot(uy - u, vy - v))
    lvl = np.clip(np.floor(np.log2(np.maximum(fp, 1.0))), 0, MIP_LEVELS - 1).astype(np.int8)
    pyr = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)]
    for _ in range(MIP_LEVELS - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    lum = np.zeros(len(idx), np.float32)
    for l in range(MIP_LEVELS):
        m = lvl == l
        if m.any():
            s = 2.0 ** l
            lum[m] = bilinear(pyr[l], (u[m] + 0.5) / s - 0.5, (v[m] + 0.5) / s - 0.5)
    road = c == 1
    fill = float(np.median(lum[road])) if road.sum() >= 50 else float(np.median(lum))
    L = np.full(N, fill, np.float32); L[idx] = lum
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(round(TOPHAT_M / RES)) | 1,) * 2)
    th = cv2.morphologyEx(L.reshape(NX, NY), cv2.MORPH_TOPHAT, se).ravel()[idx]
    base = th[road] if road.sum() >= 50 else th
    if PAINT_RULE == "p95":
        thr = max(3.0, float(np.percentile(base, 95)))
    else:
        thr = max(10.0, 2.5 * float(np.median(np.abs(base - np.median(base)))))
    paint[idx] = (th > thr) & (c != 255)
    return code.reshape(NX, NY), ev.reshape(NX, NY), paint.reshape(NX, NY), {"thr": round(thr, 1)}


def build_layer(C, sg, img, cls_small, ego_small, occ_small=None):
    """(codes uint8 [NX, NY], stats): the layer with thin paint trimmed to the bright cells."""
    code, _, paint, st = layer_fields(C, sg, img, cls_small, ego_small, occ_small=occ_small)
    thin = np.isin(code, THIN)
    trim = thin & ~paint
    code = code.copy(); code[trim] = 1
    return code, {"thin_cells": int(thin.sum()), "trimmed": int(trim.sum()), "thr": st["thr"]}


def world_cells(T, cam_t, wx0, wy0, WW, WH):
    """World cells within R_MAX of a camera and inside its frame's layer: (world i, world j, layer i, layer j, range)."""
    cw = T[:2, :2] @ cam_t[:2] + T[:2, 3]
    i0 = max(0, int((cw[0] - R_MAX - wx0) / RES)); i1 = min(WW, int((cw[0] + R_MAX - wx0) / RES) + 1)
    j0 = max(0, int((cw[1] - R_MAX - wy0) / RES)); j1 = min(WH, int((cw[1] + R_MAX - wy0) / RES) + 1)
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    fi, fj = ii.ravel(), jj.ravel()
    q = (np.c_[wx0 + (fi + 0.5) * RES, wy0 + (fj + 0.5) * RES] - T[:2, 3]) @ T[:2, :2]      # world -> rig of that frame
    li = np.floor((q[:, 0] - LX[0]) / RES).astype(np.int64); lj = np.floor((q[:, 1] - LY[0]) / RES).astype(np.int64)
    inside = (li >= 0) & (li < NX) & (lj >= 0) & (lj < NY)
    q = q[inside]
    return fi[inside], fj[inside], li[inside], lj[inside], np.hypot(q[:, 0] - cam_t[0], q[:, 1] - cam_t[1]).astype(np.float32)


def main():
    c8 = sys.argv[1]; npz_dir = Path(sys.argv[2]); out_dir = Path(sys.argv[3]); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]
    sd = V2 / f"seq_{c8}"
    toks = [str(f["tok"]) for f in frames]
    meta0 = json.loads((sd / toks[0] / "meta.json").read_text(encoding="utf-8"))
    sha, tref0 = meta0["clip_sha12"], meta0["t_ref_us"]
    cams = [c for c in R4.VIEWS if f"cls_{c}" in frames[0]]
    FRONT_CAM = "CAM_FW" if "CAM_FW" in cams else "CAM_F0"
    SIDE = [c for c in (("CAM_CL", "CAM_CR", "CAM_RL", "CAM_RR") if "CAM_CL" in cams else ("CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2")) if c in cams]
    SIDE_NAME = {"CAM_L0": "front-left", "CAM_R0": "front-right", "CAM_L2": "rear-left", "CAM_R2": "rear-right",
                 "CAM_CL": "cross-left", "CAM_CR": "cross-right", "CAM_RL": "rear-left", "CAM_RR": "rear-right"}
    LEGEND = (1, 2, 3, 6, 4, 5, 7) if "pts_7" in frames[0] else (1, 2, 3, 6, 4, 5)
    Ts = [f["T_world_rig"] for f in frames]; Pp = np.array([T[:2, 3] for T in Ts])
    # ---- world map extent
    wx0, wy0 = Pp.min(axis=0) - (R_MAX + 12); wx1, wy1 = Pp.max(axis=0) + (R_MAX + 12)
    WW, WH = int(np.ceil((wx1 - wx0) / RES)), int(np.ceil((wy1 - wy0) / RES))
    wcls = np.full((WW, WH), 255, np.uint8); wrng = np.full((WW, WH), np.inf, np.float32)
    if COMPOSITE == "majority_v65":
        vk = np.zeros((8, WW, WH), np.float32); vedge = np.zeros((WW, WH), np.float32); vnear = np.zeros((WW, WH), np.float32)
    if COMPOSITE == "front_priority":
        wcls_f = np.full((WW, WH), 255, np.uint8); wrng_f = np.full((WW, WH), np.inf, np.float32); seen_f = np.zeros((WW, WH), bool)
    if COMPOSITE.startswith("surface_vote"):
        vdrv = np.zeros((WW, WH), np.float32); vwlk = np.zeros((WW, WH), np.float32); vbg = np.zeros((WW, WH), np.float32)
    stats = {"layers": 0, "thin_cells": 0, "trimmed": 0}
    for n, f in enumerate(frames):
        fd = sd / toks[n]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        lp = fd / "lidar.npy"
        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
        sg = GS.smooth_grid(grid, fb)
        T = Ts[n]
        boxes = np.load(BOX_ROOT / f"seq_{c8}" / toks[n] / "gt.npz")["boxes"] if (COMPOSITE.endswith("_occ") or COMPOSITE.endswith("_occ2")) else None
        raw = dict(np.load(RAW_DIR / files[n].name, allow_pickle=True)) if (RAW_DIR is not None and COMPOSITE.endswith("_occ2")) else None
        for cam in cams:
            img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
            C = CM.Camera.from_calib(c, fr["cam_order"].index(cam), img.shape[1], img.shape[0])
            occ = agent_occluders(C, boxes) if boxes is not None else None
            if raw is not None:
                zeroed = (raw[f"cls_{cam}"] != 0) & (f[f"cls_{cam}"] == 0)
                occ = zeroed if occ is None else (occ | zeroed)
            codes, st = build_layer(C, sg, img, f[f"cls_{cam}"], f.get(f"ego_{cam}"), occ)
            if occ is not None:
                stats["agent_occluded_px"] = stats.get("agent_occluded_px", 0) + int(occ.sum())
            stats["layers"] += 1; stats["thin_cells"] += st.get("thin_cells", 0); stats["trimmed"] += st.get("trimmed", 0)
            # accumulate into the world map by INVERSE lookup (world cells inside this layer's footprint -> layer cell)
            flat_i, flat_j, li, lj, dist = world_cells(T, C.t, wx0, wy0, WW, WH)
            code = codes[li, lj]
            cur_c = wcls[flat_i, flat_j]; cur_r = wrng[flat_i, flat_j]
            labelled = (code >= 1) & (code <= 7)
            take = labelled & (dist < cur_r)
            seen_bg = (code == 0) & (cur_c == 255)               # seen, no class: marks the cell observed, never overrides a label
            wcls[flat_i[take], flat_j[take]] = code[take]; wrng[flat_i[take], flat_j[take]] = dist[take]
            wcls[flat_i[seen_bg], flat_j[seen_bg]] = 0
            if COMPOSITE == "majority_v65":
                wv = (1.0 / (1.0 + (dist / 8.0) ** 2)).astype(np.float32)
                for k_ in (1, 2, 3, 4, 6, 7):
                    mk = code == k_
                    if mk.any():
                        vk[k_, flat_i[mk], flat_j[mk]] += wv[mk]
                nr = (code != 255) & (dist <= EDGE_MAX_M)
                vnear[flat_i[nr], flat_j[nr]] += wv[nr]
                me = code == 5
                vedge[flat_i[me], flat_j[me]] += wv[me]
            if COMPOSITE == "front_priority" and cam in FRONT_SET:
                cf_ = wcls_f[flat_i, flat_j]; rf_ = wrng_f[flat_i, flat_j]
                tk = labelled & (dist < rf_)
                wcls_f[flat_i[tk], flat_j[tk]] = code[tk]; wrng_f[flat_i[tk], flat_j[tk]] = dist[tk]
                seen_f[flat_i[code != 255], flat_j[code != 255]] = True
            if COMPOSITE.startswith("surface_vote"):
                sv = code != 255
                wv = (1.0 / (1.0 + (dist[sv] / 4.0) ** 2) ** 2).astype(np.float32); cs = code[sv]; ii_, jj_ = flat_i[sv], flat_j[sv]
                vdrv[ii_, jj_] += wv * np.isin(cs, (1, 2, 3, 4, 6)); vwlk[ii_, jj_] += wv * (cs == 7); vbg[ii_, jj_] += BG_W * wv * (cs == 0)
        if n % 10 == 0:
            print(f"  layers through frame {n}: {time.time() - t0:.0f}s", flush=True)
    if COMPOSITE == "majority_v65":
        seen = wcls != 255
        dw = vk[1] + vk[2] + vk[3] + vk[4] + vk[6]; sw = vk[7]
        out = np.where(seen, 0, 255).astype(np.uint8)
        drv = (dw > 0) & (dw >= sw); wlk = (sw > 0) & (sw > dw)
        if SURFACE_MODE >= 3:
            surf = np.where(drv, 1, np.where(wlk, 2, 0)).astype(np.uint8)
            sums = [cv2.boxFilter(((surf == k_) & seen).astype(np.float32), -1, (SURFACE_MODE, SURFACE_MODE), normalize=False) for k_ in (0, 1, 2)]
            surf2 = np.argmax(np.stack(sums), axis=0).astype(np.uint8)
            has = (dw + sw) > 0
            drv = seen & has & (surf2 == 1); wlk = seen & has & (surf2 == 2)
        out[drv] = 1; out[wlk] = 7
        best_p = np.argmax(np.stack([vk[2], vk[3], vk[4], vk[6]]), axis=0); best_w = np.max(np.stack([vk[2], vk[3], vk[4], vk[6]]), axis=0)
        paint_k = np.array([2, 3, 4, 6], np.uint8)[best_p]
        thr_p = np.where(paint_k == 3, 0.15, PAINT_SHARE_ENV)                # faint night zebra stripes: a lower share for crosswalk
        pm = drv & (best_w >= thr_p * dw)
        out[pm] = paint_k[pm]
        wlk_c = cv2.morphologyEx(wlk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0     # drop sidewalk fragments < 0.5 m
        drv_c = cv2.morphologyEx(drv.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        touch = cv2.dilate(drv_c.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        e_boundary = wlk_c & touch
        n_e, lab_e, st_e, _ = cv2.connectedComponentsWithStats(e_boundary.astype(np.uint8), connectivity=8)
        e_boundary = np.isin(lab_e, np.flatnonzero(st_e[:, cv2.CC_STAT_AREA] >= 10)[1:]) if n_e > 1 else e_boundary   # edges >= 1 m of cells
        e_obs = (vnear > 0) & (vedge >= 0.25 * vnear)
        out[e_boundary | e_obs] = 5
        stats["majority_v65"] = {"cells_drivable": int(drv.sum()), "cells_sidewalk": int(wlk.sum()), "cells_paint": int(pm.sum()),
                                 "cells_edge_boundary": int(e_boundary.sum()), "cells_edge_observed": int(e_obs.sum()), "cells_edge_total": int((out == 5).sum())}
        if AGENT_DRIVABLE:
            force = np.zeros((WW, WH), bool)
            ex, ey = np.meshgrid(np.arange(-1.0, 3.95, RES), np.arange(-1.05, 1.06, RES), indexing="ij")
            ego_pts = np.c_[ex.ravel(), ey.ravel()]
            n_veh = 0
            for n_ in range(len(frames)):
                Tn_ = Ts[n_]; fps = [ego_pts]
                gp = BOX_ROOT / f"seq_{c8}" / toks[n_] / "gt.npz"
                if gp.exists():
                    g_ = np.load(gp)
                    for b in g_["boxes"][g_["labels"] == 0]:
                        x_, y_, _, l_, w_, _, yaw_ = [float(v) for v in b[:7]]
                        ax, ay = np.meshgrid(np.arange(-l_ / 2, l_ / 2 + 1e-6, RES), np.arange(-w_ / 2, w_ / 2 + 1e-6, RES), indexing="ij")
                        cy_, sy_ = np.cos(yaw_), np.sin(yaw_)
                        fps.append(np.c_[x_ + ax.ravel() * cy_ - ay.ravel() * sy_, y_ + ax.ravel() * sy_ + ay.ravel() * cy_]); n_veh += 1
                allp = np.concatenate(fps) @ Tn_[:2, :2].T + Tn_[:2, 3]
                ci_ = np.floor((allp[:, 0] - wx0) / RES).astype(np.int64); cj_ = np.floor((allp[:, 1] - wy0) / RES).astype(np.int64)
                ok_ = (ci_ >= 0) & (ci_ < WW) & (cj_ >= 0) & (cj_ < WH)
                force[ci_[ok_], cj_[ok_]] = True
            conv = force & np.isin(out, (0, 5, 7, 255))
            out[conv] = 1
            stats["agent_drivable"] = {"cells_forced": int(conv.sum()), "vehicle_boxes_used": n_veh}
        wcls = out
    if COMPOSITE == "front_priority":
        lab_f = (wcls_f >= 1) & (wcls_f <= 7)
        stats["front_priority"] = {"cells_front_labelled": int(lab_f.sum()), "cells_front_seen_background_only": int((seen_f & ~lab_f).sum()),
                                   "cells_filled_by_other_cameras": int((~seen_f & (wcls >= 1) & (wcls <= 7)).sum())}
        wcls = np.where(seen_f, np.uint8(0), wcls).astype(np.uint8)
        wcls[lab_f] = wcls_f[lab_f]; wrng = np.where(lab_f, wrng_f, np.where(seen_f, np.inf, wrng)).astype(np.float32)
    if COMPOSITE.startswith("surface_vote"):
        seen_any = (vdrv + vwlk + vbg) > 0
        surf = np.argmax(np.stack([vbg, vdrv, vwlk]), axis=0)            # 0 background, 1 drivable, 2 sidewalk-verge
        nearest = wcls.copy()
        wcls = np.where(seen_any, 0, 255).astype(np.uint8)
        drv = seen_any & (surf == 1)
        wcls[drv] = np.where(np.isin(nearest[drv], (2, 3, 4, 6)), nearest[drv], 1)
        wcls[seen_any & (surf == 2)] = 7
        wcls[nearest == 5] = 5                                              # edges: the nearest labelled view, as v6.1
        stats["surface_vote"] = {"cells_drivable_v61": int(np.isin(nearest, (1, 2, 3, 4, 6)).sum()), "cells_drivable_v62": int(np.isin(wcls, (1, 2, 3, 4, 6)).sum()),
                                 "cells_sidewalk_v61": int((nearest == 7).sum()), "cells_sidewalk_v62": int((wcls == 7).sum())}
    np.savez_compressed(out_dir / "worldmap.npz", cls=wcls, rng=np.where(np.isfinite(wrng), wrng, 0).astype(np.float16),
                        origin=np.array([wx0, wy0]), res=RES, clip_sha12=sha, ver=str(frames[0].get("ver", "v?")), composite=COMPOSITE)
    print(f"world map {WW}x{WH} cells, {stats}  {time.time() - t0:.0f}s", flush=True)
    stats["paint_rule"] = PAINT_RULE; stats["crosswalk_stripes_trimmed"] = 3 in THIN
    if NO_FRAMES:
        stats["seconds"] = round(time.time() - t0, 1); stats["world_cells_labelled"] = int(((wcls >= 1) & (wcls <= 7)).sum())
        stats["world_cells_per_class"] = {R4.NAME[k]: int((wcls == k).sum()) for k in range(1, 8)}
        (out_dir / "render_v5_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        print("world map only (NO_FRAMES)", stats)
        return
    # ---- BEV grid of the rendered frame (rows forward-up, cols left-on-the-left), and the whole-clip map image
    xs = np.arange(BX[1] - RES / 2, BX[0], -RES); ys = np.arange(BY[1] - RES / 2, BY[0], -RES)
    GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]
    lab = np.where(wcls == 255, 0, wcls)
    near_r = np.where(np.isfinite(wrng), wrng, 99.0)
    small = cv2.resize(lab, (WH // 4, WW // 4), interpolation=cv2.INTER_NEAREST); small_r = cv2.resize(near_r, (WH // 4, WW // 4), interpolation=cv2.INTER_NEAREST)
    mimg_full = R4.paint_bev(small.T[::-1], small_r.T[::-1])       # world map, north (y) up, x to the right
    for n in range(len(frames)):
        Tn = Ts[n]; fd = sd / toks[n]
        w = XY @ Tn[:2, :2].T + Tn[:2, 3]
        ci = np.floor((w[:, 0] - wx0) / RES).astype(np.int64); cj = np.floor((w[:, 1] - wy0) / RES).astype(np.int64)
        ok = (ci >= 0) & (ci < WW) & (cj >= 0) & (cj < WH)
        bcls = np.zeros(len(XY), np.uint8); brng = np.full(len(XY), 99.0, np.float32)
        bcls[ok] = lab[ci[ok], cj[ok]]; brng[ok] = near_r[ci[ok], cj[ok]]
        bcls = bcls.reshape(GX.shape); brng = brng.reshape(GX.shape)
        meta = json.loads((fd / "meta.json").read_text(encoding="utf-8"))
        canvas = Image.new("RGB", (1920, 1080), (14, 17, 16)); d = ImageDraw.Draw(canvas)
        f = frames[n]
        f0img = np.asarray(Image.open(fd / "images" / f"{FRONT_CAM}.jpg").convert("RGB"))
        canvas.paste(R4.overlay(f0img, display_cls(f0img, f[f"cls_{FRONT_CAM}"], f.get(f"ego_{FRONT_CAM}"))).resize((1180, 664)), (10, 64))
        d.text((18, 70), "front", fill=(255, 255, 255), font=R4.font(18, True))
        for q_, cam in enumerate(SIDE):
            im = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
            canvas.paste(R4.overlay(im, display_cls(im, f[f"cls_{cam}"], f.get(f"ego_{cam}"))).resize((290, 163)), (10 + q_ * 297, 740))
            d.text((16 + q_ * 297, 744), SIDE_NAME[cam], fill=(255, 255, 255), font=R4.font(15, True))
        canvas.paste(Image.fromarray(R4.paint_bev(bcls, brng)), (1210, 64))
        ex, ey = 1210 + int(BY[1] / RES), 64 + int(BX[1] / RES)
        d.polygon([(ex, ey - 12), (ex - 8, ey + 8), (ex + 8, ey + 8)], fill=(255, 255, 255))
        d.text((1210, 668), "GT BEV · all cams & frames · 0.1 m", fill=(200, 205, 202), font=R4.font(15))
        mim = Image.fromarray(mimg_full); s = min(380 / mim.width, 600 / mim.height)
        mim = mim.resize((max(1, int(mim.width * s)), max(1, int(mim.height * s))), Image.LANCZOS)
        ox, oy = 1530 + (380 - mim.width) // 2, 64 + (600 - mim.height) // 2
        d.rectangle([1530, 64, 1910, 664], fill=(22, 26, 24)); canvas.paste(mim, (ox, oy))
        for m2 in range(len(frames)):
            px = ox + (Pp[m2, 0] - wx0) / (RES * 4) * s; py = oy + (mim.height / s - (Pp[m2, 1] - wy0) / (RES * 4)) * s
            rr = 2.5 if m2 == n else 1.2
            d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(255, 255, 255) if m2 <= n else (120, 124, 122))
        d.text((1530, 672), "whole-clip map · world frame", fill=(200, 205, 202), font=R4.font(15))
        yy = 700
        d.text((1210, yy), "in this BEV (m², seen within 10 m + farther)", fill=(240, 242, 241), font=R4.font(17, True)); yy += 26
        for k in LEGEND:
            a_s = float(((bcls == k) & (brng <= R4.NEAR_M)).sum()) * RES ** 2; a_t = float(((bcls == k) & (brng > R4.NEAR_M)).sum()) * RES ** 2
            d.rectangle([1210, yy + 2, 1226, yy + 18], fill=R4.COL[k]); d.text((1236, yy), R4.NAME[k], fill=(230, 232, 231), font=R4.font(16))
            d.text((1420, yy), f"{a_s:7.1f} + {a_t:6.1f}", fill=(230, 232, 231), font=R4.font(16)); yy += 23
        ver = str(frames[0].get("ver", "v?"))
        d.text((10, 12), f"SAM3-only road map {ver} · {len(cams)} native cameras · metric BEV  ·  clip {sha}  ·  t = {(meta['t_ref_us'] - tref0) / 1e6:5.1f} s",
               fill=(240, 242, 241), font=R4.font(26, True))
        d.text((10, 42), "paint = brighter than the asphalt within a 1.5 m ground disc · every BEV cell reads its pixel through its camera model from the nearest view",
               fill=(170, 178, 174), font=R4.font(14))
        for q_, k in enumerate(LEGEND):                                           # two rows of four: no label overlap
            xx, yy_ = 10 + (q_ % 4) * 297, 912 + (q_ // 4) * 26
            d.rectangle([xx, yy_, xx + 20, yy_ + 20], fill=R4.COL[k]); d.text((xx + 28, yy_), R4.NAME[k], fill=(230, 232, 231), font=R4.font(16))
        d.text((10, 966), "BEV: solid = nearest view within 10 m of its camera · faded = only seen from farther · labels may use future frames (ground truth, not inference).",
               fill=(150, 158, 154), font=R4.font(14))
        d.text((10, 988), "Geometry only (not semantics): calibration, ego poses, LiDAR ground height. Camera overlays: paint trimmed with a 127 px image top-hat (display).",
               fill=(150, 158, 154), font=R4.font(14))
        canvas.save(out_dir / f"f{n:04d}.png")
    stats["seconds"] = round(time.time() - t0, 1); stats["world_cells_labelled"] = int(((wcls >= 1) & (wcls <= 7)).sum())
    stats["world_cells_per_class"] = {R4.NAME[k]: int((wcls == k).sum()) for k in range(1, 8)}
    (out_dir / "render_v5_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("rendered", len(frames), stats)


TOPHAT_DISPLAY = cv2.getStructuringElement(cv2.MORPH_RECT, (127, 127))     # rect: 0.03 s vs 0.64 s (ellipse) per 1920x1080 on Thor


def display_cls(img, cls_small, ego_small):
    """Camera overlay classes: SAM3 raster at full size, thin paint trimmed with a 127 px top-hat, ego body blanked."""
    cls = cv2.resize(cls_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if ego_small is not None:
        cls[cv2.resize(ego_small.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] = 0
    Y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    th = cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, TOPHAT_DISPLAY)
    road = cls == 1
    mad = float(np.median(np.abs(th[road] - np.median(th[road])))) if road.any() else 3.0
    cls[np.isin(cls, THIN) & (th <= max(10.0, 2.5 * mad))] = 1
    return cls


if __name__ == "__main__":
    V2 = Path(os.environ.get("SAM3MAP_ROOT", str(V2)))
    main()
