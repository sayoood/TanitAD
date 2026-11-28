"""Diagnostics for three PI questions on the night clip at t = 17.2 s (token index 86):
  (1) noise at the roundabout exit: per BEV cell of that frame's window, how many labelled observations of the whole clip
      exist, how many DIFFERENT classes they carry, and whether the nearest-view label equals the range-weighted majority
      label -- front-camera-only arm and all-camera arm; plus the two maps drawn side by side
  (2) the colour between crosswalk stripes: CAM_FW's SAM3 crosswalk instance mask (evidence bit 2), the crosswalk class
      after refine (v6), after consensus (v61), and the bright paint (127 px top-hat)
  (3) the left curb: along CAM_FW frames 76-92, the drivable mask, sidewalk/grass (bit 128) and curb (bit 64) instance
      masks and the final edges; the edge rule needs the drivable boundary within ~3 px (960x540 raster) of a non-drivable
      mask, so the measured gap per frame says why an edge did or did not form
Usage: exit_edge_xwalk_diag.py <c8> <token index> <out dir>"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
import sam3map_render_v4 as R4
import sam3map_render_v5 as R5

SM = Path("/home/nvidia/sam3map"); ROOT = SM / "native7"
c8, J, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
RES, BX, BY = 0.10, (-20.0, 40.0), (-15.0, 15.0)
xs = np.arange(BX[1] - RES / 2, BX[0], -RES); ys = np.arange(BY[1] - RES / 2, BY[0], -RES)
GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]
rep = {}

# ---------------- (1) label agreement per BEV cell
def agreement(npz_dir, tag):
    files = sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]
    Tn = frames[J]["T_world_rig"]; W = XY @ Tn[:2, :2].T + Tn[:2, 3]                    # world xy of the window cells
    cnt = np.zeros((len(XY), 8), np.float32); near_c = np.full(len(XY), 255, np.uint8); near_r = np.full(len(XY), np.inf, np.float32)
    nobs = np.zeros(len(XY), np.int32); seen = np.zeros(len(XY), bool)
    cams = [k[4:] for k in frames[0] if k.startswith("cls_")]
    for n, f in enumerate(frames):
        fd = ROOT / f"seq_{c8}" / str(f["tok"]); T = f["T_world_rig"]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
        q = (W - T[:2, 3]) @ T[:2, :2]
        if np.hypot(*(T[:2, 3] - Tn[:2, 3])) > 60:
            continue
        for cam in cams:
            C = CM.Camera.from_calib(c, fr["cam_order"].index(cam))
            img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
            codes, _ = R5.build_layer(C, sg, img, f[f"cls_{cam}"], f.get(f"ego_{cam}"))
            li = np.floor((q[:, 0] - R5.LX[0]) / RES).astype(np.int64); lj = np.floor((q[:, 1] - R5.LY[0]) / RES).astype(np.int64)
            ok = (li >= 0) & (li < R5.NX) & (lj >= 0) & (lj < R5.NY)
            code = np.full(len(q), 255, np.uint8); code[ok] = codes[li[ok], lj[ok]]
            dist = np.hypot(q[:, 0] - C.t[0], q[:, 1] - C.t[1]).astype(np.float32)
            lab = (code >= 1) & (code <= 7)
            seen |= code != 255
            w = (1.0 / (1.0 + (dist / 8.0) ** 2)).astype(np.float32)
            idx = np.flatnonzero(lab)
            np.add.at(cnt, (idx, code[idx].astype(np.int64)), w[idx]); nobs[idx] += 1
            tk = lab & (dist < near_r); near_c[tk] = code[tk]; near_r[tk] = dist[tk]
    maj = np.where(cnt[:, 1:].sum(axis=1) > 0, np.argmax(cnt[:, 1:], axis=1) + 1, 255).astype(np.uint8)
    distinct = (cnt[:, 1:] > 0).sum(axis=1)
    labelled = nobs > 0
    exitreg = labelled & (XY[:, 0] >= 5) & (XY[:, 0] <= 30) & (XY[:, 1] <= 0)           # ahead-right: the exit in that frame
    rep[tag] = {"cells_labelled": int(labelled.sum()), "median_labelled_observations_per_cell": float(np.median(nobs[labelled])),
                "share_cells_with_2plus_classes": round(float((distinct[labelled] >= 2).mean()), 3),
                "share_cells_nearest_ne_weighted_majority": round(float((near_c[labelled] != maj[labelled]).mean()), 3),
                "exit_region_share_cells_with_2plus_classes": round(float((distinct[exitreg] >= 2).mean()), 3),
                "exit_region_share_nearest_ne_majority": round(float((near_c[exitreg] != maj[exitreg]).mean()), 3)}

    def draw(codes):
        img = np.zeros(GX.shape + (3,), np.uint8); img[:] = (14, 16, 15)
        cc = codes.reshape(GX.shape)
        img[(cc == 0) | ((cc == 255) & seen.reshape(GX.shape))] = (58, 62, 60)
        for k in (7, 1, 2, 4, 6, 3):
            img[cc == k] = R4.COL[k]
        img[cv2.dilate((cc == 5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = R4.COL[5]
        return Image.fromarray(img)
    return draw(near_c), draw(maj)


sheet1 = Image.new("RGB", (1300, 690), (10, 12, 11)); d1 = ImageDraw.Draw(sheet1)
for q_, (tag, nd) in enumerate((("front_only", SM / f"{c8}_v61f"), ("all_7", SM / f"{c8}_v61"))):
    a, b = agreement(nd, tag)
    sheet1.paste(a, (10 + q_ * 640, 60)); sheet1.paste(b, (330 + q_ * 640, 60))
    d1.text((10 + q_ * 640, 10), f"{tag}: NEAREST view (current rule)", fill=(240, 240, 240), font=R4.font(15, True))
    d1.text((330 + q_ * 640, 30), "weighted MAJORITY of labelled views", fill=(240, 240, 240), font=R4.font(15, True))
sheet1.save(out / "noise_nearest_vs_majority.png")

# ---------------- (2) crosswalk region vs paint, CAM_FW at token J
f6 = np.load(SM / f"{c8}_v6" / f"{J:03d}.npz", allow_pickle=True); f61 = np.load(SM / f"{c8}_v61" / f"{J:03d}.npz", allow_pickle=True)
raw = np.load(SM / f"{c8}_v6raw" / f"{J:03d}.npz", allow_pickle=True)
fd = ROOT / f"seq_{c8}" / str(f6["tok"])
img = np.asarray(Image.open(fd / "images" / "CAM_FW.jpg").convert("RGB"))
small = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
Y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
th = cv2.resize(cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (127, 127))), (960, 540), interpolation=cv2.INTER_AREA)
road = f6["cls_CAM_FW"] == 1
thr = max(10.0, 2.5 * float(np.median(np.abs(th[road] - np.median(th[road]))))) if road.any() else 10.0
paint = th > thr
xw_ev = (raw["evid_CAM_FW"] & 2) > 0; c3_v6 = f6["cls_CAM_FW"] == 3; c3_v61 = f61["cls_CAM_FW"] == 3


def tint(mask, col):
    a = small.astype(np.float32).copy(); a[mask] = 0.4 * a[mask] + 0.6 * np.array(col); return Image.fromarray(a.astype(np.uint8))


panels = [("SAM3 crosswalk instance masks (on-road)", tint(xw_ev, (0, 220, 210))), ("crosswalk class after refine (v6)", tint(c3_v6, (0, 220, 210))),
          ("crosswalk class after consensus (v6.1)", tint(c3_v61, (0, 220, 210))), ("bright paint (top-hat) inside v6.1 crosswalk", tint(c3_v61 & paint, (255, 255, 0)))]
sheet2 = Image.new("RGB", (1920, 1110), (10, 12, 11)); d2 = ImageDraw.Draw(sheet2)
for q_, (title, im) in enumerate(panels):
    x0, y0 = (q_ % 2) * 960, (q_ // 2) * 555
    sheet2.paste(im, (x0, y0 + 15)); d2.text((x0 + 6, y0), title, fill=(255, 255, 255), font=R4.font(14, True))
sheet2.save(out / "crosswalk_region_vs_paint.png")
rep["crosswalk_token"] = {"px_crosswalk_evidence": int(xw_ev.sum()), "px_class3_v6": int(c3_v6.sum()), "px_class3_v61": int(c3_v61.sum()),
                          "share_of_v61_class3_that_is_bright_paint": round(float((c3_v61 & paint).sum() / max(c3_v61.sum(), 1)), 3),
                          "share_of_v61_class3_outside_sam3_crosswalk_masks": round(float((c3_v61 & ~xw_ev).sum() / max(c3_v61.sum(), 1)), 3)}

# ---------------- (3) the left curb, CAM_FW frames J-10 .. J+6
rows = []
sheet3 = Image.new("RGB", (1920, 3 * 280), (10, 12, 11)); d3 = ImageDraw.Draw(sheet3)
for q_, jj in enumerate(range(J - 10, J + 7, 3)):
    fr6 = np.load(SM / f"{c8}_v6" / f"{jj:03d}.npz", allow_pickle=True); frr = np.load(SM / f"{c8}_v6raw" / f"{jj:03d}.npz", allow_pickle=True)
    fdj = ROOT / f"seq_{c8}" / str(fr6["tok"])
    im = cv2.resize(np.asarray(Image.open(fdj / "images" / "CAM_FW.jpg").convert("RGB")), (960, 540), interpolation=cv2.INTER_AREA)
    cl = fr6["cls_CAM_FW"]; ev = frr["evid_CAM_FW"]
    drive = np.isin(cl, (1, 2, 3, 4, 6)); walk = (ev & 128) > 0; curbm = (ev & 64) > 0; edge = cl == 5
    left = np.zeros_like(drive); left[270:, :480] = True                                  # lower-left quarter: the island curb in these frames
    bnd = drive & (cv2.dilate((~drive).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & left
    dist_nd = cv2.distanceTransform((~(walk | curbm)).astype(np.uint8), cv2.DIST_L2, 3)
    g = dist_nd[bnd]
    rows.append({"token_index": jj, "left_boundary_px": int(bnd.sum()), "median_gap_to_sidewalk_grass_curb_px": round(float(np.median(g)), 1) if len(g) else None,
                 "share_boundary_within_3px": round(float((g <= 3).mean()), 3) if len(g) else None,
                 "left_edge_px": int((edge & left).sum()), "left_curb_mask_px": int((curbm & left).sum()), "left_sidewalk_grass_px": int((walk & left).sum())})
    a = im.astype(np.float32)
    for m, col in ((drive, (70, 120, 200)), (walk, (110, 190, 90)), (curbm, (255, 60, 220)), (edge, (235, 40, 40))):
        a[m] = 0.45 * a[m] + 0.55 * np.array(col)
    pan = Image.fromarray(a.astype(np.uint8)).resize((640, 360)).crop((0, 100, 640, 360))
    x0, y0 = (q_ % 3) * 640, (q_ // 3) * 280
    sheet3.paste(pan, (x0, y0 + 20))
    d3.text((x0 + 6, y0 + 2), f"token {jj}: gap median {rows[-1]['median_gap_to_sidewalk_grass_curb_px']} px, within 3 px {rows[-1]['share_boundary_within_3px']}, edge px {rows[-1]['left_edge_px']}",
            fill=(255, 255, 255), font=R4.font(13, True))
sheet3.save(out / "left_curb_masks.png")
rep["left_curb"] = rows
(out / "diag.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps(rep, indent=1)); print("ZZDIAG-DONEZZ")
