"""Frames for the PI's long quality video (2026-09-14: "Combined the proven measures and give me a rendered long video to confirm
quality"). Per front frame: TOP the map produced by the combined approved measures drawn into the front camera, BOTTOM the same frame
with the map of today's pipeline (the exact reference arm), middle both BEVs cut at the frame (x -20..40 m, y +-15 m, 0.1 m), right
the legend, the clip's measured bars (fast vs reference) and its ground source. Display rules as the approved reprojection videos v2:
paint where that arm's own SAM3 raster of this frame says road-level surface (dilated 4 working pixels), full ego mask, no boxes.
A title card (1 s) opens each clip. JPEG frames.
Usage: render_fast_long.py <out dir> <root> <cam> <c8> <fast tag> <ref tag> <fast label> <bars json> <clip i> <clip n> <ground label>"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
import gt_reproject as GR
import sam3map_render_v4 as R4

out, root, cam, c8, ftag, rtag, flabel, bars_path, ci, cn, ground = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6],
                                                                   sys.argv[7], Path(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10]), sys.argv[11])
SM = Path("/home/nvidia/sam3map")
out.mkdir(parents=True, exist_ok=True)
RES, BX, BY = 0.10, (-20.0, 40.0), (-15.0, 15.0)
xs = np.arange(BX[1] - RES / 2, BX[0], -RES); ys = np.arange(BY[1] - RES / 2, BY[0], -RES)
GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]
SEEN_BG, NEVER = np.array((58, 62, 60), np.uint8), np.array((14, 16, 15), np.uint8)
bars = json.loads(bars_path.read_text()) if bars_path.exists() else {}


def bev(wm, T):
    cls, (x0, y0), res = wm["cls"], wm["origin"], float(wm["res"])
    w = XY @ T[:2, :2].T + T[:2, 3]
    i = np.floor((w[:, 0] - x0) / res).astype(np.int64); j = np.floor((w[:, 1] - y0) / res).astype(np.int64)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    code = np.full(len(XY), 255, np.uint8); code[ok] = cls[i[ok], j[ok]]
    code = code.reshape(GX.shape)
    img = np.zeros(code.shape + (3,), np.uint8); img[:] = NEVER
    img[code == 0] = SEEN_BG
    for k in (7, 1, 2, 4, 6, 3):
        img[code == k] = R4.COL[k]
    img[cv2.dilate((code == 5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = R4.COL[5]
    return img


def sam3_ground(cls_small, hw):
    g = (cls_small >= 1) & (cls_small <= 7)
    g = cv2.resize(g.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).ravel() > 0


wf = dict(np.load(SM / f"render5_{c8}_{ftag}r" / "worldmap.npz", allow_pickle=True))
wr = dict(np.load(SM / f"render5_{c8}_{rtag}r" / "worldmap.npz", allow_pickle=True))
disp_f, disp_r = GR.display_world(wf["cls"]), GR.display_world(wr["cls"])
ff = sorted((SM / f"{c8}_{ftag}c").glob("[0-9][0-9][0-9].npz")); rf = sorted((SM / f"{c8}_{rtag}c").glob("[0-9][0-9][0-9].npz"))
assert [f.name for f in ff] == [f.name for f in rf] and len(ff) > 0
meta0 = json.loads((root / f"seq_{c8}" / str(np.load(rf[0], allow_pickle=True)["tok"]) / "meta.json").read_text())
sha = meta0.get("clip_sha12", "?")

# ---- title card, 5 frames (1 s at 5 fps)
card = Image.new("RGB", (1920, 1080), (14, 17, 16)); d = ImageDraw.Draw(card)
d.text((120, 330), f"clip {ci} of {cn}  ·  {sha}", fill=(240, 242, 241), font=R4.font(56, True))
d.text((120, 420), f"front-camera SAM3 map  ·  {flabel}", fill=(210, 214, 212), font=R4.font(32))
d.text((120, 480), f"ground: {ground}  ·  {len(ff)} frames, 5 per second", fill=(170, 178, 174), font=R4.font(28))
b = bars.get(c8)
if b:
    d.text((120, 560), f"vs today's pipeline: map cells identical {100 * b['N1']:.2f} %  ·  edge IoU {b.get('edge', float('nan')):.3f}  ·  frame rasters {100 * b['N4']:.2f} %",
           fill=(170, 178, 174), font=R4.font(26))
for q in range(5):
    card.save(out / f"f{q:04d}.jpg", quality=90)

for n, (fa, fr_) in enumerate(zip(ff, rf)):
    za = np.load(fa, allow_pickle=True); zr = np.load(fr_, allow_pickle=True)
    tok = str(zr["tok"]); T = zr["T_world_rig"]; fd = root / f"seq_{c8}" / tok
    meta = json.loads((fd / "meta.json").read_text(encoding="utf-8"))
    c = np.load(fd / "calib.npz"); frj = json.loads((fd / "frame.json").read_text())
    lidar = np.load(fd / "lidar.npy")
    grid, fb = P.ground_grid(lidar.astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    canvas = Image.new("RGB", (1920, 1080), (14, 17, 16)); d = ImageDraw.Draw(canvas)
    d.text((10, 8), f"SAM3 front-camera ground-truth map · clip {ci}/{cn} {sha} · t = {(meta['t_ref_us'] - meta0['t_ref_us']) / 1e6:5.1f} s · ground: {ground}",
           fill=(240, 242, 241), font=R4.font(24, True))
    d.text((10, 40), "colour = map class under the pixel's ground point · painted only where that arm's SAM3 sees a road-level surface in this frame (no boxes)",
           fill=(170, 178, 174), font=R4.font(14))
    img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
    C = CM.Camera.from_calib(c, frj["cam_order"].index(cam), img.shape[1], img.shape[0])
    ego = zr[f"ego_{cam}"] if f"ego_{cam}" in zr.files else None
    wxy, keep0, hw = GR.camera_ground(C, sg, T, 2, lidar=None, ego_small=ego)
    small = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    for y0, z, disp, wm, title in ((88, za, disp_f, wf, "COMBINED APPROVED MEASURES · map in the front camera"), (590, zr, disp_r, wr, "TODAY'S PIPELINE (reference) · map in the front camera")):
        keep = keep0 & sam3_ground(z[f"cls_{cam}"], hw)
        cls_px = GR.lookup(wxy, keep, hw, disp, wm["origin"], float(wm["res"]))
        canvas.paste(R4.overlay(small, cls_px).resize((850, 478)), (10, y0))
        d.text((10, y0 - 22), title, fill=(255, 255, 255), font=R4.font(16, True))
    for x0, wm, label in ((880, wf, "COMBINED · BEV"), (1200, wr, "TODAY · BEV")):
        canvas.paste(Image.fromarray(bev(wm, T)), (x0, 88))
        d.text((x0, 66), label, fill=(255, 255, 255), font=R4.font(16, True))
        ex, ey = x0 + int(BY[1] / RES), 88 + int(BX[1] / RES)
        d.polygon([(ex, ey - 12), (ex - 8, ey + 8), (ex + 8, ey + 8)], fill=(255, 255, 255))
    x0, yy = 1530, 66
    for k in (1, 2, 3, 6, 4, 5, 7):
        d.rectangle([x0, yy + 2, x0 + 16, yy + 18], fill=R4.COL[k]); d.text((x0 + 24, yy), R4.NAME[k], fill=(230, 232, 231), font=R4.font(15)); yy += 22
    d.rectangle([x0, yy + 2, x0 + 16, yy + 18], fill=tuple(SEEN_BG)); d.text((x0 + 24, yy), "BEV: seen, no map class", fill=(230, 232, 231), font=R4.font(15)); yy += 22
    d.rectangle([x0, yy + 2, x0 + 16, yy + 18], fill=tuple(NEVER), outline=(90, 90, 90)); d.text((x0 + 24, yy), "BEV: never seen", fill=(230, 232, 231), font=R4.font(15)); yy += 40
    d.text((880, 700), "combined = " + flabel, fill=(200, 205, 202), font=R4.font(14))
    if b:
        d.text((880, 728), "this clip, combined vs today (whole clip)", fill=(240, 242, 241), font=R4.font(16, True))
        lines = [f"map cells identical  {100 * b['N1']:.2f} %", f"edge IoU  {b.get('edge', float('nan')):.3f}   ·   crosswalk IoU  {b.get('crosswalk', float('nan')):.3f}",
                 f"line IoU  {b.get('line', float('nan')):.3f}   ·   drivable IoU  {b.get('drivable', float('nan')):.3f}", f"frame rasters identical  {100 * b['N4']:.2f} %",
                 (f"s per frame: today {b['s_ref']:.2f} · combined {b.get('s_fast', float('nan')):.2f}" if "s_ref" in b
                  else f"s per frame, combined measures: {b.get('s_fast', float('nan')):.2f}")]
        for q, t in enumerate(lines):
            d.text((880, 756 + 24 * q), t, fill=(210, 214, 212), font=R4.font(15))
    d.text((10, 1072 - 14), "Ground truth uses future frames and the ground surface (label time only); inference is vision-only.", fill=(150, 158, 154), font=R4.font(13))
    canvas.save(out / f"f{n + 5:04d}.jpg", quality=90)
print(f"ZZRENDERLONG-{c8}-{len(ff) + 5}ZZ")
