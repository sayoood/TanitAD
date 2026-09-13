"""Side-by-side video frames: the SAM3 ground-truth map from the FRONT CAMERA ONLY vs from ALL 7 native cameras, same
clip, same frame, same pipeline (refine v6 -> consensus -> renderer v5 world map), drawn identically.
Both BEV panels are the non-causal world map cut at the frame's pose (x -20..40 m forward-up, y +-15 m, 0.1 m cells),
drawn in FULL colour (no range fading): coloured = labelled class, grey = seen but no map class, near-black = never seen
by that arm. Display only: edge cells dilated by one cell in both panels. LiDAR checks of both arms are printed.
Usage: compare_front_vs_all.py <c8> <front npz dir> <front render dir> <all npz dir> <all render dir> <out dir> <front score json> <all score json>
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sam3map_render_v4 as R4
import sam3map_render_v5 as R5

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
c8, fnpz, frdr, anpz, ardr, out = [Path(a) if i else a for i, a in enumerate(sys.argv[1:7])]
fscore, ascore = json.loads(Path(sys.argv[7]).read_text()), json.loads(Path(sys.argv[8]).read_text())
out.mkdir(parents=True, exist_ok=True)
RES, BX, BY = 0.10, (-20.0, 40.0), (-15.0, 15.0)
xs = np.arange(BX[1] - RES / 2, BX[0], -RES); ys = np.arange(BY[1] - RES / 2, BY[0], -RES)
GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]
SEEN_BG, NEVER = np.array((58, 62, 60), np.uint8), np.array((14, 16, 15), np.uint8)


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
    edge = cv2.dilate((code == 5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    img[edge] = R4.COL[5]
    return img, code


def row(score, arm):
    r = score.get(arm, {})
    return [r.get(k, {}).get("value", "-") if isinstance(r.get(k), dict) else "-" for k in ("MAP_A2", "MAP_B", "MAP_C", "MAP_E")]


wf = dict(np.load(frdr / "worldmap.npz", allow_pickle=True)); wa = dict(np.load(ardr / "worldmap.npz", allow_pickle=True))
ffiles = sorted(fnpz.glob("[0-9][0-9][0-9].npz")); afiles = sorted(anpz.glob("[0-9][0-9][0-9].npz"))
assert [f.name for f in ffiles] == [f.name for f in afiles]
meta0 = None
for n, (ff, af) in enumerate(zip(ffiles, afiles)):
    zf = np.load(ff, allow_pickle=True); za = np.load(af, allow_pickle=True)
    tok = str(za["tok"]); assert str(zf["tok"]) == tok
    T = za["T_world_rig"]; fd = ROOT / f"seq_{c8}" / tok
    meta = json.loads((fd / "meta.json").read_text(encoding="utf-8")); meta0 = meta0 or meta
    canvas = Image.new("RGB", (1920, 1080), (14, 17, 16)); d = ImageDraw.Draw(canvas)
    d.text((10, 10), f"SAM3 map · FRONT CAMERA ONLY vs ALL 7 CAMERAS · same pipeline · clip {meta['clip_sha12']} · t = {(meta['t_ref_us'] - meta0['t_ref_us']) / 1e6:5.1f} s",
           fill=(240, 242, 241), font=R4.font(26, True))
    d.text((10, 42), "BEV = the ground-truth world map at this frame (all frames of the clip), full colour, 0.1 m · grey: seen, no map class · black: never seen by that arm",
           fill=(170, 178, 174), font=R4.font(14))
    img = np.asarray(Image.open(fd / "images" / "CAM_FW.jpg").convert("RGB"))
    canvas.paste(R4.overlay(img, R5.display_cls(img, zf["cls_CAM_FW"], zf["ego_CAM_FW"] if "ego_CAM_FW" in zf.files else None)).resize((960, 540)), (10, 70))
    d.text((18, 76), "front camera (front-only arm classes)", fill=(255, 255, 255), font=R4.font(16, True))
    for q, cam in enumerate(("CAM_CL", "CAM_CR", "CAM_RL", "CAM_RR")):
        im = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
        canvas.paste(R4.overlay(im, R5.display_cls(im, za[f"cls_{cam}"], za[f"ego_{cam}"] if f"ego_{cam}" in za.files else None)).resize((236, 133)), (10 + q * 241, 620))
        d.text((14 + q * 241, 623), cam.replace("CAM_", ""), fill=(255, 255, 255), font=R4.font(13, True))
    d.text((10, 757), "the four other cameras used by the all-camera arm (plus the two tele cameras)", fill=(170, 178, 174), font=R4.font(13))
    bf, cf = bev(wf, T); ba, ca = bev(wa, T)
    for x0, im_, label in ((1000, bf, "FRONT CAMERA ONLY"), (1320, ba, "ALL 7 CAMERAS")):
        canvas.paste(Image.fromarray(im_), (x0, 100))
        d.text((x0, 72), label, fill=(255, 255, 255), font=R4.font(18, True))
        ex, ey = x0 + int(BY[1] / RES), 100 + int(BX[1] / RES)
        d.polygon([(ex, ey - 12), (ex - 8, ey + 8), (ex + 8, ey + 8)], fill=(255, 255, 255))
    yy = 720
    for x0, code in ((1000, cf), (1320, ca)):
        seen = float((code != 255).mean())
        d.text((x0, 706), f"seen {100 * seen:.0f} % of window", fill=(200, 205, 202), font=R4.font(14))
    d.text((1640, 100), "LiDAR checks (96 frames)", fill=(240, 242, 241), font=R4.font(17, True))
    d.text((1640, 124), "A2 obst.on road ↓ · B path on road ↑", fill=(170, 178, 174), font=R4.font(12))
    d.text((1640, 140), "C vehicles on road ↑ · E edges on curb ↑", fill=(170, 178, 174), font=R4.font(12))
    yy = 168
    for title, sc, arm in (("front · GT map", fscore, "SAM3_GT_worldmap"), ("all 7 · GT map", ascore, "SAM3_GT_worldmap"),
                           ("front · clip vote", fscore, "SAM3_clipvote"), ("all 7 · clip vote", ascore, "SAM3_clipvote")):
        v = row(sc, arm)
        d.text((1640, yy), title, fill=(230, 232, 231), font=R4.font(15, True)); yy += 20
        d.text((1650, yy), f"A2 {v[0]}  B {v[1]}", fill=(210, 214, 212), font=R4.font(14)); yy += 18
        d.text((1650, yy), f"C {v[2]}  E {v[3]}", fill=(210, 214, 212), font=R4.font(14)); yy += 26
    for q, k in enumerate((1, 2, 3, 6, 4, 5, 7)):
        xx, y2 = 10 + (q % 4) * 240, 790 + (q // 4) * 26
        d.rectangle([xx, y2, xx + 20, y2 + 20], fill=R4.COL[k]); d.text((xx + 28, y2), R4.NAME[k], fill=(230, 232, 231), font=R4.font(16))
    d.rectangle([10, 852, 30, 872], fill=tuple(SEEN_BG)); d.text((38, 852), "seen, no map class", fill=(230, 232, 231), font=R4.font(16))
    d.rectangle([250, 852, 270, 872], fill=tuple(NEVER), outline=(90, 90, 90)); d.text((278, 852), "never seen", fill=(230, 232, 231), font=R4.font(16))
    for x0, code in ((1000, cf), (1320, ca)):
        y3 = 740
        for k in (1, 2, 3, 6, 4, 5, 7):
            d.rectangle([x0, y3 + 2, x0 + 14, y3 + 16], fill=R4.COL[k])
            d.text((x0 + 20, y3), f"{R4.NAME[k][:14]:14s} {float((code == k).sum()) * RES ** 2:6.1f} m²", fill=(220, 224, 222), font=R4.font(13)); y3 += 19
    canvas.save(out / f"f{n:04d}.png")
print("ZZCOMPARE-DONEZZ", len(ffiles))
