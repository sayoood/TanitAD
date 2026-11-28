"""Review sheet on the PI's hard frames: for one clip, each listed frame x each variant world map (the ground-truth map cut
at that frame, x -20..40 m forward-up, y +-15 m, 0.1 m, full colour, edges dilated one cell for display), side by side,
with the front camera image of the frame on the left. Variants that do not exist on disk are skipped.
Usage: hard_frames_sheet.py <c8> <npz dir for poses> <out png> <frames comma> <label=render dir> [<label=render dir> ...]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval")
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sam3map_render_v4 as R4

SM = Path("/home/nvidia/sam3map")
c8, npz_dir, out, frames_s = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
variants = [(a.split("=", 1)[0], Path(a.split("=", 1)[1])) for a in sys.argv[5:]]
variants = [(l, d) for l, d in variants if (d / "worldmap.npz").exists()]
frames = [int(x) for x in frames_s.split(",")]
RES, BX, BY = 0.10, (-20.0, 40.0), (-15.0, 15.0)
xs = np.arange(BX[1] - RES / 2, BX[0], -RES); ys = np.arange(BY[1] - RES / 2, BY[0], -RES)
GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]
W_ = {l: dict(np.load(d / "worldmap.npz", allow_pickle=True)) for l, d in variants}
files = sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))


def bev(wm, T):
    cls, (x0, y0), res = wm["cls"], wm["origin"], float(wm["res"])
    w = XY @ T[:2, :2].T + T[:2, 3]
    i = np.floor((w[:, 0] - x0) / res).astype(np.int64); j = np.floor((w[:, 1] - y0) / res).astype(np.int64)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    code = np.full(len(XY), 255, np.uint8); code[ok] = cls[i[ok], j[ok]]; code = code.reshape(GX.shape)
    img = np.zeros(code.shape + (3,), np.uint8); img[:] = (14, 16, 15); img[code == 0] = (58, 62, 60)
    for k in (7, 1, 2, 4, 6, 3):
        img[code == k] = R4.COL[k]
    img[cv2.dilate((code == 5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = R4.COL[5]
    return img


PW, PH = 150, 300                                              # half-size BEV panels
sheet = Image.new("RGB", (500 + len(variants) * (PW + 8), len(frames) * (PH + 30) + 40), (12, 14, 13)); d = ImageDraw.Draw(sheet)
d.text((8, 8), f"clip {json.loads((SM / 'native7' / f'seq_{c8}' / str(np.load(files[0], allow_pickle=True)['tok']) / 'meta.json').read_text())['clip_sha12']} · hard frames × map variants (ground-truth map at the frame, 0.1 m)",
       fill=(235, 238, 236), font=R4.font(16, True))
for q, j in enumerate(frames):
    z = np.load(files[j], allow_pickle=True); T = z["T_world_rig"]; tok = str(z["tok"])
    y0 = 40 + q * (PH + 30)
    img = Image.open(SM / "native7" / f"seq_{c8}" / tok / "images" / "CAM_FW.jpg").convert("RGB").resize((480, 270))
    sheet.paste(img, (8, y0 + 20)); d.text((8, y0 + 2), f"token {j} (t = {j * 0.2:.1f} s) front camera", fill=(220, 224, 222), font=R4.font(13, True))
    for v, (label, _) in enumerate(variants):
        x0 = 500 + v * (PW + 8)
        sheet.paste(Image.fromarray(bev(W_[label], T)).resize((PW, PH), Image.NEAREST), (x0, y0 + 20))
        d.text((x0, y0 + 2), label, fill=(220, 224, 222), font=R4.font(12, True))
        ex, ey = x0 + PW // 2, y0 + 20 + int(BX[1] / RES / 2)
        d.polygon([(ex, ey - 6), (ex - 4, ey + 4), (ex + 4, ey + 4)], fill=(255, 255, 255))
sheet.save(out)
print("ZZSHEET-OK", out, [l for l, _ in variants])
