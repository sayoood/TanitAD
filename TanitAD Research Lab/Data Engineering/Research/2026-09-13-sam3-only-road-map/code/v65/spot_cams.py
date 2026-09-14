"""Raw camera crops of one world spot (a box in the rig frame of frame J) over several frames and cameras, thin outline only.
Usage: spot_cams.py <c8> <npz dir> <J> <x0,x1,y0,y1> <out png> <cam:frame> [...]"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
from PIL import Image, ImageDraw
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
import sam3map_render_v4 as R4
ROOT = Path("/home/nvidia/sam3map/native7")
c8, nd, J, box, out = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), [float(v) for v in sys.argv[4].split(",")], Path(sys.argv[5])
x0, x1, y0, y1 = box
T = np.load(nd / f"{J:03d}.npz", allow_pickle=True)["T_world_rig"]
corners_w = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]]) @ T[:2, :2].T + T[:2, 3]
edge_w = np.concatenate([a + np.linspace(0, 1, 40)[:, None] * (b - a) for a, b in zip(corners_w, np.roll(corners_w, -1, axis=0))])
tiles = []
for arg in sys.argv[6:]:
    cam, fr_ = arg.split(":"); jj = int(fr_)
    zz = np.load(nd / f"{jj:03d}.npz", allow_pickle=True); fdd = ROOT / f"seq_{c8}" / str(zz["tok"])
    img = Image.open(fdd / "images" / f"{cam}.jpg").convert("RGB")
    c = np.load(fdd / "calib.npz"); fr = json.loads((fdd / "frame.json").read_text())
    C = CM.Camera.from_calib(c, fr["cam_order"].index(cam), img.width, img.height)
    grid, fb = P.ground_grid(np.load(fdd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    Tj = zz["T_world_rig"]
    q = (edge_w - Tj[:2, 3]) @ Tj[:2, :2]
    u, v, ok = C.project_rig(np.c_[q, GS.height(q, sg)])
    if ok.sum() < 10:
        continue
    d = ImageDraw.Draw(img)
    pts = [(float(a), float(b)) for a, b, k in zip(u, v, ok) if k]
    d.line(pts, fill=(255, 0, 255), width=2)
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    cx0, cx1 = max(min(us) - 200, 0), min(max(us) + 200, img.width); cy0, cy1 = max(min(vs) - 120, 0), min(max(vs) + 120, img.height)
    im = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
    s = 420 / im.height; im = im.resize((max(1, int(im.width * s)), 420))
    tiles.append((f"{cam} frame {jj}", im))
W = sum(t[1].width + 10 for t in tiles) + 10
sheet = Image.new("RGB", (min(W, 3600), 460), (12, 14, 13)); dd = ImageDraw.Draw(sheet)
x = 10
for label, im in tiles:
    if x + im.width > sheet.width:
        break
    sheet.paste(im, (x, 30)); dd.text((x, 8), label, fill=(230, 232, 231), font=R4.font(14, True)); x += im.width + 10
sheet.save(out); print("ZZCAMS-OK", len(tiles))
