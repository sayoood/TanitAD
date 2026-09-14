"""One spot of the map, every layer of evidence side by side (PI 2026-09-14 on the box-free videos: hatched area shown red,
noisy yellow/red border, crosswalk stripes not separated).
For a box in the rig frame of frame J: the world map of each listed version (0.1 m, drawn 4x), the vote fields of the delivered
map (renderer v5m FIELDS=1) as heat maps, and the front camera at two frames with the box outlined on the ground.
Fields: sidewalk share sw/(dw+sw); hatched share vk6/dw after trimming and vraw6/obs before; line vk2/dw; crosswalk vk3/dw;
bright share vpf/obs (observations whose top-hat beats the layer's paint threshold); mean top-hat / threshold vth/obs;
edge share vedge/vnear; background share vbg/obs.
Usage: spot_diag.py <c8> <npz dir> <J> <x0,x1,y0,y1> <out png> <fields render dir> <label=render dir> [...]"""
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

ROOT = Path("/home/nvidia/sam3map/native7")
c8, nd, J, box, out, fdir = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), [float(v) for v in sys.argv[4].split(",")], Path(sys.argv[5]), Path(sys.argv[6])
variants = [(a.split("=", 1)[0], Path(a.split("=", 1)[1])) for a in sys.argv[7:]]
x0, x1, y0, y1 = box
RES = 0.1; S = 4
z = np.load(nd / f"{J:03d}.npz", allow_pickle=True); T = z["T_world_rig"]; tok = str(z["tok"])
xs = np.arange(x1 - RES / 2, x0, -RES); ys = np.arange(y1 - RES / 2, y0, -RES)
GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]
W = XY @ T[:2, :2].T + T[:2, 3]


def sample(arr, origin, res, fill=0):
    i = np.floor((W[:, 0] - origin[0]) / res).astype(np.int64); j = np.floor((W[:, 1] - origin[1]) / res).astype(np.int64)
    ok = (i >= 0) & (i < arr.shape[0]) & (j >= 0) & (j < arr.shape[1])
    o = np.full(len(W), fill, arr.dtype if arr.dtype != np.float16 else np.float32); o[ok] = arr[i[ok], j[ok]]
    return o.reshape(GX.shape)


def paint_cls(code):
    img = np.zeros(code.shape + (3,), np.uint8); img[:] = (14, 16, 15); img[code == 0] = (58, 62, 60)
    for k in (7, 1, 2, 4, 6, 3, 5):
        img[code == k] = R4.COL[k]
    return img


def heat(v, lo=0.0, hi=1.0):
    t = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]


panels = []
for label, d in variants:
    wm = np.load(d / "worldmap.npz", allow_pickle=True)
    panels.append((label, paint_cls(sample(wm["cls"], wm["origin"], float(wm["res"]), 255))))
F = np.load(fdir / "worldmap_fields.npz", allow_pickle=True)
g = {k: sample(F[k].astype(np.float32), F["origin"], float(F["res"])) for k in F.files if k.startswith("v")}
dw = g["vk1"] + g["vk2"] + g["vk3"] + g["vk4"] + g["vk6"]; sw = g["vk7"]; eps = 1e-6
fp = [("sidewalk share", heat(sw / (dw + sw + eps))), ("hatched / road (trimmed)", heat(g["vk6"] / (dw + eps), 0, 0.5)),
      ("hatched / obs (raw SAM3)", heat(g["vraw6"] / (g["vobs"] + eps), 0, 0.5)), ("line / road", heat(g["vk2"] / (dw + eps), 0, 0.5)),
      ("crosswalk / road", heat(g["vk3"] / (dw + eps), 0, 0.5)), ("bright share (image)", heat(g["vpf"] / (g["vobs"] + eps), 0, 0.6)),
      ("top-hat / threshold", heat(g["vth"] / (g["vobs"] + eps), 0, 2.0)), ("edge share (near)", heat(g["vedge"] / (g["vnear"] + eps), 0, 0.5)),
      ("background share", heat(g["vbg"] / (g["vobs"] + eps), 0, 1.0))]
panels += fp
ph, pw = GX.shape[0] * S, GX.shape[1] * S
cols = 6
rows_n = (len(panels) + cols - 1) // cols
cam_h = 300
sheet = Image.new("RGB", (cols * (pw + 10) + 10, rows_n * (ph + 26) + cam_h + 60), (12, 14, 13)); d = ImageDraw.Draw(sheet)
d.text((10, 6), f"clip {json.loads((ROOT / f'seq_{c8}' / tok / 'meta.json').read_text())['clip_sha12']} · frame {J} · rig box x {x0}..{x1} y {y0}..{y1} m (forward up, left left)",
       fill=(235, 238, 236), font=R4.font(15, True))
for q, (label, im) in enumerate(panels):
    xx, yy = 10 + (q % cols) * (pw + 10), 30 + (q // cols) * (ph + 26)
    sheet.paste(Image.fromarray(np.ascontiguousarray(im)).resize((pw, ph), Image.NEAREST), (xx, yy + 18))
    d.text((xx, yy), label, fill=(220, 224, 222), font=R4.font(12, True))
yc = 30 + rows_n * (ph + 26)
for q, jj in enumerate((max(J - 10, 0), J)):
    zz = np.load(nd / f"{jj:03d}.npz", allow_pickle=True); fdd = ROOT / f"seq_{c8}" / str(zz["tok"])
    img = np.asarray(Image.open(fdd / "images" / "CAM_FW.jpg").convert("RGB"))
    c = np.load(fdd / "calib.npz"); fr = json.loads((fdd / "frame.json").read_text())
    C = CM.Camera.from_calib(c, fr["cam_order"].index("CAM_FW"), img.shape[1], img.shape[0])
    grid, fb = P.ground_grid(np.load(fdd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    Tj = zz["T_world_rig"]
    corners_w = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]]) @ T[:2, :2].T + T[:2, 3]
    edge_pts = []
    for a, b in zip(corners_w, np.roll(corners_w, -1, axis=0)):
        for t_ in np.linspace(0, 1, 30):
            edge_pts.append(a + t_ * (b - a))
    q_rig = (np.array(edge_pts) - Tj[:2, 3]) @ Tj[:2, :2]
    u, v, ok = C.project_rig(np.c_[q_rig, GS.height(q_rig, sg)])
    im = Image.fromarray(img); dd = ImageDraw.Draw(im)
    pts = [(float(uu), float(vv)) for uu, vv, kk in zip(u, v, ok) if kk]
    if len(pts) > 2:
        dd.line(pts, fill=(255, 0, 255), width=5)
        us, vs = [p_[0] for p_ in pts], [p_[1] for p_ in pts]
        cx0, cx1 = max(min(us) - 250, 0), min(max(us) + 250, img.shape[1]); cy0, cy1 = max(min(vs) - 150, 0), min(max(vs) + 150, img.shape[0])
        im = im.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
    sc = cam_h / im.height
    im = im.resize((max(1, int(im.width * sc)), cam_h))
    sheet.paste(im, (10 + q * (sheet.width // 2), yc + 20))
    d.text((10 + q * (sheet.width // 2), yc + 2), f"CAM_FW frame {jj}", fill=(220, 224, 222), font=R4.font(12, True))
sheet.save(out)
print("ZZSPOT-OK", out)
