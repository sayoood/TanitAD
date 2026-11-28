"""Does SAM3 separate zebra stripes better on a zoomed crop than on the full image? For listed frames / cameras: "crosswalk stripe"
(>= 0.4) on the full 1920x1080 image vs on a crop around the crossing (the crossing cells of the world map projected into the image,
box grown 15 %, at least 800 px wide, 16:9), the crop mask pasted back. Also the image white top-hat inside each mask vs inside the
crossing's gaps (the same bar/gap test as xwalk_reproj_contrast.py, per image) and the mask area. A sheet shows both masks.
Usage: stripe_zoom_test.py <c8> <npz dir> <fields render dir> <out png> <cam:frame> [...]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sam3map_extract_v6 as E
import camera_model as CM
import ground_surface as GS
import sam3_paint as P
import gt_reproject as GR
import sam3map_render_v4 as R4

ROOT = Path("/home/nvidia/sam3map/native7")
c8, nd, fdir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
F = np.load(fdir / "worldmap_fields.npz", allow_pickle=True)
vk = {k: F[f"vk{k}"].astype(np.float32) for k in (1, 2, 3, 4, 6, 7)}
dw = vk[1] + vk[2] + vk[3] + vk[4] + vk[6]; drv = (dw > 0) & (dw >= vk[7])
x_any = (vk[3] > 0) | (F["vraw3"].astype(np.float32) > 0)
cross = ((cv2.morphologyEx(x_any.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) & drv).astype(np.uint8)
proc, _ = E.S.build(conf=0.25)
tiles, rep = [], []
for arg in sys.argv[5:]:
    cam, fr_ = arg.split(":"); j = int(fr_)
    z = np.load(nd / f"{j:03d}.npz", allow_pickle=True); fd = ROOT / f"seq_{c8}" / str(z["tok"]); T = z["T_world_rig"]
    img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"); W, H = img.size
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    C = CM.Camera.from_calib(c, fr["cam_order"].index(cam), W, H)
    grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    wxy, keep, hw = GR.camera_ground(C, sg, T, 2, lidar=None, ego_small=z[f"ego_{cam}"] if f"ego_{cam}" in z.files else None)
    inx = GR.lookup(wxy, keep, hw, cross, F["origin"], float(F["res"])) > 0                   # 540 x 960
    if inx.sum() < 500:
        rep.append((arg, "no crossing in view")); continue
    vv, uu = np.nonzero(inx); u0, u1, v0, v1 = uu.min() * 2, uu.max() * 2, vv.min() * 2, vv.max() * 2
    cw = max(800, int((u1 - u0) * 1.15)); ch = max(int(cw * 9 / 16), int((v1 - v0) * 1.15)); cw = max(cw, int(ch * 16 / 9))
    cu, cv_ = (u0 + u1) // 2, (v0 + v1) // 2
    x0 = int(np.clip(cu - cw // 2, 0, max(W - cw, 0))); y0 = int(np.clip(cv_ - ch // 2, 0, max(H - ch, 0))); x1, y1 = min(x0 + cw, W), min(y0 + ch, H)
    t0 = time.time()
    st_full = proc.set_image(img)
    mf = np.zeros((H, W), bool)
    for s, m in E.instances(proc, st_full, "crosswalk stripe", 0.4):
        mf |= m
    crop = img.crop((x0, y0, x1, y1))
    st_crop = proc.set_image(crop)
    mc_ = np.zeros((y1 - y0, x1 - x0), bool)
    for s, m in E.instances(proc, st_crop, "crosswalk stripe", 0.4):
        mc_ |= m
    mc = np.zeros((H, W), bool); mc[y0:y1, x0:x1] = mc_
    dt = time.time() - t0
    Y = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    th = cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (127, 127)))
    region = cv2.resize(inx.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    row = {"view": arg, "crop_px": [x0, y0, x1, y1], "seconds": round(dt, 1)}
    for name, m in (("full", mf), ("zoom", mc)):
        bar = m & region; gap = region & ~m
        row[name] = {"mask_px_in_crossing": int(bar.sum()), "bar_share_of_crossing": round(float(bar.sum()) / max(int(region.sum()), 1), 3),
                     "bar_tophat": round(float(th[bar].mean()), 1) if bar.any() else None, "gap_tophat": round(float(th[gap].mean()), 1) if gap.any() else None}
    rep.append((arg, row))
    base = np.asarray(img).astype(np.float32)
    for name, m, col in (("full image", mf, (0, 230, 230)), ("zoomed crop", mc, (255, 0, 255))):
        a = base.copy(); a[m] = 0.35 * a[m] + 0.65 * np.array(col)
        t = Image.fromarray(a.astype(np.uint8)).crop((x0, y0, x1, y1))
        t = t.resize((int(t.width * 420 / t.height), 420))
        tiles.append((f"{arg} · SAM3 on {name}", t))
Wt = max(sum(t.width + 10 for _, t in tiles[i:i + 2]) for i in range(0, len(tiles), 2)) + 10 if tiles else 10
sheet = Image.new("RGB", (Wt, (len(tiles) // 2) * 450 + 10), (12, 14, 13)); d = ImageDraw.Draw(sheet)
for i in range(0, len(tiles), 2):
    x = 10
    for label, t in tiles[i:i + 2]:
        sheet.paste(t, (x, (i // 2) * 450 + 28)); d.text((x, (i // 2) * 450 + 6), label, fill=(230, 232, 231), font=R4.font(14, True)); x += t.width + 10
sheet.save(out)
for arg, row in rep:
    print(row)
print("ZZZOOM-DONEZZ")
