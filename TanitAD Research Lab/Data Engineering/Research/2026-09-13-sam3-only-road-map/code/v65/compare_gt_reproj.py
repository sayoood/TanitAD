"""Side-by-side video frames, FRONT CAMERA ONLY vs ALL 7 CAMERAS, with each arm's ground-truth WORLD MAP drawn back into
the camera images (gt_reproject.py: ray -> smooth LiDAR ground -> world cell; occluded by LiDAR obstacles, tracked agent
boxes and the ego body). What is painted in a camera is exactly what the BEV ground truth says -- no per-frame SAM3 mask --
so misplaced edges, gaps and spill are visible against the real image.
Layout: left, the front camera twice (front-only map above, all-camera map below); middle, both BEVs (the world map cut at
the frame, x -20..40 m forward-up, y +-15 m, 0.1 m, full colour); below them the four other cameras with the all-camera map;
right, legend, LiDAR checks and the PI checks (pi_checks.py).
Usage: compare_gt_reproj.py <c8> <front npz> <front render> <all npz> <all render> <out dir> <front score json> <all score json>
                            [<pi checks json> <front label> <all label>]"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
import sam3map_render_v4 as R4
import sam3map_render_v5c as R5
import gt_reproject as GR

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7")); V2 = Path("/home/nvidia/qwendrive/v2")
c8, fnpz, frdr, anpz, ardr, out = [Path(a) if i else a for i, a in enumerate(sys.argv[1:7])]
fscore, ascore = json.loads(Path(sys.argv[7]).read_text()), json.loads(Path(sys.argv[8]).read_text())
pic = json.loads(Path(sys.argv[9]).read_text()) if len(sys.argv) > 9 and Path(sys.argv[9]).exists() else None
flab, alab = (sys.argv[10], sys.argv[11]) if len(sys.argv) > 11 else (None, None)
out.mkdir(parents=True, exist_ok=True)
RES, BX, BY = 0.10, (-20.0, 40.0), (-15.0, 15.0)
xs = np.arange(BX[1] - RES / 2, BX[0], -RES); ys = np.arange(BY[1] - RES / 2, BY[0], -RES)
GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]
SEEN_BG, NEVER = np.array((58, 62, 60), np.uint8), np.array((14, 16, 15), np.uint8)
SIDE = ("CAM_CL", "CAM_CR", "CAM_RL", "CAM_RR")
SIDE_NAME = {"CAM_CL": "cross-left", "CAM_CR": "cross-right", "CAM_RL": "rear-left", "CAM_RR": "rear-right"}


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
    return img, code


def val(sc, arm, k):
    r = sc.get(arm, {}).get(k)
    return r.get("value", "-") if isinstance(r, dict) else "-"


wf = dict(np.load(frdr / "worldmap.npz", allow_pickle=True)); wa = dict(np.load(ardr / "worldmap.npz", allow_pickle=True))
disp_f, disp_a = GR.display_world(wf["cls"]), GR.display_world(wa["cls"])
ffiles = sorted(fnpz.glob("[0-9][0-9][0-9].npz")); afiles = sorted(anpz.glob("[0-9][0-9][0-9].npz"))
assert [f.name for f in ffiles] == [f.name for f in afiles]
meta0 = None
ONLY = {int(x) for x in os.environ["FRAMES"].split(",")} if os.environ.get("FRAMES") else None
for n, (ff, af) in enumerate(zip(ffiles, afiles)):
    zf = np.load(ff, allow_pickle=True); za = np.load(af, allow_pickle=True)
    tok = str(za["tok"]); assert str(zf["tok"]) == tok
    T = za["T_world_rig"]; fd = ROOT / f"seq_{c8}" / tok
    meta = json.loads((fd / "meta.json").read_text(encoding="utf-8")); meta0 = meta0 or meta
    if ONLY is not None and n not in ONLY:
        continue
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    lidar = np.load(fd / "lidar.npy")
    grid, fb = P.ground_grid(lidar.astype(np.float64)); sg = GS.smooth_grid(grid, fb)
    gp = V2 / f"seq_{c8}" / tok / "gt.npz"
    boxes = np.load(gp)["boxes"] if gp.exists() else np.zeros((0, 7))
    canvas = Image.new("RGB", (1920, 1080), (14, 17, 16)); d = ImageDraw.Draw(canvas)
    d.text((10, 8), f"SAM3 ground-truth map drawn INTO the cameras · FRONT CAMERA ONLY vs ALL 7 · clip {meta['clip_sha12']} · t = {(meta['t_ref_us'] - meta0['t_ref_us']) / 1e6:5.1f} s",
           fill=(240, 242, 241), font=R4.font(24, True))
    d.text((10, 38), "every coloured camera pixel = the BEV ground-truth class under that pixel's ground point (LiDAR ground; agents, ego body and LiDAR obstacles in front are left unpainted)",
           fill=(170, 178, 174), font=R4.font(14))
    # ---- front camera, both maps
    img = np.asarray(Image.open(fd / "images" / "CAM_FW.jpg").convert("RGB"))
    C = CM.Camera.from_calib(c, fr["cam_order"].index("CAM_FW"), img.shape[1], img.shape[0])
    wxy, keep, hw = GR.camera_ground(C, sg, T, 2, lidar=lidar, ego_small=za["ego_CAM_FW"] if "ego_CAM_FW" in za.files else None,
                                     occ_small=R5.agent_occluders(C, boxes))
    small = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    for y0, disp, wm, title in ((86, disp_f, wf, "FRONT CAMERA ONLY · its ground-truth map"), (568, disp_a, wa, "ALL 7 CAMERAS · their ground-truth map")):
        cls_px = GR.lookup(wxy, keep, hw, disp, wm["origin"], float(wm["res"]))
        canvas.paste(R4.overlay(small, cls_px).resize((800, 450)), (10, y0))
        d.text((10, y0 - 22), title + " · drawn into the front camera", fill=(255, 255, 255), font=R4.font(16, True))
    # ---- BEVs
    bf, cf = bev(wf, T); ba, ca = bev(wa, T)
    for x0, im_, label, code in ((830, bf, "FRONT ONLY · BEV", cf), (1150, ba, "ALL 7 · BEV", ca)):
        canvas.paste(Image.fromarray(im_), (x0, 86))
        d.text((x0, 64), label, fill=(255, 255, 255), font=R4.font(16, True))
        ex, ey = x0 + int(BY[1] / RES), 86 + int(BX[1] / RES)
        d.polygon([(ex, ey - 12), (ex - 8, ey + 8), (ex + 8, ey + 8)], fill=(255, 255, 255))
        d.text((x0, 688), f"seen {100 * float((code != 255).mean()):.0f} % of window", fill=(200, 205, 202), font=R4.font(13))
    # ---- the four other cameras, all-camera map
    for q, cam in enumerate(SIDE):
        im = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
        Cs = CM.Camera.from_calib(c, fr["cam_order"].index(cam), im.shape[1], im.shape[0])
        w2, k2, h2 = GR.camera_ground(Cs, sg, T, 4, lidar=lidar, ego_small=za[f"ego_{cam}"] if f"ego_{cam}" in za.files else None,
                                      occ_small=R5.agent_occluders(Cs, boxes))
        cp = GR.lookup(w2, k2, h2, disp_a, wa["origin"], float(wa["res"]))
        sm = cv2.resize(im, (h2[1], h2[0]), interpolation=cv2.INTER_AREA)
        x0, y0 = 830 + (q % 2) * 310, 724 + (q // 2) * 175
        canvas.paste(R4.overlay(sm, cp).resize((300, 169)), (x0, y0))
        d.text((x0 + 4, y0 + 2), SIDE_NAME[cam] + " · all-7 map", fill=(255, 255, 255), font=R4.font(12, True))
    d.text((830, 706), "the other cameras, all-camera map drawn in", fill=(200, 205, 202), font=R4.font(13))
    # ---- legend and checks
    x0, yy = 1480, 64
    for k in (1, 2, 3, 6, 4, 5, 7):
        d.rectangle([x0, yy + 2, x0 + 16, yy + 18], fill=R4.COL[k]); d.text((x0 + 24, yy), R4.NAME[k], fill=(230, 232, 231), font=R4.font(15)); yy += 22
    d.rectangle([x0, yy + 2, x0 + 16, yy + 18], fill=tuple(SEEN_BG)); d.text((x0 + 24, yy), "BEV: seen, no map class", fill=(230, 232, 231), font=R4.font(15)); yy += 22
    d.rectangle([x0, yy + 2, x0 + 16, yy + 18], fill=tuple(NEVER), outline=(90, 90, 90)); d.text((x0 + 24, yy), "BEV: never seen", fill=(230, 232, 231), font=R4.font(15)); yy += 34
    d.text((x0, yy), "LiDAR checks, 96 frames", fill=(240, 242, 241), font=R4.font(16, True)); yy += 22
    d.text((x0, yy), "A2 tall obstacles on road ↓ (true map 0.0003)", fill=(170, 178, 174), font=R4.font(12)); yy += 16
    d.text((x0, yy), "C vehicles on road ↑ · E edges on curb ↑", fill=(170, 178, 174), font=R4.font(12)); yy += 22
    for title, sc in (("front only", fscore), ("all 7", ascore)):
        d.text((x0, yy), title, fill=(230, 232, 231), font=R4.font(15, True)); yy += 20
        d.text((x0 + 10, yy), f"A2 {val(sc, 'SAM3_GT_worldmap', 'MAP_A2')}   C {val(sc, 'SAM3_GT_worldmap', 'MAP_C')}   E {val(sc, 'SAM3_GT_worldmap', 'MAP_E')}",
               fill=(210, 214, 212), font=R4.font(14)); yy += 18
        o = sc.get("observed_only", {})
        d.text((x0 + 10, yy), f"inside own camera coverage: A2 {val(o, 'SAM3_GT_worldmap', 'MAP_A2')}", fill=(170, 178, 174), font=R4.font(12)); yy += 24
    if pic is not None and flab in pic["variants"] and alab in pic["variants"]:
        yy += 6
        d.text((x0, yy), "PI checks (whole clip, <= 30 m of path)", fill=(240, 242, 241), font=R4.font(16, True)); yy += 22
        for title, lab in (("front only", flab), ("all 7", alab)):
            r = pic["variants"][lab]["roi"]
            d.text((x0, yy), title, fill=(230, 232, 231), font=R4.font(15, True)); yy += 20
            d.text((x0 + 10, yy), f"noise fragments / 1000 m²  {r['fragments_per_1000m2']}", fill=(210, 214, 212), font=R4.font(13)); yy += 17
            d.text((x0 + 10, yy), f"LiDAR curbs with an edge  {r['curb_recall_edge_within_0.6m']}", fill=(210, 214, 212), font=R4.font(13)); yy += 17
            d.text((x0 + 10, yy), f"edges on a LiDAR curb/obstacle  {r['edge_precision_within_0.6m']}", fill=(210, 214, 212), font=R4.font(13)); yy += 22
    d.text((10, 1040), "Ground truth may use future frames, LiDAR ground height and tracked boxes (label time only); inference is vision-only.",
           fill=(150, 158, 154), font=R4.font(13))
    canvas.save(out / f"f{n:04d}.png")
    if n % 20 == 0:
        print(f"  frame {n}", flush=True)
print("ZZCMPREPROJ-DONEZZ", len(ffiles))
