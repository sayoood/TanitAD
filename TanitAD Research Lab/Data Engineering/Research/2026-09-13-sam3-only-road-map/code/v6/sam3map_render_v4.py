"""Render the SAM3-only map video, v4 BEV: inverse sampling + paint-pixel refinement + nearest-view compositing.

Why (PI 2026-09-13: "the arrows are not recognisable on the BEV despite they are well segmented in the image"; checked
with ipm_diag.py / bev_v4_proto.py on that frame): the camera image warped onto the ground shows the arrows clearly, so
calibration and ground height were right; the loss came from (1) FORWARD scattering stride-2 samples into cells, which
leaves scan-line holes, (2) dilating and voting over 11 frames whose registration differs by up to ~1 m (front-only
clips), and (3) SAM3 masks ~2x coarser than the image.
  refine     inside SAM3 line / arrow-text / hatched masks keep only pixels brighter than the asphalt (white top-hat
             31 px > max(10, 2.5 x road MAD)); the rest of such a mask becomes drivable; crosswalk stays a region
  sample     every BEV cell (0.10 m, drawn 1:1) projects into each camera and reads the refined class -- no holes
  composite  BEV now: frames of the past 2 s, every camera; each cell keeps the class seen from the NEAREST camera range
             (a z-buffer, not a vote -> no smear); solid if that range <= 10 m, faded otherwise
  clip map   the same nearest-view rule accumulated over every frame so far (world frame, 0.25 m)
Classes: 1 drivable, 2 lane/road line, 3 crosswalk, 4 arrow/text, 5 non-drivable edge, 6 hatched area.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import sam3_paint as P
import camera_model as CM                                      # pinhole virtual views or native f-theta cameras
import ground_surface as GS                                    # smooth bilinear ground (no 2 m seams)

V2 = Path("/home/nvidia/qwendrive/v2")
COL = {1: (70, 120, 200), 2: (255, 214, 0), 3: (0, 220, 210), 4: (255, 60, 220), 5: (235, 40, 40), 6: (255, 140, 0), 7: (110, 190, 90)}
NAME = {1: "drivable road", 2: "lane / road line", 3: "crosswalk", 4: "arrow / text", 5: "non-drivable edge", 6: "hatched area", 7: "sidewalk / verge"}
ALPHA = {1: 0.30, 2: 0.9, 3: 0.75, 4: 0.9, 5: 0.9, 6: 0.8, 7: 0.40}
VIEWS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2",
         "CAM_FW", "CAM_CL", "CAM_CR", "CAM_RL", "CAM_RR", "CAM_RT", "CAM_FT"]                # virtual rig, then native cameras
BEV_RES, BX, BY = 0.10, (-20.0, 40.0), (-15.0, 15.0)
MAP_RES, MAP_R = 0.25, 35.0
NEAR_M, PAST, EDGE_MAX_M = 10.0, 10, 12.0
BG = np.array((22, 26, 24), np.float32)
THIN = (2, 4, 6)
TOPHAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))


def font(sz, bold=False):
    for n in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def refine(img, cls_small):
    cls = cv2.resize(cls_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    Y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    th = cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, TOPHAT)
    road = cls == 1
    mad = float(np.median(np.abs(th[road] - np.median(th[road])))) if road.any() else 3.0
    thin = np.isin(cls, THIN)
    cls[thin & (th <= max(10.0, 2.5 * mad))] = 1
    return cls


def sample(V, xy_rig):
    """(class, range) of rig-frame ground cells as seen by camera view V (range inf where not observed); the ground
    height is the smooth surface, the projection the view's own camera model."""
    Pk = np.c_[xy_rig, GS.height(xy_rig, V["sg"])]
    u, v, ok = V["C"].project_rig(Pk)
    cls = np.zeros(len(u), np.uint8)
    cls[ok] = V["ref"][v[ok].astype(int), u[ok].astype(int)]
    dist = np.hypot(Pk[:, 0] - V["C"].t[0], Pk[:, 1] - V["C"].t[1])
    cls[(cls == 5) & (dist > EDGE_MAX_M)] = 0            # a thin edge seen from far covers many cells: take edges only near
    rng = np.where(ok & (cls > 0), dist, np.inf)
    return cls, rng


def paint_bev(cls, rng):
    img = np.zeros(cls.shape + (3,), np.float32); img[:] = BG
    for k in (7, 1, 2, 4, 6, 3, 5):
        m = cls == k
        col = np.array(COL[k], np.float32)
        img[m & (rng <= NEAR_M)] = col
        img[m & (rng > NEAR_M)] = 0.45 * col + 0.55 * BG
    return img.clip(0, 255).astype(np.uint8)


def overlay(img, cls):
    a = img.astype(np.float32)
    for k in (7, 1, 2, 3, 4, 6, 5):
        m = cls == k
        a[m] = (1 - ALPHA[k]) * a[m] + ALPHA[k] * np.array(COL[k], np.float32)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def main():
    c8 = sys.argv[1]; npz_dir = Path(sys.argv[2]); out_dir = Path(sys.argv[3]); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]
    sd = V2 / f"seq_{c8}"
    toks = [str(f["tok"]) for f in frames]
    meta0 = json.loads((sd / toks[0] / "meta.json").read_text(encoding="utf-8"))
    sha, t0 = meta0["clip_sha12"], meta0["t_ref_us"]
    cams = [c for c in VIEWS if f"cls_{c}" in frames[0]]
    FRONT_CAM = "CAM_FW" if "CAM_FW" in cams else "CAM_F0"
    front_only = cams == [FRONT_CAM]
    SIDE = ("CAM_CL", "CAM_CR", "CAM_RL", "CAM_RR") if "CAM_CL" in cams else ("CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2")
    SIDE_NAME = {"CAM_L0": "front-left", "CAM_R0": "front-right", "CAM_L2": "rear-left", "CAM_R2": "rear-right",
                 "CAM_CL": "cross-left", "CAM_CR": "cross-right", "CAM_RL": "rear-left", "CAM_RR": "rear-right"}
    LEGEND = (1, 2, 3, 6, 4, 5, 7) if "pts_7" in frames[0] else (1, 2, 3, 6, 4, 5)     # v6 adds sidewalk / verge; older maps unchanged
    LEG_DX = 168 if len(LEGEND) == 7 else 197
    # ---- per frame, per view: calibration, ground, refined full-res class raster
    FV = []
    for n, f in enumerate(frames):
        fd = sd / toks[n]
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        lp = fd / "lidar.npy"
        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
        sg = GS.smooth_grid(grid, fb)
        views = []
        for cam in cams:
            i = fr["cam_order"].index(cam)
            img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
            views.append({"cam": cam, "C": CM.Camera.from_calib(c, i, img.shape[1], img.shape[0]), "sg": sg,
                          "ref": refine(img, f[f"cls_{cam}"])})
        FV.append(views)
    print(f"refined {len(frames)} frames x {len(cams)} views", flush=True)
    # ---- BEV grid (frame rig), rows = forward up, cols = left on the left
    xs = np.arange(BX[1] - BEV_RES / 2, BX[0], -BEV_RES); ys = np.arange(BY[1] - BEV_RES / 2, BY[0], -BEV_RES)
    GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]
    # ---- clip map (world)
    Ts = [f["T_world_rig"] for f in frames]; Pp = np.array([T[:2, 3] for T in Ts])
    mx0, my0 = Pp.min(axis=0) - 45; mx1, my1 = Pp.max(axis=0) + 45
    MW, MH = int((mx1 - mx0) / MAP_RES), int((my1 - my0) / MAP_RES)
    wi, wj = np.meshgrid(np.arange(MH), np.arange(MW), indexing="ij")
    WXY = np.c_[mx0 + (wj.ravel() + 0.5) * MAP_RES, my0 + (wi.ravel() + 0.5) * MAP_RES]
    map_cls = np.zeros(MH * MW, np.uint8); map_rng = np.full(MH * MW, np.inf)
    for n in range(len(frames)):
        f = frames[n]; Tn = f["T_world_rig"]
        # clip map update with frame n (cells within MAP_R of the rig)
        near = np.flatnonzero(np.hypot(WXY[:, 0] - Tn[0, 3], WXY[:, 1] - Tn[1, 3]) <= MAP_R)
        q = (np.c_[WXY[near], np.zeros(len(near)), np.ones(len(near))] @ np.linalg.inv(Tn).T)[:, :2]
        for V in FV[n]:
            c_, r_ = sample(V, q)
            take = r_ < map_rng[near]
            map_cls[near[take]] = c_[take]; map_rng[near[take]] = r_[take]
        # BEV now: nearest view over the past 2 s
        best_c = np.zeros(len(XY), np.uint8); best_r = np.full(len(XY), np.inf)
        for k in range(n, max(-1, n - PAST - 1), -1):
            rel = np.linalg.inv(Ts[k]) @ Tn
            qk = (np.c_[XY, np.zeros(len(XY)), np.ones(len(XY))] @ rel.T)[:, :2]
            for V in FV[k]:
                c_, r_ = sample(V, qk)
                take = r_ < best_r
                best_c[take] = c_[take]; best_r[take] = r_[take]
        bcls = best_c.reshape(GX.shape); brng = best_r.reshape(GX.shape)
        fd = sd / toks[n]
        meta = json.loads((fd / "meta.json").read_text(encoding="utf-8"))
        canvas = Image.new("RGB", (1920, 1080), (14, 17, 16)); d = ImageDraw.Draw(canvas)
        byc = {V["cam"]: V for V in FV[n]}
        f0img = np.asarray(Image.open(fd / "images" / f"{FRONT_CAM}.jpg").convert("RGB"))
        canvas.paste(overlay(f0img, byc[FRONT_CAM]["ref"]).resize((1180, 664)), (10, 64))
        d.text((18, 70), "front", fill=(255, 255, 255), font=font(18, True))
        if front_only:
            d.text((10, 742), "front camera only on this clip (no side cameras / LiDAR on disk): the ground comes from the ego's own path.",
                   fill=(200, 205, 202), font=font(16))
        else:
            for q_, cam in enumerate(SIDE):
                im = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
                canvas.paste(overlay(im, byc[cam]["ref"]).resize((290, 163)), (10 + q_ * 297, 740))
                d.text((16 + q_ * 297, 744), SIDE_NAME[cam],
                       fill=(255, 255, 255), font=font(15, True))
        canvas.paste(Image.fromarray(paint_bev(bcls, brng)), (1210, 64))
        ex, ey = 1210 + int(BY[1] / BEV_RES), 64 + int(BX[1] / BEV_RES)
        d.polygon([(ex, ey - 12), (ex - 8, ey + 8), (ex + 8, ey + 8)], fill=(255, 255, 255))
        d.text((1210, 668), "BEV now · nearest view, past 2 s · 0.1 m", fill=(200, 205, 202), font=font(15))
        mimg = paint_bev(map_cls.reshape(MH, MW), map_rng.reshape(MH, MW))[::-1].copy()
        mim = Image.fromarray(mimg); s = min(380 / mim.width, 600 / mim.height)
        mim = mim.resize((max(1, int(mim.width * s)), max(1, int(mim.height * s))), Image.LANCZOS if s < 1 else Image.NEAREST)
        ox, oy = 1530 + (380 - mim.width) // 2, 64 + (600 - mim.height) // 2
        d.rectangle([1530, 64, 1910, 664], fill=(22, 26, 24)); canvas.paste(mim, (ox, oy))
        for m2 in range(n + 1):
            px = ox + (Pp[m2, 0] - mx0) / MAP_RES * s; py = oy + (MH - (Pp[m2, 1] - my0) / MAP_RES) * s
            d.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=(255, 255, 255))
        d.text((1530, 672), "clip map so far · nearest view · world frame", fill=(200, 205, 202), font=font(15))
        yy = 700
        d.text((1210, yy), "in the current BEV (m², solid + faded)", fill=(240, 242, 241), font=font(17, True)); yy += 26
        for k in LEGEND:
            a_s = float(((bcls == k) & (brng <= NEAR_M)).sum()) * BEV_RES ** 2; a_t = float(((bcls == k) & (brng > NEAR_M)).sum()) * BEV_RES ** 2
            d.rectangle([1210, yy + 2, 1226, yy + 18], fill=COL[k]); d.text((1236, yy), NAME[k], fill=(230, 232, 231), font=font(16))
            d.text((1420, yy), f"{a_s:7.1f} + {a_t:6.1f}", fill=(230, 232, 231), font=font(16)); yy += 23
        ver = str(frames[0].get("ver", "v3"))
        label = {"v5": "v5 · native 120° camera · smooth ground", "v6": "v6 · all native cameras · smooth ground"}.get(ver, f"{ver} extraction · v4 BEV")
        d.text((10, 12), f"SAM3-only road map {label}  ·  clip {sha}  ·  t = {(meta['t_ref_us'] - t0) / 1e6:5.1f} s", fill=(240, 242, 241), font=font(26, True))
        d.text((10, 42), "paint kept only where brighter than the asphalt · BEV: every cell reads its pixel from the nearest camera view (no scatter, no vote)",
               fill=(170, 178, 174), font=font(14))
        for q_, k in enumerate(LEGEND):
            xx = 10 + q_ * LEG_DX
            d.rectangle([xx, 930, xx + 20, 950], fill=COL[k]); d.text((xx + 28, 930), NAME[k], fill=(230, 232, 231), font=font(16))
        d.text((10, 966), "BEV: solid = seen within 10 m of a camera · faded = only seen from farther · never uses future frames.", fill=(150, 158, 154), font=font(14))
        d.text((10, 988), "Geometry only (not semantics): calibration, ego poses, ground height (LiDAR, or the ego path on front-only clips).",
               fill=(150, 158, 154), font=font(14))
        canvas.save(out_dir / f"f{n:04d}.png")
    print("rendered", len(frames))


if __name__ == "__main__":
    main()
