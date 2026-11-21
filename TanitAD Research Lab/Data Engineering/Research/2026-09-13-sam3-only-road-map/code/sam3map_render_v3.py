"""Render the v3 SAM3-only map video: camera overlays + a CAUSAL, RANGE-AWARE BEV + the clip map so far.

Why the fusion changed (measured 2026-09-13, align_bias2.py / align_class.py): the lifting geometry is self-consistent
(median signed offset -0.001 m at 8-16 m), but far SAM3 masks spill -- at night only 54-65 % of crosswalk points seen
from 8-16 m land within 0.5 m of the same crosswalk seen from < 8 m (day 91-93 %). The v1/v2 BEV fused +-1 s of frames
at every range, so it drew far spills metres off, and showed paint the front image had not seen yet.
  window     BEV now = the past 2 s + the current frame (never the future); clip map = every frame so far
  weights    a hit observed within 10 m of its camera counts 1, from 10-25 m counts 0.35, beyond 25 m 0
  classes    per cell the paint class with the highest weighted score (ties: crosswalk > hatched > arrow/text > line)
  drawing    SOLID when the winning class has at least one hit from within 10 m, FADED (tentative) otherwise
Classes: 1 drivable, 2 lane/road line, 3 crosswalk, 4 arrow/text, 5 non-drivable edge, 6 hatched area.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

V2 = Path(r"C:/Users/Admin/qwenvis/v2")
CLIPS = {}                                                     # c8 -> full clip id, filled from meta.json at run time
COL = {1: (70, 120, 200), 2: (255, 214, 0), 3: (0, 220, 210), 4: (255, 60, 220), 5: (235, 40, 40), 6: (255, 140, 0)}
NAME = {1: "drivable road", 2: "lane / road line", 3: "crosswalk", 4: "arrow / text", 5: "non-drivable edge", 6: "hatched area"}
ALPHA = {1: 0.35, 2: 0.85, 3: 0.75, 4: 0.85, 5: 0.9, 6: 0.75}
BEV_RES, BX, BY = 0.2, (-20.0, 40.0), (-15.0, 15.0)
MAP_RES = 0.25
NEAR_M, FAR_M, FAR_W = 10.0, 25.0, 0.35
PAINT_ORDER = (3, 6, 4, 2)                                     # tie-break priority, strongest first
PAST = 10                                                      # frames = 2 s at 5 Hz
BG = (22, 26, 24)


def font(sz, bold=False):
    for n in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def overlay(img, cls):
    a = np.asarray(img.resize((cls.shape[1], cls.shape[0]))).astype(np.float32)
    for k in (1, 2, 3, 4, 6, 5):
        m = cls == k
        a[m] = (1 - ALPHA[k]) * a[m] + ALPHA[k] * np.array(COL[k], np.float32)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def to_frame(xy_world, Tinv):
    return (np.c_[xy_world, np.zeros(len(xy_world)), np.ones(len(xy_world))] @ Tinv.T)[:, :2]


def grid_index(q, res, x0, y0, H, W, dilate=False):
    i = np.floor((q[:, 1] - y0) / res).astype(np.int64); j = np.floor((q[:, 0] - x0) / res).astype(np.int64)
    if dilate:                                                  # 1-cell tolerance to frame-to-frame misregistration
        off = np.array([-1, 0, 1])
        i = (i[:, None, None] + off[None, :, None] + 0 * off[None, None, :]).ravel()
        j = (j[:, None, None] + 0 * off[None, :, None] + off[None, None, :]).ravel()
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    return np.unique(i[ok] * W + j[ok])


def accumulate(frames, sel, Tinv, res, x0, y0, H, W, dilate_paint=False):
    near = {k: np.zeros(H * W, np.int16) for k in range(1, 7)}; far = {k: np.zeros(H * W, np.int16) for k in range(1, 7)}
    for n in sel:
        f = frames[n]
        for k in range(1, 7):
            p = f.get(f"pts_{k}")
            if p is None or not len(p):
                continue
            r = f[f"rng_{k}"].astype(np.float32)
            q = to_frame(p, Tinv)
            for store, m in ((near, r <= NEAR_M), (far, (r > NEAR_M) & (r <= FAR_M))):
                if m.any():
                    store[k][grid_index(q[m], res, x0, y0, H, W, dilate=dilate_paint and k != 1)] += 1
    return near, far


def decide(near, far, H, W, need_road, need_thin):
    """-> (cls, solid) rasters."""
    cls = np.zeros(H * W, np.uint8); solid = np.zeros(H * W, bool)
    road = (near[1] + far[1]) >= need_road
    cls[road] = 1; solid[road] = True
    score = np.stack([near[k] + FAR_W * far[k] for k in PAINT_ORDER])             # (4, H*W)
    best = np.argmax(score, axis=0)                                                # first max wins -> priority order
    top = score[best, np.arange(H * W)]
    paint = top >= need_thin
    kbest = np.array(PAINT_ORDER)[best]
    cls[paint] = kbest[paint]
    near_best = np.stack([near[k] for k in PAINT_ORDER])[best, np.arange(H * W)]
    solid[paint] = near_best[paint] >= 1
    edge = (near[5] + FAR_W * far[5]) >= need_thin
    cls[edge] = 5; solid[edge] = near[5][edge] >= 1
    cls = cls.reshape(H, W); solid = solid.reshape(H, W)
    road2 = ndi.binary_closing(cls == 1, structure=np.ones((5, 5))) & (cls == 0)
    cls[road2] = 1; solid[road2] = True
    return cls, solid


def paint_image(cls, solid):
    img = np.zeros(cls.shape + (3,), np.float32); img[:] = BG
    for k in (1, 2, 3, 4, 6, 5):
        m = cls == k
        col = np.array(COL[k], np.float32)
        img[m & solid] = col
        img[m & ~solid] = 0.45 * col + 0.55 * np.array(BG, np.float32)
    return img.clip(0, 255).astype(np.uint8)


def main():
    c8 = sys.argv[1]
    npz_dir = Path(sys.argv[2]); out_dir = Path(sys.argv[3]); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]
    toks = [str(f["tok"]) for f in frames]
    meta0 = json.loads((V2 / f"seq_{c8}" / toks[0] / "meta.json").read_text(encoding="utf-8"))
    sha, t0 = meta0["clip_sha12"], meta0["t_ref_us"]
    Ts = [f["T_world_rig"] for f in frames]
    P = np.array([T[:2, 3] for T in Ts])
    mx0, my0 = P.min(axis=0) - 45; mx1, my1 = P.max(axis=0) + 45
    MW, MH = int((mx1 - mx0) / MAP_RES), int((my1 - my0) / MAP_RES)
    BH, BW = int(round((BY[1] - BY[0]) / BEV_RES)), int(round((BX[1] - BX[0]) / BEV_RES))
    I4 = np.eye(4)
    FRONT = "cls_CAM_L0" not in frames[0]
    # front camera only: near paint is visible for 2-3 frames before it passes under the bonnet, so the rig's
    # "20 % of the window" vote starved the BEV; one near observation (1-cell tolerance) is enough there
    THIN_FRAC, MAP_THIN, DIL = (0.0, 1.0, True) if FRONT else (0.2, 2.0, False)
    cn = {k: np.zeros(MH * MW, np.int16) for k in range(1, 7)}; cf = {k: np.zeros(MH * MW, np.int16) for k in range(1, 7)}
    for n in range(len(frames)):
        f = frames[n]
        nn, ff = accumulate(frames, [n], I4, MAP_RES, mx0, my0, MH, MW, DIL)        # world frame, this frame only
        for k in range(1, 7):
            cn[k] += nn[k]; cf[k] += ff[k]
        fd = V2 / f"seq_{c8}" / toks[n]
        meta = json.loads((fd / "meta.json").read_text(encoding="utf-8"))
        canvas = Image.new("RGB", (1920, 1080), (14, 17, 16)); d = ImageDraw.Draw(canvas)
        front_only = "cls_CAM_L0" not in f
        side = () if front_only else ("CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2")
        cams = {c: overlay(Image.open(fd / "images" / f"{c}.jpg").convert("RGB"), f[f"cls_{c}"]) for c in ("CAM_F0",) + side}
        if front_only:
            canvas.paste(cams["CAM_F0"].resize((1180, 664)), (10, 64))
            d.text((10, 742), "front camera only on this clip (no side cameras / LiDAR on disk): the ground comes from the ego's own path,",
                   fill=(200, 205, 202), font=font(16))
            d.text((10, 764), "so the BEV covers what the front camera sees.", fill=(200, 205, 202), font=font(16))
        else:
            canvas.paste(cams["CAM_F0"].resize((1180, 664)), (10, 64))
        for q, c in enumerate(side):
            canvas.paste(cams[c].resize((290, 163)), (10 + q * 297, 740))
            d.text((16 + q * 297, 744), {"CAM_L0": "front-left", "CAM_R0": "front-right", "CAM_L2": "rear-left", "CAM_R2": "rear-right"}[c],
                   fill=(255, 255, 255), font=font(15, True))
        d.text((18, 70), "front", fill=(255, 255, 255), font=font(18, True))
        # ---- BEV now: causal, range-aware
        sel = list(range(max(0, n - PAST), n + 1))
        near, far = accumulate(frames, sel, np.linalg.inv(f["T_world_rig"]), BEV_RES, BX[0], BY[0], BH, BW, DIL)
        cls, solid = decide(near, far, BH, BW, max(1, int(np.ceil(0.3 * len(sel)))), max(1.0, THIN_FRAC * len(sel)))
        img = paint_image(cls, solid).transpose(1, 0, 2)[::-1, ::-1]
        bev = Image.fromarray(np.ascontiguousarray(img)); bev = bev.resize((bev.width * 2, bev.height * 2), Image.NEAREST)
        canvas.paste(bev, (1210, 64))
        ex, ey = 1210 + int((BY[1] - 0) / BEV_RES) * 2, 64 + int((BX[1] - 0) / BEV_RES) * 2
        d.polygon([(ex, ey - 12), (ex - 8, ey + 8), (ex + 8, ey + 8)], fill=(255, 255, 255))
        d.text((1210, 668), "BEV now · past 2 s · 30 × 60 m", fill=(200, 205, 202), font=font(15))
        # ---- clip map so far
        mcls, msolid = decide(cn, cf, MH, MW, 3, MAP_THIN)
        mim = Image.fromarray(paint_image(mcls, msolid)[::-1].copy())
        s = min(380 / mim.width, 600 / mim.height)
        mim = mim.resize((max(1, int(mim.width * s)), max(1, int(mim.height * s))), Image.NEAREST)
        ox, oy = 1530 + (380 - mim.width) // 2, 64 + (600 - mim.height) // 2
        d.rectangle([1530, 64, 1910, 664], fill=BG); canvas.paste(mim, (ox, oy))
        for m2 in range(n + 1):
            px = ox + (P[m2, 0] - mx0) / MAP_RES * s; py = oy + (MH - (P[m2, 1] - my0) / MAP_RES) * s
            d.ellipse([px - 1.6, py - 1.6, px + 1.6, py + 1.6], fill=(255, 255, 255))
        d.text((1530, 672), "clip map so far · world frame", fill=(200, 205, 202), font=font(15))
        # ---- stats
        st = json.loads(str(f["stats"])); refine = st.pop("_refine_v2", None)
        acc, x2h, symdrop = {}, 0, 0
        for cam_st in st.values():
            for p_, v_ in cam_st["accepted"].items():
                acc[p_] = acc.get(p_, 0) + v_
            x2h += cam_st.get("crosswalk_to_hatched", 0) + cam_st.get("hatched_standalone", 0)
            symdrop += cam_st.get("symbol_dropped_dashed_solid", 0) + cam_st.get("symbol_dropped_footprint", 0)
        yy = 700
        d.text((1210, yy), "in the current BEV (m², solid + faded)", fill=(240, 242, 241), font=font(17, True)); yy += 26
        for k in (1, 2, 3, 6, 4, 5):
            a_s = float(((cls == k) & solid).sum()) * BEV_RES ** 2; a_t = float(((cls == k) & ~solid).sum()) * BEV_RES ** 2
            d.rectangle([1210, yy + 2, 1226, yy + 18], fill=COL[k]); d.text((1236, yy), NAME[k], fill=(230, 232, 231), font=font(16))
            d.text((1420, yy), f"{a_s:7.1f} + {a_t:6.1f}", fill=(230, 232, 231), font=font(16)); yy += 23
        yy += 8
        d.text((1210, yy), f"SAM3 decisions this frame ({len(st)} view{'s' if len(st) > 1 else ''})", fill=(240, 242, 241), font=font(17, True)); yy += 26
        for p_ in ("lane marking", "road marking", "stop line", "crosswalk", "zebra crossing", "pedestrian crossing", "arrow painted on road", "text painted on road"):
            d.text((1236, yy), p_, fill=(210, 214, 212), font=font(15)); d.text((1440, yy), f"{acc.get(p_, 0):4d}", fill=(210, 214, 212), font=font(15)); yy += 19
        d.text((1236, yy + 4), f"crosswalk → hatched area {x2h} · arrow/text dropped as dash/line {symdrop}", fill=(170, 178, 174), font=font(15))
        # ---- header, legend, notes
        d.text((10, 12), f"SAM3-only road map v3  ·  clip {sha}  ·  t = {(meta['t_ref_us'] - t0) / 1e6:5.1f} s", fill=(240, 242, 241), font=font(26, True))
        d.text((10, 42), "crosswalk needs 3+ stripes · stripes under diagonal-stripe evidence → hatched area · arrow/text dropped when a dashed/solid line explains it",
               fill=(170, 178, 174), font=font(14))
        for q, k in enumerate((1, 2, 3, 6, 4, 5)):
            xx = 10 + q * 197
            d.rectangle([xx, 930, xx + 20, 950], fill=COL[k]); d.text((xx + 28, 930), NAME[k], fill=(230, 232, 231), font=font(16))
        d.text((10, 966), "BEV: solid = seen within 10 m of a camera · faded = only seen from farther (tentative) · the BEV never uses future frames.",
               fill=(150, 158, 154), font=font(14))
        d.text((10, 988), "Geometry only (not semantics): calibration, ego poses, LiDAR ground height · ego bonnet and ego-attached artefacts removed.",
               fill=(150, 158, 154), font=font(14))
        canvas.save(out_dir / f"f{n:04d}.png")
    print("rendered", len(frames))


if __name__ == "__main__":
    main()
