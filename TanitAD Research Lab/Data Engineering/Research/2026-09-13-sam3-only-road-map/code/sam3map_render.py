"""Render the SAM3-only map video frames: camera pixel overlays + BEV map fused over +-1 s + the clip map so far.

Everything semantic comes from SAM3 (sam3map_extract.py); geometry only places it. Clip ids appear as sha12 only.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

V2 = Path(r"C:/Users/Admin/qwenvis/v2")
CLIPS = {"4fbd97b6a4b7": "4fbd97b6a4b7", "73495082f98b": "73495082f98b",
         "0d90d20036a3": "0d90d20036a3", "6924358fafe0": "6924358fafe0"}
COL = {1: (70, 120, 200), 2: (255, 214, 0), 3: (0, 220, 210), 4: (255, 60, 220), 5: (235, 40, 40)}
NAME = {1: "drivable road", 2: "lane / road line", 3: "crosswalk", 4: "arrow / text", 5: "non-drivable edge"}
ALPHA = {1: 0.35, 2: 0.85, 3: 0.75, 4: 0.85, 5: 0.9}
BEV_RES, BX, BY = 0.2, (-20.0, 40.0), (-15.0, 15.0)          # forward-biased current map: 60 m x 30 m
MAP_RES = 0.25
VER, NOTE = "", ""                                            # set per run from the npz directory name (v1 / v2)


def font(sz, bold=False):
    for n in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def overlay(img, cls):
    a = np.asarray(img.resize((cls.shape[1], cls.shape[0]))).astype(np.float32)
    for k in (1, 2, 3, 4, 5):
        m = cls == k
        a[m] = (1 - ALPHA[k]) * a[m] + ALPHA[k] * np.array(COL[k], np.float32)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def to_rig(xy_world, T):
    q = np.c_[xy_world, np.zeros(len(xy_world)), np.ones(len(xy_world))] @ np.linalg.inv(T).T
    return q[:, :2]


def raster_counts(pts_rig, res, bx, by):
    H, W = int(round((by[1] - by[0]) / res)), int(round((bx[1] - bx[0]) / res))
    m = np.zeros((H, W), bool)
    i = np.floor((pts_rig[:, 1] - by[0]) / res).astype(int); j = np.floor((pts_rig[:, 0] - bx[0]) / res).astype(int)
    k = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    m[i[k], j[k]] = True
    return m


def fused_classes(frames, j, half, res, bx, by, T):
    """Distinct-frame hit counts per class over j-half..j+half; drivable needs >= 3 frames, everything else >= 2."""
    H, W = int(round((by[1] - by[0]) / res)), int(round((bx[1] - bx[0]) / res))
    hits = {k: np.zeros((H, W), np.int16) for k in range(1, 6)}
    win = frames[max(0, j - half): j + half + 1]
    need_road, need_thin = max(1, int(np.ceil(0.3 * len(win)))), max(1, int(np.ceil(0.2 * len(win))))
    for f in win:
        for k in range(1, 6):
            p = f[f"pts_{k}"]
            if len(p):
                hits[k] += raster_counts(to_rig(p, T), res, bx, by)
    from scipy import ndimage as ndi
    cls = np.zeros((H, W), np.uint8)
    cls[ndi.binary_closing(hits[1] >= need_road, structure=np.ones((5, 5)))] = 1     # far-range sampling gaps
    for k in (2, 3, 4, 5):
        cls[ndi.binary_closing(hits[k] >= need_thin, structure=np.ones((3, 3)))] = k
    return cls


def bev_image(cls, scale, bg=(22, 26, 24)):
    """(Y, X) raster -> image with forward UP and left LEFT."""
    img = np.zeros(cls.shape + (3,), np.uint8); img[:] = bg
    for k in (1, 2, 3, 4, 5):
        img[cls == k] = COL[k]
    img = img.transpose(1, 0, 2)[::-1, ::-1]
    im = Image.fromarray(np.ascontiguousarray(img))
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def main():
    c8 = sys.argv[1]
    npz_dir = Path(sys.argv[2]); out_dir = Path(sys.argv[3]); out_dir.mkdir(parents=True, exist_ok=True)
    sel = sys.argv[4] if len(sys.argv) > 4 else "all"
    global VER, NOTE
    if npz_dir.name.endswith("_v2"):
        VER, NOTE = " v2", "v2: ego bonnet removed by a SAM3 'car hood' mask, the ego-attached patch removed, edges kept only on the road boundary."
    else:
        VER, NOTE = "", "Known artefact: SAM3's road mask includes parts of the ego bonnet."
    sha = hashlib.sha256(CLIPS[c8].encode()).hexdigest()[:12]
    files = sorted(npz_dir.glob("*.npz"))
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]
    idx = [int(f.stem) for f in files]
    toks = [str(f["tok"]) for f in frames]
    t0 = json.loads((V2 / f"seq_{c8}" / toks[0] / "meta.json").read_text(encoding="utf-8"))["t_ref_us"]
    # clip map extent in world frame, from the ego path
    Ts = [f["T_world_rig"] for f in frames]
    P = np.array([T[:2, 3] for T in Ts])
    mx0, my0 = P.min(axis=0) - 45; mx1, my1 = P.max(axis=0) + 45
    MW, MH = int((mx1 - mx0) / MAP_RES), int((my1 - my0) / MAP_RES)
    map_hits = {k: np.zeros((MH, MW), np.int16) for k in range(1, 6)}
    todo = range(len(frames)) if sel == "all" else [idx.index(int(x)) for x in sel.split(",")]
    for n in range(len(frames)):
        f = frames[n]
        for k in range(1, 6):                                   # clip map accumulates EVERY frame up to now
            p = f[f"pts_{k}"]
            if len(p):
                i = np.floor((p[:, 1] - my0) / MAP_RES).astype(int); jj = np.floor((p[:, 0] - mx0) / MAP_RES).astype(int)
                ok = (i >= 0) & (i < MH) & (jj >= 0) & (jj < MW)
                sub = np.zeros((MH, MW), bool); sub[i[ok], jj[ok]] = True
                map_hits[k] += sub
        if n not in todo:
            continue
        tok = toks[n]; fd = V2 / f"seq_{c8}" / tok
        meta = json.loads((fd / "meta.json").read_text(encoding="utf-8"))
        canvas = Image.new("RGB", (1920, 1080), (14, 17, 16))
        d = ImageDraw.Draw(canvas)
        # ---- cameras: F0 large, four small
        cams = {c: overlay(Image.open(fd / "images" / f"{c}.jpg").convert("RGB"), f[f"cls_{c}"]) for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2")}
        canvas.paste(cams["CAM_F0"].resize((1180, 664)), (10, 64))
        for q, c in enumerate(("CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2")):
            canvas.paste(cams[c].resize((290, 163)), (10 + q * 297, 740))
            d.text((16 + q * 297, 744), {"CAM_L0": "front-left", "CAM_R0": "front-right", "CAM_L2": "rear-left", "CAM_R2": "rear-right"}[c],
                   fill=(255, 255, 255), font=font(15, True))
        d.text((18, 70), "front", fill=(255, 255, 255), font=font(18, True))
        # ---- BEV now (fused +-1 s), forward-biased window
        T = f["T_world_rig"]
        cls = fused_classes(frames, n, 5, BEV_RES, BX, BY, T)
        bev = bev_image(cls, 2)                                  # 150 x 300 cells -> 300 x 600 px, 10 px/m
        canvas.paste(bev, (1210, 64))
        ex, ey = 1210 + int((BY[1] - 0) / BEV_RES) * 2, 64 + int((BX[1] - 0) / BEV_RES) * 2
        d.polygon([(ex, ey - 12), (ex - 8, ey + 8), (ex + 8, ey + 8)], fill=(255, 255, 255))
        d.text((1210, 668), "BEV now · fused ±1 s · 30 × 60 m", fill=(200, 205, 202), font=font(15))
        # ---- clip map so far (world frame), fitted into 380 x 600, ego trail
        mcls = np.zeros((MH, MW), np.uint8)
        mcls[map_hits[1] >= 3] = 1
        for k in (2, 3, 4, 5):
            mcls[map_hits[k] >= 2] = k
        mimg = np.zeros((MH, MW, 3), np.uint8); mimg[:] = (22, 26, 24)
        for k in (1, 2, 3, 4, 5):
            mimg[mcls == k] = COL[k]
        mim = Image.fromarray(mimg[::-1].copy())
        s = min(380 / mim.width, 600 / mim.height)
        mim = mim.resize((max(1, int(mim.width * s)), max(1, int(mim.height * s))), Image.NEAREST)
        ox, oy = 1530 + (380 - mim.width) // 2, 64 + (600 - mim.height) // 2
        d.rectangle([1530, 64, 1910, 664], fill=(22, 26, 24))
        canvas.paste(mim, (ox, oy))
        for m2 in range(n + 1):
            px = ox + (P[m2, 0] - mx0) / MAP_RES * s; py = oy + (MH - (P[m2, 1] - my0) / MAP_RES) * s
            d.ellipse([px - 1.6, py - 1.6, px + 1.6, py + 1.6], fill=(255, 255, 255))
        d.text((1530, 672), "clip map so far · world frame · 0.25 m", fill=(200, 205, 202), font=font(15))
        # ---- stats panel: areas in the current BEV and accepted SAM3 instances
        area = {k: float((cls == k).sum()) * BEV_RES * BEV_RES for k in (1, 2, 3, 4, 5)}
        st = json.loads(str(f["stats"]))
        refine = st.pop("_refine_v2", None)
        acc = {}
        for cam_st in st.values():
            for p_, v_ in cam_st["accepted"].items():
                acc[p_] = acc.get(p_, 0) + v_
        rej = sum(cam_st["rejected_offroad"] for cam_st in st.values())
        zc = sum(cam_st["rejected_zebra_cap"] for cam_st in st.values())
        yy = 700
        d.text((1210, yy), "in the current BEV (m²)", fill=(240, 242, 241), font=font(17, True)); yy += 26
        for k in (1, 2, 3, 4, 5):
            d.rectangle([1210, yy + 2, 1226, yy + 18], fill=COL[k])
            d.text((1236, yy), f"{NAME[k]}", fill=(230, 232, 231), font=font(16))
            d.text((1440, yy), f"{area[k]:8.1f}", fill=(230, 232, 231), font=font(16)); yy += 23
        yy += 10
        d.text((1210, yy), "SAM3 instances accepted this frame (7 views)", fill=(240, 242, 241), font=font(17, True)); yy += 28
        for p_ in ("lane marking", "road marking", "stop line", "crosswalk", "zebra crossing", "arrow painted on road", "text painted on road"):
            d.text((1236, yy), p_, fill=(210, 214, 212), font=font(15))
            d.text((1440, yy), f"{acc.get(p_, 0):4d}", fill=(210, 214, 212), font=font(15)); yy += 20
        d.text((1236, yy + 4), f"rejected off-road {rej} · crosswalk area cap {zc}", fill=(170, 178, 174), font=font(15))
        if refine is not None:
            r_ = {q: sum(v[q] for v in refine.values()) for q in ("R1_px", "R2_components", "R3_px")}
            d.text((1236, yy + 26), f"v2 removed: bonnet {r_['R1_px']} px · ego patch {r_['R2_components']} · off-boundary edge {r_['R3_px']} px",
                   fill=(170, 178, 174), font=font(15))
        # ---- header + legend
        d.text((10, 12), f"SAM3-only road map{VER}  ·  clip {sha}  ·  t = {(meta['t_ref_us'] - t0) / 1e6:5.1f} s", fill=(240, 242, 241), font=font(26, True))
        d.text((10, 42), "prompts: road · sidewalk/grass/curb/guardrail · lane & road marking · stop line · crosswalk · arrow/text on road   |   "
                         "markings kept only on the road; edges = road boundary touching non-drivable", fill=(170, 178, 174), font=font(14))
        for q, k in enumerate((1, 2, 3, 4, 5)):
            xx = 10 + q * 236
            d.rectangle([xx, 930, xx + 20, 950], fill=COL[k]); d.text((xx + 28, 930), NAME[k], fill=(230, 232, 231), font=font(17))
        d.text((10, 992), NOTE, fill=(150, 158, 154), font=font(14))
        d.text((10, 970), "Geometry only (not semantics): calibration, ego poses, LiDAR ground height. Virtual nuPlan-rig views of PhysicalAI cameras.",
               fill=(150, 158, 154), font=font(14))
        canvas.save(out_dir / f"f{n:04d}.png")
    print("rendered", len(list(todo)))


if __name__ == "__main__":
    main()
