"""BEV v4 prototype on one frame: inverse sampling + paint-pixel refinement + nearest-view compositing.

Why (ipm_diag.py on the PI's example, front clip, t = 10 s): the image warped onto the ground shows the arrows
clearly (projection is right), but (1) forward-scattered stride-2 samples leave scan-line holes, (2) the front-only
fusion dilated and unioned 11 misregistered frames into blobs, (3) SAM3's masks are ~2x coarser than the image.
  refine    inside a SAM3 paint mask (line / arrow-text / hatched) keep only pixels brighter than the local asphalt:
            white top-hat of the luminance (31 px), above max(10, 2.5 x the road's median absolute deviation); the rest
            of the mask falls back to drivable. Crosswalk stays a region.
  sample    every BEV cell (0.05 m) projects into the image and reads its class -> no holes
  composite over the past 2 s, each cell takes the class seen from the NEAREST camera range (a z-buffer), not a vote
Panels: B (current single-frame class warp) | G (refined single frame) | H (refined, nearest-view composite over 2 s)
| E_new (same as H at the render's 0.20 m).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ipm_diag import load, ground_z, cell_centres, COL

PAINT_THIN = (2, 4, 6)


def refine(F):
    """Full-res refined class raster from the stored 960x540 raster + the image."""
    cls = cv2.resize(F["cls"], (1920, 1080), interpolation=cv2.INTER_NEAREST)
    Y = cv2.cvtColor(F["img"], cv2.COLOR_RGB2GRAY).astype(np.float32)
    th = cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    road = cls == 1
    mad = float(np.median(np.abs(th[road] - np.median(th[road])))) if road.any() else 3.0
    bright = th > max(10.0, 2.5 * mad)
    out = cls.copy()
    thin = np.isin(cls, PAINT_THIN)
    out[thin & ~bright] = 1
    return out


def sample_frame(F, XYj, Tj, refined):
    """Class + camera range for the frame-j BEV cells as seen from frame F."""
    rel = np.linalg.inv(F["T"]) @ Tj
    XYk = (np.c_[XYj.reshape(-1, 2), np.zeros(XYj.size // 2), np.ones(XYj.size // 2)] @ rel.T)[:, :2]
    Pk = np.c_[XYk, ground_z(XYk, F["grid"], F["fb"])]
    cc = (Pk - F["t"]) @ F["R"]
    z = cc[:, 2]
    good = z > 0.5
    u = np.where(good, F["K"][0, 0] * cc[:, 0] / np.where(good, z, 1) + F["K"][0, 2], -1)
    v = np.where(good, F["K"][1, 1] * cc[:, 1] / np.where(good, z, 1) + F["K"][1, 2], -1)
    ok = good & (u >= 0) & (u < 1920) & (v >= 0) & (v < 1080)
    cls = np.zeros(len(u), np.uint8)
    cls[ok] = refined[v[ok].astype(int), u[ok].astype(int)]
    rng = np.where(ok & (cls > 0), np.hypot(Pk[:, 0] - F["t"][0], Pk[:, 1] - F["t"][1]), np.inf)
    return cls.reshape(XYj.shape[:2]), rng.reshape(XYj.shape[:2])


def colour(cls):
    img = np.zeros(cls.shape + (3,), np.uint8); img[:] = (22, 26, 24)
    for k in (1, 2, 4, 6, 3, 5):
        img[cls == k] = COL[k]
    return img


def main(root, npz_dir, c8, j, cam, x0, x1, y0, y1, out_png):
    Fj = load(root, npz_dir, c8, j, cam)
    panels = []
    for res, tag in ((0.05, "0.05 m"), (0.20, "0.20 m")):
        XY = cell_centres(x0, x1, y0, y1, res)
        best_cls = np.zeros(XY.shape[:2], np.uint8); best_rng = np.full(XY.shape[:2], np.inf)
        single_raw = single_ref = None
        for k in range(j, max(-1, j - 11), -1):
            Fk = Fj if k == j else load(root, npz_dir, c8, k, cam)
            ref = refine(Fk)
            c, r = sample_frame(Fk, XY, Fj["T"], ref)
            if k == j and res == 0.05:
                raw = cv2.resize(Fk["cls"], (1920, 1080), interpolation=cv2.INTER_NEAREST)
                single_raw, _ = sample_frame(Fk, XY, Fj["T"], raw)
                single_ref = c
            take = r < best_rng
            best_cls[take] = c[take]; best_rng[take] = r[take]
        if res == 0.05:
            panels += [("B  current class warp, this frame", colour(single_raw)), ("G  refined paint, this frame", colour(single_ref)),
                       ("H  refined, nearest-view composite, past 2 s", colour(best_cls))]
        else:
            H0 = panels[0][1].shape
            panels.append(("E_new  same composite at 0.20 m", cv2.resize(colour(best_cls), (H0[1], H0[0]), interpolation=cv2.INTER_NEAREST)))
    wpan, hpan = panels[0][1].shape[1], panels[0][1].shape[0]
    sheet = Image.new("RGB", (len(panels) * (wpan + 10), hpan + 40), (14, 17, 16)); dr = ImageDraw.Draw(sheet)
    fnt = ImageFont.truetype("DejaVuSans.ttf", 13)
    for q, (name, im) in enumerate(panels):
        sheet.paste(Image.fromarray(im), (q * (wpan + 10), 36)); dr.text((q * (wpan + 10) + 4, 8), name, fill=(230, 232, 231), font=fnt)
    sheet.save(out_png)
    print("saved", out_png)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]), a[4], float(a[5]), float(a[6]), float(a[7]), float(a[8]), a[9])
