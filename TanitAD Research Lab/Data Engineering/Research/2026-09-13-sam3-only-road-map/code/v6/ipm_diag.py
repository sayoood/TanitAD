"""Why are painted arrows unrecognisable in the BEV although SAM3 segments them well? Check each stage on ONE frame.

Panels (forward up, left left, same ground window):
  A  RGB warped onto the ground by INVERSE sampling (each BEV cell -> its image pixel, bilinear): if the painted arrow
     is an arrow here, calibration + ground height are right
  B  the class raster sampled the same way (nearest) over panel A
  C  the CURRENT forward path: stride-2 pixel samples lifted by P.lift and binned into 0.20 m cells (one frame)
  D  forward path binned at 0.05 m
  E  the render's front-only fusion: 0.20 m cells, 1-cell dilation, union over the past 2 s (11 frames)
  F  RGB inverse-warped from the frame 1 s EARLIER into this frame through the poses, blended 50/50 with A (ghosting =
     pose / ground inconsistency between frames)
Plus a numeric round trip: ground points projected into the camera and lifted back by P.lift.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import sam3_paint as P

COL = {1: (70, 120, 200), 2: (255, 214, 0), 3: (0, 220, 210), 4: (255, 60, 220), 5: (235, 40, 40), 6: (255, 140, 0)}


def load(root, npz_dir, c8, j, cam):
    sd = Path(root) / f"seq_{c8}"
    toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    fd = sd / toks[j]
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    i = fr["cam_order"].index(cam)
    K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
    grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64))
    img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
    d = np.load(Path(npz_dir) / f"{j:03d}.npz", allow_pickle=True)
    return dict(K=K, R=R, t=t, grid=grid, fb=fb, img=img, cls=d[f"cls_{cam}"], T=d["T_world_rig"],
                pts={k: d[f"pts_{k}"] for k in range(1, 7) if f"pts_{k}" in d})


def ground_z(xy, grid, fb):
    ci = np.clip(np.floor((xy[..., 0] + 50) / 2).astype(int), 0, 49); cj = np.clip(np.floor((xy[..., 1] + 50) / 2).astype(int), 0, 49)
    g = grid[ci * 50 + cj]
    return np.where(np.isfinite(g), g, fb)


def cell_centres(x0, x1, y0, y1, res):
    xs = np.arange(x1 - res / 2, x0, -res)            # rows: forward UP
    ys = np.arange(y1 - res / 2, y0, -res)            # cols: left on the LEFT
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X, Y], axis=-1)


def project(F, Pr):
    cc = (Pr - F["t"]) @ F["R"]
    z = np.where(cc[..., 2] > 0.5, cc[..., 2], np.nan)
    u = F["K"][0, 0] * cc[..., 0] / z + F["K"][0, 2]; v = F["K"][1, 1] * cc[..., 1] / z + F["K"][1, 2]
    return u, v


def inverse(F, XY):
    Pr = np.concatenate([XY, ground_z(XY, F["grid"], F["fb"])[..., None]], axis=-1)
    u, v = project(F, Pr)
    ok = np.isfinite(u) & (u >= 0) & (u < 1920) & (v >= 0) & (v < 1080)
    mu, mv = np.where(ok, u, -1).astype(np.float32), np.where(ok, v, -1).astype(np.float32)
    rgb = cv2.remap(F["img"], mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cls = cv2.remap(F["cls"], mu / 2, mv / 2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rgb, cls, ok


def paint_over(rgb, cls, alpha=0.6):
    a = rgb.astype(np.float32).copy()
    for k in (2, 3, 4, 6, 5):
        m = cls == k
        a[m] = (1 - alpha) * a[m] + alpha * np.array(COL[k], np.float32)
    return a.clip(0, 255).astype(np.uint8)


def forward_bins(xy_rig_by_class, x0, x1, y0, y1, res, dilate=False):
    H, W = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    out = np.zeros((H, W), np.uint8)
    for k in (1, 2, 4, 6, 3, 5):
        p = xy_rig_by_class.get(k)
        if p is None or not len(p):
            continue
        r = np.floor((x1 - p[:, 0]) / res).astype(int); c = np.floor((y1 - p[:, 1]) / res).astype(int)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        m = np.zeros((H, W), np.uint8); m[r[ok], c[ok]] = 1
        if dilate and k != 1:
            m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        out[m > 0] = k
    img = np.zeros((H, W, 3), np.uint8); img[:] = (22, 26, 24)
    for k, col in COL.items():
        img[out == k] = col
    return img


def main(root, npz_dir, c8, j, cam, x0, x1, y0, y1, out_png):
    F = load(root, npz_dir, c8, j, cam)
    # ---- round trip: ground points -> pixels -> P.lift
    XY = cell_centres(x0, x1, y0, y1, 0.5).reshape(-1, 2)
    Pr = np.c_[XY, ground_z(XY, F["grid"], F["fb"])]
    u, v = project(F, Pr[None])
    u, v = u[0], v[0]
    ok = np.isfinite(u) & (u >= 1) & (u < 1919) & (v >= 1) & (v < 1079)
    errs = []
    for (uu, vv, p) in zip(u[ok][:400], v[ok][:400], Pr[ok][:400]):
        m = np.zeros((1080, 1920), bool)
        iu, iv = int(uu // 2 * 2), int(vv // 2 * 2)                 # the stride-2 sample P.lift will take
        m[iv, iu] = True
        xy = P.lift(m, F["K"], F["R"], F["t"], F["grid"], F["fb"], stride=2)
        if len(xy):
            # expected: the ground point of that sample's pixel centre (iu+1, iv+1), not of (uu, vv)
            errs.append(np.hypot(*(xy[0] - p[:2])))
    errs = np.array(errs)
    print(f"round trip ({cam}): n {len(errs)}, median {np.median(errs):.3f} m, p90 {np.percentile(errs, 90):.3f} m "
          f"(includes the <= 1 px quantisation of the stride-2 sample)")
    # ---- panels
    res_i = 0.05
    XYi = cell_centres(x0, x1, y0, y1, res_i)
    rgbA, clsA, okA = inverse(F, XYi)
    A = rgbA; B = paint_over(rgbA, clsA)
    Tinv = np.linalg.inv(F["T"])
    rig = {k: (np.c_[p, np.zeros(len(p)), np.ones(len(p))] @ Tinv.T)[:, :2] for k, p in F["pts"].items()}
    C = forward_bins(rig, x0, x1, y0, y1, 0.20)
    D = forward_bins(rig, x0, x1, y0, y1, 0.05)
    npz = sorted(Path(npz_dir).glob("[0-9][0-9][0-9].npz"))
    union = {}
    for jj in range(max(0, j - 10), j + 1):
        dd = np.load(npz[jj], allow_pickle=True)
        for k in range(1, 7):
            if f"pts_{k}" in dd and len(dd[f"pts_{k}"]):
                q = (np.c_[dd[f"pts_{k}"], np.zeros(len(dd[f"pts_{k}"])), np.ones(len(dd[f"pts_{k}"]))] @ Tinv.T)[:, :2]
                union.setdefault(k, []).append(q)
    E = forward_bins({k: np.concatenate(v) for k, v in union.items()}, x0, x1, y0, y1, 0.20, dilate=True)
    Fe = load(root, npz_dir, c8, max(0, j - 5), cam)
    Tk = Fe["T"]; rel = np.linalg.inv(Tk) @ F["T"]                 # frame-j rig -> frame-k rig
    XYk = (np.c_[XYi.reshape(-1, 2), np.zeros(XYi.size // 2), np.ones(XYi.size // 2)] @ rel.T)[:, :2].reshape(XYi.shape)
    rgbK, _, okK = inverse(Fe, XYk)
    G = (0.5 * rgbA.astype(np.float32) + 0.5 * rgbK.astype(np.float32)).astype(np.uint8)
    Hh = A.shape[0]
    panels = [("A  image warped onto ground (0.05 m)", A), ("B  class raster warped the same way", B),
              ("C  current: lifted samples in 0.20 m cells", cv2.resize(C, (A.shape[1], Hh), interpolation=cv2.INTER_NEAREST)),
              ("D  lifted samples in 0.05 m cells", D),
              ("E  render fusion: 0.20 m, dilated, past 2 s", cv2.resize(E, (A.shape[1], Hh), interpolation=cv2.INTER_NEAREST)),
              ("F  this frame + 1 s earlier, warped", G)]
    wpan = A.shape[1]
    sheet = Image.new("RGB", (wpan * 6 + 50, Hh + 40), (14, 17, 16)); dr = ImageDraw.Draw(sheet)
    try:
        fnt = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        fnt = ImageFont.load_default()
    for q, (name, im) in enumerate(panels):
        sheet.paste(Image.fromarray(im), (q * (wpan + 10), 36)); dr.text((q * (wpan + 10) + 4, 8), name, fill=(230, 232, 231), font=fnt)
    sheet.save(out_png)
    print("saved", out_png)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]), a[4], float(a[5]), float(a[6]), float(a[7]), float(a[8]), a[9])
