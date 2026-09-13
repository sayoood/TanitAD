"""Ground-projection check on the native 120 deg front camera: ground model and calibration, scored without rewarding blur.

For frame j and the frame 1 s earlier, the same ground window (0.05 m) is warped straight from the native f-theta
images. Agreement = normalised cross-correlation of their GRADIENT magnitudes over one FIXED common mask (cells 5-25 m
from the rig that every tested setting sees) -- a smeared warp lowers gradients in both and cannot score higher, unlike
the mean-luminance difference whose optimum ran to the sweep edge (ipm_native.py).
Swept: ground model {per-2 m-cell (current), smooth bilinear}, camera pitch offset, camera height offset.
"""
import math, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
from ftheta_pinhole import FTheta, quat_to_R
from build_front_seq import pose_at, path_ground, D
from ipm_native import rot_pitch, frames_at
import sam3_paint as P
import ground_surface as GS


def zfun(kind, grid, fb):
    if kind == "cell":
        def f(xy):
            ci = np.clip(np.floor((xy[:, 0] + 50) / 2).astype(int), 0, 49); cj = np.clip(np.floor((xy[:, 1] + 50) / 2).astype(int), 0, 49)
            g = grid[ci * 50 + cj]
            return np.where(np.isfinite(g), g, fb)
        return f
    sg = GS.smooth_grid(grid, fb)
    return lambda xy: GS.height(xy, sg)


def warp(img, ft, R_s, t_s, zf, XY, shape, dpitch=0.0, dz=0.0):
    Pc = (np.c_[XY, zf(XY)] - (t_s + np.array([0.0, 0.0, dz]))) @ (R_s @ rot_pitch(dpitch))
    x, y, zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    rho = np.hypot(x, y); th = np.arctan2(rho, zc); r = ft.r_of_theta(th)
    k = np.where(rho > 1e-9, r / np.maximum(rho, 1e-9), 0)
    u = ft.cx + x * k; v = ft.cy + y * k
    ok = (zc > 0.3) & (u >= 0) & (u < img.shape[1] - 1) & (v >= 0) & (v < img.shape[0] - 1)
    mu = np.where(ok, u, -1).astype(np.float32).reshape(shape); mv = np.where(ok, v, -1).astype(np.float32).reshape(shape)
    return cv2.remap(img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), ok


def grad(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)).reshape(-1)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))


def main(clip, chunk, j, x0, x1, y0, y1, res, out_png):
    idf = pd.read_parquet(D / "calib" / f"camera_intrinsics.chunk_{chunk:04d}.parquet").reset_index()
    edf = pd.read_parquet(D / "calib" / f"sensor_extrinsics.chunk_{chunk:04d}.parquet").reset_index()
    r = idf[(idf["clip_id"] == clip) & (idf["camera_name"] == "camera_front_wide_120fov")].iloc[0]
    ft = FTheta(poly=tuple(float(r[f"fw_poly_{i}"]) for i in range(5)), cx=float(r["cx"]), cy=float(r["cy"]), width=int(r["width"]), height=int(r["height"]))
    e = edf[(edf["clip_id"] == clip) & (edf["sensor_name"] == "camera_front_wide_120fov")].iloc[0]
    R_s = quat_to_R(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"])); t_s = np.array([float(e["x"]), float(e["y"]), float(e["z"])])
    ego = pd.read_parquet(D / "egomotion_alpamayo" / f"{clip}.parquet").sort_values("timestamp").reset_index(drop=True)
    ts = pd.read_parquet(D / "frontwide" / f"{clip}.timestamps.parquet")["timestamp"].to_numpy(np.float64)
    t_refs = np.arange(ts[0] + 0.5e6, ts[-1] - 0.5e6, 0.2e6)
    ij, ik = int(np.argmin(np.abs(ts - t_refs[j]))), int(np.argmin(np.abs(ts - t_refs[j - 5])))
    imgs = frames_at(D / "frontwide" / f"{clip}.mp4", {ij, ik})
    Tj, Tk = pose_at(ego, ts[ij]), pose_at(ego, ts[ik])
    gj = P.ground_grid(path_ground(ego, Tj, ts[ij]).astype(np.float64)); gk = P.ground_grid(path_ground(ego, Tk, ts[ik]).astype(np.float64))
    xs = np.arange(x1 - res / 2, x0, -res); ys = np.arange(y1 - res / 2, y0, -res)
    GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]; shape = GX.shape
    XYk = (np.c_[XY, np.zeros(len(XY)), np.ones(len(XY))] @ (np.linalg.inv(Tk) @ Tj).T)[:, :2]
    dist = np.hypot(XY[:, 0], XY[:, 1])
    settings = [(kind, dp, dz) for kind in ("cell", "smooth") for dp in (-1.0, -0.5, 0.0, 0.5, 1.0) for dz in (0.0,)]
    settings += [("smooth", 0.0, dz) for dz in (-0.10, 0.10)]
    res_w = {}
    common = (dist >= 5) & (dist <= 25)
    for s in settings:
        a, oka = warp(imgs[ij], ft, R_s, t_s, zfun(s[0], *gj), XY, shape, s[1], s[2])
        b, okb = warp(imgs[ik], ft, R_s, t_s, zfun(s[0], *gk), XYk, shape, s[1], s[2])
        res_w[s] = (a, b); common &= oka & okb
    print(f"frames {ij} / {ik} (1 s apart); fixed common mask {int(common.sum())} cells at 5-25 m")
    for s in settings:
        a, b = res_w[s]
        print(f"  ground {s[0]:6s} pitch {s[1]:+.1f} deg  height {s[2]:+.2f} m   gradient NCC {ncc(grad(a)[common], grad(b)[common]):.4f}")
    A0, B0 = res_w[("cell", 0.0, 0.0)]; A1, B1 = res_w[("smooth", 0.0, 0.0)]
    panels = [("per-cell ground (current)", A0), ("smooth ground", A1),
              ("per-cell: + 1 s earlier", ((A0.astype(np.float32) + B0) / 2).astype(np.uint8)),
              ("smooth: + 1 s earlier", ((A1.astype(np.float32) + B1) / 2).astype(np.uint8))]
    W_, H_ = A0.shape[1], A0.shape[0]
    sheet = Image.new("RGB", (2 * W_ + 10, 2 * H_ + 68), (14, 17, 16)); dr = ImageDraw.Draw(sheet)
    fnt = ImageFont.truetype("DejaVuSans.ttf", 14)
    for q, (name, im) in enumerate(panels):
        x_, y_ = (q % 2) * (W_ + 10), (q // 2) * (H_ + 34)
        sheet.paste(Image.fromarray(im), (x_, y_ + 30)); dr.text((x_ + 6, y_ + 8), name, fill=(230, 232, 231), font=fnt)
    sheet.save(out_png); print("saved", out_png)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]), float(a[3]), float(a[4]), float(a[5]), float(a[6]), float(a[7]), a[8])
