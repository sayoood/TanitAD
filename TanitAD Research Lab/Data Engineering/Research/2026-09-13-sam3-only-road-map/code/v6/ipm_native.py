"""Warp the NATIVE front-wide f-theta frame (full 120 deg) onto the ground, and test calibration by frame-to-frame
agreement.

(1) native vs virtual: the SAM3 input so far was the virtual nuPlan pinhole view (fx 1545 @ 1920 -> 63.7 deg), itself an
    interpolated reprojection of the native image; here each ground cell (0.05 m) is mapped straight into the native
    f-theta image (one bilinear lookup): cell -> rig -> camera (R_s, t_s) -> theta -> r(theta) -> pixel
(2) calibration: the same ground window warped from this frame and from the frame 1 s earlier (through the poses) must
    agree where both see the road. Mean absolute luminance difference over the common road area, swept over a pitch
    offset (and a camera-height offset) of the camera; a correct calibration has its minimum at 0.
Output: ipm_native_<c8>_<j>.png + printed sweep.
"""
import json, math, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map/eval")
import numpy as np
import pandas as pd
import av
import cv2
from PIL import Image, ImageDraw, ImageFont
from ftheta_pinhole import FTheta, quat_to_R
from build_front_seq import pose_at, path_ground, D

sys.path.insert(0, "/home/nvidia/sam3paint")
import sam3_paint as P


SHAPE = [None]


def rot_pitch(deg):
    a = math.radians(deg)                       # rotate the camera about its own x (right) axis
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def frames_at(mp4, idx):
    out = {}
    with av.open(str(mp4)) as cont:
        st = cont.streams.video[0]
        for i, fr in enumerate(cont.decode(st)):
            if i in idx:
                out[i] = fr.to_ndarray(format="rgb24")
            if i >= max(idx):
                break
    return out


def warp(img, ft, R_s, t_s, grid, fb, XY, dpitch=0.0, dz=0.0, shape=None):
    ci = np.clip(np.floor((XY[:, 0] + 50) / 2).astype(int), 0, 49); cj = np.clip(np.floor((XY[:, 1] + 50) / 2).astype(int), 0, 49)
    g = grid[ci * 50 + cj]; z = np.where(np.isfinite(g), g, fb)
    Pc = (np.c_[XY, z] - (t_s + np.array([0, 0, dz]))) @ (R_s @ rot_pitch(dpitch))     # rig -> camera
    x, y, zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    rho = np.hypot(x, y); th = np.arctan2(rho, zc)
    r = ft.r_of_theta(th)
    k = np.where(rho > 1e-9, r / np.maximum(rho, 1e-9), 0)
    u = ft.cx + x * k; v = ft.cy + y * k
    ok = (zc > 0.3) & (u >= 0) & (u < img.shape[1] - 1) & (v >= 0) & (v < img.shape[0] - 1)
    mu = np.where(ok, u, -1).astype(np.float32).reshape(SHAPE[0]); mv = np.where(ok, v, -1).astype(np.float32).reshape(SHAPE[0])
    return cv2.remap(img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), ok


def main(clip, chunk, j, x0, x1, y0, y1, res):
    c8 = clip[:8]
    idf = pd.read_parquet(D / "calib" / f"camera_intrinsics.chunk_{chunk:04d}.parquet").reset_index()
    edf = pd.read_parquet(D / "calib" / f"sensor_extrinsics.chunk_{chunk:04d}.parquet").reset_index()
    r = idf[(idf["clip_id"] == clip) & (idf["camera_name"] == "camera_front_wide_120fov")].iloc[0]
    ft = FTheta(poly=tuple(float(r[f"fw_poly_{i}"]) for i in range(5)), cx=float(r["cx"]), cy=float(r["cy"]), width=int(r["width"]), height=int(r["height"]))
    e = edf[(edf["clip_id"] == clip) & (edf["sensor_name"] == "camera_front_wide_120fov")].iloc[0]
    R_s = quat_to_R(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"])); t_s = np.array([float(e["x"]), float(e["y"]), float(e["z"])])
    rmax = math.hypot(max(ft.cx, ft.width - ft.cx), max(ft.cy, ft.height - ft.cy))
    print(f"native camera: {ft.width}x{ft.height}, poly {tuple(round(p, 3) for p in ft.poly)}, max theta {math.degrees(ft.theta_of_r(ft.width / 2, hi=2.2)) * 2:.1f} deg across the width, "
          f"height {t_s[2]:.3f} m")
    ego = pd.read_parquet(D / "egomotion_alpamayo" / f"{clip}.parquet").sort_values("timestamp").reset_index(drop=True)
    ts = pd.read_parquet(D / "frontwide" / f"{clip}.timestamps.parquet")["timestamp"].to_numpy(np.float64)
    t_refs = np.arange(ts[0] + 0.5e6, ts[-1] - 0.5e6, 0.2e6)
    tj, tk = t_refs[j], t_refs[j - 5]
    ij, ik = int(np.argmin(np.abs(ts - tj))), int(np.argmin(np.abs(ts - tk)))
    imgs = frames_at(D / "frontwide" / f"{clip}.mp4", {ij, ik})
    Tj, Tk = pose_at(ego, ts[ij]), pose_at(ego, ts[ik])
    gj = P.ground_grid(path_ground(ego, Tj, ts[ij]).astype(np.float64)); gk = P.ground_grid(path_ground(ego, Tk, ts[ik]).astype(np.float64))
    xs = np.arange(x1 - res / 2, x0, -res); ys = np.arange(y1 - res / 2, y0, -res)
    GX, GY = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[GX.ravel(), GY.ravel()]; SHAPE[0] = GX.shape
    rel = np.linalg.inv(Tk) @ Tj
    XYk = (np.c_[XY, np.zeros(len(XY)), np.ones(len(XY))] @ rel.T)[:, :2]
    Wj, okj = warp(imgs[ij], ft, R_s, t_s, *gj, XY)
    Wk, okk = warp(imgs[ik], ft, R_s, t_s, *gk, XYk)
    both = okj & okk & (np.hypot(XY[:, 0], XY[:, 1]) < 25)
    Lj = cv2.cvtColor(Wj.reshape(GX.shape + (3,)), cv2.COLOR_RGB2GRAY).reshape(-1).astype(np.float32)
    print(f"frames {ij} and {ik} (1 s apart), common ground cells {int(both.sum())}")
    print(" pitch offset   dz 0.00   (mean |luminance difference|)")
    best = None
    for dp in (-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5):
        a, oka = warp(imgs[ij], ft, R_s, t_s, *gj, XY, dpitch=dp)
        b, okb = warp(imgs[ik], ft, R_s, t_s, *gk, XYk, dpitch=dp)
        m = oka & okb & (np.hypot(XY[:, 0], XY[:, 1]) < 25)
        la = cv2.cvtColor(a.reshape(GX.shape + (3,)), cv2.COLOR_RGB2GRAY).reshape(-1).astype(np.float32)
        lb = cv2.cvtColor(b.reshape(GX.shape + (3,)), cv2.COLOR_RGB2GRAY).reshape(-1).astype(np.float32)
        err = float(np.mean(np.abs(la[m] - lb[m]))); print(f"   {dp:+.2f} deg   {err:6.2f}   (n {int(m.sum())})")
        if best is None or err < best[1]:
            best = (dp, err)
    print(f" best pitch offset {best[0]:+.2f} deg")
    for dz in (-0.10, -0.05, 0.05, 0.10):
        a, oka = warp(imgs[ij], ft, R_s, t_s, *gj, XY, dpitch=best[0], dz=dz)
        b, okb = warp(imgs[ik], ft, R_s, t_s, *gk, XYk, dpitch=best[0], dz=dz)
        m = oka & okb & (np.hypot(XY[:, 0], XY[:, 1]) < 25)
        la = cv2.cvtColor(a.reshape(GX.shape + (3,)), cv2.COLOR_RGB2GRAY).reshape(-1).astype(np.float32)
        lb = cv2.cvtColor(b.reshape(GX.shape + (3,)), cv2.COLOR_RGB2GRAY).reshape(-1).astype(np.float32)
        print(f"   height offset {dz:+.2f} m at best pitch: {float(np.mean(np.abs(la[m] - lb[m]))):6.2f}")
    A = Wj.reshape(GX.shape + (3,)); F = (0.5 * A.astype(np.float32) + 0.5 * Wk.reshape(GX.shape + (3,)).astype(np.float32)).astype(np.uint8)
    sheet = Image.new("RGB", (A.shape[1] * 2 + 10, A.shape[0] + 34), (14, 17, 16)); dr = ImageDraw.Draw(sheet)
    fnt = ImageFont.truetype("DejaVuSans.ttf", 14)
    sheet.paste(Image.fromarray(A), (0, 34)); sheet.paste(Image.fromarray(F), (A.shape[1] + 10, 34))
    dr.text((6, 8), f"native 120 deg f-theta frame warped onto ground ({res} m)", fill=(230, 232, 231), font=fnt)
    dr.text((A.shape[1] + 16, 8), "this frame + 1 s earlier, blended", fill=(230, 232, 231), font=fnt)
    out = f"/home/nvidia/sam3map/ipm_native_{c8}_{j}.png"; sheet.save(out); print("saved", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]), float(a[3]), float(a[4]), float(a[5]), float(a[6]), float(a[7]))
