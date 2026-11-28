"""v5f video v3: FULL 256-anchor plan-fan BEV, plan_fan.py conventions
(PI: 'the whole anchor fan visualization like the refc visualization').

Joins /workspace/v5f_viz (frames/gt/v0/goal/wgt-top24) with /workspace/v5f_fanfull
(full fan [T,256,20,2] + softmax probs + sel_idx + raw anchor vocabulary).
Layers per taniteval/plan_fan.py: (1) vocabulary shadow — all raw anchors faint
grey; (2) scored fan — ALL 256 refined proposals, viridis by softmax prob on the
FIXED log scale (P_FLOOR 1e-4), alpha+width scale with score, ascending order;
(3) top-8 emphasis; (4) SELECTED halo + white core; (5) GT dashed green; (6)
colorbar + HUD + 10 m rings. Camera pane: GT + selected only (v2 request kept).
Frames + mp4 on /tmp. Requires: episode-count and per-episode frame-count parity
between the two dumps (asserted; fails loud).
"""
import argparse
import glob
import math
import os
import shutil
import subprocess

import numpy as np

W_SUB, H_SUB = 624, 176
F_REF = 305.5774907364391
CAM_HM = 1.5
SC = 2
CAM_W, CAM_H_PX = W_SUB * SC, H_SUB * SC
BEV_H = 420
BEV_XR = 60.0
BEV_S = (BEV_H - 44) / BEV_XR

P_FLOOR = 1e-4
LOG_FLOOR = math.log10(P_FLOOR)
VIRIDIS = ((68, 1, 84), (72, 40, 120), (62, 73, 137), (49, 104, 142),
           (38, 130, 142), (31, 158, 137), (53, 183, 121), (110, 206, 88),
           (181, 222, 43), (253, 231, 37))
TOP_K_EMPH = 8


def viridis(t):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    x = t * (len(VIRIDIS) - 1)
    i = min(int(x), len(VIRIDIS) - 2)
    f = x - i
    a, b = VIRIDIS[i], VIRIDIS[i + 1]
    return tuple(int(round(a[j] + f * (b[j] - a[j]))) for j in range(3))


def p_to_t(p):
    if p <= P_FLOOR:
        return 0.0
    return (math.log10(p) - LOG_FLOOR) / (0.0 - LOG_FLOOR)


def project(pts):
    fwd = np.maximum(pts[:, 0], 0.3)
    lat = -pts[:, 1]
    phi = np.arctan2(lat, fwd)
    rho = np.hypot(lat, fwd)
    return ((W_SUB - 1) / 2.0 + F_REF * phi,
            (H_SUB - 1) / 2.0 + F_REF * (CAM_HM / rho))


def main(a):
    from PIL import Image, ImageDraw, ImageFont
    fr_dir = "/tmp/v5f_ff_frames"
    shutil.rmtree(fr_dir, ignore_errors=True)
    os.makedirs(fr_dir)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_s = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = font_s = ImageFont.load_default()

    anc_p = os.path.join(a.fanfull, "anchors.npz")
    anchors = (np.load(anc_p)["anchors"].astype(np.float64)
               if os.path.exists(anc_p) else None)
    if anchors is not None and anchors.ndim == 3:
        anchors = anchors[..., :2]

    viz_eps = sorted(glob.glob(f"{a.dump}/ep*.npz"))
    ff_eps = sorted(glob.glob(f"{a.fanfull}/ep*.npz"))
    assert len(viz_eps) == len(ff_eps), (len(viz_eps), len(ff_eps))

    H_TOT = CAM_H_PX + BEV_H
    cx_bev = CAM_W // 2
    y0 = CAM_H_PX
    n = 0
    for epf, ffp in zip(viz_eps, ff_eps):
        d = np.load(epf)
        f = np.load(ffp)
        T = d["frame"].shape[0]
        assert f["fan"].shape[0] == T, (epf, ffp, f["fan"].shape, T)
        for i in range(T):
            img = d["frame"][i]
            if img.ndim == 3 and img.shape[-1] == 9:
                img = img[..., -3:]
            if img.ndim == 3 and img.shape[0] in (3, 9):
                img = img[-3:].transpose(1, 2, 0)
            canvas = Image.new("RGB", (CAM_W, H_TOT), (10, 12, 16))
            cam = Image.fromarray(img).resize((CAM_W, CAM_H_PX), Image.LANCZOS)
            canvas.paste(cam, (0, 0))
            dr = ImageDraw.Draw(canvas, "RGBA")

            fan = f["fan"][i].astype(np.float64)          # [256, 20, 2]
            probs = f["probs"][i].astype(np.float64)      # [256]
            sel = int(f["sel"][i])
            gt = d["gt"][i].astype(np.float64)
            fe = np.linalg.norm(fan - gt[None], axis=-1).mean(-1)
            orc = int(fe.argmin())

            def cam_line(pts, color, width):
                c, r = project(pts)
                dr.line(list(zip(c * SC, r * SC)), fill=color, width=width)
            cam_line(gt, (238, 51, 119, 255), 4)
            cam_line(fan[sel], (118, 255, 40, 255), 5)
            dr.rectangle([0, 0, CAM_W, 26], fill=(0, 0, 0, 180))
            dr.text((8, 4),
                    f"v5f-30k plan fan · sel ADE {fe[sel]:.2f} m · oracle-in-fan "
                    f"{fe[orc]:.2f} m · v0 {d['v0'][i]*3.6:.0f} km/h · "
                    f"modes>1% {int((probs > 0.01).sum())} · T0",
                    fill=(255, 255, 255), font=font)

            dr.rectangle([0, y0, CAM_W, H_TOT], fill=(10, 12, 16, 255))
            for rr in range(10, int(BEV_XR) + 1, 10):
                py = (y0 + BEV_H - 24) - rr * BEV_S
                dr.line([cx_bev - 320, py, cx_bev + 320, py],
                        fill=(40, 44, 54), width=1)
                dr.text((cx_bev + 326, py - 7), f"{rr} m",
                        fill=(105, 112, 124), font=font_s)
            dr.line([cx_bev, y0 + 8, cx_bev, y0 + BEV_H - 12],
                    fill=(40, 44, 54), width=1)

            def bev_xy(pts):
                px = cx_bev - pts[:, 1] * BEV_S
                py = (y0 + BEV_H - 24) - pts[:, 0] * BEV_S
                return px, py

            def bev_line(pts, color, width, dash=False):
                px, py = bev_xy(pts)
                xy = list(zip(px, py))
                if dash:
                    for k in range(0, len(xy) - 1, 2):
                        dr.line(xy[k:k + 2], fill=color, width=width)
                else:
                    dr.line(xy, fill=color, width=width)

            # 1) vocabulary shadow: raw anchors, faint grey
            if anchors is not None:
                for k in range(anchors.shape[0]):
                    bev_line(anchors[k], (90, 95, 105, 26), 1)
            # 2) scored fan: ALL 256 refined, viridis by prob, ascending
            order = np.argsort(probs)
            for k in order:
                t = p_to_t(float(probs[k]))
                r, g_, b_ = viridis(t)
                aph = int(30 + 190 * t)
                bev_line(fan[k], (r, g_, b_, aph), 1 + int(2 * t))
            # 3) top-8 emphasis with waypoint dots
            top8 = np.argsort(-probs)[:TOP_K_EMPH]
            for k in top8:
                t = p_to_t(float(probs[k]))
                r, g_, b_ = viridis(t)
                bev_line(fan[k], (r, g_, b_, 235), 3)
                px, py = bev_xy(fan[k][4::5])
                for x_, y_ in zip(px, py):
                    dr.ellipse([x_ - 2, y_ - 2, x_ + 2, y_ + 2],
                               fill=(r, g_, b_, 235))
            # 4) selected: colour halo under white core
            bev_line(fan[sel], (118, 255, 40, 255), 7)
            bev_line(fan[sel], (255, 255, 255, 255), 3)
            # 5) GT dashed + oracle marker
            bev_line(gt, (60, 220, 60, 255), 3, dash=True)
            bev_line(fan[orc], (0, 220, 255, 200), 2, dash=True)
            ex, ey = cx_bev, y0 + BEV_H - 24
            dr.polygon([(ex, ey - 10), (ex - 6, ey + 4), (ex + 6, ey + 4)],
                       fill=(255, 255, 255))
            # 6) colorbar (fixed log scale)
            bx0, by0 = 12, y0 + 10
            for px_ in range(180):
                r, g_, b_ = viridis(px_ / 179.0)
                dr.line([bx0 + px_, by0, bx0 + px_, by0 + 12],
                        fill=(r, g_, b_, 255))
            dr.text((bx0, by0 + 14), "p=1e-4", fill=(150, 156, 168), font=font_s)
            dr.text((bx0 + 150, by0 + 14), "p=1", fill=(150, 156, 168), font=font_s)
            dr.text((bx0 + 200, by0),
                    "all 256 refined proposals · shadow = raw vocabulary · "
                    "sel = halo+white · GT dashed green · oracle dashed cyan",
                    fill=(185, 190, 200), font=font_s)

            g4 = d["goal"][i] if "goal" in d.files else None
            if g4 is not None and np.any(g4):
                graded = float(np.tanh(g4[2] * 60.0))
                cls = ("LEFT" if graded > 0.762
                       else ("RIGHT" if graded < -0.762 else "STRAIGHT"))
                dr.rectangle([0, H_TOT - 28, CAM_W, H_TOT], fill=(0, 0, 0, 200))
                dr.text((8, H_TOT - 24),
                        f"TACTICAL  target v(5s) {float(g4[3])*3.6:5.0f} km/h · "
                        f"curv3s {g4[1]:+.4f}/m · ttm {g4[0]:4.1f}s",
                        fill=(255, 220, 120), font=font)
                bx0b, bx1b, byb = CAM_W - 330, CAM_W - 30, H_TOT - 14
                dr.line([bx0b, byb, bx1b, byb], fill=(120, 120, 120), width=3)
                mid = (bx0b + bx1b) / 2
                dr.line([mid, byb - 6, mid, byb + 6], fill=(160, 160, 160), width=2)
                pxm = mid - graded * (bx1b - bx0b) / 2
                dr.ellipse([pxm - 6, byb - 6, pxm + 6, byb + 6], fill=(0, 220, 255))
                dr.text((bx0b, H_TOT - 28), f"STRATEGIC route {cls} ({graded:+.2f})",
                        fill=(0, 220, 255), font=font_s)

            canvas.save(f"{fr_dir}/f{n:06d}.png")
            n += 1
        print(f"[ff-render] {epf} done ({n})", flush=True)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-framerate", str(a.fps), "-i",
                    f"{fr_dir}/f%06d.png", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "21", "-threads", "4", "-pix_fmt", "yuv420p",
                    "/tmp/v5f_planfan.mp4"], check=True)
    subprocess.run([ff, "-y", "-i", "/tmp/v5f_planfan.mp4", "-vf", "scale=624:-2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                    "-threads", "4", "-pix_fmt", "yuv420p",
                    "/tmp/v5f_planfan_compact.mp4"], check=True)
    print(f"[ff-render] {n} frames", flush=True)
    print("PLANFAN_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/workspace/v5f_viz")
    ap.add_argument("--fanfull", default="/workspace/v5f_fanfull")
    ap.add_argument("--fps", type=int, default=10)
    main(ap.parse_args())
