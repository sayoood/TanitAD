"""v5f video v2 (PI request 2026-08-10): clean camera overlay + BEV fan pane.

Camera pane (top, 1248x352 = 2x upscale of the 176x624 sub-frame): ONLY the GT
trajectory (pink) and the v5f FINAL selected trajectory (bold green). No fan.
BEV pane (bottom, ego frame, forward = up): the full dumped fan (top-24 of 256,
weight-alpha green), selected bold green, fan-oracle dashed cyan, GT pink,
10 m range rings, HUD + tactical/strategic goal strip.
Consumes the EXISTING /workspace/v5f_viz dump (57 episodes); render-only, CPU.
Frames + mp4 go to /tmp (container disk) to stay off MooseFS during writes.
"""
import argparse
import glob
import os
import shutil
import subprocess

import numpy as np

W_SUB, H_SUB = 624, 176
F_REF = 305.5774907364391
CAM_H = 1.5
SC = 2
CAM_W, CAM_H_PX = W_SUB * SC, H_SUB * SC          # 1248 x 352
BEV_H = 380
BEV_XR = 60.0                                      # metres ahead shown
BEV_S = (BEV_H - 40) / BEV_XR                      # px per metre (~5.7)


def project(pts):
    fwd = np.maximum(pts[:, 0], 0.3)
    lat = -pts[:, 1]
    phi = np.arctan2(lat, fwd)
    rho = np.hypot(lat, fwd)
    col = (W_SUB - 1) / 2.0 + F_REF * phi
    row = (H_SUB - 1) / 2.0 + F_REF * (CAM_H / rho)
    return col, row


def bev_xy(pts, cx):
    """ego (x fwd, y left) -> BEV pixel (forward = up, left = left)."""
    px = cx - pts[:, 1] * BEV_S
    py = (BEV_H - 20) - pts[:, 0] * BEV_S
    return px, py


def main(a):
    from PIL import Image, ImageDraw, ImageFont
    fr_dir = "/tmp/v5f_bev_frames"
    shutil.rmtree(fr_dir, ignore_errors=True)
    os.makedirs(fr_dir)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_s = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = font_s = ImageFont.load_default()
    n = 0
    H_TOT = CAM_H_PX + BEV_H
    cx_bev = CAM_W // 2
    for epf in sorted(glob.glob(f"{a.dump}/ep*.npz")):
        d = np.load(epf)
        T = d["frame"].shape[0]
        for i in range(T):
            img = d["frame"][i]
            if img.ndim == 3 and img.shape[-1] == 9:
                img = img[..., -3:]
            if img.ndim == 3 and img.shape[0] in (3, 9):
                img = img[-3:].transpose(1, 2, 0)
            canvas = Image.new("RGB", (CAM_W, H_TOT), (12, 14, 18))
            cam = Image.fromarray(img).resize((CAM_W, CAM_H_PX), Image.LANCZOS)
            canvas.paste(cam, (0, 0))
            dr = ImageDraw.Draw(canvas, "RGBA")

            fan = d["fan"][i].astype(np.float64)       # [24, 20, 2]
            gt = d["gt"][i].astype(np.float64)         # [20, 2]
            wgt = d["wgt"][i].astype(np.float64)
            fe = np.linalg.norm(fan - gt[None], axis=-1).mean(-1)
            orc = int(fe.argmin())

            # ---- camera overlay: GT + selected ONLY --------------------------
            def cam_line(pts, color, width, dash=False):
                c, r = project(pts)
                xy = list(zip(c * SC, r * SC))
                if dash:
                    for k in range(0, len(xy) - 1, 2):
                        dr.line(xy[k:k + 2], fill=color, width=width)
                else:
                    dr.line(xy, fill=color, width=width)
            cam_line(gt, (238, 51, 119, 255), 4)
            cam_line(fan[0], (118, 255, 40, 255), 5)   # top-weight = deployed
            dr.rectangle([0, 0, CAM_W, 26], fill=(0, 0, 0, 180))
            dr.text((8, 4), "v5f-30k  ·  GT (pink) vs selected plan (green)  ·  "
                    f"sel ADE {d['sel_ade'][i]:.2f} m · v0 {d['v0'][i]*3.6:.0f} km/h"
                    "  ·  T0 (plan from context, no future actions)",
                    fill=(255, 255, 255), font=font)

            # ---- BEV pane ----------------------------------------------------
            y0 = CAM_H_PX
            dr.rectangle([0, y0, CAM_W, H_TOT], fill=(12, 14, 18, 255))
            # range rings + centreline
            for rr in range(10, int(BEV_XR) + 1, 10):
                py = (y0 + BEV_H - 20) - rr * BEV_S
                dr.line([cx_bev - 300, py, cx_bev + 300, py],
                        fill=(45, 50, 60), width=1)
                dr.text((cx_bev + 306, py - 7), f"{rr} m",
                        fill=(110, 118, 130), font=font_s)
            dr.line([cx_bev, y0 + 10, cx_bev, y0 + BEV_H - 10],
                    fill=(45, 50, 60), width=1)

            def bev_line(pts, color, width, dash=False):
                px, py = bev_xy(pts, cx_bev)
                xy = list(zip(px, y0 + py))
                if dash:
                    for k in range(0, len(xy) - 1, 2):
                        dr.line(xy[k:k + 2], fill=color, width=width)
                else:
                    dr.line(xy, fill=color, width=width)
            wmax = max(float(wgt.max()), 1e-6)
            for k in range(fan.shape[0] - 1, 0, -1):   # fan first, selected last
                aph = int(28 + 150 * float(wgt[k]) / wmax)
                bev_line(fan[k], (118, 185, 0, aph), 2)
            bev_line(gt, (238, 51, 119, 255), 4)
            bev_line(fan[orc], (0, 220, 255, 255), 3, dash=True)
            bev_line(fan[0], (118, 255, 40, 255), 5)
            # ego marker
            ex, ey = cx_bev, y0 + BEV_H - 20
            dr.polygon([(ex, ey - 10), (ex - 6, ey + 4), (ex + 6, ey + 4)],
                       fill=(255, 255, 255))
            dr.text((8, y0 + 6),
                    f"BEV · fan top-{fan.shape[0]} of 256 (weight-alpha) · "
                    f"selected (green) · fan-oracle (cyan dashed) "
                    f"{d['orc_ade'][i]:.2f} m · GT (pink)",
                    fill=(200, 205, 215), font=font_s)

            # tactical / strategic strip (goal head scalars)
            g4 = d["goal"][i] if "goal" in d.files else None
            if g4 is not None and np.any(g4):
                CURV_TURN = 1.0 / 60.0
                graded = float(np.tanh(g4[2] / CURV_TURN))
                cls = ("LEFT" if graded > 0.762
                       else ("RIGHT" if graded < -0.762 else "STRAIGHT"))
                dr.rectangle([0, H_TOT - 28, CAM_W, H_TOT], fill=(0, 0, 0, 200))
                dr.text((8, H_TOT - 24),
                        f"TACTICAL  target v(5s) {float(g4[3])*3.6:5.0f} km/h · "
                        f"curv3s {g4[1]:+.4f}/m · ttm {g4[0]:4.1f}s",
                        fill=(255, 220, 120), font=font)
                bx0, bx1, by = CAM_W - 330, CAM_W - 30, H_TOT - 14
                dr.line([bx0, by, bx1, by], fill=(120, 120, 120), width=3)
                mid = (bx0 + bx1) / 2
                dr.line([mid, by - 6, mid, by + 6], fill=(160, 160, 160), width=2)
                px = mid - graded * (bx1 - bx0) / 2
                dr.ellipse([px - 6, by - 6, px + 6, by + 6], fill=(0, 220, 255))
                dr.text((bx0, H_TOT - 28), f"STRATEGIC route {cls} ({graded:+.2f})",
                        fill=(0, 220, 255), font=font_s)

            canvas.save(f"{fr_dir}/f{n:06d}.png")
            n += 1
        print(f"[bev] {epf} done ({n} frames)", flush=True)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-framerate", str(a.fps), "-i",
                    f"{fr_dir}/f%06d.png", "-c:v", "libx264", "-preset",
                    "medium", "-crf", "21", "-threads", "4", "-pix_fmt",
                    "yuv420p", "/tmp/v5f_bev.mp4"], check=True)
    subprocess.run([ff, "-y", "-i", "/tmp/v5f_bev.mp4", "-vf", "scale=624:-2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                    "-threads", "4", "-pix_fmt", "yuv420p",
                    "/tmp/v5f_bev_compact.mp4"], check=True)
    print(f"[bev] {n} frames {n/a.fps:.0f}s", flush=True)
    print("BEVRENDER_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/workspace/v5f_viz")
    ap.add_argument("--fps", type=int, default=10)
    main(ap.parse_args())
