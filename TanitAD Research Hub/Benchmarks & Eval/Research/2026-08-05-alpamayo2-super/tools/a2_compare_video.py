"""Comparison reel: Alpamayo 2 Super vs TanitAD flagship, same clip, same t0.

Each clip is one segment. Panels:
  LEFT   our flagship's actual input — ONE 256x256 front crop — with its 2 s
         prediction and the GT projected in. This is the whole sensor budget it
         gets, and showing it is the point.
  MIDDLE metric BEV, calibration-independent, carrying BOTH predictions against
         the SAME ground truth: Alpamayo (green), flagship (orange), GT (pink).
  RIGHT  Alpamayo's Chain-of-Causation text, both ADEs, and the caveats.

⛔ THE BANNER IS NOT DECORATION. This is not a fair fight and the frame says so:
34.3 B vs <0.3 B, 6 cameras x 1920x1080 vs one 256x256 crop, Alpamayo truncated
from 6.4 s to 2 s, NF4-quantised, and the clips are from a dataset Alpamayo lists
as TRAINING data. A viewer who sees only the BEV would conclude something the
evidence does not support.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

W_CAM, W_BEV, W_TXT = 448, 430, 470
H_BAN, PAD = 82, 10
COL_ALP = (118, 185, 0)        # NVIDIA green
COL_FS = (255, 122, 61)        # our orange
COL_GT = (238, 51, 119)        # pink, as in Alpamayo's own palette


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def bev(size, gt, alp, fs, fonts):
    w, h = size
    im = Image.new("RGB", (w, h), (8, 11, 15))
    d = ImageDraw.Draw(im)
    allp = [p for p in (gt, alp, fs) if p is not None and len(p)]
    xmax = max(12.0, max(float(np.max(p[:, 0])) for p in allp) * 1.1)
    ymax = max(4.0, max(float(np.max(np.abs(p[:, 1]))) for p in allp) * 1.3)
    pad = 34
    cx, by, top = w // 2, h - pad, pad + 16

    def m2px(X, Y):
        return (cx - (Y / ymax) * ((w / 2) - pad),
                by - (max(X, 0.0) / xmax) * (by - top))

    step = 10 if xmax > 25 else 5
    r = step
    while r <= xmax + 0.1:
        _, py = m2px(r, 0)
        d.line([(12, py), (w - 12, py)], fill=(34, 42, 52))
        d.text((14, py - 12), f"{r:g} m", fill=(96, 106, 120), font=fonts["tiny"])
        r += step
    d.line([(cx, top), (cx, by)], fill=(44, 54, 66))
    for path, col, wd in ((gt, COL_GT, 6), (alp, COL_ALP, 3), (fs, COL_FS, 3)):
        if path is None or len(path) < 2:
            continue
        pts = [m2px(float(p[0]), float(p[1])) for p in path]
        d.line(pts, fill=col, width=wd)
        x, y = pts[-1]
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=col, width=3)
    d.polygon([(cx - 6, by), (cx + 6, by), (cx, by - 12)], fill=(232, 236, 242))
    d.text((12, 8), "BEV · metres · calibration-independent · 2 s horizon",
           fill=(150, 160, 175), font=fonts["tiny"])
    for i, (lab, col) in enumerate((("ground truth", COL_GT),
                                    ("Alpamayo 2 Super", COL_ALP),
                                    ("TanitAD flagship", COL_FS))):
        d.line([(14, h - 46 + i * 14), (34, h - 46 + i * 14)], fill=col, width=4)
        d.text((40, h - 53 + i * 14), lab, fill=(200, 208, 218), font=fonts["tiny"])
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-json", required=True)
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--flagship-json", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds-per-clip", type=float, default=2.5)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    ffmpeg = _ffmpeg()

    from taniteval.corpus_overlay import FlatProjector
    from taniteval.data import load_frames
    from taniteval.flagship_overlay import _font

    fonts = {"big": _font(19), "hud": _font(15), "sub": _font(13), "tiny": _font(11)}
    cmp_ = json.load(open(a.compare_json))
    fs_rows = {r["i"]: r for r in json.load(open(a.flagship_json)) if "pred" in r}
    proj = FlatProjector(128.0)

    W = PAD + W_CAM + PAD + W_BEV + PAD + W_TXT + PAD
    H = H_BAN + W_CAM + PAD
    frames_dir = os.path.join(os.path.dirname(a.out) or ".", "_cmp_frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)
    n_out = 0
    per_clip = max(1, int(a.seconds_per_clip * a.fps))

    for row in cmp_["rows"]:
        i = row["i"]
        z = np.load(os.path.join(a.traj_dir, f"traj_{i:04d}.npz"))
        P = np.asarray(z["pred_xyz"]).reshape(-1, z["pred_xyz"].shape[-2], 3)[0]
        G = np.asarray(z["gt_xyz"]).reshape(-1, z["gt_xyz"].shape[-2], 3)[0]
        alp = P[:20, :2]
        fsr = fs_rows.get(i)
        if fsr is None:
            continue
        fsp = np.asarray(fsr["pred"])[:20]
        gt = np.asarray(fsr["gt"])[:20]          # our GT, our t0
        ep = load_frames([f"{a.corpus}/ep_{i:05d}.pt"])[0]
        t = fsr["window_last"]
        rgb = torch.as_tensor(ep.feats[t, -3:]).permute(1, 2, 0).numpy()

        canvas = Image.new("RGB", (W, H), (6, 9, 12))
        d = ImageDraw.Draw(canvas)
        cam = Image.fromarray(rgb).resize((W_CAM, W_CAM), Image.LANCZOS)
        cd = ImageDraw.Draw(cam)
        for path, col, wd in ((gt, COL_GT, 6), (fsp, COL_FS, 3)):
            pts = proj(path)
            if len(pts) >= 2:
                cd.line(pts, fill=col, width=wd)
        canvas.paste(cam, (PAD, H_BAN))
        canvas.paste(bev((W_BEV, W_CAM), gt, alp, fsp, fonts),
                     (PAD + W_CAM + PAD, H_BAN))

        # banner
        d.rectangle([0, 0, W, H_BAN], fill=(10, 14, 19))
        d.text((PAD, 5), f"Alpamayo 2 Super  vs  TanitAD flagship-v1arch   ·   "
                         f"clip {row['clip_id'][:8]}   ·   t0 5.1 s",
               fill=(233, 237, 243), font=fonts["big"])
        d.text((PAD, 30), "⛔ NOT LIKE-FOR-LIKE: 34.3B vs <0.3B params · 6 cameras "
                          "@1920x1080 vs ONE 256x256 crop · Alpamayo truncated "
                          "6.4s->2.0s · NF4-quantised (not NVIDIA-validated)",
               fill=(235, 180, 90), font=fonts["tiny"])
        d.text((PAD, 46), "⛔ CONTAMINATION UNRESOLVED: these clips are "
                          "PhysicalAI-AV, which Alpamayo lists as TRAINING data — "
                          "any advantage may be contamination, not capability.",
               fill=(235, 180, 90), font=fonts["tiny"])
        d.text((PAD, 62), "LEFT = the flagship's ENTIRE sensor input. Alpamayo "
                          "additionally sees 5 more cameras at full HD.",
               fill=(150, 160, 175), font=fonts["tiny"])

        # text panel
        x0 = PAD + W_CAM + PAD + W_BEV + PAD
        d.rectangle([x0, H_BAN, x0 + W_TXT, H_BAN + W_CAM], fill=(10, 14, 19),
                    outline=(40, 48, 58))
        y = H_BAN + 12
        d.text((x0 + 12, y), "Alpamayo Chain-of-Causation", fill=COL_ALP,
               font=fonts["hud"]); y += 24
        words, line = (row.get("cot") or "").split(), ""
        for wd_ in words:
            if d.textlength(line + " " + wd_, font=fonts["sub"]) > W_TXT - 30:
                d.text((x0 + 12, y), line, fill=(233, 237, 243), font=fonts["sub"])
                y += 18; line = wd_
            else:
                line = (line + " " + wd_).strip()
        if line:
            d.text((x0 + 12, y), line, fill=(233, 237, 243), font=fonts["sub"])
        y += 34
        for lab, val, col in (
                ("ADE @2s  Alpamayo", f"{row['alp_ade_2s']:.3f} m", COL_ALP),
                ("ADE @2s  flagship", f"{row['fs_ade_2s']:.3f} m", COL_FS),
                ("Alpamayo native 6.4 s", f"{row['alp_ade_full_6p4s']:.3f} m", (150, 160, 175)),
                ("speed bias  Alpamayo", f"{row['alp_speed_bias']:+.3f} m/s", COL_ALP),
                ("speed bias  flagship", f"{row['fs_speed_bias']:+.3f} m/s", COL_FS),
                ("ego speed v0", f"{row['v0_mps']:.1f} m/s", (150, 160, 175)),
                ("t0 alignment", f"{row['align_err_s']:+.3f} s", (150, 160, 175))):
            d.text((x0 + 12, y), lab, fill=(150, 160, 175), font=fonts["tiny"])
            d.text((x0 + 250, y - 2), val, fill=col, font=fonts["sub"])
            y += 22
        y += 6
        d.text((x0 + 12, y), "⚠️ Each model is scored against ITS OWN ground",
               fill=(120, 130, 145), font=fonts["tiny"]); y += 14
        d.text((x0 + 12, y), "truth at ITS OWN t0 (|dt| ≤ 0.4 s, our 0.8 s stride).",
               fill=(120, 130, 145), font=fonts["tiny"])

        for _ in range(per_clip):
            canvas.save(os.path.join(frames_dir, f"f{n_out:05d}.png"))
            n_out += 1
        print(f"[{i}] {row['clip_id'][:8]} alp {row['alp_ade_2s']:.3f} "
              f"fs {row['fs_ade_2s']:.3f}", flush=True)

    subprocess.run([ffmpeg, "-y", "-r", str(a.fps),
                    "-i", os.path.join(frames_dir, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                    "-movflags", "+faststart", a.out], check=True,
                   capture_output=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"[video] {a.out}  {n_out} frames  {n_out/a.fps:.0f}s  {W}x{H}")


if __name__ == "__main__":
    main()
