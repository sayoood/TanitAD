"""Prepare attention-clip panels for the pitch film.

Reads the programme's own clip v8f_hierarchy_attention.mp4 (read-only copy exported from
git 24b97ba9), crops the three camera panels (S, T, O), removes the clip's own white
percentage labels (so the film shows no number that is not in FACTS_PITCH.md) by
normalised-convolution inpainting, and saves PNG sequences.  CPU only.
"""
import os, subprocess, shutil, sys
import numpy as np
from PIL import Image, ImageFilter

ASSETS = "F:/TanitAD-scratch/pitch/assets"
SRC = ASSETS + "/v8f_hierarchy_attention.mp4"
OUT = ASSETS + "/att"
ROWS = {"A": (10, 12), "B": (24, 12)}          # start s, duration s  (clip 1 junction, clip 2 curve)
PANELS = {"S": 8, "T": 648, "O": 1288}         # interior left edge (px), measured on the source frame
PW, Y0, Y1 = 624, 98, 346


def fill_local(a, mask):
    """Normalised-convolution fill of masked pixels, restricted to one grid cell."""
    valid = (~mask).astype(np.float32)
    if valid.sum() < 20:
        return a
    out, filled = a.copy(), valid.copy()
    num = a * valid[..., None]
    for r in (5, 10, 20, 40):
        nb = np.stack([np.asarray(Image.fromarray(np.clip(num[..., c], 0, 255).astype(np.uint8), "L").filter(ImageFilter.BoxBlur(r))).astype(np.float32) for c in range(3)], axis=2)
        db = np.asarray(Image.fromarray((valid * 255).astype(np.uint8), "L").filter(ImageFilter.BoxBlur(r))).astype(np.float32) / 255.0
        ok = (db > 0.12) & mask & (filled < 0.5)
        out[ok] = (nb / np.maximum(db[..., None], 1e-3))[ok]
        filled[ok] = 1.0
    return out


def clean(img):
    a = np.asarray(img).astype(np.float32)
    white = (a.min(axis=2) > 200)
    m = Image.fromarray((white * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(17))
    mask = (np.asarray(m) > 0)
    h, w = mask.shape
    out = a.copy()
    cw, ch = w / 10.0, h / 4.0
    for r in range(4):
        for c in range(10):
            y0, y1, x0, x1 = int(round(r * ch)), int(round((r + 1) * ch)), int(round(c * cw)), int(round((c + 1) * cw))
            mk = mask[y0:y1, x0:x1]
            if mk.any():
                out[y0:y1, x0:x1] = fill_local(a[y0:y1, x0:x1], mk)
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    blur = res.filter(ImageFilter.GaussianBlur(1.0))
    mk = Image.fromarray((mask * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1.5))
    return Image.composite(blur, res, mk)


def main():
    tmp = ASSETS + "/att_tmp"
    os.makedirs(OUT, exist_ok=True)
    for row, (ss, dur) in ROWS.items():
        d = f"{tmp}/{row}"
        os.makedirs(d, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(ss), "-t", str(dur), "-i", SRC,
                        "-vf", "fps=4", f"{d}/f_%03d.png"], check=True)
        files = sorted(f for f in os.listdir(d) if f.startswith("f_"))
        os.makedirs(f"{OUT}/{row}", exist_ok=True)
        for i, f in enumerate(files):
            fr = Image.open(f"{d}/{f}").convert("RGB")
            for p, x0 in PANELS.items():
                crop = fr.crop((x0, Y0, x0 + PW, Y1))
                clean(crop).save(f"{OUT}/{row}/{p}_{i:03d}.png")
        print(row, len(files), "frames", flush=True)


if __name__ == "__main__":
    main()
