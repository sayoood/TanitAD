"""Verify a corrected pinhole projection for cosmos: does the GT ego path track
the lane on a curve + vanish at the true horizon? Sweep (CY, H)."""
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
from taniteval.cam_overlay import ego_future_path  # noqa
from tanitad.data.mixing import load_episode        # noqa

ROOT = "/root/valdata/cosmos-val-e8f3cef4976b"
OUT = Path("/root/taniteval/results/videos/_calib_probe")
OUT.mkdir(parents=True, exist_ok=True)
UP = 2
S = 256 * UP
F_EFF, CX = 265.83, 128.0


def _font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def project(pts, cy, H, f=F_EFF, cx=CX):
    out = []
    for p in pts:
        X, Y = float(p[0]), float(p[1])
        if X < 1.2:
            continue
        out.append(((cx - f * Y / X) * UP, (cy + f * H / X) * UP))
    return out


F = _font(13)
# (idx, frame) pairs: curve clip and straight highway
jobs = [(18, 10), (18, 6), (17, 10), (0, 12)]
combos = [(180, 1.5, (110, 235, 131)), (180, 2.0, (255, 190, 60)),
          (178, 1.25, (80, 170, 255)), (185, 1.8, (240, 90, 200))]
for idx, t in jobs:
    ep = load_episode(str(sorted(Path(ROOT).glob("ep_*.pt"))[idx]), mmap=True)
    poses = ep.poses.float()
    rgb = ep.frames[t, -3:].permute(1, 2, 0).numpy()
    im = Image.fromarray(rgb).resize((S, S), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    gt = ego_future_path(poses, t, min(20, poses.shape[0] - t - 1))
    y = 20
    for cy, H, col in combos:
        g = project(gt, cy, H)
        if len(g) >= 2:
            d.line(g, fill=col, width=3)
        for x, yy in g[::4]:
            d.ellipse([x-3, yy-3, x+3, yy+3], fill=col)
        d.line([(0, cy*UP), (S, cy*UP)], fill=col, width=1)
        d.text((6, y), f"CY={cy} H={H}", fill=col, font=F)
        y += 15
    d.text((6, 4), f"cosmos ep{idx} f{t}", fill=(255, 255, 255), font=F)
    p = OUT / f"fit_ep{idx:02d}_f{t:03d}.png"
    im.save(p)
    print(f"saved {p}")
