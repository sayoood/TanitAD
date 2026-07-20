"""Empirical cosmos camera-projection probe.

Renders a few cosmos-val stills with the GT ego-future-path projected under the
current flat-ground pinhole (CY horizon, CAM_H) so we can SEE whether the path
tracks the road / lane and where the true horizon sits. Also draws candidate
horizon rows so the vertical offset can be read off directly.
"""
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")

from taniteval.cam_overlay import ego_future_path, CAM_H, F_EFF  # noqa: E402
from tanitad.data.mixing import load_episode                      # noqa: E402

ROOT = "/root/valdata/cosmos-val-e8f3cef4976b"
OUT = Path("/root/taniteval/results/videos/_calib_probe")
OUT.mkdir(parents=True, exist_ok=True)
UP = 2
S = 256 * UP


def _font(sz):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def project_cy(pts, cy, cam_h=CAM_H, f=F_EFF, cx=128.0):
    out = []
    for p in pts:
        X, Y = float(p[0]), float(p[1])
        if X < 1.2:
            continue
        out.append(((cx - f * Y / X) * UP, (cy + f * cam_h / X) * UP))
    return out


def main():
    idxs = [int(a) for a in sys.argv[1:]] or [17, 27]
    files = sorted(Path(ROOT).glob("ep_*.pt"))
    F = _font(13)
    for idx in idxs:
        ep = load_episode(str(files[idx]), mmap=True)
        poses = ep.poses.float()
        T = ep.frames.shape[0]
        print(f"ep{idx}: T={T} poses={tuple(poses.shape)} "
              f"v0_mean={float(poses[:,3].mean()):.1f} "
              f"net_yaw_deg={float((poses[-1,2]-poses[0,2])*57.3):.1f}")
        for t in (T // 3, T // 2):
            rgb = ep.frames[t, -3:].permute(1, 2, 0).numpy()
            im = Image.fromarray(rgb).resize((S, S), Image.LANCZOS).convert("RGB")
            d = ImageDraw.Draw(im)
            gt = ego_future_path(poses, t, min(20, T - t - 1))
            # candidate horizon rows: 128 (current) plus a low one
            for cy, col in ((128.0, (90, 160, 255)), (170.0, (255, 210, 70))):
                d.line([(0, cy * UP), (S, cy * UP)], fill=col, width=1)
                d.text((4, cy * UP + 2), f"row {int(cy)}", fill=col, font=F)
                g = project_cy(gt, cy)
                if len(g) >= 2:
                    d.line(g, fill=col, width=3)
            # GT with current model (green, CY=128) drawn thicker on top
            g = project_cy(gt, 128.0)
            if len(g) >= 2:
                d.line(g, fill=(110, 235, 131), width=4)
            for x, y in g[::4]:
                d.ellipse([x-3, y-3, x+3, y+3], fill=(110, 235, 131))
            d.text((4, 4), f"cosmos ep{idx} f{t} · green=GT@row128 · "
                   f"yellow=GT@row170", fill=(233, 237, 243), font=F)
            p = OUT / f"probe_ep{idx:02d}_f{t:03d}.png"
            im.save(p)
            print(f"  saved {p}")


if __name__ == "__main__":
    main()
