"""Validation-gate diagnostic stills for the cosmos exact per-clip calibration.

Per ep: mid frame with
  GREEN   exact-chain projection of the CACHED ep poses (what videos would show)
  MAGENTA exact-chain projection of RAW vehicle_pose, chunk-offset @ native rate
          (video frame g=3*(i+2); pose index = 121*chunk + g)   [timing check]
  YELLOW  computed horizon line (row + small cross at horizon col)
  RED     the old global hack row 180 (reference)

Gate is judged visually: GT path ON the ego lane, vanishing AT the horizon.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")

from taniteval.cam_overlay import ego_future_path  # noqa: E402
from tanitad.data.mixing import load_episode        # noqa: E402

ROOT = Path("/root/valdata/cosmos-val-e8f3cef4976b")
PAIRS = Path("/root/cosmos_data/pairs")
CAL = json.loads(Path("/root/taniteval/results/cosmos_calib.json").read_text())
OUT = Path("/root/taniteval/results/videos/_calib_exact")
OUT.mkdir(parents=True, exist_ok=True)
UP = 2
S = 256 * UP


def font(sz):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


F = font(13)


class Projector:
    """Exact chain: vehicle-frame 3D point -> cached-256 pixel."""

    def __init__(self, e):
        self.R = np.array(e["R"])
        self.t = np.array(e["t"])
        self.fx, self.fy, self.cx, self.cy = e["fx"], e["fy"], e["cx"], e["cy"]
        self.vt, self.sx = e["vcrop_top"], e["sx"]
        self.c, self.top, self.left = e["crop_c"], e["crop_top"], e["crop_left"]

    def __call__(self, pts3):
        """[N,3] vehicle-frame (x fwd, y left, z up, origin at ground) -> list[(u,v)] 256-px."""
        out = []
        for p in np.asarray(pts3, dtype=np.float64):
            pc = self.R.T @ (p - self.t)
            if pc[2] < 1.0:
                continue
            u = self.fx * pc[0] / pc[2] + self.cx
            v = self.fy * pc[1] / pc[2] + self.cy
            ug = u * self.sx
            vg = (v - self.vt) * self.sx
            out.append((((ug - self.left) * 256.0 / self.c),
                        ((vg - self.top) * 256.0 / self.c)))
        return out


def raw_future_path(cid, chunk, ep_frame_i, n_fut=60):
    """Future ego path from RAW vehicle_pose with the chunk offset, in the
    vehicle frame at the video-matched pose. [M,3]."""
    veh_dir = PAIRS / "vehicle_pose" / cid
    files = sorted(veh_dir.glob("*.vehicle_pose.npy"))
    M = np.stack([np.load(f).reshape(4, 4) for f in files])
    p = 121 * chunk + 3 * (ep_frame_i + 2)
    if p >= len(M) - 2:
        return None
    T0inv = np.linalg.inv(M[p])
    fut = []
    for q in range(p + 1, min(p + 1 + n_fut, len(M))):
        fut.append((T0inv @ M[q])[:3, 3])
    return np.array(fut)


def main():
    eps = [int(a) for a in sys.argv[1:]] or None
    files = sorted(ROOT.glob("ep_*.pt"))
    for f in files:
        idx = int(f.stem.split("_")[1])
        if eps is not None and idx not in eps:
            continue
        e = CAL[str(idx)]
        if not e.get("calib"):
            print(f"ep{idx:02d}: NO CALIB ({e['reason']}) — would be gate-disabled")
            continue
        pr = Projector(e)
        ep = load_episode(str(f), mmap=True)
        poses = ep.poses.float()
        T = ep.frames.shape[0]
        for t in (T // 3, 2 * T // 3):
            rgb = ep.frames[t, -3:].permute(1, 2, 0).numpy()
            im = Image.fromarray(rgb).resize((S, S), Image.LANCZOS).convert("RGB")
            d = ImageDraw.Draw(im)
            # old hack row 180 (red) + exact horizon (yellow)
            d.line([(0, 180 * UP), (S, 180 * UP)], fill=(220, 60, 60), width=1)
            hr, hc = e["horizon_row_256"], e["horizon_col_256"]
            d.line([(0, hr * UP), (S, hr * UP)], fill=(250, 220, 60), width=1)
            d.line([(hc * UP, (hr - 6) * UP), (hc * UP, (hr + 6) * UP)],
                   fill=(250, 220, 60), width=1)
            # cached-pose GT path (green), z=0 ground
            gt = ego_future_path(poses, t, min(40, T - t - 1)).numpy()
            gt3 = np.concatenate([gt, np.zeros((len(gt), 1))], 1)
            g = [(u * UP, v * UP) for u, v in pr(gt3)]
            if len(g) >= 2:
                d.line(g, fill=(110, 235, 131), width=3)
            for u, v in g[::4]:
                d.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(110, 235, 131))
            # raw chunk-corrected path (magenta)
            raw = raw_future_path(e["clip"], e["chunk"], t)
            if raw is not None:
                m = [(u * UP, v * UP) for u, v in pr(raw)]
                if len(m) >= 2:
                    d.line(m, fill=(235, 80, 235), width=2)
            d.text((6, 4), f"ep{idx:02d} f{t:03d} ch{e['chunk']} "
                           f"f_eff={e['f_eff_256']} hrow={hr}", fill=(255, 255, 255), font=F)
            d.text((6, 20), "green=cached-GT  magenta=raw+chunkfix  "
                            "yellow=exact horizon  red=old180", fill=(255, 255, 255), font=F)
            p = OUT / f"gate_ep{idx:02d}_f{t:03d}.png"
            im.save(p)
            print("saved", p)


if __name__ == "__main__":
    main()
