"""TanitEval — front-camera trajectory overlays + MP4.

Projects ego-frame trajectories (GT future path + REF-B's predicted waypoints)
into the front-wide camera image and renders per-frame overlays -> stills and
an MP4 (PNG sequence + ffmpeg).

Camera model: the epcache 256x256 crop was cut around the per-clip principal
point with the f-theta v2 correction; its effective focal is ~265.8 px with the
principal point at the crop centre. Near the centre (retained hfov ~51 deg) a
pinhole approximation is fine for visualization. Ground plane at camera height
H below the lens, pitch ~0 (rig-cy centring puts the horizon ~mid-image).

  u = cx - f * Y / X          (Y left -> image left)
  v = cy + f * H / X          (ground point at forward distance X)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")

F_EFF, CX, CY = 265.83, 128.0, 128.0
CAM_H = 1.5                      # metres above ground (front-wide rig approx)
UP = 2                           # render upscale
COL_GT = (120, 255, 140)
COL_PRED = (45, 212, 191)
COL_PRED_PT = (255, 120, 110)


def ego_future_path(poses, t, k=20):
    """Future GT positions t+1..t+k in the ego frame of pose t. [k,2] (x fwd, y left)."""
    x0, y0, yaw = float(poses[t, 0]), float(poses[t, 1]), float(poses[t, 2])
    c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    fut = poses[t + 1:t + 1 + k, :2] - torch.tensor([x0, y0])
    return torch.stack([c * fut[:, 0] + s * fut[:, 1],
                        -s * fut[:, 0] + c * fut[:, 1]], dim=1)


def project(pts):
    """Ego-frame ground points [N,2] -> image px [N,2] (only X>1.2 m kept)."""
    out = []
    for p in pts:
        X, Y = float(p[0]), float(p[1])
        if X < 1.2:
            continue
        out.append(((CX - F_EFF * Y / X) * UP, (CY + F_EFF * CAM_H / X) * UP))
    return out


def draw_frame(rgb_hwc, gt_path, pred_wp, label=""):
    im = Image.fromarray(rgb_hwc).resize((256 * UP, 256 * UP),
                                         Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    g = project(gt_path)
    if len(g) >= 2:
        d.line(g, fill=COL_GT, width=3)
    for x, y in g[::5]:
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=COL_GT)
    p = project(pred_wp)
    if len(p) >= 2:
        d.line(p, fill=COL_PRED, width=4)
    for x, y in p:
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=COL_PRED_PT, width=3)
    d.rectangle([0, 0, 256 * UP, 22], fill=(10, 14, 19))
    d.text((8, 5), f"GT path (green) · REF-B pred (teal)  {label}",
           fill=(233, 237, 243))
    return im


@torch.no_grad()
def refb_episode_predictions(model, ep, device, window=8, batch=16,
                             speed_input=True):
    """Stride-1 REF-B waypoint predictions for every frame. Returns dict t->wp[4,2]."""
    fr, T = ep.frames, ep.frames.shape[0]
    preds = {}
    starts = list(range(0, T - window - 20))
    for i in range(0, len(starts), batch):
        ch = starts[i:i + batch]
        last = torch.tensor([t + window - 1 for t in ch])
        fw = torch.stack([torch.as_tensor(fr[t:t + window])
                          for t in ch]).to(device).float().div_(255.0)
        v0 = ep.poses[last, 3].to(device) if speed_input else None
        out = model(fw, nav_cmd=None, v0=v0)
        wp = torch.stack([out["waypoints"][k] for k in (5, 10, 15, 20)],
                         dim=1).cpu()
        for j, t in enumerate(ch):
            preds[t + window - 1] = wp[j]
    return preds


def render_episode(model, ep, out_dir, name, device, fps=10, max_frames=160):
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    preds = refb_episode_predictions(model, ep, device)
    ts = sorted(preds.keys())[:max_frames]
    for n, t in enumerate(ts):
        rgb = ep.frames[t, -3:].permute(1, 2, 0).numpy()   # current RGB HWC
        gt = ego_future_path(ep.poses, t, 20)
        im = draw_frame(rgb, gt, preds[t], label=f"t={t / 10:.1f}s")
        im.save(out / f"f{n:04d}.png")
    mp4 = Path(out_dir) / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-r", str(fps), "-i", str(out / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    str(mp4)], check=True, capture_output=True)
    return str(mp4), len(ts)


def main():
    from tanitad.data.mixing import load_episode
    from taniteval import loaders
    from taniteval.registry import MODELS
    device = "cuda"
    e = [m for m in MODELS if m["key"] == "refb"][0]
    model = loaders.load(e, device)["model"]
    files = sorted(Path("/root/valdata/physicalai-val-0c5f7dac3b11").glob("ep_*.pt"))
    eps = [load_episode(str(f), mmap=True) for f in files[:40]]
    # pick the two most dynamic (net heading change) + one straight
    def net_turn(ep):
        return abs(float(ep.poses[-1, 2] - ep.poses[0, 2]))
    order = sorted(range(len(eps)), key=lambda i: -net_turn(eps[i]))
    picks = [(order[0], "turn1"), (order[1], "turn2"), (order[-1], "straight")]
    for idx, tag in picks:
        mp4, n = render_episode(model, eps[idx], "/root/taniteval/results/video",
                                f"refb_{tag}_ep{idx}", device)
        print(f"[video] {mp4} ({n} frames)", flush=True)
    print("VIDEO_DONE", flush=True)


if __name__ == "__main__":
    main()
