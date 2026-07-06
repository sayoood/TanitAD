"""Visualize an episode: frames with action/ground-truth overlay -> MP4 + contact sheet.

Per frame overlay (bottom bar):
  - steering dial: a line rotated by the road-wheel angle (x5 exaggeration)
  - accel bar: green (accelerating) / red (braking), scaled to +-4 m/s^2
  - text: v [m/s], steer [deg], accel [m/s^2], step index
  - trajectory inset (top-right): pose (x, y) path with current position dot
The frame shown is the CURRENT frame (last 3 channels of the D-015 stack).

Usage:
  python scripts/visualize_episode.py --source comma2k19 --path <segment_dir> --out ep.mp4
  python scripts/visualize_episode.py --source physicalai --path <r0_root> --clip-index 0 --out ep.mp4
  python scripts/visualize_episode.py --source pt --path episode.pt --out ep.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

BAR_H = 72
SHEET_COLS = 4


def load_episode(source: str, path: str, clip_index: int):
    if source == "comma2k19":
        from tanitad.data.comma2k19 import build_episode
        return build_episode(Path(path), size=256, max_steps=200)
    if source == "physicalai":
        from tanitad.data.physicalai import build_episode, discover_r0_clips
        clips = discover_r0_clips(path)
        assert clips, f"no R0 clips under {path}"
        return build_episode(clips[clip_index], size=256)
    if source == "pt":
        from tanitad.data.mixing import load_episode as load_pt
        return load_pt(path)
    raise ValueError(source)


def frame_to_pil(frame9: torch.Tensor) -> Image.Image:
    """Current frame = LAST 3 channels of the D-015 stack (1ch toy: replicate)."""
    c = frame9.shape[0]
    rgb = frame9[-3:] if c >= 3 else frame9.expand(3, -1, -1)
    if rgb.dtype == torch.uint8:
        arr = rgb.permute(1, 2, 0).numpy()
    else:
        arr = (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr).resize((512, 512), Image.BILINEAR)


def draw_overlay(img: Image.Image, action, pose, poses_xy, t: int,
                 total: int) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + BAR_H), (12, 12, 16))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    steer, accel = float(action[0]), float(action[1])
    v = float(pose[3])

    # steering dial (x5 exaggeration for visibility)
    cx, cy, r = 60, h + BAR_H // 2, 26
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(120, 120, 130), width=2)
    ang = -steer * 5.0 + math.pi / 2
    d.line([cx, cy, cx + r * math.cos(ang), cy - r * math.sin(ang)],
           fill=(80, 180, 255), width=4)

    # accel bar: center-anchored, +-4 m/s^2 full-scale
    bx, bw2 = 140, 90
    frac = max(-1.0, min(1.0, accel / 4.0))
    d.rectangle([bx, cy - 8, bx + 2 * bw2, cy + 8], outline=(120, 120, 130))
    color = (60, 200, 90) if frac >= 0 else (230, 70, 70)
    x0, x1 = sorted([bx + bw2, bx + bw2 + frac * bw2])
    d.rectangle([x0, cy - 7, x1, cy + 7], fill=color)

    d.text((bx + 2 * bw2 + 16, cy - 22),
           f"v {v:5.1f} m/s   steer {math.degrees(steer):+6.2f} deg   "
           f"accel {accel:+5.2f} m/s^2   t {t}/{total}",
           fill=(230, 230, 235))

    # trajectory inset (top-right)
    if poses_xy is not None and len(poses_xy) > 2:
        iw = 130
        pad = 8
        box = [w - iw - pad, pad, w - pad, pad + iw]
        d.rectangle(box, fill=(10, 10, 14), outline=(120, 120, 130))
        xy = poses_xy - poses_xy.min(axis=0)
        span = max(float(xy.max()), 1e-3)
        pts = [(box[0] + 6 + p[0] / span * (iw - 12),
                box[3] - 6 - p[1] / span * (iw - 12)) for p in xy]
        d.line(pts, fill=(90, 160, 240), width=2)
        px, py = pts[min(t, len(pts) - 1)]
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 200, 60))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["comma2k19", "physicalai", "pt"],
                    required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--out", default="episode_viz.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    ep = load_episode(args.source, args.path, args.clip_index)
    poses_xy = ep.poses[:, :2].numpy()
    n = min(args.max_frames, ep.frames.shape[0])

    import av
    first = draw_overlay(frame_to_pil(ep.frames[0]), ep.actions[0], ep.poses[0],
                         poses_xy, 0, n)
    with av.open(args.out, "w") as container:
        stream = container.add_stream("h264", rate=args.fps)
        stream.width, stream.height = first.size
        stream.pix_fmt = "yuv420p"
        sheet_tiles = []
        for t in range(n):
            img = draw_overlay(frame_to_pil(ep.frames[t]), ep.actions[t],
                               ep.poses[t], poses_xy, t, n)
            if t % max(1, n // (SHEET_COLS * 4)) == 0 and len(sheet_tiles) < 16:
                sheet_tiles.append(img.resize((img.width // 2, img.height // 2)))
            frame = av.VideoFrame.from_image(img)
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)

    tw, th = sheet_tiles[0].size
    rows = (len(sheet_tiles) + SHEET_COLS - 1) // SHEET_COLS
    sheet = Image.new("RGB", (SHEET_COLS * tw, rows * th), (0, 0, 0))
    for i, tile in enumerate(sheet_tiles):
        sheet.paste(tile, ((i % SHEET_COLS) * tw, (i // SHEET_COLS) * th))
    sheet_path = str(Path(args.out).with_suffix("")) + "_sheet.png"
    sheet.save(sheet_path)
    print(f"wrote {args.out} ({n} frames) and {sheet_path}")


if __name__ == "__main__":
    main()
