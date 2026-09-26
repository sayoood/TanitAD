"""Contact sheet (PNG) from a rendered mp4 -- because *.mp4 is gitignored in this repo, the
repo carries a 3x2 tile of evenly spaced frames per video (with frame indices), and the mp4
itself goes to the PI directly.

Usage: python contact_sheet.py <video.mp4> <out.png> [--n 6]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def sheet(video: Path, out: Path, n: int = 6, cols: int = 3, scale: float = 0.5) -> None:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit(f"no frames readable in {video}")
    idx = [int(round(i * (total - 1) / max(n - 1, 1))) for i in range(n)]
    tiles = []
    for k in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.rectangle(fr, (0, 0), (150, 26), (0, 0, 0), -1)
        cv2.putText(fr, f"frame {k}/{total-1}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(fr)
    cap.release()
    if not tiles:
        raise SystemExit("no tiles")
    h, w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"SHEET {out}  tiles={len(tiles)}  from {total} frames  {canvas.shape[1]}x{canvas.shape[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    sheet(Path(a.video), Path(a.out), a.n)
