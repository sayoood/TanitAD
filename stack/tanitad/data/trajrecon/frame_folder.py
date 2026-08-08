"""Adapter that makes a folder of extracted frames look like a video stream.

The already-synchronised output stores one directory per frame holding
``frame.jpg`` and a ``frame_metadata.json`` whose ``frame_offset_msec`` is the
container PTS.  That is enough to run the same optical-flow sync and camera
calibration used on the MP4, so recordings whose original video is no longer at
hand can still be processed and validated.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameFolderVideo:
    """Duck-type of :class:`trajlib.timesync.VideoInfo` backed by JPEG frames."""

    path: str
    width: int
    height: int
    duration: float
    nb_frames: int
    avg_fps: float
    creation_time_utc: float | None
    pts: np.ndarray
    frame_paths: list = field(default_factory=list)

    stem_epoch_ms = None

    def iter_gray(self, width: int, height: int, first: int = 0, count: int | None = None):
        import cv2
        last = len(self.frame_paths) if count is None else min(first + count, len(self.frame_paths))
        for i in range(first, last):
            img = cv2.imread(self.frame_paths[i], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            yield cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def read_bgr(self, index: int):
        import cv2
        return cv2.imread(self.frame_paths[index], cv2.IMREAD_COLOR)


def load_frame_folder(root: str, limit: int | None = None,
                      image_name: str = "frame.jpg") -> FrameFolderVideo:
    """Scan ``root`` for per-frame directories and order them by PTS."""
    entries = []
    for d in sorted(os.listdir(root)):
        sub = os.path.join(root, d)
        if not os.path.isdir(sub):
            continue
        img = os.path.join(sub, image_name)
        meta = os.path.join(sub, "frame_metadata.json")
        if not (os.path.exists(img) and os.path.exists(meta)):
            continue
        with open(meta) as f:
            m = json.load(f)
        entries.append((float(m["frame_offset_msec"]) / 1000.0, img))

    entries.sort(key=lambda e: e[0])
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise FileNotFoundError(f"no frame directories found under {root}")

    pts = np.array([e[0] for e in entries], dtype=float)
    paths = [e[1] for e in entries]

    import cv2
    probe = cv2.imread(paths[0])
    h, w = probe.shape[:2]
    dur = float(pts[-1] - pts[0])
    return FrameFolderVideo(
        path=root, width=int(w), height=int(h), duration=dur,
        nb_frames=len(paths), avg_fps=(len(paths) - 1) / dur if dur > 0 else 0.0,
        creation_time_utc=None, pts=pts, frame_paths=paths,
    )
