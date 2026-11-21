"""Shared utilities for the YouTube-IDM pilot (privacy-safe harvest -> episode
contract). Runs on pod3 with PYTHONPATH=/workspace/TanitAD/stack so it can reuse
the EXACT preprocessing the training corpora use:

  * tanitad.data.calib.focal_crop_resize  -> canonical effective focal F_REF=266
  * tanitad.data.comma2k19.stack_frames   -> the 3-frame [t-2,t-1,t] 9-channel stack

The only YouTube-specific pieces live here:
  * CC-license verification (yt-dlp `license` field)
  * face + license-plate blur (OpenCV Haar cascades) applied to FULL-RES frames
    BEFORE the canonical crop/resize, so identifiable regions are destroyed
    before any pixel is stored
  * fps-agnostic resampling to 10 Hz by nearest-timestamp selection
  * a cheap shot-cut filter so compilation cuts don't inject bogus ego-motion

DESIGN (privacy): the ONLY imagery that ever touches disk is the transient,
downscaled (256x256), face/plate-blurred stacked frames used to compute latents;
those are deleted after encoding. The PERSISTENT artifacts are latents (z; 2048-d
vectors, not human-viewable), pseudo-labels (numbers), and URL+timestamp pointers
to already-public CC uploads. No raw video and no full-res frames are kept.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

# ---- exact training-corpus preprocessing (import from the stack) ------------ #
from tanitad.data.calib import focal_crop_resize          # canonical focal crop
from tanitad.data.comma2k19 import stack_frames           # 3-frame 9ch stack

CC_LICENSE = "Creative Commons Attribution license (reuse allowed)"
TARGET_HZ = 10.0
SIZE = 256
N_STACK = 3
# YouTube dashcams have UNKNOWN intrinsics. We assume a nominal horizontal FOV
# and crop to the SAME canonical half-angle comma2k19 achieves (f_eff=266). This
# is the pilot's key geometry approximation and a NAMED domain-shift source: if
# the true HFOV differs, the apparent-motion scale (hence pseudo speed) is biased.
DEFAULT_HFOV_DEG = 100.0

import math


def nominal_focal_px(width: int, hfov_deg: float) -> float:
    return width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


# --------------------------------------------------------------------------- #
# license                                                                     #
# --------------------------------------------------------------------------- #
def is_creative_commons(info: dict) -> bool:
    """True only if yt-dlp's metadata marks the upload as the reuse-allowed CC
    license. Anything else (incl. None / 'Standard YouTube License') is refused."""
    return info.get("license") == CC_LICENSE


# --------------------------------------------------------------------------- #
# privacy: face + license-plate blur (Haar cascades on the full-res frame)     #
# --------------------------------------------------------------------------- #
class Anonymizer:
    """Detect faces + license plates (+ bodies as a face-miss backstop) with
    OpenCV Haar cascades and Gaussian-blur each region. Applied to the full-res
    RGB frame BEFORE the downscale, so plate/face detail never reaches disk."""

    def __init__(self, cascade_dir: str | None = None, blur_bodies: bool = True,
                 det_max_side: int = 512, detect_every: int = 2):
        import cv2
        self.detect_every = max(1, detect_every)   # re-detect every Nth frame; carry
        self._last_boxes = []                      # forward boxes between (10 Hz)
        self._i = 0
        base = cascade_dir or cv2.data.haarcascades
        def load(name, fallback_dir="/workspace/tmp/yt_pilot/cascades"):
            p = os.path.join(base, name)
            if not os.path.exists(p):
                p = os.path.join(fallback_dir, name)
            c = cv2.CascadeClassifier(p)
            return None if c.empty() else c
        self.cv2 = cv2
        self.det_max_side = det_max_side          # detect on a downscaled gray for speed
        # lean set: frontal+profile faces, one plate, one body backstop. Blur is
        # still applied at FULL RES; only DETECTION runs on the downscaled gray.
        self.face = [c for c in (load("haarcascade_frontalface_default.xml"),
                                 load("haarcascade_profileface.xml")) if c]
        self.plate = [c for c in (load("haarcascade_russian_plate_number.xml"),) if c]
        self.body = ([c for c in (load("haarcascade_upperbody.xml"),) if c]
                     if blur_bodies else [])
        if not self.face or not self.plate:
            raise RuntimeError(
                "privacy pass unavailable: face and/or plate cascades failed to "
                "load — refusing to store footage (escalate per brief).")
        self.stats = {"faces": 0, "plates": 0, "bodies": 0, "frames": 0}

    def reset(self):
        """Reset per-video stats + carry-forward state (call before each video)."""
        self.stats = {"faces": 0, "plates": 0, "bodies": 0, "frames": 0}
        self._last_boxes = []
        self._i = 0

    def _detect(self, gray, cascades, min_size):
        boxes = []
        for c in cascades:
            for (x, y, w, h) in c.detectMultiScale(gray, scaleFactor=1.15,
                                                   minNeighbors=4,
                                                   minSize=min_size):
                boxes.append((int(x), int(y), int(w), int(h)))
        return boxes

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb [H,W,3] uint8 -> same, with faces/plates/bodies blurred in place.
        Detection runs on a gray image downscaled so its long side <= det_max_side
        (boxes scaled back to full res); blur is applied at full res."""
        cv2 = self.cv2
        H, W = rgb.shape[:2]
        self.stats["frames"] += 1
        if self._i % self.detect_every == 0:              # re-detect
            scale = min(1.0, self.det_max_side / float(max(H, W)))
            gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gray = (cv2.resize(gray_full, (int(W * scale), int(H * scale)))
                    if scale < 1.0 else gray_full)
            inv = 1.0 / scale
            raw = (self._detect(gray, self.face, (16, 16)),
                   self._detect(gray, self.plate, (14, 7)),
                   self._detect(gray, self.body, (24, 48)) if self.body else [])
            self.stats["faces"] += len(raw[0])
            self.stats["plates"] += len(raw[1])
            self.stats["bodies"] += len(raw[2])
            self._last_boxes = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv))
                                for group in raw for (x, y, w, h) in group]
        self._i += 1
        for (x, y, w, h) in self._last_boxes:             # carry-forward on skip frames
            pad = int(0.25 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            roi = rgb[y0:y1, x0:x1]
            if roi.size:
                k = max(9, (max(roi.shape[:2]) // 3) | 1)     # odd kernel
                rgb[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
        return rgb


# --------------------------------------------------------------------------- #
# decode + resample + anonymize + canonical crop  -> [T,3,256,256] uint8        #
# --------------------------------------------------------------------------- #
def decode_canonical(mp4_path: str, anonymizer: Anonymizer, *,
                     hfov_deg: float = DEFAULT_HFOV_DEG, size: int = SIZE,
                     target_hz: float = TARGET_HZ,
                     max_frames: int | None = None) -> tuple[torch.Tensor, dict]:
    """Decode video, resample to ~target_hz by nearest presentation timestamp,
    blur faces/plates on the full-res RGB, then canonical focal crop+resize to
    [T,3,size,size] uint8. Streams frame-by-frame — never holds full-res video.
    Returns (frames_u8 [T,3,size,size], meta)."""
    import av
    frames = []
    src_fps = None
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        try:
            src_fps = float(stream.average_rate) if stream.average_rate else None
        except Exception:
            src_fps = None
        dt = 1.0 / target_hz
        next_t = 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb)
            if t + 1e-6 < next_t:
                continue                                     # skip until next grid point
            rgb = frame.to_ndarray(format="rgb24")           # H W 3 uint8
            rgb = anonymizer(np.ascontiguousarray(rgb))      # blur in place
            H, W = rgb.shape[:2]
            f_px = nominal_focal_px(W, hfov_deg)
            vt = torch.from_numpy(rgb).permute(2, 0, 1)[None]  # [1,3,H,W]
            canon = focal_crop_resize(vt, f_px, size)          # [1,3,size,size] u8
            frames.append(canon[0])
            next_t += dt
            if max_frames is not None and len(frames) >= max_frames:
                break
    if not frames:
        raise RuntimeError("no frames decoded")
    vid = torch.stack(frames)                                 # [T,3,size,size] u8
    meta = {"src_fps": src_fps, "n_frames_10hz": int(vid.shape[0]),
            "hfov_assumed_deg": hfov_deg, "size": size,
            "anon": dict(anonymizer.stats)}
    return vid, meta


# --------------------------------------------------------------------------- #
# shot-cut score (compilation / scene-change filter)                          #
# --------------------------------------------------------------------------- #
def shotcut_score(vid_u8: torch.Tensor) -> float:
    """Max normalized frame-to-frame mean-abs-difference over a clip. High values
    flag a scene cut / compilation splice (which would inject bogus ego-motion
    into the pseudo-labels). Continuous driving stays low."""
    x = vid_u8.float().mean(1)                                # [T,S,S] luma-ish
    d = (x[1:] - x[:-1]).abs().mean(dim=(1, 2))               # [T-1]
    if d.numel() == 0:
        return 0.0
    med = float(d.median()) + 1e-3
    return float((d.max() / med))


# --------------------------------------------------------------------------- #
# pointer record (persistent, no imagery)                                     #
# --------------------------------------------------------------------------- #
def clip_pointer(info: dict, clip_idx: int, start_frame: int, n_frames: int,
                 target_hz: float, meta: dict, extra: dict | None = None) -> dict:
    start_s = start_frame / target_hz
    rec = {
        "video_id": info.get("id"),
        "url": info.get("webpage_url") or f"https://www.youtube.com/watch?v={info.get('id')}",
        "title": (info.get("title") or "")[:200],
        "uploader": info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "license": info.get("license"),
        "is_cc": is_creative_commons(info),
        "clip_idx": clip_idx,
        "start_frame_10hz": start_frame,
        "start_time_s": round(start_s, 2),
        "end_time_s": round(start_s + n_frames / target_hz, 2),
        "n_frames_10hz": n_frames,
        "src_fps": meta.get("src_fps"),
        "hfov_assumed_deg": meta.get("hfov_assumed_deg"),
        "harvested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        rec.update(extra)
    return rec
