#!/usr/bin/env python3
"""CPU sky pre-screen (BEHAVIOUR_SCORECARD 3.1 QC bullet 1): sky-visible fraction of the TOP THIRD of CAM_F0.

ESTIMATED proxy, UNVERIFIED that it predicts the lifter's failure (the sim's QC render decides).  Method (colour + texture, no learned model):
  1. downscale the 1920x1080 frame to 480x270 (area), keep rows [0, 90) = the top third;
  2. HSV (OpenCV, H in [0,180) = degrees/2);
  3. sky pixel = smooth (3x3 local std of the gray value <= 5.0 of 255)  AND  EITHER
        clear/blue sky : hue 95..130 (190..260 deg), S >= 0.12*255, V >= 0.35*255
        overcast sky   : S <= 0.12*255 and V >= 0.70*255   (bright, grey-white)
     the smoothness test removes bright facades and lit road markings; the overcast branch can still accept a white wall (documented false positive);
  4. fraction = sky pixels / pixels of the top third.
The scorecard rule: fraction >= 0.5 on >= 90 % of the frames of the first 6 s (13 key frames at 2 Hz incl. t0).
Thresholds above were fixed BEFORE looking at candidate frames; they were checked by eye on 16 frames of the cruise bundle (see SKY_METHOD_CHECK.png).
"""
from __future__ import annotations

import numpy as np
import cv2

SMALL = (480, 270)
TOP_ROWS = 90
STD_MAX = 5.0
BLUE_H = (95, 130)
BLUE_S_MIN, BLUE_V_MIN = 0.12 * 255, 0.35 * 255
GREY_S_MAX, GREY_V_MIN = 0.12 * 255, 0.70 * 255
FRAC_MIN, SHARE_MIN = 0.5, 0.9


def sky_mask(bgr):
    sm = cv2.resize(bgr, SMALL, interpolation=cv2.INTER_AREA)[:TOP_ROWS]
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV).astype(np.float32)
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m1 = cv2.blur(g, (3, 3))
    m2 = cv2.blur(g * g, (3, 3))
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (H >= BLUE_H[0]) & (H <= BLUE_H[1]) & (S >= BLUE_S_MIN) & (V >= BLUE_V_MIN)
    grey = (S <= GREY_S_MAX) & (V >= GREY_V_MIN)
    return (std <= STD_MAX) & (blue | grey), sm


def sky_fraction(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    m, _ = sky_mask(img)
    return float(m.mean())


def verdict(fracs):
    """fracs: list of per-frame fractions (None = frame missing)."""
    have = [f for f in fracs if f is not None]
    if not have:
        return dict(n_frames=0, n_ge_half=0, share=None, pass_proxy=None)
    n_ok = sum(f >= FRAC_MIN for f in have)
    return dict(n_frames=len(have), n_ge_half=n_ok, share=round(n_ok / len(have), 3), pass_proxy=bool(n_ok / len(have) >= SHARE_MIN),
                frac_min=round(min(have), 3), frac_median=round(float(np.median(have)), 3), frac_max=round(max(have), 3))
