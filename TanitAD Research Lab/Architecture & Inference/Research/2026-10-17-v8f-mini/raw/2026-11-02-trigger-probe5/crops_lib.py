"""crops_lib.py -- geometry + decode layer of trigger probe 5 (PREREG_FROZEN.md section 2).  Dev box, CPU.
Native-scale cylindrical frame 768x1920 (f_ref = 3 x 305.5775, hfov 120); TL / LEAD crops cut by slicing the rectification GRID before grid_sample (pixel-identical to slicing the full rectification)."""
from __future__ import annotations

import io
import os
import sys

import numpy as np

STACK = os.environ.get("TANITAD_STACK", "E:/Projects/TanitAD/stack")
if STACK not in sys.path:
    sys.path.insert(0, STACK)

PAI_ROOT = "C:/Users/Admin/tanitad-data/physicalai"
MP4_DIR = f"{PAI_ROOT}/camera/camera_front_wide_120fov"
HFOV = 120.0
FULL_H, FULL_W = 768, 1920            # native-scale cylindrical frame (3 x the cache's 256x640)
CACHE_H, CACHE_W = 256, 640           # the cache's own frame
BOXES = {"TL": (0, 336, 384, 1536), "LEAD": (272, 608, 480, 1440)}      # (r0, r1, c0, c1) in the 768x1920 frame
CROPS = ("TL", "LEAD")
JPEG_Q = 95


def frame_full():
    from tanitad.data.calib import CanonicalFrame
    return CanonicalFrame.from_hfov(HFOV, FULL_H, FULL_W, "cylindrical")


def frame_cache():
    from tanitad.data.calib import CanonicalFrame
    return CanonicalFrame.from_hfov(HFOV, CACHE_H, CACHE_W, "cylindrical")


def check_boxes():
    for k, (r0, r1, c0, c1) in BOXES.items():
        assert all(v % 16 == 0 for v in (r0, r1, c0, c1)), k
        assert 0 <= r0 < r1 <= FULL_H and 0 <= c0 < c1 <= FULL_W, k
    return {k: ((r1 - r0) // 16, (c1 - c0) // 16) for k, (r0, r1, c0, c1) in BOXES.items()}


def frame_idx_for(timestamps_parquet, target_hz=10.0):
    """The v2ep recipe (v2_compressed._resampled): 10 Hz query times on the timestamp track -> raw mp4 frame index of v2ep frame i."""
    import pandas as pd
    ts = pd.read_parquet(timestamps_parquet)
    tcol = next(c for c in ts.columns if "time" in c.lower())
    t_frames = ts[tcol].to_numpy(np.float64)
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * target_hz), 4)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    return np.searchsorted(t_frames, t_query).clip(0, len(t_frames) - 1)


def raw_frames(mp4, raw_idx, threads=2):
    """uint8 [n,3,1080,1920] rgb of the raw mp4 frames in raw_idx (order kept)."""
    import av
    import torch
    need = sorted(set(int(i) for i in raw_idx))
    got = {}
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        st.thread_count = threads
        last = need[-1]
        for fi, vf in enumerate(c.decode(st)):
            if fi in set(need):
                got[fi] = torch.from_numpy(vf.to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous()
            if fi >= last:
                break
    assert len(got) == len(need), (mp4, len(got), len(need))
    return torch.stack([got[int(i)] for i in raw_idx])


def rectify(vid, intr, frame, box=None):
    """cylindrical_rectify of calib.py, optionally restricted to a box of the output frame by slicing the grid FIRST (grid_sample is pointwise in the output)."""
    import torch
    import torch.nn.functional as F
    from tanitad.data.calib import cylindrical_grid
    h, w = int(vid.shape[-2]), int(vid.shape[-1])
    grid, mask = cylindrical_grid(intr, h, w, frame)
    if box is not None:
        r0, r1, c0, c1 = box
        grid, mask = grid[:, r0:r1, c0:c1], mask[r0:r1, c0:c1]
    out = F.grid_sample(vid.float(), grid.expand(vid.shape[0], -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=False)
    out = out * mask.to(out.dtype)
    return out.clamp(0, 255).to(torch.uint8)


def intr_for(clip):
    from tanitad.data.physicalai import intrinsics_for_clip
    intr = intrinsics_for_clip(clip, PAI_ROOT)
    assert intr.per_clip, clip
    return intr


def jpeg_bytes(chw_u8):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(chw_u8.permute(1, 2, 0).numpy())).save(b, "JPEG", quality=JPEG_Q, subsampling=0)
    return b.getvalue()
