#!/usr/bin/env python3
"""CROPREAD data-side kit: shared pieces (geometry of the windows, frame plan, decode, encoder path, pooling).

Everything that the cache's own build does is IMPORTED from the stack tree, not re-written:
  rays / f-theta projection / grid mask       tanitad.data.calib (cylindrical_rays, ftheta_project_rays, cylindrical_grid)
  decode + frame choice                       scripts/v2_compressed.py (_decode_cropped_selected; the frame-index lines of _resampled are
                                              restated in frame_plan() with the source lines named, since _resampled itself builds the whole frame)
  encoder path                                the cache builder pull_encode_pooled.py (build_model, pool_field) and the ship module
                                              dinov3_fp8_encode_ship.py (encode_episode, lines 108-129): the SAME arithmetic, restated for an
                                              array of images of any size in Encoder.tokens()
No frame pixel is written anywhere by this module.
"""
import importlib.util
import math
import os
import sys

import numpy as np

POD_STACK = "/workspace/v8proof/tree/stack"
LOCAL_STACK = "E:/Projects/TanitAD/stack"
STACK = os.environ.get("TANITAD_STACK") or (POD_STACK if os.path.isdir(POD_STACK) else LOCAL_STACK)
PEP_PATH = os.environ.get("V8F_LORA_BUILDER", "/workspace/v8proof/stageA/code/pull_encode_pooled.py")
SHIP_PATH = os.environ.get("V8F_LORA_SHIP", os.path.join(STACK, "scripts", "dinov3_fp8_encode_ship.py"))

TARGET_HZ = 10.0                  # tanitad/data/physicalai.py:64
FULL_H, FULL_W, FULL_HFOV = 256, 640, 120.0     # pod_build_b1_epcache.py:83-84
WIN_PX = 256
ARMS = ("D", "B")
# arm D fixed windows (card section 3): (size_deg, azimuth_left_deg, elevation_up_deg); order W32 then W16
ARM_D_WINDOWS = ((32.0, 0.0, -6.0), (16.0, 0.0, -2.0))
SIZES_DEG = (32.0, 16.0)


def setup_paths():
    for p in (STACK, os.path.join(STACK, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return STACK


setup_paths()


def sha256_file(path, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_ids(path):
    """One clip id per line; a second column (role, position) is ignored unless it is the position column of train_order."""
    out = []
    for ln in open(path, encoding="utf-8"):
        t = ln.split()
        if not t:
            continue
        out.append(t[-1] if (len(t) == 2 and t[0].isdigit()) else t[0])
    return out


# ------------------------------------------------------------------------------------------------ frame plan
def frame_plan(ts_parquet):
    """The builder's frame choice (scripts/v2_compressed.py:_resampled, lines 111-121, verbatim arithmetic).

    Returns dict: t_frames [us], t_query [us] (n_target), frame_idx (video frame index per episode frame, n_target),
    cache_idx = frame_idx[0::2] (cache index j <-> v2ep frame 2j: pull_encode_pooled.py:82 and
    dinov3_fp8_encode_ship.py:119 `range(0, len(lens), 2)`), n_ep (= n_target), n_cache = ceil(n_ep / 2).
    """
    import pandas as pd
    ts = pd.read_parquet(ts_parquet)
    tcol = next(c for c in ts.columns if "time" in c.lower())
    t_frames = ts[tcol].to_numpy(np.float64)
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * TARGET_HZ), 4)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    frame_idx = np.searchsorted(t_frames, t_query).clip(0, len(t_frames) - 1)
    return {"t_frames": t_frames, "t_query": t_query, "frame_idx": frame_idx, "cache_idx": frame_idx[0::2],
            "n_ep": int(n_target), "n_cache": int(math.ceil(n_target / 2)), "unit": unit}


def decode_selected(mp4, need_sorted, threads=None):
    """Yield (video_frame_index, uint8 tensor [3,H,W]) for the indices in `need_sorted` (a sorted unique list), decoded with the
    same PyAV settings and the same rgb24 conversion as scripts/v2_compressed.py:_decode_cropped_selected (lines 83-96)."""
    import av
    import torch
    need = set(int(i) for i in need_sorted)
    last = max(need)
    with av.open(str(mp4)) as container:
        st = container.streams.video[0]
        st.thread_type = "AUTO"
        st.thread_count = int(threads if threads is not None else os.environ.get("PAI_DECODE_THREADS", "1"))
        fi = 0
        for vframe in container.decode(st):
            if fi in need:
                yield fi, torch.from_numpy(vframe.to_ndarray(format="rgb24")).permute(2, 0, 1)
            fi += 1
            if fi > last:
                break


# ------------------------------------------------------------------------------------------------ windows
def window_frame(size_deg, px=WIN_PX):
    """A square cylindrical CanonicalFrame whose HORIZONTAL angular size is exactly `size_deg` (azimuth is linear in the column:
    f_ref = (px/2) / radians(size/2), the same solve as CanonicalFrame.from_hfov, calib.py:174-183). The vertical axis is the
    builder's own per-column pinhole (calib.py:154-157), so the vertical half-angle is atan((px/2)/f_ref)."""
    from tanitad.data.calib import CanonicalFrame
    return CanonicalFrame.from_hfov(float(size_deg), int(px), int(px), "cylindrical")


def rotation_az_el(az_left_deg, el_up_deg):
    """Camera-axes rotation that points the centre ray at (azimuth left-positive, elevation up-positive), as float32 [...,3,3].

    Camera axes are the builder's (+x right, +y down, +z boresight; calib.py:906-913). R = Ry(-az) @ Rx(el); the centre ray then is
    (-cos(el) sin(az), -sin(el), cos(el) cos(az)); its cylinder coordinates are azimuth = atan2(-x, z) = az (left positive) and
    tan(elevation) = -y / hypot(x, z) = tan(el). So (az, el) ARE the coordinates of the rectified frame: azimuth linear in the
    column, row offset f_ref * tan(el), azimuth 0 the centre column, elevation 0 the centre row (the optical axis; the builder
    applies no mount pitch)."""
    import torch
    az = torch.as_tensor(np.asarray(az_left_deg, dtype=np.float64)).float() * (math.pi / 180.0)
    el = torch.as_tensor(np.asarray(el_up_deg, dtype=np.float64)).float() * (math.pi / 180.0)
    az, el = torch.broadcast_tensors(az, el)
    ca, sa, ce, se = torch.cos(-az), torch.sin(-az), torch.cos(el), torch.sin(el)
    z0, o0 = torch.zeros_like(az), torch.ones_like(az)
    ry = torch.stack([torch.stack([ca, z0, sa], -1), torch.stack([z0, o0, z0], -1), torch.stack([-sa, z0, ca], -1)], -2)
    rx = torch.stack([torch.stack([o0, z0, z0], -1), torch.stack([z0, ce, -se], -1), torch.stack([z0, se, ce], -1)], -2)
    return ry @ rx


def window_grid(intr, h, w, frame, az_left_deg=0.0, el_up_deg=0.0, device="cpu"):
    """grid_sample grid + observed mask for ONE window per (az, el) given as scalars or [T] arrays.

    Same chain as calib.cylindrical_grid (calib.py:937-957): rays of `frame` (cylindrical_rays) -> f-theta projection
    (ftheta_project_rays) -> native px scaled to the decoded (h, w) -> [-1,1]. The only addition is the rotation of the rays to the
    window's centre direction. With az = el = 0 NO rotation is applied, so a window whose `frame` is the 256x640 build frame returns
    exactly cylindrical_grid's grid and mask (pinned in test_enc.py). Returns (grid [T,H,W,2], mask [T,H,W] bool)."""
    import torch
    from tanitad.data.calib import cylindrical_rays, ftheta_project_rays
    x, y, z = cylindrical_rays(frame, device=device)               # [H,W] each
    az = np.atleast_1d(np.asarray(az_left_deg, dtype=np.float64))
    el = np.atleast_1d(np.asarray(el_up_deg, dtype=np.float64))
    az, el = np.broadcast_arrays(az, el)
    T = az.shape[0]
    if np.all(az == 0.0) and np.all(el == 0.0):
        xs, ys, zs = x.expand(T, -1, -1), y.expand(T, -1, -1), z.expand(T, -1, -1)
    else:
        P = torch.stack([x, y, z], -1).reshape(1, -1, 3)             # [1,HW,3]
        R = rotation_az_el(az, el).to(P.device)                      # [T,3,3]
        Q = P @ R.transpose(-1, -2)                                  # [T,HW,3]  (R @ p for each ray)
        xs, ys, zs = (Q[..., i].reshape(T, frame.height, frame.width) for i in range(3))
    u_n, v_n = ftheta_project_rays(intr, xs, ys, zs)
    sx, sy = w / float(intr.width), h / float(intr.height)
    u, v = u_n * sx, v_n * sy
    mask = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1) & (zs > 0)
    gx = u / (w - 1) * 2.0 - 1.0
    gy = v / (h - 1) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1), mask


def window_sample(vid_u8, grid, mask):
    """[T,3,h,w] uint8 + (grid [T,H,W,2], mask [T,H,W]) -> [T,3,H,W] uint8, the builder's sampling call
    (calib.cylindrical_rectify, lines 990-998: bilinear, zeros padding, align_corners=False, mask multiply, clamp, uint8)."""
    import torch
    import torch.nn.functional as F
    out = F.grid_sample(vid_u8.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = out * mask.unsqueeze(1).to(out.dtype)
    return out.clamp(0, 255).to(torch.uint8)


# ------------------------------------------------------------------------------------------------ encoder
def load_builder():
    """(pep, ship): the cache builder module and its encoder module, loaded exactly as cache_pod.load_builder does."""
    spec = importlib.util.spec_from_file_location("pull_encode_pooled", PEP_PATH)
    pep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pep)
    ship = pep.load_encoder_module(SHIP_PATH)
    return pep, ship


class Encoder:
    """The cache builder's encoder path for an array of images: bf16 weights, no autocast, batches of `bs` (the builder uses 8),
    x = img/255 in bf16, (x - mean)/std, last_hidden_state, first n_special (=5: class + 4 registers) tokens separated."""

    def __init__(self, device="cuda", bs=8):
        import torch
        self.torch = torch
        self.pep, self.ship = load_builder()
        self.model, self.mean, self.std, self.n_special = self.pep.build_model(device, "bf16")
        self.device, self.bs = device, int(bs)

    def hidden(self, imgs_u8):
        """uint8 [N,3,H,W] -> float32 cpu [N, n_special + (H/16)(W/16), 1024]."""
        torch = self.torch
        outs = []
        with torch.inference_mode():
            for j in range(0, imgs_u8.shape[0], self.bs):
                x = imgs_u8[j:j + self.bs].to(self.device).to(self.mean.dtype) / 255.0
                h = self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state
                outs.append(h.float().cpu())
        return torch.cat(outs) if outs else torch.zeros(0, 1, 1024)

    def window_tokens(self, wins_u8):
        """uint8 [N,3,256,256] -> float32 [N,65,1024]: token 0 = class token, tokens 1..64 = the 16x16 patch tokens mean-pooled
        2x2 to 8x8 in row-major order (fp32 pooling, as pull_encode_pooled.pool_field)."""
        h = self.hidden(wins_u8)
        N = h.shape[0]
        cls = h[:, 0:1, :]
        p = h[:, self.n_special:, :]
        assert p.shape[1] == 256, p.shape
        pooled = p.reshape(N, 8, 2, 8, 2, p.shape[-1]).mean(dim=(2, 4)).reshape(N, 64, p.shape[-1])
        out = self.torch.cat([cls, pooled], 1)
        assert bool(self.torch.isfinite(out).all())
        return out

    def full_pooled(self, frames_u8, gh=8, gw=20):
        """The cache's own pooling of the full 256x640 frame: float32 [N, gh*gw, 1024] (pool_field, fp32, before the fp8 cast)."""
        h = self.hidden(frames_u8)
        f = h[:, self.n_special:, :]
        assert f.shape[1] == 640, f.shape
        return self.pep.pool_field(f, gh, gw)


# ------------------------------------------------------------------------------------------------ stage layout
class Layout:
    """Where stage_clips.py puts a clip's inputs under --root (the mp4 sits below a directory named r0 so that the stack's
    _physicalai_root_of(mp4) resolves <root>/calibration/physicalai_front_wide_intrinsics.csv)."""

    def __init__(self, root):
        self.root = root
        self.cam_root = f"{root}/r0"
        self.ledger = f"{root}/ledger.jsonl"

    def mp4(self, c):
        return f"{self.cam_root}/camera/{c}.mp4"

    def ts(self, c):
        return f"{self.root}/ts/{c}.timestamps.parquet"

    def ego(self, c):
        return f"{self.root}/ego/{c}.parquet"

    def obs(self, c):
        return f"{self.root}/obs/{c}.parquet"

    @property
    def intr_csv(self):
        return f"{self.root}/calibration/physicalai_front_wide_intrinsics.csv"

    @property
    def ext_csv(self):
        return f"{self.root}/calibration/front_wide_extrinsics.csv"
