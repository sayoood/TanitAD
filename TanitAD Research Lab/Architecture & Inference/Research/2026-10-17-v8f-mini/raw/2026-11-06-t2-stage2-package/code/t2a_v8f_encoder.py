"""E2 -- the in-loop DINOv3-L encoder for the v8F-mini T2 re-perception rollout.

It reproduces, from ONE front-wide frame, the canon that built box 2's training cache (`cache_full`) and the eval-141 cache.
The route is MEASURED, not assumed (`raw/2026-10-25-lora-route/ROUTE.md`, `manifest_*.jsonl`: every row `source=local256`,
`enc_dtype=bf16`, `pipeline=ra1a-pooled-v1`, 0 rows thor416):

  1. f-theta 1920x1080 frame + PER-CLIP intrinsics -> 256x640 CYLINDER, hfov 120 deg, f_ref 305.5774907364391, ONE
     `cylindrical_rectify` (bilinear grid_sample, no prefilter, float -> uint8 by truncation). The same call the kit's
     `_BasePolicy._canon_cyl` makes. (A frame that is ALREADY the cylinder -- the cache's own v2ep PNG -- skips this step.)
  2. x = uint8.to(bf16) / 255 ; (x - mean) / std with the ImageNet constants of the model's processor.
  3. DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, HF `AutoModel`, bf16) -> last_hidden_state, the first
     5 tokens (CLS + 4 registers) dropped -> fp32 [640, 1024] (a 16 x 40 patch grid, row-major).
  4. fp32 average over equal 4 x 4 token cells -> [40, 1024] (a 4 x 10 grid), THEN `.to(float8_e4m3fn)` (pool-then-quantise).

The model reads the stored fp8 as fp16 (`ClipStore.get`: `o[grid].to(torch.float16)`); `InLoopEncoder.tokens` returns the fp8 tensor
and `as_model_rows` does that cast, so the policy sees exactly what a cache read would give.

WEIGHTS come from a local offline snapshot only (HF_HUB_OFFLINE=1, `local_files_only=True`); no token file is read and no network
call is made. A snapshot path on the G: mount is refused. DEVICE: the cache is GPU bf16; on CPU the same bf16 forward differs from
it by a device floor of rel-L2 ~0.03 (ROUTE.md, measured), which is the residue the G2 gate reports.
"""
from __future__ import annotations

import glob
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
GH, GW = 16, 40                  # the patch grid of the 256x640 cylinder
POOL_GH, POOL_GW = 4, 10         # the pooled grid the v8F cache stores (pool4x10)
N_SPECIAL = 5                    # CLS + 4 registers
FRAME_HW = (256, 640)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
HF_HUB_DEFAULT = "C:/Users/Admin/.cache/huggingface/hub"


def refuse_g_path(p) -> str:
    s = str(p).replace("\\", "/")
    if s.upper().startswith("G:") or "meine ablage" in s.lower() or "saybou" in s.lower():
        raise RuntimeError(f"REFUSED: {s!r} is on the G: mount (the programme moved to D:/E:/F:)")
    return str(p)


def find_local_snapshot(hub: str = HF_HUB_DEFAULT) -> str:
    """The newest local snapshot of the DINOv3-L model that holds both config.json and model.safetensors. Offline: a directory listing only."""
    root = refuse_g_path(hub)
    cands = sorted(glob.glob(os.path.join(root, "models--" + MID.replace("/", "--"), "snapshots", "*")))
    cands = [c for c in cands if os.path.isfile(os.path.join(c, "model.safetensors")) and os.path.isfile(os.path.join(c, "config.json"))]
    if not cands:
        raise FileNotFoundError(f"no offline snapshot of {MID} under {root}; this encoder never downloads")
    return cands[-1]


def pool_field(f: torch.Tensor, gh: int = POOL_GH, gw: int = POOL_GW) -> torch.Tensor:
    """[T, 640, D] fp32 (row-major 16x40) -> [T, gh*gw, D] by non-overlapping average pooling, cells row-major
    (`pull_encode_pooled.pool_field`, re-spelled; the G2 test pins it against an independent loop)."""
    T, N, D = f.shape
    assert N == GH * GW and GH % gh == 0 and GW % gw == 0
    return f.reshape(T, gh, GH // gh, gw, GW // gw, D).mean(dim=(2, 4)).reshape(T, gh * gw, D)


def fp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> float8_e4m3fn -> fp32: what a pooled cache entry holds (e4m3fn has no inf: > 448 becomes NaN, which is asserted away)."""
    q = x.to(torch.float8_e4m3fn)
    assert bool(torch.isfinite(q.float()).all()), "non-finite after fp8 e4m3"
    return q.float()


def as_model_rows(tok_fp8: torch.Tensor) -> torch.Tensor:
    """The policy's feature row from a stored fp8 token set: the cache reader's own cast (fp8 -> fp16)."""
    return tok_fp8.to(torch.float16)


def rel_l2(a, b) -> float:
    a, b = torch.as_tensor(a).double().flatten(), torch.as_tensor(b).double().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def cosine(a, b) -> float:
    a, b = torch.as_tensor(a).double().flatten(), torch.as_tensor(b).double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


class InLoopEncoder:
    """front-wide frame -> pooled fp8 tokens [40, 1024]. Stateless apart from the loaded weights."""

    def __init__(self, snapshot: str | None = None, dtype: str = "bf16", device: str = "cpu", n_special: int = N_SPECIAL):
        from transformers import AutoImageProcessor, AutoModel
        self.snapshot = refuse_g_path(snapshot or find_local_snapshot())
        self.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.device = torch.device(device)
        proc = AutoImageProcessor.from_pretrained(self.snapshot, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.snapshot, local_files_only=True, dtype=self.dtype).eval().to(self.device)
        self.mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(self.device, self.dtype)
        self.std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(self.device, self.dtype)
        n_true = 1 + getattr(self.model.config, "num_register_tokens", 0)
        if tuple(proc.image_mean) != IMAGENET_MEAN or n_true != N_SPECIAL:
            raise RuntimeError(f"encoder geometry differs from the cache's: mean {tuple(proc.image_mean)}, special tokens {n_true}")
        self.n_special = int(n_special)          # a test may pass a wrong value to prove the gate sees it
        self.frame = None
        self.canon = {}

    # -- step 1 ------------------------------------------------------------------------------------------------------
    def rectify(self, frame_hwc_u8: np.ndarray, intr) -> torch.Tensor:
        """f-theta [H, W, 3] uint8 + `FThetaIntrinsics(per_clip=True)` -> uint8 [3, 256, 640] cylinder (the kit's `_canon_cyl` call)."""
        from tanitad.data.calib import PHYSICALAI_WIDE120_256x640, cylindrical_rectify
        fr = PHYSICALAI_WIDE120_256x640
        vid = torch.from_numpy(np.ascontiguousarray(frame_hwc_u8))[None].permute(0, 3, 1, 2)
        cyl = cylindrical_rectify(vid, intr, fr)
        f_eff, obs = float(cylindrical_rectify.last_f_eff), float(cylindrical_rectify.last_observed_frac)
        if abs(f_eff - fr.f_ref) > 1e-9 or not (obs > 0.85) or tuple(cyl.shape[-2:]) != FRAME_HW:
            raise RuntimeError(f"256x640 cylinder self-check FAILED (f_eff={f_eff}, observed_frac={obs}, shape={tuple(cyl.shape)})")
        self.canon = {"mode": "cyl256x640", "f_eff": f_eff, "f_ref": fr.f_ref, "observed_frac": obs, "projection": fr.projection,
                      "intr_cx": float(intr.cx), "intr_cy": float(intr.cy), "intr_per_clip": bool(intr.per_clip)}
        return cyl[0]

    # -- steps 2-3 (`dinov3_fp8_encode_ship.encode_episode` body) -------------------------------------------------------
    def encode_cyl(self, cyl_u8: torch.Tensor, batch: int = 8) -> torch.Tensor:
        """uint8 [N, 3, 256, 640] -> fp32 [N, 640, 1024] patch field."""
        if tuple(cyl_u8.shape[1:]) != (3, *FRAME_HW) or cyl_u8.dtype != torch.uint8:
            raise ValueError(f"expected uint8 [N,3,256,640], got {tuple(cyl_u8.shape)} {cyl_u8.dtype}")
        out = []
        with torch.inference_mode():
            for j in range(0, cyl_u8.shape[0], batch):
                x = cyl_u8[j:j + batch].to(self.device).to(self.mean.dtype) / 255.0
                h = self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state
                out.append(h[:, self.n_special:, :].cpu())
        f = torch.cat(out).float()
        assert f.shape[1:] == (GH * GW, 1024), f"token grid {tuple(f.shape[1:])}: a wrong special-token count shifts it"
        assert bool(torch.isfinite(f).all()) and float(f.abs().mean()) > 1e-3
        return f

    # -- step 4 ----------------------------------------------------------------------------------------------------
    def tokens(self, frame, intr=None, *, already_cyl: bool = False) -> torch.Tensor:
        """One frame -> fp8 [40, 1024]. `already_cyl`: `frame` is the uint8 [3,256,640] (or [256,640,3]) cylinder; else an f-theta [H,W,3] frame and `intr`."""
        if already_cyl:
            c = torch.as_tensor(np.asarray(frame)) if not torch.is_tensor(frame) else frame
            cyl = c.permute(2, 0, 1) if c.shape[-1] == 3 else c
        else:
            if intr is None:
                raise ValueError("an f-theta frame needs its per-clip intrinsics")
            cyl = self.rectify(frame, intr)
        f = self.encode_cyl(cyl[None].contiguous())
        return pool_field(f).to(torch.float8_e4m3fn)[0].contiguous()

    def __call__(self, frame, intr=None):
        return self.tokens(frame, intr)


class StubEncoder:
    """The blocker-shaped stand-in: used where weights are absent. Returns a fixed fp8 token set derived from the frame bytes (NOT features)."""
    snapshot = None
    canon = {"mode": "stub"}

    def tokens(self, frame, intr=None, *, already_cyl: bool = False):
        h = int(np.asarray(frame).astype(np.uint64).sum() % (2 ** 31))
        g = torch.Generator().manual_seed(h)
        return (torch.randn(POOL_GH * POOL_GW, 1024, generator=g)).to(torch.float8_e4m3fn)

    __call__ = lambda self, frame, intr=None: self.tokens(frame, intr)
