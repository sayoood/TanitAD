"""LoRA on the frozen DINOv3 ViT-L/16: the on-the-fly frame encoder of ``PREREG_V8F_LORA`` (plan §13.5 M4 / AR-2).

WHAT THIS IS. v8F-mini reads CACHED pooled DINOv3-L tokens (``pool4x10`` fp8, ``[T, 40, 1024]``), so no gradient has ever reached
the features T decides from. This module runs the SAME encoder on the fly, with rank-r LoRA adapters on the query and value
projections of every block, and returns exactly the tensor the cache holds, so the heads read the same kind of input B's heads did.

THE CACHE BUILDER'S OWN PATH (``Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py`` + ``stack/scripts/dinov3_fp8_encode_ship.py``,
READ, NOT EDITED -- ``pull_encode_pooled`` records the encoder script's sha256 in every cache's ``meta.json``, so editing it would change the
provenance of every future cache):
  * decode: PNG frames of the v2ep, cache step ``g`` <-> raw frame ``2 g``                                   (``encode_episode`` :113-117)
  * normalise IN THE ENCODER'S DTYPE: ``(x.to(bf16) / 255 - mean_bf16) / std_bf16``, mean/std from the model's own processor   (:122-123)
  * DINOv3 ViT-L/16, bf16 weights, the 256x640 cylindrical frame fed directly (16 x 40 patches), CLS + 4 registers dropped      (:124-125)
  * fp32, ``pool_field`` -> 4x10 average cells, THEN fp8 e4m3 quantisation                                   (``payload_from_field``)
``pool_field`` is IMPORTED from the builder (never re-implemented). The decode + normalise lines (six of them) are MIRRORED here, because the
builder fuses them with ``torch.inference_mode()`` (no autograd) in one function; the mirror is pinned three ways: a runtime bit-for-bit test
against the builder's own ``encode_episode`` (``tests/test_v8f_lora.py``), the sha256 of ``encode_episode``'s source
(:data:`BUILDER_ENCODE_EPISODE_SHA256`), and the constants below being the builder's own. A change to the builder turns those red.

THE IDENTITY PROPERTY (the arm's anchor): ``B`` is zero-initialised, so at step 0 the LoRA branch adds an exact 0.0 and the encoder's output is
the frozen model's BIT FOR BIT (same device, dtype, micro-batch). Everything after step 0 is the lever.

ONE LEVER, ZERO GLOBAL-RNG DRAW: ``A`` is initialised from a private ``torch.Generator`` -- building the encoder never touches the global RNG,
so the heads' initialisation (``torch.manual_seed(seed)`` then ``V8FMini(...)``) is B's, exactly.

EXACT ACCUMULATION (:func:`cached_grad_step`). Plain micro-batch accumulation is NOT exact for this model, although the ViT has no batch
statistics: the heads' losses are masked means (``sum / mask.sum()``) and the WM auxiliary carries SIGReg, a statistic of the whole batch. So the
heads' loss is evaluated ONCE on the full effective batch (like B), its gradient w.r.t. the pooled features is cached, and the encoder
re-runs in micro-batches to pull that gradient back into the LoRA matrices (the "gradient cache" of Gao et al. 2021). The result equals the
full-batch gradient up to float order, whatever the micro-batch size.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import inspect
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
#: the builder's own constants (pull_encode_pooled.py:78-79, 85): the 256x640 cylindrical frame is a 16 x 40 grid of 16-px patches
GH, GW = 16, 40
FRAME_H, FRAME_W = 256, 640
GRIDS = {"4x10": (4, 10), "8x20": (8, 20)}
#: mirrored decode / normalise constants (dinov3_fp8_encode_ship.py:122-123). They are module attributes ON PURPOSE: the mutation harness
#: changes ONE of them and the identity control must break.
PIXEL_SCALE = 255.0
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
N_SPECIAL = 5                      # CLS + 4 registers (pull_encode_pooled.py:230: assert n_special == 5)
#: sha256 of ``inspect.getsource(encode_episode)`` of the builder at the tip this module was written against (190be80a). If the builder's
#: function changes, :func:`assert_builder_unchanged` fails and the mirror above must be re-proved before anything trains.
BUILDER_ENCODE_EPISODE_SHA256 = "57f200072b2812ca524ee09461b308ddb45973aabc09e29fcad269ab5156308f"
#: the safetensors the cache builder encoded with (``meta.json`` ``model_safetensors_sha256`` of the dev-box builds, MEASURED 2026-10-17,
#: ``C:/Users/Admin/refav1_hj_cache/test_runscript/train/meta.json``). The trainer REFUSES any other file (PREREG_V8F_LORA refusal 4).
BUILDER_MODEL_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
DEFAULT_TARGETS = ("q", "v")
_PROJ = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}

_BUILDER: dict = {}


# --------------------------------------------------------------------------------------------- the builder, by file path
def builder_paths() -> tuple[Path, Path]:
    pep = Path(os.environ.get("V8F_LORA_BUILDER") or (
        REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "code" / "pull_encode_pooled.py"))
    ship = Path(os.environ.get("V8F_LORA_SHIP") or (REPO / "stack" / "scripts" / "dinov3_fp8_encode_ship.py"))
    return pep, ship


def load_builder():
    """-> ``(pull_encode_pooled, dinov3_fp8_encode_ship)``, both loaded by FILE PATH exactly as the builder loads its encoder
    (``pull_encode_pooled.load_encoder_module``, which also shims the ``truststore`` import). FAILS LOUD when a path is missing: there is no
    fallback copy of ``pool_field``, because a silent fallback is a pooling that is not the cache's."""
    if _BUILDER:
        return _BUILDER["pep"], _BUILDER["ship"]
    pep_p, ship_p = builder_paths()
    for p in (pep_p, ship_p):
        if not p.is_file():
            raise FileNotFoundError(f"v8f_lora_encoder: the cache builder file {p} does not exist (set V8F_LORA_BUILDER / V8F_LORA_SHIP, or run from a full "
                                    f"checkout). Looked for: {pep_p} and {ship_p}")
    spec = importlib.util.spec_from_file_location("pull_encode_pooled", str(pep_p))
    pep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pep)
    ship = pep.load_encoder_module(str(ship_p))
    _BUILDER.update(pep=pep, ship=ship)
    return pep, ship


def builder_pool_field() -> Callable:
    """The builder's ``pool_field`` -- IMPORTED, not re-implemented."""
    return load_builder()[0].pool_field


def builder_encode_episode_sha256() -> str:
    return hashlib.sha256(inspect.getsource(load_builder()[1].encode_episode).encode("utf-8")).hexdigest()


def assert_builder_unchanged() -> str:
    """The mirror's source pin: the builder's ``encode_episode`` is the one this module was proved against."""
    got = builder_encode_episode_sha256()
    if got != BUILDER_ENCODE_EPISODE_SHA256:
        raise AssertionError(f"the cache builder's encode_episode changed (sha256 {got} != pinned {BUILDER_ENCODE_EPISODE_SHA256}): re-prove the "
                             f"decode + normalise mirror in v8f_lora_encoder.py against it before any run reads this encoder")
    return got


# --------------------------------------------------------------------------------------------- decode + normalise (mirror)
def decode_raw_frames(o: dict, raw_idx: Sequence[int]) -> torch.Tensor:
    """v2ep dict -> uint8 ``[n, 3, H, W]`` of the given RAW frame indices (PNG). The builder's decode, for a subset of frames."""
    import torchvision
    assert str(o["codec"]) == "png", "the cache builder decodes PNG v2ep only"
    buf, lens = o["jpeg_buf"], o["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    return torch.stack([torchvision.io.decode_png(buf[offs[j]:offs[j + 1]].clone()) for j in raw_idx])


def decode_cache_frames(o: dict, cache_idx: Sequence[int]) -> torch.Tensor:
    """cache step ``g`` <-> raw frame ``2 g`` (``encode_episode``: ``range(0, len(lens), 2)``)."""
    return decode_raw_frames(o, [2 * int(g) for g in cache_idx])


def n_cache_frames(o: dict) -> int:
    return (int(o["jpeg_len"].shape[0]) + 1) // 2           # ceil(T_ep / 2), the manifest's n_frames


def mean_std(dtype: torch.dtype, device="cpu", from_processor: bool = False):
    """The builder's ``mean`` / ``std`` buffers: ``torch.tensor(proc.image_mean).view(1,3,1,1).to(device, dtype)`` (so the ROUNDING to the encoder's
    dtype is the builder's). ``from_processor`` reads them from the model's own preprocessor (what the builder does) and asserts they are the constants."""
    if from_processor:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(MID, local_files_only=True)
        assert tuple(proc.image_mean) == (0.485, 0.456, 0.406) and tuple(proc.image_std) == (0.229, 0.224, 0.225), "preprocessor constants changed"
        m, s = proc.image_mean, proc.image_std
    else:
        m, s = IMAGE_MEAN, IMAGE_STD
    return (torch.tensor(m).view(1, 3, 1, 1).to(device, dtype), torch.tensor(s).view(1, 3, 1, 1).to(device, dtype))


def normalise(frames_u8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """``encode_episode`` :122-123, verbatim ops: ``x = frames.to(device).to(mean.dtype) / 255.0; (x - mean) / std``."""
    x = frames_u8.to(mean.device).to(mean.dtype) / PIXEL_SCALE
    return (x - mean) / std


# --------------------------------------------------------------------------------------------- fp8 (straight-through)
class _Fp8STE(torch.autograd.Function):
    """The cache's quantiser as a differentiable op: forward = fp8 e4m3 round trip (EXACTLY ``pool.to(fp8).float()``, the value the model reads from
    the cache after ``fp8 -> fp16 -> fp32``); backward = identity (straight-through)."""

    @staticmethod
    def forward(ctx, x):
        return x.to(torch.float8_e4m3fn).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        return g


def fp8_ste(x: torch.Tensor) -> torch.Tensor:
    return _Fp8STE.apply(x)


# --------------------------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """``y = W x + b + (alpha / r) * B (A x)`` on a frozen ``nn.Linear``. ``B`` zero-initialised (the identity), ``A`` kaiming-uniform(a = sqrt 5)
    from a PRIVATE generator. The adapter branch runs in the adapter's dtype (fp32) and is added in the base output's dtype. Dropout 0 by default."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0, gen: torch.Generator | None = None):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base, self.r, self.alpha, self.scaling = base, int(r), float(alpha), float(alpha) / int(r)
        self.in_features, self.out_features = base.in_features, base.out_features
        for p in self.base.parameters():
            p.requires_grad_(False)
        a = torch.empty(self.r, self.in_features, dtype=torch.float32)
        bound = 1.0 / math.sqrt(self.in_features)                   # kaiming_uniform_(a=sqrt(5)): U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        a.uniform_(-bound, bound, generator=gen)
        self.lora_A = nn.Parameter(a.to(base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, dtype=torch.float32, device=base.weight.device))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        """``(alpha / r) * B @ A``, fp32 ``[out, in]``."""
        return (self.lora_B.float() @ self.lora_A.float()) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.merged:
            return y
        h = self.drop(x.to(self.lora_A.dtype))
        return y + (F.linear(F.linear(h, self.lora_A), self.lora_B) * self.scaling).to(y.dtype)

    def extra_repr(self):
        return f"r={self.r}, alpha={self.alpha}, in={self.in_features}, out={self.out_features}, merged={self.merged}"


class LoRAFusedQKV(nn.Module):
    """LoRA on the q and v SLICES of one fused ``qkv`` linear map ``[3E, E]`` (rows ``[0:E]`` = q, ``[E:2E]`` = k, ``[2E:3E]`` = v). k is untouched."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, targets: Sequence[str] = ("q", "v"), dropout: float = 0.0,
                 gen: torch.Generator | None = None):
        super().__init__()
        if base.out_features % 3:
            raise ValueError("a fused qkv map must have out_features divisible by 3")
        self.base, self.E = base, base.out_features // 3
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.targets = tuple(t for t in targets if t in ("q", "k", "v"))
        if not self.targets:
            raise ValueError(f"fused qkv: targets {targets} contain none of q / k / v")
        self.slices = nn.ModuleDict({t: _Slice(base.in_features, self.E, r, alpha, dropout, gen, base.weight.device) for t in self.targets})
        self.merged = False

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        E = self.E
        parts = [y[..., 0:E], y[..., E:2 * E], y[..., 2 * E:3 * E]]
        for t, i in (("q", 0), ("k", 1), ("v", 2)):
            if t in self.slices:
                parts[i] = parts[i] + self.slices[t](x).to(y.dtype)
        return torch.cat(parts, dim=-1)

    def delta_weight(self) -> torch.Tensor:
        d = torch.zeros(self.base.weight.shape, dtype=torch.float32, device=self.base.weight.device)
        for t, i in (("q", 0), ("k", 1), ("v", 2)):
            if t in self.slices:
                d[i * self.E:(i + 1) * self.E] = self.slices[t].delta_weight()
        return d


class _Slice(nn.Module):
    def __init__(self, in_f, out_f, r, alpha, dropout, gen, device):
        super().__init__()
        self.scaling = float(alpha) / int(r)
        a = torch.empty(int(r), in_f, dtype=torch.float32)
        bound = 1.0 / math.sqrt(in_f)
        a.uniform_(-bound, bound, generator=gen)
        self.lora_A = nn.Parameter(a.to(device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, int(r), dtype=torch.float32, device=device))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def delta_weight(self):
        return (self.lora_B.float() @ self.lora_A.float()) * self.scaling

    def forward(self, x):
        return F.linear(F.linear(self.drop(x.to(self.lora_A.dtype)), self.lora_A), self.lora_B) * self.scaling


def inject_lora(model: nn.Module, r: int, alpha: float, targets: Sequence[str] = DEFAULT_TARGETS, dropout: float = 0.0,
                gen: torch.Generator | None = None) -> int:
    """Freeze every parameter of ``model`` and wrap the attention projections in LoRA. Returns the number of attention blocks wrapped.

    * a block with SEPARATE ``q_proj`` / ``k_proj`` / ``v_proj`` linears (HF DINOv3) -> :class:`LoRALinear` on the ``targets`` among them;
    * a block with ONE fused ``qkv`` linear -> :class:`LoRAFusedQKV`: only the q and v SLICES are adapted (PREREG_V8F_LORA).
    Targets must be a subset of ``q k v o`` (``o`` = ``o_proj`` / ``proj``); an empty or unknown target is an error, never a silent no-op."""
    tg = tuple(dict.fromkeys(targets))
    bad = [t for t in tg if t not in _PROJ]
    if bad or not tg:
        raise ValueError(f"LoRA targets must be a non-empty subset of {sorted(_PROJ)}, got {tuple(targets)}")
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, (LoRALinear, LoRAFusedQKV, _Slice)):
            continue
        if all(isinstance(getattr(mod, a, None), nn.Linear) for a in ("q_proj", "k_proj", "v_proj")):
            for t in tg:
                attr = _out_attr(mod) if t == "o" else _PROJ[t]
                lin = getattr(mod, attr, None)
                if not isinstance(lin, nn.Linear):
                    raise ValueError(f"{name}: no linear for LoRA target {t!r} ({attr})")
                setattr(mod, attr, LoRALinear(lin, r, alpha, dropout, gen))
            n += 1
        elif isinstance(getattr(mod, "qkv", None), nn.Linear):
            mod.qkv = LoRAFusedQKV(mod.qkv, r, alpha, [t for t in tg if t in "qkv"], dropout, gen)
            if "o" in tg:
                attr = _out_attr(mod)
                setattr(mod, attr, LoRALinear(getattr(mod, attr), r, alpha, dropout, gen))
            n += 1
    if n == 0:
        raise ValueError("inject_lora: no attention block (q_proj/k_proj/v_proj or fused qkv) found -- nothing was adapted")
    return n


def _out_attr(mod: nn.Module) -> str:
    for a in ("o_proj", "proj", "out_proj"):
        if isinstance(getattr(mod, a, None), nn.Linear):
            return a
    raise ValueError("no output projection (o_proj / proj / out_proj) for LoRA target 'o'")


def lora_modules(model: nn.Module) -> list:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (LoRALinear, _Slice))]


@torch.no_grad()
def _fold(model: nn.Module) -> None:
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear) and not m.merged:
            m.base.weight.data = (m.base.weight.data.float() + m.delta_weight().to(m.base.weight.device)).to(m.base.weight.dtype)
            m.merged = True
        elif isinstance(m, LoRAFusedQKV) and not m.merged:
            m.base.weight.data = (m.base.weight.data.float() + m.delta_weight()).to(m.base.weight.dtype)
            m.merged = True


def _unwrap(model: nn.Module) -> nn.Module:
    """Replace every merged LoRA wrapper by its folded ``nn.Linear``: the result has exactly the frozen model's modules and parameter count."""
    for name, mod in list(model.named_modules()):
        for cname, ch in list(mod.named_children()):
            if isinstance(ch, (LoRALinear, LoRAFusedQKV)):
                assert ch.merged, "unwrap before merge"
                setattr(mod, cname, ch.base)
    return model


# --------------------------------------------------------------------------------------------- the encoder
def file_sha256(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def model_safetensors_path() -> Path:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(MID, "model.safetensors")
    if not isinstance(p, str):
        raise FileNotFoundError(f"{MID} model.safetensors is not in the local HF cache (looked via huggingface_hub.try_to_load_from_cache; "
                                f"also {Path.home() / '.cache/huggingface/hub'}). No download is attempted.")
    return Path(p)


class DinoV3LoRAEncoder(nn.Module):
    """Frozen DINOv3 backbone + LoRA, producing the cache's tensor. ``backbone`` is the HF ``DINOv3ViTModel`` (or any module with the same
    attention naming); weights are NOT touched except by the LoRA matrices. ``quantize`` = the cache's fp8 e4m3 round trip (straight-through)."""

    def __init__(self, backbone: nn.Module, mean: torch.Tensor, std: torch.Tensor, *, n_special: int = N_SPECIAL, r: int = 32,
                 alpha: float = 32.0, targets: Sequence[str] = DEFAULT_TARGETS, dropout: float = 0.0, seed: int = 0, grid: str = "4x10",
                 quantize: bool = True, weights_sha256: str | None = None):
        super().__init__()
        if grid not in GRIDS:
            raise ValueError(f"grid must be one of {sorted(GRIDS)}")
        self.backbone, self.n_special, self.grid, self.quantize = backbone, int(n_special), grid, bool(quantize)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.r, self.alpha, self.targets, self.dropout, self.seed = int(r), float(alpha), tuple(targets), float(dropout), int(seed)
        gen = torch.Generator().manual_seed(self.seed)                    # PRIVATE: the global RNG (the heads' init) is never touched
        self.n_blocks = inject_lora(self.backbone, self.r, self.alpha, self.targets, self.dropout, gen)
        self.weights_sha256 = weights_sha256
        self._ckpt = False

    # ---- construction ------------------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, *, r=32, alpha=32.0, targets=DEFAULT_TARGETS, dropout=0.0, seed=0, grid="4x10", quantize=True, device="cpu",
                        dtype=torch.bfloat16, verify_sha: bool = True) -> "DinoV3LoRAEncoder":
        """The builder's ``build_model``: local HF cache only (no token, no download), weights cast to ``dtype`` (bf16: the shipped cache's)."""
        from transformers import AutoModel
        model = AutoModel.from_pretrained(MID, local_files_only=True, dtype=dtype).eval().to(device)
        m, s = mean_std(dtype, device, from_processor=True)
        n_special = 1 + getattr(model.config, "num_register_tokens", 0)
        assert n_special == N_SPECIAL, "encoder geometry changed"
        sha = file_sha256(model_safetensors_path()) if verify_sha else None
        if verify_sha and sha != BUILDER_MODEL_SHA256:
            raise RuntimeError(f"the local {MID} safetensors sha256 {sha} is not the cache builder's {BUILDER_MODEL_SHA256}")
        return cls(model, m, s, n_special=n_special, r=r, alpha=alpha, targets=targets, dropout=dropout, seed=seed, grid=grid, quantize=quantize,
                   weights_sha256=sha)

    # ---- bookkeeping -------------------------------------------------------------------------------------------
    def lora_parameters(self) -> list:
        return [p for n, p in self.named_parameters() if "lora_" in n]

    def trainable_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def frozen_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if not p.requires_grad))

    def expected_trainable(self) -> int:
        """rank x (in + out) per adapted projection x targets x blocks -- from the BUILT modules, so a wrong rank / target / block count is visible."""
        tot = 0
        for _, m in lora_modules(self):
            tot += int(m.lora_A.numel() + m.lora_B.numel())
        return tot

    def assert_frozen(self) -> None:
        bad = [n for n, p in self.named_parameters() if p.requires_grad and "lora_" not in n]
        if bad:
            raise AssertionError(f"non-LoRA parameters are trainable: {bad[:5]}")

    def built_levers(self) -> dict:
        ls = lora_modules(self)
        return {"encoder": "lora", "lora_r": self.r, "lora_alpha": self.alpha, "lora_targets": list(self.targets), "lora_dropout": self.dropout,
                "lora_seed": self.seed, "lora_blocks": self.n_blocks, "lora_modules": len(ls),
                "lora_trainable": self.trainable_count(), "lora_expected_trainable": self.expected_trainable(),
                "lora_B_zero_at_build": bool(all(float(m.lora_B.detach().abs().sum()) == 0.0 for _, m in ls)),
                "frozen_params": self.frozen_count(), "grid": self.grid, "fp8_ste": self.quantize,
                "weights_sha256": self.weights_sha256, "builder_model_sha256": BUILDER_MODEL_SHA256,
                "backbone_dtype": str(next(self.backbone.parameters()).dtype), "checkpointing": self._ckpt}

    # ---- checkpointing -----------------------------------------------------------------------------------------
    def train(self, mode: bool = True):
        """⛔ THE BACKBONE IS ALWAYS IN EVAL MODE. MEASURED 2026-10-25: HF ``DINOv3ViTRopePositionEmbedding.forward`` applies
        ``augment_patches_center_coordinates(..., rescale=config.pos_embed_rescale)`` (2.0 in the shipped config) whenever ``self.training`` -- a RANDOM
        rescale of the RoPE coordinates. A ``model.train()`` would therefore feed the heads features unlike the cache's (which the builder made under
        ``.eval()``), differently at every call, and the identity control would fail only on the NEXT eval. Gradient checkpointing needs the *layers* in
        train mode, not the RoPE module: see :meth:`set_checkpointing`."""
        super().train(mode)
        self.backbone.eval()
        if self._ckpt:
            self._layers_train()
        return self

    def _layers_train(self) -> None:
        from transformers.modeling_layers import GradientCheckpointingLayer
        for m in self.backbone.modules():
            if isinstance(m, GradientCheckpointingLayer):
                m.train()               # the layers only; RoPE / embeddings stay eval

    def set_checkpointing(self, on: bool) -> None:
        cfg = getattr(self.backbone, "config", None)
        if on:
            if cfg is not None and (float(getattr(cfg, "attention_dropout", 0.0)) != 0.0 or float(getattr(cfg, "drop_path_rate", 0.0)) != 0.0):
                raise ValueError("checkpointing runs the blocks in train mode, which is only numerically neutral with attention_dropout = drop_path_rate = 0")
            self.backbone.eval()
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self._ckpt = True
            self._layers_train()
        else:
            self.backbone.gradient_checkpointing_disable()
            self.backbone.eval()
            self._ckpt = False

    # ---- the encoder -------------------------------------------------------------------------------------------
    def pixel_values(self, frames_u8: torch.Tensor) -> torch.Tensor:
        return normalise(frames_u8, self.mean, self.std)

    def field(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """uint8 ``[N, 3, H, W]`` -> fp32 patch field ``[N, 640, D]`` (CLS + registers dropped): ``encode_episode``'s ``f`` for these frames."""
        h = self.backbone(pixel_values=self.pixel_values(frames_u8)).last_hidden_state
        return h[:, self.n_special:, :].float()

    def pooled(self, frames_u8: torch.Tensor, grid: str | None = None, quantize: bool | None = None) -> torch.Tensor:
        """uint8 frames -> the cache's ``pool<grid>`` rows, fp32 ``[N, gh*gw, D]``: the builder's ``pool_field`` on the fp32 field, then the fp8 round trip."""
        gh, gw = GRIDS[grid or self.grid]
        p = builder_pool_field()(self.field(frames_u8), gh, gw)
        return fp8_ste(p) if (self.quantize if quantize is None else quantize) else p

    forward = pooled

    # ---- LoRA state / merge ------------------------------------------------------------------------------------
    def lora_state_dict(self) -> dict:
        return {"meta": {"r": self.r, "alpha": self.alpha, "targets": list(self.targets), "blocks": self.n_blocks, "dropout": self.dropout,
                         "seed": self.seed, "weights_sha256": self.weights_sha256},
                "tensors": {n: p.detach().cpu().clone() for n, p in self.named_parameters() if "lora_" in n}}

    def load_lora_state_dict(self, sd: dict) -> None:
        mt = sd["meta"]
        for k, want in (("r", self.r), ("alpha", self.alpha), ("targets", list(self.targets)), ("blocks", self.n_blocks)):
            if mt[k] != want:
                raise ValueError(f"LoRA checkpoint {k} {mt[k]} != built {want}")
        mine = dict(self.named_parameters())
        if set(sd["tensors"]) != {n for n in mine if "lora_" in n}:
            raise ValueError("LoRA checkpoint parameter names differ from the built encoder's")
        with torch.no_grad():
            for n, t in sd["tensors"].items():
                mine[n].copy_(t.to(mine[n].device, mine[n].dtype))

    def merge(self) -> nn.Module:
        """The deployable encoder: a COPY of the backbone with ``B A`` folded into ``W`` (``W + (alpha/r) B A``, fp32 sum rounded to the weight's
        dtype) and the wrappers removed -- exactly the frozen encoder's modules and parameter count. ``self`` is not modified."""
        m = copy.deepcopy(self.backbone)
        if getattr(m, "gradient_checkpointing", False):
            m.gradient_checkpointing_disable()
        m.eval()
        _fold(m)
        return _unwrap(m)

    def merged_encoder(self) -> "DinoV3LoRAEncoder":
        """A plain-backbone twin (no adapters) around :meth:`merge`: ``pooled`` / ``field`` of the merged weights, with the same preprocessing."""
        bb = self.merge()
        twin = _PlainEncoder(bb, self.mean.clone(), self.std.clone(), n_special=self.n_special, grid=self.grid, quantize=self.quantize)
        return twin


class _PlainEncoder(DinoV3LoRAEncoder):
    """The merged / frozen backbone with the cache's preprocessing and pooling, no adapters."""

    def __init__(self, backbone, mean, std, *, n_special=N_SPECIAL, grid="4x10", quantize=True):
        nn.Module.__init__(self)
        self.backbone, self.n_special, self.grid, self.quantize = backbone, int(n_special), grid, bool(quantize)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.r, self.alpha, self.targets, self.dropout, self.seed, self.n_blocks = 0, 0.0, (), 0.0, 0, 0
        self.weights_sha256, self._ckpt = None, False
        for p in self.backbone.parameters():
            p.requires_grad_(False)


def frozen_plain_encoder(*, device="cpu", dtype=torch.bfloat16, grid="4x10", quantize=True) -> _PlainEncoder:
    """The cache builder's encoder, as a module: no adapters at all (the reference for the identity tests)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MID, local_files_only=True, dtype=dtype).eval().to(device)
    m, s = mean_std(dtype, device, from_processor=True)
    return _PlainEncoder(model, m, s, n_special=1 + getattr(model.config, "num_register_tokens", 0), grid=grid, quantize=quantize)


# --------------------------------------------------------------------------------------------- unique frames of a batch
def unique_frames(clips: Sequence[str], idx: np.ndarray) -> tuple[list, np.ndarray]:
    """``clips[b]`` and ``idx [B, F]`` (cache steps, already clamped into the clip) -> ``(keys, frame_map)`` with ``keys`` the unique ``(clip, step)``
    pairs in first-seen order and ``frame_map [B, F]`` the index of each into ``keys``. Frames shared by windows of one clip (and a clip's clamped
    frame 0) are encoded ONCE; autograd sums their gradients through the gather."""
    seen: dict = {}
    fm = np.empty(idx.shape, np.int64)
    for b, c in enumerate(clips):
        for f in range(idx.shape[1]):
            k = (c, int(idx[b, f]))
            j = seen.get(k)
            if j is None:
                j = seen[k] = len(seen)
            fm[b, f] = j
    return list(seen), fm


# --------------------------------------------------------------------------------------------- the exact accumulation
def encode_chunked(enc: DinoV3LoRAEncoder, frames_u8: torch.Tensor, chunk: int, *, grad: bool, device) -> torch.Tensor:
    """Encode ``[U, 3, H, W]`` uint8 in chunks of ``chunk`` frames -> fp32 ``[U, n_tok, D]`` on ``device`` (no autograd unless ``grad``)."""
    outs = []
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx:
        for s in range(0, frames_u8.shape[0], chunk):
            outs.append(enc.pooled(frames_u8[s:s + chunk].to(device)))
    return torch.cat(outs)


def _sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def cached_grad_step(enc: DinoV3LoRAEncoder, frames_u8: torch.Tensor, frame_map: torch.Tensor, run_loss: Callable[[torch.Tensor], dict], *,
                     chunk: int, device, timing: bool = False) -> dict:
    """ONE exact forward + backward of the full effective batch through heads AND encoder, with the encoder run in micro-batches of ``chunk`` frames.

    1. encode all unique frames with NO graph (micro-batched) -> ``pooled [U, n_tok, D]``;
    2. ``leaf = pooled`` as a leaf that requires grad; ``feats = leaf[frame_map]`` is the model's ``[B, F, n_tok, D]`` input; ``run_loss(feats)``
       evaluates the heads' loss on the FULL batch and ``total.backward()`` fills the heads' grads and ``leaf.grad``;
    3. re-encode each micro-batch WITH graph and backpropagate ``leaf.grad[chunk]`` into the LoRA matrices.
    ``run_loss`` returns the model's ``loss`` output dict (``out['terms']['total']`` is back-propagated here). Returns that dict plus ``leaf_grad_abs_sum`` (and, with
    ``timing``, the wall time of the three phases -- synchronised on CUDA -- for the AR-2 projection)."""
    import time
    tm = {}
    if timing:
        _sync(device)
        t0 = time.perf_counter()
    pooled = encode_chunked(enc, frames_u8, chunk, grad=False, device=device)
    if timing:
        _sync(device)
        tm["phase1_encode_nograd_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
    if not bool(torch.isfinite(pooled).all()):
        raise FloatingPointError("the on-the-fly encoder produced non-finite pooled features (fp8 e4m3fn has no inf: values > 448 become NaN)")
    leaf = pooled.detach().requires_grad_(True)
    out = run_loss(leaf[frame_map.to(leaf.device)])
    out["terms"]["total"].backward()
    g = leaf.grad
    if g is None:
        raise RuntimeError("the heads' loss did not reach the encoder's output (leaf.grad is None): the features are detached on the way to the loss")
    if timing:
        _sync(device)
        tm["heads_fwd_bwd_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
    for s in range(0, frames_u8.shape[0], chunk):
        p = enc.pooled(frames_u8[s:s + chunk].to(device))
        torch.autograd.backward(p, g[s:s + chunk])
    out["leaf_grad_abs_sum"] = float(g.abs().sum())
    if timing:
        _sync(device)
        tm["phase2_encode_grad_s"] = time.perf_counter() - t0
        tm["n_unique_frames"] = int(frames_u8.shape[0])
        out["timing"] = tm
    return out
