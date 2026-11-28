"""The MINIMAL relaxation that lets the frozen deployed-v1 trunk accept a non-256, non-square input.

⚠️ **This is a LOCAL SHIM. `stack/` is NOT modified by this stream.** It exists so the FOV x
resolution x aspect-ratio question can be MEASURED against the real v1 weights before anyone
proposes a training-path change. What it patches is exactly the scope of that change:

| where | today | what has to give |
|---|---|---|
| `encoder.ViTEncoder.__init__` | `self.grid_hw = image_size // patch_size`; `n_tokens = grid_hw**2` | one grid dim -> two (`gh`, `gw`) |
| `encoder.ViTEncoder.pos` | learned `[1, 256, 768]` | ⭐ **a CHECKPOINT-SHAPED tensor** — it must be resized, so this is the one part that is not free |
| `readout.SpatialGridReadout.__init__` | `assert hw*hw == n_tokens`; `AvgPool2d(hw // grid)` | square assert -> (gh, gw); fixed pool -> adaptive pool to (4, 4) |

Everything else in the encoder is shape-agnostic: `patch` is a strided `Conv2d`, the blocks are
attention + MLP over an arbitrary token count, and `norm` is a `LayerNorm` over `d_model`.

**Positional-embedding transfer.** The 16x16 learned grid is bicubically resampled to (gh, gw) —
the standard ViT resolution-transfer recipe. ⚠️ **DECLARED BIAS, one-directional:** these weights
were trained at 256 px / 51.4 deg, so every non-native shape is evaluated under a train/test shape
shift that can only HURT it. A higher-resolution arm that WINS under this handicap is therefore
strong evidence; a higher-resolution arm that merely ties is weak evidence and must not be read as
"resolution does not help" — it may be reading the handicap. Any adopted shape would be RETRAINED,
not shimmed.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, os.environ.get(
    "TANITAD_STACK", r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack"))

TRUNK = os.environ.get(
    "V1_TRUNK",
    r"C:\Users\Admin\AppData\Local\Temp\claude"
    r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
    r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\sitclf\v1_trunk.pt")


def resample_pos(pos: Tensor, gh_src: int, gw_src: int, gh: int, gw: int) -> Tensor:
    """[1, gh_src*gw_src, D] -> [1, gh*gw, D] by bicubic resampling of the 2-D token grid."""
    if (gh_src, gw_src) == (gh, gw):
        return pos.clone()
    b, n, d = pos.shape
    assert n == gh_src * gw_src, f"pos has {n} tokens, expected {gh_src}x{gw_src}"
    g = pos.reshape(b, gh_src, gw_src, d).permute(0, 3, 1, 2)          # [1, D, gh, gw]
    g = F.interpolate(g, size=(gh, gw), mode="bicubic", align_corners=False)
    return g.permute(0, 2, 3, 1).reshape(b, gh * gw, d)


class ShapedReadout(nn.Module):
    """`SpatialGridReadout` with the square assert removed and an ADAPTIVE 4x4 pool.

    On a square 16x16 grid with grid=4 the adaptive pool is EXACTLY `AvgPool2d(4)`, so this is
    bit-identical to the deployed readout at the deployed shape — verified by `verify_identity()`.
    """

    def __init__(self, gh: int, gw: int, d_model: int, grid: int = 4, d_readout: int = 128):
        super().__init__()
        self.gh, self.gw, self.grid = gh, gw, grid
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.proj = nn.Linear(d_model, d_readout)
        self.out_dim = grid * grid * d_readout

    def forward(self, tokens: Tensor) -> Tensor:
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, self.gh, self.gw)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x).flatten(1)


class ShapedEncoder(nn.Module):
    """`ViTEncoder` with a (gh, gw) token grid and a resampled positional embedding."""

    def __init__(self, src_enc, gh: int, gw: int):
        super().__init__()
        self.cfg = src_enc.cfg
        self.gh, self.gw = gh, gw
        self.n_tokens = gh * gw
        self.patch = src_enc.patch
        gs = src_enc.grid_hw
        self.pos = nn.Parameter(resample_pos(src_enc.pos.data, gs, gs, gh, gw),
                                requires_grad=False)
        self.blocks = src_enc.blocks
        self.norm = src_enc.norm

    def forward(self, x: Tensor) -> Tensor:
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = t + self.pos
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)


class ShapedTrunk(nn.Module):
    """encoder + readout, exposing `.encode` and `.state_dim` like `WorldModel`."""

    def __init__(self, enc, ro):
        super().__init__()
        self.encoder, self.readout = enc, ro
        self.state_dim = ro.out_dim

    def encode(self, x: Tensor) -> Tensor:
        return self.readout(self.encoder(x))


def _reference_trunk(device: str):
    """The deployed encoder + readout built DIRECTLY from `stack`, bypassing `fourbrain.WorldModel`.

    ⚠️ Deliberate: on 2026-07-27 a sibling stream was mid-edit making the model layer non-square
    aware, and `fourbrain.py` momentarily referenced a `ReadoutConfig` field `config.py` did not yet
    have. Constructing `ViTEncoder` + `SpatialGridReadout` directly keeps this sweep reproducible
    against the CHECKPOINT rather than against a file being edited underneath it.
    """
    from tanitad.config import flagship4b_config
    from tanitad.models.encoder import ViTEncoder
    from tanitad.models.readout import SpatialGridReadout

    pay = torch.load(TRUNK, map_location="cpu", weights_only=False)
    trunk = pay["trunk"]
    cfg = flagship4b_config()
    enc = ViTEncoder(cfg.encoder)
    enc.load_state_dict({k[len("encoder."):]: v for k, v in trunk.items()
                         if k.startswith("encoder.")}, strict=True)   # STRICT -> v1's weights
    # read the readout's own shape OFF THE CHECKPOINT, never off a config that may have moved
    d_readout = int(trunk["readout.proj.weight"].shape[0])
    grid = int(cfg.readout.grid)
    ro = SpatialGridReadout(enc.n_tokens, cfg.encoder.d_model, grid=grid, d_readout=d_readout)
    ro.load_state_dict({k[len("readout."):]: v for k, v in trunk.items()
                        if k.startswith("readout.")}, strict=True)
    assert sum(p.numel() for p in enc.parameters()) == pay["encoder_params"]
    assert sum(p.numel() for p in ro.parameters()) == pay["readout_params"]
    return enc.to(device).eval(), ro.to(device).eval(), cfg, grid, d_readout


def build_trunk(image_h: int = 256, image_w: int = 256, device: str = "cuda",
                trunk_path: str | None = None) -> ShapedTrunk:
    """The DEPLOYED-v1 encoder + readout, STRICT-loaded, reshaped to (image_h, image_w)."""
    src_enc, src_ro, cfg, grid, d_readout = _reference_trunk(device)
    p = cfg.encoder.patch_size
    assert image_h % p == 0 and image_w % p == 0, f"h,w must be multiples of {p}"
    gh, gw = image_h // p, image_w // p
    enc = ShapedEncoder(src_enc, gh, gw)
    ro = ShapedReadout(gh, gw, cfg.encoder.d_model, grid=grid, d_readout=d_readout)
    ro.proj.load_state_dict(src_ro.proj.state_dict())
    m = ShapedTrunk(enc, ro).to(device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def verify_identity(device: str = "cuda", tol: float = 0.0) -> dict:
    """C-FID for the shim: at the DEPLOYED shape it must be BIT-IDENTICAL to the real trunk.

    A shim that silently changes the deployed arm would make every contrast in the sweep a
    comparison against the wrong baseline, so this check is run before any sweep number is emitted
    and a failure is fatal, not a warning.
    """
    enc, ro, _cfg, _g, _d = _reference_trunk(device)

    class _Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc, self.ro = enc, ro

        def encode(self, x):
            return self.ro(self.enc(x))

    world = _Ref().to(device).eval()
    shim = build_trunk(256, 256, device)
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randint(0, 255, (4, 9, 256, 256), generator=g, dtype=torch.uint8).to(device)
    with torch.no_grad():
        a = world.encode(x.float().div(255.0))
        b = shim.encode(x.float().div(255.0))
    d = (a - b).abs().max().item()
    return {"max_abs_diff": d, "bit_identical": bool(d <= tol),
            "shape": list(a.shape), "state_dim": int(a.shape[-1])}


if __name__ == "__main__":
    print(verify_identity())
    for hw in ((256, 256), (256, 640), (640, 640)):
        m = build_trunk(*hw)
        x = torch.randint(0, 255, (2, 9, hw[0], hw[1]), dtype=torch.uint8, device="cuda")
        with torch.no_grad():
            y = m.encode(x.float().div(255.0))
        print(hw, "tokens", m.encoder.n_tokens, "->", tuple(y.shape))
        del m
        torch.cuda.empty_cache()
