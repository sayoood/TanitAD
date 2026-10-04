"""A small CPU rig for the refcv8 WP-B tests: a REAL RefCV3Model with the refcv7 planner stack -- DDIM sampler in
control space, v0-conditioned alat vocabulary, the residual prior (ha0_ext_pose), ego history, F1-F6, the refcv6
behaviour decoder over agent slots + a stand-in BEV branch -- built through the trainer's OWN argv pin for the
decoder half, then the scene half exactly as `test_refcv7_model._cfg` builds it (timm resnet18 at 64x64,
pretrained=False: no download).
"""
from __future__ import annotations

import dataclasses as _dc
import importlib.util
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_STACK = _HERE.parent
for _p in (str(_STACK), str(_STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.refs import refcv6_tactical as v6tac  # noqa: E402

D_BEV = 8
F1_F6 = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
         "--f5-emitting-conf", "--f6-w-u0-zero"]
A_LON = (-3.0, -1.5, 0.0, 1.0)
A_LAT = (-2.5, -1.0, 0.0, 1.0, 2.5)


def trainer():
    """The trainer as an importable module (`stack/scripts` is on sys.path), so its classes PICKLE -- a DataLoader
    worker receives exactly such a pickled dataset."""
    import importlib
    return importlib.import_module("refc_v3_train")


class FakeBranch(torch.nn.Module):
    """`test_refcv7_model._FakeBranch`, verbatim contract."""

    def __init__(self, d_bev: int = D_BEV):
        super().__init__()
        self.lift = None
        self.map_branch = None
        self.box_mem = None
        self.box_dec = None
        self.d_bev = int(d_bev)
        self.proj = torch.nn.Linear(1, self.d_bev)

    def forward(self, fmap_s16, grid=None, valid=None):
        b = fmap_s16.shape[0]
        pooled = fmap_s16.mean(dim=(1, 2, 3)).reshape(b, 1, 1)
        feats = self.proj(pooled).expand(b, 6, self.d_bev)
        return {"bev_feats": feats.transpose(1, 2).reshape(b, self.d_bev, 2, 3), "bev_tokens": feats}


def grid() -> torch.Tensor:
    return torch.cartesian_prod(torch.tensor(A_LON), torch.tensor(A_LAT))     # 20 anchors incl. (0, 0)


def build(T, refcv8: bool, seed: int = 0, **r8kw):
    from tanitad.refs.refc_agents import AgentSeamConfig
    argv = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned", "--anchor-control-units", "alat",
            "--n-anchors", str(len(A_LON) * len(A_LAT)), "--ego-history", "--residual-prior", "ha0_ext_pose",
            "--out", "_r8_rig_unused"] + F1_F6
    args = T.build_parser().parse_args(argv)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    cfg.tac_vocab_version = "v7.0"
    cfg.core.encoder = _dc.replace(cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k",
                                   trunk_pretrained=False, in_channels=3, image_size=64, image_width=None)
    cfg.core.agents = AgentSeamConfig(enable=True)
    cfg.core.decoder.cross_agent = True
    cfg.tac_decoder_v6 = True
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(d_model=32, n_layers=1, n_heads=4, ff_mult=2,
                                                      d_agent=int(cfg.core.decoder.d), d_bev=D_BEV,
                                                      sources=("agent", "bev"))
    if refcv8:
        cfg.refcv8.enable = True
        for k, v in r8kw.items():
            setattr(cfg.refcv8, k, v)
    torch.manual_seed(seed)
    m = T.v3.RefCV3Model(cfg)
    m._perception = FakeBranch(D_BEV)
    with torch.no_grad():
        m.core.decoder.anchor_controls.copy_(grid())
    return cfg, m


def batch(cfg, b: int = 3, seed: int = 1):
    enc = cfg.core.encoder
    h, w = enc.image_hw()
    W = int(cfg.core.window)
    g = torch.Generator().manual_seed(seed)
    frames = torch.rand(b, W, int(enc.in_channels), h, w, generator=g)
    v_s = torch.tensor([6.0, 12.0, 3.0, 20.0, 8.0][:b])
    om = torch.tensor([0.2, -0.1, 0.3, 0.05, 0.0][:b])
    t = torch.arange(W, dtype=torch.float32)
    ph = torch.zeros(b, W, 4)
    ph[:, :, 3] = v_s[:, None]
    ph[:, :, 2] = om[:, None] * 0.1 * t[None]
    ph[:, :, 0] = v_s[:, None] * 0.1 * t[None]
    nav = torch.tensor([0, 1, 2, 0, 1][:b])
    return {"frames": frames, "nav_cmd": nav, "v0": ph[:, -1, 3].clone(), "pose_hist": ph}


def forward(cfg, m, bt, seed: int = 2, train: bool = False, **kw):
    m.train(train)
    torch.manual_seed(seed)
    ph = bt["pose_hist"]
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        m.core.set_ego_window(ph, int(ph.shape[1]))
        return m(bt["frames"], nav_cmd=bt["nav_cmd"], v0=bt["v0"], steps=cfg.core.decoder.diffusion_steps, **kw)


def copy_into(src, dst) -> list[str]:
    """Load src's state into dst: strict on every src key; returns the dst-only (NEW) keys."""
    sd = src.state_dict()
    missing, unexpected = dst.load_state_dict(sd, strict=False)
    if unexpected:
        raise AssertionError(f"unexpected keys {unexpected[:5]}")
    return list(missing)


def turn_angle(deg: float) -> float:
    return math.radians(deg)
