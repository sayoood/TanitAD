"""Shared model builder for the 2026-09-22 refcv6 TACTICAL/NAV/HIERARCHY review.

⛔ WHY A SHARED BUILDER AND NOT A COPY PER PROBE. Five probes that each write
their own config are five chances to measure five different models and call the
difference a finding. One builder, one recipe, and every probe prints the
config fingerprint it actually ran.

⛔ NO GEOMETRY LITERAL AND NO SILENT DEFAULT: every flag this review depends on
is set HERE and echoed in `fingerprint()`, so a reader can tell an arm that was
never configured from one that was configured and refused.

Recipe follows `stack/tests/test_refcv6_bev_tactical_wiring.py::_model` (the
in-repo recipe for a real `RefCV3Model` with the timm trunk) — deliberately,
so a difference between my numbers and the repo's is a difference in the
QUESTION, not in the model.
"""
from __future__ import annotations

import dataclasses as _dc
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve()
while ROOT.name != "TanitAD":
    ROOT = ROOT.parent
for _p in (str(ROOT / "stack"), str(ROOT / "stack" / "scripts"),
           str(ROOT / "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.refs import refc_v3 as v3            # noqa: E402
from tanitad.refs import refcv6_tactical as v6tac  # noqa: E402
from tanitad.refs.refc_agents import AgentSeamConfig  # noqa: E402


class FakeBranch(nn.Module):
    """A perception branch stand-in with ONE real parameter, so "did gradient
    reach the branch / the trunk" is answerable without a timm download.

    It mirrors `refcv6_perception_branch.PerceptionBranch`'s OUTPUT CONTRACT
    only (`bev_tokens` / `bev_feats`), which is what `_bev_hook` consumes.
    `lift = None` so `_bev_hook` does not demand per-clip lift geometry.
    """

    lift = None

    def __init__(self, d_bev: int, n_cells: int = 12):
        super().__init__()
        self.d_bev, self.n_cells = int(d_bev), int(n_cells)
        self.w = nn.Parameter(torch.randn(1, dtype=torch.float32) * 0.1 + 1.0)

    def forward(self, fmap_s16, grid=None, valid=None):
        b = fmap_s16.shape[0]
        # carry the trunk's graph: pool the real feature map, then project by a
        # fixed (non-parameter) matrix so the branch itself has exactly 1 param.
        pooled = fmap_s16.flatten(2).mean(dim=-1)            # [B, C]
        c = pooled.shape[-1]
        g = torch.Generator().manual_seed(1234)
        proj = torch.randn(c, self.d_bev, generator=g,
                           dtype=pooled.dtype) / (c ** 0.5)
        tok = (pooled @ proj.to(pooled.device)) * self.w      # [B, d_bev]
        tok = tok.unsqueeze(1).expand(b, self.n_cells, self.d_bev).contiguous()
        return {"bev_tokens": tok,
                "bev_feats": tok.transpose(1, 2).reshape(
                    b, self.d_bev, self.n_cells, 1)}


def build(*, d_bev: int = 16, tac: bool = True, attach_branch: bool = True,
          no_strategic: bool = False, max_speed: bool = True,
          nav_compliance: bool = True, detach_bev: bool = False,
          image_size: int = 64, sampler: str = "none",
          refcv6_flags=None, tac8_prior: bool = False,
          behaviour_sel: bool = False):
    """A real `RefCV3Model`. Returns `(model, cfg)`.

    `sampler="ddim"` builds the refcv5 WP-4 / refcv6 denoising loop, which is
    what makes the SPEC §5 phrase *"every denoising pass"* measurable at all:
    with `sampler="none"` the decoder stack runs ONCE and the phrase has no
    referent in the forward.
    """
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.tac_vocab_version = "v7.0"
    cfg.core.encoder = _dc.replace(
        cfg.core.encoder, trunk="timm", trunk_name="resnet18.a1_in1k",
        trunk_pretrained=False, in_channels=3, image_size=int(image_size),
        image_width=None)
    cfg.core.agents = AgentSeamConfig(enable=True)
    cfg.core.decoder.cross_agent = True
    cfg.core.no_strategic = bool(no_strategic)
    cfg.core.ego_valid_channel = True   # precondition of ego_state_inject
    cfg.ego_state_inject = True     # E11' - v0/a0 reach the condition
    cfg.core.decoder.sampler = str(sampler)
    if sampler == "ddim":
        # ⛔ The DDIM sampler REFUSES a fixed-path bank: its state IS the
        # control sequence, and a zero `anchor_controls` would centre the
        # anchored Gaussian on "do nothing". So the v0-conditioned vocabulary
        # is turned on and the controls are given a real spread — the same
        # recipe `stack/tests/test_refc_sampler.py::_v0_decoder` uses.
        cfg.core.anchors.v0_conditioned = True
    if refcv6_flags is not None:
        cfg.core.decoder.refcv6 = refcv6_flags
    if nav_compliance:
        cfg.core.graft_nav_compliance = True
        cfg.core.nav_compliance_tau_rad = 0.35   # any positive tau; stated
    if behaviour_sel:
        # SPEC §4 (b): the valid-behaviour set gates SELECTION.
        cfg.core.graft_behaviour_sel = True
    if tac8_prior:
        # SPEC §4 (b): the lat/lon posterior REPLACES the image-only lat3/lon3
        # as the anchor prior through a NEW zero-init 8 -> n_anchors graft.
        cfg.core.graft_tac8_prior = True
    if tac:
        cfg.tac_decoder_v6 = True
        cfg.tac_decoder_bev_detach = bool(detach_bev)
        cfg.max_speed_onehot_v6 = bool(max_speed)
        cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(
            d_model=64, n_layers=2, n_heads=4, ff_mult=2,
            d_agent=int(cfg.core.decoder.d), d_bev=int(d_bev),
            sources=("agent",) if d_bev <= 0 else ("agent", "bev"))
    model = v3.RefCV3Model(cfg)
    if sampler == "ddim":
        dec = model.core.decoder
        n = int(dec.anchors.shape[0])
        with torch.no_grad():
            dec.anchor_controls.copy_(torch.stack(
                [torch.linspace(-2.0, 2.0, n),
                 torch.linspace(-1.5, 1.5, n)], dim=-1))
    if tac and attach_branch and d_bev > 0:
        model._perception = FakeBranch(int(d_bev))
    model.eval()
    return model, cfg


def fingerprint(model, cfg) -> dict:
    enc = cfg.core.encoder
    return {
        "trunk": f"{enc.trunk}:{enc.trunk_name}",
        "image_hw": list(enc.image_hw()),
        "tac_decoder_v6_built": getattr(model, "tac_decoder_v6", None) is not None,
        "tac_sources": (list(model.tac_decoder_v6.cfg.sources)
                        if getattr(model, "tac_decoder_v6", None) is not None
                        else None),
        "d_bev": (int(model.tac_decoder_v6.cfg.d_bev)
                  if getattr(model, "tac_decoder_v6", None) is not None else None),
        "perception_attached": getattr(model, "_perception", None) is not None,
        "no_strategic": bool(getattr(cfg.core, "no_strategic", False)),
        "max_speed_onehot_v6": bool(getattr(cfg, "max_speed_onehot_v6", False)),
        "graft_nav_compliance": bool(getattr(cfg.core, "graft_nav_compliance", False)),
        "sampler": str(getattr(cfg.core.decoder, "sampler", "none")),
        "diffusion_steps": int(getattr(cfg.core.decoder, "diffusion_steps", -1)),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


def batch(model, cfg, b: int = 4, *, nav: int | None = 1,
          v_max_ms: float = 13.9, seed: int = 0):
    """One CPU batch of inputs for `RefCV3Model.forward`."""
    torch.manual_seed(int(seed))
    enc = cfg.core.encoder
    h, w = enc.image_hw()
    frames = torch.rand(b, int(cfg.core.window), int(enc.in_channels), h, w)
    kw = dict(
        v0=torch.full((b,), 8.0),
        ego_state=torch.tensor([[8.0, 0.3, 0.01, 0.0, 1.0]]).repeat(b, 1),
        v_max_ms=torch.full((b,), float(v_max_ms)),
        v_max_valid=torch.ones(b, dtype=torch.bool),
    )
    if nav is not None:
        kw["nav_cmd"] = torch.full((b,), int(nav), dtype=torch.long)
    return frames, kw
