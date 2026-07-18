"""REF-A improvement variants (Phase-2 prep) -- ISOLATED copy, does NOT touch
tanitad.refs.refa. Two model-side fixes as toggles:

  FIX 2  TemporalGridAdapter : motion-aware adapter -- per-frame DINO grid PLUS
         frame-to-frame delta, grid-pooled + projected. out_dim == DinoGridAdapter
         (grid^2*d_readout = 2048) so the predictor / state_dim are unchanged.
  FIX 1b aux heads           : ego-speed + yaw-rate regressors on the latent
         (built by build_aux_heads, kept OUTSIDE the model like the metric heads
         -> a grid-adapter checkpoint's ckpt['model'] stays vanilla-loadable).

RefAModelPlus subclasses the real RefAModel; adapter_kind='grid'/'pool' behave
exactly as upstream, adapter_kind='temporal' swaps in the motion adapter.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from tanitad.models.readout import SpatialGridReadout
from tanitad.refs.refa import RefAModel


class TemporalGridAdapter(nn.Module):
    """[.,N,D] grid + frame-to-frame delta -> SpatialGridReadout(2D) -> [.,out].

    Window path (forward_window, [b,W,N,D]) computes real temporal deltas along
    W; single-frame path (forward_single, [b,N,D]) uses a zero delta. out_dim is
    grid*grid*d_readout == the DinoGridAdapter's, so state_dim is unchanged."""

    def __init__(self, n_tokens: int = 256, d_in: int = 768, grid: int = 4,
                 d_readout: int = 128):
        super().__init__()
        self.readout = SpatialGridReadout(n_tokens, d_in * 2, grid=grid,
                                          d_readout=d_readout)
        self.out_dim = self.readout.out_dim

    def _fuse(self, std: Tensor, delta: Tensor) -> Tensor:
        x = torch.cat([std, delta], dim=-1)             # [.,N,2D]
        lead = x.shape[:-2]
        return self.readout(x.reshape(-1, *x.shape[-2:])).reshape(*lead,
                                                                  self.out_dim)

    def forward_window(self, std_win: Tensor) -> Tensor:   # [b,W,N,D] standardized
        delta = torch.zeros_like(std_win)
        delta[:, 1:] = std_win[:, 1:] - std_win[:, :-1]
        return self._fuse(std_win, delta)

    def forward_single(self, std: Tensor) -> Tensor:       # [b,N,D] standardized
        return self._fuse(std, torch.zeros_like(std))


class RefAModelPlus(RefAModel):
    """RefAModel + optional temporal adapter (adapter_kind='temporal')."""

    def __init__(self, pred_cfg=None, adapter_kind: str = "grid",
                 n_tokens: int = 256, *, grid: int = 4,
                 grid_d_readout: int = 128, **kw):
        temporal = adapter_kind == "temporal"
        super().__init__(pred_cfg,
                         adapter_kind=("grid" if temporal else adapter_kind),
                         n_tokens=n_tokens, grid=grid,
                         grid_d_readout=grid_d_readout, **kw)
        if temporal:
            self.adapter = TemporalGridAdapter(n_tokens, kw.get("d_dino", 768),
                                               grid=grid, d_readout=grid_d_readout)
            assert self.adapter.out_dim == self.state_dim, \
                (self.adapter.out_dim, self.state_dim)
            self.adapter_kind = "temporal"

    def encode(self, feats: Tensor) -> Tensor:
        if self.adapter_kind == "temporal":
            std = self.standardizer(feats)
            return (self.adapter.forward_window(std) if std.dim() == 4
                    else self.adapter.forward_single(std))
        return super().encode(feats)

    def encode_window(self, feats: Tensor) -> Tensor:
        if self.adapter_kind == "temporal":
            return self.adapter.forward_window(self.standardizer(feats))
        return super().encode(feats)                    # grid handles [b,W,N,D]


def build_aux_heads(state_dim: int, device: str = "cpu", hidden: int = 256,
                    which=("speed", "yaw")) -> dict:
    """FIX 1b/accel: per-property regressors on the latent (ego-speed, yaw-rate,
    longitudinal accel). Kept OUTSIDE the model (saved under ckpt keys
    'aux_speed'/'aux_yaw'/'aux_accel', like the metric heads)."""
    def mk():
        return nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(),
                             nn.Linear(hidden, 1))
    return {f"aux_{w}": mk().to(device) for w in which}
