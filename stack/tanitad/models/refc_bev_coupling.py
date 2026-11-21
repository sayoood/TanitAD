"""refcv6 §1: DiffusionDrive **coupling (1)** -- BEV features at the candidate's
own waypoints.

``SPEC_REFCV6_V2.md`` §1, the decoder's per-layer reads::

    (1) BEV sampled at the candidate's own waypoints        [DiffusionDrive coupling (1)]
    (2) agent slots addressed by waypoint                   [WP-B]
    (3) image tokens by content                             [direct trunk access]

(2) already exists end to end in ``refc.py`` (``attach_wp_index`` /
``_agent_index`` / ``CrossAttnLayer._agent_bias``) and needs only its
``agent_pos`` argument wired -- see ``integration/refc_wiring.patch``. (1) is
this module.

## ⚠️ What the RELEASED DiffusionDrive actually does

``…/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/blocks.py:80-108``
(primary source, read 2026-09-16)::

    normalized_trajectory = traj_points.clone()
    normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
    normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x
    normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y
    attention_weights = self.attention_weights(queries)
    attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)
    value = self.value_proj(bev_feature)
    grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
    sampled_features = torch.nn.functional.grid_sample(value, grid, mode='bilinear', ...)

⭐ **The sample positions are the waypoints themselves. There are NO learned
offsets** -- the only learned thing about WHERE it looks is the softmax over the
``num_points`` waypoints (`attention_weights`), computed from the query. That is
the default here (:attr:`BEVCouplingConfig.learned_offsets` ``= False``) and it
is what a "DiffusionDrive coupling (1)" arm must run.

⛔ **Learned offsets are OUR EXTENSION.** They are implemented, switchable and
**flagged**: :meth:`BEVWaypointSampler.provenance` returns
``"ddv2-faithful"`` or ``"tanitad-extension"``, and
:attr:`BEVWaypointSampler.is_extension` is what a trainer stamps into
``config.json``. An arm that runs with offsets on and reports itself as
"DiffusionDrive's coupling" is a mislabelled arm, not a better one.

## The grid, exactly once

The metre -> ``grid_sample`` map is :func:`waypoints_to_bev_grid`, and it is the
BEV twin of ``bev_lift.pixel_to_feature_grid``. Cell centres come from
``bev_raster._cell_centers`` (``bev_raster.py:115-120``)::

    x_centre[i] = (i + 0.5) * cell_m                 row    (the [B,C,X,Y] X axis)
    y_centre[j] = -y_half_m + (j + 0.5) * cell_m     column (the Y axis)

so the fractional indices of a metric point are ``ix = x/cell - 0.5`` and
``iy = (y + y_half)/cell - 0.5``, and ``align_corners=False`` normalisation is
``(2k + 1)/n - 1``. ⛔ ``grid_sample``'s LAST grid component indexes the tensor's
SECOND-TO-LAST axis: with a ``[B, C, X, Y]`` BEV map the grid is ``(gy_from_y,
gx_from_x)`` -- the transposition DiffusionDrive writes as ``[..., [1, 0]]``.
Getting it backwards is a silent left-right/forward mix-up whenever ``X == Y``;
here it is a shape-preserving wrong answer, so ``tests/test_refcv6_bev_coupling.py``
carries the mutation.

## Removability

The output is ``gate * f(...)`` with ``gate`` a **zero-init scalar**
(the ``agent_gate`` / ``ctx_to_cond`` discipline). With ``gate == 0`` the module
returns its input queries **bitwise** -- not approximately: the addition of an
exactly-zero tensor is exact in IEEE754 for every finite input, and
:meth:`BEVWaypointSampler.forward` short-circuits BEFORE ``grid_sample`` when the
coupling is disabled, so no backend dispatch can differ either (the reason
``CrossAttnLayer._agent_bias`` returns ``None`` rather than a zero mask).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.data.bev_raster import GRID_DEFAULT, BEVGrid

__all__ = [
    "BEVCouplingConfig", "BEVWaypointSampler", "waypoints_to_bev_grid",
    "DDV2_BLOCKS_CITATION",
]

DDV2_BLOCKS_CITATION = (
    "TanitAD Research Lab/Architecture & Inference/Research/"
    "2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/blocks.py:80-108"
)


def waypoints_to_bev_grid(wp: Tensor, grid: BEVGrid = GRID_DEFAULT) -> Tensor:
    """Metric ego-frame waypoints ``[..., 2] = (x fwd, y LEFT)`` ->
    ``grid_sample`` coordinates for a ``[B, C, X, Y]`` BEV map, ``[..., 2]``.

    ``align_corners=False``. Returns ``(g_col, g_row)`` in ``grid_sample``'s own
    order: the FIRST component indexes the last tensor axis (Y / lateral) and
    the SECOND indexes the second-to-last (X / forward). See the module
    docstring for why that ordering is the whole trap.
    """
    if wp.shape[-1] != 2:
        raise ValueError(f"waypoints must end in 2 (x, y), got {tuple(wp.shape)}")
    nx, ny = grid.shape
    cell = float(grid.cell_m)
    ix = wp[..., 0] / cell - 0.5
    iy = (wp[..., 1] + float(grid.y_half_m)) / cell - 0.5
    g_col = (2.0 * iy + 1.0) / ny - 1.0        # -> the Y (last) axis
    g_row = (2.0 * ix + 1.0) / nx - 1.0        # -> the X (second-to-last) axis
    return torch.stack([g_col, g_row], dim=-1)


@dataclass(frozen=True)
class BEVCouplingConfig:
    """Declared decisions of :class:`BEVWaypointSampler`.

    ``learned_offsets``: ⚠️ **OUR EXTENSION, off by default.** The released
    DiffusionDrive samples at the waypoints themselves (module docstring). With
    it on, each (query, waypoint) also predicts a metric ``(dx, dy)`` offset,
    bounded by ``offset_max_m`` through a ``tanh``, and the module reports itself
    as ``tanitad-extension``.

    ``padding_mode``: DD uses ``'zeros'`` (``blocks.py:100``), which is kept --
    a waypoint outside the 60 m x +-16 m grid must read as "nothing there", not
    as a copy of the nearest edge cell.
    """

    enable: bool = True
    d_model: int = 256
    d_bev: int = 96
    n_points: int = 8
    learned_offsets: bool = False
    offset_max_m: float = 2.0
    padding_mode: str = "zeros"
    grid: BEVGrid = GRID_DEFAULT

    def __post_init__(self) -> None:
        if self.padding_mode not in ("zeros", "border", "reflection"):
            raise ValueError(f"padding_mode {self.padding_mode!r} is not a "
                             f"grid_sample mode")
        if int(self.n_points) < 1:
            raise ValueError(f"n_points must be >= 1, got {self.n_points}")
        if float(self.offset_max_m) <= 0:
            raise ValueError("offset_max_m must be > 0")

    def as_dict(self) -> dict:
        return {"enable": bool(self.enable), "d_model": int(self.d_model),
                "d_bev": int(self.d_bev), "n_points": int(self.n_points),
                "learned_offsets": bool(self.learned_offsets),
                "offset_max_m": float(self.offset_max_m),
                "padding_mode": str(self.padding_mode),
                "grid": list(self.grid.shape),
                "provenance": ("tanitad-extension" if self.learned_offsets
                               else "ddv2-faithful"),
                "released_source": DDV2_BLOCKS_CITATION}


class BEVWaypointSampler(nn.Module):
    """``(queries, waypoints, bev) -> queries`` with the BEV read at the
    candidate's own waypoints, behind a zero-init gate.

    ``queries`` ``[B, N, d_model]`` · ``wp`` ``[B, N, S, 2]`` metres, ego frame
    · ``bev`` ``[B, d_bev, X, Y]``.

    Follows ``blocks.py:88-108`` term for term: a query-conditioned softmax over
    the ``S`` waypoints, a 1x1 ``value_proj`` of the BEV map, bilinear
    ``grid_sample`` at the waypoints, the weighted sum, an ``output_proj``, and a
    residual add. The two departures are declared: the gate (so the coupling is
    removable and provably inert when off) and the optional offsets (flagged).

    ⛔ ``S`` need not equal ``n_points``: the softmax head is built for
    ``n_points`` and REFUSES another length rather than resizing, because a
    resize would silently re-weight a different set of waypoints.
    """

    def __init__(self, cfg: BEVCouplingConfig | None = None):
        super().__init__()
        self.cfg = cfg or BEVCouplingConfig()
        c = self.cfg
        self.attention_weights = nn.Linear(c.d_model, c.n_points)
        self.value_proj = nn.Conv2d(c.d_bev, c.d_model, 1)
        self.output_proj = nn.Linear(c.d_model, c.d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.offset_head = (nn.Linear(c.d_model, c.n_points * 2)
                            if c.learned_offsets else None)
        if self.offset_head is not None:
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)

    # ---- provenance ------------------------------------------------------- #
    @property
    def is_extension(self) -> bool:
        """True when this build departs from the released DiffusionDrive."""
        return bool(self.cfg.learned_offsets)

    def provenance(self) -> str:
        return "tanitad-extension" if self.is_extension else "ddv2-faithful"

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def param_breakdown(self) -> dict:
        out = {n: int(sum(p.numel() for p in m.parameters()))
               for n, m in (("attention_weights", self.attention_weights),
                            ("value_proj", self.value_proj),
                            ("output_proj", self.output_proj))}
        out["gate"] = int(self.gate.numel())
        out["offset_head"] = (0 if self.offset_head is None
                              else int(sum(p.numel() for p
                                           in self.offset_head.parameters())))
        out["total"] = self.n_params
        return out

    # ---- forward ---------------------------------------------------------- #
    def sample_positions(self, queries: Tensor, wp: Tensor) -> Tensor:
        """The metric positions actually sampled: the waypoints, plus the
        bounded offsets when this is the extension build."""
        if self.offset_head is None:
            return wp
        b, n = wp.shape[:2]
        d = self.offset_head(queries).reshape(b, n, self.cfg.n_points, 2)
        return wp + torch.tanh(d) * float(self.cfg.offset_max_m)

    def forward(self, queries: Tensor, wp: Tensor, bev: Tensor | None) -> Tensor:
        if not self.cfg.enable or bev is None:
            return queries                      # bitwise identity, no dispatch
        if queries.dim() != 3:
            raise ValueError(f"queries must be [B, N, d], got {tuple(queries.shape)}")
        if wp.dim() != 4 or wp.shape[-1] != 2:
            raise ValueError(f"wp must be [B, N, S, 2], got {tuple(wp.shape)}")
        if wp.shape[:2] != queries.shape[:2]:
            raise ValueError(f"wp {tuple(wp.shape[:2])} and queries "
                             f"{tuple(queries.shape[:2])} disagree on [B, N]")
        if wp.shape[2] != self.cfg.n_points:
            raise ValueError(
                f"this sampler weights {self.cfg.n_points} waypoints but got "
                f"{wp.shape[2]}. ⛔ Not resized: the softmax would then weight a "
                f"different set of positions under the same learned head.")
        if bev.dim() != 4 or bev.shape[1] != self.cfg.d_bev:
            raise ValueError(f"bev must be [B, {self.cfg.d_bev}, X, Y], got "
                             f"{tuple(bev.shape)}")
        if tuple(bev.shape[2:]) != tuple(self.cfg.grid.shape):
            raise ValueError(f"bev is {tuple(bev.shape[2:])}, the configured BEV "
                             f"grid is {tuple(self.cfg.grid.shape)}")
        b, n, s, _ = wp.shape
        pos = self.sample_positions(queries, wp)
        g = waypoints_to_bev_grid(pos, self.cfg.grid).to(bev.dtype)
        w = self.attention_weights(queries).softmax(-1)          # [B, N, S]
        value = self.value_proj(bev)                             # [B, d, X, Y]
        sampled = F.grid_sample(value, g.reshape(b, n, s, 2), mode="bilinear",
                                padding_mode=self.cfg.padding_mode,
                                align_corners=False)             # [B, d, N, S]
        out = (w.unsqueeze(1) * sampled).sum(dim=-1)             # [B, d, N]
        out = self.output_proj(out.permute(0, 2, 1).contiguous())
        return queries + self.gate * out


if __name__ == "__main__":
    for lo in (False, True):
        m = BEVWaypointSampler(BEVCouplingConfig(learned_offsets=lo))
        print(f"BEVWaypointSampler(learned_offsets={lo}) [{m.provenance()}]: "
              f"{m.n_params:,} params  {m.param_breakdown()}")
