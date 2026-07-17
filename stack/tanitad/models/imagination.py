"""H15 — Imagination in unobserved areas (Phase 0 scope, D-008).

The capability: like a human driver, the model keeps an internal belief about
traffic it is NOT currently observing (occluded agents, gated-off sensors,
areas outside the active field of view). This is the principled enabler of H2
(Attention-based Modality Steering): a sensor may only be powered down when
the imagination's uncertainty in its field of view is low.

Three mechanisms, trained jointly with the world model:

1. Sector-masked imagination training. Whole spatial sectors of the input
   frame are hidden (simulating a gated sensor / occlusion); the model must
   still predict the next-step latent content of those cells from context and
   dynamics. This converts masking from an augmentation into an *imagination
   objective*.

2. Latent advection prior (Deep Think 2). Each token cell carries a learned
   2-D flow on the token grid; hidden cells evolve semi-Lagrangian:
   z_hat_{t+1}(x) = z_t(x - v(x)dt). Object permanence by construction — a
   pedestrian latent behind a bus keeps moving while unobserved.

3. Epistemic gating. A per-cell log-variance head; the imagination loss is a
   heteroscedastic Gaussian NLL, so the model must KNOW where it cannot know.
   Uncertainty grows with occlusion duration and is exported as the trigger
   signal for modality steering (H2), fallback margins (brain 4), and the
   LOPS metric.

Gate D9 (Phase 0): advected-prior cosine similarity in hidden sectors must
beat a shuffled-cell baseline by a clear margin, and log-variance must be
higher in hidden cells than visible ones (calibration sanity). Object-level
LOPS on scripted occluders (Ghost Cut-Through) lands with the MetaDrive
wrapper (WP2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.models.encoder import Block

SECTORS = ("left", "right", "top", "bottom")


def sector_mask(frames: Tensor, grid_hw: int,
                generator: torch.Generator | None = None
                ) -> tuple[Tensor, Tensor]:
    """Hide one random half-sector per sample.

    frames: [B, C, H, W]. Returns (masked frames, token visibility [B, N])
    with N = grid_hw**2, 1.0 = visible, 0.0 = hidden.
    """
    b, _, h, w = frames.shape
    device = frames.device
    masked = frames.clone()
    vis = torch.ones(b, grid_hw, grid_hw, device=device)
    idx = torch.randint(0, len(SECTORS), (b,), generator=generator, device=device)
    for i in range(b):
        s = SECTORS[int(idx[i])]
        if s == "left":
            masked[i, :, :, : w // 2] = 0.0
            vis[i, :, : grid_hw // 2] = 0.0
        elif s == "right":
            masked[i, :, :, w // 2:] = 0.0
            vis[i, :, grid_hw // 2:] = 0.0
        elif s == "top":
            masked[i, :, : h // 2, :] = 0.0
            vis[i, : grid_hw // 2, :] = 0.0
        else:
            masked[i, :, h // 2:, :] = 0.0
            vis[i, grid_hw // 2:, :] = 0.0
    return masked, vis.reshape(b, -1)


def advect(tokens: Tensor, flow: Tensor, grid_hw: int) -> Tensor:
    """Semi-Lagrangian warp of a token grid by a per-cell flow (in cell units).

    tokens: [B, N, D], flow: [B, N, 2] (dx, dy). value'(x) = value(x - v(x)).
    """
    b, n, d = tokens.shape
    x = tokens.transpose(1, 2).reshape(b, d, grid_hw, grid_hw)
    ys, xs = torch.meshgrid(
        torch.arange(grid_hw, device=tokens.device, dtype=tokens.dtype),
        torch.arange(grid_hw, device=tokens.device, dtype=tokens.dtype),
        indexing="ij")
    base = torch.stack([xs, ys], dim=-1)                       # [h, w, 2]
    f = flow.reshape(b, grid_hw, grid_hw, 2)
    pos = base.unsqueeze(0) - f                                # sample source
    pos = 2.0 * pos / max(grid_hw - 1, 1) - 1.0                # -> [-1, 1]
    warped = F.grid_sample(x, pos, mode="bilinear",
                           padding_mode="border", align_corners=True)
    return warped.flatten(2).transpose(1, 2)


class ImaginationField(nn.Module):
    """Belief maintenance over the token grid: advect -> refine -> quantify."""

    def __init__(self, d_model: int, grid_hw: int, depth: int = 3,
                 n_heads: int = 12):
        super().__init__()
        self.grid_hw = grid_hw
        self.flow_head = nn.Sequential(
            nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, 2))
        nn.init.zeros_(self.flow_head[-1].weight)              # start at identity
        nn.init.zeros_(self.flow_head[-1].bias)
        self.vis_emb = nn.Embedding(2, d_model)                # hidden/visible tag
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d_model)
        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, 1))

    def forward(self, tokens: Tensor, vis: Tensor) -> tuple[Tensor, Tensor]:
        """tokens [B, N, D] from the (partially masked) current frame;
        vis [B, N] in {0, 1}. Returns (imagined next tokens [B, N, D],
        per-cell log-variance [B, N])."""
        flow = self.flow_head(tokens)
        prior = advect(tokens, flow, self.grid_hw)
        x = prior + self.vis_emb(vis.long())
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # numerics guard (prod-opt 2026-07-17): bounded logvar - exp overflow
        return x, self.logvar_head(x).squeeze(-1).clamp(-10.0, 10.0)


def imagination_nll(pred: Tensor, target: Tensor, logvar: Tensor, vis: Tensor,
                    observed_weight: float = 0.1) -> Tensor:
    """Heteroscedastic Gaussian NLL, emphasis on hidden cells.

    pred/target: [B, N, D]; logvar/vis: [B, N]. Hidden cells (vis=0) carry
    full weight — that is the imagination objective; visible cells get a small
    consistency weight so the field stays anchored where it can see.
    """
    err2 = (pred - target).pow(2).mean(dim=-1)                 # [B, N]
    logvar = logvar.clamp(-10.0, 10.0)   # defensive exp-overflow guard
    nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)
    w = (1.0 - vis) + observed_weight * vis
    return (w * nll).sum() / w.sum().clamp_min(1e-8)


@torch.no_grad()
def d9_rows(pred: Tensor, target: Tensor, logvar: Tensor, vis: Tensor) -> dict:
    """Gate D9 evidence rows (with the shuffled-cell baseline as instrument).

    - hidden_cosine: cosine(pred, target) on hidden cells
    - shuffled_cosine: same but targets shuffled across cells (chance floor)
    - calibration_gap: mean logvar(hidden) - mean logvar(visible)  (must be > 0)
    """
    hidden = vis < 0.5
    if hidden.sum() == 0:
        return {"hidden_cosine": float("nan"), "shuffled_cosine": float("nan"),
                "calibration_gap": float("nan")}
    p = F.normalize(pred[hidden], dim=-1)
    t = F.normalize(target[hidden], dim=-1)
    perm = torch.randperm(t.shape[0], device=t.device)
    return {
        "hidden_cosine": float((p * t).sum(-1).mean()),
        "shuffled_cosine": float((p * t[perm]).sum(-1).mean()),
        "calibration_gap": float(logvar[hidden].mean() - logvar[~hidden].mean()),
    }
