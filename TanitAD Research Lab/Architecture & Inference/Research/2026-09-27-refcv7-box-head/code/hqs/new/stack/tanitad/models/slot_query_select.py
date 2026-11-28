"""refcv7 A14 (SPEC_REFCV7 §19): HEATMAP QUERY SELECTION (HQS) for the BOX head -- DINO-style mixed query selection.

THE DEFECT IT FIXES (MEASURED, the one-frame ladder, ``…/2026-09-27-refcv7-box-head/raw/diag/gbo_ladder.json``): on ONE
fixed frame the slot decoder's Hungarian assignment never stabilises -- only 0.4-6.5 % of targets keep their slot over
25 steps in every rung (MAIN, frozen trunk, lr 2e-5, BCE presence, no deep supervision, 100 queries, and the refcv6 head
itself) -- so no slot can learn "I am matched" and presence sits at its base rate. That is the known bipartite-matching
instability of learned, UN-ANCHORED queries with absolute box regression (DN-DETR, DINO, anchored queries).

THE LEVER (verbatim from A14):
  * HEATMAP: 2 conv layers -> 1 channel on the box memory's BEV features (the planner grid, 60 m x +-16 m at 0.5 m;
    under the canonical config NEW-2's pooled BEV), prior 0.01. Target: CenterNet Gaussian splats of the VIS-1
    POSITIVE centres with IGNORE cells masked; loss: penalty-reduced focal (alpha 2, beta 4) / the number of
    positives, weight 1.0, INSIDE the box3d term.
  * SELECTION: per frame, 3x3 max-pool NMS, then top-K with K = n_queries. Each selected cell is that query's ANCHOR.
    The query CONTENT stays the learned table; its POSITION is a sine embedding of the anchor through an MLP, added
    at every decoder layer (to the self-attention query/key and the cross-attention query -- DETR's ``query_pos``).
    Anchors are detached; the selection is non-differentiable.
  * BOX CENTRE = anchor + tanh(raw) x 4 m. Every other field decodes exactly as today.

``--slot-query-select learned`` (the DEFAULT) never reaches this module: the branch builds no heat head, no position
embed, and calls ``box_dec(mem)`` exactly as before (pinned by a digest test and a raise-on-touch test).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: "learned" = the A9 build (DEFAULT, bit-identical); "heatmap" = HQS (A14); "learned_ref" = LEARNED REFERENCE
#: POINTS -- one trainable (x, y) anchor per query (DAB-DETR / Anchor-DETR static anchors), the same position
#: embed and anchor-relative centre, no heatmap (the one-frame bench's best arm, 2026-09-27).
QUERY_SELECT: tuple[str, ...] = ("learned", "heatmap", "learned_ref")
HEAT_PRIOR = 0.01
ANCHOR_OFFSET_M = 4.0
NMS_KERNEL = 3
CN_ALPHA, CN_BETA = 2.0, 4.0
HEAT_LOSS_W = 1.0
#: CenterPoint's BEV Gaussian: radius from the footprint in CELLS at min overlap 0.1, never below 2 cells
GAUSS_MIN_OVERLAP = 0.1
GAUSS_MIN_RADIUS = 2
POS_FREQS = 64            # sine frequencies per coordinate -> 4 x 64 = 256 features


# --------------------------------------------------------------------------------------------------------- #
# the grid                                                                                                  #
# --------------------------------------------------------------------------------------------------------- #
def grid_centres(grid, device=None) -> Tensor:
    """``[H, W, 2]`` (x, y) metres of every cell centre of a ``bev_raster.BEVGrid`` (rows = x from 0 forward,
    columns = y from -y_half left-to-right) -- the SAME centres ``bev_raster.cell_centers_xy`` rasterises against."""
    nx, ny = grid.shape
    xc = (torch.arange(nx, dtype=torch.float32, device=device) + 0.5) * float(grid.cell_m)
    yc = -float(grid.y_half_m) + (torch.arange(ny, dtype=torch.float32, device=device) + 0.5) * float(grid.cell_m)
    return torch.stack(torch.meshgrid(xc, yc, indexing="ij"), dim=-1)


def gaussian_radius(l_cells: float, w_cells: float, min_overlap: float = GAUSS_MIN_OVERLAP) -> int:
    """CenterNet's ``gaussian_radius`` on the box FOOTPRINT (length x width, in cells), as CenterPoint uses it."""
    height, width = float(l_cells), float(w_cells)
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * c1)) / 2
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2 ** 2 - 16 * c2)) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return max(GAUSS_MIN_RADIUS, int(min(r1, r2, r3)))


@torch.no_grad()
def draw_targets(box: Tensor, pos: Tensor, ign: Tensor | None, grid) -> dict:
    """``box`` [B, A, >=4] (cx, cy, l, w metres), ``pos`` / ``ign`` [B, A] bool -> ``{"heat" [B, H, W] in [0, 1],
    "ignore" [B, H, W] bool, "n_pos" int, "n_pos_outside" int}``. A POSITIVE's centre cell reads exactly 1.0;
    cells inside an IGNORE row's Gaussian are masked unless a positive's Gaussian also covers them."""
    B = int(box.shape[0])
    nx, ny = grid.shape
    cell = float(grid.cell_m)
    heat = torch.zeros(B, nx, ny, dtype=torch.float32, device=box.device)
    ignore = torch.zeros(B, nx, ny, dtype=torch.bool, device=box.device)
    n_pos = n_out = 0

    def splat(b, cx, cy, l, w, into_heat: bool):
        ix, iy = int(math.floor(cx / cell)), int(math.floor((cy + float(grid.y_half_m)) / cell))
        if not (0 <= ix < nx and 0 <= iy < ny):
            return False
        r = gaussian_radius(max(l / cell, 1.0), max(w / cell, 1.0))
        sigma = (2 * r + 1) / 6.0
        x0, x1, y0, y1 = max(0, ix - r), min(nx, ix + r + 1), max(0, iy - r), min(ny, iy + r + 1)
        dx = torch.arange(x0, x1, device=box.device, dtype=torch.float32)[:, None] - ix
        dy = torch.arange(y0, y1, device=box.device, dtype=torch.float32)[None, :] - iy
        g = torch.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
        if into_heat:
            heat[b, x0:x1, y0:y1] = torch.maximum(heat[b, x0:x1, y0:y1], g)
            heat[b, ix, iy] = 1.0
        else:
            ignore[b, x0:x1, y0:y1] = True
        return True
    for b in range(B):
        for a in torch.nonzero(pos[b], as_tuple=False).flatten().tolist():
            cx, cy, l, w = (float(v) for v in box[b, a, :4])
            if splat(b, cx, cy, l, w, True):
                n_pos += 1
            else:
                n_out += 1
        if ign is not None:
            for a in torch.nonzero(ign[b], as_tuple=False).flatten().tolist():
                cx, cy, l, w = (float(v) for v in box[b, a, :4])
                splat(b, cx, cy, l, w, False)
    ignore &= heat <= 0.0                   # a positive's Gaussian wins over an IGNORE region
    return {"heat": heat, "ignore": ignore, "n_pos": n_pos, "n_pos_outside": n_out}


def centernet_focal(logits: Tensor, heat: Tensor, ignore: Tensor | None = None, n_pos: int | None = None) -> Tensor:
    """CenterNet's penalty-reduced focal loss (alpha 2, beta 4) summed over cells / max(n_pos, 1). ``logits`` [B, H, W];
    ``heat`` the Gaussian target (exactly 1 at a positive's centre); ``ignore`` cells contribute nothing."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    posm = heat >= 1.0
    keep = torch.ones_like(heat, dtype=torch.bool) if ignore is None else ~ignore
    pos_l = -torch.log(p) * (1 - p) ** CN_ALPHA
    neg_l = -torch.log(1 - p) * p ** CN_ALPHA * (1 - heat) ** CN_BETA
    total = (pos_l * posm).sum() + (neg_l * (~posm & keep)).sum()
    n = int(posm.sum()) if n_pos is None else int(n_pos)
    return total / float(max(n, 1))


# --------------------------------------------------------------------------------------------------------- #
# the modules                                                                                               #
# --------------------------------------------------------------------------------------------------------- #
class BEVHeatHead(nn.Module):
    """2 conv layers -> 1 channel: ``[B, C, H, W]`` BEV features -> ``[B, H, W]`` centre logits (prior 0.01)."""

    def __init__(self, d_bev: int):
        super().__init__()
        self.conv1 = nn.Conv2d(int(d_bev), int(d_bev), 3, padding=1)
        self.conv2 = nn.Conv2d(int(d_bev), 1, 1)
        with torch.no_grad():
            self.conv2.bias.fill_(math.log(HEAT_PRIOR / (1.0 - HEAT_PRIOR)))

    def forward(self, bev: Tensor) -> Tensor:
        return self.conv2(F.gelu(self.conv1(bev)))[:, 0]


class LearnedRefPoints(nn.Module):
    """``learned_ref``: one TRAINABLE (x, y) anchor per query, initialised uniform over the planner grid (a seeded
    draw from the module's own generator, so the init does not consume the global RNG stream)."""

    def __init__(self, n_queries: int, grid, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        k = int(n_queries)
        xy = torch.stack([torch.rand(k, generator=g) * float(grid.x_fwd_m),
                          (torch.rand(k, generator=g) * 2 - 1) * float(grid.y_half_m)], dim=-1)
        self.xy = nn.Parameter(xy)

    def forward(self, batch: int) -> Tensor:
        return self.xy[None].expand(int(batch), -1, -1)


class AnchorPosEmbed(nn.Module):
    """The anchor's POSITION as a query embedding: sine features of the anchor (normalised to the grid) -> MLP."""

    def __init__(self, d_model: int, grid, n_freqs: int = POS_FREQS):
        super().__init__()
        self.x_fwd_m, self.y_half_m = float(grid.x_fwd_m), float(grid.y_half_m)
        self.register_buffer("freqs", 2.0 * math.pi * torch.arange(1, int(n_freqs) + 1, dtype=torch.float32),
                             persistent=False)
        self.mlp = nn.Sequential(nn.Linear(4 * int(n_freqs), int(d_model)), nn.ReLU(),
                                 nn.Linear(int(d_model), int(d_model)))

    def forward(self, anchors_xy: Tensor) -> Tensor:
        u = anchors_xy[..., 0:1] / self.x_fwd_m
        v = (anchors_xy[..., 1:2] + self.y_half_m) / (2.0 * self.y_half_m)
        f = self.freqs.to(anchors_xy.dtype)
        feats = torch.cat([torch.sin(u * f), torch.cos(u * f), torch.sin(v * f), torch.cos(v * f)], dim=-1)
        return self.mlp(feats)


# --------------------------------------------------------------------------------------------------------- #
# selection + the anchored decoder pass                                                                     #
# --------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def select_anchors(logits: Tensor, k: int, grid) -> tuple[Tensor, Tensor, Tensor]:
    """3x3 max-pool NMS on sigmoid(logits), then the top-``k`` cells per frame, sorted by score (DINO).
    -> (anchors_xy [B, k, 2] metres, scores [B, k], flat cell index [B, k]); all DETACHED."""
    p = torch.sigmoid(logits.detach().float())
    pooled = F.max_pool2d(p[:, None], NMS_KERNEL, stride=1, padding=NMS_KERNEL // 2)[:, 0]
    p_nms = torch.where(p == pooled, p, torch.zeros_like(p))
    B = int(p.shape[0])
    k = int(k)
    if k > p_nms[0].numel():
        raise ValueError(f"top-{k} over a {tuple(p.shape[1:])} grid")
    scores, idx = torch.topk(p_nms.reshape(B, -1), k, dim=1, sorted=True)
    centres = grid_centres(grid, device=logits.device).reshape(-1, 2)
    return centres[idx], scores, idx


def heat_grid_of(planner_grid, heat_logits: Tensor):
    """The grid the heatmap lives on: the planner grid's extent at the heat tensor's OWN resolution (the BEV encoder
    is stride 1 and the pool lands on the planner grid, so this is the 0.5 m grid -- asserted, never assumed)."""
    from tanitad.data.bev_raster import BEVGrid
    H, W = int(heat_logits.shape[-2]), int(heat_logits.shape[-1])
    cell = float(planner_grid.x_fwd_m) / H
    if abs(2.0 * float(planner_grid.y_half_m) / W - cell) > 1e-6:
        raise ValueError(f"heatmap {H}x{W} is not a square-cell grid over {planner_grid}")
    return BEVGrid(x_fwd_m=float(planner_grid.x_fwd_m), y_half_m=float(planner_grid.y_half_m), cell_m=cell)


def choose_anchors(branch, heat_logits: Tensor) -> tuple[Tensor, Tensor]:
    """The branch's anchors for this forward: the heatmap's top-K (HQS). A module-level seam so the one-frame red
    arm ``anchors_removed`` (A14) can swap in learned reference points WITHOUT a product flag."""
    grid = heat_grid_of(branch.cfg.planner_grid, heat_logits)
    a, s, _ = select_anchors(heat_logits, int(branch.box_dec.n_queries), grid)
    return a, s


def _layer_with_pos(layer: nn.TransformerDecoderLayer, x: Tensor, mem: Tensor, qpos: Tensor) -> Tensor:
    """One PRE-NORM ``nn.TransformerDecoderLayer`` step with ``qpos`` added to the self-attention query/key and to the
    cross-attention query (DETR's ``query_pos``), through the layer's OWN modules -- no new parameter."""
    if not bool(getattr(layer, "norm_first", False)):
        raise RuntimeError("HQS's per-layer query position assumes the pre-norm decoder layer (norm_first=True)")
    h = layer.norm1(x)
    q = h + qpos
    x = x + layer.dropout1(layer.self_attn(q, q, h, need_weights=False)[0])
    h = layer.norm2(x)
    x = x + layer.dropout2(layer.multihead_attn(h + qpos, mem, mem, need_weights=False)[0])
    h = layer.norm3(x)
    return x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(h)))))


def anchored_decode(dec, raw: Tensor, anchors_xy: Tensor) -> dict:
    """``dec.decode(raw)`` with the box CENTRE re-read as anchor + tanh(raw) x 4 m; every other field unchanged."""
    from tanitad.models.agent_slots import SLOT_SLICES
    out = dec.decode(raw)
    s = SLOT_SLICES
    cx = anchors_xy[..., 0].to(raw.dtype) + torch.tanh(raw[..., s["cx"]].squeeze(-1)) * ANCHOR_OFFSET_M
    cy = anchors_xy[..., 1].to(raw.dtype) + torch.tanh(raw[..., s["cy"]].squeeze(-1)) * ANCHOR_OFFSET_M
    out["box"] = torch.cat([torch.stack([cx, cy], dim=-1), out["box"][..., 2:]], dim=-1)
    out["anchors"] = anchors_xy
    return out


def anchored_forward(dec, memory: Tensor, anchors_xy: Tensor, qpos: Tensor) -> dict:
    """The slot decoder's forward with anchored queries: the SAME memory projection, query table, layers, norm and
    head as ``dec.forward``; ``qpos`` [B, K, d_model] added at every layer; per-layer outputs under ``"aux"`` when
    ``dec.deep_supervision`` (each decoded against the same anchors)."""
    if memory.ndim != 3 or memory.shape[1] != dec.n_memory or memory.shape[2] != dec.d_memory:
        raise ValueError(f"memory must be [B, {dec.n_memory}, {dec.d_memory}], got {tuple(memory.shape)}")
    if dec.blocks.norm is not None:
        raise RuntimeError("HQS assumes no stack-level norm (as deep supervision does)")
    b = int(memory.shape[0])
    if tuple(anchors_xy.shape) != (b, dec.n_queries, 2) or tuple(qpos.shape) != (b, dec.n_queries, dec.d_model):
        raise ValueError(f"anchors {tuple(anchors_xy.shape)} / qpos {tuple(qpos.shape)} do not fit "
                         f"[{b}, {dec.n_queries}, 2] / [{b}, {dec.n_queries}, {dec.d_model}]")
    mem = dec.mem_proj(memory) + dec.mem_pos.to(memory.dtype)
    x = dec.queries.to(memory.dtype).expand(b, -1, -1)
    qpos = qpos.to(memory.dtype)
    raws = []
    for layer in dec.blocks.layers:
        x = _layer_with_pos(layer, x, mem, qpos)
        raws.append(dec.head(dec.norm(x)))
    out = anchored_decode(dec, raws[-1], anchors_xy)
    if bool(getattr(dec, "deep_supervision", False)):
        out["aux"] = [anchored_decode(dec, r, anchors_xy) for r in raws[:-1]]
    return out


# --------------------------------------------------------------------------------------------------------- #
# the loss term (inside box3d)                                                                              #
# --------------------------------------------------------------------------------------------------------- #
def heat_term(heat_logits: Tensor, box: Tensor, pos: Tensor, ign: Tensor | None, grid) -> dict:
    """The heatmap's loss for one batch: ``{"loss", "n_pos", "n_pos_outside"}`` (the loss is UNWEIGHTED;
    ``HEAT_LOSS_W`` is applied where it is added into the box term)."""
    t = draw_targets(box, pos, ign, grid)
    loss = centernet_focal(heat_logits, t["heat"], t["ignore"])
    return {"loss": loss, "n_pos": t["n_pos"], "n_pos_outside": t["n_pos_outside"],
            "n_ignore_cells": int(t["ignore"].sum())}
