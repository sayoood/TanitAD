"""Auxiliary drivable-area head for REFe, as a SELF-CONTAINED module (SPEC_AUX_DRIVABLE_HEAD.md sections 2-4, 6-7).

NOT WIRED INTO ANY TRAINER. `model.py`, `train.py`, `scorer_finetune.py` and the launch-gate code are untouched (frozen while RL-S1 ships);
the wiring is a later agent's job from `raw/2026-10-13-aux-head/WIRING_PLAN.md`. Proof: `refe/test_aux_drivable_head.py` -> `ZZTEST_PASS <n>`.

WHAT IT IS
  * `DrivableAuxHead`   40 x 32 learnable BEV queries (2 m cells over x in [-8, 72), y in [-32, 32)) + a Fourier encoding of the cell
                        centre; 2 x CrossBlock(256, 8, ctx 256) (REFe's own class); per cell a mask logit and a signed distance
                        6 m * tanh(.). 2,440,450 parameters (= the prototype's `head_proto_selftest.out`). ONE input tensor: the
                        image-derived context `visual_ctx` [B, 7680, 256] (tap "visual") or `scene_ctx` [B, 64, 256] (tap "scene").
                        No ego, goal, navigation command or map: admissibility SPEC 6.
  * `block_stats` / `build_targets`   int8 SDF [B, 160, 128] (0.1 m unit, + inside, index [ix, iy]) -> the head's 2 m targets as
                        4 x 4 BLOCK STATISTICS: mask fraction (share of the 16 fine cells with SDF > 0) and the block mean of the
                        SDF clipped to +-6 m (clip BEFORE the mean), plus a `known` mask (a frame without a label has no cell).
  * `aux_loss`          L_aux = BCE(logit, mask fraction) + w_sdf * Huber(sdf/6, target/6, beta 0.1), mean over cells, over the frames
                        with a label; a frame without one contributes exactly 0 (loss and gradient).
  * H1 / H2 (SPEC 2.4) H1 (`feed="off"`, the default): the scorer does not consume the prediction; the scoring path is the model's
                        own arithmetic. H2 (`feed="on"`): per proposal THREE scalars from the DETACHED predicted SDF sampled
                        bilinearly at the footprint corners over the 0.1 s grid (min over corner-steps; mean at the footprint centre;
                        share of corner-steps below 0) enter `score_q_mlp`'s first layer. Implemented as an ADDITIVE projection
                        `proj: Linear(3, 256, bias=False)` added to the first layer's pre-activation, which is EXACTLY a 63-wide first
                        layer whose last three columns are `proj.weight` (tested), but leaves the checkpoint's `score_q_mlp.0.weight`
                        [256, 60] untouched, so u400 still loads STRICTLY. `feed_init="zero"` (default) makes H2 start as the
                        base scorer's exact function. An out-of-extent corner is "unknown": SDF = +6 m, flag returned (telemetry).
  * `AuxLabelSource`    reads targets by (log_name, token) from the label store (`aux_label_store.py`), ONE shard per log (cached),
                        REFUSES a store whose manifest `store_sha256` is not the pinned one, whose manifest is internally
                        inconsistent, whose manifest reports a navtest leak, or whose shard bytes do not hash to the manifest.
  * `add_aux_args` / `AuxDrivableConfig.from_args` / `aux_refusals` / `AUX_IDENTITY_ARGS`   the six flags of SPEC 4, defaults OFF.

ADMISSIBILITY (SPEC 6.1): this file imports no map API and no label-store code at import time (the store reader is imported lazily
inside `AuxLabelSource`), so the planner/eval path that builds the head never reaches the label path.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from model import CrossBlock  # noqa: E402  (REFe's own cross-attention block, the class `dec` and `score_dec` use)
import field_loss as FL  # noqa: E402  (geometry only: NAVSIM-representation poses and the footprint corners; no map API)

# ---- the label geometry (SPEC 3.1; asserted equal to aux_label_store's raster spec by the test) -----------------------
X0, X1, Y0, Y1 = -8.0, 72.0, -32.0, 32.0                  # rear-axle frame at t0, x forward, y left
FINE_RES, SDF_UNIT_M = 0.5, 0.1
NX_F, NY_F = 160, 128
CELL_M, SDF_SCALE_M = 2.0, 6.0                            # the head's grid and the SDF bound (SPEC 2.2)
N_FEED = 3                                                # H2: min SDF, mean SDF at the centre, share below 0

# ---- the pinned store (MEASURED: manifest.json written 2026-10-12 21:30:10) ---------------------------------------------
PINNED_STORE_SHA256 = "4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a"
PINNED_STORE_KEYS = 103139
DEFAULT_STORE = "D:/Projects/TanitAD-artifacts/aux_drivable/label_store"

# the dests of the six flags (SPEC 4: "add every one to IDENTITY_ARGS, so a resume with a different head is refused")
AUX_IDENTITY_ARGS = ("aux_drv", "w_aux_drv", "aux_drv_tap", "aux_drv_depth", "aux_drv_feed", "aux_drv_labels", "aux_drv_shuffle_seed")


# ======================================================================================================================
# config (G-HYG: frozen + validated, so a typo'd attribute or keyword RAISES instead of silently doing nothing)
# ======================================================================================================================
@dataclass(frozen=True)
class AuxDrivableConfig:
    tap: str = "visual"          # "visual" -> visual_ctx [B, 7680, 256] (primary) | "scene" -> scene_ctx [B, 64, 256] (registered variant)
    depth: int = 2
    feed: str = "off"            # "off" = H1 (representation only, the default) | "on" = H2 (the scorer consumes the prediction)
    w_aux: float = 0.0           # ⛔ DEFAULT 0.0 (SPEC 4): a head-less arm must not silently carry a weight; an arm that means it NAMES it
    feed_init: str = "zero"      # H2's projection: "zero" (function-preserving) | "normal" (std 0.02)
    ctx_dim: int = 256
    width: int = 256
    heads: int = 8
    cell: float = CELL_M
    sdf_scale: float = SDF_SCALE_M
    w_sdf: float = 1.0
    seed: int = 0                # the head's OWN init stream (see `build_head`)

    def __post_init__(self):
        if self.tap not in ("visual", "scene"):
            raise ValueError(f"tap must be 'visual' or 'scene', got {self.tap!r}")
        if self.feed not in ("off", "on"):
            raise ValueError(f"feed must be 'off' or 'on', got {self.feed!r}")
        if self.feed_init not in ("zero", "normal"):
            raise ValueError(f"feed_init must be 'zero' or 'normal', got {self.feed_init!r}")
        if int(self.depth) < 1:
            raise ValueError("depth must be >= 1")
        b = self.cell / FINE_RES
        if abs(b - round(b)) > 1e-9 or round(b) < 1 or NX_F % round(b) or NY_F % round(b):
            raise ValueError(f"cell {self.cell} m must be an integer multiple of the {FINE_RES} m label cell and tile the {NX_F}x{NY_F} grid")
        if self.w_aux < 0.0 or self.w_sdf < 0.0:
            raise ValueError("loss weights must be >= 0")

    @classmethod
    def from_args(cls, a) -> "AuxDrivableConfig":
        return cls(tap=a.aux_drv_tap, depth=a.aux_drv_depth, feed=a.aux_drv_feed, w_aux=a.w_aux_drv, seed=int(a.seed) if hasattr(a, "seed") else 0)


def add_aux_args(ap) -> None:
    """The six flags of SPEC 4 (+ the enable switch). EVERY default disables the mechanism; an arm that tests it must name it in its argv."""
    ap.add_argument("--aux-drv", action="store_true", help="build the auxiliary drivable-area head (default OFF)")
    ap.add_argument("--w-aux-drv", type=float, default=0.0, help="weight of L_aux in the total loss (default 0.0 = no gradient; SPEC 2.3 initial 0.1)")
    ap.add_argument("--aux-drv-tap", default="visual", choices=("visual", "scene"), help="the tensor the head reads (visual_ctx | scene_ctx)")
    ap.add_argument("--aux-drv-depth", type=int, default=2, help="CrossBlocks in the head")
    ap.add_argument("--aux-drv-feed", default="off", choices=("off", "on"), help="H1 (off) | H2 (on): does the scorer consume the prediction")
    ap.add_argument("--aux-drv-labels", default="", help="the label store directory (manifest.json + shards/)")
    ap.add_argument("--aux-drv-shuffle-seed", type=int, default=None, help="the LABEL-SHUFFLED control: a derangement of the label keys with this seed")


def aux_refusals(a) -> list:
    """A command line that names the head's levers inconsistently never starts (the `epo_refusals` pattern)."""
    bad = []
    on = bool(getattr(a, "aux_drv", False))
    if not on:
        for nm, off_val in (("w_aux_drv", 0.0), ("aux_drv_feed", "off"), ("aux_drv_labels", ""), ("aux_drv_shuffle_seed", None)):
            if getattr(a, nm, off_val) != off_val:
                bad.append(f"--{nm.replace('_', '-')} is set but --aux-drv is not: the lever would silently do nothing")
        return bad
    if not getattr(a, "aux_drv_labels", ""):
        bad.append("--aux-drv needs --aux-drv-labels (the label store)")
    if getattr(a, "w_aux_drv", 0.0) < 0.0:
        bad.append("--w-aux-drv must be >= 0")
    if getattr(a, "aux_drv_shuffle_seed", None) is not None and getattr(a, "aux_drv_feed", "off") == "on":
        bad.append("--aux-drv-shuffle-seed with --aux-drv-feed on would feed the scorer a deliberately wrong map: the shuffle control is an H1 arm")
    return bad


# ======================================================================================================================
# the head
# ======================================================================================================================
class DrivableAuxHead(nn.Module):
    def __init__(self, ctx_dim: int = 256, width: int = 256, heads: int = 8, depth: int = 2,
                 cell: float = CELL_M, sdf_scale: float = SDF_SCALE_M, n_freq: int = 4):
        super().__init__()
        self.cell, self.sdf_scale = cell, sdf_scale
        self.nx, self.ny = int(round((X1 - X0) / cell)), int(round((Y1 - Y0) / cell))
        xs = X0 + (torch.arange(self.nx) + 0.5) * cell
        ys = Y0 + (torch.arange(self.ny) + 0.5) * cell
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        ux, uy = (gx - X0) / (X1 - X0), (gy - Y0) / (Y1 - Y0)                    # [0, 1]
        k = 2.0 ** torch.arange(n_freq) * math.pi
        feats = [ux.reshape(-1, 1), uy.reshape(-1, 1)]
        for u in (ux, uy):
            feats += [torch.sin(u.reshape(-1, 1) * k), torch.cos(u.reshape(-1, 1) * k)]
        self.register_buffer("pos_feat", torch.cat(feats, -1), persistent=False)        # [nx*ny, 2 + 4*n_freq]
        self.pos_mlp = nn.Linear(self.pos_feat.shape[-1], width)
        self.q = nn.Parameter(torch.zeros(1, self.nx * self.ny, width))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.blocks = nn.ModuleList([CrossBlock(width, heads, ctx_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 2)

    def forward(self, ctx):
        B = ctx.shape[0]
        q = self.q.expand(B, -1, -1) + self.pos_mlp(self.pos_feat)[None]
        for blk in self.blocks:
            q = blk(q, ctx)
        o = self.out(self.norm(q))
        logit = o[..., 0].reshape(B, self.nx, self.ny)
        sdf = self.sdf_scale * torch.tanh(o[..., 1]).reshape(B, self.nx, self.ny)    # metres, bounded
        return logit, sdf


def build_head(cfg: AuxDrivableConfig) -> DrivableAuxHead:
    """Build the head from its OWN seeded stream. The global CPU RNG is restored on exit, so building the head does not consume (or depend
    on) the trunk's stream: the "detached => bit-identical trunk" control (SPEC 4 / 7) needs this. `fork_rng` rather than a `Generator`
    argument because `nn.Linear`'s default init takes no generator."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(cfg.seed))
        return DrivableAuxHead(ctx_dim=cfg.ctx_dim, width=cfg.width, heads=cfg.heads, depth=cfg.depth, cell=cfg.cell, sdf_scale=cfg.sdf_scale)


# ======================================================================================================================
# targets and loss
# ======================================================================================================================
@dataclass(frozen=True)
class AuxTargets:
    mask_frac: torch.Tensor      # [B, nx, ny] float in [0, 1]: share of the block's fine cells with SDF > 0
    sdf_m: torch.Tensor          # [B, nx, ny] metres: block mean of the fine SDF clipped to +-sdf_scale
    known: torch.Tensor          # [B, nx, ny] bool: the cell exists in the label's extent AND the frame has a label


def block_stats(label_i8: torch.Tensor, cell: float = CELL_M, sdf_scale: float = SDF_SCALE_M):
    """[B, 160, 128] int8 SDF (0.1 m units, + inside, index [ix, iy]) -> (mask fraction, block-mean clipped SDF in m), each [B, nx, ny]."""
    if label_i8.dtype != torch.int8 or tuple(label_i8.shape[-2:]) != (NX_F, NY_F) or label_i8.dim() != 3:
        raise ValueError(f"expected int8 [B, {NX_F}, {NY_F}], got {label_i8.dtype} {tuple(label_i8.shape)}")
    b = int(round(cell / FINE_RES))
    s = label_i8.float() * SDF_UNIT_M
    B = s.shape[0]
    s = s.view(B, NX_F // b, b, NY_F // b, b)
    frac = (s > 0).float().mean((2, 4))
    sdf = s.clamp(-sdf_scale, sdf_scale).mean((2, 4))
    return frac, sdf


def build_targets(label_i8: torch.Tensor, valid: torch.Tensor | None = None, cell: float = CELL_M, sdf_scale: float = SDF_SCALE_M) -> AuxTargets:
    """The head's targets plus the extent/validity mask: every coarse cell of a LABELLED frame is inside the label's extent
    (the label IS the window x in [-8, 72), y in [-32, 32)), and a frame without a label has no known cell at all."""
    frac, sdf = block_stats(label_i8, cell, sdf_scale)
    B = frac.shape[0]
    if valid is None:
        valid = torch.ones(B, dtype=torch.bool, device=frac.device)
    known = valid.to(frac.device).bool()[:, None, None].expand_as(frac)
    return AuxTargets(frac, sdf, known)


def aux_loss(logit, sdf_pred, label_i8, valid=None, *, cell: float = CELL_M, sdf_scale: float = SDF_SCALE_M, w_sdf: float = 1.0):
    """-> (L_aux, bce, huber), each a scalar. `valid` [B] bool: a frame without a label contributes exactly 0 (loss AND gradient; NaN-safe)."""
    t = build_targets(label_i8, valid, cell, sdf_scale)
    bce = F.binary_cross_entropy_with_logits(logit.float(), t.mask_frac, reduction="none")
    hub = F.smooth_l1_loss(sdf_pred.float() / sdf_scale, t.sdf_m / sdf_scale, beta=0.1, reduction="none")
    kn = t.known
    n_cells = kn.sum((1, 2)).clamp(min=1).float()
    zero = torch.zeros_like(bce)
    bce_f = torch.where(kn, bce, zero).sum((1, 2)) / n_cells
    hub_f = torch.where(kn, hub, zero).sum((1, 2)) / n_cells
    has = kn.any(-1).any(-1).float()
    denom = has.sum().clamp(min=1.0)
    bce_m, hub_m = bce_f.sum() / denom, hub_f.sum() / denom
    return bce_m + w_sdf * hub_m, bce_m, hub_m


@torch.no_grad()
def head_metrics(logit, sdf_pred, label_i8, valid=None, *, cell: float = CELL_M, sdf_scale: float = SDF_SCALE_M) -> dict:
    """The perception-proof read (S1): mask IoU / accuracy at the 2 m grid and the SDF MAE in metres, over the labelled frames."""
    t = build_targets(label_i8, valid, cell, sdf_scale)
    tm, pm, kn = t.mask_frac >= 0.5, logit > 0, t.known
    inter, union = (tm & pm & kn).sum().item(), ((tm | pm) & kn).sum().item()
    nk = max(int(kn.sum().item()), 1)
    return {"iou": inter / max(union, 1), "acc": ((tm == pm) & kn).sum().item() / nk,
            "sdf_mae_m": float(((sdf_pred.float() - t.sdf_m).abs() * kn).sum().item() / nk), "n_cells": int(kn.sum().item())}


# ======================================================================================================================
# H2: the feed
# ======================================================================================================================
def sample_pred_sdf(sdf_pred: torch.Tensor, pts: torch.Tensor, cell: float = CELL_M, sdf_scale: float = SDF_SCALE_M):
    """Bilinear lookup of the head's predicted SDF. sdf_pred [B, nx, ny] (cell centres x = X0 + (i + .5) * cell), pts [B, N, 2] (x, y) in the
    label frame -> (s [B, N] metres, in_extent [B, N] bool). A point outside the window [X0, X1) x [Y0, Y1) is UNKNOWN: SDF = +sdf_scale
    and in_extent False (SPEC 3.1). Inside the window, the outermost half cell replicates the border value."""
    img = sdf_pred.transpose(1, 2).unsqueeze(1)                                            # [B, 1, ny, nx]: H = y, W = x
    gx = 2.0 * (pts[..., 0] - X0) / (X1 - X0) - 1.0
    gy = 2.0 * (pts[..., 1] - Y0) / (Y1 - Y0) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2).to(img.dtype)                          # [B, N, 1, 2]
    s = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False).reshape(pts.shape[0], pts.shape[1])
    in_ext = (pts[..., 0] >= X0) & (pts[..., 0] < X1) & (pts[..., 1] >= Y0) & (pts[..., 1] < Y1)
    s = torch.where(in_ext, s, torch.full_like(s, sdf_scale))
    return s, in_ext


def feed_scalars(sdf_pred: torch.Tensor, trajs: torch.Tensor, cell: float = CELL_M, sdf_scale: float = SDF_SCALE_M):
    """sdf_pred [B, nx, ny] metres, trajs [B, K, 20, 3] (x, y, yaw of the rear axle, step n at t = 0.2 (n+1) s)
    -> (feats [B, K, 3], oob_share [B, K]).
      feats[..., 0] = min over the 40 x 4 corner-steps of the predicted SDF (metres)
      feats[..., 1] = mean over the 40 steps of the predicted SDF at the footprint CENTRE (metres)
      feats[..., 2] = share of the 40 x 4 corner-steps with predicted SDF < 0
    Steps t = 0.1 ... 4.0 s (t = 0 is the same footprint for every proposal and is excluded, as in `field_loss`); poses in NAVSIM's own
    representation (`FL.poses_10hz_navsim`). BOTH inputs are detached: the head is trained by map labels only and the scorer cannot
    teach it to game the score. `oob_share` = share of the sampled points outside the head's extent (telemetry, not a feature)."""
    sdf = sdf_pred.detach().float()
    tr = trajs.detach().float()
    B, K = tr.shape[0], tr.shape[1]
    poses = FL.poses_10hz_navsim(tr)[..., 1:, :]                                           # [B, K, 40, 3]
    corners = FL.footprint_corners(poses)                                                  # [B, K, 40, 4, 2]
    ch, sh = torch.cos(poses[..., 2]), torch.sin(poses[..., 2])
    centre = torch.stack([poses[..., 0] + FL.REAR_AXLE_TO_CENTER * ch, poses[..., 1] + FL.REAR_AXLE_TO_CENTER * sh], dim=-1)   # [B, K, 40, 2]
    n_c = corners.shape[2] * corners.shape[3]
    pts = torch.cat([corners.reshape(B, K, n_c, 2), centre], dim=2).reshape(B, K * (n_c + centre.shape[2]), 2)
    s, inb = sample_pred_sdf(sdf, pts, cell, sdf_scale)
    s, inb = s.reshape(B, K, -1), inb.reshape(B, K, -1)
    s_corner, s_centre = s[..., :n_c], s[..., n_c:]
    feats = torch.stack([s_corner.min(-1).values, s_centre.mean(-1), (s_corner < 0).float().mean(-1)], dim=-1)
    return feats, 1.0 - inb.float().mean(-1)


class DrivableFeed(nn.Module):
    """H2's learnable part: `proj(feats)` is ADDED to `score_q_mlp[0]`'s pre-activation. Equal to a 63-wide first layer whose last three
    columns are `proj.weight` (the base layer keeps its [256, 60] weight and so loads the base checkpoint strictly)."""

    def __init__(self, width: int = 256, init: str = "zero", seed: int = 0):
        super().__init__()
        self.proj = nn.Linear(N_FEED, width, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 1)
            if init == "zero":
                nn.init.zeros_(self.proj.weight)
            else:
                nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, sdf_pred, trajs, cell: float = CELL_M, sdf_scale: float = SDF_SCALE_M):
        feats, _ = feed_scalars(sdf_pred, trajs, cell, sdf_scale)
        return self.proj(feats.to(self.proj.weight.dtype))


# ======================================================================================================================
# the container the trainer would hold
# ======================================================================================================================
class AuxDrivable(nn.Module):
    """head + (H2 only) feed + loss composition + the scoring path. The head is `None`-free: a head-less arm simply does not build this."""

    def __init__(self, cfg: AuxDrivableConfig):
        super().__init__()
        self.cfg = cfg
        self.w_aux = float(cfg.w_aux)
        self.head = build_head(cfg)
        self.feed = DrivableFeed(cfg.width, cfg.feed_init, cfg.seed) if cfg.feed == "on" else None

    # -- head -------------------------------------------------------------------------------------------------------
    def predict(self, ctx):
        return self.head(ctx)

    def loss(self, logit, sdf_pred, label_i8, valid=None):
        return aux_loss(logit, sdf_pred, label_i8, valid, cell=self.cfg.cell, sdf_scale=self.cfg.sdf_scale, w_sdf=self.cfg.w_sdf)

    def combine(self, l_task, l_aux):
        """Total = l_task + w_aux * l_aux (SPEC 2.3). ALWAYS multiplied in, even at w_aux = 0, so a zero weight yields gradient tensors of
        EXACTLY 0.0 (a short-circuit would read None and prove nothing about the wiring)."""
        return l_task + self.w_aux * l_aux

    # -- the scoring path (H1 / H2) ----------------------------------------------------------------------------------
    def score(self, scorer, trajs, sctx, sdf_pred=None):
        """Score a complete proposal set. `scorer` is any module with score_q_mlp / score_dec / score_head (REFe, or scorer_finetune's
        ScorerCopy). H1: the model's own arithmetic, `sdf_pred` is IGNORED. H2: the three detached scalars enter the first layer."""
        x = trajs.flatten(2)
        if self.feed is None:
            h = scorer.score_q_mlp(x)
        else:
            mlp = scorer.score_q_mlp
            h = mlp[0](x) + self.feed(sdf_pred, trajs, self.cfg.cell, self.cfg.sdf_scale)
            for m in list(mlp)[1:]:
                h = m(h)
        for blk in scorer.score_dec:
            h = blk(h, sctx)
        return scorer.score_head(h)

    # -- facts the launch gate compares with the declared config (G-DVB) -------------------------------------------------
    def built_facts(self) -> dict:
        return {"head_built": self.head is not None, "depth": len(self.head.blocks), "cell": self.head.cell, "x0": X0, "y0": Y0,
                "nx": self.head.nx, "ny": self.head.ny, "n_params": sum(p.numel() for p in self.head.parameters()),
                "feed_built": self.feed is not None, "scorer_in_width": 60 + (self.feed.proj.in_features if self.feed is not None else 0),
                "w_aux": self.w_aux, "ctx_dim": self.head.blocks[0].cross.kdim}


def tap_is(ctx: torch.Tensor, expected: torch.Tensor) -> bool:
    """G-DVB: the head's K/V IS the declared tensor (same storage), not a copy or a different tensor."""
    return ctx.data_ptr() == expected.data_ptr() and ctx.shape == expected.shape


# ======================================================================================================================
# the label-shuffled control
# ======================================================================================================================
def derangement(keys: list, seed: int) -> list:
    """A permutation of `keys` with ZERO fixed points and the same multiset: one random n-cycle (always a derangement, no rejection loop).
    shuffled[i] != keys[i] for every i; only the image <-> label alignment is destroyed."""
    n = len(keys)
    if n < 2:
        raise ValueError("a derangement needs >= 2 keys")
    import random
    rng = random.Random(int(seed))
    order = list(range(n))
    rng.shuffle(order)
    out = [None] * n
    for pos in range(n):
        out[order[pos]] = keys[order[(pos + 1) % n]]
    return out


# ======================================================================================================================
# the label store reader
# ======================================================================================================================
class StoreRefused(RuntimeError):
    """The label store is not the pinned one, or is internally inconsistent, or a shard's bytes do not hash to the manifest."""


def _als():
    """Lazy import of the store reader: the head module is on the planner's import path, the label store is not (SPEC 6.1)."""
    import aux_label_store as ALS
    return ALS


def _parse_shard_bytes(data: bytes, ALS):
    """The shard format of `aux_label_store.write_shard`, parsed from the SAME bytes that were hashed (no re-read between check and use)."""
    m = len(ALS.MAGIC)
    if data[:m] != ALS.MAGIC:
        raise StoreRefused("shard: bad magic")
    (hl,) = struct.unpack("<I", data[m:m + 4])
    head = json.loads(data[m + 4:m + 4 + hl].decode("utf-8"))
    base = m + 4 + hl
    blobs = {}
    for e in head["entries"]:
        b = data[base + e["o"]:base + e["o"] + e["l"]]
        if len(b) != e["l"]:
            raise StoreRefused(f"shard: truncated blob for {e['t']}")
        blobs[e["t"]] = b
    return head, blobs


class AuxLabelSource:
    """(log_name, token) -> the stored int8 SDF [160, 128]. One shard per log is read once and kept (`max_logs` bounds the cache; None = keep
    all, <= 351 MB compressed for the full store)."""

    def __init__(self, store, pinned_sha256: str = PINNED_STORE_SHA256, *, verify_shards: bool = True, max_logs: int | None = None,
                 shuffle_seed: int | None = None, pinned_keys: int | None = PINNED_STORE_KEYS):
        ALS = _als()
        self.store = Path(store)
        mp = self.store / "manifest.json"
        if not mp.is_file():
            raise StoreRefused(f"no manifest.json under {self.store}")
        man = json.loads(mp.read_text(encoding="utf-8"))
        if man.get("schema") != ALS.SCHEMA:
            raise StoreRefused(f"manifest schema {man.get('schema')!r} != {ALS.SCHEMA!r}")
        if man.get("raster") != ALS.raster_spec():
            raise StoreRefused("manifest raster spec differs from aux_label_store.raster_spec()")
        got = man.get("store_sha256")
        if got != pinned_sha256:
            raise StoreRefused(f"store_sha256 {str(got)[:16]}... is not the pinned {pinned_sha256[:16]}...")
        files = man.get("files") or {}
        again = hashlib.sha256("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()).hexdigest()
        if again != got:
            raise StoreRefused("manifest is internally inconsistent: store_sha256 does not hash its own files map")
        if pinned_keys is not None and man.get("counts", {}).get("stored_keys") != pinned_keys:
            raise StoreRefused(f"manifest stored_keys {man.get('counts', {}).get('stored_keys')} != pinned {pinned_keys}")
        leak = man.get("leakage")
        if leak is not None and (leak.get("token_intersection") or leak.get("log_intersection")):
            raise StoreRefused(f"manifest reports a navtest intersection: {leak}")
        self.manifest, self._files, self.verify_shards, self.max_logs = man, files, bool(verify_shards), max_logs
        self._cache: OrderedDict = OrderedDict()
        self.stats = {"shard_reads": 0, "labels": 0, "missing_key": 0, "missing_log": 0}
        self.shuffle_seed, self._key_map = shuffle_seed, None

    # -- keys ---------------------------------------------------------------------------------------------------------
    def all_keys(self) -> list:
        """Every (log, token) in the store, sorted (reads the shard HEADERS only)."""
        keys = []
        for rel in sorted(self._files):
            head, _ = _als().read_shard(self.store / rel, with_blobs=False)
            keys += [(head["log"], e["t"]) for e in head["entries"]]
        return sorted(keys)

    def key_map(self) -> dict:
        """The shuffle control's key -> key map (built lazily; a derangement of ALL keys)."""
        if self._key_map is None:
            ks = self.all_keys()
            self._key_map = dict(zip(ks, derangement(ks, self.shuffle_seed)))
        return self._key_map

    # -- shards -----------------------------------------------------------------------------------------------------
    def _blobs(self, log: str):
        if log in self._cache:
            self._cache.move_to_end(log)
            return self._cache[log]
        rel = f"shards/{log}.aux1"
        ent = self._files.get(rel)
        if ent is None:
            self.stats["missing_log"] += 1
            return None
        data = (self.store / rel).read_bytes()
        if self.verify_shards and hashlib.sha256(data).hexdigest() != ent["sha256"]:
            raise StoreRefused(f"{rel}: bytes do not hash to the manifest's sha256")
        ALS = _als()
        head, blobs = _parse_shard_bytes(data, ALS)
        if head.get("schema") != ALS.SCHEMA or head.get("raster") != ALS.raster_spec() or head.get("log") != log:
            raise StoreRefused(f"{rel}: header is not this store's schema / raster / log")
        self.stats["shard_reads"] += 1
        self._cache[log] = blobs
        if self.max_logs is not None and len(self._cache) > self.max_logs:
            self._cache.popitem(last=False)
        return blobs

    def label(self, log: str, token: str):
        """-> int8 [160, 128] (+ inside, 0.1 m unit, index [ix, iy]) or None when the store has no label for the key."""
        if self.shuffle_seed is not None:
            log, token = self.key_map()[(log, token)]
        blobs = self._blobs(log)
        if blobs is None:
            return None
        blob = blobs.get(token)
        if blob is None:
            self.stats["missing_key"] += 1
            return None
        self.stats["labels"] += 1
        return _als().decode_sdf(blob).copy()

    def labels(self, keys) -> tuple:
        """keys: [(log_name, token)] -> (label int8 [B, 160, 128], valid bool [B]); a key with no label reads zeros and valid False."""
        out = np.zeros((len(keys), NX_F, NY_F), dtype=np.int8)
        valid = np.zeros(len(keys), dtype=bool)
        for i, (log, tok) in enumerate(keys):
            lab = self.label(log, tok)
            if lab is not None:
                out[i], valid[i] = lab, True
        return torch.from_numpy(out), torch.from_numpy(valid)

    def verify_all(self) -> dict:
        """CONTENT check of the whole store: sha256 of every shard file against the manifest (the pinned store_sha256 is only as good as this)."""
        bad = [rel for rel, ent in sorted(self._files.items())
               if hashlib.sha256((self.store / rel).read_bytes()).hexdigest() != ent["sha256"]]
        return {"files": len(self._files), "mismatch": len(bad), "first_bad": bad[:3]}
