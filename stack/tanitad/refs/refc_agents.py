"""refcv5 WP-6 / ``E-AGT-HEAD`` — the AGENT-TOKEN SEAM between the environment
head and the planner decoder, plus the monocular supervision that seam needs.

⭐ **The PI's named priority (2026-09-05):** *"Let's follow environment
extraction from the front camera and train it based on GT data."* This module is
the **seam**, not a second detector.

## ⛔ What this module deliberately does NOT contain, and why

Three complete components already exist in this repo. Re-implementing any of
them is how two geometries drift apart (the `advect` precedent, retired
2026-07-27), so every one of them is **imported**:

| already exists | file | what it gives |
|---|---|---|
| DETR slot decoder, exact Hungarian, DETR set loss, join→targets | ``tanitad.models.agent_slots`` | ``AgentSlotDecoder``, ``hungarian``, ``match_slots``, ``slot_set_loss``, ``targets_from_join`` |
| rig ⇄ canonical-cylindrical projection, both directions | ``tanitad.data.rig_projection`` | ``RigCamera``, ``project_rig_to_frame``, ``frame_to_cam_ray``, ``ground_intersection`` |
| the 10-class enum, the BEV grid, the FOV mask | ``tanitad.data.bev_raster`` | ``ALL_CLASSES``, ``GRID_DEFAULT``, ``fov_mask`` |

⚠️ **The class enum in particular is IMPORTED, never re-listed.** A first draft
of this file hand-wrote a plausible-looking 10-class tuple containing
``bicycle``, ``motorcycle`` and ``train_or_tram_car`` — **none of which exist in
the corpus**, which really carries ``other_vehicle``, ``stroller`` and
``animal``. It would have trained a head against three classes that can never
appear and silently dropped three that do. The enum has exactly one spelling,
in ``bev_raster``, and it is the label side's.

## What IS new here — the four gaps the audit named

1. :class:`AgentTokenEmbed` — turn decoded agent slots into the tokens the
   planner decoder cross-attends. **This seam did not exist**: ``AgentSlotDecoder``
   emits a set of boxes and nothing consumes them, and ``V6Config.agent_slots``
   is in ``LADDER_UNTRAINED_GROUPS`` (it trains in no stage).
2. :func:`monocular_projection_loss` — the **image-plane** term. ``slot_set_loss``
   is entirely BEV ``(cx, cy, l, w)``; a MONOCULAR head supervised only in BEV is
   asked to regress range with no term in the space it can actually see.
3. :func:`ground_range_prior` — the free metric anchor. Rig ``z = 0`` **is** the
   road plane (MEASURED over 87,481 cuboids: ground-standing bottom faces
   −0.05 to −0.13 m, ``protruding_object`` +1.68 m), so a pixel plus the plane
   already fixes range. No learned depth needed for the prior.
4. :func:`degrade_boxes` / :class:`OracleAgentEmbed` — the ``E-AGT-ORACLE`` and
   ``E-AGT-BUDGET`` rungs, which must run **before** any detector GPU-day.

## The binding rules this module is written against

| stage | what may be used | here |
|---|---|---|
| label derivation | ego, other agents, maps, future poses | ``obstacle.offline`` cuboids are **TRAIN-TIME TARGETS** |
| **inference** | ⛔ **VISION ONLY** | ``AgentSlotDecoder.forward`` takes ONE argument — the spatial memory. There is no parameter through which a label could arrive. |

⛔ An arm that reads the join at inference is **refused, not fixed**. The
oracle path below exists to price the ceiling and is marked inadmissible as a
capability claim wherever it appears.

⚠️ Frames are **256×640 CYLINDRICAL**, ``f_ref`` 305.577, HFOV **120°**. The
pinhole formula gives 92.6° here and looks entirely plausible. Nothing in this
module computes a pixel itself — every projection goes through
``rig_projection``, whose ``CanonicalFrame`` carries the projection so no call
site chooses a formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.data.bev_raster import ALL_CLASSES, GRID_DEFAULT
from tanitad.data.rig_projection import RigCamera  # noqa: F401  (re-export)
from tanitad.models.agent_slots import (
    AGENT_CLASSES, N_QUERIES_DEFAULT, AgentSlotDecoder, SlotDecodeRanges,
    match_slots, slot_set_loss,
)

__all__ = [
    "AGENT_CLASSES", "N_AGENT_CLASSES", "AgentSeamConfig", "AgentTokenEmbed",
    "OracleAgentEmbed", "build_agent_head", "degrade_boxes",
    "monocular_projection_loss", "ground_range_prior", "slot_pad_mask",
    "TOKEN_FEAT_DIM",
]

#: 10 — imported, never re-listed. See the module docstring's ⚠️.
N_AGENT_CLASSES = len(ALL_CLASSES)

#: per-slot geometry features handed to the token embedder:
#: (cx/x_range, cy/y_range, l, w, yaw_sin, yaw_cos, range/x_range,
#:  log1p(range)/5, cos bearing, sin bearing, 1/(1+range), v_rel_x, v_rel_y,
#:  yaw_rate_rel, presence)
TOKEN_FEAT_DIM = 15


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AgentSeamConfig:
    """WP-6 ``E-AGT-HEAD``. Every default is OFF — a build that does not ask
    for agent tokens constructs nothing and is bit-identical to refcv4b,
    including RNG draw order."""

    #: build the monocular head and feed its tokens to the decoder
    enable: bool = False
    #: ⛔ E-AGT-ORACLE / E-AGT-BUDGET: feed GROUND-TRUTH boxes instead of the
    #: head's. INADMISSIBLE as a capability claim — it reads a privileged label
    #: at inference. Prices the ceiling before any detector GPU-day is spent.
    oracle: bool = False
    #: E-AGT-BUDGET knobs: degrade the oracle until separation dies. The sigma
    #: that comes out IS the detector specification.
    oracle_sigma_range_m: float = 0.0
    oracle_miss_rate: float = 0.0

    #: ⚠️ ``agent_slots.N_QUERIES_DEFAULT`` is 16 and is a DECLARED PLACEHOLDER.
    #: MEASURED on the val40 join: 195,805 boxes over 7,400 frames = **26.5
    #: agents/frame MEAN**, so 16 makes ``match_slots`` drop targets on most
    #: frames. 32 is this seam's default and the p99 must be measured with
    #: ``scripts/measure_agent_density.py`` and recorded in the prereg before a
    #: full-scale run. DD uses 30, so 32 also keeps the audit gap on its axis.
    queries: int = 32
    d_model: int = 256
    depth: int = 3
    n_heads: int = 8
    #: the head's param band is enforced by AgentSlotDecoder (2-4 M). A toy
    #: rig may waive it; a real arm may NOT.
    enforce_band: bool = True

    #: the monocular image-plane auxiliary (gap #2). 0.0 turns it off, which
    #: is a BEV-only head and is a different arm — say which one you ran.
    w_project: float = 0.0
    #: ground-plane range prior (gap #3), weight on |range - plane range|.
    w_ground: float = 0.0
    #: gate the tokens by presence, so a slot the head does not believe in
    #: contributes nothing. Threshold is on sigmoid(presence_logit).
    presence_gate: float = 0.5
    #: ⛔ hard-mask slots below the threshold instead of soft-scaling. Soft is
    #: the default because a hard mask has zero gradient to the presence head
    #: through the planner loss.
    presence_hard: bool = False

    def as_dict(self) -> dict:
        """Serialised into ``config.json['seams']['agents']`` — a run record
        that cannot rebuild its own model config is not a run record."""
        return {
            "enable": bool(self.enable), "oracle": bool(self.oracle),
            "oracle_sigma_range_m": float(self.oracle_sigma_range_m),
            "oracle_miss_rate": float(self.oracle_miss_rate),
            "queries": int(self.queries), "d_model": int(self.d_model),
            "depth": int(self.depth), "n_heads": int(self.n_heads),
            "enforce_band": bool(self.enforce_band),
            "w_project": float(self.w_project),
            "w_ground": float(self.w_ground),
            "presence_gate": float(self.presence_gate),
            "presence_hard": bool(self.presence_hard),
            "n_classes": int(N_AGENT_CLASSES),
            "classes": list(AGENT_CLASSES),
            "n_queries_default_upstream": int(N_QUERIES_DEFAULT),
        }


def build_agent_head(cfg: AgentSeamConfig, d_memory: int, n_memory: int
                     ) -> AgentSlotDecoder:
    """The head, built from the EXISTING ``AgentSlotDecoder``.

    ``n_memory`` is the conv map's token count (``gh * gw``) — the decoder's
    positional table is per-token, so this is a geometry, not a resize.
    """
    return AgentSlotDecoder(
        d_memory=int(d_memory), n_memory=int(n_memory),
        n_queries=int(cfg.queries), d_model=int(cfg.d_model),
        depth=int(cfg.depth), n_heads=int(cfg.n_heads),
        enforce_band=bool(cfg.enforce_band))


# ---------------------------------------------------------------------------
# The seam that did not exist: slots -> planner tokens
# ---------------------------------------------------------------------------
def slot_features(box: Tensor, yaw_vec: Tensor, rates: Tensor,
                  presence: Tensor,
                  ranges: SlotDecodeRanges | None = None) -> Tensor:
    """``-> [B, N, TOKEN_FEAT_DIM]`` — the geometry a PLANNER token must carry.

    ⭐ Why explicit geometry and not just the query feature: the decoder acts on
    a fan of trajectories **in metres**, and every LONGITUDINAL metric the
    programme is bound to report (headway, time-gap, TTC) is a function of
    **range and bearing**. A token that hides those inside a learned embedding
    makes cross-attention re-derive them from scratch.

    ``box`` ``[B, N, 4]`` = (cx, cy, l, w) in METRES (``AgentSlotDecoder``'s
    decode) · ``yaw_vec`` ``[B, N, 2]`` unit (sin, cos) · ``rates``
    ``[B, N, 3]`` = (v_rel_x, v_rel_y, yaw_rate_rel) · ``presence``
    ``[B, N]`` in ``[0, 1]``.
    """
    r = ranges or SlotDecodeRanges()
    cx, cy = box[..., 0], box[..., 1]
    rng = torch.sqrt(cx * cx + cy * cy).clamp_min(1e-3)
    return torch.stack([
        cx / r.x_fwd_m, cy / r.y_half_m,
        box[..., 2], box[..., 3],
        yaw_vec[..., 0], yaw_vec[..., 1],
        rng / r.x_fwd_m,
        torch.log1p(rng) / 5.0,
        cx / rng, cy / rng,                 # bearing (cos, sin)
        1.0 / (1.0 + rng),
        rates[..., 0], rates[..., 1], rates[..., 2],
        presence,
    ], dim=-1)


class AgentTokenEmbed(nn.Module):
    """Decoded agent slots -> the ``[B, N, d_dec]`` tokens the planner decoder
    cross-attends, plus the ``[B, N]`` padding mask it needs.

    ⛔ **The padding convention is torch's**: ``pad[b, n] == True`` means slot
    ``n`` is PADDING and must NOT be attended. Inverting it silently attends to
    nothing and reads exactly like a dead seam.
    """

    def __init__(self, d_out: int, cfg: AgentSeamConfig,
                 ranges: SlotDecodeRanges | None = None):
        super().__init__()
        self.cfg = cfg
        self.ranges = ranges or SlotDecodeRanges()
        d = int(cfg.d_model)
        self.geo = nn.Sequential(
            nn.Linear(TOKEN_FEAT_DIM + N_AGENT_CLASSES, d), nn.GELU(),
            nn.Linear(d, d_out))
        self.norm = nn.LayerNorm(d_out)

    def forward(self, slots: dict) -> tuple[Tensor, Tensor]:
        """``slots`` is ``AgentSlotDecoder.forward``'s output (or the oracle's).

        Returns ``(tokens [B, N, d_out], pad [B, N] bool)``.
        """
        presence = torch.sigmoid(slots["presence_logit"])          # [B, N]
        cls_p = torch.softmax(slots["cls_logits"], dim=-1)         # [B, N, C]
        feats = slot_features(slots["box"], slots["yaw_vec"], slots["rates"],
                              presence, self.ranges)
        tok = self.norm(self.geo(torch.cat([feats, cls_p], dim=-1)))
        thr = float(self.cfg.presence_gate)
        if self.cfg.presence_hard:
            pad = presence < thr
            # a row with NOTHING above threshold is handled by CrossAttnLayer,
            # which un-masks fully-padded rows and zeroes their contribution.
            return tok, pad
        # SOFT (default): scale by presence so the presence head receives
        # gradient THROUGH the planner loss, and mark only structurally absent
        # slots as padding.
        tok = tok * presence.unsqueeze(-1).to(tok.dtype)
        return tok, slot_pad_mask(slots)


def slot_pad_mask(slots: dict) -> Tensor:
    """``[B, N]`` bool, True where the slot is structurally absent.

    For a LEARNED head every slot exists (presence is soft), so this is
    all-False; for the ORACLE path it is ``~valid``, because a padded GT row is
    genuinely not an agent.
    """
    v = slots.get("valid")
    if v is None:
        return torch.zeros(slots["box"].shape[:2], dtype=torch.bool,
                           device=slots["box"].device)
    return ~v


# ---------------------------------------------------------------------------
# E-AGT-ORACLE / E-AGT-BUDGET
# ---------------------------------------------------------------------------
def degrade_boxes(box: Tensor, valid: Tensor, sigma_range_m: float = 0.0,
                  miss_rate: float = 0.0,
                  generator: torch.Generator | None = None
                  ) -> tuple[Tensor, Tensor]:
    """``E-AGT-BUDGET`` — degrade ORACLE boxes to a stated accuracy budget.

    Noise is applied **along the RANGE direction only**. That is a modelling
    choice with a reason: on a 120° cylindrical frame the image COLUMN fixes
    bearing almost exactly (``col = (W-1)/2 + f_ref·azimuth``, linear), while
    range is what a monocular detector actually gets wrong. Noising both axes
    isotropically would price a detector nobody would build.

    ``box`` ``[B, N, 4]`` = (cx, cy, l, w) metres · ``valid`` ``[B, N]`` bool.

    ⭐ **The control that must read a known value:** σ = 0 and miss = 0 return
    the inputs **unchanged objects**, so the ORACLE arm is exactly the
    zero-degradation point of the BUDGET sweep and not a separate code path.
    """
    if sigma_range_m <= 0.0 and miss_rate <= 0.0:
        return box, valid
    out = box.clone()
    if sigma_range_m > 0.0:
        cx, cy = out[..., 0], out[..., 1]
        rng = torch.sqrt(cx * cx + cy * cy).clamp_min(1e-3)
        eps = torch.randn(rng.shape, device=box.device, dtype=box.dtype,
                          generator=generator) * float(sigma_range_m)
        scale = (rng + eps).clamp_min(0.1) / rng
        out[..., 0] = cx * scale
        out[..., 1] = cy * scale
    v = valid
    if miss_rate > 0.0:
        keep = torch.rand(valid.shape, device=valid.device,
                          generator=generator) >= float(miss_rate)
        v = valid & keep
    return out, v


class OracleAgentEmbed(nn.Module):
    """``E-AGT-ORACLE`` / ``E-AGT-BUDGET`` — tokens from GT boxes, no detector.

    ⛔ **INADMISSIBLE as a capability claim.** It reads a privileged label at
    inference. Its ONLY job is to answer *"if the detector were perfect, would
    the planner move at all?"* before any GPU-day is spent training one — and
    per the ladder, if it does not separate on **LONGITUDINAL and TACTICAL**,
    the whole mechanism is refused for zero further GPU-days.

    Emits the SAME dict shape ``AgentSlotDecoder`` does, so
    :class:`AgentTokenEmbed` consumes either without a branch.
    """

    def __init__(self, cfg: AgentSeamConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, box: Tensor, yaw: Tensor, cls: Tensor, valid: Tensor,
                rates: Tensor | None = None,
                generator: torch.Generator | None = None) -> dict:
        """``box [B,N,4]`` (cx, cy, l, w) m · ``yaw [B,N]`` rad ·
        ``cls [B,N]`` long (-1 = unknown) · ``valid [B,N]`` bool."""
        c = self.cfg
        box, valid = degrade_boxes(box, valid, c.oracle_sigma_range_m,
                                   c.oracle_miss_rate, generator)
        yaw_vec = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)
        # a GT box is present with certainty; a padded one with none.
        big = box.new_tensor(20.0)
        presence_logit = torch.where(valid, big, -big)
        onehot = F.one_hot(cls.clamp_min(0), N_AGENT_CLASSES).to(box.dtype)
        # unknown class (-1) -> uniform, never class 0 (which would invent
        # "automobile" for every unlabelled row).
        unk = (cls < 0).unsqueeze(-1)
        onehot = torch.where(unk, torch.full_like(onehot,
                                                  1.0 / N_AGENT_CLASSES),
                             onehot)
        cls_logits = torch.log(onehot.clamp_min(1e-9))
        if rates is None:
            rates = box.new_zeros(*box.shape[:2], 3)
        return {"box": box, "yaw_vec": yaw_vec, "yaw": yaw,
                "cls_logits": cls_logits, "presence_logit": presence_logit,
                "rates": rates, "valid": valid,
                "occ_logit": box.new_zeros(box.shape[:2])}


# ---------------------------------------------------------------------------
# Gap #2 — the IMAGE-PLANE term a monocular head needs
# ---------------------------------------------------------------------------
def monocular_projection_loss(pred_box: Tensor, tgt_box: Tensor,
                              cam: RigCamera, match: dict,
                              z_center_m: float = 0.75) -> dict:
    """L1 between the PROJECTED pixel centres of matched predicted and target
    boxes, in the canonical cylindrical frame.

    ⭐ **Why this term exists.** ``slot_set_loss``'s box terms are entirely BEV
    metres. A MONOCULAR head has no direct observation of range — it sees a
    column (bearing, near-exact) and a row (which, given the road plane, IS the
    range). Supervising only in BEV asks it to regress the hard axis with no
    term in the space it can actually see, and lets a 2 m range error at 50 m
    cost the same as at 5 m although the image evidence differs by an order of
    magnitude. The projection term is naturally range-weighted: it is large
    where the image says the head is wrong.

    ⛔ **Everything geometric goes through ``rig_projection``.** The pinhole FOV
    formula is WRONG on this projection and nothing here re-derives a pixel.

    ``z_center_m`` lifts the BEV centre to a box CENTRE height above the road
    plane, because rig ``z = 0`` is the road (MEASURED) and a box centred on the
    road would project below every real detection. 0.75 m ≈ half a car's height;
    it is a declared constant, not a fitted one, and it cancels to first order
    in the DIFFERENCE of two projections at similar range.

    Returns ``{"loss", "n"}`` — ``n`` is the matched count the loss was
    computed over, RETURNED so a panel can print its own ``n``.
    """
    device = pred_box.device
    losses: list[Tensor] = []
    n = 0
    for b, (rows, cols) in enumerate(zip(match["rows"], match["cols"])):
        if rows.numel() == 0:
            continue
        pb = pred_box[b, rows.to(device)]                  # [M, 4]
        tb = tgt_box[b, cols.to(device)]                   # [M, 4]
        z = pb.new_full((pb.shape[0], 1), float(z_center_m))
        p3 = torch.cat([pb[:, :1], pb[:, 1:2], z], dim=-1)
        t3 = torch.cat([tb[:, :1], tb[:, 1:2], z], dim=-1)
        pc, pr, pv = cam.project(p3.to(cam.t_cam_in_rig.dtype))
        tc, tr, tv = cam.project(t3.to(cam.t_cam_in_rig.dtype))
        keep = pv & tv
        if not bool(keep.any()):
            continue
        # normalise by the frame so column and row errors are commensurate and
        # the term is scale-free across the 256x640 / 176x624 frames.
        w = float(cam.frame.width)
        h = float(cam.frame.height)
        d = ((pc - tc).abs() / w + (pr - tr).abs() / h)[keep]
        losses.append(d.to(pred_box.dtype).mean())
        n += int(keep.sum())
    if not losses:
        return {"loss": pred_box.new_zeros(()), "n": 0}
    return {"loss": torch.stack(losses).mean(), "n": n}


# ---------------------------------------------------------------------------
# Gap #3 — the free metric anchor
# ---------------------------------------------------------------------------
def ground_range_prior(pred_box: Tensor, cam: RigCamera,
                       z_center_m: float = 0.75) -> dict:
    """Consistency between a slot's predicted range and the range its own
    projected FOOT point implies under the road plane.

    ⭐ Rig ``z = 0`` **is** the road plane — MEASURED over 87,481 cuboids, not
    assumed (ground-standing classes' bottom faces read −0.05 to −0.13 m;
    ``protruding_object``, which by definition does not stand on the ground,
    reads +1.68 m). So a pixel plus the plane already fixes metric range and the
    monocular scale ambiguity does not apply to us. This term says: *project the
    box's foot to a pixel, back-project that pixel to the road, and the range
    you get back must be the range you predicted.*

    It costs no label at all — it is a **self-consistency** term, so it may run
    on windows the join does not cover (the ~NO_LABEL frames past ~20 s, which
    are a large share of every clip).

    Returns ``{"loss", "n", "frac_hit"}``.
    """
    dt = cam.t_cam_in_rig.dtype
    cx, cy = pred_box[..., 0], pred_box[..., 1]
    foot = torch.stack([cx, cy, torch.zeros_like(cx)], dim=-1).to(dt)
    col, row, valid = cam.project(foot)
    p_rig, hits = cam.ground_intersection(col, row)
    ok = valid & hits
    if not bool(ok.any()):
        return {"loss": pred_box.new_zeros(()), "n": 0, "frac_hit": 0.0}
    r_pred = torch.sqrt(cx * cx + cy * cy).to(dt)
    r_back = torch.linalg.vector_norm(p_rig[..., :2], dim=-1)
    d = ((r_pred - r_back).abs() / r_pred.clamp_min(1.0))[ok]
    return {"loss": d.to(pred_box.dtype).mean(), "n": int(ok.sum()),
            "frac_hit": float(ok.to(torch.float32).mean())}


# ---------------------------------------------------------------------------
# The composed training loss
# ---------------------------------------------------------------------------
def agent_losses(slots: dict, tgt: dict, cfg: AgentSeamConfig,
                 cam: RigCamera | None = None,
                 weights: dict | None = None) -> dict:
    """``slot_set_loss`` + the two monocular terms, per term, with their ``n``.

    ⛔ Per-term, never pooled into one score — the four-families discipline's
    sibling rule: a composite hides exactly the trade-off one wants to see.
    """
    match = match_slots(slots, tgt)
    out = slot_set_loss(slots, tgt, match=match, weights=weights)
    total = out["total"]
    if cam is not None and cfg.w_project > 0.0:
        pj = monocular_projection_loss(slots["box"], tgt["box"], cam, match)
        out["loss_project"] = pj["loss"]
        out["n"]["project"] = pj["n"]
        total = total + cfg.w_project * pj["loss"]
    if cam is not None and cfg.w_ground > 0.0:
        gp = ground_range_prior(slots["box"], cam)
        out["loss_ground"] = gp["loss"]
        out["n"]["ground"] = gp["n"]
        out["ground_frac_hit"] = gp["frac_hit"]
        total = total + cfg.w_ground * gp["loss"]
    out["total"] = total
    out["n_dropped"] = int(sum(match["n_dropped"]))
    out["n_target"] = int(sum(match["n_target"]))
    return out
