"""refcv9 arm B3 (REFCV9_DESIGN 2.4 rows *presence target* / *denoising* / *training-time NMS* / *class*; R4, R11) -- the box head's
de-duplication, trained in. Four independent, opt-in levers (one flag each, so B3 vs B3' is one variable; OFF = the tip, bit-identical):

  --r9-box-quality     QUALITY-AWARE PRESENCE. A matched slot's presence target is  q = exp(-d_centre / 1 m) * yawness  instead of 1
                       (unmatched slots keep 0). The focal loss is the generalised (quality) focal form  alpha_t * |t - p|^gamma * BCE
                       -- identical to ``slot_presence.sigmoid_focal_elementwise`` for t in {0, 1}, with the optimum at p = t for a soft t.
  --r9-box-dn          QUERY DENOISING. 3 DN groups; per GT one noised POSITIVE and 2 range-modulated NEGATIVES; TRAINING ONLY. The DN
                       queries ride through the SAME decoder in ONE forward under a block-diagonal attention mask: matching queries see
                       matching queries only, each DN group sees its own group only -- no leak in either direction.
  --r9-box-class-sqrt  CE class weights = sqrt(inverse-frequency), normalised to mean 1 (R11), replacing the pure inverse-frequency vector.
  --r9-box-inloss-nms  (B3') BEV-NMS at tau 0.1 INSIDE the loss (DenseBEV): between decoder layers, NMS-suppressed slots are masked out
                       of the other queries' self-attention keys; the Hungarian match of every supervised layer is taken over the NMS
                       SURVIVORS, so a suppressed duplicate can only be a presence NEGATIVE.

Pinned by the design: sigma 1 m, 3 groups, 2 negatives per GT, tau 0.1. NOT pinned by the design (my choice, constants below, each
stamped): the yaw-ness form, the DN noise scales and the form of the "range modulation".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "QUALITY_SIGMA_M", "YAWNESS_FORM", "DN_GROUPS", "DN_NEG_PER_GT", "NMS_TAU", "BoxB3Config", "yawness", "quality_target",
    "qfl_elementwise", "sqrt_temper", "dn_attn_mask", "build_dn_pack", "run_decoder", "bev_nms_keep",
    "refined_box3d_losses_b3", "box3d_loss_row_b3", "dn_loss", "presence_term_b3",
]

QUALITY_SIGMA_M: float = 1.0                  # PINNED (2.4): presence target = exp(-d_centre / 1 m)
YAWNESS_FORM: str = "(1 + cos(dyaw)) / 2"     # NOT pinned: Sparse4Dv3's yaw-ness is a cosine similarity; mapped to [0, 1] for a sigmoid target
DN_GROUPS: int = 3                            # PINNED (2.4)
DN_NEG_PER_GT: int = 2                        # PINNED (2.4)
NMS_TAU: float = 0.1                          # PINNED (2.4): BEV-NMS tau 0.1 (DenseBEV T5a)
NMS_P_FLOOR: float = 0.05                     # slots below this presence never suppress and are never suppressed
DN_MAX_GT: int = 24                           # per window (nearest first, as match_slots drops the farthest)
DN_RANGE_SCALE_M: float = 30.0                # "range-modulated": noise scale = 1 + range / 30 m
DN_POS_SIGMA_M: float = 0.3                   # positive centre noise (per axis, before the range scale)
DN_NEG_MIN_M: float = 2.0                     # negatives are displaced 2-4 m (x the range scale), i.e. outside the 2 m AP tolerance
DN_NEG_MAX_M: float = 4.0
DN_SIZE_NOISE: float = 0.1                    # relative l/w noise
DN_YAW_NOISE_RAD: float = 0.1
DN_LOSS_W: float = 1.0
DN_FEAT_DIM: int = 6                          # cx/x_fwd, cy/y_half, l/10, w/4, sin, cos of the noised box


@dataclass(frozen=True)
class BoxB3Config:
    quality: bool = False
    dn: bool = False
    class_sqrt: bool = False
    inloss_nms: bool = False

    @property
    def any_loss_change(self) -> bool:
        return self.quality or self.dn or self.inloss_nms

    def as_dict(self) -> dict:
        return {"quality": self.quality, "dn": self.dn, "class_sqrt": self.class_sqrt, "inloss_nms": self.inloss_nms}


# --------------------------------------------------------------------------------------------------------- #
# quality-aware presence                                                                                    #
# --------------------------------------------------------------------------------------------------------- #
def yawness(pred_yaw_vec: Tensor, tgt_yaw: Tensor) -> Tensor:
    """``(1 + cos dyaw) / 2`` in [0, 1] from the predicted unit (sin, cos) pair and the target yaw."""
    ts, tc = torch.sin(tgt_yaw), torch.cos(tgt_yaw)
    cos_d = pred_yaw_vec[..., 0] * ts + pred_yaw_vec[..., 1] * tc
    return (1.0 + cos_d.clamp(-1.0, 1.0)) / 2.0


@torch.no_grad()
def quality_target(pred_xy: Tensor, pred_yaw_vec: Tensor, tgt_xy: Tensor, tgt_yaw: Tensor) -> Tensor:
    """``exp(-||pred - tgt|| / 1 m) * yawness`` per matched pair. A TARGET: never differentiated."""
    d = (pred_xy.float() - tgt_xy.float()).norm(dim=-1)
    return (torch.exp(-d / QUALITY_SIGMA_M) * yawness(pred_yaw_vec.float(), tgt_yaw.float())).to(pred_xy.dtype)


def qfl_elementwise(logit: Tensor, target: Tensor, alpha: float | None = None, gamma: float | None = None) -> Tensor:
    """``alpha_t * |t - p|^gamma * BCE(logit, t)`` per element (alpha, gamma = the registered focal literals)."""
    from tanitad.models.slot_presence import FOCAL_ALPHA, FOCAL_GAMMA
    a = FOCAL_ALPHA if alpha is None else float(alpha)
    g = FOCAL_GAMMA if gamma is None else float(gamma)
    t = target.to(logit.dtype)
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, t, reduction="none")
    return (a * t + (1.0 - a) * (1.0 - t)) * (t - p).abs().pow(g) * ce


def presence_term_b3(logit: Tensor, match: dict, quality: list | None, ignore_w: Tensor | None = None) -> dict:
    """``slot_presence.presence_term(mode="focal")`` with the matched slots' target = ``quality[b]`` (rows aligned with
    ``match["rows"][b]``) instead of 1. ``quality=None`` is the hard-target term (the identity control, pinned by a test)."""
    tgt = torch.zeros_like(logit)
    matched = torch.zeros_like(logit, dtype=torch.bool)
    for b, r in enumerate(match["rows"]):
        if r.numel():
            r = r.to(logit.device)
            tgt[b, r] = 1.0 if quality is None else quality[b].to(logit.dtype)
            matched[b, r] = True
    n_m = int(matched.sum())
    w = torch.ones_like(logit) if ignore_w is None else ignore_w.to(logit.dtype).clone()
    w[matched] = 1.0
    n_ex = int(((w == 0) & ~matched).sum())
    loss = (qfl_elementwise(logit, tgt) * w).sum() / float(max(n_m, 1))
    return {"loss": loss, "n_matched": n_m, "n_exempt": n_ex}


# --------------------------------------------------------------------------------------------------------- #
# class weights                                                                                             #
# --------------------------------------------------------------------------------------------------------- #
def sqrt_temper(w: Tensor) -> Tensor:
    """``sqrt(w) / mean(sqrt(w))``: the sqrt inverse-frequency rule (PON / PanopticBEV) applied to the inverse-frequency vector."""
    s = torch.as_tensor(w, dtype=torch.float32).clamp_min(0.0).sqrt()
    return s / s.mean().clamp_min(1e-12)


# --------------------------------------------------------------------------------------------------------- #
# BEV-NMS (axis-aligned enclosing rectangle of the rotated box)                                             #
# --------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def _aabb(box: Tensor, yaw: Tensor) -> Tensor:
    c, s = torch.cos(yaw).abs(), torch.sin(yaw).abs()
    hx = 0.5 * (box[..., 2] * c + box[..., 3] * s)
    hy = 0.5 * (box[..., 2] * s + box[..., 3] * c)
    return torch.stack([box[..., 0] - hx, box[..., 1] - hy, box[..., 0] + hx, box[..., 1] + hy], dim=-1)


@torch.no_grad()
def bev_nms_keep(pred: dict, tau: float = NMS_TAU, p_floor: float = NMS_P_FLOOR) -> Tensor:
    """``[B, N]`` bool, True = SURVIVOR of greedy BEV-NMS at IoU ``tau`` (score = sigmoid(presence)). Slots under ``p_floor`` neither
    suppress nor are suppressed."""
    p = pred["presence_logit"].detach().float().sigmoid()
    bx = _aabb(pred["box"].detach().float(), pred["yaw"].detach().float())
    B, N = p.shape
    keep = torch.ones(B, N, dtype=torch.bool, device=p.device)
    for b in range(B):
        idx = (p[b] >= p_floor).nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        idx = idx[torch.argsort(p[b][idx], descending=True)]
        bb = bx[b][idx]
        area = ((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])).clamp_min(1e-9)
        lt = torch.maximum(bb[:, None, :2], bb[None, :, :2])
        rb = torch.minimum(bb[:, None, 2:], bb[None, :, 2:])
        inter = (rb - lt).clamp_min(0).prod(-1)
        iou = inter / (area[:, None] + area[None, :] - inter)
        alive = torch.ones(idx.numel(), dtype=torch.bool, device=p.device)
        for i in range(idx.numel()):
            if not bool(alive[i]):
                continue
            kill = iou[i] > float(tau)
            kill[: i + 1] = False
            alive &= ~kill
        keep[b, idx[~alive]] = False
    return keep


# --------------------------------------------------------------------------------------------------------- #
# denoising: the pack, the mask, the joint forward                                                           #
# --------------------------------------------------------------------------------------------------------- #
def dn_attn_mask(n_match: int, n_groups: int, group_size: int, device=None) -> Tensor:
    """``[T, T]`` bool, True = NOT allowed to attend. Matching queries see matching queries only; each DN group sees its own group
    only. Both directions are blocked: no GT-derived DN content reaches a matching query, and none of the matching queries'
    state reaches a DN query."""
    t = n_match + n_groups * group_size
    blk = torch.ones(t, t, dtype=torch.bool, device=device)
    blk[:n_match, :n_match] = False
    for g in range(n_groups):
        s = n_match + g * group_size
        blk[s:s + group_size, s:s + group_size] = False
    return blk


def build_dn_pack(box: Tensor, yaw: Tensor, cls: Tensor, pos_valid: Tensor, row_ok: Tensor, *, x_fwd_m: float, y_half_m: float,
                  qpos_module: nn.Module | None = None, generator: torch.Generator | None = None) -> dict | None:
    """The DN queries of one batch from the GT (``box`` [B, A, 4] = cx, cy, l, w; ``yaw``/``cls`` [B, A]; ``pos_valid`` [B, A] = the
    POSITIVE target rows; ``row_ok`` [B] = the window carries labels). ``None`` when no window has a positive.

    Layout per window (``Gm`` = the largest positive count, capped at ``DN_MAX_GT``): ``DN_GROUPS`` groups of ``(1 + DN_NEG_PER_GT) * Gm``
    queries, ``[positives | negatives 1 | negatives 2]`` inside a group."""
    with torch.no_grad():                                      # the GT-derived noise is a label-side input; ``qpos`` below keeps its graph
        pack = _build_dn_pack_nograd(box, yaw, cls, pos_valid, row_ok, x_fwd_m, y_half_m, generator)
    if pack is None:
        return None
    pack["qpos"] = qpos_module(pack["anchors"]) if qpos_module is not None else None   # WITH grad: box_qpos is trained by it
    return pack


def _build_dn_pack_nograd(box, yaw, cls, pos_valid, row_ok, x_fwd_m, y_half_m, generator):
    B, A = pos_valid.shape
    dev = box.device
    ok = pos_valid.to(torch.bool) & row_ok.to(dev).to(torch.bool)[:, None]
    gm = min(int(ok.sum(1).max()) if B else 0, DN_MAX_GT)
    if gm == 0:
        return None
    # nearest-first selection of up to gm GT per window
    rng = box[..., :2].norm(dim=-1).masked_fill(~ok, float("inf"))
    order = torch.argsort(rng, dim=1)[:, :gm]                                       # [B, gm]
    take = lambda x: torch.gather(x, 1, order if x.dim() == 2 else order[..., None].expand(-1, -1, x.shape[-1]))
    gb, gy, gc, gv = take(box), take(yaw), take(cls), take(ok)                       # gv: real GT among the gm
    G, k = DN_GROUPS, 1 + DN_NEG_PER_GT
    gs = k * gm

    def u(*shape):
        return torch.rand(*shape, device=dev, generator=generator)

    scale = 1.0 + gb[..., :2].norm(dim=-1) / DN_RANGE_SCALE_M                        # [B, gm]
    feats, anchors, cl, is_pos, is_neg, t_box, t_yaw = [], [], [], [], [], [], []
    for g in range(G):
        for j in range(k):
            ang = u(B, gm) * (2.0 * math.pi)
            if j == 0:                                                               # the noised POSITIVE
                n = torch.randn(B, gm, 2, device=dev, generator=generator) * DN_POS_SIGMA_M * scale[..., None]
            else:                                                                    # a range-modulated NEGATIVE
                rad = (DN_NEG_MIN_M + u(B, gm) * (DN_NEG_MAX_M - DN_NEG_MIN_M)) * scale
                n = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], dim=-1)
            xy = gb[..., :2] + n
            lw = gb[..., 2:4] * (1.0 + (u(B, gm, 2) * 2.0 - 1.0) * DN_SIZE_NOISE)
            yw = gy + (u(B, gm) * 2.0 - 1.0) * DN_YAW_NOISE_RAD
            feats.append(torch.stack([xy[..., 0] / x_fwd_m, xy[..., 1] / y_half_m, lw[..., 0] / 10.0, lw[..., 1] / 4.0,
                                      torch.sin(yw), torch.cos(yw)], dim=-1))
            anchors.append(xy)
            cl.append(gc.clamp_min(0))
            is_pos.append(gv if j == 0 else torch.zeros_like(gv))
            is_neg.append(gv if j > 0 else torch.zeros_like(gv))
            t_box.append(gb)
            t_yaw.append(gy)
    cat = lambda xs: torch.cat(xs, dim=1)
    pack = {"feat": cat(feats), "anchors": cat(anchors), "cls": cat(cl), "is_pos": cat(is_pos), "is_neg": cat(is_neg),
            "t_box": cat(t_box), "t_yaw": cat(t_yaw), "n_groups": G, "group_size": gs, "gmax": gm,
            "x_fwd_m": float(x_fwd_m), "y_half_m": float(y_half_m)}
    # the class target of a positive is the GT class (-1 = unknown class stays unknown)
    pack["t_cls"] = cat([gc for _ in range(G * k)])
    return pack


def _layer_masked(layer: nn.TransformerDecoderLayer, x: Tensor, mem: Tensor, qpos: Tensor, attn_mask: Tensor | None) -> Tensor:
    """``slot_query_select._layer_with_pos`` with a self-attention ``attn_mask`` (True = blocked); zero ``qpos`` = the plain layer."""
    if not bool(getattr(layer, "norm_first", False)):
        raise RuntimeError("B3's masked decoder pass assumes the pre-norm decoder layer (norm_first=True)")
    h = layer.norm1(x)
    q = h + qpos
    x = x + layer.dropout1(layer.self_attn(q, q, h, attn_mask=attn_mask, need_weights=False)[0])
    h = layer.norm2(x)
    x = x + layer.dropout2(layer.multihead_attn(h + qpos, mem, mem, need_weights=False)[0])
    h = layer.norm3(x)
    return x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(h)))))


def _decode(dec, raw: Tensor, anchors_xy: Tensor | None) -> dict:
    if anchors_xy is None:
        return dec.decode(raw)
    from tanitad.models.slot_query_select import anchored_decode
    return anchored_decode(dec, raw, anchors_xy)


def _mask_for_layer(base: Tensor | None, supp: Tensor | None, b: int, t: int, n_heads: int, device) -> Tensor | None:
    if supp is None:
        return base
    blk = (base[None].expand(b, t, t).clone() if base is not None else torch.zeros(b, t, t, dtype=torch.bool, device=device))
    blk[:, :, : supp.shape[1]] |= supp[:, None, :]
    ar = torch.arange(t, device=device)
    blk[:, ar, ar] = False                                                          # a suppressed query still attends itself
    return blk.repeat_interleave(n_heads, dim=0)


def run_decoder(dec, memory: Tensor, *, dn: dict | None = None, nms_tau: float | None = None,
                anchors_xy: Tensor | None = None, qpos: Tensor | None = None) -> dict:
    """The slot decoder's forward with (a) DN queries appended under :func:`dn_attn_mask` and/or (b) between-layer BEV-NMS key masks.
    Returns the usual decoded dict of the MATCHING queries (``aux`` when ``dec.deep_supervision``) and returns the DN decode under
    ``out["_dn_out"]`` for the loss (``select_slots`` passes a non-tensor entry through; the loss selects with ``sel``). ``anchors_xy``/``qpos`` given = the learned_ref (anchored) mode."""
    if memory.ndim != 3 or memory.shape[1] != dec.n_memory or memory.shape[2] != dec.d_memory:
        raise ValueError(f"memory must be [B, {dec.n_memory}, {dec.d_memory}], got {tuple(memory.shape)}")
    if dec.blocks.norm is not None:
        raise RuntimeError("B3 assumes no stack-level norm (as deep supervision does)")
    b, dt = int(memory.shape[0]), memory.dtype
    mem = dec.mem_proj(memory) + dec.mem_pos.to(dt)
    xm = dec.queries.to(dt).expand(b, -1, -1)
    n = int(xm.shape[1])
    qm = qpos.to(dt) if qpos is not None else torch.zeros_like(xm)
    base = None
    if dn is not None:
        if getattr(dec, "dn_embed", None) is None:
            raise RuntimeError("a DN pack reached a decoder built without DN (enable_b3(dn=True) was never called)")
        xd = dec.dn_embed(dn["feat"].to(dt)) + dec.dn_cls_embed(dn["cls"])
        qd = dn["qpos"].to(dt) if (qpos is not None and dn.get("qpos") is not None) else torch.zeros_like(xd)
        x, qp = torch.cat([xm, xd], dim=1), torch.cat([qm, qd], dim=1)
        base = dn_attn_mask(n, int(dn["n_groups"]), int(dn["group_size"]), x.device)
    else:
        x, qp = xm, qm
    t = int(x.shape[1])
    nh = int(dec.blocks.layers[0].self_attn.num_heads)
    n_layers = len(dec.blocks.layers)
    raws_m, raws_d, supp = [], [], None
    for li, layer in enumerate(dec.blocks.layers):
        x = _layer_masked(layer, x, mem, qp, _mask_for_layer(base, supp, b, t, nh, x.device))
        raw = dec.head(dec.norm(x))
        raws_m.append(raw[:, :n])
        raws_d.append(raw[:, n:])
        if nms_tau is not None and li < n_layers - 1:
            supp = ~bev_nms_keep(_decode(dec, raws_m[-1], anchors_xy), float(nms_tau))
    out = _decode(dec, raws_m[-1], anchors_xy)
    if bool(getattr(dec, "deep_supervision", False)):
        out["aux"] = [_decode(dec, r, anchors_xy) for r in raws_m[:-1]]
    if dn is not None:
        out["_dn_out"] = {"pack": dn, "layers": [_decode(dec, r, dn["anchors"] if anchors_xy is not None else None) for r in raws_d]}
    return out


# --------------------------------------------------------------------------------------------------------- #
# the DN loss                                                                                               #
# --------------------------------------------------------------------------------------------------------- #
def _sel_dn(dn_out: dict, sel: Tensor | None) -> dict:
    if sel is None:
        return dn_out
    f = lambda v: v.index_select(0, sel) if (torch.is_tensor(v) and v.dim() > 0) else v
    pk = {k: f(v) for k, v in dn_out["pack"].items()}
    return {"pack": pk, "layers": [{k: f(v) for k, v in lay.items() if torch.is_tensor(v)} for lay in dn_out["layers"]]}


def dn_loss(dn_out: dict, cfg: BoxB3Config, *, cls_class_weight=None, sel: Tensor | None = None) -> dict:
    """Sum over the supervised layers of the DN set loss. Positives: presence target (quality if ``cfg.quality`` else 1), L1 centre and
    size, ``1 - cos`` yaw, class CE; negatives: presence target 0 only. Normalised by the number of DN positives (clamped 1)."""
    from tanitad.models.agent_slots import SLOT_LOSS_W
    from tanitad.models.slot_presence import FOCAL_PRESENCE_W
    d = _sel_dn(dn_out, sel)
    pk = d["pack"]
    pos, neg = pk["is_pos"], pk["is_neg"]
    n_pos = int(pos.sum())
    zero = pk["feat"].new_zeros(())
    total, parts = zero, {"presence": zero, "centre": zero, "size": zero, "yaw": zero, "cls": zero}
    for lay in d["layers"]:
        tp = torch.zeros_like(lay["presence_logit"])
        if cfg.quality and n_pos:
            q = quality_target(lay["box"][..., :2][pos], lay["yaw_vec"][pos], pk["t_box"][..., :2][pos], pk["t_yaw"][pos])
            tp[pos] = q.to(tp.dtype)
        else:
            tp[pos] = 1.0
        w = (pos | neg).to(tp.dtype)
        l_pres = (qfl_elementwise(lay["presence_logit"], tp) * w).sum() / float(max(n_pos, 1))
        l_c = (lay["box"][..., :2][pos] - pk["t_box"][..., :2][pos]).abs().sum() / float(max(n_pos, 1)) if n_pos else zero
        l_s = (lay["box"][..., 2:][pos] - pk["t_box"][..., 2:][pos]).abs().sum() / float(max(n_pos, 1)) if n_pos else zero
        yt = torch.stack([torch.sin(pk["t_yaw"][pos]), torch.cos(pk["t_yaw"][pos])], dim=-1) if n_pos else None
        l_y = (1.0 - (lay["yaw_vec"][pos] * yt).sum(-1)).sum() / float(max(n_pos, 1)) if n_pos else zero
        l_k = zero
        if n_pos:
            ct = pk["t_cls"][pos]
            ok = ct >= 0
            if bool(ok.any()):
                cw = None if cls_class_weight is None else cls_class_weight.to(device=lay["cls_logits"].device,
                                                                               dtype=lay["cls_logits"].dtype)
                ce = F.cross_entropy(lay["cls_logits"][pos][ok], ct[ok], reduction="sum", weight=cw)
                l_k = ce / (float(ok.sum()) if cw is None else float(cw[ct[ok]].sum()))
        total = total + FOCAL_PRESENCE_W * l_pres + SLOT_LOSS_W["centre"] * l_c + SLOT_LOSS_W["size"] * l_s \
            + SLOT_LOSS_W["yaw"] * l_y + SLOT_LOSS_W["cls"] * l_k
        for k_, v_ in (("presence", l_pres), ("centre", l_c), ("size", l_s), ("yaw", l_y), ("cls", l_k)):
            parts[k_] = parts[k_] + v_
    return {"loss": total, "parts": parts, "n_pos": n_pos, "n_neg": int(neg.sum()), "n_layers": len(d["layers"])}


# --------------------------------------------------------------------------------------------------------- #
# the refined box3d loss with the B3 levers                                                                  #
# --------------------------------------------------------------------------------------------------------- #
def _match_view(p: dict, keep: Tensor | None) -> dict:
    """The prediction dict the Hungarian match reads: NMS-suppressed slots are moved far away with a hard-negative presence, so the
    match is taken over the survivors. The LOSS still reads the real ``p``."""
    if keep is None:
        return p
    sup = ~keep
    box = p["box"].clone()
    box[..., :2] = torch.where(sup[..., None], torch.full_like(box[..., :2], 1.0e4), box[..., :2])
    pl = torch.where(sup, torch.full_like(p["presence_logit"], -1.0e4), p["presence_logit"])
    return {**p, "box": box, "presence_logit": pl}


def refined_box3d_losses_b3(slots: dict, tgt: dict, cfg: BoxB3Config, *, presence_loss: str, vis1: bool, vis: dict | None = None,
                            weights: dict | None = None, cls_class_weight=None, sel: Tensor | None = None) -> dict:
    """``slot_presence.refined_box3d_losses`` (``visible_filter=True``, focal presence) with the B3 levers; same keys, plus ``loss_dn``."""
    from tanitad.models import slot_presence as sp
    from tanitad.models.agent_slots import match_slots
    from tanitad.models.box3d_head import box3d_set_loss
    from tanitad.refs.refc_agents import visible_target_filter
    if presence_loss != "focal":
        raise ValueError("B3 is defined on the focal (refined) presence path: --slot-presence-loss focal")
    ignore, counts = None, {"target_prefilter": int(tgt["valid"].sum())}
    if vis1:
        from tanitad.data.vis1 import vis1_split
        if vis is None:
            raise ValueError("[box-b3] vis1=True but no visibility block was passed (REFUSE, do not skip)")
        sp_ = vis1_split(tgt, n_full=vis["n_full"], n_vis=vis["n_vis"], vis_known=vis["known"])
        tpos, ignore = sp_["pos"], sp_["ignore"]
        counts.update({f"vis1_{k}": v for k, v in sp_["counts"].items()})
        counts["target_visible"] = int(sp_["in_filter"].sum())
    else:
        tpos = visible_target_filter(tgt)
        counts["target_visible"] = int(tpos["valid"].sum())
    if tpos.get("zh_mask") is not None:
        tpos = {**tpos, "zh_mask": tpos["zh_mask"].to(torch.bool) & tpos["valid"].to(torch.bool)}
    w0 = {**(weights or {}), "presence": 0.0}
    per, n_sup = [], 0
    for p in sp.layer_preds(slots):
        keep = bev_nms_keep(p) if cfg.inloss_nms else None
        if keep is not None:
            n_sup += int((~keep).sum())
        m = match_slots(_match_view(p, keep), tpos, presence_cost="focal")
        base = box3d_set_loss(p, tpos, match=m, weights=w0, cls_class_weight=cls_class_weight, visible_filter=False)
        iw = sp.ignore_presence_weight(p["box"], tgt["box"], ignore) if ignore is not None else None
        q = None
        if cfg.quality:
            q = []
            for b in range(int(p["box"].shape[0])):
                r, c = m["rows"][b].to(p["box"].device), m["cols"][b].to(p["box"].device)
                q.append(quality_target(p["box"][b][r][:, :2], p["yaw_vec"][b][r], tpos["box"][b][c][:, :2], tpos["yaw"][b][c])
                         if r.numel() else p["box"].new_zeros(0))
        pr = presence_term_b3(p["presence_logit"], m, q, iw)
        per.append({"base": base, "presence": pr, "match": m, "total": base["total"] + sp.FOCAL_PRESENCE_W * pr["loss"]})
    last = per[-1]
    out = {k: v for k, v in last["base"].items() if k.startswith("loss_")}
    out["loss_presence"] = last["presence"]["loss"]
    out["total"] = sum(x["total"] for x in per)
    for i, x in enumerate(per):
        out[f"loss_layer{i}"] = x["total"]
        out[f"loss_presence_layer{i}"] = x["presence"]["loss"]
    n = dict(last["base"].get("n") or {})
    n.update(counts)
    n.update({"layers": len(per), "presence_exempt": int(last["presence"]["n_exempt"]), "presence_matched": int(last["presence"]["n_matched"]),
              "nms_suppressed": n_sup})
    dn_out = slots.get("_dn_out")
    if cfg.dn:
        if dn_out is None:
            raise ValueError("[box-b3] --r9-box-dn is on and the loss ran with no DN output: the DN pack never reached the forward "
                             "(REFUSE, do not skip -- the tac_goal_tok_head class)")
        dl = dn_loss(dn_out, cfg, cls_class_weight=cls_class_weight, sel=sel)
        out["loss_dn"] = dl["loss"]
        out["total"] = out["total"] + DN_LOSS_W * dl["loss"]
        n.update({"dn_pos": dl["n_pos"], "dn_neg": dl["n_neg"], "dn_layers": dl["n_layers"]})
    n["dropped_not_visible"] = int(n.get("target_prefilter", 0)) - int(n.get("target_visible", 0))
    out["n"] = n
    out["match"] = last["match"]
    out["_weights"] = {**dict(last["base"].get("_weights") or {}), "presence": sp.FOCAL_PRESENCE_W}
    out["_refine"] = sp.refine_stamp(presence_loss, vis1, n_layers=len(per))
    out["_visible_filter"] = True
    return out


def box3d_loss_row_b3(slots: dict, tgt: dict, cfg: BoxB3Config, *, weights: dict | None = None, cls_class_weight=None,
                      presence_loss: str = "focal", vis1: bool = False, vis: dict | None = None, sel: Tensor | None = None) -> dict:
    """``refcv6_perception_branch.box3d_loss_row``'s row (same keys) over :func:`refined_box3d_losses_b3`. The A14 heatmap term is not
    built here: ``--r9-box-*`` is REFUSED with ``--slot-query-select heatmap`` at argv time."""
    from tanitad.models import slot_presence as sp
    r = refined_box3d_losses_b3(slots, tgt, cfg, presence_loss=presence_loss, vis1=vis1, vis=vis, weights=weights,
                                cls_class_weight=cls_class_weight, sel=sel)
    row = {"loss": r["total"]}
    for k, v in r.items():
        if k.startswith("loss_") and torch.is_tensor(v):
            row[f"box3d_{k[5:]}"] = v
    for k, v in (r.get("n") or {}).items():
        row[f"box3d_n_{k}"] = float(v)
    row["box3d_visible_filter"] = 1.0
    row["box3d_presence_focal"] = 1.0
    row["box3d_vis1"] = 1.0 if vis1 else 0.0
    row["box3d_presence_frac_confident"] = sp.presence_sanity(slots["presence_logit"])["frac_confident"]
    return row
