"""+R6 -- DENOISING QUERIES for the 3-D box head (SPEC_REFCV7 §15.5, A10.1): TRAINING ONLY, default OFF.

A10.1 pre-registered this as the next arm IF G-BOX-OVERFIT's MAIN arm fails criterion 2 or 3 while presence sits at its
base rate (the slots did not specialise). It is BANKED UNLANDED until that happens.

WHAT IT IS (DN-DETR / DINO contrastive denoising, adapted to a content-query 3-D slot decoder):
  * every VIS-1 POSITIVE of the batch is copied into ``DN_GROUPS`` groups; in each group it appears twice --
    a POSITIVE query (its box noised INSIDE ``DN_BOX_NOISE`` of its half-extent / size / 45 deg yaw) and a NEGATIVE
    query (noised between 1x and 2x that) -- with its class label flipped to a random class with probability
    ``DN_LABEL_NOISE`` (DN-DETR's label noise);
  * a small embedder (:class:`DenoiseQueryEmbed`, owned by the perception branch, NOT the decoder, so the decoder's
    §6 parameter band is untouched) turns (noised box, noised label) into a query vector;
  * the DN queries run through the box decoder's OWN modules (``mem_proj``/``mem_pos``/``blocks``/``norm``/``head``/
    ``decode``) in a SEPARATE pass: each group attends to itself only (a block-diagonal self-attention mask) and to the
    memory. ⚠️ Declared simplification vs DN-DETR, where the DN part may also see the matching part: here it cannot,
    so the matching pass is untouched BY CONSTRUCTION (A10.1's "attention-masked from the matching queries" in its
    strictest form) and nothing is computed twice;
  * losses at EVERY decoder layer (R2): a positive reconstructs its GT (presence focal target 1, class CE with the
    run's class weights, centre / size L1 in metres, yaw 1 - cos, z / h L1), a negative is "no object" (presence focal
    target 0). No Hungarian: the assignment is fixed by construction. Normalised by the number of DN positives.
Inference is unchanged: the DN pass exists only when the branch is training and ``dn_groups > 0``.

PUBLISHED: DN parity at 50 % of the epochs and +1.9 AP (DN-DETR, S3); CDN +0.5 AP and fewer duplicates (DINO, S4);
used by StreamPETR and Sparse4D v3 (literature pass §3 D7).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["DN_GROUPS_ARM", "DN_BOX_NOISE", "DN_LABEL_NOISE", "DN_YAW_NOISE_RAD", "DN_LOSS_W", "DenoiseQueryEmbed",
           "make_dn_queries", "dn_attention_mask", "dn_forward", "dn_losses"]

#: the A10.1 arm's literals (DN-DETR: 5 groups, box noise 0.4, label noise 0.2).
DN_GROUPS_ARM: int = 5
DN_BOX_NOISE: float = 0.4
DN_LABEL_NOISE: float = 0.2
DN_YAW_NOISE_RAD: float = math.pi / 4.0
DN_LOSS_W: float = 1.0


class DenoiseQueryEmbed(nn.Module):
    """(noised box, noised label) -> a decoder query. 18 inputs: cx/x_range, cy/y_range, log l, log w, sin/cos yaw,
    cz/z_range, log h, and the 10-class one-hot; plus a learned DN indicator. Training only."""

    def __init__(self, d_model: int, n_classes: int, *, x_range: float, y_range: float, z_range: float):
        super().__init__()
        self.n_classes = int(n_classes)
        self.x_range, self.y_range, self.z_range = float(x_range), float(y_range), float(z_range)
        self.mlp = nn.Sequential(nn.Linear(8 + self.n_classes, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.indicator = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.indicator, std=0.02)

    def forward(self, box: Tensor, yaw: Tensor, cz: Tensor, h: Tensor, cls: Tensor) -> Tensor:
        """``box`` [B, D, 4] (cx, cy, l, w) m, ``yaw``/``cz``/``h`` [B, D], ``cls`` [B, D] long (-1 = unknown)."""
        f = torch.stack([box[..., 0] / self.x_range, box[..., 1] / self.y_range,
                         torch.log(box[..., 2].clamp_min(0.05)), torch.log(box[..., 3].clamp_min(0.05)),
                         torch.sin(yaw), torch.cos(yaw), cz / self.z_range, torch.log(h.clamp_min(0.05))], dim=-1)
        oh = F.one_hot(cls.clamp_min(0), self.n_classes).to(f.dtype)
        oh = torch.where((cls < 0)[..., None], torch.full_like(oh, 1.0 / self.n_classes), oh)
        return self.mlp(torch.cat([f, oh], dim=-1)) + self.indicator.to(f.dtype)


def _noise(tgt: dict, sel: Tensor, scale_lo: float, scale_hi: float, gen: torch.Generator):
    """Noised copies of the rows ``sel`` [B, A] of ``tgt``: |relative noise| in [scale_lo, scale_hi) per axis."""
    box = tgt["box"]
    dev = box.device

    def mag(shape):
        u = torch.rand(shape, generator=gen, device="cpu").to(dev) * (scale_hi - scale_lo) + scale_lo
        s = torch.where(torch.rand(shape, generator=gen, device="cpu").to(dev) < 0.5, -1.0, 1.0)
        return u * s
    B, A = sel.shape
    dx = mag((B, A)) * box[..., 2] / 2.0
    dy = mag((B, A)) * box[..., 3] / 2.0
    sl = 1.0 + mag((B, A))
    sw = 1.0 + mag((B, A))
    nb = torch.stack([box[..., 0] + dx, box[..., 1] + dy, (box[..., 2] * sl).clamp_min(0.1),
                      (box[..., 3] * sw).clamp_min(0.1)], dim=-1)
    nyaw = tgt["yaw"] + mag((B, A)) * DN_YAW_NOISE_RAD
    return nb, nyaw


def make_dn_queries(embed: DenoiseQueryEmbed, tgt_pos: dict, *, groups: int, gen: torch.Generator) -> dict:
    """The DN query block for a batch whose POSITIVES are ``tgt_pos["valid"]`` [B, A].

    Layout per element: ``groups`` x [A_max positives | A_max negatives] (A_max = the batch's most positives).
    Returns ``{"q" [B, D, d], "pad" [B, D] bool (True = padding), "gt" [B, D] long (the GT row), "is_pos" [B, D],
    "A_max", "groups"}``."""
    valid = tgt_pos["valid"].to(torch.bool)
    B = int(valid.shape[0])
    A_max = int(valid.sum(1).max()) if B else 0
    if A_max == 0:
        return {"q": None, "A_max": 0, "groups": int(groups)}
    dev = tgt_pos["box"].device
    # the positives of each element, packed to the front (their original row index kept)
    order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)[:, :A_max]          # [B, A_max]
    ok = torch.gather(valid, 1, order)
    g = {k: torch.gather(tgt_pos[k], 1, order) for k in ("yaw", "cls")}
    g["box"] = torch.gather(tgt_pos["box"], 1, order[..., None].expand(-1, -1, 4))
    cz = torch.gather(tgt_pos["cz"], 1, order) if tgt_pos.get("cz") is not None else torch.zeros_like(g["yaw"])
    hh = torch.gather(tgt_pos["h"], 1, order) if tgt_pos.get("h") is not None else torch.full_like(g["yaw"], 1.6)
    qs, pads, gts, pos = [], [], [], []
    for _ in range(int(groups)):
        for is_pos, lo, hi in ((True, 0.0, DN_BOX_NOISE), (False, DN_BOX_NOISE, 2.0 * DN_BOX_NOISE)):
            nb, ny = _noise(g, ok, lo, hi, gen)
            cls = g["cls"].clone()
            flip = (torch.rand(cls.shape, generator=gen).to(dev) < DN_LABEL_NOISE) & (cls >= 0)
            rnd = torch.randint(0, embed.n_classes, cls.shape, generator=gen).to(dev)
            cls = torch.where(flip, rnd, cls)
            qs.append(embed(nb, ny, cz, hh, cls))
            pads.append(~ok)
            gts.append(order)
            pos.append(torch.full_like(ok, is_pos))
    return {"q": torch.cat(qs, 1), "pad": torch.cat(pads, 1), "gt": torch.cat(gts, 1), "is_pos": torch.cat(pos, 1),
            "A_max": A_max, "groups": int(groups)}


def dn_attention_mask(groups: int, A_max: int, device=None) -> Tensor:
    """[D, D] bool, True = NOT allowed: each group (2 x A_max queries) sees only itself (block-diagonal)."""
    n = 2 * int(A_max)
    D = int(groups) * n
    m = torch.ones(D, D, dtype=torch.bool, device=device)
    for k in range(int(groups)):
        m[k * n:(k + 1) * n, k * n:(k + 1) * n] = False
    return m


def dn_forward(dec, memory: Tensor, dn: dict) -> list:
    """The DN pass through the decoder's OWN modules; returns the per-layer decodes (last layer last)."""
    q = dn["q"]
    mem = dec.mem_proj(memory) + dec.mem_pos.to(memory.dtype)
    mask = dn_attention_mask(dn["groups"], dn["A_max"], device=q.device)
    # ⛔ an element with NO positive has every key of every group padded: a fully masked softmax row is NaN, and
    # NaN x 0 is NaN in any loss. Its (dummy) queries are therefore left unpadded among themselves -- they carry no
    # loss either way (``dn_losses`` selects with ``torch.where``, never a multiplication).
    kp = dn["pad"].clone()
    kp[~(~dn["pad"]).any(1)] = False
    x = q.to(memory.dtype)
    outs = []
    for layer in dec.blocks.layers:
        x = layer(x, mem, tgt_mask=mask, tgt_key_padding_mask=kp)
        outs.append(dec.decode(dec.head(dec.norm(x))))
    return outs


def dn_losses(outs: list, tgt_pos: dict, dn: dict, *, cls_class_weight=None) -> dict:
    """Per-layer reconstruction: positives -> their GT (presence 1, class, box, yaw, z/h), negatives -> presence 0."""
    from tanitad.models import slot_presence as SP
    from tanitad.models.slot_presence import sigmoid_focal_elementwise
    pad = dn["pad"]
    is_pos = dn["is_pos"] & ~pad
    is_neg = ~dn["is_pos"] & ~pad
    n_pos = max(int(is_pos.sum()), 1)
    gt = dn["gt"]
    B, D = gt.shape
    tb = torch.gather(tgt_pos["box"], 1, gt[..., None].expand(-1, -1, 4))
    ty = torch.gather(tgt_pos["yaw"], 1, gt)
    tc = torch.gather(tgt_pos["cls"], 1, gt)
    has_z = tgt_pos.get("zh_mask") is not None
    if has_z:
        tz = torch.gather(tgt_pos["cz"], 1, gt)
        th = torch.gather(tgt_pos["h"], 1, gt)
        tm = torch.gather(tgt_pos["zh_mask"].to(torch.bool), 1, gt) & is_pos
    total = 0.0
    parts = {}
    def sel(x, m):                               # NaN-safe masked sum (never x * 0)
        return torch.where(m, x, torch.zeros_like(x)).sum()
    for i, p in enumerate(outs):
        tp = is_pos.to(p["presence_logit"].dtype)
        l_pres = sel(sigmoid_focal_elementwise(p["presence_logit"], tp), is_pos | is_neg) / n_pos
        l_ctr = sel((p["box"][..., :2] - tb[..., :2]).abs().sum(-1), is_pos) / n_pos
        l_size = sel((p["box"][..., 2:] - tb[..., 2:]).abs().sum(-1), is_pos) / n_pos
        yv = torch.stack([torch.sin(ty), torch.cos(ty)], -1)
        l_yaw = sel(1.0 - (p["yaw_vec"] * yv).sum(-1), is_pos) / n_pos
        ok = is_pos & (tc >= 0)
        if bool(ok.any()):
            cw = None if cls_class_weight is None else cls_class_weight.to(p["cls_logits"].device,
                                                                            p["cls_logits"].dtype)
            ce = F.cross_entropy(p["cls_logits"][ok], tc[ok], weight=cw, reduction="sum")
            l_cls = ce / (float(ok.sum()) if cw is None else float(cw[tc[ok]].sum()))
        else:
            l_cls = p["presence_logit"].sum() * 0.0
        # the SAME presence weight as the matching queries (read at call time, so the harness's presence_w0 arm
        # zeroes BOTH presence terms -- otherwise the DN term alone could rescue a must-fail arm)
        tot = SP.FOCAL_PRESENCE_W * l_pres + l_cls + l_ctr + l_size + l_yaw
        if has_z and "cz" in p and bool(tm.any()):
            nz = max(int(tm.sum()), 1)
            l_z = sel((p["cz"] - tz).abs(), tm) / nz
            l_h = sel((p["h"] - th).abs(), tm) / nz
            tot = tot + l_z + l_h
            parts[f"dn_z_layer{i}"] = l_z
        parts[f"dn_layer{i}"] = tot
        total = total + tot
    last = len(outs) - 1
    parts.update({"dn_presence": l_pres, "dn_centre": l_ctr, "dn_cls": l_cls, "dn_size": l_size, "dn_yaw": l_yaw,
                  "dn_n_pos": float(int(is_pos.sum())), "dn_n_neg": float(int(is_neg.sum())),
                  "dn_n_layers": float(len(outs)), "dn_last_layer": float(last)})
    return {"total": DN_LOSS_W * total, "parts": parts}
