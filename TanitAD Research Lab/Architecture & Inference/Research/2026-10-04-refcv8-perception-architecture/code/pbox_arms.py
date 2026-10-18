"""WP-D P-BOX arms (PREREG_WPD_PROBES.md, registered sha256 c054190b…): the refcv7 box decoder retrained on the FROZEN
box memory cached by `pbox_capture.py`, one variable per arm.

Every arm starts from the refcv7-50,400 decoder (`decoder.pt`: box_dec, learned reference points, query-position
embed), trains 4,000 steps at batch 16 on the TRAIN cache (AdamW lr 1e-4 constant to step 3,599, cosine to 0 over the
final 400, wd 1e-4, clip 10, seed), then writes the trainer's own detection packs (`detection_metrics.window_packs`,
last decoder layer) on EVAL-DIAG and on TRAIN-DIAG (the registered P = R gate fit set). The metrics are computed
off-GPU by `pbox_analyze.py`.

The loss is the launch loss (`slot_presence.refined_box3d_losses`: focal presence alpha 0.25 gamma 2 x 2.0, deep
supervision over the 3 layers, VIS-1 POSITIVE / IGNORE with the 2 m presence exemption, `box3d_set_loss` terms) rebuilt
from the SAME library functions so that one arm can change one thing (pinned by control C-loss: on PB0 the rebuilt
total equals the library's `refined_box3d_losses` total on the same slots, rel. err <= 1e-6).

Arms (registered; deviations disclosed in RESULT): PB0 control · PB0r seed 1 · PB1 contrastive denoising (centre noise
only: our anchors are (x, y), size / yaw have no input path) · PB2 K = 4 query groups · PB3 varifocal quality presence
q = exp(-d^2/2) · PB4 per-layer anchor refinement · PB4h heatmap query selection (fresh heat head) · PB6 = PB1+PB2+PB3+PB4
· PB7 ray position embedding on the image tokens · PB8 0.5 m BEV sampling at the anchor · must-fail R-pres0 (presence
weight 0), R-pres0z (the A1 proposal: + presence row reset), R-memshuf (memory rolled across the batch, train AND eval),
R-qshuf (PB3 with q permuted among the batch's matched slots).
Controls run first with --controls: C-fwd (this file's decoder pass == `slot_query_select.anchored_forward`),
C-loss (above), C-cache (step-0 EVAL AP@2 m of the cached memory vs the diagnostics' T0 0.2483, +-0.003).
Env: PYTHONPATH=<launch tree>/stack (the decoder modules unpickle from it). ⛔ sha12 only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = {
    "PB0": {}, "PB0r": {"seed": 1}, "PB1": {"dn": True}, "PB2": {"groups": 4}, "PB3": {"quality": True},
    "PB4": {"refine": True}, "PB4h": {"hqs": True},
    "PB6": {"dn": True, "groups": 4, "quality": True, "refine": True},
    "PB7": {"petr": True}, "PB8": {"bevsample": True},
    "R-pres0": {"pres_w": 0.0}, "R-pres0z": {"pres_w": 0.0, "pres_reset": True},
    "R-memshuf": {"memshuf": True}, "R-qshuf": {"quality": True, "qshuf": True},
}
TGT_KEYS = ("box", "yaw", "cls", "valid", "occ", "rates", "rates_mask", "cz", "h", "zh_mask")
VIS_KEYS = ("n_full", "n_vis", "known")
FOCAL_W = 2.0
DN_GROUPS, DN_POS_MAX, DN_NEG_LO, DN_NEG_HI, DN_PMAX = 3, 0.4, 0.4, 1.0, 64


# --------------------------------------------------------------------------------------------------------- #
# the cache                                                                                                  #
# --------------------------------------------------------------------------------------------------------- #
class Cache:
    def __init__(self, d: Path):
        recs = []
        for p in sorted(d.glob("shard_*.pt")):
            recs += torch.load(p, map_location="cpu", weights_only=False)
        if not recs:
            raise SystemExit(f"[pbox] no shards in {d}")
        self.n = len(recs)
        self.sha12 = [r["sha12"] for r in recs]
        self.t = [r["t"] for r in recs]
        self.mi_q = torch.stack([r["mem_img_q"] for r in recs])
        self.mi_lo = torch.stack([r["mem_img_lo"] for r in recs])
        self.mi_sc = torch.stack([r["mem_img_sc"] for r in recs])
        self.mb_q = torch.stack([r["mem_bev_q"] for r in recs])
        self.mb_lo = torch.stack([r["mem_bev_lo"] for r in recs])
        self.mb_sc = torch.stack([r["mem_bev_sc"] for r in recs])
        self.bv_q = torch.stack([r["bev_q"] for r in recs])
        self.bv_lo = torch.stack([r["bev_lo"] for r in recs])
        self.bv_sc = torch.stack([r["bev_sc"] for r in recs])
        self.tgt = [r["tgt"] for r in recs]
        self.vis = [r["vis"] for r in recs]
        self.R = torch.stack([r["R_cam_to_rig"] for r in recs])
        self.tcam = torch.stack([r["t_cam_in_rig"] for r in recs])
        self.live_pl = torch.stack([r["live_presence_logit"] for r in recs])

    def batch(self, idx, dev, with_bev=True):
        idx = torch.as_tensor(idx, dtype=torch.long)
        mi = self.mi_q[idx].to(dev).float() * self.mi_sc[idx].to(dev) + self.mi_lo[idx].to(dev)
        mb = self.mb_q[idx].to(dev).float() * self.mb_sc[idx].to(dev) + self.mb_lo[idx].to(dev)
        out = {"mem": torch.cat([mi, mb], 1), "R": self.R[idx].to(dev), "tcam": self.tcam[idx].to(dev)}
        if with_bev:
            b = self.bv_q[idx].to(dev).float()
            C = b.shape[1]
            out["bev"] = (b.reshape(len(idx), C, -1) * self.bv_sc[idx].to(dev) +
                          self.bv_lo[idx].to(dev)).reshape(b.shape)
        A = max(1, max(int(self.tgt[i]["valid"].shape[0]) for i in idx.tolist()))
        tg, vs = {}, {}
        for k in TGT_KEYS:
            ref = self.tgt[int(idx[0])][k]
            z = torch.zeros((len(idx), A) + tuple(ref.shape[1:]), dtype=ref.dtype)
            for j, i in enumerate(idx.tolist()):
                v = self.tgt[i][k]
                z[j, :v.shape[0]] = v
            tg[k] = z.to(dev)
        for k in VIS_KEYS:
            ref = self.vis[int(idx[0])][k]
            z = torch.zeros((len(idx), A) + tuple(ref.shape[1:]), dtype=ref.dtype)
            for j, i in enumerate(idx.tolist()):
                v = self.vis[i][k]
                z[j, :v.shape[0]] = v
            vs[k] = z.to(dev)
        out["tgt"], out["vis"] = tg, vs
        return out


# --------------------------------------------------------------------------------------------------------- #
# the decoder pass                                                                                           #
# --------------------------------------------------------------------------------------------------------- #
def layer_fwd(layer, x, mem, qpos, mask=None):
    """`slot_query_select._layer_with_pos` with an optional self-attention mask (bool, True = blocked)."""
    h = layer.norm1(x)
    q = h + qpos
    x = x + layer.dropout1(layer.self_attn(q, q, h, attn_mask=mask, need_weights=False)[0])
    h = layer.norm2(x)
    x = x + layer.dropout2(layer.multihead_attn(h + qpos, mem, mem, need_weights=False)[0])
    h = layer.norm3(x)
    return x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(h)))))


def slice_slots(d, s, e):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.dim() >= 2:
            out[k] = v[:, s:e]
        else:
            out[k] = v
    return out


class Arm(nn.Module):
    def __init__(self, D, cfg, dev):
        super().__init__()
        self.cfg = cfg
        self.dec = copy.deepcopy(D["box_dec"]).to(dev)
        self.refpts = copy.deepcopy(D["box_refpts"]).to(dev)
        self.qpos = copy.deepcopy(D["box_qpos"]).to(dev)
        # ⛔ the modules were saved off the EVAL loader's model, where requires_grad is False everywhere (MEASURED: the
        # first PB0 job died on "optimizer got an empty parameter list" AFTER every control passed)
        for m_ in (self.dec, self.refpts, self.qpos):
            for p_ in m_.parameters():
                p_.requires_grad_(True)
        self.n_img = int(D["n_img_tokens"])
        self.grid = D["planner_grid"]
        self.K = int(cfg.get("groups", 1))
        g = torch.Generator().manual_seed(1234)
        self.xq = nn.ParameterList()
        self.xa = nn.ParameterList()
        for _ in range(self.K - 1):
            self.xq.append(nn.Parameter(self.dec.queries.detach().clone()))
            jit = (torch.rand(self.refpts.xy.shape, generator=g) * 2 - 1).to(dev)
            self.xa.append(nn.Parameter(self.refpts.xy.detach().clone() + jit))
        if cfg.get("dn"):
            self.dn_content = nn.Parameter(self.dec.queries.detach().mean(1, keepdim=True).clone())
        if cfg.get("hqs"):
            from tanitad.models.slot_query_select import BEVHeatHead
            self.heat = BEVHeatHead(96).to(dev)
        if cfg.get("petr"):
            self.pe = nn.Sequential(nn.Linear(6, 256), nn.ReLU(), nn.Linear(256, 256)).to(dev)
            nn.init.zeros_(self.pe[2].weight)
            nn.init.zeros_(self.pe[2].bias)
            fr = D["frame"]
            th, tw = D["image_hw_tokens"]
            rows = torch.arange(th, dtype=torch.float32) * (fr["height"] / th)
            cols = torch.arange(tw, dtype=torch.float32) * (fr["width"] / tw)
            rr, cc = torch.meshgrid(rows, cols, indexing="ij")
            az = (cc - (fr["width"] - 1) / 2.0) / fr["f_ref"]
            el = (rr - (fr["height"] - 1) / 2.0) / fr["f_ref"]
            d = torch.stack([torch.sin(az), el, torch.cos(az)], -1).reshape(-1, 3)
            self.register_buffer("ray_cam", F.normalize(d, dim=-1).to(dev))
        if cfg.get("bevsample"):
            self.bproj = nn.Linear(96 * 9, 256).to(dev)
            nn.init.zeros_(self.bproj.weight)
            nn.init.zeros_(self.bproj.bias)
        if cfg.get("pres_reset"):
            from tanitad.models.agent_slots import SLOT_SLICES
            s = SLOT_SLICES["presence"]
            with torch.no_grad():
                self.dec.head.weight[s] = 0.0
                self.dec.head.bias[s] = math.log(0.01 / 0.99)

    def bev_sample(self, bev, anchors):
        B, N, _ = anchors.shape
        offs = torch.tensor([[dx, dy] for dx in (-0.5, 0.0, 0.5) for dy in (-0.5, 0.0, 0.5)], device=bev.device)
        p = anchors[:, :, None, :] + offs[None, None]                       # [B, N, 9, 2]
        gx = (p[..., 1] + float(self.grid.y_half_m)) / (2 * float(self.grid.y_half_m)) * 2 - 1
        gy = p[..., 0] / float(self.grid.x_fwd_m) * 2 - 1
        s = F.grid_sample(bev, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="zeros",
                          align_corners=False)                               # [B, 96, N, 9]
        return self.bproj(s.permute(0, 2, 1, 3).reshape(B, N, -1))

    def forward(self, bat, train=True, dn=None):
        from tanitad.models.slot_query_select import anchored_decode, heat_grid_of, select_anchors
        dec = self.dec
        mem = bat["mem"]
        B = mem.shape[0]
        if self.cfg.get("memshuf") and B > 1:
            mem = torch.roll(mem, 1, 0)
        mem_p = dec.mem_proj(mem) + dec.mem_pos.to(mem.dtype)
        if self.cfg.get("petr"):
            d_rig = torch.einsum("bij,nj->bni", bat["R"], self.ray_cam)        # [B, 1664, 3]
            feat = torch.cat([d_rig, (bat["tcam"] / 2.0)[:, None, :].expand_as(d_rig)], -1)
            mem_p = torch.cat([mem_p[:, :self.n_img] + self.pe(feat), mem_p[:, self.n_img:]], 1)
        extra = {}
        if self.cfg.get("hqs"):
            heat_logits = self.heat(bat["bev"])
            grid = heat_grid_of(self.grid, heat_logits)
            a0 = select_anchors(heat_logits, int(dec.n_queries), grid)[0]   # (anchors, scores, idx), detached
            extra.update({"heat_logits": heat_logits, "heat_grid": grid})
        else:
            a0 = self.refpts(B)
        Kg = self.K if train else 1
        qs = [dec.queries.expand(B, -1, -1)] + [self.xq[k].expand(B, -1, -1) for k in range(Kg - 1)]
        As = [a0] + [self.xa[k][None].expand(B, -1, -1) for k in range(Kg - 1)]
        Nq = int(dec.n_queries)
        n_dn = 0
        if dn is not None:
            qs.append(self.dn_content.expand(B, dn["anchors"].shape[1], -1))
            As.append(dn["anchors"])
            n_dn = int(dn["anchors"].shape[1])
        x = torch.cat(qs, 1)
        anchors = torch.cat(As, 1)
        L = x.shape[1]
        mask = None
        if Kg > 1 or n_dn:
            allowed = torch.zeros(B, L, L, dtype=torch.bool, device=x.device)
            for k in range(Kg):
                allowed[:, k * Nq:(k + 1) * Nq, k * Nq:(k + 1) * Nq] = True
            if n_dn:
                o = Kg * Nq
                gsz = n_dn // DN_GROUPS
                for gi in range(DN_GROUPS):
                    s, e = o + gi * gsz, o + (gi + 1) * gsz
                    allowed[:, s:e, s:e] = dn["valid"][:, None, gi * gsz:(gi + 1) * gsz] & \
                        dn["valid"][:, gi * gsz:(gi + 1) * gsz, None]
                    allowed[:, s:e, :Nq] = True                       # DN sees the matching group 0
                idx = torch.arange(L, device=x.device)
                allowed[:, idx, idx] = True                           # nobody attends to nothing
            mask = (~allowed).repeat_interleave(dec.blocks.layers[0].self_attn.num_heads, 0)
        qpos = self.qpos(anchors)
        outs = []
        for layer in dec.blocks.layers:
            if self.cfg.get("bevsample"):
                x = x + self.bev_sample(bat["bev"], anchors)
            x = layer_fwd(layer, x, mem_p, qpos, mask)
            raw = dec.head(dec.norm(x))
            o = anchored_decode(dec, raw, anchors)
            outs.append(o)
            if self.cfg.get("refine"):
                anchors = o["box"][..., :2].detach()
                qpos = self.qpos(anchors)
        groups = []
        for k in range(Kg):
            lay = [slice_slots(o, k * Nq, (k + 1) * Nq) for o in outs]
            g = dict(lay[-1])
            g["aux"] = lay[:-1]
            groups.append(g)
        dn_out = [slice_slots(o, Kg * Nq, L) for o in outs] if n_dn else None
        return groups, dn_out, extra


# --------------------------------------------------------------------------------------------------------- #
# losses                                                                                                     #
# --------------------------------------------------------------------------------------------------------- #
def targets(bat):
    from tanitad.data.vis1 import vis1_split
    t = bat["tgt"]
    sp = vis1_split(t, n_full=bat["vis"]["n_full"], n_vis=bat["vis"]["n_vis"], vis_known=bat["vis"]["known"])
    tpos = sp["pos"]
    if tpos.get("zh_mask") is not None:
        tpos = {**tpos, "zh_mask": tpos["zh_mask"].to(torch.bool) & tpos["valid"].to(torch.bool)}
    return tpos, sp["ignore"]


def vfl_presence(p, m, tpos, iw, qshuf, gen):
    from tanitad.models.slot_presence import FOCAL_ALPHA, FOCAL_GAMMA
    logit = p["presence_logit"]
    tgt = torch.zeros_like(logit)
    matched = torch.zeros_like(logit, dtype=torch.bool)
    rows, qs = [], []
    for b, (r, c) in enumerate(zip(m["rows"], m["cols"])):
        if r.numel():
            r, c = r.to(logit.device), c.to(logit.device)
            d = (p["box"][b, r, :2].detach() - tpos["box"][b, c, :2]).norm(dim=-1)
            rows.append((b, r))
            qs.append(torch.exp(-d * d / 2.0))
    if qs:
        qcat = torch.cat(qs)
        if qshuf:
            qcat = qcat[torch.randperm(qcat.numel(), generator=gen).to(qcat.device)]
        k = 0
        for b, r in rows:
            tgt[b, r] = qcat[k:k + r.numel()]
            matched[b, r] = True
            k += r.numel()
    n_m = int(matched.sum())
    w = iw.to(logit.dtype).clone()
    w[matched] = 1.0
    bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
    pr = torch.sigmoid(logit)
    neg = (1.0 - FOCAL_ALPHA) * pr.pow(FOCAL_GAMMA) * bce
    pos = tgt * bce
    el = torch.where(matched, pos, neg)
    return (el * w).sum() / float(max(n_m, 1))


def set_loss(group, tpos, tgt_real, ignore, cfg, cw, gen):
    from tanitad.models.agent_slots import match_slots
    from tanitad.models.box3d_head import box3d_set_loss
    from tanitad.models.slot_presence import ignore_presence_weight, layer_preds, presence_term
    pw = float(cfg.get("pres_w", FOCAL_W))
    tot = 0.0
    for p in layer_preds(group):
        m = match_slots(p, tpos, presence_cost="focal")
        base = box3d_set_loss(p, tpos, match=m, weights={"presence": 0.0}, cls_class_weight=cw,
                              visible_filter=False)
        iw = ignore_presence_weight(p["box"], tgt_real["box"], ignore)
        if cfg.get("quality"):
            pr = vfl_presence(p, m, tpos, iw, bool(cfg.get("qshuf")), gen)
        else:
            pr = presence_term(p["presence_logit"], m, mode="focal", ignore_w=iw)["loss"]
        tot = tot + base["total"] + pw * pr
    return tot


def make_dn(tpos, gen, dev):
    """CDN queries: per group, Pmax noised positives then Pmax noised negatives (centre noise only)."""
    B = tpos["valid"].shape[0]
    pos_idx = [tpos["valid"][b].nonzero(as_tuple=False).flatten()[:DN_PMAX] for b in range(B)]
    Pm = max(int(p.numel()) for p in pos_idx)
    if Pm == 0:
        return None
    anchors = torch.zeros(B, DN_GROUPS * 2 * Pm, 2, device=dev)
    valid = torch.zeros(B, DN_GROUPS * 2 * Pm, dtype=torch.bool, device=dev)
    for gi in range(DN_GROUPS):
        o = gi * 2 * Pm
        for b in range(B):
            pi = pos_idx[b]
            n = int(pi.numel())
            if not n:
                continue
            ctr = tpos["box"][b, pi, :2]
            ext = tpos["box"][b, pi, 2:4].max(-1).values.clamp_min(0.5)[:, None]
            u = (torch.rand(n, 2, generator=gen) * 2 - 1).to(dev)
            anchors[b, o:o + n] = ctr + u * DN_POS_MAX * ext
            mag = (DN_NEG_LO + torch.rand(n, 2, generator=gen) * (DN_NEG_HI - DN_NEG_LO)).to(dev)
            sgn = torch.where(torch.rand(n, 2, generator=gen) < 0.5, -1.0, 1.0).to(dev)
            anchors[b, o + Pm:o + Pm + n] = ctr + sgn * mag * ext
            valid[b, o:o + n] = True
            valid[b, o + Pm:o + Pm + n] = True
    return {"anchors": anchors, "valid": valid, "pos_idx": pos_idx, "Pm": Pm}


def dn_loss(dn_out, dn, tpos, cw):
    from tanitad.models.box3d_head import box3d_set_loss
    from tanitad.models.slot_presence import sigmoid_focal_elementwise
    Pm = dn["Pm"]
    tot = 0.0
    for gi in range(DN_GROUPS):
        o = gi * 2 * Pm
        for p in dn_out:
            pg = slice_slots(p, o, o + 2 * Pm)
            rows = [torch.arange(int(pi.numel()), dtype=torch.long) for pi in dn["pos_idx"]]
            cols = [pi.cpu() for pi in dn["pos_idx"]]
            m = {"rows": rows, "cols": cols, "n_target": [int(r.numel()) for r in rows],
                 "n_dropped": [0] * len(rows)}
            base = box3d_set_loss(pg, tpos, match=m, weights={"presence": 0.0}, cls_class_weight=cw,
                                  visible_filter=False)
            lg = pg["presence_logit"]
            tg = torch.zeros_like(lg)
            tg[:, :Pm] = 1.0
            vm = dn["valid"][:, o:o + 2 * Pm].to(lg.dtype)
            n_pos = max(1, sum(int(r.numel()) for r in rows))
            pres = (sigmoid_focal_elementwise(lg, tg) * vm).sum() / float(n_pos)
            tot = tot + base["total"] + FOCAL_W * pres
    return tot / DN_GROUPS


def arm_loss(model, bat, cfg, cw, gen):
    tpos, ign = targets(bat)
    dn = make_dn(tpos, gen, bat["mem"].device) if cfg.get("dn") else None
    groups, dn_out, extra = model(bat, train=True, dn=dn)
    tot = sum(set_loss(g, tpos, bat["tgt"], ign, cfg, cw, gen) for g in groups) / len(groups)
    if dn is not None:
        tot = tot + dn_loss(dn_out, dn, tpos, cw)
    if "heat_logits" in extra:
        from tanitad.models.slot_query_select import HEAT_LOSS_W, heat_term
        ht = heat_term(extra["heat_logits"], bat["tgt"]["box"], tpos["valid"].to(torch.bool), ign,
                       extra["heat_grid"])
        tot = tot + HEAT_LOSS_W * ht["loss"]
    return tot


# --------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
def eval_packs(model, cache, dev, cw, bs=32):
    from tanitad.eval import detection_metrics as det
    model.eval()
    packs = []
    for s in range(0, cache.n, bs):
        idx = list(range(s, min(cache.n, s + bs)))
        bat = cache.batch(idx, dev, with_bev=bool(model.cfg.get("hqs") or model.cfg.get("bevsample")))
        groups, _, _ = model(bat, train=False)
        last = {k: v for k, v in groups[0].items() if k != "aux"}
        vis = {"n_full": bat["vis"]["n_full"], "n_vis": bat["vis"]["n_vis"], "known": bat["vis"]["known"]}
        pk = det.window_packs(last, bat["tgt"], vis, presence_cost="focal", cls_weight=cw)
        for j, p in enumerate(pk):
            p.pop("ep", None)
            p["sha12"] = cache.sha12[idx[j]]
            p["t"] = cache.t[idx[j]]
        packs += pk
    model.train()
    return packs


def ap2(packs):
    from tanitad.eval import detection_metrics as det
    return float(det.summarise(packs, "box3d")["eval_box3d_ap2m"])


def run_controls(D, caches, dev, out):
    from tanitad.models.slot_presence import refined_box3d_losses
    from tanitad.models.slot_query_select import anchored_forward
    cw = D["cls_class_weight"]
    cw = cw.to(dev) if cw is not None else None
    m = Arm(D, {}, dev)
    m.eval()
    tr = caches["train"]
    bat = tr.batch(list(range(16)), dev)
    with torch.no_grad():
        groups, _, _ = m(bat, train=False)
        ref = anchored_forward(m.dec, bat["mem"], m.refpts(16), m.qpos(m.refpts(16)))
    d_fwd = float((groups[0]["presence_logit"] - ref["presence_logit"]).abs().max())
    d_box = float((groups[0]["box"] - ref["box"]).abs().max())
    m.train()
    groups, _, _ = m(bat, train=True)
    tpos, ign = targets(bat)
    mine = set_loss(groups[0], tpos, bat["tgt"], ign, {}, cw, torch.Generator().manual_seed(0))
    lib = refined_box3d_losses(groups[0], bat["tgt"], presence_loss="focal", vis1=True, vis=bat["vis"],
                               visible_filter=True, cls_class_weight=cw)["total"]
    rel = float((mine - lib).abs() / lib.abs().clamp_min(1e-12))
    ev = eval_packs(m, caches["eval"], dev, cw)
    a = ap2(ev)
    with torch.no_grad():
        m.eval()
        n_live = min(64, caches["eval"].n)
        pl_cached = m(caches["eval"].batch(list(range(n_live)), dev, with_bev=False), train=False)[0][0][
            "presence_logit"]
        live_d = float((caches["eval"].live_pl[:n_live].to(dev) - pl_cached).abs().max())
        m.train()
    rec = {"C_fwd_max_abs_presence": d_fwd, "C_fwd_max_abs_box": d_box, "C_fwd_pass": d_fwd <= 1e-5 and d_box <= 1e-5,
           "C_loss_rel_err": rel, "C_loss_pass": rel <= 1e-6,
           "C_cache_ap2_step0": a, "C_cache_expected": 0.2483, "C_cache_pass": abs(a - 0.2483) <= 0.003,
           "C_cache_live_vs_cached_presence_max_abs_64win": live_d, "n_eval_windows": caches["eval"].n,
           "n_train_windows": caches["train"].n}
    (out / "controls.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    with open(out / "step0_eval_packs.pkl", "wb") as fh:
        pickle.dump(ev, fh, protocol=4)
    print(f"[pbox] controls {rec}", flush=True)
    return rec


def train_arm(name, D, caches, dev, out, steps, bs, lr, budget_s):
    cfg = dict(ARMS[name])
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = Arm(D, cfg, dev)
    m.train()
    cw = D["cls_class_weight"]
    cw = cw.to(dev) if cw is not None else None
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    theta0 = torch.cat([p.detach().reshape(-1).float().cpu() for p in m.dec.parameters()])
    tr = caches["train"]
    need_bev = bool(cfg.get("hqs") or cfg.get("bevsample"))
    perm, ptr, t0, log, start = None, 0, time.time(), [], 0
    sp = out / "state.pt"
    if sp.exists():                       # RESUME (a job stops at its wall budget; the step count defines the arm)
        st = torch.load(sp, map_location="cpu", weights_only=False)
        m.load_state_dict(st["arm"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"])
        torch.set_rng_state(st["torch_rng"])
        perm, ptr, log, start = st["perm"], st["ptr"], st["log"], st["step"]
        print(f"[pbox] {name} resumed at step {start}", flush=True)

    def save_state(step_next):
        torch.save({"arm": m.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                    "torch_rng": torch.get_rng_state(), "perm": perm, "ptr": ptr, "log": log, "step": step_next},
                   str(sp) + ".tmp")
        os.replace(str(sp) + ".tmp", sp)
    for step in range(start, steps):
        if perm is None or ptr + bs > len(perm):
            perm, ptr = torch.randperm(tr.n, generator=gen).tolist(), 0
        idx = perm[ptr:ptr + bs]
        ptr += bs
        f = 1.0 if step < steps - 400 else 0.5 * (1 + math.cos(math.pi * (step - (steps - 400)) / 400))
        for g in opt.param_groups:
            g["lr"] = lr * f
        bat = tr.batch(idx, dev, with_bev=need_bev)
        loss = arm_loss(m, bat, cfg, cw, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 10.0))
        opt.step()          # ⛔ MISSING in the first PB0 job: it trained nothing (EVAL AP equalled step 0 to 7 dp)
        if step % 200 == 0 or step == steps - 1:
            el = time.time() - t0
            log.append({"step": step, "loss": float(loss.detach()), "grad_norm": gn, "lr": lr * f, "s": el})
            print(f"[pbox] {name} step {step} loss {float(loss.detach()):.4f} gn {gn:.2f} {el:.0f}s", flush=True)
        if (step + 1) % 500 == 0:
            save_state(step + 1)
        if time.time() - t0 > budget_s and step + 1 < steps:
            save_state(step + 1)
            print(f"[pbox] {name} PARTIAL at step {step + 1} (budget) -- resumable", flush=True)
            return 2
    if time.time() - t0 > budget_s + 200:
        save_state(steps)
        print(f"[pbox] {name} trained; eval next job (budget)", flush=True)
        return 2
    res = {}
    for split in ("eval", "train_diag"):
        pk = eval_packs(m, caches[split], dev, cw)
        with open(out / f"{split}_packs.pkl", "wb") as fh:
            pickle.dump(pk, fh, protocol=4)
        res[f"{split}_ap2m_quick"] = ap2(pk)
    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, out / "arm_state.pt")
    theta1 = torch.cat([p.detach().reshape(-1).float().cpu() for p in m.dec.parameters()])
    dtheta = float((theta1 - theta0).norm() / theta0.norm().clamp_min(1e-12))
    if dtheta == 0.0 and steps > 0:
        raise SystemExit(f"[pbox] {name}: CONTROL C-step FAILED -- the decoder did not change in {steps} steps")
    rec = {"arm": name, "cfg": cfg, "steps": steps, "batch": bs, "lr": lr, "seed": seed, "log": log,
           "C_step_rel_param_change": dtheta,
           "train_wall_s": time.time() - t0, **res, "n_params_trainable": int(sum(p.numel() for p in params))}
    (out / "record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[pbox] DONE {name} eval AP@2m(quick) {res['eval_ap2m_quick']:.4f} wall {rec['train_wall_s']:.0f}s",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--budget-s", type=float, default=1900)
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    root = Path(a.cache)
    D = torch.load(root / "decoder.pt", map_location="cpu", weights_only=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    caches = {s: Cache(root / s) for s in ("train", "eval", "train_diag") if (root / s).exists()}
    print(f"[pbox] caches { {k: v.n for k, v in caches.items()} }", flush=True)
    if a.controls:
        run_controls(D, caches, dev, out)
    rc = 0
    for name in [x for x in a.arms.split(",") if x]:
        od = out / name
        if (od / "record.json").exists():
            print(f"[pbox] {name} already banked -- skip", flush=True)
            continue
        od.mkdir(parents=True, exist_ok=True)
        rc = max(rc, train_arm(name, D, caches, dev, od, a.steps, a.batch, a.lr, a.budget_s) or 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
