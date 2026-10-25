"""Level-wide critics and losses (paper sec:train). The default vocabulary loss is the level-wide soft-target softmax;
masked/unmasked InfoNCE are the registered ablation (H-HC4)."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import HiCAP, LEVELS


def lr_rule(width: int, batch: int) -> float:
    return 2e-3 * math.sqrt(1024 / width) * math.sqrt(batch / 1024)


def soft_target(dist: torch.Tensor, sigma: float = 0.5, valid: torch.Tensor | None = None) -> torch.Tensor:
    """y(a) ∝ exp(-d^2 / 2 sigma^2) over candidates (VADv2 distance-weighted target). dist [B,K] in metres."""
    z = -(dist ** 2) / (2 * sigma ** 2)
    if valid is not None: z = z.masked_fill(~valid, -1e9)
    return torch.softmax(z, -1)


def subtree_sum(y_leaf: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
    """Ancestor summation: y_parent = sum over leaves in the subtree. y_leaf [B,K], member [K] class id -> [B,C]."""
    C = int(member.max().item()) + 1
    out = y_leaf.new_zeros(y_leaf.shape[0], C)
    return out.index_add(1, member, y_leaf)


def level_logits(s, A, tau, logq=None, beta=0.0, mask=None):
    z = tau * s @ A.t()
    if logq is not None and beta != 0.0: z = z + beta * logq
    if mask is not None: z = z.masked_fill(~mask, -1e9)
    return z


def level_loss(z: torch.Tensor, y: torch.Tensor, ignore: torch.Tensor | None = None) -> torch.Tensor:
    """-sum_v y(v) log softmax(z)(v), with y a soft target; rows flagged `ignore` are dropped."""
    l = -(y * F.log_softmax(z, -1)).sum(-1)
    if ignore is not None:
        keep = ~ignore
        return l[keep].mean() if keep.any() else z.sum() * 0.0
    return l.mean()


def hard_to_soft(label: torch.Tensor, n: int, smooth: float = 0.0):
    """label [B] with -100 = IGNORE -> (soft one-hot [B,n], ignore mask)."""
    ign = label < 0
    y = F.one_hot(label.clamp(min=0), n).float()
    if smooth: y = y * (1 - smooth) + smooth / n
    return y, ign


def infonce_bidirectional(s: torch.Tensor, a: torch.Tensor, tau: torch.Tensor, action_id: torch.Tensor | None = None):
    """CLM-style in-batch InfoNCE. With action_id the same-action false negatives are masked (changes the optimum,
    Remark after Prop. 1: beta must then be fitted); without, the optimum is PMI (Corollary: bias)."""
    z = tau * s @ a.t()
    if action_id is not None:
        same = action_id[:, None] == action_id[None, :]
        off = same & ~torch.eye(len(s), dtype=torch.bool, device=s.device)
        z = z.masked_fill(off, -1e9)
    t = torch.arange(len(s), device=s.device)
    return 0.5 * (F.cross_entropy(z, t) + F.cross_entropy(z.t(), t))


def consistency_kl(p_cascade: torch.Tensor, z_level: torch.Tensor):
    """KL( sg[p_cascade] || softmax(z_level) )."""
    return F.kl_div(F.log_softmax(z_level, -1), p_cascade.detach(), reduction="batchmean")


def goal_loss(logits, y, w, excl_pairs, pos_weight_cap=50.0, lam_excl=0.1, pos_rate=None):
    """BCE over the 22 goals on supervised entries only (w=1); positive weight capped; exclusivity penalty on pairs (i,j)."""
    pw = torch.ones(logits.shape[-1], device=logits.device)
    if pos_rate is not None: pw = ((1 - pos_rate) / pos_rate.clamp(min=1e-6)).clamp(max=pos_weight_cap)
    l = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none") * w
    out = l.sum() / w.sum().clamp(min=1.0)
    p = torch.sigmoid(logits)
    if excl_pairs is not None and len(excl_pairs):
        i, j = excl_pairs[:, 0], excl_pairs[:, 1]
        out = out + lam_excl * (p[:, i] * p[:, j]).mean()
    return out


@dataclass
class Stage:
    name: str
    hard_negatives: bool
    replay: float
    lam: dict


HARD_W = 2.0                                  # weight of near-miss negatives (ADE in [0.25, 1.0] m) in the stage-2 normaliser (HYPOTHESIS)

STAGES = {
    1: Stage("stage1", False, 0.0, dict(voc=1.0, cons=0.0, goal=0.2, cost=0.0, ref=0.0, cfg=0.0, aud=0.0, img=0.0)),
    2: Stage("stage2", True, 0.4, dict(voc=1.0, cons=0.1, goal=0.2, cost=0.0, ref=0.0, cfg=0.0, aud=0.0, img=0.0)),
    3: Stage("stage3", True, 0.4, dict(voc=1.0, cons=0.1, goal=0.2, cost=0.3, ref=0.3, cfg=0.1, aud=0.1, img=0.1)),
}


class VocabTensors(nn.Module):
    """Buffers describing the vocabularies; node embeddings are recomputed from the towers each step."""
    def __init__(self, strat_text, strat_args, lat_text, lat_args, lon_text, lon_args,
                 path_feat, path_cls, prof_feat, prof_cls, logq: dict | None = None, strat_mask=None, cell_bias=None):
        super().__init__()
        for k, v in dict(strat_text=strat_text, strat_args=strat_args, lat_text=lat_text, lat_args=lat_args,
                         lon_text=lon_text, lon_args=lon_args, path_feat=path_feat, prof_feat=prof_feat).items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        self.register_buffer("path_cls", torch.as_tensor(path_cls, dtype=torch.long))
        self.register_buffer("prof_cls", torch.as_tensor(prof_cls, dtype=torch.long))
        n1 = len(strat_text)
        self.register_buffer("strat_mask", torch.as_tensor(strat_mask if strat_mask is not None else np.ones(n1, bool)))
        nl, no = len(lat_text), len(lon_text)
        self.register_buffer("cell_bias", torch.as_tensor(cell_bias if cell_bias is not None else np.zeros((nl, no), np.float32)))
        lq = logq or {}
        for k, n in (("s1", n1), ("s2lat", nl), ("s2lon", no), ("s3P", len(path_cls)), ("s3V", len(prof_cls))):
            self.register_buffer("logq_" + k, torch.as_tensor(lq.get(k, np.zeros(n)), dtype=torch.float32))


def node_embeddings(m: HiCAP, V: VocabTensors):
    return dict(s1=m.tower1(V.strat_text, V.strat_args), s2lat=m.tower_lat(V.lat_text, V.lat_args),
                s2lon=m.tower_lon(V.lon_text, V.lon_args), s3P=m.tower_path(V.path_feat, V.path_cls),
                s3V=m.tower_prof(V.prof_feat, V.prof_cls))


def cell_logits(m: HiCAP, s2lat, s2lon, A_lat, A_lon, bias_mask):
    """f2 = tau <s_lat,a_lat> + tau <s_lon,a_lon> + b(c_lat,c_lon);  [B, n_lat, n_lon]. bias_mask has -inf for never-occurring pairs."""
    zl = m.tau("s2lat") * s2lat @ A_lat.t(); zo = m.tau("s2lon") * s2lon @ A_lon.t()
    return zl.unsqueeze(2) + zo.unsqueeze(1) + (m.pair_bias + bias_mask).unsqueeze(0)


class HiCAPLoss(nn.Module):
    """Batch dict (labels): strat_y [B], lat_y [B], lon_y [B] (-100 ignore), yP [B,K_P], yQ [B,K_V] (soft, from distances),
    goal_y/goal_w [B,22], goal_pt [B,3,3] + goal_pt_w [B,3], optional cand_p/cand_q [B,K], cost_t [B,K,3], refine_t [B,K,16]."""
    def __init__(self, beta: dict | None = None, excl_pairs=None, sigma_smooth=0.0):
        super().__init__(); self.beta = beta or {}; self.excl = excl_pairs; self.smooth = sigma_smooth

    def forward(self, m: HiCAP, V: VocabTensors, batch: dict, stage: int = 1, width: int | None = None):
        st = STAGES[stage]; lam = st.lam
        s, g_logits, g_pt, scene = m.encode(batch, width)
        A = node_embeddings(m, V)
        if width is not None: A = {k: F.normalize(v[..., :width], dim=-1) for k, v in A.items()}
        parts = {}
        # Level 1 (strategic) -- label, masked for empirically absent classes
        y1, ig1 = hard_to_soft(batch["strat_y"], A["s1"].shape[0], self.smooth)
        z1 = level_logits(s["s1"], A["s1"], m.tau("s1"), V.logq_s1, self.beta.get("s1", 0.0), V.strat_mask.unsqueeze(0))
        parts["L1"] = level_loss(z1, y1, ig1)
        # Level 2 -- factored marginals + joint cell
        yl, igl = hard_to_soft(batch["lat_y"], A["s2lat"].shape[0], self.smooth)
        yo, igo = hard_to_soft(batch["lon_y"], A["s2lon"].shape[0], self.smooth)
        zl = level_logits(s["s2lat"], A["s2lat"], m.tau("s2lat"), V.logq_s2lat, self.beta.get("s2lat", 0.0))
        zo = level_logits(s["s2lon"], A["s2lon"], m.tau("s2lon"), V.logq_s2lon, self.beta.get("s2lon", 0.0))
        parts["L2lat"] = level_loss(zl, yl, igl); parts["L2lon"] = level_loss(zo, yo, igo)
        zc = cell_logits(m, s["s2lat"], s["s2lon"], A["s2lat"], A["s2lon"], V.cell_bias)
        ok = ~(igl | igo)
        if ok.any():
            tgt = (batch["lat_y"].clamp(min=0) * zc.shape[2] + batch["lon_y"].clamp(min=0))
            parts["L2cell"] = F.cross_entropy(zc.flatten(1)[ok], tgt[ok])
        else:
            parts["L2cell"] = zc.sum() * 0.0
        # Level 3 -- factored path / profile critics with LEVEL-WIDE negatives and soft targets
        zP = level_logits(s["s3P"], A["s3P"], m.tau("s3P"), V.logq_s3P, self.beta.get("s3P", 0.0))
        zQ = level_logits(s["s3V"], A["s3V"], m.tau("s3V"), V.logq_s3V, self.beta.get("s3V", 0.0))
        zPt, zQt = zP, zQ
        if st.hard_negatives and "hardP" in batch:                              # stage 2+: near-miss negatives get extra weight in the normaliser
            zPt = zP + math.log(HARD_W) * batch["hardP"]; zQt = zQ + math.log(HARD_W) * batch["hardQ"]
        parts["L3P"] = level_loss(zPt, batch["yP"]); parts["L3V"] = level_loss(zQt, batch["yQ"])
        voc = sum(parts.values())
        total = lam["voc"] * voc
        # consistency: cascade marginal p(class) * p(path | class) of the path factor vs the level-wide critic (stop-grad on the cascade)
        if lam["cons"] > 0:
            cls = V.path_cls
            p_cls = F.softmax(zl, -1)[:, cls]                                   # [B,K_P] lateral-class mass of each path's class
            same = (cls.unsqueeze(0) == cls.unsqueeze(1)).float()                # [K_P,K_P]
            zc_ = zP.unsqueeze(1) + torch.log(same.unsqueeze(0).clamp(min=1e-30))  # [B,K_P(row=path's class),K_P]
            p_in = F.softmax(zc_, -1).diagonal(dim1=1, dim2=2)                   # p(path | class(path))
            p_cascade = (p_cls * p_in)
            p_cascade = p_cascade / p_cascade.sum(-1, keepdim=True).clamp(min=1e-9)
            parts["cons"] = consistency_kl(p_cascade, zP)
            total = total + lam["cons"] * parts["cons"]
        if lam["goal"] > 0:
            parts["goal"] = goal_loss(g_logits, batch["goal_y"], batch["goal_w"], self.excl)
            if "goal_pt" in batch:
                parts["goal_pt"] = (F.smooth_l1_loss(g_pt, batch["goal_pt"], reduction="none").mean(-1) * batch["goal_pt_w"]).sum() / batch["goal_pt_w"].sum().clamp(min=1)
            total = total + lam["goal"] * (parts["goal"] + parts.get("goal_pt", 0.0))
        if lam["cost"] > 0 and "cand_p" in batch and "cost_t" in batch:     # needs the agent-track join (not in this dataset yet)
            ap = A["s3P"][batch["cand_p"]]; aq = A["s3V"][batch["cand_q"]]
            sp = (s["s3P"].unsqueeze(1) * s["s3V"].unsqueeze(1))
            feat = sp * ap * aq
            pred = torch.stack([h(feat).squeeze(-1) for h in m.cost], -1)
            parts["cost"] = F.smooth_l1_loss(pred, batch["cost_t"])
            total = total + lam["cost"] * parts["cost"]
        if lam["ref"] > 0 and "cand_p" in batch and "refine_t" in batch:
            ap = A["s3P"][batch["cand_p"]]; aq = A["s3V"][batch["cand_q"]]
            r = m.refine(torch.cat([s["s3P"].unsqueeze(1).expand_as(ap) * ap, s["s3V"].unsqueeze(1).expand_as(aq) * aq], -1))
            parts["ref"] = F.l1_loss(r, batch["refine_t"])
            total = total + lam["ref"] * parts["ref"]
        return total, parts
