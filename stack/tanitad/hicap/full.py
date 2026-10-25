"""The complete HiCAP: core (multi-hierarchy + CLM-style critics) + HiCAP-I imagination heads (E-head, Tier 2) + elastic sensing
(camera gate, configuration embedding). `FullLoss` adds the terms of the paper's total loss:
L = sum_l lam_l L^voc_l + lam_c L^cons + lam_g L^goal + lam_k L^cost + lam_r L^ref + lam_i L^img + lam_a L^aud + lam_f L^cfg.
Each extra term is active only when its supervision is present in the batch (agent tracks, counterfactual costs, per-option errors)."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .model import HiCAP, HiCAPConfig
from .losses import HiCAPLoss, STAGES, node_embeddings, VocabTensors
from .imagination import EHead, Tier2, exo_loss, tier2_loss, tier0_target
from .elastic import CameraGate, ConfigEmbedding, gate_loss, cfg_loss, N_OPT


class HiCAPFull(HiCAP):
    def __init__(self, c: HiCAPConfig | None = None):
        super().__init__(c)
        d = self.c.d
        self.ehead = EHead(d); self.tier2 = Tier2(d, 4); self.camera_gate = CameraGate(d); self.cfg_emb = ConfigEmbedding(d)
        self.cfg_proj = torch.nn.Linear(d, d)

    def imagination_params(self): return [p for n, p in self.named_parameters() if n.startswith(("ehead", "tier2"))]

    def encode_cfg(self, batch, cfg_opt, width=None):
        """Budget-conditioned readouts: the configuration embedding is added to every level query via the last token state (frozen
        backbone: the nesting lives in the trainable pooler/heads)."""
        b = dict(batch); b["last"] = batch["last"]
        s, g, gp, scene = self.encode(b, width)
        e = F.normalize(self.cfg_proj(self.cfg_emb(cfg_opt)), dim=-1) * 0.1
        return {k: F.normalize(v + e[..., : v.shape[-1]], dim=-1) for k, v in s.items()}, g, gp, scene


class FullLoss(HiCAPLoss):
    def forward(self, m: HiCAPFull, V: VocabTensors, batch: dict, stage: int = 1, width=None):
        total, parts = super().forward(m, V, batch, stage, width)
        lam = STAGES[stage].lam
        s, _, _, _ = m.encode(batch, width); A = node_embeddings(m, V)
        # --- L^img: exogenous descriptors (E-head reads the state only; no counterfactual labels needed)
        if lam["img"] > 0 and "exo_gap" in batch:
            out = m.ehead(s["s2lon"].detach())                       # trained separately from the selector: no gradient into it
            tgt = dict(gap=batch["exo_gap"], closing=batch["exo_close"], lacc=batch["exo_acc"], mode=batch["exo_mode"], onset=batch["exo_onset"],
                       has_lead=batch["exo_has_lead"], w=batch["exo_w"])
            parts["exo"] = exo_loss(out, tgt); total = total + lam["img"] * parts["exo"]
        # --- Tier 2 counterfactual heads (family-masked: G2)
        if lam["img"] > 0 and "cf_cost" in batch and "cand_p" in batch:
            v = A["s3P"][batch["cand_p"]] * A["s3V"][batch["cand_q"]]
            c = m.tier2(s["s3P"].detach(), v)
            parts["tier2"] = tier2_loss(c, batch["cf_cost"], batch["cf_mask"]); total = total + lam["img"] * parts["tier2"]
        # --- Tier 0: listwise soft outcome-aware target over the candidate set (training only, zero inference cost)
        if "J_teacher" in batch and "cand_p" in batch:
            ap, aq = A["s3P"][batch["cand_p"]], A["s3V"][batch["cand_q"]]
            z = m.tau("s3P") * (s["s3P"].unsqueeze(1) * ap).sum(-1) + m.tau("s3V") * (s["s3V"].unsqueeze(1) * aq).sum(-1)
            y = tier0_target(batch["J_teacher"], batch["star"])
            parts["tier0"] = -(y * F.log_softmax(z, -1)).sum(-1).mean(); total = total + parts["tier0"]
        # --- elastic: cost-sensitive gate on cached per-option errors, and L^cfg on reduced configurations
        if "E_opt" in batch:
            front = s["s2lon"]
            lg = m.camera_gate(front.detach(), batch["prev_opt"])
            parts["gate"] = gate_loss(lg, batch["E_opt"], batch["crit"]); total = total + parts["gate"]
        if lam["cfg"] > 0 and "cfg_opt" in batch and "lat_y" in batch:
            ok = batch["lat_y"] >= 0
            if ok.any():
                s_c, *_ = m.encode_cfg(batch, batch["cfg_opt"], width)
                parts["cfg"] = lam["cfg"] * cfg_loss(s_c["s2lat"][ok], s["s2lat"][ok], A["s2lat"], batch["lat_y"][ok]); total = total + parts["cfg"]
        return total, parts
