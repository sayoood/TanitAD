"""The complete HiCAP: core (multi-hierarchy + CLM-style critics) + HiCAP-I imagination heads (E-head, Tier 2) + elastic sensing
(camera gate, configuration embedding) + v10 knowledge (country readout on s1, per-country traffic-rule retrieval and a zero-init gated
memory read into s1/s2; paper sec:rules). `FullLoss` adds the terms of the paper's total loss:
L = sum_l lam_l L^voc_l + lam_c L^cons + lam_g L^goal + lam_k L^cost + lam_r L^ref + lam_i L^img + lam_a L^aud + lam_f L^cfg.
+ lam_country L^country (v10). Each extra term is active only when its supervision is present in the batch (agent tracks, counterfactual
costs, per-option errors, country labels). Knowledge is active only after `set_knowledge` (cached frozen-text embeddings of the rules and
country nodes); before that, and with the memory gate at 0, the model is bit-exactly the one without it."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .model import HiCAP, HiCAPConfig, LEVELS
from .losses import HiCAPLoss, STAGES, node_embeddings, VocabTensors
from .imagination import EHead, Tier2, exo_loss, tier2_loss, tier0_target
from .elastic import CameraGate, ConfigEmbedding, gate_loss, cfg_loss, N_OPT
from .rules import RuleBook, RuleMemory, GatedMemory, CountryReadout
from .audio import AudioBridge, BridgeConfig

LAM_COUNTRY = 0.1                                                     # paper sec:rules, lambda_c; active from stage 1 (a strategic critic)


class HiCAPFull(HiCAP):
    def __init__(self, c: HiCAPConfig | None = None, book: RuleBook | None = None, bridge: BridgeConfig | None = None):
        super().__init__(c)
        d = self.c.d
        self.ehead = EHead(d); self.tier2 = Tier2(d, 4); self.camera_gate = CameraGate(d); self.cfg_emb = ConfigEmbedding(d)
        self.cfg_proj = torch.nn.Linear(d, d)
        # v10 knowledge: rule/country embeddings live in the frozen backbone's text space (width d_text); set by set_knowledge
        self.book = book or RuleBook.load()
        self.country_readout = CountryReadout(d, self.c.d_text)
        self.rule_memory = RuleMemory(d, self.c.d_text, self.book.membership())
        self.gated_memory = GatedMemory(d, self.c.d_text, self.c.heads, LEVELS)
        self.register_buffer("rule_emb", torch.zeros(0, self.c.d_text), persistent=False)
        self.register_buffer("country_emb", torch.zeros(0, self.c.d_text), persistent=False)
        # v10 audio (optional): frozen-tower features -> trainable bridge -> 8 tokens in the backbone's text space -> GatedAudio
        self.audio_bridge = None
        if bridge is not None:
            if not (bridge.d_v == self.c.d_v == self.c.d_audio):
                raise ValueError(f"audio bridge needs d_v == HiCAPConfig.d_v == d_audio (got {bridge.d_v}, {self.c.d_v}, {self.c.d_audio})")
            self.audio_bridge = AudioBridge(bridge)

    def set_knowledge(self, rule_emb, country_emb):
        """Cached frozen-text-path embeddings of RuleBook.texts() [R,d_text] and RuleBook.country_nodes() [C,d_text] (pod job E33;
        tests use knowledge.hashed_text_embedding). Order must match the RuleBook version the model was built with."""
        rule_emb, country_emb = torch.as_tensor(rule_emb, dtype=torch.float32), torch.as_tensor(country_emb, dtype=torch.float32)
        assert rule_emb.shape[0] == len(self.book) and country_emb.shape[0] == self.rule_memory.M.shape[0]
        self.rule_emb, self.country_emb = rule_emb.to(self.rule_emb.device), country_emb.to(self.country_emb.device)

    @property
    def has_knowledge(self) -> bool: return self.rule_emb.shape[0] > 0

    def audio_params(self): return [] if self.audio_bridge is None else list(self.audio_bridge.parameters())

    def knowledge_params(self): return [p for n, p in self.named_parameters() if n.startswith(("country_readout", "rule_memory", "gated_memory"))]

    def encode_k(self, batch, width=None):
        """encode() plus the knowledge read: country posterior from the s1 query BEFORE memory (predicted arm, or one-hot of
        batch['country_supplied'] in the registered GNSS arm), top-k rule retrieval keyed on s2lat, gated memory into s1/s2 only
        (never the goal query). Returns (s, goal_logits, goal_pt, scene, k) with k = dict(country_logits, p_country, rules) or {}."""
        audio, ap = batch.get("audio"), batch.get("audio_present")
        if self.audio_bridge is not None and batch.get("audio_feat") is not None:
            if ap is None or bool(ap.any()):                       # no sample with audio -> branch skipped (bit-exact null)
                audio = self.audio_bridge(batch["audio_feat"], batch.get("audio_spatial"), batch.get("audio_pad"))
        q, scene = self.fusion(batch["vlm"], batch["last"], batch["age_s"], batch["frames"], batch["ego"], batch["nav"],
                               batch["nav_args"], batch["route"], batch.get("ego_present"), batch.get("nav_present"),
                               batch.get("route_present"), audio, ap)
        k = {}
        if self.has_knowledge:
            i1, i2 = LEVELS.index("s1"), LEVELS.index("s2lat")
            logits = self.country_readout(q[:, i1], self.country_emb)
            sup = batch.get("country_supplied")
            p = F.one_hot(sup.long(), self.country_emb.shape[0]).float() if sup is not None else torch.softmax(logits, -1)
            rules = self.rule_memory(q[:, i2], p, self.rule_emb)
            q = self.gated_memory(q, rules)
            k = dict(country_logits=logits, p_country=p, rules=rules)
        s = {kk: self.heads[kk](q[:, i], width) for i, kk in enumerate(LEVELS) if kk != "goal"}
        g = self.goal_head(q[:, LEVELS.index("goal")])
        return s, g[:, : self.c.n_goals], g[:, self.c.n_goals:].view(-1, 3, 3), scene, k

    def encode(self, batch, width=None): return self.encode_k(batch, width)[:4]

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
        s, _, _, _, kn = m.encode_k(batch, width); A = node_embeddings(m, V)
        # --- L^country (v10): contrastive readout of the strategic query over the country nodes; label only, never an input
        if kn and "country_y" in batch:
            parts["country"] = m.country_readout.loss(kn["country_logits"], batch["country_y"]); total = total + LAM_COUNTRY * parts["country"]
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
