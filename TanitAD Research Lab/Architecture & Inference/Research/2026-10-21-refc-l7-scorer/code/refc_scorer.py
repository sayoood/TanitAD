"""L7 -- a learned, disentangled per-candidate SCORER / SELECTOR for refc's own fan (scorer9, 2026-10-21).

Design: ``../DESIGN.md``.  Pre-registration draft: ``../PREREG_L7_SCORER_DRAFT.md``.

WHAT IT IS
    REFe's / DrivoR's scoring decoder re-pointed at refc's 149-candidate fan.  A candidate enters ONLY as its re-embedded
    waypoints (``refcv7_heads.trajectory_features``, imported unchanged), passed DETACHED, so the scorer can neither read
    the generator's latent nor train the generator.  Three nested arms share this one module:

      mode "E"  geometry + v0                                   the ego-scalar-only FLOOR (no vision at all)
      mode "F"  E + the model's own per-candidate outputs        cheap: needs no scene tokens in the bank
      mode "S"  E + cross-attention to DETACHED scene tokens     REFe / DrivoR form; ``refcv7_heads.SceneMemory`` +
                                                                 ``refcv7_heads._Decoder`` imported unchanged, NO
                                                                 cross-candidate self-attention by default

    Heads are per benchmark (``PAI_HEADS`` / ``NAVSIM_HEADS``); targets are built OUTSIDE this file
    (``l7_targets_pai.py`` reuses ``refcv7_oracle``; ``l7_targets_navsim.py`` reuses REFe's ``navsim_pdm_targets``).

ADMISSIBILITY (DESIGN sec. 4) IS ENFORCED HERE, NOT DOCUMENTED
    * :func:`assemble_inputs` is the ONLY path from a record to the scorer; it REFUSES every key in
      :data:`FORBIDDEN_INPUT_KEYS` (nav, the fed set-speed ceiling, z_tac / cond, GT, agent futures, maps, PDM values,
      tactical labels, anything situation-classifier) and every key it does not know.
    * :meth:`L7Scorer.forward` refuses a candidate tensor that carries grad (the stop-gradient is the design).

SELECTION
    :func:`select` reproduces ``refc_v3.e9_rank`` (``refc_v3.py:1025-1053``) on the masks it is given -- reach, then the
    set-speed ceiling with the reach-only fallback, then the emit mask with its fallback -- and takes the argmax.  Ties
    resolve to the LOWEST index (``torch.argmax`` returns the first maximal value), so a constant scorer emits the lowest
    emittable candidate: control K1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: refc's waypoint grid in seconds (``config.horizons`` [5,10,15,20,30,40,50,60] ticks at 0.1 s)
SLOT_T_REFC: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
#: refc's base fan size (117 anchors); candidates >= this index are refcv8's allocated extras
N_BASE_REFC: int = 117

PAI_HEADS: tuple[str, ...] = ("imit", "along", "cross", "nc", "ttc", "dac", "comf", "ep")
PAI_REG_HEADS: tuple[str, ...] = ("along", "cross")
NAVSIM_HEADS: tuple[str, ...] = ("nc", "dac", "ep", "ttc", "c", "ddc")     # REFe train.ScorerBank.COMPONENTS order

#: the model-output features mode "F" may read (each is a function of the model's heads at inference)
MODEL_FEATURES: tuple[str, ...] = ("sel_z", "ref_z", "sel_logp", "sel_rank", "is_alloc", "mbr_logrisk", "in_top32")
OPTIONAL_MODEL_FEATURES: tuple[str, ...] = ("r8_sub0", "r8_sub1", "r8_sub2", "r8_sub3")

#: everything a scorer input may be assembled from
ALLOWED_INPUT_KEYS: frozenset = frozenset({
    "traj", "anchor_traj", "v0",
    "sel_score", "refined_logits", "r8_sub_logits", "reach_keep", "ceil_keep", "r8_emit_keep",
    "bev_tokens", "bev_pad", "agent_tokens", "agent_pad",
})
#: never an input (DESIGN sec. 4).  A label, a route-/future-derived channel, or a mixture containing one.
FORBIDDEN_INPUT_KEYS: frozenset = frozenset({
    # route / command: on PhysicalAI derived from the ego's own logged future
    "nav", "nav_cmd", "nav_cmd_sel", "nav_args", "r8_nav", "rc", "route", "goal_point", "goal_point_tac", "gp_point",
    # the fed set-speed ceiling: on PhysicalAI the max of the ego's ACTUAL future speed (max_speed_input.py:12-21)
    "v_max_ms", "v_max_valid", "v_lim", "vmax", "max_speed", "max_speed_ms", "v_limit_ms", "gt_max_speed_6s",
    # mixtures that contain nav / the ceiling / ego
    "z_tac", "cond", "r7_cond", "g_str", "g_tac", "tac_goal_logits", "r8_phi",
    # labels
    "g", "gt", "gt_valid", "gt_future", "r7_agent_fut", "r7_agent_fut_valid", "r7_agent_fut_label", "agents_fut",
    "drivable", "sam3", "map_gt", "pdm", "pdms", "tac_goal_y", "tac_goal_w", "lat_v7", "lon_v7",
    # steering-derived kinematics (NOT IMU): config.ego derives curvature / yaw rate from tan(steer)
    "curvature", "kappa0", "yaw_rate_steer",
    # situation classifier, in any form
    "situation", "sit_logits", "sit_probs", "situation_logits",
})


class InadmissibleInput(ValueError):
    """Raised when a forbidden or unknown key is offered to the scorer."""


def assert_admissible(keys: Sequence[str]) -> None:
    bad = sorted(k for k in keys if k in FORBIDDEN_INPUT_KEYS)
    if bad:
        raise InadmissibleInput(f"[L7] FORBIDDEN scorer input(s) {bad} (DESIGN sec. 4: route-/future-derived or a label)")
    unknown = sorted(k for k in keys if k not in ALLOWED_INPUT_KEYS)
    if unknown:
        raise InadmissibleInput(f"[L7] UNKNOWN scorer input(s) {unknown}: declare them in ALLOWED_INPUT_KEYS with their "
                                f"provenance before use")


# ---------------------------------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------------------------------
def _row_z(x: Tensor, keep: Tensor | None) -> Tensor:
    """Per-row z-score over the finite (and kept) entries; non-finite / dropped entries -> 0."""
    ok = torch.isfinite(x)
    if keep is not None:
        ok = ok & keep
    xf = torch.where(ok, x, torch.zeros_like(x))
    n = ok.sum(-1, keepdim=True).clamp_min(1)
    mu = xf.sum(-1, keepdim=True) / n
    var = (torch.where(ok, (x - mu) ** 2, torch.zeros_like(x))).sum(-1, keepdim=True) / n
    return torch.where(ok, (x - mu) / var.sqrt().clamp_min(1e-6), torch.zeros_like(x))


def mbr_risk(fan2s: Tensor, score: Tensor, keep: Tensor | None, k: int = 32, t: float = 1.0) -> tuple[Tensor, Tensor]:
    """The frozen MBR32 picker's statistic for EVERY candidate.

    ``fan_mbr_x8.mbr_pick`` (sha256 94cef175..., read 2026-10-21): take the top-``k`` candidates by ``score``, weight them
    p = softmax(score / t) over that set, and pick argmin over the set of risk_c = sum_j p_j * ADE(c, j) on the 2 s grid.
    Here the risk is computed for all N candidates against the same weighted top-k set, and ``in_top`` marks the set, so
    the MBR32 pick == argmin of risk over ``in_top`` -- the rule lies INSIDE mode F's hypothesis class.
    ``fan2s`` [B, N, 4, 2]; returns (risk [B, N] metres, in_top [B, N] bool)."""
    s = score.clone()
    s = torch.where(torch.isfinite(s), s, torch.full_like(s, -1e30))
    if keep is not None:
        s = torch.where(keep, s, torch.full_like(s, -1e30))
    kk = min(int(k), s.shape[1])
    top = torch.topk(s, kk, dim=1, sorted=True).indices                         # [B, k]
    c = torch.gather(fan2s, 1, top[:, :, None, None].expand(-1, -1, *fan2s.shape[2:]))   # [B, k, 4, 2]
    d = (fan2s[:, :, None] - c[:, None]).norm(dim=-1).mean(-1)                   # [B, N, k]
    lg = torch.gather(s, 1, top).double() / float(t)
    p = torch.softmax(lg - lg.max(1, keepdim=True).values, dim=1).to(d.dtype)
    risk = (d * p[:, None, :]).sum(-1)
    in_top = torch.zeros_like(s, dtype=torch.bool).scatter_(1, top, True)
    return risk, in_top


def model_features(fan: Tensor, sel_score: Tensor, refined_logits: Tensor, keep: Tensor | None = None,
                   r8_sub_logits: Tensor | None = None, n_base: int = N_BASE_REFC) -> Tensor:
    """Mode F's per-candidate model-output features [B, N, len(MODEL_FEATURES) (+4)].  All GT-free."""
    B, N = sel_score.shape
    s = torch.where(torch.isfinite(sel_score), sel_score, torch.full_like(sel_score, -1e30))
    sk = s if keep is None else torch.where(keep, s, torch.full_like(s, -1e30))
    logp = torch.log_softmax(sk.double(), dim=1).to(sel_score.dtype).clamp_min(-50.0)
    rank = torch.argsort(torch.argsort(-sk, dim=1, stable=True), dim=1).to(sel_score.dtype) / max(N - 1, 1)
    is_alloc = (torch.arange(N, device=sel_score.device) >= int(n_base)).to(sel_score.dtype)[None].expand(B, N)
    risk, in_top = mbr_risk(fan[:, :, :4], sel_score, keep, 32, 1.0)
    feats = [_row_z(sel_score, keep), _row_z(refined_logits, keep), logp / 10.0, rank, is_alloc,
             torch.log1p(risk / 0.5), in_top.to(sel_score.dtype)]
    if r8_sub_logits is not None:
        feats += [r8_sub_logits[..., j].clamp(-20, 20) / 5.0 for j in range(r8_sub_logits.shape[-1])]
    return torch.stack(feats, -1)


def assemble_inputs(rec: Mapping[str, Tensor], mode: str, use_r8_sub: bool = False) -> dict:
    """The ONLY path from a record (a dict of tensors for B windows) to :meth:`L7Scorer.forward` kwargs."""
    assert_admissible(list(rec.keys()))
    fan = rec["anchor_traj"] if "anchor_traj" in rec else rec["traj"]
    keep = rec.get("reach_keep")
    keep = None if keep is None else keep.to(torch.bool)
    out = {"traj": fan.detach(), "v0": rec["v0"]}
    if mode == "F":
        out["model_feats"] = model_features(fan.detach(), rec["sel_score"], rec["refined_logits"], keep,
                                            rec.get("r8_sub_logits") if use_r8_sub else None)
    if mode == "S":
        out["scene"] = {k: rec[k + "_tokens"].detach() for k in ("bev", "agent") if (k + "_tokens") in rec}
        out["scene_pads"] = {k: rec.get(k + "_pad") for k in out["scene"]}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the module
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class L7Config:
    mode: str = "F"                         # "E" | "F" | "S"
    heads: tuple[str, ...] = PAI_HEADS
    d_model: int = 128
    n_layers: int = 2                       # mode S decoder layers; modes E / F use an MLP trunk of this depth
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0
    self_attn: bool = False                 # REFe's set attention is a registered SECONDARY arm only
    slot_t: tuple[float, ...] = SLOT_T_REFC
    n_model_feats: int = len(MODEL_FEATURES)
    source_dims: dict = field(default_factory=dict)   # mode S: {"bev": 96, "agent": 256} read from the capture

    def __post_init__(self):
        if self.mode not in ("E", "F", "S"):
            raise ValueError(f"L7Config.mode must be E, F or S, got {self.mode!r}")
        if self.mode == "S" and not self.source_dims:
            raise ValueError("mode S needs source_dims (the widths of the detached scene sources)")


class L7Scorer(nn.Module):
    def __init__(self, cfg: L7Config):
        super().__init__()
        from tanitad.refs import refcv7_heads as H        # imported unchanged (trajectory_features, SceneMemory, _Decoder)
        self._H = H
        self.cfg = cfg
        d = cfg.d_model
        S = len(cfg.slot_t)
        n_in = S * 4 + 1 + (cfg.n_model_feats if cfg.mode == "F" else 0)
        self.embed = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        if cfg.mode == "S":
            hc = H.Refcv7HeadConfig(d_model=d, n_layers=cfg.n_layers, n_heads=cfg.n_heads, ffn_mult=cfg.ffn_mult,
                                    dropout=cfg.dropout, n_slots=S, scorer_self_attn=cfg.self_attn)
            self.memory = H.SceneMemory(dict(cfg.source_dims), d)
            self.trunk = H._Decoder(hc, self_attn=cfg.self_attn)
        else:
            self.memory = None
            layers = []
            for _ in range(cfg.n_layers):
                layers += [nn.LayerNorm(d), nn.Linear(d, d * cfg.ffn_mult // 2), nn.GELU(),
                           nn.Linear(d * cfg.ffn_mult // 2, d)]
            self.trunk = nn.ModuleList([nn.Sequential(*layers[i:i + 4]) for i in range(0, len(layers), 4)])
            self.out_norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({k: nn.Linear(d, 1) for k in cfg.heads})
        self.register_buffer("slot_t", torch.tensor(cfg.slot_t, dtype=torch.float32), persistent=False)

    def forward(self, traj: Tensor, v0: Tensor, model_feats: Tensor | None = None,
                scene: dict | None = None, scene_pads: dict | None = None) -> dict[str, Tensor]:
        if traj.requires_grad and torch.is_grad_enabled():
            raise ValueError("[L7] pass the candidates DETACHED (stop-gradient to the generator is the design)")
        x = self._H.trajectory_features(traj.to(torch.float32), self.slot_t, v0.to(torch.float32))
        if self.cfg.mode == "F":
            if model_feats is None:
                raise ValueError("[L7] mode F needs model_feats (use assemble_inputs)")
            x = torch.cat([x, model_feats.to(x.dtype)], -1)
        elif model_feats is not None:
            raise ValueError(f"[L7] mode {self.cfg.mode} must not receive model_feats (the floor would read the model)")
        h = self.embed(x)
        if self.cfg.mode == "S":
            if not scene:
                raise ValueError("[L7] mode S needs the detached scene sources")
            if any(t.requires_grad and torch.is_grad_enabled() for t in scene.values()):
                raise ValueError("[L7] pass the scene tokens DETACHED")
            mem, kp = self.memory(scene, scene_pads)
            h = self.trunk(h, mem, kp)
        else:
            if scene:
                raise ValueError(f"[L7] mode {self.cfg.mode} must not receive scene tokens")
            for blk in self.trunk:
                h = h + blk(h)
            h = self.out_norm(h)
        return {k: self.heads[k](h).squeeze(-1) for k in self.cfg.heads}


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


# ---------------------------------------------------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------------------------------------------------
def soft_target(cost: Tensor, keep: Tensor | None, temperature: float) -> Tensor:
    """p* proportional to exp(-cost / T) over the kept, finite-cost candidates (Hydra-MDP Eq 8 / TNT psi)."""
    ok = torch.isfinite(cost)
    if keep is not None:
        ok = ok & keep
    c = torch.where(ok, cost, torch.full_like(cost, float("inf")))
    return torch.softmax(-c / float(temperature), dim=-1)


def listwise_imitation_loss(logit: Tensor, cost: Tensor, keep: Tensor | None, temperature: float = 0.5) -> Tensor:
    """-sum_k p*_k log softmax(logit)_k over the kept set; rows with no finite cost drop out."""
    ok = torch.isfinite(cost)
    if keep is not None:
        ok = ok & keep
    rows = ok.any(-1)
    if not bool(rows.any()):
        return logit.sum() * 0.0
    with torch.no_grad():
        tgt = soft_target(cost, ok, temperature)
    lp = torch.log_softmax(logit.masked_fill(~ok, float("-inf")), dim=-1)
    per = -(tgt * lp.masked_fill(~ok, 0.0)).sum(-1)
    return per[rows].mean()


def regression_loss(pred: Tensor, err_m: Tensor, mask: Tensor | None = None, scale_m: float = 0.5,
                    beta: float = 0.1) -> Tensor:
    """Huber on log1p(err / scale): the along / cross auxiliary heads."""
    y = torch.log1p(err_m.clamp_min(0) / float(scale_m))
    ok = torch.isfinite(y)
    if mask is not None:
        ok = ok & mask
    if not bool(ok.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[ok], y[ok], beta=float(beta))


def bce_heads_loss(out: Mapping[str, Tensor], targets: Mapping[str, Tensor], names: Sequence[str],
                   keep: Tensor | None = None) -> tuple[Tensor, dict]:
    """BCE-with-logits per sub-score head on its own mask (REFe train.py:1732-1735 / refcv7_heads.scorer_loss form):
    ``targets[name]`` in [0, 1] (soft allowed), ``targets[name + '_mask']`` bool.  Unweighted mean over the heads present."""
    tot, log, n = None, {}, 0
    for k in names:
        if k not in out or k not in targets:
            continue
        m = targets.get(k + "_mask")
        m = torch.ones_like(targets[k], dtype=torch.bool) if m is None else m.to(torch.bool)
        if keep is not None:
            m = m & keep
        if not bool(m.any()):
            log[k] = float("nan")
            continue
        l = F.binary_cross_entropy_with_logits(out[k][m], targets[k][m].to(out[k].dtype))
        log[k] = float(l.detach())
        tot = l if tot is None else tot + l
        n += 1
    if tot is None:
        tot = sum(v.sum() for v in out.values()) * 0.0
        n = 1
    return tot / n, log


def pai_loss(out: Mapping[str, Tensor], tg: Mapping[str, Tensor], keep: Tensor | None, temperature: float = 0.5,
             w_imit: float = 1.0, w_reg: float = 0.5, w_bce: float = 1.0) -> tuple[Tensor, dict]:
    li = listwise_imitation_loss(out["imit"], tg["cost"], keep, temperature)
    lr = sum(regression_loss(out[k], tg[k], keep) for k in PAI_REG_HEADS if k in out and k in tg)
    lb, lg = bce_heads_loss(out, tg, [h for h in PAI_HEADS if h not in ("imit",) + PAI_REG_HEADS], keep)
    tot = w_imit * li + w_reg * lr + w_bce * lb
    lg.update(imit=float(li.detach()), reg=float(torch.as_tensor(lr).detach()), total=float(tot.detach()))
    return tot, lg


# ---------------------------------------------------------------------------------------------------------------------
# selection rules + the e9_rank masks
# ---------------------------------------------------------------------------------------------------------------------
def rule_imit(out: Mapping[str, Tensor]) -> Tensor:
    """R-IMIT (primary, PhysicalAI)."""
    return out["imit"]


def rule_safe(out: Mapping[str, Tensor], keep: Tensor | None = None) -> Tensor:
    """R-SAFE (secondary): log softmax(imit) + log sigma(nc) + log sigma(dac) -- PDMS-shaped gates on the imitation term."""
    lp = torch.log_softmax(out["imit"] if keep is None else out["imit"].masked_fill(~keep, -1e9), dim=-1)
    s = lp
    for g in ("nc", "dac"):
        if g in out:
            s = s + F.logsigmoid(out[g])
    return s


#: NAVSIM v1 PDMS weights, as REFe planner.V1_W (planner.py:520): (EP, TTC, C)
V1_W: tuple[float, float, float] = (5.0, 5.0, 2.0)


def rule_navsim_v1(out: Mapping[str, Tensor]) -> Tensor:
    """REFe ``REFePlanner.aggregate`` rule ``navsim_v1`` (planner.py:556-560), restated over this module's head names:
    sigma(NC) * sigma(DAC) * (5 sigma(EP) + 5 sigma(TTC) + 2 sigma(C)) / 12.  DDC does not enter (weight 0 in v1)."""
    p = {k: torch.sigmoid(out[k]) for k in ("nc", "dac", "ep", "ttc", "c")}
    return p["nc"] * p["dac"] * (V1_W[0] * p["ep"] + V1_W[1] * p["ttc"] + V1_W[2] * p["c"]) / sum(V1_W)


def e9_masked(score: Tensor, reach_keep: Tensor | None = None, ceil_keep: Tensor | None = None,
              emit_keep: Tensor | None = None) -> Tensor:
    """``refc_v3.e9_rank`` (refc_v3.py:1025-1053), statement for statement, on an arbitrary score."""
    rank = score
    if reach_keep is not None:
        rank = score.masked_fill(~reach_keep.to(torch.bool), float("-inf"))
    if ceil_keep is not None:
        both = rank.masked_fill(~ceil_keep.to(torch.bool), float("-inf"))
        dead = ~torch.isfinite(both).any(dim=1)
        rank = torch.where(dead[:, None], rank, both)
    if emit_keep is not None:
        ek = emit_keep.to(torch.bool)
        r2 = rank.masked_fill(~ek, float("-inf"))
        dead = ~torch.isfinite(r2).any(dim=1)
        rank = torch.where(dead[:, None], score.masked_fill(~ek, float("-inf")), r2)
    return rank


def select(score: Tensor, reach_keep: Tensor | None = None, ceil_keep: Tensor | None = None,
           emit_keep: Tensor | None = None) -> Tensor:
    """argmax of the e9-masked score; ties -> the LOWEST index (torch.argmax returns the first maximal value)."""
    return e9_masked(score, reach_keep, ceil_keep, emit_keep).argmax(dim=1)


def load_scorer(path: str, map_location: str = "cpu") -> L7Scorer:
    """A trained scorer from ``train_l7.py``'s ``best.pt`` ({"cfg": ..., "state_dict": ...}); eval mode, frozen."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(ck["cfg"])
    cfg["heads"] = tuple(cfg["heads"])
    cfg["slot_t"] = tuple(cfg["slot_t"])
    m = L7Scorer(L7Config(**cfg))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def attach(model: nn.Module, path: str, select_live: bool = True, rule: str = "imit") -> L7Scorer:
    """Attach a trained L7 scorer to a ``RefCV3Model`` (integration.diff reads ``model.l7_scorer`` / ``l7_select`` /
    ``l7_rule``).  ``select_live=False`` scores and exports ``sel_idx_l7`` without changing the emitted plan."""
    if rule not in ("imit", "safe"):
        raise ValueError(f"[L7] rule must be 'imit' or 'safe', got {rule!r}")
    sc = load_scorer(path, map_location=str(next(model.parameters()).device))
    model.l7_scorer = sc
    model.l7_select = bool(select_live)
    model.l7_rule = rule
    return sc


if __name__ == "__main__":       # --params: print the parameter count of each arm at the registered widths
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", action="store_true")
    ap.add_argument("--bev-dim", type=int, default=96)
    ap.add_argument("--agent-dim", type=int, default=256)
    a = ap.parse_args()
    rep = {}
    for mode in ("E", "F", "S"):
        cfg = L7Config(mode=mode, source_dims=({"bev": a.bev_dim, "agent": a.agent_dim} if mode == "S" else {}))
        rep[f"L7-{mode} (PAI heads)"] = n_params(L7Scorer(cfg))
    cfg = L7Config(mode="S", heads=NAVSIM_HEADS, source_dims={"bev": a.bev_dim, "agent": a.agent_dim})
    rep["L7-NS (NAVSIM heads)"] = n_params(L7Scorer(cfg))
    print(json.dumps(rep, indent=1))
