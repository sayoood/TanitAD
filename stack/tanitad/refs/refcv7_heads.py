"""refcv7 — DrivoR's two proven planning heads, adapted to the REF-C hierarchy.

Source: *Driving on Registers* (DrivoR, valeo.ai, CVPR 2026, arXiv 2601.05083, lib-banked),
read in full 2026-09-19. The ablations that make these two heads worth porting (navval
PDMS, one run each, so read as DIRECTION, not as effect size):

* 64 learned trajectory queries vs 1: 80.1 -> 90.0; one token per WHOLE trajectory vs one
  token per pose: 83.9 -> 90.0 (DrivoR Tab. 5, Tab. 12);
* a scorer in its OWN decoder, fed the decoded trajectory RE-EMBEDDED behind a
  STOP-GRADIENT, predicting SEPARATE sub-scores: shared 84.7 -> separate 86.8 ->
  disentangled 90.0; one total score 88.2 vs 6 sub-scores 90.0 (Tab. 6).

What refcv7 changes, and why (each is a deliberate deviation, not a transcription error):

1. **Condition.** DrivoR adds ego status (poses, velocities, accelerations, command) to every
   query. refcv7 adds the refcv6 CONDITION instead: nav one-hot (MANDATORY, PI 2026-09-16),
   the max-speed one-hot INPUT, v0 and the PAST-only ego-history embedding. Nothing from the
   future and no situation-classifier output may enter it (PI 2026-08-03 rulings).
2. **Waypoints.** The REF-C slot grid (V3_HORIZONS: 0.5 ... 6.0 s, NON-uniform), so WTA
   proposals, the 117 anchors and the diffusion fan are all directly comparable candidates.
3. **Scorer self-attention is OFF by default.** DrivoR's scorer is a vanilla decoder, so its
   candidates attend to each other and a candidate's score depends on its companions. refcv7
   uses the scorer as a test-time SEARCH reward (``refcv7_toad``), and a reward must be a
   function of the trajectory alone. ``scorer_self_attn=True`` restores DrivoR's form as an
   arm.
4. **Sub-scores** are refcv7's PhysicalAI oracle (``refcv7_oracle.SUBSCORES``), not NAVSIM's.

Both heads read the SAME scene tokens (shared trunk; DrivoR lets the scorer's gradient reach
the encoder too) through SEPARATE key/value projections.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .refcv7_oracle import SUBSCORES


@dataclass
class Refcv7HeadConfig:
    d_model: int = 256          # DrivoR: decoders d = 256, FFN x4, 4 layers
    n_layers: int = 4
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    n_queries: int = 64         # DrivoR Tab. 5: plateau at 64
    n_slots: int = 8
    cond_dim: int = 0           # width of the refcv6 condition vector (0 = none)
    v_scale_ms: float = 10.0    # per-slot displacement scale of the WTA head
    scorer_self_attn: bool = False
    subscores: tuple[str, ...] = SUBSCORES


class SceneMemory(nn.Module):
    """Project each scene-token source to ``d_model`` and tag it with a learned type
    embedding, then concatenate: ``[B, sum(N_i), d]``.

    Sources are passed BY NAME (e.g. ``img``, ``bev``, ``agents``) so a head can never
    silently read a different source than the one it was built for.
    """

    def __init__(self, source_dims: dict[str, int], d_model: int):
        super().__init__()
        self.names = tuple(sorted(source_dims))
        self.proj = nn.ModuleDict({k: nn.Linear(source_dims[k], d_model)
                                   for k in self.names})
        self.type_emb = nn.ParameterDict({k: nn.Parameter(torch.zeros(1, 1, d_model))
                                          for k in self.names})
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sources: dict[str, Tensor],
                pads: dict[str, Tensor | None] | None = None
                ) -> tuple[Tensor, Tensor | None]:
        """Returns ``(memory [B, N, d], key_pad [B, N] bool | None)``; ``pads[k]`` is
        True on PADDED tokens (the agent slots' convention), which attention ignores."""
        missing = [k for k in self.names if k not in sources]
        if missing:
            raise KeyError(f"scene sources missing: {missing} (built for {self.names})")
        parts = [self.proj[k](sources[k]) + self.type_emb[k] for k in self.names]
        mem = self.norm(torch.cat(parts, dim=1))
        pads = pads or {}
        if not any(pads.get(k) is not None for k in self.names):
            return mem, None
        masks = []
        for k in self.names:
            pk = pads.get(k)
            n = sources[k].shape[1]
            masks.append(pk.to(torch.bool) if pk is not None else
                         torch.zeros(mem.shape[0], n, dtype=torch.bool, device=mem.device))
        kp = torch.cat(masks, dim=1)
        # a row whose EVERY key is padded would make softmax NaN: unmask it (it then
        # attends to zero-content pads, which is the honest "no scene" reading)
        allpad = kp.all(dim=1, keepdim=True)
        return mem, kp & ~allpad


class _DecoderLayer(nn.Module):
    """Pre-norm decoder layer: [self-attn] -> cross-attn(scene) -> FFN."""

    def __init__(self, d: int, h: int, ffn: int, dropout: float, self_attn: bool):
        super().__init__()
        self.use_self = self_attn
        if self_attn:
            self.n0 = nn.LayerNorm(d)
            self.sa = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.ca = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, q: Tensor, mem: Tensor, key_pad: Tensor | None = None) -> Tensor:
        if self.use_self:
            x = self.n0(q)
            q = q + self.sa(x, x, x, need_weights=False)[0]
        x = self.n1(q)
        q = q + self.ca(x, mem, mem, key_padding_mask=key_pad, need_weights=False)[0]
        return q + self.ff(self.n2(q))


class _Decoder(nn.Module):
    def __init__(self, cfg: Refcv7HeadConfig, self_attn: bool):
        super().__init__()
        d = cfg.d_model
        self.layers = nn.ModuleList(_DecoderLayer(d, cfg.n_heads, d * cfg.ffn_mult,
                                                  cfg.dropout, self_attn)
                                    for _ in range(cfg.n_layers))
        self.out_norm = nn.LayerNorm(d)

    def forward(self, q: Tensor, mem: Tensor, key_pad: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            q = layer(q, mem, key_pad)
        return self.out_norm(q)


# ---------------------------------------------------------------------------
# 1. WTA proposal decoder
# ---------------------------------------------------------------------------
class WTAProposalDecoder(nn.Module):
    """DrivoR's trajectory decoder: ``n_queries`` learned queries, each decoded from ONE
    token into a whole trajectory of ``n_slots`` waypoints, trained winner-takes-all.

    Output parameterisation: per-slot displacements scaled by ``slot_dt * v_scale`` and
    cumulatively summed, so a fresh head starts near "standing still" rather than at
    random positions tens of metres away.
    """

    def __init__(self, cfg: Refcv7HeadConfig, source_dims: dict[str, int],
                 slot_t: tuple[float, ...]):
        super().__init__()
        if len(slot_t) != cfg.n_slots:
            raise ValueError(f"slot_t has {len(slot_t)} entries, n_slots={cfg.n_slots}")
        d = cfg.d_model
        self.cfg = cfg
        self.memory = SceneMemory(source_dims, d)
        # DrivoR initialises its queries N(0, 1e-6); a tiny std keeps them symmetric-broken
        self.queries = nn.Parameter(torch.randn(1, cfg.n_queries, d) * 1e-3)
        self.cond = nn.Linear(cfg.cond_dim, d) if cfg.cond_dim > 0 else None
        self.decoder = _Decoder(cfg, self_attn=True)   # queries DO see each other (DrivoR)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.n_slots * 2))
        t = torch.tensor((0.0,) + tuple(slot_t))
        self.register_buffer("slot_dt", (t[1:] - t[:-1]).float(), persistent=False)

    def forward(self, sources: dict[str, Tensor], cond: Tensor | None = None,
                pads: dict[str, Tensor | None] | None = None) -> Tensor:
        mem, kp = self.memory(sources, pads)
        B = mem.shape[0]
        q = self.queries.expand(B, -1, -1)
        if self.cond is not None:
            if cond is None:
                raise ValueError("this head was built with a condition; pass cond")
            q = q + self.cond(cond)[:, None, :]
        q = self.decoder(q, mem, kp)
        delta = self.head(q).view(B, self.cfg.n_queries, self.cfg.n_slots, 2)
        step = delta * (self.slot_dt * self.cfg.v_scale_ms)[None, None, :, None]
        return torch.cumsum(step, dim=2)                         # [B, Q, S, 2]


def wta_loss(proposals: Tensor, gt: Tensor, gt_valid: Tensor | None = None,
             extra_target: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Winner-takes-all L1 (DrivoR eq. 1/2): only the closest proposal is supervised.

    ``proposals`` [B, Q, S, 2], ``gt`` [B, S, 2], ``gt_valid`` [B, S].
    ``extra_target`` [B, S, 2] is DrivoR's optional "accelerated" second target (eq. 2) —
    OFF by default in refcv7: DrivoR measured it +0.6 on NAVSIM-v1 but -1.6 on the
    out-of-distribution NAVSIM-v2 warm-up (Tab. 7).
    Returns (scalar loss, winner index [B]).
    """
    if gt_valid is None:
        gt_valid = torch.ones(gt.shape[:2], dtype=torch.bool, device=gt.device)
    m = gt_valid[:, None, :, None].to(proposals.dtype)
    denom = m.sum(dim=(2, 3)).clamp_min(1.0)
    err = ((proposals - gt[:, None]).abs() * m).sum(dim=(2, 3)) / denom        # [B, Q]
    if extra_target is not None:
        err = err + ((proposals - extra_target[:, None]).abs() * m).sum(dim=(2, 3)) / denom
    best = err.argmin(dim=1)
    loss = err.gather(1, best[:, None]).squeeze(1)
    has_gt = gt_valid.any(dim=1)
    if not bool(has_gt.any()):
        return proposals.sum() * 0.0, best
    return loss[has_gt].mean(), best


# ---------------------------------------------------------------------------
# 2. disentangled scorer
# ---------------------------------------------------------------------------
def trajectory_features(traj: Tensor, slot_t: Tensor, v0: Tensor) -> Tensor:
    """What the scorer is allowed to know about a candidate: its waypoints plus the
    per-slot speed and heading implied by them (no latent from the generator).
    ``traj`` [B, M, S, 2] -> [B, M, S * 4 + 1]."""
    B, M, S, _ = traj.shape
    z = traj.new_zeros(B, M, 1, 2)
    d = torch.cat([z, traj], dim=2).diff(dim=2)
    t = torch.cat([slot_t.new_zeros(1), slot_t]).to(traj)
    v = d.norm(dim=-1) / (t[1:] - t[:-1]).clamp_min(1e-3)
    hd = torch.atan2(d[..., 1], d[..., 0])
    feats = torch.cat([traj / 30.0, (v / 15.0)[..., None], hd[..., None]], dim=-1)
    return torch.cat([feats.flatten(2), (v0.to(traj) / 15.0)[:, None, None].expand(B, M, 1)],
                     dim=-1)


class DisentangledScorer(nn.Module):
    """DrivoR's scoring decoder, refcv7-adapted.

    * The candidate enters ONLY as its re-embedded waypoints, and the caller must pass it
      DETACHED: the scorer can neither see the generator's latent nor train the generator
      (DrivoR's "disentanglement"). :meth:`forward` asserts this.
    * Its own scene projections, and the refcv6 condition added to every query, so the
      score is nav- and set-speed-aware (PI: selection must follow nav).
    * One logit per oracle sub-score.
    """

    def __init__(self, cfg: Refcv7HeadConfig, source_dims: dict[str, int],
                 slot_t: tuple[float, ...]):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.memory = SceneMemory(source_dims, d)
        n_feat = cfg.n_slots * 4 + 1
        self.embed = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Linear(d, d))
        self.cond = nn.Linear(cfg.cond_dim, d) if cfg.cond_dim > 0 else None
        self.decoder = _Decoder(cfg, self_attn=cfg.scorer_self_attn)
        self.heads = nn.ModuleDict({k: nn.Linear(d, 1) for k in cfg.subscores})
        self.register_buffer("slot_t", torch.tensor(slot_t, dtype=torch.float32),
                             persistent=False)

    def encode_scene(self, sources: dict[str, Tensor],
                     pads: dict[str, Tensor | None] | None = None
                     ) -> tuple[Tensor, Tensor | None]:
        """The scene memory (and its key-pad mask), computed ONCE per frame and reused
        by every score call — why test-time search costs ~2 ms, not a second forward
        (TOAD §4)."""
        return self.memory(sources, pads)

    def score(self, mem: tuple[Tensor, Tensor | None], traj: Tensor, v0: Tensor,
              cond: Tensor | None = None) -> dict[str, Tensor]:
        mem, kp = mem
        if traj.requires_grad and torch.is_grad_enabled():
            raise ValueError("DisentangledScorer: pass the candidates DETACHED "
                             "(stop-gradient to the generator is the design)")
        q = self.embed(trajectory_features(traj, self.slot_t, v0))
        if self.cond is not None:
            if cond is None:
                raise ValueError("this scorer was built with a condition; pass cond")
            q = q + self.cond(cond)[:, None, :]
        # Without self-attention every op in the decoder is per-query (cross-attention,
        # LayerNorm, FFN), so each candidate's score is already a function of that
        # candidate and the scene alone — asserted by test_refcv7_heads.py.
        h = self.decoder(q, mem, kp)
        return {k: self.heads[k](h).squeeze(-1) for k in self.cfg.subscores}

    def forward(self, sources: dict[str, Tensor], traj: Tensor, v0: Tensor,
                cond: Tensor | None = None,
                pads: dict[str, Tensor | None] | None = None) -> dict[str, Tensor]:
        return self.score(self.encode_scene(sources, pads), traj, v0, cond)


def scorer_loss(logits: dict[str, Tensor], oracle: dict[str, Tensor],
                weights: dict[str, float] | None = None) -> tuple[Tensor, dict[str, float]]:
    """BCE of each sub-score logit against the oracle value, on its mask only
    (DrivoR eq. 3, all lambda_c = 1 by default). Returns (loss, per-sub-score log)."""
    total = None
    log: dict[str, float] = {}
    for k, lg in logits.items():
        mask = oracle.get(k + "_mask")
        tgt = oracle[k].to(lg.dtype)
        if mask is None:
            mask = torch.ones_like(tgt, dtype=torch.bool)
        if not bool(mask.any()):
            log[k] = float("nan")
            continue
        l = F.binary_cross_entropy_with_logits(lg[mask], tgt[mask])
        log[k] = float(l.detach())
        w = 1.0 if weights is None else float(weights.get(k, 1.0))
        total = w * l if total is None else total + w * l
    if total is None:
        total = sum(lg.sum() for lg in logits.values()) * 0.0
    return total, log
