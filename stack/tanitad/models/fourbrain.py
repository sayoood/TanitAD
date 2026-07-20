"""The 4B stack composition (H1): operative + tactical + strategic + fallback.

Phase 0 scope:
- Operative: encoder -> spatial readout -> action-conditioned predictor (full).
- Tactical: discrete maneuver vocabulary, imagine-and-select — imagine each
  candidate maneuver's post-latent, decode with the imagination-calibrated
  probe, score against sub-goal + safety costs (full, minimal form).
- Strategic: latent transition graph over VQ-like codes (minimal: k-means
  centroids + observed-transition adjacency + Dijkstra). Port target from
  ALPS-4B; here a functional skeleton.
- Fallback (brain 4): imagination-error self-monitor (A9) + collapse watchdog;
  exposes alarms and an MRC hook. Runs out-of-gradient.

Layer frequencies (thinking fast/slow): operative every step, tactical every
N_tac steps, strategic every N_str steps — enforced by the caller/runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from tanitad.config import (StackConfig, StrategicPolicyConfig,
                            TacticalPolicyConfig)
from tanitad.models.encoder import ViTEncoder
from tanitad.models.imagination import ImaginationField
from tanitad.models.inverse_dynamics import InverseDynamicsHead
from tanitad.models.predictor import CausalBlock, OperativePredictor
from tanitad.models.readout import RidgeProbe, SpatialGridReadout
from tanitad.models.sigreg import SigReg


def _causal_mask(w: int, device) -> Tensor:
    return torch.triu(torch.ones(w, w, device=device, dtype=torch.bool),
                      diagonal=1)


class StrategicPolicy(nn.Module):
    """Trained strategic brain (D-030) — ports REF-B rev2 ``StrategicHead``.

    A causal transformer over the operative STATE window, FiLM-conditioned on a
    nav-command embedding it owns, that emits (a) a context token ``ctx`` [B,
    d_ctx] — the FiLM cond of the tactical brain — and (b) route-heading logits
    (route_left/straight/right) for the auxiliary route CE (direct supervision).

    Operates on the compact state, so it composes on the flagship's ViT+readout
    state and REF-A's frozen-DINO adapter state alike (shared brain). Owns
    ``nav_emb`` so it is self-contained: ``forward(states, nav_cmd_long)``.
    """

    def __init__(self, cfg: StrategicPolicyConfig, state_dim: int, window: int,
                 ego_input: bool = False):
        super().__init__()
        self.cfg, self.window = cfg, window
        d = cfg.d_model
        self.nav_emb = nn.Embedding(cfg.n_commands, cfg.d_cmd)
        # v2 lever 1: proprioceptive [v0, yr0] added to the nav embedding
        # (leakage-safe, t<=0 only; mirrors the operative v0 channel + the
        # refbpatch speed_emb pattern). None when off -> state_dict identical.
        self.ego_emb = nn.Linear(2, cfg.d_cmd) if ego_input else None
        self.in_proj = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            CausalBlock(d, cfg.n_heads, cond_dim=cfg.d_cmd)
            for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.ctx_proj = nn.Linear(d, cfg.d_ctx)
        self.route_head = nn.Linear(d, cfg.n_route)

    def forward(self, states: Tensor, nav_cmd: Tensor,
                ego: Tensor | None = None) -> dict[str, Tensor]:
        b, w, _ = states.shape
        assert w == self.window, f"window mismatch: {w} != {self.window}"
        cmd = self.nav_emb(nav_cmd)                            # [B, d_cmd]
        if self.ego_emb is not None and ego is not None:
            cmd = cmd + self.ego_emb(ego.to(cmd.dtype))        # [B,2]->[B,d_cmd]
        x = self.in_proj(states) + self.pos[:, :w]
        cond = cmd.unsqueeze(1).expand(-1, w, -1)              # [B, W, d_cmd]
        mask = _causal_mask(w, states.device)
        for blk in self.blocks:
            x = blk(x, cond, mask)
        h = self.norm(x[:, -1])
        return {"ctx": self.ctx_proj(h), "route_logits": self.route_head(h)}


# ============================================================================
# TIME-anchored tactical decoder (v2 lever 8) — DiffusionDrive-style multi-anchor
# ============================================================================
# The FPS anchor-vocabulary math mirrors tanitad.refs.refc.{synth_anchor_pool,
# furthest_point_sample, default_anchors} (the proven REF-C/REF-B recipe). It is
# duplicated here ON PURPOSE: refs/ is a SEPARATE package ("never entangled with
# the main model", refs/__init__ docstring), so the flagship's own tactical
# decoder must not import from a reference model. The functions are pure +
# deterministic, so the two copies yield byte-identical vocabularies for equal
# args (128 anchors over a 4096-rollout pool, seed 0).

def _synth_anchor_pool(horizons: tuple[int, ...], pool_size: int, seed: int,
                       dt: float = 0.1, device: str = "cpu") -> Tensor:
    """Random unicycle rollouts -> ego-frame trajectory pool [pool_size, S, 2]
    (ego starts at origin heading +x, so world positions ARE the ego waypoints)."""
    g = torch.Generator(device=device).manual_seed(seed)
    m = pool_size
    v = torch.rand(m, generator=g, device=device) * 30.0            # 0..30 m/s
    yaw_rate = (torch.rand(m, generator=g, device=device) - 0.5) * 0.7   # +-0.35
    accel = (torch.rand(m, generator=g, device=device) - 0.5) * 6.0     # +-3
    x = torch.zeros(m, device=device)
    y = torch.zeros(m, device=device)
    yaw = torch.zeros(m, device=device)
    max_h = max(horizons)
    pos = torch.zeros(m, max_h + 1, 2, device=device)
    for t in range(1, max_h + 1):
        x = x + v * torch.cos(yaw) * dt
        y = y + v * torch.sin(yaw) * dt
        yaw = yaw + yaw_rate * dt
        v = (v + accel * dt).clamp_min(0.0)
        pos[:, t, 0] = x
        pos[:, t, 1] = y
    return torch.stack([pos[:, h] for h in horizons], dim=1)        # [m, S, 2]


def _furthest_point_sample(pool: Tensor, n: int, seed: int = 0) -> Tensor:
    """Greedy FPS in flattened-L2 space [M, S, 2] -> [n, S, 2]: SPREADS the
    vocabulary over the trajectory manifold (covers the rare curves) instead of
    clustering on the straight majority the way k-means would. Deterministic."""
    m = pool.shape[0]
    if n > m:
        raise ValueError(f"cannot FPS {n} anchors from a pool of {m}")
    flat = pool.reshape(m, -1)
    g = torch.Generator(device=flat.device).manual_seed(seed)
    first = int(torch.randint(m, (1,), generator=g, device=flat.device))
    chosen = [first]
    dist = ((flat - flat[first]) ** 2).sum(dim=-1)
    for _ in range(n - 1):
        nxt = int(torch.argmax(dist))
        chosen.append(nxt)
        dist = torch.minimum(dist, ((flat - flat[nxt]) ** 2).sum(dim=-1))
    return pool[torch.tensor(chosen, device=pool.device)]


def default_time_anchors(horizons: tuple[int, ...], n_anchors: int,
                         pool_size: int = 4096, seed: int = 0,
                         device: str = "cpu") -> Tensor:
    """Built-in TIME-anchored vocabulary: FPS over a synthetic unicycle pool,
    deterministic (fixed seed) so two independently-built TacticalPolicies share
    anchors byte-for-byte. Anchors are TIME-parameterized over ``horizons`` (the
    tactical ``waypoint_horizons`` — the same 2 s ego-frame sub-waypoint slots the
    unimodal wp_heads produced)."""
    pool = _synth_anchor_pool(horizons, pool_size, seed, device=device)
    return _furthest_point_sample(pool, n_anchors, seed=seed).contiguous()


class _AnchorFiLM(nn.Module):
    """Live feature-wise linear modulation (mirrors refs.refc.FiLM zero_init=False)
    so the summary-token condition steers the anchor decoder from step 0."""

    def __init__(self, cond_dim: int, d: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class _AnchorCrossAttnLayer(nn.Module):
    """Anchor-query cross-attention block (pre-norm, residual): cross-attend the
    processed state-window tokens, then a FiLM(summary)-modulated MLP. Mirrors
    refs.refc.CrossAttnLayer but over the temporal window (KV = [B, W, d]) rather
    than a conv feature map."""

    def __init__(self, d: int, n_heads: int, cond_dim: int, ff_mult: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_f = nn.LayerNorm(d)
        self.film = _AnchorFiLM(cond_dim, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(),
                                 nn.Linear(ff_mult * d, d))

    def forward(self, q: Tensor, kv: Tensor, cond: Tensor) -> Tensor:
        h = self.norm_q(q)
        q = q + self.cross(h, kv, kv, need_weights=False)[0]
        q = q + self.mlp(self.film(self.norm_f(q), cond.unsqueeze(1)))
        return q


class AnchoredTacticalDecoder(nn.Module):
    """TIME-anchored (DiffusionDrive-style) tactical trajectory decoder — the
    multi-anchor REPLACEMENT for ``TacticalPolicy.wp_heads`` (classifier mode, the
    REF-C ``steps=0`` floor). A fixed FPS vocabulary of ego-frame trajectory
    ANCHORS whose queries cross-attend the processed state window and emit a
    per-anchor confidence + per-anchor offset; the winning mode's anchor+offset is
    the 2 s trajectory. This preserves MULTIMODALITY (the exact 3.4 m tactical
    weakness the unimodal per-horizon heads suffered) — the trainer assigns GT to
    its NEAREST anchor and applies anchor-cls CE + winner-takes-all L1 on that
    anchor only (never pulling every anchor to GT).

    H19 (maneuver->anchor prior): the tactical ``maneuver_head`` logits reweight
    the anchor confidences through a learned projection, guarded by a ZERO-INIT
    learnable SCALAR gate (``h19_gate``) so the prior starts as a strict no-op and
    ramps only if training earns it (like REF-C's zero-init grafts / the gated-
    intent lever). A scalar gate — NOT a LayerNorm: the REF-B run showed a
    LayerNorm pins the prior norm at sqrt(N) and can over-bias the confidences.

    forward(kv, cond, maneuver_logits) -> {anchor_logits [B, N], anchor_traj
    [B, N, S, 2], offset [B, N, S, 2], traj [B, S, 2] (selected), sel_idx [B],
    conf_norm, prior_norm, n_modes (diagnostics for the train log)}.
    """

    def __init__(self, d: int, horizons: tuple[int, ...], *,
                 n_maneuvers: int, n_anchors: int = 128, n_heads: int = 8,
                 layers: int = 2, ff_mult: int = 4, pool_size: int = 4096,
                 seed: int = 0):
        super().__init__()
        self.horizons = tuple(horizons)
        self.n_steps = len(self.horizons)
        # Anchor vocabulary — a persistent buffer (travels with the checkpoint).
        anchors = default_time_anchors(self.horizons, n_anchors, pool_size, seed)
        self.register_buffer("anchors", anchors)              # [N, S, 2]
        self.traj_proj = nn.Linear(self.n_steps * 2, d)       # anchor traj -> query
        self.layers = nn.ModuleList(
            _AnchorCrossAttnLayer(d, n_heads, d, ff_mult) for _ in range(layers))
        self.conf_head = nn.Linear(d, 1)                      # per-anchor conf
        self.offset_head = nn.Linear(d, self.n_steps * 2)     # per-anchor offset
        # H19: maneuver logits -> anchor prior (LIVE projection) scaled by a
        # ZERO-INIT scalar gate (no-op at init; ramps only if earned).
        self.maneuver_to_anchor = nn.Linear(n_maneuvers, n_anchors, bias=False)
        self.h19_gate = nn.Parameter(torch.zeros(()))

    def load_anchors(self, anchors: Tensor) -> None:
        """Install an externally-built anchor vocabulary. Shape must match
        [N, n_steps, 2] of the constructed decoder."""
        if tuple(anchors.shape) != tuple(self.anchors.shape):
            raise ValueError(f"anchor shape {tuple(anchors.shape)} != decoder "
                             f"{tuple(self.anchors.shape)}")
        self.anchors.copy_(anchors.to(self.anchors.dtype))

    def forward(self, kv: Tensor, cond: Tensor, maneuver_logits: Tensor) -> dict:
        """kv [B, W, d] processed state-window tokens; cond [B, d] summary token;
        maneuver_logits [B, n_maneuvers] the tactical maneuver distribution."""
        b = kv.shape[0]
        anchors = self.anchors.to(kv.dtype)                   # [N, S, 2]
        n = anchors.shape[0]
        x0 = anchors[None].expand(b, n, self.n_steps, 2)
        q = self.traj_proj(x0.reshape(b, n, -1))              # [B, N, d]
        for layer in self.layers:
            q = layer(q, kv, cond)
        conf = self.conf_head(q).squeeze(-1)                  # [B, N]
        offset = self.offset_head(q).reshape(b, n, self.n_steps, 2)
        x = anchors[None] + offset                            # [B, N, S, 2]
        # H19 maneuver prior (log-space), zero-init gated (a no-op at start).
        prior = self.maneuver_to_anchor(
            torch.log_softmax(maneuver_logits, dim=-1))       # [B, N]
        gated_prior = self.h19_gate * prior
        conf = conf + gated_prior
        idx = conf.argmax(dim=1)                              # [B] (detached select)
        traj = x[torch.arange(b, device=x.device), idx]       # [B, S, 2]
        return {"anchor_logits": conf, "anchor_traj": x, "offset": offset,
                "traj": traj, "sel_idx": idx,
                "conf_norm": conf.detach().norm(dim=-1).mean(),
                "prior_norm": gated_prior.detach().norm(dim=-1).mean(),
                "n_modes": int(idx.unique().numel())}


class TacticalPolicy(nn.Module):
    """Trained tactical brain (D-030) — ports REF-B rev2 ``TacticalHead`` and
    EXTENDS it with a target-latent goal head.

    A causal transformer over the operative STATE window, FiLM-conditioned on
    the strategic context token, that emits the tactical decision:
      - ``maneuver_logits`` [B, M] — distribution over the maneuver vocabulary;
      - ``waypoints`` {k: [B, 2]}  — the tactical GOAL position: 2 s ego-frame
        sub-waypoints (direct per-horizon heads; with ``anchor_tactical`` this is
        a shim read off the selected anchor trajectory — see below);
      - ``target_latent`` [B, S]   — the tactical GOAL latent (JEPA target at the
        2 s horizon — the "where the world should be" the operative aims for);
      - ``intent`` [B, d_intent]   — the token that FiLM-conditions the operative
        predictor, closing the hierarchy.

    State-dim-agnostic (shared brain). ``d_cond`` is the strategic context dim.

    ``anchor_tactical`` (v2 lever 8): REPLACE the unimodal ``wp_heads`` with an
    :class:`AnchoredTacticalDecoder` (TIME-anchored, DiffusionDrive-style) — the
    multimodal cure for the 3.4 m tactical weakness. Off (default) is byte-
    identical to the pre-lever brain (``wp_heads`` present, no anchor params). When
    on the forward ALSO emits ``anchor_logits`` / ``anchor_traj`` / ``offset`` /
    ``sel_idx`` and the ``waypoints`` dict becomes a shim off the selected traj.
    """

    def __init__(self, cfg: TacticalPolicyConfig, state_dim: int, window: int,
                 d_cond: int, ego_input: bool = False,
                 anchor_tactical: bool = False):
        super().__init__()
        self.cfg, self.window = cfg, window
        d = cfg.d_model
        self.in_proj = nn.Linear(state_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            CausalBlock(d, cfg.n_heads, cond_dim=d_cond)
            for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.maneuver_head = nn.Linear(d, cfg.n_maneuvers)
        # v2 lever 8: the TIME-anchored multi-anchor decoder REPLACES the unimodal
        # per-horizon wp_heads when gated on. OFF builds wp_heads in the SAME slot
        # (byte-identical init + state_dict; v1 ckpts resume); ON builds the
        # anchored decoder instead (no dead wp_heads). Both expose a `waypoints`
        # dict downstream (the anchored path via a shim off the selected traj).
        self.anchor_tactical = anchor_tactical
        if anchor_tactical:
            self.wp_heads = None
            self.anchor_decoder = AnchoredTacticalDecoder(
                d, tuple(cfg.waypoint_horizons), n_maneuvers=cfg.n_maneuvers,
                n_heads=cfg.n_heads)
        else:
            self.wp_heads = nn.ModuleDict(
                {str(k): nn.Linear(d, 2) for k in cfg.waypoint_horizons})
            self.anchor_decoder = None
        self.target_latent_head = nn.Linear(d, state_dim)
        self.intent_proj = nn.Linear(d, cfg.d_intent)
        # v2 lever 1: [v0, yr0] added to the strategic ctx cond (the tactical
        # wp heads were speed-starved: 3.38 m vs operative 0.628).
        self.ego_emb = nn.Linear(2, d_cond) if ego_input else None

    def forward(self, states: Tensor, ctx: Tensor,
                ego: Tensor | None = None) -> dict:
        if self.ego_emb is not None and ego is not None:
            ctx = ctx + self.ego_emb(ego.to(ctx.dtype))
        b, w, _ = states.shape
        assert w == self.window, f"window mismatch: {w} != {self.window}"
        x = self.in_proj(states) + self.pos[:, :w]
        cond = ctx.unsqueeze(1).expand(-1, w, -1)             # [B, W, d_cond]
        mask = _causal_mask(w, states.device)
        for blk in self.blocks:
            x = blk(x, cond, mask)
        h = self.norm(x[:, -1])
        man_logits = self.maneuver_head(h)
        out = {
            "maneuver_logits": man_logits,
            "target_latent": self.target_latent_head(h),
            "intent": self.intent_proj(h),
        }
        if self.anchor_decoder is not None:
            # KV = the processed state window (temporal analog of REF-C's conv
            # map); cond = the summary token h; maneuver logits drive the H19 prior.
            dec = self.anchor_decoder(x, h, man_logits)
            out.update(dec)
            # Back-compat shim: per-horizon points read off the SELECTED anchor
            # trajectory so downstream code/metrics using tac["waypoints"] still
            # work. Each waypoints[k] is the ego-frame position at horizon k — now
            # a point on ONE temporally-coherent anchor traj rather than an
            # independent per-horizon regression (see the report's semantics note).
            out["waypoints"] = {k: dec["traj"][:, i]
                                for i, k in enumerate(self.cfg.waypoint_horizons)}
        else:
            out["waypoints"] = {k: self.wp_heads[str(k)](h)
                                for k in self.cfg.waypoint_horizons}
        return out


def run_hierarchy(model, states: Tensor, actions: Tensor,
                  nav_cmd: Tensor | None = None,
                  ego: Tensor | None = None) -> dict:
    """Compose the 4-brain hierarchy on a compact state window (shared by the
    flagship WorldModel and REF-A — both hold the same brains).

    Flow (RECOVERY_PLAN §C wiring): strategic ``ctx`` --(FiLM)--> tactical
    ``intent`` --(FiLM)--> operative predictor. ``nav_cmd`` None -> ``follow``
    (index 0). Returns strategic + tactical outputs plus the operative ``preds``
    (k -> imagined latent), all conditioned down the chain.

    Cadence (thinking fast/slow): at deployment the caller recomputes the
    strategic ``ctx`` every ``strategic_policy.cadence`` (N_str) operative ticks
    and the tactical ``intent`` every ``tactical_policy.cadence`` (N_tac) ticks,
    reusing the cached tokens in between and calling only ``model.predictor(
    states, actions, intent=cached_intent)`` on operative-only ticks. Training
    runs all brains every step for dense gradients."""
    if model.strategic_policy is None or model.tactical_policy is None:
        raise ValueError("run_hierarchy needs both trained policy brains")
    b = states.shape[0]
    if nav_cmd is None:
        nav_cmd = torch.zeros(b, dtype=torch.long, device=states.device)
    strat = model.strategic_policy(states, nav_cmd, ego=ego)
    tac = model.tactical_policy(states, strat["ctx"], ego=ego)
    preds = model.predictor(states, actions, intent=tac["intent"])
    return {"ctx": strat["ctx"], "route_logits": strat["route_logits"],
            "maneuver_logits": tac["maneuver_logits"],
            "waypoints": tac["waypoints"], "target_latent": tac["target_latent"],
            "intent": tac["intent"], "preds": preds}


class WorldModel(nn.Module):
    """Encoder + readout + operative predictor + tactical predictor (optional)
    + trained tactical/strategic policy brains (optional, D-030) + H15
    imagination field (optional) + inverse dynamics + SIGReg.

    With ``cfg.tactical_policy``/``cfg.strategic_policy`` enabled this is the
    FULL trained-and-wired 4-brain flagship (strategic ctx --FiLM--> tactical
    intent --FiLM--> operative). With both None it is the base world model,
    unchanged — and a base model still loads a 4b checkpoint (the policy keys are
    simply extra), so ``evaluate_checkpoint`` / ``driving_diagnostic`` are
    unaffected.
    """

    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(cfg.encoder)
        self.readout = SpatialGridReadout(
            self.encoder.n_tokens, cfg.encoder.d_model,
            grid=cfg.readout.grid, d_readout=cfg.readout.d_readout)
        self.state_dim = self.readout.out_dim
        # The trained policy brains are a matched set (the hierarchy needs the
        # strategic context to condition tactical): enable both or neither.
        if (cfg.tactical_policy is None) != (cfg.strategic_policy is None):
            raise ValueError("tactical_policy and strategic_policy must be "
                             "enabled together (the 4-brain hierarchy)")
        intent_dim = (cfg.tactical_policy.d_intent
                      if cfg.tactical_policy is not None else None)
        # Operative predictor — intent-conditioned when the tactical brain is on.
        # v2 lever 7: ReZero gate on the intent term (H26 — ungated intent
        # diluted the action cond). Off => byte-identical state_dict.
        self.predictor = OperativePredictor(
            cfg.predictor, self.state_dim, intent_dim=intent_dim,
            gated_intent=getattr(cfg, "v2_gated_intent", False))
        # Tactical brain, parametric dynamics part: same predictor family at
        # maneuver horizons (8/16 steps) — the ALPS-4B tactical role.
        self.tactical_pred = (OperativePredictor(cfg.tactical_pred, self.state_dim)
                              if cfg.tactical_pred is not None else None)
        # Trained strategic transformer + tactical policy (D-030 recovery).
        _ego = bool(getattr(cfg, "v2_ego_to_planners", False))
        self.strategic_policy = (
            StrategicPolicy(cfg.strategic_policy, self.state_dim,
                            cfg.predictor.window, ego_input=_ego)
            if cfg.strategic_policy is not None else None)
        self.tactical_policy = (
            TacticalPolicy(cfg.tactical_policy, self.state_dim,
                           cfg.predictor.window,
                           d_cond=cfg.strategic_policy.d_ctx, ego_input=_ego,
                           anchor_tactical=getattr(cfg, "v2_anchor_tactical",
                                                   False))
            if cfg.tactical_policy is not None else None)
        # v2 lever 4: decode the 2 s trajectory FROM the imagined goal latent
        # (goal cos=0.885 while linear wp heads sat at 3.38 m — the decode was
        # the bottleneck, not the imagination). None when off.
        self.goal_traj_head = (nn.Sequential(
            nn.Linear(self.state_dim * 2, 512), nn.GELU(),
            nn.Linear(512, 2 * len(cfg.tactical_policy.waypoint_horizons)))
            if getattr(cfg, "v2_goal_decode", False)
            and cfg.tactical_policy is not None else None)
        # H15: belief maintenance over unobserved sectors (D-008).
        self.imagination = (ImaginationField(cfg.encoder.d_model,
                                             self.encoder.grid_hw,
                                             depth=cfg.h15.depth,
                                             n_heads=cfg.encoder.n_heads)
                            if cfg.h15.enabled else None)
        self.inv_dyn = InverseDynamicsHead(
            self.state_dim, cfg.predictor.action_dim,
            hidden=max(256, min(1024, self.state_dim // 2)))
        self.sigreg = SigReg(cfg.loss.sigreg.n_slices, cfg.loss.sigreg.beta)

    def encode_tokens(self, frames: Tensor) -> Tensor:
        """frames [B, C, H, W] -> token grid [B, N, D] (H15 operates here)."""
        return self.encoder(frames)

    def encode(self, frames: Tensor) -> Tensor:
        """frames [B, C, H, W] -> compact state [B, S]."""
        return self.readout(self.encoder(frames))

    def encode_window(self, frames: Tensor) -> Tensor:
        """frames [B, W, C, H, W'] -> states [B, W, S]."""
        b, w = frames.shape[:2]
        flat = frames.reshape(b * w, *frames.shape[2:])
        return self.encode(flat).reshape(b, w, -1)

    def imagine(self, states: Tensor, actions: Tensor,
                intent: Tensor | None = None) -> dict[int, Tensor]:
        """Causal window -> imagined future states per horizon (optionally
        conditioned by the tactical intent token)."""
        return self.predictor(states, actions, intent=intent)

    def hierarchy(self, states: Tensor, actions: Tensor,
                  nav_cmd: Tensor | None = None) -> dict:
        """Run the full strategic->tactical->operative chain (see
        :func:`run_hierarchy`)."""
        return run_hierarchy(self, states, actions, nav_cmd)


@dataclass
class MonitorReport:
    imag_relative: float          # ||z_hat - z||/scale ; <1 = better than persistence
    effective_rank: float
    collapse_alarm: bool
    ood_alarm: bool


class FallbackMonitor:
    """Brain 4 (Phase 0 form): out-of-gradient self-monitor + MRC hook.

    - imagination error (A9): free OOD/anomaly signal while driving,
    - collapse watchdog: effective rank / variance / NaN on latents,
    - mrc_hook: callable executed on alarm (in sim: deterministic brake profile).
    """

    def __init__(self, imag_threshold: float = 1.0, rank_floor: float = 4.0,
                 mrc_hook=None):
        self.imag_threshold = imag_threshold
        self.rank_floor = rank_floor
        self.mrc_hook = mrc_hook

    @staticmethod
    def _effective_rank(z: Tensor) -> float:
        zc = z - z.mean(0, keepdim=True)
        s = torch.linalg.svdvals(zc.double())
        p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
        return float(torch.exp(-(p * p.log()).sum()))

    @torch.no_grad()
    def step(self, z_pred: Tensor, z_true: Tensor, z_prev: Tensor) -> MonitorReport:
        scale = (z_true - z_prev).norm(dim=-1).mean().clamp_min(1e-8)
        imag_rel = float((z_pred - z_true).norm(dim=-1).mean() / scale)
        erank = self._effective_rank(z_true)
        collapse = bool(erank < self.rank_floor or not torch.isfinite(z_true).all())
        ood = bool(imag_rel > self.imag_threshold)
        if (collapse or ood) and self.mrc_hook is not None:
            self.mrc_hook(collapse=collapse, ood=ood, imag_relative=imag_rel)
        return MonitorReport(imag_rel, erank, collapse, ood)


@dataclass
class Maneuver:
    """A tactical candidate: a short action sequence (primitive).

    ``maneuver_class`` (optional) is the candidate's index in the maneuver
    vocabulary, used by :meth:`TacticalSelector.propose_and_score` to read the
    trained policy's preference for this candidate (the PROPOSE prior)."""
    name: str
    actions: Tensor               # [K, action_dim]
    maneuver_class: int | None = None


class TacticalSelector:
    """Imagine-and-select over a discrete maneuver vocabulary (H1/H2 pattern).

    For each candidate maneuver: roll the operative predictor forward, decode
    the imagined end-latent with the imagination-calibrated probe (A3), score
    distance-to-subgoal + action cost. Milliseconds: K batched predictor
    passes, no pixels, no diffusion, no CEM.
    """

    def __init__(self, world: WorldModel, probe_imag: RidgeProbe,
                 comfort_weight: float = 0.01):
        self.world = world
        self.probe = probe_imag
        self.comfort_weight = comfort_weight

    @torch.no_grad()
    def select(self, states: Tensor, past_actions: Tensor,
               maneuvers: list[Maneuver], subgoal_xy: Tensor) -> tuple[int, Tensor]:
        """states [1, W, S], past_actions [1, W, A], subgoal_xy [2].

        Returns (best index, per-maneuver scores). Phase 0 scoring: predicted
        ego displacement toward sub-goal + comfort (action magnitude). Safety
        costs (TTC proxy) attach here once the obstacle probe lands.
        """
        scores = []
        for m in maneuvers:
            s, a = states.clone(), past_actions.clone()
            # Apply the maneuver's actions step by step, feeding predictions back.
            for k in range(m.actions.shape[0]):
                a = torch.roll(a, -1, dims=1)
                a[:, -1] = m.actions[k]
                z_next = self.world.imagine(s, a)[1]          # 1-step head
                s = torch.roll(s, -1, dims=1)
                s[:, -1] = z_next
            decoded = self.probe.predict(s[:, -1])            # -> ego (x, y, ...)
            dist = (decoded[0, :2] - subgoal_xy.to(decoded.dtype)).norm()
            comfort = m.actions.pow(2).mean()
            scores.append(dist + self.comfort_weight * comfort)
        scores_t = torch.stack(scores)
        return int(scores_t.argmin()), scores_t

    @torch.no_grad()
    def propose_and_score(self, states: Tensor, past_actions: Tensor,
                          maneuvers: list[Maneuver], subgoal_xy: Tensor,
                          step_readout, ctx: Tensor,
                          prior_weight: float = 1.0) -> tuple[int, Tensor]:
        """D-030 tactical selection: the TRAINED policy PROPOSES, the GROUNDED
        rollout SCORES.

        The trained :class:`TacticalPolicy` scores the maneuver vocabulary (its
        distribution is the PROPOSE prior + it emits the intent token). For each
        candidate we roll the intent-conditioned operative predictor under the
        maneuver's action primitive, decode each transition's per-step metric
        Δpose with the GROUNDED ``step_readout`` (StepDisplacementReadout, not the
        old ridge probe), accumulate SE(2) to the endpoint, and SCORE distance-
        to-subgoal + comfort minus the policy log-prior for the candidate's
        class. Returns ``(best_idx, scores)`` (lower is better).

        ``states``/``past_actions`` [1, W, *]; ``ctx`` [1, d_ctx] the strategic
        context; ``subgoal_xy`` [2] the tactical goal in the ego frame."""
        from tanitad.models.metric_dynamics import accumulate_se2
        assert self.world.tactical_policy is not None, \
            "propose_and_score needs a trained tactical policy"
        tac = self.world.tactical_policy(states, ctx)
        intent = tac["intent"]
        logp = torch.log_softmax(tac["maneuver_logits"][0], dim=-1)
        scores = []
        for m in maneuvers:
            s, a = states.clone(), past_actions.clone()
            dposes = []
            for k in range(m.actions.shape[0]):
                a = torch.roll(a, -1, dims=1)
                a[:, -1] = m.actions[k]
                z_next = self.world.predictor(s, a, intent=intent)[1]
                dposes.append(step_readout(s[:, -1], z_next))
                s = torch.roll(s, -1, dims=1)
                s[:, -1] = z_next
            endpoint = accumulate_se2(torch.stack(dposes, dim=1))[:, -1]   # [1,2]
            dist = (endpoint[0] - subgoal_xy.to(endpoint.dtype)).norm()
            comfort = m.actions.pow(2).mean()
            prior = (logp[m.maneuver_class] if m.maneuver_class is not None
                     else torch.zeros((), device=states.device))
            scores.append(dist + self.comfort_weight * comfort
                          - prior_weight * prior)
        scores_t = torch.stack(scores)
        return int(scores_t.argmin()), scores_t


class StrategicGraph:
    """Latent transition graph (A6): topological memory of the driven network.

    Minimal Phase 0 skeleton: nodes = code indices from k-means over pooled
    latents; edges = observed transitions with visit-count costs; route =
    Dijkstra. The ALPS-4B implementation (+58 % topology edge) is the port
    reference. Re-routing trigger: imagination diverges from the edge's
    expectation (plan-vs-imagination divergence).
    """

    def __init__(self):
        self.edges: dict[int, dict[int, float]] = {}

    def observe_transition(self, code_a: int, code_b: int, cost: float = 1.0):
        if code_a == code_b:
            return
        nbrs = self.edges.setdefault(code_a, {})
        # Empirical cost: running mean biased toward cheaper (more traveled) edges.
        nbrs[code_b] = min(nbrs.get(code_b, cost), cost)
        self.edges.setdefault(code_b, {})

    def route(self, start: int, goal: int) -> list[int] | None:
        import heapq
        dist = {start: 0.0}
        prev: dict[int, int] = {}
        pq: list[tuple[float, int]] = [(0.0, start)]
        seen: set[int] = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == goal:
                path = [u]
                while u in prev:
                    u = prev[u]
                    path.append(u)
                return path[::-1]
            for v, w in self.edges.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        return None
