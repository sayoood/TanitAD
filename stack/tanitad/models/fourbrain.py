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


class TacticalPolicy(nn.Module):
    """Trained tactical brain (D-030) — ports REF-B rev2 ``TacticalHead`` and
    EXTENDS it with a target-latent goal head.

    A causal transformer over the operative STATE window, FiLM-conditioned on
    the strategic context token, that emits the tactical decision:
      - ``maneuver_logits`` [B, M] — distribution over the maneuver vocabulary;
      - ``waypoints`` {k: [B, 2]}  — the tactical GOAL position: 2 s ego-frame
        sub-waypoints (direct per-horizon heads, no recursion);
      - ``target_latent`` [B, S]   — the tactical GOAL latent (JEPA target at the
        2 s horizon — the "where the world should be" the operative aims for);
      - ``intent`` [B, d_intent]   — the token that FiLM-conditions the operative
        predictor, closing the hierarchy.

    State-dim-agnostic (shared brain). ``d_cond`` is the strategic context dim.
    """

    def __init__(self, cfg: TacticalPolicyConfig, state_dim: int, window: int,
                 d_cond: int, ego_input: bool = False):
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
        self.wp_heads = nn.ModuleDict(
            {str(k): nn.Linear(d, 2) for k in cfg.waypoint_horizons})
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
        return {
            "maneuver_logits": self.maneuver_head(h),
            "waypoints": {k: self.wp_heads[str(k)](h)
                          for k in self.cfg.waypoint_horizons},
            "target_latent": self.target_latent_head(h),
            "intent": self.intent_proj(h),
        }


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
        self.predictor = OperativePredictor(cfg.predictor, self.state_dim,
                                            intent_dim=intent_dim)
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
                           d_cond=cfg.strategic_policy.d_ctx, ego_input=_ego)
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
