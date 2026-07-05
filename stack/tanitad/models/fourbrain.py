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

from tanitad.config import StackConfig
from tanitad.models.encoder import ViTEncoder
from tanitad.models.inverse_dynamics import InverseDynamicsHead
from tanitad.models.predictor import OperativePredictor
from tanitad.models.readout import RidgeProbe, SpatialGridReadout
from tanitad.models.sigreg import SigReg


class WorldModel(nn.Module):
    """Encoder + readout + operative predictor + inverse dynamics + SIGReg."""

    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(cfg.encoder)
        self.readout = SpatialGridReadout(
            self.encoder.n_tokens, cfg.encoder.d_model,
            grid=cfg.readout.grid, d_readout=cfg.readout.d_readout)
        self.state_dim = self.readout.out_dim
        self.predictor = OperativePredictor(cfg.predictor, self.state_dim)
        self.inv_dyn = InverseDynamicsHead(self.state_dim, cfg.predictor.action_dim)
        self.sigreg = SigReg(cfg.loss.sigreg.n_slices, cfg.loss.sigreg.beta)

    def encode(self, frames: Tensor) -> Tensor:
        """frames [B, C, H, W] -> compact state [B, S]."""
        return self.readout(self.encoder(frames))

    def encode_window(self, frames: Tensor) -> Tensor:
        """frames [B, W, C, H, W'] -> states [B, W, S]."""
        b, w = frames.shape[:2]
        flat = frames.reshape(b * w, *frames.shape[2:])
        return self.encode(flat).reshape(b, w, -1)

    def imagine(self, states: Tensor, actions: Tensor) -> dict[int, Tensor]:
        """Causal window -> imagined future states per horizon."""
        return self.predictor(states, actions)


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
    """A tactical candidate: a short action sequence (primitive)."""
    name: str
    actions: Tensor               # [K, action_dim]


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
