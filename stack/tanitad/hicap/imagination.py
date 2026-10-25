"""HiCAP-I: identifiable, tiered imagination (paper sec:imag). Tier 0 soft outcome-aware targets (training only), Tier 1 exogenous
E-head + analytic ego cost (every tick), Tier 2 gated counterfactual cost heads with a two-head-disagreement VETO (top-8, learned
benefit gate). Guards G1-G9 are implemented where they are code (G1 mask after learned terms, G2 coverage mask, G4 veto-not-trigger,
G5 confidence from the BASE score only). No gradient flows from these heads into the frozen backbone (cached tokens), and the E-head
reads the state only (G9)."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import MLP, HiCAPConfig

H_T = (0.5, 1.0, 2.0, 3.0)                     # horizons of the exogenous descriptors (s)
MODE_ACC = (-3.0, 0.0, 1.5, -6.0)              # lead behaviour modes: brakes, holds, accelerates, brakes hard (m/s^2, constants)
N_MODE = len(MODE_ACC)
FAMILIES = ("safety", "comfort", "progress", "lateral")


# ------------------------------------------------------------------ Tier 0
def tier0_target(J_teacher: torch.Tensor, star: torch.Tensor, T: float = 1.0, mix: float = 0.7, valid: torch.Tensor | None = None):
    """y(a) = 0.7 softmax(-J/T) + 0.3 onehot(a*). J_teacher [B,K] from logged other-agent futures (labels may use them). Zero inference cost."""
    z = -J_teacher / T
    if valid is not None: z = z.masked_fill(~valid, -1e9)
    return mix * torch.softmax(z, -1) + (1 - mix) * F.one_hot(star, J_teacher.shape[-1]).float()


def teacher_costs(traj: np.ndarray, agents_xy: np.ndarray | None, v0: float, dt: float = 0.1) -> np.ndarray:
    """Per-family teacher cost of ONE candidate [4] (safety, comfort, progress, lateral) against non-reactive logged agent tracks.
    traj [T,2] ego frame; agents_xy [A,T,2] logged agent positions in the same frame (None = no agents -> safety 0)."""
    T = len(traj)
    safety = 0.0
    if agents_xy is not None and len(agents_xy):
        d = np.linalg.norm(agents_xy - traj[None], axis=-1).min(1)                 # per-agent min distance over time
        safety = float(np.maximum(0.0, 3.0 - d.min()) ** 2)                        # penalise clearance < 3 m
    v = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], traj]), axis=0), axis=1) / dt
    a = np.diff(np.concatenate([[v0], v])) / dt; jerk = np.diff(a) / dt
    comfort = float((np.maximum(np.abs(a) - 3.0, 0) ** 2).mean() + 0.01 * (np.maximum(np.abs(jerk) - 5.0, 0) ** 2).mean())
    progress = float(max(0.0, v0 * T * dt - np.linalg.norm(traj[-1])) / max(v0 * T * dt, 1.0))
    lateral = float(np.abs(traj[:, 1]).max() / 10.0)
    return np.array([safety, comfort, progress, lateral], np.float32)


# ------------------------------------------------------------------ Tier 1
class EHead(nn.Module):
    """R^512 -> R^1024 -> R^24 reading the STATE only: lead gap / closing speed / lead accel at 0.5,1,2,3 s (12), log-sigma of the gap (4),
    mode logits (4), onset quantiles q10/q50/q90 (3), has-lead logit (1)."""
    def __init__(self, d=512):
        super().__init__(); self.m = MLP(d, 2 * d, 24)
    def forward(self, s):
        o = self.m(s)
        return dict(gap=F.softplus(o[:, 0:4]) * 10.0, closing=o[:, 4:8], lacc=o[:, 8:12], logsig=o[:, 12:16], mode=o[:, 16:20],
                    onset=torch.sort(F.softplus(o[:, 20:23]), -1).values * 3.0, has_lead=o[:, 23])


def exo_loss(out: dict, tgt: dict):
    """L^exo: Huber on descriptors + CE on mode + quantile (pinball) on onset, all on supervised rows (tgt['w'] in {0,1}); has-lead BCE.
    tgt: gap [B,4], closing [B,4], lacc [B,4], mode [B] (-100 ignore), onset [B], has_lead [B], w [B]."""
    w = tgt["w"]; hl = tgt["has_lead"]
    m = (w * hl)
    huber = sum(F.smooth_l1_loss(out[k], tgt[k], reduction="none").mean(-1) for k in ("gap", "closing", "lacc"))
    L = (huber * m).sum() / m.sum().clamp(min=1.0)
    ok = (tgt["mode"] >= 0) & (m > 0)
    if ok.any(): L = L + F.cross_entropy(out["mode"][ok], tgt["mode"][ok])
    q = torch.tensor([0.1, 0.5, 0.9], device=w.device)
    e = tgt["onset"].unsqueeze(-1) - out["onset"]
    pin = torch.maximum(q * e, (q - 1) * e).mean(-1)
    L = L + (pin * m).sum() / m.sum().clamp(min=1.0)
    L = L + F.binary_cross_entropy_with_logits(out["has_lead"], hl, reduction="none").mul(w).sum() / w.sum().clamp(min=1.0)
    return L


def _lead_pos(out: dict, i: int, v0: float, t: torch.Tensor, acc: float, onset: float) -> torch.Tensor:
    """Lead absolute longitudinal position L(t) for sample i, one mode and onset: baseline from the descriptors (gap_h + v0 t_h, linearly
    interpolated over horizons) plus 0.5 a_mode (t - onset)_+^2."""
    th = np.array(H_T, np.float32)
    knots = (out["gap"][i].detach().numpy() + v0 * th).astype(np.float32)          # lead position if the ego held its speed
    slope = (knots[-1] - knots[-2]) / (th[-1] - th[-2])
    tt = t.numpy()
    L0 = np.interp(tt, th, knots) + np.where(tt > th[-1], slope * (tt - th[-1]), 0.0) - np.where(tt < th[0], (th[0] - tt) * slope, 0.0)
    L0 = torch.from_numpy(L0.astype(np.float32))
    return L0 + 0.5 * acc * torch.clamp(t - onset, min=0) ** 2


def tier1_cost(out: dict, i: int, trajs: np.ndarray, v0: float, dt: float = 0.1, ttc_thr: float = 2.5, gap_thr: float = 4.0) -> np.ndarray:
    """J1(s,a) = E_{mode, onset}[ c(e_a, x_{m,tau}) ] for sample i over candidates trajs [K,T,2]. Marginalises mode AND onset timing (G7);
    c is an analytic time-to-collision + clearance cost on the candidate's own arc length. Returns [K] >= 0; zero if no lead."""
    K, T = trajs.shape[:2]
    if float(torch.sigmoid(out["has_lead"][i])) < 0.5: return np.zeros(K, np.float32)
    t = (torch.arange(T) + 1).float() * dt
    s_ego = torch.from_numpy(np.cumsum(np.linalg.norm(np.diff(np.concatenate([np.zeros((K, 1, 2)), trajs], 1), axis=1), axis=-1), 1).astype(np.float32))   # [K,T]
    pm = torch.softmax(out["mode"][i], -1); ons = out["onset"][i]
    J = torch.zeros(K)
    for m in range(N_MODE):
        for q, wq in zip(ons, (0.3, 0.4, 0.3)):
            L = _lead_pos(out, i, v0, t, MODE_ACC[m], float(q))
            gap = L.unsqueeze(0) - s_ego                                     # [K,T]
            close = -(gap[:, 1:] - gap[:, :-1]) / dt                          # closing speed
            ttc = gap[:, 1:] / close.clamp(min=1e-3)
            ttc = torch.where(close > 1e-3, ttc, torch.full_like(ttc, 99.0))
            c = F.relu(ttc_thr - ttc.min(1).values) ** 2 + F.relu(gap_thr - gap.min(1).values) ** 2 * 0.1
            J = J + float(pm[m]) * wq * c
    return J.numpy().astype(np.float32)


# ------------------------------------------------------------------ Tier 2
class Tier2(nn.Module):
    """Two cost heads on phi(s,a) = norm(s + R v(a)) regressing per-family COUNTERFACTUAL costs; |head1 - head2| is a veto on
    out-of-vocabulary candidates (never a benefit trigger, G4); a learned gate decides whether Tier 2 runs (target <= 30 % of ticks)."""
    def __init__(self, d=512, n_fam=4):
        super().__init__()
        self.R = nn.Linear(d, d, bias=False)
        self.h = nn.ModuleList([MLP(d, d, n_fam) for _ in range(2)])
        self.gate = MLP(d + 2, d // 2, 1)

    def phi(self, s, v): return F.normalize(s.unsqueeze(1) + self.R(v), dim=-1)                    # s [B,d], v [B,K,d]
    def forward(self, s, v):
        p = self.phi(s, v); return torch.stack([F.softplus(h(p)) for h in self.h])                 # [2,B,K,4]
    def veto(self, c, thr: float): return (c[0] - c[1]).abs().max(-1).values > thr                 # [B,K]
    def benefit_logit(self, s, margin, entropy): return self.gate(torch.cat([s, margin.unsqueeze(-1), entropy.unsqueeze(-1)], -1)).squeeze(-1)


def tier2_loss(c: torch.Tensor, cf: torch.Tensor, mask: torch.Tensor):
    """Regress counterfactual family costs where they exist (G2: mask[b,k,f]=0 where a family has no counterfactual supervision)."""
    l = sum(F.smooth_l1_loss(c[i], cf, reduction="none") for i in range(2)) * mask
    return l.sum() / mask.sum().clamp(min=1.0)


def gate_target(base_regret: torch.Tensor, tier2_regret: torch.Tensor, margin: float = 0.0):
    """Benefit gate label: 1 where adding Tier 2 reduces regret by more than `margin` (needs counterfactual labels)."""
    return ((base_regret - tier2_regret) > margin).float()


# ------------------------------------------------------------------ composition (decision time)
def compose_scores(base: torch.Tensor, J1: torch.Tensor | None, J2: torch.Tensor | None, beta1: float, beta2: float,
                   feasible: torch.Tensor | None = None, veto: torch.Tensor | None = None, veto_pen: float = 1e3):
    """score = base - beta1 J1 - beta2 J2 (J >= 0), then G1: analytic feasibility mask AFTER every learned term; G4: veto removes
    out-of-vocabulary candidates. Each term has a zero-weight ablation (beta = 0)."""
    s = base.clone()
    if J1 is not None and beta1: s = s - beta1 * J1
    if J2 is not None and beta2: s = s - beta2 * J2
    if veto is not None: s = s - veto_pen * veto.float()
    if feasible is not None: s = s.masked_fill(~feasible, -1e9)
    return s


def confidence(base: torch.Tensor) -> torch.Tensor:
    """G5: confidence is computed from the BASE score only; the combined softmax is never exposed."""
    p = torch.softmax(base, -1); return p.max(-1).values
