"""Elastic sensing (paper sec:elastic): camera tiers, vision-only gate G-V, cost-sensitive full-information gate loss, asymmetric
keep-alive controller with shadow-mode prefetch, camera-dropout curriculum, budget-conditioned configuration embedding and L^cfg.
The gate maps a situation to a camera set, so it is a situation classifier in disguise: G-V reads ONLY the pooled front-wide
embedding and its own previous camera set; G-E (ego) is an ablation pending a ruling; G-T (previous tactical token) is NOT provided."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import MLP

CAMERAS = ("front_wide", "front_tele", "cross_left", "cross_right", "rear_left", "rear_right", "rear_tele")
FRONT, PERIPH = 160, 64                          # tokens per camera
# tier -> camera subsets the gate may choose (T0 front only; T1 + one; T2 + two; T3 all seven = probe / degraded state)
OPTIONS = ([0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 1, 2], [0, 1, 3], [0, 2, 3], list(range(7)))
N_OPT = len(OPTIONS)


def option_tokens(j: int) -> int: return FRONT + PERIPH * (len(OPTIONS[j]) - 1)
def option_cost(j: int) -> float: return option_tokens(j) / FRONT           # camera units relative to front-only


class CameraGate(nn.Module):
    """G-V: input = pooled front-wide embedding (d) + one-hot previous option (keep-alive state)."""
    def __init__(self, d=512):
        super().__init__(); self.m = MLP(d + N_OPT, d // 2, N_OPT)
    def forward(self, front_emb, prev_opt):
        return self.m(torch.cat([front_emb, F.one_hot(prev_opt, N_OPT).float()], -1))


def gate_loss(logits: torch.Tensor, E: torch.Tensor, criticality: torch.Tensor, lam: float = 0.05):
    """Full-information cost-sensitive loss: sum_j p_j [E_j(x) * crit(x) + lam cost_j]. E [B,N_OPT] = per-option error of the frozen
    backbone + heads on cached features; criticality [B] weights a wrong camera (cheap in cruise, costly at a junction)."""
    p = torch.softmax(logits, -1); cost = torch.tensor([option_cost(j) for j in range(N_OPT)], device=logits.device)
    return (p * (E * criticality.unsqueeze(-1) + lam * cost)).sum(-1).mean()


@dataclass
class KeepAlive:
    """Escalate in ONE tick; de-escalate only after `down` consecutive ticks with a margin; never smooth symmetrically. A newly enabled
    camera is encoded in SHADOW mode (not used) for `shadow` ticks before first use."""
    down: int = 3; margin: float = 0.10; shadow: int = 2
    cur: int = 0; _below: int = 0; _shadow: dict = field(default_factory=dict)

    def step(self, probs: np.ndarray) -> dict:
        want = int(np.argmax(probs)); cost = lambda j: option_cost(j)
        if cost(want) > cost(self.cur) or (want != self.cur and cost(want) == cost(self.cur) and probs[want] - probs[self.cur] > self.margin):
            self.cur = want; self._below = 0                                           # escalate immediately
        elif want != self.cur and probs[want] - probs[self.cur] > self.margin:
            self._below += 1
            if self._below >= self.down: self.cur = want; self._below = 0
        else:
            self._below = 0
        used, shadowed = [], []
        for c in OPTIONS[self.cur]:
            n = self._shadow.get(c, 0)
            if c == 0 or n >= self.shadow: used.append(c)
            else: shadowed.append(c)
            self._shadow[c] = n + 1
        for c in list(self._shadow):
            if c not in OPTIONS[self.cur]: self._shadow[c] = 0
        return dict(option=self.cur, cameras_used=used, cameras_shadow=shadowed, tokens=FRONT + PERIPH * (len(used) - 1))


def dropout_option(rng: np.random.Generator, oracle: int, return_kind: bool = False):
    """Camera-dropout curriculum: 40 % oracle, 20 % oracle + one random extra, 15 % all, 25 % front + random 0-2 extras. The sampled camera
    SET is mapped to the smallest available option that contains it (else the best overlap)."""
    u = rng.random()
    if u < 0.40: kind, s = "oracle", set(OPTIONS[oracle])
    elif u < 0.60: kind, s = "oracle+1", set(OPTIONS[oracle]) | {int(rng.integers(1, 7))}
    elif u < 0.75: kind, s = "all", set(range(7))
    else:
        k = int(rng.integers(0, 3)); kind, s = "front+rand", {0, *(rng.choice(np.arange(1, 7), size=k, replace=False).tolist() if k else [])}
    sup = [j for j in range(N_OPT) if s <= set(OPTIONS[j])]
    j = min(sup, key=lambda j: len(OPTIONS[j])) if sup else max(range(N_OPT), key=lambda j: len(set(OPTIONS[j]) & s))
    return (j, kind) if return_kind else j


class ConfigEmbedding(nn.Module):
    """Budget-conditioned configuration embedding appended to the pooled state; the full-camera embedding is a distillation target."""
    def __init__(self, d=512): super().__init__(); self.e = nn.Embedding(N_OPT, d)
    def forward(self, opt): return self.e(opt)


def cfg_loss(s_c: torch.Tensor, s_full: torch.Tensor, A: torch.Tensor, y: torch.Tensor, widths=(64, 128, 256), tau: float = 20.0, beta: float = 1.0):
    """L^cfg = sum_d InfoNCE(f_d(e_c), a_d) + beta (1 - cos(e_c, sg e_full)); s_c / s_full [B,d] unit readouts, A [V,d] node table, y [B] targets."""
    L = 0.0
    for w in widths:
        z = tau * F.normalize(s_c[:, :w], dim=-1) @ F.normalize(A[:, :w], dim=-1).t()
        L = L + F.cross_entropy(z, y)
    return L + beta * (1 - F.cosine_similarity(s_c, s_full.detach(), dim=-1)).mean()
