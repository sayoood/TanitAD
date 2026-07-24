"""Supervised predictive NON-CAUSAL Inverse-Dynamics (IDM) head on a FROZEN
encoder — the cheapest discriminating experiment for the IDM/YouTube line
(TanitAD Research Hub/Architecture & Inference/IDM_VIDEO_PRETRAIN_DESIGN §5).

The head is a small bidirectional temporal transformer over a window of encoder
latents ``z_{t-k..t+k}`` (k=4 → 9 frames) that reads out, at the window CENTER
``t``: continuous ``speed / yaw_rate / steer / long_accel`` and the 2 s metric
ego-frame trajectory at horizons ``{5,10,15,20}``. Continuous regression (Huber +
trajectory L2), NOT a discrete codebook — the design's option (a).

DESIGN INVARIANTS (see the pre-registration note):
  * The encoder is FROZEN and PURELY VISUAL: ``encode_window`` uses only
    ``encoder`` + ``readout`` and takes NO action/speed channel, so the head is a
    readout on ``z`` alone. Everything here operates on cached ``z [T, state_dim]``.
  * NON-CAUSAL: no attention mask — the labeler sees past AND future frames (VPT's
    trick; we only ever label offline). The center token is the prediction site.
  * Ground truth is CAN-derived kinematics already in the episode contract
    (``poses [T,4] = x,y,yaw,v``; ``actions [T,2] = steer_road_rad, accel_mps2``).

This module is dependency-light (torch + numpy) on purpose, so it imports on the
pods' older tree and on the dev box alike. The pod-side encode/orchestration lives
in ``run_idm_proof.py``; this file holds the model, targets, losses, metrics and a
generic ``train_head`` so it is unit-testable on CPU with synthetic latents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Fixed target order — the columns of every scalar tensor produced here.
SCALAR_NAMES: tuple[str, ...] = ("speed", "yaw_rate", "steer", "long_accel")
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 15, 20)   # 0.5/1/1.5/2 s @ 10 Hz
DT = 0.1                                               # 10 Hz contract


# --------------------------------------------------------------------------- #
# geometry helpers (repo `_ego` convention: +x forward, +y left; yaw about +z) #
# --------------------------------------------------------------------------- #
def wrap_to_pi(a: Tensor) -> Tensor:
    """Wrap angles to (-pi, pi] so yaw differences across +-pi stay small."""
    return a - (2 * math.pi) * torch.floor((a + math.pi) / (2 * math.pi))


def ego_frame(dxy: Tensor, yaw: Tensor) -> Tensor:
    """World displacement [...,2] -> ego frame of ``yaw`` (d1_probe `_ego`)."""
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


# --------------------------------------------------------------------------- #
# targets from the episode contract                                           #
# --------------------------------------------------------------------------- #
def scalar_targets_at(poses: Tensor, actions: Tensor, t: Tensor) -> Tensor:
    """CAN ground-truth scalars at center indices ``t`` [N] -> [N, 4] in
    SCALAR_NAMES order: (speed, yaw_rate, steer, long_accel).

    ``yaw_rate`` is a CENTERED finite difference of the (wrapped) heading —
    non-causal, which is exactly the labeler's regime. Needs ``t-1`` and ``t+1``
    valid (the caller's window guarantees it)."""
    speed = poses[t, 3]
    steer = actions[t, 0]
    accel = actions[t, 1]
    yaw_rate = wrap_to_pi(poses[t + 1, 2] - poses[t - 1, 2]) / (2.0 * DT)
    return torch.stack([speed, yaw_rate, steer, accel], dim=-1)


def traj_targets_at(poses: Tensor, t: Tensor,
                    horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> Tensor:
    """Ego-frame 2 s waypoint targets at center indices ``t`` [N] ->
    [N, len(horizons), 2]. Waypoint at horizon h = ego_frame(xy[t+h]-xy[t],
    yaw[t]) — the ``refb_labels.waypoint_targets`` convention exactly."""
    yaw0 = poses[t, 2]
    xy0 = poses[t, :2]
    wps = [ego_frame(poses[t + h, :2] - xy0, yaw0) for h in horizons]
    return torch.stack(wps, dim=1)


def valid_centers(T: int, k: int, horizons: tuple[int, ...], stride: int) -> Tensor:
    """Center indices ``t`` with a full [t-k, t+k] latent window, ``t-1``/``t+1``
    for the centered yaw-rate, and ``t+max(horizons)`` future poses for the
    trajectory. Strided."""
    max_h = max(horizons)
    lo = max(k, 1)
    hi = T - 1 - max(k, max_h)          # inclusive upper bound
    if hi < lo:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(lo, hi + 1, stride, dtype=torch.long)


def build_windows(z: Tensor, poses: Tensor, actions: Tensor, *, k: int = 4,
                  horizons: tuple[int, ...] = DEFAULT_HORIZONS, stride: int = 2
                  ) -> tuple[Tensor, Tensor, Tensor]:
    """One cached episode -> (Zwin [N, 2k+1, D], scalars [N,4], traj [N,H,2]).

    ``z`` [T, D] frozen per-frame latents; ``poses`` [T,4]; ``actions`` [T,2].
    NON-CAUSAL windows centred on each valid ``t``. Empty tensors when the
    episode is too short (caller concatenates and drops empties)."""
    T = z.shape[0]
    t = valid_centers(T, k, horizons, stride)
    if t.numel() == 0:
        D = z.shape[1]
        H = len(horizons)
        return (z.new_zeros(0, 2 * k + 1, D), z.new_zeros(0, 4),
                z.new_zeros(0, H, 2))
    # gather the [t-k, t+k] window for every center (advanced indexing)
    offs = torch.arange(-k, k + 1)
    idx = t[:, None] + offs[None, :]                    # [N, 2k+1]
    Zwin = z[idx]                                        # [N, 2k+1, D]
    scal = scalar_targets_at(poses, actions, t)          # [N, 4]
    traj = traj_targets_at(poses, t, horizons)           # [N, H, 2]
    return Zwin, scal, traj


# --------------------------------------------------------------------------- #
# the head                                                                    #
# --------------------------------------------------------------------------- #
class IDMHead(nn.Module):
    """Small NON-CAUSAL temporal transformer over the latent window -> center
    readout of the 4 scalars + the 2 s ego trajectory. ~a few M params."""

    def __init__(self, state_dim: int = 2048, d_model: int = 256, depth: int = 3,
                 n_heads: int = 4, window: int = 9, n_scalars: int = 4,
                 horizons: tuple[int, ...] = DEFAULT_HORIZONS):
        super().__init__()
        self.window = window
        self.center = window // 2
        self.horizons = tuple(horizons)
        self.in_proj = nn.Linear(state_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.scalar_head = nn.Linear(d_model, n_scalars)
        self.traj_head = nn.Linear(d_model, 2 * len(self.horizons))

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """z [B, W, state_dim] -> {"scalars" [B,4], "traj" [B,H,2]}. No mask =
        bidirectional (non-causal)."""
        b, w, _ = z.shape
        x = self.in_proj(z) + self.pos[:, :w]
        x = self.blocks(x)                               # bidirectional
        h = self.norm(x[:, self.center])
        return {"scalars": self.scalar_head(h),
                "traj": self.traj_head(h).reshape(b, len(self.horizons), 2)}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# standardiser + loss + metrics                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Standardizer:
    """Per-scalar train mean/std for a balanced Huber loss (R²/MAE stay in raw
    physical units)."""
    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, scalars: Tensor) -> "Standardizer":
        mean = scalars.mean(0)
        std = scalars.std(0).clamp_min(1e-6)
        return cls(mean, std)

    def norm(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x)) / self.std.to(x)


def idm_loss(pred: dict[str, Tensor], scal: Tensor, traj: Tensor,
             std: Standardizer, *, traj_weight: float = 1.0,
             traj_scale: float = 10.0, huber_beta: float = 1.0
             ) -> dict[str, Tensor]:
    """Huber on STANDARDISED scalars + smooth-L1 on the trajectory scaled to
    O(1) by ``traj_scale`` metres. Without the scale the raw-metre trajectory
    (~10 m at 2 s) swamps the standardised scalar term and the speed/yaw/steer
    heads never train — measured on the CPU smoke."""
    ps = std.norm(pred["scalars"])
    ts = std.norm(scal)
    scal_l = F.huber_loss(ps, ts, delta=huber_beta)
    traj_l = F.smooth_l1_loss(pred["traj"] / traj_scale, traj / traj_scale,
                              beta=huber_beta)
    return {"loss": scal_l + traj_weight * traj_l,
            "scalar_loss": scal_l.detach(), "traj_loss": traj_l.detach()}


def r2_score(pred: Tensor, gt: Tensor) -> float:
    """1 - SS_res/SS_tot per column, but here called per-scalar 1-D vectors."""
    gt = gt.double()
    pred = pred.double()
    ss_res = ((gt - pred) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)


def traj_ade(pred_traj: Tensor, gt_traj: Tensor) -> float:
    """ADE@2s: mean over windows of the mean over horizons of the L2 waypoint
    error (metres). ``[N, H, 2]`` inputs."""
    de = (pred_traj.double() - gt_traj.double()).norm(dim=-1)   # [N, H]
    return float(de.mean())


@torch.no_grad()
def evaluate(head: IDMHead, Z: Tensor, scal: Tensor, traj: Tensor, *,
             device: str = "cpu", batch: int = 1024) -> dict:
    """R² per scalar + per-horizon de + ADE@2s + MAE, on a held-out window set."""
    head.eval()
    preds_s, preds_t = [], []
    for i in range(0, Z.shape[0], batch):
        out = head(Z[i:i + batch].to(device))
        preds_s.append(out["scalars"].cpu())
        preds_t.append(out["traj"].cpu())
    ps = torch.cat(preds_s) if preds_s else scal.new_zeros(0, 4)
    pt = torch.cat(preds_t) if preds_t else traj.new_zeros(0, *traj.shape[1:])
    r2 = {SCALAR_NAMES[j]: r2_score(ps[:, j], scal[:, j])
          for j in range(len(SCALAR_NAMES))}
    mae = {SCALAR_NAMES[j]: float((ps[:, j].double() - scal[:, j].double())
                                  .abs().mean())
           for j in range(len(SCALAR_NAMES))}
    de = (pt.double() - traj.double()).norm(dim=-1).mean(0)       # [H]
    return {"n": int(Z.shape[0]), "r2": r2, "mae": mae,
            "ade_2s": traj_ade(pt, traj),
            "de_per_horizon": [float(x) for x in de]}


def train_head(train: tuple[Tensor, Tensor, Tensor],
               val_sets: dict[str, tuple[Tensor, Tensor, Tensor]], *,
               state_dim: int, horizons: tuple[int, ...] = DEFAULT_HORIZONS,
               epochs: int = 8, batch: int = 256, lr: float = 3e-4,
               wd: float = 0.01, traj_weight: float = 1.0, seed: int = 0,
               device: str = "cpu", log=print) -> dict:
    """Fit an ``IDMHead`` on ``train`` (Z, scalars, traj), evaluate each named
    val set. Returns {"val": {name: metrics}, "params": P, "train_n": N}. The
    standardiser is fit on TRAIN only (no eval leakage)."""
    torch.manual_seed(seed)
    Ztr, Str, Ttr = train
    std = Standardizer.fit(Str)
    head = IDMHead(state_dim=state_dim, horizons=horizons).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = Ztr.shape[0]
    steps_per = max(1, n // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = head(Ztr[idx].to(device))
            ld = idm_loss(out, Str[idx].to(device), Ttr[idx].to(device), std,
                          traj_weight=traj_weight)
            opt.zero_grad(set_to_none=True)
            ld["loss"].backward()
            opt.step()
            sched.step()
            tot += float(ld["loss"].detach()) * len(idx)
        log(f"[idm] epoch {ep+1}/{epochs} train_loss {tot / n:.4f}")
    val = {name: evaluate(head, Zv, Sv, Tv, device=device)
           for name, (Zv, Sv, Tv) in val_sets.items()}
    return {"val": val, "params": count_params(head), "train_n": int(n),
            "scalar_mean": [float(x) for x in std.mean],
            "scalar_std": [float(x) for x in std.std]}


# --------------------------------------------------------------------------- #
# CPU self-test (import-free smoke): synthetic latents that ENCODE the target  #
# so a healthy head must drive R² up — proves finite + differentiable + learns #
# --------------------------------------------------------------------------- #
def _synthetic_episode(T: int, D: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    # BOUNDED, shared-distribution across episodes (a random walk would let the
    # val episodes drift outside the train range and tank R² for reasons that
    # have nothing to do with the head). A hidden state drives BOTH the latent
    # and the kinematics, so the inverse map exists and a healthy head recovers
    # it; distinct phases/freqs per seed keep episodes non-identical.
    t = torch.arange(T).float()
    ph = torch.rand(4, generator=g) * 6.28
    fr = 0.05 + 0.10 * torch.rand(4, generator=g)
    hidden = torch.stack([torch.sin(fr[j] * t + ph[j]) for j in range(4)], dim=1)
    v = 8.0 + 4.0 * hidden[:, 0]                          # 4..12 m/s, bounded
    yaw = 0.25 * hidden[:, 1]
    x = torch.cumsum(v * torch.cos(yaw) * DT, dim=0)
    y = torch.cumsum(v * torch.sin(yaw) * DT, dim=0)
    poses = torch.stack([x, y, yaw, v], dim=1).float()
    steer = 0.08 * hidden[:, 2]
    accel = (v[1:] - v[:-1]) / DT
    accel = torch.cat([accel[:1], accel])
    actions = torch.stack([steer, accel], dim=1).float()
    z = torch.cat([hidden, 0.3 * torch.randn(T, D - 4, generator=g)], dim=1).float()
    return z, poses, actions


def _self_test() -> None:
    torch.manual_seed(0)
    D = 64
    tr, va = [], []
    for s in range(16):
        z, p, a = _synthetic_episode(160, D, s)
        (tr if s < 12 else va).append(build_windows(z, p, a, k=4))
    def cat(lst):
        return (torch.cat([x[0] for x in lst]), torch.cat([x[1] for x in lst]),
                torch.cat([x[2] for x in lst]))
    Ztr, Str, Ttr = cat(tr)
    Zva, Sva, Tva = cat(va)
    assert torch.isfinite(Ztr).all() and torch.isfinite(Str).all()
    head = IDMHead(state_dim=D)
    out = head(Ztr[:8])
    assert out["scalars"].shape == (8, 4) and out["traj"].shape == (8, 4, 2)
    std = Standardizer.fit(Str)
    ld = idm_loss(out, Str[:8], Ttr[:8], std)
    assert torch.isfinite(ld["loss"]), "non-finite loss"
    ld["loss"].backward()                                # differentiable
    gnorm = sum(float(p.grad.norm()) for p in head.parameters()
                if p.grad is not None)
    assert math.isfinite(gnorm) and gnorm > 0, "no/NaN gradient"
    res = train_head((Ztr, Str, Ttr), {"val": (Zva, Sva, Tva)}, state_dim=D,
                     epochs=15, batch=64)
    r2 = res["val"]["val"]["r2"]
    ade = res["val"]["val"]["ade_2s"]
    print("[idm-selftest] params", res["params"], "windows",
          Ztr.shape[0], "->", {k: round(v, 3) for k, v in r2.items()},
          "ade", round(ade, 3))
    # SMOKE CONTRACT (not a science result): the full pipeline
    # build_windows -> head -> loss -> train -> evaluate is finite, differentiable,
    # and demonstrably FITS a target end-to-end. steer here is a clean direct
    # readout of a latent dim (identical pathway to speed), so its recovery proves
    # the machinery; the real quantitative R² lands in the pod GPU runs on CAN GT.
    assert math.isfinite(ade), "non-finite ADE"
    best = max(r2.values())
    assert best > 0.8, f"no scalar learned (best R² {best:.3f}) — pipeline broken"
    print("[idm-selftest] PASS (finite, differentiable, fits a target end-to-end)")


if __name__ == "__main__":
    _self_test()
