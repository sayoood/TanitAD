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


def speed_seq_targets_at(poses: Tensor, t: Tensor, k: int) -> Tensor:
    """CAN speed at EVERY window position ``t-k..t+k`` -> [N, 2k+1].

    Supervising the whole window rather than only its centre is what lets
    ``long_accel`` be DERIVED instead of regressed — see
    :func:`derive_long_accel` for why that matters.
    """
    offs = torch.arange(-k, k + 1, device=t.device)
    return poses[t[:, None] + offs[None, :], 3]


def derive_long_accel(speed_seq: Tensor, center: int, dt: float = DT) -> Tensor:
    """Centred difference of the predicted speed sequence -> long_accel [N].

    WHY THIS EXISTS — MEASURED, not assumed. On the comma2k19 val cache
    (30 episodes, 8,940 windows) the CAN ``long_accel`` channel is recoverable
    from the TRUE speed track by exactly this finite difference at **R² 0.902**
    (pooled corr 0.951). Yet the shipped head, which predicts ``speed`` at
    **R² 0.86**, regresses ``long_accel`` as an independent fourth linear
    readout of the centre token and scores **R² −0.15 to −0.42** on the banked
    held-out read — i.e. worse than emitting the training mean
    (``…/2026-07-27-fleet-sync-idm-steer/raw/idm5_ensemble.json``, every seed and
    both domains).

    The information is present: the window is NON-CAUSAL and spans t−k..t+k, so
    both v(t−1) and v(t+1) are inside it. Nothing in the architecture or the loss
    said that ``long_accel`` and the change in ``speed`` are the same physical
    quantity, so the head learned them as unrelated targets and got one wrong.
    Deriving the channel imposes that identity by construction.

    ⛔ **REFUTED ON THE REAL CORPUS — DO NOT ENABLE `speed_seq` TO FIX
    `long_accel`.** MEASURED 2026-08-03, 50 content-clean comma2k19 episodes,
    EPISODE-DISJOINT 33/17 split, 3 seeds, 100 epochs, paired episode-cluster
    bootstrap B=2000
    (`…/incoming/2026-08-03-idm-derived-accel/results_idm_derived_accel.json`):
    deriving is **SEPARATED-WORSE**, ΔR² ``long_accel`` **−0.2530
    [−0.4831, −0.0997]** (−0.2061 → −0.4591), and it also costs ``yaw_rate``
    (−0.1142 [−0.3635, −0.0688]).

    WHY, and it is the same 5× that motivates the term above: at R² 0.72 the
    speed channel still has MAE **4.04 m/s**, and a 0.2 s centred difference of
    two such predictions carries ~5× that error against a target whose own MAE is
    **0.45 m/s²**. The identity is real; the head's speed track is nowhere near
    accurate enough to exploit it.

    ⭐ The refutation produced the better diagnosis: on the same run the BASELINE's
    ``long_accel`` is **not separated** from a control whose latent↔target link
    was destroyed (−0.0984 [−0.3087, +0.0179]) while ``speed`` (+0.7187) and
    ``yaw_rate`` (+0.2252) are. **The channel carries no recoverable information
    from the frozen v1 latents at this scale**, so no reparameterisation of the
    head can repair it — the lever is the representation or the data, not this.

    The code stays because it is off by default, cheap, and its contract is
    tested; it is kept as the settled negative result, not as a recommendation.
    """
    if speed_seq.shape[-1] < center + 2 or center < 1:
        raise ValueError(f"need a window position each side of centre {center}, "
                         f"got width {speed_seq.shape[-1]}")
    return (speed_seq[..., center + 1] - speed_seq[..., center - 1]) / (2.0 * dt)


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
                 horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                 speed_seq: bool = False):
        super().__init__()
        self.window = window
        self.center = window // 2
        self.horizons = tuple(horizons)
        #: read speed at EVERY window position, so ``long_accel`` can be derived
        #: from it (:func:`derive_long_accel`) instead of regressed separately.
        #: Off by default: every banked checkpoint predates it and must keep
        #: loading and scoring exactly as before.
        self.speed_seq = bool(speed_seq)
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
        self.speed_seq_head = nn.Linear(d_model, 1) if self.speed_seq else None

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """z [B, W, state_dim] -> {"scalars" [B,4], "traj" [B,H,2]}, plus
        ``"speed_seq" [B,W]`` and the derived ``"long_accel" [B]`` when the
        speed-sequence readout is enabled. No mask = bidirectional (non-causal).
        """
        b, w, _ = z.shape
        x = self.in_proj(z) + self.pos[:, :w]
        x = self.blocks(x)                               # bidirectional
        xn = self.norm(x)
        h = xn[:, self.center]
        out = {"scalars": self.scalar_head(h),
               "traj": self.traj_head(h).reshape(b, len(self.horizons), 2)}
        if self.speed_seq_head is not None:
            # the readout runs on EVERY token, not just the centre — the window
            # is bidirectional, so v(t-1) and v(t+1) are already in scope
            out["speed_seq"] = self.speed_seq_head(xn).squeeze(-1)     # [B, W]
            out["long_accel"] = derive_long_accel(out["speed_seq"], self.center)
        return out

    def deployed_scalars(self, out: dict[str, Tensor]) -> Tensor:
        """The 4 scalars as they would actually be USED [B,4].

        Identical to ``out["scalars"]`` for a legacy head. With the speed-sequence
        readout on, ``long_accel`` is replaced by the derived channel — the point
        of the fix — so callers never have to remember which column to swap.
        """
        s = out["scalars"]
        if "long_accel" not in out:
            return s
        j = SCALAR_NAMES.index("long_accel")
        return torch.cat([s[:, :j], out["long_accel"][:, None], s[:, j + 1:]], 1)


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
             traj_scale: float = 10.0, huber_beta: float = 1.0,
             speed_seq: Tensor | None = None, speed_seq_weight: float = 1.0
             ) -> dict[str, Tensor]:
    """Huber on STANDARDISED scalars + smooth-L1 on the trajectory scaled to
    O(1) by ``traj_scale`` metres. Without the scale the raw-metre trajectory
    (~10 m at 2 s) swamps the standardised scalar term and the speed/yaw/steer
    heads never train — measured on the CPU smoke.

    ``speed_seq`` [N, W] is the per-window-position CAN speed. When given (and
    the head emits one), it adds a Huber term on the standardised speed
    SEQUENCE. The ``long_accel`` column of ``scal`` is then supervised ONLY
    through that sequence — the direct readout is left out of the loss, because
    keeping both would let the head satisfy the accel target the broken way and
    the arm would no longer isolate the fix.
    """
    ps = std.norm(pred["scalars"])
    ts = std.norm(scal)
    j = SCALAR_NAMES.index("long_accel")
    use_seq = speed_seq is not None and "speed_seq" in pred
    if use_seq:
        keep = [c for c in range(ts.shape[1]) if c != j]
        scal_l = F.huber_loss(ps[:, keep], ts[:, keep], delta=huber_beta)
    else:
        scal_l = F.huber_loss(ps, ts, delta=huber_beta)
    traj_l = F.smooth_l1_loss(pred["traj"] / traj_scale, traj / traj_scale,
                              beta=huber_beta)
    total = scal_l + traj_weight * traj_l
    out = {"scalar_loss": scal_l.detach(), "traj_loss": traj_l.detach()}
    if use_seq:
        i = SCALAR_NAMES.index("speed")
        mu, sd = std.mean.to(ps)[i], std.std.to(ps)[i]
        seq_l = F.huber_loss((pred["speed_seq"] - mu) / sd,
                             (speed_seq.to(ps) - mu) / sd, delta=huber_beta)
        # ⚠️ The derived channel is a REPARAMETERISATION, not a substitute for
        # supervision. Differencing two window positions multiplies whatever
        # error the speed sequence has by 1/(2*dt) = 5x, so a head trained ONLY
        # on absolute speeds derives a WORSE accel than a direct readout —
        # MEASURED on the synthetic contract: derived 0.6134 vs direct 0.8755
        # when the accel target is left out of the loss. Supervising the derived
        # channel on its own target, in its own standardised units, is what makes
        # the physics a prior rather than a handicap.
        aj = (pred["long_accel"] - std.mean.to(ps)[j]) / std.std.to(ps)[j]
        acc_l = F.huber_loss(aj, ts[:, j], delta=huber_beta)
        total = total + speed_seq_weight * seq_l + acc_l
        out["speed_seq_loss"] = seq_l.detach()
        out["derived_accel_loss"] = acc_l.detach()
    out["loss"] = total
    return out


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
def predict(head: IDMHead, Z: Tensor, *, device: str = "cpu",
            batch: int = 1024) -> dict[str, Tensor]:
    """Batched forward -> {"scalars" [N,4] AS DEPLOYED, "traj" [N,H,2], ...}.

    ``scalars`` is :meth:`IDMHead.deployed_scalars`, so a speed-sequence head is
    scored on its DERIVED ``long_accel`` — the channel it would actually emit —
    while ``scalars_direct`` keeps the raw readout for the side-by-side.
    """
    head.eval()
    keys = ["scalars", "traj", "scalars_direct"]
    if head.speed_seq_head is not None:
        keys.append("speed_seq")
    acc: dict[str, list] = {k: [] for k in keys}
    for i in range(0, Z.shape[0], batch):
        out = head(Z[i:i + batch].to(device))
        acc["scalars"].append(head.deployed_scalars(out).cpu())
        acc["scalars_direct"].append(out["scalars"].cpu())
        acc["traj"].append(out["traj"].cpu())
        if "speed_seq" in acc:
            acc["speed_seq"].append(out["speed_seq"].cpu())
    return {k: (torch.cat(v) if v else torch.zeros(0)) for k, v in acc.items()}


@torch.no_grad()
def evaluate(head: IDMHead, Z: Tensor, scal: Tensor, traj: Tensor, *,
             device: str = "cpu", batch: int = 1024) -> dict:
    """R² per scalar + per-horizon de + ADE@2s + MAE, on a held-out window set.

    With the speed-sequence readout on, ``r2``/``mae`` report the DERIVED
    ``long_accel`` (what the head deploys) and ``r2_direct`` additionally
    reports the untouched linear readout, so the fix's effect is visible in one
    result rather than needing two runs.
    """
    p = predict(head, Z, device=device, batch=batch)
    ps, pt = p["scalars"], p["traj"]
    if ps.numel() == 0:
        ps, pt = scal.new_zeros(0, 4), traj.new_zeros(0, *traj.shape[1:])
    r2 = {SCALAR_NAMES[j]: r2_score(ps[:, j], scal[:, j])
          for j in range(len(SCALAR_NAMES))}
    mae = {SCALAR_NAMES[j]: float((ps[:, j].double() - scal[:, j].double())
                                  .abs().mean())
           for j in range(len(SCALAR_NAMES))}
    de = (pt.double() - traj.double()).norm(dim=-1).mean(0)       # [H]
    res = {"n": int(Z.shape[0]), "r2": r2, "mae": mae,
           "ade_2s": traj_ade(pt, traj),
           "de_per_horizon": [float(x) for x in de]}
    if head.speed_seq_head is not None and p["scalars_direct"].numel():
        pd_ = p["scalars_direct"]
        res["r2_direct"] = {SCALAR_NAMES[j]: r2_score(pd_[:, j], scal[:, j])
                            for j in range(len(SCALAR_NAMES))}
        res["speed_seq_on"] = True
    return res


def fit_head(train, *, state_dim: int,
             horizons: tuple[int, ...] = DEFAULT_HORIZONS,
             epochs: int = 8, batch: int = 256, lr: float = 3e-4,
             wd: float = 0.01, traj_weight: float = 1.0, seed: int = 0,
             device: str = "cpu", log=print, speed_seq: bool = False,
             speed_seq_weight: float = 1.0, head_kw: dict | None = None
             ) -> tuple["IDMHead", Standardizer, dict]:
    """Fit an ``IDMHead`` and hand back the MODULE, its standardiser and metadata.

    :func:`train_head` wraps this and returns only the JSON-serialisable part,
    because its result is dumped straight to disk by callers
    (``run_idm_proof.py:387``) and must never carry an ``nn.Module``. Anything
    that needs the weights — a per-window prediction dump for a paired bootstrap,
    a checkpoint save — calls ``fit_head``.

    ``speed_seq=True`` enables the per-position speed readout and derives
    ``long_accel`` from it. ``train`` may then carry a FOURTH element, the
    ``[N, W]`` speed-sequence target from :func:`speed_seq_targets_at`; a
    3-tuple keeps working unchanged, which is what makes every existing caller
    and every banked checkpoint safe.
    """
    torch.manual_seed(seed)
    Ztr, Str, Ttr = train[0], train[1], train[2]
    Qtr = train[3] if len(train) > 3 else None
    if speed_seq and Qtr is None:
        raise ValueError("speed_seq=True needs the [N, W] speed-sequence target "
                         "as a fourth element of `train`")
    std = Standardizer.fit(Str)
    # ``head_kw`` carries d_model / depth / n_heads / window. It defaults to
    # empty so the production geometry is unchanged; a CPU test needs a small
    # head, and without this the only way to get one was to bypass the fitter —
    # which would then no longer be the code under test.
    head = IDMHead(state_dim=state_dim, horizons=horizons,
                   speed_seq=speed_seq, **(head_kw or {})).to(device)
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
                          traj_weight=traj_weight,
                          speed_seq=None if Qtr is None else Qtr[idx].to(device),
                          speed_seq_weight=speed_seq_weight)
            opt.zero_grad(set_to_none=True)
            ld["loss"].backward()
            opt.step()
            sched.step()
            tot += float(ld["loss"].detach()) * len(idx)
        log(f"[idm] epoch {ep+1}/{epochs} train_loss {tot / n:.4f}")
    return head, std, {"params": count_params(head), "train_n": int(n),
                       "speed_seq": bool(speed_seq),
                       "scalar_mean": [float(x) for x in std.mean],
                       "scalar_std": [float(x) for x in std.std]}


def train_head(train, val_sets: dict, *, state_dim: int,
               horizons: tuple[int, ...] = DEFAULT_HORIZONS, **kw) -> dict:
    """Fit an ``IDMHead`` on ``train`` (Z, scalars, traj[, speed_seq]) and
    evaluate each named val set.

    Returns a JSON-serialisable ``{"val": {name: metrics}, "params": P,
    "train_n": N, ...}``. The standardiser is fit on TRAIN only (no eval
    leakage). Use :func:`fit_head` when the fitted module itself is needed.
    """
    head, _std, meta = fit_head(train, state_dim=state_dim, horizons=horizons, **kw)
    dev = kw.get("device", "cpu")
    val = {name: evaluate(head, v[0], v[1], v[2], device=dev)
           for name, v in val_sets.items()}
    return {"val": val, **meta}


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
