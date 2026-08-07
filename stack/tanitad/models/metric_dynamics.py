"""Metric ego-motion grounding for the world-model latent (fix-B1, dynamics mode).

WHY THIS EXISTS
---------------
The driving diagnostic proved the JEPA latent does NOT encode metric ego-
trajectory: oracle in-distribution decode ceiling ADE@1s = 1.65 m, while a
trivial constant-velocity predictor gets 0.28 m and the held-out MLP probe 3.89 m
(10-15x worse than trivial, everywhere, incl. straight highway). Root cause
(mechanistic): the objective optimizes JEPA latent-future prediction + SigReg
anti-collapse + imagination ranking (that is why D2 PASSES) but has NO loss that
forces the latent to hold *metric* ego-motion — the waypoint readout was a
post-hoc ridge probe, never trained.

THE FIX (Sayed directive — preserve the SSL character)
------------------------------------------------------
Do NOT bolt on a supervised absolute-waypoint regressor (that leans
imitation-learning and would not reshape the dynamics). Instead ground the
ACTION-CONDITIONED DYNAMICS in metric space, using ONLY proprioceptive, free
signals (CAN actions + IMU/GNSS ego-pose odometry — NO human labels), and let the
trajectory EMERGE from a forward rollout:

- ``MetricInverseDynamics``: from a latent PAIR ``(x_t, x_{t+k})`` regress the
  METRIC relative ego-pose ``(Δx, Δy, Δyaw)`` measured from odometry (``_ego``
  convention). This is the metre-scale generalization of the action inverse-
  dynamics head (which only had to recover the action *class*). It forces the
  ENCODER latent to encode metric ego-motion — the diagnosed missing signal that
  the 1.65 m ceiling measures.

- ``StepDisplacementReadout``: from a SINGLE latent transition ``(x_t, x_{t+1})``
  decode the per-step metric Δpose. Applied to each latent produced by rolling
  the operative predictor forward under the TRUE action sequence, then
  ACCUMULATED by SE(2) dead-reckoning (:func:`accumulate_se2`) into a trajectory.
  So the eval trajectory is the rollout of grounded dynamics, not a one-shot
  regression. Calibrated (A3 doctrine) on the predictor's own rolled latents via
  the forward-consistency loss.

Both heads are deliberately SEPARATE ``nn.Module``s (never registered on
``WorldModel``) so a fine-tune checkpoint's ``["model"]`` stays a vanilla
``WorldModel`` state dict and ``evaluate_checkpoint`` / ``driving_diagnostic``
load it unchanged.

The geometry helpers (``ego_delta``, ``wrap_angle``, ``relative_ego_pose``,
``accumulate_se2``) are pure and match the ``_ego`` convention of
``scripts/d1_probe_capacity.py`` / ``scripts/driving_diagnostic.py`` byte-for-
byte, so the training targets and the eval metric use one definition.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


# --------------------------------------------------------------------------- #
# Straight-through gradient SCALE (Part A loss-rebalance, 2026-07-18)          #
# --------------------------------------------------------------------------- #
def grad_scale(x: Tensor, alpha: float) -> Tensor:
    """Straight-through gradient scale: forward returns ``x`` UNCHANGED, backward
    multiplies the gradient w.r.t. ``x`` by ``alpha`` (``0 <= alpha <= 1``).

    This is the research's ``x*alpha + x.detach()*(1-alpha)`` written as
    ``x + (alpha-1)*(x - x.detach())`` — algebraically identical (expand:
    ``x*alpha + (1-alpha)*x.detach()``) but computed in a straight-through order
    so the forward value is BIT-EXACTLY ``x`` for every ``alpha`` (``x - x.detach()``
    is a real zero in the forward, so nothing perturbs the fitted value). Only the
    gradient flowing back INTO ``x`` is scaled; any parameter downstream of ``x``
    (e.g. an MLP head reading it) sees the identical input value and therefore an
    unchanged gradient.

    ``alpha == 1.0`` short-circuits to ``x`` itself: a strict no-op (identical
    graph, loss, and gradients), so the default-off path is provably byte-identical
    to pre-change — the safety contract for an unattended multi-day run.
    """
    if alpha == 1.0:
        return x
    return x + (alpha - 1.0) * (x - x.detach())


# --------------------------------------------------------------------------- #
# Pure SE(2) geometry — the ONE ego-frame convention (matches d1_probe._ego)   #
# --------------------------------------------------------------------------- #
def ego_delta(dxy: Tensor, yaw: Tensor) -> Tensor:
    """Rotate a world-frame displacement into the ego frame at departure ``yaw``.

    ``dxy[..., 2]`` world displacement, ``yaw[...]`` departure heading (rad).
    Rotation by ``-yaw`` — identical to ``_ego`` in ``d1_probe_capacity.py`` and
    ``driving_diagnostic.py``. Broadcasts: ``yaw`` may be ``[...]`` or
    ``[..., 1]`` against ``dxy`` of ``[..., k, 2]``.
    """
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


def wrap_angle(a: Tensor) -> Tensor:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def relative_ego_pose(pose_a: Tensor, pose_b: Tensor) -> Tensor:
    """Metric relative ego-pose from ``pose_a`` to ``pose_b``, in a's ego frame.

    poses ``[..., 4]`` = (x, y, yaw, v). Returns ``[..., 3]`` = (Δx, Δy, Δyaw):
    the position delta rotated into a's heading plus the wrapped heading change.
    This is exactly the odometry target the metric heads regress.
    """
    dxy = ego_delta(pose_b[..., :2] - pose_a[..., :2], pose_a[..., 2])
    dyaw = wrap_angle(pose_b[..., 2] - pose_a[..., 2]).unsqueeze(-1)
    return torch.cat([dxy, dyaw], dim=-1)


def accumulate_se2(step_dpose: Tensor) -> Tensor:
    """Dead-reckon per-step ego Δposes into absolute ego waypoints (origin frame).

    ``step_dpose`` ``[B, K, 3]`` = (Δx, Δy, Δyaw), each expressed in the ego frame
    of the PREVIOUS step. Returns cumulative positions ``[B, K, 2]`` in the
    origin (step-0) ego frame. SE(2) composition::

        P_j = P_{j-1} + Rot(Ψ_{j-1}) @ [Δx_j, Δy_j];   Ψ_j = Ψ_{j-1} + Δyaw_j

    with ``P_0 = 0, Ψ_0 = 0``. Feeding the TRUE per-step Δposes reproduces the GT
    ego waypoints ``ego_delta(pos_j - pos_0, yaw_0)`` exactly (unit-tested), so the
    accumulated rollout and the diagnostic's waypoint metric use one geometry.
    """
    assert step_dpose.dim() == 3 and step_dpose.shape[-1] == 3, \
        f"step_dpose must be [B, K, 3], got {tuple(step_dpose.shape)}"
    b, k, _ = step_dpose.shape
    pos = torch.zeros(b, 2, device=step_dpose.device, dtype=step_dpose.dtype)
    psi = torch.zeros(b, device=step_dpose.device, dtype=step_dpose.dtype)
    out = []
    for j in range(k):
        c, s = torch.cos(psi), torch.sin(psi)
        dx, dy = step_dpose[:, j, 0], step_dpose[:, j, 1]
        pos = pos + torch.stack([c * dx - s * dy, s * dx + c * dy], dim=-1)
        psi = psi + step_dpose[:, j, 2]
        out.append(pos)
    return torch.stack(out, dim=1)                       # [B, K, 2]


def gt_step_dposes(pose_last: Tensor, future_poses: Tensor, k: int) -> Tensor:
    """True per-step ego Δposes for a rollout of ``k`` steps.

    ``pose_last`` ``[B, 4]`` (window last frame), ``future_poses`` ``[B, H, 4]``
    (H >= k). Returns ``[B, k, 3]`` where row j is
    ``relative_ego_pose(pose_{j-1}, pose_j)`` — the odometry Δpose of transition j
    in the frame at step j-1. Accumulating these reproduces the GT waypoints.
    """
    prev = torch.cat([pose_last.unsqueeze(1), future_poses[:, :k - 1]], dim=1)
    cur = future_poses[:, :k]
    return relative_ego_pose(prev, cur)                  # [B, k, 3]


def gt_ego_waypoints(pose_last: Tensor, future_poses: Tensor,
                     steps) -> Tensor:
    """GT ego-frame waypoints at ``steps`` (1-based, in latent-transition units).

    ``pose_last`` ``[B, 4]``, ``future_poses`` ``[B, H, 4]``. Returns
    ``[B, len(steps), 2]`` = ``ego_delta(pos_{last+k} - pos_last, yaw_last)`` — the
    same target the diagnostic decodes. ``future_poses[k-1]`` is k steps ahead.
    """
    p0 = pose_last[:, :2].unsqueeze(1)                   # [B,1,2]
    yaw0 = pose_last[:, 2:3]                              # [B,1]
    idx = torch.tensor([s - 1 for s in steps], device=future_poses.device)
    tgt = future_poses.index_select(1, idx)[..., :2]     # [B,len,2]
    return ego_delta(tgt - p0, yaw0)


# --------------------------------------------------------------------------- #
# Heads                                                                        #
# --------------------------------------------------------------------------- #
class MetricInverseDynamics(nn.Module):
    """(x_t, x_{t+k}) latent pair -> metric relative ego-pose (Δx, Δy, Δyaw).

    Grounds the encoder latent in metre-scale ego-motion (attacks the 1.65 m
    oracle ceiling). LayerNorm over the concatenated pair keeps the two states'
    scales comparable; a 2-hidden-layer GELU MLP maps to the 3-D pose delta.
    """

    def __init__(self, state_dim: int, pose_dim: int = 3, hidden: int = 512):
        super().__init__()
        self.pose_dim = pose_dim
        self.net = nn.Sequential(
            nn.LayerNorm(2 * state_dim),
            nn.Linear(2 * state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, pose_dim),
        )

    def forward(self, z_t: Tensor, z_tk: Tensor) -> Tensor:
        return self.net(torch.cat([z_t, z_tk], dim=-1))


class StepDisplacementReadout(nn.Module):
    """Single latent transition (x_t, x_{t+1}) -> per-step metric Δpose.

    Applied to each rolled predictor latent at train (forward-consistency) and
    eval (trajectory rollout). Smaller than MetricInverseDynamics: a per-step
    displacement is a simpler map than an arbitrary-horizon one.
    """

    def __init__(self, state_dim: int, pose_dim: int = 3, hidden: int = 512):
        super().__init__()
        self.pose_dim = pose_dim
        self.net = nn.Sequential(
            nn.LayerNorm(2 * state_dim),
            nn.Linear(2 * state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, pose_dim),
        )

    def forward(self, z_t: Tensor, z_next: Tensor) -> Tensor:
        return self.net(torch.cat([z_t, z_next], dim=-1))


# --------------------------------------------------------------------------- #
# Forward rollout — the trajectory EMERGES from grounded action-conditioned    #
# dynamics (reuses the D-027 K-step rollout mechanism, TRUE actions).          #
# --------------------------------------------------------------------------- #
def rollout_decode(predictor, states: Tensor, actions: Tensor,
                   future_actions: Tensor | None,
                   step_readout: StepDisplacementReadout, k: int
                   ) -> tuple[Tensor, Tensor]:
    """Roll ``predictor`` forward ``k`` steps under the TRUE action sequence,
    decode each transition's per-step metric Δpose, accumulate SE(2).

    ``states`` ``[B, W, S]``, ``actions`` ``[B, W, A]`` (window), ``future_actions``
    ``[B, H, A]`` (H >= k-1) or None (zero-order-hold). Returns
    ``(waypoints [B, k, 2], step_dpose [B, k, 3])``. Gradients flow into the
    encoder (via ``states``), the predictor, and ``step_readout`` — so the whole
    dynamics chain is shaped to be metrically consistent with proprioception.
    """
    win_s, win_a = states, actions
    dposes = []
    for j in range(k):
        z_hat = predictor(win_s, win_a)[1]               # 1-step head -> z_{t+j+1}
        dposes.append(step_readout(win_s[:, -1], z_hat))
        if j < k - 1:
            a_next = (future_actions[:, j] if future_actions is not None
                      else win_a[:, -1])
            win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], dim=1)
    step_dpose = torch.stack(dposes, dim=1)              # [B, k, 3]
    return accumulate_se2(step_dpose), step_dpose


def rollout_transitions(predictor, states: Tensor, actions: Tensor,
                        future_actions: Tensor | None, k: int
                        ) -> list[tuple[Tensor, Tensor]]:
    """Roll ``predictor`` ``k`` steps under the TRUE action sequence and return
    the per-step ``(z_prev, z_hat)`` latent transitions (the input pairs a step
    readout decodes). Rolls the predictor ONCE so the hierarchical grounding can
    decode SEVERAL per-level step readouts on the SAME rolled latents instead of
    re-rolling per level (the REF-A rollout-reuse pattern). Byte-identical to the
    roll inside :func:`rollout_decode` (unit-pinned)."""
    win_s, win_a = states, actions
    trans: list[tuple[Tensor, Tensor]] = []
    for j in range(k):
        z_hat = predictor(win_s, win_a)[1]               # 1-step head
        trans.append((win_s[:, -1], z_hat))
        if j < k - 1:
            a_next = (future_actions[:, j] if future_actions is not None
                      else win_a[:, -1])
            win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], dim=1)
    return trans


def decode_transitions(step_readout: StepDisplacementReadout,
                       trans: list[tuple[Tensor, Tensor]], k: int
                       ) -> tuple[Tensor, Tensor]:
    """Decode the first ``k`` shared rollout transitions with ``step_readout``,
    accumulate SE(2). Returns ``(waypoints [B, k, 2], step_dpose [B, k, 3])``.
    ``decode_transitions(sr, rollout_transitions(...k), k)`` reproduces
    :func:`rollout_decode` exactly (same latents, same readout)."""
    step_dpose = torch.stack([step_readout(trans[j][0], trans[j][1])
                              for j in range(k)], dim=1)
    return accumulate_se2(step_dpose), step_dpose


# --------------------------------------------------------------------------- #
# H18 hierarchical grounding — one (invdyn, step-readout) pair per brain level #
# --------------------------------------------------------------------------- #
class HierarchicalGrounding(nn.Module):
    """Metric-dynamics grounding heads at EVERY level of the 4-brain hierarchy.

    One ``(MetricInverseDynamics, StepDisplacementReadout)`` pair per level:
      - ``op``  operative  — per-step Δpose (fine, short horizon);
      - ``tac`` tactical   — the 2 s sub-goal trajectory consequence;
      - ``str`` strategic  — the long-horizon coarse place-to-place consequence.

    Every head reads the compact STATE, so the container is state-dim-agnostic
    and shared by the flagship (ViT+readout state) and REF-A (frozen-DINO adapter
    state). Kept OUTSIDE the world model (as in ``finetune_traj``) and saved
    under its own checkpoint key, so a vanilla model still loads a grounded
    checkpoint. The per-level grounding loss is :func:`grounding_losses`.
    """

    LEVELS = ("op", "tac", "str")

    def __init__(self, state_dim: int, hidden: int = 512):
        super().__init__()
        self.invdyn = nn.ModuleDict(
            {lvl: MetricInverseDynamics(state_dim, hidden=hidden)
             for lvl in self.LEVELS})
        self.step = nn.ModuleDict(
            {lvl: StepDisplacementReadout(state_dim, hidden=hidden)
             for lvl in self.LEVELS})


def grounding_losses(grounding: HierarchicalGrounding, predictor,
                     states: Tensor, fut_states: Tensor, actions: Tensor,
                     future_actions: Tensor | None, pose_last: Tensor,
                     future_poses: Tensor, idx_of: dict[int, int],
                     level_cfg: dict[str, tuple[tuple[int, ...], int]],
                     pose_scale: float, invdyn_weight: float = 1.0,
                     fwd_weight: float = 1.0, fwd_step_weight: float = 0.5,
                     invdyn_gradscale: float = 1.0
                     ) -> tuple[Tensor, dict, dict]:
    """H18 grounding at all levels (op/tac/str), mirroring finetune_traj.

    ``level_cfg[level] = (invdyn_horizons, fwd_k)``. Each level contributes:
      (a) metric-inverse-dynamics on REAL latent pairs ``(z_t, z_{t+k})`` at that
          level's horizons -> odometry relative ego-pose (grounds the ENCODER at
          that timescale);
      (b) forward-metric-consistency: decode this level's step readout on the
          predictor's TRUE-action rollout, accumulate SE(2), match the odometry
          ego-trajectory (grounds ENCODER + PREDICTOR + this level's readout).

    The operative predictor is rolled ONCE to ``max(fwd_k)`` and every level
    decodes on the shared transitions. Metre errors are divided by ``pose_scale``
    so the loss weights act on O(1) quantities (finetune_traj convention).
    Returns ``(total_loss, parts, log)``; gradients reach the encoder, the
    predictor, and EACH level's invdyn+step params (test-pinned).

    ``invdyn_gradscale`` (Part A loss-rebalance, 2026-07-18): a straight-through
    gradient scale applied to the ENCODER-latent inputs ``(z_t, fut_states)`` as
    they feed term (a) — and term (a) ONLY. It softly decouples the static
    metric-inverse-dynamics probe from the encoder trunk (``1.0`` = today exactly,
    ``0.0`` = a full probe-detach of (a)) WITHOUT touching (i) the invdyn HEAD
    params (they read the identical forward value, so their gradient is unchanged
    — full-rate readout probes), (ii) term (b) forward-consistency (the rollout
    reads ``states``/``trans``, never the scaled views — the 0.033 m fwd-ADE is
    produced here and stays fully attached), or (iii) JEPA/SIGReg (outside this
    fn). Default ``1.0`` is byte-identical to pre-change."""
    device = states.device
    k_max = max(fwd_k for _, fwd_k in level_cfg.values())
    trans = rollout_transitions(predictor, states, actions, future_actions, k_max)
    z_t = states[:, -1]
    # Part A: soft gradient-decouple ONLY term (a)'s encoder path. z_t_mid feeds
    # the invdyn REAL-PAIR heads; the fut-state partner is scaled at the call
    # site. `trans`/`states` (term b rollout) are read UNSCALED above, so the
    # forward-consistency metric stays fully attached to the encoder.
    z_t_mid = grad_scale(z_t, invdyn_gradscale)
    total = torch.zeros((), device=device)
    parts: dict[str, Tensor] = {}
    log: dict[str, float] = {}
    for lvl, (inv_horizons, fwd_k) in level_cfg.items():
        # (a) metric inverse dynamics on REAL pairs -> grounds the encoder
        loss_mid = torch.zeros((), device=device)
        mid_de = 0.0
        for kh in inv_horizons:
            dpose = grounding.invdyn[lvl](
                z_t_mid, grad_scale(fut_states[:, idx_of[kh - 1]],
                                    invdyn_gradscale))
            tgt = relative_ego_pose(pose_last, future_poses[:, kh - 1])
            loss_mid = loss_mid \
                + ((dpose[..., :2] - tgt[..., :2]) / pose_scale).pow(2).mean() \
                + wrap_angle(dpose[..., 2] - tgt[..., 2]).pow(2).mean()
            mid_de += float((dpose[..., :2] - tgt[..., :2]).detach()
                            .norm(dim=-1).mean())
        loss_mid = loss_mid / len(inv_horizons)
        mid_de /= len(inv_horizons)
        # (b) forward metric consistency on the SHARED rollout
        pred_wp, pred_step = decode_transitions(grounding.step[lvl], trans, fwd_k)
        gt_wp = gt_ego_waypoints(pose_last, future_poses, range(1, fwd_k + 1))
        gt_step = gt_step_dposes(pose_last, future_poses, fwd_k)
        loss_acc = ((pred_wp - gt_wp) / pose_scale).pow(2).mean()
        loss_step = ((pred_step[..., :2] - gt_step[..., :2]) / pose_scale).pow(2).mean() \
            + wrap_angle(pred_step[..., 2] - gt_step[..., 2]).pow(2).mean()
        loss_fwd = loss_acc + fwd_step_weight * loss_step
        fwd_ade = float((pred_wp.detach() - gt_wp).norm(dim=-1).mean())
        total = total + invdyn_weight * loss_mid + fwd_weight * loss_fwd
        parts[f"{lvl}_mid"] = loss_mid
        parts[f"{lvl}_fwd"] = loss_fwd
        log[f"g_{lvl}_mid"] = loss_mid.item()
        log[f"g_{lvl}_fwd"] = loss_fwd.item()
        log[f"g_{lvl}_mid_de_m"] = round(mid_de, 4)
        log[f"g_{lvl}_fwd_ade_m"] = round(fwd_ade, 4)
    return total, parts, log


# --------------------------------------------------------------------------- #
# UNICYCLE STEP READOUT — Option 2: the trajectory decoder, on a FROZEN trunk   #
#                                                                              #
# ⭐ WHAT v1arch's "trajectory head" ACTUALLY IS. `flagship-v1arch-v2bal-30k`   #
# has NO anchored-diffusion head — its checkpoint holds only encoder,          #
# predictor, readout, imagination, inv_dyn and the two policies. The 20        #
# waypoints come from `rollout_decode(predictor, ..., step_readout, k=20)`:    #
# the operative predictor rolled 20 steps in LATENT space, each transition     #
# decoded by `StepDisplacementReadout` into a free (dx, dy, dyaw).             #
# ⇒ The DECODER is the head. It is 6.32 M parameters against a 178 M frozen    #
# encoder+predictor, which is why Option 2 is hours rather than days.          #
#                                                                              #
# ⛔ WHY A FREE (dx, dy, dyaw) DECODE PRODUCES THE MEASURED DEFECTS.           #
# Nothing couples the three channels or ties them to the ego's real speed:     #
#   * dx_j is free, so the implied speed can jump between steps -> jerk RMS    #
#     52.13 against a human 1.71 (MEASURED, 6,834 windows);                    #
#   * dx_1 is free of the ego's true v0 -> a launch transient of 1.98 m/s^2    #
#     against a 0.42 instrument floor;                                         #
#   * dy_j is free, i.e. the decoder may translate the ego SIDEWAYS. A road    #
#     vehicle cannot. Every metre of that is a heading error by construction.  #
#   * dyaw_j is independent of speed, so the decoder can turn while stopped.   #
#                                                                              #
# ⭐ THE UNICYCLE REMOVES ALL FOUR AS REPRESENTABLE STATES, and it does so     #
# while emitting the SAME [B, K, 3] `step_dpose`, so `accumulate_se2` — the    #
# geometry that is already unit-pinned against the GT waypoints — is unchanged.#
# `dy == 0` IS the non-holonomic constraint. That is the whole idea.           #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# The four design choices below are MEASURED, not preferences. All statistics  #
# are the HUMAN's own controls recovered from 39 OOD-val clips x 20 steps      #
# (2026-08-06). Each constant here is a fact about the target distribution.    #
# --------------------------------------------------------------------------- #

# ⛔ WIDENED TO THE TRAIN CORPUS 2026-08-06. The first version of these constants came
# from 39 VAL clips = 780 samples; they now come from 1,500 train episodes / 33,004
# windows / 660,080 samples — 846x more — on the trainer's own stride-8 window grid.
# ⚠️ THE VAL ESTIMATE WAS MATERIALLY WRONG ON THE CHANNEL THAT MATTERS: yaw-rate std
# 0.06930 -> 0.13091, nearly 2x. The val split under-represented turning, and using it
# would have left the yaw-rate channel ~2x under-scaled — the exact conditioning defect
# the scaling exists to remove. A 780-sample estimate of a tailed quantity is not an
# estimate. Artifact: `.../2026-08-06-v1-defect-triage/results/control_stats_train.json`.
#: std of the human's per-step acceleration (m/s^2). MEASURED 0.88361 (train).
ACCEL_SCALE = 0.884
#: std of the human's per-step yaw rate (rad/s). MEASURED 0.13091 (train).
YAWRATE_SCALE = 0.131
#: std of the STEP-TO-STEP CHANGE in each — the natural scale of a delta head.
#: MEASURED (train): accel 0.16537, yaw-rate 0.02548.
#: ⭐ The delta/absolute ratios are 0.187 and 0.195 with lag-1 autocorrelations of +0.983
#: and +0.981 — i.e. the delta-prediction argument is STRONGER on train than it looked on
#: val (where yaw-rate read 0.50 / +0.874). A delta head's output scale is ~5x smaller
#: than a level head's, and that scale IS the jerk.
DACCEL_SCALE = 0.165
DYAWRATE_SCALE = 0.025
#: ego speed normaliser, matching the trainers' convention.
SPEED_SCALE = 10.0


class UnicycleStepReadout(nn.Module):
    """Latent transition -> per-step ``(accel, yaw_rate)``, integrated as a unicycle.

    ⭐ FOUR DESIGN CHOICES, EACH FROM A MEASUREMENT OF THE TARGET DISTRIBUTION. The
    naive version — two raw linear outputs read as (accel, curvature) — is badly
    conditioned in three separate ways, and none of them is visible without looking at
    what the human's controls actually do.

    **(1) PER-CHANNEL OUTPUT SCALING.** MEASURED: accel std **0.80438**, curvature std
    **0.02091** — a **38.5x** scale ratio. One ``Linear(hidden, 2)`` gives both channels
    the same initial gradient scale, so the curvature channel is ~38x under-resolved and
    spends early training being dominated by accel. Each channel is emitted in units of
    its own target std.

    **(2) YAW RATE, NOT CURVATURE.** MEASURED: curvature kurtosis **38.9** (Gaussian is
    3.0) and its low-speed tail is **9.5x** the high-speed one — because
    ``kappa = yaw_rate / v`` explodes as ``v -> 0``. Yaw rate is far better behaved:
    kurtosis **10.4**, low/high-speed tail ratio only **2.3x**. Regressing a target
    whose variance changes 9.5x with an input the head is not even given is a bad
    objective; yaw rate is the well-conditioned one.
    ⛔ **AND THE NON-HOLONOMIC PROPERTY IS KEPT ANYWAY** by bounding the yaw rate at
    ``|v| * curvature_limit``: at ``v = 0`` the bound is 0, so the ego still cannot turn
    in place. Predicting yaw rate WITHOUT that bound would quietly restore the exact
    defect the unicycle exists to remove.

    **(3) SPEED AS AN INPUT.** The conditional distribution of the target depends
    strongly on ``v`` (the 9.5x tail ratio above IS that dependence). The displacement
    readout sees only ``(z_t, z_next)`` and must infer speed from the latents. Feeding
    the carried speed makes the head's job conditional instead of marginal.
    ⚠️ Not privileged information: ``v0`` is already a model input under
    ``speed_input=True``, and the carried ``v`` is that plus the head's own outputs.

    **(4) PREDICT THE DELTA, NOT THE LEVEL.** MEASURED, and this is the biggest one:
    the step-to-step change in accel has std **0.17494** against an absolute std of
    **0.80438** — a **4.6x easier target** — with a lag-1 autocorrelation of **+0.977**.
    Yaw rate is the same story (0.03459 vs 0.06930, corr +0.874).
    ⇒ A delta head's natural output scale IS THE JERK, so smoothness is the DEFAULT
    rather than something a barrier term has to fight the head for. This is the
    structural version of the jerk fix.

    Every option can be switched off, because an arm that changed four things at once
    and improved would be **unattributable** — the same failure as the ``--v2``
    conflation. The ablation is one flag per claim.

    ⛔ **(5) AND THE COUNTER-MEASURE THE FIRST FOUR MADE NECESSARY.** Choices (3) and (4)
    each open a path that BYPASSES THE WORLD MODEL. With a lag-1 autocorrelation of
    **+0.977**, ``a_j ~= a_{j-1}`` is an excellent predictor using **no latents at all**;
    and ``v`` alone reconstructs most of a mostly-straight, mostly-constant-speed
    corpus. A head that took those two shortcuts would be a **driving-dynamics
    predictor wearing a decoder's name** — it would score well and carry none of the
    world model's content, which is the opposite of what this arm is for.
    ⇒ ``shortcut_dropout`` randomly blanks the ``(v, a_prev, yr_prev)`` INPUT columns
    (the integrator still uses the real values — only the head's READ of them is
    dropped), so the latent pathway must carry that information on those samples.
    ⇒ ``detach_feedback`` stops the gradient through the recurrence, so the head cannot
    learn a self-consistent autoregressive trajectory that ignores the latents.
    ⚠️ Dropping the LATENTS instead would be exactly backwards: that teaches robustness
    to missing latents, i.e. it REWARDS the shortcut.

    ⭐ And the head's output is already the RESIDUAL OVER CONSTANT VELOCITY by
    construction — zero output = "hold v0, go straight" — so every departure from CV is
    something the head chose, and `wm_reliance.py` can attribute it.
    """

    def __init__(self, state_dim: int, hidden: int = 512, *,
                 predict_delta: bool = True, speed_input: bool = True,
                 curvature_limit: float = 0.33,
                 shortcut_dropout: float = 0.1, detach_feedback: bool = True):
        super().__init__()
        self.predict_delta = bool(predict_delta)
        self.speed_input = bool(speed_input)
        self.curvature_limit = float(curvature_limit)
        self.shortcut_dropout = float(shortcut_dropout)
        self.detach_feedback = bool(detach_feedback)
        # (3) + (4): the head sees the carried speed and its own previous controls,
        # which is what makes a delta prediction well-posed at all.
        extra = (1 if self.speed_input else 0) + (2 if self.predict_delta else 0)
        self.in_dim = 2 * state_dim + extra
        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # ⛔ Zero-init the OUTPUT layer only. At init the decode is then exactly
        # "hold the true v0, go straight" — kinematically valid. A randomly
        # initialised control head integrates its own noise TWICE and starts from a
        # physically absurd trajectory, which is a far worse basin to descend from.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t: Tensor, z_next: Tensor, v: Tensor,
                a_prev: Tensor, yr_prev: Tensor) -> tuple[Tensor, Tensor]:
        """-> ``(accel [B], yaw_rate [B])`` for this step, in physical units.

        ``v`` is the carried speed; ``a_prev`` / ``yr_prev`` are the previous step's
        emitted controls (zeros at step 0)."""
        # ⛔ The feedback the head READS is detachable and droppable; the values the
        # INTEGRATOR uses are always the real ones. Blanking the read is what forces the
        # latent pathway to carry the information — blanking the integrator would just
        # break the physics.
        a_in = a_prev.detach() if self.detach_feedback else a_prev
        yr_in = yr_prev.detach() if self.detach_feedback else yr_prev
        keep = 1.0
        if self.training and self.shortcut_dropout > 0.0:
            keep = (torch.rand_like(v) >= self.shortcut_dropout).to(v.dtype)
        feat = [z_t, z_next]
        if self.speed_input:
            feat.append((keep * v / SPEED_SCALE).unsqueeze(-1))
        if self.predict_delta:
            feat.append((keep * a_in / ACCEL_SCALE).unsqueeze(-1))
            feat.append((keep * yr_in / YAWRATE_SCALE).unsqueeze(-1))
        raw = self.net(torch.cat(feat, dim=-1))                # [B, 2]
        if self.predict_delta:
            a = a_prev + raw[:, 0] * DACCEL_SCALE
            yr = yr_prev + raw[:, 1] * DYAWRATE_SCALE
        else:
            a = raw[:, 0] * ACCEL_SCALE
            yr = raw[:, 1] * YAWRATE_SCALE
        # (2) the bound that keeps "cannot turn in place" while predicting yaw rate
        cap = v.abs() * self.curvature_limit
        return a, yr.clamp(-cap, cap)

    @classmethod
    def warm_start_from(cls, sr: "StepDisplacementReadout", state_dim: int,
                        hidden: int = 512, **kw) -> "UnicycleStepReadout":
        """Copy the trained trunk's FIRST LINEAR for the latent columns; the extra
        input columns and the output layer stay fresh.

        ⭐ The trunk has already learned to read a latent transition — the expensive
        half. ⚠️ The input width now differs (extra speed / previous-control columns),
        so only the ``2 * state_dim`` latent columns can be copied; the rest are left at
        their init. That partial copy is stated rather than hidden, and the shapes are
        CHECKED — a mismatch raises instead of returning a random module wearing the
        name 'warm-started'.
        """
        m = cls(state_dim, hidden, **kw)
        if sr.net[1].weight.shape[1] != 2 * state_dim:
            raise ValueError(
                f"displacement readout expects {sr.net[1].weight.shape[1]} inputs, not "
                f"2*state_dim={2 * state_dim} — refusing a 'warm start' that is random")
        if sr.net[1].weight.shape[0] != m.net[1].weight.shape[0]:
            raise ValueError(
                f"hidden mismatch {sr.net[1].weight.shape[0]} vs "
                f"{m.net[1].weight.shape[0]} — refusing a random 'warm start'")
        with torch.no_grad():
            m.net[1].weight[:, :2 * state_dim].copy_(sr.net[1].weight)
            m.net[1].bias.copy_(sr.net[1].bias)
            m.net[0].weight[:2 * state_dim].copy_(sr.net[0].weight)
            m.net[0].bias[:2 * state_dim].copy_(sr.net[0].bias)
        return m


def unicycle_step_dpose(controls: Tensor, v0: Tensor, dt: float = 0.1,
                        accel_limit: float | None = None,
                        curvature_limit: float | None = None) -> Tensor:
    """``controls`` [B, K, 2] = (accel, CURVATURE) + ``v0`` [B] -> ``step_dpose``
    [B, K, 3] in the convention :func:`accumulate_se2` consumes.

    ``dx = v*dt``, ``dy = 0`` (non-holonomic), ``dyaw = v*kappa*dt``, then
    ``v += a*dt`` clamped at zero.

    ⛔ THE UPDATE ORDER MATCHES ``accumulate_se2`` AND ``rollout_unicycle``: the step
    advances on the speed and heading held at its START, and ``v`` updates last.
    Integrating with the post-update speed would lengthen every trajectory — an
    over-progress bias, the defect this exists to remove.

    ⚠️ This is the CURVATURE-parameterised primitive, kept for the oracle
    reconstruction and the ablation arm. The trained head emits YAW RATE instead — see
    :class:`UnicycleStepReadout` (2) for the measured reason.
    """
    from tanitad.models.kinematic import _squash

    if controls.dim() != 3 or controls.shape[-1] != 2:
        raise ValueError(f"controls must be [B, K, 2], got {tuple(controls.shape)}")
    b, k, _ = controls.shape
    a, kap = controls[..., 0], controls[..., 1]
    if accel_limit is not None:
        a = _squash(a, accel_limit)
    if curvature_limit is not None:
        kap = _squash(kap, curvature_limit)
    v = torch.as_tensor(v0, dtype=controls.dtype, device=controls.device).reshape(b)
    out = []
    for j in range(k):
        out.append(torch.stack([v * dt, torch.zeros_like(v), v * kap[:, j] * dt], -1))
        v = (v + a[:, j] * dt).clamp_min(0.0)
    return torch.stack(out, dim=1)


def rollout_decode_unicycle(predictor, states: Tensor, actions: Tensor,
                            future_actions: Tensor | None,
                            readout: UnicycleStepReadout, k: int, v0: Tensor,
                            dt: float = 0.1) -> tuple[Tensor, Tensor]:
    """Drop-in for :func:`rollout_decode` with a unicycle decode.

    Returns ``(waypoints [B, k, 2], step_dpose [B, k, 3])`` — identical signature, and
    the waypoints come from the SAME :func:`accumulate_se2`, so nothing downstream
    (metrics, overlays, the four-family instrument) needs to know which decoder ran.

    ⚠️ THE LATENT ROLL IS BYTE-IDENTICAL to :func:`rollout_decode`'s — same predictor,
    same action bookkeeping. Only the decode differs. An ablation between the two
    decoders must not also be an ablation of the rollout, or it is unattributable.

    ⛔ The speed and the previous controls are carried ACROSS steps, which is what makes
    the delta parameterisation well-posed. The loop was already sequential, so this
    costs nothing.
    """
    win_s, win_a = states, actions
    b = states.shape[0]
    v = torch.as_tensor(v0, dtype=states.dtype, device=states.device).reshape(b)
    a_prev = torch.zeros_like(v)
    yr_prev = torch.zeros_like(v)
    out = []
    for j in range(k):
        z_hat = predictor(win_s, win_a)[1]
        a_j, yr_j = readout(win_s[:, -1], z_hat, v, a_prev, yr_prev)
        out.append(torch.stack([v * dt, torch.zeros_like(v), yr_j * dt], dim=-1))
        v = (v + a_j * dt).clamp_min(0.0)
        a_prev, yr_prev = a_j, yr_j
        if j < k - 1:
            a_next = (future_actions[:, j] if future_actions is not None
                      else win_a[:, -1])
            win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], a_next.unsqueeze(1)], dim=1)
    step_dpose = torch.stack(out, dim=1)                  # [B, k, 3]
    return accumulate_se2(step_dpose), step_dpose
