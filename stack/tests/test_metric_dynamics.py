"""fix-B1 unit tests: metric-dynamics grounding + fine-tune step (CPU, fast).

Covers the pivot's required checks:
  - SE(2) accumulation of TRUE per-step Δposes reproduces the GT ego waypoints
    (straight AND curved synthetic episodes) — the rollout geometry is correct;
  - metric-inverse-dynamics gradient reaches the ENCODER from the metric loss
    ALONE (the grounding signal actually shapes the latent);
  - the forward-consistency rollout loss reaches BOTH encoder and predictor;
  - a full fine-tune STEP runs with finite losses on synthetic episodes in both
    --mode dynamics and --mode head;
  - D2-relevant outputs (imagined next-latent) are still produced after a step;
  - head/readout output shapes and horizons.
"""

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from finetune_traj import build_heads, compute_losses  # noqa: E402

from tanitad.config import smoke_config  # noqa: E402
from tanitad.data._contract import EpisodeWindowDataset  # noqa: E402
from tanitad.data.toy_driving import ToyEpisode, generate_episode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.models.metric_dynamics import (MetricInverseDynamics,  # noqa: E402
                                            StepDisplacementReadout,
                                            accumulate_se2, ego_delta,
                                            gt_ego_waypoints, gt_step_dposes,
                                            relative_ego_pose, rollout_decode,
                                            wrap_angle)
from tanitad.models.traj_head import TrajectoryHead  # noqa: E402
from tanitad.train.train_worldmodel import _needed_future_indices  # noqa: E402

K = 20


def _batch(n=4, steps=60, max_h=max(K, 6)):
    cfg = smoke_config()
    eps = [generate_episode(i, steps=steps, size=cfg.encoder.image_size)
           for i in range(n)]
    ds = EpisodeWindowDataset(eps, window=cfg.predictor.window, max_horizon=max_h)
    return cfg, default_collate([ds[i] for i in range(min(8, len(ds)))])


# --------------------------------------------------------------------------- #
# Geometry: accumulation reproduces GT waypoints (straight + curved)          #
# --------------------------------------------------------------------------- #
def _straight_poses(T=40, v=3.0):
    t = torch.arange(T, dtype=torch.float32)
    x = v * t
    return torch.stack([x, torch.zeros(T), torch.zeros(T),
                        torch.full((T,), v)], dim=-1)


def _curved_poses(T=40, speed=3.0, omega=0.05):
    poses = [torch.tensor([0.0, 0.0, 0.0, speed])]
    for _ in range(T - 1):
        x, y, yaw, v = poses[-1]
        nyaw = yaw + omega
        poses.append(torch.tensor([x + speed * math.cos(yaw),
                                   y + speed * math.sin(yaw), nyaw, speed]))
    return torch.stack(poses)


def test_accumulate_matches_gt_straight_and_curved():
    for poses in (_straight_poses(), _curved_poses()):
        last = 5
        pose_last = poses[last:last + 1]                     # [1,4]
        future = poses[last + 1:last + 1 + K].unsqueeze(0)   # [1,K,4]
        gt_wp = gt_ego_waypoints(pose_last, future, range(1, K + 1))
        step_dp = gt_step_dposes(pose_last, future, K)
        acc = accumulate_se2(step_dp)
        assert torch.allclose(acc, gt_wp, atol=1e-4), \
            (acc - gt_wp).abs().max().item()


def test_relative_ego_pose_roundtrip():
    # displacement straight ahead in ego frame == distance along x, zero y.
    a = torch.tensor([[1.0, 2.0, math.pi / 2, 5.0]])         # heading +y
    b = torch.tensor([[1.0, 5.0, math.pi / 2, 5.0]])         # moved +3 in world y
    d = relative_ego_pose(a, b)                              # ego frame of a
    assert torch.allclose(d[0, :2], torch.tensor([3.0, 0.0]), atol=1e-5)
    assert abs(float(d[0, 2])) < 1e-6
    # ego_delta of zero displacement is zero
    assert torch.allclose(ego_delta(torch.zeros(1, 2), torch.tensor([0.3])),
                          torch.zeros(1, 2))


# --------------------------------------------------------------------------- #
# Heads: shapes / horizons                                                     #
# --------------------------------------------------------------------------- #
def test_head_and_readout_shapes():
    S, B = 512, 4
    mid = MetricInverseDynamics(S)
    step = StepDisplacementReadout(S)
    z = torch.randn(B, S)
    assert mid(z, z).shape == (B, 3)
    assert step(z, z).shape == (B, 3)
    head = TrajectoryHead(S, horizons=(5, 10, 15, 20))
    assert head(torch.randn(B, S)).shape == (B, 4, 2)
    head2 = TrajectoryHead(S, horizons=(5, 10, 15, 20), n_extra_states=2)
    out = head2(torch.randn(B, S), [torch.randn(B, S), torch.randn(B, S)])
    assert out.shape == (B, 4, 2)


# --------------------------------------------------------------------------- #
# Gradient: metric-inverse-dynamics ALONE reaches the encoder                  #
# --------------------------------------------------------------------------- #
def test_metric_invdyn_grad_reaches_encoder():
    torch.manual_seed(0)
    cfg, batch = _batch()
    world = WorldModel(cfg)
    mid = MetricInverseDynamics(world.state_dim)
    needed_fut, idx_of = _needed_future_indices(cfg)

    frames = batch["frames"]
    fut = batch["future_frames"]
    future_poses = batch["future_poses"].float()
    pose_last = batch["pose_last"].float()

    states = world.encode_window(frames)
    fut_states = world.encode_window(fut[:, needed_fut])
    z_t = states[:, -1]
    dpose = mid(z_t, fut_states[:, idx_of[0]])               # k=1 pair
    tgt = relative_ego_pose(pose_last, future_poses[:, 0])
    loss = ((dpose[..., :2] - tgt[..., :2]).pow(2).mean()
            + wrap_angle(dpose[..., 2] - tgt[..., 2]).pow(2).mean())

    world.zero_grad(); mid.zero_grad()
    loss.backward()
    g = world.encoder.patch.weight.grad
    assert g is not None and float(g.abs().sum()) > 0, \
        "metric-invdyn loss did not propagate into the encoder"


def test_forward_rollout_grad_reaches_encoder_and_predictor():
    torch.manual_seed(0)
    cfg, batch = _batch()
    world = WorldModel(cfg)
    step_ro = StepDisplacementReadout(world.state_dim)
    fwd_k = 6

    states = world.encode_window(batch["frames"])
    actions = batch["actions"]
    fut_actions = batch["future_actions"]
    future_poses = batch["future_poses"].float()
    pose_last = batch["pose_last"].float()

    pred_wp, _ = rollout_decode(world.predictor, states, actions, fut_actions,
                                step_ro, fwd_k)
    gt_wp = gt_ego_waypoints(pose_last, future_poses, range(1, fwd_k + 1))
    loss = (pred_wp - gt_wp).pow(2).mean()
    world.zero_grad(); step_ro.zero_grad()
    loss.backward()
    ge = world.encoder.patch.weight.grad
    gp = world.predictor.heads["1"].weight.grad
    assert ge is not None and float(ge.abs().sum()) > 0, "no encoder grad"
    assert gp is not None and float(gp.abs().sum()) > 0, "no predictor grad"


# --------------------------------------------------------------------------- #
# Full fine-tune step: finite losses, both modes; D2 outputs preserved         #
# --------------------------------------------------------------------------- #
def _step_and_check(mode):
    torch.manual_seed(0)
    cfg, batch = _batch()
    world = WorldModel(cfg)
    heads = build_heads(mode, world.state_dim, cfg, "cpu")
    needed_fut, idx_of = _needed_future_indices(cfg)
    params = list(world.parameters())
    for h in heads.values():
        params += list(h.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)

    total, log, parts = compute_losses(
        world, heads, batch, cfg, needed_fut, idx_of, mode=mode,
        invdyn_weight=2.0, traj_weight=1.0, fwd_k=6, fwd_step_weight=0.5,
        mid_horizons=list(cfg.predictor.horizons), pose_scale=10.0, device="cpu")

    assert torch.isfinite(total), f"{mode}: non-finite total loss"
    for name, v in parts.items():
        assert torch.isfinite(v).all(), f"{mode}: non-finite part {name}"
    # SSL core terms present and finite (D2 protection: JEPA + SigReg kept)
    for k in ("pred", "sigreg", "inv"):
        assert k in log and math.isfinite(log[k])

    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

    # D2-relevant output still produced: imagined next-latent finite, right shape
    with torch.no_grad():
        states = world.encode_window(batch["frames"])
        z_imag1 = world.imagine(states, batch["actions"])[1]
    assert z_imag1.shape == (batch["frames"].shape[0], world.state_dim)
    assert torch.isfinite(z_imag1).all()
    return log


def test_finetune_step_dynamics():
    log = _step_and_check("dynamics")
    assert "mid" in log and "fwd" in log
    assert "fwd_ade_m" in log and math.isfinite(log["fwd_ade_m"])


def test_finetune_step_head():
    log = _step_and_check("head")
    assert "traj" in log and math.isfinite(log["traj_ade_m"])
