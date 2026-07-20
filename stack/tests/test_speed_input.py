"""v0 speed-input tests (review 2026-07-18) — CPU, synthetic.

The PROVEN operative fix (flagship-speed 0.628 m vs nospeed 2.918 m; ckpt
forensics: act_emb.0.weight (768, 3)) was trained by a pod-side trainer that
was never committed. These tests pin the committed implementation:
  (a) default-off == pre-change EXACTLY (action_dim 2, no v0 channel appended);
  (b) speed_input + action_dim=3 -> joint loss runs finite, backward reaches
      the widened act_emb;
  (c) SEMANTICS: the appended channel is pose_last[:,3]/10.0 (SPEED_SCALE
      contract), constant across the window AND the future actions — the t=0
      speed only, never a future speed (leakage-safe);
  (d) under v2 the ego-dropout keep-mask zeroes the v0 channel JOINTLY with
      the planner ego vector (training-only; eval feeds the true v0);
  (e) the trainer's wiring path (dataclasses.replace on predictor +
      tactical_pred) threads action_dim=3 into WorldModel.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_flagship4b import FlagshipWindowDataset  # noqa: E402

from tanitad.config import flagship4b_smoke_config  # noqa: E402
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights,  # noqa: E402
                                           build_grounding, flagship_loss,
                                           horizon_plan)

FAST = dict(op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)


def _poses(T: int, dt=0.1, v0=8.0, yaw_rate=0.0, accel=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _episode(T, eid, yaw_rate=0.0, accel=0.0, size=64):
    g = torch.Generator().manual_seed(100 + eid)
    frames = [torch.rand(1, size, size, generator=g) for _ in range(T)]
    poses = _poses(T, yaw_rate=yaw_rate, accel=accel)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, 0.1, eid)


def _speed_cfg(smoke=True, **v2):
    """smoke config + speed_input + action_dim 3 (the trainer's wiring path)."""
    cfg = flagship4b_smoke_config()
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    for k, v in v2.items():
        setattr(cfg, k, v)
    return cfg


def _batch(cfg, n=4, T=260):
    # accelerating episodes so future speeds DIFFER from v0 (leakage guard)
    plan = horizon_plan(cfg, **FAST)
    eps = [_episode(T, 0, yaw_rate=0.06, accel=-1.0),
           _episode(T, 1, yaw_rate=-0.06, accel=0.8),
           _episode(T, 2, accel=-1.2), _episode(T, 3, accel=1.0)]
    ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    batch = default_collate([ds[i] for i in range(n)])
    return plan, batch


def _run_loss(cfg, m, plan, batch):
    grounding = build_grounding(m.state_dim, hidden=32)
    states = m.encode_window(batch["frames"])
    fut_states = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    return flagship_loss(
        m, grounding, batch, states, fut_states, plan, cfg,
        weights=LossWeights(), sigreg_variant="full_relaxed",
        sigreg_free_dims=cfg.loss.sigreg.free_dims, pose_scale=10.0,
        fwd_step_weight=0.5, device="cpu")


def _capture_predictor_actions(m):
    """Wrap the operative predictor's forward to record its actions arg."""
    captured = []
    orig = m.predictor.forward

    def spy(states, actions, **kw):
        captured.append(actions.detach().clone())
        return orig(states, actions, **kw)

    m.predictor.forward = spy
    return captured


# (a) default off == pre-change: 2-channel actions reach the predictor
def test_off_is_prechange_two_channels():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    assert cfg.speed_input is False          # default-off
    m = WorldModel(cfg)
    assert m.predictor.act_emb[0].in_features == 2
    plan, batch = _batch(cfg)
    captured = _capture_predictor_actions(m)
    total, _, _ = _run_loss(cfg, m, plan, batch)
    assert torch.isfinite(total)
    assert all(a.shape[-1] == 2 for a in captured)


# (b) on: 3-channel forward + backward reach the widened act_emb
def test_on_runs_finite_and_backward():
    torch.manual_seed(0)
    cfg = _speed_cfg()
    m = WorldModel(cfg)
    assert m.predictor.act_emb[0].in_features == 3
    plan, batch = _batch(cfg)
    # FiLM is zero-init (identity start) -> at a fresh init NO grad reaches
    # act_emb; nudge the FiLM projections off zero so conditioning is live.
    for blk in m.predictor.blocks:
        torch.nn.init.normal_(blk.film.to_scale_shift.weight, std=0.01)
    captured = _capture_predictor_actions(m)
    total, _, _ = _run_loss(cfg, m, plan, batch)
    assert torch.isfinite(total)
    assert captured and all(a.shape[-1] == 3 for a in captured)
    total.backward()
    g = m.predictor.act_emb[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g[:, 2].abs().sum() > 0           # the v0 column actually trains


# (c) semantics: 3rd channel == pose_last speed / 10, constant, past-only
def test_v0_semantics_constant_t0_leakage_safe():
    torch.manual_seed(0)
    cfg = _speed_cfg()
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    captured = _capture_predictor_actions(m)
    _run_loss(cfg, m, plan, batch)
    v0 = batch["pose_last"][:, 3:4].float() / 10.0            # [B, 1]
    for a in captured:                       # every predictor call (incl. roll)
        chan = a[..., 2]                     # [B, K]
        assert torch.allclose(chan, v0.expand_as(chan), atol=1e-6)
    # leakage guard: accelerating episodes -> future speeds differ from v0,
    # so a constant==v0 channel proves no future speed was fed
    fut_speed = batch["future_poses"][:, -1, 3:4].float() / 10.0
    assert (fut_speed - v0).abs().max() > 1e-3


# (d) v2 ego-dropout zeroes the v0 channel jointly (training-only)
def test_ego_dropout_joint_zeroes_v0_channel():
    torch.manual_seed(0)
    cfg = _speed_cfg(v2_ego_to_planners=True, v2_ego_dropout=1.0)
    m = WorldModel(cfg)
    plan, batch = _batch(cfg)

    m.train()
    captured = _capture_predictor_actions(m)
    _run_loss(cfg, m, plan, batch)
    assert captured and all((a[..., 2] == 0).all() for a in captured), \
        "p=1.0 ego-dropout must zero the v0 action channel in training"

    m.eval()                                 # eval: true v0 always fed
    captured.clear()
    _run_loss(cfg, m, plan, batch)
    v0 = batch["pose_last"][:, 3:4].float() / 10.0
    assert captured and all(
        torch.allclose(a[..., 2], v0.expand_as(a[..., 2]), atol=1e-6)
        for a in captured)


# (e) meta-device build: the trainer's replace() path threads action_dim=3
def test_trainer_wiring_path_threads_action_dim():
    cfg = _speed_cfg()
    with torch.device("meta"):
        m = WorldModel(cfg)
    assert m.predictor.act_emb[0].in_features == 3
    if getattr(cfg, "tactical_pred", None) is not None:
        assert m.tactical_pred.act_emb[0].in_features == 3
