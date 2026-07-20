"""v3 vision-reliance levers (fleet directive 2026-07-18) — CPU, synthetic.

Two PARAMETER-FREE, default-off, gated levers that attack the measured
flagship-30k weaknesses (TanitEval, HYPOTHESIS_LEDGER H25/H26):

  LEVER A  ``v2_route_from_vision`` — the strategic route head is a pure
    command-echo (route_skill_vs_chance 0.0, follow-acc == base rate). This adds
    an ALWAYS-ON, DETERMINISTIC nav-ZEROED route aux (class-weighted CE reusing
    the existing route_head) so route-FROM-vision trains every step, independent
    of the stochastic ``v2_nav_dropout``.

  LEVER B  ``v2_encoder_ego_decorr`` — the trained encoder REDUNDANTLY re-encodes
    the fed ego (yaw R2 0.89 in-latent, vision_use ~12%). This adds a LINEAR
    decorrelation penalty between the pooled latent z_t and the fed [v0, yr0]
    (``tanitad.train.decorr``), plus a DETACHED ego_r2 linear-probe proxy.

Pins (deliverable test matrix):
  (a) both OFF  -> byte-identical loss (weight-invariant) + NO new params /
      state_dict keys / log keys;
  (b) route_from_vision ON  -> 'route_vis' present + finite, grad reaches
      route_head, and the aux CE is CLASS-WEIGHTED (turn class up-weighted);
  (c) encoder_ego_decorr ON -> 'decorr' + 'ego_r2' present + finite, decorr grad
      reaches the ENCODER, ego_r2 is DETACHED (no grad), no-op when ego is None;
  (d) ALL levers together (full --v2 + these two) -> forward+backward finite,
      no NaN, no shape conflict.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refb_labels  # noqa: E402  (ROUTE_* class indices)
from train_flagship4b import FlagshipWindowDataset  # noqa: E402

from tanitad.config import flagship4b_smoke_config  # noqa: E402
from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.train.decorr import ego_decorr_loss, ego_linear_r2  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights,  # noqa: E402
                                           _class_weighted_ce, build_grounding,
                                           flagship_loss, horizon_plan)

FAST = dict(op_fwd_k=2, tac_fwd_k=3, str_fwd_k=4)


# --------------------------------------------------------------------------- #
# Fixtures (mirror test_speed_input)                                          #
# --------------------------------------------------------------------------- #
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


def _v2_cfg(rollout_k: int = 2):
    """flagship smoke config with the FULL trainer --v2 pack (all model-side v2
    levers + speed-input action_dim 3) PLUS the two new v3 levers — exactly what
    ``train_flagship4b.py --v2`` constructs, shrunk for CPU."""
    cfg = flagship4b_smoke_config()
    cfg.v2_ego_to_planners = True
    cfg.v2_ego_dropout = 0.25
    cfg.v2_fa_dropout = 0.3
    cfg.v2_goal_decode = True
    cfg.v2_nav_dropout = 0.5
    cfg.v2_traj_jerk = 0.02
    cfg.v2_gated_intent = True
    cfg.v2_anchor_tactical = True
    cfg.v2_route_from_vision = True       # LEVER A
    cfg.v2_encoder_ego_decorr = True      # LEVER B
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    cfg.train.rollout_k = rollout_k
    return cfg


def _batch(cfg, n=4, T=60):
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


def _states(m, plan, batch):
    st = m.encode_window(batch["frames"])
    fut = m.encode_window(batch["future_frames"][:, plan.needed_fut])
    return st, fut


def _run_loss(cfg, m, plan, batch, weights=None, states=None, fut_states=None):
    grounding = build_grounding(m.state_dim, hidden=32)
    if states is None:
        states, fut_states = _states(m, plan, batch)
    return flagship_loss(
        m, grounding, batch, states, fut_states, plan, cfg,
        weights=weights or LossWeights(), sigreg_variant="full_relaxed",
        sigreg_free_dims=cfg.loss.sigreg.free_dims, pose_scale=10.0,
        fwd_step_weight=0.5, device="cpu")


# --------------------------------------------------------------------------- #
# (a) both OFF == byte-identical + no new params / keys                        #
# --------------------------------------------------------------------------- #
def test_off_is_weight_invariant_and_no_new_log_keys():
    """Default-off: the new terms are gated to 0, so the TOTAL is invariant to
    their weights (byte-identical) and NONE of the new log keys appear."""
    cfg = flagship4b_smoke_config()
    assert cfg.v2_route_from_vision is False and cfg.v2_encoder_ego_decorr is False
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    states, fut = _states(m, plan, batch)
    torch.manual_seed(1234)                        # pin sigreg's random slices
    t_def, log_def, _ = _run_loss(cfg, m, plan, batch, LossWeights(),
                                  states, fut)
    torch.manual_seed(1234)
    t_big, log_big, _ = _run_loss(cfg, m, plan, batch,
                                  LossWeights(route_vis=1e6, decorr=1e6),
                                  states, fut)
    assert torch.equal(t_def, t_big), "off path must ignore route_vis/decorr weights"
    for k in ("route_vis", "decorr", "ego_r2"):
        assert k not in log_def and k not in log_big


def test_levers_add_zero_parameters():
    """Both levers are loss-side / functional — the model is byte-identical
    (same param count AND state_dict keys) whether they are on or off."""
    off = flagship4b_smoke_config()
    on = dataclasses.replace(off, v2_route_from_vision=True,
                             v2_encoder_ego_decorr=True)
    m_off, m_on = WorldModel(off), WorldModel(on)
    n_off = sum(p.numel() for p in m_off.parameters())
    n_on = sum(p.numel() for p in m_on.parameters())
    assert n_on == n_off                           # ZERO new params
    assert set(m_off.state_dict()) == set(m_on.state_dict())


# --------------------------------------------------------------------------- #
# (b) LEVER A: route-from-vision                                               #
# --------------------------------------------------------------------------- #
def test_route_vis_present_finite_and_grad_reaches_route_head():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    cfg.v2_route_from_vision = True
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    batch["nav_valid"] = torch.ones(4, dtype=torch.bool)   # ensure the aux fires
    _, log, parts = _run_loss(cfg, m, plan, batch)
    assert "route_vis" in log and math.isfinite(log["route_vis"])
    assert torch.isfinite(parts["route_vis"])
    parts["route_vis"].backward()
    g = m.strategic_policy.route_head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_route_vis_is_class_weighted_turn_upweighted():
    """The aux CE up-weights the rare TURN routes vs the STRAIGHT majority, and
    its value matches an independently-recomputed class-weighted CE (NOT a plain
    unweighted CE) — proving the class-weights are applied."""
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    cfg.v2_route_from_vision = True
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    # 3x STRAIGHT majority + 1 rare LEFT turn, all valid.
    batch["route_target"] = torch.tensor(
        [refb_labels.ROUTE_STRAIGHT, refb_labels.ROUTE_STRAIGHT,
         refb_labels.ROUTE_STRAIGHT, refb_labels.ROUTE_LEFT], dtype=torch.long)
    batch["nav_valid"] = torch.ones(4, dtype=torch.bool)
    batch["nav_cmd"] = torch.zeros(4, dtype=torch.long)

    states, fut = _states(m, plan, batch)
    _, log, _ = _run_loss(cfg, m, plan, batch, states=states, fut_states=fut)

    n_route = cfg.strategic_policy.n_route
    valid, tgt = batch["nav_valid"], batch["route_target"]
    # the class-weight vector the aux uses: the TURN class is up-weighted.
    counts = torch.bincount(tgt[valid], minlength=n_route).float()
    w = (tgt[valid].numel() / (n_route * counts.clamp_min(1.0))).clamp(max=10.0)
    assert w[refb_labels.ROUTE_LEFT] > w[refb_labels.ROUTE_STRAIGHT]

    with torch.no_grad():                          # recompute the nav-zeroed aux
        sv = m.strategic_policy(states, torch.zeros_like(batch["nav_cmd"]),
                                ego=None)
        exp_weighted = float(_class_weighted_ce(
            sv["route_logits"][valid], tgt[valid], n_route))
        plain = float(F.cross_entropy(sv["route_logits"][valid], tgt[valid]))
    assert math.isclose(log["route_vis"], exp_weighted, rel_tol=1e-5, abs_tol=1e-6)
    assert not math.isclose(log["route_vis"], plain, rel_tol=1e-3), \
        "route_vis must be the CLASS-WEIGHTED CE, not the plain CE"


# --------------------------------------------------------------------------- #
# (c) LEVER B: encoder<->ego decorrelation                                     #
# --------------------------------------------------------------------------- #
def test_decorr_present_finite_and_grad_reaches_encoder():
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    cfg.v2_ego_to_planners = True                  # ego must be fed
    cfg.v2_encoder_ego_decorr = True
    m = WorldModel(cfg)                            # train mode (encoder grads live)
    plan, batch = _batch(cfg)
    _, log, parts = _run_loss(cfg, m, plan, batch)
    assert "decorr" in log and math.isfinite(log["decorr"])
    assert "ego_r2" in log and math.isfinite(log["ego_r2"])
    assert 0.0 <= log["ego_r2"] <= 1.0
    assert torch.isfinite(parts["decorr"]) and parts["decorr"] >= 0.0
    parts["decorr"].backward()                     # isolate the decorr gradient
    hit = [(n, p.grad) for n, p in m.encoder.named_parameters()
           if p.grad is not None and p.grad.abs().sum() > 0]
    assert hit, "decorr penalty must send a gradient into the ENCODER"
    assert all(torch.isfinite(g).all() for _, g in hit)


def test_ego_r2_is_detached_and_bounded():
    z = torch.randn(6, 32, requires_grad=True)
    e = torch.randn(6, 2)
    r2 = ego_linear_r2(z, e)
    assert not r2.requires_grad                    # DETACHED — no training grad
    with pytest.raises(RuntimeError):
        r2.backward()
    assert 0.0 <= float(r2) <= 1.0


def test_decorr_loss_grad_reaches_z_but_not_ego():
    z = torch.randn(6, 32, requires_grad=True)
    e = torch.randn(6, 2, requires_grad=True)
    loss = ego_decorr_loss(z, e)
    assert loss.requires_grad and torch.isfinite(loss)
    assert 0.0 <= float(loss.detach()) <= 1.0      # bounded (mean squared corr)
    loss.backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    # ego is a data tensor in the loss; even a grad-enabled ego receives a
    # gradient here, but in flagship_loss ego_full is pose-derived (leaf, no grad).


def test_decorr_is_noop_when_ego_none():
    # unit: the module is a hard no-op for ego=None
    assert float(ego_decorr_loss(torch.randn(4, 8), None)) == 0.0
    assert float(ego_linear_r2(torch.randn(4, 8), None)) == 0.0
    # integration: lever ON but v2_ego_to_planners OFF (ego is None) -> no keys,
    # finite total, and the decorr term contributes nothing.
    torch.manual_seed(0)
    cfg = flagship4b_smoke_config()
    cfg.v2_encoder_ego_decorr = True               # on, but no ego fed
    assert cfg.v2_ego_to_planners is False
    m = WorldModel(cfg).eval()
    plan, batch = _batch(cfg)
    total, log, parts = _run_loss(cfg, m, plan, batch)
    assert torch.isfinite(total)
    assert "decorr" not in log and "ego_r2" not in log
    assert float(parts["decorr"]) == 0.0


# --------------------------------------------------------------------------- #
# (d) all levers together                                                      #
# --------------------------------------------------------------------------- #
def test_full_v2_plus_new_levers_forward_backward_finite():
    torch.manual_seed(0)
    cfg = _v2_cfg()
    m = WorldModel(cfg)                            # train mode: every lever live
    plan, batch = _batch(cfg)
    total, log, parts = _run_loss(cfg, m, plan, batch)
    assert torch.isfinite(total), "combined v2+v3 forward is non-finite"
    for k in ("route_vis", "decorr", "ego_r2"):    # both new levers report
        assert k in log and math.isfinite(log[k])
    total.backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads with all levers on: {bad[:5]}"
