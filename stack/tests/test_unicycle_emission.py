"""W4 :class:`UnicycleEmission` standalone smoke — shapes, bounds, convention.

CPU-only, no checkpoint, no corpus: this is the part of
``train_v58f_unicycle_head.py`` that IS runnable off-pod. The full-model path
(frozen v5f trunk + head + v2 corpora) is pod-side and is NOT exercised here.

What is pinned:
  * output shapes and finiteness;
  * feasibility BY CONSTRUCTION: |a| <= 4 m/s^2, |kappa| <= 0.2 1/m, and the
    derived census criterion |yaw_rate| = |kappa*v| <= 0.33*v + 0.05;
  * zero-init warm start == constant-velocity straight rollout (the CV
    baseline), against a hand-derived closed form;
  * the discretisation MATCHES the v1.6/v1.7 convention
    (train_unicycle_readout.py:247-260 through accumulate_se2), verified
    against an INDEPENDENT hand-rolled python loop — not against the function
    under test;
  * gradients reach the emission MLP through the integrated waypoints.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_v58f_unicycle_head import (A_MAX, DT, KAPPA_MAX,  # noqa: E402
                                      UnicycleEmission, unicycle_rollout)

B, N, K, F = 2, 5, 20, 384


def _emission(feat_dim=F, seed=0):
    torch.manual_seed(seed)
    m = UnicycleEmission(feat_dim=feat_dim, k=K)
    # perturb the zero-init final layer so bounds are exercised off the origin
    with torch.no_grad():
        for p in m.net[-1].parameters():
            p.normal_(std=2.0)
    return m


def test_shapes_finite_and_bounded():
    m = _emission()
    feat = torch.randn(B, N, F)
    v0 = torch.rand(B) * 15.0
    a, kappa, wp = m(feat, v0)
    assert a.shape == (B, N, K)
    assert kappa.shape == (B, N, K)
    assert wp.shape == (B, N, K, 2)
    assert torch.isfinite(wp).all() and torch.isfinite(a).all() \
        and torch.isfinite(kappa).all()
    assert (a.abs() <= A_MAX + 1e-5).all()
    assert (kappa.abs() <= KAPPA_MAX + 1e-5).all()


def test_census_feasibility_by_construction():
    """|a|<=4 AND |yaw_rate|<=0.33*v+0.05 — the W2 census criteria the v5f fan
    violates on ~97 % of steps must hold on EVERY emitted step here."""
    m = _emission(seed=1)
    a, kappa, _ = m(torch.randn(B, N, F) * 3, torch.rand(B) * 20.0)
    _, v_pre = unicycle_rollout(a, kappa, torch.rand(B) * 20.0)
    yaw_rate = kappa * v_pre
    assert (a.abs() <= A_MAX + 1e-5).all()
    assert (yaw_rate.abs() <= 0.33 * v_pre + 0.05 + 1e-5).all()


def test_zero_init_is_constant_velocity_straight():
    """Fresh module (zero-init final layer) -> a=kappa=0 -> wp_k = (v0*dt*k, 0)."""
    torch.manual_seed(0)
    m = UnicycleEmission(feat_dim=F, k=K)          # NO perturbation
    v0 = torch.tensor([3.0, 12.5])
    a, kappa, wp = m(torch.randn(B, N, F), v0)
    assert torch.allclose(a, torch.zeros_like(a))
    assert torch.allclose(kappa, torch.zeros_like(kappa))
    ks = torch.arange(1, K + 1, dtype=torch.float32)
    expect_x = v0[:, None, None] * DT * ks[None, None]
    assert torch.allclose(wp[..., 0], expect_x, atol=1e-5)
    assert torch.allclose(wp[..., 1], torch.zeros_like(wp[..., 1]), atol=1e-6)


def test_rollout_matches_readout_convention_hand_loop():
    """Independent scalar re-derivation of the train_unicycle_readout.py:247-260
    discretisation (translate at PREVIOUS heading with PRE-update speed, then
    turn, then update speed) against unicycle_rollout."""
    torch.manual_seed(2)
    a = torch.randn(1, 1, K).clamp(-A_MAX, A_MAX)
    kappa = torch.randn(1, 1, K).mul(0.05).clamp(-KAPPA_MAX, KAPPA_MAX)
    v0 = torch.tensor([7.0])
    wp, v_pre = unicycle_rollout(a, kappa, v0)
    # hand loop
    v, psi, x, y = 7.0, 0.0, 0.0, 0.0
    for k in range(K):
        dx = v * DT
        dyaw = float(kappa[0, 0, k]) * v * DT
        x += dx * math.cos(psi)
        y += dx * math.sin(psi)
        psi += dyaw
        assert abs(v - float(v_pre[0, 0, k])) < 1e-5
        v = max(0.0, v + float(a[0, 0, k]) * DT)
        assert abs(x - float(wp[0, 0, k, 0])) < 1e-4, f"x at step {k}"
        assert abs(y - float(wp[0, 0, k, 1])) < 1e-4, f"y at step {k}"


def test_speed_clamps_at_zero():
    """Hard braking cannot integrate through v=0 into reverse."""
    a = torch.full((1, 1, K), -A_MAX)
    kappa = torch.zeros(1, 1, K)
    wp, v_pre = unicycle_rollout(a, kappa, torch.tensor([1.0]))
    assert (v_pre >= 0.0).all()
    dx = torch.diff(torch.cat([torch.zeros(1, 1, 1), wp[..., 0]], dim=-1),
                    dim=-1)
    assert (dx >= -1e-6).all()          # never moves backwards


def test_gradients_reach_the_mlp():
    m = _emission(seed=3)
    feat = torch.randn(B, N, F)
    _, _, wp = m(feat, torch.rand(B) * 10.0)
    wp.pow(2).sum().backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_anchor_fallback_dim():
    """--cond anchor conditions on flattened anchor_traj (K*2) + v0."""
    m = _emission(feat_dim=K * 2, seed=4)
    fan = torch.randn(B, N, K, 2)
    a, kappa, wp = m(fan.flatten(2), torch.rand(B) * 10.0)
    assert wp.shape == (B, N, K, 2)
    assert (a.abs() <= A_MAX + 1e-5).all()
