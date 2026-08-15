"""Kinematic loss terms and the unicycle decode — the retrain path for v1arch.

⛔ WHY THESE EXIST. `v15_losses` supervises trajectories with pure position L1, so
heading, curvature, acceleration and jerk are unconstrained. `test_position_l1_is_blind_
to_heading` is the load-bearing one: it constructs two paths that agree in position to
centimetres while disagreeing in heading, and shows the existing loss cannot tell them
apart while the new terms can. If that test ever fails, the premise of the whole change
is wrong.
"""
import math

import torch

from tanitad.models.kinematic import (A2S_CURVATURE_LIMIT, kinematic_losses,
                                      rollout_unicycle, unicycle_decode,
                                      unicycle_controls_from_path)


def _arc(v=10.0, kappa=0.0, K=20, dt=0.1, B=1):
    c = torch.zeros(B, K, 2, dtype=torch.float64)
    c[..., 1] = kappa
    s0 = torch.tensor([[0.0, 0.0, 0.0, v]], dtype=torch.float64).repeat(B, 1)
    return rollout_unicycle(s0, c, dt)[..., :2]


def test_identical_paths_score_zero():
    p = _arc(kappa=0.04)
    L = kinematic_losses(p, p)
    for k, v in L.items():
        assert float(v) < 1e-9, (k, float(v))


def test_position_l1_is_blind_to_heading():
    """⭐ THE LOAD-BEARING TEST — the premise of the whole change.

    A gently oscillating path and a straight one can agree in POSITION to
    centimetres while their per-step tangents differ every step. The existing
    position-L1 term barely moves; the heading term does. If L1 alone could see
    this, no new loss would be needed."""
    K, v, dt = 20, 10.0, 0.1
    x = torch.arange(1, K + 1, dtype=torch.float64) * v * dt
    straight = torch.stack([x, torch.zeros(K, dtype=torch.float64)], -1)[None]
    # +-2 cm lateral ripple: tiny in position, large in tangent
    ripple = straight.clone()
    ripple[0, :, 1] = 0.02 * torch.tensor([1.0, -1.0] * (K // 2), dtype=torch.float64)

    l1 = float((ripple - straight).abs().mean())
    L = kinematic_losses(ripple, straight)
    # MEASURED: position error 0.0100 m (one centimetre) against a heading error of
    # 2.23 deg. The trained loss sees the first and is blind to the second.
    assert l1 <= 0.011, l1
    assert math.degrees(float(L["heading"])) > 2.0, math.degrees(float(L["heading"]))
    # the ratio is the point: per unit of loss, heading moves ~4x more than position
    assert float(L["heading"]) / max(l1, 1e-9) > 3.0


def test_net_yaw_is_the_sampling_independent_term():
    """net_yaw must react to a genuinely different total turn, and must NOT react
    to the same curve merely sampled at different speeds — that is the property
    that made it the honest lateral metric when re-timing was measured."""
    base = _arc(v=10.0, kappa=0.05)
    same_curve_faster = _arc(v=14.0, kappa=0.05 * 10.0 / 14.0)   # same kappa-per-metre
    different_turn = _arc(v=10.0, kappa=0.10)
    assert float(kinematic_losses(different_turn, base)["net_yaw"]) > \
        float(kinematic_losses(same_curve_faster, base)["net_yaw"])


def test_barriers_are_barriers_not_shrinkage():
    """⛔ Only the EXCESS above the limit is penalised. A plain lambda*jerk**2 would
    punish legitimate emergency braking, i.e. train the arm to be smooth when it
    should be decisive."""
    K = 20
    gentle = torch.zeros(1, K, 2, dtype=torch.float64)
    gentle[..., 0] = 1.0                                  # 1 m/s^2, well inside
    p_gentle = rollout_unicycle(
        torch.tensor([[0.0, 0.0, 0.0, 8.0]], dtype=torch.float64), gentle)[..., :2]
    L = kinematic_losses(p_gentle, p_gentle, accel_limit=2.689, jerk_limit=6.369)
    assert float(L["accel"]) == 0.0 and float(L["jerk"]) == 0.0

    violent = torch.zeros(1, K, 2, dtype=torch.float64)
    violent[0, ::2, 0] = 8.0
    violent[0, 1::2, 0] = -8.0                            # thrash
    p_bad = rollout_unicycle(
        torch.tensor([[0.0, 0.0, 0.0, 12.0]], dtype=torch.float64), violent)[..., :2]
    Lb = kinematic_losses(p_bad, p_bad, accel_limit=2.689, jerk_limit=6.369)
    assert float(Lb["accel"]) > 0.0 and float(Lb["jerk"]) > 0.0


def test_stopped_steps_are_masked_out_of_heading():
    """A stopped ego has no tangent; its heading flips freely. Training against
    that is training against noise."""
    K = 20
    stopped = torch.zeros(1, K, 2, dtype=torch.float64)
    jitter = torch.zeros(1, K, 2, dtype=torch.float64)
    jitter[0, :, 1] = 1e-4 * torch.tensor([1.0, -1.0] * (K // 2), dtype=torch.float64)
    L = kinematic_losses(jitter, stopped)
    assert float(L["heading"]) == 0.0
    assert float(L["net_yaw"]) == 0.0


def test_losses_are_differentiable():
    p = _arc(kappa=0.03).requires_grad_(True)
    t = _arc(kappa=0.05)
    total = sum(kinematic_losses(p, t).values())
    total.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert float(p.grad.abs().max()) > 0.0


# --------------------------------------------------------------------------- #
# unicycle_decode                                                              #
# --------------------------------------------------------------------------- #

def test_zero_delta_reproduces_the_anchor():
    """⛔ The identity property. If a zero correction did not return the anchor, the
    head would have to learn to undo the decode before it could learn anything, and
    every anchor in the vocabulary would be unreachable."""
    base = _arc(v=9.0, kappa=0.04)
    v0 = torch.tensor([9.0], dtype=torch.float64)
    out = unicycle_decode(base, torch.zeros_like(base), v0)
    assert torch.allclose(out, base, atol=2e-3), float((out - base).abs().max())


def test_output_is_feasible_by_construction():
    """A wild delta must still produce a bounded, finite trajectory — that is the
    whole point of composing in control space."""
    base = _arc(v=10.0, kappa=0.02)
    delta = torch.full_like(base, 500.0)
    out = unicycle_decode(base, delta, torch.tensor([10.0], dtype=torch.float64))
    assert torch.isfinite(out).all()
    k = unicycle_controls_from_path(out)[..., 1]
    assert float(k.abs().max()) <= A2S_CURVATURE_LIMIT * 1.5 + 1e-6, float(k.abs().max())


def test_decode_keeps_gradient_when_saturated():
    """⛔ The dead-head trap. rollout_unicycle uses softsign, so a head that
    initialises far outside the bounds can still learn back in. A hard clamp here
    would give exactly zero gradient and the head would never recover."""
    base = _arc(v=10.0)
    delta = torch.full_like(base, 300.0).requires_grad_(True)
    unicycle_decode(base, delta, torch.tensor([10.0], dtype=torch.float64)).sum().backward()
    assert torch.isfinite(delta.grad).all()
    assert float(delta.grad.abs().max()) > 0.0


def test_decode_starts_at_the_true_v0():
    """The launch transient is unrepresentable: the first displacement is v0*dt."""
    base = _arc(v=10.0)
    out = unicycle_decode(base, torch.zeros_like(base),
                          torch.tensor([4.0], dtype=torch.float64))
    assert abs(float(torch.linalg.norm(out[0, 0])) / 0.1 - 4.0) < 1e-6


def test_decode_shape_contract():
    base = _arc()
    try:
        unicycle_decode(base, torch.zeros(1, 19, 2, dtype=torch.float64),
                        torch.zeros(1, dtype=torch.float64))
    except ValueError:
        return
    raise AssertionError("accepted mismatched delta shape")


def test_kin_dt_is_mandatory_and_the_grid_matters():
    """⛔ THE 25x TRAP. `out['traj']` is on the head's HORIZON grid — four waypoints at
    0.5 s for the default (5,10,15,20) — not twenty at 0.1 s. Accel scales as 1/dt^2,
    so assuming 0.1 inflates it 25x and jerk 125x: the barriers would fire on ordinary
    driving and train the arm to crawl. `four_families._DT_CONTRACT` documents the same
    trap; the programme has already published wrong speed numbers to it once.

    A wrong dt does not crash — it silently trains the wrong objective. So the loss
    REFUSES to guess."""
    from tanitad.models.flagship_v15 import v15_losses

    B, N, S = 2, 4, 4
    out = {"anchor_traj": torch.randn(B, N, S, 2),
           "anchor_logits": torch.randn(B, N), "sel_score": torch.randn(B, N),
           "sel_idx": torch.zeros(B, dtype=torch.long)}
    out["traj"] = out["anchor_traj"][:, 0]
    anch, tgt = torch.randn(N, S, 2), torch.randn(B, S, 2)

    # zero weights: dt is irrelevant and must NOT be demanded
    v15_losses(out, anch, tgt)

    try:
        v15_losses(out, anch, tgt, kin_weights={"jerk": 0.1})
    except ValueError as e:
        assert "kin_dt" in str(e)
    else:
        raise AssertionError("accepted a kinematic weight with no kin_dt")

    # and the value genuinely matters: 1/dt^2 on accel between 0.1 and 0.5
    a = v15_losses(out, anch, tgt, kin_weights={"accel": 1.0}, kin_dt=0.1)
    b = v15_losses(out, anch, tgt, kin_weights={"accel": 1.0}, kin_dt=0.5)
    assert float(a["kin_accel"]) > float(b["kin_accel"]) * 5.0, \
        (float(a["kin_accel"]), float(b["kin_accel"]))
    assert a["kin_dt"] == 0.1 and b["kin_dt"] == 0.5


def test_backward_is_finite_on_a_batch_containing_a_stopped_path():
    """⛔ THE TRAP THAT KILLED THE FIRST TRAINING RUN (2026-08-06). atan2 and norm have
    NaN GRADIENTS at exactly (0,0) even though their forward values are fine, and
    masking the OUTPUT does not help: 0 * NaN = NaN in backward. Every eval pass was
    clean — none of them call backward() — and the train corpus contains stopped
    episodes (v = 0.00), so the run NaN'd from its first logged step. Third costume of
    the F-5/6/7 sqrt-relu trap (`kamm_circle_violation`)."""
    K = 20
    pred = torch.zeros(2, K, 2, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        pred[1, :, 0] = torch.arange(1, K + 1, dtype=torch.float64)  # one moving path
    tgt = torch.zeros(2, K, 2, dtype=torch.float64)
    tgt[1, :, 0] = torch.arange(1, K + 1, dtype=torch.float64) * 1.1
    L = kinematic_losses(pred, tgt)
    total = sum(L.values())
    total.backward()
    assert torch.isfinite(total), L
    assert torch.isfinite(pred.grad).all(), "NaN gradient on the stopped path"

    # and through the inverse map, which the accel/jerk barriers use
    pred2 = torch.zeros(1, K, 2, dtype=torch.float64, requires_grad=True)
    unicycle_controls_from_path(pred2).pow(2).sum().backward()
    assert torch.isfinite(pred2.grad).all()
