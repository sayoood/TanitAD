"""The unicycle (accel, curvature) action space — leverage idea #1's kernel.

⭐ WHY THESE TESTS AND NOT AN EVAL. The measured motivation is that our arm's
`accel_mae_mps2` is 1.7644 against Alpamayo's 0.5077 (39 paired OOD-val clips,
2026-08-06). Before any GPU is spent on a control head, the integrator and its
inverse must be proven to ROUND-TRIP — otherwise a later training result would be
uninterpretable: a worse arm could mean a worse action space or a wrong integrator,
and nothing would separate them.

The round-trip is the load-bearing test. Everything else pins a trap that has a name.
"""
import torch

from tanitad.models.kinematic import (
    A2S_ACCEL_LIMIT, A2S_CURVATURE_LIMIT, entry_speed_mismatch, rollout_bicycle,
    rollout_unicycle, unicycle_controls_from_path)


def test_roundtrip_path_to_controls_to_path():
    """⭐ THE TEST. Integrate controls -> path -> recover controls -> integrate again.
    The two paths must agree, or every downstream conclusion about the action space is
    confounded by the integrator."""
    torch.manual_seed(0)
    B, K = 4, 20
    controls = torch.stack([
        torch.linspace(-1.5, 1.5, K).repeat(B, 1),           # accel ramp
        torch.linspace(-0.05, 0.05, K).repeat(B, 1),          # gentle curvature sweep
    ], dim=-1)
    v0 = torch.full((B,), 12.0)
    state0 = torch.stack([torch.zeros(B), torch.zeros(B), torch.zeros(B), v0], dim=-1)

    path = rollout_unicycle(state0, controls)[..., :2]
    recovered = unicycle_controls_from_path(path)
    path2 = rollout_unicycle(state0, recovered)[..., :2]

    # ⚠️ Tolerance is on the PATH, not the controls. The inverse recovers CHORD-average
    # controls (documented), so control-space equality is not the contract; producing
    # the same trajectory is.
    # ⛔ This is the test that caught the OFF-BY-ONE: because the integrator advances
    # position on the PRE-update speed, step k's displacement reveals the speed before
    # accel[k]. A naive `(speed[k] - speed[k-1])/dt` inverse drifted 1.2233 m over 2 s
    # and every recovered control was shifted one step late.
    assert torch.allclose(path, path2, atol=1e-3), \
        f"round-trip drift {float((path - path2).abs().max()):.4f} m"


def test_straight_line_has_zero_curvature_and_zero_accel():
    """A constant-speed straight path must imply no turn and no acceleration —
    the negative control that would catch a sign or axis error."""
    K, v = 20, 10.0
    path = torch.stack([torch.arange(1, K + 1) * v * 0.1, torch.zeros(K)], dim=-1)[None]
    c = unicycle_controls_from_path(path)
    assert c[..., 0].abs().max() < 1e-4, "straight constant-speed path implies accel"
    assert c[..., 1].abs().max() < 1e-4, "straight path implies curvature"


def test_launch_transient_is_its_own_number_not_folded_into_accel():
    """⛔ THE LAUNCH TRANSIENT IS THE DEFECT WE ARE HUNTING, and it must not hide
    inside the accel profile. A path that starts at 10 m/s from an entry speed of
    5 m/s is not reachable by ANY control under this integrator — the first
    displacement is forced to use v0. So the recovered accel is legitimately ~0 (the
    path really is constant-speed) and the discontinuity surfaces separately."""
    K, v0, v_path = 20, 5.0, 10.0
    path = torch.stack([torch.arange(1, K + 1) * v_path * 0.1, torch.zeros(K)], dim=-1)[None]
    c = unicycle_controls_from_path(path)
    assert c[..., 0].abs().max() < 1e-4, "constant-speed path implies no acceleration"
    # (10 - 5) m/s over one 0.1 s step -> 50 m/s^2 of unreachable launch transient
    assert abs(float(entry_speed_mismatch(path, torch.tensor([v0]))[0]) - 50.0) < 1e-3


def test_entry_speed_mismatch_is_zero_when_the_path_agrees_with_v0():
    """The negative control: an arm that leaves the origin at exactly the ego's speed
    has no transient, and must not be charged for one."""
    K, v = 20, 7.5
    path = torch.stack([torch.arange(1, K + 1) * v * 0.1, torch.zeros(K)], dim=-1)[None]
    assert abs(float(entry_speed_mismatch(path, torch.tensor([v]))[0])) < 1e-3


def test_yaw_rate_is_v_times_curvature():
    """The defining identity. A stopped vehicle cannot turn — that is correct physics,
    not an edge case to patch around."""
    B, K = 2, 5
    controls = torch.zeros(B, K, 2)
    controls[..., 1] = 0.1                                     # constant curvature
    moving = torch.tensor([[0.0, 0.0, 0.0, 10.0]]).repeat(B, 1)
    stopped = torch.tensor([[0.0, 0.0, 0.0, 0.0]]).repeat(B, 1)
    yaw_moving = rollout_unicycle(moving, controls)[..., 2]
    yaw_stopped = rollout_unicycle(stopped, controls)[..., 2]
    assert torch.allclose(yaw_moving[:, 0], torch.full((B,), 10.0 * 0.1 * 0.1))
    assert float(yaw_stopped.abs().max()) == 0.0, "a stopped unicycle must not turn"


def test_limits_saturate_smoothly_and_keep_gradient():
    """⛔ A hard clamp has ZERO gradient outside the range, so a head initialised
    outside it can never learn back in — the silent dead-head. This pins that the
    squash bounds the control WITHOUT reintroducing that cliff."""
    controls = torch.zeros(1, 3, 2, requires_grad=True)
    with torch.no_grad():
        controls[..., 0] = 500.0                               # far outside the limit
        controls[..., 1] = 50.0
    state0 = torch.tensor([[0.0, 0.0, 0.0, 8.0]])
    out = rollout_unicycle(state0, controls, accel_limit=A2S_ACCEL_LIMIT,
                           curvature_limit=A2S_CURVATURE_LIMIT)
    out.sum().backward()
    assert torch.isfinite(controls.grad).all()
    # ⛔ THIS IS THE TEST THAT REJECTED tanh. MEASURED 2026-08-06: at this 51x
    # overshoot `1 - tanh(51)**2` is EXACTLY 0.0 in float32 — tanh does not remove the
    # dead-head cliff, it moves it out to where nobody tests. Softsign's 1/x^2 decay
    # leaves ~3.7e-4 here: small, representable, and recoverable.
    assert float(controls.grad.abs().max()) > 0.0, "saturated head has no gradient"
    # and the integrated speed respects the bound: v0 + a_max*dt per step
    v = out.detach()[0, :, 3]
    assert float(v[0]) <= 8.0 + A2S_ACCEL_LIMIT * 0.1 + 1e-4


def test_unlimited_is_the_default_and_passes_controls_through():
    """A head that bounds its own output must not be squashed twice."""
    controls = torch.zeros(1, 2, 2)
    controls[..., 0] = 3.0
    state0 = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    v = rollout_unicycle(state0, controls)[0, :, 3]
    assert abs(float(v[0]) - 0.3) < 1e-6                       # 3 m/s^2 * 0.1 s
    assert abs(float(v[1]) - 0.6) < 1e-6


def test_speed_never_goes_negative():
    """Braking past a standstill must stop, not reverse — `clamp_min(0)` matches
    `rollout_bicycle`, and reversing would silently corrupt along-track metrics."""
    controls = torch.zeros(1, 10, 2)
    controls[..., 0] = -20.0
    state0 = torch.tensor([[0.0, 0.0, 0.0, 5.0]])
    st = rollout_unicycle(state0, controls)
    assert float(st[..., 3].min()) >= 0.0
    assert float(st[0, -1, 0]) >= float(st[0, -2, 0]), "ego moved backwards"


def test_agrees_with_bicycle_on_the_equivalent_control():
    """⛔ The two integrators must differ ONLY in the action space, never in the
    integration order — otherwise a bicycle-vs-unicycle ablation measures the
    integrator. curvature = tan(steer)/L is the exact correspondence."""
    B, K, L = 3, 12, 2.7
    steer = torch.full((B, K), 0.08)
    accel = torch.full((B, K), 0.5)
    state0 = torch.tensor([[0.0, 0.0, 0.0, 9.0]]).repeat(B, 1)
    bike = rollout_bicycle(state0, torch.stack([accel, steer], -1), wheelbase=L)
    uni = rollout_unicycle(state0, torch.stack([accel, torch.tan(steer) / L], -1))
    assert torch.allclose(bike, uni, atol=1e-5), \
        f"integrators diverge by {float((bike - uni).abs().max()):.2e}"


def test_shape_contract_is_enforced_not_broadcast():
    """A [B, K, 3] control silently broadcasting would be a wrong-answer-shaped bug."""
    for bad in (torch.zeros(1, 4, 3), torch.zeros(1, 4, 1)):
        try:
            rollout_unicycle(torch.zeros(1, 4), bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted controls of shape {tuple(bad.shape)}")


def test_stopped_ego_implies_zero_curvature_not_a_huge_number():
    """⛔ THE TRAP THIS PINS, and it produced a real number before it was caught.
    At v ~ 0 the heading change is zero under ANY curvature, so curvature is
    UNDETERMINED — but `dh / ds` with a tiny ds returns something enormous. MEASURED
    2026-08-06 over the 39 paired OOD-val clips, the ungated inverse reported an
    implied-curvature MAE of 1.6e6 and 7.6e3 1/m for the two arms. That is not a large
    error, it is a meaningless one, and it was on its way into a report."""
    K = 20
    stopped = torch.zeros(1, K, 2)                             # ego never moves
    c = unicycle_controls_from_path(stopped)
    assert float(c[..., 1].abs().max()) == 0.0, "stopped ego given a curvature"
    assert torch.isfinite(c).all()

    # a crawling, jittery path is the same class of trap and must also be gated
    torch.manual_seed(1)
    crawl = torch.cumsum(torch.randn(1, K, 2) * 1e-3, dim=1)
    assert float(unicycle_controls_from_path(crawl)[..., 1].abs().max()) == 0.0


def test_moving_curvature_is_unaffected_by_the_gate():
    """The negative control: the gate must not silently zero a real turn."""
    B, K = 2, 20
    controls = torch.zeros(B, K, 2)
    controls[..., 1] = 0.05
    state0 = torch.tensor([[0.0, 0.0, 0.0, 12.0]]).repeat(B, 1)
    path = rollout_unicycle(state0, controls)[..., :2]
    rec = unicycle_controls_from_path(path)
    assert abs(float(rec[0, 0, 1]) - 0.05) < 1e-3, float(rec[0, 0, 1])
