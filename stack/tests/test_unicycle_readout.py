"""Option 2 — the unicycle trajectory decoder on a frozen trunk.

⛔ WHAT THIS REPLACES AND WHY. `flagship-v1arch-v2bal-30k` has no trajectory head; its
20 waypoints come from `rollout_decode(predictor, ..., StepDisplacementReadout, k=20)`,
which emits a FREE (dx, dy, dyaw) per step. Nothing couples the three channels or ties
them to the ego's real speed, and all four measured defects follow directly. The
load-bearing tests here are the ones that show those states are now UNREPRESENTABLE —
`test_no_sideways_translation`, `test_cannot_turn_in_place`,
`test_first_step_is_exactly_v0` — because a loss term can be out-weighted, but a
representation cannot be argued with.
"""
import math

import torch
from torch import nn

from tanitad.models.metric_dynamics import (StepDisplacementReadout,
                                            UnicycleStepReadout,
                                            accumulate_se2,
                                            rollout_decode,
                                            rollout_decode_unicycle,
                                            unicycle_step_dpose)


class _Predictor(nn.Module):
    """Minimal stand-in: window of states -> next latent. The tuple shape matches
    what `rollout_decode` indexes ([1] is the 1-step head)."""

    def __init__(self, s):
        super().__init__()
        self.lin = nn.Linear(s, s)

    def forward(self, win_s, win_a):
        return None, self.lin(win_s[:, -1])


def test_first_step_is_exactly_v0():
    """⭐ The launch transient becomes UNREPRESENTABLE. MEASURED on v1arch: 1.9825 m/s²
    against a 0.5487 instrument floor, because dx_1 is free of the ego's real speed."""
    B, K = 3, 20
    v0 = torch.tensor([4.0, 11.0, 0.0], dtype=torch.float64)
    ctrl = torch.randn(B, K, 2, dtype=torch.float64) * 3.0
    dp = unicycle_step_dpose(ctrl, v0, dt=0.1)
    assert torch.allclose(dp[:, 0, 0], v0 * 0.1, atol=1e-12)


def test_no_sideways_translation():
    """⛔ THE NON-HOLONOMIC CONSTRAINT, and the one a free decoder violates silently.
    A road vehicle cannot translate sideways; every metre of `dy` a free readout emits
    is a heading error by construction."""
    ctrl = torch.randn(4, 20, 2, dtype=torch.float64) * 5.0
    dp = unicycle_step_dpose(ctrl, torch.full((4,), 9.0, dtype=torch.float64))
    assert float(dp[..., 1].abs().max()) == 0.0


def test_cannot_turn_in_place():
    """yaw_rate = v*kappa, so a stopped ego cannot rotate. A free (dx,dy,dyaw) decode
    can, and nothing in a position loss objects."""
    ctrl = torch.zeros(1, 10, 2, dtype=torch.float64)
    ctrl[..., 1] = 0.3                                  # large curvature demanded
    dp = unicycle_step_dpose(ctrl, torch.zeros(1, dtype=torch.float64))
    assert float(dp[..., 2].abs().max()) == 0.0
    assert float(dp[..., 0].abs().max()) == 0.0         # and it does not move


def test_speed_never_reverses():
    ctrl = torch.zeros(1, 20, 2, dtype=torch.float64)
    ctrl[..., 0] = -30.0                                 # brake far past a standstill
    dp = unicycle_step_dpose(ctrl, torch.tensor([5.0], dtype=torch.float64))
    assert float(dp[..., 0].min()) >= 0.0


def test_constant_speed_zero_curvature_is_a_straight_line():
    """The negative control: zero controls from v0 must give a straight constant-speed
    path — which is exactly what a zero-initialised readout emits at step 0."""
    K, v = 20, 8.0
    dp = unicycle_step_dpose(torch.zeros(1, K, 2, dtype=torch.float64),
                             torch.tensor([v], dtype=torch.float64))
    wp = accumulate_se2(dp)
    assert torch.allclose(wp[0, :, 1], torch.zeros(K, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(wp[0, :, 0],
                          torch.arange(1, K + 1, dtype=torch.float64) * v * 0.1,
                          atol=1e-10)


def test_curvature_produces_the_expected_arc():
    """Sanity against closed form: constant v and kappa trace a circle of radius 1/kappa,
    so the net yaw over K steps is v*kappa*K*dt."""
    K, v, kap = 20, 10.0, 0.05
    ctrl = torch.zeros(1, K, 2, dtype=torch.float64)
    ctrl[..., 1] = kap
    dp = unicycle_step_dpose(ctrl, torch.tensor([v], dtype=torch.float64))
    assert abs(float(dp[..., 2].sum()) - v * kap * K * 0.1) < 1e-10


def test_zero_init_means_the_decoder_starts_kinematically_valid():
    """⛔ A randomly-initialised control head integrates its own noise TWICE and starts
    from a physically absurd trajectory — a far worse basin. Zero-init makes step 0
    exactly 'hold v0, go straight'."""
    m = UnicycleStepReadout(state_dim=16)
    z, v, zr = torch.randn(5, 16), torch.full((5,), 9.0), torch.zeros(5)
    a, yr = m(z, z, v, zr, zr)
    assert float(a.abs().max()) == 0.0 and float(yr.abs().max()) == 0.0


def test_warm_start_copies_the_latent_columns_and_refuses_a_mismatch():
    """⭐ The trunk has already learned to read a latent transition — the expensive
    half. ⚠️ Only the 2*state_dim LATENT columns can be copied now that the head also
    takes speed and previous controls; that partial copy is stated, not hidden. And a
    shape mismatch must RAISE, never silently return a random module wearing the name
    'warm-started'."""
    sr = StepDisplacementReadout(state_dim=16, hidden=64)
    m = UnicycleStepReadout.warm_start_from(sr, state_dim=16, hidden=64)
    assert torch.equal(m.net[1].weight[:, :32], sr.net[1].weight)
    assert torch.equal(m.net[0].weight[:32], sr.net[0].weight)
    assert float(m.net[-1].weight.abs().max()) == 0.0        # head still zero
    try:
        UnicycleStepReadout.warm_start_from(sr, state_dim=16, hidden=128)
    except ValueError:
        return
    raise AssertionError("accepted a trunk shape mismatch")


def test_rollout_matches_the_displacement_rollout_shapes_exactly():
    """⛔ The latent roll must be byte-identical to `rollout_decode`'s — same predictor,
    same action bookkeeping — or an ablation between the two DECODERS would also be an
    ablation of the ROLLOUT and the result would be unattributable."""
    B, W, S, A, K = 2, 4, 16, 3, 20
    torch.manual_seed(0)
    pred = _Predictor(S).double()
    states = torch.randn(B, W, S, dtype=torch.float64)
    acts = torch.randn(B, W, A, dtype=torch.float64)
    fut = torch.randn(B, K, A, dtype=torch.float64)
    sr = StepDisplacementReadout(S, hidden=32).double()
    uni = UnicycleStepReadout(S, hidden=32).double()
    v0 = torch.tensor([7.0, 3.0], dtype=torch.float64)

    wp_a, dp_a = rollout_decode(pred, states, acts, fut, sr, K)
    wp_b, dp_b = rollout_decode_unicycle(pred, states, acts, fut, uni, K, v0)
    assert wp_a.shape == wp_b.shape == (B, K, 2)
    assert dp_a.shape == dp_b.shape == (B, K, 3)
    # zero-init unicycle -> straight line at v0, which the free readout will not match
    assert torch.allclose(wp_b[:, :, 1], torch.zeros(B, K, dtype=torch.float64),
                          atol=1e-12)


def test_gradients_reach_the_readout_and_not_a_frozen_trunk():
    """Option 2's contract: the trunk is frozen, the decoder trains."""
    B, W, S, A, K = 2, 4, 16, 3, 8
    pred = _Predictor(S).double()
    for p in pred.parameters():
        p.requires_grad_(False)
    uni = UnicycleStepReadout(S, hidden=32).double()
    nn.init.normal_(uni.net[-1].weight, std=0.01)         # break the zero-init
    states = torch.randn(B, W, S, dtype=torch.float64)
    acts = torch.randn(B, W, A, dtype=torch.float64)
    wp, _ = rollout_decode_unicycle(pred, states, acts, None, uni, K,
                                    torch.full((B,), 9.0, dtype=torch.float64))
    wp.pow(2).mean().backward()
    assert uni.net[-1].weight.grad is not None
    assert float(uni.net[-1].weight.grad.abs().max()) > 0.0
    assert all(p.grad is None for p in pred.parameters())


def test_limits_are_optional_and_off_by_default():
    """An unbounded decode must pass controls through — bounding is the caller's
    choice, and squashing twice is not a no-op."""
    ctrl = torch.zeros(1, 3, 2, dtype=torch.float64)
    ctrl[..., 0] = 4.0
    dp = unicycle_step_dpose(ctrl, torch.zeros(1, dtype=torch.float64))
    assert abs(float(dp[0, 1, 0]) - 4.0 * 0.1 * 0.1) < 1e-12    # v after one step
    dp_l = unicycle_step_dpose(ctrl, torch.zeros(1, dtype=torch.float64),
                               accel_limit=1.0)
    assert float(dp_l[0, 1, 0]) < float(dp[0, 1, 0])


def test_shape_contract():
    for bad in (torch.zeros(20, 2), torch.zeros(1, 20, 3)):
        try:
            unicycle_step_dpose(bad, torch.zeros(1))
        except ValueError:
            continue
        raise AssertionError(f"accepted {tuple(bad.shape)}")


# --------------------------------------------------------------------------- #
# The four MEASURED design choices — one test per claim, so an arm that changed  #
# four things at once is never the only evidence for any of them.               #
# --------------------------------------------------------------------------- #

def test_output_channels_are_scaled_to_their_own_target_std():
    """⛔ MEASURED on the TRAIN corpus (660,080 samples): accel std 0.88361 vs yaw-rate
    std 0.13091 — a 6.7x ratio, and against CURVATURE (std 0.03207) it is 27.6x. With one
    shared output layer the lateral channel is badly under-resolved. A unit raw output
    must land on each channel's own target scale.

    ⚠️ The constants moved when the corpus widened (yaw-rate std 0.06930 -> 0.13091,
    nearly 2x), so this test asserts the RELATIONSHIP, not a frozen literal — a test that
    pins the number would have to be edited every time the estimate improves, which is
    how a test stops being a check and becomes a copy."""
    from tanitad.models.metric_dynamics import ACCEL_SCALE, YAWRATE_SCALE
    m = UnicycleStepReadout(state_dim=8, hidden=16, predict_delta=False,
                            speed_input=False)
    with torch.no_grad():
        m.net[-1].bias.copy_(torch.tensor([1.0, 1.0]))     # unit raw output
    z, v, zr = torch.zeros(1, 8), torch.full((1,), 20.0), torch.zeros(1)
    a, yr = m(z, z, v, zr, zr)
    assert abs(float(a) - ACCEL_SCALE) < 1e-6
    assert abs(float(yr) - YAWRATE_SCALE) < 1e-6
    assert ACCEL_SCALE / YAWRATE_SCALE > 4.0               # the imbalance is real


def test_yaw_rate_parameterisation_still_cannot_turn_in_place():
    """⛔ THE TRAP IN CHOICE (2). Yaw rate is the better-conditioned target (kurtosis
    10.4 vs curvature's 38.9), but predicting it WITHOUT the |v|*kappa_max bound would
    quietly restore turn-in-place — the exact defect the unicycle exists to remove."""
    m = UnicycleStepReadout(state_dim=8, hidden=16, predict_delta=False,
                            speed_input=False, curvature_limit=0.33)
    with torch.no_grad():
        m.net[-1].bias.copy_(torch.tensor([0.0, 100.0]))   # demand a huge yaw rate
    z, zr = torch.zeros(1, 8), torch.zeros(1)
    _, yr_stopped = m(z, z, torch.zeros(1), zr, zr)
    _, yr_moving = m(z, z, torch.full((1,), 12.0), zr, zr)
    assert float(yr_stopped.abs()) == 0.0                  # v=0 -> bound is 0
    assert abs(float(yr_moving) - 12.0 * 0.33) < 1e-5      # bounded, not unbounded


def test_speed_is_actually_an_input():
    """Choice (3): the target's conditional distribution depends strongly on v (its
    low-speed tail is 9.5x the high-speed one). A head that ignores v is fitting the
    marginal."""
    torch.manual_seed(0)
    m = UnicycleStepReadout(state_dim=8, hidden=16, speed_input=True,
                            predict_delta=False)
    with torch.no_grad():
        nn.init.normal_(m.net[-1].weight, std=1.0)
    z, zr = torch.randn(1, 8), torch.zeros(1)
    a_slow, _ = m(z, z, torch.full((1,), 1.0), zr, zr)
    a_fast, _ = m(z, z, torch.full((1,), 18.0), zr, zr)
    assert abs(float(a_slow) - float(a_fast)) > 1e-6

    off = UnicycleStepReadout(state_dim=8, hidden=16, speed_input=False,
                              predict_delta=False)
    with torch.no_grad():
        nn.init.normal_(off.net[-1].weight, std=1.0)
    b_slow, _ = off(z, z, torch.full((1,), 1.0), zr, zr)
    b_fast, _ = off(z, z, torch.full((1,), 18.0), zr, zr)
    assert abs(float(b_slow) - float(b_fast)) < 1e-9       # the ablation is a real off


def test_delta_head_is_smooth_by_default():
    """⭐ CHOICE (4), the biggest. MEASURED: accel delta std 0.17494 against an absolute
    std of 0.80438 — a 4.6x easier target — with lag-1 autocorrelation +0.977. A delta
    head's natural output scale IS THE JERK, so smoothness is the DEFAULT rather than
    something a barrier has to fight the head for.

    With random weights, the ABSOLUTE head's step-to-step accel change is ~its full
    output scale; the DELTA head's is ~its (much smaller) delta scale."""
    from tanitad.models.metric_dynamics import ACCEL_SCALE, DACCEL_SCALE
    torch.manual_seed(0)
    S, K = 8, 20
    zs = [torch.randn(1, S) for _ in range(K + 1)]

    def run(delta: bool):
        m = UnicycleStepReadout(state_dim=S, hidden=32, predict_delta=delta,
                                speed_input=False)
        with torch.no_grad():
            nn.init.normal_(m.net[-1].weight, std=1.0)
        a_prev = yr_prev = torch.zeros(1)
        v, accs = torch.full((1,), 12.0), []
        for j in range(K):
            a, yr = m(zs[j], zs[j + 1], v, a_prev, yr_prev)
            accs.append(float(a)); a_prev, yr_prev = a, yr
        d = torch.tensor(accs)[1:] - torch.tensor(accs)[:-1]
        return float(d.abs().mean())

    assert run(True) < run(False), (run(True), run(False))
    assert DACCEL_SCALE < ACCEL_SCALE / 3.0                # 0.175 vs 0.804


def test_every_choice_can_be_switched_off_for_the_ablation():
    """⛔ An arm that changed four things at once and improved would be UNATTRIBUTABLE —
    the --v2 conflation failure. One flag per claim."""
    S = 8
    base = UnicycleStepReadout(S, hidden=16, predict_delta=False, speed_input=False)
    assert base.in_dim == 2 * S
    assert UnicycleStepReadout(S, 16, predict_delta=True, speed_input=False).in_dim == 2 * S + 2
    assert UnicycleStepReadout(S, 16, predict_delta=False, speed_input=True).in_dim == 2 * S + 1
    assert UnicycleStepReadout(S, 16, predict_delta=True, speed_input=True).in_dim == 2 * S + 3
