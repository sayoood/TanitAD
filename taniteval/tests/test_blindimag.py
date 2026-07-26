"""Certification for :mod:`taniteval.blindimag` — M3, with a FAILING input.

The single claim this whole stream rests on is that the program's grounded
rollout is ALREADY a blind-imagination drive. If ``blind_rollout``'s default
configuration ever stops being ``rollout_decode``, every arm downstream is
measuring a different instrument and the comparison to ``wm_canary_ade_2s`` /
``ade_0_2s`` becomes false. ``test_imagination_is_bit_identical_to_rollout_decode``
is that guarantee.

Per M3 every guard here is shown to be capable of FAILING: the mutation tests
feed a deliberately wrong instrument and assert the equality check rejects it.
"""
import math

import pytest
import torch

from taniteval import blindimag as bi
from tanitad.models.metric_dynamics import (StepDisplacementReadout,
                                            accumulate_se2, gt_step_dposes,
                                            rollout_decode)

S, A, WIN, B, K = 24, 3, 8, 5, 7


class _StubPredictor(torch.nn.Module):
    """A deterministic 1-step predictor with the WorldModel predictor's contract
    (``__call__(states, actions) -> {horizon: [B, S]}``). Depends on BOTH the
    window and the actions, so a test that swaps either can actually fail."""

    def __init__(self, s=S, a=A):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(s * WIN + a * WIN, s)

    def forward(self, states, actions):
        x = torch.cat([states.flatten(1), actions.flatten(1)], dim=-1)
        return {1: torch.tanh(self.lin(x))}


def _fixture(seed=0):
    torch.manual_seed(seed)
    states = torch.randn(B, WIN, S)
    actions = torch.randn(B, WIN, A) * 0.1
    fut = torch.randn(B, K, A) * 0.1
    obs = torch.randn(B, K, S)
    pred = _StubPredictor().eval()
    ro = StepDisplacementReadout(S).eval()
    return states, actions, fut, obs, pred, ro


# --------------------------------------------------------------------------- #
# THE certification                                                            #
# --------------------------------------------------------------------------- #
def test_imagination_is_bit_identical_to_rollout_decode():
    """``blind_rollout`` default == ``metric_dynamics.rollout_decode``, exactly."""
    states, actions, fut, _obs, pred, ro = _fixture()
    with torch.no_grad():
        wp_ref, dp_ref = rollout_decode(pred, states, actions, fut, ro, K)
    out = bi.blind_rollout(pred, states, actions, ro, K,
                           state_source="imagination",
                           action_source="true_future", future_actions=fut)
    assert torch.equal(out["waypoints"], wp_ref)
    assert torch.equal(out["step_dpose"], dp_ref)


def test_hold_last_matches_rollout_decode_zero_order_hold():
    """``action_source='hold_last'`` == ``rollout_decode(future_actions=None)``."""
    states, actions, _fut, _obs, pred, ro = _fixture()
    with torch.no_grad():
        wp_ref, _ = rollout_decode(pred, states, actions, None, ro, K)
    out = bi.blind_rollout(pred, states, actions, ro, K,
                           action_source="hold_last")
    assert torch.equal(out["waypoints"], wp_ref)


def test_accumulate_matches_metric_dynamics():
    torch.manual_seed(3)
    dp = torch.randn(B, K, 3) * 0.3
    pos, psi = bi.accumulate_se2_pose(dp)
    assert torch.equal(pos, accumulate_se2(dp))
    assert torch.allclose(psi, dp[..., 2].cumsum(dim=1), atol=1e-6)


# --------------------------------------------------------------------------- #
# M3: the guard must be able to FAIL. Feed it a deliberately wrong instrument.  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state_source", ["frozen_last", "full_obs"])
def test_other_state_sources_are_NOT_rollout_decode(state_source):
    """A control arm that silently equalled the imagination arm would make the
    whole experiment vacuous. Assert the arms genuinely differ."""
    states, actions, fut, obs, pred, ro = _fixture()
    with torch.no_grad():
        wp_ref, _ = rollout_decode(pred, states, actions, fut, ro, K)
    out = bi.blind_rollout(pred, states, actions, ro, K,
                           state_source=state_source,
                           action_source="true_future",
                           future_actions=fut, obs_states=obs)
    assert not torch.allclose(out["waypoints"], wp_ref, atol=1e-6)


def test_first_step_is_identical_across_state_sources():
    """Before anything is appended, every arm must decode the SAME first step —
    they only diverge from step 2. If they differed at step 1 the arms would not
    be sharing an observed window and no contrast would be attributable."""
    states, actions, fut, obs, pred, ro = _fixture()
    outs = [bi.blind_rollout(pred, states, actions, ro, K, state_source=s,
                             action_source="true_future", future_actions=fut,
                             obs_states=obs)["step_dpose"][:, 0]
            for s in ("imagination", "frozen_last", "full_obs")]
    for o in outs[1:]:
        assert torch.equal(outs[0], o)


def test_frozen_last_window_really_freezes():
    """After W-1 appends the frozen arm's window must be entirely copies of the
    last real percept — 'the world stopped', mechanically."""
    states, actions, _f, _o, pred, ro = _fixture()
    k_long = WIN + 4                              # long enough to fully flush
    fut = torch.randn(B, k_long, A) * 0.1
    obs = torch.randn(B, k_long, S)
    seen = {}

    class _Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self.n = inner, 0

        def forward(self, s, a):
            seen[self.n] = s.clone()
            self.n += 1
            return self.inner(s, a)

    bi.blind_rollout(_Spy(pred), states, actions, ro, k_long,
                     state_source="frozen_last", action_source="true_future",
                     future_actions=fut, obs_states=obs)
    last_win = seen[WIN]                       # after W appends
    z = states[:, -1]
    for w in range(WIN):
        assert torch.equal(last_win[:, w], z)


# --------------------------------------------------------------------------- #
# The action inverse                                                           #
# --------------------------------------------------------------------------- #
def test_kinematic_inverse_recovers_the_corpus_steer_definition():
    """The corpus builds ``steer = atan(WHEELBASE * curvature)``. Feeding the
    inverse a Δpose generated from a known (v, kappa) must return that steer."""
    v = torch.tensor([8.0, 15.0])
    kappa = torch.tensor([0.01, -0.004])
    dyaw = kappa * v * bi.DT
    dpose = torch.stack([v * bi.DT, torch.zeros(2), dyaw], dim=-1)
    steer, accel, v_out = bi.kinematic_action_from_dpose(dpose, v)
    assert torch.allclose(v_out, v, atol=1e-5)
    assert torch.allclose(accel, torch.zeros(2), atol=1e-5)
    assert torch.allclose(steer, torch.atan(bi.WHEELBASE * kappa), atol=1e-5)


def test_kinematic_inverse_clamps_are_reachable():
    """C13: a clamp that can never fire is not a clamp. Show both fire."""
    dpose = torch.tensor([[3.0, 2.0, 1.0]])
    steer, accel, _ = bi.kinematic_action_from_dpose(dpose, torch.tensor([0.0]))
    assert float(accel) == pytest.approx(bi.ACCEL_CLAMP)
    assert float(steer) == pytest.approx(bi.STEER_CLAMP)


def test_gt_kinematic_uses_true_dposes_not_the_models():
    """The convention control must be insensitive to the model's own output."""
    states, actions, fut, obs, pred, ro = _fixture()
    gt = torch.randn(B, K, 3) * 0.1
    v0 = torch.rand(B) * 10 + 3
    a = bi.blind_rollout(pred, states, actions, ro, K,
                         action_source="gt_kinematic", gt_step_dpose=gt,
                         v_last=v0)["fed_actions"]
    b = bi.blind_rollout(pred, states, actions, ro, K,
                         state_source="full_obs", action_source="gt_kinematic",
                         gt_step_dpose=gt, v_last=v0, obs_states=obs)["fed_actions"]
    assert torch.equal(a[:, :, :2], b[:, :, :2])


# --------------------------------------------------------------------------- #
# Ground truth / floors at arbitrary horizon                                    #
# --------------------------------------------------------------------------- #
def _poses(T=60, seed=1):
    torch.manual_seed(seed)
    yaw = torch.cumsum(torch.randn(T) * 0.02, 0)
    v = 8.0 + torch.randn(T) * 0.3
    x = torch.cumsum(v * torch.cos(yaw) * bi.DT, 0)
    y = torch.cumsum(v * torch.sin(yaw) * bi.DT, 0)
    return torch.stack([x, y, yaw, v], dim=-1)


def test_gt_dense_matches_driving_diagnostic():
    from driving_diagnostic import gt_ego_waypoints
    p = _poses()
    last = torch.tensor([8, 17, 25])
    pos, _ = bi.gt_dense_path(p, last, 12)
    ref = gt_ego_waypoints(p, last, wp_steps=tuple(range(1, 13)))
    assert torch.allclose(pos, ref, atol=1e-5)


def test_gt_step_dposes_dense_matches_metric_dynamics():
    p = _poses()
    last = torch.tensor([8, 17, 25])
    k = 12
    fut = torch.stack([p[int(t) + 1:int(t) + 1 + k] for t in last])
    ref = gt_step_dposes(p[last], fut, k)
    assert torch.allclose(bi.gt_step_dposes_dense(p, last, k), ref, atol=1e-5)


def test_gt_step_dposes_accumulate_back_to_gt_path():
    """Sanity of the whole geometry chain: accumulating the TRUE Δposes must
    reproduce the TRUE waypoints. If this fails, no arm's path means anything."""
    p = _poses()
    last = torch.tensor([8, 17, 25])
    pos, _ = bi.gt_dense_path(p, last, 20)
    acc, _ = bi.accumulate_se2_pose(bi.gt_step_dposes_dense(p, last, 20))
    assert torch.allclose(acc, pos, atol=1e-4)


def test_cv_dense_matches_baseline_waypoints():
    from driving_diagnostic import baseline_waypoints
    p = _poses()
    last = torch.tensor([8, 17, 25])
    cv = bi.cv_dense_path(p, last, 20)
    ref = baseline_waypoints(p, last)["constant_velocity"]        # steps 5/10/15/20
    assert torch.allclose(cv[:, [4, 9, 14, 19]], ref, atol=1e-5)


def test_hold_v0_matches_go_straight():
    from driving_diagnostic import baseline_waypoints
    p = _poses()
    last = torch.tensor([8, 17, 25])
    hv = bi.hold_v0_dense_path(p, last, 20)
    ref = baseline_waypoints(p, last)["go_straight"]
    assert torch.allclose(hv[:, [4, 9, 14, 19]], ref, atol=1e-5)


def test_window_starts_matches_clhorizon():
    from taniteval.clhorizon import horizon_windows
    for T in (188, 198, 205):
        for k in (20, 60, 185):
            assert len(bi.window_starts(T, k)) == horizon_windows(T, k)


def test_path_deviation_is_zero_on_the_logged_path():
    """A perfect prediction must show zero lateral and zero heading deviation —
    the control that proves the deviation instrument is not measuring noise."""
    p = _poses()
    last = torch.tensor([8, 17, 25])
    pos, yaw = bi.gt_dense_path(p, last, 20)
    lat, dpsi = bi.path_deviation(pos, yaw, pos, yaw)
    assert float(lat.max()) < 1e-4
    assert float(dpsi.max()) < 1e-3


# --------------------------------------------------------------------------- #
# The peek policies (E-IMAG-4)                                                  #
# --------------------------------------------------------------------------- #
def test_uniform_peek_duty_cycle_is_what_it_claims():
    """The realised duty cycle must equal 1/T' — measured, never assumed."""
    states, actions, fut, _o, pred, ro = _fixture()
    k_long = 40
    fut = torch.randn(B, k_long, A) * 0.1
    obs = torch.randn(B, k_long, S)
    for period in (2, 5, 10):
        out = bi.blind_rollout(pred, states, actions, ro, k_long,
                               action_source="true_future", future_actions=fut,
                               obs_states=obs, peek_period=period)
        realised = out["peek_mask"].float().mean().item()
        assert abs(realised - 1.0 / period) < 1.5 / k_long


def test_peek_period_1_equals_full_observation():
    """Peeking every single step IS the full-observation arm. If these differed,
    the peek machinery and the ceiling arm would not be on one surface."""
    states, actions, _f, _o, pred, ro = _fixture()
    k_long = 20
    fut = torch.randn(B, k_long, A) * 0.1
    obs = torch.randn(B, k_long, S)
    a = bi.blind_rollout(pred, states, actions, ro, k_long,
                         action_source="true_future", future_actions=fut,
                         obs_states=obs, peek_period=1)["waypoints"]
    b = bi.blind_rollout(pred, states, actions, ro, k_long,
                         state_source="full_obs", action_source="true_future",
                         future_actions=fut, obs_states=obs)["waypoints"]
    assert torch.equal(a, b)


def test_oracle_peek_bar_is_reachable_in_both_directions():
    """C13 on the oracle trigger: an impossible bar must never fire and a zero
    bar must always fire. A trigger that cannot do both carries no information."""
    states, actions, _f, _o, pred, ro = _fixture()
    k_long = 20
    obs = torch.randn(B, k_long, S)
    gt = torch.randn(B, k_long, 3) * 0.1
    v0 = torch.rand(B) * 10 + 3
    never = bi.blind_rollout(pred, states, actions, ro, k_long,
                             action_source="gt_kinematic", gt_step_dpose=gt,
                             v_last=v0, obs_states=obs, peek_oracle_bar=1e9)
    always = bi.blind_rollout(pred, states, actions, ro, k_long,
                              action_source="gt_kinematic", gt_step_dpose=gt,
                              v_last=v0, obs_states=obs, peek_oracle_bar=-1.0)
    assert never["peek_mask"].sum() == 0
    assert always["peek_mask"][:, :-1].all()


def test_no_peek_policy_leaves_the_default_path_bit_identical():
    """The peek machinery must not perturb the certified default."""
    states, actions, fut, _o, pred, ro = _fixture()
    with torch.no_grad():
        wp_ref, _ = rollout_decode(pred, states, actions, fut, ro, K)
    out = bi.blind_rollout(pred, states, actions, ro, K, future_actions=fut)
    assert torch.equal(out["waypoints"], wp_ref)
    assert out["peek_mask"].sum() == 0


def test_path_deviation_recovers_a_known_offset():
    """C13 on the deviation instrument: inject a 1.5 m lateral shift and require
    it to be recovered, so a reading of 'inside the envelope' can actually fail."""
    p = _poses(seed=4)
    last = torch.tensor([8, 20])
    pos, yaw = bi.gt_dense_path(p, last, 20)
    shifted = pos.clone()
    shifted[..., 1] += 1.5
    lat, _ = bi.path_deviation(shifted, yaw, pos, yaw)
    assert abs(float(lat[:, 5:].mean()) - 1.5) < 0.25


# =========================================================================== #
# T_BLIND RUNG 1 — the ACTION FILTER. Certified in BOTH directions: the two
# endpoints must reduce EXACTLY to arms that already exist, and every non-null
# setting must actually change the fed action. A filter that is silently a no-op
# would produce a flat, confident, wrong intervention curve.
# =========================================================================== #
def _kin_fixture(seed=0, k=12):
    torch.manual_seed(seed)
    states = torch.randn(B, WIN, S)
    actions = torch.randn(B, WIN, A) * 0.1
    obs = torch.randn(B, k, S)
    pred = _StubPredictor().eval()
    ro = StepDisplacementReadout(S).eval()
    v0 = torch.rand(B) * 8 + 2
    return states, actions, obs, pred, ro, v0, k


def _roll(spec, *, seed=0, k=12, state_source="imagination", **kw):
    states, actions, obs, pred, ro, v0, k = _kin_fixture(seed, k)
    return bi.blind_rollout(pred, states, actions, ro, k,
                            state_source=state_source, action_source=spec,
                            obs_states=obs, v_last=v0, **kw)


@pytest.mark.parametrize("spec,expect", [
    ("own_kinematic", ("own_kinematic", {})),
    ("own_kinematic|blend=0.25", ("own_kinematic", {"blend": 0.25})),
    ("gt_kinematic|ema=0.8|every=3",
     ("gt_kinematic", {"ema": 0.8, "every": 3})),
])
def test_parse_action_source(spec, expect):
    assert bi.parse_action_source(spec) == expect


@pytest.mark.parametrize("bad", [
    "own_kinematic|nosuchknob=1",      # a dropped knob is how a sweep lies
    "own_kinematic|blend=1.5",         # out of range
    "own_kinematic|every=0",           # would divide-by-zero silently
    "own_kinematic|chan=throttle",     # not a channel
    "hold_last|blend=0.5",             # nothing to filter
    "true_future|ema=0.5",
])
def test_parse_action_source_rejects(bad):
    with pytest.raises(ValueError):
        bi.parse_action_source(bad)


def test_blend_zero_is_bit_identical_to_the_unfiltered_arm():
    a = _roll("own_kinematic")
    b = _roll("own_kinematic|blend=0.0")
    assert torch.equal(a["waypoints"], b["waypoints"])
    assert torch.equal(a["fed_actions"], b["fed_actions"])


def test_every_1_is_bit_identical_to_the_unfiltered_arm():
    """The `every` knob must have a null setting AND a non-null one."""
    a = _roll("own_kinematic")
    assert torch.equal(a["waypoints"], _roll("own_kinematic|every=1")["waypoints"])
    assert not torch.equal(a["waypoints"],
                           _roll("own_kinematic|every=4")["waypoints"])


def test_blend_one_reduces_exactly_to_hold_last():
    """⭐ The upper endpoint of the intervention axis IS the measured ceiling arm.

    ``a_hold0`` is the constant ``hold_last`` feeds forever, so the blend at
    alpha = 1 is that hold — algebraically, not approximately. This is what lets
    the sweep's endpoints be checked against arms that already exist.
    """
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    hold = bi.blind_rollout(pred, states, actions, ro, k,
                            action_source="hold_last")
    blend1 = bi.blind_rollout(pred, states, actions, ro, k,
                              action_source="own_kinematic|blend=1.0",
                              obs_states=obs, v_last=v0)
    assert torch.equal(hold["waypoints"], blend1["waypoints"])
    assert torch.equal(hold["fed_actions"], blend1["fed_actions"])
    assert torch.equal(hold["pred_speed"], blend1["pred_speed"])


@pytest.mark.parametrize("spec", [
    "own_kinematic|blend=0.5", "own_kinematic|ema=0.8",
    "own_kinematic|every=5", "own_kinematic|steer_clip=0.001",
    "own_kinematic|accel_clip=0.05", "own_kinematic|chan=steer",
    "own_kinematic|chan=accel", "own_kinematic|own_before=3",
    "own_kinematic|own_after=3",
])
def test_every_filter_actually_changes_the_fed_action(spec):
    """ANTI-NO-OP. A knob that changes nothing is a knob that was not read."""
    base = _roll("own_kinematic")
    got = _roll(spec)
    assert not torch.equal(base["fed_actions"], got["fed_actions"]), spec
    assert not torch.equal(base["waypoints"], got["waypoints"]), spec


def test_clip_filter_binds_the_band_it_names():
    out = _roll("own_kinematic|steer_clip=0.004|accel_clip=0.05")
    fa = out["fed_actions"][:, :-1]        # the last row is the carried window action
    assert float(fa[..., 0].abs().max()) <= 0.004 + 1e-6
    assert float(fa[..., 1].abs().max()) <= 0.05 + 1e-6


def test_switch_filters_hold_exactly_the_segment_they_name():
    m = 4
    hold0 = _kin_fixture()[1][:, -1]
    before = _roll(f"own_kinematic|own_before={m}")["fed_actions"]
    after = _roll(f"own_kinematic|own_after={m}")["fed_actions"]
    # own_before: steps >= m are the held action; own_after: steps < m are
    assert torch.allclose(before[:, m:-1, :2], hold0[:, None, :2].expand_as(before[:, m:-1, :2]))
    assert torch.allclose(after[:, :m, :2], hold0[:, None, :2].expand_as(after[:, :m, :2]))
    assert not torch.allclose(before[:, :m, :2], hold0[:, None, :2].expand_as(before[:, :m, :2]))


def test_action_filter_leaves_the_speed_channel_alone():
    out = _roll("own_kinematic|blend=0.5")
    v0c = _kin_fixture()[1][:, -1, 2]
    assert torch.allclose(out["fed_actions"][..., 2],
                          v0c[:, None].expand_as(out["fed_actions"][..., 2]))


def test_reconstruct_kinematic_actions_matches_fed_actions():
    """⭐ The 599-window action statistics are RECONSTRUCTED from a dump that
    stores ``psi``/``pred_speed`` but not ``fed_actions``. That reconstruction is
    proved here, not assumed."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture(seed=1, k=15)
    out = bi.blind_rollout(pred, states, actions, ro, k,
                           action_source="own_kinematic",
                           obs_states=obs, v_last=v0)
    steer, accel = bi.reconstruct_kinematic_actions(out["psi"],
                                                    out["pred_speed"], v0)
    fed = out["fed_actions"]
    assert torch.allclose(steer[:, :k - 1], fed[:, :k - 1, 0], atol=1e-6)
    assert torch.allclose(accel[:, :k - 1], fed[:, :k - 1, 1], atol=1e-6)


def test_reconstruction_can_fail_on_the_wrong_input():
    """C13 on the reconstruction: feeding it the WRONG speed must break it."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture(seed=1, k=15)
    out = bi.blind_rollout(pred, states, actions, ro, k,
                           action_source="own_kinematic",
                           obs_states=obs, v_last=v0)
    steer, accel = bi.reconstruct_kinematic_actions(out["psi"],
                                                    out["pred_speed"] * 1.5, v0)
    fed = out["fed_actions"]
    assert not torch.allclose(accel[:, :k - 1], fed[:, :k - 1, 1], atol=1e-6)


# =========================================================================== #
# T_BLIND RUNG 1 — the PLANNER action source (`wp_to_control`), the R1 row      #
# =========================================================================== #
def _plan_const(w):
    """A plan_fn returning a FIXED ego-frame lookahead target, so the controller
    can be exercised without a trained tactical brain on CPU."""
    def f(win_s, v):
        return torch.as_tensor(w, dtype=v.dtype).expand(win_s.shape[0], 2).clone()
    return f


@pytest.mark.parametrize("spec,expect", [
    ("planner", ("planner", {})),
    ("planner|vsrc=decoded", ("planner", {"vsrc": "decoded"})),
    ("planner|look=gt", ("planner", {"look": "gt"})),
    ("planner|vsrc=ctrl|look=plan", ("planner", {"vsrc": "ctrl", "look": "plan"})),
])
def test_parse_planner_action_source(spec, expect):
    assert bi.parse_action_source(spec) == expect


@pytest.mark.parametrize("bad", [
    "own_kinematic|vsrc=ctrl",        # planner knob on a kinematic base
    "planner|blend=0.5",              # filter knob on the planner base
    "planner|vsrc=nope",
    "planner|look=nope",
])
def test_parse_planner_rejects(bad):
    with pytest.raises(ValueError):
        bi.parse_action_source(bad)


def test_planner_requires_its_inputs():
    """FAIL LOUD: the planner cannot silently fall back to another source."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    with pytest.raises(ValueError):                       # no plan_fn
        bi.blind_rollout(pred, states, actions, ro, k, action_source="planner",
                         obs_states=obs, v_last=v0)
    with pytest.raises(ValueError):                       # look=gt, no gt_pos
        bi.blind_rollout(pred, states, actions, ro, k,
                         action_source="planner|look=gt",
                         obs_states=obs, v_last=v0)
    with pytest.raises(ValueError):                       # no v_last
        bi.blind_rollout(pred, states, actions, ro, k, action_source="planner",
                         obs_states=obs, plan_fn=_plan_const([10.0, 0.0]))


def test_planner_feeds_exactly_wp_to_control():
    """THE LOAD-BEARING IDENTITY. The first fed action must equal what
    `closedloop.wp_to_control` returns for the same target and speed — the arm
    IS the deployed controller, not a re-derivation of it."""
    from taniteval.closedloop import wp_to_control
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    w = [12.0, 1.5]
    out = bi.blind_rollout(pred, states, actions, ro, k, action_source="planner",
                           obs_states=obs, v_last=v0, plan_fn=_plan_const(w))
    st, ac = wp_to_control(torch.as_tensor([w], dtype=v0.dtype).expand(B, 2), v0)
    assert torch.equal(out["fed_actions"][:, 0, 0], st)
    assert torch.equal(out["fed_actions"][:, 0, 1], ac)


def test_planner_speed_bookkeeping_is_closed_loop_rollouts():
    """`v <- clamp_min(v + accel*DT, 0)` — the controller integrates its OWN
    command. Checked at step 2, where a wrong `v` changes the action."""
    from taniteval.closedloop import wp_to_control
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    w = [12.0, 1.5]
    out = bi.blind_rollout(pred, states, actions, ro, k, action_source="planner",
                           obs_states=obs, v_last=v0, plan_fn=_plan_const(w))
    a0 = out["fed_actions"][:, 0, 1]
    v1 = (v0 + a0 * bi.DT).clamp_min(0.0)
    _, a1 = wp_to_control(torch.as_tensor([w], dtype=v0.dtype).expand(B, 2), v1)
    assert torch.equal(out["fed_actions"][:, 1, 1], a1)


def _plan_track(dv, y):
    """A plan_fn whose implied target speed is `v + dv`, so `wp_to_control`'s
    accel does NOT sit on the +-3 clamp. ⚠️ With a CONSTANT far-away target the
    accel saturates for every input speed and the arms become indistinguishable —
    which is the very saturation mechanism Rung 1 identified, and it would make an
    anti-no-op test pass vacuously."""
    from taniteval.closedloop import DT as _dt, LOOKAHEAD_STEP as _L

    def f(win_s, v):
        return torch.stack([(v + dv) * (_L * _dt),
                            torch.full_like(v, float(y))], dim=-1)
    return f


def test_planner_vsrc_decoded_is_a_different_arm():
    """ANTI-NO-OP: the `vsrc` knob must actually move the fed action."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    kw = dict(obs_states=obs, v_last=v0, plan_fn=_plan_track(0.5, 0.2))
    a = bi.blind_rollout(pred, states, actions, ro, k,
                         action_source="planner", **kw)["fed_actions"]
    b = bi.blind_rollout(pred, states, actions, ro, k,
                         action_source="planner|vsrc=decoded", **kw)["fed_actions"]
    assert not torch.allclose(a, b)


def test_planner_is_not_the_kinematic_inverse_nor_hold():
    """ANTI-NO-OP against BOTH endpoints — a planner arm silently equal to
    `own_kinematic` or to `hold_last` would produce a flat, confident, wrong
    curve, which is exactly how Rung 1's plumbing self-test earns its place."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    kw = dict(obs_states=obs, v_last=v0)
    p = bi.blind_rollout(pred, states, actions, ro, k, action_source="planner",
                         plan_fn=_plan_const([12.0, 1.5]), **kw)["waypoints"]
    o = bi.blind_rollout(pred, states, actions, ro, k,
                         action_source="own_kinematic", **kw)["waypoints"]
    h = bi.blind_rollout(pred, states, actions, ro, k,
                         action_source="hold_last", **kw)["waypoints"]
    assert (p - o).abs().max() > 1e-3
    assert (p - h).abs().max() > 1e-3


def test_planner_gt_lookahead_tracks_the_true_path():
    """The `look=gt` ORACLE must actually consume gt_pos: perturbing the true
    path must move the fed action. (It reads the future — diagnostic only.)"""
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    gt = torch.randn(B, k, 2).cumsum(1) + 5.0
    kw = dict(obs_states=obs, v_last=v0)
    a = bi.blind_rollout(pred, states, actions, ro, k,
                         action_source="planner|look=gt", gt_pos=gt,
                         **kw)["fed_actions"]
    b = bi.blind_rollout(pred, states, actions, ro, k,
                         action_source="planner|look=gt", gt_pos=gt + 3.0,
                         **kw)["fed_actions"]
    assert not torch.allclose(a, b)


def test_planner_running_pose_matches_accumulate_se2_pose():
    """The `look=gt` frame transform uses a running pose computed INSIDE the
    loop. It must be the same pose `accumulate_se2_pose` reports, or the oracle
    target is expressed in the wrong frame."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    gt = torch.randn(B, k, 2).cumsum(1) + 5.0
    out = bi.blind_rollout(pred, states, actions, ro, k,
                           action_source="planner|look=gt", obs_states=obs,
                           v_last=v0, gt_pos=gt)
    pos, psi = bi.accumulate_se2_pose(out["step_dpose"])
    assert torch.equal(pos, out["waypoints"])
    assert torch.equal(psi, out["psi"])


def test_planner_speed_channel_is_constant_by_default():
    """Every Rung 1 arm holds `v0` constant; the planner primary must too, or it
    is not comparable to them."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    out = bi.blind_rollout(pred, states, actions, ro, k, action_source="planner",
                           obs_states=obs, v_last=v0,
                           plan_fn=_plan_const([12.0, 1.5]))
    v0c = actions[:, -1, 2]
    assert torch.allclose(out["fed_actions"][:, :-1, 2],
                          v0c[:, None].expand(B, k - 1))


def test_planner_update_speed_channel_uses_the_controller_speed():
    """`closedloop.build_action` feeds v_tracked/SPEED_SCALE. Reported
    separately, never mixed into the primary — so it must differ from it."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    kw = dict(obs_states=obs, v_last=v0, plan_fn=_plan_const([12.0, 1.5]))
    a = bi.blind_rollout(pred, states, actions, ro, k, action_source="planner",
                         **kw)["fed_actions"]
    b = bi.blind_rollout(pred, states, actions, ro, k, action_source="planner",
                         update_speed_channel=True, **kw)["fed_actions"]
    assert not torch.allclose(a[..., 2], b[..., 2])


def test_no_planner_leaves_every_pre_existing_path_bit_identical():
    """The whole extension must be inert on every call site that predates it."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    for spec in ("own_kinematic", "hold_last", "own_kinematic|blend=0.25",
                 "own_kinematic|ema=0.8"):
        a = bi.blind_rollout(pred, states, actions, ro, k, action_source=spec,
                             obs_states=obs, v_last=v0)["waypoints"]
        b = bi.blind_rollout(pred, states, actions, ro, k, action_source=spec,
                             obs_states=obs, v_last=v0, plan_fn=None,
                             gt_pos=None)["waypoints"]
        assert torch.equal(a, b)
