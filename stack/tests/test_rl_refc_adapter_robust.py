"""Tests for the refcv4b/refcv5 RL binding and the timing-robust contact term.

Two things are pinned here, and both are the *directional* kind of test — the kind that
fails when the code is wrong rather than merely when it is absent:

1. ``refc_adapter.assert_conditioning`` REFUSES a batch that drops a channel the
   checkpoint's own config says it was trained with. The bug it prevents is silent: the
   forward returns a well-formed fan from a differently-conditioned policy and nothing
   raises (`refc_v3.py:1000` only guards the opposite direction).
2. ``robust_contact`` reduces EXACTLY to ``rewards._collision`` at zero shift, and its
   SIGN runs the way the measured surface says (`dt < 0` = lead earlier along its own
   path = closer to a following ego = the risk direction).
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tanitad.rl import rewards as R                                   # noqa: E402
from tanitad.rl import robust_contact as RC                           # noqa: E402
from tanitad.rl import refc_adapter as A                              # noqa: E402
from tanitad.rl.config import PostTrainConfig                         # noqa: E402


# --------------------------------------------------------------------------------------
# robust_contact
# --------------------------------------------------------------------------------------

def _straight_lead(x0=6.0, v=4.0, S=5, dt=0.5, B=2):
    """A lead ahead of the ego, moving +x at constant speed."""
    t = torch.arange(S, dtype=torch.float32) * dt
    xy = torch.stack([x0 + v * t, torch.zeros(S)], dim=-1)
    return xy.expand(B, 1, S, 2).clone()


def _ego(v=8.0, S=5, dt=0.5, B=2, N=3):
    """An ego closing on the lead, straight ahead."""
    t = torch.arange(S, dtype=torch.float32) * dt
    xy = torch.stack([v * t, torch.zeros(S)], dim=-1)
    return xy.expand(B, N, S, 2).clone()


def test_zero_shift_is_bitwise_the_point_estimate():
    """⭐ THE T1 CONTROL. A robustness term that does not reduce to the point estimate at
    zero perturbation is measuring something else."""
    traj, lead = _ego(), _straight_lead()
    ctx = {"lead_path": lead, "dt": 0.5}
    point = R.COMPONENTS["collision"](traj, ctx)
    robust = RC.robust_contact(traj, {**ctx, "robust_shifts_s": (0.0,)})
    assert torch.equal(point, robust), "zero-shift must be the point estimate exactly"


def test_negative_shift_moves_the_lead_closer_to_a_following_ego():
    """SIGN AS GEOMETRY. dt < 0 samples EARLIER indices, so the lead sits further back
    along its own path -- nearer a follower. Asserted on POSITIONS, not on a label."""
    lead = _straight_lead()
    dt = 0.5
    back = RC.shift_track(lead, -dt, dt)
    fwd = RC.shift_track(lead, +dt, dt)
    # interior points only: the edges are clamped on purpose
    assert (back[..., 1:-1, 0] < lead[..., 1:-1, 0]).all(), "dt<0 must move the lead BACK"
    assert (fwd[..., 1:-1, 0] > lead[..., 1:-1, 0]).all(), "dt>0 must move the lead FORWARD"


def test_risk_direction_cannot_score_better_than_the_point_estimate():
    """The measured surface is MONOTONE toward dt<0 (0.034277 -> 0.066243). For a
    closing ego the risk-shifted contact must be at least as bad as dt=0."""
    traj, lead = _ego(v=9.0), _straight_lead(x0=5.0, v=2.0)
    ctx = {"lead_path": lead, "dt": 0.5}
    at0 = RC.contact_under_shift(traj, ctx, 0.0)
    risk = RC.contact_under_shift(traj, ctx, -1.0)
    # component is in [-1, 0]; -1 is contact, so "worse" means <=
    assert (risk <= at0 + 1e-6).all(), "the risk direction must not look safer"


def test_shift_past_the_end_clamps_and_never_invents_a_lead():
    """An extrapolated lead position is a FABRICATED obstacle. Clamp, do not extrapolate."""
    lead = _straight_lead(S=5, dt=0.5)
    far = RC.shift_track(lead, -99.0, 0.5)
    assert torch.allclose(far, lead[..., :1, :].expand_as(lead)), "must hold the first point"
    far2 = RC.shift_track(lead, +99.0, 0.5)
    assert torch.allclose(far2, lead[..., -1:, :].expand_as(lead)), "must hold the last point"


def test_absence_of_a_lead_is_neutral_not_a_penalty():
    """rewards.py:152-171 -- absence means no constraint therefore no penalty."""
    traj = _ego()
    out = RC.robust_contact(traj, {"dt": 0.5})
    assert torch.equal(out, torch.zeros_like(out))
    assert RC.make_robust_contact_component().neutral == 0.0


def test_weights_are_normalised_and_bad_ones_refused():
    traj, lead = _ego(), _straight_lead()
    ctx = {"lead_path": lead, "dt": 0.5}
    a = RC.robust_contact(traj, {**ctx, "robust_shifts_s": (-1.0, 0.0),
                                 "robust_shift_weights": (1.0, 1.0)})
    b = RC.robust_contact(traj, {**ctx, "robust_shifts_s": (-1.0, 0.0),
                                 "robust_shift_weights": (3.0, 3.0)})
    assert torch.allclose(a, b), "weights must be normalised, not absolute"
    with pytest.raises(ValueError):
        RC.robust_contact(traj, {**ctx, "robust_shifts_s": ()})
    with pytest.raises(ValueError):
        RC.robust_contact(traj, {**ctx, "robust_shifts_s": (0.0,),
                                 "robust_shift_weights": (1.0, 2.0)})
    with pytest.raises(ValueError):
        RC.robust_contact(traj, {**ctx, "robust_shifts_s": (0.0,),
                                 "robust_shift_weights": (0.0,)})


def test_component_bounds_are_declared_and_respected():
    comp = RC.make_robust_contact_component()
    assert (comp.lo, comp.hi) == (-1.0, 0.0)
    assert comp.hackable_alone is True, "standing still never contacts -- say so"
    traj, lead = _ego(v=30.0), _straight_lead(x0=1.0, v=0.0)
    v = comp(traj, {"lead_path": lead, "dt": 0.5})
    assert float(v.min()) >= -1.0 and float(v.max()) <= 0.0


def test_the_robust_term_can_see_risk_the_point_estimate_cannot():
    """The headroom argument, as a test: a fan that is clean at dt=0 need not be clean
    under timing error. If this ever reads equal, the term has stopped doing its job."""
    # ego 9 m/s closing on a 6 m/s lead 9 m ahead: the gap shrinks 9 -> 3 m and never
    # reaches the 2 m contact radius, so dt=0 is CLEAN. Rolling the lead back along its
    # own path by 1 s puts it where the ego arrives -> contact. Same two paths, same
    # predicate; only our belief about WHEN the lead is there changed.
    lead = _straight_lead(x0=9.0, v=6.0)
    traj = _ego(v=9.0)
    ctx = {"lead_path": lead, "dt": 0.5}
    point = float(R.COMPONENTS["collision"](traj, ctx).mean())
    robust = float(RC.robust_contact(traj, ctx).mean())
    assert point == 0.0, "fixture must be clean at dt=0 or it tests nothing"
    assert robust < 0.0, "the shifted grid must expose contact the point estimate misses"


# --------------------------------------------------------------------------------------
# refc_adapter -- the conditioning contract
# --------------------------------------------------------------------------------------

class _Cfg:
    def __init__(self, **kw):
        self.ego_state_inject = kw.get("ego_state_inject", False)
        self.nav_inject = kw.get("nav_inject", True)


class _Model:
    def __init__(self, **kw):
        self.cfg = _Cfg(**kw)


def test_requirements_are_read_from_the_models_own_config():
    """M51: a NAME is not provenance. The requirement comes from the declaring field."""
    assert A.conditioning_requirements(_Model(ego_state_inject=True))["ego_state"] is True
    assert A.conditioning_requirements(_Model())["ego_state"] is False


def test_a_v4b_style_build_refuses_a_batch_with_no_ego_state():
    """⛔ THE BUG THIS PREVENTS IS SILENT: forward would return a well-formed fan from a
    differently-conditioned policy and nothing would raise."""
    m = _Model(ego_state_inject=True)
    with pytest.raises(A.ConditioningError) as ei:
        A.assert_conditioning(m, {"frames": torch.zeros(1), "v0": torch.zeros(1)})
    assert "ego_state" in str(ei.value)


def test_the_same_build_accepts_the_batch_once_the_channel_is_present():
    m = _Model(ego_state_inject=True)
    req = A.assert_conditioning(m, {"frames": torch.zeros(1),
                                    "ego_state": torch.zeros(1, 5)})
    assert req["ego_state"] is True


def test_a_v3_style_build_does_not_demand_ego_state():
    m = _Model(ego_state_inject=False)
    assert A.assert_conditioning(m, {"frames": torch.zeros(1)})["ego_state"] is False


def test_nav_cmd_is_never_asserted_because_refc_is_evaluated_with_nav_none():
    """Asserting nav_cmd would refuse the programme's own standard eval arm."""
    m = _Model(nav_inject=True)
    A.assert_conditioning(m, {"frames": torch.zeros(1)})          # must not raise
    assert A._REQUIRING_FLAG["nav_cmd"] is None


def test_no_requirement_names_a_config_field_that_does_not_exist():
    """⛔ A guard keyed on a misspelt flag reads False forever -- a check that can never
    fire. Every non-None flag must be a real attribute of the real config class."""
    from tanitad.refs.refc_v3 import RefCV3Config
    for key, flag in A._REQUIRING_FLAG.items():
        if flag is not None:
            assert hasattr(RefCV3Config, flag), (
                f"{key} -> {flag!r} is not a field of RefCV3Config; the guard would "
                f"silently never fire")


def test_model_without_cfg_is_refused_rather_than_guessed():
    with pytest.raises(A.ConditioningError):
        A.conditioning_requirements(object())


def test_forward_kwargs_plumbs_every_channel_the_forward_accepts():
    """The original adapter passed 5 of 8. Pin the full set against the real signature."""
    import inspect
    from tanitad.refs.refc_v3 import RefCV3Model
    sig = set(inspect.signature(RefCV3Model.forward).parameters) - {"self", "frames", "steps"}
    assert set(A.FORWARD_KEYS) == sig, (
        f"FORWARD_KEYS {sorted(A.FORWARD_KEYS)} != forward's optional params {sorted(sig)}")
    kw = A.forward_kwargs({"frames": 1, "v0": 2, "ego_state": 3}, PostTrainConfig())
    assert set(kw) == sig | {"steps"}
    assert kw["ego_state"] == 3 and kw["nav_known"] is None


def test_sample_fn_asserts_conditioning_on_the_first_batch():
    m = _Model(ego_state_inject=True)
    fn = A.make_refc_sample_fn(m, PostTrainConfig())
    with pytest.raises(A.ConditioningError):
        fn({"frames": torch.zeros(1, 1)}, PostTrainConfig())


def test_strict_conditioning_can_be_turned_off_but_must_be_typed():
    """A deliberate un-conditioned ablation is allowed -- it just cannot be an accident."""
    m = _Model(ego_state_inject=True)
    fn = A.make_refc_sample_fn(m, PostTrainConfig(), strict_conditioning=False)
    with pytest.raises((AttributeError, TypeError, KeyError)):
        # it gets past the conditioning guard and dies in the mock's forward instead,
        # which is the proof that the guard was the only thing stopping it
        fn({"frames": torch.zeros(1, 1)}, PostTrainConfig())


def test_reusable_pieces_are_re_exported_not_forked():
    """sample_offsets/gt_context are model-agnostic; two copies would drift."""
    from tanitad.rl import refcv3_adapter as V3
    assert A.sample_offsets is V3.sample_offsets
    assert A.gt_context is V3.gt_context
