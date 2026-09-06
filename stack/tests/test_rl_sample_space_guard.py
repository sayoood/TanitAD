"""The sample-space refusal, PROVED BY MUTATION -- and the silent-no-op logp.

⛔ WHY BY MUTATION AND NOT BY INSPECTION. A guard that is only read is not a
guard: this programme has an AST census that read **0 suspects on BOTH the fixed
and the broken trainer**. Every test below REINTRODUCES the defect and requires
the refusal to fire, and each also shows the guard is DISCRIMINATING -- the
legitimate metre-space arm is still allowed, or the refusal would be a
tautology.

THE DEFECT THESE PIN, MEASURED 2026-09-06:
  1. `refcv3_adapter.sample_offsets` scales the emitted OFFSET WAYPOINTS and has
     no control space, so EVERY arm in the table explored in METRE space -- which
     `E-DDA-3c` section 4 pre-registers as the DELIBERATE REGRESSION `reg_metre`.
     Launching `rl` executed the regression under the hypothesis' name.
  2. `control_space.sample_control_space`'s original `logp` was the density of the
     TWO SCALARS, a function of the DRAW alone: `requires_grad` False, no
     `grad_fn`, and `(-(logp*adv).sum()).backward()` raised *"element 0 of
     tensors does not require grad"*. Wiring it in gives a crash or a policy
     gradient that is IDENTICALLY ZERO -- a silent no-op arm.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.rl import control_space as CS                      # noqa: E402
from tanitad.rl.config import (SAMPLE_SPACES, PostTrainConfig,  # noqa: E402
                               SampleSpaceRefusal,
                               assert_arm_sample_space)


HONEST = {"sample_space": "control", "hypothesis_arm": True}
REGRESSION = {"sample_space": "metre", "hypothesis_arm": False}


# --------------------------------------------------------------------------- #
# 1. THE REFUSAL, EACH BRANCH REACHED BY MUTATING A GOOD SPEC                    #
# --------------------------------------------------------------------------- #
def test_the_honest_arm_is_allowed():
    assert assert_arm_sample_space("rl", dict(HONEST)) == ("control", True)


def test_MUTATION_hypothesis_arm_in_metre_space_is_REFUSED():
    """The exact pre-fix state: `rl`, metre space. It must not be launchable."""
    broken = dict(HONEST)
    broken["sample_space"] = "metre"          # <- reintroduce the defect
    with pytest.raises(SampleSpaceRefusal) as e:
        assert_arm_sample_space("rl", broken)
    assert "reg_metre" in str(e.value)
    assert "REFUSING TO LAUNCH" in str(e.value)


def test_MUTATION_undeclared_arm_is_REFUSED():
    for missing in ("sample_space", "hypothesis_arm"):
        broken = dict(HONEST)
        broken.pop(missing)
        with pytest.raises(SampleSpaceRefusal, match="does not DECLARE"):
            assert_arm_sample_space("new_arm", broken)


def test_MUTATION_unknown_space_is_REFUSED():
    broken = dict(HONEST)
    broken["sample_space"] = "waypoint"
    with pytest.raises(SampleSpaceRefusal, match="must be one of"):
        assert_arm_sample_space("rl", broken)


def test_the_guard_is_DISCRIMINATING_not_a_blanket_ban():
    """`reg_metre` IS metre space and MUST still be launchable -- it is the
    pre-registered regression. A guard that refused it too would prove nothing."""
    assert assert_arm_sample_space("reg_metre", dict(REGRESSION)) == ("metre", False)


def test_every_legal_space_is_reachable():
    for sp in SAMPLE_SPACES:
        assert assert_arm_sample_space(
            "x", {"sample_space": sp, "hypothesis_arm": False})[0] == sp


# --------------------------------------------------------------------------- #
# 2. THE FIELD IS VALIDATED AND RECORDED                                        #
# --------------------------------------------------------------------------- #
def _cfg(**kw):
    base = dict(method="grpo", group_size=4, reward_weights={"progress": 1.0},
                out_dir=".", steps=1)
    base.update(kw)
    return PostTrainConfig(**base)


def test_config_refuses_an_unknown_sample_space():
    c = _cfg(sample_space="waypoint")
    with pytest.raises(ValueError, match="sample_space"):
        c.validate()


def test_config_accepts_both_and_records_it():
    for sp in SAMPLE_SPACES:
        c = _cfg(sample_space=sp)
        c.validate()
        assert c.to_dict()["sample_space"] == sp, "the record must carry the space"


# --------------------------------------------------------------------------- #
# 3. THE SILENT-NO-OP logp, MEASURED AND THEN REFUSED                           #
# --------------------------------------------------------------------------- #
def _fixture(requires_grad=True):
    state0 = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    leaf = torch.zeros(1, 2, 4, 2, requires_grad=requires_grad)
    controls = leaf + torch.tensor([0.5, 0.01])
    return state0, leaf, controls


def test_MEASURED_the_scalar_logp_carries_NO_policy_gradient():
    """The defect, reproduced: the scalar density is a function of the DRAW."""
    state0, _leaf, controls = _fixture()
    _traj, logp, _ctl = CS.sample_control_space(
        state0, controls, group=4, dt=0.5, logp_mode="scalar")
    assert not logp.requires_grad and logp.grad_fn is None
    with pytest.raises(RuntimeError, match="does not require grad"):
        (-(logp * torch.randn_like(logp)).sum()).backward()


def test_the_guard_REFUSES_the_scalar_logp():
    state0, _leaf, controls = _fixture()
    _traj, logp, _ctl = CS.sample_control_space(
        state0, controls, group=4, dt=0.5, logp_mode="scalar")
    with pytest.raises(CS.ControlSpaceError, match="SILENT"):
        CS.assert_carries_policy_gradient(logp)


def test_the_policy_logp_moves_the_POLICY_and_the_guard_passes_it():
    state0, leaf, controls = _fixture()
    _traj, logp, _ctl = CS.sample_control_space(
        state0, controls, group=4, dt=0.5, logp_mode="policy")
    CS.assert_carries_policy_gradient(logp)
    (-(logp * torch.randn_like(logp)).sum()).backward()
    assert leaf.grad is not None
    assert float(leaf.grad.abs().sum()) > 0.0


def test_unknown_logp_mode_is_refused():
    state0, _leaf, controls = _fixture()
    with pytest.raises(CS.ControlSpaceError, match="logp_mode"):
        CS.sample_control_space(state0, controls, group=4, dt=0.5,
                                logp_mode="reinforce")


# --------------------------------------------------------------------------- #
# 4. THE WHOLE CONTROL-SPACE PATH, END TO END FROM A WAYPOINT FAN                #
# --------------------------------------------------------------------------- #
def test_waypoint_fan_round_trips_through_the_inverse_map():
    """`controls_from_path` -> `roll_controls` must reproduce a path the unicycle
    can actually fly. The inverse is the programme's SINGLE one, imported."""
    v0 = 10.0
    path = torch.tensor([[[5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0]]])
    ctl = CS.controls_from_path(path[None], dt=0.5)              # [1, 1, 4, 2]
    st0 = torch.tensor([[[0.0, 0.0, 0.0, v0]]])
    back = CS.roll_controls(st0, ctl, dt=0.5)
    assert torch.allclose(back, path[None], atol=1e-3)


def test_control_space_samples_are_FLYABLE_and_metre_space_ones_need_not_be():
    """The discriminating property. A control-space draw reads EXACTLY 0.0
    envelope violation; the metre-space arm on the same fan does violate, so
    'flyable by construction' is a claim with content."""
    from tanitad.rl.config import PostTrainConfig as PTC
    from tanitad.rl.refcv3_adapter import sample_offsets
    torch.manual_seed(0)
    v0 = 12.0
    fan = torch.tensor([[5.0, 0.0], [11.0, 0.4], [18.0, 1.2], [26.0, 2.6]])
    fan = fan.reshape(1, 1, 4, 2)
    ctl0 = CS.controls_from_path(fan, dt=0.5)
    st0 = torch.tensor([[0.0, 0.0, 0.0, v0]])
    _traj, _logp, ctl = CS.sample_control_space(st0, ctl0, group=8, dt=0.5,
                                                sigma=25 * CS.SIGMA_EXPLORE_FLOOR)
    assert float(CS.envelope_violation(ctl).max()) == 0.0

    cfg = PTC(method="grpo", group_size=8, noise_mode="two_scalar", noise_scale=1.0,
              reward_weights={"progress": 1.0}, out_dir=".", steps=1,
              sample_space="metre")
    off = fan.clone()
    samp, _lp = sample_offsets(off, cfg)
    ctl_m = CS.controls_from_path(samp, dt=0.5)
    assert float(CS.envelope_violation(ctl_m).max()) > 0.0, (
        "the metre-space arm did not violate the envelope on this fixture, so "
        "'flyable by construction' would be a tautology here")
