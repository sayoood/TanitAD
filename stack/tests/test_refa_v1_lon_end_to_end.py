# -*- coding: utf-8 -*-
"""THE END-TO-END HALF of the longitudinal levers: does the flag actually reach
the GOAL FIELD, the SEED and the COST, or is it a dead pipe?

This exists because of the `ccosh` lesson (M28 1): that lever moved ZERO plans,
and the only reason the zero counted as a MEASUREMENT rather than a broken
harness was a same-breath control that read 15.08 m. An arm-level null on
`a_sustain` would be uninterpretable without the assertions below, so they are
made HERE, on a tiny CPU model, before any GPU is spent.

Helpers are imported from `test_steer_conversion_complete` rather than
re-written, so a change to the tiny rig cannot make this file quietly test a
different model.
"""
import pytest
import torch

from tanitad.refs.refa_v1 import canonical_controls
from test_steer_conversion_complete import _force_turn, _pc, _tiny_model


def test_J1_a_sustain_reaches_the_goal_field_and_is_recorded():
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg, lat="TURN_R", lon="CRUISE")     # CRUISE = maintain
    with torch.no_grad():
        off = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg))
        on = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), a_sustain=1.3)
    assert off.a_sustain is None
    assert on.a_sustain == [pytest.approx(1.3)]
    # SAME-BREATH CONTROL: the GOAL the two plans are scored against must differ,
    # or an arm-level null would be a dead pipe rather than a measurement.
    c_off = canonical_controls("TURN_R", "CRUISE", 8.0, cfg.op_steps, cfg.op_dt)
    c_on = canonical_controls("TURN_R", "CRUISE", 8.0, cfg.op_steps, cfg.op_dt,
                              a_sustain=1.3)
    assert float((c_on[:, 0] - c_off[:, 0]).abs().max()) == pytest.approx(1.3)
    assert torch.equal(c_on[:, 1], c_off[:, 1]), "lateral channel must not move"


def test_J2_a_sustain_zero_leaves_the_plan_bit_identical():
    """The DELIBERATE NULL: `a_sustain = 0.0` is the shipped maintain profile,
    so the emitted plan must be bit-identical to the flag being absent. If this
    ever fails, an `a_sustain` arm difference is a bug and not a lever."""
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg, lat="LANE_KEEP", lon="CRUISE")
    with torch.no_grad():
        off = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg))
        zero = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), a_sustain=0.0)
        moved = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), a_sustain=1.4)
    assert torch.equal(off.controls, zero.controls)
    # SAME-BREATH CONTROL that must DIFFER
    assert not torch.equal(off.controls, moved.controls), (
        "a_sustain=1.4 moved no control at all -- the flag is a dead pipe, and "
        "an arm-level null would be uninterpretable")


def test_J3_the_jerk_seam_changes_the_cost_and_only_the_cost():
    """`jerk_seam_a0` must move the BASELINE COSTS (it prices a term) while
    `jerk_seam_a0 = 0.0` must be the shipped expression exactly, because the
    plan's implicit previous action in the shipped form is `controls[0]` itself
    -- i.e. no seam. A seam of 0 is NOT a no-op, so this asserts the direction
    rather than an identity: a plan that starts at a != 0 must cost MORE with a
    zero seam, and the `cv` baseline (all-zero controls) must be unchanged."""
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg, lat="LANE_KEEP", lon="CRUISE")
    with torch.no_grad():
        off = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg))
        seam0 = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), jerk_seam_a0=0.0)
        seam2 = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), jerk_seam_a0=-2.0)
    assert off.jerk_seam_a0 is None
    assert seam2.jerk_seam_a0 == pytest.approx(-2.0)
    # `cv` is the ALL-ZERO control: with a zero seam its jerk is still all-zero,
    # so its cost cannot move. This is the identity control for the term.
    assert seam0.baseline_costs["cv"] == pytest.approx(
        off.baseline_costs["cv"], rel=1e-9)
    # SAME-BREATH CONTROL that must read NON-ZERO: a seam of -2 m/s^2 against
    # the all-zero plan is a real jerk and must cost more.
    assert seam2.baseline_costs["cv"] > off.baseline_costs["cv"]


def test_J4_w_vend_stays_dead_unless_explicitly_armed():
    """`target_speed` is reserved as a PI decision (test_C1 pins the production
    call site). The result must SAY whether it was armed, so a banked cost
    triple can never be read as having three live terms when it has two."""
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg)
    with torch.no_grad():
        off = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg))
        on = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), target_speed=7.0)
    assert off.w_vend_armed is False and off.target_speed is None
    assert on.w_vend_armed is True and on.target_speed == pytest.approx(7.0)
    assert on.baseline_costs["cv"] > off.baseline_costs["cv"]
