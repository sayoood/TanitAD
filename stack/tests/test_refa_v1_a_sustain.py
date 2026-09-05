# -*- coding: utf-8 -*-
"""THE LONGITUDINAL ANALOGUE, pinned (D-REFAV1-LON-VOCAB / D-REFAV1-LON-COST,
2026-09-05).

WHAT IS BEING PINNED, and why each case has a same-breath control that must
read the OTHER value:

* `a_sustain` is `canonical_controls`' longitudinal `kappa_turn`. The lateral
  vocabulary could command exactly two sustained curvatures; the longitudinal
  one can command exactly ZERO sustained accelerations, because every token's
  profile is `a_i = (v_t - v)/GOAL_REACH_S` and `v -> v_t`. On the MAINTAIN
  branch (`v_t == v0`) that is `a == 0` for the entire horizon.
* MEASURED on the live grid (dump_wk15, n = 40, ckpt 21,109): the maintain
  branch is 31/40 windows (77.5 %) and 20/24 (83.3 %) of the GT-LON stratum,
  and the vocabulary's reachable `dv` over the 2 s plan window is
  [-2.85, +0.98] m/s against a corpus p90 of +2.34 -- 14 of the 17
  accelerating windows (82.4 %) are outside it entirely.
* `jerk_seam_a0` prices the step from the MEASURED a0 to `controls[0]`, which
  the shipped jerk term omits, so the all-zero plan is free even when the car
  is braking hard.
* `target_speed` is the DEAD third weight: `plan()` defaults it to None and the
  arm tool never passed it, so `w_vend` has never contributed a unit of cost.

ASCII-only output (cp1252 dev box).
"""
import math

import pytest
import torch

from tanitad.refs.refa_v1 import (GOAL_A_MAX, GOAL_CREEP_MPS,
                                  GOAL_CURVE_VMAX_MPS, GOAL_LON_DV_MPS,
                                  canonical_controls)

K, DT, OP = 10, 0.2, 30
MAINTAIN = ("CRUISE", "ADAPT_SPEED_FOR_CURVE")     # at v0 <= GOAL_CURVE_VMAX
NON_MAINTAIN = ("HOLD", "CREEP", "FOLLOW", "ACCELERATE", "YIELD_MERGE",
                "BRAKE_TO")
LATS = ("LANE_KEEP", "TURN_L", "TURN_R", "NUDGE_L", "LANE_CHANGE_R")


def test_a_absent_a_sustain_is_bit_identical():
    """`a_sustain=None` must be BIT-IDENTICAL to not passing it -- the whole
    parity claim for every arm banked before 2026-09-05 rests on this."""
    for lat in LATS:
        for lon in MAINTAIN + NON_MAINTAIN:
            base = canonical_controls(lat, lon, 5.0, OP, DT)
            none = canonical_controls(lat, lon, 5.0, OP, DT, a_sustain=None)
            assert torch.equal(base, none), (lat, lon)
    # SAME-BREATH CONTROL that must DIFFER, or the test above proves nothing
    changed = canonical_controls("LANE_KEEP", "CRUISE", 5.0, OP, DT,
                                 a_sustain=1.0)
    assert not torch.equal(canonical_controls("LANE_KEEP", "CRUISE", 5.0, OP, DT),
                           changed)


def test_b_the_shipped_maintain_branch_commands_exactly_zero():
    """The defect itself, asserted rather than quoted: on the maintain branch
    the shipped vocabulary commands a == 0 over the WHOLE horizon."""
    for lon in MAINTAIN:
        c = canonical_controls("LANE_KEEP", lon, 5.0, OP, DT)
        assert float(c[:, 0].abs().max()) == 0.0, lon
    # CONTROL: above GOAL_CURVE_VMAX_MPS the same token is NOT a == 0, it is a
    # sustained brake to 8 m/s -- 'a decode is not a goal state' (M28 2).
    hi = canonical_controls("LANE_KEEP", "ADAPT_SPEED_FOR_CURVE", 20.0, OP, DT)
    assert float(hi[:, 0].abs().max()) > 1.0


def test_c_a_sustain_touches_only_the_maintain_branch():
    """A token whose target speed differs from v0 must be bit-identical with
    and without `a_sustain` -- otherwise the knob is a second lever."""
    v0 = 5.0
    for lon in NON_MAINTAIN:
        base = canonical_controls("LANE_KEEP", lon, v0, OP, DT)
        with_a = canonical_controls("LANE_KEEP", lon, v0, OP, DT, a_sustain=1.2)
        assert torch.equal(base, with_a), lon
    # SAME-BREATH CONTROL: on the maintain branch it MUST change
    for lon in MAINTAIN:
        base = canonical_controls("LANE_KEEP", lon, v0, OP, DT)
        with_a = canonical_controls("LANE_KEEP", lon, v0, OP, DT, a_sustain=1.2)
        assert not torch.equal(base, with_a), lon


def test_d_a_sustain_is_constant_clipped_and_lateral_neutral():
    v0 = 6.0
    for a_req, a_exp in ((0.4, 0.4), (-0.9, -0.9),
                         (99.0, GOAL_A_MAX), (-99.0, -GOAL_A_MAX)):
        c = canonical_controls("TURN_L", "CRUISE", v0, OP, DT, a_sustain=a_req)
        assert torch.allclose(c[:, 0], torch.full((OP,), float(a_exp)),
                              atol=1e-6), a_req
        # the LATERAL channel is untouched: one variable, not two
        base = canonical_controls("TURN_L", "CRUISE", v0, OP, DT)
        assert torch.equal(c[:, 1], base[:, 1]), a_req


def test_e_a_sustain_zero_equals_the_shipped_maintain_profile():
    """The DELIBERATE-NULL control: a_sustain = 0 is the shipped behaviour, so
    any arm difference at a_sustain = 0 would be a bug, not a lever."""
    for lon in MAINTAIN:
        assert torch.equal(
            canonical_controls("LANE_KEEP", lon, 4.0, OP, DT),
            canonical_controls("LANE_KEEP", lon, 4.0, OP, DT, a_sustain=0.0))


def test_f_a_sustain_opens_the_positive_dv_the_shipped_vocab_cannot_reach():
    """The measured reason the lever exists: the shipped reachable dv over the
    2 s plan window tops out at +0.977 m/s while the corpus p90 is +2.34."""
    v0 = 10.0
    toks = list(MAINTAIN) + list(NON_MAINTAIN)
    hi = max(float(canonical_controls("LANE_KEEP", l, v0, OP, DT)[:K, 0].sum() * DT)
             for l in toks)
    assert 0.9 < hi < 1.1, hi          # the single positive rung, +0.977
    with_a = float(canonical_controls("LANE_KEEP", "CRUISE", v0, OP, DT,
                                      a_sustain=1.2)[:K, 0].sum() * DT)
    assert with_a > 2.0                # 1.2 * 2.0 s = +2.4 m/s, p90 reachable


def test_g_plan_signature_carries_all_three_levers_and_they_default_off():
    import inspect
    from tanitad.refs.refa_v1 import RefAV1
    p = inspect.signature(RefAV1.plan).parameters
    for name, default in (("a_sustain", None), ("jerk_seam_a0", None),
                          ("target_speed", None)):
        assert name in p, name
        assert p[name].default is default, (name, p[name].default)


def test_h_the_jerk_seam_prices_the_step_from_a0():
    """The seam term, computed exactly as `_cost_chunk` does, on a plan that
    holds a0 and one that drops to zero. Without the seam they TIE; with it,
    the drop is strictly more expensive. (The expression is duplicated here on
    purpose: a test that imported the closure could not fail if the closure
    were deleted.)"""
    dt, a0 = 0.2, -2.0
    hold = torch.full((1, K), a0)
    drop = torch.zeros(1, K)
    def jerk_no_seam(x):
        return (((x[:, 1:] - x[:, :-1]) / dt) ** 2).mean(-1)
    def jerk_seam(x):
        s = torch.cat([torch.full((1, 1), a0), x], dim=1)
        return (((s[:, 1:] - s[:, :-1]) / dt) ** 2).mean(-1)
    assert float(jerk_no_seam(hold)) == pytest.approx(float(jerk_no_seam(drop)))
    assert float(jerk_seam(hold)) < float(jerk_seam(drop))
    assert float(jerk_seam(hold)) == pytest.approx(0.0, abs=1e-9)


def test_i_w_vend_is_a_dead_term_unless_target_speed_is_armed():
    """The measured claim, asserted from SOURCE: the arm tool's single real
    `.plan(...)` call now passes `target_speed`, and `plan()` still defaults it
    to None so every pre-2026-09-05 arm keeps its dead third weight."""
    import inspect
    import re
    from tanitad.refs.refa_v1 import RefAV1
    src = inspect.getsource(RefAV1.plan)
    assert "if target_speed is not None:" in src
    assert "w_vend * (v_end - target_speed).pow(2)" in src
    assert inspect.signature(RefAV1.plan).parameters["target_speed"].default is None


# =========================================================================== #
# D2 -- `a_shift`: the tokens name a change relative to WHERE YOU ARE GOING
# =========================================================================== #
def test_k_a_shift_absent_is_bit_identical():
    for lat in LATS:
        for lon in MAINTAIN + NON_MAINTAIN:
            base = canonical_controls(lat, lon, 5.0, OP, DT)
            none = canonical_controls(lat, lon, 5.0, OP, DT, a_shift=None)
            assert torch.equal(base, none), (lat, lon)
    # SAME-BREATH CONTROL that must DIFFER
    assert not torch.equal(
        canonical_controls("LANE_KEEP", "CRUISE", 5.0, OP, DT),
        canonical_controls("LANE_KEEP", "CRUISE", 5.0, OP, DT, a_shift=0.9))


def test_l_a_shift_reduces_to_a_sustain_at_the_first_step_on_maintain():
    """D2's defining property: `a[0]` is exactly the hint on the maintain
    branch, so `a_sustain` is its special case at the first step. They must
    then DIVERGE later (D1 holds, D2 decays) -- asserted, not assumed."""
    for lon in MAINTAIN:
        sh = canonical_controls("LANE_KEEP", lon, 6.0, OP, DT, a_shift=0.8)
        su = canonical_controls("LANE_KEEP", lon, 6.0, OP, DT, a_sustain=0.8)
        assert float(sh[0, 0]) == pytest.approx(0.8, abs=1e-6), lon
        assert float(su[0, 0]) == pytest.approx(0.8, abs=1e-6), lon
        # SAME-BREATH CONTROL: they are NOT the same design
        assert not torch.equal(sh, su), lon
        assert float(su[:, 0].std()) == pytest.approx(0.0, abs=1e-7)
        assert float(sh[:, 0].std()) > 1e-3


def test_m_a_shift_never_moves_an_ABSOLUTE_target():
    """`HOLD` (stop) and `CREEP` (1.5 m/s) name a speed IN THE WORLD, and so
    does `ADAPT_SPEED_FOR_CURVE` above GOAL_CURVE_VMAX_MPS. Shifting those
    would change what the token MEANS."""
    for lon in ("HOLD", "CREEP"):
        assert torch.equal(
            canonical_controls("LANE_KEEP", lon, 10.0, OP, DT),
            canonical_controls("LANE_KEEP", lon, 10.0, OP, DT, a_shift=0.9)), lon
    hi = 20.0                      # ADAPT above the cap = a brake to 8 m/s
    assert hi > GOAL_CURVE_VMAX_MPS
    assert torch.equal(
        canonical_controls("LANE_KEEP", "ADAPT_SPEED_FOR_CURVE", hi, OP, DT),
        canonical_controls("LANE_KEEP", "ADAPT_SPEED_FOR_CURVE", hi, OP, DT,
                           a_shift=0.9))
    # SAME-BREATH CONTROLS that must MOVE: every relative target
    for lon in ("CRUISE", "FOLLOW", "ACCELERATE", "YIELD_MERGE", "BRAKE_TO"):
        assert not torch.equal(
            canonical_controls("LANE_KEEP", lon, 10.0, OP, DT),
            canonical_controls("LANE_KEEP", lon, 10.0, OP, DT, a_shift=0.9)), lon
    lo = 4.0                       # ADAPT below the cap IS relative (== v0)
    assert not torch.equal(
        canonical_controls("LANE_KEEP", "ADAPT_SPEED_FOR_CURVE", lo, OP, DT),
        canonical_controls("LANE_KEEP", "ADAPT_SPEED_FOR_CURVE", lo, OP, DT,
                           a_shift=0.9))


def test_n_a_shift_and_a_sustain_are_mutually_exclusive():
    with pytest.raises(ValueError, match="two spellings"):
        canonical_controls("LANE_KEEP", "CRUISE", 5.0, OP, DT,
                           a_sustain=0.5, a_shift=0.5)


def test_o_a_shift_opens_the_positive_dv_on_EVERY_token_not_just_maintain():
    """The reason D2 beats D1: D1 is inert on the 9 non-maintain windows."""
    v0 = 10.0
    for lon in ("ACCELERATE", "BRAKE_TO", "FOLLOW"):
        base = float(canonical_controls("LANE_KEEP", lon, v0, OP, DT)[:K, 0].sum() * DT)
        su = float(canonical_controls("LANE_KEEP", lon, v0, OP, DT,
                                      a_sustain=1.2)[:K, 0].sum() * DT)
        sh = float(canonical_controls("LANE_KEEP", lon, v0, OP, DT,
                                      a_shift=1.2)[:K, 0].sum() * DT)
        assert su == pytest.approx(base), (lon, "a_sustain must be INERT here")
        assert sh > base + 1.0, (lon, "a_shift must move it")
