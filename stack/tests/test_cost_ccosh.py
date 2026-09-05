"""D-REFAV1-COST-GEOMETRY - `cost_metric="ccosh"`, ccos WITH THE HOLD BRANCH.

THE DEFECT THIS REPAIRS, and it is a KNOWN VALUE, not a suspicion. Under
``"ccos"`` the do-nothing candidate IS ``z_ref``, so its centred vector is the
ZERO vector and its cost is ``1.0`` EXACTLY - the worst-but-one value in the
range - for doing exactly what a HOLD goal asked. `test_cost_ccos.py::test_f_*`
pins that degeneracy on purpose and its own docstring says the hold branch is a
pre-registration item deliberately not taken there. This file takes it, as a
SEPARATE metric, so that no banked ``ccos`` number moves by a byte.

MEASURED, and the reason it is not cosmetic (`.../Research/
2026-09-05-refav1-cost-geometry/raw/cost_scale.txt`, n = 40 windows /
8 episodes, ckpt step 21,109, `dump_ccos_argmax`, cost weights (0, 0, 64.297)
so the cost IS the goal term):

  * median ``basecost_cv``  = **1.0**        (the degeneracy, end to end)
  * median winner cost      = **3.11e-05**
  * median ``max|kappa|``   = **0.08** = exactly ``GOAL_KAPPA_TURN``, on
    windows whose decoded goal is LANE_KEEP as often as not.

THE REPAIR. When ``||g - z_ref|| <= CCOS_HOLD_REL * ||z_ref||`` the goal IS the
hold field, the centred direction is float32 noise, and the cost falls back to
the pinned ``"chord"`` - the DISTANCE form - under which the do-nothing
candidate correctly scores ~0 and a moving candidate is charged for moving.

TIER: n/a (arithmetic + a tiny random-init model). EVIDENCE CLASS: MEASURED.
"""
import pytest
import torch
import torch.nn.functional as F

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import (CCOS_HOLD_REL, COST_METRICS, RefAV1,
                                  RefAV1Config, _CHORD_EPS,
                                  _check_cost_metric, _goal_term)
from tanitad.refs.refa_v1_plan import PlanConfig

#: MEASURED relative ``||g - z_ref||`` on a HOLD window (COST_METRICS note).
MEASURED_HOLD_REL = 4.18e-08


def _shipped_cos(zt, g):
    return 1.0 - F.cosine_similarity(zt.flatten(1), g.flatten(1), dim=-1)


def _shipped_chord(zt, g):
    x, y = zt.flatten(1), g.flatten(1)
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(_CHORD_EPS)
    yn = y / y.norm(dim=-1, keepdim=True).clamp_min(_CHORD_EPS)
    return (xn - yn).norm(dim=-1)


def _shipped_ccos(zt, g, z_ref):
    x, y = zt.flatten(1), g.flatten(1)
    r = z_ref.flatten(1)
    return 1.0 - F.cosine_similarity(x - r, y - r, dim=-1)


def _cfg(**kw) -> RefAV1Config:
    base = dict(tac_vocab_version="v7.0", d_enc=16, d_state=16, n_tokens=8,
                op_layers=1, op_heads=2, op_window=2, tac_layers=1,
                tac_queries=4, str_dim=8, str_layers=1,
                strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1,
                                                    n_heads=2, d_ctx=16,
                                                    d_cmd=8),
                tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1,
                                                  n_heads=2, d_intent=16))
    base.update(kw)
    return RefAV1Config(**base)


def _model(seed: int = 0, **kw) -> RefAV1:
    torch.manual_seed(seed)
    m = RefAV1(_cfg(**kw)).eval()
    m.std.fit(torch.randn(256, m.cfg.d_enc))
    return m


def _pc(c: RefAV1Config) -> PlanConfig:
    return PlanConfig(horizon=c.plan_steps, dt=c.op_dt, seed=0,
                      n_samples=8, n_iters=2, n_elites=4)


def _window(c: RefAV1Config, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, c.op_window, c.n_tokens, c.d_enc, generator=g),
            10.0, torch.randint(0, 4, (1,), generator=g))


# --------------------------------------------------------------------------- #
#  A - the three pre-existing branches are UNTOUCHED, bit for bit              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(5, 1, 64), (3, 4, 16), (7, 64, 32)])
def test_a_the_three_existing_branches_are_bit_identical(shape):
    """The addition may not move ANY banked number. Re-implemented here so a
    change inside `_goal_term` cannot hide behind the function it changed."""
    g_ = torch.Generator().manual_seed(11)
    zt = torch.randn(*shape, generator=g_)
    go = torch.randn(*shape, generator=g_)
    zr = torch.randn(1, *shape[1:], generator=g_)
    assert torch.equal(_goal_term(zt, go, "cos"), _shipped_cos(zt, go))
    assert torch.equal(_goal_term(zt, go, "chord"), _shipped_chord(zt, go))
    assert torch.equal(_goal_term(zt, go, "ccos", zr),
                       _shipped_ccos(zt, go, zr))


def test_b_the_tuple_and_the_validator_carry_the_fourth_branch():
    assert COST_METRICS == ("cos", "chord", "ccos", "ccosh")
    assert _check_cost_metric("ccosh") == "ccosh"
    with pytest.raises(ValueError):
        _check_cost_metric("ccosh2")


def test_b2_ccosh_without_a_reference_fails_LOUDLY():
    zt = torch.randn(3, 4, 8)
    with pytest.raises(ValueError, match="ccosh"):
        _goal_term(zt, zt.clone(), "ccosh")


# --------------------------------------------------------------------------- #
#  C - the degeneracy, and that ccosh removes it                               #
# --------------------------------------------------------------------------- #
def test_c_hold_goal_do_nothing_scores_ZERO_and_ccos_scores_ONE():
    """THE KNOWN-VALUE PAIR, in one breath.

    Goal == reference == candidate (a HOLD goal, honoured exactly):
      * ``ccos``  = **1.0** EXACTLY - the degeneracy, by construction
      * ``ccosh`` = **0.0** EXACTLY - the do-nothing candidate is right

    The same-breath CONTROL is that the two numbers come from the SAME call
    shape on the SAME tensors, so a 0.0 that came from an unevaluated branch
    would show up as a 0.0 on the ccos side too."""
    g_ = torch.Generator().manual_seed(3)
    r = torch.randn(1, 4, 32, generator=g_) * 315.0
    zt = r.expand(6, -1, -1).contiguous()
    goal = r.clone()
    c_ccos = _goal_term(zt, goal.expand(6, -1, -1), "ccos", r)
    c_hold = _goal_term(zt, goal.expand(6, -1, -1), "ccosh", r)
    assert torch.all(c_ccos == 1.0), f"ccos control did not read 1.0: {c_ccos}"
    assert torch.all(c_hold == 0.0), f"ccosh did not read 0.0: {c_hold}"


def test_d_hold_goal_ccos_has_NO_SIGNAL_AT_ALL_and_ccosh_ranks_correctly():
    """THE DEFECT, sharper than "the ordering is wrong": on an exactly-held
    goal ``g - z_ref`` is the ZERO vector, so `F.cosine_similarity` returns 0
    for EVERY candidate and ``ccos`` is the CONSTANT 1.0 across the whole
    population - peak-to-peak **exactly 0.0**. The search then has no signal at
    all and its winner is whatever the tie-break returns. Under ``ccosh`` the
    same population is discriminated and the do-nothing candidate is the unique
    minimum."""
    g_ = torch.Generator().manual_seed(17)
    r = torch.randn(1, 4, 32, generator=g_) * 315.0
    goal = r.clone()
    moves = torch.randn(64, 4, 32, generator=g_) * 3.0
    zt = torch.cat([r, r + moves], 0)               # row 0 = do nothing
    gexp = goal.expand(zt.shape[0], -1, -1)
    c = _goal_term(zt, gexp, "ccosh", r)
    assert float(c[0]) == 0.0
    assert int(c.argmin()) == 0, "a moving candidate beat the hold candidate"
    assert float(c[1:].min()) > 0.0
    assert float(c.max() - c.min()) > 0.0, "ccosh did not discriminate either"
    # the CONTROL that must read a KNOWN value: ccos is constant 1.0
    c2 = _goal_term(zt, gexp, "ccos", r)
    assert torch.all(c2 == 1.0), f"ccos control did not read 1.0: {c2}"
    assert float(c2.max() - c2.min()) == 0.0, (
        "ccos discriminated on an exactly-held goal - the degeneracy this "
        "test is built on does not hold, so the repair measures nothing")


def test_d2_hold_goal_AT_THE_MEASURED_NOISE_FLOOR_ccos_ranks_by_noise():
    """The realistic form of `test_d`: on the real checkpoint ``g - z_ref`` is
    not exactly zero, it is float32 rounding at **4.18e-08 relative**
    (MEASURED, COST_METRICS). The cosine against a noise direction is then
    O(1/sqrt(D)) noise, so the hold candidate does NOT win - which is how
    curvature ends up non-zero on ~90 % of GT-straight windows. ``ccosh``
    ranks the same population correctly."""
    g_ = torch.Generator().manual_seed(41)
    r = torch.randn(1, 4, 32, generator=g_) * 315.0
    d = torch.randn(1, 4, 32, generator=g_)
    goal = r + d / d.norm() * r.norm() * MEASURED_HOLD_REL
    moves = torch.randn(64, 4, 32, generator=g_) * 3.0
    zt = torch.cat([r, r + moves], 0)
    gexp = goal.expand(zt.shape[0], -1, -1)
    c2 = _goal_term(zt, gexp, "ccos", r)
    assert int(c2.argmin()) != 0, (
        "ccos ranked the hold candidate first at the measured noise floor - "
        "this draw does not exhibit the defect, so the test measures nothing")
    c = _goal_term(zt, gexp, "ccosh", r)
    assert int(c.argmin()) == 0, "ccosh did not rank the hold candidate first"


def test_e_a_REAL_goal_is_ccos_bit_for_bit():
    """Above the gate, ccosh must not perturb the repaired metric at all."""
    g_ = torch.Generator().manual_seed(23)
    r = torch.randn(1, 4, 32, generator=g_) * 315.0
    d = torch.randn(1, 4, 32, generator=g_)
    goal = r + d / d.norm() * r.norm() * 1e-2       # 1e-2 relative >> the gate
    zt = r + torch.randn(16, 4, 32, generator=g_)
    a = _goal_term(zt, goal.expand(16, -1, -1), "ccos", r)
    b = _goal_term(zt, goal.expand(16, -1, -1), "ccosh", r)
    assert torch.equal(a, b)


@pytest.mark.parametrize("rel,expect_hold", [
    (MEASURED_HOLD_REL, True),                       # the MEASURED hold value
    (CCOS_HOLD_REL * 0.5, True),
    (CCOS_HOLD_REL * 2.0, False),
    (1e-3, False),
    (1e-2, False),
])
def test_f_the_gate_sits_where_it_is_documented(rel, expect_hold):
    """The gate is a SCALE and it is auditable. It must swallow the measured
    hold value (4.18e-08 relative, i.e. float32 epsilon) and must NOT swallow
    anything that could be a real direction."""
    assert CCOS_HOLD_REL == 1e-5
    g_ = torch.Generator().manual_seed(29)
    r = torch.randn(1, 4, 32, generator=g_) * 315.0
    d = torch.randn(1, 4, 32, generator=g_)
    goal = r + d / d.norm() * r.norm() * rel
    zt = r + torch.randn(8, 4, 32, generator=g_)
    a = _goal_term(zt, goal.expand(8, -1, -1), "ccos", r)
    b = _goal_term(zt, goal.expand(8, -1, -1), "ccosh", r)
    if expect_hold:
        assert not torch.equal(a, b), f"rel={rel}: gate did NOT fire"
        assert torch.equal(b, _shipped_chord(zt, goal.expand(8, -1, -1)))
    else:
        assert torch.equal(a, b), f"rel={rel}: gate fired when it must not"


def test_g_the_default_still_changes_nothing():
    """`plan()`'s default metric is unchanged and nothing selects ccosh."""
    m = _model(seed=7)
    feats, v0, nav = _window(m.cfg, seed=7)
    pc = _pc(m.cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc)
        b = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc, cost_metric="cos")
    assert a.cost_metric == "cos"
    assert torch.equal(a.controls, b.controls)


def test_h_plan_under_ccosh_runs_and_stamps_its_metric_END_TO_END():
    """The end-to-end plumbing assertion: the flag reaches `plan()`, the
    result carries it, and the run does not crash on a real search."""
    m = _model(seed=13)
    feats, v0, nav = _window(m.cfg, seed=13)
    pc = _pc(m.cfg)
    with torch.no_grad():
        res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc,
                     cost_metric="ccosh")
    assert res.cost_metric == "ccosh"
    assert res.controls.shape == (m.cfg.plan_steps, m.cfg.a_dim)
    assert torch.isfinite(res.controls).all()
