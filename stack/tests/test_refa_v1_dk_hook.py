"""The distance-keeping term's HOOK into `refa_v1.plan` (``D-REFAV1-DK-COST``).

`test_refav1_lon_cost.py` pins the term as arithmetic. This file pins the only
two things that can go wrong once it is wired into the planner:

⛔ **A. PARITY.** Every refav1 arm banked before 2026-09-06 ran without this
term. If the default path moved by a single bit, every cross-arm comparison in
the programme would silently stop being one. `dk_spec=None` must be
BIT-IDENTICAL.

⚠️ **B. THE CONTROL THAT KEEPS A INTERESTING.** A parity test passes trivially
when nothing runs — the "guards need mutation, not inspection" rule. So every
parity assertion here is paired with a SAME-BREATH arm that MUST differ: an
armed spec on a tight gap has to change the plan, or the hook is dead code and
the parity result means nothing.

⭐ C. The COVERAGE distinction the record depends on: a spec with a positive
weight on a window with NO LEAD is **not armed for that window**. `res.dk_armed`
reports the per-window truth, `res.dk_spec` the declaration.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.refs.refa_v1 import (DistanceKeepingSpec, RefAV1, RefAV1Config,
                                  StrategicPolicyConfig, TacticalPolicyConfig)
from tanitad.refs.refa_v1_plan import PlanConfig


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


def _model(seed: int = 0) -> RefAV1:
    torch.manual_seed(seed)
    m = RefAV1(_cfg()).eval()
    m.std.fit(torch.randn(256, m.cfg.d_enc))
    return m


def _window(c: RefAV1Config, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, c.op_window, c.n_tokens, c.d_enc, generator=g),
            12.0, torch.randint(0, 4, (1,), generator=g))


def _pc(c: RefAV1Config) -> PlanConfig:
    return PlanConfig(horizon=c.plan_steps, dt=c.op_dt, seed=0,
                      n_samples=32, n_iters=3, n_elites=8)


def _plan(m, feats, v0, nav, **kw):
    kw.setdefault("cost_metric", "ccos")
    return m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg), **kw)


# ------------------------------------------------------------------ #
# A. PARITY, each paired with a same-breath control that MUST differ
# ------------------------------------------------------------------ #
def test_a_none_is_bit_identical():
    m = _model()
    feats, v0, nav = _window(m.cfg)
    base = _plan(m, feats, v0, nav)
    same = _plan(m, feats, v0, nav, dk_spec=None, lead_gap_m=4.0)
    assert torch.equal(base.controls, same.controls)
    assert float(base.cost) == float(same.cost)
    assert base.dk_armed is False and same.dk_armed is False

    # ⚠️ SAME-BREATH CONTROL: the identical call with the term ARMED on the same
    # tight gap must NOT be bit-identical, or the parity above is vacuous.
    spec = DistanceKeepingSpec(w_dk=5.0, gap_source="unit-test")
    moved = _plan(m, feats, v0, nav, dk_spec=spec, lead_gap_m=4.0)
    assert moved.dk_armed is True
    assert not torch.equal(base.controls, moved.controls)


def test_a_zero_weight_is_bit_identical():
    """A declared-but-unarmed spec (`w_dk = 0`) is still the shipped path."""
    m = _model()
    feats, v0, nav = _window(m.cfg, seed=3)
    base = _plan(m, feats, v0, nav)
    off = _plan(m, feats, v0, nav,
                dk_spec=DistanceKeepingSpec(w_dk=0.0), lead_gap_m=3.0)
    assert torch.equal(base.controls, off.controls)
    assert off.dk_armed is False
    on = _plan(m, feats, v0, nav,
               dk_spec=DistanceKeepingSpec(w_dk=5.0, gap_source="unit-test"), lead_gap_m=3.0)
    assert not torch.equal(base.controls, on.controls)      # control


@pytest.mark.parametrize("gap", [None, float("nan")])
def test_a_no_lead_is_bit_identical(gap):
    """⭐ Free flow is not a failure: an armed spec on a window with no lead
    must leave the plan exactly where it was."""
    m = _model()
    feats, v0, nav = _window(m.cfg, seed=5)
    base = _plan(m, feats, v0, nav)
    spec = DistanceKeepingSpec(w_dk=25.0, gap_source="unit-test")
    out = _plan(m, feats, v0, nav, dk_spec=spec, lead_gap_m=gap)
    assert out.dk_armed is False
    assert torch.equal(base.controls, out.controls)
    # control: the SAME spec on a real tight gap must move
    moved = _plan(m, feats, v0, nav, dk_spec=spec, lead_gap_m=3.0)
    assert not torch.equal(base.controls, moved.controls)


def test_a_wide_gap_is_bit_identical():
    """A gap far above ``d0 + tau*v0`` is a zero for the one-sided penalty on
    every candidate, so it cannot re-rank -- the term never rewards closing."""
    m = _model()
    feats, v0, nav = _window(m.cfg, seed=7)
    base = _plan(m, feats, v0, nav)
    spec = DistanceKeepingSpec(w_dk=25.0, gap_source="unit-test")
    wide = _plan(m, feats, v0, nav, dk_spec=spec, lead_gap_m=400.0)
    assert wide.dk_armed is True                     # armed, but priced at zero
    assert torch.equal(base.controls, wide.controls)


# ------------------------------------------------------------------ #
# B. the term does what it is for
# ------------------------------------------------------------------ #
def test_b_tight_gap_slows_the_plan():
    """The plan under a tight gap must command LESS longitudinal progress than
    the same plan without the term."""
    m = _model()
    feats, v0, nav = _window(m.cfg, seed=11)
    base = _plan(m, feats, v0, nav)
    tight = _plan(m, feats, v0, nav,
                  dk_spec=DistanceKeepingSpec(w_dk=10.0, gap_source="unit-test"),
                  lead_gap_m=3.0)
    assert float(tight.controls[..., 0].sum()) < float(base.controls[..., 0].sum())


def test_b_tighter_gap_is_monotone():
    m = _model()
    feats, v0, nav = _window(m.cfg, seed=13)
    spec = DistanceKeepingSpec(w_dk=10.0, gap_source="unit-test")
    a = _plan(m, feats, v0, nav, dk_spec=spec, lead_gap_m=20.0)
    b = _plan(m, feats, v0, nav, dk_spec=spec, lead_gap_m=2.0)
    assert float(b.controls[..., 0].sum()) <= float(a.controls[..., 0].sum())


# ------------------------------------------------------------------ #
# C. the declaration travels, and coverage is per-window
# ------------------------------------------------------------------ #
def test_c_spec_and_gap_travel_on_the_result():
    m = _model()
    feats, v0, nav = _window(m.cfg, seed=17)
    spec = DistanceKeepingSpec(w_dk=2.0, tau_target_s=1.2, d0_m=4.0,
                               gap_source="oracle_label")
    res = _plan(m, feats, v0, nav, dk_spec=spec, lead_gap_m=6.25)
    assert res.dk_spec["w_dk"] == 2.0
    assert res.dk_spec["tau_target_s"] == 1.2
    assert res.dk_spec["gap_source"] == "oracle_label"
    assert res.lead_gap_m == pytest.approx(6.25)
    assert res.dk_armed is True


def test_c_armed_is_per_window_not_per_spec():
    """⛔ The distinction that keeps coverage honest: the SAME spec is armed on
    a lead window and NOT armed on a free-flow one."""
    m = _model()
    feats, v0, nav = _window(m.cfg, seed=19)
    spec = DistanceKeepingSpec(w_dk=2.0, gap_source="oracle_label")
    assert _plan(m, feats, v0, nav, dk_spec=spec, lead_gap_m=6.0).dk_armed is True
    off = _plan(m, feats, v0, nav, dk_spec=spec, lead_gap_m=None)
    assert off.dk_armed is False
    assert off.dk_spec["w_dk"] == 2.0        # declared, but not armed HERE
