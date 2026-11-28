"""`goal_keeps_seed` — the DE-CONFOUNDED supplied-goal path.

⭐ WHY THIS FILE EXISTS. `plan(goal_field=...)` sets `goal_action = None`, and
`goal_action` is the ONLY thing that puts the canonical seed into the iCEM
pool. So the "oracle goal" arm the programme banked (`cl_oraclegoal`, ADE
7.4708) differs from the shipped arm in **two** things — the goal AND the
missing seed — and therefore **bounds nothing**. MEASURED on the Stage-B dump:
`goal_source` = `supplied` 142/142 against `tactical_imagined` 142/142 for
`cl` (`.../2026-09-05-refav1-make-it-drive/ORACLE_GOAL.md`).

`goal_keeps_seed=True` keeps the head's own seed and replaces ONLY the goal
field, so the goal becomes the single difference from `cl`.

⛔ THE PARITY CLAIM IS LOAD-BEARING. Every arm banked before this change ran
without the argument. `test_a_*` pins that the default path is bit-identical.

⚠️ Parity tests rot into vacuity — a test that passes because nothing ran at
all proves nothing. Every parity assertion here is paired with a same-breath
control that MUST differ.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.refs.refa_v1 import (RefAV1, RefAV1Config, StrategicPolicyConfig,
                                  TacticalPolicyConfig)
from tanitad.refs.refa_v1_plan import PlanConfig
from tanitad.models.v6 import tactical_lat_actions


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
            10.0, torch.randint(0, 4, (1,), generator=g))


def _pc(c: RefAV1Config) -> PlanConfig:
    return PlanConfig(horizon=c.plan_steps, dt=c.op_dt, seed=0,
                      n_samples=32, n_iters=3, n_elites=8)


def _plan(m, feats, v0, nav, **kw):
    """`cost_metric` defaults to ``"ccos"`` for the same reason as
    `test_refa_v1_lat_bias.py`: under ``"cos"`` a decoded TURN is refused on
    0/38 real windows, so a test written against it asserts that nothing
    happens and passes for the wrong reason."""
    kw.setdefault("cost_metric", "ccos")
    return m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg),
                  model_action_units="kappa", **kw)


def _oracle_field(m):
    """The same construction `refav1_arm.py` uses: the adapter/standardiser
    image of the (here: synthetic) future at the plan horizon."""
    hp = m.cfg.plan_steps
    fut = torch.randn(1, hp, m.cfg.n_tokens, m.cfg.d_enc,
                      generator=torch.Generator().manual_seed(7))
    return m.adapter(m.std(fut))[:, hp - 1]


# --------------------------------------------------------------------------- #
#  a. PARITY — the default path is untouched                                   #
# --------------------------------------------------------------------------- #
def test_a_absent_flag_is_bit_identical_to_false():
    """Not passing the argument == passing False, on the NORMAL (no
    `goal_field`) path — the one every banked arm ran."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    a = _plan(m, feats, v0, nav)
    b = _plan(m, feats, v0, nav, goal_keeps_seed=False)
    assert torch.equal(a.controls, b.controls)
    assert a.goal_source == b.goal_source == "tactical_imagined"
    # ⛔ SAME-BREATH CONTROL: the comparison must be capable of showing a
    # difference at all, or the equality above is vacuous.
    c = _plan(m, feats, v0, nav, goal_field=_oracle_field(m))
    assert not torch.equal(a.controls, c.controls), \
        "a supplied goal changed nothing — this test cannot detect anything"


def test_a2_flag_is_inert_when_no_goal_field_is_supplied():
    """`goal_keeps_seed` only means something beside a supplied field. With
    none it must be a no-op, or it is a second lever hiding in the flag."""
    m = _model()
    feats, v0, nav = _window(m.cfg, seed=3)
    a = _plan(m, feats, v0, nav, goal_keeps_seed=False)
    b = _plan(m, feats, v0, nav, goal_keeps_seed=True)
    assert torch.equal(a.controls, b.controls)
    assert a.goal_source == b.goal_source == "tactical_imagined"


# --------------------------------------------------------------------------- #
#  b. THE CONFOUND, REPRODUCED — so the fix is measured against it             #
# --------------------------------------------------------------------------- #
def test_b_supplied_goal_alone_drops_the_seed():
    """⛔ THE DEFECT `cl_oraclegoal` CARRIES. A supplied goal leaves
    `goal_action = None`, so the canonical seed never enters the pool. This
    pins it so a future refactor cannot quietly "fix" the confounded arm and
    make the banked 7.4708 unreproducible."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    r = _plan(m, feats, v0, nav, goal_field=_oracle_field(m))
    assert r.goal_source == "supplied"
    assert r.goal_action is None, "the seed-less confound is gone; re-read " \
                                  "ORACLE_GOAL.md before changing this test"
    # same-breath control: the normal path DOES build the seed
    n = _plan(m, feats, v0, nav)
    assert n.goal_source == "tactical_imagined" and n.goal_action is not None


# --------------------------------------------------------------------------- #
#  c. THE FIX — the seed comes back, and ONLY the goal differs from `cl`       #
# --------------------------------------------------------------------------- #
def test_c_keeps_seed_restores_the_seed_and_stamps_provenance():
    m = _model()
    feats, v0, nav = _window(m.cfg)
    r = _plan(m, feats, v0, nav, goal_field=_oracle_field(m),
              goal_keeps_seed=True)
    assert r.goal_source == "supplied+seed"
    assert r.goal_action is not None
    assert getattr(r, "goal_keeps_seed") is True
    # ⭐ the seed is the SHIPPED one: identical tokens and identical canonical
    # controls to the normal arm on the same window, because the head is fed
    # the same inputs. If this ever fails the arm is not a single-variable
    # manipulation and cannot bound anything either.
    n = _plan(m, feats, v0, nav)
    assert r.goal_action["lat"] == n.goal_action["lat"]
    assert r.goal_action["lon"] == n.goal_action["lon"]
    assert torch.equal(r.goal_action["controls"], n.goal_action["controls"])


def test_c2_keeps_seed_actually_changes_the_plan():
    """A flag whose effect the search cannot see is not a de-confounding. The
    seed re-entering the pool must move the returned controls away from the
    seed-less supplied arm.

    ⚠️ THE BIAS IS NOT DECORATION. On this random-init model the head decodes
    `LANE_KEEP`/`CRUISE`, whose `canonical_controls` are EXACTLY ZERO — which
    is bit-identical to the `cv` baseline the pool already carries, so adding
    it provably cannot change anything and the test would fail for a reason
    that says nothing about the flag. MEASURED: both arms returned all-zero
    controls. Forcing a TURN_L decode gives the seed something to carry, and
    the assertion below then tests the mechanism instead of the window."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    gf = _oracle_field(m)
    bias = torch.zeros(len(tactical_lat_actions("v7.0")))
    bias[tactical_lat_actions("v7.0").index("TURN_L")] = 50.0
    plain = _plan(m, feats, v0, nav, goal_field=gf, lat_logit_bias=bias)
    kept = _plan(m, feats, v0, nav, goal_field=gf, goal_keeps_seed=True,
                 lat_logit_bias=bias)
    # ⛔ the premise, asserted rather than assumed: the seed is NON-ZERO, so
    # the pool really did change.
    assert kept.goal_action["lat"] == "TURN_L"
    assert float(kept.goal_action["controls"][:, 1].abs().max()) > 0.0
    assert not torch.equal(plain.controls, kept.controls)


def test_c3_the_goal_is_the_only_difference_from_cl():
    """⭐ THE ATTRIBUTION CLAIM, PINNED. Against the shipped arm, the kept-seed
    oracle differs in `goal_source` (hence the goal field) and in NOTHING the
    result reports about the search: same cost metric, same weights, same
    action units, same goal time grid."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    n = _plan(m, feats, v0, nav)
    r = _plan(m, feats, v0, nav, goal_field=_oracle_field(m),
              goal_keeps_seed=True)
    for attr in ("cost_metric", "cost_weights", "goal_time_grid",
                 "goal_space"):
        assert getattr(n, attr, None) == getattr(r, attr, None), attr
    assert n.goal_source != r.goal_source


# --------------------------------------------------------------------------- #
#  d. THE GUARD — `goal_time_grid="plan"` would destroy the supplied field     #
# --------------------------------------------------------------------------- #
def test_d_plan_grid_is_refused_not_silently_wrong():
    """Under `goal_time_grid="plan"` the goal is RE-ROLLED from the seed, so a
    kept-seed oracle arm would silently stop being an oracle arm. It must
    raise rather than produce a mislabelled number."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    with pytest.raises(ValueError, match="goal_keeps_seed"):
        _plan(m, feats, v0, nav, goal_field=_oracle_field(m),
              goal_keeps_seed=True, goal_time_grid="plan")
    # same-breath control: the same call WITHOUT the flag is accepted, so the
    # refusal is about the combination and not about `goal_time_grid` itself.
    ok = _plan(m, feats, v0, nav, goal_field=_oracle_field(m),
               goal_time_grid="plan")
    assert ok.goal_source == "supplied"
