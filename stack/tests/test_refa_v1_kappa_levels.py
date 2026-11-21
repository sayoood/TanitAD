"""M15's L=3 lateral vocabulary — the level set, its chooser, and its parity.

⭐ WHAT M15 APPROVED. `canonical_controls` gains THREE sustained curvature
magnitudes at the corpus's own quantiles instead of one. MEASURED on the dense
bank (4,520 windows, 644 real turns): expressible turns **0.3866 -> 1.0000**,
medAE-on-turns **0.01551 -> 0.00421** (3.7x), at the same RMSE.

⛔⛔ AND WHAT IT DID NOT SUPPLY: A CHOOSER. That table is an ORACLE bound. The
v7.0 head has ONE `TURN_L` and ONE `TURN_R` slot and `kappa_turn` never touches
its logits — MEASURED consequence: `turns goaled correctly` is **0.2811 for
every `kappa_turn`**. So a level set cannot be selected by the trained head, and
`goal_kappa_hint` is REQUIRED rather than defaulted: a default would hide which
tier the arm is (a true-kappa hint is T0; a predicted one is T1 and composes
with its own error).

⛔ PARITY IS THE LOAD-BEARING CLAIM. Every refav1 number banked before
2026-09-05 is under the shipped single magnitude. `test_a_*` pins that
`goal_kappa_levels=None` is bit-identical, and every parity assertion is paired
with a same-breath control that MUST differ — a parity test that passes because
nothing ran proves nothing.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.refs.refa_v1 import (GOAL_KAPPA_TURN, GOAL_KAPPA_TURN_LEVELS,
                                  GOAL_KAPPA_VOCAB_SHIPPED, RefAV1,
                                  RefAV1Config, StrategicPolicyConfig,
                                  TacticalPolicyConfig, canonical_controls,
                                  canonical_controls_levels,
                                  choose_kappa_level, goal_kappa_vocab_id)
from tanitad.refs.refa_v1_plan import PlanConfig
from tanitad.models.v6 import tactical_lat_actions

LAT = list(tactical_lat_actions("v7.0"))
K, DT = 30, 0.2


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
    kw.setdefault("cost_metric", "ccos")
    return m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg),
                  model_action_units="kappa", **kw)


# --------------------------------------------------------------------------- #
#  a. PARITY — the shipped path is untouched                                   #
# --------------------------------------------------------------------------- #
def test_a_absent_level_set_is_bit_identical():
    m = _model()
    feats, v0, nav = _window(m.cfg)
    a = _plan(m, feats, v0, nav)
    b = _plan(m, feats, v0, nav, goal_kappa_levels=None)
    assert torch.equal(a.controls, b.controls)
    assert a.goal_kappa_vocab == b.goal_kappa_vocab == GOAL_KAPPA_VOCAB_SHIPPED
    assert a.goal_kappa_levels is None
    # ⛔ SAME-BREATH CONTROL: a level set that is NOT the shipped magnitude must
    # change the goal's control profile, or the parity above is vacuous.
    c = _plan(m, feats, v0, nav, goal_kappa_levels=(0.5,), goal_kappa_hint=0.5,
              lat_logit_bias=_turn_bias())
    d = _plan(m, feats, v0, nav, lat_logit_bias=_turn_bias())
    assert not torch.equal(c.goal_action["controls"],
                           d.goal_action["controls"])


def _turn_bias() -> torch.Tensor:
    b = torch.zeros(len(LAT))
    b[LAT.index("TURN_L")] = 50.0
    return b


def test_a2_single_level_equal_to_shipped_reproduces_the_shipped_profile():
    """A one-level set at exactly `GOAL_KAPPA_TURN` must be BIT-IDENTICAL to
    the shipped path. If it is not, the level machinery is not a pure
    generalisation and every banked number is at risk."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    bias = _turn_bias()
    a = _plan(m, feats, v0, nav, lat_logit_bias=bias)
    b = _plan(m, feats, v0, nav, lat_logit_bias=bias,
              goal_kappa_levels=(GOAL_KAPPA_TURN,), goal_kappa_hint=999.0)
    assert a.goal_action["lat"] == b.goal_action["lat"] == "TURN_L"
    assert torch.equal(a.goal_action["controls"], b.goal_action["controls"])
    assert torch.equal(a.controls, b.controls)
    # ⭐ and the STAMP agrees, because a one-level set at the shipped magnitude
    # IS the shipped action space — the id names the space, not the call.
    assert a.goal_kappa_vocab == b.goal_kappa_vocab == GOAL_KAPPA_VOCAB_SHIPPED
    # same-breath control: a DIFFERENT magnitude gets a different id, so the
    # stamp is not a constant.
    assert goal_kappa_vocab_id((0.02,)) != GOAL_KAPPA_VOCAB_SHIPPED


# --------------------------------------------------------------------------- #
#  b. THE LEVEL SET SURFACE                                                    #
# --------------------------------------------------------------------------- #
def test_b_levels_are_the_measured_quantiles():
    """The sizing is DERIVED from the road (`vocab_design.json`, row
    'L=3 at corpus quantiles'), not chosen. Pinned so a later edit is a
    deliberate act."""
    assert GOAL_KAPPA_TURN_LEVELS == (0.01155, 0.01942, 0.05805)
    assert len(GOAL_KAPPA_TURN_LEVELS) == 3
    assert GOAL_KAPPA_TURN not in GOAL_KAPPA_TURN_LEVELS


def test_b2_canonical_controls_levels_stacks_and_only_TURN_varies():
    c = canonical_controls_levels("TURN_L", "CRUISE", 8.0, K, DT,
                                  GOAL_KAPPA_TURN_LEVELS)
    assert c.shape == (3, K, 2)
    got = sorted({float(r[0, 1]) for r in c})
    assert got == pytest.approx(sorted(GOAL_KAPPA_TURN_LEVELS), rel=1e-6)
    # ⛔ CONTROL: a non-TURN token must be INVARIANT to the level set, or the
    # level set is a second lever acting where it was never approved to act.
    for tok in ("LANE_KEEP", "NUDGE_L", "LANE_CHANGE_R"):
        z = canonical_controls_levels(tok, "CRUISE", 8.0, K, DT,
                                      GOAL_KAPPA_TURN_LEVELS)
        assert torch.equal(z[0], z[1]) and torch.equal(z[1], z[2]), tok
        assert torch.equal(z[0], canonical_controls(tok, "CRUISE", 8.0, K, DT))


def test_b3_chooser_picks_the_nearest_magnitude():
    lv = GOAL_KAPPA_TURN_LEVELS
    assert choose_kappa_level(lv, 0.011) == lv[0]
    assert choose_kappa_level(lv, 0.020) == lv[1]
    assert choose_kappa_level(lv, 0.090) == lv[2]
    # sign-blind: the token carries the direction, the level carries magnitude
    assert choose_kappa_level(lv, -0.090) == lv[2]
    with pytest.raises(ValueError):
        choose_kappa_level((), 0.02)


def test_b4_vocab_id_distinguishes_the_action_spaces():
    assert goal_kappa_vocab_id(None) == GOAL_KAPPA_VOCAB_SHIPPED == "L1-0.08"
    assert goal_kappa_vocab_id(GOAL_KAPPA_TURN_LEVELS).startswith("L3-")
    # a one-level set AT the shipped magnitude is the shipped space and keeps
    # its id; any other magnitude, and any wider set, must not.
    assert goal_kappa_vocab_id((0.08,)) == goal_kappa_vocab_id(None)
    assert goal_kappa_vocab_id((0.02,)) != goal_kappa_vocab_id(None)
    assert goal_kappa_vocab_id((0.08, 0.02)) != goal_kappa_vocab_id(None)


# --------------------------------------------------------------------------- #
#  c. THE CHOOSER IS MANDATORY — the tier must not be hidden                   #
# --------------------------------------------------------------------------- #
def test_c_level_set_without_a_hint_is_refused():
    m = _model()
    feats, v0, nav = _window(m.cfg)
    with pytest.raises(ValueError, match="goal_kappa_hint"):
        _plan(m, feats, v0, nav, goal_kappa_levels=GOAL_KAPPA_TURN_LEVELS)
    # same-breath control: WITH a hint the same call is accepted, so the
    # refusal is about the missing chooser and not about the level set.
    ok = _plan(m, feats, v0, nav, goal_kappa_levels=GOAL_KAPPA_TURN_LEVELS,
               goal_kappa_hint=0.02)
    assert ok.goal_kappa_levels == GOAL_KAPPA_TURN_LEVELS


def test_c2_level_set_and_scalar_are_mutually_exclusive():
    m = _model()
    feats, v0, nav = _window(m.cfg)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _plan(m, feats, v0, nav, goal_kappa_turn=0.08,
              goal_kappa_levels=GOAL_KAPPA_TURN_LEVELS, goal_kappa_hint=0.02)


def test_c3_the_hint_actually_selects_the_level():
    """⭐ The mechanism, end to end: two hints that fall on different levels
    must produce different goal curvature on a TURN window."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    bias = _turn_bias()
    lo = _plan(m, feats, v0, nav, lat_logit_bias=bias,
               goal_kappa_levels=GOAL_KAPPA_TURN_LEVELS, goal_kappa_hint=0.011)
    hi = _plan(m, feats, v0, nav, lat_logit_bias=bias,
               goal_kappa_levels=GOAL_KAPPA_TURN_LEVELS, goal_kappa_hint=0.090)
    assert lo.goal_action["kappa_turn_used"] == GOAL_KAPPA_TURN_LEVELS[0]
    assert hi.goal_action["kappa_turn_used"] == GOAL_KAPPA_TURN_LEVELS[2]
    assert lo.goal_action["kappa_vocab"] == hi.goal_action["kappa_vocab"]
    assert not torch.equal(lo.goal_action["controls"],
                           hi.goal_action["controls"])
    # ⛔ CONTROL: on a LANE_KEEP window the hint must change NOTHING — the
    # level set may only act where the token asks for a sustained curvature.
    # ⚠️ The token is FORCED, not assumed: this random-init model decodes
    # `TURN_R` on the unbiased window, so a control written against "whatever
    # it happens to decode" would have silently tested a TURN again.
    lk = torch.zeros(len(LAT))
    lk[LAT.index("LANE_KEEP")] = 50.0
    lk_lo = _plan(m, feats, v0, nav, lat_logit_bias=lk,
                  goal_kappa_levels=GOAL_KAPPA_TURN_LEVELS, goal_kappa_hint=0.011)
    lk_hi = _plan(m, feats, v0, nav, lat_logit_bias=lk,
                  goal_kappa_levels=GOAL_KAPPA_TURN_LEVELS, goal_kappa_hint=0.090)
    assert lk_lo.goal_action["lat"] == "LANE_KEEP"
    assert torch.equal(lk_lo.goal_action["controls"],
                       lk_hi.goal_action["controls"])
    assert torch.equal(lk_lo.controls, lk_hi.controls)
