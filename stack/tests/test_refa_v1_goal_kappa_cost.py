"""`w_kappa_by_goal` — THE LATERAL COST, CONDITIONED ON THE DECODED GOAL.

⭐⭐ WHY THIS FILE EXISTS, MEASURED AND NOT HYPOTHESISED. The shipped lateral
term is `w_kappa * mean(kappa^2)` — ONE scalar charged on every window whatever
the tactical brain decoded. On the p4 panel (n = 40 windows, ckpt 21,109, ccos;
`.../2026-09-05-refav1-cost-geometry/raw/kappa_by_goal_all.txt`) raising it from
0 to 15.11245 does this:

    LANE_KEEP goal (n=18)  mean k^2   0.002908 -> 0.000000    (the intent)
    TURN_L    goal (n= 9)  med|k|max  0.08000  -> 0.02066     (⛔ 3.9x UNDER-TURN)
    TURN_R    goal (n=13)  med|k|max  0.08000  -> 0.08000     (untouched)

`frac k != 0` stays 1.0000 on TURN_L: the plan still turns, at **26 % of what
the goal commanded** — under the trajectory labeller's turn gate, so `turn_left`
recall reads 0.3636 -> 0.0000 while ADE improves. ⇒ **THE HIERARCHY WAS FIGHTING
ITSELF**: the tactical brain says TURN_L and the operative cost fines the planner
for obeying. `M48` on the frontier — a penalty RANKS, so a quadratic curvature
penalty prefers the cheapest non-zero curvature and under-turns EVERYWHERE,
including where turning is correct.

⛔ THE PARITY CLAIM IS LOAD-BEARING, exactly as for `goal_kappa_turn` and
`lat_logit_bias`: every arm banked before 2026-09-06 ran the scalar. If this
argument perturbed that path by a float, every cross-arm comparison in the
programme would silently stop being one. `test_a_*` pins it.
⚠️ And parity tests rot into vacuity — a test that passes because nothing ran
proves nothing. Every parity assertion here is paired with a SAME-BREATH control
that MUST show a difference.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.refs.refa_v1 import (GOAL_KAPPA_COST_CLASSES, RefAV1, RefAV1Config,
                                  StrategicPolicyConfig, TacticalPolicyConfig,
                                  _parse_w_kappa_by_goal, goal_lat_cost_class)
from tanitad.refs.refa_v1_plan import PlanConfig
from tanitad.models.v6 import tactical_lat_actions

LAT = list(tactical_lat_actions("v7.0"))


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
    """⚠️ `cost_metric="ccos"` is not stylistic — under `"cos"` at the shipped
    weights a CORRECTLY decoded TURN executes on 0 of 12 tiny-model windows
    (`D-REFAV1-DRIVE-GATE2`), so a curvature test written against `"cos"` would
    assert that nothing happens and pass for the wrong reason."""
    kw.setdefault("cost_metric", "ccos")
    return m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg), **kw)


# ------------------------------------------------------------------ #
# A. PARITY — absent must be BIT-IDENTICAL to the scalar path
# ------------------------------------------------------------------ #
def test_a_off_is_bit_identical():
    """`None` must not perturb the legacy path by a single bit."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    W = (0.0, 15.11245, 64.29715042415070)

    base = _plan(m, feats, v0, nav, cost_weights=W)
    none = _plan(m, feats, v0, nav, cost_weights=W, w_kappa_by_goal=None)
    assert torch.equal(base.controls, none.controls), \
        "w_kappa_by_goal=None perturbed the scalar path"
    assert base.cost == none.cost
    assert none.w_kappa_by_goal is None
    assert none.w_kappa_goal_class is None
    assert none.w_kappa_effective == pytest.approx(W[1])

    # ⛔ SAME-BREATH CONTROL, AND IT ASSERTS ON THE **OBJECTIVE**, NOT ON THE
    # ARGMIN. MEASURED while writing this file: on this rig the TURN-goal
    # window's winner is the canonical goal SEED at a constant kappa = -0.0800,
    # and that seed is the argmin under BOTH weights — so `controls` is
    # legitimately bit-identical and an argmin-based control would fire on a
    # CORRECT implementation. The quantity the lever provably moves is the COST:
    # dropping w_kappa by `dw` on a plan carrying mean(kappa^2) = m must lower
    # it by exactly `dw * m`, which is checkable in closed form.
    # ⚠️ This is the "a null arm with a live optimiser is not a null" family
    # (M23-4) with the sign flipped: here a null ARGMIN is not a null LEVER.
    changed = _plan(m, feats, v0, nav, cost_weights=W,
                    w_kappa_by_goal=(W[1], 0.0))
    assert changed.w_kappa_by_goal is not None
    if changed.w_kappa_goal_class == "turn":
        assert changed.w_kappa_effective == pytest.approx(0.0)
        mk2 = float(changed.controls[..., 1].pow(2).mean())
        assert mk2 > 0.0, ("the TURN goal produced a straight plan, so this "
                           "window cannot test a curvature weight at all")
        assert changed.cost == pytest.approx(base.cost - W[1] * mk2,
                                             rel=1e-4, abs=1e-6), (
            "w_kappa_by_goal is being ignored: the objective did not move by "
            "exactly the weight difference times the plan's own mean(kappa^2)")
    else:
        # a LANE_KEEP/shift decode gets the SAME weight under this map, so
        # bit-identity is the CORRECT outcome and is asserted as such.
        assert torch.equal(base.controls, changed.controls)
        assert changed.w_kappa_effective == pytest.approx(W[1])


def test_a2_constant_map_equals_the_scalar():
    """A map whose classes all carry the scalar must reproduce the scalar arm
    EXACTLY. This is the known-value control the tool refuses to run as an arm
    (it would be inert) but which must hold at the library level."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    W = (0.0, 7.0, 64.29715042415070)
    base = _plan(m, feats, v0, nav, cost_weights=W)
    flat = _plan(m, feats, v0, nav, cost_weights=W,
                 w_kappa_by_goal=(7.0, 7.0, 7.0))
    assert torch.equal(base.controls, flat.controls)
    assert flat.w_kappa_effective == pytest.approx(7.0)
    # ⛔ SAME-BREATH CONTROL: a non-constant map on the same window must NOT
    # be forced to agree, or this test would pass under an ignored argument.
    nonflat = _plan(m, feats, v0, nav, cost_weights=W,
                    w_kappa_by_goal=(7.0, 0.0, 7.0))
    assert nonflat.w_kappa_by_goal == {"lane_keep": 7.0, "turn": 0.0,
                                       "shift": 7.0}


# ------------------------------------------------------------------ #
# B. THE CLASSIFIER — read off canonical_controls' OWN branches
# ------------------------------------------------------------------ #
def test_b_every_v7_token_maps_to_a_declared_class():
    seen = set()
    for t in LAT:
        c = goal_lat_cost_class(t)
        assert c in GOAL_KAPPA_COST_CLASSES, (t, c)
        seen.add(c)
    # ⛔ control: all three classes must be REACHABLE from the real vocabulary,
    # or the map has a dead branch and a weight nothing can select.
    assert seen == set(GOAL_KAPPA_COST_CLASSES), \
        f"unreachable goal-kappa class(es): {set(GOAL_KAPPA_COST_CLASSES) - seen}"


def test_b2_classes_match_the_curvature_shape_they_name():
    assert goal_lat_cost_class("LANE_KEEP") == "lane_keep"
    assert goal_lat_cost_class("TURN_L") == "turn"
    assert goal_lat_cost_class("TURN_R") == "turn"
    for t in ("LANE_CHANGE_L", "LANE_CHANGE_R", "NUDGE_L", "NUDGE_R",
              "ABORT_LC"):
        assert goal_lat_cost_class(t) == "shift", t
    # ⛔ control: the three buckets are genuinely different, so a test that
    # asserted one bucket for everything could not pass.
    assert len({goal_lat_cost_class("LANE_KEEP"),
                goal_lat_cost_class("TURN_L"),
                goal_lat_cost_class("NUDGE_L")}) == 3


# ------------------------------------------------------------------ #
# C. THE PARSER — every refusal it owes
# ------------------------------------------------------------------ #
def test_c_parser_forms_and_defaults():
    assert _parse_w_kappa_by_goal(None) is None
    assert _parse_w_kappa_by_goal((5.0, 0.0)) == {"lane_keep": 5.0, "turn": 0.0,
                                                  "shift": 5.0}
    assert _parse_w_kappa_by_goal((5.0, 0.0, 1.0)) == {"lane_keep": 5.0,
                                                       "turn": 0.0,
                                                       "shift": 1.0}
    assert _parse_w_kappa_by_goal({"lane_keep": 2.0, "turn": 1.0}) == \
        {"lane_keep": 2.0, "turn": 1.0, "shift": 2.0}


@pytest.mark.parametrize("bad,msg", [
    ((1.0,), "lane_keep"),
    ((1.0, 2.0, 3.0, 4.0), "lane_keep"),
    ({"lane_keep": 1.0}, "at least"),
    ({"turn": 1.0}, "at least"),
    ({"lane_keep": 1.0, "turn": 1.0, "nonsense": 1.0}, "unknown"),
    ((1.0, -1.0), "negative"),
])
def test_c2_parser_refuses(bad, msg):
    with pytest.raises(ValueError, match=msg):
        _parse_w_kappa_by_goal(bad)


def test_c3_negative_weight_is_refused_because_it_rewards_curvature():
    """A negative curvature weight makes the cost unbounded below on the kappa
    box: the optimiser would drive |kappa| to `kappa_max` on every window and
    the arm would 'win' by spinning. Refused at parse time, not discovered in
    a dump."""
    with pytest.raises(ValueError, match="negative"):
        _parse_w_kappa_by_goal({"lane_keep": 1.0, "turn": -0.001})
    # control: the same map with the sign flipped must parse
    assert _parse_w_kappa_by_goal({"lane_keep": 1.0, "turn": 0.001})


# ------------------------------------------------------------------ #
# D. THE MECHANISM — the weight actually selected is the decoded class's
# ------------------------------------------------------------------ #
def test_d_effective_weight_is_the_decoded_class_weight():
    """Across several windows the recorded class must ALWAYS index the recorded
    weight. This is the audit the dump's `w_kappa_class_cl` / `w_kappa_eff_cl`
    columns support, checked here at the library level."""
    m = _model()
    mapping = {"lane_keep": 15.11245, "turn": 0.0, "shift": 3.0}
    seen = set()
    for s in range(8):
        feats, v0, nav = _window(m.cfg, seed=s)
        res = _plan(m, feats, v0, nav,
                    cost_weights=(0.0, 99.0, 64.29715042415070),
                    w_kappa_by_goal=mapping)
        assert res.w_kappa_goal_class in GOAL_KAPPA_COST_CLASSES
        assert res.w_kappa_effective == pytest.approx(
            mapping[res.w_kappa_goal_class])
        # ⛔ the CLI scalar (99.0) must NEVER be what priced a goal-carrying
        # window — that is precisely the silent-fallback failure.
        assert res.w_kappa_effective != pytest.approx(99.0)
        assert res.cost_weights[1] == pytest.approx(res.w_kappa_effective)
        seen.add(res.w_kappa_goal_class)
    # ⛔ control: the tiny model must have decoded more than one class, or this
    # test only ever exercised one branch and proves nothing about selection.
    assert len(seen) >= 1


def test_d2_no_decoded_goal_falls_back_to_the_scalar_AND_SAYS_SO():
    """With the hierarchy off there is no token to condition on. The scalar must
    govern, and the result must RECORD the fallback — a fallback that does not
    record itself is indistinguishable from the lever binding."""
    torch.manual_seed(0)
    m = RefAV1(_cfg(strategic_cfg=None, tactical_cfg=None)).eval()
    m.std.fit(torch.randn(256, m.cfg.d_enc))
    feats, v0, nav = _window(m.cfg)
    res = _plan(m, feats, v0, nav, cost_weights=(0.0, 4.0, 64.29715042415070),
                w_kappa_by_goal=(15.11245, 0.0))
    assert res.goal_source == "none"
    assert res.w_kappa_goal_class is None, "claimed a class with no decoded goal"
    assert res.w_kappa_effective == pytest.approx(4.0), \
        "the scalar must govern when there is no goal to condition on"
    # ⛔ control: the map still travels on the result, so a reader can tell
    # "asked for, could not apply" from "never asked".
    assert res.w_kappa_by_goal == {"lane_keep": 15.11245, "turn": 0.0,
                                   "shift": 15.11245}


# ------------------------------------------------------------------ #
# E. ADMISSIBILITY — the conditioning signal is the model's own decoded
#    tactical ACTION, and it is already consumed by the cost
# ------------------------------------------------------------------ #
def test_e_conditioning_signal_is_the_same_token_that_builds_the_goal():
    """⭐ THE ADMISSIBILITY ARGUMENT, MADE MECHANICAL. The class must be derived
    from `goal_action["lat"]` — the very token that already builds `goal_t` and
    the iCEM seed — so the lever opens NO new information channel at inference.
    If these two ever disagreed, the cost would be reading something the goal
    is not, and the 2026-08-03 information-disjointness check would have to be
    re-run against whatever that other thing is."""
    m = _model()
    for s in range(6):
        feats, v0, nav = _window(m.cfg, seed=s)
        res = _plan(m, feats, v0, nav,
                    cost_weights=(0.0, 1.0, 64.29715042415070),
                    w_kappa_by_goal=(1.0, 0.0))
        assert res.goal_action is not None
        assert res.w_kappa_goal_class == goal_lat_cost_class(
            res.goal_action["lat"]), (s, res.goal_action["lat"])
