"""BACKLOG R26 + R27 — the planner->model crossing, in UNITS and in TIME.

Two defects were MEASURED on 2026-09-03 by ``taniteval/tools/cost_surface_probe.py``
(``Research/2026-09-03-refav1-cost-surface/``) and are repaired in ``refa_v1.py``:

* **R26 (units).** ``as_command`` was called at exactly ONE site — inside
  ``plan()``'s ``_cost_chunk`` — so ``model_action_units="steer"`` converted the
  CANDIDATE while ``_imagine_tactical_goal`` still rolled the GOAL in raw curvature.
  Candidate and goal were compared across two action conventions, and the converted
  canonical turn landed ONE float32 ULP FURTHER from its own goal (goal term
  5.96e-08 -> 1.19e-07; advantage over ``cv`` 0 -> -5.96e-08).
* **R27 (time).** With ``plan_level="tactical"`` the cost rolled ``self.tactical`` on
  all ``H`` OPERATIVE actions, so the 2.0 s plan was imagined as 6.0 s and candidate
  action ``j`` landed on tactical step ``j`` — while the goal's action ``j`` is
  operative step ``j*stride``, which is also the only grid the predictor was ever
  TRAINED on (``forward``'s ``tac_a``).

⚠️ WHAT THIS IS NOT. ``D-REFAV1-BOUNDARY-NULL`` measured that completing the unit
conversion moves the cost's minimum on **0 of 140** windows on both banked
checkpoints (``share = +0.0000 [0.0000, 0.0000]``, zero-width paired
episode-cluster interval). These tests pin a CORRECTNESS property and a DIRECTION.
No test here claims, or may be quoted as claiming, a driving improvement.

Both repairs are behind call-site flags whose DEFAULTS are the legacy path, and the
first two tests exist to prove that default is byte-identical.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.models.kinematic import STEER_WHEELBASE_M
from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
from tanitad.refs.refa_v1 import (COST_TIME_GRIDS, GOAL_KAPPA_TURN, RefAV1,
                                  RefAV1Config)
from tanitad.refs.refa_v1_plan import PlanConfig

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: float32's step next to cos == 1. The goal term is ``1 - cosine_similarity`` in
#: float32, so every comparison below carries these as tolerance —
#: `H-REFAV1-COST-FLAT` measured the lateral axis sitting AT this noise floor.
ULP = 5.9604645e-08
#: ⛔ ``1 - cos(x, x)`` IS NOT 0.0 IN FLOAT32 — it is a small ULP multiple, and
#: the self-similarity of a real terminal field is the noise floor, not zero.
#: The cost-surface probe's control C3 failed its pre-registered *exactly-0.0*
#: form at **2.00 / 4.00 ULPs** on the two banked checkpoints, and that failure
#: IS the measurement. Reproduced here on a tiny random model at 2.00 ULPs.
#: ⇒ any assertion that a self-comparison is "zero" must carry this budget, or
#: it is asserting float32 does not exist.
FLOOR_ULPS = 4


# --------------------------------------------------------------------------- #
# fixtures                                                                      #
# --------------------------------------------------------------------------- #
def _tiny_cfg(**kw):
    return RefAV1Config(
        tac_vocab_version="v7.0", d_enc=32, n_tokens=8, d_state=32,
        op_layers=1, op_heads=2, op_window=4, tac_layers=1, tac_queries=4,
        str_dim=16, str_layers=1,
        strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                            d_ctx=16, d_cmd=8),
        tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                          d_intent=16), **kw)


def _tiny_model(**kw):
    torch.manual_seed(0)
    cfg = _tiny_cfg(**kw)
    m = RefAV1(cfg).eval()
    feats = torch.randn(1, cfg.op_window, cfg.n_tokens, cfg.d_enc)
    return m, cfg, feats


def _pc(cfg):
    return PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt, n_samples=8,
                      n_iters=1, n_elites=2, seed=0)


class _ConstHead(nn.Module):
    """A decoder head that always argmaxes to ``idx`` — so a test can ask for a
    manoeuvre instead of hoping a random init decodes one."""

    def __init__(self, n: int, idx: int):
        super().__init__()
        self.n, self.idx = n, idx

    def forward(self, x):
        out = torch.full((x.shape[0], self.n), -10.0, device=x.device,
                         dtype=x.dtype)
        out[:, self.idx] = 10.0
        return out


def _force_turn(m, cfg, lat="TURN_R", lon="CRUISE"):
    """Pin the tactical decode so the imagined goal CARRIES CURVATURE.

    Without this the goal is whatever a random init decodes, and on the real
    checkpoints that is ``LANE_KEEP`` on 140/140 windows
    (`D-REFAV1-GOAL-DEGENERATE`) — i.e. zero curvature, where a units repair is
    trivially a no-op and nothing is being tested."""
    lat_v = tactical_lat_actions(cfg.tac_vocab_version)
    lon_v = tactical_lon_actions_v(cfg.tac_vocab_version)
    m.lat_head = _ConstHead(len(lat_v), lat_v.index(lat))
    m.lon_head = _ConstHead(len(lon_v), lon_v.index(lon))
    return m


def _brains_at(m, feats, v0, nav=None):
    field = m.encode(feats)
    last = m._last_state(field)
    brains = m._run_brains(field.mean(dim=-2), nav)
    v0_t = torch.as_tensor([v0], dtype=torch.float32, device=feats.device)
    return last, brains, v0_t


def _spy_on(obj, name):
    """Record every tensor passed as the first positional arg of ``obj.name``.

    ``nn.Module.__setattr__`` refuses a non-Module for a submodule name but a
    bound method is a plain instance attribute, so this shadows cleanly."""
    calls = []
    orig = getattr(obj, name)

    def wrapped(first, *a, **kw):
        calls.append(first.detach().clone())
        return orig(first, *a, **kw)

    setattr(obj, name, wrapped)
    return calls, (lambda: setattr(obj, name, orig))


def _spy_rollout_actions(pred):
    """Record the ACTION tensor of every ``pred.rollout`` call (arg 2)."""
    calls = []
    orig = pred.rollout

    def wrapped(field, actions, *a, **kw):
        calls.append(actions.detach().clone())
        return orig(field, actions, *a, **kw)

    pred.rollout = wrapped
    return calls, (lambda: setattr(pred, "rollout", orig))


# =========================================================================== #
# A — UNITS (R26)
# =========================================================================== #
def test_A1_default_OFF_is_byte_identical_on_a_real_rollout():
    """The three spellings of "do nothing" must agree bit for bit."""
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg))
        b = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), model_action_units="kappa")
        c = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), model_action_units="kappa",
                   cost_time_grid="dense")
    for other in (b, c):
        assert torch.equal(a.controls, other.controls)
        assert a.source == other.source
        assert a.cost == other.cost                      # float equality, not approx
        assert a.baseline_costs == other.baseline_costs
        assert a.fine_costs == other.fine_costs
    assert a.model_action_units == "kappa" and a.cost_time_grid == "dense"


def test_A2_the_default_goal_equals_the_LEGACY_expression_verbatim():
    """OFF-identity proved against the pre-2026-09-03 source line, not against
    another spelling of the new code: ``augment_actions(ctrl, v0)[:, ::stride]
    [:, :tac_steps]`` (refa_v1.py:1683 before the repair)."""
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg)
    last, brains, v0_t = _brains_at(m, feats, 8.0)
    with torch.no_grad():
        goal_new, ga = m._imagine_tactical_goal(last, brains, v0_t)
        stride = m._stride(cfg.tac_dt)
        legacy_acts = m.augment_actions(ga["controls"], v0_t)[:, ::stride]
        legacy_acts = legacy_acts[:, :cfg.tac_steps]
        goal_legacy = m.tactical.rollout(m._tac_field(last), legacy_acts,
                                         intent=brains["intent"], last_only=True)
    assert torch.equal(goal_new, goal_legacy)
    # and the provenance the seed pool reads stays in GEOMETRY
    assert float(ga["controls"][..., 1].abs().max()) == pytest.approx(
        GOAL_KAPPA_TURN, abs=1e-7)


def test_A3_ON_converts_BOTH_sides_in_one_plan_call():
    """The defect, stated as an observation: with ``"steer"`` EVERY tensor that
    crosses into the model — the goal's and the candidates' — must be
    ``arctan(L*kappa)`` of what the ``"kappa"`` run handed over."""
    out = {}
    for units in ("kappa", "steer"):
        m, cfg, feats = _tiny_model()
        _force_turn(m, cfg)
        calls, restore = _spy_on(m, "augment_actions")
        with torch.no_grad():
            m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), model_action_units=units)
        restore()
        out[units] = calls

    ck, cs = out["kappa"], out["steer"]
    assert len(ck) == len(cs) >= 2, "goal + at least one candidate chunk"

    # the GOAL is the first crossing (built before the search) and is the one
    # site the shipped code did NOT convert.
    g_k, g_s = ck[0], cs[0]
    assert g_k.shape == (1, cfg.op_steps, cfg.a_dim)
    assert float(g_k[..., 1].abs().max()) > 0.0, "the goal must carry curvature"
    assert torch.equal(g_k[..., 0], g_s[..., 0]), "accel must be untouched"
    assert torch.equal(g_s[..., 1], torch.atan(STEER_WHEELBASE_M * g_k[..., 1]))
    assert not torch.equal(g_k[..., 1], g_s[..., 1])

    # ...and every remaining crossing is a CANDIDATE chunk, already converted
    # before the repair. Channel 1 must be the arctan of a curvature inside the
    # search box, i.e. strictly smaller in magnitude and same sign.
    for a_k, a_s in zip(ck[1:], cs[1:]):
        assert a_k.shape[1] == cfg.plan_steps
        assert torch.equal(a_k[..., 0], a_s[..., 0])
        assert torch.equal(a_s[..., 1],
                           torch.atan(STEER_WHEELBASE_M * a_k[..., 1]))


def test_A4_a_candidate_in_the_SAME_units_reproduces_its_own_goal_exactly():
    """The identity the repair restores: the control profile that BUILT the goal,
    put back through the boundary under the SAME spelling, is the goal.

    ⛔ Under the shipped half-conversion the same profile under ``"steer"`` was
    scored against a goal built in ``"kappa"`` and did NOT reproduce it — which
    is the whole defect."""
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg)
    last, brains, v0_t = _brains_at(m, feats, 8.0)
    stride, z0 = m._stride(cfg.tac_dt), m._tac_field(last)

    def roll(ctrl, units):
        acts = m._model_actions(ctrl, v0_t, units)[:, ::stride][:, :cfg.tac_steps]
        return m.tactical.rollout(z0, acts, intent=brains["intent"],
                                  last_only=True)

    with torch.no_grad():
        for units in ("kappa", "steer"):
            goal, ga = m._imagine_tactical_goal(last, brains, v0_t, units=units)
            assert torch.equal(roll(ga["controls"], units), goal)
        # the cross-units pairing the shipped code performed
        goal_k, ga = m._imagine_tactical_goal(last, brains, v0_t, units="kappa")
        assert not torch.equal(roll(ga["controls"], "steer"), goal_k), (
            "the half-conversion must be observable — if this passes the "
            "predictor is laterally blind and the test proves nothing")


def test_A5_the_turns_advantage_over_the_straight_line_does_not_get_WORSE():
    """DIRECTION ONLY — no magnitude is claimed, and the measured magnitude on
    the real checkpoints is exactly zero (`D-REFAV1-BOUNDARY-NULL`).

    Under consistent units the turn that built the goal sits at the goal term's
    floor (``1 - cos(x, x)``), so it can never score worse than the straight
    line. The half-conversion breaks that: MEASURED on 25 real ``TURN_R``
    windows, the turn's advantage over ``cv`` went 0 -> -5.96e-08."""
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg)
    last, brains, v0_t = _brains_at(m, feats, 8.0)
    stride, z0 = m._stride(cfg.tac_dt), m._tac_field(last)

    def roll(ctrl, units):
        acts = m._model_actions(ctrl, v0_t, units)[:, ::stride][:, :cfg.tac_steps]
        return m.tactical.rollout(z0, acts, intent=brains["intent"],
                                  last_only=True)

    def c_goal(zk, goal):
        return float(1.0 - F.cosine_similarity(zk.flatten(1), goal.flatten(1),
                                               dim=-1))

    with torch.no_grad():
        goal_k, ga = m._imagine_tactical_goal(last, brains, v0_t, units="kappa")
        goal_s, _ = m._imagine_tactical_goal(last, brains, v0_t, units="steer")
        turn = ga["controls"]
        straight = torch.zeros_like(turn)

        # OFF (legacy, one convention on both sides) and FIXED-ON (one
        # convention on both sides) — the turn is the minimiser in both.
        for goal, units in ((goal_k, "kappa"), (goal_s, "steer")):
            c_turn = c_goal(roll(turn, units), goal)
            c_str = c_goal(roll(straight, units), goal)
            assert c_turn <= c_str + ULP, (units, c_turn, c_str)
            # it IS the floor — up to float32's self-similarity budget, which is
            # a measured 2-4 ULPs and not 0.0 (see FLOOR_ULPS).
            assert c_turn <= FLOOR_ULPS * ULP, (units, c_turn)

        # the TRAP as shipped: candidate converted, goal not. The turn's own
        # cost can only go UP, because cos(x, x) is the maximum of cos.
        c_turn_trap = c_goal(roll(turn, "steer"), goal_k)
        c_turn_fix = c_goal(roll(turn, "steer"), goal_s)
        assert c_turn_fix <= c_turn_trap + ULP, (c_turn_fix, c_turn_trap)


def test_A6_the_units_flag_is_still_refused_loudly_and_reaches_imagined_goal():
    m, cfg, feats = _tiny_model()
    with pytest.raises(ValueError):
        m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), model_action_units="kappa_ish")
    with pytest.raises(ValueError):
        m.imagined_goal(feats, v0=8.0, model_action_units="curvature")
    with torch.no_grad():
        g1, _ = m.imagined_goal(feats, v0=8.0)
        g2, _ = m.imagined_goal(feats, v0=8.0, model_action_units="kappa")
    assert torch.equal(g1, g2)


# =========================================================================== #
# B — TIME (R27)
# =========================================================================== #
def test_B1_the_time_flag_default_is_byte_identical_and_refused_loudly():
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg)
    with torch.no_grad():
        a = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg))
        b = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), cost_time_grid="dense")
    assert torch.equal(a.controls, b.controls)
    assert a.cost == b.cost and a.baseline_costs == b.baseline_costs
    assert COST_TIME_GRIDS == ("dense", "tactical")
    with pytest.raises(ValueError):
        m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), cost_time_grid="operative")


def test_B2_the_corrected_index_map_is_the_GOALs_map_then_a_hold():
    """Pure arithmetic on the shipped config — no model, no checkpoint."""
    cfg = RefAV1Config(tac_vocab_version="v7.0")
    stride = max(1, int(round(cfg.tac_dt / cfg.op_dt)))
    h = cfg.plan_steps
    assert (stride, h, cfg.tac_steps, cfg.op_steps) == (3, 10, 10, 30)

    cand_fixed = [min(j * stride, h - 1) for j in range(cfg.tac_steps)]
    goal_map = list(range(0, cfg.op_steps, stride))[:cfg.tac_steps]
    cand_dense = list(range(h))                       # today's behaviour

    assert cand_fixed == [0, 3, 6, 9, 9, 9, 9, 9, 9, 9]
    assert goal_map == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    assert cand_dense == list(range(10))
    # the repair puts the candidate on the goal's grid while the plan lasts...
    assert cand_fixed[:4] == goal_map[:4]
    # ...and today's dense feed does not, from the second step on
    assert cand_dense[1] != goal_map[1]
    # ⛔ the RESIDUAL the repair does NOT remove: the plan is 2.0 s of a 6.0 s
    # goal, so the two maps must diverge after the 4th tactical step. This is a
    # `plan_horizon_s` design question, escalated, not patched.
    assert cand_fixed[4:] != goal_map[4:]


def test_B3_ON_regrids_the_TACTICAL_rollout_and_holds_the_last_action():
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg)
    seen = {}
    for grid in ("dense", "tactical"):
        calls, restore = _spy_rollout_actions(m.tactical)
        with torch.no_grad():
            m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), cost_time_grid=grid)
        restore()
        # call 0 is the GOAL rollout (identical under both grids by construction)
        seen[grid] = calls

    assert torch.equal(seen["dense"][0], seen["tactical"][0]), "goal must not move"
    cand_d = seen["dense"][1:]
    cand_t = seen["tactical"][1:]
    assert cand_d and cand_t

    # under the repair every candidate's tail is the HELD last plan action
    for a in cand_t:
        assert a.shape[1] == cfg.tac_steps
        for j in range(4, cfg.tac_steps):
            assert torch.equal(a[:, j], a[:, 3]), j
    # ...and under today's dense feed it is not (the search is coloured noise)
    assert any(not torch.equal(a[:, j], a[:, 3])
               for a in cand_d for j in range(4, cfg.tac_steps))


def test_B4_the_time_grid_is_a_NO_OP_where_it_must_be():
    """Two controls that must read a known value.

    (a) A CONSTANT candidate: both index maps select the same values, so the
        cost cannot move. (Only true with ``speed_channel=False`` — see (c).)
    (b) The coarse->fine re-score rolls the OPERATIVE predictor, whose step IS
        ``op_dt``; it has no defect and must be untouched.
    (c) ⚠️ With the speed channel ON the regrid DOES move the input, and that is
        CORRECT: a tactical step opening at 1.8 s must see the speed at 1.8 s,
        not the speed at 0.6 s. Pinned so it is never "fixed"."""
    m, cfg, feats = _tiny_model()               # speed_channel=False by default
    assert cfg.speed_channel is False and cfg.a_in_dim == cfg.a_dim
    _force_turn(m, cfg)
    stride = m._stride(cfg.tac_dt)
    idx = torch.tensor([min(j * stride, cfg.plan_steps - 1)
                        for j in range(cfg.tac_steps)])
    const = torch.zeros(1, cfg.plan_steps, cfg.a_dim)
    const[..., 0], const[..., 1] = 1.25, 0.03
    with torch.no_grad():
        a_const = m._model_actions(const, None, "kappa")
    assert torch.equal(a_const.index_select(1, idx), a_const), "(a)"

    # (b) the fine re-score is on the operative predictor
    calls, restore = _spy_rollout_actions(m.operative)
    with torch.no_grad():
        m.plan(feats, v0=8.0, plan_cfg=_pc(cfg), cost_time_grid="tactical")
    restore()
    assert calls, "verify_on_operative must have re-scored"
    for a in calls:
        assert a.shape[1] == cfg.plan_steps, "the operative roll must stay dense"

    # (c) the speed channel is time-dependent, so the regrid moves it
    m3, cfg3, _ = _tiny_model(speed_channel=True)
    assert cfg3.a_in_dim == cfg3.a_dim + 1
    v0_t = torch.tensor([8.0])
    with torch.no_grad():
        a3 = m3._model_actions(const, v0_t, "kappa")
    assert not torch.equal(a3.index_select(1, idx), a3), "(c)"
    assert torch.equal(a3[..., :cfg3.a_dim].index_select(1, idx),
                       a3[..., :cfg3.a_dim]), "(c): only the speed channel moves"


def test_B5_the_two_flags_are_ORTHOGONAL_and_compose():
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg)
    with torch.no_grad():
        r = m.plan(feats, v0=8.0, plan_cfg=_pc(cfg),
                   model_action_units="steer", cost_time_grid="tactical")
    assert r.model_action_units == "steer" and r.cost_time_grid == "tactical"
    assert float(r.controls[..., 1].abs().max()) <= _pc(cfg).kappa_max + 1e-6, (
        "the DEPLOYED plan must still be curvature — the conversion is a model "
        "boundary, not a change of search space")


# =========================================================================== #
# C — T4 (`target_speed`): dead, and NOT deleted
# =========================================================================== #
def _plan_call_keywords(path: pathlib.Path) -> list[set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "plan"):
            out.append({k.arg for k in node.keywords if k.arg})
    return out


@pytest.mark.parametrize("rel", ["taniteval/tools/refav1_arm.py",
                                 "taniteval/tools/cost_surface_probe.py"])
def test_C1_no_production_plan_call_site_passes_target_speed(rel):
    """T4 (``0.10*(v_end - target_speed)^2``) is DEAD in every banked read: the
    shipped cost has THREE live terms, not four.

    ⛔ This is an ast walk, not a grep, because ``cost_surface_probe.py``
    contains the string ``target_speed`` in its own re-implementation while its
    ``model.plan(...)`` call does not pass it — a grep would read as a false
    positive. And it asserts the call was FOUND, so a rename cannot make the
    claim vacuously true (the `36-features` rot, in miniature)."""
    p = ROOT / rel
    if not p.exists():
        pytest.skip(f"{rel} not present")
    calls = _plan_call_keywords(p)
    assert calls, f"no `.plan(...)` call found in {rel} — the pin went stale"
    for kws in calls:
        assert "target_speed" not in kws, (rel, sorted(kws))


def test_C2_T4_is_exactly_zero_when_absent_and_hand_computable_when_supplied():
    """The behavioural half: on the ``cv`` baseline (zero controls) ``v_end`` is
    ``v0``, so ``target_speed=v0`` must change nothing at all, and a 1 m/s miss
    must add exactly ``0.10 * 1.0**2``.

    ⇒ REMOVAL WAS REJECTED, deliberately: T4 is *unused by production*, which is
    weaker than *unreachable* — ``test_refa_v1.py:413`` and
    ``test_refa_v1_speed_channel.py:213`` both exercise it. Wiring it to the
    decoded ``lon`` token would give the tactical brain's LONGITUDINAL decision a
    second channel into the cost and change every banked number; that is a PI
    decision, and it is ESCALATED, not taken here."""
    m, cfg, feats = _tiny_model()
    _force_turn(m, cfg)
    v0 = 8.0
    with torch.no_grad():
        none = m.plan(feats, v0=v0, plan_cfg=_pc(cfg))
        same = m.plan(feats, v0=v0, plan_cfg=_pc(cfg), target_speed=v0)
        off = m.plan(feats, v0=v0, plan_cfg=_pc(cfg), target_speed=v0 - 1.0)
    for k in ("cv", "hold_v0"):
        assert none.baseline_costs[k] == same.baseline_costs[k], k
        assert (off.baseline_costs[k] - none.baseline_costs[k]
                == pytest.approx(0.10 * 1.0 ** 2, abs=1e-6)), k
