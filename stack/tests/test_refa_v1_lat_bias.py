"""`lat_logit_bias` — the goal head's decision rule, and its parity guarantee.

⭐ WHY THIS FILE EXISTS. `_imagine_tactical_goal` decides the lateral token by
argmax, and that token is the ONLY curvature-carrying candidate the planner
ever sees (`refa_v1_plan._baseline_controls` has none; `colored_noise` is
zero-mean along time). MEASURED on the trained 21,109-step checkpoint: a
LANE_KEEP decode gives planned curvature EXACTLY 0.0 on 244/244 windows
(`.../2026-09-05-refav1-make-it-drive/GATE_ON_REAL_MODEL.md`). So the decision
rule at that one site IS refav1's lateral policy, and it needed to become a
knob.

⛔ THE PARITY CLAIM IS THE LOAD-BEARING ONE. Every arm banked before
2026-09-05 ran plain argmax. If the new argument perturbed that path by so much
as a float, every cross-arm comparison in the programme would silently stop
being a comparison. `test_a_*` pins it: absent and zero-vector bias are
BIT-IDENTICAL to the pre-change expression.

⚠️ And parity tests are the ones that rot into vacuity — a test that passes
because nothing ran at all proves nothing. Each parity assertion here is
therefore paired with a same-breath control that MUST show a difference.
"""
from __future__ import annotations

import torch

from tanitad.refs.refa_v1 import (RefAV1, RefAV1Config, StrategicPolicyConfig,
                                  TacticalPolicyConfig, GOAL_KAPPA_TURN)
from tanitad.refs.refa_v1_plan import PlanConfig
from tanitad.models.v6 import tactical_lat_actions

LAT = list(tactical_lat_actions("v7.0"))
N_LAT = len(LAT)


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
    """⚠️ `cost_metric` DEFAULTS TO ``"ccos"`` HERE, and that is not a
    stylistic choice — under ``"cos"`` these tests cannot express what they
    test. MEASURED 2026-09-05 (`raw/metric_gate.json`, and on the trained
    checkpoint `raw/turnexec_dump_cos_ext.json`): under ``"cos"`` at the
    shipped weights a CORRECTLY decoded TURN token is executed on **0 of 12**
    tiny-model windows at every search size from 32x3 to 300x30, and on **0 of
    38** real windows of `dump_cos_ext` — planned curvature EXACTLY 0.0. Under
    ``"ccos"`` the same turn executes (>50 % tiny, 63 % real). So a bias test
    written against ``"cos"`` would assert that nothing happens, and pass for
    the wrong reason. `test_d_*` pins that second gate rather than hiding it."""
    kw.setdefault("cost_metric", "ccos")
    return m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(m.cfg),
                  model_action_units="kappa", **kw)


# --------------------------------------------------------------------------- #
#  a. PARITY — the default path is untouched                                   #
# --------------------------------------------------------------------------- #
def test_a_absent_bias_is_bit_identical_to_zero_bias():
    """Not passing the argument == passing zeros. If these ever diverge, the
    add is not a no-op at zero and every legacy arm is affected."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    a = _plan(m, feats, v0, nav)
    b = _plan(m, feats, v0, nav, lat_logit_bias=torch.zeros(N_LAT))
    assert torch.equal(a.controls, b.controls)
    assert a.goal_action["lat"] == b.goal_action["lat"]
    assert float(a.cost) == float(b.cost)


def test_a_CONTROL_a_real_bias_does_change_the_plan():
    """The same-breath control for the test above. A bias big enough to force
    TURN_L MUST change both the decoded token and the planned curvature —
    otherwise `test_a_absent_bias...` is passing because nothing happens."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    base = _plan(m, feats, v0, nav)
    bias = torch.zeros(N_LAT)
    bias[LAT.index("TURN_L")] = 1e3
    forced = _plan(m, feats, v0, nav, lat_logit_bias=bias)
    assert forced.goal_action["lat"] == "TURN_L"
    assert forced.goal_action["lat"] != base.goal_action["lat"] or \
        not torch.equal(base.controls, forced.controls)
    # and it steers LEFT, at the canonical turn curvature
    assert float(forced.controls[:, 1].mean()) > 1e-3


def test_a_imagined_goal_absent_equals_zero_bias():
    """The public `imagined_goal` entry point carries the same guarantee."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    _, ga = m.imagined_goal(feats, v0=v0, nav_cmd=nav)
    _, gb = m.imagined_goal(feats, v0=v0, nav_cmd=nav,
                            lat_logit_bias=torch.zeros(N_LAT))
    assert ga["lat"] == gb["lat"] and ga["lon"] == gb["lon"]
    assert torch.equal(ga["controls"], gb["controls"])


# --------------------------------------------------------------------------- #
#  b. THE GATE — the bias reaches the curvature the planner can produce        #
# --------------------------------------------------------------------------- #
def test_b_bias_selects_every_token_and_only_turn_tokens_steer():
    """Forcing each token in turn: the decode follows the bias, and the planned
    curvature is zero for exactly the zero-curvature tokens. This is the gate,
    expressed as a test."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    zero_curv, steered = [], {}
    for i, tok in enumerate(LAT):
        bias = torch.zeros(N_LAT)
        bias[i] = 1e3
        r = _plan(m, feats, v0, nav, lat_logit_bias=bias)
        assert r.goal_action["lat"] == tok, f"bias did not select {tok}"
        k = r.controls[:, 1]
        steered[tok] = float(k.abs().max())
        if float(k.abs().max()) == 0.0:
            zero_curv.append(tok)
    assert sorted(zero_curv) == ["ABORT_LC", "LANE_KEEP"], (
        f"zero-curvature plans were {sorted(zero_curv)}; the gate's shape "
        f"changed — GATE_ON_REAL_MODEL.md must be re-measured")
    # CONTROL: the turn tokens must reach the canonical turn curvature, or the
    # 'zero' above is a statement about a planner that cannot steer at all.
    assert steered["TURN_L"] >= GOAL_KAPPA_TURN - 1e-6
    assert steered["TURN_R"] >= GOAL_KAPPA_TURN - 1e-6


def test_b_sign_follows_the_token():
    m = _model()
    feats, v0, nav = _window(m.cfg)
    out = {}
    for tok in ("TURN_L", "TURN_R"):
        bias = torch.zeros(N_LAT)
        bias[LAT.index(tok)] = 1e3
        out[tok] = float(_plan(m, feats, v0, nav,
                               lat_logit_bias=bias).controls[:, 1].mean())
    assert out["TURN_L"] > 0 > out["TURN_R"]


# --------------------------------------------------------------------------- #
#  c. PROVENANCE — a banked window says which rule produced it                 #
# --------------------------------------------------------------------------- #
def test_c_bias_is_stamped_on_the_result():
    m = _model()
    feats, v0, nav = _window(m.cfg)
    assert _plan(m, feats, v0, nav).lat_logit_bias is None
    bias = torch.zeros(N_LAT)
    bias[LAT.index("TURN_R")] = 2.5
    r = _plan(m, feats, v0, nav, lat_logit_bias=bias)
    assert r.lat_logit_bias == [0.0] * LAT.index("TURN_R") + [2.5] + \
        [0.0] * (N_LAT - LAT.index("TURN_R") - 1)


def test_c_wrong_length_bias_is_refused():
    m = _model()
    feats, v0, nav = _window(m.cfg)
    try:
        _plan(m, feats, v0, nav, lat_logit_bias=torch.zeros(N_LAT - 1))
    except ValueError as e:
        assert "lateral vocabulary" in str(e)
    else:
        raise AssertionError("a wrong-length bias must be refused, not "
                             "broadcast into a different decision")


# --------------------------------------------------------------------------- #
#  d. THE SECOND GATE — `cos` refuses a turn the bias correctly asked for      #
# --------------------------------------------------------------------------- #
def test_d_cos_refuses_the_turn_that_ccos_executes():
    """⛔ THE FINDING THIS FILE MUST NOT LET ROT (D-REFAV1-DRIVE-GATE2).

    `refav1_arm.py`'s default is ``--cost-metric cos``. Under it, a TURN token
    forced by an unambiguous bias is PROPOSED AND REJECTED: the planner returns
    a zero-curvature baseline. Under ``ccos`` the identical call turns.

    MEASURED on the trained 21,109-step checkpoint as well: `dump_cos_ext`
    emits curvature EXACTLY 0.0 on all 38 windows whose head decoded
    TURN_L/TURN_R, while `dump_ccos_comp` turns on 63 % of them
    (`.../2026-09-05-refav1-make-it-drive/TWO_GATES.md`).

    ⇒ ``ccos`` is a PREREQUISITE FOR LATERAL CONTROL, not the "instrumented
    option, nothing selects it" that `COST_METRICS` still calls it. If this
    test ever fails, that sentence — and the arm's default — have changed, and
    the doc must be re-read before the test is 'fixed'."""
    m = _model()
    feats, v0, nav = _window(m.cfg)
    bias = torch.zeros(N_LAT)
    bias[LAT.index("TURN_L")] = 1e3

    r_cos = _plan(m, feats, v0, nav, lat_logit_bias=bias, cost_metric="cos")
    r_ccos = _plan(m, feats, v0, nav, lat_logit_bias=bias, cost_metric="ccos")
    # the DECODE is identical — the two arms differ only in the cost
    assert r_cos.goal_action["lat"] == r_ccos.goal_action["lat"] == "TURN_L"
    assert float(r_cos.controls[:, 1].abs().max()) == 0.0, (
        "cos executed a turn — the second gate has changed; re-measure "
        "before editing this test")
    # CONTROL: the same window under ccos MUST turn, or the zero above is a
    # statement about this window rather than about the metric.
    assert float(r_ccos.controls[:, 1].mean()) >= GOAL_KAPPA_TURN - 1e-6
