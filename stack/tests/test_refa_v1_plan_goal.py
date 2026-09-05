"""plan()'s DEFAULT goal is the tactical brain's own imagined 6 s field
(design change #8) — and the tie-break that stops a constant brake.

⛔ THE DEFECT (MEASURED by the T1-adapter build, `test_refav1_arm.py`: 51/51
windows, 27/27 in its CLI smoke): with ``goal_field=None`` the cost was jerk +
curvature only, every zero-curvature constant-acceleration candidate scored
EXACTLY 0, `icem_plan`'s floor loop keeps the LAST tie, and the deployed plan
was ``baseline:decel_1.5`` — a constant −1.5 m/s² brake — on every window,
while `plan()`'s docstring promised "the tactical brain's own imagined 6 s
field".

WHAT IS NOW TRUE, each pinned below:
  * the goal is imagined from vision + nav (through `_run_brains`, the one
    shared nav site) + the measured v0: the trained factored heads decode the
    intent into a (lat, lon) token, `canonical_controls` turns the token into
    a profile, and the TACTICAL predictor rolls its own field 10 x 0.6 s;
  * ONE goal space — the tactical query field [Q, d]; operative terminal
    fields and supplied operative goals are pooled through `_tac_field`;
  * tie-break: among baselines tied at the winning cost `hold_v0` (then `cv`)
    wins — do nothing beats braking when the cost cannot tell them apart;
  * provenance on the result: ``res.goal_source`` / ``goal_space`` /
    ``goal_action``;
  * the canonical controls behind the goal SEED the search (`seed_pool`) —
    without that, `icem_plan`'s zero-mean coloured noise cannot express a
    sustained curvature and the planner returned `hold_v0` on 24/24 windows
    against a TURN goal at both init scales (MEASURED 2026-09-02).

⚠️ THE LOAD-BEARING TEST RUNS TWO REGIMES, both random-init. At the SHIPPED
init the residual heads are 1e-3-scaled (`RESIDUAL_HEAD_INIT_SCALE`) and kappa
enters `act` at ~0.08 raw units (20x smaller than a), so every rollout sits
within ~1e-3 of the start field and the smoothness terms outweigh any goal
difference: the floor wins by a property of the INIT, not of the planner. So
the shipped-init regime pins the DEFECT SIGNATURE (no all-tied windows, never
a brake by tie), and an "informative world" regime (heads at unit scale, the
deliberate-regression init, and the action pathway compensated for kappa's raw
scale) pins that the search WINS. MEASURED 2026-09-02, 24 windows, planner
8x1 (the adapter's settings): gains (1,1) -> floor 24/24 on model seeds 0/1/2;
(1,1000) -> cem 6/23/10; (50,1000) -> cem 24/24/14; decel-by-tie 0 in all 24
(seed x gain x planner) cells.
"""
import collections
import math

import pytest
import torch

import tanitad.refs.refa_v1 as refa_v1_mod
from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
from tanitad.refs.refa_v1 import (GOAL_A_MAX, GOAL_KAPPA_TURN, SPEED_SCALE_MPS,
                                  RefAV1, RefAV1Config, canonical_controls)
from tanitad.refs.refa_v1_plan import PlanConfig

N_WIN = 24


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
    """The T1 adapter's own planner settings (`test_refav1_arm._args`)."""
    return PlanConfig(horizon=c.plan_steps, dt=c.op_dt, seed=0,
                      n_samples=8, n_iters=1, n_elites=4)


def _windows(c: RefAV1Config, n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        out.append((torch.randn(1, c.op_window, c.n_tokens, c.d_enc, generator=g),
                    float(torch.rand(1, generator=g) * 20.0),
                    torch.randint(0, 4, (1,), generator=g)))
    return out


def _informative(m: RefAV1, act_gain: float = 50.0,
                 head_gain: float = 1000.0) -> None:
    """A random-init world that RESPONDS to actions (module docstring)."""
    with torch.no_grad():
        for pred in (m.operative, m.tactical):
            pred.act[0].weight.mul_(act_gain)
            pred.head[-1].weight.mul_(head_gain)
            pred.head[-1].bias.mul_(head_gain)


# ------------------------------------------------------- (a) determinism --
def test_a_the_imagined_goal_is_deterministic_and_has_the_stated_shape():
    m = _model()
    c = m.cfg
    feats, v0, nav = _windows(c, 1)[0]
    g1, a1 = m.imagined_goal(feats, v0=v0, nav_cmd=nav)
    g2, a2 = m.imagined_goal(feats, v0=v0, nav_cmd=nav)
    assert g1.shape == (1, c.tac_queries, c.d_state)       # tactical query field
    assert torch.equal(g1, g2)
    assert a1["lat"] == a2["lat"] and a1["lon"] == a2["lon"]
    assert torch.equal(a1["controls"], a2["controls"])
    assert a1["controls"].shape == (1, c.op_steps, 2)
    assert a1["lat"][0] in tactical_lat_actions("v7.0")
    assert a1["lon"][0] in tactical_lon_actions_v("v7.0")
    # nav reaches the goal through `_run_brains`, the one shared nav site
    g3, _ = m.imagined_goal(feats, v0=v0,
                            nav_cmd=torch.tensor([(int(nav) + 1) % 4]))
    assert not torch.equal(g1, g3)
    # and plan() itself is deterministic on the same (feats, nav, v0)
    r1 = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c))
    r2 = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c))
    assert torch.equal(r1.controls, r2.controls) and r1.source == r2.source
    assert r1.goal_source == "tactical_imagined"
    # no hierarchy => no tactical imagination, refused by name
    with pytest.raises(ValueError, match="hierarchy"):
        _model(0, strategic_cfg=None, tactical_cfg=None).imagined_goal(
            feats, v0=v0)


# ------------------------------------------- (b) THE LOAD-BEARING ONE --
def test_b_THE_LOAD_BEARING_ONE_no_brake_by_tie_and_the_search_beats_the_floor():
    m = _model(0)
    c = m.cfg
    wins = _windows(c, N_WIN)
    # --- regime 1: the shipped init — the defect signature must be GONE ---
    tied_all = decel_by_tie = 0
    sources = collections.Counter()
    for feats, v0, nav in wins:
        res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c))
        assert res.goal_source == "tactical_imagined"
        assert res.goal_space == "tactical_query_field"
        bc = res.baseline_costs
        if max(bc.values()) - min(bc.values()) < 1e-12:
            tied_all += 1                    # the defect: all baselines at 0
        if (res.source == "baseline:decel_1.5"
                and not bc["decel_1.5"] < bc["hold_v0"]):
            decel_by_tie += 1                # the brake chosen by tie order
        sources[res.source] += 1
    assert tied_all == 0, sources            # the goal term is live everywhere
    assert decel_by_tie == 0, sources        # a brake only when the cost says so
    # --- regime 2: an informative world — the search must WIN --------------
    m = _model(0)
    _informative(m)
    src2 = collections.Counter(
        m.plan(f, v0=v, nav_cmd=n, plan_cfg=_pc(c)).source for f, v, n in wins)
    assert src2["cem"] > N_WIN // 2, src2    # MEASURED 24/24 at this seed


# ------------------------------------------------------- (c) tie-break --
def test_c_with_every_candidate_tied_the_chosen_baseline_is_hold_v0_not_decel():
    m = _model(0)
    c = m.cfg

    def blind(field, actions, intent=None, last_only=False):
        return (field if last_only
                else field[:, None].expand(-1, actions.shape[1], -1, -1))

    m.operative.rollout = blind          # every candidate lands on the same
    m.tactical.rollout = blind           # field => the goal term cannot rank
    feats, v0, nav = _windows(c, 1)[0]
    res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c))
    assert math.isclose(res.baseline_costs["decel_1.5"],
                        res.baseline_costs["hold_v0"])          # they DID tie
    assert res.source == "baseline:hold_v0"
    assert torch.equal(res.controls, torch.zeros_like(res.controls))
    # the goal-free path (no hierarchy) is the EXACT pre-fix situation — every
    # zero-curvature constant-accel candidate at exactly 0 — and now holds
    # speed instead of braking
    m3 = _model(0, strategic_cfg=None, tactical_cfg=None)
    c3 = m3.cfg
    res3 = m3.plan(torch.randn(1, c3.op_window, c3.n_tokens, c3.d_enc),
                   v0=5.0, plan_cfg=_pc(c3))
    assert res3.goal_source == "none" and res3.goal_action is None
    assert res3.baseline_costs["decel_1.5"] == 0.0
    assert res3.baseline_costs["hold_v0"] == 0.0
    assert res3.source == "baseline:hold_v0"
    assert torch.equal(res3.controls, torch.zeros_like(res3.controls))


# --------------------------------------------------- (d) speed channel --
def test_d_the_goal_rollout_receives_the_integrated_speed_channel():
    m = _model(0, speed_channel=True)
    c = m.cfg
    seen = []
    orig = m.tactical.rollout

    def spy(field, actions, intent=None, last_only=False):
        seen.append(actions.clone())
        return orig(field, actions, intent=intent, last_only=last_only)

    m.tactical.rollout = spy
    feats, _, nav = _windows(c, 1)[0]
    v0 = 12.0
    g, ga = m.imagined_goal(feats, v0=v0, nav_cmd=nav)
    a = seen[0]
    assert a.shape == (1, c.tac_steps, 3)
    stride = int(round(c.tac_dt / c.op_dt))
    expect = m.augment_actions(ga["controls"], torch.tensor([v0]))
    expect = expect[:, ::stride][:, :c.tac_steps]      # exactly forward's tac_a
    assert torch.equal(a, expect)
    assert float(a[0, 0, 2]) * SPEED_SCALE_MPS == pytest.approx(v0)
    g2, _ = m.imagined_goal(feats, v0=v0 + 5.0, nav_cmd=nav)
    assert not torch.equal(g, g2)                        # the channel is live
    # and plan() under the speed channel: the goal rollout, the search and the
    # re-score all consume 3-wide inputs
    seen.clear()
    res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c))
    assert seen and all(x.shape[-1] == 3 for x in seen)
    assert res.goal_source == "tactical_imagined"
    assert res.controls.shape == (c.plan_steps, 2)


# ------------------------------------------------------ (e) provenance --
def test_e_goal_provenance_is_stamped_on_the_result():
    m = _model(0)
    c = m.cfg
    feats, v0, nav = _windows(c, 1)[0]
    res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c))
    assert res.goal_source == "tactical_imagined"
    # ⭐ `kappa_turn_used` / `kappa_vocab` joined the dict with M15's level set
    # (2026-09-05). Kept as an EXACT set on purpose: a dump reader indexes these
    # keys, and a silent addition is how a provenance field starts being
    # inferred instead of read.
    assert set(res.goal_action) == {"lat", "lon", "controls",
                                    "kappa_turn_used", "kappa_vocab"}
    # the SHIPPED path names the shipped action space and commands no level
    assert res.goal_action["kappa_vocab"] == "L1-0.08"
    assert res.goal_action["kappa_turn_used"] is None
    assert res.goal_action["controls"].shape == (c.op_steps, 2)
    g_a = torch.randn(1, c.n_tokens, c.d_state)          # operative-space goal
    sup = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c), goal_field=g_a)
    assert sup.goal_source == "supplied" and sup.goal_action is None
    sup2 = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c),
                  goal_field=g_a * -1.0)
    assert sup.baseline_costs != sup2.baseline_costs   # the supplied goal is used
    # the fine re-score compares in the SAME space: it ran and reported
    assert sup.fine_costs and "plan" in sup.fine_costs


# ------------------------------------------------------------ (f) seed --
def test_f_the_decoded_action_seeds_the_search_population(monkeypatch):
    captured = {}
    real = refa_v1_mod.icem_plan

    def spy(cost_fn, **kw):
        captured.update(kw)
        return real(cost_fn, **kw)

    monkeypatch.setattr(refa_v1_mod, "icem_plan", spy)
    m = _model(0)
    c = m.cfg
    feats, v0, nav = _windows(c, 1)[0]
    res = m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c))
    sp = captured["seed_pool"]
    assert sp is not None and sp.shape == (1, c.plan_steps, 2)
    assert torch.equal(sp[0], res.goal_action["controls"][:c.plan_steps])
    # multimodal proposals and the seed coexist
    m2 = _model(0, proposal_k=3)
    m2.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c))
    assert captured["seed_pool"].shape == (3, c.plan_steps, 2)
    # a supplied goal seeds nothing (no decoded action behind it)
    m.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=_pc(c),
           goal_field=torch.randn(1, c.n_tokens, c.d_state))
    assert captured["seed_pool"] is None


# --------------------------------------------- (g) canonical controls --
def test_g_canonical_controls_follow_the_emitter_definitions():
    K, dt = 30, 0.2
    zeros = torch.zeros(K, 2)

    def v_of(ctrl, v0):
        return v0 + torch.cumsum(ctrl[:, 0], 0) * dt

    assert torch.equal(canonical_controls("LANE_KEEP", "CRUISE", 10.0, K, dt), zeros)
    brake = canonical_controls("LANE_KEEP", "BRAKE_TO", 10.0, K, dt)
    assert float(brake[0, 0]) == -GOAL_A_MAX                # floor rate first
    v = v_of(brake, 10.0)
    assert 7.0 <= float(v[-1]) <= 7.3 and float(v.min()) >= 7.0 - 1e-6
    hold = canonical_controls("LANE_KEEP", "HOLD", 1.0, K, dt)
    v = v_of(hold, 1.0)
    assert float(v.min()) >= -1e-6 and float(v[-1]) < 0.1   # to a stop, never below 0
    assert float(canonical_controls("LANE_KEEP", "ACCELERATE", 10.0, K, dt)[0, 0]) > 0
    assert float(canonical_controls("LANE_KEEP", "CREEP", 10.0, K, dt)[0, 0]) == -GOAL_A_MAX
    assert float(canonical_controls("LANE_KEEP", "ADAPT_SPEED_FOR_CURVE", 5.0, K, dt)[0, 0]) == 0.0
    assert float(canonical_controls("LANE_KEEP", "ADAPT_SPEED_FOR_CURVE", 12.0, K, dt)[0, 0]) < 0
    tl = canonical_controls("TURN_L", "CRUISE", 5.0, K, dt)[:, 1]
    assert torch.all(tl[:20] == GOAL_KAPPA_TURN) and torch.all(tl[20:] == 0)
    tr = canonical_controls("TURN_R", "CRUISE", 5.0, K, dt)[:, 1]
    assert torch.all(tr[:20] == -GOAL_KAPPA_TURN)
    lc = canonical_controls("LANE_CHANGE_R", "CRUISE", 10.0, K, dt)[:, 1]
    kap = 2 * 1.75 / (10.0 ** 2 * 2.0 ** 2)
    assert torch.allclose(lc[:10], torch.full((10,), -kap))
    assert torch.allclose(lc[10:20], torch.full((10,), kap))
    assert torch.all(lc[20:] == 0)
    nd = canonical_controls("NUDGE_L", "CRUISE", 10.0, K, dt)[:, 1]
    assert torch.allclose(nd[:5], torch.full((5,), 0.01))
    assert torch.allclose(nd[5:10], torch.full((5,), -0.01))
    assert torch.all(nd[10:] == 0)
    slow = canonical_controls("LANE_CHANGE_L", "CRUISE", 0.0, K, dt)[:, 1]
    assert float(slow.abs().max()) <= 0.2                  # kappa_max at v0 = 0
    assert torch.equal(canonical_controls("NO_SUCH", "NOPE", 10.0, K, dt), zeros)
    # the v6.0 vocabulary is covered too
    for lat in tactical_lat_actions("v6.0"):
        for lon in tactical_lon_actions_v("v6.0"):
            assert torch.isfinite(canonical_controls(lat, lon, 7.0, K, dt)).all()
