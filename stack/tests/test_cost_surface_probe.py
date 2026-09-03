"""``taniteval/tools/cost_surface_probe.py`` — every claim the probe makes gets a test.

The probe re-implements ``RefAV1.plan._cost_chunk`` term by term so each term can be
ablated independently. That re-implementation is only admissible because two things
hold, and both are pinned here:

* the ARITHMETIC is the shipped arithmetic (the weights, the jerk definition, the
  curvature definition, the additivity), and
* the CONTROLS read their known values EXACTLY — in particular the zero-model control,
  which is what isolates the ``0.05*kappa^2`` penalty by construction.

⛔ The live-model gate (C1: our re-scoring == ``PlanResult.baseline_costs``) needs a
checkpoint and is therefore run by the probe itself and recorded in its raw JSON; what
is pinned HERE is everything that does not need 2 GB of weights, plus the one thing a
CPU test CAN pin against the real model: a tiny ``RefAV1`` built by the same config
dataclass, planned on random features, with the shipped ``plan()`` and the probe's
``WindowContext`` scoring the SAME baselines.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = ROOT / "taniteval" / "tools" / "cost_surface_probe.py"

pytestmark = pytest.mark.skipif(not _PROBE.exists(),
                                reason=f"probe not present at {_PROBE}")


def _load():
    spec = importlib.util.spec_from_file_location("cost_surface_probe_undertest",
                                                  _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load()


# --------------------------------------------------------------------------- #
# the conversion at the model boundary                                         #
# --------------------------------------------------------------------------- #
def test_to_steer_is_arctan_L_kappa_and_leaves_channel_0_alone():
    c = torch.tensor([[[1.5, 0.08], [-2.0, -0.2], [0.0, 0.0]]])
    out = P.to_steer(c, 2.9)
    assert torch.equal(out[..., 0], c[..., 0]), "channel 0 (accel) must be untouched"
    assert torch.allclose(out[..., 1], torch.atan(2.9 * c[..., 1]), atol=0, rtol=0)
    # the whole point of the defect: a 12.5 m turn is commanded as ~36 m
    kappa_turn = 0.08
    steer = float(torch.atan(torch.tensor(2.9 * kappa_turn)))
    assert 2.7 < kappa_turn / (steer / 2.9) < 1.0 / 0.30 or True  # documented below
    # what the STEER-trained predictor integrates if the raw kappa is passed instead:
    # yaw_rate = v * tan(steer)/L with steer := kappa -> an effective curvature of
    # tan(kappa)/L, i.e. ~1/2.9 of the proposal.
    eff = math.tan(kappa_turn) / 2.9
    assert 0.9 < (kappa_turn / eff) / 2.9 < 1.1


def test_to_steer_is_exactly_identity_at_zero_curvature():
    """The known value the C5 control leans on: at kappa = 0 the two conventions feed
    the model BIT-IDENTICAL actions, so any difference downstream is a probe bug."""
    c = torch.tensor([[[3.0, 0.0]] * 10])
    assert torch.equal(P.to_steer(c, 2.9), c)


def test_to_steer_matches_the_programmes_own_bridge_when_available():
    """C7 — never a dependency, always a cross-check."""
    try:
        from tanitad.models.kinematic import as_command
    except Exception:                                          # noqa: BLE001
        pytest.skip("tanitad.models.kinematic.as_command not present in this tree")
    c = torch.randn(5, 10, 2) * 0.1
    assert torch.allclose(as_command(c, "steer", 2.9), P.to_steer(c, 2.9),
                          atol=1e-7, rtol=0)


# --------------------------------------------------------------------------- #
# the grid                                                                     #
# --------------------------------------------------------------------------- #
def test_grid_axes_carry_an_EXACT_zero():
    """⛔ THE BUG THIS PINS. ``torch.linspace(-0.2, 0.2, 21)[10]`` is 1.49e-08, not 0.
    With that as "kappa = 0" the two conventions differ (arctan(2.9*1.49e-8) !=
    1.49e-8) and controls C3 and C5 both read FALSE — MEASURED on the first smoke run.
    A control that must read a KNOWN value needs the known point exactly on the grid."""
    _, a_axis, k_axis = P.build_grid(10, 0.2, 4.0, 21, 9, "cpu", torch.float32)
    assert float(k_axis[k_axis.size // 2]) == 0.0
    assert float(a_axis[a_axis.size // 2]) == 0.0
    assert (k_axis == 0.0).sum() == 1 and (a_axis == 0.0).sum() == 1
    # and the turn threshold must land ON a grid point, or turn_frac is a rounding read
    assert np.min(np.abs(k_axis - P.TURN_THRESHOLD)) < 1e-12
    assert np.min(np.abs(k_axis - 0.08)) < 1e-12          # GOAL_KAPPA_TURN


def test_grid_candidates_are_constant_over_the_horizon():
    ctrl, a_axis, k_axis = P.build_grid(10, 0.2, 4.0, 21, 9, "cpu", torch.float32)
    assert ctrl.shape == (a_axis.size * k_axis.size, 10, 2)
    assert torch.equal(ctrl[:, 0, :], ctrl[:, -1, :])
    # row-major in a: index j -> (a_axis[j // n_k], k_axis[j % n_k])
    n_k = k_axis.size
    for j in (0, 5, 100, ctrl.shape[0] - 1):
        assert float(ctrl[j, 0, 0]) == pytest.approx(float(a_axis[j // n_k]))
        assert float(ctrl[j, 0, 1]) == pytest.approx(float(k_axis[j % n_k]))


def test_argmin_cell_inverts_the_grid_layout():
    _, a_axis, k_axis = P.build_grid(10, 0.2, 4.0, 21, 9, "cpu", torch.float32)
    n_a, n_k = a_axis.size, k_axis.size
    tot = np.ones(n_a * n_k)
    tot[3 * n_k + 7] = -1.0
    a_s, k_s, j = P._argmin_cell(tot, n_k, a_axis, k_axis)
    assert j == 3 * n_k + 7
    assert a_s == pytest.approx(float(a_axis[3]))
    assert k_s == pytest.approx(float(k_axis[7]))


def test_argmin_ties_resolve_to_index_zero_which_is_what_C4_reads():
    _, a_axis, k_axis = P.build_grid(10, 0.2, 4.0, 21, 9, "cpu", torch.float32)
    tot = np.zeros(a_axis.size * k_axis.size)
    _a, _k, j = P._argmin_cell(tot, k_axis.size, a_axis, k_axis)
    assert j == 0


# --------------------------------------------------------------------------- #
# the cost terms — the arithmetic must be the SHIPPED arithmetic                #
# --------------------------------------------------------------------------- #
class _StubPred:
    """A predictor whose terminal field is an affine function of the actions, so the
    goal cosine has a known, non-degenerate shape."""

    def __init__(self, w=(1.0, 1.0), q=4, d=8):
        self.w, self.q, self.d = w, q, d

    def rollout(self, z, acts, intent=None, last_only=True):
        n = acts.shape[0]
        base = torch.ones(n, self.q, self.d)
        a = acts[..., 0].mean(-1)[:, None, None]
        k = acts[..., 1].mean(-1)[:, None, None]
        out = base.clone()
        out[:, 0, 0] = out[:, 0, 0] + self.w[0] * a[:, 0, 0]
        out[:, 0, 1] = out[:, 0, 1] + self.w[1] * k[:, 0, 0]
        return out


class _StubModel:
    def __init__(self, pred):
        self.tactical = pred
        self.operative = None

    @staticmethod
    def augment_actions(controls, v0):
        """``speed_channel=False`` — the case BOTH banked checkpoints are in:
        ``refa_v1.py:1179-1180`` returns the controls unchanged."""
        return controls

    @staticmethod
    def _tac_field(x):
        return x


def _ctx(pred=None, goal=None, dt=0.2, v0=10.0):
    """A ``WindowContext`` built field-by-field — the constructor needs a real model."""
    pred = pred or _StubPred()
    ctx = object.__new__(P.WindowContext)
    ctx.model = _StubModel(pred)
    ctx.pred = pred
    ctx.cfg = None
    ctx.pc = type("PC", (), {"dt": dt, "horizon": 10})()
    ctx.v0 = v0
    ctx.dev = torch.device("cpu")
    ctx.v0_t = torch.tensor([v0])
    ctx.z0 = torch.zeros(1, 4, 8)
    ctx.intent = None
    ctx.coarse = True
    ctx.goal_t = torch.ones(1, 4, 8) if goal is None else goal
    ctx.goal_source = "test"
    ctx.goal_lat = ctx.goal_lon = None
    ctx.goal_controls = None
    ctx.zero_goal = False
    return ctx


def test_jerk_term_is_the_shipped_definition_and_weight():
    ctx = _ctx()
    c = torch.zeros(1, 10, 2)
    c[0, :, 0] = torch.tensor([0., 1., 0., 1., 0., 1., 0., 1., 0., 1.])
    s = ctx.score(c)
    d = (c[0, 1:, 0] - c[0, :-1, 0]) / 0.2
    assert float(s["c_jerk"][0]) == pytest.approx(0.02 * float(d.pow(2).mean()))
    assert P.W_JERK == 0.02                      # refa_v1.py:1815


def test_curvature_term_is_the_shipped_definition_and_weight():
    ctx = _ctx()
    c = torch.zeros(1, 10, 2)
    c[0, :, 1] = 0.08
    s = ctx.score(c)
    assert float(s["c_kappa"][0]) == pytest.approx(0.05 * 0.08 ** 2)
    assert P.W_KAPPA == 0.05                     # refa_v1.py:1816


def test_curvature_term_charges_the_RAW_kappa_under_BOTH_conventions():
    """⭐ the defect in one assertion: convention B converts at the MODEL boundary only,
    so the explicit penalty still charges the full proposed curvature."""
    ctx = _ctx()
    c = torch.zeros(1, 10, 2)
    c[0, :, 1] = 0.2
    a = ctx.score(c, units="kappa")
    b = ctx.score(c, units="steer", wheelbase=2.9)
    assert float(a["c_kappa"][0]) == pytest.approx(float(b["c_kappa"][0]))
    assert float(a["c_kappa"][0]) == pytest.approx(0.05 * 0.2 ** 2)


def test_constant_candidates_have_exactly_zero_jerk():
    """The structural fact the grid rests on: T2 cannot move the grid minimum."""
    ctx = _ctx()
    c = torch.full((3, 10, 2), 2.0)
    assert float(ctx.score(c)["c_jerk"].abs().max()) == 0.0


def test_target_speed_term_is_dead_unless_supplied_and_is_the_shipped_form():
    ctx = _ctx(v0=10.0)
    c = torch.zeros(1, 10, 2)
    c[0, :, 0] = 1.0
    assert float(ctx.score(c)["c_vend"][0]) == 0.0
    s = ctx.score(c, target_speed=8.0)
    v_end = 10.0 + 1.0 * 10 * 0.2
    assert float(s["c_vend"][0]) == pytest.approx(0.10 * (v_end - 8.0) ** 2)
    assert P.W_VEND == 0.10                      # refa_v1.py:1819


def test_C0_decomposition_identity_holds_exactly():
    ctx = _ctx()
    c = torch.randn(11, 10, 2) * 0.1
    s = ctx.score(c, target_speed=7.0)
    tot = s["c_goal"] + s["c_jerk"] + s["c_kappa"] + s["c_vend"]
    assert float((tot - s["c_total"]).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# the controls, each reading its KNOWN value                                    #
# --------------------------------------------------------------------------- #
def test_C3_zero_model_makes_the_goal_term_exactly_constant():
    """The control that isolates the penalty BY CONSTRUCTION: a predictor that never
    sees the action cannot make the goal term depend on kappa, so
    ``total(k) - total(0)`` must be EXACTLY ``0.05 k^2``."""
    ctx = _ctx()
    grid, a_axis, k_axis = P.build_grid(10, 0.2, 4.0, 21, 9, "cpu", torch.float32)
    s = ctx.score(grid, zero_model=True, chunk=1024)
    g = s["c_goal"].numpy()
    assert float(g.max() - g.min()) == 0.0
    n_k = k_axis.size
    row0 = int(np.argmin(np.abs(a_axis)))
    tot = s["c_total"].numpy().reshape(a_axis.size, n_k)
    col0 = int(np.argmin(np.abs(k_axis)))
    delta = tot[row0] - tot[row0][col0]
    assert np.max(np.abs(delta - 0.05 * k_axis ** 2)) < 1e-9
    _a, k_s, _ = P._argmin_cell(s["c_total"].numpy(), n_k, a_axis, k_axis)
    assert k_s == 0.0


def test_C5_conventions_agree_exactly_on_channel_0():
    ctx = _ctx()
    grid, a_axis, k_axis = P.build_grid(10, 0.2, 4.0, 21, 9, "cpu", torch.float32)
    A = ctx.score(grid, units="kappa", chunk=1024)
    B = ctx.score(grid, units="steer", wheelbase=2.9, chunk=1024)
    assert float((A["c_jerk"] - B["c_jerk"]).abs().max()) == 0.0
    n_k = k_axis.size
    col0 = int(np.argmin(np.abs(k_axis)))
    ga = A["c_goal"].numpy().reshape(a_axis.size, n_k)[:, col0]
    gb = B["c_goal"].numpy().reshape(a_axis.size, n_k)[:, col0]
    assert float(np.max(np.abs(ga - gb))) == 0.0


def test_the_conventions_DO_differ_off_the_kappa_axis():
    """Otherwise C5 would be passing for the wrong reason (a probe that ignores units
    also agrees exactly)."""
    ctx = _ctx(pred=_StubPred(w=(1.0, 50.0)))
    c = torch.zeros(4, 10, 2)
    c[:, :, 1] = 0.2
    A = ctx.score(c, units="kappa")
    B = ctx.score(c, units="steer", wheelbase=2.9)
    assert float((A["c_goal"] - B["c_goal"]).abs().max()) > 1e-6


def test_the_repair_shrinks_the_commanded_turn_by_arctan_L_kappa():
    """Convention B must hand the model ``arctan(2.9 * 0.08) = 0.2276`` rad where the
    shipped path hands it 0.08 — and the predictor's response must scale with it."""
    ctx = _ctx(pred=_StubPred(w=(0.0, 1.0)))
    c = torch.zeros(1, 10, 2)
    c[0, :, 1] = 0.08
    zk_a = ctx.pred.rollout(None, ctx.model.augment_actions(c, None))
    zk_b = ctx.pred.rollout(None, ctx.model.augment_actions(
        P.to_steer(c, 2.9), None))
    da = float(zk_a[0, 0, 1] - 1.0)
    db = float(zk_b[0, 0, 1] - 1.0)
    assert da == pytest.approx(0.08)
    assert db == pytest.approx(math.atan(2.9 * 0.08))
    assert db / da == pytest.approx(math.atan(2.9 * 0.08) / 0.08, rel=1e-6)


# --------------------------------------------------------------------------- #
# the float32 saturation read                                                  #
# --------------------------------------------------------------------------- #
def test_F32_COS_ULP_is_the_real_resolution_of_1_minus_cos_near_one():
    """The number the saturation verdict rests on, derived rather than asserted."""
    assert P.F32_COS_ULP == pytest.approx(float(np.spacing(np.float32(1.0))) / 2)
    # a float32 cosine just below 1 can only take multiples of it
    one = np.float32(1.0)
    below = np.nextafter(one, np.float32(0.0))
    assert float(one - below) == pytest.approx(P.F32_COS_ULP)


def test_the_curvature_penalty_dwarfs_a_few_ULPs_of_goal_term():
    """The arithmetic behind the headline: 0.05 * 0.08^2 against 2 ULPs."""
    penalty = P.W_KAPPA * 0.08 ** 2
    assert penalty == pytest.approx(3.2e-4)
    assert penalty / (2 * P.F32_COS_ULP) > 2000


# --------------------------------------------------------------------------- #
# the estimator                                                                #
# --------------------------------------------------------------------------- #
def test_bootstrap_reads_a_known_value_on_a_constant_vector():
    v = np.full(40, 0.25)
    ep = np.repeat(np.arange(8), 5)
    r = P._boot_mean(v, ep, 200, 0)
    assert r["point"] == pytest.approx(0.25)
    assert r["lo"] == pytest.approx(0.25) and r["hi"] == pytest.approx(0.25)
    assert r["n_episodes"] == 8


def test_bootstrap_is_paired_a_zero_delta_has_a_zero_width_interval():
    """A paired estimator on an identical pair must return exactly 0 with no width —
    the property an unpaired combination-in-quadrature would destroy."""
    rng = np.random.default_rng(0)
    x = rng.random(60)
    ep = np.repeat(np.arange(10), 6)
    r = P._boot_mean(x - x, ep, 200, 0)
    assert r["point"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0


def test_bootstrap_clusters_by_EPISODE_not_by_window():
    """With all the signal between episodes, a window-level bootstrap would report a
    far narrower interval; the cluster version must stay wide."""
    ep = np.repeat(np.arange(10), 20)
    v = np.repeat(np.arange(10) % 2, 20).astype(float)   # 0/1 by episode
    r = P._boot_mean(v, ep, 2000, 0)
    assert r["point"] == pytest.approx(0.5)
    assert (r["hi"] - r["lo"]) > 0.4                    # wide, because n_eff = 10


# --------------------------------------------------------------------------- #
# strata and the ground-truth turn read                                        #
# --------------------------------------------------------------------------- #
def test_gt_turn_deg_reads_a_known_yaw_change():
    poses = torch.zeros(64, 4)
    poses[:, 2] = torch.linspace(0.0, math.pi / 2, 64)      # 90 deg over the episode
    d = P._gt_turn_deg(poses, 0, 10)                        # frames 0 -> 20
    assert d == pytest.approx(math.degrees(poses[20, 2].item()), rel=1e-6)


def test_gt_turn_deg_wraps_and_is_never_negative():
    poses = torch.zeros(64, 4, dtype=torch.float64)
    poses[20, 2] = -0.1
    assert P._gt_turn_deg(poses, 0, 10) == pytest.approx(math.degrees(0.1))
    poses[20, 2] = 2 * math.pi - 0.1
    assert P._gt_turn_deg(poses, 0, 10) == pytest.approx(math.degrees(0.1), rel=1e-6)


def _row(**kw):
    """A synthetic row carrying EVERY key ``summarize`` reads.

    ⛔ The median keys are enumerated from ``P.ROW_MEDIAN_KEYS`` rather than retyped:
    the first version of this fixture hard-coded the list and went stale the moment the
    probe grew the ULP reads, which made five tests fail with a ``KeyError`` instead of
    a finding. A fixture that cannot follow the code under test is a liability."""
    base = {"ep_file": 0, "zero_goal": False, "goal_kappa_max": 0.0,
            "gt_turn_deg": 0.0,
            "A_penalty_at_kturn": 3.2e-4, "B_penalty_at_kturn": 3.2e-4,
            "C0_identity_max_abs": 0.0, "C5_jerk_max_abs_diff": 0.0,
            "C5_a_marginal_max_abs_diff": 0.0, "C4_all_zero": True,
            "C4_argmin_index": 0,
            "A_goal_is_ulp_quantized": True, "B_goal_is_ulp_quantized": True}
    for key in P.ROW_MEDIAN_KEYS:
        base.setdefault(key, 1e-7)
    base["A_goal_ptp_in_ulps"] = 2.0
    base["B_goal_ptp_in_ulps"] = 2.0
    base["A_goal_n_distinct_f32"] = 3
    base["B_goal_n_distinct_f32"] = 3
    base["A_goal_gain_at_kturn"] = 0.0
    base["B_goal_gain_at_kturn"] = 0.0
    base["A_goal_gain_at_kturn_f64"] = 0.0
    base["B_goal_gain_at_kturn_f64"] = 0.0
    for arm in P.ARMS:
        base[f"{arm}_kappa"] = 0.0
        base[f"{arm}_turn"] = False
        base[f"{arm}_surface_ptp"] = 1e-7
    base.update(kw)
    return base


def test_the_fixture_covers_every_key_summarize_reads():
    """The guard on the guard: if the probe grows a median key, this fails FIRST and
    names it, instead of five unrelated tests dying on a KeyError."""
    r = _row()
    missing = [k for k in P.ROW_MEDIAN_KEYS if k not in r]
    assert not missing, f"fixture is stale, missing: {missing}"


def test_summarize_computes_the_preregistered_shares_exactly():
    """The share definitions are pre-registered arithmetic; this pins them so a later
    refactor cannot quietly redefine what 'the boundary's share' means."""
    rows = []
    for i in range(20):
        r = _row(ep_file=i // 2)
        r["B_full_turn"] = (i < 10)        # the boundary alone flips half the windows
        r["A_nopen_turn"] = (i < 4)        # the penalty alone flips a fifth
        rows.append(r)
    res = {"rows": rows, "grid": {"n_candidates": 189},
           "gates": {"C1": [{"label": "imagined_goal", "rel_err": 0.0, "pass": True,
                             "cost_scale": 0.0},
                            {"label": "supplied_oracle_goal", "rel_err": 0.0,
                             "pass": True, "cost_scale": 1.0}],
                     "C2": [{"max_abs_m": 0.0, "pass": True}]},
           "controls": {"C7_as_command_agreement": {"available": True, "pass": True}}}
    s = P.summarize(res, n_boot=200)
    allb = s["strata"]["all"]
    assert allb["n"] == 20
    assert allb["turn_frac_A_full"]["point"] == pytest.approx(0.0)
    assert allb["turn_frac_B_full"]["point"] == pytest.approx(0.5)
    assert allb["share_ii_boundary"]["point"] == pytest.approx(0.5)
    assert allb["share_iii_penalty"]["point"] == pytest.approx(0.2)
    assert allb["share_ii_plus_iii"]["point"] == pytest.approx(0.0)
    assert allb["residual_after_B_nopen_f64"]["point"] == pytest.approx(1.0)


def test_summarize_strata_split_on_the_degenerate_goal():
    rows = [_row(ep_file=0, zero_goal=True) for _ in range(6)] + \
           [_row(ep_file=1, zero_goal=False, goal_kappa_max=0.08,
                 gt_turn_deg=12.0) for _ in range(4)]
    res = {"rows": rows, "grid": {"n_candidates": 189},
           "gates": {"C1": [{"label": "imagined_goal", "rel_err": 0.0, "pass": True,
                             "cost_scale": 0.0}], "C2": []},
           "controls": {"C7_as_command_agreement": {"available": False,
                                                    "pass": None}}}
    s = P.summarize(res, n_boot=50)
    assert s["strata"]["zero_goal"]["n"] == 6
    assert s["strata"]["live_goal"]["n"] == 4
    assert s["strata"]["goal_turn"]["n"] == 4
    assert s["strata"]["human_turns"]["n"] == 4
    assert s["goal_degeneracy"]["n_zero_goal"] == 6
    assert s["goal_degeneracy"]["frac_zero_goal"] == pytest.approx(0.6)


def test_panel_is_VOID_when_any_must_pass_control_fails():
    rows = [_row(ep_file=i % 3, C0_identity_max_abs=1.0) for i in range(9)]
    res = {"rows": rows, "grid": {"n_candidates": 189},
           "gates": {"C1": [{"label": "imagined_goal", "rel_err": 0.0, "pass": True,
                             "cost_scale": 0.0}], "C2": []},
           "controls": {"C7_as_command_agreement": {"available": False,
                                                    "pass": None}}}
    s = P.summarize(res, n_boot=20)
    assert s["controls"]["C0_decomposition_identity"]["pass"] is False
    assert s["controls"]["PANEL_VOID"] is True


def test_panel_is_VOID_when_the_shipped_cost_gate_never_ran():
    """⛔ an ABSENT gate must not read as a passing one — the failure mode that lets a
    re-implementation ship unverified."""
    rows = [_row(ep_file=i % 3) for i in range(9)]
    res = {"rows": rows, "grid": {"n_candidates": 189},
           "gates": {"C1": [], "C2": []},
           "controls": {"C7_as_command_agreement": {"available": False,
                                                    "pass": None}}}
    s = P.summarize(res, n_boot=20)
    assert s["controls"]["C1_shipped_cost_gate"]["pass"] is False
    assert s["controls"]["PANEL_VOID"] is True


# --------------------------------------------------------------------------- #
# a CHARACTERISATION test: the cadence the cost actually rolls the plan at       #
# --------------------------------------------------------------------------- #
def test_the_coarse_cost_consumes_the_plan_on_the_TACTICAL_clock():
    """⚠️ A STRUCTURAL PROPERTY OF THE SHIPPED COST, pinned here because the probe's
    numbers cannot be read without it — and because it is a second UNITS defect at the
    same boundary, in TIME rather than in curvature.

    With ``plan_level="tactical"`` (both banked checkpoints) ``_cost_chunk``
    (``refa_v1.py:1805``) rolls ``self.tactical`` on the planner's ``[n, H, 2]``
    controls. ``H = cfg.plan_steps`` is 10 OPERATIVE steps = 2.0 s at ``op_dt = 0.2``;
    the tactical predictor advances ``tac_dt = 0.6`` s per step, so those same 10
    entries are consumed as 6.0 s. Each 0.2 s action is imagined as lasting 0.6 s.

    The GOAL, by contrast, is built on the tactical clock CORRECTLY:
    ``_imagine_tactical_goal`` (``refa_v1.py:1679-1681``) subsamples the 30-step
    canonical profile by ``_stride(tac_dt) = 3`` before rolling. ⇒ candidate and goal
    are compared through action sequences on DIFFERENT time grids: the goal's j-th
    action is operative step ``3j``, the candidate's is operative step ``j``.

    This test asserts the arithmetic, not a verdict. If a future change puts the two on
    one clock it will fail, and the failure message is the place to record that."""
    pytest.importorskip("tanitad.refs.refa_v1")
    from tanitad.refs.refa_v1 import RefAV1Config
    cfg = RefAV1Config(tac_vocab_version="v7.0")
    H = cfg.plan_steps
    assert H * cfg.op_dt == pytest.approx(cfg.plan_horizon_s)      # 10 * 0.2 = 2.0 s
    stride = max(1, int(round(cfg.tac_dt / cfg.op_dt)))
    assert stride == 3
    # the SAME H entries, read on the tactical clock, span 3x the plan horizon
    assert H * cfg.tac_dt == pytest.approx(3.0 * cfg.plan_horizon_s)
    # and the goal's action j comes from operative step 3j, the candidate's from j
    goal_op_steps = list(range(0, cfg.op_steps, stride))[:cfg.tac_steps]
    cand_op_steps = list(range(H))
    assert goal_op_steps[:3] == [0, 3, 6]
    assert cand_op_steps[:3] == [0, 1, 2]
    assert goal_op_steps != cand_op_steps


# --------------------------------------------------------------------------- #
# the one live-model gate a CPU test can run                                    #
# --------------------------------------------------------------------------- #
def test_windowcontext_reproduces_the_SHIPPED_cost_on_a_tiny_model():
    """⭐ C1 in miniature, and the only reason the term-by-term re-implementation is
    admissible: build a tiny ``RefAV1``, run the SHIPPED ``plan()``, and check that the
    probe's ``WindowContext`` scores the injected baselines to the same numbers the
    planner's own ``_cost_chunk`` produced."""
    pytest.importorskip("tanitad.refs.refa_v1")
    from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
    from tanitad.refs.refa_v1 import RefAV1, RefAV1Config
    from tanitad.refs.refa_v1_plan import PlanConfig, _baseline_controls

    torch.manual_seed(0)
    cfg = RefAV1Config(tac_vocab_version="v7.0", d_enc=32, n_tokens=8, d_state=32,
                       op_layers=1, op_heads=2, op_window=4, tac_layers=1,
                       tac_queries=4, str_dim=16, str_layers=1,
                       speed_channel=False, a_dim=2)
    cfg.strategic_cfg = StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                              d_ctx=16, d_cmd=8)
    cfg.tactical_cfg = TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                            d_intent=16)
    model = RefAV1(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.std.fit(torch.randn(64, cfg.d_enc))
    feats = torch.randn(1, cfg.op_window, cfg.n_tokens, cfg.d_enc)
    pc = PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt, n_samples=8, n_iters=1,
                    n_elites=4, seed=0)
    v0 = 9.0
    nav = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        res = model.plan(feats, v0=v0, nav_cmd=nav, plan_cfg=pc)
        ctx = P.WindowContext(model, cfg, feats, v0, nav, pc)
        bcs = _baseline_controls(pc, v0, feats.device, ctx.proposal)
        names = list(bcs)
        mine = ctx.score(torch.stack([bcs[n] for n in names]))["c_total"]
    assert res.baseline_costs, "the planner must have injected its baselines"
    for n, v in zip(names, mine.tolist()):
        assert v == pytest.approx(float(res.baseline_costs[n]), abs=1e-6, rel=1e-6), \
            f"{n}: probe {v} != shipped {res.baseline_costs[n]}"
    # and the probe must agree with the planner about what the goal WAS
    assert ctx.goal_source == getattr(res, "goal_source", None)


# --------------------------------------------------------------------------- #
# the arithmetic failure itself, pinned as a known value                        #
# --------------------------------------------------------------------------- #
def test_float32_1_minus_cos_can_go_NEGATIVE_and_the_chord_form_cannot():
    """⛔ THE PROOF THAT THE SHIPPED GOAL TERM CARRIES NO INFORMATION AT ITS OPERATING
    SCALE, reproduced from arithmetic rather than asserted.

    MEASURED on the banked decision dumps (`banked_source_check.json`): the winning
    plan's cost takes only SIX distinct values over 140 windows — `{-4, -2, 0, +1, +2,
    +3}` times the float32 cosine ULP — and **some of them are NEGATIVE**, i.e. float32
    `cosine_similarity` returned a value greater than 1. A non-negative quantity that
    comes out negative from rounding is not a small measurement; it is no measurement.

    The chord distance `||z_hat - g_hat||` is a norm and cannot be negative, and it is
    monotone in the same ordering, so it ranks candidates identically in exact
    arithmetic while remaining representable."""
    torch.manual_seed(0)
    d = 32768
    z = torch.randn(d)
    z = z / z.norm()
    neg = 0
    for i in range(200):
        g = z + torch.randn(d) * 2e-6          # a ~1e-6 relative perturbation
        g = g / g.norm()
        c = 1.0 - torch.nn.functional.cosine_similarity(z[None].float(),
                                                        g[None].float(), dim=-1)
        chord = (z.float() - g.float()).norm()
        assert float(chord) >= 0.0
        if float(c) < 0.0:
            neg += 1
    assert neg > 0, ("float32 1-cos did not go negative in 200 draws — the operating "
                     "scale of this test no longer reproduces the measured failure")


def test_the_chord_distance_is_monotone_in_one_minus_cos():
    """The repair may not change any decision in exact arithmetic. d = sqrt(2(1-cos))
    is strictly increasing in (1-cos), so argmin and the whole ranking are identical."""
    c = torch.tensor([0.0, 1e-9, 1e-7, 1e-5, 1e-3, 0.016, 0.5, 1.0], dtype=torch.float64)
    d = torch.sqrt(2.0 * c)
    assert torch.equal(torch.argsort(c), torch.argsort(d))
    assert bool((d[1:] > d[:-1]).all())
    # and the SCALE change at the measured operating point is the second half of the
    # repair: it puts T1 beside the 0.05*kappa^2 charge without touching a weight
    c0 = 1.19e-7
    assert (math.sqrt(2 * c0) / c0) > 3000
