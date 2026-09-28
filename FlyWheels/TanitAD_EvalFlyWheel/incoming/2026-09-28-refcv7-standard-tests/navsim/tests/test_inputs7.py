"""refcv7's NavSim inputs, each against a LITERAL or ANALYTIC target, each with an arm that must go
RED. TANITAD VENV, CPU, no checkpoint:  pytest -q tests/test_inputs7.py

Expectations are LITERALS (or closed-form sums written here, independent of the code under test) —
CLAUDE.md: "a check that shares the defect it checks for is green forever".
"""
from __future__ import annotations

import ast
import copy
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import refcv7_bridge as R7  # noqa: E402
import torch  # noqa: E402
from tanitad.models.ego_history import ego_channels_from_poses  # noqa: E402

TIMES = [-1.5, -1.0, -0.5, 0.0]
TRAINER = os.path.join(R7.boot7.TREE, "stack", "scripts", "refc_v3_train.py")


def _es(x, y, h, vx, vy=0.0, ax=0.0, ay=0.0, cmd=(0, 1, 0, 0)):
    return {"ego_pose": [x, y, h], "ego_velocity": [vx, vy], "ego_acceleration": [ax, ay],
            "driving_command": list(cmd)}


def _const_accel(a=2.0, v_t0=10.0, times=TIMES):
    return [_es(v_t0 * t + 0.5 * a * t * t, 0.0, 0.0, v_t0 + a * t) for t in times]


def _hist(st, arm="R7_A1", times=TIMES):
    return R7.ego_history_poses(R7.declare7(st, arm), times)


# --------------------------------------------------------------------------- #
# 0. the tree under test                                                        #
# --------------------------------------------------------------------------- #
def test_the_tree_is_the_clean_archive():
    import tanitad
    t = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
    assert t.startswith(R7.boot7.TREE.lower() + "/stack/"), t
    assert os.path.abspath(R7.R6.__file__).replace("\\", "/").lower().startswith(R7.boot7.TREE.lower())


# --------------------------------------------------------------------------- #
# 1. arms -> the refcv6 enforcement templates (LITERAL)                          #
# --------------------------------------------------------------------------- #
HIST = ("ego_pose[1]", "ego_pose[2]", "ego_pose[3]",
        "ego_velocity[1]", "ego_velocity[2]", "ego_velocity[3]")
EXPECT = {  # arm -> (declared, frames, nav, vmax, filter, seed)
    "R7_A1": (HIST + ("driving_command[3]",), "ST", "cmd", "map", True, 0),
    "R7_A1_s1": (HIST + ("driving_command[3]",), "ST", "cmd", "map", True, 1),
    "R7_FILTOFF": (HIST + ("driving_command[3]",), "ST", "cmd", "map", False, 0),
    "R7_BLIND": (HIST + ("driving_command[3]",), "BLIND", "cmd", "map", True, 0),
    "R7_NAVOFF": (HIST, "ST", None, "map", True, 0),
    "R7_VMAXOFF": (HIST + ("driving_command[3]",), "ST", "cmd", "off", True, 0),
    "R7_A1NT": (HIST + ("driving_command[3]",), "NT", "cmd", "map", True, 0),
    "R7_VMAXORACLE": (HIST + ("driving_command[3]",), "ST", "cmd", "oracle", True, 0),
}


@pytest.mark.parametrize("arm", sorted(EXPECT))
def test_arm_specs_are_the_literal(arm):
    d, fr, nav, vm, fl, sd = EXPECT[arm]
    s = R7.ARMS7[arm]
    assert tuple(sorted(R7.declared(arm))) == tuple(sorted(d))
    assert (s["frames"], s["nav"], s["vmax"], s["filter"], s["seed"]) == (fr, nav, vm, fl, sd)


def test_no_arm_declares_acceleration():
    """refcv7's NavSim feed reads NO ego_acceleration field (the prior is POSE-derived)."""
    for arm in R7.ARMS7:
        assert not any(x.startswith("ego_acceleration") for x in R7.declared(arm))


def test_REGRESSION_a_mismatched_template_is_refused(monkeypatch):
    monkeypatch.setitem(R7.R6_TEMPLATE, "R7_NAVOFF", "R6_A1")      # would feed the command
    with pytest.raises(R7.RefusedInput):
        R7.declare7(_const_accel(), "R7_NAVOFF")


def _inputs(st, arm):
    d = R7.declare7(st, arm)
    return (R7.ego_history_poses(d, TIMES).tobytes(), R7.v0_of(d),
            R7.nav_input(d, arm)["nav_index"])


@pytest.mark.parametrize("arm", ["R7_A1", "R7_NAVOFF", "R7_VMAXOFF", "R7_BLIND", "R7_FILTOFF"])
def test_K5_undeclared_fields_are_invisible(arm):
    rng = np.random.default_rng(0)
    base = _const_accel()
    ref = _inputs(base, arm)
    decl = set(R7.declared(arm))
    for _ in range(20):
        st = copy.deepcopy(base)
        for i in range(4):
            for f in R7.FIELDS:
                if f"{f}[{i}]" in decl:
                    continue
                st[i][f] = (list(np.eye(4)[rng.integers(4)]) if f == "driving_command"
                            else list(rng.normal(size=len(st[i][f])) * 50))
        assert _inputs(st, arm) == ref


@pytest.mark.parametrize("field", ["ego_pose[2]", "ego_velocity[3]", "driving_command[3]"])
def test_K6_declared_fields_are_visible(field):
    base = _const_accel()
    st = copy.deepcopy(base)
    f, i = field[:-1].split("[")
    i = int(i)
    st[i][f] = [0, 0, 1, 0] if f == "driving_command" else [v + 3.0 for v in st[i][f]]
    assert _inputs(st, "R7_A1") != _inputs(base, "R7_A1")


# --------------------------------------------------------------------------- #
# 2. the ego window (refcv6 construction, re-read through the R7 path)           #
# --------------------------------------------------------------------------- #
def test_hist_constant_acceleration_reads_2_mps2():
    h = _hist(_const_accel(2.0, 10.0))
    np.testing.assert_allclose(h[:, 3], [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0], atol=1e-5)
    ch = ego_channels_from_poses(torch.from_numpy(h)[None], 8, dt=0.1)[0].numpy()
    np.testing.assert_allclose(ch[1:, 1], 2.0, atol=1e-3)


@pytest.mark.parametrize("times", [[-2.0, -1.5, -0.5, 0.0], [-2.0, -1.5, -1.0, 0.0]])
def test_hist_dropped_frame_amendment_A1(times):
    st = [_es(10.0 * t + t * t, 0.0, 0.0, 10.0 + 2.0 * t) for t in times]
    h = _hist(st, times=times)
    np.testing.assert_allclose(h[:, 3], [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0], atol=1e-5)


def test_hist_refuses_extrapolation():
    with pytest.raises(R7.RefusedInput):
        _hist(_const_accel(), times=[-1.0, -0.6, -0.3, 0.0])


# --------------------------------------------------------------------------- #
# 3. THE PRIOR on the NavSim window (ha0_ext_pose) — LITERALS                    #
# --------------------------------------------------------------------------- #
def _prior(st, times=TIMES):
    d = R7.declare7(st, "R7_A1")
    return R7.prior_model_free(R7.ego_history_poses(d, times), R7.v0_of(d))


def test_prior_constant_acceleration_a0_is_2():
    p = _prior(_const_accel(2.0, 10.0))
    assert p["a0"] == pytest.approx(2.0, abs=1e-4)
    assert p["kappa0"] == pytest.approx(0.0, abs=1e-9)


def test_prior_path_constant_acceleration_is_the_euler_sum():
    """a0 = 2, kappa0 = 0, v0 = 10, forward Euler at 0.1 s, speed updated LAST
    (kinematic.rollout_unicycle): x_n = 0.1 * (10 n + 2 * 0.1 * n (n - 1) / 2)."""
    p = _prior(_const_accel(2.0, 10.0))
    np.testing.assert_allclose(p["path"][:, 0], [5.2, 10.9, 17.1, 23.8, 38.7, 55.6, 74.5, 95.4],
                               atol=2e-3)
    np.testing.assert_allclose(p["path"][:, 1], 0.0, atol=1e-6)


def test_prior_path_constant_speed_is_the_cv_line():
    st = [_es(12.0 * t, 0.0, 0.0, 12.0) for t in TIMES]
    p = _prior(st)
    assert p["a0"] == pytest.approx(0.0, abs=1e-5) and p["kappa0"] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(p["path"][:, 0], [6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 60.0, 72.0],
                               atol=1e-4)


def test_prior_constant_yaw_rate_kappa_is_omega_over_v():
    st = [_es(0.0, 0.0, -0.2 * t, 8.0) for t in TIMES]          # yaw rate -0.2 rad/s at 8 m/s
    p = _prior(st)
    assert p["kappa0"] == pytest.approx(-0.025, abs=1e-5)
    assert p["a0"] == pytest.approx(0.0, abs=1e-5)
    # the Euler circle, written as its closed-form sum (phi = v * kappa * dt per tick)
    phi, v = -0.2 * 0.1, 8.0
    want = [(v * 0.1 * sum(math.cos(i * phi) for i in range(n)),
             v * 0.1 * sum(math.sin(i * phi) for i in range(n))) for n in R7.HORIZONS]
    np.testing.assert_allclose(p["path"], np.asarray(want), atol=2e-3)


def test_prior_speed_floor_and_cap():
    slow = [_es(0.0, 0.0, -0.2 * t, 1.0) for t in TIMES]        # 1 m/s < the 2 m/s floor
    assert _prior(slow)["kappa0"] == pytest.approx(-0.1, abs=1e-5)
    fast_turn = [_es(0.0, 0.0, -1.0 * t, 2.0) for t in TIMES]   # -1 rad/s at 2 m/s -> -0.5 -> cap
    assert _prior(fast_turn)["kappa0"] == pytest.approx(-0.3, abs=1e-6)


def test_prior_is_frame_invariant():
    """A rigid transform of the whole history (rotate 0.7 rad, shift) leaves (a0, kappa0) — and so
    P, which is rolled from the ego origin — unchanged: the prior reads DIFFERENCES only."""
    base = [_es(0.0, 0.0, -0.2 * t, 8.0 + 1.5 * t) for t in TIMES]
    c, s = math.cos(0.7), math.sin(0.7)
    moved = [_es(100 + c * e["ego_pose"][0] - s * e["ego_pose"][1],
                 -50 + s * e["ego_pose"][0] + c * e["ego_pose"][1], e["ego_pose"][2] + 0.7,
                 e["ego_velocity"][0]) for e in base]
    a, b = _prior(base), _prior(moved)
    assert a["a0"] == pytest.approx(b["a0"], abs=1e-5)
    assert a["kappa0"] == pytest.approx(b["kappa0"], abs=1e-6)
    np.testing.assert_allclose(a["path"], b["path"], atol=1e-4)


def test_REGRESSION_prior_on_the_wrong_grid_goes_red():
    """The defect: the 2 Hz states packed as CONSECUTIVE 0.1 s samples. The prior's a0 then reads
    (10 - 9) / 0.1 = 10 m/s^2 for a 2 m/s^2 history — the literal must NOT be met."""
    d = R7.declare7(_const_accel(2.0, 10.0), "R7_A1")
    v = [R7.speed_of(d[f"ego_velocity[{i}]"]) for i in (1, 2, 3)]
    wrong = np.zeros((8, 4), np.float32)
    wrong[:, 3] = v[0]
    wrong[-3:, 3] = v
    p = R7.prior_model_free(wrong, v[-1])
    assert p["a0"] == pytest.approx(10.0, abs=1e-3)
    assert abs(p["a0"] - 2.0) > 1.0, "the regression arm failed to go red"


# --------------------------------------------------------------------------- #
# 4. the max-speed ceiling the filter reads — LITERALS                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mps,ceil", [(11.17568, 13.888888888888889),     # 25 mph -> 50 km/h
                                      (15.6464, 27.77777777777778),       # 35 mph -> 100 km/h
                                      (6.7056, 8.333333333333334)])       # 15 mph -> 30 km/h
def test_ceiling_of_the_map_limit(mps, ceil):
    vm = R7.max_speed_input({"status": "limit", "speed_limit_mps": mps}, "R7_A1")
    assert vm["v_max_valid"] == 1.0
    assert R7.ceiling_ms(vm) == pytest.approx(ceil, abs=1e-9)


def test_ceiling_unknown_and_withheld_are_infinite():
    assert R7.ceiling_ms(R7.max_speed_input(None, "R7_A1")) == float("inf")
    assert R7.ceiling_ms(R7.max_speed_input({"status": "limit", "speed_limit_mps": 11.17568},
                                            "R7_VMAXOFF")) == float("inf")


# --------------------------------------------------------------------------- #
# 5. the forward contract: EXACTLY compute_losses_v3's keywords (AST, literal)   #
# --------------------------------------------------------------------------- #
LITERAL_KWARGS = ("nav_cmd", "v0", "steps", "lan", "ego_state", "nav_args", "v_max_ms",
                  "v_max_valid", "agent_gt", "perception_grid", "perception_valid")


def _model_call_kwargs(src: str) -> tuple:
    """The keywords of the ``model(frames, ...)`` call inside ``compute_losses_v3``."""
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "compute_losses_v3")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "model"]
    assert len(calls) == 1, f"{len(calls)} model(...) calls in compute_losses_v3"
    return tuple(k.arg for k in calls[0].keywords)


def _set_ego_window_kwargs(src: str) -> tuple:
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "compute_losses_v3")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "set_ego_window"]
    assert len(calls) == 1
    return len(calls[0].args), tuple(k.arg for k in calls[0].keywords)


def test_forward_kwargs_are_the_trainers():
    src = open(TRAINER, encoding="utf-8").read()
    assert _model_call_kwargs(src) == LITERAL_KWARGS
    assert R7.FORWARD_KWARGS == LITERAL_KWARGS
    assert _set_ego_window_kwargs(src) == (2, ("actions",))


def test_forward_kwargs7_emits_exactly_the_contract():
    g = torch.zeros(4, 400, 240, 2)
    v = torch.ones(4, 400, 240, dtype=torch.bool)
    kw = R7.forward_kwargs7(None, {"nav_index": 2}, 9.5, {"v_max_ms": 11.2, "v_max_valid": 1.0},
                            g, v, 2, "cpu")
    assert tuple(kw) == LITERAL_KWARGS
    assert int(kw["nav_cmd"][0]) == 2 and float(kw["v0"][0]) == pytest.approx(9.5)
    assert kw["lan"] is None and kw["ego_state"] is None and kw["nav_args"] is None
    assert kw["agent_gt"] is None and tuple(kw["perception_grid"].shape) == (1, 4, 400, 240, 2)
    kw0 = R7.forward_kwargs7(None, {"nav_index": None}, 1.0, {"v_max_ms": 0.0, "v_max_valid": 0.0},
                             g, v, 2, "cpu")
    assert kw0["nav_cmd"] is None and float(kw0["v_max_valid"][0]) == 0.0


def test_REGRESSION_a_trainer_that_adds_a_keyword_goes_red():
    src = open(TRAINER, encoding="utf-8").read()
    mutated = src.replace("perception_grid=_pgrid, perception_valid=_pvalid)",
                          "perception_grid=_pgrid, perception_valid=_pvalid, new_input=x)", 1)
    assert mutated != src, "the mutation did not apply -- the trainer's call site moved"
    assert _model_call_kwargs(mutated) != LITERAL_KWARGS


# --------------------------------------------------------------------------- #
# 6. the 0.25 m lift: the model's bank, a NavSim rig through the SAME call        #
# --------------------------------------------------------------------------- #
def _level_rig(cam_z_ego=1.5, cam_x=1.7, pitch_down=0.0):
    fwd = np.array([math.cos(pitch_down), 0.0, -math.sin(pitch_down)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return {"rig_key": "synthetic", "f0_sensor2lidar_translation": [cam_x, 0.0, cam_z_ego],
            "virtual_R_cam_to_lidar": np.stack([right, down, fwd], axis=1).tolist()}


@pytest.fixture(scope="module")
def hbank():
    """A 0.25 m bank built EXACTLY as train() builds ``model._lift_bank_hires``
    (``HiresLiftGeometryBank(table, frame, cfg=MapHiresConfig(<the run's extent>),
    equalize_bottom_rows=43)``) over a table holding one synthetic clip = the NavSim level rig."""
    from tanitad.models import map_head_hires as mhr
    from tanitad.models.trunk_shapes import FRAME_416x1024
    cam = R7.RIG6.rig_camera(_level_rig(), -0.35, FRAME_416x1024)
    cfg = mhr.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0)
    b = mhr.HiresLiftGeometryBank({"synthetic-clip": cam}, frame=FRAME_416x1024, cfg=cfg,
                                  equalize_bottom_rows=43)
    return b, cam


def test_bank_params_are_the_runs(hbank):
    b, _ = hbank
    p = R7.bank_params(b)
    assert p["stride"] == 8 and p["heights_m"] == (0.0, 0.5, 1.5, 2.5)
    assert (p["grid"].x_fwd_m, p["grid"].y_half_m, p["grid"].cell_m) == (100.0, 30.0, 0.25)
    assert p["equalize_bottom_rows"] == 43 and int((~p["observed"]).sum()) == 43 * 1024


def test_KL_lift_navsim_camera_through_the_banks_own_call(hbank):
    """The bridge's geometry for a camera == the bank's own ``geometry(ep)`` for a clip holding
    the same camera: bit-exact, grid AND valid."""
    from tanitad.data.v2_dataset import stable_episode_id
    b, cam = hbank
    g0, v0 = b.geometry(int(stable_episode_id("synthetic-clip")))
    g1, v1 = R7.lift_geometry_for_camera(cam, b)
    assert torch.equal(g0, g1) and torch.equal(v0, v1)
    g2, v2, s = R7.hires_lift_for_rig(_level_rig(), -0.35, b)
    assert torch.equal(g0, g2) and torch.equal(v0, v2)
    assert tuple(g2.shape) == (4, 400, 240, 2)
    assert s["cam_height_m"] == pytest.approx(1.85, abs=1e-12)


def test_lift_bottom_43_rows_are_unobserved_analytic(hbank):
    """Level camera 1.85 m above the road, x 1.7 m: a road point d ahead lands on row
    207.5 + f * 1.85 / (d - 1.7). It leaves the image (row > 415) for d < 1.7 + 4.34 m and
    enters the masked 43 rows (row >= 373) for d < 1.7 + 5.465 m. So on the centre line at
    z = 0: 6.25 <= x < 7.1 m is IMAGED but MASKED (invalid), x >= 7.5 m valid."""
    b, _ = hbank
    v = b.geometry(int(__import__("tanitad.data.v2_dataset", fromlist=["x"])
                       .stable_episode_id("synthetic-clip")))[1]
    cols = (119, 120)                                      # y = -0.125 / +0.125 m
    xi = lambda x: int(x / 0.25)                           # noqa: E731  cell i covers [0.25 i, 0.25 (i+1))
    for x in (6.3, 6.6, 6.9):
        assert not bool(v[0, xi(x), cols[0]]) and not bool(v[0, xi(x), cols[1]]), x
    for x in (7.6, 10.0, 25.0):
        assert bool(v[0, xi(x), cols[0]]) and bool(v[0, xi(x), cols[1]]), x


def test_REGRESSION_lift_without_the_43_row_mask_goes_red(hbank):
    from tanitad.models import map_head_hires as mhr
    from tanitad.models.trunk_shapes import FRAME_416x1024
    b, cam = hbank
    nb = mhr.HiresLiftGeometryBank({"synthetic-clip": cam}, frame=FRAME_416x1024,
                                   cfg=mhr.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0,
                                                          y_half_m=30.0),
                                   equalize_bottom_rows=0)
    v = R7.lift_geometry_for_camera(cam, nb)[1]
    assert bool(v[0, int(6.6 / 0.25), 119]), "the unmasked lift must see the masked rows"
    assert not torch.equal(v, R7.lift_geometry_for_camera(cam, b)[1])


def test_REGRESSION_the_stride16_lift_goes_red(hbank):
    """Feeding refcv6's stride-16 geometry (rig6.lift_geometry) where refcv7's 0.25 m lift is
    expected: a different tensor, never silently accepted."""
    b, _ = hbank
    g16, v16, _ = R7.RIG6.lift_geometry(_level_rig(), -0.35, stride=16)
    g8 = R7.hires_lift_for_rig(_level_rig(), -0.35, b)[0]
    assert tuple(g16.shape) != tuple(g8.shape)


def test_REGRESSION_road_offset_ignored_goes_red(hbank):
    b, _ = hbank
    good = R7.hires_lift_for_rig(_level_rig(), -0.35, b)
    bad = R7.hires_lift_for_rig(_level_rig(), 0.0, b)
    assert good[2]["cam_height_m"] - bad[2]["cam_height_m"] == pytest.approx(0.35, abs=1e-12)
    assert not torch.equal(good[0], bad[0])
