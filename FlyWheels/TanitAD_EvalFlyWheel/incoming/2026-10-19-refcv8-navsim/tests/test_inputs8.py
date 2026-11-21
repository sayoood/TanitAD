"""refcv8's NavSim inputs, each against a LITERAL or ANALYTIC target, each with an arm that must go RED.  TANITAD VENV, CPU, no
checkpoint:   pytest -q tests/test_inputs8.py        (tests/test_mutation8.py proves the RED arms in fresh interpreters)

Expectations are LITERALS (or closed-form sums written here, independent of the code under test) -- CLAUDE.md: "a check that
shares the defect it checks for is green forever".  Scope of this file: the FEED (command -> token, r8_nav, speed row, ego window,
prior, forward contract, lift, re-timing, seam format, re-stitch geometry).  The model forward on a real checkpoint is
``tests/test_model_seam8.py``.
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "code"))
import refcv8_bridge as B  # noqa: E402
import torch  # noqa: E402
from tanitad.models.ego_history import ego_channels_from_poses  # noqa: E402

TREE = B.boot8.TREE
TRAINER = os.path.join(TREE, "stack", "scripts", "refc_v3_train.py")
R8TRAIN = os.path.join(TREE, "stack", "tanitad", "train", "refcv8_train.py")
V9BUILD = os.path.join(TREE, "stack", "scripts", "build_v9_labels.py")
TIMES = [-1.5, -1.0, -0.5, 0.0]


def _es(x, y, h, vx, vy=0.0, ax=0.0, ay=0.0, cmd=(0, 1, 0, 0)):
    return {"ego_pose": [x, y, h], "ego_velocity": [vx, vy], "ego_acceleration": [ax, ay], "driving_command": list(cmd)}


def _const_accel(a=2.0, v_t0=10.0, times=TIMES, cmd=(0, 1, 0, 0)):
    return [_es(v_t0 * t + 0.5 * a * t * t, 0.0, 0.0, v_t0 + a * t, cmd=cmd) for t in times]


def _src(path):
    return open(path, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# 0. the tree under test                                                        #
# --------------------------------------------------------------------------- #
def test_the_tree_is_the_launch_archive():
    import tanitad
    t = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
    assert t.startswith(TREE.lower() + "/stack/"), t
    assert os.path.abspath(B.R6.__file__).replace("\\", "/").lower().startswith(TREE.lower())


# --------------------------------------------------------------------------- #
# 1. arms -> the refcv6 enforcement templates (LITERAL)                          #
# --------------------------------------------------------------------------- #
HIST = ("ego_pose[1]", "ego_pose[2]", "ego_pose[3]", "ego_velocity[1]", "ego_velocity[2]", "ego_velocity[3]")
EXPECT = {  # arm -> (declared, frames, nav, speed, seed)
    "R8_LEGAL": (HIST + ("driving_command[3]",), "ST", "cmd", "legal", 0),
    "R8_LEGAL_s1": (HIST + ("driving_command[3]",), "ST", "cmd", "legal", 1),
    "R8_PRIV": (HIST + ("driving_command[3]",), "ST", "cmd", "map", 0),
    "R8_NT": (HIST + ("driving_command[3]",), "NT", "cmd", "legal", 0),
    "R8_BLIND": (HIST + ("driving_command[3]",), "BLIND", "cmd", "legal", 0),
    "R8_NAVOFF": (HIST, "ST", None, "legal", 0),
}


@pytest.mark.parametrize("arm", sorted(EXPECT))
def test_arm_specs_are_the_literal(arm):
    d, fr, nav, sp, sd = EXPECT[arm]
    s = B.ARMS8[arm]
    assert tuple(sorted(B.declared(arm))) == tuple(sorted(d))
    assert (s["frames"], s["nav"], s["speed"], s["seed"]) == (fr, nav, sp, sd)
    assert sorted(B.ARMS8) == sorted(EXPECT)


def test_no_arm_declares_acceleration_or_curvature():
    """a0 and yaw rate are derived INSIDE the forward from the pose window (ha0_ext_pose); no field carries them."""
    for arm in B.ARMS8:
        assert not any(x.startswith("ego_acceleration") for x in B.declared(arm))


def test_REGRESSION_a_mismatched_template_is_refused(monkeypatch):
    monkeypatch.setitem(B.R6_TEMPLATE, "R8_NAVOFF", "R6_A1")        # would feed the command
    with pytest.raises(B.RefusedInput):
        B.declare8(_const_accel(), "R8_NAVOFF")


def _inputs(st, arm):
    d = B.declare8(st, arm)
    return (B.ego_history_poses(d, TIMES).tobytes(), B.v0_of(d), B.nav_input8(d, arm)["nav_index"],
            B.r8_nav_row().numpy().tobytes(), B.speed_input8({"status": "limit", "speed_limit_mps": 11.17568}, arm)["v_max_ms"])


@pytest.mark.parametrize("arm", ["R8_LEGAL", "R8_NAVOFF", "R8_PRIV", "R8_BLIND"])
def test_K5_undeclared_fields_are_invisible(arm):
    """Every UNDECLARED field -- including the driving command of the 3 EARLIER frames (the clip-level-token defect of
    R8-EVAL-NAV-FEED-1 could only enter through them) -- is randomised; the model inputs must be byte-identical."""
    rng = np.random.default_rng(0)
    base = _const_accel()
    ref = _inputs(base, arm)
    decl = set(B.declared(arm))
    for _ in range(20):
        st = copy.deepcopy(base)
        for i in range(4):
            for f in B.FIELDS:
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
    st[i][f] = [1, 0, 0, 0] if f == "driving_command" else [v + 3.0 for v in st[i][f]]
    assert _inputs(st, "R8_LEGAL") != _inputs(base, "R8_LEGAL")


# --------------------------------------------------------------------------- #
# 2. the NAV TOKEN: command -> {FOLLOW 0, TURN_L 1, TURN_R 2}, LITERAL           #
# --------------------------------------------------------------------------- #
ONEHOT_EXPECT = [([1, 0, 0, 0], 1, "TURN_L"),      # NavSim left      -> TURN_L
                 ([0, 1, 0, 0], 0, "FOLLOW"),      # NavSim straight  -> FOLLOW (NOT 'straight' = 3: the trainer never feeds 3)
                 ([0, 0, 1, 0], 2, "TURN_R"),      # NavSim right     -> TURN_R
                 ([0, 0, 0, 1], 0, "FOLLOW")]      # NavSim unknown   -> FOLLOW (the LEGAL fallback)


@pytest.mark.parametrize("cmd,tok,name", ONEHOT_EXPECT)
def test_nav_command_to_token_literal(cmd, tok, name):
    st = _const_accel(cmd=tuple(cmd))
    n = B.nav_input8(B.declare8(st, "R8_LEGAL"), "R8_LEGAL")
    assert (n["nav_index"], n["name"]) == (tok, name)
    assert n["navsim_argmax"] == cmd.index(1)


@pytest.mark.parametrize("bad", [[1, 1, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]])
def test_non_one_hot_command_is_refused(bad):
    st = _const_accel(cmd=tuple(bad))
    with pytest.raises(B.RefusedInput):
        B.nav_input8(B.declare8(st, "R8_LEGAL"), "R8_LEGAL")


def test_navoff_withholds_the_token():
    n = B.nav_input8(B.declare8(_const_accel(), "R8_NAVOFF"), "R8_NAVOFF")
    assert n["nav_index"] is None


def test_the_token_is_the_t0_command_not_an_earlier_one():
    """Time-local by construction: commands that DIFFER across the 4 history frames -- the token follows frame 3 (t0) only."""
    st = _const_accel()
    for i, c in enumerate(([1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0])):
        st[i]["driving_command"] = c
    assert B.nav_input8(B.declare8(st, "R8_LEGAL"), "R8_LEGAL")["nav_index"] == 0           # t0 = straight -> FOLLOW
    st[3]["driving_command"] = [0, 0, 1, 0]
    assert B.nav_input8(B.declare8(st, "R8_LEGAL"), "R8_LEGAL")["nav_index"] == 2           # t0 = right -> TURN_R


def test_vocabulary_source_lines_are_pinned():
    """The three source lines the mapping rests on, read from THIS tree's files (a literal substring each)."""
    assert "NAV_FOLLOW, NAV_TURN_L, NAV_TURN_R = 0, 1, 2" in _src(V9BUILD)                         # build_v9_labels.py:260
    assert "R8_V9_NAV_TO_NAV_CMD = {0: 0, 1: 1, 2: 2}" in _src(TRAINER)                            # refc_v3_train.py:4686
    assert 'NAVSIM_LEGAL_NAV_CMD = {"left": 1, "forward": 0, "right": 2, "unknown": 0}' in _src(R8TRAIN)   # refcv8_train.py:71-72
    from tanitad.refs import refb
    assert tuple(refb.NAV_COMMANDS[:3]) == ("follow", "left", "right")
    from tanitad.train import refcv8_train
    assert dict(refcv8_train.NAVSIM_LEGAL_NAV_CMD) == B.NAVSIM_LEGAL_NAV_CMD == {"left": 1, "forward": 0, "right": 2, "unknown": 0}
    assert B.NAV_NAMES8 == ("FOLLOW", "TURN_L", "TURN_R")


@pytest.mark.parametrize("cmd", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
def test_token_agrees_with_E2s_independent_route(cmd):
    """The refcv7 path (E2's ``nav_index_from_command`` -> ``refb.NAV_COMMANDS.index``) and this port's trainer-derived map are two
    independent derivations; they must agree on every one-hot."""
    e2_idx, _, _ = B.B2.nav_index_from_command(cmd)
    mine = B.nav_input8(B.declare8(_const_accel(cmd=tuple(cmd)), "R8_LEGAL"), "R8_LEGAL")["nav_index"]
    assert mine == e2_idx


# --------------------------------------------------------------------------- #
# 3. r8_nav = the UNKNOWN row, exactly as training's dropout path builds it      #
# --------------------------------------------------------------------------- #
def test_r8_nav_row_is_exactly_zero():
    row = B.r8_nav_row()
    assert tuple(row.shape) == (1, 6) and row.dtype == torch.float32
    assert torch.equal(row, torch.zeros(1, 6, dtype=torch.float32))


def test_unknown_row_equals_the_dropout_path_even_with_nonzero_args():
    """``refcv8_train.r8_before_forward`` (:560-567): ``known &= ~(gen.rand((b,)) < p)`` then ``scale_nav(nav, known)``.  Replayed with
    p = 1.0 (every row dropped) on a row whose v9 args are NON-zero: the output is the bridge's row, bit for bit."""
    from tanitad.train import refcv8_train as tr
    src = _src(R8TRAIN)
    assert 'fwd["r8_nav"] = scale_nav(nav, known)' in src and "known &= ~(gen.rand((b,), dev) <" in src
    assert "known = torch.zeros_like(known)" in src
    g = torch.Generator().manual_seed(0)
    raw = torch.tensor([[40.0, 120.0, 90.0, 1.0, 150.0]])
    known = torch.ones(1, dtype=torch.bool)
    known &= ~(torch.rand((1,), generator=g) < 1.0)
    assert not bool(known.any())
    assert torch.equal(tr.scale_nav(raw, known), B.r8_nav_row())


def test_scale_nav_known_row_value():
    """A KNOWN row, closed form: (log1p(d_next/50), log1p(d_end/50), dyaw/90, side, log1p(lookahead/50), 1)."""
    from tanitad.train import refcv8_train as tr
    out = tr.scale_nav(torch.tensor([[40.0, 120.0, 90.0, 1.0, 150.0]]), torch.ones(1, dtype=torch.bool))[0].tolist()
    want = [math.log1p(40 / 50), math.log1p(120 / 50), 1.0, 1.0, math.log1p(150 / 50), 1.0]
    np.testing.assert_allclose(out, want, atol=1e-6)


def test_r8_rc_is_not_fed():
    """``--r8-no-rc``: the trainer never writes ``fwd['r8_rc']``; the bridge's keyword set has no r8_rc either."""
    assert "r8_rc" not in B.FORWARD_KWARGS


# --------------------------------------------------------------------------- #
# 4. the speed input: LEGAL = the trained unknown row; PRIV = the containing bin  #
# --------------------------------------------------------------------------- #
def test_legal_speed_is_the_unknown_row():
    for rec in (None, {"status": "limit", "speed_limit_mps": 11.17568}, {"status": "none"}):
        v = B.speed_input8(rec, "R8_LEGAL")
        assert (v["v_max_ms"], v["v_max_valid"]) == (0.0, 0.0)
        assert B.ceiling_ms(v) == float("inf")


@pytest.mark.parametrize("mps,kmh", [(11.17568, 50.0),     # 25 mph = 40.2 km/h -> the 50 bin
                                     (15.6464, 100.0),     # 35 mph = 56.3 km/h -> the 100 bin
                                     (6.7056, 30.0)])      # 15 mph = 24.1 km/h -> the 30 bin
def test_priv_speed_snaps_to_the_containing_bin(mps, kmh):
    v = B.speed_input8({"status": "limit", "speed_limit_mps": mps}, "R8_PRIV")
    assert v["v_max_valid"] == 1.0 and v["bin_kmh"] == kmh
    assert v["v_max_ms"] == pytest.approx(np.float32(kmh / 3.6), abs=0)       # the Python-float64 quotient, cast once
    assert B.ceiling_ms(v) == pytest.approx(kmh / 3.6, abs=1e-9)


def test_scalar_float64_ceiling_rule_misreads_the_snapped_value():
    """The trap ``ceiling_ms`` avoids: the float32 cast of 50/3.6 read through the float64 SCALAR rule lands in the 100 bin."""
    from tanitad.refs.refcv6_max_speed import SPEED_MAX_STEPS_MS_V6, speed_max_bin
    v32 = float(np.float32(50.0 / 3.6))
    assert v32 > 50.0 / 3.6 + 1e-9 and speed_max_bin(v32)[0] == 2 and SPEED_MAX_STEPS_MS_V6[2] == pytest.approx(100 / 3.6)
    assert B.ceiling_ms({"v_max_ms": v32, "v_max_valid": 1.0}) == pytest.approx(50.0 / 3.6, abs=1e-9)


def test_priv_without_a_limit_is_the_unknown_row():
    for rec in (None, {"status": "none"}):
        v = B.speed_input8(rec, "R8_PRIV")
        assert (v["v_max_ms"], v["v_max_valid"]) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# 5. the ego window and THE PRIOR on the NavSim window (ha0_ext_pose) -- LITERALS  #
# --------------------------------------------------------------------------- #
def _hist(st, arm="R8_LEGAL", times=TIMES):
    return B.ego_history_poses(B.declare8(st, arm), times)


def _prior(st, times=TIMES):
    d = B.declare8(st, "R8_LEGAL")
    return B.prior_model_free(B.ego_history_poses(d, times), B.v0_of(d))


def test_hist_constant_acceleration_reads_2_mps2():
    h = _hist(_const_accel(2.0, 10.0))
    np.testing.assert_allclose(h[:, 3], [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0], atol=1e-5)
    ch = ego_channels_from_poses(torch.from_numpy(h)[None], 8, dt=0.1)[0].numpy()
    np.testing.assert_allclose(ch[1:, 1], 2.0, atol=1e-3)


def test_hist_refuses_extrapolation():
    with pytest.raises(B.RefusedInput):
        _hist(_const_accel(), times=[-1.0, -0.6, -0.3, 0.0])


def test_prior_constant_acceleration_a0_is_2():
    p = _prior(_const_accel(2.0, 10.0))
    assert p["a0"] == pytest.approx(2.0, abs=1e-4)
    assert p["kappa0"] == pytest.approx(0.0, abs=1e-9)


def test_prior_path_constant_acceleration_is_the_euler_sum():
    """a0 = 2, kappa0 = 0, v0 = 10, forward Euler at 0.1 s, speed updated LAST: x_n = 0.1 * (10 n + 2 * 0.1 * n (n - 1) / 2)."""
    p = _prior(_const_accel(2.0, 10.0))
    np.testing.assert_allclose(p["path"][:, 0], [5.2, 10.9, 17.1, 23.8, 38.7, 55.6, 74.5, 95.4], atol=2e-3)
    np.testing.assert_allclose(p["path"][:, 1], 0.0, atol=1e-6)


def test_straight_constant_speed_history_gives_the_cv_line():
    """THE straight constant-speed known value: 12 m/s, zero yaw rate -> a0 = kappa0 = 0 and the prior is the constant-velocity line
    at the 8 knot instants (0.5 ... 6 s): 6, 12, 18, 24, 36, 48, 60, 72 m."""
    st = [_es(12.0 * t, 0.0, 0.0, 12.0) for t in TIMES]
    p = _prior(st)
    assert p["a0"] == pytest.approx(0.0, abs=1e-5) and p["kappa0"] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(p["path"][:, 0], [6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 60.0, 72.0], atol=1e-4)
    np.testing.assert_allclose(p["path"][:, 1], 0.0, atol=1e-6)


def test_prior_constant_yaw_rate_kappa_is_omega_over_v():
    st = [_es(0.0, 0.0, -0.2 * t, 8.0) for t in TIMES]                 # yaw rate -0.2 rad/s at 8 m/s
    p = _prior(st)
    assert p["kappa0"] == pytest.approx(-0.025, abs=1e-5)
    phi, v = -0.2 * 0.1, 8.0
    want = [(v * 0.1 * sum(math.cos(i * phi) for i in range(n)), v * 0.1 * sum(math.sin(i * phi) for i in range(n)))
            for n in B.HORIZONS]
    np.testing.assert_allclose(p["path"], np.asarray(want), atol=2e-3)


def test_REGRESSION_prior_on_the_wrong_grid_goes_red():
    """The defect: the 2 Hz states packed as CONSECUTIVE 0.1 s samples -> a0 reads 10 m/s^2 for a 2 m/s^2 history."""
    d = B.declare8(_const_accel(2.0, 10.0), "R8_LEGAL")
    v = [B.speed_of(d[f"ego_velocity[{i}]"]) for i in (1, 2, 3)]
    wrong = np.zeros((8, 4), np.float32)
    wrong[:, 3] = v[0]
    wrong[-3:, 3] = v
    p = B.prior_model_free(wrong, v[-1])
    assert p["a0"] == pytest.approx(10.0, abs=1e-3) and abs(p["a0"] - 2.0) > 1.0


# --------------------------------------------------------------------------- #
# 6. the forward contract: the trainer's keywords (AST, literal)                 #
# --------------------------------------------------------------------------- #
LITERAL_FWD = ("nav_cmd", "v0", "steps", "lan", "ego_state", "nav_args", "v_max_ms", "v_max_valid", "agent_gt",
               "perception_grid", "perception_valid")


def _fn(src, name):
    return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)


def _fwd_call_kwargs(src):
    calls = [n for n in ast.walk(_fn(src, "compute_losses_v3")) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_fwd"]
    assert len(calls) == 1, f"{len(calls)} _fwd(...) calls in compute_losses_v3"
    return tuple(k.arg for k in calls[0].keywords)


def _fwd_dict_keys(src):
    """Every ``fwd["<key>"] = ...`` store inside ``r8_before_forward`` (the dict merged OVER the call's keywords)."""
    return {n.slice.value for n in ast.walk(_fn(src, "r8_before_forward"))
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name)
            and n.value.id == "fwd" and isinstance(n.slice, ast.Constant)}


def _set_ego_window_kwargs(src):
    calls = [n for n in ast.walk(_fn(src, "compute_losses_v3")) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "set_ego_window"]
    assert len(calls) == 1
    return len(calls[0].args), tuple(k.arg for k in calls[0].keywords)


def test_forward_kwargs_are_the_trainers():
    src = _src(TRAINER)
    assert _fwd_call_kwargs(src) == LITERAL_FWD
    # r8_before_forward adds r8_nav (and r8_rc, which --r8-no-rc suppresses) and REPLACES v_max_ms / v_max_valid
    assert _fwd_dict_keys(_src(R8TRAIN)) == {"r8_nav", "r8_rc", "v_max_ms", "v_max_valid"}
    assert B.FORWARD_KWARGS == LITERAL_FWD + ("r8_nav",)
    assert _set_ego_window_kwargs(src) == (2, ("actions",))


def test_forward_kwargs8_emits_exactly_the_contract():
    g = torch.zeros(4, 400, 240, 2)
    v = torch.ones(4, 400, 240, dtype=torch.bool)
    nav = B.nav_input8(B.declare8(_const_accel(cmd=(1, 0, 0, 0)), "R8_LEGAL"), "R8_LEGAL")
    kw = B.forward_kwargs8(None, nav, 9.5, B.speed_input8(None, "R8_LEGAL"), g, v, 2, "cpu")
    assert tuple(kw) == B.FORWARD_KWARGS
    assert int(kw["nav_cmd"][0]) == 1 and float(kw["v0"][0]) == pytest.approx(9.5)
    assert kw["lan"] is None and kw["ego_state"] is None and kw["nav_args"] is None and kw["agent_gt"] is None
    assert float(kw["v_max_ms"][0]) == 0.0 and float(kw["v_max_valid"][0]) == 0.0
    assert tuple(kw["perception_grid"].shape) == (1, 4, 400, 240, 2)
    assert torch.equal(kw["r8_nav"], torch.zeros(1, 6)) and kw["r8_nav"].dtype == torch.float32
    kw0 = B.forward_kwargs8(None, {"nav_index": None}, 1.0, B.speed_input8(None, "R8_LEGAL"), g, v, 2, "cpu")
    assert kw0["nav_cmd"] is None and torch.equal(kw0["r8_nav"], torch.zeros(1, 6))


def test_REGRESSION_a_trainer_that_adds_a_keyword_goes_red():
    src = _src(TRAINER)
    mutated = src.replace("perception_grid=_pgrid, perception_valid=_pvalid)",
                          "perception_grid=_pgrid, perception_valid=_pvalid, new_input=x)", 1)
    assert mutated != src, "the mutation did not apply -- the trainer's call site moved"
    assert _fwd_call_kwargs(mutated) != LITERAL_FWD


def test_REGRESSION_a_trainer_that_adds_an_r8_key_goes_red():
    src = _src(R8TRAIN)
    mutated = src.replace('fwd["r8_nav"] = scale_nav(nav, known)', 'fwd["r8_nav"] = scale_nav(nav, known)\n        fwd["r8_new"] = nav', 1)
    assert mutated != src
    assert _fwd_dict_keys(mutated) != {"r8_nav", "r8_rc", "v_max_ms", "v_max_valid"}


# --------------------------------------------------------------------------- #
# 7. the 0.25 m lift: the model's bank, a NavSim rig through the SAME call        #
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
    """A 0.25 m bank built EXACTLY as train() builds ``model._lift_bank_hires`` over one synthetic clip = the NavSim level rig."""
    from tanitad.models import map_head_hires as mhr
    from tanitad.models.trunk_shapes import FRAME_416x1024
    cam = B.RIG6.rig_camera(_level_rig(), -0.35, FRAME_416x1024)
    cfg = mhr.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0)
    b = mhr.HiresLiftGeometryBank({"synthetic-clip": cam}, frame=FRAME_416x1024, cfg=cfg, equalize_bottom_rows=43)
    return b, cam


def test_bank_params_are_the_runs(hbank):
    b, _ = hbank
    p = B.bank_params(b)
    assert p["stride"] == 8 and p["heights_m"] == (0.0, 0.5, 1.5, 2.5)
    assert (p["grid"].x_fwd_m, p["grid"].y_half_m, p["grid"].cell_m) == (100.0, 30.0, 0.25)
    assert p["equalize_bottom_rows"] == 43 and int((~p["observed"]).sum()) == 43 * 1024


def test_KL_lift_navsim_camera_through_the_banks_own_call(hbank):
    from tanitad.data.v2_dataset import stable_episode_id
    b, cam = hbank
    g0, v0 = b.geometry(int(stable_episode_id("synthetic-clip")))
    g1, v1 = B.lift_geometry_for_camera(cam, b)
    assert torch.equal(g0, g1) and torch.equal(v0, v1)
    g2, v2, s = B.hires_lift_for_rig(_level_rig(), -0.35, b)
    assert torch.equal(g0, g2) and torch.equal(v0, v2)
    assert tuple(g2.shape) == (4, 400, 240, 2)
    assert s["cam_height_m"] == pytest.approx(1.85, abs=1e-12)


def test_REGRESSION_the_stride16_lift_goes_red(hbank):
    b, _ = hbank
    g16, _, _ = B.RIG6.lift_geometry(_level_rig(), -0.35, stride=16)
    assert tuple(g16.shape) != tuple(B.hires_lift_for_rig(_level_rig(), -0.35, b)[0].shape)


# --------------------------------------------------------------------------- #
# 8. the frame contract and the RE-STITCH geometry                               #
# --------------------------------------------------------------------------- #
def test_frame_contract_is_cylindrical_488_924():
    """416x1024 CYLINDRICAL, f_ref = 305.5774907364391 * 1024/640 = 488.924 px, hfov 120.000 deg.  The PINHOLE formula
    2*atan((W/2)/f) would give 92.6 deg and look plausible -- the trap this test documents."""
    from tanitad.data.calib import cylindrical_rays
    from tanitad.models.trunk_shapes import FRAME_416x1024 as FR
    assert (FR.height, FR.width, FR.projection) == (416, 1024, "cylindrical")
    assert FR.f_ref == pytest.approx(305.5774907364391 * 1024 / 640, abs=1e-9) and FR.f_ref == pytest.approx(488.924, abs=1e-3)
    assert FR.hfov_deg == pytest.approx(120.0, abs=1e-9)
    assert math.degrees(2 * math.atan((1024 / 2) / FR.f_ref)) == pytest.approx(92.6, abs=0.1)       # the WRONG formula
    xc, _, zc = (t.double().numpy() for t in cylindrical_rays(FR))
    az = np.degrees(np.arctan2(xc[208], zc[208]))
    u = (FR.width - 1) / 2 + np.radians(az) * FR.f_ref
    assert float(np.abs(u - np.arange(FR.width)).max()) < 1e-3                                      # column LINEAR in azimuth


WARMUP_BANK = "C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/warmup_two_stage"
NAVSIM_ROOT = "D:/Archive/devbox-C/navsim/data/openscene"


@pytest.mark.skipif(not (os.path.isdir(WARMUP_BANK) and os.path.isdir(f"{NAVSIM_ROOT}/warmup_two_stage/sensor_blobs")),
                    reason="the NavSim warmup raw camera blobs / 416 bank are not on this box")
def test_restitch_reproduces_the_landed_smoke_geometry():
    """The landed ``camera_smoke_r8.py`` S2 on ONE warmup scene (the first key of the bank, as it picks): re-stitching its raw
    CAM_L0/F0/R0 jpgs through ``frames416.build_one`` is BIT-IDENTICAL to the bank the refcv7 suite fed its model -- sha16
    ``d44e7bc225ec67a7`` (NAVSIM_EVAL_PLAN.md section 4, MEASURED), max |d| 0 px, observed fraction 0.8817."""
    import hashlib
    import pathlib

    import pandas as pd
    import frames416 as F4
    from tanitad.models.trunk_shapes import FRAME_416x1024
    F4.configure(FRAME_416x1024)
    wb = pathlib.Path(WARMUP_BANK)
    prov = pd.read_parquet(wb / "frames_provenance.parquet")
    key = sorted(prov.scene_token)[0]
    sc = F4.E2BF.load(pathlib.Path(NAVSIM_ROOT) / "warmup_two_stage" / "synthetic_scene_pickles" / f"{key}.pkl")
    cams = [fr["camera_dict"] for fr in sc["frames"]]
    keep = int(json.load(open(wb / "BUILD_REPORT.json", encoding="utf-8"))["keep"])
    arr, src, rk, obs, _ = F4.build_one(cams, pathlib.Path(NAVSIM_ROOT) / "warmup_two_stage" / "sensor_blobs", {}, keep)
    assert arr.shape[1:] == (416, 1024, 3) and arr.dtype == np.uint8
    sha = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    assert sha == "d44e7bc225ec67a7"
    banked = np.load(wb / "frames" / f"{key}.npy")
    assert int(np.abs(arr.astype(np.int16) - banked.astype(np.int16)).max()) == 0
    assert obs == pytest.approx(0.8817, abs=5e-4)


# --------------------------------------------------------------------------- #
# 9. the output: 8 knots -> 8 NavSim poses, and the RE-TIMING stage               #
# --------------------------------------------------------------------------- #
def test_knots_to_navsim_resampling_literal():
    """Constant velocity 10 m/s straight: a not-a-knot cubic spline reproduces a LINE exactly, so the poses at 0.5 ... 4.0 s are
    5, 10, ..., 40 m (2.5 s and 3.5 s are the interpolated ones: 25, 35), y = 0, heading 0."""
    knots = np.stack([10.0 * np.array(B.KNOT_T_S), np.zeros(8)], -1)
    poses = B.knots_to_navsim(knots)
    assert poses.shape == (8, 3)
    np.testing.assert_allclose(poses[:, 0], [5, 10, 15, 20, 25, 30, 35, 40], atol=1e-5)
    np.testing.assert_allclose(poses[:, 1:], 0.0, atol=1e-6)


def _stage_and_table(tmp_path, gains):
    import hashlib
    p = tmp_path / "gains.json"
    p.write_text(json.dumps({"name": "TEST", "knots_s": list(B.KNOT_T_S), "gains": list(gains)}), encoding="utf-8")
    (tmp_path / "gains.json.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "  gains.json\n", encoding="utf-8")
    return p


def test_retime_has_no_default_table():
    with pytest.raises(B.RefusedInput):
        B.load_retime("", "")


def test_retime_refuses_a_table_whose_sha_sidecar_disagrees(tmp_path):
    p = _stage_and_table(tmp_path, [1.0] * 8)
    (tmp_path / "gains.json.sha256").write_text("0" * 64 + "  gains.json\n", encoding="utf-8")
    with pytest.raises(Exception, match="sha256"):
        B.load_retime("", str(p))


def test_retime_gain_one_is_the_identity_and_gain_half_is_the_midpoint(tmp_path):
    """Straight plan at 10 m/s, prior at 5 m/s along the same line.  g = 1: the plan, bit for bit.  g = 0.5: s' = L + 0.5 (M - L) =
    7.5 t -> 3.75, 7.5, 11.25, 15, 22.5, 30, 37.5, 45 m on the plan's own line.  g = 0: the prior's arc length, 5 t."""
    t = np.array(B.KNOT_T_S)
    plan = np.stack([10.0 * t, np.zeros(8)], -1).astype(np.float32)
    prior = np.stack([5.0 * t, np.zeros(8)], -1).astype(np.float32)
    for g, want in ((1.0, 10.0 * t), (0.5, 7.5 * t), (0.0, 5.0 * t)):
        d = tmp_path / f"g{g}"
        d.mkdir()
        RS, table, rec = B.load_retime("", str(_stage_and_table(d, [g] * 8)))
        out = B.retime_plan8(RS, table, plan, prior)
        np.testing.assert_allclose(out[:, 0], want, atol=1e-5)
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-9)
        assert rec["call"].startswith("8-slot") and len(rec["stage_sha256"]) == 64 and len(rec["table_sha256"]) == 64
    # g = 1 is the plan BIT FOR BIT (float32 plan cast to float64)
    d = tmp_path / "gid"
    d.mkdir()
    RS1, table1, _ = B.load_retime("", str(_stage_and_table(d, [1.0] * 8)))
    assert np.array_equal(B.retime_plan8(RS1, table1, plan, prior), plan.astype(np.float64))


# --------------------------------------------------------------------------- #
# 10. the seam format                                                            #
# --------------------------------------------------------------------------- #
def test_seam_writer_marks_a_missing_token_partial(tmp_path):
    import run_bridge8 as RB
    toks = {t: {"fingerprint": f"fp{t}"} for t in ("a", "b", "c")}
    rec = {t: {"token": t, "source": "refcv8", "poses": np.zeros((8, 3)).tolist(), "knots": np.zeros((8, 2)).tolist()}
           for t in ("a", "b")}
    m = RB.write_seam(str(tmp_path), "R8_LEGAL", ["a", "b", "c"], rec, toks, {})
    assert m["partial"] is True and m["n_missing"] == 1 and m["n_model_rows"] == 2
    z = np.load(tmp_path / "seam_R8_LEGAL.npz")
    assert z["poses"].shape == (2, 8, 3) and z["knots"].shape == (2, 8, 2) and list(z["sampling"]) == [8, 0.5]
    rec["c"] = {"token": "c", "source": "cv_standin"}
    m2 = RB.write_seam(str(tmp_path), "R8_LEGAL", ["a", "b", "c"], rec, toks, {})
    assert m2["partial"] is False and m2["n_cv_standin_rows"] == 1


def test_the_runners_refuse_a_mutated_environment():
    import subprocess
    env = dict(os.environ, R8_MUTATION="wrong_token", CUDA_VISIBLE_DEVICES="-1")
    for script, extra in (("run_navsim_refcv8.py", ["--ckpt", "nonexistent.pt"]),
                          ("run_bridge8.py", ["--split", "navtest", "--arms", "R8_LEGAL", "--inputs", "x", "--road-plane", "x",
                                              "--ckpt", "x", "--config", "x", "--out", os.path.join(PKG, "raw", "_never")])):
        r = subprocess.run([sys.executable, os.path.join(PKG, "code", script)] + extra, capture_output=True, text=True,
                           env=env, timeout=300)
        assert r.returncode != 0 and "R8_MUTATION" in (r.stdout + r.stderr), script
