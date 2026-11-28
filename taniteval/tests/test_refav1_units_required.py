"""HJ-UNITS (BACKLOG; RETRACTION_LOG R46, 2026-10-17): the unit of action channel 1 is REQUIRED.

THE HAZARD. The recorded v2ep channel ``actions[:, 0]`` is a road-wheel STEER ANGLE,
``arctan(L * kappa)`` with L = 2.9 (``physicalai.py`` writes it; ``kinematic.STEER_WHEELBASE_M``
carries the contract). Every integrator in the programme wants a CURVATURE. A caller that took
a defaulted ``action_units="kappa"`` integrated the angle as a curvature -- an echo control
over-rotated about 2.9x (5.356 m instead of 2.838 m) that manufactured a fake -1.90 m "lever".
The unit is a property of WHERE THE CONTROLS CAME FROM, so the CALLER must state it.

THREE SURFACES carry that hazard in live code and are pinned here, the same way:

  * ``taniteval/tools/refav1_arm.py::paths_from_controls``  (landed 9939892; pinned elsewhere by
    ``stack/tests/test_steer_curvature_interface.py`` B3a-c, repeated here so ONE file states the class)
  * ``taniteval/taniteval/floors_kin.py::const_control_paths``
  * ``taniteval/taniteval/floors_kin.py::ha0_ext_paths``  (its own main input ``kin[:, 2]`` IS the
    recorded steer angle, so a default of "kappa" was wrong by construction for its main caller)

HJ-UNITS-2 (2026-10-18) EXTENDS THE SAME CLASS TO FOUR MORE SURFACES AND THE CLI (families g, h, i):

  * ``stack/tanitad/refs/refa_v1_plan.py::unicycle_paths``  (``action_units``; the shared integrator)
  * ``RefAV1.plan``  (``model_action_units``)  -- the planner->model crossing: the unit is a property of
    the CHECKPOINT (what its action channel 1 was trained on), which ``plan`` cannot see
  * ``RefAV1.imagined_goal``  (``model_action_units``) and ``RefAV1._imagine_tactical_goal`` (``units``)
  * ``taniteval/tools/cost_surface_probe.py::WindowContext.score``  (``units``; also now REFUSES an unknown
    value instead of treating "anything but kappa" as steer)
  * ``refav1_arm.py --action-units``  -- no default; a rollout without it exits 2 naming both values;
    ``--analyze-only`` reads the unit from the dump manifest and is not blocked

FAMILIES
  (a) omitting the unit raises TypeError and the parameter is keyword-only with NO default
  (b) an unknown / non-string unit raises ValueError naming BOTH legal values, even for zero controls
  (c) KNOWN ANSWER: a constant steer integrated as "steer" == its equivalent curvature
      ``tan(steer) / STEER_WHEELBASE_M`` integrated as "kappa" == the closed form of the integrator's
      own scheme, to a tolerance DERIVED from float epsilon and the path scale; plus the CONTROLS that
      the tolerance can tell a wrong unit (and a wrong wheelbase) apart
  (d) MUTATION: re-adding the old ``= "kappa"`` default (in-process on the source text, and in a
      TEMP COPY of the tree running the real test (a) in a subprocess) turns test (a) RED
  (e) CENSUS: every live call of these integrators names its unit; with a positive control and a
      mutation on a real file, because a census that reads 0 suspects on BOTH the fixed and the
      broken code proves nothing
  (g) the HJ-UNITS-2 surfaces: (a) + (b) + a closed-form known answer for the shared integrator and for
      the model boundary, and the (d) mutation arms in-process and in a temp copy
  (h) the CLI flag, behaviourally (the real ``main``) and by mutation
  (i) the CLI census over live docs / launchers (an invocation of ``refav1_arm.py`` states the unit)

INSTFIX / HJ-UNITS-3 (2026-10-18) CLOSES THE LAST SILENT UNIT DEFAULTS ON THE ROLLOUT CLIs (families j, k, l, m):

  * ``taniteval/tools/actdiv_anchored.py --action-units``  -- was ``"kappa"`` (the LEGACY direction: the swept level is
    fed to the model unconverted); now required for ``--family refav1`` (exit 2), not read by ``--family v7`` / ``--selftest``
  * ``taniteval/tools/refcv3_arm.py --action-units`` and ``taniteval/tools/openloop_suite.py --action-units`` -- were
    ``"steer"`` (the safe direction, but silent); they integrate the RECORDED channel for the ``ha`` floors, so they are
    required for a rollout (``--analyze-only`` / ``--arm-json`` are not asked)
  * the two LIVE callers that relied on that default now state it: ``taniteval/taniteval/bench/internal_t1.py::build_cmd``
    and ``taniteval/tools/run_hierarchy_panel.py::roll``
  (j) actdiv CLI, behaviourally (the real ``main``), by AST, and by mutation (in-process and in a temp copy)
  (k) refcv3_arm + openloop_suite CLI, same three ways
  (l) the DECLARATION CENSUS: every live ``--action-units`` declaration has ``default=None`` (one dead-flag exception,
      proved dead), with positive / negative controls and a mutation on every real file
  (m) the invocation census generalised to all four CLIs, and the two live callers

ASCII-only assertion text: this dev box's console is cp1252.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import math
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_TESTS = Path(__file__).resolve().parent
_TANI = _TESTS.parent                       # <repo>/taniteval
_REPO = _TANI.parent                        # <repo>
_ARM_PATH = _TANI / "tools" / "refav1_arm.py"
_FK_PATH = _TANI / "taniteval" / "floors_kin.py"
_PLAN_PATH = _REPO / "stack" / "tanitad" / "refs" / "refa_v1_plan.py"
_V1_PATH = _REPO / "stack" / "tanitad" / "refs" / "refa_v1.py"
_PROBE_PATH = _TANI / "tools" / "cost_surface_probe.py"

from taniteval import floors_kin as fk                                       # noqa: E402
from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig       # noqa: E402
from tanitad.models.kinematic import STEER_WHEELBASE_M                       # noqa: E402
from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v   # noqa: E402
from tanitad.refs import refa_v1 as _v1_live                                 # noqa: E402
from tanitad.refs import refa_v1_plan as _plan_live                          # noqa: E402

DT = 0.2
N = 30
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
def _load_module(path: Path, name: str, source: str | None = None):
    """Import ``path`` as a fresh module; ``source`` overrides its text (the mutation arm)."""
    src = path.read_text(encoding="utf-8") if source is None else source
    mod = types.ModuleType(name)
    mod.__file__ = str(path)                  # refav1_arm derives its repo paths from __file__
    sys.modules[name] = mod
    try:
        exec(compile(src, str(path), "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


@pytest.fixture(scope="module")
def arm():
    mod = _load_module(_ARM_PATH, "_hj_units_arm_live")
    yield mod
    sys.modules.pop("_hj_units_arm_live", None)


#: HAND-WRITTEN controls (no RNG): (accel, channel 1), K = 10.
_CONTROLS = [[0.5, 0.0], [0.25, 0.02], [0.0, 0.04], [-0.25, 0.06], [-0.5, 0.08],
             [0.4, 0.06], [0.2, 0.04], [0.0, 0.02], [-0.2, 0.0], [-0.4, -0.02]]


def _surfaces(arm_mod, fk_mod) -> dict:
    """surface name -> ``call(**unit_kwargs)``; the omit case passes NO kwargs."""
    c = torch.tensor(_CONTROLS, dtype=torch.float32)
    return {
        "refav1_arm.paths_from_controls":
            lambda **kw: arm_mod.paths_from_controls(c, 9.0, DT, 10, **kw),
        "floors_kin.const_control_paths":
            lambda **kw: fk_mod.const_control_paths(np.array([9.0]), 0.3, 0.05, DT, 10, **kw),
        "floors_kin.ha0_ext_paths":
            lambda **kw: fk_mod.ha0_ext_paths(np.array([[9.0, 0.3, 0.05]]), DT, 10, **kw),
    }


_NAMES = ["refav1_arm.paths_from_controls", "floors_kin.const_control_paths",
          "floors_kin.ha0_ext_paths"]


def _function(arm_mod, fk_mod, name):
    mod = arm_mod if name.startswith("refav1_arm.") else fk_mod
    return getattr(mod, name.split(".", 1)[1])


def _omission_rejected(call) -> bool:
    """True iff a call that omits ``action_units`` is REFUSED with a TypeError naming it."""
    try:
        call()
    except TypeError as e:
        return "action_units" in str(e)
    return False


# =========================================================================== #
# (a) omission raises; keyword-only; no default                                #
# =========================================================================== #
@pytest.mark.parametrize("name", _NAMES)
def test_a_omitted_units_raises_TypeError(arm, name):
    call = _surfaces(arm, fk)[name]
    assert _omission_rejected(call), name + ": a call without action_units must raise TypeError"


@pytest.mark.parametrize("name", _NAMES)
def test_a2_unit_parameter_is_keyword_only_with_no_default(arm, name):
    prm = inspect.signature(_function(arm, fk, name)).parameters["action_units"]
    assert prm.default is inspect.Parameter.empty, (name, prm.default)
    assert prm.kind is inspect.Parameter.KEYWORD_ONLY, (name, prm.kind)


# =========================================================================== #
# (b) an unknown value raises ValueError naming BOTH legal values              #
# =========================================================================== #
_BAD = [None, "", "Kappa", "STEER", "curvature", "radians", "kappa|steer", 0, ("kappa",)]


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("bad", _BAD, ids=[repr(b) for b in _BAD])
def test_b_unknown_units_raise_ValueError_naming_both_legal_values(arm, name, bad):
    call = _surfaces(arm, fk)[name]
    with pytest.raises(ValueError) as ei:
        call(action_units=bad)
    msg = str(ei.value)
    assert "kappa" in msg and "steer" in msg, msg


def test_b2_zero_controls_still_have_to_name_a_legal_unit(arm):
    """Zero controls are 0 in either unit (so the value cannot change the path), but a
    caller that never decided still must not pass: the unit is stated, not inferred."""
    z = torch.zeros(10, 2)
    with pytest.raises(TypeError):
        arm.paths_from_controls(z, 9.0, DT, 10)
    with pytest.raises(ValueError):
        arm.paths_from_controls(z, 9.0, DT, 10, action_units="nonsense")
    with pytest.raises(ValueError):
        fk.const_control_paths(np.array([9.0]), 0.0, 0.0, DT, 10, action_units="nonsense")
    a = arm.paths_from_controls(z, 9.0, DT, 10, action_units="kappa")
    b = arm.paths_from_controls(z, 9.0, DT, 10, action_units="steer")
    assert torch.equal(a, b)                                  # moot for zero controls, by identity


# =========================================================================== #
# (c) KNOWN ANSWER                                                             #
# =========================================================================== #
def _closed_form(v0: float, kappa: float, dt: float, n: int) -> np.ndarray:
    """The integrator's OWN scheme in closed form, for a = 0 (constant speed).

    ``rollout_unicycle`` / ``floors_kin``: x += v cos(yaw) dt; y += v sin(yaw) dt; yaw += v kappa dt
    (position on the PRE-update heading). With theta = v*kappa*dt the heading after j steps is
    j*theta, so the m-th output (m = 1..n) is the geometric sum
        z_m = v dt * sum_{i=0}^{m-1} exp(i*i*theta) = v dt * [sin(m theta/2) / sin(theta/2)] * exp(i (m-1) theta/2)
    (the sin form, not (e^{i m theta}-1)/(e^{i theta}-1): the quotient cancels catastrophically as
    theta -> 0, and this form is accurate to a few eps at every theta).
    """
    theta = v0 * kappa * dt
    m = np.arange(1, n + 1, dtype=np.float64)
    if abs(theta) < 1e-12:
        z = v0 * dt * m.astype(np.complex128)
    else:
        z = v0 * dt * (np.sin(m * theta / 2.0) / np.sin(theta / 2.0)) * np.exp(1j * (m - 1.0) * theta / 2.0)
    return np.stack([z.real, z.imag], -1)


def _tolerance(eps: float, path: np.ndarray, psi_max: float) -> float:
    """tol = eps * (R + 1) * (n + psi_max * (n + 2))        [metres]

    DERIVED, not tuned. With R the largest coordinate of the path (+1 m floor) and n the steps:
      * the running x / y sums round n times, at most eps*R/2 each          -> eps*R*n/2
      * the converted curvature carries one relative eps into the heading   -> eps*psi_max*R
      * the heading sum rounds n times, at most eps*psi_max/2 each, and a heading error dpsi
        moves the position by at most R*dpsi                                -> eps*psi_max*R*n/2
      * the trig roundings are smaller than the first term
    The sum is eps*R*(n/2 + psi_max*(1 + n/2)); the formula above is exactly 2x that (the safety).
    """
    r = float(np.abs(path).max())
    n = path.shape[0]
    return eps * (r + 1.0) * (n + psi_max * (n + 2.0))


_STEER_CASES = [(10.0, 0.02), (10.0, 0.10), (12.0, -0.13), (6.0, 0.05)]


@pytest.mark.parametrize("v0,steer", _STEER_CASES)
def test_c_constant_steer_equals_equivalent_kappa_equals_closed_form(arm, v0, steer):
    L = STEER_WHEELBASE_M                                      # from source, never retyped
    kappa = math.tan(steer) / L
    want = _closed_form(v0, kappa, DT, N)
    psi_max = abs(v0 * kappa * DT * N)

    # float64 numpy (floors_kin): "steer" and the equivalent "kappa" and the closed form
    p_steer = fk.const_control_paths(np.array([v0]), 0.0, steer, DT, N, action_units="steer")[0]
    p_kappa = fk.const_control_paths(np.array([v0]), 0.0, kappa, DT, N, action_units="kappa")[0]
    tol64 = _tolerance(EPS64, want, psi_max)
    assert np.abs(p_steer - want).max() <= tol64, (np.abs(p_steer - want).max(), tol64)
    assert np.abs(p_kappa - want).max() <= tol64
    assert np.abs(p_steer - p_kappa).max() <= tol64

    # float32 torch (refav1_arm.paths_from_controls): same three-way identity
    ctrl_steer = torch.tensor([[0.0, steer]] * N, dtype=torch.float32)
    ctrl_kappa = torch.tensor([[0.0, kappa]] * N, dtype=torch.float32)
    t_steer = arm.paths_from_controls(ctrl_steer, v0, DT, N, action_units="steer")[0].double().numpy()
    t_kappa = arm.paths_from_controls(ctrl_kappa, v0, DT, N, action_units="kappa")[0].double().numpy()
    tol32 = _tolerance(EPS32, want, psi_max)
    assert np.abs(t_steer - want).max() <= tol32, (np.abs(t_steer - want).max(), tol32)
    assert np.abs(t_kappa - want).max() <= tol32
    assert np.abs(t_steer - t_kappa).max() <= tol32


@pytest.mark.parametrize("v0,steer", _STEER_CASES)
def test_c2_the_tolerance_can_tell_a_wrong_unit_and_a_wrong_wheelbase_apart(arm, v0, steer):
    """The CONTROL for (c): an equality inside a tolerance proves nothing unless the same
    tolerance REJECTS the defect it exists to catch. Integrating the steer angle AS a curvature
    (the R46 error) and using a vehicle's true wheelbase instead of the encoder's (kinematic.py:
    the release's clips average 3.085 m, NOT 2.9) must both sit far outside it."""
    L = STEER_WHEELBASE_M
    want = _closed_form(v0, math.tan(steer) / L, DT, N)
    psi_max = abs(v0 * math.tan(steer) / L * DT * N)
    ctrl = torch.tensor([[0.0, steer]] * N, dtype=torch.float32)
    tol32 = _tolerance(EPS32, want, psi_max)
    tol64 = _tolerance(EPS64, want, psi_max)

    wrong_unit = arm.paths_from_controls(ctrl, v0, DT, N, action_units="kappa")[0].double().numpy()
    assert np.abs(wrong_unit - want).max() > 1000.0 * tol32, np.abs(wrong_unit - want).max()

    L_true = 3.085                                            # kinematic.py: "mean 3.085, NOT ONE is 2.9"
    wrong_L = _closed_form(v0, math.tan(steer) / L_true, DT, N)
    assert np.abs(wrong_L - want).max() > 100.0 * tol32, np.abs(wrong_L - want).max()
    assert np.abs(wrong_L - want).max() > 1e6 * tol64


def test_c3_float32_torch_and_float64_numpy_integrators_agree_with_acceleration(arm):
    """Two independent implementations of the same scheme, with accel != 0 (no closed form needed):
    they must agree to the derived float32 tolerance in BOTH units."""
    v0, a, steer = 8.0, 0.4, 0.08
    ctrl = torch.tensor([[a, steer]] * N, dtype=torch.float32)
    for units in ("steer", "kappa"):
        t = arm.paths_from_controls(ctrl, v0, DT, N, action_units=units)[0].double().numpy()
        f = fk.const_control_paths(np.array([v0]), a, steer, DT, N, action_units=units)[0]
        kap = steer if units == "kappa" else math.tan(steer) / STEER_WHEELBASE_M
        psi_max = (v0 + a * DT * N) * abs(kap) * DT * N
        tol = _tolerance(EPS32, f, psi_max)
        assert np.abs(t - f).max() <= tol, (units, np.abs(t - f).max(), tol)


def test_c4_the_two_wheelbase_constants_are_the_same_number():
    assert fk.STEER_WHEELBASE_M == STEER_WHEELBASE_M == 2.9


# =========================================================================== #
# (d) MUTATION: re-adding the old default turns (a) RED                        #
# =========================================================================== #
_ARM_NEEDLE = "action_units: str):"
_FK_NEEDLE = "*, action_units: str) -> np.ndarray:"


def _mutated_sources() -> tuple[str, str]:
    arm_src = _ARM_PATH.read_text(encoding="utf-8")
    fk_src = _FK_PATH.read_text(encoding="utf-8")
    assert arm_src.count(_ARM_NEEDLE) == 1, "refav1_arm signature moved: update this mutation"
    assert fk_src.count(_FK_NEEDLE) == 2, "floors_kin signatures moved: update this mutation"
    return (arm_src.replace(_ARM_NEEDLE, 'action_units: str = "kappa"):'),
            fk_src.replace(_FK_NEEDLE, '*, action_units: str = "kappa") -> np.ndarray:'))


def test_d_mutation_in_process_restoring_the_default_turns_the_omission_check_RED(arm):
    m_arm_src, m_fk_src = _mutated_sources()
    m_arm = _load_module(_ARM_PATH, "_hj_units_arm_mutant", m_arm_src)
    m_fk = _load_module(_FK_PATH, "_hj_units_fk_mutant", m_fk_src)
    try:
        live, mutant = _surfaces(arm, fk), _surfaces(m_arm, m_fk)
        for name in _NAMES:
            assert _omission_rejected(live[name]), name + ": the live code must refuse an omission"
            assert not _omission_rejected(mutant[name]), (
                name + ": with the default restored the omission is ACCEPTED -- test (a) is RED")
            # the mutation really IS 'the old default': omitting == stating "kappa", bit for bit
            got, want = mutant[name](), mutant[name](action_units="kappa")
            assert np.array_equal(np.asarray(got), np.asarray(want)), name
            # and the units still differ, so the silent default is not a no-op
            steer = np.asarray(mutant[name](action_units="steer"), dtype=np.float64)
            assert np.abs(steer - np.asarray(got, dtype=np.float64)).max() > 1e-3, name
        # parameter introspection goes RED too
        for name in _NAMES:
            prm = inspect.signature(_function(m_arm, m_fk, name)).parameters["action_units"]
            assert prm.default == "kappa"
    finally:
        sys.modules.pop("_hj_units_arm_mutant", None)
        sys.modules.pop("_hj_units_fk_mutant", None)


def _run_pytest_in_copy(tmp: Path, mutate: bool):
    """Copy the minimum tree to ``tmp``, optionally restore the defaults, and run the REAL test (a)
    (this file, selected by -k) in a subprocess. Returns (returncode, combined output)."""
    root = tmp / ("mutant" if mutate else "clean") / "taniteval"
    (root / "tests").mkdir(parents=True)
    shutil.copytree(_TANI / "taniteval", root / "taniteval",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(_TANI / "tools", root / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(_TANI / "conftest.py", root / "conftest.py")
    shutil.copy2(Path(__file__), root / "tests" / Path(__file__).name)
    if mutate:
        m_arm_src, m_fk_src = _mutated_sources()
        (root / "tools" / "refav1_arm.py").write_text(m_arm_src, encoding="utf-8", newline="\n")
        (root / "taniteval" / "floors_kin.py").write_text(m_fk_src, encoding="utf-8", newline="\n")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["HJ_UNITS_NESTED"] = "1"                              # recursion guard, see test (d2)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "stack"), str(_REPO / "stack" / "scripts")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "-k", "test_a_ or test_a2_", str(root / "tests" / Path(__file__).name)]
    p = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def test_d2_mutation_in_a_temp_copy_turns_the_real_omission_check_RED(tmp_path):
    """The deliberate-regression arm, end to end: in a TEMP COPY of the tree restore ``= "kappa"``
    on all three signatures and run this file's own test (a). Control first: the CLEAN copy is
    GREEN, so the RED below is the mutation and not the harness."""
    if os.environ.get("HJ_UNITS_NESTED") == "1":
        pytest.skip("nested run: this test is not selected inside its own subprocess")
    rc_clean, out_clean = _run_pytest_in_copy(tmp_path, mutate=False)
    assert rc_clean == 0, out_clean[-3000:]
    assert "6 passed" in out_clean, out_clean[-1500:]            # 3 x test_a_ + 3 x test_a2_
    rc_mut, out_mut = _run_pytest_in_copy(tmp_path, mutate=True)
    assert rc_mut != 0, "the mutant tree still passed test (a):\n" + out_mut[-3000:]
    for name in _NAMES:
        assert f"test_a_omitted_units_raises_TypeError[{name}]" in out_mut, out_mut[-3000:]
        assert f"test_a2_unit_parameter_is_keyword_only_with_no_default[{name}]" in out_mut
    assert "6 failed" in out_mut, out_mut[-1500:]


# =========================================================================== #
# (e) CENSUS: every live call names its unit                                   #
# =========================================================================== #
#: callee name -> keyword names that count as STATING the unit.
#: ``const_control_paths`` is also the name of ``stack/tanitad/models/p10_prior.py``'s torch port,
#: whose unit is a bool ``steer_units`` (two in-file calls, both explicit).
#: HJ-UNITS-2 adds the shared integrator, the planner->model crossing, and the probe's re-scoring.
#: ``plan`` and ``score`` are COMMON names (``policy.plan(frames, intr, v, nav)`` is a different API;
#: ``score`` is everywhere), so ``_selected`` claims them only where they mean the unit-bearing callee.
_CALLEES = {
    "paths_from_controls": {"action_units"},
    "ha0_ext_paths": {"action_units"},
    "const_control_paths": {"action_units", "steer_units"},
    "integrate_select": {"action_units"},
    "unicycle_paths": {"action_units"},
    "imagined_goal": {"model_action_units"},
    "_imagine_tactical_goal": {"units"},
    "plan": {"model_action_units"},
    "score": {"units"},
}
_LIVE_ROOTS = ["stack/tanitad", "stack/scripts", "taniteval/taniteval", "taniteval/tools", "tools"]


def _selected(node: ast.Call, name: str, label: str) -> bool:
    """Is this call the unit-bearing callee? Generic names are claimed only where they mean it."""
    if name == "plan":                  # RefAV1.plan(feats, *, v0=...): the policy adapters' plan() has no v0 kw
        return any(k.arg == "v0" for k in node.keywords)
    if name == "score":                 # cost_surface_probe.WindowContext.score, always called on `ctx`
        f = node.func
        return (label.endswith("cost_surface_probe.py") and isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name) and f.value.id == "ctx")
    return True


def _callee_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _census_detail(src: str, label: str) -> tuple[dict, list[str]]:
    """({callee: calls scanned}, [offending call sites]). A call with ``**kwargs`` cannot be verified -> flagged."""
    tree = ast.parse(src)
    by, bad = {}, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node)
        if name not in _CALLEES or not _selected(node, name, label):
            continue
        by[name] = by.get(name, 0) + 1
        kws = {k.arg for k in node.keywords}
        if None in kws or not (kws & _CALLEES[name]):
            bad.append(f"{label}:{node.lineno} {name}(...)")
    return by, bad


def _census_source(src: str, label: str) -> tuple[int, list[str]]:
    """(calls scanned, [offending call sites])."""
    by, bad = _census_detail(src, label)
    return sum(by.values()), bad


_REPO_BY_CALLEE: dict = {}


def _census_repo() -> tuple[int, list[str], list[str]]:
    scanned, bad, files = 0, [], []
    _REPO_BY_CALLEE.clear()
    for rel in _LIVE_ROOTS:
        root = _REPO / rel
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            parts = set(path.relative_to(_REPO).parts)
            if parts & {"tests", "results", "incoming", "__pycache__"}:
                continue
            try:
                src = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not any(n in src for n in _CALLEES):
                continue
            label = str(path.relative_to(_REPO)).replace("\\", "/")
            try:
                by, b = _census_detail(src, label)
            except SyntaxError as ex:                     # unverifiable == flagged, never skipped
                bad.append(f"{label}: UNPARSEABLE ({ex.msg} line {ex.lineno})")
                continue
            n = sum(by.values())
            for k, v in by.items():
                _REPO_BY_CALLEE[k] = _REPO_BY_CALLEE.get(k, 0) + v
            if n:
                files.append(label)
            scanned += n
            bad += b
    return scanned, bad, files


def test_e_census_every_live_call_names_its_unit():
    scanned, bad, files = _census_repo()
    assert scanned >= 15, f"census is vacuous: only {scanned} calls scanned in {files}"
    assert not bad, "live call(s) that do not state action_units:\n  " + "\n  ".join(bad)


def test_e2_census_positive_control_a_call_without_the_unit_is_flagged():
    omitted = "import taniteval.floors_kin as fk\nx = fk.ha0_ext_paths(kin, 0.1, 30)\n"
    stated = "import taniteval.floors_kin as fk\nx = fk.ha0_ext_paths(kin, 0.1, 30, action_units='steer')\n"
    splat = "x = fk.ha0_ext_paths(kin, 0.1, 30, **kw)\n"
    assert _census_source(omitted, "ctl")[1] == ["ctl:2 ha0_ext_paths(...)"]
    assert _census_source(stated, "ctl") == (1, [])
    assert len(_census_source(splat, "ctl")[1]) == 1            # unverifiable == flagged


def test_e3_census_mutation_removing_the_keyword_from_a_REAL_file_is_caught():
    path = _TANI / "tools" / "refav1_add_floor.py"
    src = path.read_text(encoding="utf-8")
    n0, bad0 = _census_source(src, "refav1_add_floor.py")
    assert n0 >= 1 and not bad0, (n0, bad0)

    class _Strip(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if _callee_name(node) == "paths_from_controls":
                node.keywords = [k for k in node.keywords if k.arg != "action_units"]
            return node

    mutated = ast.unparse(ast.fix_missing_locations(_Strip().visit(copy.deepcopy(ast.parse(src)))))
    n1, bad1 = _census_source(mutated, "refav1_add_floor.py[mutant]")
    assert n1 == n0 and len(bad1) == n0, (n1, bad1)


# --------------------------------------------------------------------------- #
# (e, extended) HJ-UNITS-2 census: the new required arguments                  #
# --------------------------------------------------------------------------- #
#: Minimum live calls the census must SCAN per callee. A callee that scans 0 would pass vacuously --
#: the failure this guards is the census silently ceasing to see a surface (e.g. the selector
#: rejecting every ``plan``). MEASURED on the tip at the time of writing; ``>=`` so a new caller is fine.
_MIN_SCANNED = {"unicycle_paths": 1, "plan": 2, "_imagine_tactical_goal": 4, "score": 5,
                "paths_from_controls": 3, "ha0_ext_paths": 1}


def test_e4_census_covers_every_new_required_argument_in_live_code():
    scanned, bad, files = _census_repo()
    assert not bad, "live call(s) that do not state their unit:\n  " + "\n  ".join(bad)
    for callee, floor in _MIN_SCANNED.items():
        got = _REPO_BY_CALLEE.get(callee, 0)
        assert got >= floor, f"census scanned only {got} live {callee}() call(s) (< {floor}): {_REPO_BY_CALLEE}"
    # the two files whose callers HJ-UNITS-2 had to fix are in the scan
    assert "taniteval/tools/cost_surface_probe.py" in files and "taniteval/tools/refav1_arm.py" in files, files


def test_e5_census_positive_controls_for_the_new_callees_and_the_selector():
    for src, name in [
            ("x = model.plan(feats, v0=8.0, plan_cfg=pc)\n", "plan"),
            ("x = model.plan(feats, v0=8.0, **kw)\n", "plan"),                       # unverifiable == flagged
            ("x = m.imagined_goal(feats, v0=8.0)\n", "imagined_goal"),
            ("x = m._imagine_tactical_goal(last, brains, v0_t)\n", "_imagine_tactical_goal"),
            ("x = unicycle_paths(c, v0, dt)\n", "unicycle_paths"),
            ("x = ctx.score(grid, chunk=64)\n", "score")]:
        label = "tools/cost_surface_probe.py" if name == "score" else "ctl.py"
        n, bad = _census_source(src, label)
        assert n == 1 and len(bad) == 1, (name, n, bad)
    stated = [("model.plan(feats, v0=8.0, plan_cfg=pc, model_action_units='steer')", "plan"),
              ("m.imagined_goal(feats, v0=8.0, model_action_units='kappa')", "imagined_goal"),
              ("m._imagine_tactical_goal(last, brains, v0_t, units='kappa')", "_imagine_tactical_goal"),
              ("unicycle_paths(c, v0, dt, action_units='kappa')", "unicycle_paths"),
              ("ctx.score(grid, units='steer')", "score")]
    for src, name in stated:
        label = "tools/cost_surface_probe.py" if name == "score" else "ctl.py"
        assert _census_source("x = " + src + "\n", label) == (1, []), src
    # SELECTIVITY (the negative control): another API's plan() / score() is NOT claimed
    assert _census_source("x = policy.plan(frames, intr, v, nav, nav_known=None)\n", "ctl.py") == (0, [])
    assert _census_source("x = ctx.score(grid)\n", "stack/scripts/other.py") == (0, [])
    assert _census_source("x = self.score(grid)\n", "tools/cost_surface_probe.py") == (0, [])


def test_e6_census_mutation_stripping_the_keyword_from_a_REAL_call_is_caught():
    """For every new callee that a real live file calls: strip its unit keyword with an AST transform
    and require the census to flag EXACTLY those calls (and nothing was flagged before)."""
    targets = [("taniteval/tools/refav1_arm.py", "plan"),
               ("taniteval/tools/refav1_arm.py", "unicycle_paths"),
               ("taniteval/tools/cost_surface_probe.py", "plan"),
               ("taniteval/tools/cost_surface_probe.py", "_imagine_tactical_goal"),
               ("taniteval/tools/cost_surface_probe.py", "score"),
               ("stack/tanitad/refs/refa_v1.py", "_imagine_tactical_goal")]
    for rel, callee in targets:
        src = (_REPO / rel).read_text(encoding="utf-8")
        by0, bad0 = _census_detail(src, rel)
        assert by0.get(callee, 0) >= 1 and not bad0, (rel, callee, by0, bad0)
        kwset = _CALLEES[callee]

        class _Strip(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if _callee_name(node) == callee and _selected(node, callee, rel):
                    node.keywords = [k for k in node.keywords if k.arg not in kwset]
                return node

        mutated = ast.unparse(ast.fix_missing_locations(_Strip().visit(copy.deepcopy(ast.parse(src)))))
        by1, bad1 = _census_detail(mutated, rel)
        assert by1.get(callee) == by0[callee], (rel, callee, by0, by1)
        assert len(bad1) == by0[callee] and all(callee in b for b in bad1), (rel, callee, bad1)


# =========================================================================== #
# (g) HJ-UNITS-2 surfaces                                                      #
# =========================================================================== #
def _tiny_cfg(v1_mod):
    return v1_mod.RefAV1Config(
        tac_vocab_version="v7.0", d_enc=32, n_tokens=8, d_state=32,
        op_layers=1, op_heads=2, op_window=4, tac_layers=1, tac_queries=4,
        str_dim=16, str_layers=1,
        strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1, n_heads=2, d_ctx=16, d_cmd=8),
        tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1, n_heads=2, d_intent=16))


class _ConstHead(torch.nn.Module):
    def __init__(self, n: int, idx: int):
        super().__init__()
        self.n, self.idx = n, idx

    def forward(self, x):
        out = torch.full((x.shape[0], self.n), -10.0, device=x.device, dtype=x.dtype)
        out[:, self.idx] = 10.0
        return out


def _tiny_model(v1_mod, plan_mod):
    """(model, cfg, feats) built from the GIVEN modules (live or mutant). The decode is pinned to a
    TURN so the imagined goal CARRIES CURVATURE -- with LANE_KEEP a unit change is a no-op."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(v1_mod)
    m = v1_mod.RefAV1(cfg).eval()
    lat_v, lon_v = tactical_lat_actions(cfg.tac_vocab_version), tactical_lon_actions_v(cfg.tac_vocab_version)
    m.lat_head = _ConstHead(len(lat_v), lat_v.index("TURN_R"))
    m.lon_head = _ConstHead(len(lon_v), lon_v.index("CRUISE"))
    feats = torch.randn(1, cfg.op_window, cfg.n_tokens, cfg.d_enc)
    pc = v1_mod.PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt, n_samples=8, n_iters=1, n_elites=2, seed=0)
    return m, cfg, feats, pc


#: surface -> the name of its unit parameter
_NEW_PARAM = {"refa_v1_plan.unicycle_paths": "action_units",
              "RefAV1.plan": "model_action_units",
              "RefAV1.imagined_goal": "model_action_units",
              "RefAV1._imagine_tactical_goal": "units",
              "cost_surface_probe.WindowContext.score": "units"}
_NEW_NAMES = list(_NEW_PARAM)


def _new_surfaces(v1_mod, plan_mod, probe_mod) -> dict:
    """surface name -> (``call(**unit_kwargs)``, function object); the omit case passes NO kwargs."""
    m, cfg, feats, pc = _tiny_model(v1_mod, plan_mod)
    with torch.no_grad():
        field = m.encode(feats)
        last = m._last_state(field)
        brains = m._run_brains(field.mean(dim=-2), None)
        ctx = probe_mod.WindowContext(m, cfg, feats, 8.0, None, pc)       # a REAL context, no stub
    v0_t = torch.as_tensor([8.0], dtype=torch.float32)
    c = torch.tensor(_CONTROLS, dtype=torch.float32)[None]
    return {
        "refa_v1_plan.unicycle_paths":
            (lambda **kw: plan_mod.unicycle_paths(c, torch.tensor([9.0]), DT, **kw), plan_mod.unicycle_paths),
        "RefAV1.plan":
            (lambda **kw: m.plan(feats, v0=8.0, plan_cfg=pc, **kw), v1_mod.RefAV1.plan),
        "RefAV1.imagined_goal":
            (lambda **kw: m.imagined_goal(feats, v0=8.0, **kw), v1_mod.RefAV1.imagined_goal),
        "RefAV1._imagine_tactical_goal":
            (lambda **kw: m._imagine_tactical_goal(last, brains, v0_t, **kw), v1_mod.RefAV1._imagine_tactical_goal),
        "cost_surface_probe.WindowContext.score":
            (lambda **kw: ctx.score(c, **kw), probe_mod.WindowContext.score),
    }


@pytest.fixture(scope="module")
def new_live():
    probe = _load_module(_PROBE_PATH, "_hj_units_probe_live")
    yield _new_surfaces(_v1_live, _plan_live, probe)
    sys.modules.pop("_hj_units_probe_live", None)


def _omission_rejected_for(call, param: str) -> bool:
    """True iff a call that omits the unit is REFUSED with a TypeError naming it. Any other outcome --
    including a different exception raised AFTER argument binding -- means the omission got through."""
    try:
        call()
    except TypeError as e:
        return param in str(e)
    except Exception:                                          # noqa: BLE001
        return False
    return False


@pytest.mark.parametrize("name", _NEW_NAMES)
def test_g_omitted_units_raises_TypeError(new_live, name):
    call, _ = new_live[name]
    assert _omission_rejected_for(call, _NEW_PARAM[name]), (
        name + ": a call without " + _NEW_PARAM[name] + " must raise TypeError")


@pytest.mark.parametrize("name", _NEW_NAMES)
def test_g2_unit_parameter_is_keyword_only_with_no_default(new_live, name):
    _, fn = new_live[name]
    prm = inspect.signature(fn).parameters[_NEW_PARAM[name]]
    assert prm.default is inspect.Parameter.empty, (name, prm.default)
    assert prm.kind is inspect.Parameter.KEYWORD_ONLY, (name, prm.kind)


@pytest.mark.parametrize("name", _NEW_NAMES)
@pytest.mark.parametrize("bad", _BAD, ids=[repr(b) for b in _BAD])
def test_g3_unknown_units_raise_ValueError_naming_both_legal_values(new_live, name, bad):
    call, _ = new_live[name]
    with pytest.raises(ValueError) as ei:
        with torch.no_grad():
            call(**{_NEW_PARAM[name]: bad})
    msg = str(ei.value)
    assert "kappa" in msg and "steer" in msg, msg


def test_g4_unicycle_paths_known_answer_against_the_closed_form(new_live):
    """The shared integrator itself, under the SAME derived tolerance as (c): constant steer integrated as
    "steer" == its equivalent curvature as "kappa" == the closed form; the R46 error (steer fed as kappa)
    and a wrong wheelbase sit orders of magnitude outside the same tolerance."""
    L = STEER_WHEELBASE_M
    for v0, steer in _STEER_CASES:
        kappa = math.tan(steer) / L
        want = _closed_form(v0, kappa, DT, N)
        psi_max = abs(v0 * kappa * DT * N)
        tol = _tolerance(EPS32, want, psi_max)
        cs = torch.tensor([[[0.0, steer]] * N], dtype=torch.float32)
        ck = torch.tensor([[[0.0, kappa]] * N], dtype=torch.float32)
        v = torch.tensor([v0])
        ps = _plan_live.unicycle_paths(cs, v, DT, action_units="steer")[0].double().numpy()
        pk = _plan_live.unicycle_paths(ck, v, DT, action_units="kappa")[0].double().numpy()
        assert np.abs(ps - want).max() <= tol and np.abs(pk - want).max() <= tol, (v0, steer, tol)
        assert np.abs(ps - pk).max() <= tol
        wrong = _plan_live.unicycle_paths(cs, v, DT, action_units="kappa")[0].double().numpy()
        assert np.abs(wrong - want).max() > 1000.0 * tol, (v0, steer)          # the R46 error is not hidden
        wrong_L = _plan_live.unicycle_paths(cs, v, DT, action_units="steer", wheelbase=3.085)[0].double().numpy()
        assert np.abs(wrong_L - want).max() > 100.0 * tol, (v0, steer)         # nor is a wrong wheelbase


def test_g5_the_model_boundary_known_answer_steer_is_arctan_of_Lkappa_and_kappa_is_pass_through():
    """``_model_actions`` is the ONE planner->model crossing. Known answer: "steer" writes
    ``arctan(2.9 * kappa)`` into channel 1 (to 2 float32 eps of |L kappa|), "kappa" returns channel 1
    untouched -- and the same tolerance REJECTS the unconverted value and a wrong wheelbase."""
    m, cfg, feats, pc = _tiny_model(_v1_live, _plan_live)
    kappas = torch.tensor([[-0.2, -0.08, 0.0, 0.01, 0.08, 0.2]], dtype=torch.float32).T[None, :, 0]  # [1, 6]
    ctrl = torch.stack([torch.zeros_like(kappas), kappas], dim=-1)                                   # [1, 6, 2]
    with torch.no_grad():
        out_s = m._model_actions(ctrl, torch.tensor([8.0]), "steer")[..., 1].double()
        out_k = m._model_actions(ctrl, torch.tensor([8.0]), "kappa")[..., 1].double()
    k64 = kappas.double()
    want = torch.atan(STEER_WHEELBASE_M * k64)
    tol = 2.0 * EPS32 * (STEER_WHEELBASE_M * k64.abs()) + 1e-12
    assert bool((out_s - want).abs().le(tol).all()), (out_s, want)
    assert torch.equal(out_k, k64)                                             # pass-through, bit for bit
    nz = k64.abs() > 0
    assert bool((out_k - want).abs()[nz].gt(1000.0 * tol[nz]).all())           # pass-through is NOT arctan
    wrong_L = torch.atan(3.085 * k64)
    assert bool((wrong_L - want).abs()[nz].gt(100.0 * tol[nz]).all())          # nor is a wrong wheelbase


# --- (d, extended) mutation: restore each new default -------------------------
_PLAN_NEEDLE = ("                   action_units: str,\n"
                "                   wheelbase: float = STEER_WHEELBASE_M) -> Tensor:")
_V1_NEEDLES = [
    ("                               *, units: str,\n", "                               *, units: str = \"kappa\",\n"),
    ("model_action_units: str,\n                      nav_cmd: Tensor | None = None,",
     "model_action_units: str = \"kappa\",\n                      nav_cmd: Tensor | None = None,"),
    ("cost_chunk: int = 64, model_action_units: str,\n",
     "cost_chunk: int = 64, model_action_units: str = \"kappa\",\n"),
]


_PROBE_NEEDLE = "    def score(self, controls, *, units: str, wheelbase: float = 2.9,"


def _mutated_new_sources() -> tuple[str, str, str]:
    """(refa_v1_plan.py, refa_v1.py, cost_surface_probe.py) with the pre-HJ-UNITS-2 ``= "kappa"`` defaults restored."""
    plan_src = _PLAN_PATH.read_text(encoding="utf-8")
    v1_src = _V1_PATH.read_text(encoding="utf-8")
    probe_src = _PROBE_PATH.read_text(encoding="utf-8")
    assert probe_src.count(_PROBE_NEEDLE) == 1, "cost_surface_probe.score signature moved: update this mutation"
    probe_src = probe_src.replace(_PROBE_NEEDLE, _PROBE_NEEDLE.replace("units: str,", 'units: str = "kappa",'))
    assert plan_src.count(_PLAN_NEEDLE) == 1, "refa_v1_plan signature moved: update this mutation"
    plan_src = plan_src.replace(_PLAN_NEEDLE, _PLAN_NEEDLE.replace("action_units: str,", 'action_units: str = "kappa",'))
    for old, new in _V1_NEEDLES:
        assert v1_src.count(old) == 1, ("refa_v1 signature moved: update this mutation", old)
        v1_src = v1_src.replace(old, new)
    return plan_src, v1_src, probe_src


def test_g6_mutation_in_process_restoring_the_defaults_turns_the_omission_check_RED(new_live):
    plan_src, v1_src, probe_src = _mutated_new_sources()
    m_plan = _load_module(_PLAN_PATH, "_hj_units_plan_mutant", plan_src)
    m_v1 = _load_module(_V1_PATH, "_hj_units_v1_mutant", v1_src)
    m_probe = _load_module(_PROBE_PATH, "_hj_units_probe_mutant", probe_src)
    try:
        with torch.no_grad():
            mutant = _new_surfaces(m_v1, m_plan, m_probe)
            for name in _NEW_NAMES:
                param = _NEW_PARAM[name]
                call, fn = new_live[name]
                assert _omission_rejected_for(call, param), name + ": the live code must refuse an omission"
                mcall, mfn = mutant[name]
                assert not _omission_rejected_for(mcall, param), (
                    name + ": with the default restored the omission is ACCEPTED -- test (g) is RED")
                assert inspect.signature(mfn).parameters[param].default == "kappa"
                # the mutation really IS 'the old default': omitting == stating "kappa", bit for bit
                got, want = mcall(), mcall(**{param: "kappa"})
                assert _same(got, want), name
                # and the units still differ, so the silent default is not a no-op
                steer = mcall(**{param: "steer"})
                assert not _same(got, steer), name + ": the unit does not matter on this surface -- vacuous"
    finally:
        sys.modules.pop("_hj_units_plan_mutant", None)
        sys.modules.pop("_hj_units_v1_mutant", None)
        sys.modules.pop("_hj_units_probe_mutant", None)


def _same(a, b) -> bool:
    """Equality of whatever a surface returns: a tensor, a (tensor, dict) pair, a score dict, or a PlanResult."""
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, tuple):
        return torch.equal(a[0], b[0])
    if isinstance(a, dict):
        return all(torch.equal(a[k], b[k]) for k in a if isinstance(a[k], torch.Tensor))
    return torch.equal(a.controls, b.controls) and a.cost == b.cost and a.baseline_costs == b.baseline_costs


def _run_pytest_in_copy_v2(tmp: Path, mutate: str | None, select: str):
    """Like ``_run_pytest_in_copy`` but also copies ``stack/tanitad`` so the temp tree is SELF-CONTAINED
    (the mutant ``refa_v1*.py`` must be the ones the nested run imports). ``mutate`` is None | "new" | "cli" |
    a key of ``_CLI_MUTANTS`` ("actdiv" | "refcv3" | "openloop").
    Returns (returncode, combined output)."""
    tag = "clean" if mutate is None else "mutant_" + mutate
    root = tmp / tag
    tani = root / "taniteval"
    (tani / "tests").mkdir(parents=True)
    ign = shutil.ignore_patterns("__pycache__")
    shutil.copytree(_TANI / "taniteval", tani / "taniteval", ignore=ign)
    shutil.copytree(_TANI / "tools", tani / "tools", ignore=ign)
    shutil.copytree(_REPO / "stack" / "tanitad", root / "stack" / "tanitad", ignore=ign)
    shutil.copy2(_TANI / "conftest.py", tani / "conftest.py")
    shutil.copy2(Path(__file__), tani / "tests" / Path(__file__).name)
    if mutate == "new":
        plan_src, v1_src, probe_src = _mutated_new_sources()
        (root / "stack" / "tanitad" / "refs" / "refa_v1_plan.py").write_text(plan_src, encoding="utf-8", newline="\n")
        (root / "stack" / "tanitad" / "refs" / "refa_v1.py").write_text(v1_src, encoding="utf-8", newline="\n")
        (tani / "tools" / "cost_surface_probe.py").write_text(probe_src, encoding="utf-8", newline="\n")
    elif mutate == "cli":
        (tani / "tools" / "refav1_arm.py").write_text(_mutated_cli_source(), encoding="utf-8", newline="\n")
    elif mutate in _CLI_MUTANTS:
        (tani / "tools" / _CLI_MUTANTS[mutate][0]).write_text(_mutated_tool_source(mutate), encoding="utf-8", newline="\n")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["HJ_UNITS_NESTED"] = "1"                              # recursion guard
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "stack"), str(_REPO / "stack" / "scripts")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", select,
           str(tani / "tests" / Path(__file__).name)]
    p = subprocess.run(cmd, cwd=str(tani), env=env, capture_output=True, text=True, timeout=900)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def test_g7_mutation_in_a_temp_copy_turns_the_real_omission_check_RED(tmp_path):
    """The deliberate-regression arm, end to end, for the five HJ-UNITS-2 surfaces: in a SELF-CONTAINED
    temp copy restore ``= "kappa"`` and run this file's own (g) checks. The CLEAN copy is the control."""
    if os.environ.get("HJ_UNITS_NESTED") == "1":
        pytest.skip("nested run: this test is not selected inside its own subprocess")
    sel = "test_g_omitted or test_g2_unit"
    rc_clean, out_clean = _run_pytest_in_copy_v2(tmp_path, None, sel)
    assert rc_clean == 0, out_clean[-3000:]
    assert "10 passed" in out_clean, out_clean[-1500:]           # 5 x test_g_ + 5 x test_g2_
    rc_mut, out_mut = _run_pytest_in_copy_v2(tmp_path, "new", sel)
    assert rc_mut != 0, "the mutant tree still passed test (g):\n" + out_mut[-3000:]
    for name in _NEW_NAMES:
        assert f"test_g_omitted_units_raises_TypeError[{name}]" in out_mut, out_mut[-3000:]
        assert f"test_g2_unit_parameter_is_keyword_only_with_no_default[{name}]" in out_mut
    assert "10 failed" in out_mut, out_mut[-1500:]


# =========================================================================== #
# (h) the CLI flag, behaviourally                                              #
# =========================================================================== #
_CLI_NEEDLE = 'ap.add_argument("--action-units", choices=("kappa", "steer"), default=None,'
_ROLLOUT = ["--ckpt", "c.pt", "--cache", "x", "--episodes", "x", "--dump-dir", "d", "--out", "o.json"]


class _Reached(Exception):
    """Raised by the sentinel that replaces the first stage AFTER the unit gate."""


def _cli_source() -> str:
    return _ARM_PATH.read_text(encoding="utf-8")


def _mutated_cli_source() -> str:
    src = _cli_source()
    assert src.count(_CLI_NEEDLE) == 1, "refav1_arm --action-units declaration moved: update this mutation"
    return src.replace(_CLI_NEEDLE, 'ap.add_argument("--action-units", choices=("kappa", "steer"), default="kappa",')


def _cli_gate(mod, argv, monkeypatch):
    """Run ``mod.main(argv)`` with the stage after the unit gate replaced by a sentinel.
    Returns "REACHED" if the gate let it through, else the SystemExit code."""
    monkeypatch.setattr(mod, "preflight_analysis_imports", lambda *a, **k: (_ for _ in ()).throw(_Reached()))
    try:
        mod.main(argv)
    except _Reached:
        return "REACHED"
    except SystemExit as e:
        return e.code
    return "RETURNED"


def test_h_cli_rollout_without_units_is_refused_naming_both_legal_values(arm, capsys, monkeypatch):
    assert _cli_gate(arm, list(_ROLLOUT), monkeypatch) == 2
    err = capsys.readouterr().err
    assert "--action-units" in err and "kappa" in err and "steer" in err, err
    assert "required" in err.lower() or "REQUIRED" in err, err


def test_h2_cli_a_bad_value_is_refused_by_argparse(arm, capsys, monkeypatch):
    assert _cli_gate(arm, _ROLLOUT + ["--action-units", "radians"], monkeypatch) == 2
    err = capsys.readouterr().err
    assert "kappa" in err and "steer" in err, err


def test_h3_cli_a_stated_unit_passes_the_gate(arm, monkeypatch):
    for units in ("kappa", "steer"):
        assert _cli_gate(arm, _ROLLOUT + ["--action-units", units], monkeypatch) == "REACHED", units


def test_h4_cli_analyze_only_needs_no_unit(arm, monkeypatch):
    """``--analyze-only`` reads the unit from the dump manifest and never integrates a control."""
    assert _cli_gate(arm, ["--analyze-only", "dump", "--out", "o.json"], monkeypatch) == "REACHED"


def test_h5_cli_declaration_has_no_default_by_ast():
    tree = ast.parse(_cli_source())
    decl = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _callee_name(n) == "add_argument"
            and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--action-units"]
    assert len(decl) == 1, len(decl)
    kw = {k.arg: k.value for k in decl[0].keywords}
    assert isinstance(kw.get("default"), ast.Constant) and kw["default"].value is None, ast.dump(kw.get("default"))
    assert "kappa" in ast.unparse(kw["choices"]) and "steer" in ast.unparse(kw["choices"])


def test_h6_cli_mutation_in_process_restoring_the_default_lets_a_forgetful_launch_through(monkeypatch):
    live = _load_module(_ARM_PATH, "_hj_units_arm_cli_live")
    mutant = _load_module(_ARM_PATH, "_hj_units_arm_cli_mutant", _mutated_cli_source())
    try:
        assert _cli_gate(live, list(_ROLLOUT), monkeypatch) == 2
        # with the old default restored the forgetful launch line is ACCEPTED -- test (h) is RED
        assert _cli_gate(mutant, list(_ROLLOUT), monkeypatch) == "REACHED"
    finally:
        sys.modules.pop("_hj_units_arm_cli_live", None)
        sys.modules.pop("_hj_units_arm_cli_mutant", None)


def test_h7_cli_mutation_in_a_temp_copy_turns_the_real_check_RED(tmp_path):
    if os.environ.get("HJ_UNITS_NESTED") == "1":
        pytest.skip("nested run: this test is not selected inside its own subprocess")
    sel = "test_h_cli"
    rc_clean, out_clean = _run_pytest_in_copy_v2(tmp_path, None, sel)
    assert rc_clean == 0 and "1 passed" in out_clean, out_clean[-3000:]
    rc_mut, out_mut = _run_pytest_in_copy_v2(tmp_path, "cli", sel)
    assert rc_mut != 0 and "test_h_cli_rollout_without_units_is_refused_naming_both_legal_values" in out_mut, out_mut[-3000:]
    assert "1 failed" in out_mut, out_mut[-1500:]


# =========================================================================== #
# (i) the CLI census: live docs / launchers that invoke refav1_arm.py          #
# =========================================================================== #
_CLI_GLOBS = ["taniteval/tools/*.md", "taniteval/tools/*.sh", "taniteval/tools/*.ps1",
              "stack/scripts/*.sh", "stack/scripts/*.ps1", "tools/*.sh", "tools/*.ps1", "tools/*.md"]


def _logical_commands(text: str):
    """Yield (first line number, joined command text) for every shell command, joining ``\\`` continuations."""
    cur, start = [], 0
    for i, ln in enumerate(text.split("\n"), 1):
        if not cur:
            start = i
        stripped = ln.rstrip()
        if stripped.endswith("\\"):
            cur.append(stripped[:-1])
            continue
        cur.append(stripped)
        yield start, " ".join(c.strip() for c in cur)
        cur = []
    if cur:
        yield start, " ".join(c.strip() for c in cur)


def _cli_census_text(text: str, label: str) -> tuple[int, list[str]]:
    """(rollout invocations scanned, [those that omit --action-units]). An invocation is a logical command
    that runs ``refav1_arm.py`` under a python interpreter; ``--analyze-only`` / ``--help`` never read the unit."""
    scanned, bad = 0, []
    run = re.compile(r"(?:^|[\s/\\])(?:python[0-9.]*(?:\.exe)?|py|\$PY|\"\$PY\")\s+(?:\S+\s+)*?\S*refav1_arm\.py(?:\s|$)")
    for ln, cmd in _logical_commands(text):
        if "refav1_arm.py" not in cmd or not run.search(cmd):
            continue
        if "--analyze-only" in cmd or "--help" in cmd:
            continue
        scanned += 1
        if "--action-units" not in cmd:
            bad.append(f"{label}:{ln}")
    return scanned, bad


def test_i_cli_census_every_live_invocation_of_refav1_arm_states_the_unit():
    scanned, bad, files = 0, [], []
    for pat in _CLI_GLOBS:
        for path in sorted(_REPO.glob(pat)):
            label = str(path.relative_to(_REPO)).replace("\\", "/")
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            n, b = _cli_census_text(text, label)
            if n:
                files.append(label)
            scanned += n
            bad += b
    assert scanned >= 2, f"CLI census is vacuous: {scanned} invocations in {files}"
    assert not bad, "live invocation(s) of refav1_arm.py without --action-units:\n  " + "\n  ".join(bad)


def test_i2_cli_census_positive_and_negative_controls():
    omitted = "python taniteval/tools/refav1_arm.py \\\n  --ckpt a \\\n  --out b\n"
    stated = "python taniteval/tools/refav1_arm.py \\\n  --ckpt a \\\n  --action-units steer \\\n  --out b\n"
    analyze = "python taniteval/tools/refav1_arm.py --analyze-only d --out b\n"
    prose = "see refav1_arm.py for the flag list\n"
    thor = "/home/nvidia/venvs/tanitad-train/bin/python taniteval/tools/refav1_arm.py \\\n  --ckpt a\n"
    assert _cli_census_text(omitted, "ctl") == (1, ["ctl:1"])
    assert _cli_census_text(stated, "ctl") == (1, [])
    assert _cli_census_text(analyze, "ctl") == (0, [])                 # never reads the unit
    assert _cli_census_text(prose, "ctl") == (0, [])                   # selectivity
    assert _cli_census_text(thor, "ctl") == (1, ["ctl:1"])             # a full interpreter path is still seen


def test_i3_cli_census_mutation_removing_the_flag_from_a_REAL_doc_is_caught():
    path = _TANI / "tools" / "REFAV1_ARM.md"
    text = path.read_text(encoding="utf-8")
    n0, bad0 = _cli_census_text(text, "REFAV1_ARM.md")
    assert n0 >= 1 and not bad0, (n0, bad0)
    mutated = "\n".join(ln for ln in text.split("\n") if ln.strip() != "--action-units steer \\")
    assert mutated != text
    n1, bad1 = _cli_census_text(mutated, "REFAV1_ARM.md")
    assert n1 == n0 and len(bad1) == n0, (n1, bad1)


# =========================================================================== #
# INSTFIX / HJ-UNITS-3 -- the last silent unit defaults on the rollout CLIs     #
# =========================================================================== #
_ACTDIV_PATH = _TANI / "tools" / "actdiv_anchored.py"
_REFCV3_PATH = _TANI / "tools" / "refcv3_arm.py"
_OPENLOOP_PATH = _TANI / "tools" / "openloop_suite.py"
_RENDER_CV3_PATH = _TANI / "tools" / "render_refcv3_video.py"
_PANEL_PATH = _TANI / "tools" / "run_hierarchy_panel.py"
_INTERNAL_T1_PATH = _TANI / "taniteval" / "bench" / "internal_t1.py"

#: key -> (tool file, declaration as it stands, the same declaration with the OLD default restored)
_CLI_MUTANTS = {
    "actdiv": ("actdiv_anchored.py",
               'ap.add_argument("--action-units", default=None, choices=ACTION_UNITS_CHOICES,',
               'ap.add_argument("--action-units", default="kappa", choices=ACTION_UNITS_CHOICES,'),
    "refcv3": ("refcv3_arm.py",
               'ap.add_argument("--action-units", choices=("kappa", "steer"), default=None,',
               'ap.add_argument("--action-units", choices=("kappa", "steer"), default="steer",'),
    "openloop": ("openloop_suite.py",
                 'roll.add_argument("--action-units", choices=("kappa", "steer"), default=None,',
                 'roll.add_argument("--action-units", choices=("kappa", "steer"), default="steer",'),
}


def _mutated_tool_source(key: str) -> str:
    """The tool's source with its pre-INSTFIX ``--action-units`` default restored (the deliberate regression)."""
    fname, needle, old_default = _CLI_MUTANTS[key]
    src = (_TANI / "tools" / fname).read_text(encoding="utf-8")
    assert src.count(needle) == 1, f"{fname}: --action-units declaration moved: update this mutation"
    return src.replace(needle, old_default)


class _Stop(Exception):
    """Raised by the sentinel that replaces the first stage AFTER a unit gate."""


def _stop(*_a, **_k):
    raise _Stop()


def _gate(mod, argv, monkeypatch, sentinels):
    """Run ``mod.main(argv)`` with each named module-level stage after the gate replaced by a sentinel.
    Returns "REACHED" if the gate let the run through, the SystemExit code if it refused, else ("RETURNED", rv)."""
    for name, repl in sentinels.items():
        monkeypatch.setattr(mod, name, repl)
    try:
        rv = mod.main(argv)
    except _Stop:
        return "REACHED"
    except SystemExit as e:
        return e.code
    return ("RETURNED", rv)


# --------------------------------------------------------------------------- #
# (j) actdiv_anchored.py --action-units                                         #
# --------------------------------------------------------------------------- #
_ADIV_REFAV1 = ["--family", "refav1", "--arm", "a=x.pt", "--out", "o.json"]
_ADIV_SENT = {"refav1_main": _stop, "resolve_stack": _stop}


@pytest.fixture(scope="module")
def actdiv():
    mod = _load_module(_ACTDIV_PATH, "_instfix_actdiv_live")
    yield mod
    sys.modules.pop("_instfix_actdiv_live", None)


def test_j_actdiv_refav1_without_units_is_refused_naming_both_legal_values(actdiv, capsys, monkeypatch):
    assert _gate(actdiv, list(_ADIV_REFAV1), monkeypatch, _ADIV_SENT) == 2
    err = capsys.readouterr().err
    assert "--action-units" in err and "kappa" in err and "steer" in err, err
    assert "REQUIRED" in err and "R46" in err, err


def test_j2_actdiv_a_bad_value_is_refused_by_argparse(actdiv, capsys, monkeypatch):
    assert _gate(actdiv, _ADIV_REFAV1 + ["--action-units", "radians"], monkeypatch, _ADIV_SENT) == 2
    err = capsys.readouterr().err
    assert "kappa" in err and "steer" in err, err


def test_j3_actdiv_a_stated_unit_passes_the_gate(actdiv, monkeypatch):
    for units in ("kappa", "steer"):
        assert _gate(actdiv, _ADIV_REFAV1 + ["--action-units", units], monkeypatch, _ADIV_SENT) == "REACHED", units


def test_j4_actdiv_the_v7_family_and_selftest_are_not_asked_for_a_unit(actdiv, monkeypatch):
    """v7 never reads a unit (``steer`` is refused there: the encoding wheelbase does not travel); --selftest is synthetic."""
    sent = dict(_ADIV_SENT, selftest=lambda *a, **k: 0)
    assert _gate(actdiv, ["--family", "v7"], monkeypatch, sent) == "REACHED"                       # absent: fine
    assert _gate(actdiv, ["--family", "v7", "--action-units", "kappa"], monkeypatch, sent) == "REACHED"   # the old no-op
    refused = _gate(actdiv, ["--family", "v7", "--action-units", "steer"], monkeypatch, sent)
    assert isinstance(refused, str) and "refav1-only" in refused, refused
    assert _gate(actdiv, ["--selftest"], monkeypatch, sent) == ("RETURNED", 0)                     # no unit, no refusal


def test_j5_actdiv_declaration_has_no_default_by_ast(actdiv):
    decl = _declarations(_ACTDIV_PATH.read_text(encoding="utf-8"))
    assert len(decl) == 1, decl
    assert decl[0]["default"] == "<None>" and decl[0]["choices"] == "ACTION_UNITS_CHOICES", decl
    assert tuple(actdiv.ACTION_UNITS_CHOICES) == ("kappa", "steer"), actdiv.ACTION_UNITS_CHOICES    # what the name resolves to


def test_j6_actdiv_a_hand_built_namespace_cannot_slip_a_missing_unit_past_the_library(actdiv):
    """The CLI gate is not the only door: ``apply_action_units`` itself refuses None (naming both values)."""
    cands = actdiv.candidate_grid_abs((0.02, 0.1), (0.5,), actdiv.REFAV1_CHANNELS)
    with pytest.raises(ValueError) as ei:
        actdiv.apply_action_units(cands, None)
    assert "kappa" in str(ei.value) and "steer" in str(ei.value) and "R46" in str(ei.value), str(ei.value)
    out, prov = actdiv.apply_action_units(cands, "kappa")                                          # control: stated passes
    assert out is cands and prov["action_units"] == "kappa"


def test_j7_actdiv_mutation_in_process_restoring_the_default_lets_a_forgetful_launch_through(actdiv, monkeypatch):
    mutant = _load_module(_ACTDIV_PATH, "_instfix_actdiv_mutant", _mutated_tool_source("actdiv"))
    try:
        assert _gate(actdiv, list(_ADIV_REFAV1), monkeypatch, _ADIV_SENT) == 2
        assert _gate(mutant, list(_ADIV_REFAV1), monkeypatch, _ADIV_SENT) == "REACHED"           # RED: the silent default is back
    finally:
        sys.modules.pop("_instfix_actdiv_mutant", None)


def test_j8_actdiv_mutation_in_a_temp_copy_turns_the_real_check_RED(tmp_path):
    if os.environ.get("HJ_UNITS_NESTED") == "1":
        pytest.skip("nested run: this test is not selected inside its own subprocess")
    sel = "test_j_actdiv_refav1_without_units"
    rc_clean, out_clean = _run_pytest_in_copy_v2(tmp_path, None, sel)
    assert rc_clean == 0 and "1 passed" in out_clean, out_clean[-3000:]
    rc_mut, out_mut = _run_pytest_in_copy_v2(tmp_path, "actdiv", sel)
    assert rc_mut != 0 and "test_j_actdiv_refav1_without_units_is_refused_naming_both_legal_values" in out_mut, out_mut[-3000:]
    assert "1 failed" in out_mut, out_mut[-1500:]


# --------------------------------------------------------------------------- #
# (k) refcv3_arm.py and openloop_suite.py --action-units                        #
# --------------------------------------------------------------------------- #
_CV3_ROLLOUT = ["--ckpt", "c.pt", "--episodes", "x", "--labels", "l", "--dump-dir", "d", "--out", "o.json", "--no-lead-block"]
_CV3_SENT = {"run_dump": _stop, "analyze_refcv3": _stop}


@pytest.fixture(scope="module")
def refcv3():
    mod = _load_module(_REFCV3_PATH, "_instfix_refcv3_live")
    yield mod
    sys.modules.pop("_instfix_refcv3_live", None)


@pytest.fixture(scope="module")
def openloop():
    mod = _load_module(_OPENLOOP_PATH, "_instfix_openloop_live")
    yield mod
    sys.modules.pop("_instfix_openloop_live", None)


def test_k_refcv3_rollout_without_units_is_refused_naming_both_legal_values(refcv3, capsys, monkeypatch):
    assert _gate(refcv3, list(_CV3_ROLLOUT), monkeypatch, _CV3_SENT) == 2
    err = capsys.readouterr().err
    assert "--action-units" in err and "kappa" in err and "steer" in err and "REQUIRED" in err, err


def test_k2_refcv3_a_bad_value_is_refused_a_stated_value_and_analyze_only_pass(refcv3, capsys, monkeypatch):
    assert _gate(refcv3, _CV3_ROLLOUT + ["--action-units", "radians"], monkeypatch, _CV3_SENT) == 2
    assert "kappa" in capsys.readouterr().err
    for units in ("kappa", "steer"):
        assert _gate(refcv3, _CV3_ROLLOUT + ["--action-units", units], monkeypatch, _CV3_SENT) == "REACHED", units
    # --analyze-only reads the unit from the dump manifest and is not asked
    assert _gate(refcv3, ["--analyze-only", "dump", "--out", "o.json", "--no-lead-block"], monkeypatch, _CV3_SENT) == "REACHED"


def test_k3_refcv3_declaration_has_no_default_by_ast():
    decl = _declarations(_REFCV3_PATH.read_text(encoding="utf-8"))
    assert len(decl) == 1 and decl[0]["default"] == "<None>", decl


def test_k4_refcv3_mutation_in_process_restoring_the_default_lets_a_forgetful_launch_through(refcv3, monkeypatch):
    mutant = _load_module(_REFCV3_PATH, "_instfix_refcv3_mutant", _mutated_tool_source("refcv3"))
    try:
        assert _gate(refcv3, list(_CV3_ROLLOUT), monkeypatch, _CV3_SENT) == 2
        assert _gate(mutant, list(_CV3_ROLLOUT), monkeypatch, _CV3_SENT) == "REACHED"
    finally:
        sys.modules.pop("_instfix_refcv3_mutant", None)


def test_k5_refcv3_mutation_in_a_temp_copy_turns_the_real_check_RED(tmp_path):
    if os.environ.get("HJ_UNITS_NESTED") == "1":
        pytest.skip("nested run: this test is not selected inside its own subprocess")
    sel = "test_k_refcv3_rollout_without_units"
    rc_clean, out_clean = _run_pytest_in_copy_v2(tmp_path, None, sel)
    assert rc_clean == 0 and "1 passed" in out_clean, out_clean[-3000:]
    rc_mut, out_mut = _run_pytest_in_copy_v2(tmp_path, "refcv3", sel)
    assert rc_mut != 0 and "test_k_refcv3_rollout_without_units_is_refused_naming_both_legal_values" in out_mut, out_mut[-3000:]
    assert "1 failed" in out_mut, out_mut[-1500:]


def _ol_argv(tmp_path, *extra):
    return ["--out-dir", str(tmp_path / "out"), "--tag", "t", *extra]


def test_k6_openloop_rollout_without_units_is_refused_the_other_sources_are_not_asked(openloop, capsys, monkeypatch, tmp_path):
    sent = {"acquire_record": _stop}
    assert _gate(openloop, _ol_argv(tmp_path, "--ckpt", "c.pt", "--episodes", "x"), monkeypatch, sent) == 2
    err = capsys.readouterr().err
    assert "--action-units" in err and "kappa" in err and "steer" in err and "REQUIRED" in err, err
    for units in ("kappa", "steer"):
        assert _gate(openloop, _ol_argv(tmp_path, "--ckpt", "c.pt", "--episodes", "x", "--action-units", units),
                     monkeypatch, sent) == "REACHED", units
    assert _gate(openloop, _ol_argv(tmp_path, "--analyze-only", "dump"), monkeypatch, sent) == "REACHED"      # zero GPU
    assert _gate(openloop, _ol_argv(tmp_path, "--arm-json", "rec.json"), monkeypatch, sent) == "REACHED"      # no rollout
    assert _gate(openloop, _ol_argv(tmp_path, "--ckpt", "c.pt", "--episodes", "x", "--action-units", "radians"),
                 monkeypatch, sent) == 2


def test_k7_openloop_declaration_has_no_default_by_ast_and_the_default_is_the_mutation_target():
    decl = _declarations(_OPENLOOP_PATH.read_text(encoding="utf-8"))
    assert len(decl) == 1 and decl[0]["default"] == "<None>", decl
    mutant = _declarations(_mutated_tool_source("openloop"))
    assert len(mutant) == 1 and mutant[0]["default"] == "'steer'", mutant


def test_k8_openloop_mutation_in_process_and_in_a_temp_copy(monkeypatch, tmp_path):
    mutant = _load_module(_OPENLOOP_PATH, "_instfix_openloop_mutant", _mutated_tool_source("openloop"))
    try:
        sent = {"acquire_record": _stop}
        assert _gate(mutant, _ol_argv(tmp_path, "--ckpt", "c.pt", "--episodes", "x"), monkeypatch, sent) == "REACHED"
    finally:
        sys.modules.pop("_instfix_openloop_mutant", None)
    if os.environ.get("HJ_UNITS_NESTED") == "1":
        return
    sel = "test_k6_openloop_rollout_without_units"
    rc_clean, out_clean = _run_pytest_in_copy_v2(tmp_path / "tc", None, sel)
    assert rc_clean == 0 and "1 passed" in out_clean, out_clean[-3000:]
    rc_mut, out_mut = _run_pytest_in_copy_v2(tmp_path / "tc", "openloop", sel)
    assert rc_mut != 0 and "test_k6_openloop_rollout_without_units_is_refused_the_other_sources_are_not_asked" in out_mut, out_mut[-3000:]


# --------------------------------------------------------------------------- #
# (l) the DECLARATION CENSUS: every live `--action-units` declaration           #
# --------------------------------------------------------------------------- #
def _declarations(src: str) -> list[dict]:
    """Every ``<x>.add_argument("--action-units", ...)`` in the source: {line, default, required, choices}.
    ``default`` is ``'<None>'`` for ``default=None`` OR no ``default=`` at all with ``required=True``; ``'<implicit None>'`` for
    no default and not required; else the unparsed default expression."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and _callee_name(node) == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--action-units"):
            kw = {k.arg: k.value for k in node.keywords}
            if "default" in kw:
                d = kw["default"]
                default = "<None>" if (isinstance(d, ast.Constant) and d.value is None) else ast.unparse(d)
            else:
                req = kw.get("required")
                default = "<None>" if (isinstance(req, ast.Constant) and req.value is True) else "<implicit None>"
            out.append({"line": node.lineno, "default": default,
                        "choices": ast.unparse(kw["choices"]) if "choices" in kw else None})
    return out


#: Files allowed to declare ``--action-units`` WITH a default, and WHY (checked below, not asserted by the prose).
_DEAD_FLAG = {"taniteval/tools/render_refcv3_video.py"}
_DECL_ROOTS = ["stack/tanitad", "stack/scripts", "taniteval/taniteval", "taniteval/tools", "tools"]


def _declaration_census(files: dict | None = None):
    """({label: [declarations]}, [offences]).  ``files`` overrides the live scan (the controls)."""
    found, bad = {}, []
    if files is None:
        files = {}
        for rel in _DECL_ROOTS:
            root = _REPO / rel
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.py")):
                if set(path.relative_to(_REPO).parts) & {"tests", "results", "incoming", "__pycache__"}:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                if "--action-units" in text:
                    files[str(path.relative_to(_REPO)).replace("\\", "/")] = text
    for label, text in sorted(files.items()):
        try:
            decl = _declarations(text)
        except SyntaxError as ex:                                    # unverifiable == flagged, never skipped
            bad.append(f"{label}: UNPARSEABLE ({ex.msg} line {ex.lineno})")
            continue
        if decl:
            found[label] = decl
        for d in decl:
            if d["default"] != "<None>" and label not in _DEAD_FLAG:
                bad.append(f"{label}:{d['line']} --action-units default={d['default']}")
    return found, bad


def test_l_census_every_live_action_units_declaration_has_no_default():
    found, bad = _declaration_census()
    assert not bad, "live --action-units declaration(s) with a silent default:\n  " + "\n  ".join(bad)
    # the census must SEE the four rollout CLIs it exists for (a census that scans 0 passes vacuously)
    for must in ("taniteval/tools/refav1_arm.py", "taniteval/tools/actdiv_anchored.py",
                 "taniteval/tools/refcv3_arm.py", "taniteval/tools/openloop_suite.py"):
        assert must in found, (must, sorted(found))
    assert len(found) >= 5, sorted(found)


def test_l2_census_positive_and_negative_controls():
    bad_src = 'ap.add_argument("--action-units", choices=("kappa", "steer"), default="steer")\n'
    ok_src = 'ap.add_argument("--action-units", choices=("kappa", "steer"), default=None)\n'
    req_src = 'ap.add_argument("--action-units", choices=("kappa", "steer"), required=True)\n'
    implicit = 'ap.add_argument("--action-units", choices=("kappa", "steer"))\n'
    other = 'ap.add_argument("--kin-mode", default="steer")\nap.add_argument("--action-units-note", default="kappa")\n'
    assert _declaration_census({"ctl/bad.py": bad_src})[1] == ["ctl/bad.py:1 --action-units default='steer'"]
    assert _declaration_census({"ctl/ok.py": ok_src, "ctl/req.py": req_src})[1] == []
    assert _declaration_census({"ctl/implicit.py": implicit})[1] == ["ctl/implicit.py:1 --action-units default=<implicit None>"]
    assert _declaration_census({"ctl/other.py": other}) == ({}, [])                  # selectivity: other flags are not claimed
    assert len(_declaration_census({"ctl/broken.py": "def (:\n"})[1]) == 1            # unparseable == flagged


def test_l3_census_mutation_restoring_a_default_in_each_REAL_file_is_caught():
    for key in _CLI_MUTANTS:
        rel = "taniteval/tools/" + _CLI_MUTANTS[key][0]
        found0, bad0 = _declaration_census({rel: (_REPO / rel).read_text(encoding="utf-8")})
        assert rel in found0 and not bad0, (rel, found0, bad0)
        found1, bad1 = _declaration_census({rel: _mutated_tool_source(key)})
        assert len(bad1) == 1 and bad1[0].startswith(rel), (key, bad1)
    arm_rel = "taniteval/tools/refav1_arm.py"
    arm_src = _mutated_cli_source()
    assert len(_declaration_census({arm_rel: arm_src})[1]) == 1


def test_l4_the_one_default_that_remains_is_a_flag_nothing_reads():
    """``render_refcv3_video.py --action-units`` keeps ``default="steer"`` because the flag is DEAD there: the namespace
    attribute is never read in the file, and the only function in ``refcv3_arm.py`` that consumes a unit (``run_dump``) is never
    called from it.  Prove both from the AST, so the exception cannot outlive its reason."""
    render = ast.parse(_RENDER_CV3_PATH.read_text(encoding="utf-8"))
    reads = [n.lineno for n in ast.walk(render) if isinstance(n, ast.Attribute) and n.attr == "action_units"]
    assert reads == [], f"render_refcv3_video.py now READS action_units at {reads}: it must drop the default"
    strs = [n.value for n in ast.walk(render) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert strs.count("action_units") == 0
    arm = ast.parse(_REFCV3_PATH.read_text(encoding="utf-8"))
    consumers = set()
    for fn in ast.walk(arm):
        if isinstance(fn, ast.FunctionDef):
            if any(isinstance(n, ast.Attribute) and n.attr == "action_units" for n in ast.walk(fn)):
                consumers.add(fn.name)
    assert consumers == {"main", "run_dump"}, consumers                  # main = the gate; run_dump = the one integrator
    called_from_render = {n.func.attr for n in ast.walk(render)
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                          and isinstance(n.func.value, ast.Name) and n.func.value.id == "arm"}
    assert "run_dump" not in called_from_render and "main" not in called_from_render, sorted(called_from_render)
    assert len(called_from_render) >= 3, sorted(called_from_render)       # the scan sees the arm calls (not vacuous)
    # mutation: make render READ the flag -> the premise breaks
    mutated = ast.parse(_RENDER_CV3_PATH.read_text(encoding="utf-8") + "\n_x = a.action_units\n")
    assert [n for n in ast.walk(mutated) if isinstance(n, ast.Attribute) and n.attr == "action_units"]


# --------------------------------------------------------------------------- #
# (m) the invocation census, generalised; and the two live callers              #
# --------------------------------------------------------------------------- #
#: script -> (flags that mean "this invocation never reads the unit", scope regex or None)
_CLI_SPECS = {
    "refav1_arm.py": (("--analyze-only", "--help"), None),
    "refcv3_arm.py": (("--analyze-only", "--help"), None),
    "openloop_suite.py": (("--analyze-only", "--arm-json", "--help"), None),
    "actdiv_anchored.py": (("--help", "--selftest"), r"--family[ =]refav1"),
}
_CLI_GLOBS_M = ["taniteval/tools/*.md", "taniteval/tools/*.sh", "taniteval/tools/*.ps1", "taniteval/tools/*.cmd",
                "stack/scripts/*.sh", "stack/scripts/*.ps1", "stack/scripts/*.cmd", "tools/*.sh", "tools/*.ps1",
                "tools/*.md", "tools/*.cmd"]


def _cli_census_script(text: str, label: str, script: str) -> tuple[int, list[str]]:
    """(unit-reading invocations of ``script`` scanned, [those that omit --action-units]).  An invocation is a logical
    command (``\\`` / backtick / ``^`` continuations joined) that runs the script under a python interpreter."""
    exempt, scope = _CLI_SPECS[script]
    run = re.compile(r"(?:^|[\s/\\\"'&])(?:python[0-9.]*(?:\.exe)?|py|\$PY|\"\$PY\"|%PY%|\"%PY%\"|\$py)\s+(?:\S+\s+)*?[\"']?\S*"
                     + re.escape(script) + r"[\"']?(?:\s|$)")
    amp = re.compile(r"&\s*\"?[^\"\n]*" + re.escape(script))
    scanned, bad = 0, []
    cur, start, cmds = [], 0, []
    for i, ln in enumerate(text.split("\n"), 1):
        if not cur:
            start = i
        st = ln.rstrip()
        if st.endswith("\\") or st.endswith("`") or st.endswith("^"):
            cur.append(st[:-1])
            continue
        cur.append(st)
        cmds.append((start, " ".join(c.strip() for c in cur)))
        cur = []
    if cur:
        cmds.append((start, " ".join(c.strip() for c in cur)))
    for ln, cmd in cmds:
        if script not in cmd or not (run.search(cmd) or amp.search(cmd)):
            continue
        if scope and not re.search(scope, cmd):
            continue
        if any(x in cmd for x in exempt):
            continue
        scanned += 1
        if "--action-units" not in cmd:
            bad.append(f"{label}:{ln}")
    return scanned, bad


def test_m_cli_census_every_live_invocation_of_the_four_rollout_CLIs_states_the_unit():
    scanned, bad, seen_files = {s: 0 for s in _CLI_SPECS}, [], 0
    for pat in _CLI_GLOBS_M:
        for path in sorted(_REPO.glob(pat)):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            seen_files += 1
            label = str(path.relative_to(_REPO)).replace("\\", "/")
            for script in _CLI_SPECS:
                n, b = _cli_census_script(text, label, script)
                scanned[script] += n
                bad += [f"{x} ({script})" for x in b]
    assert seen_files >= 5, f"census is blind: it read {seen_files} files"
    assert scanned["refav1_arm.py"] >= 2 and scanned["refcv3_arm.py"] >= 2 and scanned["openloop_suite.py"] >= 2, scanned
    assert not bad, "live invocation(s) without --action-units:\n  " + "\n  ".join(bad)


def test_m2_cli_census_controls_use_the_real_frozen_forms():
    """Positive / negative controls built from the FORMS the frozen callers use (a .cmd with %PY%, a .ps1 with & and backticks)."""
    cmd_form = ('"%PY%" taniteval\\tools\\actdiv_anchored.py --family refav1 --arm "a=C:\\x\\ckpt.pt" --n-perm 200 --out "%RAW%\\o.json"\n')
    ps1_form = ('& $py "$wt\\taniteval\\tools\\actdiv_anchored.py" `\n    --family refav1 `\n    --arm "a=C:\\x" `\n    --out $out\n')
    ps1_ok = ps1_form.replace("--family refav1 `", "--family refav1 `\n    --action-units steer `")
    v7 = '"%PY%" taniteval\\tools\\actdiv_anchored.py --assets "%AS%" --arms a,b --out o.json\n'
    sh_form = "python taniteval/tools/actdiv_anchored.py --family refav1 \\\n  --arm a=b \\\n  --out o.json\n"
    assert _cli_census_script(cmd_form, "ctl.cmd", "actdiv_anchored.py") == (1, ["ctl.cmd:1"])
    assert _cli_census_script(ps1_form, "ctl.ps1", "actdiv_anchored.py") == (1, ["ctl.ps1:1"])
    assert _cli_census_script(ps1_ok, "ctl.ps1", "actdiv_anchored.py") == (1, [])
    assert _cli_census_script(sh_form, "ctl.sh", "actdiv_anchored.py") == (1, ["ctl.sh:1"])
    assert _cli_census_script(v7, "ctl.cmd", "actdiv_anchored.py") == (0, [])                      # v7: no unit exists
    assert _cli_census_script("see actdiv_anchored.py --family refav1 for details\n", "ctl.md", "actdiv_anchored.py") == (0, [])
    roll = "python taniteval/tools/refcv3_arm.py \\\n  --ckpt a \\\n  --out b\n"
    assert _cli_census_script(roll, "ctl", "refcv3_arm.py") == (1, ["ctl:1"])
    assert _cli_census_script(roll.replace("--ckpt a", "--ckpt a --action-units steer"), "ctl", "refcv3_arm.py") == (1, [])
    assert _cli_census_script("python taniteval/tools/refcv3_arm.py --analyze-only d --out b\n", "ctl", "refcv3_arm.py") == (0, [])
    suite = "python taniteval/tools/openloop_suite.py --arm-json r.json --out-dir o\n"
    assert _cli_census_script(suite, "ctl", "openloop_suite.py") == (0, [])                        # no rollout
    assert _cli_census_script(suite.replace("--arm-json r.json", "--ckpt c --episodes e"), "ctl", "openloop_suite.py") == (1, ["ctl:1"])


def test_m3_cli_census_mutation_removing_the_flag_from_a_REAL_doc_is_caught():
    for rel, script in (("taniteval/tools/REFCV3_ARM.md", "refcv3_arm.py"), ("taniteval/tools/OPENLOOP_SUITE.md", "openloop_suite.py")):
        text = (_REPO / rel).read_text(encoding="utf-8")
        n0, bad0 = _cli_census_script(text, rel, script)
        assert n0 >= 2 and not bad0, (rel, n0, bad0)
        mutated = re.sub(r"--action-units\s+(?:steer|kappa)\s*", "", text)
        assert mutated != text
        n1, bad1 = _cli_census_script(mutated, rel, script)
        assert n1 == n0 and len(bad1) == n0, (rel, n0, n1, bad1)


def test_m4_openloop_suite_docstring_usage_states_the_unit():
    """The module docstring's `--ckpt` usage example is a live invocation the .md globs do not reach."""
    doc = ast.get_docstring(ast.parse(_OPENLOOP_PATH.read_text(encoding="utf-8")))
    n, bad = _cli_census_script(doc, "openloop_suite.py[docstring]", "openloop_suite.py")
    assert n == 1 and not bad, (n, bad)
    n2, bad2 = _cli_census_script(doc.replace("--action-units steer", ""), "openloop_suite.py[docstring]", "openloop_suite.py")
    assert n2 == 1 and len(bad2) == 1, (n2, bad2)


def _import_internal_t1():
    """Import the bench module under its package name (it uses a relative import)."""
    import importlib
    return importlib.import_module("taniteval.bench.internal_t1")


def test_m5_live_caller_internal_t1_rollout_states_the_unit_and_analyze_only_does_not(tmp_path):
    import argparse
    T = _import_internal_t1()
    a = argparse.Namespace(ckpt="C:/x/ckpt.pt", episodes="C:/cache", labels="C:/l.jsonl.gz", config=None, grid="2s",
                           episodes_n=0, window_stride=0, analyze_only=None, n_boot=10, seed=0, t1_args="")
    cmd = T.build_cmd(a, tmp_path, "cpu", tmp_path / "dump")
    assert cmd[cmd.index("--action-units") + 1] == "steer" and cmd.count("--action-units") == 1, cmd
    a.t1_args = "--action-units kappa"                                                  # an operator override rides LAST and wins
    cmd = T.build_cmd(a, tmp_path, "cpu", tmp_path / "dump")
    assert cmd[-1] == "kappa" and cmd[-2] == "--action-units", cmd
    a.t1_args, a.analyze_only = "", str(tmp_path / "banked")
    assert "--action-units" not in T.build_cmd(a, tmp_path, "cpu", tmp_path / "dump")
    # mutation: strip the literal from the real source and the AST probe (below) sees it gone
    src = _INTERNAL_T1_PATH.read_text(encoding="utf-8")
    assert src.count('"--action-units", "steer"') == 1
    assert '"--action-units"' not in src.replace('"--action-units", "steer"', "")


def test_m6_live_caller_run_hierarchy_panel_rollout_states_the_unit_and_the_recover_branch_does_not(tmp_path, monkeypatch):
    import argparse
    panel = _load_module(_PANEL_PATH, "_instfix_panel_live")
    try:
        cmds = []
        monkeypatch.setattr(panel, "run", lambda cmd, dry: cmds.append(list(cmd)) or 0)
        monkeypatch.setattr(panel, "record_ok", lambda path: (False, "absent"))
        a = argparse.Namespace(workdir=str(tmp_path), ckpt="c.pt", episodes="e", labels="l", episodes_n=0)
        panel.roll("FULL", ["--with-navflip"], a, dry=False)                           # a ROLLOUT: no dump exists yet
        assert len(cmds) == 1 and cmds[0][cmds[0].index("--action-units") + 1] == "steer", cmds
        assert "--analyze-only" not in cmds[0] and "--with-navflip" in cmds[0], cmds[0]
        dump = tmp_path / "dump_FULL"
        dump.mkdir()
        (dump / "ep000.npz").write_bytes(b"x")
        cmds.clear()
        panel.roll("FULL", ["--with-navflip"], a, dry=False)                           # the dump exists: RE-ANALYSE, never re-roll
        assert len(cmds) == 1 and "--analyze-only" in cmds[0] and "--action-units" not in cmds[0], cmds
    finally:
        sys.modules.pop("_instfix_panel_live", None)
