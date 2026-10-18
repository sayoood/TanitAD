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

ASCII-only assertion text: this dev box's console is cp1252.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import math
import os
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

from taniteval import floors_kin as fk                                       # noqa: E402
from tanitad.models.kinematic import STEER_WHEELBASE_M                       # noqa: E402

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
_CALLEES = {
    "paths_from_controls": {"action_units"},
    "ha0_ext_paths": {"action_units"},
    "const_control_paths": {"action_units", "steer_units"},
    "integrate_select": {"action_units"},
}
_LIVE_ROOTS = ["stack/tanitad", "stack/scripts", "taniteval/taniteval", "taniteval/tools", "tools"]


def _callee_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _census_source(src: str, label: str) -> tuple[int, list[str]]:
    """(calls scanned, [offending call sites]). A call with ``**kwargs`` cannot be verified -> flagged."""
    tree = ast.parse(src)
    scanned, bad = 0, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node)
        if name not in _CALLEES:
            continue
        scanned += 1
        kws = {k.arg for k in node.keywords}
        if None in kws or not (kws & _CALLEES[name]):
            bad.append(f"{label}:{node.lineno} {name}(...)")
    return scanned, bad


def _census_repo() -> tuple[int, list[str], list[str]]:
    scanned, bad, files = 0, [], []
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
                n, b = _census_source(src, label)
            except SyntaxError as ex:                     # unverifiable == flagged, never skipped
                bad.append(f"{label}: UNPARSEABLE ({ex.msg} line {ex.lineno})")
                continue
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
