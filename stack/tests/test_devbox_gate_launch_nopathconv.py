"""The dev-box gate launches hand native python a THOR-path --path-map LITERALLY and keep MSYS_NO_PATHCONV out of the gate's environment.

MEASURED 2026-11-28 11:28Z (MM 4a3d05, D1 chain af1ce565): the dev-box complement refused in 74 s with
  GATE ERROR: --path-map FROM 'C:\\Program Files\\Git\\home\\nvidia\\r8ns\\r9e3\\runs\\r9e3-K2DC02' matches no data path of the argv
`env -u MSYS_NO_PATHCONV` (there since 2026-11-03 so the variable never reaches pytest) re-enabled MSYS argument conversion for the gate's own
exec. No earlier arm carried a --path-map, so nothing showed it, and test_r9gate3_study_e3's PMAP test echoes through a bash BUILTIN, which
MSYS never converts. These tests cross the real exec boundary: Git bash -> native python, through each script's OWN shim line.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import test_launch_gate_e1rig_e1rec2 as R2  # noqa: E402

GATE3 = ROOT / "ops" / "r9e1_gate3"
SCRIPTS = ("r9gate3_devbox_complement.sh", "r9gate3_devbox_pinned.sh")
PMAP = "/home/nvidia/r8ns/r9e3/runs/r9e3-K2DC02=F:/refcv9_e3/gate/d1_src"     # D1's own map (F:/refcv9_e3/gate/pathmap_K2DC02_d1.txt)
PROBE = "import os, sys; print(repr(sys.argv[1:])); print(os.environ.get('MSYS_NO_PATHCONV', '<absent>'))"
WIN = pytest.mark.skipif(sys.platform != "win32", reason="MSYS argument conversion exists only under Git bash on Windows")


def _src(name: str) -> str:
    return (GATE3 / name).read_text(encoding="utf-8")


def _shim_line(name: str) -> str:
    ls = [l.strip() for l in _src(name).splitlines() if l.strip().startswith("NOPC_SHIM=")]
    assert len(ls) == 1, (name, ls)
    return ls[0]


def _gate_line(name: str) -> str:
    ls = [l for l in _src(name).splitlines() if '/stack/scripts/launch_gate.py" run' in l and not l.lstrip().startswith("#")]
    assert len(ls) == 1, (name, ls)
    return ls[0]


def _run(body: str) -> list[str]:
    r = subprocess.run([R2.BASH, "-c", "export MSYS_NO_PATHCONV=1\n" + body, "_", Path(sys.executable).as_posix(), PMAP, PROBE],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0, r.stderr[-600:]
    return r.stdout.strip().splitlines()[-2:]


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_gate_starts_through_the_shim_and_never_under_env_u(name):
    line, src = _gate_line(name), _src(name)
    assert '"$DEVPY" -I -c "$NOPC_SHIM" ' in line and "env -u MSYS_NO_PATHCONV" not in line, line
    assert "export MSYS_NO_PATHCONV=1" in src and src.index("NOPC_SHIM=") < src.index(line)


@R2.needs_bash
@pytest.mark.parametrize("name", SCRIPTS)
def test_END_TO_END_the_scripts_own_shim_passes_the_thor_map_literally_and_drops_the_variable(name):
    argv, nopc = _run(_shim_line(name) + '\n"$1" -I -c "$NOPC_SHIM" -c "$3" --path-map "$2"')
    assert argv == repr(["--path-map", PMAP]), argv
    assert nopc == "<absent>", nopc                                         # it never reaches the gate, so never pytest


@R2.needs_bash
@pytest.mark.parametrize("name", SCRIPTS)
def test_REGRESSION_ARM_a_shim_that_does_not_pop_leaks_the_variable(name):
    bad = _shim_line(name).replace('os.environ.pop("MSYS_NO_PATHCONV", None); ', "")
    assert bad != _shim_line(name)
    argv, nopc = _run(bad + '\n"$1" -I -c "$NOPC_SHIM" -c "$3" --path-map "$2"')
    assert nopc == "1", nopc


@WIN
@R2.needs_bash
def test_REGRESSION_ARM_env_u_rewrites_the_thor_map_the_measured_af1c_failure():
    argv, nopc = _run('env -u MSYS_NO_PATHCONV "$1" -c "$3" --path-map "$2"')
    assert argv != repr(["--path-map", PMAP]) and "Program Files" in argv, argv   # MSYS rewrote /home/nvidia/... under the Git root
    assert nopc == "<absent>", nopc
