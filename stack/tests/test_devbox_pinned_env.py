"""The dev-box pinned driver (`stack/ops/r9e1_gate3/r9gate3_devbox_pinned.sh`) runs the launch gate's G-SUITE-PINNED with MSYS_NO_PATHCONV UNSET.

MEASURED 2026-11-03 (MM 207ebd): the driver exports MSYS_NO_PATHCONV=1 for its own ssh / scp of Thor paths, and the export reached pytest:
`tests/test_r9_e1_floor.py` failed 6 / 28 in the C0 pinned run at 5761cd2 -- every `r9e1_read.sh` dry run handed native python an unconverted
/f/... path ("can't open ...r9e1_read_state.py: [Errno 2]"). The same test in the same archive passes without the variable and fails with it,
and the whole file reads 28 / 28 on a fresh tree without it. The gate call therefore runs under `env -u MSYS_NO_PATHCONV`; the driver's own
ssh / scp keep it. A dev-box operator environment must never leak into the suite the gate judges.
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

DRIVER = ROOT / "ops" / "r9e1_gate3" / "r9gate3_devbox_pinned.sh"


def _gate_lines(src: str) -> list:
    return [l for l in src.splitlines() if '/stack/scripts/launch_gate.py" run' in l and not l.lstrip().startswith("#")]


def test_the_pinned_gate_runs_with_msys_no_pathconv_unset():
    src = DRIVER.read_text(encoding="utf-8")
    gl = _gate_lines(src)
    assert len(gl) == 1, gl                                                    # ONE gate invocation in the driver
    line = gl[0]
    assert "env -u MSYS_NO_PATHCONV" in line, line
    assert line.index("env -u MSYS_NO_PATHCONV") < line.index('"$DEVPY"'), line   # the unset wraps the gate's python, not a later command


def test_the_driver_still_exports_it_for_its_own_ssh_and_scp():
    src = DRIVER.read_text(encoding="utf-8")
    assert "export MSYS_NO_PATHCONV=1" in src                                  # the Thor paths in its ssh / scp must stay unconverted
    assert any(l.lstrip().startswith(("ssh ", "scp ")) or " ssh -n" in l or "scp -q" in l for l in src.splitlines())


def test_a_mutated_driver_without_the_unset_is_caught():
    src = DRIVER.read_text(encoding="utf-8")
    bad = src.replace("env -u MSYS_NO_PATHCONV ", "")
    assert bad != src
    assert not any("env -u MSYS_NO_PATHCONV" in l for l in _gate_lines(bad))


@R2.needs_bash
def test_env_u_really_removes_the_variable_for_the_child():
    out = subprocess.run([R2.BASH, "-c", 'export MSYS_NO_PATHCONV=1; env -u MSYS_NO_PATHCONV bash -c \'echo "[${MSYS_NO_PATHCONV-unset}]"\''],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[unset]", out.stdout
