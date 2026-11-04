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
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[unset]", out.stdout


# --- the suite-work archive (MM 2026-11-03): G-SUITE-PINNED extracts a ~12.5 GB full-commit archive per arm into gsw_<ARM> and the driver never
# deleted one -- F: fell 120 -> 43 GB in ~2.5 h. A SHIPPED arm now queues its own for deletion (PENDING) and purges it; a residue is retried
# before every later arm and at the end (gsw_B2 took 3 passes over ~7 min); a FAILed arm keeps its own for diagnosis; every arm first waits
# for room on the stage drive and is refused by name (FAILZZ DISK-LOW) rather than run a suite into a full disk.

def _lines(src: str) -> list:
    return [l for l in src.splitlines() if not l.lstrip().startswith("#")]


def _loop_span(lines: list) -> tuple:
    i = next(k for k, l in enumerate(lines) if l.startswith("for ARM in "))
    j = next(k for k in range(i + 1, len(lines)) if lines[k] == "done")
    return i, j


def _queues_after_ok(src: str) -> bool:
    lines = _lines(src)
    i, j = _loop_span(lines)
    body = lines[i + 1:j]
    ok = [k for k, l in enumerate(body) if 'mark "ZZGS3-PINNED-${ARM}-OKZZ' in l]
    q = [k for k, l in enumerate(body) if 'PENDING="$PENDING $SW"' in l]
    return len(ok) == 1 and len(q) == 1 and q[0] > ok[0] and "purge" in body[q[0]]


def test_a_shipped_arm_queues_and_purges_its_suite_work_after_the_ok_mark():
    src = DRIVER.read_text(encoding="utf-8")
    assert _queues_after_ok(src)
    fn = [l for l in _lines(src) if l.startswith("purge() {")]
    assert len(fn) == 1 and 'rm -rf "${d:?}"' in fn[0], fn


def test_a_mutated_driver_that_keeps_or_predeletes_the_archive_is_caught():
    src = DRIVER.read_text(encoding="utf-8")
    kept = "\n".join(l for l in src.splitlines() if 'PENDING="$PENDING $SW"' not in l)
    assert kept != src and not _queues_after_ok(kept)                           # never queued: the F: fill
    lines = src.splitlines()
    r = next(k for k, l in enumerate(lines) if 'PENDING="$PENDING $SW"' in l)
    q_line = lines.pop(r)
    lines.insert(next(k for k, l in enumerate(lines) if l.startswith('  case "$PST"')), q_line)
    assert not _queues_after_ok("\n".join(lines))                               # queued before the verdict: a FAILed arm loses its own


def test_every_arm_purges_then_waits_for_disk_before_the_gate_extracts():
    lines = _lines(DRIVER.read_text(encoding="utf-8"))
    i, j = _loop_span(lines)
    body = lines[i + 1:j]
    gate = next(k for k, l in enumerate(body) if '/stack/scripts/launch_gate.py" run' in l)
    purge = [k for k, l in enumerate(body) if l.strip().startswith("purge;")]
    refuse = [k for k, l in enumerate(body) if "FAILZZ DISK-LOW" in l and "continue" in l]
    mk = [k for k, l in enumerate(body) if l.strip() == 'mkdir -p "$SW"']
    assert len(purge) == 1 and len(refuse) == 1 and len(mk) == 1, (purge, refuse, mk)
    assert purge[0] < refuse[0] < mk[0] < gate                                  # free earlier residue, refuse, else make the dir, then extract
    tail = lines[j + 1:]
    fin = next(k for k, l in enumerate(tail) if "purge" in l and "seq" in l)
    assert fin < next(k for k, l in enumerate(tail) if "ZZGS3-PINNED-ALLDONEZZ" in l)  # a bounded final purge before ALLDONE


def _purge_defs() -> str:
    src = DRIVER.read_text(encoding="utf-8")
    return "\n".join(l for l in src.splitlines() if l.startswith(('PENDING=""', "purge() {")))   # the driver's OWN definitions


@R2.needs_bash
def test_purge_deletes_queued_dirs_and_keeps_a_residue_queued(tmp_path):
    a, b = tmp_path / "gsw_A", tmp_path / "gsw_B"
    for d in (a, b):
        (d / "x").mkdir(parents=True)
        (d / "x" / "f.py").write_text("1")
    (a / "x" / "f.py").chmod(0o444)                                             # a read-only file, as in the archive
    script = _purge_defs() + '\nPENDING="$PENDING $1 $2"; purge; echo "[${PENDING# }]"'
    out = subprocess.run([R2.BASH, "-c", script, "_", a.as_posix(), b.as_posix()], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]" and not a.exists() and not b.exists(), (out.stdout, a.exists(), b.exists())
    (a / "x").mkdir(parents=True)
    stuck = _purge_defs() + '\nrm() { return 1; }\nPENDING=" $1"; purge; echo "[${PENDING# }]"'   # an rm that cannot delete (a held tree)
    out = subprocess.run([R2.BASH, "-c", stuck, "_", a.as_posix()], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == f"[{a.as_posix()}]" and a.exists(), out.stdout   # the residue stays queued for the next pass


@R2.needs_bash
def test_free_gb_reads_an_integer_on_this_platform(tmp_path):
    src = DRIVER.read_text(encoding="utf-8")
    fn = next(l for l in src.splitlines() if l.startswith("free_gb() {"))      # the driver's OWN definition, executed
    out = subprocess.run([R2.BASH, "-c", fn + '\nfree_gb "$1"', "_", str(tmp_path)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().isdigit(), repr(out.stdout)
