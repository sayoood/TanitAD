"""The gate3 chain runs the Thor suite with the runner of ITS OWN commit, at the study's worker count (MM 4a3d05, 2026-11-15).

Before this, `r9gate3_chain.sh` stage B copied `thor_gate_runner2.py` + `gate_verdict.py` from a FIXED Thor folder written on 2026-10-15
(`/home/nvidia/gate_fb47a583763f/`), so a runner change in the repo never reached the chain: landing the parallel runner alone would have
changed nothing on Thor. MEASURED 2026-11-15: the fixed copy, the r9e2 launch tree's copy and the repo's blob at 9697fa4d were byte-identical
(runner md5 4ef75118, verdict d8897a80), so switching the source changes no file the current suite runs.

Proved here: the chain's stage B copies both files from `$T/stack/ops/gate` (the commit's launch tree) and no executable line names the fixed
folder; the runner command passes `THOR_GATE_WORKERS=${R9_GATE_WORKERS:-1}`; the study file sets R9_GATE_WORKERS 1 / 1 / 3 for e1 / e2 / e3
(by EXECUTION: the file is sourced); the runner the chain now copies is the one that reads THOR_GATE_WORKERS.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_launch_gate_e1rig_e1rec2 as R2  # noqa: E402

GATE3 = ROOT / "ops" / "r9e1_gate3"
CHAIN = GATE3 / "r9gate3_chain.sh"
STUDY = GATE3 / "r9gate3_study.sh"
RUNNER = ROOT / "ops" / "gate" / "thor_gate_runner2.py"
VERDICT = ROOT / "ops" / "gate" / "gate_verdict.py"
BASH = R2.BASH
needs_bash = R2.needs_bash


def _code_lines(p: Path):
    """the executable lines of a shell script (comments and blank lines dropped)."""
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _stage_b(p: Path):
    txt = p.read_text(encoding="utf-8")
    a, b = txt.index("# ---- B:"), txt.index("# ---- C:")
    return [ln.strip() for ln in txt[a:b].splitlines() if ln.strip() and not ln.strip().startswith("#")]


def test_stage_b_copies_runner_and_verdict_from_the_commit_tree():
    cps = [ln for ln in _stage_b(CHAIN) if ln.startswith("cp ") and "thor_gate_runner2.py" in ln]
    assert len(cps) == 1, cps
    cp = cps[0]
    assert "$T/stack/ops/gate/thor_gate_runner2.py" in cp and "$T/stack/ops/gate/gate_verdict.py" in cp, cp
    assert "stop NO-RUNNER" in cp, "a missing runner must STOP the chain, never fall back"


def test_no_executable_line_names_the_fixed_thor_copy():
    bad = [ln for ln in _code_lines(CHAIN) if "gate_fb47a583763f" in ln]
    assert bad == [], bad


def test_runner_command_passes_the_study_worker_count():
    runs = [ln for ln in _stage_b(CHAIN) if "thor_gate_runner2.py $G" in ln and "nohup" in ln]
    assert len(runs) == 1, runs
    assert "THOR_GATE_WORKERS=${R9_GATE_WORKERS:-1}" in runs[0], runs[0]
    assert "CUDA_VISIBLE_DEVICES=" in runs[0] and "nice -n 19" in runs[0], runs[0]   # still CPU-only, lowest priority


def test_the_copied_runner_is_the_one_that_reads_the_variable():
    assert RUNNER.is_file() and VERDICT.is_file()
    src = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ.get("THOR_GATE_WORKERS", "1")' in src


def _workers(env: dict | None):
    e = dict(os.environ, MSYS_NO_PATHCONV="1")
    e.pop("R9_STUDY", None)
    e.pop("R9_GATE_WORKERS", None)
    e.update(env or {})
    prog = "set -u; . '%s' || exit $?; echo \"W=[${R9_GATE_WORKERS-UNSET}] S=[$R9_STUDY]\"" % STUDY.as_posix()
    r = subprocess.run([BASH, "-c", prog], capture_output=True, text=True, encoding="utf-8", env=e, timeout=120,
                       stdin=subprocess.DEVNULL)
    m = re.search(r"W=\[(.*?)\] S=\[(.*?)\]", r.stdout.replace("\r", ""))
    return r.returncode, (m.group(1), m.group(2)) if m else None, r.stdout + r.stderr


@needs_bash
@pytest.mark.parametrize("study,want", [(None, "1"), ("e1", "1"), ("e2", "1"), ("e3", "3")])
def test_study_worker_counts(study, want):
    rc, got, out = _workers(None if study is None else {"R9_STUDY": study})
    assert rc == 0, out
    assert got is not None, out
    assert got[0] == want, (study, got, out)
    assert got[1] == (study or "e1"), (study, got, out)
