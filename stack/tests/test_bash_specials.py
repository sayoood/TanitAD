"""No operational shell script may assign to a bash SPECIAL variable.

MEASURED 2026-10-21 (PREREG_V8F_TAC Amendment 1): ``chain_v8f_tacgrp.sh`` held its eight group names in ``GROUPS=...``.
``GROUPS`` is a bash builtin: assignment to it is SILENTLY IGNORED (no error, rc 0) and ``$GROUPS`` expands to the user's gid
(197121 on the dev box). The 32 evals finished, and the pull step then asked box 2 for ``eval_M1_s0-tacorc-197121.npz``. Nothing
in ``bash -n`` or a dry run catches it, because the script is syntactically fine and the variable "exists".

Two checks: the mechanism itself (bash really does ignore the assignment, so the rule guards something real), and a scan of every
box-2 / pod / stack shell script for an assignment or ``for`` loop over one of these names.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
#: bash variables whose assignment is ignored, reset, or changes behaviour (bash(1), "Shell Variables"); UID / EUID / PPID are
#: readonly (an error, not silent, but just as wrong to use as a name)
SPECIALS = ("GROUPS", "FUNCNAME", "BASHPID", "BASH_SUBSHELL", "BASH_ARGV0", "BASH_COMMAND", "HISTCMD", "LINENO", "RANDOM", "SRANDOM",
            "SECONDS", "EPOCHSECONDS", "EPOCHREALTIME", "DIRSTACK", "PIPESTATUS", "UID", "EUID", "PPID")
_NAMES = "|".join(SPECIALS)
PAT = re.compile(rf"^\s*(?:(?:export|local|readonly|declare(?:\s+-[a-zA-Z]+)*)\s+)?(?P<n>{_NAMES})(?:\[[^\]]*\])?\+?=|"
                 rf"\bfor\s+(?P<f>{_NAMES})\s+in\b|\bread\s+(?:-[a-zA-Z]+\s+)*(?P<r>{_NAMES})\b", re.M)
GLOBS = ("TanitAD Research Lab/*/Research/*/code/box2_ops/*.sh", "FlyWheels/*/incoming/*/code/box2_ops/*.sh",
         "stack/scripts/*.sh", "stack/scripts/pod_ops/*.sh", "stack/ops/*.sh")


def offenders(text: str) -> list:
    out = []
    for m in PAT.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        if text[text.rfind("\n", 0, m.start()) + 1:m.start()].lstrip().startswith("#"):
            continue
        out.append((line, m.group("n") or m.group("f") or m.group("r")))
    return out


def test_detector_red_and_green():
    assert offenders('GROUPS="a b"\n') == [(1, "GROUPS")]
    assert offenders('  export SECONDS=0\n') == [(1, "SECONDS")]
    assert offenders('for RANDOM in 1 2; do :; done\n') == [(1, "RANDOM")]
    assert offenders('read -r UID\n') == [(1, "UID")]
    assert offenders('declare -a GROUPS=(a b)\n') == [(1, "GROUPS")]
    # GREEN: a renamed list, a use (not an assignment), a comment, a longer name that merely starts with a special
    assert offenders('TAC_GROUPS="a b"\nfor g in $TAC_GROUPS; do :; done\necho $SECONDS\n# GROUPS=x is a bash builtin\n'
                     'GROUPS_N=3\nRANDOM_SEED=1\n') == []


def _working_bash():
    """the first bash that runs: on Windows, PATH's ``bash`` can be System32's WSL launcher, which fails with no distro installed"""
    for c in ("C:/Program Files/Git/bin/bash.exe", shutil.which("bash"), "/bin/bash"):
        if not c or not Path(c).exists():
            continue
        try:
            r = subprocess.run([c, "-c", "echo ok"], capture_output=True, text=True, timeout=30, encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout.strip() == "ok":
            return c
    return None


BASH = _working_bash()


@pytest.mark.skipif(BASH is None, reason="no working bash")
def test_bash_really_ignores_groups_assignment():
    r = subprocess.run([BASH, "-c", 'GROUPS="goal latdec"; echo "[$GROUPS]"'], capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert r.returncode == 0
    assert r.stdout.strip() != "[goal latdec]", "bash honoured the assignment: the trap this test guards no longer exists"


def test_no_operational_script_assigns_a_bash_special():
    files = sorted({p for g in GLOBS for p in REPO.glob(g)})
    assert len(files) >= 20, f"only {len(files)} shell scripts found: the globs no longer match the repo layout"
    bad = [f"{p.relative_to(REPO)}:{ln}: {n}" for p in files for ln, n in offenders(p.read_text(encoding="utf-8", errors="replace"))]
    assert not bad, ("a shell script assigns a bash special variable (bash ignores or redefines it; see TRAPS.md, GROUPS): "
                     + "; ".join(bad))
