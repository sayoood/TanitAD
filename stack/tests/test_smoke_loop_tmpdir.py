"""Every test that runs ``train_flagship_v4.smoke_loop`` must pass ``tmp_dir``.

Without it ``smoke_loop`` falls back to ``tempfile.mkdtemp(prefix="v4smoke_")`` (stack/scripts/train_flagship_v4.py), and
nothing removes that directory. MEASURED 2026-11-26 on Thor: 113 ``/tmp/v4smoke_*`` dirs, 2,516 MB, one per gate-suite run
since 2026-10-14, all from ``test_v5_trainer_v2_val.py::test_THE_LOOP_READS_THE_GATE_and_stops_early``, the one caller that
passed no ``tmp_dir``. They were deleted on the PI's go; this test keeps them from coming back.

The check is static (an AST scan of ``stack/tests/*.py``), so it costs nothing. The scan is proven to work in both directions:
it must FLAG a bare call in a planted source, must PASS a call with ``tmp_dir=`` (keyword or first positional argument), and
must FIND the real callers (a scan that reads zero calls is a broken scan, not a clean suite).
"""
from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent
#: smoke_loop's first positional parameter is tmp_dir (train_flagship_v4.py: ``def smoke_loop(tmp_dir=None, heldout_gate=None, ...)``)
FIRST_POSITIONAL = "tmp_dir"


def bare_smoke_loop_calls(source: str, name: str = "<src>") -> tuple[int, list[str]]:
    """(number of smoke_loop calls, the ones that pass no tmp_dir) in one module's source."""
    tree = ast.parse(source, filename=name)
    n, bad = 0, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if fname != "smoke_loop":
            continue
        n += 1
        has = bool(node.args) or any(k.arg == FIRST_POSITIONAL for k in node.keywords)
        if not has:
            bad.append(f"{name}:{node.lineno}")
    return n, bad


def test_the_scan_flags_a_bare_call_and_passes_a_tmp_dir_call():
    planted = (
        "import train_flagship_v4 as T4\n"
        "def test_a(tmp_path):\n"
        "    T4.smoke_loop(heldout_gate=None)\n"            # bare: must be flagged
        "    T4.smoke_loop(tmp_dir=str(tmp_path))\n"        # keyword: fine
        "    T4.smoke_loop(str(tmp_path), heldout_gate=1)\n"  # positional: fine
        "    smoke_loop()\n"                                # bare name call: must be flagged
    )
    n, bad = bare_smoke_loop_calls(planted, "planted.py")
    assert n == 4
    assert bad == ["planted.py:3", "planted.py:6"]


def test_every_test_caller_of_smoke_loop_passes_tmp_dir():
    n_total, offenders = 0, []
    for p in sorted(TESTS.glob("test_*.py")):
        if p.name == Path(__file__).name:
            continue
        src = p.read_text(encoding="utf-8")
        if "smoke_loop" not in src:
            continue
        n, bad = bare_smoke_loop_calls(src, p.name)
        n_total += n
        offenders += bad
    # control: the real suite calls smoke_loop several times (test_train_flagship_v4.py, test_v5_trainer_v2_val.py)
    assert n_total >= 6, f"the scan found only {n_total} smoke_loop calls: it is not reading the suite"
    assert not offenders, ("smoke_loop called without tmp_dir (leaks a /tmp/v4smoke_* dir per run; pass "
                           f"tmp_dir=str(tmp_path)): {offenders}")
