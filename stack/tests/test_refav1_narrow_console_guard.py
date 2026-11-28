"""``refav1_arm``'s narrow-console guard must stay REACHABLE, not merely present.

WHY THIS TEST EXISTS.

``taniteval/tools/refav1_arm.py`` carries 252 lines containing non-ASCII, nine of
them in argparse help strings.  On a default Windows console (**cp1252**) a single
``print`` of any of them raises ``UnicodeEncodeError`` and takes the process with
it.  ``_survive_a_narrow_console()`` degrades the markers instead of dying, and
``main()`` calls it before anything else.

An escalation reached the Master Mind on 2026-09-06 claiming this file carried
"a live cp1252 ``UnicodeEncodeError`` that kills the CLI after the bootstrap is
paid for, on every run".  **It does not, and this test is why that is now a
claim somebody can check instead of re-litigate.**  MEASURED: with the guard
removed the defect reproduces; with it applied the same call succeeds; and
``--help`` exits **0** with 20,136 bytes of stdout and an EMPTY stderr under a
cp1252 console with ``PYTHONIOENCODING`` unset.

⛔ THE POINT IS THE TWO-STATE CONTROL, NOT THE PASS.  A test that only asserted
"the guard runs without crashing" would pass just as happily on a stream that was
never narrow in the first place -- it would be certifying the fixture, not the
guard.  So STATE A deliberately reintroduces the defect and REQUIRES it to fail.
If STATE A ever stops raising, this test reports INCONCLUSIVE rather than green:
the fixture has stopped reproducing the condition the guard exists for.
Same family as ``test_rl_forward_keys_cover_signature``'s mutation controls.

⚠️ LOADED BY PATH, DELIBERATELY.  ``import taniteval.tools.refav1_arm`` raises
``KeyError`` inside ``_load_unlocked`` on this repo -- the outer ``taniteval/``
has no ``__init__.py``, so it resolves as a NAMESPACE package and the submodule
import fails.  Running the file as a script works, which is exactly why ``--help``
passes while an import-based test would fail for an unrelated reason and look
like the encoding defect.  Loading by path exercises the same code the CLI runs.
"""

import importlib.util
import io
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "taniteval", "tools", "refav1_arm.py")

#: A profile that reaches BOTH marker paths: the unconditional em dash in the
#: header line, and the U+26A0 warning that only fires when an arm is degenerate.
#: The ``ha0`` floor is degenerate on every real run, so this is the live shape.
_TP = {
    "n_windows": 10,
    "arms": {
        "ha0": {"n": 10, "straight_frac": 0.5, "const_speed_frac": 0.5,
                "trivial_frac": 0.9, "identical_to": {}},
    },
    "degenerate_arms": ["ha0"],
}


def _module():
    if not os.path.exists(_SRC):
        pytest.skip("refav1_arm.py not present at %s" % _SRC)
    spec = importlib.util.spec_from_file_location("refav1_arm_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refav1_arm_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _narrow_stream():
    """A stream that encodes exactly like a default Windows console."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict",
                            newline="")


def _call(apply_guard):
    """Call the printer on a forced-cp1252 stream. Returns None or the exception."""
    mod = _module()
    buf = _narrow_stream()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf, buf
    try:
        if apply_guard:
            mod._survive_a_narrow_console()
        mod._print_trivial_profile(_TP)
        return None
    except Exception as exc:  # noqa: BLE001
        return exc
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_state_a_the_defect_still_reproduces_without_the_guard():
    """The load-bearing half. Without this, STATE B certifies nothing."""
    exc = _call(apply_guard=False)
    assert isinstance(exc, UnicodeEncodeError), (
        "STATE A did not reproduce the defect (got {0!r}). The fixture has stopped "
        "exercising a narrow console, so the guard is no longer under test -- treat "
        "a green STATE B as INCONCLUSIVE until this reproduces again.".format(exc)
    )


def test_state_b_the_guard_survives_the_same_stream():
    """The claim itself: the guard degrades the markers instead of dying."""
    exc = _call(apply_guard=True)
    assert exc is None, (
        "_survive_a_narrow_console() did not save the call: {0!r}. The CLI dies on "
        "a cp1252 console, which is the default one.".format(exc)
    )


def test_main_calls_the_guard_before_anything_can_print():
    """Correctness and WIRING are different claims, and only the first was tested.

    A guard that exists but is never called is decoration; that is the exact
    failure class this repo has already paid for twice.
    """
    import inspect
    src = inspect.getsource(_module().main)
    first = [ln.strip() for ln in src.splitlines() if ln.strip()][1]
    assert "_survive_a_narrow_console()" in first, (
        "main()'s first statement is {0!r}, not the narrow-console guard. Anything "
        "printed before it dies on a cp1252 console.".format(first)
    )


def test_the_file_really_does_carry_markers_that_would_break():
    """Vacuity gate: if the markers were all removed, every test above is moot."""
    with io.open(_SRC, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    non_ascii_lines = [ln for ln in text.splitlines() if any(ord(c) > 127 for c in ln)]
    assert len(non_ascii_lines) > 50, (
        "only {0} lines carry non-ASCII; if the markers are gone the guard is no "
        "longer load-bearing and this whole module should be deleted rather than "
        "left passing vacuously.".format(len(non_ascii_lines))
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
