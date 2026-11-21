"""`in_head.py` must distinguish ABSENT from INCONCLUSIVE.

THE DEFECT IT GUARDS.  On this mount a failed git-object read and a genuine absence are
byte-identical: both give empty stdout and a non-zero exit.  Four separate readers have
reported refcv5's model seams "not in HEAD" from a read taken during a mount flap; the
files were in HEAD every time.

MUTATION PROOF.  `test_dead_channel_is_inconclusive_not_absent` is the whole point and
FAILS on any implementation that omits the control -- which is exactly what every one of
those four readers did.  The remaining tests are the known-value controls: without them
a checker that always returned INCONCLUSIVE would pass the first test.

ASCII only.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.normpath(os.path.join(_HERE, "..", "scripts", "in_head.py"))


def _load():
    if not os.path.isfile(_TOOL):
        pytest.skip(f"in_head.py not present at {_TOOL}")
    spec = importlib.util.spec_from_file_location("_in_head_under_test", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dead_channel_is_inconclusive_not_absent(monkeypatch):
    """THE POINT: a dead channel must NOT read as absence."""
    m = _load()
    monkeypatch.setattr(m, "_show", lambda p: None)          # every read fails
    monkeypatch.setattr(sys, "argv", ["in_head.py", "some/real/path.py"])
    rc = m.main()
    assert rc == m.EXIT_INCONCLUSIVE, (
        "a dead channel must return INCONCLUSIVE, never ABSENT -- conflating them is "
        "the defect this tool exists to remove")
    assert rc != m.EXIT_SOME_ABSENT


def test_live_channel_reports_genuine_absence(monkeypatch):
    """Known-value control: with the channel proven, absence is real."""
    m = _load()
    monkeypatch.setattr(m, "_show",
                        lambda p: ("x\n" * 500) if p == m.CONTROL_PATH else None)
    monkeypatch.setattr(sys, "argv", ["in_head.py", "gone.py"])
    assert m.main() == m.EXIT_SOME_ABSENT


def test_live_channel_reports_presence(monkeypatch):
    """Known-value control: a present path reads present."""
    m = _load()
    monkeypatch.setattr(m, "_show", lambda p: "y\n" * 500)
    monkeypatch.setattr(sys, "argv", ["in_head.py", "there.py"])
    assert m.main() == m.EXIT_ALL_PRESENT


def test_short_control_is_inconclusive(monkeypatch):
    """A control that reads but is too SHORT is still not a proven channel."""
    m = _load()
    monkeypatch.setattr(m, "_show",
                        lambda p: "z\n" * 3 if p == m.CONTROL_PATH else "q\n" * 9)
    monkeypatch.setattr(sys, "argv", ["in_head.py", "anything.py"])
    assert m.main() == m.EXIT_INCONCLUSIVE


def test_exit_codes_are_distinct():
    """INCONCLUSIVE must never collide with ABSENT, or callers cannot tell them apart."""
    m = _load()
    codes = {m.EXIT_ALL_PRESENT, m.EXIT_SOME_ABSENT, m.EXIT_INCONCLUSIVE}
    assert len(codes) == 3
