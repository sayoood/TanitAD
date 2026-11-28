"""The step-aware model-as-trained stamp (PI ruling 2026-09-26: fixes from step 34,500). Literal
expectations; the boundary step itself is PRE-switch (the run stopped AT the 34,500 checkpoint)."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import model_stamp6 as M6  # noqa: E402


def test_pre_switch_steps():
    for s in (1000, 5000, 30000, 34500, "34500"):
        assert M6.stamp(s).startswith('"F3 detach-only, F4 on the last layer only; tactical labels ~0.37 s early"')


def test_post_switch_steps():
    for s in (34501, 34550, 50400):
        assert M6.stamp(s).startswith('"hybrid: F3 cascade loss + true label clock from step 34,500"')


def test_unknown_step_is_never_silently_pre_or_post():
    assert M6.stamp(None).startswith("UNKNOWN")
    assert M6.stamp("final").startswith("UNKNOWN")


def test_switch_warning_only_when_straddling():
    assert M6.switch_warning(5000, 30000) is None
    assert M6.switch_warning(34501, 50400) is None
    assert M6.switch_warning(5000, 50400).startswith("PRE- vs POST-SWITCH")
    assert M6.switch_warning(30000, 34501).startswith("PRE- vs POST-SWITCH")
    assert M6.switch_warning(None, 50400).startswith("UNKNOWN")


def test_REGRESSION_an_off_by_one_switch_goes_red(monkeypatch):
    """Mutation: a switch at 34,499 would stamp the stop checkpoint itself as fixed — must differ."""
    monkeypatch.setattr(M6, "SWITCH_STEP", 34499)
    assert not M6.stamp(34500).startswith('"F3 detach-only')
