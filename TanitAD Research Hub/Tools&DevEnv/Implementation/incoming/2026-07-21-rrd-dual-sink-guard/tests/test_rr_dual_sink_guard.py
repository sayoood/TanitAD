"""Falsifiers for the rerun dual-sink guard.

Run standalone:  pytest <this package>/tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rr_dual_sink_guard import check_sinks, DUAL_SINK_MSG   # noqa: E402


def test_both_sinks_raise():
    with pytest.raises(ValueError) as e:
        check_sinks("a.rrd", 9090)
    assert "3,314x" in str(e.value)


def test_rrd_only_is_allowed():
    check_sinks("a.rrd", None)


def test_serve_only_is_allowed():
    check_sinks(None, 9090)


def test_explicit_optout_is_honoured():
    """A stub .rrd is a legitimate choice — as long as it was chosen."""
    check_sinks("a.rrd", 9090, allow_stub_rrd=True)


def test_message_names_both_the_measurement_and_the_workaround():
    """A guard that only says 'no' costs the next person the same hour."""
    assert "10,593,179" in DUAL_SINK_MSG and "twice" in DUAL_SINK_MSG
