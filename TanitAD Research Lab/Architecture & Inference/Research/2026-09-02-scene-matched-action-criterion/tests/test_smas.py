"""Pin E-ARCH-SMAS-1's headline numbers and its control panel.

⭐ The controls are the load-bearing part: a criterion proposal whose
no-information control does not read the known value is an opinion.

    pytest -q test_smas.py
"""
import io
import json
import os

import pytest

RAW = os.path.join(os.path.dirname(__file__), "..", "raw", "smas.json")


@pytest.fixture(scope="module")
def r():
    if not os.path.exists(RAW):
        pytest.skip(f"artifact not present: {RAW}")
    return json.load(io.open(RAW, encoding="utf-8"))


def test_every_control_reads_its_known_value(r):
    """⛔ If this fails, nothing else in the package is admissible."""
    assert r["controls_all_pass"] is True, r["controls"]
    ni = r["controls"]["no_information_control"]
    assert ni["raw_ratio"] == 0.0 and ni["smas"] == 0.0
    assert all(r["C0_identity_control"].values()), r["C0_identity_control"]


def test_the_ratio_change_is_mostly_the_denominator(r):
    """F1 — the finding."""
    d = r["one_variable_decomposition_k8control_to_k60"]
    assert d["share_of_change_from_scene"] > 0.70, d
    assert d["share_of_change_from_action"] < 0.30, d
    # the decomposition must reconstruct the observed ratio, or the two reads
    # are not on the same instrument
    assert d["reconstruction_abs_err"] < 1e-3, d


def test_the_stated_10x_bar_actually_demanded_more(r):
    """F2 — a pre-registered bar that floats is not a pre-registered bar."""
    b = r["prereg_bar"]
    assert b["stated_ratio_bar"] == 10.0
    assert b["effective_ACTION_bar_actually_imposed"] > 16.0, b


def test_a_doubled_action_response_would_have_read_as_flat(r):
    """F3 — the argument for changing the criterion."""
    got = r["hypothetical_reads_under_the_RAW_ratio"][
        "action_factor_2.0_reads_ratio_factor"]
    assert 1.0 < got < 1.25, got          # "+17 %, within noise"


def test_the_fix_does_NOT_rescue_the_arm(r):
    """F4 — stated so the package cannot be read as overselling."""
    assert r["smas"]["factor"] < 1.0, r["smas"]
    assert 0.80 < r["smas"]["factor"] < 0.86, r["smas"]
