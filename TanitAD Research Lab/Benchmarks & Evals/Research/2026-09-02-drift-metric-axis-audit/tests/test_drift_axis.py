"""Pin E-BE-DRIFT-1's flatness result and its control discipline.

    pytest -q test_drift_axis.py
"""
import io
import json
import os

import pytest

RAW = os.path.join(os.path.dirname(__file__), "..", "raw", "drift_axis.json")


@pytest.fixture(scope="module")
def r():
    if not os.path.exists(RAW):
        pytest.skip(f"artifact not present: {RAW}")
    return json.load(io.open(RAW, encoding="utf-8"))


def test_no_read_was_scored_with_a_failing_control(r):
    """⛔ The constant column must read EXACTLY 0.0 or the read is excluded."""
    assert r["controls"]["constant_control_required"] == 0.0
    assert r["controls"]["reads_not_scored_due_to_control_failure"] == []


def test_all_three_arms_and_both_bands_are_present(r):
    ax = r["k_axis"]
    assert len(ax) == 6, sorted(ax)
    for arm in ("postrain30k", "k8clip05p30k", "k60clip05p30k"):
        assert any(k.startswith(arm) for k in ax), arm


def test_drift_is_flat_across_a_15x_horizon_change(r):
    """F3 — the finding: a transporting latent should degrade with horizon."""
    for name, v in r["k_axis"].items():
        assert v["horizon_ratio"] == 15, (name, v)
        assert abs(v["relative_change"]) < 0.03, (name, v)


def test_the_shrinking_sample_is_recorded_not_hidden(r):
    """k=60 roughly halves the usable windows; a curve that hides that is not one."""
    for name, v in r["k_axis"].items():
        assert v["n_rows_high"] < v["n_rows_low"], (name, v)


def test_the_metric_definition_is_carried_with_the_numbers(r):
    """F1 — the whole package exists because the metric was known by its NAME."""
    d = r["_what_drift_is"]
    assert "z_{t+k} - z_t" in d
    assert "NOT a start-vs-end" in d
    assert "PCA" in r["_caveat_pca_basis"]
