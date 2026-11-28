"""Pin E-DATA-HORIZON-1's headline numbers against its own banked artifact.

⭐ WHY A TEST AND NOT JUST THE RESULT: `RESULT.md` is prose and can be edited; the
artifact can be regenerated or corrupted. These assertions fail if the two ever
disagree, which is the failure the programme keeps paying for (a number in a doc
that no longer matches the JSON it cites).

    pytest -q test_horizon_budget.py
"""
import io
import json
import os

import pytest

RAW = os.path.join(os.path.dirname(__file__), "..", "raw", "horizon_budget.json")


@pytest.fixture(scope="module")
def r():
    if not os.path.exists(RAW):
        pytest.skip(f"artifact not present: {RAW}")
    return json.load(io.open(RAW, encoding="utf-8"))


def test_episode_frames_are_about_200_not_120(r):
    """F1 — the finding that refutes the trainer docstring's premise."""
    T = r["episodes"]["T_unique"]
    assert min(T) >= 195, T
    assert 120 not in T, "T=120 is the refuted docstring value"
    assert 195 <= r["episodes"]["T_mean"] <= 210


def test_the_corpus_is_not_exhausted_at_K6(r):
    """F2 — the docstring says K=6 (max_horizon 120) yields ZERO windows."""
    row = r["window_budget_t_max"]["120"]
    assert row["windows_per_episode_mean"] > 70, row
    assert row["episodes_contributing_zero_windows"] == 0, row


def test_derived_max_horizon_differs_while_the_args_key_does_not(r):
    """F3 — what an args-level diff structurally cannot see."""
    arms = r["arm_configs"]
    for a in arms.values():
        assert a["args_max_horizon_AS_DIFFED"] is None, a
    assert arms["v7tiny_postrain30k"]["max_horizon_DERIVED"] == 20
    assert arms["v7tiny_k60clip05p30k"]["max_horizon_DERIVED"] == 60


def test_the_free_shift_budget_does_NOT_decorrelate_the_steering(r):
    """F4 — the whole point: affordable is not the same as admissible.

    Free budget at the O11 re-run's settings is s <= max_horizon - o11_k = 16.
    """
    free_shift = 20 - 4
    steer_at_free = r["action_autocorrelation_r_by_lag"][str(free_shift)]["steer_road_rad"]
    assert steer_at_free > 0.7, (
        f"steer r={steer_at_free} at the free shift — if this ever drops below "
        "0.7 the recommendation to raise --max-horizon should be revisited")
    need_25 = r["min_decorrelating_shift_steps"]["steer_road_rad@|r|<=0.25"]
    assert need_25 > free_shift, (need_25, free_shift)
    assert 35 < need_25 < 50, need_25


def test_accel_is_not_the_binding_channel(r):
    a = r["min_decorrelating_shift_steps"]
    assert a["accel_mps2@|r|<=0.25"] < a["steer_road_rad@|r|<=0.25"]
