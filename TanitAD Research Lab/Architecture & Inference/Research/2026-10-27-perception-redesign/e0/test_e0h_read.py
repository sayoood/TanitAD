"""Known-answer test for e0h_read.interp_cross (tier 2): literal expectation, not an expression over the code under test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0h_read as R  # noqa: E402


def test_interpolated_half_crossing_is_0p8():
    # BASE 0.18 -> level 0.09; rate 0.12 at p 0.7 and 0.06 at p 0.9: 0.7 + (0.12 - 0.09) / (0.12 - 0.06) * 0.2 = 0.8 (literal)
    r = R.interp_cross([0.5, 0.7, 0.9, 1.0], [0.18, 0.12, 0.06, 0.02], 0.09)
    assert r["kind"] == "interpolated" and abs(r["p"] - 0.8) < 1e-12 and r["bracket"] == [0.7, 0.9]


def test_never_reached_and_already_reached():
    assert R.interp_cross([0.5, 1.0], [0.18, 0.12], 0.09)["p"] is None
    assert R.interp_cross([0.5, 1.0], [0.05, 0.02], 0.09)["kind"] == "at_or_below_first_grid_point"
