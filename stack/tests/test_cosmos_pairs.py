"""Tests for the matched-weather-pair selection (scripts/cosmos_pairs.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cosmos_pairs import parse_member, select_pairs  # noqa: E402


def test_parse_member():
    base = "ab12_604490799000_604510799000"
    assert parse_member(f"x/{base}_0_Rainy.mp4") == (base, 0, "rainy")
    assert parse_member(f"{base}_1_Golden_hour.mp4") == (base, 1, "golden_hour")
    assert parse_member(f"{base}_1.mp4") is None       # no weather suffix
    assert parse_member("caption.json") is None


def test_select_pairs_requires_clear_and_degraded():
    groups = {
        ("a", 0): {"sunny": "a_0_Sunny.mp4", "foggy": "a_0_Foggy.mp4",
                   "rainy": "a_0_Rainy.mp4"},
        ("b", 1): {"night": "b_1_Night.mp4"},                # degraded only
        ("c", 0): {"morning": "c_0_Morning.mp4"},            # clear only
        ("d", 0): {"sunny": "d_0_Sunny.mp4", "snowy": "d_0_Snowy.mp4"},
    }
    wanted = select_pairs(groups, n_pairs=10)
    keys = set(wanted.values())
    assert keys == {("a", 0), ("d", 0)}                # only true pairs
    assert len(wanted) == 4                            # one clear + one degraded each
    # richer group ("a", 3 variants) must survive an n_pairs=1 cut
    wanted1 = select_pairs(groups, n_pairs=1)
    assert set(wanted1.values()) == {("a", 0)}
