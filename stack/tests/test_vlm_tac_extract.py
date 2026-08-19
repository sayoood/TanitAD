"""Tests for the GPU-free logic of the VLM extractor.

The parts worth pinning are the ones whose failure is SILENT and expensive:
resume (a wrong answer re-runs GPU-days, or worse, skips clips) and the sign
gate (a wrong answer changes which clips can reach ROUTE_TO at all).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vlm_tac_extract as X  # noqa: E402


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #
def test_load_done_on_a_missing_file_is_empty_not_an_error():
    """First run: no output yet. This must be "nothing done", never a crash."""
    assert X.load_done("/nonexistent/nope.jsonl") == set()


def test_load_done_keys_on_clip_AND_kind(tmp_path):
    p = tmp_path / "o.jsonl"
    p.write_text(
        json.dumps({"clip_id": "a", "kind": "lon", "raw": "x"}) + "\n"
        + json.dumps({"clip_id": "a", "kind": "lane", "raw": "x"}) + "\n"
        + json.dumps({"clip_id": "b", "kind": "lon", "raw": "x"}) + "\n",
        encoding="utf-8")
    done = X.load_done(str(p))
    assert done == {("a", "lon"), ("a", "lane"), ("b", "lon")}
    # ⛔ per-CLIP resume would skip ("b","lane"), which was never generated
    assert ("b", "lane") not in done


def test_load_done_survives_a_torn_last_line(tmp_path):
    """The run died mid-write. Resume must skip the fragment and keep the rest,
    not crash — otherwise a crash makes the output permanently unreadable."""
    p = tmp_path / "o.jsonl"
    p.write_text(json.dumps({"clip_id": "a", "kind": "lon"}) + "\n"
                 + '{"clip_id": "b", "kind": "la',        # torn
                 encoding="utf-8")
    assert X.load_done(str(p)) == {("a", "lon")}


def test_load_done_ignores_error_records_so_they_are_not_retried_forever(tmp_path):
    """An error record still marks the pair attempted. Re-running a
    deterministically failing generation on every resume would livelock the
    run; the failure is visible in the artifact for the operator to act on."""
    p = tmp_path / "o.jsonl"
    p.write_text(json.dumps({"clip_id": "a", "kind": "lon", "error": "boom"}) + "\n",
                 encoding="utf-8")
    assert ("a", "lon") in X.load_done(str(p))


# --------------------------------------------------------------------------- #
# the sign gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lane,expected", [
    ("Turn Left", True), ("Turn Right", True),
    ("turn left", True),                       # case-insensitive
    ("  Turn Right  ", True),                  # the taxonomy has stray spaces
    ("Lane Keep", False), ("Right Lane Change", False),
    ("Slightly Shift Left", False), (None, False),
])
def test_sign_call_is_gated_on_the_turn_classes(lane, expected):
    """⭐ A SCOPE rule, not a sampling shortcut: ROUTE_TO asserts a turn was
    route-determined, so it is unreachable where the ego does not turn."""
    clip = {"clip_id": "c", "alpamayo": {"lane": lane}}
    assert X.wants_sign(clip, sign_on_turns=True) is expected


def test_all_sign_overrides_the_gate():
    clip = {"clip_id": "c", "alpamayo": {"lane": "Lane Keep"}}
    assert X.wants_sign(clip, sign_on_turns=False) is True


def test_a_clip_with_no_alpamayo_block_does_not_crash_the_gate():
    assert X.wants_sign({"clip_id": "c"}, sign_on_turns=True) is False


def test_gate_matches_the_taxonomy_tokens_we_actually_map():
    """Pins the gate against the SAME strings tac_str_labels maps to turns, so
    the two cannot drift apart."""
    from tanitad.lake.tac_str_labels import ALPAMAYO_LANE_TO_LAT
    turns = {k for k, v in ALPAMAYO_LANE_TO_LAT.items()
             if v in ("TURN_L", "TURN_R")}
    assert turns == set(X.TURN_LANES), (turns, X.TURN_LANES)
