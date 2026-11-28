"""The emitted label set must pass the guard that judges it.

⭐ WHY THIS TEST EXISTS. The emitter and `label_guard` each encoded "at rest"
and "accelerating away" with THEIR OWN constants (0.5 / 5.0 m/s vs 1.0 / 2.0).
Clips landing in the gap made the emitter's own output fail its own guard —
9 labels, then 5 after a partial fix. Two constants for one fact always drift;
this pins that they are shared.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tanitad.data import label_guard as LG

LABELS = Path("C:/Users/Admin/tanitad-wt/_s2build/labels_geom/s2_labels_geom.jsonl")


def _guard(r):
    m = r["manoeuvre"]
    return LG.check(clip_id=r["clip_id"], g_str=r["g_str"]["token"],
                    a_str=r["a_str"]["token"], peak_yaw_deg=m["peak_yaw_deg"],
                    v_at_key_ms=m["v_at_key"], v_end_ms=m["v_end"],
                    v_min_future_ms=m["v_min"], tac_lat=r["a_tac"]["lat"],
                    tac_lon=r["a_tac"]["lon"], lateral_class=m["lateral_class"],
                    stop_type=m["stop_type"])


@pytest.mark.skipif(not LABELS.exists(), reason="emitted label set not present")
def test_the_emitter_output_passes_its_own_guard():
    rows = [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    assert rows, "no labels emitted"
    refused = [r["clip_id"] for r in rows
               if r["manoeuvre"] and _guard(r).refused]
    assert not refused, (f"{len(refused)}/{len(rows)} emitted labels are REFUSED "
                         f"by label_guard: {refused[:5]}")


@pytest.mark.skipif(not LABELS.exists(), reason="emitted label set not present")
def test_no_label_is_a_silent_guess():
    """Every label states its provenance and the horizon it was derived over."""
    rows = [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    for r in rows:
        assert r["g_str"]["provenance"] in ("geometry", "abstain")
        assert r["a_str"]["provenance"] in ("geometry", "abstain")
        assert "available_s" in r["horizon"]
        # a short horizon MUST abstain rather than guess
        if r["horizon"]["available_s"] < 8.0:
            assert r["g_str"]["token"] == "NONE_ABSTAIN", r["clip_id"]


@pytest.mark.skipif(not LABELS.exists(), reason="emitted label set not present")
def test_lateral_never_comes_from_alpamayo():
    """The lateral axis is at chance in Alpamayo; it may not supervise here."""
    rows = [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    for r in rows:
        assert "GEOMETRY ONLY" in r["_provenance"]["lateral"]


@pytest.mark.skipif(not LABELS.exists(), reason="emitted label set not present")
def test_a_vlm_reason_never_creates_a_stop_that_geometry_did_not_find():
    rows = [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    for r in rows:
        if r["g_tac"]["lon_token"] == "STOP_POINT":
            assert r["manoeuvre"] is None or r["manoeuvre"]["n_stop_episodes"] > 0 \
                or r["g_tac"]["lon_args"].get("hold_s", 0) > 0, r["clip_id"]
