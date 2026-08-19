"""Tests for the compose stage — where every rule is actually applied."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vlm_tac_compose as C  # noqa: E402


def _payload(tmp, clip_id="c1", lane="Lane Keep", lon="Gentle Deceleration"):
    p = tmp / "payload.json"
    p.write_text(json.dumps({
        "sampling": {}, "frame_times_s": [], "_parity_gate": {"in_deployed_val": 0},
        "clips": [{"clip_id": clip_id,
                   "alpamayo": {"lane": lane, "longitudinal": lon,
                                "lateral": "x", "cot": "y"},
                   "ego": {"v0_ms": 10.0, "v_end_ms": 5.0},
                   "calls": {}}]}), encoding="utf-8")
    return p


def _raw(tmp, recs):
    p = tmp / "raw.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    return p


def _run(tmp, payload, raw):
    out, cen = tmp / "labels.jsonl", tmp / "census.json"
    C.main(["--raw", str(raw), "--payload", str(payload),
            "--out", str(out), "--census", str(cen)])
    labels = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    return labels, json.loads(cen.read_text(encoding="utf-8"))


def test_only_the_echo_free_pass_is_read(tmp_path):
    """⛔ The with_ego pass is a CONTROL. A control that can become the label is
    not a control. The extractor only writes no_ego records, and compose must
    never reach for anything else."""
    raw = _raw(tmp_path, [{"clip_id": "c1", "kind": "lane", "raw": "</think>\nVERDICT: LEFT",
                           "hit_cap": False, "closed_think": True}])
    labels, cen = _run(tmp_path, _payload(tmp_path), raw)
    assert labels[0]["g_str"]["value"] == "LANE_TARGET"


def test_a_capped_generation_abstains_and_is_counted(tmp_path):
    """The number that must never be read as 'the model cannot do this'."""
    raw = _raw(tmp_path, [{"clip_id": "c1", "kind": "lane",
                           "raw": "reasoning... VERDICT: LEFT but wait",
                           "hit_cap": True, "closed_think": False}])
    labels, cen = _run(tmp_path, _payload(tmp_path), raw)
    assert cen["quality"]["hit_cap"] == 1
    assert cen["verdicts"]["lane"]["ABSTAIN"] == 1
    assert labels[0]["g_str"]["value"] == "ABSTAIN"


def test_a_reason_without_a_referent_does_not_become_a_label(tmp_path):
    raw = _raw(tmp_path, [{"clip_id": "c1", "kind": "lon",
                           "raw": "</think>\nVERDICT: FOLLOW",   # no a) line
                           "hit_cap": False, "closed_think": True}])
    labels, cen = _run(tmp_path, _payload(tmp_path), raw)
    assert labels[0]["a_tac_lon"]["value"] == "ABSTAIN"
    assert "VLM_REASON_WITHOUT_REFERENT" in labels[0]["flags"]


def test_a_parroted_referent_survives_into_the_artifact(tmp_path):
    """⚠️ compose() has no opinion on prompt-parroting — it is about the PROMPT,
    not the label algebra — so the flag must be attached here or it is lost."""
    raw = _raw(tmp_path, [{"clip_id": "c1", "kind": "lon",
                           "raw": "</think>\na) the white van ahead in our lane\n"
                                  "VERDICT: FOLLOW",
                           "hit_cap": False, "closed_think": True}])
    labels, cen = _run(tmp_path, _payload(tmp_path), raw)
    assert "REFERENT_ECHOED_PROMPT_EXAMPLE" in labels[0]["flags"]
    assert cen["quality"]["referent_echoed_prompt"] == 1


def test_missing_generations_are_counted_not_silently_skipped(tmp_path):
    """A clip with no records must appear in the census as absent — otherwise a
    half-finished run reports a clean, smaller corpus."""
    labels, cen = _run(tmp_path, _payload(tmp_path), _raw(tmp_path, []))
    assert cen["generations"]["lon/absent"] == 1
    assert cen["generations"]["lane/absent"] == 1
    assert len(labels) == 1 and labels[0]["a_tac_lon"]["value"] == "ABSTAIN"


def test_the_lateral_contradiction_flag_reaches_the_census(tmp_path):
    raw = _raw(tmp_path, [{"clip_id": "c1", "kind": "lane",
                           "raw": "</think>\nVERDICT: CURRENT",
                           "hit_cap": False, "closed_think": True}])
    pl = _payload(tmp_path, lane="Right Lane Change")
    labels, cen = _run(tmp_path, pl, raw)
    assert cen["flags"]["LAT_LANE_TARGET_DISAGREEMENT"] == 1


def test_the_parity_gate_record_rides_into_the_census(tmp_path):
    """A label census that cannot state its own parity provenance is not
    decision-grade."""
    _, cen = _run(tmp_path, _payload(tmp_path), _raw(tmp_path, []))
    assert cen["parity_gate"] == {"in_deployed_val": 0}
