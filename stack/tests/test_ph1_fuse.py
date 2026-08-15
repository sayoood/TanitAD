"""PH1 fuser — the jurisdiction rules must hold structurally, not by habit.

Pins: tracks come only from SAM3 frames; the red-light/ego-stop check emits a
CONFLICT (not a merge) when the sources disagree; vocab tokens are always from
the real v6 lists; the goal/situation disjointness assert holds; and the fused
record's inference whitelist never contains the ego layer.
"""
import json
import sys
from pathlib import Path

import pytest

# the suite's standard scripts-dir insert; without it this file only imported
# when an alphabetically-earlier test had already done the insert
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ph1_fuse import build_tracks, corroborate, emit_vocab, main  # noqa: E402

V2 = {
    "clip_id": "c1", "_frame_wh": [448, 179],
    "scene": {"illumination": "day", "weather": "clear", "road_type": "urban",
              "domain": "urban", "lanes_visible": 2},
    "signs": {"n_signs": 1, "signs": [
        {"kind": "light", "text": "", "state": "red", "applies_to_ego": True}]},
    "symbols": {"goal_kind": "follow_main_road",
                "actions": [{"verb": "hold_corridor", "direction": "none"}]},
    "ego_state": {"v_now_ms": 8.0, "motion": "steady", "turning": "straight"},
    "speed_profile": {"v_min_future": 7.5, "stops": 0},
    "situations": {"lane_change": False, "intersection": False},
    "route": {"token": "KEEP"},
}
SAM3 = {"clip_id": "c1", "frames": {
    "0": {"det": [{"concept": "car", "score": 0.9,
                   "box_xyxy": [10, 10, 50, 50], "mask_area_px": 100}]},
    "6": {"det": [{"concept": "car", "score": 0.9,
                   "box_xyxy": [12, 11, 54, 52], "mask_area_px": 160}]},
}}


def test_tracks_link_and_classify_dynamics():
    tr = build_tracks(SAM3["frames"])
    assert len(tr) == 1 and tr[0]["n_frames"] == 2
    assert tr[0]["dynamics"] == "approaching"       # area 100 -> 160 (>1.3x)
    assert tr[0]["src"] == "sam3"


def test_red_light_without_stop_is_a_conflict_not_a_merge():
    cor, conflicts = corroborate(V2, SAM3, build_tracks(SAM3["frames"]))
    assert cor["red_light_vs_stop"]["verdict"] == "conflict"
    assert any(c["check"] == "red_light_vs_stop" for c in conflicts)


def test_red_light_with_stop_corroborates():
    v2 = json.loads(json.dumps(V2))
    v2["speed_profile"]["stops"] = 1
    cor, conflicts = corroborate(v2, SAM3, [])
    assert cor["red_light_vs_stop"]["verdict"] == "corroborated"
    assert not any(c["check"] == "red_light_vs_stop" for c in conflicts)


def test_vocab_tokens_come_from_the_real_lists():
    from tanitad.models.v6 import (STRATEGIC_GOAL_TOKENS,
                                   TACTICAL_LAT_ACTIONS, TACTICAL_LON_ACTIONS)
    vocab, _ = emit_vocab(V2, None)
    assert vocab["g_str"]["token"] in STRATEGIC_GOAL_TOKENS
    assert vocab["g_tac_lat"]["token"] in TACTICAL_LAT_ACTIONS
    assert vocab["g_tac_lon"]["token"] in TACTICAL_LON_ACTIONS


def test_unmapped_goal_becomes_abstain_plus_conflict():
    v2 = json.loads(json.dumps(V2))
    v2["symbols"]["goal_kind"] = "teleport_home"
    vocab, conflicts = emit_vocab(v2, None)
    assert vocab["g_str"]["token"] == "NONE_ABSTAIN"
    assert any(c["check"] == "goal_kind_unmapped" for c in conflicts)


def test_end_to_end_record_and_the_leak_whitelist(tmp_path):
    (tmp_path / "ego").mkdir()
    v2p = tmp_path / "v2.json"
    v2p.write_text(json.dumps([V2]))
    s3p = tmp_path / "sam3.json"
    s3p.write_text(json.dumps({"engine": "sam3", "n_clips": 1,
                            "clips": [SAM3]}))
    out = tmp_path / "fused"
    assert main(["--v2-json", str(v2p), "--sam3", str(s3p),
                 "--ego-root", str(tmp_path / "ego"), "--out", str(out)]) == 0
    rec = json.loads((out / "c1.json").read_text())
    assert rec["schema_version"] == "ph1-fused-v1"
    # labels may use ego; INFERENCE IS VISION-ONLY — ego must not be whitelisted
    assert "ego" not in rec["inference_admissible"]
    assert "alpamayo" not in rec["inference_admissible"]
    assert set(rec["inference_admissible"]) == {"perception", "semantics"}
    assert rec["semantics"]["sign_text_status"] == "pending_g1_gate"
    assert "situation" not in json.dumps(rec["vocab"]).lower()
    assert (out / "_summary.json").exists()


def test_resume_skips_existing(tmp_path):
    (tmp_path / "ego").mkdir()
    (tmp_path / "v2.json").write_text(json.dumps([V2]))
    (tmp_path / "sam3.json").write_text(json.dumps([SAM3]))
    out = tmp_path / "fused"
    args = ["--v2-json", str(tmp_path / "v2.json"), "--sam3",
            str(tmp_path / "sam3.json"), "--ego-root", str(tmp_path / "ego"),
            "--out", str(out)]
    main(args)
    first = (out / "c1.json").read_text()
    main(args)                                     # second run must not rewrite
    assert (out / "c1.json").read_text() == first
    assert json.loads((out / "_summary.json").read_text())["n_fused"] == 0


def test_speed_sign_units_are_not_assumed():
    """'20' corroborates under km/h OR mph — the unit is recorded ambiguous,
    never silently assumed (US corpus, but the pilot text was bare digits)."""
    v2 = json.loads(json.dumps(V2))
    v2["signs"]["signs"] = [{"kind": "speed", "text": "20", "state": "none",
                             "applies_to_ego": True}]
    v2["speed_profile"]["v_min_future"] = 5.0      # ~= 20 km/h -> plausible
    cor, _ = corroborate(v2, SAM3, [])
    assert cor["speed_sign_vs_ego"]["verdict"] == "corroborated"


def test_production_wrapper_format_is_parsed(tmp_path):
    """The real sam3.json is a metadata wrapper with records under "clips".
    The first fuse run matched 0 of 600 because of this — pinned."""
    (tmp_path / "ego").mkdir()
    (tmp_path / "v2.json").write_text(json.dumps([V2]))
    (tmp_path / "s.json").write_text(json.dumps(
        {"engine": "sam3", "frame_stride": 6, "clips": [SAM3]}))
    out = tmp_path / "fused"
    main(["--v2-json", str(tmp_path / "v2.json"), "--sam3",
          str(tmp_path / "s.json"), "--ego-root", str(tmp_path / "ego"),
          "--out", str(out)])
    rec = json.loads((out / "c1.json").read_text())
    assert rec["perception"]["tracks"], "wrapper records must produce tracks"


def test_zero_sam3_records_refuses_loudly(tmp_path):
    (tmp_path / "ego").mkdir()
    (tmp_path / "v2.json").write_text(json.dumps([V2]))
    (tmp_path / "s.json").write_text(json.dumps({"engine": "sam3"}))
    with pytest.raises(SystemExit, match="0 SAM3 records"):
        main(["--v2-json", str(tmp_path / "v2.json"), "--sam3",
              str(tmp_path / "s.json"), "--ego-root", str(tmp_path / "ego"),
              "--out", str(tmp_path / "fused")])


def test_missing_situations_is_not_computable_not_conflict():
    v2 = json.loads(json.dumps(V2))
    v2["scene"]["road_type"] = "junction"
    del v2["situations"]
    cor, conflicts = corroborate(v2, SAM3, [])
    assert cor["scene_vs_situations"]["verdict"] == "not_computable"
    assert not any(c.get("check") == "scene_vs_situations" for c in conflicts)


def _v2_second_clip():
    v2b = json.loads(json.dumps(V2))
    v2b["clip_id"] = "c2"
    v2b["symbols"] = {"goal_kind": "route_to", "goal_evidence_sign": 0,
                      "actions": []}
    del v2b["signs"]                       # keep the partial-clip case minimal
    return v2b


def test_missing_sam3_without_flag_refuses_loudly(tmp_path):
    """A v2 clip with no SAM3 record is a partial, and a partial must be
    NAMED, not silent (the val-600 run silently fused 4 such clips)."""
    (tmp_path / "ego").mkdir()
    (tmp_path / "v2.json").write_text(json.dumps([V2, _v2_second_clip()]))
    (tmp_path / "s.json").write_text(json.dumps([SAM3]))     # c1 only
    with pytest.raises(SystemExit, match="NO SAM3"):
        main(["--v2-json", str(tmp_path / "v2.json"), "--sam3",
              str(tmp_path / "s.json"), "--ego-root", str(tmp_path / "ego"),
              "--out", str(tmp_path / "fused")])


def test_missing_sam3_with_flag_is_a_named_partial(tmp_path):
    """With --missing-sam3-ok REASON the clip fuses from its other layers,
    perception carries the absence, and no SAM3-dependent check fabricates
    a verdict from the empty detector."""
    (tmp_path / "ego").mkdir()
    (tmp_path / "v2.json").write_text(json.dumps([V2, _v2_second_clip()]))
    (tmp_path / "s.json").write_text(json.dumps([SAM3]))     # c1 only
    out = tmp_path / "fused"
    assert main(["--v2-json", str(tmp_path / "v2.json"), "--sam3",
                 str(tmp_path / "s.json"), "--ego-root", str(tmp_path / "ego"),
                 "--missing-sam3-ok", "B184_SAM3_ABSENT",
                 "--out", str(out)]) == 0
    rec = json.loads((out / "c2.json").read_text())
    assert rec["perception"]["absent"] == "B184_SAM3_ABSENT"
    assert rec["perception"]["tracks"] == []
    # urban + zero tracks must NOT fabricate flagged_empty_urban, and the
    # route_to goal must NOT be scored provisional by a detector that never ran
    assert rec["corroboration"]["census_vs_scene"]["verdict"] == "not_computable"
    assert rec["corroboration"]["goal_evidence"]["verdict"] == "not_computable"
    # the clip WITH sam3 is untouched by the flag
    rec1 = json.loads((out / "c1.json").read_text())
    assert "absent" not in rec1["perception"] and rec1["perception"]["tracks"]
    summ = json.loads((out / "_summary.json").read_text())
    assert summ["sam3_missing"] == 1
    assert summ["missing_sam3_reason"] == "B184_SAM3_ABSENT"


def test_ego_spine_recomputed_from_npz(tmp_path):
    import numpy as np
    from ph1_fuse import ego_from_npz
    poses = np.zeros((50, 4), dtype=np.float32)
    poses[:, 3] = np.linspace(8.0, 0.2, 50)          # decelerating to a stop
    np.savez(tmp_path / "c9.npz", poses=poses, actions=np.zeros((50, 2)))
    spine = ego_from_npz(str(tmp_path / "c9.npz"))
    assert spine["speed_profile"]["stops"] >= 1
    assert spine["speed_profile"]["v_min_future"] < 0.5
    assert spine["speed_profile"]["src"] == "ego_npz"
