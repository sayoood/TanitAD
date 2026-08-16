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


# ============================================================================
# S2 v1 — geometry-primary strategic layer (S2_STRATEGIC_GAP §6 items 2/3/6)
# ============================================================================
EA_TURN_LEFT = {
    "route": {"token": "turn_left", "token_valid": True, "dist_m": 27.3,
              "arc_m": 118.4, "maneuver_dyaw_rad": 1.68},
    "lane_change_events": [], "speed_events": [],
    "speed_profile": {"v_t0_ms": 8.2, "v_min_future_ms": 3.1,
                      "v_max_future_ms": 12.0, "net_dv_ms": 0.4,
                      "stops": False},
    "peak_kappa_per_m": 0.11,
}
EA_FOLLOW = {
    "route": {"token": "follow", "token_valid": True, "dist_m": None,
              "arc_m": 240.0, "maneuver_dyaw_rad": 0.0},
    "lane_change_events": [], "speed_events": [],
    "speed_profile": {"v_t0_ms": 13.0, "v_min_future_ms": 11.5,
                      "v_max_future_ms": 14.0, "net_dv_ms": 0.2,
                      "stops": False},
    "peak_kappa_per_m": 0.01,
}


def test_dead_ego_lateral_vote_now_lands():
    """Item 6: ego_past_state emits turning='turning_left'/'turning_right'
    (ph0_v2.py:400-401) but the vote map keyed 'left'/'right' — MEASURED
    null on 36/36 turning clips. Negative control first: the OLD mapping
    really was dead for these values; then the fix lands the vote."""
    old_map = {"left": "NUDGE_L", "right": "NUDGE_R", "straight": "LANE_KEEP"}
    assert old_map.get("turning_left") is None          # the measured defect
    v2 = json.loads(json.dumps(V2))
    v2["ego_state"]["turning"] = "turning_left"
    del v2["signs"]                     # no red light: lat vote is the point
    vocab, _ = emit_vocab(v2, None)
    votes = dict((s, t) for s, t in
                 [tuple(v) for v in vocab["g_tac_lat"]["votes"]]
                 if s == "ego")
    assert votes["ego"] == "NUDGE_L"    # the vote LANDS now
    v2["ego_state"]["turning"] = "turning_right"
    vocab, _ = emit_vocab(v2, None)
    assert ["ego", "NUDGE_R"] in vocab["g_tac_lat"]["votes"]
    v2["ego_state"]["turning"] = "straight"             # unchanged behavior
    vocab, _ = emit_vocab(v2, None)
    assert ["ego", "LANE_KEEP"] in vocab["g_tac_lat"]["votes"]


def test_geometry_primary_turn_beats_vlm_follow():
    """The 44-mislabel class: valid turn geometry + VLM 'follow_main_road'
    must label TURN_LEFT with the dist arg set, VLM recorded as disagreeing."""
    vocab, _ = emit_vocab(V2, None, engine_a=EA_TURN_LEFT)
    g = vocab["g_str"]
    assert g["token"] == "TURN_LEFT" and g["token_id"] == 4
    assert g["provenance"] == "path" and g["src"] == "engine_a"
    assert g["args"][0] == 27.3 and g["arg_mask"][0] == 1
    assert g["corroboration"]["vlm_goal_kind"] == "follow_main_road"
    assert g["corroboration"]["agrees"] is False


def test_route_to_is_remapped_on_turn_geometry_and_abstained_without():
    """⛔ ROUTE_TO stays GATED: geometry-backed claims remap to their turn;
    a claim with no junction geometry ABSTAINS with a reason. ROUTE_TO
    itself is never emitted."""
    v2 = json.loads(json.dumps(V2))
    v2["symbols"]["goal_kind"] = "route_to"
    vocab, _ = emit_vocab(v2, None, engine_a=EA_TURN_LEFT)
    g = vocab["g_str"]
    assert g["token"] == "TURN_LEFT"
    assert g["corroboration"]["remapped_from_route_to"] is True
    vocab, _ = emit_vocab(v2, None, engine_a=EA_FOLLOW)
    g = vocab["g_str"]
    assert g["token"] == "NONE_ABSTAIN" and "reason" in g
    # even with no engine A at all, route_to must not be emitted as a guess
    vocab, _ = emit_vocab(v2, None, engine_a=None)
    assert vocab["g_str"]["token"] == "NONE_ABSTAIN"


def test_null_route_dist_never_fabricates_an_arg():
    """The 8dc5d14d… row: a validated turn whose maneuver window sits at the
    clip edge has dist_m=None — the arg slot stays UNSET, never 0.0-invented."""
    ea = json.loads(json.dumps(EA_TURN_LEFT))
    ea["route"]["dist_m"] = None
    vocab, _ = emit_vocab(V2, None, engine_a=ea)
    g = vocab["g_str"]
    assert g["token"] == "TURN_LEFT"
    assert g["arg_mask"] == [0] * 8


def test_a_str_prepare_stop_from_lonmode_event():
    ea = json.loads(json.dumps(EA_FOLLOW))
    ea["speed_events"] = [{"token": "stop_at_point", "t_start_s": 1.0,
                           "t_end_s": 2.0, "stop_dist_m": 21.0, "dv": -6.0,
                           "arc_from_t0_m": 12.5}]
    ea["speed_profile"]["stops"] = True
    v2 = json.loads(json.dumps(V2))
    v2["symbols"]["actions"] = [{"verb": "prepare_stop", "direction": "none"}]
    vocab, _ = emit_vocab(v2, None, engine_a=ea)
    a = vocab["a_str"]
    assert a["token"] == "PREPARE_STOP"
    within = a["args"][4]                       # slot 4 = within_m
    assert a["arg_mask"][4] == 1 and within == 12.5
    # g_str also becomes STOP_AT (no junction outranks it here)
    assert vocab["g_str"]["token"] == "STOP_AT"
    # the VLM verb corroborates through the geometric checker
    verbs = a["corroboration"]["vlm_verbs"]
    assert verbs and verbs[0]["geometry"] == "ok"
    assert a["corroboration"]["agrees"] is True


def test_lane_change_gate_requires_displacement_and_valid_follow():
    """§4.2: latmaneuver fires on 120/201 clips — implausible. LANE_TARGET /
    PREPARE_LANE_CHANGE require route follow+valid AND >= 3.0 m displacement."""
    ea = json.loads(json.dumps(EA_FOLLOW))
    ea["lane_change_events"] = [{"token": "lc_left", "t_start_s": 1.0,
                                 "t_end_s": 2.0, "lat_m": 1.2,
                                 "arc_from_t0_m": 15.0}]
    vocab, _ = emit_vocab(V2, None, engine_a=ea)
    assert vocab["g_str"]["token"] == "FOLLOW_MAIN_ROAD"   # 1.2 m: gated out
    ea["lane_change_events"][0]["lat_m"] = 3.4
    vocab, _ = emit_vocab(V2, None, engine_a=ea)
    g = vocab["g_str"]
    assert g["token"] == "LANE_TARGET"
    assert g["args"][0] == 1.0 and g["arg_mask"][0] == 1   # +1 = left
    assert g["args"][1] == 15.0 and g["arg_mask"][1] == 1  # deadline arc
    assert vocab["a_str"]["token"] == "PREPARE_LANE_CHANGE"
    # on turn geometry the same event must NOT mint a lane change
    ea2 = json.loads(json.dumps(EA_TURN_LEFT))
    ea2["lane_change_events"] = [dict(ea["lane_change_events"][0])]
    vocab, _ = emit_vocab(V2, None, engine_a=ea2)
    assert vocab["g_str"]["token"] == "TURN_LEFT"


def test_vlm_primary_fallback_without_engine_a_is_tagged():
    """Legacy inputs (no engine A anywhere): the VLM token is emitted but
    tagged vlm-fused — stated, never silently defaulted."""
    v2 = json.loads(json.dumps(V2))
    v2["symbols"]["goal_kind"] = "turn_left"
    vocab, _ = emit_vocab(v2, None)
    g = vocab["g_str"]
    assert g["token"] == "TURN_LEFT"
    assert g["provenance"] == "vlm-fused" and g["src"] == "vlm"


def test_provenance_names_the_ego_prompt_and_unwhitelists_semantics(tmp_path):
    """§5 defect 1: 353/353 v2 records ran with ego-past + Engine A in the
    prompt — `_provenance.vlm = "vision"` was a lie, and `semantics` must not
    be whitelisted as inference-admissible for those records."""
    (tmp_path / "ego").mkdir()
    v2 = json.loads(json.dumps(V2))
    v2["_ego_prompt_mode"] = "past"
    (tmp_path / "v2.json").write_text(json.dumps([v2]))
    (tmp_path / "s.json").write_text(json.dumps([SAM3]))
    out = tmp_path / "fused"
    assert main(["--v2-json", str(tmp_path / "v2.json"), "--sam3",
                 str(tmp_path / "s.json"), "--ego-root", str(tmp_path / "ego"),
                 "--out", str(out)]) == 0
    rec = json.loads((out / "c1.json").read_text())
    assert rec["_provenance"]["vlm"] == \
        "vision+ego-past-prompt+engineA-prompt"
    assert rec["inference_admissible"] == ["perception"]
    assert "_inference_admissible_note" in rec


def test_engine_a_sidecar_reaches_the_vocab(tmp_path):
    (tmp_path / "ego").mkdir()
    (tmp_path / "v2.json").write_text(json.dumps([V2]))
    (tmp_path / "s.json").write_text(json.dumps([SAM3]))
    side = tmp_path / "ea.jsonl"
    side.write_text(json.dumps({"clip_id": "c1", "engine_a": EA_TURN_LEFT})
                    + "\n")
    out = tmp_path / "fused"
    assert main(["--v2-json", str(tmp_path / "v2.json"), "--sam3",
                 str(tmp_path / "s.json"), "--ego-root", str(tmp_path / "ego"),
                 "--engine-a", str(side), "--out", str(out)]) == 0
    rec = json.loads((out / "c1.json").read_text())
    assert rec["vocab"]["g_str"]["token"] == "TURN_LEFT"
    assert rec["ego"]["engine_a"]["route"]["token"] == "turn_left"
    assert rec["_provenance"]["engine_a"].startswith("privileged")
    # disjointness survives the richer vocab block
    assert "situation" not in json.dumps(rec["vocab"]).lower()


def test_uturn_confound_abstains_and_clean_uturn_maps_left():
    import s2_derive
    ea = json.loads(json.dumps(EA_TURN_LEFT))
    ea["route"].update(token="u_turn", uturn_roundabout_confounded=True)
    g = s2_derive.derive_g_str(ea, {})
    assert g["token"] == "NONE_ABSTAIN" and "confounded" in g["reason"]
    ea["route"]["uturn_roundabout_confounded"] = False
    g = s2_derive.derive_g_str(ea, {})
    assert g["token"] == "TURN_LEFT"


def test_resume_cruise_when_stopped_then_launching():
    import s2_derive
    ea = json.loads(json.dumps(EA_FOLLOW))
    ea["speed_profile"].update(v_t0_ms=0.1, v_max_future_ms=9.5, stops=True)
    ea["speed_events"] = [
        {"token": "hold_stop", "t_start_s": 0.0, "t_end_s": 1.0,
         "stop_dist_m": 0.0, "dv": 0.0, "arc_from_t0_m": 0.0},
        {"token": "launch", "t_start_s": 2.0, "t_end_s": 3.0,
         "stop_dist_m": None, "dv": 6.0, "arc_from_t0_m": 1.0}]
    a = s2_derive.derive_a_str(ea, {})
    assert a["token"] == "RESUME_CRUISE"
    assert a["args"][0] == 9.5 and a["arg_mask"][0] == 1
    # and the initial hold_stop must not become a STOP_AT goal
    g = s2_derive.derive_g_str(ea, {})
    assert g["token"] != "STOP_AT"


def test_derive_never_reads_situations():
    """Structural disjointness: a poisoned engine_a with a situations block
    must produce byte-identical labels to the same engine_a without it."""
    import s2_derive
    clean = s2_derive.derive_g_str(EA_TURN_LEFT, {})
    poisoned_ea = json.loads(json.dumps(EA_TURN_LEFT))
    poisoned_ea["situations"] = {"intersection": True, "lane_change": True}
    poisoned = s2_derive.derive_g_str(poisoned_ea, {})
    assert clean == poisoned
    assert "situation" not in json.dumps(poisoned).lower()
