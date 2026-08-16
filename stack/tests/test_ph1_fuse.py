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

import s2_derive  # noqa: E402
from ph1_fuse import (build_tracks, census_phrase, census_state,  # noqa: E402
                      corroborate, emit_vocab, lane_phrase, main,
                      scenario_line)

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
    """⛔ THIS PIN WAS INVERTED 2026-08-16, and the old form is kept below as
    the negative control it now needs.

    It used to assert `vocab["g_tac_lat"]["token"] in TACTICAL_LAT_ACTIONS` —
    i.e. it PINNED a field named for the **`g_tac` GOAL** vocabulary while
    holding a member of the **`a_tac` ACTION** vocabulary. The two are
    DISJOINT: `TACTICAL_GOAL_TOKENS_LAT` is
    (ANCHOR_GOAL, CORRIDOR_OFFSET, EVADE_IN_CORRIDOR, LAT_UNCONSTRAINED) and
    contains no `LANE_KEEP`. So the assertion passed forever while any
    consumer joining `g_tac_lat` to a goal head silently received actions —
    the S2 `"no agents"` class, *a shape that reads like the thing it is not*
    (`TACTICAL_LABEL_VALIDATION.md` §1.3, third defect).

    The pin changed because the FIELD was renamed to what it holds. It is not
    deleted: the same content is now pinned under `a_tac_*`, and the goal keys
    are pinned EMPTY-WITH-A-REASON, because nothing in this fuse derives them.
    """
    from tanitad.models.v6 import (STRATEGIC_GOAL_TOKENS,
                                   TACTICAL_GOAL_TOKENS_LAT,
                                   TACTICAL_GOAL_TOKENS_LON,
                                   TACTICAL_LAT_ACTIONS, TACTICAL_LON_ACTIONS)
    vocab, _ = emit_vocab(V2, None)
    assert vocab["g_str"]["token"] in STRATEGIC_GOAL_TOKENS
    # the ACTION content, under the ACTION name
    assert vocab["a_tac_lat"]["token"] in TACTICAL_LAT_ACTIONS
    assert vocab["a_tac_lon"]["token"] in TACTICAL_LON_ACTIONS
    # ⛔ the negative control: the emitted action token is NOT a goal token,
    # which is exactly why the old assertion was a mismatch and not a typo
    assert vocab["a_tac_lat"]["token"] not in TACTICAL_GOAL_TOKENS_LAT
    # the goal axes: declared unavailable, never silently filled or dropped
    for k, valid in (("g_tac_lat", TACTICAL_GOAL_TOKENS_LAT),
                     ("g_tac_lon", TACTICAL_GOAL_TOKENS_LON)):
        assert vocab[k]["token"] is None
        assert vocab[k]["unavailable_reason"]
        assert vocab[k]["token"] not in valid          # None is not a token


def test_no_known_verb_maps_to_nothing_and_unknown_ones_raise():
    """⛔ THE DEFECT THAT MADE ALL THREE POSSIBLE: a verb could map to NOTHING
    by falling off the end of a substring list, and a silent None is
    indistinguishable from a verb that was never spoken (the C77 family).

    MEASURED pre-fix (`TACTICAL_LABEL_VALIDATION.md` §1.3, 270 emissions):
    `reduce_to` — the VLM's ONLY deceleration verb — matched nothing on either
    axis, dropping 49/270 = 18.1 % of all emissions.
    """
    import ph1_fuse
    from ph0_v2 import ACTION_VERBS
    # (a) the table is TOTAL over the real closed vocabulary — not a subset
    assert set(ph1_fuse.VLM_VERB_TO_A_TAC) == set(ACTION_VERBS)
    # (b) no verb is silent on BOTH axes except the one that DECLARES it
    silent = {v for v in ACTION_VERBS
              if ph1_fuse.map_vlm_action(v, "left")[:2] == (None, None)}
    assert silent == {"prepare_exit"}, silent
    # (c) an unknown verb RAISES — it can never again become a silent None
    with pytest.raises(ph1_fuse.UnmappedActionVerb):
        ph1_fuse.map_vlm_action("teleport_home", "none")
    with pytest.raises(ph1_fuse.UnmappedActionVerb):
        ph1_fuse.map_vlm_action("prepare_lane_change", "sideways")


def test_reduce_to_the_only_deceleration_verb_now_lands():
    """Defect 1 pinned by outcome, with the pre-fix mechanism as the negative
    control: the old LON substring list really did miss `reduce_to`."""
    import ph1_fuse
    old_lon_rules = (("brake", "BRAKE_TO"), ("stop", "BRAKE_TO"),
                     ("decel", "BRAKE_TO"), ("yield", "YIELD_MERGE"),
                     ("merge", "YIELD_MERGE"), ("creep", "CREEP"),
                     ("hold", "HOLD"), ("wait", "HOLD"), ("follow", "FOLLOW"),
                     ("cruise", "CRUISE"), ("accel", "CRUISE"))
    for d in ("none", "left", "right"):
        txt = f"reduce_to_{d}"
        assert not any(s in txt for s, _ in old_lon_rules), "control broken"
        assert ph1_fuse.map_vlm_action("reduce_to", d)[1] == "BRAKE_TO"
    v2 = json.loads(json.dumps(V2))
    v2["symbols"]["actions"] = [{"verb": "reduce_to", "direction": "none"}]
    del v2["signs"]
    vocab, _ = emit_vocab(v2, None)
    assert ["vlm", "BRAKE_TO"] in vocab["a_tac_lon"]["votes"]


def test_hold_corridor_is_lateral_and_makes_no_longitudinal_claim():
    """Defect 2. `hold_corridor` matched the LON substring `"hold"` → the
    LONGITUDINAL token `HOLD` on 159/270 = 58.9 % of emissions, which (with
    defect 1) made the VLM's longitudinal leg a CONSTANT: κ exactly 0.0000
    against BOTH other legs, n=162. A constant is not a weak signal — it is
    no signal, and it was counted as a vote."""
    import ph1_fuse
    for d in ("none", "left", "right"):
        lat, lon, _ = ph1_fuse.map_vlm_action("hold_corridor", d)
        assert lat == "LANE_KEEP"
        assert lon is None, "a LATERAL verb must cast no LONGITUDINAL vote"
    v2 = json.loads(json.dumps(V2))          # V2's action IS hold_corridor
    del v2["signs"]
    vocab, _ = emit_vocab(v2, None)
    assert ["vlm", "LANE_KEEP"] in vocab["a_tac_lat"]["votes"]
    assert ["vlm", None] in vocab["a_tac_lon"]["votes"]
    assert not any(t == "HOLD" for s, t in vocab["a_tac_lon"]["votes"])


def test_alpamayo_axes_are_parsed_and_the_reasoning_cannot_vote():
    """Defect 3: the Alpamayo leg ran the SAME substrings over
    `json.dumps(meta_action)[:400]` — a blob containing the free-text `cot`.
    A rationale could out-vote the axis it was meant to explain.

    The adversarial case below is real in shape: `cot` strings like *"Keep
    distance to the lead vehicle…"* (n=282) and *"Slow down for the roundabout
    because of the yield sign ahead"* (n=61) are top-of-table in
    `raw/a1_alpamayo_taxonomy.json`.
    """
    import ph1_fuse
    raw = json.dumps({
        "raw_outputs": ["Longitudinal: Maintain Speed. Lateral: Go Straight. "
                        "Lane: Lane Keep."],
        "cot": ["Stop and yield to the merging vehicle, then follow it"]})
    axes = ph1_fuse.parse_alpamayo_axes(raw)
    assert axes["longitudinal"] == "Maintain Speed"
    assert axes["lane"] == "Lane Keep"
    lat, lon, notes = ph1_fuse.map_alpamayo_axes(axes)
    assert lat == "LANE_KEEP"
    # ⛔ the cot says stop/yield/follow; NONE of them may become a vote
    assert lon is None and "reason" in notes["lon"]
    # and the axis itself is what abstained, not a parse failure
    assert "magnitude-typed" in notes["lon"]


def test_alpamayo_turn_has_no_v6_token_and_is_never_bent_into_a_neighbour():
    """⚠️ `Turn Left`/`Turn Right` are 186 of 4,729 clips (3.94 %) and
    `TACTICAL_LAT_ACTIONS` has NO `TURN_*` member. The honest emission is an
    abstain with a named reason — `NUDGE_L` would be a fabrication. The
    vocabulary is REPORTED, never edited (the tuples size embedding tables)."""
    import ph1_fuse
    from tanitad.models.v6 import TACTICAL_LAT_ACTIONS
    assert not any(t.startswith("TURN") for t in TACTICAL_LAT_ACTIONS)
    for v in ("Turn Left", "Turn Right"):
        lat, _, notes = ph1_fuse.map_alpamayo_axes({"lane": v})
        assert lat is None and "no v6 token" in notes["lat"]
    # a value outside the observed taxonomy is a FINDING, not a coercion
    with pytest.raises(ph1_fuse.UnmappedActionVerb):
        ph1_fuse.map_alpamayo_axes({"lane": "Pirouette Left"})


def test_ego_and_vlm_are_ONE_block_and_can_never_corroborate_each_other():
    """⛔ PRIORITY-2 PIN. MEASURED (`TACTICAL_LABEL_VALIDATION.md` §1.4):
    `_ego_prompt_mode == 'past'` on 201/201, and the block printed into the
    VLM's prompt carries `motion` and `turning` — EXACTLY the two fields the
    ego voter reads. Signature: VLM↔ego LAT κ 0.7608 vs Alpamayo↔VLM 0.1717.

    ⇒ A 2-of-3 majority could be carried by ONE SOURCE COUNTED TWICE. This
    pins that it no longer can: agreement between ego and the VLM is ONE
    vote, and `corroborated` stays False without an INDEPENDENT block.
    """
    import ph1_fuse
    # ego and vlm agreeing perfectly: one block, one vote, NOT corroborated
    out = ph1_fuse.block_vote([("ego", "LANE_KEEP"), ("vlm", "LANE_KEEP")],
                              ("LANE_KEEP", "NUDGE_L"))
    assert out["token"] == "LANE_KEEP"
    assert out["n_blocks_speaking"] == 1
    assert out["corroborated"] is False, "a source and its own echo"
    assert out["blocks"]["ego+vlm"]["members"] == ["ego", "vlm"]
    # the old 2-of-3 helper is GONE, not renamed
    assert not hasattr(ph1_fuse, "majority")
    # an INDEPENDENT third leg agreeing is what corroboration means
    out = ph1_fuse.block_vote(
        [("ego", "LANE_KEEP"), ("vlm", "LANE_KEEP"),
         ("alpamayo", "LANE_KEEP")], ("LANE_KEEP", "NUDGE_L"))
    assert out["corroborated"] is True and out["n_blocks_speaking"] == 2
    # blocks disagreeing: Alpamayo decides (full camera rig, external) and the
    # disagreement is RECORDED, never averaged
    out = ph1_fuse.block_vote(
        [("ego", "LANE_KEEP"), ("vlm", "LANE_KEEP"),
         ("alpamayo", "NUDGE_L")], ("LANE_KEEP", "NUDGE_L"))
    assert out["token"] == "NUDGE_L" and out["provenance"] == "alpamayo"
    assert out["corroborated"] is False
    assert out["blocks"]["ego+vlm"]["token"] == "LANE_KEEP"
    # ⛔ two ego+vlm members disagreeing is a FINDING (the VLM departing from
    # its own prompt), so the block abstains rather than picking one
    out = ph1_fuse.block_vote([("ego", "LANE_KEEP"), ("vlm", "NUDGE_L")],
                              ("LANE_KEEP", "NUDGE_L"))
    assert out["token"] is None and out["n_blocks_speaking"] == 0
    assert out["blocks"]["ego+vlm"]["internal_disagreement"] is True


def test_two_of_three_can_no_longer_be_satisfied_by_ego_plus_vlm():
    """The end-to-end half of the pin, through `emit_vocab`: ego and the VLM
    both say LANE_KEEP and Alpamayo says otherwise. Under the retired
    majority that was 2 votes to 1 and LANE_KEEP won."""
    v2 = json.loads(json.dumps(V2))
    del v2["signs"]
    alp = {"meta_action": json.dumps({
        "raw_outputs": ["Longitudinal: Maintain Speed. Lateral: Steer Left. "
                        "Lane: Left Lane Change."], "cot": ["Move over"]})}
    vocab, _ = emit_vocab(v2, alp)
    lat = vocab["a_tac_lat"]
    assert [s for s, t in lat["votes"] if t == "LANE_KEEP"] == ["ego", "vlm"]
    assert lat["token"] == "LANE_CHANGE_L"       # NOT the 2-vote echo
    assert lat["provenance"] == "alpamayo" and lat["corroborated"] is False


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
                 [tuple(v) for v in vocab["a_tac_lat"]["votes"]]
                 if s == "ego")
    assert votes["ego"] == "NUDGE_L"    # the vote LANDS now
    v2["ego_state"]["turning"] = "turning_right"
    vocab, _ = emit_vocab(v2, None)
    assert ["ego", "NUDGE_R"] in vocab["a_tac_lat"]["votes"]
    v2["ego_state"]["turning"] = "straight"             # unchanged behavior
    vocab, _ = emit_vocab(v2, None)
    assert ["ego", "LANE_KEEP"] in vocab["a_tac_lat"]["votes"]


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


def test_geometric_gate_can_no_longer_produce_either_lane_change_token():
    """⛔ THE §LC PIN (PI ruling 2026-08-16, verbatim: "stop emitting
    lane_target/Prepare Lane change from geometric gate").

    The PI adjudicated 18 of the 19 LANE_TARGET labels this path emitted on
    aug120 and called 14 wrong. NO amount of observed lateral displacement,
    at any magnitude, may mint either token — this is removal, not retuning,
    so the test sweeps a range that brackets and far exceeds the old 3.0 m
    gate. The clip falls through to its ROUTE token instead.
    """
    for lat in (1.2, 3.4, 7.0, 25.0):
        ea = json.loads(json.dumps(EA_FOLLOW))
        ea["lane_change_events"] = [{"token": "lc_left", "t_start_s": 1.0,
                                     "t_end_s": 2.0, "lat_m": lat,
                                     "arc_from_t0_m": 15.0}]
        vocab, _ = emit_vocab(V2, None, engine_a=ea)
        assert vocab["g_str"]["token"] == "FOLLOW_MAIN_ROAD", lat
        assert vocab["a_str"]["token"] != "PREPARE_LANE_CHANGE", lat
    # ... and the same on a right-hand event and on turn geometry
    ea["lane_change_events"][0]["token"] = "lc_right"
    vocab, _ = emit_vocab(V2, None, engine_a=ea)
    assert vocab["g_str"]["token"] == "FOLLOW_MAIN_ROAD"
    ea2 = json.loads(json.dumps(EA_TURN_LEFT))
    ea2["lane_change_events"] = [dict(ea["lane_change_events"][0])]
    vocab, _ = emit_vocab(V2, None, engine_a=ea2)
    assert vocab["g_str"]["token"] == "TURN_LEFT"


def test_lane_target_is_never_emitted_by_any_path():
    """§LC: `LANE_TARGET` left g_str emission ENTIRELY — a lane change is a
    MEANS (a_str), not a GOAL. Geometry, VLM-primary fallback and the
    ROUTE_TO remap set are all checked. ⚠️ The token STAYS in the v6
    vocabulary (the tuples size embedding tables, v6.py:3281) — this pins
    EMISSION, and the vocabulary assert below pins that it is still there."""
    from tanitad.models.v6 import STRATEGIC_GOAL_TOKENS as V6G
    assert "LANE_TARGET" in V6G, "vocabulary must NOT lose the token"
    ea = json.loads(json.dumps(EA_FOLLOW))
    ea["lane_change_events"] = [{"token": "lc_left", "t_start_s": 1.0,
                                 "t_end_s": 2.0, "lat_m": 9.9,
                                 "arc_from_t0_m": 15.0}]
    v2 = json.loads(json.dumps(V2))
    for goal_kind in ("lane_target", "follow_main_road", "route_to"):
        v2["symbols"]["goal_kind"] = goal_kind
        for engine_a in (ea, None):
            vocab, _ = emit_vocab(v2, None, engine_a=engine_a)
            assert vocab["g_str"]["token"] != "LANE_TARGET", \
                (goal_kind, engine_a is not None)
    # the VLM claiming it outright abstains WITH A REASON, never silently
    v2["symbols"]["goal_kind"] = "lane_target"
    vocab, _ = emit_vocab(v2, None, engine_a=None)
    assert vocab["g_str"]["token"] == "NONE_ABSTAIN"
    assert "LANE_TARGET" in vocab["g_str"]["reason"]


def test_observed_lane_change_is_recorded_as_corroboration_not_source():
    """§LC: observed-but-unrequired is a REAL category. The observation must
    survive in the record — dropping it would trade one silent failure for
    another — but flagged `is_label_source: False`."""
    ea = json.loads(json.dumps(EA_FOLLOW))
    ea["lane_change_events"] = [{"token": "lc_right", "t_start_s": 1.0,
                                 "t_end_s": 2.0, "lat_m": -4.4,
                                 "arc_from_t0_m": 15.0}]
    vocab, _ = emit_vocab(V2, None, engine_a=ea)
    for blk in (vocab["g_str"], vocab["a_str"]):
        obs = blk["corroboration"]["observed_lane_change"]
        assert obs["token"] == "lc_right" and obs["lat_m"] == -4.4
        assert obs["is_label_source"] is False
    req = vocab["a_str"]["corroboration"]["lane_change_requirement"]
    assert req["required"] is None            # UNKNOWN, never False-by-default
    assert set(req["missing"]) == set(s2_derive.LANE_CONTEXT_INPUTS)
    assert "lane context unavailable" in vocab["a_str"]["reason"]


def test_prepare_lane_change_needs_a_ROUTE_SERVING_requirement():
    """§LC part 2: the token is admissible ONLY when the route (or the main
    road, no route set) demands a lane the ego is not in — and then its
    direction/deadline come from the ROUTE, never from the observation."""
    ea = json.loads(json.dumps(EA_FOLLOW))                # NO lateral event
    ctx = {"n_lanes_same_direction": 3, "ego_lane_idx": 0,
           "route_lane_idx": 2, "lane_continues": True}
    a = s2_derive.derive_a_str(ea, None, lane_context=ctx)
    assert a["token"] == "PREPARE_LANE_CHANGE"
    assert a["args"][0] == 1.0                            # +1 = left (0 -> 2)
    assert a["sources"] == ["lane_context.route_serving:follow_main_road"]
    # the ego already in the serving lane => NOT required, even with a big
    # observed displacement present
    ea2 = json.loads(json.dumps(EA_FOLLOW))
    ea2["lane_change_events"] = [{"token": "lc_left", "t_start_s": 1.0,
                                  "t_end_s": 2.0, "lat_m": 8.0,
                                  "arc_from_t0_m": 15.0}]
    ok = {**ctx, "ego_lane_idx": 2}
    a2 = s2_derive.derive_a_str(ea2, None, lane_context=ok)
    assert a2["token"] != "PREPARE_LANE_CHANGE"
    assert a2["corroboration"]["lane_change_requirement"]["required"] is False


def test_lane_change_requirement_cannot_read_the_observation():
    """⛔ The structural half: `lane_change_requirement()` must be UNABLE to
    reach `lane_change_events`. Same discipline as the situations allowlist —
    a rule that only holds by habit has already failed here once."""
    bare = json.loads(json.dumps(EA_FOLLOW))
    ctx = {"n_lanes_same_direction": 2, "ego_lane_idx": 0,
           "route_lane_idx": 0, "lane_continues": True}
    for lat in (-99.0, -4.0, 0.0, 4.0, 99.0):
        for tok in ("lc_left", "lc_right"):
            ea = json.loads(json.dumps(EA_FOLLOW))
            ea["lane_change_events"] = [{"token": tok, "t_start_s": 1.0,
                                         "t_end_s": 2.0, "lat_m": lat,
                                         "arc_from_t0_m": 15.0}]
            for c in (None, ctx):
                assert s2_derive.lane_change_requirement(ea, c) == \
                    s2_derive.lane_change_requirement(bare, c), \
                    f"requirement moved when only the OBSERVATION did ({tok} {lat})"
                assert (s2_derive.derive_a_str(ea, None, lane_context=c)["token"]
                        == s2_derive.derive_a_str(bare, None,
                                                  lane_context=c)["token"])


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


# =========================================================================== #
# ⛔ C77 — AN ABSENT MEASUREMENT MUST NEVER RENDER AS A CONFIDENT NEGATIVE     #
# =========================================================================== #
def test_absent_perception_never_renders_as_a_finding_about_the_world():
    """The PI's `03ba450b` note: *"not agent said by the pipeline. In the
    picture of frame 9 there is clear a car as incoming traffic"*.

    `", ".join(census) or "no agents"` turned an EMPTY census into a positive
    claim. MEASURED on fused aug120: "no agents" on 119/201 records, of which
    115 (96.6 %) ALREADY carried `perception.absent` — the structured layer
    knew and the prose contradicted it. Absence and a measured zero are
    different claims and must render differently.
    """
    # (a) detector never ran for this clip
    cen = census_state([], absent_reason="sam3 absent for this clip")
    assert cen["state"] == "unavailable" and cen["counts"] is None
    assert cen["n_agents"] is None       # ⛔ never 0 — that would be a count
    assert "UNAVAILABLE" in census_phrase(cen)
    # (b) detector ran but processed no frames — also unavailability
    assert census_state([], n_frames=0)["state"] == "unavailable"
    # (c) detector ran and returned nothing — a MEASUREMENT of the detector
    cen = census_state([], n_frames=8)
    assert cen["state"] == "measured" and cen["n_agents"] == 0
    assert census_phrase(cen) == "0 agents detected"
    # ⛔ the regression that started this: the old literal must be gone from
    # every branch, so it can never be read as a claim about the world again
    for c in (census_state([], absent_reason="x"), census_state([], n_frames=0),
              census_state([], n_frames=8)):
        assert census_phrase(c) != "no agents"


def test_scenario_line_carries_the_census_state_not_a_bare_negative():
    tracks = build_tracks(SAM3["frames"])
    line = scenario_line(V2, tracks, n_frames=len(SAM3["frames"]))
    assert "1 car" in line
    absent = scenario_line(V2, [], absent_reason="sam3 absent for this clip")
    assert "UNAVAILABLE" in absent and "no agents" not in absent
    empty = scenario_line(V2, [], n_frames=8)
    assert "0 agents detected" in empty


def test_empty_census_cannot_produce_a_flagged_empty_verdict():
    """The corroboration leg of the same rule: `flagged_empty_urban` is a
    FINDING and an absent detector may not produce one."""
    cor, _ = corroborate(V2, {}, [], sam3_absent=True)
    assert cor["census_vs_scene"]["verdict"] == "not_computable"
    # SAM3 present but zero frames processed: still not a finding
    cor, _ = corroborate(V2, {"frames": {}}, [], sam3_absent=False)
    assert cor["census_vs_scene"]["verdict"] == "not_computable"
    # SAM3 ran with frames and found no agents: NOW it is a measured flag
    cor, _ = corroborate(V2, SAM3, [], sam3_absent=False)
    assert cor["census_vs_scene"]["verdict"] == "flagged_empty_urban"


def test_lane_count_phrase_names_its_scope():
    """⚠️ `lanes_visible` is B1-defined as the EGO'S CARRIAGEWAY only
    (ph0_v2.py:140). The PI read `"rural 1-lane"` as "a one-lane road" and
    objected — correctly, of the phrase. The count was right by its own
    definition; the RENDERING hid the scope. It must not be ambiguous."""
    assert lane_phrase({"road_type": "rural", "lanes_visible": 1}) == \
        "rural 1-lane-ego-carriageway"
    # B1's documented 0 = "unclear" escape is an ABSENT count, not "0 lanes"
    out = lane_phrase({"road_type": "urban", "lanes_visible": 0})
    assert "UNCLEAR" in out and "0-lane" not in out
    assert "UNCLEAR" in lane_phrase({"road_type": "urban"})


def test_lane_ends_requires_a_change_but_never_invents_a_direction():
    """§LC edge: the ego lane IS the serving lane but does not continue (lane
    ends / forced merge). A change IS required; WHICH WAY is not derivable
    from these inputs, so the direction slot stays UNSET — the same args
    discipline as the `8dc5d14d…` null-dist_m row, never a fabricated ±1."""
    ctx = {"n_lanes_same_direction": 2, "ego_lane_idx": 1,
           "route_lane_idx": 1, "lane_continues": False}
    req = s2_derive.lane_change_requirement(EA_FOLLOW, ctx)
    assert req["required"] is True and req["direction"] is None
    a = s2_derive.derive_a_str(EA_FOLLOW, None, lane_context=ctx)
    assert a["token"] == "PREPARE_LANE_CHANGE"
    assert a["arg_mask"][0] == 0 and a["args"][0] == 0.0   # UNSET, not 0.0-meant


def test_fused_record_carries_the_machine_readable_census(tmp_path):
    """The census must reach the RECORD, not only the prose — downstream
    should read a field, never parse `scenario_description`. Both states are
    exercised end-to-end through `main()`."""
    (tmp_path / "ego").mkdir()
    (tmp_path / "v2.json").write_text(json.dumps([V2, _v2_second_clip()]))
    (tmp_path / "s.json").write_text(json.dumps({"engine": "sam3",
                                                 "clips": [SAM3]}))
    out = tmp_path / "fused"
    assert main(["--v2-json", str(tmp_path / "v2.json"), "--sam3",
                 str(tmp_path / "s.json"), "--ego-root", str(tmp_path / "ego"),
                 "--out", str(out), "--missing-sam3-ok", "no sam3 for c2"]) == 0
    c1 = json.loads((out / "c1.json").read_text())["perception"]["census"]
    assert c1["state"] == "measured" and c1["counts"] == {"car": 1}
    c2 = json.loads((out / "c2.json").read_text())["perception"]["census"]
    assert c2["state"] == "unavailable"
    assert c2["counts"] is None and c2["n_agents"] is None
    # and the prose agrees with the field on BOTH clips
    assert "no agents" not in json.loads(
        (out / "c2.json").read_text())["scenario_description"]


def test_legacy_two_arg_scenario_line_still_reports_agents_it_has():
    """⚠️ `colab/s2_lab_lib.py:832` calls `scenario_line(r, tracks)` with two
    positional args and CANNOT pass `n_frames`. Collapsing `None` into 0 would
    make that caller say "unavailable" with agents plainly in the census — the
    C77 defect inverted. Tracks are proof the detector ran."""
    tracks = build_tracks(SAM3["frames"])
    line = scenario_line(V2, tracks)                    # no n_frames, as legacy
    assert "1 car" in line and "UNAVAILABLE" not in line
    cen = census_state(tracks)
    assert cen["state"] == "measured" and cen["n_agents"] == 1
    # empty census + unknown frame count is genuinely undecidable -> unavailable
    cen0 = census_state([])
    assert cen0["state"] == "unavailable" and cen0["n_agents"] is None
    assert "indistinguishable" in cen0["reason"]
    # an EXPLICIT zero frame count keeps its own, more specific reason
    assert census_state([], n_frames=0)["reason"] == \
        "no frames processed by the detector"
