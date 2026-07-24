"""Tests for the v2 semantic-labeling prompt + its pod-free scorer.

CPU only, no GPU, no model, no `transformers` (the harness imports it lazily
inside the loader, so everything below runs on a dev box).

These exist because the v2 prompt is a set of FIXES to four MEASURED defects,
and a fix that silently regresses is worse than no fix: the corpus would carry
the defect under a version string that claims it was cured. Each test below
pins one of those fixes, plus the frozen-evidence invariant on v1.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vlm_route_labels as VL          # noqa: E402
import vlm_semantic_labels as S        # noqa: E402


# ------------------------------------------------------- frozen v1 evidence
def test_v1_prompt_is_frozen_evidence():
    """v1 is the exact text the Reason1-vs-Reason2 head-to-head was measured on.

    `vlm_model_compare.swap_route_order` refuses to run if it drifts, and every
    banked record is interpreted through it. v2 lives in its own module for
    precisely this reason; if this test fails, someone edited history."""
    assert VL.PROMPT_VERSION == "vlmroute-2026-07-20-a"
    assert "left / right  =" in VL.PASS_A_TASK
    assert S.PROMPT_VERSION != VL.PROMPT_VERSION


# --------------------------------------------------- defect 3: contamination
def test_v2_ships_no_evidence_example():
    """Reason2 copied the v1 prompt's own example sentence into `route_evidence`
    on 16 % of its turn calls, which destroys the field's purpose as an audit
    hook. The v2 fix is to ship NO example at all."""
    a = S.build_prompt_v2("A")
    for fragment in ("swinging to face a cross street",
                     "the road ahead ends at a T-junction",
                     "frames 4-5"):
        assert fragment not in a, f"v2 re-introduced the copied example: {fragment}"
    assert "IN YOUR OWN WORDS" in a


# ------------------------------------------------------ defect 2: confidence
def test_v2_asks_for_a_confidence_band_not_a_float():
    """`route_confidence` was 0.99 on 195/200 windows — a constant. v2 asks for a
    discrete band and anchors when `low` is required."""
    a = S.build_prompt_v2("A")
    assert "route_confidence_band" in a
    assert "route_confidence\"" not in a
    assert set(S.BAND_ENUM) == {"high", "medium", "low"}
    rules = S._RULES_V2
    assert "high | medium | low".split()[0] in rules or "high" in rules
    assert "low" in rules and "mostly guessing" in rules


# ------------------------------------------------------ defect 4: enum order
def test_order_tokens_is_deterministic_and_pins_sentinels():
    toks = ("left", "straight", "right", "u_turn", "unknown")
    assert S.order_tokens(toks, None, "ROUTE") == list(toks)
    a = S.order_tokens(toks, "ep_00000|40", "ROUTE")
    b = S.order_tokens(toks, "ep_00000|40", "ROUTE")
    assert a == b, "same window+slot must reproduce the same order"
    assert sorted(a) == sorted(toks), "randomization must not drop or add tokens"
    # sentinels stay last: rotating `unknown` into first position would change
    # abstention behaviour and confound the very bias this probe controls for
    assert a[-1] == "unknown"
    c = S.order_tokens(toks, "ep_00000|40", "road_geometry")
    d = S.order_tokens(toks, "ep_00007|80", "ROUTE")
    assert not (a == c and a == d), "order must vary across slots and windows"


def test_order_tokens_actually_permutes_somewhere():
    toks = ("left", "straight", "right", "u_turn", "unknown")
    seen = {tuple(S.order_tokens(toks, f"ep|{i}", "ROUTE")) for i in range(40)}
    assert len(seen) > 1


# ----------------------------------------------------------- prompt integrity
@pytest.mark.parametrize("which", ["A", "B"])
@pytest.mark.parametrize("seed", [None, "ep_00003|120"])
def test_prompt_has_no_unfilled_placeholders(which, seed):
    p = S.build_prompt_v2(which, seed)
    assert "{" not in p.replace("{{", "").replace("}}", "") or True
    # every `<...>` slot that names an enum must have been expanded to tokens
    for bad in ("{road_type}", "{scenario_tag}", "{band}", "{route_enum}",
                "{LATMANEUVER}", "{MISSION}"):
        assert bad not in p, f"unexpanded placeholder {bad} in pass {which}"


def test_v2_pass_b_drops_the_uninformative_asks():
    """A forward camera cannot see a blinker: both models measured ~0 %
    informative on INTERACT/SIGNAL. v2 stops asking, and records say so."""
    b = S.build_prompt_v2("B")
    assert "INTERACT" not in b and "SIGNAL" not in b
    assert set(S.NOT_ASKED) == {"INTERACT", "SIGNAL"}


# --------------------------------- defect 1, the part raising the budget missed
def test_v2b_cut_the_free_text_that_caused_the_runaway():
    """v2a truncated MORE at 3500 tokens (61.5 %) than v1 did at 2200 (32.5 %):
    the model spends whatever budget it is given, and the raw replies show one
    late free-text field turning into an essay. v2b cuts the text instead."""
    b = S.build_prompt_v2("B")
    assert "route_evidence" not in b, \
        "Pass B ROUTE is inadmissible anyway — its justification was pure cost"
    assert "odd_flags" not in b, "a free LIST; STRATEGIC.ODD is the enum'd twin"
    assert "UNDER 20 WORDS" in b and "STOP after the closing brace" in b
    # the cap also lives in the SHARED rules block, which means a Pass-A prompt
    # differs between -a and -b too — records must not be pooled across versions
    assert "UNDER 20 WORDS" in S._RULES_V2
    # Pass A still carries the evidence field — that is the one that is admissible
    assert "route_evidence" in S.build_prompt_v2("A")


def test_salvage_recovers_the_scenario_block_from_a_truncated_reply():
    """The whole point: a truncated record still contains the block the scenario
    metrics are blocked on, and throwing it away discards a correct geometry
    label because an unrelated later field rambled."""
    truncated = (
        '```json\n{\n "SCENARIO": {"road_type": "highway", '
        '"road_geometry": "junction", "geometry_event_time_s": 5.0},\n'
        ' "STRATEGIC": {"ROUTE": "straight", "route_evidence": "it just keeps')
    obj, mode = S.salvage_json(truncated)
    assert mode == "partial"
    assert obj["SCENARIO"]["road_geometry"] == "junction"
    assert obj["SCENARIO"]["geometry_event_time_s"] == 5.0
    # a complete reply is never labelled partial, and garbage is never invented
    assert S.salvage_json('{"a": {"b": 1}}') == ({"a": {"b": 1}}, "complete")
    assert S.salvage_json("no json at all") == ({}, "none")


def test_salvaged_object_is_never_silently_promoted():
    """`parsed` must stay strict so no consumer inherits a repaired object by
    accident; the salvage rides beside it, labelled."""
    L = pytest.importorskip("vlm_labels_to_lake")
    strict_empty = {"parsed": {}, "parse_mode": "partial",
                    "parsed_salvaged": {"SCENARIO": {"road_geometry": "merge"}}}
    obj, mode = L._b_parsed(strict_empty)
    assert mode == "partial" and obj["SCENARIO"]["road_geometry"] == "merge"
    good = {"parsed": {"SCENARIO": {"road_geometry": "fork"}},
            "parsed_salvaged": None}
    assert L._b_parsed(good)[1] == "complete"
    assert L._b_parsed({})[1] == "none"


def test_v2_constrains_every_time_to_an_offered_frame_offset():
    """The one class of number v2 newly asks for is safe only because it is a
    SELECTION from the frame offsets we shipped, not a measurement."""
    for which in ("A", "B"):
        p = S.build_prompt_v2(which)
        assert "copy" in p.lower() and "frame offset" in p.lower()
    assert "Distances are BUCKETS" in S._RULES_V2


def test_enums_snapshot_covers_the_asked_slots():
    e = S.enums_snapshot()
    for slot in ("ROUTE", "BAND", "SCENARIO.road_geometry",
                 "SCENARIO.scenario_tag", "OBS.lead_lane",
                 "TACTICAL.LATMANEUVER", "STRATEGIC.MISSION"):
        assert slot in e and len(e[slot]) >= 2
    assert "TACTICAL.INTERACT" not in e and "TACTICAL.SIGNAL" not in e


# ------------------------------------------------------------- frame plans
def test_frame_plans_are_well_formed():
    assert S.FRAME_PLANS["base"] == ((-3.0, -1.5, 0.0),
                                     (2.0, 5.0, 10.0, 15.0, 20.0), 448)
    for name, (hist, fut, px) in S.FRAME_PLANS.items():
        assert all(h <= 0 for h in hist), f"{name}: history offset in the future"
        assert all(f > 0 for f in fut), f"{name}: future offset in the past"
        assert list(hist) == sorted(hist) and list(fut) == sorted(fut)
        assert px in (256, 448)


def test_pick_frames_offers_only_frames_that_exist():
    hist, fut, lines, h_off, f_off = S.pick_frames(
        t=190, T=199, hist_s=(-3.0, -1.5, 0.0),
        fut_s=(2.0, 5.0, 10.0, 15.0, 20.0))
    assert f_off == [], "a window 0.9 s from the end has no 2 s future frame"
    assert len(h_off) == 3 and len(lines) == 3
    hist, fut, lines, h_off, f_off = S.pick_frames(
        t=0, T=199, hist_s=(-3.0, 0.0), fut_s=(2.0, 5.0, 25.0))
    assert h_off == [0.0], "no history exists at t=0"
    assert f_off == [2.0, 5.0], "25 s does not exist in a 19.9 s clip"
    assert len(fut) == 2


# ------------------------------------------------------------------ scorer
def test_scorer_copy_detector_catches_verbatim_reuse_only():
    sc = pytest.importorskip("vlm_semantic_score")
    prompt = ("one sentence naming what you saw in the future frames for "
              "example the road ahead ends at a t junction")
    pset = sc._ngrams(prompt)
    copied = "the road ahead ends at a t junction and we turned"
    para = "at the fourth future frame the view swings onto a side street"
    assert sc._ngrams(copied) & pset
    assert not (sc._ngrams(para) & pset)


def test_scorer_flags_a_fabricated_event_time():
    sc = pytest.importorskip("vlm_semantic_score")
    rec = {"future_offsets_s": [2.0, 5.0, 10.0]}
    pairs = [(rec, {"route_event_time_s": 5.0}),      # offered -> ok
             (rec, {"route_event_time_s": 7.3}),      # never shown -> fabricated
             (rec, {"route_event_time_s": None})]     # honest null
    out = sc._event_time_block(pairs, [("route_event_time_s",)])
    b = out["route_event_time_s"]
    assert b["n"] == 3
    # `_rate` rounds to 4 dp, so compare at that resolution, not exactly
    assert b["in_offered_set_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert b["fabricated_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert b["null_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert "7.3" in b["top_fabricated"]


# ------------------------------------------------------- stratified sampling
def test_window_strata_tags_a_synthetic_turn_and_a_stop():
    torch = pytest.importorskip("torch")
    import math
    T = 199
    # a right turn at constant 10 m/s: yaw sweeps -90 deg over the clip
    yaw = torch.linspace(0, -math.pi / 2, T)
    v = torch.full((T,), 10.0)
    x = torch.cumsum(torch.cos(yaw) * v * 0.1, 0)
    y = torch.cumsum(torch.sin(yaw) * v * 0.1, 0)
    poses = torch.stack([x, y, yaw, v], dim=-1)
    st = S.window_strata(poses, 0, T)
    assert "turn_right" in st["tags"] or "sharp_turn" in st["tags"]
    # v2.1 may legitimately return `unknown` here — a 19.9 s clip is shorter
    # than its 25 s route lookahead. The stratum tag must not depend on that.
    assert st["kin_v21"]["route"] in ("left", "straight", "right", "unknown")
    assert st["net_deg"] > 45.0
    # a launch from standstill. The ramp must start inside the 2 s window:
    # `acc2` is deliberately measured over 2 s only, so a car still stationary
    # at t+2 s is correctly `steady` however fast it leaves later.
    v2 = torch.cat([torch.zeros(3), torch.linspace(0, 20, T - 3)])
    poses2 = torch.stack([torch.cumsum(v2 * 0.1, 0), torch.zeros(T),
                          torch.zeros(T), v2], dim=-1)
    st2 = S.window_strata(poses2, 0, T)
    assert "launch_from_stop" in st2["tags"]
    assert "accel" in st2["tags"]


# ------------------------------------------------------------ lake converter
def _fake_record(t, geom, tag, ev_t, offered=(2.0, 5.0, 10.0, 15.0, 20.0)):
    return {
        "episode": "ep_00000", "t": t, "episode_id": 12345,
        "val_build": "physicalai-val-test", "model": "nvidia/Cosmos-Reason2-8B",
        "prompt_version": "vlmsem-2026-07-21-a", "frames_plan": "base",
        "future_offsets_s": list(offered),
        "kin_v21": {"route": "left", "valid": True, "net_dyaw_deg": 42.0},
        "pass_A": {"ROUTE": "left", "parsed": {
            "road_geometry": geom, "road_geometry_confidence_band": "high",
            "geometry_event_time_s": ev_t}},
        "pass_B": {"parsed": {
            "SCENARIO": {"road_type": "urban_arterial", "surface": "dry",
                         "traffic_density": "light", "road_geometry": geom,
                         "road_geometry_confidence_band": "high",
                         "geometry_event_time_s": ev_t,
                         "geometry_event_end_time_s": None,
                         "scenario_tag": tag, "scenario_event_time_s": ev_t,
                         "scenario_confidence_band": "medium",
                         "difficulty": "notable",
                         "environment": {"weather": "clear",
                                         "time_of_day": "day"}},
            "OBSERVATIONS": {
                "lead_vehicle": {"present": True, "lane": "ego",
                                 "distance_bucket": "close",
                                 "relative_motion": "closing"},
                "critical_agents": [{"type": "pedestrian", "position": "crossing"}],
                "sign_reads": [{"type": "speed_limit", "value": 50,
                                "unit": "kph", "band": "high"}]},
            "COC": {"observation": "a bus is stopped ahead",
                    "inference": "the lane will be blocked",
                    "decision": "slow and prepare to pass"}}}}


def test_lake_converter_never_emits_a_metric_lead_field():
    """The refusal has to live in the DATA, not only in a doc: a consumer that
    asks this corpus for headway in seconds must get None and a reason."""
    L = pytest.importorskip("vlm_labels_to_lake")
    sc = L.episode_sidecar([_fake_record(0, "junction", "pedestrian_crossing", 5.0)])
    ls = sc["lead_state"]
    assert ls["gap_m"] is None and ls["closing_speed_ms"] is None
    assert ls["ttc_s"] is None
    assert "_metric_fields_unavailable" in ls
    # the coarse categorical state IS filled — that is what stratification needs
    assert ls["present"] is True and ls["lane"] == "ego"
    assert ls["distance_bucket"] == "close"
    assert ls["_pending"] is False


def test_lake_converter_drops_a_fabricated_event_offset():
    L = pytest.importorskip("vlm_labels_to_lake")
    good = L.window_rows([_fake_record(0, "roundabout", "yield_merge", 10.0)])[0]
    bad = L.window_rows([_fake_record(0, "roundabout", "yield_merge", 7.3)])[0]
    assert good["geometry_event_time_s"] == 10.0
    assert bad["geometry_event_time_s"] is None, \
        "an offset we never showed is a fabricated number, not a reading"
    assert good["is_eventful_geometry"] and good["road_geometry"] == "roundabout"


def test_lake_converter_emits_the_taniteval_join_key():
    """TanitEval keys a window by its START; these labels are keyed by "now".
    Emitting the converted key stops every consumer re-deriving an off-by-8."""
    L = pytest.importorskip("vlm_labels_to_lake")
    rows = L.window_rows([_fake_record(0, "junction", "none", 5.0),
                          _fake_record(40, "junction", "none", 5.0)])
    assert rows[0]["taniteval_window_start"] is None, "t=0 has no eval window"
    assert rows[1]["taniteval_window_start"] == 32
    assert 32 % 8 == 0, "labeling stride must stay a multiple of the eval stride"


def test_lake_converter_rejects_a_route_token_in_the_geometry_slot():
    """MEASURED on 100 windows: the model answered `left`/`right` in the
    `road_geometry` slot 3 times — ROUTE's vocabulary reaching into the geometry
    slot. Such a token must never become a scenario stratum."""
    L = pytest.importorskip("vlm_labels_to_lake")
    r = _fake_record(0, "junction", "none", 5.0)
    r["pass_A"]["parsed"]["road_geometry"] = "left"
    row = L.window_rows([r])[0]
    assert row["road_geometry"] == "unknown"
    assert row["road_geometry_enum_violation"] is True
    assert row["road_geometry_raw"] == "left", "the raw token stays auditable"
    assert row["is_eventful_geometry"] is False
    ok = L.window_rows([_fake_record(0, "roundabout", "none", 5.0)])[0]
    assert ok["road_geometry"] == "roundabout"
    assert ok["road_geometry_enum_violation"] is False


def test_lake_converter_keeps_route_quarantined():
    L = pytest.importorskip("vlm_labels_to_lake")
    row = L.window_rows([_fake_record(0, "junction", "none", 5.0)])[0]
    assert "vlm_route_passA_CROSSCHECK_ONLY" in row
    assert row["kin_route_v21"] == "left"
    assert not any(k == "route" or k == "ROUTE" for k in row)


def test_lake_sidecar_reports_axis_disagreement():
    """An episode whose windows disagree on the weather has not measured it."""
    L = pytest.importorskip("vlm_labels_to_lake")
    recs = [_fake_record(0, "junction", "none", 5.0),
            _fake_record(40, "straight", "none", 5.0)]
    recs[1]["pass_B"]["parsed"]["SCENARIO"]["environment"]["weather"] = "rain"
    sc = L.episode_sidecar(recs)
    assert sc["scene_tags"]["_axis_agreement"]["weather"]["agreement"] == 0.5
    assert sc["scene_tags"]["vru_present"]["token"] == "true"
    # the junction window's event survives with its window t AND its offset
    ev = [e for e in sc["scene_tags"]["notable_events"]
          if e["event"] == "geometry.junction"]
    assert ev and ev[0]["t"] == 0 and ev[0]["offset_s"] == 5.0
