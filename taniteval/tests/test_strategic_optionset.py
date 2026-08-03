"""Tests for the option-set STRATEGIC family.

⛔ THE DEFECT UNDER TEST is not a crash. It is a **number that looks like a
result**: ``route_head_eq_logged = 1.0000`` on a clip where the map admitted one
continuation at every junction. Every test here is written so that reproducing
that degeneracy FAILS, rather than merely being commented against.

The suite is deliberately built around the two REAL banked label sets when they
are present on disk (``stack/experiments/nurec-gsplat/results/strategic_gt/``) —
the synthetic fixtures pin the logic, the real labels pin the contract.
"""
import json
from pathlib import Path

import pytest

from taniteval import strategic_optionset as SO

_REPO = Path(__file__).resolve().parents[2]
_LABELS = _REPO / "stack" / "experiments" / "nurec-gsplat" / "results" / "strategic_gt"


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _event(scene, jid, entry_pose, gt_class, options, taken_road=None):
    opts = [{"road": r, "lane": -1, "class": c, "class_name": SO.ROUTE_CLASSES[c],
             "branch_dyaw_deg": 0.0} for r, c in options]
    return {
        "event_id": f"{scene}|J{jid}|{entry_pose}",
        "junction_id": str(jid), "entry_pose": entry_pose, "exit_pose": entry_pose + 10,
        "entry_arc_m": float(entry_pose), "incoming_road": "1", "incoming_lane": -1,
        "connecting_road_taken": taken_road if taken_road is not None else options[0][0],
        "route_gt_class": gt_class, "route_gt_name": SO.ROUTE_CLASSES[gt_class],
        "n_options": len(opts), "options": opts,
        "SCOREABLE": bool(len(opts) >= 2),
    }


def _report(scene, events, n_poses=200, admissible=True):
    per_pose = []
    for i in range(n_poses):
        nxt = next((e for e in events if e["entry_pose"] >= i), None)
        if nxt is None:
            per_pose.append({"pose": i, "has_decision_ahead": False, "admissible": False})
            continue
        d = float(nxt["entry_arc_m"] - i)
        per_pose.append({
            "pose": i, "has_decision_ahead": True, "event_id": nxt["event_id"],
            "dist_to_decision_point_m": round(d, 2), "n_options": nxt["n_options"],
            "route_gt_class": nxt["route_gt_class"],
            "admissible": bool(nxt["SCOREABLE"] and 0.0 <= d <= 60.0),
        })
    return {"ADMISSIBLE": admissible, "n_poses": n_poses,
            "SELFCONSISTENCY_CONTROL": {"PASS": admissible,
                                        "worst_abs_err_deg_untruncated": 0.04,
                                        "tolerance_deg": 35.0},
            "events": events, "per_pose": per_pose}


@pytest.fixture
def branchy():
    """8 scenes x 2 branching junctions — enough clusters for a bootstrap."""
    reps = {}
    for s in range(8):
        sid = f"scene{s}"
        reps[sid] = _report(sid, [
            _event(sid, 10, 40, 0, [("A", 0), ("B", 1)]),           # LEFT vs STRAIGHT
            _event(sid, 11, 120, 2, [("C", 2), ("D", 1), ("E", 0)]),  # RIGHT of 3
        ])
    reps["_refused"] = {}
    return reps


@pytest.fixture
def nightclip_shaped():
    """The degenerate case: every junction offers exactly ONE continuation."""
    reps = {}
    for s in range(4):
        sid = f"night{s}"
        reps[sid] = _report(sid, [_event(sid, 220, 60, 1, [("Z", 1)])])
    reps["_refused"] = {}
    return reps


def _oracle(reports):
    ev, _ = SO.scoreable_events(reports)
    return {e["event_id"]: {"class": e["route_gt_class"],
                            "road": e["connecting_road_taken"]} for e in ev}


# --------------------------------------------------------------------------- #
# ⭐ THE NEGATIVE CONTROL — a constant predictor must NOT score well           #
# --------------------------------------------------------------------------- #
def test_negative_control_discriminates_oracle_from_every_constant(branchy):
    ctl = SO.discrimination_control(branchy, n_boot=400)
    assert ctl["DISCRIMINATES"] is True
    assert ctl["arms"]["ORACLE"]["mean"] == 1.0
    # every constant predictor must be strictly worse than the oracle
    for name, block in ctl["arms"].items():
        if name.startswith("CONSTANT_"):
            assert block["mean"] < 1.0, f"{name} tied the oracle — the metric is degenerate"
    # and the paired test must SEPARATE oracle from the strongest of them
    assert ctl["oracle_minus_best_constant"]["separated"] is True
    assert ctl["oracle_minus_best_constant"]["delta"] > 0
    assert ctl["oracle_minus_best_constant"]["estimator"].startswith("paired_")


def test_negative_control_no_head_scores_zero_and_is_not_dropped(branchy):
    ctl = SO.discrimination_control(branchy, n_boot=200)
    assert ctl["arms"]["NO_HEAD"]["mean"] == 0.0
    assert ctl["no_head_scores_zero"] is True
    # dropping it would let an arm with NO route head beat one that tries
    assert ctl["arms"]["NO_HEAD"]["n_windows"] == ctl["n_events"]


def test_negative_control_refuses_the_night_clip_shape(nightclip_shaped):
    """⛔ THE REGRESSION THAT MATTERS: a single-option set must yield NO score."""
    ctl = SO.discrimination_control(nightclip_shaped, n_boot=200)
    assert ctl["DISCRIMINATES"] is False
    assert ctl["n_events"] == 0
    assert "single-option" in ctl["reason"] or "SCOREABLE" in ctl["reason"]
    assert "arms" not in ctl          # no accuracy is emitted at all


def test_a_constant_predictor_does_not_beat_itself(branchy):
    """The best constant scored AS an arm must not be 'separated' from itself."""
    ev, _ = SO.scoreable_events(branchy)
    always_left = {e["event_id"]: {"class": 0, "road": None} for e in ev}
    fam = SO.strategic_family(branchy, always_left, arm="ALWAYS_LEFT", n_boot=400)
    assert fam["status"] == "OK"
    assert fam["beats_best_constant"] is False
    assert fam["route_class_accuracy"]["mean"] == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# the degeneracy that produced route_head_eq_logged = 1.0000                   #
# --------------------------------------------------------------------------- #
def test_single_option_set_returns_UNAVAILABLE_not_a_free_one(nightclip_shaped):
    fam = SO.strategic_family(nightclip_shaped, _oracle(nightclip_shaped), n_boot=200)
    assert fam["status"] == "UNAVAILABLE"
    assert fam["n"] == 0
    assert fam["n_events_single_option_excluded"] == 4
    assert "route_class_accuracy" not in fam       # ⛔ no number at all
    assert "1.0000" in fam["reason"]               # names the defect it prevents


def test_single_option_events_are_excluded_from_a_mixed_set():
    sid = "mixed"
    reps = {sid: _report(sid, [_event(sid, 1, 30, 1, [("Z", 1)]),                # 1 option
                               _event(sid, 2, 100, 0, [("A", 0), ("B", 1)])]),   # 2 options
            "_refused": {}}
    ev, _ = SO.scoreable_events(reps)
    assert [e["junction_id"] for e in ev] == ["2"]
    fam = SO.strategic_family(reps, _oracle(reps), n_boot=200)
    assert fam["n_events_scoreable"] == 1
    assert fam["n_events_single_option_excluded"] == 1


def test_degenerate_label_is_flagged_even_when_accuracy_is_perfect():
    """All-LEFT labels: the oracle scores 1.0, and the block must say why that is not skill."""
    reps = {}
    for s in range(5):
        sid = f"s{s}"
        reps[sid] = _report(sid, [_event(sid, 1, 40, 0, [("A", 0), ("B", 1)])])
    reps["_refused"] = {}
    fam = SO.strategic_family(reps, _oracle(reps), n_boot=300)
    assert fam["route_class_accuracy"]["mean"] == 1.0
    assert fam["label_degenerate"] is True
    assert "constant predictor scores the same" in fam["⛔_degenerate_note"]
    # ...and the constant predictor also scores 1.0, so the arm does NOT beat it
    assert fam["floors"]["BEST_CONSTANT"]["accuracy"] == 1.0
    assert fam["beats_best_constant"] is False


# --------------------------------------------------------------------------- #
# free points, denominators, precision                                         #
# --------------------------------------------------------------------------- #
def test_missing_prediction_scores_zero_and_is_counted(branchy):
    preds = _oracle(branchy)
    dropped = sorted(preds)[:5]
    for k in dropped:
        preds.pop(k)
    fam = SO.strategic_family(branchy, preds, n_boot=200)
    assert fam["n_events_without_a_prediction"] == 5
    assert fam["route_class_accuracy"]["mean"] < 1.0    # not silently dropped


def test_none_vs_none_is_not_a_free_point():
    """⛔ A model that predicts nothing must not tie an unlabelled event."""
    sid = "u"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)])
    e["route_gt_class"] = None                 # unresolved map class
    reps = {sid: _report(sid, [e]), "_refused": {}}
    fam = SO.strategic_family(reps, {e["event_id"]: {"class": None, "road": "A"}},
                              n_boot=100)
    # road is present so the family is OK; the CLASS hit must be 0, not 1
    assert fam["route_class_accuracy"]["mean"] == 0.0


def test_per_class_reports_precision_beside_recall(branchy):
    ev, _ = SO.scoreable_events(branchy)
    always_left = {e["event_id"]: {"class": 0, "road": None} for e in ev}
    fam = SO.strategic_family(branchy, always_left, n_boot=200)
    per = fam["route_class_per_class"]
    assert per["LEFT"]["recall"] == 1.0
    assert per["LEFT"]["precision"] == pytest.approx(0.5, abs=1e-9)   # 8 of 16 fires right
    assert per["LEFT"]["n_pred"] == 16 and per["LEFT"]["n_true"] == 8
    assert per["RIGHT"]["recall"] == 0.0 and per["RIGHT"]["n_pred"] == 0
    for name, row in per.items():
        assert "precision" in row and "recall" in row, f"{name} has no precision"


def test_class_to_road_projection_states_its_denominator():
    """Three UTURN options: the class cannot name a road, so those events are EXCLUDED."""
    sid = "uturny"
    e = _event(sid, 149, 40, 0, [("48", 0), ("36", 3), ("46", 3), ("47", 3)],
               taken_road="48")
    reps = {sid: _report(sid, [e]), "_refused": {}}
    fam = SO.strategic_family(reps, {e["event_id"]: {"class": 3, "road": None}},
                              n_boot=100)
    assert fam["n_events_class_to_road_ambiguous"] == 1
    assert fam["route_choice_accuracy_road_level"]["n"] == 0
    assert "AMBIGUOUS" in fam["route_choice_note"]
    assert "denominator" in fam["route_choice_note"]


def test_uturn_gt_is_counted_as_outside_the_head_vocabulary():
    sid = "u2"
    e = _event(sid, 1, 40, 3, [("A", 3), ("B", 1)], taken_road="A")
    reps = {sid: _report(sid, [e]), "_refused": {}}
    fam = SO.strategic_family(reps, {e["event_id"]: {"class": 1, "road": None}}, n_boot=100)
    assert fam["n_events_gt_outside_head_vocabulary"] == 1


def test_scenes_failing_selfconsistency_are_refused_not_scored():
    good = _report("ok", [_event("ok", 1, 40, 0, [("A", 0), ("B", 1)])])
    bad = _report("bad", [_event("bad", 1, 40, 0, [("A", 0), ("B", 1)])], admissible=False)
    reps = SO.load_label_reports({"ok": good, "bad": bad})
    assert set(reps) == {"ok", "_refused"}
    assert "bad" in reps["_refused"]
    fam = SO.strategic_family(reps, _oracle(reps), n_boot=100)
    assert fam["n_scenes_refused_by_selfconsistency"] == 1
    assert fam["n_events_scoreable"] == 1


# --------------------------------------------------------------------------- #
# estimator + cluster                                                          #
# --------------------------------------------------------------------------- #
def test_cluster_is_the_scene_and_the_value_is_the_event(branchy):
    fam = SO.strategic_family(branchy, _oracle(branchy), n_boot=200)
    acc = fam["route_class_accuracy"]
    assert acc["estimator"] == "episode_cluster_bootstrap"
    assert acc["n_episodes"] == 8            # scenes, NOT the 1600 admissible poses
    assert acc["n_windows"] == 16            # decision events
    assert "never per pose" in fam["CLUSTER"]


def test_arm_with_no_route_decision_is_UNAVAILABLE_with_the_reason(branchy):
    ev, _ = SO.scoreable_events(branchy)
    fam = SO.strategic_family(branchy, {e["event_id"]: {"class": None, "road": None}
                                        for e in ev}, n_boot=100)
    assert fam["status"] == "UNAVAILABLE"
    assert fam["n"] == 16
    assert "ARM/JOIN gap, not a scene gap" in fam["reason"]


def test_vocabularies_are_pinned_against_a_silent_rename():
    v = SO.assert_vocabularies()
    assert v["map_classes"] == ["LEFT", "STRAIGHT", "RIGHT", "UTURN"]
    assert v["head_classes"] == ["route_left", "route_straight", "route_right"]
    assert v["classes_no_head_can_emit"] == ["UTURN"]
    # cl_metrics.py:39 uses index 3 for UNKNOWN — a different object, never mixed
    assert "UNKNOWN, NOT UTURN" in v["⛔_third_vocabulary"]


# --------------------------------------------------------------------------- #
# the closed-loop join                                                         #
# --------------------------------------------------------------------------- #
def test_ticks_join_scores_the_last_tick_before_entry():
    sid = "j"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)], taken_road="A")
    rep = _report(sid, [e])
    ticks = [{"i_gt": i, "route_head": (1 if i < 30 else 0)} for i in range(0, 40)]
    preds = SO.event_predictions_from_ticks(ticks, rep)
    assert preds[e["event_id"]]["class"] == 0
    assert preds[e["event_id"]]["at_pose"] == 39          # closest to the junction
    assert preds["_join"]["n_ticks_out_of_range"] == 0


def test_ticks_outside_the_label_pose_range_are_counted_not_wrapped():
    sid = "j2"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)])
    rep = _report(sid, [e], n_poses=50)
    ticks = [{"i_gt": i, "route_head": 0} for i in (10, 20, 999, -5)]
    preds = SO.event_predictions_from_ticks(ticks, rep)
    assert preds["_join"]["n_ticks_out_of_range"] == 2


def test_decision_lead_distance_rewards_committing_early():
    sid = "lead"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)], taken_road="A")
    rep = _report(sid, [e])
    early = [{"i_gt": i, "route_head": 0} for i in range(0, 40)]
    late = [{"i_gt": i, "route_head": (1 if i < 38 else 0)} for i in range(0, 40)]
    reps = {sid: rep, "_refused": {}}
    fe = SO.strategic_family(reps, SO.event_predictions_from_ticks(early, rep), n_boot=100)
    fl = SO.strategic_family(reps, SO.event_predictions_from_ticks(late, rep), n_boot=100)
    assert fe["decision_lead_distance_m"]["mean"] > fl["decision_lead_distance_m"]["mean"]
    assert fl["decision_lead_distance_m"]["mean"] == pytest.approx(2.0, abs=1e-6)


def test_decision_lead_is_zero_when_wrong_at_the_junction():
    sid = "lead0"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)], taken_road="A")
    rep = _report(sid, [e])
    ticks = [{"i_gt": i, "route_head": (0 if i < 35 else 1)} for i in range(0, 40)]
    reps = {sid: rep, "_refused": {}}
    fam = SO.strategic_family(reps, SO.event_predictions_from_ticks(ticks, rep), n_boot=100)
    assert fam["decision_lead_distance_m"]["mean"] == 0.0


# --------------------------------------------------------------------------- #
# wiring into the four-family block                                            #
# --------------------------------------------------------------------------- #
def test_four_families_strategic_reports_a_real_number_not_UNAVAILABLE(branchy):
    torch = pytest.importorskip("torch")
    from taniteval import four_families as FF
    win = {"pred": torch.zeros(3, 4, 2), "gt": torch.zeros(3, 4, 2), "wp_steps": [5, 10, 15, 20]}
    before = FF.all_families(win)
    assert before["strategic"]["status"] == "UNAVAILABLE"
    assert "option sets" in before["strategic"]["how_to_populate"]

    after = FF.all_families(win, optionset={"labels": branchy,
                                            "predictions": _oracle(branchy),
                                            "arm": "ORACLE", "n_boot": 200})
    s = after["strategic"]
    assert s["status"] == "OK"
    assert s["route_class_accuracy"]["mean"] == 1.0
    assert s["beats_best_constant"] is True
    assert "strategic" not in after["_families_unavailable"]


def test_optionset_can_ride_on_the_window_dict(branchy):
    torch = pytest.importorskip("torch")
    from taniteval import four_families as FF
    win = {"pred": torch.zeros(2, 4, 2), "gt": torch.zeros(2, 4, 2),
           "wp_steps": [5, 10, 15, 20],
           "optionset": {"labels": branchy, "predictions": _oracle(branchy), "n_boot": 100}}
    assert FF.all_families(win)["strategic"]["status"] == "OK"


def test_optionset_overrides_the_ego_yaw_hierarchy_path(branchy):
    """⛔ The ego-yaw label must never win over the map: it cannot see a branch."""
    from taniteval import four_families as FF
    hier = {"seam_nav_to_strategic": {"route_acc_follow": 1.0,
                                      "majority_straight_rate": 1.0}}
    s = FF.strategic({}, hier, {"labels": branchy, "predictions": _oracle(branchy),
                                "n_boot": 100})
    assert s["source"].startswith("strategic_optionset")
    assert "route_acc_follow" not in s
    assert "1.0000" in s["_supersedes"]


# --------------------------------------------------------------------------- #
# the REAL banked labels (skipped when the artifacts are not on this disk)     #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _LABELS.is_dir(), reason="banked strategic_gt labels not present")
def test_real_labels_load_and_the_night_clip_is_unscoreable():
    reps = SO.load_label_reports(_LABELS)
    night = "00040136-e651-4abd-991d-0655ccda9430"
    assert night in reps, "the night clip's label must be present"
    ev, _ = SO.scoreable_events({night: reps[night], "_refused": {}})
    assert ev == [], "the night clip must contribute NO scoreable decision event"


@pytest.mark.skipif(not _LABELS.is_dir(), reason="banked strategic_gt labels not present")
def test_real_labels_discriminate_and_a_constant_predictor_does_not_score_well():
    reps = SO.load_label_reports(_LABELS)
    ctl = SO.discrimination_control(reps, n_boot=1000)
    assert ctl["n_events"] >= 5
    assert ctl["DISCRIMINATES"] is True
    assert ctl["constant_predictor_does_not_score_well"] is True
    assert ctl["arms"][ctl["best_constant"]]["mean"] < ctl["arms"]["ORACLE"]["mean"]


@pytest.mark.skipif(not _LABELS.is_dir(), reason="banked strategic_gt labels not present")
def test_real_winner_scene_has_the_measured_option_set():
    """Pins the contract against the MEASURED artifact, not against prose."""
    p = _LABELS / "strategic_gt_7c72937c-c620-4776-9555-d57222c0081f.json"
    rep = json.loads(p.read_text())
    j149 = next(e for e in rep["events"] if e["junction_id"] == "149")
    assert j149["n_options"] == 4 and j149["SCOREABLE"] is True
    assert j149["route_gt_name"] == "LEFT" and j149["connecting_road_taken"] == "48"
    assert sorted(o["class_name"] for o in j149["options"]) == \
        ["LEFT", "UTURN", "UTURN", "UTURN"]


def test_decision_lead_is_flagged_as_right_censored_by_the_clip():
    """⚠️ A 'commits 60 m out' policy on a 40-pose approach scores ~40, not 60."""
    sid = "cens"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)], taken_road="A")
    rep = _report(sid, [e])
    ticks = [{"i_gt": i, "route_head": 0} for i in range(0, 40)]     # always correct
    reps = {sid: rep, "_refused": {}}
    fam = SO.strategic_family(reps, SO.event_predictions_from_ticks(ticks, rep), n_boot=100)
    dl = fam["decision_lead_distance_m"]
    assert dl["n_censored_by_clip"] == 1
    assert dl["mean"] == pytest.approx(dl["available_lead_m_max"], abs=1e-9)
    assert "LOWER BOUND" in dl["⚠️_censoring"]


def test_decision_lead_is_not_censored_when_the_arm_commits_late():
    sid = "cens2"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)], taken_road="A")
    rep = _report(sid, [e])
    ticks = [{"i_gt": i, "route_head": (1 if i < 30 else 0)} for i in range(0, 40)]
    reps = {sid: rep, "_refused": {}}
    fam = SO.strategic_family(reps, SO.event_predictions_from_ticks(ticks, rep), n_boot=100)
    dl = fam["decision_lead_distance_m"]
    assert dl["n_censored_by_clip"] == 0
    assert dl["mean"] < dl["available_lead_m_max"]


# --------------------------------------------------------------------------- #
# guards the REAL closed-loop data forced                                      #
# --------------------------------------------------------------------------- #
def test_a_single_scene_yields_NO_admissible_interval():
    """⛔ One cluster makes lo == hi == point. That is NO interval, not a precise one."""
    sid = "solo"
    reps = {sid: _report(sid, [_event(sid, 1, 40, 0, [("A", 0), ("B", 1)]),
                               _event(sid, 2, 120, 2, [("C", 2), ("D", 1)])]),
            "_refused": {}}
    fam = SO.strategic_family(reps, _oracle(reps), n_boot=200)
    acc = fam["route_class_accuracy"]
    assert acc["lo"] == acc["hi"] == acc["mean"]          # the degenerate shape
    assert acc["CI_NOT_ADMISSIBLE"] is True
    assert fam["beats_best_constant_ADMISSIBLE"] is False
    assert "⛔_SINGLE_SCENE" in fam
    # ...and a multi-scene block must NOT carry the flag
    multi = {f"s{i}": _report(f"s{i}", [_event(f"s{i}", 1, 40, i % 2, [("A", 0), ("B", 1)])])
             for i in range(6)}
    multi["_refused"] = {}
    fam2 = SO.strategic_family(multi, _oracle(multi), n_boot=300)
    assert "CI_NOT_ADMISSIBLE" not in fam2["route_class_accuracy"]
    assert "⛔_SINGLE_SCENE" not in fam2


def test_repeated_trials_over_one_event_are_kept_separate_not_averaged():
    """A closed-loop panel asks the SAME junction from 9 overlapping starts."""
    sid = "panel"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)], taken_road="A")
    reps = {sid: _report(sid, [e]), "_refused": {}}
    trials = [{"class": 0, "road": None}, {"class": 1, "road": None},
              {"class": 0, "road": None}, {"class": 1, "road": None}]
    fam = SO.strategic_family(reps, {e["event_id"]: trials}, n_boot=100)
    assert fam["n_events_scoreable"] == 1            # one decision in the world
    assert fam["n_decision_instances_scored"] == 4   # asked four times
    assert fam["route_class_accuracy"]["mean"] == pytest.approx(0.5, abs=1e-9)
    assert fam["route_class_accuracy"]["n_windows"] == 4


# --------------------------------------------------------------------------- #
# ⛔ the instrument defect the REAL data exposed: ONE logit key was not enough #
# --------------------------------------------------------------------------- #
def test_route_logits_are_found_under_either_arms_key():
    """MEASURED: flagship-v1 writes `s_route_logits`, refc-base writes `route_logits`.

    Reading one key made REF-C's route head invisible to the closed-loop
    STRATEGIC family for 450 ticks per run (cl_metrics.py:176).
    """
    sid = "keys"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)], taken_road="A")
    rep = _report(sid, [e])
    flagship = [{"i_gt": i, "s_route_logits": [9.0, 0.0, 0.0]} for i in range(30, 40)]
    refc = [{"i_gt": i, "route_logits": [9.0, 0.0, 0.0]} for i in range(30, 40)]
    for name, ticks, key in (("flagship", flagship, "s_route_logits"),
                             ("refc", refc, "route_logits")):
        p = SO.event_predictions_from_ticks(ticks, rep)
        assert p[e["event_id"]]["class"] == 0, f"{name}'s route head was not seen"
        assert p["_join"]["class_key_resolved"] == [key]
    assert "s_route_logits" in SO.ROUTE_LOGIT_KEYS
    assert "route_logits" in SO.ROUTE_LOGIT_KEYS


def test_a_logit_vector_is_argmaxed_and_a_decoded_int_is_taken_as_is():
    sid = "argmax"
    e = _event(sid, 1, 40, 2, [("A", 2), ("B", 1)], taken_road="A")
    rep = _report(sid, [e])
    by_logits = SO.event_predictions_from_ticks(
        [{"i_gt": 39, "route_logits": [0.1, 0.2, 5.0]}], rep)
    by_int = SO.event_predictions_from_ticks([{"i_gt": 39, "route_head": 2}], rep)
    assert by_logits[e["event_id"]]["class"] == 2
    assert by_int[e["event_id"]]["class"] == 2


def test_an_arm_whose_key_is_unknown_is_UNAVAILABLE_not_scored_as_wrong():
    """A key we have never seen must read as 'no decision', not as a wrong answer."""
    sid = "unknownkey"
    e = _event(sid, 1, 40, 0, [("A", 0), ("B", 1)])
    rep = _report(sid, [e])
    p = SO.event_predictions_from_ticks([{"i_gt": 39, "brand_new_route_key": [9, 0, 0]}], rep)
    assert p["_join"]["class_key_resolved"] is None
    fam = SO.strategic_family({sid: rep, "_refused": {}}, p, n_boot=100)
    assert fam["status"] == "UNAVAILABLE"
    assert "ARM/JOIN gap" in fam["reason"]


# --------------------------------------------------------------------------- #
# ⛔ THE SECOND DEGENERACY: an INPUT ECHO. Not constant, beats every constant,  #
# and still not a decision. MEASURED 2026-08-03: flagship-v1's route head moves #
# with `nav` at 100 % of 6 660 swept poses over 78 NuRec T1 scenes, while       #
# refc-base's moves at 0 %.                                                     #
# --------------------------------------------------------------------------- #
def test_echo_control_catches_a_head_that_relabels_its_own_input():
    sweeps = [{0: 1, 1: 0, 2: 2, 3: 1} for _ in range(20)]
    c = SO.conditioning_echo_control(sweeps)
    assert c["ECHO"] is True
    assert c["echo_rate"] == 1.0
    assert c["DETERMINISTIC_ECHO"] is True
    assert c["input_to_output_map_if_deterministic"] == {0: 1, 1: 0, 2: 2, 3: 1}


def test_echo_control_clears_a_head_that_ignores_its_conditioning():
    sweeps = [{0: c, 1: c, 2: c, 3: c} for c in (0, 1, 2, 0, 1)]
    c = SO.conditioning_echo_control(sweeps)
    assert c["ECHO"] is False
    assert c["echo_rate"] == 0.0


def test_echo_control_refuses_a_sweep_that_cannot_separate_anything():
    """A one-value 'sweep' measures nothing and must NOT report a pass."""
    c = SO.conditioning_echo_control([{1: 0} for _ in range(50)])
    assert c["ECHO"] is None
    assert c["n_usable"] == 0
    assert "MISSING CONTROL" in c["reason"]


def test_an_echo_arm_beats_every_constant_yet_is_INADMISSIBLE(branchy):
    """⭐ The regression that matters: BEST_CONSTANT cannot catch an echo.

    The echo arm is built by copying the GT class, so it scores 1.0 and clears the
    constant floor — exactly the shape the closed-loop panel reported for
    flagship-v1. The family must still refuse to call it strategic skill.
    """
    events, _ = SO.scoreable_events(branchy)
    oracle = {e["event_id"]: {"class": e["route_gt_class"], "road": None}
              for e in events}
    fam = SO.strategic_family(
        branchy, oracle, arm="ECHO", n_boot=200,
        conditioning_sweeps=[{0: 1, 1: 0, 2: 2, 3: 1} for _ in events])
    assert fam["route_class_accuracy"]["mean"] == 1.0
    assert fam["beats_best_constant"] is True          # the old guard is satisfied
    assert fam["conditioning_echo_control"]["ECHO"] is True
    assert fam["STRATEGIC_SKILL_ADMISSIBLE"] is False  # ...and the new one refuses
    assert "⛔_ECHO" in fam


def test_a_non_echo_arm_that_beats_the_constant_IS_admissible(branchy):
    events, _ = SO.scoreable_events(branchy)
    oracle = {e["event_id"]: {"class": e["route_gt_class"], "road": None}
              for e in events}
    fam = SO.strategic_family(
        branchy, oracle, arm="REAL", n_boot=200,
        conditioning_sweeps=[{0: 1, 1: 1, 2: 1, 3: 1} for _ in events])
    assert fam["conditioning_echo_control"]["ECHO"] is False
    assert fam["STRATEGIC_SKILL_ADMISSIBLE"] is fam["beats_best_constant"]


def test_missing_sweep_is_reported_as_UNTESTED_not_as_a_pass(branchy):
    events, _ = SO.scoreable_events(branchy)
    oracle = {e["event_id"]: {"class": e["route_gt_class"], "road": None}
              for e in events}
    fam = SO.strategic_family(branchy, oracle, arm="NO_SWEEP", n_boot=200)
    assert fam["STRATEGIC_SKILL_ADMISSIBLE"] is None
    assert "⚠️_ECHO_UNTESTED" in fam


def test_nav_to_route_matches_the_harness_vocabulary():
    """``closedloop_drive.NAV_NAMES`` order pinned against a silent renumbering."""
    assert SO.NAV_TO_ROUTE == {0: 1, 1: 0, 2: 2, 3: 1}
    assert SO.ROUTE_CLASSES[SO.NAV_TO_ROUTE[1]] == "LEFT"
    assert SO.ROUTE_CLASSES[SO.NAV_TO_ROUTE[2]] == "RIGHT"
    assert SO.ROUTE_CLASSES[SO.NAV_TO_ROUTE[3]] == "STRAIGHT"


def test_a_manoeuvre_the_map_does_not_admit_is_counted_separately():
    """⭐ 'wrong branch' and 'no such branch' are DIFFERENT strategic failures.

    A class-vs-class confusion scores them identically; only the option set can tell
    them apart, which is the whole reason this module exists.
    """
    ev = _event("s1", 1, 40, 0, [("r1", 0), ("r2", 1)])       # options are {LEFT, STRAIGHT}
    reports = {"s1": _report("s1", [ev]), "_refused": {}}
    fam = SO.strategic_family(
        reports, {ev["event_id"]: {"class": 2, "road": None}},   # predicts RIGHT
        arm="OFF_MAP", n_boot=100)
    assert fam["n_predictions_outside_the_option_set"] == 1
    assert fam["route_class_accuracy"]["mean"] == 0.0

    fam_ok = SO.strategic_family(
        reports, {ev["event_id"]: {"class": 1, "road": None}},   # wrong, but admitted
        arm="ON_MAP", n_boot=100)
    assert fam_ok["n_predictions_outside_the_option_set"] == 0
    assert fam_ok["route_class_accuracy"]["mean"] == 0.0
