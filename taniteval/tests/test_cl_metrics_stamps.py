"""Tests for the STAMP CONTRACT in `stack/experiments/alpasim-gsplat/cl_metrics.py`.

⛔ THE DEFECT UNDER TEST IS NOT A CRASH. It is a metric family that reports itself
ABSENT while the data is sitting in the dump. `cl_metrics` read the strategic route head
as ``ex.get("s_route_logits")``; flagship-v1 emits that name but refc-base emits
``route_logits`` — on 450/450 steps of every condition. Every published closed-loop panel
therefore carried, for REF-C:

    "route_head_eq_logged": {"n": 0,
        "reason": "this arm exposes no strategic route logits at the deploy path"}

which was false. An entire metric family of the four the programme is bound to report was
deleted by a key-name mismatch, and nothing failed.

So these tests are written so that **reproducing the silent absence FAILS**. They pin:
  1. both spellings resolve (the bug itself);
  2. a head under an UNKNOWN spelling RAISES rather than becoming None — this is the
     recurrence guard, because the producer names keys after the model's own output dict
     and the next arm will bring a third spelling;
  3. presence is decided over ALL windows, never `rows[0]` and never `any()`;
  4. a missing stamp is never IMPUTED as a class;
  5. the CIRCULAR_NAV_ECHO guard fires on a nav bijection, and reports UNIDENTIFIABLE —
     not a clearance — when nav does not vary;
  6. a labeller CRASH is never reported as "this scene has no junction".

The synthetic fixtures pin the logic; the banked rollouts pin the contract against the
real bytes that produced the published numbers.
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "stack" / "experiments" / "alpasim-gsplat"
_ROLL = _EXP / "results"
if str(_EXP) not in sys.path:                       # cl_metrics lives outside the packages
    sys.path.insert(0, str(_EXP))

CM = pytest.importorskip("cl_metrics", reason="alpasim-gsplat experiment not checked out")

_FALSE_ABSENT = "exposes no strategic route logits"


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _row(k=0, *, route_head=1, man_head=2, nav=0, route_valid=True, route_gt=1,
         net_dyaw=0.0, route_reason="ok", route_key="route_logits",
         man_key="maneuver_logits"):
    """One window shaped exactly like `per_step_metrics` emits."""
    return {
        "k": k, "i_gt": k, "trunc": False,
        "de": [0.1, 0.2, 0.3, 0.4], "ade": 0.25, "de2s": 0.4,
        "lon": [0.1, 0.1, 0.1, 0.1], "lat": [0.05, 0.05, 0.05, 0.05],
        "lon_ade": 0.1, "lat_ade": 0.05,
        "cross_track": 0.3, "dist_to_gt": 0.3,
        "speed_err": 0.2, "speed_track_err": 0.1,
        "v_gt": 8.0, "v_ego": 8.1, "v_target": 8.2,
        "heading_err": 0.01, "curv_err": 0.001, "curv_plan": 0.0, "curv_gt": 0.0,
        "yawrate_err": 0.02, "nav": nav,
        "man_plan": 0, "man_gt": 0, "man_exec": 0, "man_head": man_head,
        "route_gt": route_gt, "route_valid": route_valid, "route_reason": route_reason,
        "route_net_dyaw": net_dyaw, "route_head": route_head,
        "man_head_key": man_key, "route_head_key": route_key,
        "corridor_departure": 0,
        "headway": None, "time_gap": None, "ttc": None, "v_lead": None,
        "lead_idx": -1, "actors": [],
    }


def _rows(n=20, **kw):
    return [_row(k, **kw) for k in range(n)]


def _eids(n=20, n_ep=4):
    return np.array([i % n_ep for i in range(n)])


def _summ(n_ep=4):
    return [{"start": e, "n": 5, "driven_m": 40.0, "gt_dist_m": 42.0,
             "progress_rel": 0.95, "max_cross_track": 0.4,
             "corridor_departure_rate": 0.0} for e in range(n_ep)]


# --------------------------------------------------------------------------- #
# 1. the bug itself — both spellings must resolve                              #
# --------------------------------------------------------------------------- #
class TestBothSpellingsResolve:
    @pytest.mark.parametrize("key", ["s_route_logits", "route_logits"])
    def test_route_head_resolves_under_either_arms_spelling(self, key):
        """flagship-v1 emits `s_route_logits`, refc-base emits `route_logits`."""
        val, resolved = CM.resolve_stamp({key: [0.1, 0.7, 0.2]}, "route")
        assert val == [0.1, 0.7, 0.2]
        assert resolved == key

    def test_maneuver_head_resolves(self):
        val, key = CM.resolve_stamp({"maneuver_logits": [0.0] * 5}, "maneuver")
        assert val is not None and key == "maneuver_logits"

    def test_refc_style_extra_yields_a_scored_route_family_not_an_absence(self):
        """THE REGRESSION TEST FOR TASK #54.

        Rows stamped the REF-C way must produce a real strategic number, and the
        published false sentence must not appear anywhere in the family.
        """
        rows = _rows(20, route_key="route_logits")
        fam = CM.families(rows, _eids(20), _summ())
        rhe = fam["STRATEGIC"]["route_head_eq_logged"]
        assert "mean" in rhe, "REF-C's route head was scored away again"
        assert rhe.get("n_used") == 20
        assert _FALSE_ABSENT not in json.dumps(fam["STRATEGIC"])


# --------------------------------------------------------------------------- #
# 2. the recurrence guard — an unknown spelling must RAISE                      #
# --------------------------------------------------------------------------- #
class TestUnknownSpellingRaises:
    def test_unrecognised_width3_vector_raises_rather_than_reporting_absent(self):
        """This is how the bug came back: a new arm, a new name, a silent None."""
        with pytest.raises(CM.MissingStampError) as ei:
            CM.resolve_stamp({"maneuver_logits": [0.0] * 5,
                              "nav_route_head": [0.3, 0.3, 0.4]}, "route")
        msg = str(ei.value)
        assert "nav_route_head" in msg, "the offending key must be named"
        assert "STAMP_SPEC" in msg, "the message must say how to declare it"

    def test_unrecognised_width5_vector_raises_for_the_maneuver_stamp(self):
        with pytest.raises(CM.MissingStampError):
            CM.resolve_stamp({"tactical_head": [0.0] * 5}, "maneuver")

    def test_a_genuinely_headless_arm_still_reports_absent_without_raising(self):
        """The escape hatch must stay open or every headless arm becomes a crash."""
        val, key = CM.resolve_stamp({"waypoints_flat": [0.0] * 8}, "route")
        assert (val, key) == (None, None)

    def test_declaring_a_non_head_in_IGNORED_STAMPS_silences_the_raise(self):
        CM.IGNORED_STAMPS.add("anchor_scores")
        try:
            val, key = CM.resolve_stamp({"anchor_scores": [0.1, 0.2, 0.3]}, "route")
            assert (val, key) == (None, None)
        finally:
            CM.IGNORED_STAMPS.discard("anchor_scores")

    def test_wrong_width_under_a_known_name_raises(self):
        """Geometry is asserted before scoring, not after."""
        with pytest.raises(CM.StampError):
            CM.resolve_stamp({"route_logits": [0.5, 0.5]}, "route")

    def test_absent_family_carries_its_evidence_and_the_aliases_it_tried(self):
        rows = _rows(12, route_head=None, route_key=None)
        fam = CM.families(rows, _eids(12), _summ())
        rhe = fam["STRATEGIC"]["route_head_eq_logged"]
        assert rhe["n"] == 0
        assert rhe["n_windows_checked"] == 12
        assert "route_logits" in rhe["checked_aliases"]
        assert "s_route_logits" in rhe["checked_aliases"]


# --------------------------------------------------------------------------- #
# 3. availability is a property of the ARM, resolved over ALL windows           #
# --------------------------------------------------------------------------- #
class TestAvailabilityIsResolvedOverAllWindows:
    def test_present_on_all(self):
        assert CM.stamp_availability(_rows(10), "route_head", "route") == "present"

    def test_absent_on_all(self):
        assert CM.stamp_availability(_rows(10, route_head=None),
                                     "route_head", "route") == "absent"

    def test_partial_coverage_raises(self):
        rows = _rows(10)
        rows[3]["route_head"] = None
        with pytest.raises(CM.PartialStampError) as ei:
            CM.stamp_availability(rows, "route_head", "route")
        assert "9/10" in str(ei.value)

    def test_blanking_row_zero_no_longer_deletes_the_tactical_head_family(self):
        """MEASURED before the fix: blanking 1 stamp of 450 made the whole family
        report "this arm exposes no maneuver_head logits" while 449 windows had it."""
        rows = _rows(20)
        rows[0]["man_head"] = None
        with pytest.raises(CM.PartialStampError):
            CM.families(rows, _eids(20), _summ())

    def test_row_zero_alone_no_longer_decides_the_family_is_present(self):
        """The mirror defect: 1 stamp kept of 450 scored 449 absences as WRONG."""
        rows = _rows(20, man_head=None)
        rows[0]["man_head"] = 2
        with pytest.raises(CM.PartialStampError):
            CM.families(rows, _eids(20), _summ())

    def test_tactical_head_absent_reports_evidence_not_a_guess(self):
        fam = CM.families(_rows(12, man_head=None), _eids(12), _summ())
        mh = fam["TACTICAL"]["maneuver_head"]
        assert mh["n"] == 0 and mh["n_windows_checked"] == 12
        assert "maneuver_logits" in mh["checked_aliases"]


# --------------------------------------------------------------------------- #
# 4. a missing stamp is never imputed as a class                                #
# --------------------------------------------------------------------------- #
class TestNoImputation:
    def test_graded_proxy_does_not_score_a_missing_head_as_straight(self):
        """Before the fix `{0:1,1:0,2:2}.get(head, 0)` turned None into `straight`
        and scored it. Partial coverage now raises instead of inventing data."""
        rows = _rows(20)
        for r in rows[1:]:
            r["route_head"] = None
        with pytest.raises(CM.PartialStampError):
            CM.families(rows, _eids(20), _summ())

    def test_per_class_pr_excludes_unpaired_entries_and_says_how_many(self):
        out = CM.per_class_pr([0, 1, 2, 0], [0, None, 2, 0], ("a", "b", "c"))
        assert out["_n"] == 3
        assert out["_n_input"] == 4
        assert out["_n_dropped_unpaired"] == 1
        assert "_dropped_note" in out

    def test_per_class_pr_reports_precision_next_to_recall(self):
        """⛔ binding: a rate is never published without the price it pays."""
        out = CM.per_class_pr([0, 0, 1], [1, 1, 1], ("a", "b"))
        assert out["b"]["recall"] == 1.0
        assert out["b"]["precision"] == pytest.approx(1 / 3, abs=1e-4)   # reported to 4 dp
        assert out["b"]["n_fires"] == 3 and out["b"]["support_n_true"] == 1
        assert out["_majority_class_baseline_acc"] is not None


# --------------------------------------------------------------------------- #
# 5. the CIRCULAR_NAV_ECHO guard — wired, and identifiable                      #
# --------------------------------------------------------------------------- #
class TestNavEchoGuard:
    def _fam(self, rows):
        return CM.families(rows, _eids(len(rows)), _summ())

    def test_bijection_of_nav_is_flagged_circular_and_marked_do_not_quote(self):
        """flagship-v1 on scene 7c72937c: nav=1 -> head=0, nav=0 -> head=1, exactly."""
        rows = []
        for k in range(20):
            nav = k % 2
            rows.append(_row(k, nav=nav, route_head=(0 if nav == 1 else 1),
                             route_gt=(0 if nav == 1 else 1)))
        fam = self._fam(rows)
        g = fam["STRATEGIC"]["route_head_nav_echo_check"]
        assert g["identifiable"] is True
        assert g["head_is_deterministic_function_of_nav"] is True
        assert "CIRCULAR" in g["verdict"]
        rhe = fam["STRATEGIC"]["route_head_eq_logged"]
        assert rhe["CIRCULAR_NAV_ECHO"] is True
        assert "do_not_quote" in rhe
        # the guard must also travel to the graded proxy, not just the discrete metric
        assert fam["STRATEGIC"]["route_head_side_eq_graded_proxy"]["CIRCULAR_NAV_ECHO"] is True

    def test_the_guard_reaches_EVERY_route_head_derived_metric(self):
        """⚠️ THE GUARD WAS HALF-WIRED. Its stamping loop ran before the graded proxy and
        the per-class PR blocks existed, so `.get(k)` was None and only the discrete
        metric was ever marked. MEASURED in the published scene2 panel: flagship-v1's
        `route_head_side_eq_graded_proxy` = 0.9311 shipped with NO circularity warning
        while `route_head_eq_logged` carried one. A number derived from an echoed head is
        just as circular as the head; if a new derived key is added below the stamping
        point in future, this test fails."""
        rows = []
        for k in range(20):
            nav = k % 2
            rows.append(_row(k, nav=nav, route_head=(0 if nav == 1 else 1),
                             route_gt=(0 if nav == 1 else 1)))
        s = self._fam(rows)["STRATEGIC"]
        derived = [k for k in s
                   if k.startswith("route_head") and isinstance(s[k], dict)
                   and k != "route_head_nav_echo_check"]
        assert derived, "no route-head-derived metrics found — the test is not looking"
        unstamped = [k for k in derived if not s[k].get("CIRCULAR_NAV_ECHO")]
        assert not unstamped, (
            f"these route-head metrics were derived from a head proven to be a "
            f"deterministic echo of nav, but carry no circularity warning: {unstamped}")

    def test_a_head_that_is_not_a_function_of_nav_is_not_flagged(self):
        rows = []
        for k in range(20):
            rows.append(_row(k, nav=k % 2, route_head=k % 3))
        fam = self._fam(rows)
        g = fam["STRATEGIC"]["route_head_nav_echo_check"]
        assert g["identifiable"] is True
        assert g["head_is_deterministic_function_of_nav"] is False
        assert "not an echo" in g["verdict"]
        assert "CIRCULAR_NAV_ECHO" not in fam["STRATEGIC"]["route_head_eq_logged"]

    def test_constant_nav_is_UNIDENTIFIABLE_not_circular(self):
        """⚠️ The guard's own defect. With one nav value the test is vacuously true for
        any constant head, so it published CIRCULAR on the cut-in scene (nav==0 on
        450/450). An echo cannot be separated from a constant with one input value."""
        rows = _rows(20, nav=0, route_head=1)
        fam = self._fam(rows)
        g = fam["STRATEGIC"]["route_head_nav_echo_check"]
        assert g["identifiable"] is False
        assert g["head_is_deterministic_function_of_nav"] is None
        assert "UNIDENTIFIABLE" in g["verdict"]
        assert g["n_distinct_nav"] == 1

    def test_unidentifiable_does_not_read_as_a_clearance(self):
        """An un-run guard must not look like a passed guard."""
        fam = self._fam(_rows(20, nav=0, route_head=1))
        rhe = fam["STRATEGIC"]["route_head_eq_logged"]
        assert rhe["NAV_ECHO_UNIDENTIFIABLE"] is True
        assert "NOT cleared" in rhe["nav_echo_note"]
        assert "CIRCULAR_NAV_ECHO" not in rhe

    def test_a_perfect_score_against_a_single_class_label_is_marked_degenerate(self):
        fam = self._fam(_rows(20, route_head=1, route_gt=1, nav=0))
        rhe = fam["STRATEGIC"]["route_head_eq_logged"]
        assert rhe["mean"] == 1.0
        assert rhe["degenerate"] is True
        assert "NOT SKILL" in rhe["degenerate_note"]


# --------------------------------------------------------------------------- #
# 6. a broken labeller is never reported as a property of the scene             #
# --------------------------------------------------------------------------- #
class TestLabellerFailureIsNotASceneFact:
    def test_all_windows_errored_is_flagged_as_an_instrument_failure(self):
        rows = _rows(20, route_valid=False, route_reason="error:RuntimeError")
        fam = CM.families(rows, _eids(20), _summ())
        s = fam["STRATEGIC"]
        assert s["route_label_error_rate"] == 1.0
        assert s["route_head_eq_logged"]["INSTRUMENT_FAILURE"] is True
        assert "RAISED" in s["route_head_eq_logged"]["reason"]
        assert "no junction-scale strategic decision" not in s["route_head_eq_logged"]["reason"]
        assert "route_label_INSTRUMENT_FAILURE" in s

    def test_genuine_road_following_still_reports_the_scene_explanation(self):
        rows = _rows(20, route_valid=False, route_reason="road_following")
        fam = CM.families(rows, _eids(20), _summ())
        rhe = fam["STRATEGIC"]["route_head_eq_logged"]
        assert rhe.get("INSTRUMENT_FAILURE") is None
        assert "ROAD FOLLOWING" in rhe["reason"]
        assert fam["STRATEGIC"]["route_label_error_rate"] == 0.0


# --------------------------------------------------------------------------- #
# 7. distance-keeping must not assert a scene fact it never measured            #
# --------------------------------------------------------------------------- #
class TestUnmeasuredIsNotZero:
    def test_no_tracks_loaded_reports_NOT_MEASURED(self):
        fam = CM.families(_rows(12), _eids(12), _summ())
        dk = fam["LONGITUDINAL"]["distance_keeping"]
        assert "NOT MEASURED" in dk["reason"]
        assert dk["n_actor_observations"] == 0

    def test_tracks_loaded_but_no_agent_in_lane_is_a_real_scene_fact(self):
        rows = _rows(12)
        for r in rows:                       # agents seen, but far off to the side
            r["actors"] = [{"id": 1, "idx": 0, "xy": [0.0, 0.0], "yaw": 0.0,
                            "rig": [30.0, 25.0], "v": None, "dist": 39.0}]
        fam = CM.families(rows, _eids(12), _summ())
        dk = fam["LONGITUDINAL"]["distance_keeping"]
        assert "MEASURED" in dk["reason"] and "NOT MEASURED" not in dk["reason"]
        assert dk["n_actor_observations"] == 12


# --------------------------------------------------------------------------- #
# 8. THE CONTRACT, against the real banked bytes                                #
# --------------------------------------------------------------------------- #
_BANKED = sorted(_ROLL.glob("*/rollouts/rollouts_*.json")) if _ROLL.exists() else []


@pytest.mark.skipif(not _BANKED, reason="banked rollouts not present")
class TestAgainstTheBankedRollouts:
    def test_every_banked_dump_stamps_a_route_head_under_a_declared_alias(self):
        """The empirical claim the fix rests on: the head was ALWAYS there."""
        missing = []
        for p in _BANKED:
            d = json.loads(p.read_text())
            for rec in d["rollouts"]:
                for st in rec["steps"]:
                    val, key = CM.resolve_stamp(st["extra"], "route")
                    if val is None:
                        missing.append(p.name)
                        break
                break
        assert not missing, f"route head absent in {missing}"

    def test_the_two_arms_really_do_use_different_spellings(self):
        """If this ever fails the bug class is gone — but so is the reason for the
        alias table, and someone should check why."""
        keys = {}
        for p in _BANKED:
            d = json.loads(p.read_text())
            st = d["rollouts"][0]["steps"][0]
            _, key = CM.resolve_stamp(st["extra"], "route")
            keys.setdefault(d["arm"], set()).add(key)
        assert keys["flagship-v1"] == {"s_route_logits"}
        assert keys["refc-base"] == {"route_logits"}

    @pytest.mark.parametrize("name", ["rollouts_refc-base_empty.json"])
    def test_refc_route_family_is_scored_on_the_real_dump(self, name):
        p = next(x for x in _BANKED if x.name == name)
        _, rows, eids, summ = CM.collect(str(p), None)
        fam = CM.families(rows, eids, summ)
        rhe = fam["STRATEGIC"]["route_head_eq_logged"]
        assert "mean" in rhe and rhe["n_used"] > 0
        assert _FALSE_ABSENT not in json.dumps(fam["STRATEGIC"])
        assert fam["STRATEGIC"]["route_head_stamp_key"] == ["route_logits"]

    def test_flagship_nav_bijection_is_caught_on_the_scene_where_nav_varies(self):
        """scene2 (7c72937c) is the identifiable scene: nav takes two values there."""
        p = _ROLL / "scene2-realclose" / "rollouts" / "rollouts_flagship-v1_empty.json"
        if not p.exists():
            pytest.skip("scene2 dump not present")
        _, rows, eids, summ = CM.collect(str(p), None)
        fam = CM.families(rows, eids, summ)
        g = fam["STRATEGIC"]["route_head_nav_echo_check"]
        assert g["identifiable"] is True
        assert g["head_is_deterministic_function_of_nav"] is True
        rhe = fam["STRATEGIC"]["route_head_eq_logged"]
        assert rhe["mean"] == 1.0, "the echo scores perfectly — that is the point"
        assert rhe["CIRCULAR_NAV_ECHO"] is True

    def test_refc_on_the_same_scene_is_not_an_echo(self):
        p = _ROLL / "scene2-realclose" / "rollouts" / "rollouts_refc-base_empty.json"
        if not p.exists():
            pytest.skip("scene2 dump not present")
        _, rows, eids, summ = CM.collect(str(p), None)
        fam = CM.families(rows, eids, summ)
        g = fam["STRATEGIC"]["route_head_nav_echo_check"]
        assert g["head_is_deterministic_function_of_nav"] is False
        assert fam["STRATEGIC"]["route_head_eq_logged"]["mean"] < 0.5
