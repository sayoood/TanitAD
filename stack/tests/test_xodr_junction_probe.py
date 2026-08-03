"""Guards for the OpenDRIVE reader and the junction probe (STREAM C).

Every guard here is driven with input designed to make the code FAIL, following
the pattern the programme has found actually works.  Three of them encode
mistakes that were MEASURED during the 2026-08-03 junction survey, not
hypotheticals:

* ``test_resolve_incoming_survives_a_sub_metre_connector`` — the first version
  of the probe found the incoming road by scanning backwards over snapped
  poses.  On scene ``00040136`` junction 239 is entered through connector road
  ``189`` which is **1.02 m long** and therefore never wins a nearest-lane
  snap, so the scan reported ``incomingRoad=20`` and the option count came out
  **0** instead of **1**.  A zero there reads as "no continuation exists",
  which is a materially different and false claim.
* ``test_signed_poly_dist_is_negative_inside`` — the survey's headline
  ("distance to the nearest junction") is meaningless without a sign
  convention that actually flips.
* ``test_dist_to_junctions_is_zero_on_a_junction_lane_centre`` — the C1
  positive control, in-process, so a refactor cannot silently break it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_EXP = Path(__file__).resolve().parents[1] / "experiments" / "nurec-gsplat"
sys.path.insert(0, str(_EXP))

pytest.importorskip("numpy")

from junction_probe import (_resolve_incoming, decision_options,  # noqa: E402
                            dist_to_junctions, fit_rigid_2d, junction_surface,
                            point_in_poly, polyline_dist, signed_poly_dist,
                            tm_forward, wrap_deg)
from xodr_map import Geometry, Junction, OpenDriveMap, load_xodr  # noqa: E402


# --------------------------------------------------------------------------
# geometry primitives
# --------------------------------------------------------------------------
def test_line_geometry_is_the_closed_form():
    g = Geometry(0.0, 1.0, 2.0, math.radians(30), 10.0, "line", {})
    x, y, h = g.eval(10.0)
    assert x == pytest.approx(1.0 + 10 * math.cos(math.radians(30)), abs=1e-9)
    assert y == pytest.approx(2.0 + 10 * math.sin(math.radians(30)), abs=1e-9)
    assert h == pytest.approx(math.radians(30))


def test_arc_quarter_circle_lands_where_geometry_says():
    r = 20.0
    g = Geometry(0.0, 0.0, 0.0, 0.0, 0.5 * math.pi * r, "arc", {"curvature": 1.0 / r})
    x, y, h = g.eval(g.length)
    # starting at the origin heading +x with the centre at (0, r): a quarter
    # turn left ends at (r, r) heading +y
    assert (x, y) == pytest.approx((r, r), abs=1e-6)
    assert h == pytest.approx(math.pi / 2, abs=1e-9)


def test_arc_with_negative_curvature_turns_the_other_way():
    r = 20.0
    g = Geometry(0.0, 0.0, 0.0, 0.0, 0.5 * math.pi * r, "arc", {"curvature": -1.0 / r})
    x, y, h = g.eval(g.length)
    assert (x, y) == pytest.approx((r, -r), abs=1e-6)
    assert h == pytest.approx(-math.pi / 2, abs=1e-9)


def test_spiral_with_constant_curvature_equals_the_arc():
    """A clothoid whose curvature does not change IS an arc.  If the numeric
    integrator disagrees with the closed form, every spiral in the survey is
    wrong and nothing says so."""
    r, L = 25.0, 30.0
    arc = Geometry(0.0, 3.0, -4.0, 0.7, L, "arc", {"curvature": 1.0 / r})
    spi = Geometry(0.0, 3.0, -4.0, 0.7, L, "spiral",
                   {"curvStart": 1.0 / r, "curvEnd": 1.0 / r})
    for ds in (0.0, 7.5, 15.0, L):
        assert spi.eval(ds) == pytest.approx(arc.eval(ds), abs=1e-9)


def test_spiral_with_zero_curvature_equals_the_line():
    lin = Geometry(0.0, 0.0, 0.0, 0.3, 12.0, "line", {})
    spi = Geometry(0.0, 0.0, 0.0, 0.3, 12.0, "spiral",
                   {"curvStart": 0.0, "curvEnd": 0.0})
    assert spi.eval(12.0) == pytest.approx(lin.eval(12.0), abs=1e-9)


def test_spiral_that_really_curves_bends_towards_the_end_curvature():
    """curvStart=0 -> curvEnd>0 must end up left of the straight continuation."""
    L = 40.0
    spi = Geometry(0.0, 0.0, 0.0, 0.0, L, "spiral", {"curvStart": 0.0, "curvEnd": 0.05})
    x, y, h = spi.eval(L)
    assert y > 0.5, "a left-bending clothoid must leave the x axis"
    assert 0.0 < h < 0.05 * L, "final heading must be between straight and full arc"


def test_geometry_eval_clamps_outside_its_own_length():
    g = Geometry(0.0, 0.0, 0.0, 0.0, 5.0, "line", {})
    assert g.eval(1e9)[0] == pytest.approx(5.0)
    assert g.eval(-1e9)[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# a synthetic OpenDRIVE, written to make the lane maths fail if it is wrong
# --------------------------------------------------------------------------
_SYNTH = """<?xml version="1.0"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="synth" vendor="tanitad-test">
    <geoReference><![CDATA[+proj=tmerc +lon_0=10 +lat_0=50 +=alt_0=0 +ellps=WGS84 +units=m]]></geoReference>
  </header>
  <road name="main" length="100.0" id="1" junction="-1">
    <link><successor elementType="junction" elementId="900"/></link>
    <type s="0" type="town"><speed max="50" unit="mph"/></type>
    <planView>
      <geometry s="0" x="0" y="0" hdg="0" length="100.0"><line/></geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          </lane>
          <lane id="-2" type="driving" level="false">
            <width sOffset="0" a="3.0" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
  <road name="thru" length="20.0" id="10" junction="900">
    <link><predecessor elementType="road" elementId="1" contactPoint="end"/></link>
    <planView>
      <geometry s="0" x="100" y="0" hdg="0" length="20.0"><line/></geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
  <road name="right" length="20.0" id="11" junction="900">
    <planView>
      <geometry s="0" x="100" y="0" hdg="-1.5707963" length="20.0"><line/></geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
  <junction id="900" name="T">
    <connection id="0" incomingRoad="1" connectingRoad="10" contactPoint="start">
      <laneLink from="-1" to="-1"/>
    </connection>
    <connection id="1" incomingRoad="1" connectingRoad="11" contactPoint="start">
      <laneLink from="-2" to="-1"/>
    </connection>
  </junction>
</OpenDRIVE>
"""


@pytest.fixture()
def synth(tmp_path):
    p = tmp_path / "map.xodr"
    p.write_text(_SYNTH)
    return load_xodr(p)


def test_parses_counts_and_the_malformed_geo_token(synth):
    from xodr_map import parse_geo_reference
    assert len(synth.roads) == 3
    assert len(synth.junctions) == 1
    assert synth.internal_road_ids() == {"10", "11"}
    assert synth.roads["1"].in_junction is False
    geo = parse_geo_reference(synth.header["geoReference"])
    assert geo["lat_0"] == 50.0 and geo["lon_0"] == 10.0
    # the DeepMap exporter emits a broken '+=alt_0=0'; it must be reported, not
    # crash and not silently become a key
    assert geo["_malformed"] == ["+=alt_0=0"]
    assert "=alt_0" not in geo


def test_lane_centre_offsets_are_inner_to_outer(synth):
    r = synth.roads["1"]
    # right side: lane -1 spans t in [0, -3.5] -> centre -1.75
    #             lane -2 spans t in [-3.5, -6.5] -> centre -5.0
    assert r.lane_t_bounds(50.0, -1) == pytest.approx((0.0, -3.5))
    assert r.lane_t_bounds(50.0, -2) == pytest.approx((-3.5, -6.5))
    c1 = r.lane_center_xy(50.0, -1)
    c2 = r.lane_center_xy(50.0, -2)
    assert c1[1] == pytest.approx(-1.75)
    assert c2[1] == pytest.approx(-5.0)
    assert c1[3] == pytest.approx(3.5) and c2[3] == pytest.approx(3.0)
    assert r.lane_t_bounds(50.0, 1) is None, "there is no left lane on this road"


def test_lane_centre_rotates_with_the_reference_heading():
    """Lateral offset must be applied perpendicular to hdg, not to +y."""
    xodr = _SYNTH.replace('hdg="0" length="100.0"', 'hdg="1.5707963" length="100.0"')
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".xodr", delete=False) as fh:
        fh.write(xodr)
        p = fh.name
    m = load_xodr(p)
    c = m.roads["1"].lane_center_xy(0.0, -1)
    # heading +y, right lane centre 1.75 m to the right -> +x
    assert c[0] == pytest.approx(1.75, abs=1e-6)
    assert c[1] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# polygon / distance primitives
# --------------------------------------------------------------------------
def test_signed_poly_dist_is_negative_inside():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    P = np.array([[5, 5], [1, 5], [-3, 5], [5, 13]], float)
    d = signed_poly_dist(P, sq)
    assert d[0] == pytest.approx(-5.0)      # centre, 5 m from every edge
    assert d[1] == pytest.approx(-1.0)      # 1 m inside the left edge
    assert d[2] == pytest.approx(3.0)       # 3 m outside
    assert d[3] == pytest.approx(3.0)
    assert point_in_poly(P, sq).tolist() == [True, True, False, False]


def test_point_in_poly_handles_a_concave_shape():
    """A convex-hull shortcut would call the notch 'inside'."""
    u = np.array([[0, 0], [10, 0], [10, 10], [7, 10], [7, 3],
                  [3, 3], [3, 10], [0, 10]], float)
    P = np.array([[5, 6], [1, 6], [5, 1]], float)   # notch, left arm, base
    assert point_in_poly(P, u).tolist() == [False, True, True]


def test_polyline_dist_uses_segments_not_just_vertices():
    line = np.array([[0, 0], [100, 0]], float)
    P = np.array([[50.0, 4.0]], float)
    assert polyline_dist(P, line)[0] == pytest.approx(4.0)
    # vertex-only distance would be sqrt(50^2+4^2) ~ 50.16
    assert polyline_dist(P, line)[0] < 5.0


def test_fit_rigid_2d_recovers_a_known_pose_and_refuses_to_mirror():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 2)) * 20
    th = math.radians(37.0)
    R0 = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t0 = np.array([12.0, -5.0])
    dst = src @ R0.T + t0
    R, t, res = fit_rigid_2d(src, dst)
    assert np.allclose(R, R0, atol=1e-9)
    assert np.allclose(t, t0, atol=1e-9)
    assert res.max() < 1e-9
    assert np.linalg.det(R) == pytest.approx(1.0), "must stay a rotation, not a reflection"


def test_tm_forward_puts_the_projection_origin_at_zero():
    E, N = tm_forward(np.array([50.0]), np.array([10.0]), 50.0, 10.0)
    assert abs(float(E[0])) < 1e-6 and abs(float(N[0])) < 1e-6


def test_wrap_deg_wraps_both_ways():
    assert wrap_deg(370.0) == pytest.approx(10.0)
    assert wrap_deg(-190.0) == pytest.approx(170.0)
    assert wrap_deg(180.0) == pytest.approx(-180.0)


# --------------------------------------------------------------------------
# junction semantics — the part that decides the survey verdict
# --------------------------------------------------------------------------
def test_dist_to_junctions_is_zero_on_a_junction_lane_centre(synth):
    """C1 positive control, in-process."""
    surf, road2j = junction_surface(synth, step=0.5)
    assert set(surf) == {"900"}
    assert road2j == {"10": "900", "11": "900"}
    c = synth.roads["10"].lane_center_xy(10.0, -1)
    d, which = dist_to_junctions(np.array([[c[0], c[1]]]), surf)
    assert d[0] == pytest.approx(0.0)
    assert which[0] == "900"


def test_dist_to_junctions_discriminates_a_far_point(synth):
    """C2/C3: a metric that cannot separate near from far certifies nothing."""
    surf, _ = junction_surface(synth, step=0.5)
    near, _ = dist_to_junctions(np.array([[105.0, -1.75]]), surf)
    far, _ = dist_to_junctions(np.array([[105.0 + 500.0, -1.75]]), surf)
    assert near[0] == pytest.approx(0.0)
    assert far[0] > 100.0
    assert far[0] - near[0] > 100.0


def test_decision_options_counts_per_lane_not_per_road(synth):
    """The whole survey verdict rests on this distinction: on 00040136 junction
    220 offers TWO connecting roads from road 65, but only ONE from the lane
    the ego is actually in."""
    assert decision_options(synth, "1", -1, "900") == ["10"]        # through only
    assert decision_options(synth, "1", -2, "900") == ["11"]        # right only
    assert decision_options(synth, "1", None, "900") == ["10", "11"]
    assert len(decision_options(synth, "1", None, "900")) == 2
    assert decision_options(synth, "999", None, "900") == []
    assert decision_options(synth, "1", -1, "no-such-junction") is None


def test_resolve_incoming_survives_a_sub_metre_connector(synth):
    """REGRESSION (measured 2026-08-03, scene 00040136 junction 239).

    The incoming road must be read out of the junction's own connection table.
    Resolving it by looking back at snapped poses fails whenever the connector
    is shorter than the pose spacing — road 189 is 1.02 m — and produced an
    option count of 0, i.e. 'no continuation exists', which is false.
    """
    inc_r, inc_l, how = _resolve_incoming(synth, "900", "10", -1)
    assert (inc_r, inc_l, how) == ("1", -1, "ok")
    inc_r, inc_l, how = _resolve_incoming(synth, "900", "11", -1)
    assert (inc_r, inc_l, how) == ("1", -2, "ok")
    # and it never invents an answer
    assert _resolve_incoming(synth, "900", "no-such-road", -1)[2] == \
        "connecting-road-not-in-table"
    assert _resolve_incoming(synth, "no-such-junction", "10", -1)[2] == \
        "junction-not-in-map"


def test_a_junction_with_two_options_is_flagged_and_one_is_not(synth):
    """The survey criterion itself: >=2 lane-level continuations is a DECISION."""
    lane_level = decision_options(synth, "1", -1, "900")
    road_level = decision_options(synth, "1", None, "900")
    assert len(lane_level) == 1 and len(road_level) == 2
    # exactly the 00040136 shape: road-level choice, no lane-level choice
    assert (len(lane_level) >= 2) is False
    assert (len(road_level) >= 2) is True


# --------------------------------------------------------------------------
# the strategic label — and the trap that made every left turn wrong
# --------------------------------------------------------------------------
# A road whose REFERENCE LINE is a 90 deg arc, but which carries a laneOffset
# that exactly cancels the curvature so the DRIVEN LANE runs dead straight.
# This is not a contrived shape: MEASURED on NuRec scene 7c72937c, road 35
# carries `laneOffset a=10.495` and its reference line sweeps 40.5 deg while
# the lane centreline is straight to within 0.5 deg.
_OFFSET_ROAD = """<?xml version="1.0"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="offset" vendor="tanitad-test"/>
  <road name="curvedref" length="31.41592653589793" id="1" junction="-1">
    <planView>
      <geometry s="0" x="0" y="0" hdg="0" length="31.41592653589793">
        <arc curvature="0.05"/>
      </geometry>
    </planView>
    <lanes>
      <laneOffset s="0" a="0" b="0" c="0" d="0"/>
      <laneSection s="0">
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""


def test_travel_heading_reads_the_driven_lane_not_the_reference_line(tmp_path):
    """REGRESSION (measured 2026-08-03, scene 7c72937c road 35 / junction 149).

    ``travel_heading`` used to return ``ref_pose(s)[2]``.  On a road whose
    reference line is offset from the carriageway that is simply the wrong
    curve: the branch angle came out **+51.49 deg** for a manoeuvre the ego
    drove at **+123.53 deg**, and the map said ``LEFT`` where the truth was a
    much sharper ``LEFT``.  The mandatory self-consistency control fired on
    every left turn in the shortlist until the heading was taken from the
    sampled lane centreline instead.

    This road has ZERO offset, so it is the control case: the lane reading and
    the reference line must agree.  They agree up to the estimator's KNOWN and
    intended bias — ``travel_heading`` takes a 6 m chord rather than a single
    0.5 m segment (a segment is noisy on tight arcs), and a chord of length
    ``c`` on curvature ``k`` reports the tangent at its midpoint, i.e. short of
    the endpoint tangent by ``c*k/2`` at each end.  That is specified here
    rather than hidden behind a loose tolerance.
    """
    from strategic_gt import travel_heading
    p = tmp_path / "m.xodr"
    p.write_text(_OFFSET_ROAD)
    m = load_xodr(p)
    r = m.roads["1"]
    h_in = travel_heading(r, -1, at="entry")
    h_out = travel_heading(r, -1, at="exit")
    turn = wrap_deg(h_out - h_in)
    # the reference line turns a full 90 deg
    assert math.degrees(r.ref_pose(r.length)[2]) == pytest.approx(90.0, abs=1e-6)
    # the 6 m chord loses c*k/2 at each end: 90 - 6*0.05 rad = 90 - 17.19
    expected = 90.0 - math.degrees(6.0 * 0.05)
    assert turn == pytest.approx(expected, abs=0.5)
    assert 60.0 < turn < 90.0, "still unambiguously a large left turn"


def test_travel_heading_diverges_from_the_reference_line_when_an_offset_exists(tmp_path):
    """The discriminating case: a laneOffset that bends the driven lane AWAY
    from the reference line.  If ``travel_heading`` still read ``planView``,
    this test would report the reference line's 0 deg and miss the turn."""
    from strategic_gt import travel_heading
    # straight reference line, but a laneOffset that swings 20 m sideways ->
    # the driven lane turns, the reference line does not.
    xodr = _OFFSET_ROAD.replace('<arc curvature="0.05"/>', "<line/>")
    xodr = xodr.replace('<laneOffset s="0" a="0" b="0" c="0" d="0"/>',
                        '<laneOffset s="0" a="0" b="0.5" c="0" d="0"/>')
    p = tmp_path / "m2.xodr"
    p.write_text(xodr)
    m = load_xodr(p)
    r = m.roads["1"]
    assert math.degrees(r.ref_pose(r.length)[2]) == pytest.approx(0.0, abs=1e-9), \
        "the reference line is straight by construction"
    turn = wrap_deg(travel_heading(r, -1, at="exit") - travel_heading(r, -1, at="entry"))
    assert abs(turn) < 2.0, "b=0.5 is a constant slope, so the lane is straight too"
    # now make the offset CURVE (quadratic): the lane must bend, the ref line must not
    xodr2 = xodr.replace('a="0" b="0.5" c="0" d="0"', 'a="0" b="0" c="0.02" d="0"')
    p2 = tmp_path / "m3.xodr"
    p2.write_text(xodr2)
    r2 = load_xodr(p2).roads["1"]
    assert math.degrees(r2.ref_pose(r2.length)[2]) == pytest.approx(0.0, abs=1e-9)
    turn2 = wrap_deg(travel_heading(r2, -1, at="exit") - travel_heading(r2, -1, at="entry"))
    assert abs(turn2) > 20.0, (
        "the driven lane bends by the laneOffset even though the reference line "
        "is dead straight -- reading planView headings would report 0 deg here")


def test_travel_heading_reverses_for_a_positive_lane_id(tmp_path):
    """Lanes with a positive id run AGAINST +s.  Getting this wrong labelled a
    -0.07 deg traversal 'UTURN (-176.82 deg)' on scene 7c72937c junction 154."""
    from strategic_gt import travel_heading
    p = tmp_path / "m.xodr"
    p.write_text(_OFFSET_ROAD.replace(
        "<right>\n          <lane id=\"-1\"",
        "<left>\n          <lane id=\"1\"").replace("</right>", "</left>"))
    m = load_xodr(p)
    r = m.roads["1"]
    fwd_entry = math.degrees(r.ref_pose(0.0)[2])
    back_exit = travel_heading(r, 1, at="exit")
    # Travelling against +s, the exit of the road is at s=0 and the heading is
    # the s=0 tangent flipped 180 deg -- offset only by the same half-chord
    # bias specified above (c*k/2 = 6*0.05/2 rad = 8.59 deg).
    resid = wrap_deg(back_exit - (fwd_entry + 180.0))
    assert abs(resid) == pytest.approx(math.degrees(6.0 * 0.05) / 2, abs=0.6)
    assert abs(resid) < 15.0, "a missed 180 deg flip would show up as ~180, not ~9"
    # the falsifier: the road's own reference tangent at s=0 must be ~180 deg
    # from the travel heading, because travel runs the other way down it
    assert abs(wrap_deg(back_exit - fwd_entry)) > 150.0
    # and this road genuinely has no forward lane, so asking for one is None
    assert travel_heading(r, -1, at="exit") is None


def _ev(eid, n_options, gt_road, gt_class, scoreable=True):
    return {"event_id": eid, "n_options": n_options, "connecting_road_taken": gt_road,
            "route_gt_class": gt_class, "SCOREABLE": scoreable}


def test_score_strategic_refuses_a_set_with_no_branch():
    """The exact night-clip situation: 4 junctions, every one single-option.
    The scorer must return NO accuracy at all rather than a free 1.0."""
    from strategic_gt import score_strategic
    evs = {"night": [_ev("night|J220|153", 1, "64", 1, scoreable=False),
                     _ev("night|J230|109", 1, "199", 1, scoreable=False)]}
    preds = {"night|J220|153": {"road": "64", "class": 1},
             "night|J230|109": {"road": "199", "class": 1}}
    out = score_strategic(evs, preds)
    assert out["n_events"] == 0
    assert "route_choice_accuracy" not in out, \
        "a perfect score against a one-option label is not skill and must not be emitted"
    assert "constant predictor" in out["reason"]


def test_score_strategic_scores_only_real_branches_and_reports_the_chance_floor():
    from strategic_gt import score_strategic
    evs = {"a": [_ev("a|J1|10", 4, "48", 0), _ev("a|J2|80", 2, "115", 1),
                 _ev("a|J3|99", 1, "126", 1, scoreable=False)],
           "b": [_ev("b|J9|30", 2, "7", 2)]}
    preds = {"a|J1|10": {"road": "48", "class": 0},      # correct, 4 options
             "a|J2|80": {"road": "999", "class": 0},     # wrong,  2 options
             "b|J9|30": {"road": "7", "class": 2}}       # correct, 2 options
    out = score_strategic(evs, preds, n_boot=200)
    assert out["n_events"] == 3, "the single-option junction must be excluded"
    assert out["n_scenes"] == 2
    assert out["route_choice_accuracy"]["mean"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["route_class_accuracy"]["mean"] == pytest.approx(2 / 3, abs=1e-4)
    # chance floor = mean(1/4, 1/2, 1/2)
    assert out["chance_floor"] == pytest.approx((0.25 + 0.5 + 0.5) / 3, abs=1e-4)
    assert out["accuracy_by_branching_factor"][4] == {"n": 1, "acc": 1.0}
    assert out["accuracy_by_branching_factor"][2] == {"n": 2, "acc": 0.5}
    assert out["CLUSTER"].startswith("scene")


def test_score_strategic_counts_a_missing_prediction_as_wrong_not_as_absent():
    """An arm with no strategic head must score 0, not be silently dropped —
    dropping it would let a model with no route output beat one that tries."""
    from strategic_gt import score_strategic
    evs = {"a": [_ev("a|J1|10", 3, "48", 0), _ev("a|J2|80", 2, "115", 1)]}
    out = score_strategic(evs, {"a|J1|10": {"road": "48", "class": 0}}, n_boot=200)
    assert out["n_events"] == 2
    assert out["n_events_without_a_prediction"] == 1
    assert out["route_choice_accuracy"]["mean"] == pytest.approx(0.5, abs=1e-4)


def test_score_strategic_uses_the_programme_estimator_when_available():
    """If taniteval is importable the CI must come from the episode-cluster
    bootstrap, never from overlapping_holdout_se and never from a bare mean."""
    from strategic_gt import _ci_module, score_strategic
    ci = _ci_module()
    if ci is None:
        pytest.skip("taniteval not importable from this checkout")
    evs = {f"s{i}": [_ev(f"s{i}|J|0", 2, "x", 0)] for i in range(8)}
    preds = {f"s{i}|J|0": {"road": ("x" if i % 2 else "y"), "class": 0} for i in range(8)}
    out = score_strategic(evs, preds, n_boot=300)
    acc = out["route_choice_accuracy"]
    assert acc["estimator"] != "overlapping_holdout_se"
    assert "lo" in acc and "hi" in acc and acc["lo"] <= acc["mean"] <= acc["hi"]
    assert acc["n_episodes"] == 8, "the cluster is the SCENE"


def test_classify_branch_thresholds_are_the_documented_ones():
    from strategic_gt import ROUTE, classify_branch
    assert ROUTE[classify_branch(0.0)] == "STRAIGHT"
    assert ROUTE[classify_branch(24.9)] == "STRAIGHT"
    assert ROUTE[classify_branch(25.1)] == "LEFT"
    assert ROUTE[classify_branch(-25.1)] == "RIGHT"
    assert ROUTE[classify_branch(149.9)] == "LEFT"
    assert ROUTE[classify_branch(150.1)] == "UTURN"
    assert ROUTE[classify_branch(-179.0)] == "UTURN"
    # a wrapped input must not change the answer
    assert classify_branch(370.0) == classify_branch(10.0)
