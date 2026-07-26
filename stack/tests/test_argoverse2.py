"""Argoverse 2 adapter tests (2026-07-26).

TWO fixture families, and the split between them IS the point
-------------------------------------------------------------
AV2 ships two map flavours and they have **different schemas**:

  * ``sensor`` maps (the split that carries IMAGERY) have **NO ``centerline``
    field** — MEASURED 0 / 163 698 segments over all 1 000 sensor logs;
  * ``motion-forecasting`` maps **do** have it — MEASURED 4 201 / 4 201.

An adapter written against motion-forecasting alone passes every test it is
given and then fails on every log that has images. So the sensor fixture here is
built **without** ``centerline`` on purpose, and
``test_naive_centerline_access_raises_on_sensor_map`` asserts that the naive
access *does* blow up — i.e. the trap is real and the adapter's path is the only
safe one. Delete that test and the bug can walk back in silently.

The second sensor-only trap is encoded in the fixture too: left and right
boundaries have **different lengths** (MEASURED on 49.0 % of segments, n=19 713),
so a naive elementwise mean is wrong about half the time.

Real bytes: ``test_real_av2_*`` run against the pulled corpus when
``TANITAD_AV2_MAP_DIR`` points at it, and skip otherwise — so CI stays hermetic
while the strongest evidence is still one env var away.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from tanitad.data import argoverse2 as av2


# --------------------------------------------------------------------------- #
# Fixtures — schema-faithful to the real bytes                                 #
# --------------------------------------------------------------------------- #
def _poly(pts):
    """[(x,y,z), ...] -> AV2's on-disk [{"x":..,"y":..,"z":..}, ...]."""
    return [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in pts]


def _seg(i, left, right, successors=(), predecessors=(), *, is_intersection=False,
         left_neighbor_id=None, right_neighbor_id=None, lane_type="VEHICLE",
         centerline=None):
    d = {
        "id": i,
        "is_intersection": is_intersection,
        "lane_type": lane_type,
        "left_lane_boundary": _poly(left),
        "right_lane_boundary": _poly(right),
        "left_lane_mark_type": "SOLID_WHITE",
        "right_lane_mark_type": "NONE",
        "successors": list(successors),
        "predecessors": list(predecessors),
        "left_neighbor_id": left_neighbor_id,
        "right_neighbor_id": right_neighbor_id,
    }
    if centerline is not None:                 # ONLY motion-forecasting has this
        d["centerline"] = _poly(centerline)
    return d


def sensor_map_dict():
    """A SENSOR-flavour archive: no ``centerline``, mismatched boundary lengths.

    Topology (a T-junction with a dangling exit and a small loop):
        1 -> 2 (branch) -> {3, 4}
        3 -> 5 (intersection)
        5 -> 1                      <- cycle
        4 -> 99                     <- id 99 is NOT in this map (crop boundary)
    """
    segs = [
        # left has 2 pts, right has 3 -> the 49 % mismatch case
        _seg(1, [(0, 0, 0), (0, 10, 0)], [(4, 0, 0), (4, 5, 0), (4, 10, 0)],
             successors=[2], left_neighbor_id=6),
        _seg(2, [(0, 10, 0), (0, 20, 0)], [(4, 10, 0), (4, 20, 0)],
             successors=[3, 4], predecessors=[1]),
        _seg(3, [(0, 20, 0), (0, 30, 0)], [(4, 20, 0), (4, 30, 0)],
             successors=[5], predecessors=[2]),
        _seg(4, [(4, 20, 0), (14, 20, 0)], [(4, 24, 0), (14, 24, 0)],
             successors=[99], predecessors=[2], lane_type="BIKE"),
        _seg(5, [(0, 30, 0), (0, 0, 0)], [(4, 30, 0), (4, 0, 0)],
             successors=[1], predecessors=[3], is_intersection=True),
        _seg(6, [(-4, 0, 0), (-4, 10, 0)], [(0, 0, 0), (0, 10, 0)],
             right_neighbor_id=1),
    ]
    return {
        "drivable_areas": {"900": {"id": 900,
                                   "area_boundary": _poly([(0, 0, 0), (4, 0, 0)])}},
        "lane_segments": {str(s["id"]): s for s in segs},
        "pedestrian_crossings": {},
    }


def motion_forecasting_map_dict():
    """A MOTION-FORECASTING archive: ``centerline`` present, as in the real bytes.

    The explicit centerline is deliberately NOT the midpoint of the boundaries
    (x=1.0, not 2.0), so a test can tell whether the adapter used it or silently
    recomputed one.
    """
    segs = [_seg(1, [(0, 0, 0), (0, 10, 0)], [(4, 0, 0), (4, 10, 0)],
                 successors=[2], centerline=[(1, 0, 0), (1, 10, 0)]),
            _seg(2, [(0, 10, 0), (0, 20, 0)], [(4, 10, 0), (4, 20, 0)],
                 predecessors=[1], centerline=[(1, 10, 0), (1, 20, 0)])]
    return {"drivable_areas": {}, "pedestrian_crossings": {},
            "lane_segments": {str(s["id"]): s for s in segs}}


@pytest.fixture
def sensor_graph():
    return av2.LaneGraph.from_dict(sensor_map_dict(), log_id="LOGA", city="PIT")


@pytest.fixture
def mf_graph():
    return av2.LaneGraph.from_dict(motion_forecasting_map_dict(), log_id="SCEN1")


# --------------------------------------------------------------------------- #
# TRAP 1 — the missing `centerline` field. THE headline test.                  #
# --------------------------------------------------------------------------- #
def test_sensor_fixture_is_faithful_no_centerline_field():
    """Guard the fixture itself: if someone 'helpfully' adds a centerline to the
    sensor fixture, every trap test below silently stops testing anything."""
    raw = sensor_map_dict()["lane_segments"]
    assert len(raw) == 6
    assert all("centerline" not in s for s in raw.values()), \
        "sensor maps must have NO centerline field — that is the whole trap"


def test_naive_centerline_access_raises_on_sensor_map():
    """The bug this adapter exists to prevent, demonstrated.

    ``seg["centerline"]`` is what an adapter written against motion-forecasting
    does. On a sensor map — the split with the images — it is a KeyError.
    """
    raw = sensor_map_dict()["lane_segments"]["1"]
    with pytest.raises(KeyError):
        _ = raw["centerline"]


def test_centerline_is_derived_on_sensor_map(sensor_graph):
    """No explicit field -> the adapter DERIVES the midpoint, and says so."""
    assert sensor_graph.centerline_source(1) == "derived"
    cl = sensor_graph.centerline(1)
    assert cl.shape[1] == 3 and cl.shape[0] >= 2
    # lane 1: left x=0, right x=4  ->  centerline x=2 everywhere
    assert np.allclose(cl[:, 0], 2.0)
    assert cl[0, 1] == pytest.approx(0.0)
    assert cl[-1, 1] == pytest.approx(10.0)


def test_centerline_is_explicit_on_motion_forecasting_map(mf_graph):
    """Explicit field present -> used VERBATIM, not recomputed.

    The fixture's explicit centerline sits at x=1.0 while the boundary midpoint
    is x=2.0, so this fails loudly if the adapter ignores the shipped polyline.
    """
    assert mf_graph.centerline_source(1) == "explicit"
    cl = mf_graph.centerline(1)
    assert np.allclose(cl[:, 0], 1.0), "explicit centerline was silently recomputed"


def test_centerline_source_counts_split_by_flavour(sensor_graph, mf_graph):
    assert sensor_graph.centerline_source_counts() == {"explicit": 0, "derived": 6}
    assert mf_graph.centerline_source_counts() == {"explicit": 2, "derived": 0}


def test_every_sensor_lane_yields_a_usable_centerline(sensor_graph):
    """The failure mode is per-lane, so sweep all of them, not just lane 1."""
    for i in sensor_graph.ids():
        cl = sensor_graph.centerline(i)
        assert cl.shape[0] >= 2 and cl.shape[1] == 3
        assert np.isfinite(cl).all()


# --------------------------------------------------------------------------- #
# TRAP 2 — boundaries of different length (49.0 % of real segments)            #
# --------------------------------------------------------------------------- #
def test_midpoint_line_handles_mismatched_boundary_lengths():
    left = [(0, 0, 0), (0, 10, 0)]                       # 2 points
    right = [(4, 0, 0), (4, 5, 0), (4, 10, 0)]           # 3 points
    cl = av2.midpoint_line(left, right)
    assert cl.shape == (3, 3), "should resample onto the LONGER boundary"
    assert np.allclose(cl[:, 0], 2.0)
    assert np.allclose(cl[:, 1], [0.0, 5.0, 10.0])


def test_naive_elementwise_mean_would_have_failed():
    """Show the shortcut is genuinely broken, so the resampling is not cosmetic."""
    l = np.array([[0, 0, 0], [0, 10, 0]], float)
    r = np.array([[4, 0, 0], [4, 5, 0], [4, 10, 0]], float)
    with pytest.raises(ValueError):
        _ = 0.5 * (l + r)                                # shape mismatch


def test_interp_arc_is_uniform_in_arclength_not_index():
    """Index interpolation would give x=[0,1,10]; arc length gives [0,5,10]."""
    out = av2.interp_arc([(0, 0, 0), (1, 0, 0), (10, 0, 0)], 3)
    assert np.allclose(out[:, 0], [0.0, 5.0, 10.0])


def test_interp_arc_degenerate_inputs():
    assert av2.interp_arc([], 4).shape == (4, 3)
    assert np.allclose(av2.interp_arc([(3, 3, 3)], 3), 3.0)          # single point
    assert np.allclose(av2.interp_arc([(1, 1, 1), (1, 1, 1)], 2), 1.0)  # zero length


def test_midpoint_line_with_one_empty_boundary():
    cl = av2.midpoint_line([], [(0, 0, 0), (0, 4, 0)])
    assert cl.shape[0] == 2 and np.allclose(cl[:, 1], [0.0, 4.0])


# --------------------------------------------------------------------------- #
# TRAP 3 — dangling successors are the crop boundary, never an error           #
# --------------------------------------------------------------------------- #
def test_dangling_successor_is_terminal_not_an_error(sensor_graph):
    assert sensor_graph.get(4).successors == (99,)       # raw record keeps it
    assert sensor_graph.successors(4) == ()              # resolved view drops it
    assert sensor_graph.dangling_successors(4) == (99,)  # and stays inspectable
    routes = sensor_graph.routes_from(4, max_depth=5)
    assert routes == [(4,)], "a dangling exit must terminate the route"


def test_successor_edges_resolved_vs_raw(sensor_graph):
    raw = sum(len(s.successors) for s in sensor_graph)
    assert raw == 6                                      # 1,2x2,3,4,5
    assert len(sensor_graph.successor_edges(resolved_only=True)) == 5
    st = sensor_graph.stats()
    assert st["n_successor_refs_dangling"] == 1


def test_branch_points_resolved_only_is_the_default(sensor_graph):
    assert sensor_graph.branch_points() == [2]
    assert sensor_graph.branch_points(resolved_only=False) == [2]


def test_routes_from_terminates_on_a_cycle(sensor_graph):
    """36 % of real sensor logs contain a directed cycle; this must not hang."""
    assert sensor_graph.has_cycle() is True
    routes = sensor_graph.routes_from(1, max_depth=50)
    assert routes, "cycle must not swallow all routes"
    for r in routes:
        assert len(set(r)) == len(r), "a lane must not repeat within one route"


def test_routes_from_enumerates_the_branch(sensor_graph):
    routes = sensor_graph.routes_from(1, max_depth=3)
    assert {tuple(r) for r in routes} == {(1, 2, 3), (1, 2, 4)}


def test_routes_from_respects_max_routes(sensor_graph):
    assert len(sensor_graph.routes_from(1, max_depth=50, max_routes=1)) == 1


# --------------------------------------------------------------------------- #
# Lane ids are unique only WITHIN a local map                                  #
# --------------------------------------------------------------------------- #
def test_global_lane_key_namespaces_by_log():
    a = av2.global_lane_key("LOGA", 1)
    b = av2.global_lane_key("LOGB", 1)
    assert a != b and a == "LOGA:1"


def test_two_maps_reuse_the_same_lane_id(sensor_graph, mf_graph):
    """Both fixtures contain lane id 1 — that collision is REAL, hence the key."""
    assert 1 in sensor_graph.segments and 1 in mf_graph.segments
    assert sensor_graph.global_key(1) != mf_graph.global_key(1)


def test_unknown_lane_raises_an_actionable_error(sensor_graph):
    with pytest.raises(av2.Argoverse2MapError) as e:
        sensor_graph.get(4242)
    assert "unique only WITHIN" in str(e.value)


# --------------------------------------------------------------------------- #
# The strategic-brain signals (S1 / S2 / HP-4)                                 #
# --------------------------------------------------------------------------- #
def test_branch_options_carry_geometry_for_s1(sensor_graph):
    opts = sensor_graph.branch_options(2)
    assert {o["lane_id"] for o in opts} == {3, 4}
    for o in opts:
        assert o["entry_xyz"] and o["exit_xyz"]
        assert o["global_key"].startswith("LOGA:")
    straight = [o for o in opts if o["lane_id"] == 3][0]
    turning = [o for o in opts if o["lane_id"] == 4][0]
    # lane 3 continues +y (pi/2); lane 4 heads +x (0) -> a real, separable choice
    assert straight["heading_rad"] == pytest.approx(math.pi / 2, abs=1e-6)
    assert turning["heading_rad"] == pytest.approx(0.0, abs=1e-6)


def test_lateral_neighbours_are_the_s2_signal(sensor_graph):
    assert sensor_graph.neighbors(1) == {"left": 6, "right": None}
    assert sensor_graph.neighbors(6) == {"right": 1, "left": None}


def test_intersection_ids_are_the_hp4_signal(sensor_graph):
    assert sensor_graph.intersection_ids() == [5]


def test_merge_points(sensor_graph):
    assert sensor_graph.merge_points() == []            # no lane has in-degree 2


def test_stats_and_corpus_aggregate(sensor_graph, mf_graph):
    st = sensor_graph.stats()
    assert st["n_lane_segments"] == 6
    assert st["n_branch_points"] == 1
    assert st["n_is_intersection"] == 1
    assert st["lane_types"] == {"VEHICLE": 5, "BIKE": 1}
    assert st["has_cycle"] is True
    agg = av2.lane_graph_stats([sensor_graph, mf_graph])
    assert agg["n_maps"] == 2
    assert agg["n_lane_segments"] == 8
    assert agg["maps_with_a_branch"] == 1
    assert agg["centerline_source"] == {"derived": 6, "explicit": 2}


# --------------------------------------------------------------------------- #
# IO, errors, and the deliberate ABSENCE of a terms gate                       #
# --------------------------------------------------------------------------- #
def test_load_lane_graph_round_trip(tmp_path):
    p = tmp_path / "log_map_archive_abc123____PIT_city_71109.json"
    p.write_text(json.dumps(sensor_map_dict()), encoding="utf-8")
    g = av2.load_lane_graph(p)
    assert len(g) == 6 and g.city == "PIT" and g.log_id == "abc123"


def test_load_lane_graph_accepts_our_pullers_flat_naming(tmp_path):
    """Our puller stores archives as ``<log_id>.json``; both layouts must work."""
    p = tmp_path / "02678d04-cc9f-3148-9f95-1ba66347dff9.json"
    p.write_text(json.dumps(sensor_map_dict()), encoding="utf-8")
    g = av2.load_lane_graph(p)
    assert g.log_id == "02678d04-cc9f-3148-9f95-1ba66347dff9"


def test_missing_file_error_does_not_invent_a_terms_gate(tmp_path):
    """AV2 has NO access gate. Copying nuScenes' terms guard here would be a lie.

    The message must point at the anonymous fetch, not at an account.
    """
    with pytest.raises(av2.Argoverse2MapError) as e:
        av2.load_lane_graph(tmp_path / "nope.json")
    msg = str(e.value)
    assert "--no-sign-request" in msg
    assert "NO account" in msg and "NO Terms click" in msg
    assert "sign-up" not in msg.lower() and "register" not in msg.lower()


def test_adapter_exposes_no_terms_error_symbol():
    """A guard-by-absence: nothing named *Terms* may exist in this module."""
    assert not [n for n in dir(av2) if "terms" in n.lower()]


def test_truncated_file_error_names_the_byte_check(tmp_path):
    p = tmp_path / "log_map_archive_x.json"
    p.write_text('{"lane_segments": {"1": {"id": 1', encoding="utf-8")  # truncated
    with pytest.raises(av2.Argoverse2MapError) as e:
        av2.load_lane_graph(p)
    assert "byte count" in str(e.value)


def test_map_without_lane_segments_layer_raises():
    with pytest.raises(av2.Argoverse2MapError) as e:
        av2.LaneGraph.from_dict({"drivable_areas": {}, "pedestrian_crossings": {}})
    assert "lane_segments" in str(e.value)


def test_map_archive_path_finds_the_archive(tmp_path):
    d = tmp_path / "log1" / "map"
    d.mkdir(parents=True)
    f = d / "log_map_archive_log1____MIA_city_10.json"
    f.write_text("{}", encoding="utf-8")
    assert av2.map_archive_path(tmp_path / "log1") == f


def test_map_archive_path_missing_raises(tmp_path):
    with pytest.raises(av2.Argoverse2MapError):
        av2.map_archive_path(tmp_path / "nothing")


# --------------------------------------------------------------------------- #
# Discovery / split unit                                                       #
# --------------------------------------------------------------------------- #
def test_discover_logs_and_split_unit(tmp_path):
    for lg in ("logB", "logA"):
        (tmp_path / "val" / lg).mkdir(parents=True)
    assert av2.discover_logs(tmp_path, "val") == ["logA", "logB"]
    assert av2.discover_logs(tmp_path / "val", None) == ["logA", "logB"]
    assert av2.split_unit_of("logA") == "logA"


def test_discover_logs_missing_split_fails_loud_not_silently(tmp_path):
    """REGRESSION: the first draft fell back to ``root`` and returned the SPLIT
    names as if they were logs — a silent corpus-wide mis-ingest. It must raise,
    and the error must list what is actually there."""
    (tmp_path / "val" / "logA").mkdir(parents=True)
    with pytest.raises(av2.Argoverse2MapError) as e:
        av2.discover_logs(tmp_path, "missing_split")
    assert "val" in str(e.value)          # names the available split
    assert "split=None" in str(e.value)   # and the escape hatch


# --------------------------------------------------------------------------- #
# Ego track + calibration (feather path)                                       #
# --------------------------------------------------------------------------- #
def _write_pose_feather(log_dir: Path, n=41, hz=20.0, v=10.0):
    pd = pytest.importorskip("pandas")
    log_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(n) / hz
    df = pd.DataFrame({
        "timestamp_ns": (t * 1e9).astype(np.int64),
        "qw": np.ones(n), "qx": np.zeros(n), "qy": np.zeros(n), "qz": np.zeros(n),
        "tx_m": v * t, "ty_m": np.zeros(n), "tz_m": np.zeros(n)})
    df.to_feather(log_dir / "city_SE3_egovehicle.feather")
    return t


def test_ego_track_speed_and_units(tmp_path):
    """AV2 timestamps are NANOseconds — a microsecond assumption is a 1000x bug."""
    _write_pose_feather(tmp_path, n=41, hz=20.0, v=10.0)
    poses, t = av2.ego_track(tmp_path)
    assert poses.shape == (41, 4)
    assert float(t[-1] - t[0]) == pytest.approx(2.0, abs=1e-6)   # 41 @ 20 Hz
    assert np.allclose(poses[:, 3], 10.0, atol=1e-4)             # constant 10 m/s
    assert np.allclose(poses[:, 2], 0.0, atol=1e-6)              # identity quat


def test_ego_track_rejects_a_bad_table(tmp_path):
    pd = pytest.importorskip("pandas")
    tmp_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp_ns": [1, 2]}).to_feather(
        tmp_path / "city_SE3_egovehicle.feather")
    with pytest.raises(av2.Argoverse2MapError) as e:
        av2.ego_track(tmp_path)
    assert "missing columns" in str(e.value)


def test_ego_track_missing_table_says_the_tars_are_needed(tmp_path):
    with pytest.raises(av2.Argoverse2MapError) as e:
        av2.ego_track(tmp_path / "absent")
    assert "lane-graph pull" in str(e.value).lower()


def test_actions_from_track_shape_and_straight_line(tmp_path):
    _write_pose_feather(tmp_path, n=41, hz=20.0, v=10.0)
    poses, t = av2.ego_track(tmp_path)
    a = av2.actions_from_track(poses, t)
    assert a.shape == (41, 2)
    assert np.allclose(a[:, 0], 0.0, atol=1e-6)        # straight -> no steer
    assert np.allclose(a[:, 1], 0.0, atol=1e-3)        # constant speed -> no accel


def test_camera_intrinsics_reads_per_log_and_keeps_distortion(tmp_path):
    pd = pytest.importorskip("pandas")
    cal = tmp_path / "calibration"
    cal.mkdir(parents=True)
    pd.DataFrame({
        "sensor_name": ["ring_front_center", "ring_side_left"],
        "fx_px": [1686.0, 1690.0], "fy_px": [1686.0, 1690.0],
        "cx_px": [960.0, 970.0], "cy_px": [775.0, 780.0],
        "k1": [-0.24, -0.25], "k2": [0.09, 0.08], "k3": [-0.01, -0.02],
        "width_px": [1550, 2048], "height_px": [2048, 1550],
    }).to_feather(cal / "intrinsics.feather")
    intr = av2.camera_intrinsics_of(tmp_path, "ring_front_center")
    assert intr.fx == pytest.approx(1686.0) and intr.width == 1550
    assert intr.dist == pytest.approx((-0.24, 0.09, 0.0, 0.0, -0.01))
    with pytest.raises(av2.Argoverse2MapError) as e:
        av2.camera_intrinsics_of(tmp_path, "no_such_cam")
    assert "ring_side_left" in str(e.value)            # error lists what IS there


# --------------------------------------------------------------------------- #
# Geometry helpers are IMPORTED from nuscenes, not re-derived                  #
# --------------------------------------------------------------------------- #
def test_quaternion_helpers_are_the_same_objects_as_nuscenes():
    from tanitad.data import nuscenes as ns
    assert av2.quat_to_yaw is ns.quat_to_yaw
    assert av2.quat_to_rotmat is ns.quat_to_rotmat
    assert av2.wrap_pi is ns.wrap_pi


def test_yaw_convention_matches():
    q = [math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)]   # +45 deg yaw
    assert av2.quat_to_yaw(q) == pytest.approx(math.pi / 4, abs=1e-9)


# --------------------------------------------------------------------------- #
# REAL BYTES — opt-in via TANITAD_AV2_MAP_DIR                                  #
# --------------------------------------------------------------------------- #
_REAL = os.environ.get("TANITAD_AV2_MAP_DIR", "")
_real_maps = sorted(Path(_REAL).rglob("*.json")) if _REAL and Path(_REAL).is_dir() else []
_real_maps = [p for p in _real_maps if not p.name.startswith("_")]
_needs_real = pytest.mark.skipif(
    not _real_maps,
    reason="set TANITAD_AV2_MAP_DIR to the pulled AV2 sensor lane graphs")


@_needs_real
def test_real_av2_every_map_parses():
    bad = []
    for p in _real_maps:
        try:
            g = av2.load_lane_graph(p)
            assert len(g) > 0
        except Exception as e:                                    # noqa: BLE001
            bad.append((p.name, f"{type(e).__name__}: {e}"))
    assert not bad, f"{len(bad)} real maps failed to parse: {bad[:5]}"


@_needs_real
def test_real_av2_sensor_split_has_zero_explicit_centerlines():
    """The load-bearing trap, asserted on the real corpus rather than a fixture."""
    explicit = derived = 0
    for p in _real_maps:
        c = av2.load_lane_graph(p).centerline_source_counts()
        explicit += c["explicit"]
        derived += c["derived"]
    assert explicit == 0, f"expected 0 explicit centerlines in sensor, got {explicit}"
    assert derived > 0


@_needs_real
def test_real_av2_every_lane_yields_a_finite_centerline():
    for p in _real_maps[:50]:
        g = av2.load_lane_graph(p)
        for i in g.ids():
            cl = g.centerline(i)
            assert cl.shape[0] >= 2 and np.isfinite(cl).all(), f"{p.name}:{i}"


@_needs_real
def test_real_av2_dangling_successors_exist_and_do_not_raise():
    dang = 0
    for p in _real_maps[:50]:
        g = av2.load_lane_graph(p)
        for i in g.ids():
            dang += len(g.dangling_successors(i))
            g.routes_from(i, max_depth=3, max_routes=8)     # must not raise/hang
    assert dang > 0, "expected the crop boundary to produce dangling successors"
