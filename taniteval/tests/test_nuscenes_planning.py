"""Tests for the nuScenes open-loop planning harness (``taniteval/adapters/nuscenes_planning.py``).

⛔ NO nuScenes BYTE IS READ. Every fixture is synthetic and nuScenes-SHAPED (the 13 JSON tables,
written to a temp dir and loaded through the real ``tanitad.data.nuscenes.load_tables``).

Every expected value is a LITERAL pre-stated in the W6 package's ``SPEC.md`` — derived from a
closed form, or copied from a banked primary (PARA-Drive Table 1) — never an expression over the
code under test (CLAUDE.md: "a check that shares the defect it checks for is green forever").

⭐ Deliberate-regression arms:
  * :func:`test_MUTATION_swapped_reducers_turn_target_B_red` mutates the SOURCE (the reducer
    dispatch literal) and requires the discriminating target to FAIL while the constant-offset
    target still passes — proving target A alone could never have caught a convention swap.
  * :func:`test_MUTATION_uniad_without_the_x_flip_hits_the_mirror_cell` mutates the source again
    (drops UniAD's ``update`` x flip) and requires the lateral-sign literal to FAIL.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
import tempfile
import types

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))       # <repo>/taniteval/tests
_TE = os.path.dirname(_HERE)                             # <repo>/taniteval
_REPO = os.path.dirname(_TE)                             # <repo>
for _p in (os.path.join(_REPO, "stack"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import nuscenes_planning as NP              # noqa: E402

PKG = os.path.join(_REPO, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming",
                   "2026-09-19-nuscenes-planning-harness")
AT_T, AVG = NP.Reduction.AT_T, NP.Reduction.AVG_UP_TO_T
H = ("1s", "2s", "3s", "avg_1_2_3s")


# --------------------------------------------------------------------------- #
# SPEC §2 — analytic L2 targets                                                 #
# --------------------------------------------------------------------------- #
def _straight_gt(n=1):
    return np.tile(np.array([[0.0, 2.5 * (k + 1)] for k in range(6)]), (n, 1, 1))


def _headline(mod, k, reduction, pipeline, metric="L2_m"):
    res = mod.score(k, mod.variant_tag(getattr(mod.Reduction, reduction.name),
                                       getattr(mod.Pipeline, pipeline.name)))
    return [res.get(metric, h).value for h in H], [res.get(metric, c).value for c in mod.COLUMNS]


def _check_target_A(mod):
    gt = _straight_gt()
    pred = gt + np.array([1.5, 0.0])
    k = mod.kernel_stp3(pred, gt)
    for red in (AT_T, AVG):
        head, cols = _headline(mod, k, red, NP.Pipeline.STP3)
        assert head == [1.5, 1.5, 1.5, 1.5], (red, head)
        assert cols == [1.5] * 6, (red, cols)


def _check_target_B(mod):
    gt = _straight_gt()
    pred = gt + np.array([[0.5 * (k + 1), 0.0] for k in range(6)])
    k = mod.kernel_stp3(pred, gt)
    head_t, cols_t = _headline(mod, k, AT_T, NP.Pipeline.STP3)
    head_a, cols_a = _headline(mod, k, AVG, NP.Pipeline.STP3)
    assert head_t == [1.0, 2.0, 3.0, 2.0], head_t
    assert cols_t == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], cols_t
    assert head_a == [0.75, 1.25, 1.75, 1.25], head_a
    assert cols_a == [0.5, 0.75, 1.0, 1.25, 1.5, 1.75], cols_a


def test_target_A_constant_offset_is_1p5_in_both_conventions():
    _check_target_A(NP)


def test_target_B_linear_error_literals_differ_between_conventions():
    _check_target_B(NP)


@pytest.mark.parametrize("pipeline", ["uniad", "vad", "stp3"])
def test_target_B_holds_through_every_kernel(pipeline):
    """All three kernels compute the same per-waypoint L2 (UniAD with a full mask)."""
    gt = _straight_gt()
    pred = gt + np.array([[0.5 * (k + 1), 0.0] for k in range(6)])
    occ = np.zeros((1, 6, 200, 200), np.uint8)
    if pipeline == "uniad":
        k = NP.kernel_uniad(pred, gt, np.ones((1, 6), bool), occ)
        tag_p = NP.Pipeline.UNIAD
    elif pipeline == "vad":
        k = NP.kernel_vad(pred, gt, occ, np.ones(1, bool))
        tag_p = NP.Pipeline.VAD
    else:
        k = NP.kernel_stp3(pred, gt, occ)
        tag_p = NP.Pipeline.STP3
    assert _headline(NP, k, AT_T, tag_p)[0] == [1.0, 2.0, 3.0, 2.0]
    assert _headline(NP, k, AVG, tag_p)[0] == [0.75, 1.25, 1.75, 1.25]


# --------------------------------------------------------------------------- #
# SPEC §3 — source-level mutation arms (must go RED)                            #
# --------------------------------------------------------------------------- #
def _load_mutant(old: str, new: str) -> types.ModuleType:
    src_path = NP.__file__
    src = open(src_path, encoding="utf-8").read()
    n = src.count(old)
    assert n == 1, f"mutation anchor must match exactly once, matched {n} times: {old!r}"
    mod = types.ModuleType("nuscenes_planning_mutant")
    mod.__file__ = src_path
    sys.modules["nuscenes_planning_mutant"] = mod
    try:
        exec(compile(src.replace(old, new), src_path, "exec"), mod.__dict__)
    finally:
        sys.modules.pop("nuscenes_planning_mutant", None)
    return mod


def test_MUTATION_swapped_reducers_turn_target_B_red():
    mutant = _load_mutant(
        "_REDUCERS = {Reduction.AT_T: _reduce_at_t, Reduction.AVG_UP_TO_T: _reduce_avg_up_to_t}",
        "_REDUCERS = {Reduction.AT_T: _reduce_avg_up_to_t, Reduction.AVG_UP_TO_T: _reduce_at_t}")
    _check_target_A(mutant)                     # the non-discriminating control stays GREEN
    with pytest.raises(AssertionError):
        _check_target_B(mutant)                 # the discriminating target goes RED


def test_MUTATION_uniad_without_the_x_flip_hits_the_mirror_cell():
    mutant = _load_mutant("    p[..., 0] = -p[..., 0]\n    g[..., 0] = -g[..., 0]\n",
                          "    pass\n")
    with pytest.raises(AssertionError):
        _check_uniad_lateral_sign(mutant)
    _check_uniad_lateral_sign(NP)               # the real module passes the same literal


# --------------------------------------------------------------------------- #
# SPEC §4 — PARA-Drive Table 1 (banked `paradrive-cvpr2024`, p.4): real published vectors
# --------------------------------------------------------------------------- #
PD_L2_AT_T = (0.2788, 0.5380, 0.8239, 1.1513, 1.5322, 1.9821)   # "+ Remove averaging over time"
PD_COL_AT_T = (0.10, 0.04, 0.16, 0.39, 0.61, 1.11)


def test_paradrive_table1_L2_rows_are_related_by_exactly_the_two_reductions():
    at_t = NP.reduce_per_timestep(PD_L2_AT_T, AT_T)["headline"]
    avg = NP.reduce_per_timestep(PD_L2_AT_T, AVG)["headline"]
    for got, want in zip([at_t[h] for h in H], (0.5380, 1.1513, 1.9821, 1.2238)):
        assert abs(got - want) <= 1e-4, (got, want)
    # the VAD-protocol row of the SAME table (VAD reproduction): 0.4084 0.6980 1.0511 / 0.7192
    for got, want in zip([avg[h] for h in H], (0.4084, 0.6980, 1.0511, 0.7192)):
        assert abs(got - want) <= 1e-4, (got, want)


def test_paradrive_table1_collision_is_time_averaged_too():
    at_t = NP.reduce_per_timestep(PD_COL_AT_T, AT_T)["headline"]
    avg = NP.reduce_per_timestep(PD_COL_AT_T, AVG)["headline"]
    for got, want in zip([at_t[h] for h in H], (0.04, 0.39, 1.11, 0.51)):
        assert abs(got - want) <= 5e-3, (got, want)
    for got, want in zip([avg[h] for h in H], (0.07, 0.17, 0.40, 0.21)):
        assert abs(got - want) <= 5e-3, (got, want)


# --------------------------------------------------------------------------- #
# SPEC §5 — collision kernels, literal cells                                    #
# --------------------------------------------------------------------------- #
def test_ego_box_raster_is_the_literal_32_pixels():
    want = {(r, c) for r in range(97, 105) for c in range(98, 102)}
    got = {tuple(x) for x in NP.ego_box_rc().tolist()}
    assert got == want and len(NP.ego_box_rc()) == 32


def _one_waypoint_case(x=0.0, y=10.0):
    pred = _straight_gt()
    pred[0, 3] = [x, y]
    gt = pred.copy()
    gt[..., 0] = 20.0                           # GT 20 m to the side: never hits the obstacle
    return pred, gt


def _uniad(pred, gt, occ):
    return NP.kernel_uniad(pred, gt, np.ones((1, 6), bool), occ)


def test_uniad_known_overlap_known_miss_and_point_cell():
    pred, gt = _one_waypoint_case()
    occ = np.zeros((1, 6, 200, 200), np.uint8)
    occ[0, 3, 120, 100] = 1                     # inside rows 117..124 x cols 98..101
    k = _uniad(pred, gt, occ)
    assert k.obj_box_col[0].tolist() == [0, 0, 0, 1, 0, 0]
    assert k.obj_col[0].tolist() == [0, 0, 0, 0, 0, 0]
    occ[0, 3, 119, 99] = 1                      # the point-check cell
    assert _uniad(pred, gt, occ).obj_col[0].tolist() == [0, 0, 0, 1, 0, 0]
    miss = np.zeros_like(occ)
    miss[0, 3, 120, 102] = 1                    # one column outside the box
    km = _uniad(pred, gt, miss)
    assert km.obj_box_col.sum() == 0 and km.obj_col.sum() == 0


def _check_uniad_lateral_sign(mod):
    pred, gt = _one_waypoint_case(x=-2.0, y=10.0)          # 2 m LEFT (LiDAR x is right)
    occ = np.zeros((1, 6, 200, 200), np.uint8)
    occ[0, 3, 120, 95] = 1                      # UniAD cols = RIGHT: box cols 94..97
    k = mod.kernel_uniad(pred, gt, np.ones((1, 6), bool), occ)
    assert k.obj_box_col[0].tolist() == [0, 0, 0, 1, 0, 0], k.obj_box_col[0].tolist()
    mirror = np.zeros_like(occ)
    mirror[0, 3, 120, 104] = 1
    k2 = mod.kernel_uniad(pred, gt, np.ones((1, 6), bool), mirror)
    assert k2.obj_box_col.sum() == 0


def test_uniad_lateral_sign():
    _check_uniad_lateral_sign(NP)


def test_stp3_grid_is_the_mirror_of_uniad():
    """ST-P3's labels are in the yaw-aligned ego frame (cols = LEFT), so the same waypoint lands
    on the mirrored columns 102..105 — each kernel is right against its OWN occupancy."""
    pred, gt = _one_waypoint_case(x=-2.0, y=10.0)
    occ = np.zeros((1, 6, 200, 200), np.uint8)
    occ[0, 3, 120, 104] = 1
    assert NP.kernel_stp3(pred, gt, occ).obj_box_col[0].tolist() == [0, 0, 0, 1, 0, 0]
    wrong = np.zeros_like(occ)
    wrong[0, 3, 120, 95] = 1
    assert NP.kernel_stp3(pred, gt, wrong).obj_box_col.sum() == 0
    pred0, gt0 = _one_waypoint_case()
    pt = np.zeros_like(occ)
    pt[0, 3, 119, 99] = 1
    assert NP.kernel_stp3(pred0, gt0, pt).obj_col[0].tolist() == [0, 0, 0, 1, 0, 0]


def test_vad_rows_are_flipped_and_its_point_check_reads_cell_29_49():
    pred, gt = _one_waypoint_case()
    fv = np.ones(1, bool)
    occ = np.zeros((1, 6, 200, 200), np.uint8)
    occ[0, 3, 80, 100] = 1                      # rows 76..83 x cols 98..101
    k = NP.kernel_vad(pred, gt, occ, fv)
    assert k.obj_box_col[0].tolist() == [0, 0, 0, 1, 0, 0]
    assert k.obj_col.sum() == 0                 # the point check never looks near the ego
    origin = np.zeros_like(occ)
    origin[0, 3, 100, 100] = 1                  # an obstacle AT the grid centre
    assert NP.kernel_vad(pred, gt, origin, fv).obj_col.sum() == 0
    bug = np.zeros_like(occ)
    bug[0, 3, 29, 49] = 1                       # VERBATIM: metric_stp3.py:272-273
    assert NP.kernel_vad(pred, gt, bug, fv).obj_col[0].tolist() == [0, 0, 0, 1, 0, 0]
    lat, glat = _one_waypoint_case(x=-2.0, y=10.0)
    side = np.zeros_like(occ)
    side[0, 3, 80, 95] = 1                      # VAD cols = right: box cols 94..97
    assert NP.kernel_vad(lat, glat, side, fv).obj_box_col[0].tolist() == [0, 0, 0, 1, 0, 0]


@pytest.mark.parametrize("pipeline", ["uniad", "vad", "stp3"])
def test_gt_exclusion_rule_and_the_raw_gt_floor(pipeline):
    pred, _ = _one_waypoint_case()
    gt = pred.copy()                            # GT and plan share the waypoint at t=3
    occ = np.zeros((1, 6, 200, 200), np.uint8)
    cell = {"uniad": (120, 100), "vad": (80, 100), "stp3": (120, 100)}[pipeline]
    occ[0, 3, cell[0], cell[1]] = 1
    k = {"uniad": lambda: _uniad(pred, gt, occ),
         "vad": lambda: NP.kernel_vad(pred, gt, occ, np.ones(1, bool)),
         "stp3": lambda: NP.kernel_stp3(pred, gt, occ)}[pipeline]()
    assert k.gt_box_col[0].tolist() == [0, 0, 0, 1, 0, 0]
    assert k.obj_box_col[0].tolist() == [0, 0, 0, 0, 0, 0]     # excluded, as in every reference


def test_collision_rate_is_a_percentage_over_the_scored_samples():
    pred, gt = _one_waypoint_case()
    pred2, gt2 = np.concatenate([pred, pred]), np.concatenate([gt, gt])
    occ = np.zeros((2, 6, 200, 200), np.uint8)
    occ[0, 3, 120, 100] = 1                     # only sample 0 collides
    res = NP.score(NP.kernel_stp3(pred2, gt2, occ), NP.variant_tag(AT_T, NP.Pipeline.STP3))
    assert res.get("collision_box_pct", "2.0s").value == 50.0
    assert res.get("collision_box_pct", "2s").value == 50.0
    assert res.get("collision_box_pct", "1s").value == 0.0


def test_uniad_masking_counts_the_sample_but_scores_zero():
    gt = _straight_gt(2)
    pred = gt + np.array([1.0, 0.0])
    mask = np.array([[True] * 6, [False] * 6])
    occ = np.zeros((2, 6, 200, 200), np.uint8)
    res = NP.score(NP.kernel_uniad(pred, gt, mask, occ), NP.protocol_tag("nuScenes_OL_L2_uniad"))
    assert [res.get("L2_m", h).value for h in H] == [0.5, 0.5, 0.5, 0.5]
    res_v = NP.score(NP.kernel_vad(pred, gt, occ, np.array([True, False])),
                     NP.protocol_tag("nuScenes_OL_L2_stp3"))
    assert [res_v.get("L2_m", h).value for h in H] == [1.0, 1.0, 1.0, 1.0]


# --------------------------------------------------------------------------- #
# The API — no untagged, mixed or claim-bearing number can exist                #
# --------------------------------------------------------------------------- #
def test_score_refuses_a_missing_or_string_tag():
    k = NP.kernel_stp3(_straight_gt(), _straight_gt())
    with pytest.raises(TypeError):
        NP.score(k)                                               # noqa — no default tag
    with pytest.raises(TypeError):
        NP.score(k, "nuscenes-planning/uniad-noavg")
    with pytest.raises(NP.RefusedInput):
        NP.score(k, NP.protocol_tag("nuScenes_OL_L2_uniad"))      # UniAD tag on an ST-P3 kernel


def test_protocol_names_are_the_closed_set():
    assert sorted(NP.PROTOCOLS) == ["nuScenes_OL_L2_stp3", "nuScenes_OL_L2_uniad"]
    assert NP.protocol_tag("nuScenes_OL_L2_uniad").published_family == "uniad-noavg"
    assert NP.protocol_tag("nuScenes_OL_L2_stp3").published_family == "stp3-temavg"
    assert NP.variant_tag(AT_T, NP.Pipeline.VAD).protocol is None
    with pytest.raises(NP.RefusedInput):
        NP.protocol_tag("nuScenes_OL_L2_average")


def test_a_mixed_convention_result_cannot_be_built():
    k = NP.kernel_stp3(_straight_gt(), _straight_gt())
    a = NP.score(k, NP.variant_tag(AT_T, NP.Pipeline.STP3))
    b = NP.score(k, NP.variant_tag(AVG, NP.Pipeline.STP3))
    with pytest.raises(NP.RefusedInput):
        NP.ConventionResult(a.tag, a.rows + b.rows, a.n_samples, True)
    with pytest.raises(TypeError):
        NP.TaggedValue("uniad-noavg", "L2_m", "1s", 1.0, 1)


def test_every_serialised_row_carries_its_tag():
    k = NP.kernel_vad(_straight_gt(), _straight_gt(), np.zeros((1, 6, 200, 200), np.uint8),
                      np.ones(1, bool))
    d = NP.score(k, NP.protocol_tag("nuScenes_OL_L2_stp3")).to_dict()
    assert d["rows"] and all(r["protocol_tag"] == "nuscenes-planning/stp3-temavg+vad@1688c4b"
                             for r in d["rows"])
    assert all(r["protocol"] == "nuScenes_OL_L2_stp3" for r in d["rows"])


def _artifact(**kw):
    k = NP.kernel_vad(_straight_gt(), _straight_gt(), np.zeros((1, 6, 200, 200), np.uint8),
                      np.ones(1, bool))
    res = NP.score(k, NP.protocol_tag("nuScenes_OL_L2_stp3"))
    base = dict(arm="GT", arm_kind="reference", declared_inputs=[], command_source="none",
                split={"name": "fixture"})
    base.update(kw)
    return NP.build_artifact(res, **base)


def test_artifact_is_never_claim_bearing_and_cannot_be_asked_to_be():
    art = _artifact()
    assert art["claim_bearing"] is False
    assert "H-EVAL-6" in art["claim_bearing_reason"] and art["register_rows"] == ["H-EVAL-6", "D-BENCH-PORT"]
    with pytest.raises(TypeError):
        _artifact(claim_bearing=True)                            # no such parameter exists
    for marker in ("four_families", "headline", "ade_0_2s", "full_set", "heldout", "block"):
        assert marker not in art, marker                         # criteria_check in-scope markers
    p = art["benchmark"]["nuscenes"]["planning"]
    for key in ("protocol_tag", "gt_control", "command_source", "nonstraight"):
        assert key in p, key                                     # registry nusc.plan.* keys


@pytest.mark.parametrize("src", ["gt_future_derived", "gt_ego_fut_cmd", "ego_fut_cmd", "gt"])
def test_the_gt_derived_command_is_refused_as_an_input(src):
    with pytest.raises(NP.RefusedInput):
        _artifact(command_source=src)


def test_a_model_arm_without_its_geometry_stamp_is_refused():
    with pytest.raises(NP.RefusedInput):
        _artifact(arm="A2", arm_kind="model")


# --------------------------------------------------------------------------- #
# SPEC §6 — cv2.fillPoly port                                                   #
# --------------------------------------------------------------------------- #
def _rows(img):
    return [sorted(np.flatnonzero(r).tolist()) for r in img]


def test_fillpoly_axis_aligned_rectangle_is_closed():
    img = np.zeros((10, 10), np.uint8)
    NP.cv_fill_poly(img, [(2, 3), (5, 3), (5, 6), (2, 6)], 1)
    want = np.zeros((10, 10), np.uint8)
    want[3:7, 2:6] = 1
    assert (img == want).all() and img.sum() == 16


def test_fillpoly_diamond_literal():
    img = np.zeros((5, 5), np.uint8)
    NP.cv_fill_poly(img, [(2, 0), (4, 2), (2, 4), (0, 2)], 1)
    assert _rows(img) == [[2], [1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3], [2]]


def test_fillpoly_clips_at_the_grid_edge():
    img = np.zeros((5, 5), np.uint8)
    NP.cv_fill_poly(img, [(-3, -3), (2, -3), (2, 2), (-3, 2)], 1)
    want = np.zeros((5, 5), np.uint8)
    want[0:3, 0:3] = 1
    assert (img == want).all()


def test_fillpoly_parity_with_real_cv2():
    """The port vs REAL cv2.fillPoly 4.5.4, on this test's own 200 random quads (seed 0).

    Until 2026-09-26 this SKIPPED on the dev box ("NOT RUN: OpenCV not installed in this venv") — a
    guard that had never run. It now compares against rasters drawn ONCE by cv2 4.5.4 in a throwaway env
    and banked in ``fixtures/fillpoly_opencv454/`` (provenance and further arms, incl. mutation, in
    ``test_fillpoly_opencv454_reference.py``). ⭐ The quads are regenerated here and must EQUAL the banked
    vertices, so the reference is for THESE inputs: if numpy's stream ever changes, this says so plainly
    instead of silently comparing different polygons."""
    ref = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "fillpoly_opencv454",
                               "fillpoly_cv2_454_reference.npz"), allow_pickle=False)
    rng = np.random.default_rng(0)
    for i in range(200):
        pts = rng.integers(-20, 60, size=(4, 2))
        assert (ref["quads_polys"][i, :4] == pts).all(), f"case {i}: banked quad != this test's quad"
        a = np.zeros((40, 40), np.uint8)
        NP.cv_fill_poly(a, [tuple(p) for p in pts.tolist()], 1)
        assert (a == ref["quads_rasters"][i]).all(), pts.tolist()


# --------------------------------------------------------------------------- #
# SPEC §7 — sample sets (published counts)                                      #
# --------------------------------------------------------------------------- #
def test_expected_sample_counts_reproduce_the_published_5119_and_4819():
    assert NP.expected_sample_counts() == {"uniad": 6019, "vad": 5119, "stp3": 4819}


def test_devkit_val_split_is_the_pinned_devkit_list():
    src = open(os.path.join(PKG, "raw", "devkit_splits_b40adc4.py.txt"), encoding="utf-8").read()
    blob = hashlib.sha1(b"blob %d\0" % len(src.encode("utf-8")) + src.encode("utf-8")).hexdigest()
    assert blob == "6988b5c84bdbb1d1160da8d6b5816f89145daffe"   # GitHub blob @ b40adc4
    lists = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                lists[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert tuple(lists["val"]) == NP.DEVKIT_VAL_SCENES and len(NP.DEVKIT_VAL_SCENES) == 150
    assert tuple(lists["mini_val"]) == NP.DEVKIT_MINI_VAL_SCENES == ("scene-0103", "scene-0916")
    blk = {int(s[-4:]) for s in NP.DEVKIT_VAL_SCENES} & NP.STP3_SCENE_BLACKLIST
    assert blk == set()                                          # no val scene is blacklisted


# --------------------------------------------------------------------------- #
# Synthetic nuScenes-shaped metadata                                             #
# --------------------------------------------------------------------------- #
CATS = ["noise", "animal", "human.pedestrian.adult", "human.pedestrian.child",
        "human.pedestrian.construction_worker", "human.pedestrian.personal_mobility",
        "human.pedestrian.police_officer", "human.pedestrian.stroller",
        "human.pedestrian.wheelchair", "movable_object.barrier", "movable_object.debris",
        "movable_object.pushable_pullable", "movable_object.trafficcone",
        "static_object.bicycle_rack", "vehicle.bicycle", "vehicle.bus.bendy", "vehicle.bus.rigid",
        "vehicle.car", "vehicle.construction", "vehicle.emergency.ambulance",
        "vehicle.emergency.police", "vehicle.motorcycle", "vehicle.trailer", "vehicle.truck",
        "flat.driveable_surface", "flat.sidewalk", "flat.terrain", "flat.other", "static.manmade",
        "static.vegetation", "static.other", "vehicle.ego"]
LIDAR_T = [0.943, 0.0, 1.84]
A_LEVER = 0.943


def _q_yaw(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _q_from_R(R):
    R = np.asarray(R, float)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [w, x, y, z]


LIDAR_Q = _q_yaw(-math.pi / 2)                    # x_lidar = right, y_lidar = forward
CAM_Q = _q_from_R([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])   # camera (x right, y down, z fwd) -> ego
K_NOM = [[1266.417203046554, 0.0, 816.2670197447984], [0.0, 1266.417203046554, 491.50706579294757],
         [0.0, 0.0, 1.0]]


def _ego_straight(t):
    return 4.0 * t, 0.0, 0.0


def _ego_arc(t, R=20.0, v=5.0):
    psi = v * t / R
    return R * math.sin(psi), R * (1 - math.cos(psi)), psi


def make_tables(scenes, ego=_ego_straight, agents=()):
    """scenes: [(name, n_samples)]; agents: [{"scene", "cat", "wlh", "xy_yaw", "lidar_pts"}]
    (a static agent annotated in every sample of its scene)."""
    T = {k: [] for k in ("attribute", "calibrated_sensor", "category", "ego_pose", "instance",
                         "log", "map", "sample", "sample_annotation", "sample_data", "scene",
                         "sensor", "visibility")}
    T["sensor"] = [{"token": "sen-lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
                   {"token": "sen-cam", "channel": "CAM_FRONT", "modality": "camera"}]
    T["calibrated_sensor"] = [
        {"token": "cs-lidar", "sensor_token": "sen-lidar", "translation": LIDAR_T,
         "rotation": LIDAR_Q, "camera_intrinsic": []},
        {"token": "cs-cam", "sensor_token": "sen-cam", "translation": [1.70, 0.0, 1.51],
         "rotation": CAM_Q, "camera_intrinsic": K_NOM}]
    T["category"] = [{"token": f"cat-{i}", "name": n, "description": "", "index": i}
                     for i, n in enumerate(CATS)]
    T["visibility"] = [{"token": str(i), "level": lv, "description": ""}
                       for i, lv in enumerate(["v0-40", "v40-60", "v60-80", "v80-100"], start=1)]
    T["log"] = [{"token": "log-0", "location": "boston-seaport"}]
    t_us0 = 1_600_000_000_000_000
    for si, (name, n) in enumerate(scenes):
        stoks = [f"{name}-s{i}" for i in range(n)]
        T["scene"].append({"token": f"tok-{name}", "name": name, "log_token": "log-0",
                           "first_sample_token": stoks[0], "last_sample_token": stoks[-1],
                           "nbr_samples": n, "description": ""})
        base = t_us0 + si * 100_000_000
        for i, st in enumerate(stoks):
            ts = base + i * 500_000
            T["sample"].append({"token": st, "timestamp": ts, "scene_token": f"tok-{name}",
                                "prev": stoks[i - 1] if i else "",
                                "next": stoks[i + 1] if i + 1 < n else ""})
            for kind, dt_us, cs, key in (("sweep", -50_000, "cs-lidar", False),
                                         ("lidar", 0, "cs-lidar", True),
                                         ("cam", 0, "cs-cam", True)):
                tok = f"{st}-{kind}"
                x, y, yaw = ego((i * 500_000 + dt_us) / 1e6)
                T["ego_pose"].append({"token": f"ep-{tok}", "timestamp": ts + dt_us,
                                      "translation": [x, y, 0.0], "rotation": _q_yaw(yaw)})
                T["sample_data"].append({
                    "token": tok, "sample_token": st, "ego_pose_token": f"ep-{tok}",
                    "calibrated_sensor_token": cs, "timestamp": ts + dt_us,
                    "fileformat": "jpg" if kind == "cam" else "pcd",
                    "is_key_frame": key, "height": 900 if kind == "cam" else 0,
                    "width": 1600 if kind == "cam" else 0,
                    "filename": (f"samples/CAM_FRONT/{name}__CAM_FRONT__{ts}.jpg" if kind == "cam"
                                 else f"{'samples' if key else 'sweeps'}/LIDAR_TOP/{name}__{ts + dt_us}.pcd.bin"),
                    "prev": f"{st}-sweep" if kind == "lidar" else "",
                    "next": f"{st}-lidar" if kind == "sweep" else ""})
        for ai, ag in enumerate(a for a in agents if a["scene"] == name):
            it = f"inst-{name}-{ai}"
            T["instance"].append({"token": it, "category_token": f"cat-{CATS.index(ag['cat'])}",
                                  "nbr_annotations": n})
            atoks = [f"ann-{name}-{ai}-{i}" for i in range(n)]
            x, y, yaw = ag["xy_yaw"]
            for i in range(n):
                T["sample_annotation"].append({
                    "token": atoks[i], "sample_token": stoks[i], "instance_token": it,
                    "visibility_token": "4", "attribute_tokens": [],
                    "translation": [x, y, 0.8], "size": list(ag["wlh"]), "rotation": _q_yaw(yaw),
                    "prev": atoks[i - 1] if i else "", "next": atoks[i + 1] if i + 1 < n else "",
                    "num_lidar_pts": ag.get("lidar_pts", 10), "num_radar_pts": 0})
    return T


def _write(T, root, version):
    d = os.path.join(root, version)
    os.makedirs(d, exist_ok=True)
    for k, v in T.items():
        with open(os.path.join(d, f"{k}.json"), "w", encoding="utf-8") as fh:
            json.dump(v, fh)


# --------------------------------------------------------------------------- #
# SPEC §7 / §8 — sample sets and GT on synthetic metadata                       #
# --------------------------------------------------------------------------- #
def test_sample_sets_on_synthetic_scenes():
    m = NP.Meta(make_tables([("scene-0001", 40), ("scene-0002", 39)]))
    rows = NP.sample_sets(m, ["scene-0001", "scene-0002"])
    counts = {p: sum(r[p] for r in rows) for p in ("uniad", "vad", "stp3")}
    assert counts == {"uniad": 79, "vad": 67, "stp3": 63}
    one = NP.sample_sets(m, ["scene-0001"])
    assert {p: sum(r[p] for r in one) for p in ("uniad", "vad", "stp3")} == \
        {"uniad": 40, "vad": 34, "stp3": 32}


def test_gt_is_the_lidar_origin_and_the_lever_arm_cancels_on_a_straight():
    m = NP.Meta(make_tables([("scene-0001", 10)]))
    s0 = m.samples_of_scene["tok-scene-0001"][0]
    xy, valid, _ = NP.gt_trajectory(m, s0)
    assert valid.all()
    assert np.allclose(xy, [[0.0, 2.0 * k] for k in range(1, 7)], atol=1e-9)
    assert NP.gt_command(xy, valid) == "FORWARD"


def test_gt_on_a_left_arc_matches_the_closed_form_and_commands_left():
    m = NP.Meta(make_tables([("scene-0001", 10)], ego=_ego_arc))
    s0 = m.samples_of_scene["tok-scene-0001"][0]
    xy, valid, _ = NP.gt_trajectory(m, s0)
    R, a = 20.0, A_LEVER
    want = [(-(R * (1 - math.cos(0.125 * k)) + a * math.sin(0.125 * k)),
             R * math.sin(0.125 * k) + a * math.cos(0.125 * k) - a) for k in range(1, 7)]
    assert np.allclose(xy, want, atol=1e-9)
    assert abs(xy[5, 0] - (-6.009008)) < 1e-6                    # SPEC §8 literal
    assert NP.gt_command(xy, valid) == "LEFT"


def test_uniad_gt_is_masked_past_the_scene_end():
    m = NP.Meta(make_tables([("scene-0001", 5)]))
    s = m.samples_of_scene["tok-scene-0001"][2]
    xy, valid, _ = NP.gt_trajectory(m, s)
    assert valid.tolist() == [True, True, False, False, False, False]
    assert (xy[2:] == 0).all()


# --------------------------------------------------------------------------- #
# Occupancy builders on a parked car / a pedestrian                             #
# --------------------------------------------------------------------------- #
CAR_AHEAD = {"scene": "scene-0001", "cat": "vehicle.car", "wlh": (2.0, 4.0, 1.5),
             "xy_yaw": (12.0, 0.0, 0.0)}
PED_AHEAD = {"scene": "scene-0001", "cat": "human.pedestrian.adult", "wlh": (0.6, 0.6, 1.7),
             "xy_yaw": (8.0, 0.0, 0.0)}


def test_uniad_occupancy_rasterises_the_parked_car_and_fills_invalid_frames():
    m = NP.Meta(make_tables([("scene-0001", 10)], agents=[CAR_AHEAD]))
    ss = m.samples_of_scene["tok-scene-0001"]
    occ = NP.occupancy_uniad(m, ss[0])
    ys, xs = np.nonzero(occ[0])
    # car at LiDAR (0, 11.057), 4 m long along forward, 2 m wide: rows 118..126, cols 98..102
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (118, 126, 98, 102)
    last = NP.occupancy_uniad(m, ss[8])                          # 1 future sample left
    assert last[0].max() == 1 and (last[1:] == 255).all()


def test_the_gt_trajectory_collides_with_the_parked_car_at_the_same_steps_in_uniad_and_vad():
    m = NP.Meta(make_tables([("scene-0001", 10)], agents=[CAR_AHEAD]))
    s0 = m.samples_of_scene["tok-scene-0001"][0]
    xy, valid, _ = NP.gt_trajectory(m, s0)
    gt = xy[None]
    ku = NP.kernel_uniad(gt, gt, valid[None], NP.occupancy_uniad(m, s0)[None])
    kv = NP.kernel_vad(gt, gt, NP.occupancy_vad(m, s0)[None], np.ones(1, bool))
    assert ku.gt_box_col[0].tolist() == [0, 0, 0, 1, 1, 1]
    assert kv.gt_box_col[0].tolist() == [0, 0, 0, 1, 1, 1]
    assert ku.obj_box_col.sum() == 0 and kv.obj_box_col.sum() == 0   # GT arm: excluded


def test_pedestrians_count_in_vad_but_not_in_uniad():
    m = NP.Meta(make_tables([("scene-0001", 10)], agents=[PED_AHEAD]))
    s0 = m.samples_of_scene["tok-scene-0001"][0]
    assert NP.occupancy_uniad(m, s0)[:6].max() == 0
    assert NP.occupancy_vad(m, s0).max() == 1


def test_vad_agent_filters_range_and_lidar_points():
    far = dict(CAR_AHEAD, xy_yaw=(12.0, 20.0, 0.0))              # 20 m to the left: |x| >= 15
    empty = dict(CAR_AHEAD, lidar_pts=0)
    m1 = NP.Meta(make_tables([("scene-0001", 10)], agents=[far]))
    m2 = NP.Meta(make_tables([("scene-0001", 10)], agents=[empty]))
    s1 = m1.samples_of_scene["tok-scene-0001"][0]
    s2 = m2.samples_of_scene["tok-scene-0001"][0]
    assert NP.occupancy_vad(m1, s1).max() == 0
    assert NP.occupancy_vad(m2, s2).max() == 0
    assert NP.occupancy_uniad(m2, s2)[:6].max() == 1            # UniAD keeps 0-point boxes


def test_vad_category_index_audit_on_the_lidarseg_ordered_table():
    m = NP.Meta(make_tables([("scene-0001", 3)]))
    audit = NP.vad_category_index_audit(m)
    assert audit["n_categories"] == 32 and audit["indices_out_of_range"] == []
    assert all(n.startswith("human.") for n in audit["human_index_2_8"].values())
    assert all(n.startswith("vehicle.") for n in audit["vehicle_index_14_23"].values())
    assert audit["human_names_not_selected"] == [] and audit["vehicle_names_not_selected"] == ["vehicle.ego"]


# --------------------------------------------------------------------------- #
# SPEC §9 — the input adapter (CAM_FRONT pinhole -> cylindrical training frame)  #
# --------------------------------------------------------------------------- #
def test_geometry_stamp_matches_the_closed_form():
    from tanitad.data.calib import NUSCENES_CAM_FRONT_INTR_NOMINAL as K
    st = NP.geometry_stamp(K, NP.model_frame("wide"))
    assert (st["observed_px"], st["total_px"]) == (70737, 163840)
    assert st["observed_cols"] == [145, 488] and st["observed_rows_centre_col"] == [9, 225]
    assert abs(st["camera_hfov_deg"] - 64.5615) < 1e-3 and st["frame_hfov_deg"] == 120.0
    rc = NP.geometry_stamp(K, NP.model_frame("rig_clean"))
    assert (rc["observed_px"], rc["total_px"]) == (60399, 109824)


def test_rectify_is_a_ray_resample_with_an_honest_black_periphery():
    import torch
    from tanitad.data.calib import NUSCENES_CAM_FRONT_INTR_NOMINAL as K
    img = torch.full((1, 3, 900, 1600), 200, dtype=torch.uint8)
    out, st = NP.rectify_cam_front(img, K, NP.model_frame("wide"))
    assert tuple(out.shape) == (1, 3, 256, 640)                 # the frame, not a resize target
    o = out[0, 0].numpy()
    assert o[:, :140].max() == 0 and o[:, 495:].max() == 0      # unobserved azimuth: black
    # observed centre row intact. MEASURED 2026-09-19: calib.pinhole_rectify quantises with
    # `.to(torch.uint8)` — TRUNCATION — so float32 bilinear weights summing to 0.99999994 turn a
    # constant 200 into 199 at some pixels (a -1 LSB floor bias shared with the training frames).
    row = o[128, 150:480]
    assert row.min() >= 199 and row.max() == 200 and (row == 200).mean() > 0.9
    # calib stores the fraction as a float32 mean (1.2e-8 off); the COUNT is exact in the stamp
    assert abs(st["observed_frac_of_this_call"] - 70737 / 163840) < 1e-6
    assert st["observed_px"] == 70737


def test_history_exact_refuses_without_sweeps_and_nt_st_are_declared():
    t0 = 1_000_000_000
    keyframes = [(t0 - 500_000 * i, f"k{i}.jpg") for i in range(4)]          # 2 Hz only
    with pytest.raises(NP.RefusedInput):
        NP.history_slots(t0, keyframes, n_raw=9, construction="EXACT")
    nt = NP.history_slots(t0, keyframes, n_raw=9, construction="NT")
    assert nt["files"][-1] == "k0.jpg" and nt["max_abs_time_error_s"] == pytest.approx(0.2)
    st = NP.history_slots(t0, keyframes, n_raw=9, construction="ST")
    assert set(st["files"]) == {"k0.jpg"}
    sweeps = [(t0 - int(i * 1e6 / 12), f"s{i}.jpg") for i in range(12)]      # 12 Hz
    ex = NP.history_slots(t0, sweeps, n_raw=9, construction="EXACT")
    assert ex["max_abs_time_error_s"] <= 0.05


# --------------------------------------------------------------------------- #
# SPEC §10 — plan (our ego frame) -> LiDAR-origin trajectory                    #
# --------------------------------------------------------------------------- #
def test_straight_plan_converts_exactly():
    knots = np.array([[5.0 * t, 0.0] for t in NP.KNOT_T_S])
    xy, _ = NP.plan_to_lidar(knots, LIDAR_T, LIDAR_Q)
    assert np.allclose(xy, [[0.0, 2.5 * k] for k in range(1, 7)], atol=1e-9)


def test_arc_plan_matches_the_gt_within_1cm_and_scores_zero_l2_on_gt():
    knots = np.array([_ego_arc(t)[:2] for t in NP.KNOT_T_S])
    xy, _ = NP.plan_to_lidar(knots, LIDAR_T, LIDAR_Q)
    R, a = 20.0, A_LEVER
    want = np.array([(-(R * (1 - math.cos(0.125 * k)) + a * math.sin(0.125 * k)),
                      R * math.sin(0.125 * k) + a * math.cos(0.125 * k) - a) for k in range(1, 7)])
    assert np.abs(xy - want).max() < 0.01


# --------------------------------------------------------------------------- #
# End to end: the §1 run directory from synthetic mini_val metadata              #
# --------------------------------------------------------------------------- #
def _run(protocol, tmp):
    T = make_tables([("scene-0103", 10), ("scene-0916", 10)], agents=[
        dict(CAR_AHEAD, scene="scene-0103"), dict(PED_AHEAD, scene="scene-0916")])
    root = os.path.join(tmp, "nuscenes")
    _write(T, root, "v1.0-mini")
    out = os.path.join(tmp, "run_" + protocol)
    return NP.run_nuscenes_ol(root, protocol, "mini_val", ["GT", "STOP", "CV"], out)


def _validate_with_w1_schema(doc, name):
    try:
        from taniteval.bench import schema_check as SC           # W1's validator (read-only use)
    except Exception as e:                                       # noqa: BLE001
        pytest.skip(f"NOT RUN: W1 schema_check not importable ({type(e).__name__})")
    errs = SC.validate(doc, SC.load_schema(name))
    assert errs == [], errs[:6]


def test_end_to_end_uniad_protocol_literals():
    with tempfile.TemporaryDirectory() as tmp:
        out = _run("nuScenes_OL_L2_uniad", tmp)
        s = out["summary"]
        assert s["claim_bearing"] is False and out["bench_run"]["claim_bearing"] is False
        assert s["protocol"] == "nuScenes_OL_L2_uniad"
        gt = s["arms"]["GT"]["submetrics"]["L2_m"]
        assert gt == {"1s": 0.0, "2s": 0.0, "3s": 0.0, "avg_1_2_3s": 0.0}
        # STOP: per step 0.2*(t+1)*(9-t) over two 10-sample scenes = [1.8,3.2,4.2,4.8,5.0,4.8]
        stop = s["arms"]["STOP"]["submetrics"]["L2_m"]
        for h, want in zip(H, (3.2, 4.8, 4.8, 12.8 / 3)):
            assert stop[h] == pytest.approx(want, abs=1e-9), (h, stop[h])
        cv = s["arms"]["CV"]["submetrics"]["L2_m"]
        assert max(cv.values()) < 1e-9                           # straight at the true v0 == GT
        assert s["floors"] == ["STOP", "CV"]
        for f in ("bench_run.json", "summary.json", "scores/GT.csv", "artifacts/STOP.json",
                  "criteria/CV.txt"):
            assert os.path.isfile(os.path.join(out["run_dir"], f)), f
        art = json.load(open(os.path.join(out["run_dir"], "artifacts", "STOP.json"), encoding="utf-8"))
        assert art["claim_bearing"] is False
        assert art["benchmark"]["nuscenes"]["planning"]["gt_control"]["status"] == "OK"


def test_end_to_end_stp3_protocol_uses_vad_fut_valid_samples():
    with tempfile.TemporaryDirectory() as tmp:
        out = _run("nuScenes_OL_L2_stp3", tmp)
        s = out["summary"]
        assert s["arms"]["GT"]["headline"]["n"] == 8                # 2 scenes x (10 - 6)
        stop = s["arms"]["STOP"]["submetrics"]["L2_m"]
        for h, want in zip(H, (3.0, 5.0, 7.0, 5.0)):                # TemAvg of 2,4,..,12
            assert stop[h] == pytest.approx(want, abs=1e-9), (h, stop[h])


@pytest.mark.parametrize("protocol", ["nuScenes_OL_L2_uniad", "nuScenes_OL_L2_stp3"])
def test_run_files_validate_against_w1_schemas(protocol):
    with tempfile.TemporaryDirectory() as tmp:
        out = _run(protocol, tmp)
        _validate_with_w1_schema(out["bench_run"], "bench_run")
        _validate_with_w1_schema(out["summary"], "summary")


def _w1_cli():
    try:
        from taniteval.bench import cli as W1CLI, contract as W1C   # W1's code, read-only use
    except Exception as e:                                       # noqa: BLE001
        pytest.skip(f"NOT RUN: W1 suite not importable ({type(e).__name__}: {e})")
    return W1CLI, W1C


def _with_env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return old


def _restore_env(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.parametrize("protocol", ["nuScenes_OL_L2_uniad", "nuScenes_OL_L2_stp3"])
def test_plugin_through_the_w1_cli_writes_a_valid_run(protocol):
    W1CLI, W1C = _w1_cli()
    with tempfile.TemporaryDirectory() as tmp:
        T = make_tables([("scene-0103", 10), ("scene-0916", 10)], agents=[
            dict(CAR_AHEAD, scene="scene-0103")])
        root = os.path.join(tmp, "nuscenes")
        _write(T, root, "v1.0-mini")
        old = _with_env(TANITAD_NUSCENES_ROOT=root, TANITAD_NUSCENES_PROTOCOL=protocol)
        try:
            rc = W1CLI.main(["nuscenes_ol", "--ckpt", "none", "--split", "mini_val", "--device",
                             "cpu", "--results-root", os.path.join(tmp, "runs"), "--no-report"])
        finally:
            _restore_env(old)
        assert rc == 0
        base = os.path.join(tmp, "runs", "nuscenes_ol", "mini_val")
        (run_id,) = os.listdir(base)
        rd = os.path.join(base, run_id)
        br = json.load(open(os.path.join(rd, "bench_run.json"), encoding="utf-8"))
        sm = json.load(open(os.path.join(rd, "summary.json"), encoding="utf-8"))
        assert br["status"] == "COMPLETE" and br["claim_bearing"] is False and sm["claim_bearing"] is False
        assert br["protocol"] == protocol == sm["protocol"]
        assert W1C.validate_bench_run(br) == [] and W1C.validate_summary(sm) == []
        assert sorted(sm["arms"]) == ["CV", "GT", "STOP"] and sm["floors"] == ["STOP", "CV"]


def test_plugin_refuses_without_a_chosen_convention():
    W1CLI, _ = _w1_cli()
    with tempfile.TemporaryDirectory() as tmp:
        old = _with_env(TANITAD_NUSCENES_PROTOCOL=None, TANITAD_NUSCENES_ROOT=tmp)
        try:
            rc = W1CLI.main(["nuscenes_ol", "--ckpt", "none", "--split", "mini_val", "--device",
                             "cpu", "--results-root", os.path.join(tmp, "runs"), "--no-report"])
        finally:
            _restore_env(old)
        assert rc == 2                                            # REFUSED, never a default


def test_model_arm_plumbing_with_a_fake_model():
    """CAM_FRONT on disk -> history -> calib.pinhole_rectify -> stack_frames -> forward -> LiDAR
    frame. The fake model returns a straight 5 m/s plan and records what it was fed."""
    seen = {}

    def fake_forward(rows, decl):
        seen["rows"] = tuple(rows.shape)
        seen["decl"] = dict(decl)
        return np.array([[5.0 * t, 0.0] for t in NP.KNOT_T_S])

    with tempfile.TemporaryDirectory() as tmp:
        T = make_tables([("scene-0103", 8)])
        root = os.path.join(tmp, "nuscenes")
        _write(T, root, "v1.0-mini")
        m = NP.Meta.load(root, "v1.0-mini")
        from PIL import Image
        for sd in T["sample_data"]:
            if sd["filename"].startswith("samples/CAM_FRONT"):
                p = os.path.join(root, sd["filename"])
                os.makedirs(os.path.dirname(p), exist_ok=True)
                Image.new("RGB", (1600, 900), (90, 120, 150)).save(p, quality=95)
        with pytest.raises(NP.RefusedInput):
            NP.make_model_arm("A1_ego_cmd", nuscenes_root=root, forward=fake_forward, window_rows=4)
        spec = NP.make_model_arm("A2_vision_pure", nuscenes_root=root, forward=fake_forward,
                                 window_rows=4)
        s0 = m.samples_of_scene["tok-scene-0103"][0]
        xy = spec["plan_fn"](m, s0)
        assert seen["rows"] == (4, 9, 256, 640)                  # the trainer's own D-015 stack
        assert seen["decl"] == {"_arm": "A2_vision_pure", "_declared": []}
        assert np.allclose(xy, [[0.0, 2.5 * k] for k in range(1, 7)], atol=1e-9)
        geo = spec["geometry_fn"]()
        assert geo["observed_px"] == 70737 and geo["history"]["construction"] == "ST"
        spec3 = NP.make_model_arm("A3_ego_nocmd", nuscenes_root=root, forward=fake_forward,
                                  window_rows=4)
        spec3["plan_fn"](m, m.samples_of_scene["tok-scene-0103"][3])
        vx, vy = seen["decl"]["ego_velocity"]
        assert abs(vx - 4.0) < 1e-6 and abs(vy) < 1e-9          # 4 m/s straight, ego frame
        ev = NP.evaluate_arms(m, ["scene-0103"], "nuScenes_OL_L2_stp3", ["GT"],
                              model_arms={"A2_vision_pure": spec})
        art = NP.build_run_outputs(ev, split_name="mini_val", n_scenes=1, n_logs=1)
        a2 = art["arms"]["A2_vision_pure"]["artifact"]
        assert a2["input_geometry"]["observed_px"] == 70737 and a2["claim_bearing"] is False


def test_absent_metadata_refuses_with_the_human_steps():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(Exception) as ei:
            NP.run_nuscenes_ol(tmp, "nuScenes_OL_L2_uniad", "val", ["GT"], os.path.join(tmp, "r"))
        assert "Terms of Use" in str(ei.value) or "terms-of-use" in str(ei.value)
