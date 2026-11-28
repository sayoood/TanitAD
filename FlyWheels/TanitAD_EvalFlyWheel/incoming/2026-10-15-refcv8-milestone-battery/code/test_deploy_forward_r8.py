"""Pins for deploy_forward_r8.py and tac_retime_r8.py (pure parts).  Expectations are LITERALS; each guard has a mutation that must go RED.
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -m pytest -q test_deploy_forward_r8.py
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_deploy_gains_r8 as BD  # noqa: E402
import deploy_forward_r8 as DF  # noqa: E402
import tac_retime_r8 as TAC  # noqa: E402

S2_INST = [0.5, 1.0, 1.5, 2.0]
S2_TRAIN = [0.9673903790217738, 0.6192916835387219, 0.5511341845460037, 0.530539399347386]
S6_INST = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
S6_TRAIN = [0.6393548580383511, 0.5969720822621201, 0.6546139969520486, 0.6994940271226132, 0.7222699235551424, 0.7453601677670005]


def fz():
    return {"surfaces": {"S2": {"instants_s": list(S2_INST), "all": {"RETIME": list(S2_TRAIN)}}, "S6": {"instants_s": list(S6_INST), "all": {"RETIME": list(S6_TRAIN)}}}}


def toy_result(table_by_arm: dict) -> dict:
    """A miniature forward result whose rows are a deterministic function of the arm's S2 gain table: identical tables -> identical rows."""
    def rows(t):
        return {"mean_ade_m": float(sum(t)), "minus_ha0_ext": {"delta": float(t[0] - t[1]), "lo": float(t[1] - t[2]), "hi": float(t[2] - t[3]), "separated": bool(t[0] > t[1])},
                "NR_minus_refcv7": {"ade_m": {"delta": float(np.mean(t))}}}
    return {"seeds": {str(s): {"surfaces": {"S2": {"sets": {st: {"arms": {a: rows(t) for a, t in table_by_arm.items()}} for st in DF.SETS}}}} for s in DF.SEEDS}}


# ------------------------------------------------------------------ exact_equal
def test_exact_equal_literals():
    assert DF.exact_equal({"a": [1.0, 2.0], "b": {"c": 0.5}}, {"a": [1.0, 2.0], "b": {"c": 0.5}}) == []
    assert DF.exact_equal({"a": float("nan")}, {"a": float("nan")}) == []
    assert DF.exact_equal({"a": 0.1}, {"a": 0.1 + 1e-18}) == []                      # 1e-18 is below half an ulp: the same double
    assert len(DF.exact_equal({"a": 0.1}, {"a": float(np.nextafter(0.1, 1.0))})) == 1  # ONE ulp is a mismatch
    assert len(DF.exact_equal({"a": 1}, {"b": 1})) == 2                              # key only on one side, both directions
    assert len(DF.exact_equal([1, 2], [1, 2, 3])) == 1


# ------------------------------------------------------------------ the S2 identity control
def test_s2_identity_passes_for_identical_tables():
    d = BD.build(fz())
    res = toy_result({DF.DEPLOY_ARM: d["surfaces"]["S2"]["RETIME_DEPLOY"], DF.TRAIN_ARM: S2_TRAIN})
    r = DF.s2_identity(res, res)
    assert r["pass"] is True and r["status"] == "COMPARED" and r["n_mismatches"] == 0 and r["n_leaves_compared"] == 24   # 2 seeds x 2 sets x 6 leaves


def test_s2_identity_mutation_perturbed_s2_knot_goes_red():
    """The deliberate regression: one deployed knot moved by +0.01 -> the S2 rows differ from the train arm's -> the control must FAIL."""
    d = DF.mutate_deploy(BD.build(fz()), 1, 0.01)
    assert d["surfaces"]["S2"]["RETIME_DEPLOY"] == [S2_TRAIN[0], S2_TRAIN[1] + 0.01, S2_TRAIN[2], S2_TRAIN[3]]
    res = toy_result({DF.DEPLOY_ARM: d["surfaces"]["S2"]["RETIME_DEPLOY"], DF.TRAIN_ARM: S2_TRAIN})
    r = DF.s2_identity(res, res)
    assert r["pass"] is False and r["n_mismatches"] > 0
    ctl = {"rt_forward": True, "const_g1_exactly_os": True, "known_values": True, "s2_identity_vs_forward_trainfit": r["pass"], "s2_identity_in_file": r["pass"], "gains_pinned": True}
    assert DF.aggregate_controls(ctl) is False


def test_s2_identity_absent_reference_is_not_a_pass():
    res = toy_result({DF.DEPLOY_ARM: S2_TRAIN, DF.TRAIN_ARM: S2_TRAIN})
    r = DF.s2_identity(res, None)
    assert r["status"] == "ABSENT" and r["pass"] is False and r["n_leaves_compared"] == 0


def test_s2_identity_ignores_s6_but_sees_a_missing_cell():
    res = toy_result({DF.DEPLOY_ARM: S2_TRAIN, DF.TRAIN_ARM: S2_TRAIN})
    res["seeds"]["0"]["surfaces"]["S6"] = {"sets": {}}                                 # S6 content is never compared
    assert DF.s2_identity(res, res)["pass"] is True
    broken = copy.deepcopy(res)
    del broken["seeds"]["1"]["surfaces"]["S2"]["sets"]["all"]["arms"][DF.DEPLOY_ARM]
    r = DF.s2_identity(broken, res)
    assert r["pass"] is False and any("missing" in m for m in r["mismatches"])


# ------------------------------------------------------------------ gains per arm
def test_surface_arm_gains_literals():
    d = BD.build(fz())
    s2 = DF.surface_arm_gains(d, fz(), "S2")
    s6 = DF.surface_arm_gains(d, fz(), "S6")
    assert s2[DF.DEPLOY_ARM].tolist() == S2_TRAIN and s2[DF.TRAIN_ARM].tolist() == S2_TRAIN
    assert s6[DF.DEPLOY_ARM].tolist() == [S2_TRAIN[1], S2_TRAIN[3]] + S6_TRAIN[2:]
    assert s6[DF.TRAIN_ARM].tolist() == S6_TRAIN
    assert s6["CONST_G1"].tolist() == [1.0] * 6 and s6["CONST_G0"].tolist() == [0.0] * 6 and s6["CONST_G05"].tolist() == [0.5] * 6
    assert s2["CONST_G1"].shape == (4,)


def test_aggregate_controls():
    ok = {k: True for k in ("rt_forward", "const_g1_exactly_os", "known_values", "s2_identity_vs_forward_trainfit", "s2_identity_in_file", "gains_pinned")}
    assert DF.aggregate_controls(ok) is True
    for k in ok:
        bad = dict(ok)
        bad[k] = False
        assert DF.aggregate_controls(bad) is False
        none = dict(ok)
        none[k] = None
        assert DF.aggregate_controls(none) is False
        gone = dict(ok)
        del gone[k]
        assert DF.aggregate_controls(gone) is False                                   # a missing control is not a pass


# ------------------------------------------------------------------ tac_retime pure parts
def cell(delta, lo, hi):
    return {"delta": delta, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0)}


def all_cells(**over):
    cells = {f"{s}|{k}|s{sd}": cell(0.001, -0.004, 0.006) for s in ("S2", "S6") for k in TAC.KEYS for sd in (0, 1)}
    cells.update(over)
    return cells


def test_tac_separated_worse_is_the_accuracy_rule():
    assert TAC.separated_worse(cell(-0.03, -0.05, -0.01)) is True          # lower accuracy, CI excludes 0 -> worse
    assert TAC.separated_worse(cell(0.03, 0.01, 0.05)) is False            # a separated IMPROVEMENT is not worse (the error-metric rule lo > 0 would call it worse)
    assert TAC.separated_worse(cell(-0.03, -0.07, 0.01)) is False          # lower point estimate, not separated
    assert TAC.separated_worse({"status": "REFUSED"}) is None and TAC.separated_worse(None) is None


def test_tac_clause5_literals():
    assert TAC.clause5(all_cells())["status"] == "PASS"
    assert TAC.clause5(all_cells(**{"S6|TAC_traj_lon_correct|s1": cell(-0.04, -0.08, -0.01)}))["status"] == "FAIL"
    c = all_cells()
    del c["S2|TAC_traj_lat_correct|s0"]
    assert TAC.clause5(c)["status"] == "INCOMPLETE"
    assert TAC.clause5({})["status"] == "INCOMPLETE"
    assert TAC.clause5(all_cells(**{"S2|TAC_traj_lat_correct|s0": {"status": "REFUSED"}}))["status"] == "INCOMPLETE"   # an interval-less cell is never counted clean
    # FAIL is not masked by an INCOMPLETE elsewhere in the same cells
    c = all_cells(**{"S6|TAC_traj_lon_correct|s1": cell(-0.04, -0.08, -0.01)})
    del c["S2|TAC_traj_lat_correct|s0"]
    assert TAC.clause5(c)["status"] == "FAIL"
    assert TAC.clause5(c)["separated_worse_cells"] == ["S6|TAC_traj_lon_correct|s1"]


def test_tac_labels_analytic_and_known_values():
    """Analytic: a straight constant-speed plan labels (keep, steady); a hard-braking plan labels brake_stop; GT vs GT is exactly 1.0 on both rows."""
    dt = 0.5
    straight = np.stack([np.stack([np.arange(1, 5) * 10.0 * dt, np.zeros(4)], -1)] * 3).astype(np.float32)     # 10 m/s, 3 windows
    brake = np.stack([np.stack([np.cumsum([10.0, 6.0, 2.0, 0.0]) * dt, np.zeros(4)], -1)] * 3).astype(np.float32)  # speeds 10, 6, 2, 0 m/s
    lat_s, lon_s = TAC.labels(straight, dt)
    lat_b, lon_b = TAC.labels(brake, dt)
    from tanitad.refs import refc_tactical as RC
    assert set(lat_s.tolist()) == {RC.LAT_LANE_KEEP} and set(lon_s.tolist()) == {RC.LON_STEADY}
    assert set(lat_b.tolist()) == {RC.LAT_LANE_KEEP} and set(lon_b.tolist()) == {RC.LON_BRAKE_STOP}
    t, _ = TAC.tac_direct(straight, straight, dt)
    assert all(np.all(v == 1.0) for v in t.values())
    t2, _ = TAC.tac_direct(brake, straight, dt)                                   # the plan brakes, the GT does not: lon wrong, lat right
    assert np.all(t2["TAC_traj_lat_correct"] == 1.0) and np.all(t2["TAC_traj_lon_correct"] == 0.0)


def test_tac_cell_eq():
    banked = {"delta": -0.0021, "lo": -0.0065, "hi": 0.0023, "n_windows": 4754, "n_episodes": 139, "separated": False, "n_dropped_nonfinite": 0, "mean_new": 0.5, "mean_ref": 0.6}
    assert TAC.cell_eq(dict(banked), banked)["exact"] is True
    mut = dict(banked)
    mut["hi"] = 0.0024
    r = TAC.cell_eq(mut, banked)
    assert r["exact"] is False and r["diff"] == {"hi": (0.0024, 0.0023)}
    mut2 = dict(banked)
    mut2["mean_ref"] = math.nextafter(0.6, 1.0)
    assert TAC.cell_eq(mut2, banked)["exact"] is False                              # the means are compared exactly too


# ------------------------------------------------------------------ the DEPLOY8 npz reader (the sibling's documented contract; it must REFUSE what it cannot align)
import json as _json  # noqa: E402


def _S(n=5, hz=(5, 10, 15, 20)):
    O = np.zeros((n, len(hz), 2), np.float32)
    O[:, :, 0] = np.arange(len(hz))[None, :] + 0.5
    return {"eid": np.arange(n) // 2, "clip": np.arange(n) // 2 + 100, "ws": np.arange(n) * 10, "hz": list(hz), "X": {"O": O}, "mask": np.array([True, True, False, True, True][:n])}


def _npz(S, surf="S2", tree="t", seed=0):
    n, T = S["X"]["O"].shape[:2]
    return {f"{surf}_pos": S["X"]["O"] + 1.0, f"{surf}_clip": np.asarray(S["clip"]), f"{surf}_ws": np.asarray(S["ws"]), f"{surf}_eid": np.asarray(S["eid"]),
            f"{surf}_os": S["X"]["O"].copy(), f"{surf}_compliant": np.asarray(S["mask"]), f"{surf}_instants_s": np.array([h / 10 for h in S["hz"]]),
            "meta_json": np.array(_json.dumps({"arm": "RETIME_DEPLOY8", "tree": tree, "inference_seed": seed}))}


def test_select_deploy8_accepts_the_documented_layout_and_returns_the_positions():
    S = _S()
    P, why = TAC.select_deploy8(_npz(S), "S2", S, "t", 0)
    assert why == "ok" and P.shape == (5, 4, 2) and P.dtype == np.float32 and P[0, :, 0].tolist() == [1.5, 2.5, 3.5, 4.5]
    S6 = _S(hz=(10, 20, 30, 40, 50, 60))
    P6, why6 = TAC.select_deploy8(_npz(S6, "S6"), "S6", S6, "t", 0)
    assert why6 == "ok" and P6.shape == (5, 6, 2)


def test_select_deploy8_refuses_what_it_cannot_align():
    S = _S()
    base = _npz(S)
    for key in ("S2_pos", "S2_clip", "S2_ws", "S2_eid", "S2_os", "S2_compliant", "S2_instants_s"):
        a = dict(base)
        del a[key]
        assert TAC.select_deploy8(a, "S2", S, "t", 0)[0] is None, key                   # every contract key is required
    a = dict(base)
    a["S2_ws"] = a["S2_ws"] + 1
    assert "NOT row-aligned" in TAC.select_deploy8(a, "S2", S, "t", 0)[1]                 # window ids differ
    a = dict(base)
    a["S2_os"] = a["S2_os"] + np.float32(1e-6)                                            # os not bit-equal
    assert TAC.select_deploy8(a, "S2", S, "t", 0)[0] is None
    a = dict(base)
    a["S2_compliant"] = ~a["S2_compliant"]
    assert "ceiling mask" in TAC.select_deploy8(a, "S2", S, "t", 0)[1]
    a = dict(base)
    a["S2_instants_s"] = np.array([0.5, 1.0, 1.5, 2.5])
    assert "instants" in TAC.select_deploy8(a, "S2", S, "t", 0)[1]
    a = dict(base)
    a["S2_pos"] = a["S2_pos"][:4]
    assert TAC.select_deploy8(a, "S2", S, "t", 0)[0] is None                              # row count differs
    assert TAC.select_deploy8(base, "S2", S, "t", 1)[0] is None                           # meta names inference seed 0, this run is seed 1
    assert TAC.select_deploy8(base, "S2", S, "other_tree", 0)[0] is None
    a = dict(base)
    a["meta_json"] = np.array(_json.dumps({"arm": "RETIME_DEPLOY_PERSURFACE", "tree": "t", "inference_seed": 0}))
    assert TAC.select_deploy8(a, "S2", S, "t", 0)[0] is None                              # a file of another arm is refused
