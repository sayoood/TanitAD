#!/usr/bin/env python3
"""Tests for refe/rl_terms_label.py (RL-S1 part 2).  Run: python refe/test_rl_terms_label.py  ->  ZZTEST_PASS <n> (prints the mutation table).
Needs shapely: when the interpreter lacks it the file re-launches itself in the navsim v1.1 venv (the one the labeller runs in).

Every value is HAND-COMPUTED on synthetic geometry (arithmetic next to it), never produced by running the module.  A check that cannot go RED is not a
check: each term has mutations that edit the module's SOURCE (the snippet must occur exactly once, else the test errors -- a vacuous mutation proves
nothing) and the suite must then report >= 1 failing check.

Pacifica: length 5.176 (half 2.588), width 2.297 (half 1.1485), rear axle -> centre 1.461 (front overhang 4.049 / rear 1.127 from the rear axle).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types

NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(HERE, "rl_terms_label.py")
PT_PATH = os.path.join(HERE, "navsim_pdm_targets.py")

try:
    import shapely                                                                  # noqa: F401
except ModuleNotFoundError:
    if os.environ.get("RLT_TEST_CHILD") == "1":
        raise
    sys.exit(subprocess.call([NV_PY, os.path.abspath(__file__), *sys.argv[1:]], env=dict(os.environ, RLT_TEST_CHILD="1", PYTHONIOENCODING="utf-8")))

import numpy as np                                                                  # noqa: E402
from shapely.geometry import box                                                    # noqa: E402

sys.path.insert(0, HERE)
import rl_reward as RW                                                              # noqa: E402

C_OFF = 1.461
HL, HW = 2.588, 1.1485
BIG = 1000.0


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


def states_xyh(ys, hs=0.0, x0=10.0, dx=0.5):
    """[M, 41, 11] executed-like states from rear-axle y per step ([M, 41] or [M]) and a heading."""
    ys = np.asarray(ys, dtype=np.float64)
    if ys.ndim == 1:
        ys = np.repeat(ys[:, None], 41, axis=1)
    s = np.zeros(ys.shape + (11,))
    s[..., 0] = x0 + dx * np.arange(41)[None, :]
    s[..., 1] = ys
    s[..., 2] = hs
    return s


def states_centre(cy, h, x0=100.0):
    """one proposal [1, 41, 11] whose bbox CENTRE sits at y = cy for every step, heading h (so the rear axle is 1.461 m behind)."""
    s = np.zeros((1, 41, 11))
    s[0, :, 0] = x0 - C_OFF * np.cos(h)
    s[0, :, 1] = cy - C_OFF * np.sin(h)
    s[0, :, 2] = h
    return s


class FakeType:
    def __init__(self, name):
        self.name = name


class FakeDam:
    def __init__(self, items):
        self.map_types = [FakeType(n) for n, _ in items]
        self._geometries = [g for _, g in items]


class FakePt:
    def __init__(self, x, y, h):
        self.x, self.y, self.heading = x, y, h


class FakeCenterline:
    def __init__(self, xyh):
        self.discrete_path = [FakePt(*p) for p in xyh]


def suite(M) -> list:
    """The checks against module `M`.  Returns the FAILING check names (empty = all green)."""
    bad: list = []

    def t(name, fn):
        try:
            ok = bool(fn())
        except Exception:                                                     # noqa: BLE001 -- an exception IS a failed check
            ok = False
        if not ok:
            bad.append(name)

    def raises(fn, exc=ValueError):
        try:
            fn()
        except exc:
            return True
        return False

    close = lambda a, b, tol=1e-9: bool(np.allclose(a, b, rtol=0, atol=tol))                  # noqa: E731
    road = [box(-BIG, -10.0, BIG, 10.0)]                                                     # drivable: |y| <= 10
    xs = np.arange(0.0, 401.0, 10.0)
    line0 = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)                     # centreline = the x axis, heading 0

    def geo(drv=None, inter=(), lanes=None, cl=line0, ctx=None):
        return M.SceneGeometry(road if drv is None else drv, list(inter), [box(-BIG, -50.0, BIG, 50.0)] if lanes is None else lanes, M.CenterLine(cl), ctx)

    # ---------------------------------------------------------------- footprint
    c0 = M.bbox_coords(states_xyh([0.0])[:, :1])                                              # rear axle (10, 0), heading 0
    t("bbox_centre_east", lambda: close(c0["centers"][0, 0], [10.0 + C_OFF, 0.0]))
    t("bbox_corners_east", lambda: close(np.sort(c0["corners"][0, 0, :, 0]), sorted([11.461 + 2.588] * 2 + [11.461 - 2.588] * 2))
                                  and close(np.sort(c0["corners"][0, 0, :, 1]), sorted([HW, HW, -HW, -HW])))
    c1 = M.bbox_coords(np.array([[[0.0, 0.0, np.pi / 2]]]))                                   # facing +y: front 4.049, rear -1.127, half width 1.1485
    t("bbox_corners_north", lambda: close(np.sort(c1["corners"][0, 0, :, 1]), sorted([4.049, 4.049, -1.127, -1.127]))
                                    and close(np.sort(c1["corners"][0, 0, :, 0]), sorted([HW, HW, -HW, -HW])))
    # ---------------------------------------------------------------- the gap g(t)
    s_in = states_xyh([0.0])
    g_in = M.corner_gap(M.bbox_coords(s_in)["corners"], road)
    t("gap_inside_is_exactly_zero", lambda: g_in.shape == (1, 41) and bool((g_in == 0.0).all()))
    s_out = states_xyh([10.3 - HW])                                                         # left corners at y = 10.3 -> 0.3 outside
    t("gap_pushed_out_0p3", lambda: close(M.corner_gap(M.bbox_coords(s_out)["corners"], road), 0.3, 1e-12))
    poly14 = [box(0.0, -10.0, 14.0, 10.0)]                                                  # front corners at x = 14.049 (0.049 out), rear 8.873 in
    t("gap_is_max_over_corners", lambda: close(M.corner_gap(M.bbox_coords(np.array([[[10.0, 0.0, 0.0]]]))["corners"], poly14), 0.049, 1e-12))
    t("gap_diagonal_3_4_5", lambda: close(M.point_gap(np.array([[13.0, 14.0]]), [box(0.0, 0.0, 10.0, 10.0)]), 5.0, 1e-12))
    ab = [box(0.0, -5.0, 10.0, 5.0), box(10.0, -5.0, 20.0, 5.0)]                            # a box straddling the seam is inside the UNION
    t("gap_union_of_polygons", lambda: float(M.corner_gap(M.bbox_coords(np.array([[[8.0, 0.0, 0.0]]]))["corners"], ab).max()) == 0.0)
    t("gap_nearest_of_disjoint", lambda: close(M.point_gap(np.array([[18.0, 5.0]]), [box(0.0, 0.0, 10.0, 10.0), box(20.0, 0.0, 30.0, 10.0)]), 2.0, 1e-12))
    t("gap_no_polygons_saturates", lambda: close(M.point_gap(np.array([[0.0, 0.0]]), []), M.NO_MAP_GAP) and M.NO_MAP_GAP >= 100.0)
    t("dac_from_gap", lambda: M.dac_from_gap(np.array([[0.0, 0.0], [0.0, 0.2]])).tolist() == [True, False])
    # ---------------------------------------------------------------- D end to end: rear-axle y per step -> g -> D (rl_reward.drivable_excess)
    e_last = np.concatenate([np.zeros(21), np.full(20, 10.3 - HW)])                         # E: 0.3 outside on the last 20 steps only
    ys = np.stack([np.zeros(41), np.full(41, 10.3 - HW), np.full(41, 11.0 - HW), e_last])
    out = M.terms_from_states(states_xyh(ys), geo(), "D")
    # D = mean_t min(g / 0.5, 1):  A 0 | B 0.3/0.5 = 0.6 | C min(1.0/0.5, 1) = 1 | E 20 * 0.6 / 41 = 0.292682927
    t("D_values", lambda: close(out["d"], [0.0, 0.6, 1.0, 20 * 0.6 / 41], 1e-9))
    t("gmax_values", lambda: close(out["gmax"], [0.0, 0.3, 1.0, 0.3], 1e-9))
    t("D_zero_iff_inside", lambda: out["d"][0] == 0.0 and (out["d"][1:] > 0).all())
    t("want_D_only", lambda: set(out) == {"d", "gmax"})
    t("executed_states_required", lambda: raises(lambda: M.terms_from_states(np.zeros((2, 8, 3)), geo(), "D")))
    t("executed_states_shape_ok", lambda: not raises(lambda: M.check_executed(np.zeros((2, 41, 11)))))
    # ---------------------------------------------------------------- K: centre offset, heading error, intersection switch, straddle
    one = box(-BIG, -1.75, BIG, 1.75)
    two = [box(-BIG, -3.5, BIG, 0.0), box(-BIG, 0.0, BIG, 3.5)]
    off = M.terms_from_states(states_centre(0.5, 0.0), geo(lanes=[one]), "K")
    # offset 0.5 -> k1 = (0.5 - 0.25) / 0.75 = 1/3;  heading error 0 -> k2 = 0;  inside one lane -> straddle 0;  K = 0.6 / 3 = 0.2
    t("K_offset_only", lambda: close(off["k"], 0.2, 1e-9) and close(off["k_raw"], 0.2, 1e-9))
    t("want_K_only", lambda: set(off) == {"k", "k_raw"})
    st = M.terms_from_states(states_centre(0.5, 0.0), geo(lanes=two), "K")
    # centre on y = 0.5 -> corners at y = -0.6485 (lane 0) and 1.6485 (lane 1): two lanes hold a corner, none holds all four -> straddle 1;  K = 0.2 + 0.2
    t("K_straddle", lambda: close(st["k"], 0.4, 1e-9))
    box_int = [box(-BIG, -50.0, BIG, 50.0)]
    ii = M.terms_from_states(states_centre(0.5, 0.0), geo(inter=box_int, lanes=two), "K")
    t("K_zero_inside_intersection", lambda: close(ii["k"], 0.0, 1e-12))
    lab_ctx = {"start_id": "a", "siblings": {}, "baseline": None, "human_lc": True, "route_needs": False, "none": False}
    sup = M.terms_from_states(states_centre(0.5, 0.0), geo(lanes=two, ctx=lab_ctx), "K")
    t("K_suppressed_when_label_fires", lambda: close(sup["k"], 0.0, 1e-12) and close(sup["k_raw"], 0.4, 1e-9))
    sup_r = M.terms_from_states(states_centre(0.5, 0.0), geo(lanes=two, ctx=dict(lab_ctx, human_lc=False, route_needs=True)), "K")
    t("K_suppressed_by_route_label", lambda: close(sup_r["k"], 0.0, 1e-12))
    nol = M.terms_from_states(states_centre(0.5, 0.0), geo(lanes=two, ctx=dict(lab_ctx, human_lc=False)), "K")
    t("K_not_suppressed_without_label", lambda: close(nol["k"], 0.4, 1e-9))
    h20 = np.radians(20.0)
    k20 = M.terms_from_states(states_centre(0.0, h20), geo(lanes=[one]), "K")
    # centre on the line, heading error 20 deg -> k2 = (20 - 10) / 20 = 0.5 (offset 0 < 1.5 m);  box width at 20 deg = 2*(2.588 sin20 + 1.1485 cos20) = 3.93 > 3.5 but only ONE lane holds a corner
    t("K_heading_error_20deg", lambda: close(k20["k"], 0.4 * 0.5, 1e-9))
    pth = np.stack([xs, np.zeros_like(xs), np.full_like(xs, 3.0)], axis=1)                  # path heading column 3.0 rad, ego heading -3.0 rad
    kw = M.terms_from_states(states_centre(0.0, -3.0), geo(lanes=[one], cl=pth), "K")
    # wrap(-3.0 - 3.0) = 2 pi - 6 = 0.283185 rad = 16.2254 deg -> k2 = (16.2254 - 10) / 20;  K = 0.4 k2
    t("K_heading_wraps", lambda: close(kw["k"], 0.4 * (np.degrees(2 * np.pi - 6.0) - 10.0) / 20.0, 1e-9))
    tr = M.terms_from_states(np.array(states_xyh([0.0], hs=0.5)), geo(lanes=[box(-BIG, -50.0, BIG, 50.0)]), "K", traces=True)
    # rear axle on y = 0, heading 0.5 rad -> the bbox centre is 1.461 sin(0.5) = 0.70045 off the x axis (the scorer measures the CENTRE)
    t("K_offset_is_of_the_centre", lambda: close(tr["offset"], C_OFF * np.sin(0.5), 1e-9))
    dup = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 1.0], [20.0, 0.0, 1.0]])   # a repeated vertex with a heading jump (PDM-Closed has these)
    dd = M.CenterLine(dup).trace(np.array([[10.0, 0.5]]), np.array([0.0]))
    t("centreline_duplicate_vertex_first_wins", lambda: close(dd["dpsi_deg"], 0.0, 1e-9) and close(dd["offset"], 0.5, 1e-12))
    t("centreline_signed_lateral_left_positive", lambda: close(M.CenterLine(line0).trace(np.array([[50.0, 0.7], [50.0, -0.4]]), np.zeros(2))["lat"], [0.7, -0.4]))

    def interp_matches_scipy():
        from scipy.interpolate import interp1d
        rng = np.random.default_rng(0)
        xp = np.cumsum(rng.uniform(0.0, 1.0, 60))
        ii = rng.integers(1, 59, 8)
        xp[ii] = xp[ii - 1]                                                                  # duplicated knots
        xp = np.maximum.accumulate(xp)
        fp = rng.normal(size=(60, 3))                                                        # 2-D, like PDMPath's (x, y, heading): scipy then takes its own
        q = np.concatenate([xp[1:-1], rng.uniform(xp[0], xp[-1], 200)])                      # searchsorted-left path (1-D y is delegated to np.interp)
        ref = interp1d(xp, fp, axis=0)(q)
        return all(np.allclose(M._interp_linear(xp, fp[:, c], q), ref[:, c], rtol=0, atol=1e-12, equal_nan=True) for c in range(3))
    t("interp_equals_scipy_interp1d", interp_matches_scipy)
    t("multiple_lanes_rule", lambda: M.multiple_lanes(M.bbox_coords(states_centre(0.5, 0.0))["corners"], two).all()
                                     and not M.multiple_lanes(M.bbox_coords(states_centre(1.75, 0.0))["corners"], two).any())
    overlap = [box(-BIG, -3.5, BIG, 3.5), box(-BIG, 0.0, BIG, 3.5)]                       # lane A holds all four corners, lane B (overlapping) holds two of them
    t("multiple_lanes_single_containing_lane_wins", lambda: not M.multiple_lanes(M.bbox_coords(states_centre(0.5, 0.0))["corners"], overlap).any())
    t("inside_any_within_semantics", lambda: M.inside_any(np.array([[1.0, 1.0], [9.0, 9.0], [20.0, 20.0]]), [box(0, 0, 10, 10)]).tolist() == [True, True, False])
    # ---------------------------------------------------------------- L: lc_executed, dither, label_fires
    l0, l1 = box(0.0, 0.0, 200.0, 3.5), box(0.0, 3.5, 200.0, 7.0)
    sib = {"a": l0, "b": l1}
    base = M._sh().linestrings(np.array([[0.0, 1.75], [200.0, 1.75]]))
    px = np.linspace(5.0, 40.0, 8)
    mk = lambda y: np.stack([px, np.asarray(y, dtype=np.float64)], axis=1)                    # noqa: E731
    P = np.stack([mk(np.full(8, 1.75)),                                                       # A stays                                -> no
                  mk(np.linspace(1.75, 5.25, 8)),                                             # B ends in lane b, 3.5 m off the baseline -> yes
                  mk(np.concatenate([np.full(7, 1.75), [3.7]])),                              # C ends in lane b but 1.95 m off (< 2.0)  -> no
                  mk(np.array([1.75, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 1.75])),                  # D visits lane b, returns to the baseline -> no
                  mk(np.concatenate([np.full(7, 1.75), [4.0]]))])                             # E ends in lane b, 2.25 m off (> 2.0)      -> yes
    t("lc_executed_cases", lambda: M.executed_lane_change(P, sib, "a", base).tolist() == [False, True, False, False, True])
    wide = {"a": box(0.0, 0.0, 200.0, 6.0)}                                                  # a wide START lane: 4 m off its baseline is still the same lane
    t("lc_same_lane_far_from_baseline_is_not_a_change", lambda: M.executed_lane_change(mk(np.linspace(1.0, 5.0, 8))[None], wide, "a",
                                                                                      M._sh().linestrings(np.array([[0.0, 1.0], [200.0, 1.0]]))).tolist() == [False])
    t("lc_executed_no_context_false", lambda: not M.executed_lane_change(P, {}, None, None).any())
    t("lc_threshold_is_2m", lambda: M.LC_OFFSET_M == 2.0)
    tt = np.arange(41) * 0.1
    ramp = np.linspace(0.0, 3.5, 41)                                                         # a lane change: monotone lateral drift
    wave = 0.5 * np.sin(np.pi * tt)                                                          # +-0.5 m, period 2 s: v_max = 1.57 m/s, 4 sign changes
    swerve = np.concatenate([np.linspace(0.0, 1.0, 21), np.linspace(1.0, 0.0, 20)])          # out and back once: ONE sign change
    jitter = 0.005 * np.sin(2.0 * np.pi * tt)                                                # 5 mm, 1 Hz: |v| <= 0.02 m/s, below the 0.15 dead band
    t("dither_cases", lambda: M.dither_flag(np.stack([ramp, wave, swerve, jitter])).tolist() == [0.0, 1.0, 0.0, 0.0])
    # route label truth table
    t("route_label_truth_table", lambda: [M.route_needs_lane_change(*a) for a in
                                          ((True, False, [True]), (True, True, [True]), (False, False, [True]), (True, False, [False]), (True, False, []))]
                                         == [True, False, False, False, False])
    # L end to end on executed states: lane a (y in [0, 3.5]) -> proposal 0 stays, proposal 1 drifts to y = 5.25 (lane b), proposal 2 swerves out and back
    ctx = {"start_id": "a", "siblings": sib, "baseline": base, "human_lc": False, "route_needs": False, "none": False}
    ys3 = np.stack([np.full(41, 1.75), np.linspace(1.75, 5.25, 41), 1.75 + swerve * 0.0 + np.where(tt < 2.0, 0.5 * tt, 1.0 - 0.5 * (tt - 2.0))])
    lg = geo(drv=[box(-BIG, -20.0, BIG, 20.0)], lanes=[l0, l1], ctx=ctx)
    lo = M.terms_from_states(states_xyh(ys3, x0=5.0), lg, "L")
    t("L_end_to_end", lambda: lo["lc_executed"].tolist() == [0, 1, 0] and lo["label_fires"] is False)
    t("L_dither_end_to_end", lambda: lo["dither"].tolist() == [0, 0, 0])
    t("label_fires_human", lambda: M.terms_from_states(states_xyh(ys3, x0=5.0), geo(ctx=dict(ctx, human_lc=True)), "L")["label_fires"] is True)
    t("label_fires_route", lambda: M.terms_from_states(states_xyh(ys3, x0=5.0), geo(ctx=dict(ctx, route_needs=True)), "L")["label_fires"] is True)
    t("label_fires_neither", lambda: M.terms_from_states(states_xyh(ys3, x0=5.0), geo(ctx=ctx), "L")["label_fires"] is False)
    t("L_without_context_is_zero", lambda: M.terms_from_states(states_xyh(ys3, x0=5.0), geo(ctx=None), "L")["lc_executed"].tolist() == [0, 0, 0])
    # ---------------------------------------------------------------- the polygons the scorer uses (v1.1 DAC set), selected by type name from the scene's map
    A, B, Cc, D, Ee, Ff = (box(i * 10.0, 0.0, i * 10.0 + 5.0, 5.0) for i in range(6))
    dam = FakeDam([("ROADBLOCK", A), ("LANE", B), ("LANE_CONNECTOR", Cc), ("INTERSECTION", D), ("CARPARK_AREA", Ee), ("ROADBLOCK_CONNECTOR", Ff)])
    sg = M.scene_geometry((None, None, None, FakeCenterline(line0), None, dam), None, "DKL")
    t("dac_polygon_set_v11", lambda: sg.drivable == [A, D, Ee] and sg.intersection == [D] and sg.lanes == [B, Cc])
    t("dac_layers_constant", lambda: tuple(M.DAC_LAYERS) == ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA"))
    # ---------------------------------------------------------------- the record and its contract with rl_reward
    full = M.terms_from_states(states_xyh(ys3, x0=5.0), lg, "DKL")
    rec = M.make_record(["log_a", "tok", 7, 0], 400, 2, full)
    rec = json.loads(json.dumps(rec))
    t("record_key_and_fields", lambda: rec["kind"] == "rl_terms" and rec["key"] == ["log_a", "tok", 7, 0] and rec["ckpt_step"] == 400
                                       and set(rec) >= {"d", "gmax", "k", "k_raw", "lc_executed", "label_fires", "dither"} and all(len(rec[n]) == 3 for n in ("d", "k", "lc_executed", "dither")))
    tg = np.tile(np.array([1.0, 1.0, 0.8, 1.0, 1.0, 1.0]), (3, 1))
    R = RW.reward(tg, RW.RewardCfg(use_k=True, use_l=True), d=rec["d"], k=rec["k"],
                  lane={"lc_executed": rec["lc_executed"], "label_fires": rec["label_fires"], "dither": rec["dither"]})
    t("record_feeds_rl_reward", lambda: R["R"].shape == (3,) and np.isfinite(R["R"]).all())
    t("terms_flag_validated", lambda: M.check_terms("DKL") == "DKL" and M.check_terms("D") == "D" and raises(lambda: M.check_terms("X"))
                                      and raises(lambda: M.check_terms("")) and raises(lambda: M.check_terms("DX")))
    # ---- the consumer's contract: scorer_finetune.py (ac4d3b8) globs rl_terms_*.jsonl and validates every field with _rl_terms_record.  Both are read from ITS source.
    sf = open(os.path.join(HERE, "scorer_finetune.py"), encoding="utf-8").read()
    import ast
    import fnmatch
    import re
    glob_pat = re.search(r'^RL_TERMS_GLOB = "([^"]+)"', sf, re.M).group(1)
    t("side_file_name_matches_consumer_glob", lambda: fnmatch.fnmatch(M.side_file_name("E:/x/onpolicy_r0_m5w0.jsonl"), glob_pat)
                                                  and M.side_file_name("E:/x/onpolicy_r0_m5w0.jsonl") == "rl_terms_onpolicy_r0_m5w0.jsonl")
    fn_src = next(ast.get_source_segment(sf, n) for n in ast.parse(sf).body if isinstance(n, ast.FunctionDef) and n.name == "_rl_terms_record")
    ns: dict = {"np": np}
    exec(compile(fn_src, "scorer_finetune._rl_terms_record", "exec"), ns)
    live = ns["_rl_terms_record"]
    jrec = json.loads(json.dumps(M.make_record(["l", "t", 3, 0], 400, 2, full)))
    got = live(jrec, 3, True, True, True)
    t("consumer_accepts_record", lambda: got is not None and set(got) == {"d", "k", "lc_executed", "dither", "label_fires"}
                                         and got["lc_executed"].dtype == bool and isinstance(got["label_fires"], bool))
    t("consumer_validator_is_live", lambda: live(dict(jrec, label_fires=1), 3, True, True, True) is None and live(dict(jrec, d=jrec["d"][:2]), 3, True, True, True) is None)
    t("consumer_accepts_D_only_record", lambda: live({"d": jrec["d"]}, 3, True, False, False) is not None)
    t("shard_known_value", lambda: M.shard_of("abc", 4) == 891568578 % 4)                     # CRC-32("abc") = 0x352441C2 = 891568578
    logs = [f"2021.0{i % 9}.{i}_veh-{i}" for i in range(300)]
    sh = [M.shard_of(x, 4) for x in logs]
    t("shard_partition", lambda: set(sh) == {0, 1, 2, 3} and sh == [M.shard_of(x, 4) for x in logs])
    # ---------------------------------------------------------------- the executed states are the scorer's: textual pin against navsim_pdm_targets.score_plans
    src_m, src_pt = M.__dict__.get("__SRC__", ""), open(PT_PATH, encoding="utf-8").read()
    norm = lambda s: " ".join(s.split())                                                      # noqa: E731
    pins = ['states = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]', "for p in np.asarray(plans8, dtype=np.float32):",
            'it = transform_trajectory(Trajectory(p, S["samp8"]), ego)', 'states.append(get_trajectory_as_array(it, S["samp"], ego.time_point))',
            'S["sim"].simulate_proposals(np.stack(states, 0), ego)']
    t("executed_states_pinned_to_score_plans", lambda: all(norm(p) in norm(src_pt) and norm(p) in norm(src_m) for p in pins))
    return bad


# (name, term, old snippet, new snippet) -- each snippet must occur EXACTLY once in rl_terms_label.py
MUTATIONS = [
    ("gap_min_over_corners", "D", "return point_gap(c, polys).max(axis=-1)", "return point_gap(c, polys).min(axis=-1)"),
    ("gap_centre_instead_of_corners", "D", 'g = corner_gap(co["corners"], geo.drivable)', 'g = corner_gap(co["centers"][..., None, :], geo.drivable)'),
    ("gap_union_replaced_by_first_polygon", "D", "tree = sh.STRtree(list(polys))\n    idx, dist", "tree = sh.STRtree(list(polys)[:1])\n    idx, dist"),
    ("executed_shape_check_dropped", "D", "or s.shape[1] != EXECUTED_STEPS ", ""),
    ("executed_steps_40", "D", "EXECUTED_STEPS = 41", "EXECUTED_STEPS = 40"),
    ("dac_strict_inequality", "D", "return (np.asarray(gap) <= tol).all(axis=-1)", "return (np.asarray(gap) < tol).all(axis=-1)"),
    ("no_map_gap_zero", "D", "NO_MAP_GAP = 1.0e3", "NO_MAP_GAP = 0.0"),
    ("D_is_raw_gap", "D", 'out["d"] = RW.drivable_excess(g)', 'out["d"] = g'),
    ("gmax_is_min", "D", 'out["gmax"] = g.max(axis=-1)', 'out["gmax"] = g.min(axis=-1)'),
    ("dac_layers_plus_lane", "D", 'DAC_LAYERS = ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA")',
     'DAC_LAYERS = ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA", "LANE")'),
    ("dac_layers_plus_roadblock_connector", "D", 'DAC_LAYERS = ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA")',
     'DAC_LAYERS = ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA", "ROADBLOCK_CONNECTOR")'),
    ("dac_layers_minus_carpark", "D", 'DAC_LAYERS = ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA")',
     'DAC_LAYERS = ("ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA")'),
    ("want_ignored_for_D", "D", 'if "D" in want:', "if True:"),
    ("K_intersection_mask_dropped", "K", 'in_int = inside_any(co["centers"], geo.intersection)', 'in_int = np.zeros(co["centers"].shape[:-1], dtype=bool)'),
    ("K_straddle_not_masked_in_intersection", "K", "strad_raw = (mult & ~in_int).mean(axis=-1)", "strad_raw = mult.mean(axis=-1)"),
    ("K_label_suppression_dropped", "K", 'out["k"] = RW.lane_consistency(ct["offset"], ct["dpsi_deg"], in_int | lab,',
     'out["k"] = RW.lane_consistency(ct["offset"], ct["dpsi_deg"], in_int,'),
    ("K_straddle_label_suppression_dropped", "K", "strad_raw * (0.0 if lab else 1.0)", "strad_raw"),
    ("K_offset_of_rear_axle", "K", 'ct = geo.centerline.trace(co["centers"], s[..., 2])', 'ct = geo.centerline.trace(s[..., :2], s[..., 2])'),
    ("K_heading_wrap_dropped", "K", "dpsi = np.degrees(wrap_angle(np.asarray(headings, dtype=np.float64).reshape(-1) - hp))",
     "dpsi = np.degrees(np.asarray(headings, dtype=np.float64).reshape(-1) - hp)"),
    ("K_np_interp_instead_of_interp1d", "K", "hp = _interp_linear(self.s, self.h, s)", "hp = np.interp(s, self.s, self.h)"),
    ("K_multiple_lanes_needs_no_single_lane_dropped", "K", " & (cnt != n).all(axis=1)", ""),
    ("K_multiple_lanes_needs_two", "K", "((cnt > 0).sum(axis=1) > 1)", "((cnt > 0).sum(axis=1) > 0)"),
    ("K_signed_lateral_flipped", "K", "lat = -(c.reshape(-1, 2)[:, 0] - xp) * np.sin(hp) + (c.reshape(-1, 2)[:, 1] - yp) * np.cos(hp)",
     "lat = (c.reshape(-1, 2)[:, 0] - xp) * np.sin(hp) - (c.reshape(-1, 2)[:, 1] - yp) * np.cos(hp)"),
    ("L_offset_threshold_0", "L", "LC_OFFSET_M = 2.0", "LC_OFFSET_M = 0.0"),
    ("L_start_lane_counts_as_change", "L", "if ids[ln] != start_id:", "if True:"),
    ("L_offset_read_at_first_pose", "L", "off_end = sh.distance(sh.points(p[:, -1, :]), baseline)", "off_end = sh.distance(sh.points(p[:, 0, :]), baseline)"),
    ("L_label_or_became_and", "L", 'lab = bool(geo.lane_ctx and (geo.lane_ctx["human_lc"] or geo.lane_ctx["route_needs"]))',
     'lab = bool(geo.lane_ctx and (geo.lane_ctx["human_lc"] and geo.lane_ctx["route_needs"]))'),
    ("L_dither_needs_one_change", "L", "DITHER_MIN_CHANGES = 2", "DITHER_MIN_CHANGES = 1"),
    ("L_dither_deadband_0", "L", "DITHER_V_DEAD = 0.15", "DITHER_V_DEAD = 0.0"),
    ("L_route_label_ignores_start_lane", "L", "(not start_reaches) and ", ""),
    ("L_no_context_still_changes", "L", "if lc else np.zeros(len(s), dtype=int))", "if lc else np.ones(len(s), dtype=int))"),
    ("W_prefix_not_the_consumers", "writer", 'RL_TERMS_PREFIX = "rl_terms_"', 'RL_TERMS_PREFIX = "rlt_"'),
    ("W_label_fires_not_a_bool", "writer", 'rec["label_fires"] = bool(t["label_fires"])', 'rec["label_fires"] = int(t["label_fires"])'),
    ("W_key_truncated", "writer", '"key": list(key),', '"key": list(key)[:3],'),
    ("W_shard_off_by_one", "writer", "% n\n", "% (n + 1)\n"),
    ("W_terms_subset_unchecked", "writer", "if not terms or not set(terms) <= set(\"DKL\"):", "if False:"),
    ("W_executed_states_float64", "reach", "dtype=np.float32", "dtype=np.float64"),
]


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    n = 0
    base = load(src, "rl_terms_label_real")
    base.__SRC__ = src
    real = suite(base)
    assert not real, f"the unmutated module FAILS its own checks: {real}"
    n += 1
    print(f"{'mutation':50s} {'term':7s} caught-by (failing checks)")
    for i, (name, term, old, new) in enumerate(MUTATIONS):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        msrc = src.replace(old, new)
        try:
            mod = load(msrc, f"rl_terms_label_mut{i}")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        mod.__SRC__ = msrc
        try:
            failing = suite(mod)
        except Exception as e:                                      # noqa: BLE001 -- the mutated module breaks the suite's own setup: RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:60]}"]
        assert failing, f"mutation {name} ({term}) stayed GREEN: the suite cannot see it"
        print(f"{name:50s} {term:7s} {', '.join(failing[:3])}")
        n += 1
    assert {t for _, t, _, _ in MUTATIONS} >= {"D", "K", "L", "writer", "reach"}
    n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations + coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
