#!/usr/bin/env python3
"""test_geodac.py -- validation of geodac.py (analytic identities, zero controls, and deliberate-regression arms that must go RED).

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_geodac.py          # prints  ZZTEST_PASS <n_checks> <n_mutants>
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q test_geodac.py

How the red arms work: `checks()` runs the whole battery against the module AS IMPORTED and returns {check_name: bool}.  A mutant is a
patch of ONE function/constant of `geodac` (a flipped y axis, a wrong resolution, ...); with it applied, the named checks MUST fail.
A battery that stays green under a mutant would be blind to that bug, which is the failure this file exists to rule out.
"""
from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PK / "refe"))
import geodac as G  # noqa: E402
import navsim_dac as ND  # noqa: E402  -- the repo's own (polygon-based) NAVSIM DAC helpers: numpy only at import time

GEOM = G.GT_GEOM
XC = GEOM.x0 + (np.arange(GEOM.nx) + 0.5) * GEOM.res_x
YC = GEOM.y0 + (np.arange(GEOM.ny) + 0.5) * GEOM.res_y
XX, YY = np.meshgrid(XC, YC, indexing="ij")                                     # [nx, ny], index [ix, iy]
CLIP = 12.7


def band(lo, hi):
    """Road = lo < y < hi, exact signed distance (positive inside), clipped like the store."""
    return np.clip(np.minimum(YY - lo, hi - YY), -CLIP, CLIP)


def xplane(x_lo=None, x_hi=None):
    """Road = x_lo < x < x_hi (either bound optional), exact signed distance."""
    f = np.full(XX.shape, CLIP)
    if x_lo is not None:
        f = np.minimum(f, XX - x_lo)
    if x_hi is not None:
        f = np.minimum(f, x_hi - XX)
    return np.clip(f, -CLIP, CLIP)


def line_poses(vx=10.0, vy=0.0, h=0.0):
    t = np.arange(1, 9) * 0.5
    return np.stack([vx * t, vy * t, np.full(8, h)], axis=-1)


def static(x, y, h):
    return np.array([[x, y, h]])


def brute_dac(field, geom, poses_dense, mode):
    """An INDEPENDENT loop implementation (pure python, math.floor) of the same rule, to compare the vectorised one against."""
    bad_samples, first = 0, -1
    for k, (x, y, h) in enumerate(poses_dense):
        cx, cy = x + G.REAR_AXLE_TO_CENTER * math.cos(h), y + G.REAR_AXLE_TO_CENTER * math.sin(h)
        outs = 0
        for lo, la in ((G.HALF_LENGTH, G.HALF_WIDTH), (G.HALF_LENGTH, -G.HALF_WIDTH), (-G.HALF_LENGTH, G.HALF_WIDTH), (-G.HALF_LENGTH, -G.HALF_WIDTH)):
            px = cx + lo * math.cos(h) - la * math.sin(h)
            py = cy + lo * math.sin(h) + la * math.cos(h)
            fx, fy = (px - geom.x0) / geom.res_x - 0.5, (py - geom.y0) / geom.res_y - 0.5
            if not (-0.5 <= fx < geom.nx - 0.5 and -0.5 <= fy < geom.ny - 0.5):
                continue                                                          # unknown -> benefit of the doubt
            if mode == "nearest":
                v = field[min(max(int(math.floor(fx + 0.5)), 0), geom.nx - 1), min(max(int(math.floor(fy + 0.5)), 0), geom.ny - 1)]
            else:
                i0, j0 = int(math.floor(fx)), int(math.floor(fy))
                tx, ty = fx - i0, fy - j0
                def at(i, j):
                    return field[min(max(i, 0), geom.nx - 1), min(max(j, 0), geom.ny - 1)]
                v = at(i0, j0) * (1 - tx) * (1 - ty) + at(i0 + 1, j0) * tx * (1 - ty) + at(i0, j0 + 1) * (1 - tx) * ty + at(i0 + 1, j0 + 1) * tx * ty
            outs += int(not v > 0)
        bad_samples += outs
        if outs and first < 0:
            first = k
    return bad_samples == 0, bad_samples / (4 * len(poses_dense)), first


def dense_loop(poses8):
    """Independent re-statement of the 0.1 s grid (python loops + np.interp), used only as an oracle."""
    full = np.concatenate([np.zeros((1, 3)), np.asarray(poses8, float)], 0)
    full[:, 2] = np.unwrap(full[:, 2])
    tq, tk = np.arange(41) * 0.1, np.arange(9) * 0.5
    return np.stack([np.interp(tq, tk, full[:, c]) for c in range(3)], -1)


# =============================================================================================================== the battery
def checks() -> dict:
    r = {}

    def ok(name, cond):
        r[name] = bool(cond)

    # ---- analytic identity 1: a straight path inside a straight road passes, with exactly zero outside share --------------
    road = band(-3.5, 3.5)
    a = G.geodac(road, GEOM, line_poses())
    ok("A1_straight_inside_passes", bool(a.passed) and a.frac_outside == 0.0 and a.first_bad_step == -1 and a.n_unknown == 0)
    ok("A1_margin_is_exact", abs(a.min_margin_m - (3.5 - G.HALF_WIDTH)) < 1e-9)                          # corner at |y| = 1.1485 -> margin 2.3515

    # ---- analytic identity 2: a path that drifts out of the road fails at the analytically known step ------------------------
    # y(t) = 0.8 t, heading 0: the left corners reach y = 3.5 when y(t) = 3.5 - 1.1485 = 2.3515  ->  t = 2.9394 -> first 0.1 s step k = 30
    b = G.geodac(road, GEOM, line_poses(vx=10.0, vy=0.8))
    ok("A2_drift_fails", not bool(b.passed))
    ok("A2_first_bad_step_analytic", int(b.first_bad_step) == 30)
    # exact outside share: corner samples with y_c + 1.1485 >= 3.5 : steps 30..40, 2 left corners each -> 11 * 2 / 164
    ok("A2_frac_outside_analytic", abs(b.frac_outside - 22 / 164) < 1e-12)

    # ---- analytic identity 3: a hole in the road ---------------------------------------------------------------------------------
    hole = np.where((XX >= 20.0) & (XX < 30.0), -2.0, road)
    h = G.geodac(hole, GEOM, line_poses(vx=10.0), mode="nearest")
    exp_pass, exp_frac, exp_first = brute_dac(hole, GEOM, dense_loop(line_poses(vx=10.0)), "nearest")
    ok("A3_hole_crossing_fails", (not bool(h.passed)) and (not exp_pass))
    ok("A3_hole_matches_independent_loop", abs(h.frac_outside - exp_frac) < 1e-12 and int(h.first_bad_step) == exp_first)
    hl = G.geodac(hole, GEOM, line_poses(vx=3.0), mode="nearest")                                        # 12 m by t = 4 s: never reaches x = 20 - 4.049
    ok("A3_path_stopping_short_of_hole_passes", bool(hl.passed))

    # ---- zero controls ---------------------------------------------------------------------------------------------------------------
    allin, allout = np.full(GEOM.shape, CLIP), np.full(GEOM.shape, -CLIP)
    rng = np.random.default_rng(7)
    P = np.stack([np.stack([np.sort(rng.uniform(0, 55, 8)), rng.uniform(-12, 12, 8), rng.uniform(-1.5, 1.5, 8)], -1) for _ in range(50)])
    z_in, z_out = G.geodac(allin, GEOM, P), G.geodac(allout, GEOM, P)
    ok("Z1_all_drivable_passes_everything_exactly", bool(z_in.passed.all()) and np.all(z_in.frac_outside == 0.0) and np.all(z_in.n_unknown == 0))
    ok("Z2_all_nondrivable_fails_everything_exactly", (not z_out.passed.any()) and np.all(z_out.frac_outside == 1.0))
    ok("Z3_zero_field_is_not_inside", not G.geodac(np.zeros(GEOM.shape), GEOM, P[:3]).passed.any())     # value 0 = boundary = not inside

    # ---- footprint dimensions and orientation (static poses, as_given horizon) -----------------------------------------------------
    # road occupies -1.5 < y < 4.5: a car pointing +y has y-extent [-1.127, 4.049] -> inside; pointing -y -> [-4.049, 1.127] -> outside
    side = band(-1.5, 4.5)
    ok("F1_pointing_left_fits", bool(G.geodac(side, GEOM, static(10, 0, math.pi / 2), horizon="as_given").passed))
    ok("F1_pointing_right_does_not", not bool(G.geodac(side, GEOM, static(10, 0, -math.pi / 2), horizon="as_given").passed))
    # narrow road |y| < 1.2: width 2.297 fits at heading 0 (half-width 1.1485), the 5.176 m length does not fit across it
    narrow = band(-1.2, 1.2)
    ok("F2_width_fits_narrow_road", bool(G.geodac(narrow, GEOM, static(10, 0, 0.0), horizon="as_given").passed))
    ok("F2_length_across_narrow_road_fails", not bool(G.geodac(narrow, GEOM, static(10, 0, math.pi / 2), horizon="as_given").passed))
    # front/rear overhang: road x < 10 ; rear axle at 5.9 -> front corners 9.949 (in), at 6.1 -> 10.149 (out, the two FRONT corners only)
    half = xplane(x_hi=10.0)
    ok("F3_front_overhang_inside", bool(G.geodac(half, GEOM, static(5.9, 0, 0.0), horizon="as_given").passed))
    f3 = G.geodac(half, GEOM, static(6.1, 0, 0.0), horizon="as_given")
    ok("F3_front_overhang_outside", (not bool(f3.passed)) and abs(f3.frac_outside - 0.5) < 1e-12)       # exactly 2 of 4 corners
    # rear overhang: road x > -1.0 ; rear axle at 0.2 -> rear corners -0.927 (in); at 0.0 -> -1.127 (out)
    rear = xplane(x_lo=-1.0)
    ok("F4_rear_overhang_inside", bool(G.geodac(rear, GEOM, static(0.2, 0, 0.0), horizon="as_given").passed))
    ok("F4_rear_overhang_outside", not bool(G.geodac(rear, GEOM, static(0.0, 0, 0.0), horizon="as_given").passed))

    # ---- axis conventions -------------------------------------------------------------------------------------------------------------
    left_only = band(-2.0, 9.0)                                 # road reaches 2 m to the RIGHT of the ego and 9 m to the LEFT (+y); the t=0 anchor sits at y=0
    ok("X1_left_road_passes_when_path_is_left", bool(G.geodac(left_only, GEOM, line_poses(vx=10, vy=0.0).copy() + np.array([0, 4.0, 0])).passed))
    ok("X1_left_road_fails_when_path_is_right", not bool(G.geodac(left_only, GEOM, line_poses(vx=10, vy=0.0).copy() + np.array([0, -4.0, 0])).passed))
    fwd_road = xplane(x_lo=-2.0, x_hi=45.0)                      # a road that starts 2 m behind the rear axle and ends 45 m ahead
    ok("X2_x_axis_is_forward", bool(G.geodac(fwd_road, GEOM, line_poses(vx=10.0)).passed))
    ahead_only = xplane(x_lo=20.0, x_hi=70.0)                   # a road that only starts 20 m ahead: a path that stays near the ego must fail
    ok("X2b_road_only_far_ahead_fails_a_near_path", not bool(G.geodac(ahead_only, GEOM, line_poses(vx=1.0)).passed))
    ok("X3_resolution_scale", bool(G.geodac(xplane(x_hi=10.0), GEOM, static(5.9, 0, 0.0), horizon="as_given", mode="nearest").passed)
       and not bool(G.geodac(xplane(x_hi=10.0), GEOM, static(6.1, 0, 0.0), horizon="as_given", mode="nearest").passed))
    # same road on the 2 m predicted geometry (4x coarser): the boundary at x = 10 lies on a cell EDGE, bilinear zero crossing is exact
    cx2 = G.PRED_GEOM.x0 + (np.arange(G.PRED_GEOM.nx) + 0.5) * G.PRED_GEOM.res_x
    f2 = np.repeat((10.0 - cx2)[:, None], G.PRED_GEOM.ny, axis=1)
    ok("X4_coarse_grid_boundary_exact", bool(G.geodac(f2, G.PRED_GEOM, static(5.9, 0, 0.0), horizon="as_given").passed)
       and not bool(G.geodac(f2, G.PRED_GEOM, static(6.1, 0, 0.0), horizon="as_given").passed))

    # ---- horizon: first and last steps count ---------------------------------------------------------------------------------------
    last_only = xplane(x_hi=44.0)               # front corners at t=4.0: 40 + 4.049 = 44.049 > 44 ; at t=3.9: 43.049
    lo = G.geodac(last_only, GEOM, line_poses(vx=10.0))
    ok("H1_violation_only_at_t_4s_is_caught", (not bool(lo.passed)) and int(lo.first_bad_step) == 40)
    first_only = xplane(x_lo=-0.5)              # rear corners at t=0: -1.127 (out); at t=0.1 (x=1.0): -0.127 (in)
    fo = G.geodac(first_only, GEOM, line_poses(vx=10.0))
    ok("H2_violation_only_at_t_0_is_caught", (not bool(fo.passed)) and int(fo.first_bad_step) == 0)
    ok("H3_dense_grid_has_41_steps", G.densify(line_poses()).shape == (41, 3))

    # ---- agreement with the repo's own helpers and with an independent loop ------------------------------------------------------------
    rngp = np.random.default_rng(11)
    ps = rngp.uniform(-1, 1, (200, 8, 3)) * np.array([6.0, 2.0, 2.2]) + np.array([np.linspace(2, 40, 8), np.zeros(8), np.zeros(8)]).T
    ps[:20, :, 2] += np.linspace(0, 3.4, 8)                                                              # headings that cross +-pi
    ok("E1_densify_equals_navsim_dac_at_10hz", float(np.abs(G.densify(ps) - ND.at_10hz(ps)).max()) < 1e-12)
    ok("E2_corners_equal_navsim_dac_corners_global", float(np.abs(G.corners(G.densify(ps)) - ND.corners_global(ND.at_10hz(ps), 0.0, 0.0, 0.0)).max()) < 1e-12)
    ok("E3_densify_equals_independent_loop", float(np.abs(G.densify(ps[3]) - dense_loop(ps[3])).max()) < 1e-12)
    ok("E4_footprint_constants_equal_navsim_dac", (G.HALF_LENGTH, G.HALF_WIDTH, G.REAR_AXLE_TO_CENTER) == (ND.HALF_LENGTH, ND.HALF_WIDTH, ND.REAR_AXLE_TO_CENTER))
    fld = rngp.normal(0, 3, GEOM.shape)
    agree = True
    for i in range(30):
        for mode in ("bilinear", "nearest"):
            g = G.geodac(fld, GEOM, ps[i], mode=mode)
            bp, bf, bfirst = brute_dac(fld, GEOM, dense_loop(ps[i]), mode)
            agree &= (bool(g.passed) == bp) and abs(g.frac_outside - bf) < 1e-12 and int(g.first_bad_step) == bfirst
    ok("E5_vectorised_equals_independent_loop_random_field", agree)
    batch = G.geodac(fld, GEOM, ps[:40])
    ok("E7_geodac_equals_geodac_corners_of_densified_corners", all(
        (lambda a, b: bool(a.passed) == bool(b.passed) and a.frac_outside == b.frac_outside and a.min_margin_m == b.min_margin_m)(
            G.geodac(fld, GEOM, ps[i]), G.geodac_corners(fld, GEOM, G.corners(G.densify(ps[i])))) for i in range(10)))
    ok("E6_batch_equals_one_by_one", all(bool(batch.passed[i]) == bool(G.geodac(fld, GEOM, ps[i]).passed) for i in range(40)))

    # ---- sampling exactness -------------------------------------------------------------------------------------------------------------
    lin = 0.3 * XX - 0.7 * YY + 2.0
    pts = rngp.uniform([-7.5, -31.5], [71.5, 31.5], (500, 2))
    ok("S1_bilinear_reproduces_a_linear_field", float(np.abs(G.sample_field(lin, GEOM, pts, "bilinear") - (0.3 * pts[:, 0] - 0.7 * pts[:, 1] + 2.0)).max()) < 1e-9)
    ok("S2_nearest_returns_containing_cell", float(np.abs(G.sample_field(lin, GEOM, np.array([[0.2, 0.2], [0.3, -0.1]]), "nearest") - np.array([lin[16, 64 + 0], lin[16, 63]])).max()) < 1e-12)
    far = G.geodac(road, GEOM, static(100.0, 0.0, 0.0), horizon="as_given")                              # footprint beyond x = 72
    ok("S3_out_of_extent_is_unknown_and_follows_policy", int(far.n_unknown) == 4 and bool(far.passed)
       and not bool(G.geodac(road, GEOM, static(100.0, 0.0, 0.0), horizon="as_given", unknown="outside").passed))

    # ---- the int8 store path: metres = LSB * 0.1, sign preserved -------------------------------------------------------------------------
    q = np.round(np.clip(road, -CLIP, CLIP) / G.GT_UNIT_M).astype(np.int8)
    a8 = G.geodac(G.gt_field_m(q), GEOM, line_poses())
    b8 = G.geodac(G.gt_field_m(q), GEOM, line_poses(vx=10.0, vy=0.8))
    ok("Q1_int8_store_roundtrip_same_verdicts", bool(a8.passed) and not bool(b8.passed) and int(b8.first_bad_step) == 30)
    return r


def _cc(fn):
    return ("cell_coords", fn)


def _g(p):
    return np.asarray(p, float)


MUTANTS = {
    # name: ([(patch target, replacement), ...], [detector checks: AT LEAST ONE must go red])
    "y_axis_flipped": ([_cc(lambda g, p: ((_g(p)[..., 0] - g.x0) / g.res_x - 0.5, (-_g(p)[..., 1] - g.y0) / g.res_y - 0.5))],
                       ["X1_left_road_passes_when_path_is_left", "X1_left_road_fails_when_path_is_right"]),
    "x_axis_flipped": ([_cc(lambda g, p: ((-_g(p)[..., 0] - g.x0) / g.res_x - 0.5, (_g(p)[..., 1] - g.y0) / g.res_y - 0.5))], ["X2_x_axis_is_forward"]),
    "xy_swapped": ([_cc(lambda g, p: ((_g(p)[..., 1] - g.y0) / g.res_y - 0.5, (_g(p)[..., 0] - g.x0) / g.res_x - 0.5))],
                   ["X1_left_road_passes_when_path_is_left", "X2_x_axis_is_forward", "A1_straight_inside_passes"]),
    "wrong_resolution_2x_coarse": ([_cc(lambda g, p: ((_g(p)[..., 0] - g.x0) / (2 * g.res_x) - 0.5, (_g(p)[..., 1] - g.y0) / (2 * g.res_y) - 0.5))],
                                   ["X3_resolution_scale", "F3_front_overhang_inside", "A1_straight_inside_passes"]),
    "wrong_resolution_half": ([_cc(lambda g, p: ((_g(p)[..., 0] - g.x0) / (0.5 * g.res_x) - 0.5, (_g(p)[..., 1] - g.y0) / (0.5 * g.res_y) - 0.5))],
                              ["X3_resolution_scale", "A1_straight_inside_passes", "F3_front_overhang_inside"]),
    "origin_off_by_half_a_cell": ([_cc(lambda g, p: ((_g(p)[..., 0] - g.x0) / g.res_x, (_g(p)[..., 1] - g.y0) / g.res_y))],
                                  ["E5_vectorised_equals_independent_loop_random_field", "S1_bilinear_reproduces_a_linear_field", "X3_resolution_scale"]),
    "sign_inverted": ([("is_inside", lambda v: v < 0.0)], ["A1_straight_inside_passes", "Z1_all_drivable_passes_everything_exactly"]),
    "boundary_counts_as_inside": ([("is_inside", lambda v: v >= 0.0)], ["Z3_zero_field_is_not_inside"]),
    "no_centre_shift": ([("REAR_AXLE_TO_CENTER", 0.0)], ["F1_pointing_left_fits", "F3_front_overhang_inside", "F4_rear_overhang_inside", "E4_footprint_constants_equal_navsim_dac"]),
    "length_and_width_swapped": ([("HALF_LENGTH", G.HALF_WIDTH), ("HALF_WIDTH", G.HALF_LENGTH)],
                                 ["F2_width_fits_narrow_road", "F3_front_overhang_inside", "E4_footprint_constants_equal_navsim_dac"]),
    "any_corner_is_enough": ([("_verdict", lambda ins, unk, unknown, min_corners=4: (ins | unk).sum(-1) >= 1)], ["F3_front_overhang_outside", "A2_frac_outside_analytic"]),
    "last_step_dropped": ([("DST_N", 40)], ["H1_violation_only_at_t_4s_is_caught", "H3_dense_grid_has_41_steps"]),
    "first_step_dropped": ([("densify", lambda p: G._orig_densify(p)[..., 1:, :])], ["H2_violation_only_at_t_0_is_caught", "H3_dense_grid_has_41_steps"]),
    "heading_ignored": ([("corners", lambda p: G._orig_corners(np.concatenate([_g(p)[..., :2], np.zeros_like(_g(p)[..., :1])], -1)))],
                        ["F1_pointing_left_fits", "F2_length_across_narrow_road_fails", "E2_corners_equal_navsim_dac_corners_global"]),
}


def run_mutants() -> dict:
    G._orig_densify, G._orig_corners = G.densify, G.corners
    res = {}
    for name, (patches, detectors) in MUTANTS.items():
        with contextlib.ExitStack() as st:
            for target, repl in patches:
                st.enter_context(mock.patch.object(G, target, repl))
            try:
                out = checks()
            except Exception as e:                                    # a mutant that crashes the battery is also RED (noted as such)
                out = {c: False for c in detectors}
                out["__crashed__"] = repr(e)[:120]
        red = sorted(k for k, v in out.items() if v is False)
        res[name] = {"detectors": detectors, "detected": any(out.get(c) is False for c in detectors), "detectors_red": [c for c in detectors if out.get(c) is False],
                     "n_red": len(red), "red": red[:12]}
    return res


def test_battery_is_green():
    r = checks()
    bad = [k for k, v in r.items() if not v]
    assert not bad, bad


def test_every_mutant_goes_red():
    res = run_mutants()
    miss = {k: v for k, v in res.items() if not v["detected"]}
    assert not miss, miss


def test_baseline_is_restored_after_mutants():
    run_mutants()
    assert all(checks().values())


if __name__ == "__main__":
    base = checks()
    bad = [k for k, v in base.items() if not v]
    mut = run_mutants()
    for k, v in mut.items():
        print(f"mutant {k:28s} detected={'OK ' if v['detected'] else 'FAIL'} n_red={v['n_red']:2d}  detectors_red={v['detectors_red']}")
    after = [k for k, v in checks().items() if not v]
    miss = [k for k, v in mut.items() if not v["detected"]]
    print(f"baseline: {len(base) - len(bad)}/{len(base)} green; failing={bad}; after-mutants failing={after}; mutants not red={miss}")
    if bad or after or miss:
        print("ZZTEST_FAIL")
        sys.exit(1)
    print(f"ZZTEST_PASS {len(base)} {len(mut)}")
