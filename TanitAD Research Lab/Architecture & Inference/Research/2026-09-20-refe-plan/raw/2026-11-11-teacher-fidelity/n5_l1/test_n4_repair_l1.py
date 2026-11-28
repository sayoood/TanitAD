"""Tests for n4_repair_l1 (L1 two-sided lateral repair). NAVSIM venv python, CPU, BLAS threads 1 (n5_env).
  python -m pytest -q test_n4_repair_l1.py            (synthetic: seconds; real navtrain instances: ~1 min, marked `real`)
Synthetic tests use a perfect-tracking fake context over a hand-built two-sided corridor (the feasible answer is known by construction).
Real tests use the banked navtrain failing label instances in fixture_real_instances.json (nuPlan DB rebuild, builder scorer)."""
import json
import os

import n5_env  # noqa: F401
import numpy as np
import pytest
from shapely.geometry import box

import n4_repair as RP
import n4_repair_l1 as L1
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

VP = get_pacifica_parameters()
HALF_W = VP.width / 2.0
BASE_OK = {"dac": 1.0, "nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 1.0, "pdms": 1.0}
BASE0 = {"dac": 0.0, "nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 1.0, "pdms": 0.0}


class _RA:
    heading = 0.0


class _Ego:
    rear_axle = _RA()


class FakeCx:
    """perfect tracker: the executed path IS the plan (linear in between); DAC = all 4 corners inside `drivable`; labels(t) = identity"""

    def __init__(self, drivable, track=1.0):
        self.drivable, self.vp, self.ego, self.track = drivable, VP, _Ego(), track
        self.n_labels = 0

    def labels(self, t):
        self.n_labels += 1
        return {"R": np.array(t, float)}

    def _sim(self, plan):
        t = np.concatenate([[0.0], 0.2 * np.arange(1, 21)])
        P = np.vstack([np.zeros((1, 3)), np.asarray(plan, float)])
        S = np.zeros((41, 11))
        tk = 0.1 * np.arange(41)
        for c in range(3):
            S[:, c] = np.interp(tk, t, P[:, c] * (self.track if c == 1 else 1.0))
        return S

    def ext_dist(self, S):
        from shapely.geometry import Point
        from navsim_pdm.pdm_array_representation import state_array_to_coords_array
        c = state_array_to_coords_array(S[None], VP)[0]
        return np.array([max(0.0 if self.drivable.contains(Point(*p)) else self.drivable.distance(Point(*p)) for p in c[k, :4]) for k in range(41)])

    def score(self, plans):
        res, sims = [], []
        for p in plans:
            S = self._sim(p)
            r = dict(BASE_OK)
            r["dac"] = 1.0 if self.ext_dist(S).max() <= 0 else 0.0
            res.append(r)
            sims.append(S)
        return res, sims


def teacher(y_of_t):
    t = 0.2 * np.arange(1, 21)
    return np.stack([8.0 * t, y_of_t(t), np.zeros(20)], -1)


def corridor(width_free_early, width_free_late):
    """two-sided corridor, left and right walls; the free width is VEHICLE width + the given slack (m). Early (x < 16 m) and late (x >= 16 m)."""
    e, l = HALF_W + width_free_early / 2, HALF_W + width_free_late / 2
    return box(-5, -e, 16, e).union(box(16, -l, 60, l))


def test_two_sided_synthetic_l1_certifies_known_feasible_answer():
    # slack 0.50 m early, 0.34 m late: feasible lateral interval (eroded 0.05 m) is 0.40 wide early, 0.24 late; teacher drifts out left then right
    cx = FakeCx(corridor(0.50, 0.34))
    t20 = teacher(lambda t: np.where(t < 2.0, 0.30, -0.16))
    assert cx.score([cx.labels(t20)["R"]])[0][0]["dac"] == 0.0
    info = {}
    r = L1.make_l1()(cx, t20, BASE_OK, info=info)
    assert r is not None and r["m"]["dac"] == 1.0 and info["status"] in ("CERTIFIED",)
    assert RP.certified(cx.score([r["plan"]])[0][0], BASE_OK)


def test_zero_correction_is_bit_identical_and_passing_label_returned_unchanged():
    cx = FakeCx(corridor(1.0, 1.0))
    t20 = teacher(lambda t: 0.0 * t)
    info = {}
    r = L1.make_l1()(cx, t20, BASE_OK, info=info)
    assert info["status"] == "PASSED_AT_0" and r["param"]["iters"] == 0
    assert np.array_equal(r["plan"], t20)                    # bit-identical to the base label (identity labeller)
    assert cx.n_labels == 1


def test_zero_ramp_leaves_target_unchanged():
    z = L1.taut_ramp(-np.ones(20), np.ones(20))
    assert np.array_equal(z, np.zeros(20))


def test_sign_flipped_regression_arm_does_not_certify_synthetic():
    cx = FakeCx(corridor(0.50, 0.34))
    t20 = teacher(lambda t: np.where(t < 2.0, 0.30, -0.16))
    assert L1.make_l1(sign=-1.0)(cx, t20, BASE_OK) is None
    assert L1.make_l1(sign=+1.0)(cx, t20, BASE_OK) is not None


def test_infeasible_flag_when_corridor_narrower_than_footprint_plus_margin():
    cx = FakeCx(corridor(0.06, 0.06))                         # slack 0.06 < 2 x 0.05 margin; teacher already out
    t20 = teacher(lambda t: 0.2 + 0 * t)
    info = {}
    # fallback margin 0.005 -> feasible (slack 0.06 > 0.01); so the flag needs a corridor that is still empty: slack < 0.01
    cx2 = FakeCx(corridor(0.005, 0.005))
    r = L1.make_l1()(cx2, t20, BASE_OK, info=info)
    assert r is None and info["status"] == "INFEASIBLE" and info["iters"][0]["infeasible"]
    info2 = {}
    L1.make_l1()(cx, t20, BASE_OK, info=info2)
    assert info2["iters"][0]["relaxed_steps"], "a corridor between the margins must be solved at the fallback margin and say so"


def test_reachability_parameters_are_the_ones_the_built_object_uses():
    fn = L1.make_l1(margin=0.07, iters=3, centre_m=0.02, damp=0.25, damp_cum=True, sign=-1.0, cert="dac")
    rb = L1.readback(fn)
    assert (rb["margin"], rb["iters"], rb["centre_m"], rb["damp"], rb["damp_cum"], rb["sign"], rb["cert"]) == (0.07, 3, 0.02, 0.25, True, -1.0, "dac")
    assert rb["target"].endswith("repair_l1") and L1.readback(L1.make_l1())["margin"] == RP.MARGIN and L1.readback(L1.make_l1())["iters"] == RP.CAP
    # behavioural: margin and iters change what the object DOES, not only what it reads back
    cx = FakeCx(corridor(0.50, 0.34))
    t20 = teacher(lambda t: np.where(t < 2.0, 0.30, -0.16))
    i_def, i_big = {}, {}
    L1.make_l1()(cx, t20, BASE_OK, info=i_def)
    L1.make_l1(margin=0.30)(cx, t20, BASE_OK, info=i_big)                # margin 0.30 > half the slack: solved only through the fallback margin
    assert not i_def["iters"][0]["relaxed_steps"] and i_big["iters"][0]["relaxed_steps"]
    assert L1.make_l1(iters=0)(cx, t20, BASE_OK) is None                  # iters is the cap


def test_certificate_blocked_when_inside_but_worse():
    cx = FakeCx(corridor(1.0, 1.0))
    t20 = teacher(lambda t: 0.0 * t)
    worse = dict(BASE_OK, nc=0.0)

    class W(FakeCx):
        def score(self, plans):
            res, s = super().score(plans)
            return [dict(r, nc=0.0) for r in res], s
    info = {}
    assert L1.make_l1()(W(cx.drivable), t20, BASE_OK, info=info) is None and info["status"] == "CERT_BLOCKED"
    assert worse["nc"] < BASE_OK["nc"]


# ---------------------------------------------------------------- real navtrain instances (builder scorer, nuPlan DB rebuild)
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture_real_instances.json")


def _real_ctx(inst):
    import n4_geom as G
    for lg, tk, cx in G.contexts({inst["log"]: [inst["token"]]}):
        return cx


@pytest.mark.real
def test_real_instance_l1_certifies_and_regression_arm_does_not_and_zero_shift_is_base_label():
    fx = json.load(open(FIX, encoding="utf-8"))
    for key in ("easy", "multi_iter"):
        inst = fx[key]
        cx = _real_ctx(inst)
        t20 = np.asarray(inst["t20"], float)
        lab = cx.labels(t20)["R"]
        base = cx.score([lab])[0][0]
        assert base["dac"] == 0.0                                          # a REAL failing instance
        info = {}
        r = L1.make_l1()(cx, t20, base, info=info)
        assert r is not None and r["m"]["dac"] == 1.0 and RP.certified(cx.score([r["plan"]])[0][0], base)
        assert L1.make_l1(sign=-1.0)(cx, t20, base) is None                # the sign-flipped regression arm must NOT certify
        r2 = L1.make_l1()(cx, t20, base)                                   # determinism
        assert np.array_equal(r["plan"], r2["plan"])
        zi = {}
        z = L1.repair_l1(cx, t20, base, cert=lambda c, b: True, info=zi)    # zero correction (certificate accepts iteration 0) -> returned == base label
        assert zi["status"] == "PASSED_AT_0" and np.array_equal(z["plan"], lab)
