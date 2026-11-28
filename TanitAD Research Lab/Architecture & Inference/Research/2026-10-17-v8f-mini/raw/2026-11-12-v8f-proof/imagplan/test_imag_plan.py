"""test_imag_plan.py -- IMAGPLAN (card PREREG_V8F_PROOF_ADD40): the candidate rule, the action conversion, the wrapper and its controls, on synthetic worlds and tiny models (CPU).

Run:  CUDA_VISIBLE_DEVICES=-1 python -m pytest test_imag_plan.py -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/v8proof/imagplan/work/pt/run   (the parent of --basetemp must exist)
Env IMAGPLAN_SRC_DIR = a folder whose imag_plan.py is used instead of this folder's (the mutation runner, run_mutations.py).
Tier T1-kin (kinematic rollout on the model's own actions, logged frames); synthetic worlds only: no number here is a result about a real arm.
"""
from __future__ import annotations

import math
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("DISTRULE_TREE", "F:/TanitAD-scratch/v8proof/moved_from_c/leadsup/tree_patch")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("IMAGPLAN_SRC_DIR", HERE)
for _p in (HERE, os.path.abspath(os.path.join(HERE, "..", "distrule")), os.path.abspath(os.path.join(HERE, "..", "imaghead"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, SRC)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import imag_plan as IP  # noqa: E402
import v8f_t1_eval_imagplan as WI  # noqa: E402
import v8f_t1_eval_rule as W  # noqa: E402
import dist_rule as R  # noqa: E402
from test_dist_rule import T, world, tiny, _lead_join, _env, _ds, stub_planner, arrays_equal, TREE, PINNED_SHA  # noqa: E402,F401  (fixtures and helpers of the DISTRULE tests, read only)

assert os.path.abspath(IP.__file__).startswith(os.path.abspath(SRC)), (IP.__file__, SRC)
DT = 0.2


# ============================================================================================ an independent scalar reference of the card (no code shared with imag_plan.py)
def ref_choose(plan, v0, g_src, a_off, v_L, cset, floor="card", fallback="zero", const_gap=None):
    """-> (c, safe list, a_c at knots 5 and 10 per candidate, gap per candidate) for ONE window, written from the card text with plain loops."""
    pts = [(0.0, 0.0)] + [tuple(map(float, p)) for p in plan]
    aP = [0.0]
    for i in range(1, len(pts)):
        aP.append(aP[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    v0 = max(v0, 0.0)

    def aB(i):
        t = min(i * DT, v0 / 6.0)
        return v0 * t - 0.5 * 6.0 * t * t

    cs = sorted(set(cset), reverse=True)
    if fallback == "zero" and 0.0 not in cs:
        cs.append(0.0)
    res, safes, gaps, aa = {}, [], [], []
    for c in cs:
        def a(i, c=c):
            x = max(c * aP[i], aB(i))
            return min(aP[i], x) if floor == "cap-only" else x
        g, ok = [], True
        for i, h in ((5, 1.0), (10, 2.0)):
            gap = const_gap if const_gap is not None else g_src + a_off + v_L * h - a(i)
            v = (a(i) - a(i - 1)) / DT
            g.append(gap)
            ok = ok and gap >= max(3.0, 1.2 * v)
        res[c] = ok
        safes.append(ok)
        gaps.append(g)
        aa.append((a(5), a(10)))
    safe_cs = [c for c in cs if res[c]]
    chosen = max(safe_cs) if safe_cs else (0.0 if fallback == "zero" else None)
    return chosen, safes, aa, gaps


def straight(v, n=30, y=0.0, a=0.0):
    t = np.arange(1, n + 1) * DT
    x = v * t + 0.5 * a * t * t
    return np.stack([x, np.full(n, y)], 1).astype(np.float32)


def profile(v, a=0.0, n=30):
    """A straight plan whose speed starts at v and changes by a m/s2 (never reversing): step i covers the speed at its midpoint."""
    sp = np.maximum(v + a * (np.arange(n) + 0.5) * DT, 0.0)
    return np.stack([np.cumsum(sp * DT), np.zeros(n)], 1).astype(np.float32)


def rule(plan, v0, g_src, a_off=0.0, v_L=0.0, source="cand-cv", active=True, **kw):
    plan = np.asarray(plan)[None] if np.asarray(plan).ndim == 2 else np.asarray(plan)
    n = plan.shape[0]
    f = lambda x: np.broadcast_to(np.asarray(x, np.float64), (n,)).copy()      # noqa: E731
    return IP.imag_plan_rule(plan, f(v0), np.broadcast_to(np.asarray(active, bool), (n,)).copy(), f(g_src), f(a_off), f(v_L), source, **kw)


def arc_at(plan_row, i):
    p = np.vstack([np.zeros((1, 2)), np.asarray(plan_row, np.float64)])
    return float(np.hypot(*np.diff(p[:i + 1], axis=0).T).sum())


# ============================================================================================ constants, a_B, the candidate progress
def test_constants_and_the_card_candidate_set():
    assert (IP.D, IP.TAU, IP.B_MAX) == (3.0, 1.2, 6.0) and IP.KNOT_H == (5, 10) and IP.H_S == (1.0, 2.0)
    assert IP.CAND_CARD == (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
    assert IP.parse_cand_set("0.5,1.0,0.5") == (1.0, 0.5)
    with pytest.raises(ValueError):
        IP.parse_cand_set("1.5")
    assert IP.effective_set((1.0,), "zero") == (1.0, 0.0) and IP.effective_set((1.0,), "none") == (1.0,)


def test_braking_progress_hand_values():
    aB = IP.braking_progress(np.array([10.0, 0.0, -3.0]))
    assert aB[0, 1] == pytest.approx(10 * 0.2 - 0.5 * 6 * 0.04) and aB[0, 5] == pytest.approx(7.0) and aB[0, 30] == pytest.approx(100.0 / 12.0)
    assert (aB[1:] == 0.0).all()                                              # standing or reversing: no progress
    assert (np.diff(aB[0]) >= 0).all() and aB[0, 0] == 0.0


def test_braking_floor_c0_equals_a_B():
    for v0 in (0.0, 3.3, 10.0, 25.0):
        plan = straight(max(v0, 8.0))[None]
        aP, paths = IP.plan_arcs(plan)
        aB = IP.braking_progress(np.array([v0]))
        for floor in IP.FLOORS:
            a = IP.cand_progress(aP, aB, (0.0,), floor)
            assert np.array_equal(a[0, 0], np.minimum(aP[0], aB[0]) if floor == "cap-only" else aB[0])
        o = rule(plan, v0, g_src=-50.0, cset=(1.0, 0.0))                      # nothing is safe: the plan executed is c = 0.0
        assert o.chosen_c[0] == 0.0
        arcs = np.array([arc_at(o.plan[0], i) for i in range(1, 31)])
        assert np.abs(arcs - aB[0, 1:]).max() < 1e-5


def test_monotone_in_c_random_plans_both_floors():
    rng = np.random.default_rng(1)
    for _ in range(60):
        v0 = rng.uniform(0, 25)
        steps = np.abs(v0 * DT + np.cumsum(rng.normal(0, 0.4, 30)) * DT)             # random speed profile, including plans that brake harder than B_MAX
        plan = np.stack([np.cumsum(steps), np.cumsum(rng.normal(0, 0.05, 30))], 1)[None]
        aP, _ = IP.plan_arcs(plan)
        aB = IP.braking_progress(np.array([v0]))
        for floor in IP.FLOORS:
            a = IP.cand_progress(aP, aB, IP.CAND_CARD, floor)[0]                     # [C, 31], c descending
            assert (a[:-1] >= a[1:] - 1e-12).all(), "a smaller c must never have a larger progress"
            assert (np.diff(a, axis=1) >= -1e-12).all()


# ============================================================================================ the rule: hand cases
def test_hand_boundary_exactly_one_candidate_boundary_is_safe():
    plan = straight(10.0)                                                         # constant 10 m/s: a_P(5) = 10, a_P(10) = 20; a_B(5) = 7, a_B(10) = 25/3
    cs = IP.CAND_CARD
    o = rule(plan, 10.0, g_src=25.0, v_L=0.0)
    assert o.safe[0].tolist() == [False, False, False] + [True] * 8              # c >= 0.8: gap(10) = 25 - 20c < 1.2 * 10c
    assert o.chosen_c[0] == pytest.approx(0.7)
    assert o.a_h[0, cs.index(0.7)].tolist() == pytest.approx([7.0, 14.0]) and o.v_h[0, cs.index(0.7)].tolist() == pytest.approx([4.6, 7.0])
    assert o.gap_h[0, cs.index(0.7)].tolist() == pytest.approx([18.0, 11.0])
    assert rule(plan, 10.0, g_src=22.5, v_L=0.0).chosen_c[0] == pytest.approx(0.7)   # 8.5 >= 1.2 * 7 = 8.4
    assert rule(plan, 10.0, g_src=22.0, v_L=0.0).chosen_c[0] == pytest.approx(0.6)   # 8.0 <  8.4 : c = 0.7 is out, c = 0.6 (gap 10 >= 7.2) is in
    assert rule(plan, 10.0, g_src=80.0, v_L=0.0).chosen_c[0] == 1.0


def test_selection_largest_safe_c_and_zero_when_none_is_safe_and_set_order_does_not_matter():
    plan = straight(10.0)
    o = rule(plan, 10.0, g_src=25.0, cset=(0.3, 0.9, 0.5, 0.7), fallback="none")
    assert o.chosen_c[0] == pytest.approx(0.7)                                    # the largest safe, not the first, not the smallest
    o0 = rule(plan, 10.0, g_src=5.0)                                              # a_B(5) = 7 > 5 : even full braking cannot keep the gap
    assert not o0.any_safe[0] and o0.chosen_c[0] == 0.0 and o0.chosen_idx[0] == -1
    arcs = np.array([arc_at(o0.plan[0], i) for i in range(1, 31)])
    assert np.abs(arcs - IP.braking_progress(np.array([10.0]))[0, 1:]).max() < 1e-5    # executed = full braking
    on = rule(plan, 10.0, g_src=5.0, fallback="none")
    assert np.isnan(on.chosen_c[0]) and on.plan is not None and np.array_equal(on.plan, plan[None]) and not on.changed[0]


def test_cand_cv_gap_formula_equals_the_hand_value():
    plan = straight(10.0)
    o = rule(plan, 10.0, g_src=30.0, a_off=1.5, v_L=5.0, cset=(1.0,))
    # gap(h) = 30 + 1.5 + 5 h - a_c(h), a_c = (10, 20) at c = 1.0
    assert o.gap_h[0, 0].tolist() == pytest.approx([26.5, 21.5])
    om = rule(plan, 10.0, g_src=30.0, a_off=-1.5, v_L=5.0, cset=(1.0,))          # the own-progress term enters with its sign
    assert om.gap_h[0, 0].tolist() == pytest.approx([23.5, 18.5])
    ol = rule(plan, 10.0, g_src=30.0, a_off=1.5, v_L=5.0, cset=(1.0,), source="cand-label")
    assert np.array_equal(ol.gap_h, o.gap_h)                                      # cand-label is the same formula with the label values


def test_random_cases_equal_the_scalar_reference():
    rng = np.random.default_rng(7)
    n_diff = 0
    for floor in IP.FLOORS:
        for fb in IP.FALLBACKS:
            for _ in range(60):
                v0 = rng.uniform(0, 25)
                steps = np.abs(v0 * DT + np.cumsum(rng.normal(0, 0.3, 30)) * DT)
                plan = np.stack([np.cumsum(steps), np.cumsum(rng.normal(0, 0.05, 30))], 1)
                g, off, vl = rng.uniform(-5, 70), rng.uniform(-4, 4), rng.uniform(0, 20)
                cset = IP.CAND_CARD if rng.uniform() < 0.7 else tuple(sorted(rng.choice(IP.CAND_CARD, 4, replace=False), reverse=True))
                o = rule(plan, v0, g, off, vl, cset=cset, floor=floor, fallback=fb)
                c, safes, aa, gaps = ref_choose(plan, v0, g, off, vl, cset, floor, fb)
                assert (np.isnan(o.chosen_c[0]) and c is None) or o.chosen_c[0] == pytest.approx(c)
                assert o.safe[0].tolist() == safes
                assert np.allclose(o.a_h[0], aa) and np.allclose(o.gap_h[0], gaps)
                n_diff += int(o.changed[0])
    assert n_diff > 20                                                           # the cases are not all identities


# ============================================================================================ identity (p1) and the control options reach their value
def test_identity_set_1_alone_without_fallback_and_const_gap_100_return_the_plan_object():
    rng = np.random.default_rng(3)
    plans = np.stack([straight(v) + rng.normal(0, 0.01, (30, 2)).astype(np.float32) * 0 for v in rng.uniform(3, 20, 12)])       # constant-speed plans: no lift under the card floor either
    v0 = np.array([float(arc_at(p, 1) / DT) for p in plans])
    act = np.ones(len(plans), bool)
    args = (plans, v0, act, np.full(len(plans), 8.0), np.zeros(len(plans)), np.zeros(len(plans)))
    for floor in IP.FLOORS:
        # {1.0} alone, fallback none: every window, safe or not, keeps its plan; the SAME object comes back
        o = IP.imag_plan_rule(*args, "cand-cv", cset=(1.0,), floor=floor, fallback="none")
        assert o.plan is plans and not o.changed.any() and o.active.all()
        # a source that returns 100 m everywhere (after the gate): every candidate safe, c = 1.0, the plan is the plan
        o = IP.imag_plan_rule(*args, "cand-cv", const_gap=100.0, floor=floor)
        assert o.plan is plans and (o.chosen_c == 1.0).all() and not o.changed.any()
    # the control is not vacuous: the same windows at the real gap of 8 m DO change under the default set
    assert IP.imag_plan_rule(*args, "cand-cv").changed.any()
    # inactive windows keep their plan; all inactive returns the object
    o = IP.imag_plan_rule(plans, v0, np.zeros(len(plans), bool), *args[3:], "cand-cv")
    assert o.plan is plans and (o.chosen_idx == -2).all()


def test_the_card_floor_lifts_a_plan_that_brakes_harder_than_B_MAX_and_cap_only_does_not():
    plan = profile(10.0, a=-9.0)                                                  # 9 m/s2 braking: a_P(1) = 1.82 m below a_B(1) = 1.88 m
    oc = rule(plan, 10.0, g_src=100.0, cset=(1.0,), fallback="none", floor="card")
    oo = rule(plan, 10.0, g_src=100.0, cset=(1.0,), fallback="none", floor="cap-only")
    assert oc.changed[0] and oc.n_lift[0] > 0                                      # E1: c = 1.0 is not the plan itself
    assert not oo.changed[0] and oo.n_lift[0] == 0 and np.array_equal(oo.plan, plan[None])


def test_option_const_gap_reaches_the_gap_and_changes_the_choice():
    plan = straight(10.0)
    base = rule(plan, 10.0, g_src=80.0)
    tiny_gap = rule(plan, 10.0, g_src=80.0, const_gap=2.0)
    assert base.chosen_c[0] == 1.0 and tiny_gap.chosen_c[0] == 0.0 and np.all(tiny_gap.gap_h[0] == 2.0)
    assert np.all(rule(plan, 10.0, g_src=80.0, const_gap=100.0).gap_h[0] == 100.0)


def test_option_imag_shift_reaches_every_imagined_gap_and_cset_and_fallback_reach_the_choice():
    plan = straight(10.0)

    def planted(cand, ia):                                                       # a planted world model: the exact kinematic gap of the candidate's own plan
        out = np.zeros(cand.shape[:2] + (2,))
        for k in range(cand.shape[0]):
            for c in range(cand.shape[1]):
                out[k, c] = [60.0 - arc_at(cand[k, c], 5), 60.0 - arc_at(cand[k, c], 10)]
        return out
    o0 = rule(plan, 10.0, g_src=0.0, source="cand-imag", imag_fn=planted)
    os_ = rule(plan, 10.0, g_src=0.0, source="cand-imag", imag_fn=planted, shift_m=-40.0)
    assert np.allclose(os_.gap_h - o0.gap_h, -40.0) and np.allclose(os_.imag_raw, o0.imag_raw)       # shift reaches every gap, the raw read is untouched
    assert o0.chosen_c[0] > os_.chosen_c[0]
    assert rule(plan, 10.0, g_src=80.0, cset=(0.5,)).chosen_c[0] == 0.5            # the set reaches the choice
    assert rule(plan, 10.0, g_src=80.0, cset=(0.5,)).cset == (0.5, 0.0)
    assert rule(plan, 10.0, g_src=5.0, cset=(1.0,), fallback="none").cset == (1.0,)


def test_planted_world_model_gives_the_same_choices_as_cand_cv_with_the_true_lead_speed():
    rng = np.random.default_rng(11)
    n = 80
    v0 = rng.uniform(0, 22, n)
    plans = np.stack([profile(v, a=rng.uniform(-3, 2)) for v in v0])
    g, off, vl = rng.uniform(0, 60, n), rng.uniform(-3, 3, n), rng.uniform(0, 18, n)
    act = np.ones(n, bool)

    def planted(cand, ia):                                                       # kinematics only: the gap at h = source gap + lead travel - the candidate's own travel (from ITS plan)
        out = np.zeros(cand.shape[:2] + (2,))
        for k, b in enumerate(ia):
            for c in range(cand.shape[1]):
                out[k, c] = [g[b] + vl[b] * 1.0 - arc_at(cand[k, c], 5), g[b] + vl[b] * 2.0 - arc_at(cand[k, c], 10)]
        return out
    oi = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-imag", imag_fn=planted)
    ov = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-cv")
    assert np.array_equal(oi.chosen_idx, ov.chosen_idx) and np.array_equal(oi.safe, ov.safe)
    assert np.abs(oi.gap_h - ov.gap_h).max() < 1e-4                                  # the candidate plans are float32 (the dtype of the tool's plans): 1e-6 m of rounding
    assert np.array_equal(oi.plan, ov.plan) and len(set(ov.chosen_idx.tolist())) > 3        # the choices are varied, not all one candidate


def test_non_finite_forecast_is_never_safe_and_is_counted():
    plan = straight(10.0)
    o = rule(plan, 10.0, g_src=80.0, source="cand-imag", imag_fn=lambda cand, ia: np.full(cand.shape[:2] + (2,), np.nan))
    assert not o.any_safe[0] and o.chosen_c[0] == 0.0 and o.n_nonfinite[0] == len(o.cset)


# ============================================================================================ actions of a candidate (the pinned tool's own spelling)
def _window_states(T, world, join_path, wins=(7, 39)):
    store, tracks, J, wl = _env(T, world["root"], join_path, wins)
    states = W.attach_logged(T.make_states(store, tracks, J, wl, "deadreckon"), store, J)
    return states, store, tracks, J


def test_actions_of_a_candidate_equal_what_the_pinned_commit_writes_bit_for_bit(T, world):
    _, p = _lead_join(T, world["root"], name="join_act.npz", const_gap=20.0)
    states, *_ = _window_states(T, world, p)
    rng = np.random.default_rng(5)
    for st in states:
        for _ in range(4):
            v = rng.uniform(2, 15)
            plan = np.stack([np.cumsum(np.full(30, v * DT) + rng.normal(0, 0.05, 30)), np.cumsum(rng.normal(0, 0.03, 30))], 1)
            j = int(st.j)
            act = IP.candidate_actions(T, st, plan, 10, "executed", yaw_limit=False)     # the definition itself, without the yaw bound of note ADD40_N2 (tested on its own below)
            alog = IP.candidate_actions(T, st, plan, 10, "logged", yaw_limit=False)
            st.commit_plan(plan, 10)                                           # the pinned executor writes d['a'], d['ch1'] of frames j+1 .. j+10
            got = np.stack([np.nan_to_num(st.d["a"][j + 1:j + 10]), np.nan_to_num(st.d["ch1"][j + 1:j + 10])], 1)
            assert arrays_equal(act[1:], got), "candidate actions must be the very numbers the pinned tool writes when it executes the plan"
            assert np.isfinite(act).all() and act.shape == (10, 2)
            # the corpus' definition: the logged acceleration of frame f is the forward difference of the cache-step speed the executed world then shows (d['v'] = track speed at the even rows)
            v = np.asarray(st.d["v"][j:j + 11], np.float64)
            assert np.abs(alog[:, 0] - (v[1:] - v[:-1]) / DT).max() < 1e-3 and arrays_equal(alog[:, 1], act[:, 1])
            assert np.abs(alog[:, 0] - act[:, 0]).max() > 1e-3, "the two definitions must differ (else the flag is vacuous)"
            break
    # frame j itself is the measured action of the state (no future in it)
    st = states[0]
    act2 = IP.candidate_actions(T, st, straight(8.0).astype(np.float64), 10, "executed")
    assert act2[0].tolist() == [np.nan_to_num(st.d["a"][st.j]), np.nan_to_num(st.d["ch1"][st.j])]
    with pytest.raises(ValueError):
        IP.candidate_actions(T, st, straight(8.0).astype(np.float64), 10, "bogus")


def _planted_track(a, w, v0, n_raw=140, dt=0.1):
    """A native 10 Hz track with constant acceleration a and constant yaw rate w, exact kinematics (arc integration)."""
    t = np.arange(n_raw) * dt
    v = v0 + a * t
    yaw = w * t
    s = v0 * t + 0.5 * a * t * t
    if abs(w) < 1e-12:
        x, y = s, np.zeros_like(s)
    else:                                                                         # integrate the speed along the heading (trapezoid, exact enough: the test tolerance is stated)
        fx, fy = v * np.cos(yaw), v * np.sin(yaw)
        x = np.concatenate([[0.0], np.cumsum(0.5 * (fx[1:] + fx[:-1]) * dt)])
        y = np.concatenate([[0.0], np.cumsum(0.5 * (fy[1:] + fy[:-1]) * dt)])
    return np.stack([x, y, yaw, v], 1)


def test_action_conversion_recovers_a_known_speed_and_yaw_rate_profile(T):
    from types import SimpleNamespace
    j = 8
    for a_true, w_true, tol_a, tol_w in ((1.2, 0.0, 1e-6, 1e-6), (-2.5, 0.0, 1e-6, 1e-6), (0.8, 0.15, 0.04, 0.005)):
        tr = _planted_track(a_true, w_true, 9.0)
        pose = tr[2 * j].copy()
        d = {"pose": {j: pose}, "a": {j: a_true}, "ch1": {j: w_true}}
        st = SimpleNamespace(j=j, d=d, track=tr)
        c, s = np.cos(-pose[2]), np.sin(-pose[2])
        knots = tr[2 * (j + 1):2 * (j + 31):2, :2] - pose[:2]                      # the true path at the plan knots, ego frame at the tick
        plan = np.stack([knots[:, 0] * c - knots[:, 1] * s, knots[:, 0] * s + knots[:, 1] * c], 1)
        for definition in IP.ACTION_DEFS:
            act = IP.candidate_actions(T, st, plan, 10, definition)
            assert np.abs(act[:, 0] - a_true).max() < tol_a, (definition, a_true, w_true, act[:, 0])
            assert np.abs(act[:, 1] - w_true).max() < tol_w, (definition, a_true, w_true, act[:, 1])


# ============================================================================================ the wrapper, end to end (synthetic world, tiny checkpoint with a lead head)
def run_rollout2(T, world, join_path, planner, tick_windows=(7, 39), cfg=None, m=10, noise_seed=-1, p45=False):
    store, tracks, J, wins = _env(T, world["root"], join_path, tick_windows)
    proxy, ds = _ds(T, J, cfg)
    states = W.attach_logged(T.make_states(store, tracks, J, wins, "deadreckon"), store, J)
    WI.attach_logged_actions(T, states, store, tracks, J, p45)
    if noise_seed >= 0:
        WI.apply_noise_after_tick(states, noise_seed, m)
    if isinstance(planner, W.RulePlanner):
        planner.bind(states)
    plan0, exe = T.rollout(planner, states, ds, proxy, m=m, schedule="periodic", device="cpu")
    return plan0, np.asarray(exe), wins, states, J, store


def make_ip(T, source, J, model, **kw):
    label = W.LabelSource(J) if source == "cand-label" else None
    return WI.ImagPlanner(stub_planner, model, source, 10, T.fk, T, label=label, **kw)


def test_e2e_the_tick_state_equals_the_one_the_existing_rule_logs_and_cand_imag_runs_on_the_tiny_model(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_head2.npz", const_gap=20.0)
    ticks = tuple(range(3, 40, 4))
    rp = W.RulePlanner(stub_planner, model, "head", 10, T.fk)
    real_call = W.RulePlanner.__call__
    try:
        W.RulePlanner.__call__ = lambda self, batch: (real_call(self, batch), self.base(batch))[1]             # the rule logs, the plan stays the base plan: the same executed path as the identity run below
        run_rollout2(T, world, p, rp, ticks, cfg)
    finally:
        W.RulePlanner.__call__ = real_call
    ip = make_ip(T, "cand-imag", None, model, cset=(1.0,), fallback="none", floor="cap-only")
    out_i = run_rollout2(T, world, p, ip, ticks, cfg)
    ra, ia = rp.log.arrays(), ip.ilog.arrays()
    for k in ("ci", "t0", "j", "k", "present", "src_bad", "g_src", "c_src", "p_src", "v0", "v_l", "v_L", "a_l", "a_m", "g0", "active"):
        assert arrays_equal(ra[k], ia[k]), k
    assert ia["active"].sum() > 5 and ia["safe"].shape[1:] == (1,) and np.isfinite(ia["imag_raw"][ia["active"]]).all()
    assert (ia["chosen_idx"][ia["active"]] >= -1).all() and (ia["chosen_idx"][~ia["active"]] == -2).all()
    # the identity probe went through the whole path (rollout, head, rule) and returned every plan: the executed paths equal a plain rollout
    plain = run_rollout2(T, world, p, stub_planner, ticks, cfg)
    assert arrays_equal(out_i[1], plain[1]) and not ia["changed"].any()


def test_e2e_cand_modes_change_the_executed_path_when_the_forecast_says_unsafe_and_the_dump_is_untouched_otherwise(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    J, p = _lead_join(T, world["root"], t_lead=3, gap0=35.0, name="join_close.npz")             # a stopped lead 35 m ahead at frame 3: by frame 17 the 2 s check fails for an 8 m/s plan
    plain = run_rollout2(T, world, p, stub_planner, (7, 39), cfg)
    jl = T.D.load_join(p)
    lab = make_ip(T, "cand-label", jl, model)
    out = run_rollout2(T, world, p, lab, (7, 39), cfg)
    la = lab.ilog.arrays()
    assert la["active"].sum() > 0 and la["changed"].sum() > 0 and not arrays_equal(out[1], plain[1])
    assert (la["chosen_c"][la["changed"]] < 1.0).all()
    # the label source is the label: gap column of the log is the join's gap on the tick frame
    assert np.allclose(la["g_src"][la["active"]], [W.LabelSource(jl).read(int(c), int(j))[1] for c, j in zip(la["ci"][la["active"]], la["j"][la["active"]])])
    # the closer the lead, the lower the executed progress (the rule reaches the plan)
    prog = lambda o: np.hypot(o[1][..., -1, 0], o[1][..., -1, 1])                                 # noqa: E731
    assert prog(out).mean() < prog(plain).mean()


def test_e2e_every_control_option_reaches_the_value_it_changes(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_head3.npz", const_gap=20.0)
    ticks = tuple(range(3, 40, 4))

    def go(**kw):
        ip = make_ip(T, kw.pop("source", "cand-imag"), None, model, **kw)
        o = run_rollout2(T, world, p, ip, ticks, cfg)
        return ip.ilog.arrays(), o[1]
    base, e_base = go()
    act = base["active"]
    assert act.sum() > 5
    # p1: const gap 100 after the gate, cap-only floor: every active tick executes c = 1.0 and the executed paths equal the plain rollout
    c100, e100 = go(const_gap=100.0, floor="cap-only")
    assert np.all(c100["gap_h"][c100["active"]] == 100.0) and (c100["chosen_c"][c100["active"]] == 1.0).all() and arrays_equal(e100, run_rollout2(T, world, p, stub_planner, ticks, cfg)[1])
    # a const gap of 1 m must brake every active tick: the option reaches the gap
    c1, e1 = go(const_gap=1.0)
    assert (c1["chosen_c"][c1["active"]] == 0.0).all() and not arrays_equal(e1, e100)
    # p6: the shift reaches every imagined gap of every active tick; -1000 m brakes everything
    sh, _ = go(shift_m=-20.0)
    k0 = act & (base["k"] == 0)                                                    # tick 0 of every window: the executed paths of the runs are still identical
    assert k0.sum() > 2 and np.allclose((sh["gap_h"] - base["gap_h"])[k0], -20.0, atol=1e-9) and np.allclose(sh["imag_raw"][k0], base["imag_raw"][k0])
    s_all, _ = go(shift_m=-1000.0)
    assert (s_all["chosen_c"][s_all["active"]] == 0.0).all()
    # cand-cv and cand-imag gate on the same set of ticks (p7), and cand-cv reads no rollout (imag_raw NaN)
    cv, _ = go(source="cand-cv")
    assert np.array_equal(cv["active"], base["active"]) and np.isnan(cv["imag_raw"]).all()
    # the action definition reaches the imagined gap (tick 0: the states are identical)
    ex, _ = go(action_def="executed")
    k0e = ex["active"] & (ex["k"] == 0)
    assert k0e.sum() > 2 and np.abs(ex["imag_raw"][k0e] - base["imag_raw"][k0e]).max() > 1e-6
    # candidate set reaches the choice
    c05, _ = go(cset=(0.5,), fallback="none")
    assert set(np.unique(c05["chosen_c"][c05["active"] & c05["any_safe"]])) <= {0.5}


def test_e2e_p3_noise_after_the_last_tick_changes_no_output_and_the_noise_is_really_there(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_head4.npz", const_gap=20.0)
    ticks = tuple(range(3, 40, 4))
    a = make_ip(T, "cand-imag", None, model)
    ra = run_rollout2(T, world, p, a, ticks, cfg)
    b = make_ip(T, "cand-imag", None, model)
    rb = run_rollout2(T, world, p, b, ticks, cfg, noise_seed=3)
    assert a.ilog.arrays()["active"].sum() > 5
    assert arrays_equal(ra[1], rb[1]) and all(arrays_equal(a.ilog.arrays()[k], b.ilog.arrays()[k]) for k in ("gap_h", "chosen_c", "imag_raw"))
    # the firing control: the noise changed the stored tokens after the last tick of every window, and nothing at or before it
    store, tracks, J, wins = _env(T, world["root"], p, ticks)
    st0 = W.attach_logged(T.make_states(store, tracks, J, wins, "deadreckon"), store, J)
    f_before = [np.array(s.d["feats"], copy=True) for s in st0]
    WI.apply_noise_after_tick(st0, 3, 10)
    for s, f0 in zip(st0, f_before):
        lo = s.t0 + 30 - 10 + 1
        assert np.array_equal(s.d["feats"][:lo], f0[:lo]) and (len(f0) <= lo or not np.array_equal(s.d["feats"][lo:], f0[lo:]))
    # and noise INTO the frames a tick reads does change the output (the control that fires)
    c = make_ip(T, "cand-imag", None, model)
    store, tracks, J, wins = _env(T, world["root"], p, ticks)
    proxy, ds = _ds(T, J, cfg)
    states = W.attach_logged(T.make_states(store, tracks, J, wins, "deadreckon"), store, J)
    WI.attach_logged_actions(T, states, store, tracks, J, False)
    rng = np.random.default_rng(0)
    for s in states:
        f = np.array(s.d["feats"], copy=True)
        f[: s.t0 + 5] = rng.normal(0, 1, f[: s.t0 + 5].shape).astype(f.dtype)
        s.d["feats"] = f
    c.bind(states)
    T.rollout(c, states, ds, proxy, m=10, schedule="periodic", device="cpu")
    assert not arrays_equal(c.ilog.arrays()["imag_raw"], a.ilog.arrays()["imag_raw"])


def test_e2e_p5_imagined_gap_under_logged_actions_equals_an_independent_roll_of_the_same_states(T, world, tiny):
    import torch
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "imaghead")))
    import imag_dump as IH
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_head5.npz", const_gap=20.0)
    ip = make_ip(T, "cand-imag", None, model, p45={})
    run_rollout2(T, world, p, ip, (7, 39), cfg, p45=True)
    z = {k: np.concatenate(v) for k, v in ip.p45.items()}
    store = _env(T, world["root"], p, (7, 39))[0]
    J = T.D.load_join(p)
    assert np.isfinite(z["imag_gap_logged"]).all() and z["imag_gap_logged"].shape == (len(z["ci"]), 2)
    for i in range(len(z["ci"])):
        d = store.get(str(J["clip"][int(z["ci"][i])]))
        t0 = int(z["t0"][i])
        with torch.no_grad():
            f = torch.as_tensor(np.asarray(d["feats"][t0:t0 + 1], np.float32))[None]                   # [1, 1, n_tok, d_feat]
            z0 = model.wm.z1(model.adapter(f) + model.pos[None, None])[:, 0]
            A = model.wm.scale_actions(torch.as_tensor(np.stack([np.nan_to_num(d["a"][t0:t0 + 10]), d["ch1"][t0:t0 + 10]], -1).astype(np.float32))[None])
            Rr = IH.roll_states(torch, model.wm.stack, z0, A, IH.predictor_context(model.wm.stack))
            ref = IH.head_reads_rolled(torch, model, T.M, Rr)[0, :, 1].numpy()
        assert np.abs(ref - z["imag_gap_logged"][i]).max() < 1e-4, (i, ref, z["imag_gap_logged"][i])


def test_e2e_p4_log_holds_the_candidate_actions_of_the_logged_path_and_the_stats_function_reads_them(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_head6.npz", const_gap=20.0)
    ip = make_ip(T, "cand-cv", None, model, p45={})
    run_rollout2(T, world, p, ip, (7, 39), cfg, p45=True)
    z = {k: np.concatenate(v) for k, v in ip.p45.items()}
    assert z["act_gt_exec"].shape == z["act_gt"].shape and not np.array_equal(z["act_gt"][..., 0], z["act_gt_exec"][..., 0])
    assert z["act_gt"].shape == (len(z["ci"]), 10, 2) and np.isfinite(z["act_gt"]).all() and np.isfinite(z["act_cache"]).all() and np.isfinite(z["act_track"]).all()
    s = IP.p4_stats(z["act_gt"], z["act_gt"])
    assert s["corr_acc"] == pytest.approx(1.0) and s["median_abs_diff_acc"] == 0.0
    s2 = IP.p4_stats(z["act_gt"], z["act_gt"][::-1] * 0 + 1.0 + np.random.default_rng(0).normal(0, 1, z["act_gt"].shape))
    assert s2["corr_acc"] < 0.5 and s2["median_abs_diff_acc"] > 0.1                      # the stats function can fail
    # the logged path as the plan reproduces the executed-track channels of the same frames (the corpus' yaw-mode spelling) to the Hermite smoothing of the half-step frames
    assert np.corrcoef(z["act_gt"][..., 1].ravel(), z["act_track"][..., 1].ravel())[0, 1] > 0.9


def _cli(T, world, tiny, out_dir, extra, join_path=None, plan=None, tag="t"):
    root = world["root"]
    argv = ["--tree", TREE] + (["--imag-plan", plan] if plan else []) + extra + [
        "run", "--ckpt", tiny["ckpt"], "--tag", tag, "--tracks", os.path.join(root, "native_tracks.pt"), "--eval-cache", root,
        "--eval-grid", os.path.join(root, "eval_grid_windows.json"), "--eval-join", join_path or os.path.join(root, "join_eval.npz"), "--out-dir", str(out_dir),
        "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--batch", "8", "--device", "cpu", "--no-analyze",
        "--dump-ticks", os.path.join(str(out_dir), "ticks"), "--dump-heads"]
    return WI.main(argv)


def _dump_bytes(out_dir, tag):
    d = os.path.join(str(out_dir), f"dump_{tag}")
    res = {}
    for f in sorted(x for x in os.listdir(d) if x.startswith("ep") and x.endswith(".npz")):
        z = np.load(os.path.join(d, f))
        res[f] = {k: z[k].tobytes() for k in sorted(z.files)}
    return res


def test_cli_off_equals_the_pinned_tool_and_flags_with_off_are_refused(T, world, tiny, tmp_path):
    a, c = tmp_path / "off", tmp_path / "direct"
    assert _cli(T, world, tiny, a, [], plan="off") == 0
    root = world["root"]
    T.main(["run", "--ckpt", tiny["ckpt"], "--tag", "t", "--tracks", os.path.join(root, "native_tracks.pt"), "--eval-cache", root,
            "--eval-grid", os.path.join(root, "eval_grid_windows.json"), "--eval-join", os.path.join(root, "join_eval.npz"), "--out-dir", str(c),
            "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--batch", "8", "--device", "cpu",
            "--no-analyze", "--dump-ticks", os.path.join(str(c), "ticks"), "--dump-heads"])
    assert _dump_bytes(a, "t") == _dump_bytes(c, "t") and sorted(os.listdir(a)) == sorted(os.listdir(c))
    for extra in (["--imag-log", str(tmp_path / "l.npz")], ["--const-gap", "100"], ["--imag-shift-m", "-20"], ["--noise-after-tick", "1"], ["--cand-set", "1.0"], ["--p45-log", str(tmp_path / "p.npz")],
                  ["--fallback", "none"], ["--floor", "cap-only"], ["--action-def", "executed"]):
        with pytest.raises(SystemExit, match="would change nothing"):
            _cli(T, world, tiny, tmp_path / "x", extra, plan="off")
    with pytest.raises(SystemExit, match="cand-imag and cand-imagkin only"):
        _cli(T, world, tiny, tmp_path / "y", ["--imag-shift-m", "-20"], plan="cand-cv")


def test_cli_modes_are_reachable_from_argparse_to_the_dump_and_the_log_has_its_fields(T, world, tiny, tmp_path):
    _, p = _lead_join(T, world["root"], t_lead=3, gap0=35.0, name="join_cli2.npz")
    off = tmp_path / "off"
    assert _cli(T, world, tiny, off, [], join_path=p, plan="off") == 0
    lab = tmp_path / "lab"
    assert _cli(T, world, tiny, lab, ["--imag-log", str(lab / "ticks" / "imag_log_t.npz"), "--p45-log", str(lab / "ticks" / "p45_t.npz")], join_path=p, plan="cand-label") == 0
    z = np.load(lab / "ticks" / "imag_log_t.npz")
    assert set(WI.LOG_SCALARS + WI.LOG_VECTORS) <= set(z.files) and str(z["mode"]) == "cand-label" and str(z["pinned_sha256"]) == PINNED_SHA and len(str(z["join_sha256"])) == 64
    assert z["cand_set"].tolist() == list(IP.CAND_CARD) and z["active"].sum() > 0 and z["changed"].sum() > 0
    assert _dump_bytes(lab, "t") != _dump_bytes(off, "t")
    ticks = np.load(lab / "ticks" / "ticks_t.npz")
    assert len(ticks["plan"]) == len(z["ci"])                                    # one log row per (window, tick) of the model rollout
    p45 = np.load(lab / "ticks" / "p45_t.npz")
    assert p45["act_gt"].shape[1:] == (10, 2) and np.isnan(p45["imag_gap_logged"]).all()      # cand-label reads no rollout: the p5 column is NaN by design
    hd = tmp_path / "head"
    assert _cli(T, world, tiny, hd, ["--imag-log", str(hd / "ticks" / "imag_log_t.npz"), "--const-gap", "1e9", "--floor", "cap-only"], join_path=p, plan="cand-imag") == 0
    zh = np.load(hd / "ticks" / "imag_log_t.npz")
    assert str(zh["mode"]) == "cand-imag" and float(zh["const_gap"]) == 1e9 and zh["active"].sum() > 0 and not zh["changed"].any()
    assert _dump_bytes(hd, "t") == _dump_bytes(off, "t")                         # p1 through the CLI: a const gap of 1e9 m (the tiny model's random plans run at hundreds of m/s: 100 m is not safe for them) + cap-only = the pinned tool's dump, with the rule active


def test_pod_script_refusals_and_dry_run(tmp_path):
    import shutil
    import subprocess
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    script = os.path.join(HERE, "pod_imagplan_run.sh")
    mm, ms = tmp_path / "memory.max", tmp_path / "memory.stat"
    mm.write_text(str(60 * 2 ** 30) + "\n")
    ms.write_text("anon " + str(50 * 2 ** 30) + "\nfile 1\n")                       # only 10 GB free
    env = dict(os.environ, IP_MEMMAX_FILE=str(mm).replace("\\", "/"), IP_MEMSTAT_FILE=str(ms).replace("\\", "/"), IP_GPU_COUNT="0", IP_DRY="1", IP_W=str(tmp_path / "w").replace("\\", "/"))

    def run(args, **e):
        r = subprocess.run([bash, script.replace("\\", "/")] + args, env=dict(env, **e), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return r.returncode, r.stdout + r.stderr
    rc, out = run(["A", "Q", "B", "cand-imag"])
    assert rc == 2 and "free container memory" in out
    ms.write_text("anon " + str(10 * 2 ** 30) + "\n")
    rc, out = run(["A", "Q", "B", "cand-imag"], IP_GPU_COUNT="3")
    assert rc == 2 and "GPU" in out
    rc, out = run(["A", "Q", "B", "bogus"])
    assert rc == 2 and "MODE" in out
    rc, out = run(["A"])
    assert rc == 2 and "usage" in out


def test_yaw_limit_bounds_a_heading_flip_and_leaves_a_physical_yaw_rate_untouched(T, world):
    """Note PREREG_V8F_PROOF_ADD40_N2 (added by the lead): |yaw rate| <= KAPPA_MAX * |v| on every candidate action; a value inside the bound is untouched."""
    act = np.array([[0.1, 31.416], [0.0, -31.416], [0.2, 0.15], [0.0, 0.0]])
    v = np.array([0.05, 0.3, 8.0, 0.0])
    out, n = IP.limit_yaw(act, v)
    assert n == 2 and out[0, 1] == pytest.approx(IP.KAPPA_MAX * 0.05) and out[1, 1] == pytest.approx(-IP.KAPPA_MAX * 0.3)
    assert out[2, 1] == act[2, 1] and out[3, 1] == 0.0 and arrays_equal(out[:, 0], act[:, 0])
    # through candidate_actions: a path that jitters back and forth at standstill (the measured case of the note)
    _, p = _lead_join(T, world["root"], name="join_yaw.npz", const_gap=20.0)
    states, *_ = _window_states(T, world, p)
    st = states[0]
    rng = np.random.default_rng(11)
    jit = np.stack([np.cumsum(rng.normal(0.0, 0.006, 30)), np.cumsum(rng.normal(0.0, 0.006, 30))], 1)     # millimetre steps in random directions, above the 1 mm tangent threshold
    raw = IP.candidate_actions(T, st, jit, 10, "logged", yaw_limit=False)
    lim, n = IP.candidate_actions(T, st, jit, 10, "logged", return_n=True)
    sl = IP.candidate_track_rows(T, st, jit, 10)
    bound = IP.KAPPA_MAX * np.abs(sl[0::2, 3][0:10]) + 1e-6
    assert np.abs(raw[1:, 1]).max() > 5.0, "the planted jitter must produce a heading jump in the unlimited actions (else this test proves nothing)"
    assert n >= 1 and (np.abs(lim[:, 1]) <= bound).all()
    assert arrays_equal(lim[:, 0], raw[:, 0]), "the acceleration channel is not touched by the yaw bound"
    # a moving plan: the steps that come from the plan are not changed by the bound
    mv = straight(8.0).astype(np.float64)
    a1 = IP.candidate_actions(T, st, mv, 10, "logged")
    a0 = IP.candidate_actions(T, st, mv, 10, "logged", yaw_limit=False)
    assert arrays_equal(a1[1:], a0[1:])


# ============================================================================================ candidate family decel (note PREREG_V8F_PROOF_ADD40_N3 section 2)
def ref_decel(plan, g_src, a_off, v_L, dset, fallback="zero"):
    """Independent scalar reference of the decel family for ONE window (plain loops, from the note text): (chosen d or None, safe list over the ascending set, a at knots 5 and 10 per d)."""
    pts = [(0.0, 0.0)] + [tuple(map(float, p)) for p in plan]
    aP = [0.0]
    for i in range(1, len(pts)):
        aP.append(aP[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    ds = sorted(set(dset))
    safes, aa = [], []
    for d in ds:
        a = [0.0]
        for i in range(1, 31):
            vP = (aP[i] - aP[i - 1]) / DT
            a.append(a[-1] + max(0.0, vP - d * DT * i) * DT)
        a = [aP[i] if d == 0.0 else min(a[i], aP[i]) for i in range(31)]
        ok = True
        for i, h in ((5, 1.0), (10, 2.0)):
            gap = g_src + a_off + v_L * h - a[i]
            ok = ok and gap >= max(3.0, 1.2 * (a[i] - a[i - 1]) / DT)
        safes.append(ok)
        aa.append((a[5], a[10]))
    sd = [d for d, s in zip(ds, safes) if s]
    return (min(sd) if sd else (max(ds) if fallback == "zero" else None)), safes, aa


def test_decel_family_constants_and_set_handling():
    assert IP.DECEL_CARD == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) and IP.FAMILIES == ("scale", "decel")
    assert IP.parse_decel_set("2,0.5,2,0") == (0.0, 0.5, 2.0)                       # ascending, distinct, 0 allowed, values above 1 allowed
    for bad in ("-1", "nan", ""):
        with pytest.raises(ValueError):
            IP.parse_decel_set(bad)
    assert IP.effective_set((2.0, 0.5), "zero", "decel") == (0.5, 2.0) and IP.effective_set((2.0, 0.5), "none", "decel") == (0.5, 2.0)   # nothing is added
    assert IP.effective_set((1.0,), "zero") == (1.0, 0.0)                              # the default family is unchanged
    with pytest.raises(ValueError):
        rule(straight(10.0), 10.0, 30.0, family="bogus")
    assert rule(straight(10.0), 10.0, 30.0).family == "scale" and rule(straight(10.0), 10.0, 30.0, family="decel", cset=IP.DECEL_CARD).family == "decel"


def test_decel_hand_values_constant_plan_speed_10_and_d_2():
    plan = straight(10.0)
    seen = {}

    def spy(cand, ia):
        seen["cand"] = cand.copy()
        return np.zeros(cand.shape[:2] + (2,))
    o = rule(plan, 10.0, g_src=0.0, source="cand-imag", imag_fn=spy, family="decel", cset=(2.0,), fallback="none")
    # v_d(i) = 10 - 0.4 i ; a_d(10) = 20 - 0.4 * 0.2 * 55 = 15.6 m ; a_d(5) = 10 - 0.08 * 15 = 8.8 m ; v_d(5) = 8, v_d(10) = 6
    assert o.a_h[0, 0].tolist() == pytest.approx([8.8, 15.6], abs=1e-5) and o.v_h[0, 0].tolist() == pytest.approx([8.0, 6.0], abs=1e-5)
    assert arc_at(seen["cand"][0, 0], 10) == pytest.approx(15.6, abs=1e-4) and arc_at(seen["cand"][0, 0], 5) == pytest.approx(8.8, abs=1e-4)
    # the candidate path is the polyline of the plan (a straight line here): y stays 0, x is the progress
    assert np.abs(seen["cand"][0, 0, :, 1]).max() == 0.0 and seen["cand"][0, 0, 9, 0] == pytest.approx(15.6, abs=1e-4)
    # cand-cv gap = g + a_off + v_L h - a_d(h)
    ov = rule(plan, 10.0, g_src=30.0, a_off=1.5, v_L=5.0, family="decel", cset=(2.0,), fallback="none")
    assert ov.gap_h[0, 0].tolist() == pytest.approx([30 + 1.5 + 5 - 8.8, 30 + 1.5 + 10 - 15.6], abs=1e-5)


def test_decel_zero_is_the_plan_bit_for_bit_progress_knots_and_object():
    rng = np.random.default_rng(21)
    for _ in range(20):
        v0 = rng.uniform(0, 20)
        steps = np.abs(v0 * DT + np.cumsum(rng.normal(0, 0.4, 30)) * DT)
        plan = np.stack([np.cumsum(steps), np.cumsum(rng.normal(0, 0.05, 30))], 1)[None].astype(np.float32)
        aP, _ = IP.plan_arcs(plan)
        a = IP.decel_progress(aP, (0.0, 1.0, 6.0))
        assert np.array_equal(a[:, 0], aP) and (a[:, 1:] <= aP[:, None, :]).all()
        seen = {}

        def spy(cand, ia):
            seen["c"] = cand.copy()
            return np.full(cand.shape[:2] + (2,), 500.0)
        o = rule(plan, v0, g_src=500.0, source="cand-imag", family="decel", cset=(0.0, 3.0), fallback="none", imag_fn=spy)
        assert np.array_equal(seen["c"][0, 0], plan[0]) and seen["c"].dtype == plan.dtype      # the d = 0 candidate polyline is the own knots of the plan
        assert o.plan is plan and o.chosen_c[0] == 0.0 and not o.changed[0]
    # the plan is the object when the executed candidate is d = 0 (also with a one-element set and no fallback)
    assert rule(plan, 5.0, g_src=-50.0, family="decel", cset=(0.0,), fallback="none").plan is plan


def test_decel_monotone_in_d_speeds_never_negative_and_a_stopped_candidate_stays_stopped():
    rng = np.random.default_rng(22)
    for _ in range(60):
        v0 = rng.uniform(0, 25)
        steps = np.abs(v0 * DT + np.cumsum(rng.normal(0, 0.5, 30)) * DT)             # arbitrary speed profile, may re-accelerate
        plan = np.stack([np.cumsum(steps), np.cumsum(rng.normal(0, 0.05, 30))], 1)[None]
        aP, _ = IP.plan_arcs(plan)
        v = IP.decel_step_speeds(aP, IP.DECEL_CARD)[0]                                   # [C, N]
        a = IP.decel_progress(aP, IP.DECEL_CARD)[0]                                       # [C, N+1]
        assert (v >= 0.0).all() and (v[:-1] >= v[1:]).all() and (a[:-1] >= a[1:]).all(), "a larger d must never have a larger speed or progress"
        assert (np.diff(a, axis=1) >= -1e-12).all()
    plan = profile(10.0, a=-1.5)[None]                                                    # a plan that brakes to a stop: the candidates that stop must stay at standstill
    aP, _ = IP.plan_arcs(plan)
    a = IP.decel_progress(aP, (6.0,))[0, 0]
    stop = int(np.argmax(np.diff(a) <= 1e-12))
    assert 1 <= stop < 20 and np.allclose(np.diff(a)[stop:], 0.0, atol=1e-12)


def test_decel_selection_smallest_safe_d_largest_d_when_none_is_safe_and_fallback_none_keeps_the_plan():
    plan = straight(10.0)
    o = rule(plan, 10.0, g_src=25.0, family="decel", cset=IP.DECEL_CARD)
    c, safes, aa = ref_decel(plan, 25.0, 0.0, 0.0, IP.DECEL_CARD)
    assert o.safe[0].tolist() == safes and 0 < sum(safes) < len(safes)                  # a case where only some candidates are safe
    assert o.chosen_c[0] == pytest.approx(c) and o.chosen_c[0] == min(d for d, s in zip(IP.DECEL_CARD, safes) if s)
    assert o.cset == IP.DECEL_CARD and o.chosen_idx[0] == IP.DECEL_CARD.index(o.chosen_c[0]) and o.chosen_c[0] > 0.0
    assert np.allclose(o.a_h[0], aa, atol=1e-5)
    on = rule(plan, 10.0, g_src=5.0, family="decel", cset=IP.DECEL_CARD)                  # nothing safe (even d = 6 gives a_d(5) = 6.4 m > 5 m)
    assert not on.any_safe[0] and on.chosen_idx[0] == -1 and on.chosen_c[0] == 6.0 and on.changed[0]
    assert arc_at(on.plan[0], 10) == pytest.approx(0.2 * (8 * 10 - 1.2 * 36), abs=1e-4)   # the executed plan is the d = 6 candidate (it stops after step 8: v_d(9) = 10 - 10.8 < 0)
    assert rule(plan, 10.0, g_src=5.0, family="decel", cset=(0.5, 3.0)).chosen_c[0] == 3.0   # the largest d OF THE SET
    nn = rule(plan, 10.0, g_src=5.0, family="decel", cset=IP.DECEL_CARD, fallback="none")
    assert np.isnan(nn.chosen_c[0]) and nn.plan is not None and np.array_equal(nn.plan, plan[None]) and not nn.changed[0]
    assert rule(plan, 10.0, g_src=500.0, family="decel", cset=IP.DECEL_CARD).chosen_c[0] == 0.0     # all safe: d = 0, the plan


def test_decel_random_cases_equal_the_scalar_reference_and_floor_is_ignored():
    rng = np.random.default_rng(23)
    n_diff = 0
    for fb in IP.FALLBACKS:
        for _ in range(80):
            v0 = rng.uniform(0, 25)
            steps = np.abs(v0 * DT + np.cumsum(rng.normal(0, 0.3, 30)) * DT)
            plan = np.stack([np.cumsum(steps), np.cumsum(rng.normal(0, 0.05, 30))], 1)
            g, off, vl = rng.uniform(-5, 70), rng.uniform(-4, 4), rng.uniform(0, 20)
            cset = IP.DECEL_CARD if rng.uniform() < 0.6 else tuple(sorted(rng.choice(IP.DECEL_CARD, 4, replace=False)))
            o = rule(plan, v0, g, off, vl, cset=cset, fallback=fb, family="decel")
            c, safes, aa = ref_decel(plan, g, off, vl, cset, fb)
            assert (np.isnan(o.chosen_c[0]) and c is None) or o.chosen_c[0] == pytest.approx(c)
            assert o.safe[0].tolist() == safes and np.allclose(o.a_h[0], aa, atol=1e-6)
            assert np.array_equal(o.safe, rule(plan, v0, g, off, vl, cset=cset, fallback=fb, family="decel", floor="cap-only").safe)       # floor: no effect
            n_diff += int(o.changed[0])
    assert n_diff > 20


def test_decel_planted_world_model_gives_the_same_choices_as_cand_cv():
    rng = np.random.default_rng(24)
    n = 80
    v0 = rng.uniform(0, 22, n)
    plans = np.stack([profile(v, a=rng.uniform(-3, 2)) for v in v0])
    g, off, vl = rng.uniform(0, 60, n), rng.uniform(-3, 3, n), rng.uniform(0, 18, n)
    act = np.ones(n, bool)

    def planted(cand, ia):
        out = np.zeros(cand.shape[:2] + (2,))
        for k, b in enumerate(ia):
            for c in range(cand.shape[1]):
                out[k, c] = [g[b] + vl[b] * 1.0 - arc_at(cand[k, c], 5), g[b] + vl[b] * 2.0 - arc_at(cand[k, c], 10)]
        return out
    oi = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-imag", imag_fn=planted, family="decel", cset=IP.DECEL_CARD)
    ov = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-cv", family="decel", cset=IP.DECEL_CARD)
    assert np.array_equal(oi.chosen_idx, ov.chosen_idx) and np.array_equal(oi.safe, ov.safe) and np.abs(oi.gap_h - ov.gap_h).max() < 1e-4
    assert np.array_equal(oi.plan, ov.plan) and len(set(ov.chosen_idx.tolist())) > 2
    ol = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-label", family="decel", cset=IP.DECEL_CARD)
    assert np.array_equal(ol.gap_h, ov.gap_h)                                             # cand-label = the same formula with label values


def test_decel_actions_yaw_is_the_plan_acceleration_is_the_plan_minus_d_and_d0_is_bit_equal(T, world):
    _, p = _lead_join(T, world["root"], name="join_decel_act.npz", const_gap=20.0)
    states, *_ = _window_states(T, world, p)
    rng = np.random.default_rng(31)
    n_moving = n_stopped = 0
    for st in states:
        for v in (3.0, 9.0, 14.0):
            plan = np.stack([np.cumsum(np.full(30, v * DT) + rng.normal(0, 0.03, 30)), np.cumsum(rng.normal(0, 0.03, 30))], 1)
            base = IP.candidate_actions(T, st, plan, 10, "logged")
            sl = IP.candidate_track_rows(T, st, plan, 10)
            vt = sl[0::2, 3]
            a0, n0 = IP.decel_actions(T, st, plan, 0.0, 10, "logged", return_n=True)
            assert arrays_equal(a0, base) and a0.dtype == base.dtype == np.float32 and a0.shape == (10, 2) and n0 == IP.candidate_actions(T, st, plan, 10, "logged", return_n=True)[1]
            for d in (1.0, 2.0, 6.0):
                a = IP.decel_actions(T, st, plan, d, 10, "logged")
                assert a.dtype == np.float32 and a.shape == (10, 2) and np.isfinite(a).all()
                assert arrays_equal(a[:, 1], base[:, 1]), "the yaw channel is the one of the plan, bit for bit"
                vd = vt.astype(np.float64).copy()
                vd[1:] = np.maximum(0.0, vt[1:] - d * DT * np.arange(1, 11))
                for s in range(10):
                    if vd[s + 1] > 0.0 and vd[s] > 0.0 and (s == 0 or vt[s] - d * DT * s > 0.0):
                        n_moving += 1
                        assert abs(float(a[s, 0]) - (float(base[s, 0]) - d)) < 1e-4, (s, d)        # candidate still moving: the acceleration of the plan minus d
                    elif vd[s + 1] == 0.0 and s > 0:
                        n_stopped += 1
                        assert abs(float(a[s, 0]) + vd[s] / DT) < 1e-4                              # the step that reaches or stays at standstill
                assert not arrays_equal(a[:, 0], base[:, 0])
                b, nl = IP.decel_actions(T, st, plan, d, 10, "logged", return_n=True, base=(base, 0, np.asarray(vt, np.float64)))
                assert arrays_equal(b, a) and isinstance(nl, int)
    assert n_moving > 20 and n_stopped > 3                                                  # both branches were exercised


def test_pod_script_family_flag_in_the_dry_run_and_refusals(tmp_path):
    import shutil
    import subprocess
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash")
    script = os.path.join(HERE, "pod_imagplan_run.sh")
    mm, ms = tmp_path / "memory.max", tmp_path / "memory.stat"
    mm.write_text(str(60 * 2 ** 30) + "\n")
    ms.write_text("anon " + str(10 * 2 ** 30) + "\n")
    w = tmp_path / "w"
    (w / "runs" / "Q" / "B").mkdir(parents=True)
    (w / "runs" / "Q" / "B" / "ckpt_last.pt").write_bytes(b"x")
    (w / "data" / "cache").mkdir(parents=True)
    for f in ("tracks.pt", "grid.json", "join.npz"):
        (w / "data" / f).write_bytes(b"x")
    fw = lambda x: str(x).replace("\\", "/")      # noqa: E731
    env = dict(os.environ, IP_MEMMAX_FILE=fw(mm), IP_MEMSTAT_FILE=fw(ms), IP_GPU_COUNT="0", IP_DRY="1", IP_W=fw(w), IP_TREE=fw(TREE), IP_TRACKS=fw(w / "data" / "tracks.pt"),
               IP_GRID=fw(w / "data" / "grid.json"), IP_JOIN=fw(w / "data" / "join.npz"), IP_EVALC=fw(w / "data" / "cache"))
    env.pop("IP_FAMILY", None)

    def run(args, **e):
        r = subprocess.run([bash, fw(script)] + args, env=dict(env, **e), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return r.returncode, r.stdout + r.stderr
    rc, out = run(["A", "Q", "B", "cand-cv"], IP_FAMILY="decel", IP_CAND_SET="0,2")
    assert rc == 0 and "DRY:" in out and "--cand-family decel" in out and "--cand-set 0,2" in out, out[-600:]
    rc, out = run(["A", "Q", "B", "cand-cv"])
    assert rc == 0 and "--cand-family scale" in out and "--cand-family decel" not in out
    rc, out = run(["A", "Q", "B", "cand-cv"], IP_FAMILY="bogus")
    assert rc == 2 and "IP_FAMILY" in out
    rc, out = run(["A", "Q", "B", "cand-cv"], IP_FAMILY="decel", IP_FLOOR="cap-only")
    assert rc == 2 and "IP_FLOOR" in out
    rc, out = run(["A", "Q", "B", "off"], IP_FAMILY="decel")
    assert rc == 2 and "would change nothing" in out


def test_decel_e2e_cand_imag_reaches_the_actions_of_every_d_and_d0_equals_the_scale_identity(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_head7.npz", const_gap=20.0)
    ticks = tuple(range(3, 40, 4))
    dec = make_ip(T, "cand-imag", None, model, cset=IP.DECEL_CARD, family="decel")
    run_rollout2(T, world, p, dec, ticks, cfg)
    z = dec.ilog.arrays()
    act = z["active"]
    assert act.sum() > 5 and z["imag_raw"].shape[1:] == (len(IP.DECEL_CARD), 2) and np.isfinite(z["imag_raw"][act]).all()
    # the d of a candidate reaches the imagined gap: candidates differ from d = 0 (tick 0: the states are identical there)
    k0 = act & (z["k"] == 0)
    assert k0.sum() > 2
    dg = np.abs(z["imag_raw"][k0] - z["imag_raw"][k0][:, :1]).max((0, 2))                        # [C] largest difference to the d = 0 read
    assert dg[0] == 0.0 and (dg[1:] > 1e-6).all(), dg
    # d = 0 alone with no fallback = the actions of the plan: identical to the scale family c = 1.0 (cap-only, no fallback), read for read and path for path
    d0 = make_ip(T, "cand-imag", None, model, cset=(0.0,), fallback="none", family="decel")
    sc = make_ip(T, "cand-imag", None, model, cset=(1.0,), fallback="none", floor="cap-only")
    o0 = run_rollout2(T, world, p, d0, ticks, cfg)
    o1 = run_rollout2(T, world, p, sc, ticks, cfg)
    assert arrays_equal(d0.ilog.arrays()["imag_raw"], sc.ilog.arrays()["imag_raw"]) and arrays_equal(o0[1], o1[1]) and not d0.ilog.arrays()["changed"].any()


def test_cli_cand_family_decel_reaches_the_rule_and_set_0_without_fallback_equals_off(T, world, tiny, tmp_path):
    _, p = _lead_join(T, world["root"], t_lead=3, gap0=35.0, name="join_cli3.npz")
    off = tmp_path / "off"
    assert _cli(T, world, tiny, off, [], join_path=p, plan="off") == 0
    dec = tmp_path / "dec"
    assert _cli(T, world, tiny, dec, ["--cand-family", "decel", "--imag-log", str(dec / "ticks" / "imag_log_t.npz")], join_path=p, plan="cand-label") == 0
    z = np.load(dec / "ticks" / "imag_log_t.npz")
    assert str(z["family"]) == "decel" and z["cand_set"].tolist() == list(IP.DECEL_CARD) and z["active"].sum() > 0
    ch = z["chosen_c"][z["active"]]
    assert set(np.unique(ch[np.isfinite(ch)])) <= set(IP.DECEL_CARD) and z["gap_h"].shape[1] == len(IP.DECEL_CARD)
    assert _dump_bytes(dec, "t") != _dump_bytes(off, "t")                                       # the rule changed the executed world
    sc = tmp_path / "sc"
    assert _cli(T, world, tiny, sc, ["--imag-log", str(sc / "ticks" / "imag_log_t.npz")], join_path=p, plan="cand-label") == 0
    assert str(np.load(sc / "ticks" / "imag_log_t.npz")["family"]) == "scale"
    ident = tmp_path / "ident"
    assert _cli(T, world, tiny, ident, ["--cand-family", "decel", "--cand-set", "0", "--fallback", "none", "--imag-log", str(ident / "ticks" / "imag_log_t.npz")], join_path=p, plan="cand-label") == 0
    zi = np.load(ident / "ticks" / "imag_log_t.npz")
    assert zi["active"].sum() > 0 and not zi["changed"].any() and zi["cand_set"].tolist() == [0.0]
    assert _dump_bytes(ident, "t") == _dump_bytes(off, "t")
    with pytest.raises(SystemExit, match="would change nothing"):
        _cli(T, world, tiny, tmp_path / "x", ["--cand-family", "decel"], plan="off")
    with pytest.raises(ValueError, match="decel set"):
        _cli(T, world, tiny, tmp_path / "x2", ["--cand-family", "decel", "--cand-set", "-1"], plan="cand-cv")
    with pytest.raises(SystemExit, match="scale family only"):
        _cli(T, world, tiny, tmp_path / "y", ["--cand-family", "decel", "--floor", "cap-only"], plan="cand-cv")


# ============================================================================================ source cand-imagkin (note PREREG_V8F_PROOF_ADD40_N4 section 3; written by the lead)
def test_imagkin_gap_is_the_imagined_level_of_the_plan_plus_the_known_differences():
    rng = np.random.default_rng(31)
    n = 60
    v0 = rng.uniform(2, 22, n)
    plans = np.stack([profile(v, a=rng.uniform(-2, 1.5)) for v in v0])
    g, off, vl = rng.uniform(0, 60, n), rng.uniform(-3, 3, n), rng.uniform(0, 18, n)
    act = rng.random(n) < 0.8
    ia = np.nonzero(act)[0]
    C = len(IP.effective_set(IP.DECEL_CARD, "zero", "decel"))
    raw = rng.uniform(-10, 70, (len(ia), C, 2))                                              # an arbitrary imagination: a different gap for every candidate
    o = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-imagkin", imag_fn=lambda c, i: raw.copy(), family="decel", cset=IP.DECEL_CARD, shift_m=-1.5)
    i0 = o.cset.index(0.0)
    assert np.array_equal(o.imag_raw[ia], raw)                                               # every candidate's roll is logged
    lvl = raw[:, [i0], :] - 1.5 + off[ia][:, None, None]
    assert np.abs(o.gap_h[ia][:, [i0], :] - lvl).max() < 1e-12                               # the level: the imagined gap of the plan + shift + own progress
    kin = o.a_h[ia][:, [i0], :] - o.a_h[ia]
    assert np.abs((o.gap_h[ia] - o.gap_h[ia][:, [i0], :]) - kin).max() < 1e-9                # control k1: the response is the known one
    assert (kin >= -1e-12).all() and kin.max() > 1.0
    assert len(set(o.chosen_idx[ia].tolist())) > 2
    # the decision reads only the plan's column: the other columns of the imagination change nothing
    raw2 = np.full(raw.shape, 1e6)
    raw2[:, i0] = raw[:, i0]
    o2 = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-imagkin", imag_fn=lambda c, i: raw2.copy(), family="decel", cset=IP.DECEL_CARD, shift_m=-1.5)
    assert np.array_equal(o2.chosen_idx, o.chosen_idx) and np.array_equal(o2.plan, o.plan) and np.array_equal(o2.gap_h, o.gap_h, equal_nan=True)
    # and it is not cand-imag: with the same imagination the pure source decides differently somewhere
    oi = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-imag", imag_fn=lambda c, i: raw.copy(), family="decel", cset=IP.DECEL_CARD, shift_m=-1.5)
    assert not np.array_equal(oi.chosen_idx, o.chosen_idx)


def test_imagkin_with_a_planted_level_equals_cand_cv_and_refusals():
    rng = np.random.default_rng(32)
    n = 70
    v0 = rng.uniform(0, 22, n)
    plans = np.stack([profile(v, a=rng.uniform(-3, 2)) for v in v0])
    g, off, vl = rng.uniform(0, 60, n), rng.uniform(-3, 3, n), rng.uniform(0, 18, n)
    act = np.ones(n, bool)
    i0 = IP.effective_set(IP.DECEL_CARD, "zero", "decel").index(0.0)

    def planted(cand, ia):                                                                   # only the plan's column is planted; the others are NaN and must not matter
        out = np.full(cand.shape[:2] + (2,), np.nan)
        for k, b in enumerate(ia):
            out[k, i0] = [g[b] + vl[b] * 1.0 - arc_at(plans[b], 5), g[b] + vl[b] * 2.0 - arc_at(plans[b], 10)]
        return out
    ok_ = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-imagkin", imag_fn=planted, family="decel", cset=IP.DECEL_CARD)
    ov = IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-cv", family="decel", cset=IP.DECEL_CARD)
    assert np.abs(ok_.gap_h - ov.gap_h).max() < 1e-4 and np.array_equal(ok_.chosen_idx, ov.chosen_idx) and np.array_equal(ok_.plan, ov.plan)
    assert len(set(ov.chosen_idx.tolist())) > 2 and not ok_.n_nonfinite.any()
    with pytest.raises(ValueError):
        IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-imagkin", imag_fn=planted, family="scale")
    with pytest.raises(ValueError):
        IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-imagkin", imag_fn=planted, family="decel", cset=(0.5, 1.0, 6.0))
    with pytest.raises(ValueError):
        IP.imag_plan_rule(plans, v0, act, g, off, vl, "cand-imagkin", family="decel", cset=IP.DECEL_CARD)


def test_imagkin_e2e_runs_on_the_tiny_model_keeps_k1_and_the_p45_log_has_the_second_diagnostic(T, world, tiny):
    cfg, model = tiny["cfg"], tiny["model"]
    _, p = _lead_join(T, world["root"], name="join_head8.npz", const_gap=20.0)
    ticks = tuple(range(3, 40, 4))
    kin = make_ip(T, "cand-imagkin", None, model, cset=IP.DECEL_CARD, family="decel", p45={})
    run_rollout2(T, world, p, kin, ticks, cfg, p45=True)
    z = kin.ilog.arrays()
    act = z["active"]
    i0 = list(IP.effective_set(IP.DECEL_CARD, "zero", "decel")).index(0.0)
    assert act.sum() > 5 and np.isfinite(z["imag_raw"][act]).all() and np.isfinite(z["gap_h"][act]).all()
    d_gap = z["gap_h"][act] - z["gap_h"][act][:, [i0], :]
    d_kin = z["a_h"][act][:, [i0], :] - z["a_h"][act]
    assert np.abs(d_gap - d_kin).max() < 1e-9 and d_kin.max() > 0.5                          # control k1 on the wrapper's own log
    # at tick 0 the level equals the level of a cand-imag run of the same model (control k2: the same states and the same plan there)
    pure = make_ip(T, "cand-imag", None, model, cset=IP.DECEL_CARD, family="decel")
    run_rollout2(T, world, p, pure, ticks, cfg)
    y = pure.ilog.arrays()
    k0a, k0b = act & (z["k"] == 0), y["active"] & (y["k"] == 0)
    assert k0a.sum() > 2 and np.array_equal(k0a, k0b)
    assert np.abs(z["gap_h"][k0a][:, i0] - y["gap_h"][k0b][:, i0]).max() < 1e-4
    # the p4/p5 log carries the plan's actions and the four baselines of the second diagnostic (base + three offsets, two horizons)
    q = {k: np.concatenate(v) for k, v in kin.p45.items()}
    n = len(q["ci"])
    assert q["act_plan"].shape == (n, IP.N_ROLL, 2) and np.isfinite(q["act_plan"]).all()
    for nm in WI.P45_BASES:
        h = q["imag_gap_hyb_" + nm]
        assert h.shape == (n, 1 + len(WI.P45_OFFSETS), 2) and np.isfinite(h).all()
        assert np.abs(h[:, 1:] - h[:, :1]).max() > 1e-6                                      # an offset reaches the roll
    # pp with no offset is the plan's own roll: at tick 0 it equals the d = 0 read of the run on the active windows
    key = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(q["ci"], q["t0"]))}
    rows = [key[(int(c), int(t))] for c, t in zip(z["ci"][k0a], z["t0"][k0a])]
    assert np.abs(q["imag_gap_hyb_pp"][rows, 0] - z["imag_raw"][k0a][:, i0]).max() < 1e-4
