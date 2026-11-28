"""Self-tests of the HGATE arithmetic (analytic identities + deliberate-regression arms). CPU, numpy only, tiny.

  python -m pytest -q test_v8f_hgate.py --basetemp C:/Users/Admin/pt_hgate -p no:cacheprovider
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8f_hgate as G  # noqa: E402

from taniteval import v8f_consistency as C  # noqa: E402  (hblend/deps copy, put on sys.path by v8f_hblend)

KIN = lambda plan: C.plan_kinematics(plan)


def _world(n=200, seed=0):
    rng = np.random.default_rng(seed)
    P = np.cumsum(rng.uniform(0.5, 2.0, size=(n, 30, 1)), 1) * np.array([1.0, 0.05])[None, None] + rng.normal(size=(n, 30, 2)) * 0.01
    tau = P + rng.normal(size=(n, 30, 2)) * np.linspace(0.05, 1.5, 30)[None, :, None]
    gp = rng.uniform(0, 0.4, size=(n, 22))
    lon = rng.normal(size=(n, 7))
    lon[:, G.LON_FOLLOW] = -10.0                                   # FOLLOW never the argmax unless a test makes it so
    vhi = np.full(n, 1e3)                                          # no T4 unless a test lowers it
    return P, tau, gp, lon, vhi


# ------------------------------------------------------------------------------------------------ the triggers
def test_each_trigger_alone_and_the_ge_threshold():
    P, tau, gp, lon, vhi = _world()
    v = np.zeros((len(P), 30))
    base = G.triggers(gp, lon, vhi, v)
    assert not any(t.any() for t in base.values())
    g2 = gp.copy(); g2[0, G.STOP_I] = 0.5; g2[1, G.STOP_I] = 0.4999999
    assert G.triggers(g2, lon, vhi, v)["T2_stop"].tolist()[:3] == [True, False, False]             # '>= 0.5': equality fires
    g3 = gp.copy(); g3[0, G.TURN_L_I] = 0.5; g3[1, G.TURN_R_I] = 0.9; g3[2, G.TURN_L_I] = 0.49
    assert G.triggers(g3, lon, vhi, v)["T3_turn"].tolist()[:4] == [True, True, False, False]
    l1 = lon.copy(); l1[5, G.LON_FOLLOW] = 99.0
    t1 = G.triggers(gp, l1, vhi, v)["T1_follow"]
    assert t1[5] and t1.sum() == 1


def test_T4_reads_only_the_band_intervals_from_2s():
    P, tau, gp, lon, vhi = _world(20)
    v = np.zeros((20, 30))
    v[0, 9] = 50.0            # interval 9 = [1.8, 2.0): OUTSIDE the SPEED rule's band -> must NOT fire
    v[1, 10] = 50.0           # interval 10 = [2.0, 2.2): inside -> fires
    v[2, 29] = 50.0
    vhi = np.full(20, 10.0)
    t4 = G.triggers(gp, lon, vhi, v)["T4_vhi"]
    assert t4.tolist()[:4] == [False, True, True, False]
    # equality does not exceed; a non-positive predicted v_hi closes the gate (any speed >= 0 exceeds it)
    v2 = np.full((20, 30), 10.0)
    assert not G.triggers(gp, lon, vhi, v2)["T4_vhi"].any()
    assert G.triggers(gp, lon, np.full(20, -0.1), v2)["T4_vhi"].all()


def test_band_knot_is_the_evals_own():
    assert G.BAND_S / G.DT == 10 and C._BAND_KNOT == 10


def test_deliberate_regression_wrong_follow_index_changes_the_gate():
    P, tau, gp, lon, vhi = _world()
    lon[:, G.LON_FOLLOW] = 5.0
    ok = G.triggers(gp, lon, vhi, np.zeros((len(P), 30)))["T1_follow"]
    assert ok.all()
    lon_bad = np.roll(lon, 1, axis=1)               # mutate: FOLLOW read from the wrong column
    assert not G.triggers(gp, lon_bad, vhi, np.zeros((len(P), 30)))["T1_follow"].any()


# ------------------------------------------------------------------------------------------------ the gate and its identity controls
def test_gate_identities():
    P, tau, gp, lon, vhi = _world()
    w = np.stack([np.linspace(0.1, 0.9, 30), np.linspace(0.3, 0.8, 30)], 1)
    closed, _ = G.gate_plan(P, tau, w, gp, lon, vhi, KIN, force="closed")
    assert np.array_equal(closed, tau)                                    # (i) closed everywhere = B exactly (delta 0.0)
    openp, info = G.gate_plan(P, tau, w, gp, lon, vhi, KIN, force="open")
    assert np.array_equal(openp, G.H.apply_w(P, tau, w))                  # (ii) open with the banked w = H exactly
    none, _ = G.gate_plan(P, tau, w, gp, lon, vhi, KIN, use=())
    assert np.array_equal(none, openp)                                    # no trigger can close it -> the blend everywhere
    w1, _ = G.gate_plan(P, tau, np.ones((30, 2)), gp, lon, vhi, KIN, force="open")
    assert np.allclose(w1, tau, atol=1e-12)                               # w = 1 -> tau
    w0, _ = G.gate_plan(P, tau, np.zeros((30, 2)), gp, lon, vhi, KIN, force="open")
    assert np.allclose(w0, P, atol=1e-12)                                 # w = 0 -> P10


def test_gate_takes_B_exactly_where_a_trigger_fires_and_the_blend_elsewhere():
    P, tau, gp, lon, vhi = _world(50)
    w = np.full((30, 2), 0.5)
    gp[3, G.STOP_I] = 0.9
    lon[7, G.LON_FOLLOW] = 9.0
    plan, info = G.gate_plan(P, tau, w, gp, lon, vhi, KIN)
    cl = info["closed"]
    assert cl[3] and cl[7] and cl.sum() == 2
    assert np.array_equal(plan[cl], tau[cl]) and np.array_equal(plan[~cl], G.H.apply_w(P, tau, w)[~cl])


def test_T4_is_evaluated_on_tau_G_not_on_tau():
    """T4 must read the speed of the SHRUNK plan: with w = 0 the blend is P (slow), with w = 1 it is tau (fast)."""
    n = 10
    P = np.cumsum(np.full((n, 30, 1), 1.0), 1) * np.array([1.0, 0.0])[None, None]          # 5 m/s
    tau = np.cumsum(np.full((n, 30, 1), 4.0), 1) * np.array([1.0, 0.0])[None, None]        # 20 m/s
    gp = np.zeros((n, 22)); lon = np.zeros((n, 7)); lon[:, 0] = 1.0
    vhi = np.full(n, 10.0)
    _, i0 = G.gate_plan(P, tau, np.zeros((30, 2)), gp, lon, vhi, KIN)
    _, i1 = G.gate_plan(P, tau, np.ones((30, 2)), gp, lon, vhi, KIN)
    assert not i0["triggers"]["T4_vhi"].any() and i1["triggers"]["T4_vhi"].all()


# ------------------------------------------------------------------------------------------------ the fit restriction
def test_fit_restricted_to_inactive_windows_uses_only_them():
    rng = np.random.default_rng(3)
    n = 600
    P = rng.normal(size=(n, 30, 2)) * 10
    r = rng.normal(size=(n, 30, 2)) * np.linspace(0.05, 2.0, 30)[None, :, None] + 0.5
    tau = P + r
    active = np.arange(n) % 3 == 0
    y = np.where(active[:, None, None], tau, P + 0.4 * r)        # constrained windows follow tau (w=1); free windows keep 40 % of the residual (w=0.4)
    wG = G.H.fit_weights(tau[~active], P[~active], y[~active])["w"]
    wA = G.H.fit_weights(tau, P, y)["w"]
    wBad = G.H.fit_weights(tau[active], P[active], y[active])["w"]       # deliberate regression: the WRONG subset
    assert np.allclose(wG, 0.4, atol=1e-9)
    assert not np.allclose(wA, 0.4, atol=1e-3) and np.allclose(wBad, 1.0, atol=1e-9)


def test_fit_on_all_windows_reproduces_banked_weights_if_files_present():
    """PREREG identity (iii) on the real fit windows (skipped where the artifacts are absent)."""
    if not os.path.exists(f"{G.HB}/fit_windows_M2b_s0.npz"):
        pytest.skip("HBLEND fit windows not on this box")
    for seed in G.SEEDS:
        r = G.fit_identity_all(seed)
        assert r["pass"] and r["maxabs_vs_banked_weights"] <= 1e-12


def test_activity_builder_indexing_and_clipping():
    from tanitad.data import v8f_mini_data as D
    n_steps = 40
    N = n_steps
    J = {"off": np.array([0]), "n": np.array([n_steps]), "goal_y": np.zeros(N, np.uint32), "goal_w": np.zeros(N, np.uint32),
         "cons": np.full((N, 26), np.nan, np.float32), "lead_valid": np.zeros(N, np.int8), "has": np.ones(N, bool),
         "lead_gap": np.arange(N, dtype=np.float32), "limit": np.array([[0, 1, 0, 0]], np.float32), "limit_ok": np.array([True])}
    Xw = G.window_X(J, np.array([0, 0]), np.array([2, 38]), np.zeros((2, 30)), D)
    assert Xw["jrow"].tolist() == [2, 38]
    assert Xw["lead_gap_h"][0, 0] == 3.0 and Xw["lead_gap_h"][0, 29] == 32.0              # rows t+1 .. t+30
    assert (Xw["lead_gap_h"][1] == 39.0).all()                                            # clipped at T-1 = 39
    assert np.isclose(Xw["limit_ms"][0], 50 / 3.6)


def test_gt_activity_rules_match_the_consistency_module():
    """The activity of the three gate rules: STOP needs the bit, a finite stop distance and t_stop <= 6; TURN needs a side and >= 1 s overlap;
    LEAD needs lead_valid and a finite gap at some knot >= 10. Deliberate regression: removing the lead gap data empties LEAD."""
    class _Ev:                                   # the same two entry points the eval module exposes
        C = C
        @staticmethod
        def gt_constraints(X):
            from tanitad.data import v8f_mini_data as D
            return C.constraints_from_v9(D.v9_label_dict(X), X["limit_ms"], X["lead_gap_h"], X["s_h"])
    from tanitad.data import v8f_mini_data as D
    n = 6
    gi = {nm: i for i, nm in enumerate(C.GOAL22)}
    cons = np.full((n, 26), np.nan)
    gy = np.zeros(n, np.uint32); gw = np.zeros(n, np.uint32)
    ci = D.CONS_IDX
    # 0: STOP active ; 1: STOP bit but t_stop 7 s (inactive) ; 2: TURN_L, 2 s overlap (active) ; 3: TURN_L, 0.5 s overlap (inactive) ; 4: LEAD ; 5: nothing
    for i in (0, 1):
        gy[i] |= 1 << gi["STOP_POINT"]; gw[i] |= 1 << gi["STOP_POINT"]
        cons[i, ci["lon_d_reach_m"]] = 20.0
    cons[0, ci["lon_t_reach_s"]] = 4.0; cons[1, ci["lon_t_reach_s"]] = 7.0
    for i in (2, 3):
        gy[i] |= 1 << gi["TURN_L"]; gw[i] |= 1 << gi["TURN_L"]
    cons[2, ci["turn_t_start_s"]], cons[2, ci["turn_t_end_s"]] = 1.0, 3.0
    cons[3, ci["turn_t_start_s"]], cons[3, ci["turn_t_end_s"]] = 1.0, 1.5
    lead_valid = np.array([0, 0, 0, 0, 1, 0], np.int8)
    cons[4, ci["lead_tg_min_s"]] = 2.0
    lead_gap = np.full((n, 30), np.nan); lead_gap[4] = 30.0
    Xw = {"goal_y": gy, "goal_w": gw, "cons": cons, "lead_valid": lead_valid, "lead_gap_h": lead_gap, "s_h": np.cumsum(np.full((n, 30), 1.0), 1),
          "limit_ms": np.full(n, np.nan), "has": np.ones(n, bool)}
    plan = np.cumsum(np.full((n, 30, 1), 1.0), 1) * np.array([1.0, 0.0])[None, None]
    act = G.gt_activity(Xw, plan, _Ev)
    assert act["STOP"].tolist() == [True, False, False, False, False, False]
    assert act["TURN"].tolist() == [False, False, True, False, False, False]
    assert act["LEAD"].tolist() == [False, False, False, False, True, False]
    assert act["SPEED"].sum() == 0                                        # no limit, no v_hi in this toy world
    Xw_bad = dict(Xw); Xw_bad["lead_gap_h"] = np.full((n, 30), np.nan)
    assert not G.gt_activity(Xw_bad, plan, _Ev)["LEAD"].any()


# ------------------------------------------------------------------------------------------------ the outcome table
@pytest.mark.parametrize("c_ok,k_ok,want", [(True, True, "PASS"), (False, True, "CONSISTENCY NOT RESTORED"), (False, False, "CONSISTENCY NOT RESTORED"),
                                            (True, False, "GAIN LOST")])
def test_outcome_table_verbatim(c_ok, k_ok, want):
    assert G.outcome_of(c_ok, k_ok) == want
    assert G.NEXT[want]


def test_crosstab_counts():
    f = np.array([1, 1, 0, 0, 1], bool); g = np.array([1, 0, 1, 0, 1], bool)
    r = G.crosstab(f, g)
    assert (r["fired_and_active"], r["fired_not_active"], r["not_fired_active"], r["not_fired_not_active"]) == (2, 1, 1, 1)
    assert np.isclose(r["precision"], 2 / 3) and np.isclose(r["recall"], 2 / 3)
