import json, os, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import gym_env as g

IDX = 1  # a moving state (v0 ~ 10 m/s)


def test_state_selection_seeded_and_navtest_free():
    a, b = g.select_states(20, 10), g.select_states(20, 10)
    assert a == b and len(a) == 20 and len({s["log"] for s in a}) >= 5
    nt = g.navtest_log_names()
    assert len(nt) == 147 and not any(s["log"] in nt for s in a)


_S = None


def make_S():
    return g.GymState(g.select_states(20, 10)[IDX])


def test_action_fan_is_24(S):
    assert len(S.proposals) == 24 and S.names[:2] == ["pdm0", "pdm1"] and S.names[-1].startswith("sweep")


def test_reward_is_multiplicative_and_hard_zero(S):
    logs = S.rollout(5, "log", horizon_s=2.0)
    for r in logs:
        if r["hard"]:
            assert r["reward"] == 0.0
        else:
            import math
            assert abs(r["reward"] - r["progress_norm"] * math.prod(r["q"].values())) < 1e-12
            assert set(r["q"]) == set(g.TERMS)


def test_do_nothing_gets_zero_progress_reward(S):
    ft = S.first_traj_for("stop")
    logs = S.rollout(-1, "log", horizon_s=5.0, policy="stop", first_traj=ft)
    pdm = S.rollout(5, "log", horizon_s=5.0)
    assert sum(r["reward"] for r in logs) < sum(r["reward"] for r in pdm)   # braking momentum still earns some progress
    stopped = [r for r in logs if r["v"] < 0.05]
    assert stopped and all(r["reward"] <= 1e-3 for r in stopped)            # once stopped, reward ~ 0


def test_slowest_is_slower_than_pdm(S):
    ft = S.first_traj_for("slowest")
    slow = S.rollout(-1, "log", horizon_s=3.0, policy="slowest", first_traj=ft)
    pdm = S.rollout(5, "log", horizon_s=3.0)
    assert slow[-1]["v"] < pdm[-1]["v"]


def test_deterministic_same_process_and_fresh_process(S):
    a = [(r["reward"], r["qprod"], r["v"]) for r in S.rollout(5, "log", horizon_s=2.0)]
    b = [(r["reward"], r["qprod"], r["v"]) for r in S.rollout(5, "log", horizon_s=2.0)]
    assert a == b
    ia = [(r["reward"], r["qprod"], r["v"]) for r in S.rollout(5, "idm", horizon_s=2.0)]
    ib = [(r["reward"], r["qprod"], r["v"]) for r in S.rollout(5, "idm", horizon_s=2.0)]
    assert ia == ib


_ST = {}


def state_at(idx):
    if idx not in _ST:
        _ST[idx] = g.GymState(g.select_states(20, 10)[idx])
    return _ST[idx]


def test_v2_item1_start_pose_map_slack_states_8_and_19():
    # pilot: BOTH states ended at cycle 0 on all 24 arms x 2 modes. Cause (MEASURED, M_GYM_2_RESULT.md s1): 1-2 footprint corners of the
    # LOGGED start pose lie 0.10-0.16 m outside the union of drivable polygons; the strict all-corners test fires on step 0.
    for idx in (8, 19):
        S = state_at(idx)
        ev = S.evaluator
        assert 0.0 < ev.t0_overshoot_m < 0.2 and 0.0 < ev.map_slack_m <= g.MAX_MAP_SLACK_M and not ev.map_unreliable, (idx, ev.t0_overshoot_m)
        k = int(np.argmax(S.pdm_scores))
        slack = ev.map_slack_m
        ev.map_slack_m = 0.0                                   # = M-GYM-1 behaviour
        try:
            old = S.rollout(k, "log", horizon_s=1.0)
        finally:
            ev.map_slack_m = slack
        assert old[0]["hard"] == "off_drivable" and old[0]["cycle"] == 0
        seen, orig = [], ev.cycle
        def _spy(ctx, egos, objs, tl, npl):
            seen.extend(ev.footprint_overshoot(e) for e in egos)
            return orig(ctx, egos, objs, tl, npl)
        ev.cycle = _spy
        try:
            new = S.rollout(k, "log", horizon_s=1.0)
        finally:
            del ev.cycle
        if idx == 8:
            assert new[0]["hard"] is None and len(new) == 3, (idx, new[0]["hard"], len(new))
        else:
            # NARROWED after run 2 (state 19; M_GYM_2_RESULT.md Part A): the t0 pose (0.1555 m) is inside the slack (0.1587 m) but the
            # arm's OWN first ticks reach 0.168-0.170 m, 1 cm beyond the human log's worst: a real off-drivable event of the arm under
            # the pre-registered rule. This test guards against FALSE termination from the start pose, so an event is acceptable only
            # if the arm's measured footprint overshoot exceeds the slack in force.
            h = new[0]["hard"]
            assert h is None or (h == "off_drivable" and max(seen) > ev.map_slack_m), (idx, h, len(new), max(seen))


def test_v2_item1_collision_uses_pdm_at_fault_rule_state_2():
    # pilot: log-replay vehicle hit ego from behind-right (local -3.3, +2.4 m, ego 2.2 m/s, agent 4.8 m/s) -> "collision" at cycle 13 on all 24 arms
    S = state_at(2)
    ev = S.evaluator
    k = int(np.argmax(S.pdm_scores))
    ev.collision_rule = "legacy"
    try:
        old = S.rollout(k, "log")
    finally:
        ev.collision_rule = "navsim"
    assert old[-1]["hard"] == "collision"
    new = S.rollout(k, "log")
    assert new[-1]["hard"] != "collision", new[-1]["hard"]


# ======================================================================= M-GYM-2 tests (each fails on the pilot code)
import numpy as np
import sqlite3


def test_v2_jerk_uses_nuplan_smoothed_estimate():
    # one 0.5 m/s^2 acceleration step between two 0.1 s samples: raw difference = 5 m/s^3 > 4.13 limit (M-GYM-1 fired),
    # nuPlan's Savitzky-Golay estimate (state_extractors.py:187-248) smooths it well below the limit.
    n = 30
    ax = np.concatenate([np.zeros(n // 2), 0.5 * np.ones(n // 2)])
    ay = np.zeros(n)
    t = (np.arange(n) * 100_000).astype(np.int64)
    assert g.jerk_channels_raw(ax, ay) == {"lon_jerk"}
    viol, mx, mm = g.jerk_channels_nuplan(t, ax, ay, 5)
    assert viol == set() and mx < 4.13
    # a genuine harsh event (6 m/s^2 step over 0.3 s, i.e. ~20 m/s^3) must still fire under the smoothed estimator
    ax2 = np.concatenate([np.zeros(24), [2, 4, 6, 6, 6, 6]]).astype(float)
    viol2, mx2, _ = g.jerk_channels_nuplan(t, ax2, np.zeros(n), 6)
    assert "lon_jerk" in viol2 and mx2 > 4.13


def test_v2_brake_reference_distance_analytic():
    f = g.RewardEvaluator.brake_ref_dist
    assert abs(f(10.0, 0.0, 0.5, 2.0) - (10 * 0.5 - 0.25)) < 1e-12          # still braking inside the cycle
    assert abs(f(2.0, 0.0, 5.0, 2.0) - 1.0) < 1e-12                          # v^2/(2a) = 1 m, stops at t=1 s
    assert f(2.0, 3.0, 0.5, 2.0) == 0.0                                       # after the reference stop: no allowance


def test_v2_do_nothing_fails_the_old_bar_only_under_legacy_reward(S):
    ft = S.first_traj_for("stop")
    ev = S.evaluator
    ev.brake_baseline = None
    legacy = S.rollout(-1, "log", horizon_s=8.0, policy="stop", first_traj=ft)
    ev.brake_baseline = g.BRAKE_BASELINE_DECEL
    new = S.rollout(-1, "log", horizon_s=8.0, policy="stop", first_traj=ft)
    lm, nm = np.mean([r["reward"] for r in legacy]), np.mean([r["reward"] for r in new])
    assert lm > 0.02 >= nm, (lm, nm)
    assert all(r["progress_norm"] <= r["progress_norm_legacy"] + 1e-12 for r in new)
    for r in new:                                                               # CaRL multiplicative form kept
        if not r["hard"]:
            assert abs(r["reward"] - r["progress_norm"] * np.prod(list(r["q"].values()))) < 1e-12


def test_v2_stall_sampler_is_seeded_rule_based_and_tagged():
    a, b = g.select_states_stall(20, 10), g.select_states_stall(20, 10)
    assert a == b and len(a) == 20 and len({s["log"] for s in a}) == 10
    assert not any(s["log"] in g.navtest_log_names() for s in a)
    for s in a:
        con = sqlite3.connect(f"file:{s['db']}?mode=ro", uri=True)
        tags = {r[0] for r in con.execute("select type from scenario_tag t join lidar_pc l on l.token=t.lidar_pc_token where l.token=?",
                                          (bytes.fromhex(s["token"]),))}
        con.close()
        want = g.STOP_AND_GO_TAGS if s["sampler"] == "stop_and_go" else g.JUNCTION_TAGS
        assert tags & set(want), (s["log"], s["sampler"], tags)


def test_v2_hold_cycles_halves_replans(S):
    calls = []
    orig = g.GymPDM.compute_planner_trajectory
    g.GymPDM.compute_planner_trajectory = lambda self, pin: (calls.append(1), orig(self, pin))[1]
    try:
        S.rollout(5, "log", horizon_s=3.0)                     # 7 cycles -> 6 replans
        n1 = len(calls); calls.clear()
        S.rollout(5, "log", horizon_s=3.0, hold_cycles=2)      # replans at cycles 1,3,5 -> 3
        n2 = len(calls)
    finally:
        g.GymPDM.compute_planner_trajectory = orig
    assert (n1, n2) == (6, 3), (n1, n2)


if __name__ == "__main__":   # no pytest in the eval venv: minimal runner
    import time
    fails = 0
    S = make_S()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            try:
                fn(S) if fn.__code__.co_argcount else fn()
                print("PASS", name, "%.1fs" % (time.time() - t), flush=True)
            except Exception as e:
                fails += 1
                print("FAIL", name, repr(e), flush=True)
    print("TESTS_DONE fails=%d" % fails)
    sys.exit(fails)
