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
