#!/usr/bin/env python3
"""TF-N2 re-roll: reachability and default-identity tests for the protocol knobs. Plain asserts (no pytest in the driverl-eval venv):
  python eval/run_p4.py --log <log> --script test_tfn2_reroll_reach.py -- --pass-through
CPU only. Each control has a MUTATION arm that must make its check go RED (the check is the one the roll stage uses: knob_mismatch)."""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tfn2_reroll_roll as RR  # noqa: E402
import build_teacher_rollouts as BTR  # noqa: E402


def ns(**kw):
    d = {"route_goal_min_speed": None, "route_goal_horizon": None, "policy_interval_s": None, "pos_jerk_limit": None}
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_defaults_are_empty():
    assert RR.overrides_from_args(ns()) == {}
    assert RR.expected_knobs({}) == RR.REGISTERED
    assert BTR.TEACHER_OVERRIDES == {}


def test_flags_map_to_keys():
    ov = RR.overrides_from_args(ns(route_goal_min_speed=8.0, pos_jerk_limit=5.0, policy_interval_s=0.2, route_goal_horizon=12.0))
    assert ov == {"route_goal_min_speed_mps": 8.0, "positive_jerk_limit_when_nonnegative_acc": 5.0, "policy_interval_s": 0.2,
                  "route_goal_horizon_s": 12.0}


def test_post_init_wrapper_reaches_and_mutation_goes_red():
    class Cfg:
        positive_jerk_limit_when_nonnegative_acc = 1.0

    class Fake:
        def __init__(self):
            self.config = Cfg()

        def initialize(self, _):                                  # what DriveRLNuPlanPlanner.initialize does on its first call
            self.config.positive_jerk_limit_when_nonnegative_acc = 1.0
    p = Fake()
    BTR._install_post_init_overrides(p, {"positive_jerk_limit_when_nonnegative_acc": 5.0})
    p.initialize(None)
    assert p.config.positive_jerk_limit_when_nonnegative_acc == 5.0
    q = Fake()                                                    # MUTATION: wrapper not installed
    q.initialize(None)
    bad = RR.knob_mismatch({"positive_jerk_limit_when_nonnegative_acc": q.config.positive_jerk_limit_when_nonnegative_acc},
                           {"positive_jerk_limit_when_nonnegative_acc": 5.0})
    assert bad, "mutation (override dropped) did NOT make the check go red"
    r = Fake()                                                    # default: no override => initialize is the class method (bit-identical)
    BTR._install_post_init_overrides(r, {})
    assert "initialize" not in r.__dict__


def test_real_planner_build():
    dev = "cpu"
    for key, val in (("route_goal_min_speed_mps", 8.0), ("policy_interval_s", 0.2), ("route_goal_horizon_s", 10.0)):
        BTR.TEACHER_OVERRIDES.clear()
        BTR.TEACHER_OVERRIDES[key] = val
        pl = BTR.build_planner(dev)
        built = BTR.effective_knobs(pl)
        exp = RR.expected_knobs({key: val})
        exp.pop("positive_jerk_limit_when_nonnegative_acc")
        assert not RR.knob_mismatch(built, exp), (key, built)
        assert built[key] == val
        saved = BTR._PLANNER_CFG_KEYS                             # MUTATION: the override is dropped inside build_planner
        BTR._PLANNER_CFG_KEYS = ()
        try:
            pm = BTR.build_planner(dev)
            assert RR.knob_mismatch(BTR.effective_knobs(pm), exp), f"mutation did not go red for {key}"
        finally:
            BTR._PLANNER_CFG_KEYS = saved
    BTR.TEACHER_OVERRIDES.clear()
    d = BTR.build_planner(dev)                                    # DEFAULT build is the registered protocol, un-wrapped
    k = BTR.effective_knobs(d)
    assert not RR.knob_mismatch(k, RR.REGISTERED), k
    assert "initialize" not in d.__dict__


if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            try:
                f()
                print("PASS", n)
            except Exception as e:                                # noqa: BLE001
                fails += 1
                print("FAIL", n, type(e).__name__, e)
    print("ZZTFN2_TESTS", "FAIL" if fails else "PASS", fails)
    sys.exit(1 if fails else 0)
