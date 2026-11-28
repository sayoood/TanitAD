#!/usr/bin/env python3
"""TF-N3 combo arm RJ5H5 (--pos-jerk-limit 5.0 --policy-interval-s 0.2): reachability of BOTH knobs, plus mutation arms that must go RED.
Plain asserts (no pytest in the driverl-eval venv):  python eval/run_p4.py --log <log> --script test_tfn3_combo_reach.py -- --pass-through
CPU only. The check is the one the roll stage uses (tfn3_roll.knob_mismatch)."""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tfn3_roll as RR  # noqa: E402
import build_teacher_rollouts as BTR  # noqa: E402

JERK = "positive_jerk_limit_when_nonnegative_acc"


def ns(**kw):
    d = {"route_goal_min_speed": None, "route_goal_horizon": None, "policy_interval_s": None, "pos_jerk_limit": None}
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_defaults_are_empty():
    assert RR.overrides_from_args(ns()) == {}
    assert RR.expected_knobs({}) == RR.REGISTERED
    assert BTR.TEACHER_OVERRIDES == {}


def test_combo_flags_map_to_exactly_two_keys():
    ov = RR.overrides_from_args(ns(pos_jerk_limit=5.0, policy_interval_s=0.2))
    assert ov == {JERK: 5.0, "policy_interval_s": 0.2}, ov
    exp = RR.expected_knobs(ov)
    assert exp[JERK] == 5.0 and exp["policy_interval_s"] == 0.2
    assert exp["route_goal_min_speed_mps"] == 5.0 and exp["route_goal_horizon_s"] == 12.0      # nothing else moved


def test_jerk_wrapper_reaches_alongside_hz_and_mutation_goes_red():
    class Cfg:
        positive_jerk_limit_when_nonnegative_acc = 1.0
        policy_interval_s = 0.2                                   # the build-time knob, already in the config

    class Fake:
        def __init__(self):
            self.config = Cfg()

        def initialize(self, _):                                  # the real initialize overwrites the jerk limit from the release yaml
            self.config.positive_jerk_limit_when_nonnegative_acc = 1.0
    p = Fake()
    BTR._install_post_init_overrides(p, {JERK: 5.0})
    p.initialize(None)
    built = {JERK: p.config.positive_jerk_limit_when_nonnegative_acc, "policy_interval_s": p.config.policy_interval_s}
    assert not RR.knob_mismatch(built, {JERK: 5.0, "policy_interval_s": 0.2}), built
    q = Fake()                                                    # MUTATION A: jerk override dropped (wrapper not installed)
    q.initialize(None)
    bad = RR.knob_mismatch({JERK: q.config.positive_jerk_limit_when_nonnegative_acc, "policy_interval_s": q.config.policy_interval_s},
                           {JERK: 5.0, "policy_interval_s": 0.2})
    assert len(bad) == 1 and "positive_jerk" in bad[0], f"mutation A must go red on the jerk knob only: {bad}"


def test_real_planner_build_reaches_hz_and_mutation_goes_red():
    dev = "cpu"
    BTR.TEACHER_OVERRIDES.clear()
    BTR.TEACHER_OVERRIDES.update({JERK: 5.0, "policy_interval_s": 0.2})          # the RJ5H5 request
    pl = BTR.build_planner(dev)
    built = BTR.effective_knobs(pl)
    exp = RR.expected_knobs({JERK: 5.0, "policy_interval_s": 0.2})
    exp_build = {k: v for k, v in exp.items() if k != JERK}                      # the jerk knob is applied after initialize (wrapper)
    assert not RR.knob_mismatch(built, exp_build), built
    assert built["policy_interval_s"] == 0.2
    assert "initialize" in pl.__dict__, "RJ5H5 must install the post-initialize jerk wrapper on the BUILT planner"
    saved = BTR._PLANNER_CFG_KEYS                                                # MUTATION B: the hz override is dropped inside build_planner
    BTR._PLANNER_CFG_KEYS = ()
    try:
        pm = BTR.build_planner(dev)
        bad = RR.knob_mismatch(BTR.effective_knobs(pm), exp_build)
        assert bad and "policy_interval_s" in bad[0], f"mutation B must go red on policy_interval_s: {bad}"
    finally:
        BTR._PLANNER_CFG_KEYS = saved
    BTR.TEACHER_OVERRIDES.clear()
    d = BTR.build_planner(dev)                                                   # DEFAULT build is the registered protocol, un-wrapped
    assert not RR.knob_mismatch(BTR.effective_knobs(d), RR.REGISTERED)
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
    print("ZZTFN3_TESTS", "FAIL" if fails else "PASS", fails)
    sys.exit(1 if fails else 0)
