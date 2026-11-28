#!/usr/bin/env python3
"""TF-N3 DOSE arms RJ3 (--pos-jerk-limit 3.0) and RJ8 (8.0): reachability of the jerk knob, plus mutation arms that must go RED.
Plain asserts (no pytest in the driverl-eval venv). Run from the hold through the roll environment (nothing is written into the combo export):
  python <w>/eval/run_p4.py --log <log> --script <hold>/test_tfn3d_reach.py -- --pass-through
CPU only. The check is the one the roll stage uses (tfn3_roll.knob_mismatch); the real-roll mutation arm is the chain smoke
(TFN3_MUTATE_DROP=pos_jerk_limit on one token must write an error row KNOB-NOT-REACHED)."""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
W_EVAL = os.environ.get("TFN3_W_EVAL", "F:/tanitad_scratch/tfn3_combo/w/eval")
sys.path.insert(0, W_EVAL)
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


def test_dose_flags_map_to_exactly_one_key():
    for x in (3.0, 8.0):
        ov = RR.overrides_from_args(ns(pos_jerk_limit=x))
        assert ov == {JERK: x}, ov
        exp = RR.expected_knobs(ov)
        assert exp[JERK] == x and exp["policy_interval_s"] == 0.0          # the 5 Hz knob is NOT moved
        assert exp["route_goal_min_speed_mps"] == 5.0 and exp["route_goal_horizon_s"] == 12.0


def test_jerk_wrapper_reaches_and_mutation_goes_red():
    class Cfg:
        positive_jerk_limit_when_nonnegative_acc = 1.0
        policy_interval_s = 0.0

    class Fake:
        def __init__(self):
            self.config = Cfg()

        def initialize(self, _):                                  # the real initialize overwrites the jerk limit from the release yaml
            self.config.positive_jerk_limit_when_nonnegative_acc = 1.0
    for x in (3.0, 8.0):
        p = Fake()
        BTR._install_post_init_overrides(p, {JERK: x})
        p.initialize(None)
        assert not RR.knob_mismatch({JERK: p.config.positive_jerk_limit_when_nonnegative_acc}, {JERK: x}), x
        q = Fake()                                                # MUTATION A: override dropped (wrapper not installed) -> must go RED
        q.initialize(None)
        bad = RR.knob_mismatch({JERK: q.config.positive_jerk_limit_when_nonnegative_acc}, {JERK: x})
        assert len(bad) == 1 and "positive_jerk" in bad[0], f"mutation A must go red for {x}: {bad}"
    # the three arms must be told apart: a value-mix-up (RJ3 row reads 8.0) is also RED
    assert RR.knob_mismatch({JERK: 8.0}, {JERK: 3.0}) and RR.knob_mismatch({JERK: 3.0}, {JERK: 8.0})


def test_real_planner_build_installs_wrapper_only_when_requested():
    BTR.TEACHER_OVERRIDES.clear()
    BTR.TEACHER_OVERRIDES.update({JERK: 8.0})                                    # the RJ8 request
    pl = BTR.build_planner("cpu")
    exp = RR.expected_knobs({JERK: 8.0})
    exp_build = {k: v for k, v in exp.items() if k != JERK}                      # jerk is applied after initialize (wrapper)
    assert not RR.knob_mismatch(BTR.effective_knobs(pl), exp_build), BTR.effective_knobs(pl)
    assert "initialize" in pl.__dict__, "RJ8 must install the post-initialize jerk wrapper on the BUILT planner"
    BTR.TEACHER_OVERRIDES.clear()                                                # MUTATION: override dropped -> no wrapper
    pm = BTR.build_planner("cpu")
    assert "initialize" not in pm.__dict__, "mutation (override dropped) must leave the planner un-wrapped (RED for a dose arm)"
    assert not RR.knob_mismatch(BTR.effective_knobs(pm), RR.REGISTERED)          # default build is the registered protocol


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
    print("ZZTFN3D_TESTS", "FAIL" if fails else "PASS", fails)
    sys.exit(1 if fails else 0)
