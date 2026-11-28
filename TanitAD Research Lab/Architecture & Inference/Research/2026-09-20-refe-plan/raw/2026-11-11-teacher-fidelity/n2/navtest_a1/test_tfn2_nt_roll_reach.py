#!/usr/bin/env python3
"""TF-N2 navtest roll: reachability and default-identity tests for the ONE knob. Plain asserts (no pytest in the driverl-eval venv):
  python eval/run_p4.py --log <log> --script test_tfn2_nt_roll_reach.py -- --pass-through
CPU only. Each control has a MUTATION arm that must go RED. The end-to-end mutation (the override dropped, every row an error row, the chain's roll stage
FAIL) is run by the smoke: `TFN2_MUTATE_DROP_OVERRIDE=1` on the chain, see HANDOFF.md."""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tfn2_nt_roll as RR  # noqa: E402
import build_teacher_rollouts as BTR  # noqa: E402


def test_defaults_are_empty_and_r0_is_the_registered_build():
    assert BTR.TEACHER_OVERRIDES == {}
    assert RR.overrides_for("R0") == {}
    assert RR.expected_knobs({}) == RR.REGISTERED
    RR.install("R0")
    assert BTR.TEACHER_OVERRIDES == {}


def test_rj5_sets_exactly_the_jerk_limit_and_nothing_else():
    ov = RR.overrides_for("RJ5")
    assert ov == {"positive_jerk_limit_when_nonnegative_acc": 5.0}
    RR.install("RJ5")
    assert BTR.TEACHER_OVERRIDES == {"positive_jerk_limit_when_nonnegative_acc": 5.0}
    assert RR.expected_knobs(ov)["positive_jerk_limit_when_nonnegative_acc"] == 5.0
    assert RR.expected_knobs(ov)["route_goal_min_speed_mps"] == 5.0          # the goal-speed lever (refuted) is NOT touched


def test_no_other_arm_is_reachable():
    for arm in ("RG8", "R5HZ", "R1", "RJ10"):
        try:
            RR.overrides_for(arm)
        except KeyError:
            continue
        raise AssertionError(f"arm {arm} must not exist")


def test_mutation_drop_override_keeps_the_request_but_not_the_override():
    RR.install("RJ5", mutate=True)
    assert BTR.TEACHER_OVERRIDES == {}                                       # the build would be the registered one ...
    assert RR.knob_mismatch(RR.REGISTERED, RR.expected_knobs(RR.overrides_for("RJ5")))   # ... and the read-back check goes RED


def test_knob_mismatch_is_red_on_each_wrong_value_and_a_missing_key():
    exp = RR.expected_knobs(RR.overrides_for("RJ5"))
    good = dict(exp, max_jerk_long=5.0)
    assert RR.knob_mismatch(good, exp) == []
    for k in exp:
        bad = dict(good)
        bad[k] = bad[k] + 0.5
        assert RR.knob_mismatch(bad, exp), k
        gone = {x: v for x, v in good.items() if x != k}
        assert RR.knob_mismatch(gone, exp), k


def test_row_wrapper_adds_knobs_to_ok_rows_only():
    buf = io.StringIO()
    state = {"knobs": {"positive_jerk_limit_when_nonnegative_acc": 5.0}}
    f = RR._KnobFile(buf, state, "RJ5")
    f.write(json.dumps({"token": "a", "traj20": [1]}) + "\n")
    f.write(json.dumps({"token": "b", "error": "x"}) + "\n")
    f.flush()
    a, b = (json.loads(x) for x in buf.getvalue().splitlines())
    assert a["knobs"] == state["knobs"] and a["arm"] == "RJ5"
    assert "knobs" not in b


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            BTR.TEACHER_OVERRIDES.clear()
            v()
            n += 1
            print("PASS", k)
    print(f"ZZTFN2NT_REACH_OK {n}")
