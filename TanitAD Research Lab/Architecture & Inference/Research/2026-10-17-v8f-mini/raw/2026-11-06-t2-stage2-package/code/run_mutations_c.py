"""Deliberate-regression arms of tests/test_t2v8f_run.py: each must raise AssertionError (RED); the unmutated check must pass first."""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import test_t2v8f_run as T  # noqa: E402

PLAN = [("check_starts", m) for m in ("no_clamp", "one_delta", "one_guard_start", "guard_seed", "no_collapse")] + [("check_ground_correct", "e3_version"), ("check_ground_correct", "yaw_from_log"), ("check_dispatch", "cond_dropped")]
print("baseline starts:", T.check_starts())
print("baseline ground_correct:", T.check_ground_correct())
red = 0
for fn, mut in PLAN:
    try:
        getattr(T, fn)(mut)
        print(f"{fn}[{mut}]: GREEN  <-- the mutation was not caught")
    except (AssertionError, ValueError) as e:
        red += 1
        print(f"{fn}[{mut}]: RED ({type(e).__name__}: {str(e)[:110]})")
print(f"mutations RED: {red} of {len(PLAN)}")
sys.exit(0 if red == len(PLAN) else 1)
