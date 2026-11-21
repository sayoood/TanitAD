"""Deliberate-regression arms of tests/test_t2v8f_stage1.py. Each arm is run unmutated first (must pass), then with its mutation (must raise AssertionError = RED):
  planted_missing / changed_byte / roi_shifted / empty_draws_lead   mut="not_applied": the fault is not applied, so the assertion that the harness went RED must fail
  score_e2e                                                          mut="drop_p10_record" (pairing error -> X1 not OK) and mut="rows_missing" (S-POS rows missing)
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I code/run_mutations_stage1.py"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import test_t2v8f_stage1 as T  # noqa: E402

BASE = ["check_planted_missing", "check_changed_byte", "check_actor_control", "check_roi_shifted", "check_empty_draws_lead", "check_score_e2e"]
PLAN = [("check_planted_missing", "not_applied"), ("check_changed_byte", "not_applied"), ("check_roi_shifted", "not_applied"), ("check_empty_draws_lead", "not_applied"),
        ("check_score_e2e", "drop_p10_record"), ("check_score_e2e", "rows_missing")]
for fn in BASE:
    print(f"baseline {fn}:", str(getattr(T, fn)())[:200], flush=True)
red = 0
for fn, mut in PLAN:
    try:
        getattr(T, fn)(mut)
        print(f"{fn}[{mut}]: GREEN  <-- the mutation was not caught", flush=True)
    except AssertionError as e:
        red += 1
        print(f"{fn}[{mut}]: RED (AssertionError: {str(e)[:150]})", flush=True)
print(f"mutations RED: {red} of {len(PLAN)}")
sys.exit(0 if red == len(PLAN) else 1)
