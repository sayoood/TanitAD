"""Deliberate-regression arms of tests/test_t2v8f_stage2.py. Each arm is run unmutated first (must pass), then with its mutation (must raise AssertionError = RED).
  --fast   the hold gates (G-ID, G-MOV, G-PHYS, tie), the cell switch, the order, the argv, the footprint pin and the resumable copy (seconds)
  (default) also the stub rehearsal arms (planted missing / skip policy / changed byte / DONE skipped / executor mismatch / gate RED / builder / base == direct / hold reachable)
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I code/run_mutations_stage2.py [--fast]"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import test_t2v8f_stage2 as T  # noqa: E402

FAST = [("check_g_id", "eps_0.1"), ("check_g_mov", "slow"), ("check_g_phys", "hold_disabled"), ("check_tie_documented", "same_rule"), ("check_cell_switch", "no_restore"),
        ("check_order", "beside_first"), ("check_cell_argv", "plumbing_kept"), ("check_cell_footprint", "off_by_one"), ("check_resume", "short_is_complete"),
        ("check_scorer_classes", "pooled"), ("check_start_class_rule", "audit_rule")]
SLOW = [("check_positive", "footprint"), ("check_base_equals_direct", "hold_instead"), ("check_hold_reachable", "flag_dropped"), ("check_planted_missing", "not_applied"),
        ("check_skip_policy", "not_applied"), ("check_changed_byte", "not_applied"), ("check_done_skipped", "one_not_done"), ("check_executor_mismatch", "not_applied"),
        ("check_gate_red", "not_applied"), ("check_builder_refuses_bad_plan", "not_applied")]
plan = FAST if "--fast" in sys.argv else FAST + SLOW
for fn, _ in plan:
    T.H.uninstall(T.X)
    print(f"baseline {fn}:", str(getattr(T, fn)())[:160], flush=True)
red = 0
for fn, mut in plan:
    T.H.uninstall(T.X)
    try:
        getattr(T, fn)(mut)
        print(f"{fn}[{mut}]: GREEN  <-- the mutation was not caught", flush=True)
    except AssertionError as e:
        red += 1
        print(f"{fn}[{mut}]: RED (AssertionError: {str(e)[:150]})", flush=True)
print(f"mutations RED: {red} of {len(plan)}")
sys.exit(0 if red == len(plan) else 1)
