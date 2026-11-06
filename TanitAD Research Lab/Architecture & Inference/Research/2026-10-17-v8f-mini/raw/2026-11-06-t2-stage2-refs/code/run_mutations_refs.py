"""Deliberate-regression arms of tests/test_t2v8f_stage2_refs.py (AGENT J). Each arm is run unmutated first (must pass), then with its mutation (must raise AssertionError = RED).
  --fast   the REAL reader and the REAL caller without the npz, the relaunch-aware disk figure left unscaled (seconds)
  (default) also the stub-rehearsal arms: a missing npz left in place, a changed byte not applied, a MANIFEST2 left with its entries, a missing kit npz not removed, `skip` policy, and the
            IMPLEMENTATION mutation (the line that turns a missing npz into a refusal deleted from the driver, MANIFEST2 re-hashed so the driver still passes its own check)
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I code/run_mutations_refs.py [--fast]"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import test_t2v8f_stage2_refs as R  # noqa: E402

FAST = [("check_real_read", "no_npz"), ("check_real_caller", "no_npz"), ("check_need_scales", "unscaled")]
SLOW = [("check_artifact", "dropped"), ("check_positive_refs", "count"), ("check_refused_missing", "not_applied"), ("check_refused_missing", "driver_unchecked"),
        ("check_refused_kit", "not_applied"), ("check_refused_changed", "not_applied"), ("check_refused_nolist", "left_in"), ("check_refused_skip", "not_applied")]
plan = FAST if "--fast" in sys.argv else FAST + SLOW
for fn in dict.fromkeys(f for f, _ in plan):
    print(f"baseline {fn}:", str(getattr(R, fn)())[:160], flush=True)
red = 0
for fn, mut in plan:
    try:
        getattr(R, fn)(mut)
        print(f"{fn}[{mut}]: GREEN  <-- the mutation was not caught", flush=True)
    except AssertionError as e:
        red += 1
        print(f"{fn}[{mut}]: RED (AssertionError: {str(e)[:150]})", flush=True)
print(f"mutations RED: {red} of {len(plan)}")
sys.exit(0 if red == len(plan) else 1)
