"""Deliberate-regression arms of tests/test_t2v8f_stage3.py (AGENT K). Each arm is run unmutated first (must pass), then with its mutation (must raise AssertionError = RED).
  --fast   the arms that need no rehearsal fixture (draw, classes, refs, manifest, real guard npz, pool, footprint, artifact)
  (default) also the rehearsal arms (missing npz, changed byte, stage-2 manifest, busy file, requires, two roots + the wrong-drive driver, resume, sftp, positive)
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I code/run_mutations_stage3.py [--fast]   (the full run is long: run it detached and read logs/mutations_stage3.log)"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import test_t2v8f_stage3 as T  # noqa: E402

FAST = [("check_draw", "seed2"), ("check_classes", "threshold"), ("check_refs_real", "bad_listing"), ("check_real_guard", "no_npz"), ("check_manifest", "not_applied"),
        ("check_pool", "idx_reused"), ("check_footprint", "off_by_one"), ("check_artifact", "extra")]
SLOW = [("check_missing_npz", "not_applied"), ("check_changed_byte", "not_applied"), ("check_stage2_manifest", "not_applied"), ("check_busy_file", "not_applied"), ("check_requires", "not_applied"),
        ("check_two_roots", "wrong_drive"), ("check_resume", "none_removed"), ("check_sftp", "stage2_root"), ("check_positive", "cell_dropped")]
plan = FAST if "--fast" in sys.argv else FAST + SLOW
for fn, _ in plan:
    print(f"baseline {fn}:", str(getattr(T, fn)())[:160], flush=True)
red = 0
for fn, mut in plan:
    try:
        getattr(T, fn)(mut)
        print(f"{fn}[{mut}]: GREEN  <-- the mutation was not caught", flush=True)
    except AssertionError as e:
        red += 1
        print(f"{fn}[{mut}]: RED (AssertionError: {str(e)[:150]})", flush=True)
print(f"mutations RED: {red} of {len(plan)}")
sys.exit(0 if red == len(plan) else 1)
