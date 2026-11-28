"""Deliberate-regression arms of tests/test_t2v8f_stage4.py (AGENT N designed, AGENT O wrote). Each arm is run unmutated first (must pass), then with its mutation (must raise AssertionError = RED).
  --fast   the arms that need no rehearsal fixture (manifest, order, pairing, artifact, real npz path)
  (default) also the rehearsal arms (positive, changed byte, scenes absent, busy files, RI copy, requires, wrong manifest, missing ckpt, composite off / wrong prior, contract, footprint)
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 python -I code/run_mutations_stage4.py [--fast]   (the full run is long: run it detached and read logs/mutations_stage4.log)"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import test_t2v8f_stage4 as T  # noqa: E402

FAST = [("check_manifest", "not_applied"), ("check_order", "beside_first"), ("check_pairing", "shifted"), ("check_artifact", "extra"), ("check_real_path", "no_npz")]
SLOW = [("check_positive", "cell_dropped"), ("check_changed_byte", "not_applied"), ("check_scenes_absent", "not_applied"), ("check_busy_files", "not_applied"), ("check_ri_copy", "not_applied"),
        ("check_requires", "not_applied"), ("check_wrong_manifest", "not_applied"), ("check_missing_ckpt", "not_applied"), ("check_composite", "composite_off"), ("check_composite", "wrong_prior"),
        ("check_contract", "not_applied"), ("check_footprint", "off_by_one")]
plan = FAST if "--fast" in sys.argv else FAST + SLOW
seen = set()
for fn, _ in plan:
    if fn not in seen:
        seen.add(fn)
        print(f"baseline {fn}:", str(getattr(T, fn)())[:160], flush=True)
red = 0
for fn, mut in plan:
    try:
        getattr(T, fn)(mut)
        print(f"{fn}[{mut}]: GREEN  <-- the mutation was not caught", flush=True)
    except AssertionError as e:
        red += 1
        print(f"{fn}[{mut}]: RED (AssertionError: {str(e)[:150]})", flush=True)
    except Exception as e:                                                          # noqa: BLE001  a crash is NOT a deliberate RED: it is printed and counted as not caught
        print(f"{fn}[{mut}]: ERROR <-- crashed, not a deliberate RED ({type(e).__name__}: {str(e)[:150]})", flush=True)
print(f"mutations RED: {red} of {len(plan)}")
sys.exit(0 if red == len(plan) else 1)
