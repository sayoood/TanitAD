"""Runs each E3 gate with a deliberate regression; every one must raise AssertionError (RED). Run from t2v8f/ with the venv python -I."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import test_t2v8f_e3 as T  # noqa: E402

M = [("T2g", T.check_t2g, "knot_shift"), ("RR index", T.check_rr_index, "rr_time_index"), ("live path", T.check_live_path, "lookup_off_by_one"),
     *[("T2 cond", T.check_t2_condition, m) for m in T.T2_MUTATIONS]]
bad = 0
for name, fn, mut in M:
    fn()                                                               # the honest arm must pass first
    try:
        fn(mut=mut)
    except AssertionError as e:
        print(f"RED   {name:10s} mutation {mut}: {str(e)[:110]}")
    else:
        print(f"GREEN {name:10s} mutation {mut}: NOT CAUGHT")
        bad += 1
print("MUTATIONS ALL RED" if not bad else f"{bad} NOT CAUGHT")
sys.exit(1 if bad else 0)
