"""Runs each E5 gate (T3 zero, T3 attribution, T4 pin) with its deliberate regressions; every one must raise AssertionError (RED). Run from t2v8f/ with the venv python -I."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import test_t2v8f_e5 as T  # noqa: E402

M = [("T3 zero", T.check_t3_zero, "zero_eps"), ("T3 zero", T.check_t3_zero, "zero_is_model"), ("T3 attribution", T.check_t3_attr, "y_shifted"),
     ("T4 pin", T.check_t4, "x_half"), ("T4 pin", T.check_t4, "m1_vmin")]
bad = 0
for name, fn, mut in M:
    print("HONEST", name, fn(), flush=True)                          # the honest arm must pass first
    try:
        fn(mut=mut)
    except AssertionError as e:
        print(f"RED   {name:15s} mutation {mut}: {str(e)[:120]}", flush=True)
    else:
        print(f"GREEN {name:15s} mutation {mut}: NOT CAUGHT", flush=True)
        bad += 1
print("MUTATIONS ALL RED" if not bad else f"{bad} NOT CAUGHT")
sys.exit(1 if bad else 0)
