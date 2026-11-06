"""Mutation runner for tests/test_t2v8f_audit.py: each deliberate regression must make its check raise AssertionError (RED); an unmutated run must be green.
Run from t2v8f/: CUDA_VISIBLE_DEVICES=-1 python -I audit/run_mutations_audit.py"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE + "/../tests")
import test_t2v8f_audit as T  # noqa: E402

RUNS = [("green: self-score", T.check_self_score, None), ("green: T1 path equal", T.check_t1_path_equal, None), ("green: launch class T1", T.check_launch_class_t1, None),
        ("RED: logged future 0.1 s late", T.check_self_score, "g_late"), ("RED: logged future 0.2 s late", T.check_self_score, "g_late_knot"),
        ("RED: plan rolled one tick (executor skew=1)", T.check_self_score, "skew1"), ("RED: yaw sign flipped", T.check_self_score, "yawflip"),
        ("RED: a g built 0.1 s late vs the T1 path", T.check_t1_path_equal, "g_late"), ("RED: wrong class (moving windows)", T.check_launch_class_t1, "moving")]
bad = 0
for name, fn, mut in RUNS:
    try:
        fn(mut)
        outcome = "green"
    except AssertionError as e:
        outcome = "RED"
        msg = str(e)[:110]
    want = "green" if name.startswith("green") else "RED"
    ok = outcome == want
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {name:50s} -> {outcome}" + (f"   ({msg})" if outcome == "RED" else ""))
print("MUTATIONS", "ALL AS EXPECTED" if not bad else f"{bad} NOT AS EXPECTED")
sys.exit(1 if bad else 0)
