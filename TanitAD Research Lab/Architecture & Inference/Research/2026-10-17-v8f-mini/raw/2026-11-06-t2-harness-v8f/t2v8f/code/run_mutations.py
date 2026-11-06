"""Run each deliberate regression against its gate and require an AssertionError (RED). Usage: python -I code/run_mutations.py   (CPU)"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import test_t2v8f_gates as T  # noqa: E402

CASES = [
    (T.check_g1_tick0, "stale_history", "G1 m1: NOW one cache step too old"),
    (T.check_g1_tick0, "feat_off_by_one", "G1 m2: the NOW frame's tokens are the previous step's"),
    (T.check_g1_tick0, "no_yaw_rate", "G1 m3: the curvature/yaw-rate feature zeroed"),
    (T.check_g1_tick0, "limit_off", "G1 m4: the causal speed bucket dropped"),
    (T.check_g1_ticks, "no_deadreckon", "G1 m5: nav held at t0 instead of dead-reckoned"),
    (T.check_g1_hacc, "hacc_noop", "G1 m6: the HACC post-map has no effect"),
    (T.check_g3_zero, "returns_model", "G3 m1: the P10 arm returns the model plan"),
    (T.check_g3_zero, "prior_eps", "G3 m2: the P10 arm is off by 1e-6"),
    (T.check_g4_keys, "guard_off", "G4 m1: the key guard removed"),
]
try:
    import t2a_g2
    CASES += [(lambda mut=None, _m=m: t2a_g2.run_g2(mut=_m, assert_pass=True), m, f"G2 {m}") for m in t2a_g2.MUTATIONS]
except ImportError:
    print("G2 module not available: G2 mutations not run")
print("MUT unmutated:", end=" ")
for fn, name in ((T.check_g1_tick0, "G1"), (T.check_g1_ticks, "G1t"), (T.check_g1_hacc, "G1h"), (T.check_g3_zero, "G3"), (T.check_g4_keys, "G4")):
    fn()
    print(name, "PASS", end="; ")
print()
bad = 0
for fn, mut, why in CASES:
    try:
        fn(mut=mut)
        print(f"MUT {why}: SURVIVED (NOT RED)")
        bad += 1
    except AssertionError as e:
        print(f"MUT {why}: RED  AssertionError: {str(e)[:160]}")
    except Exception as e:  # a crash is not a RED verdict
        print(f"MUT {why}: CRASH {type(e).__name__}: {str(e)[:160]} (counted NOT RED)")
        bad += 1
print("MUTATIONS ALL RED" if bad == 0 else f"MUTATIONS NOT ALL RED ({bad})")
sys.exit(1 if bad else 0)
