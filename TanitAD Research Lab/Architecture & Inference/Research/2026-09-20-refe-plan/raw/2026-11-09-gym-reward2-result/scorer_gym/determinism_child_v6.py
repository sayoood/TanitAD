"""usage: determinism_child_v6.py IDX FLOOR CURVE_MIN_SEG_M -> prints JSON{log,idm: [run1, run2], curve_min_seg_m} (fixes ON, arm 5, fresh REWARD-2 state IDX); run1 vs run2 = in-process repeat."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_env as g
import reward2 as r2
st = r2.load_states_v6()[int(sys.argv[1])]
arc = float(sys.argv[3])
S = g.GymState(st, fixes="on", q_free_floor=float(sys.argv[2]), curve_min_seg_m=arc)
assert S.evaluator.curve_min_seg_m == arc
f = lambda m: [(r["reward"], r["qprod"], r["v"], r["progress_norm"], r["hard"]) for r in S.rollout(5, m)]
out = {m: [f(m), f(m)] for m in ("log", "idm")}
out["curve_min_seg_m"] = arc
print("JSON" + json.dumps(out))
