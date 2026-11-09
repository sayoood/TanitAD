"""usage: determinism_child_v5.py IDX FLOOR -> prints JSON{log,idm: [run1, run2]} (fixes ON, arm 5, fresh REWARD-1 state IDX); run1 vs run2 = in-process repeat."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_env as g
import reward1 as r1
st = r1.load_states_v5()[int(sys.argv[1])]
S = g.GymState(st, fixes="on", q_free_floor=float(sys.argv[2]))
f = lambda m: [(r["reward"], r["qprod"], r["v"], r["progress_norm"], r["hard"]) for r in S.rollout(5, m)]
print("JSON" + json.dumps({m: [f(m), f(m)] for m in ("log", "idm")}))
