import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_env as g
st = g.select_states(20, 10)[int(sys.argv[1])]
S = g.GymState(st)
out = {m: [(r["reward"], r["qprod"], r["v"], r["progress_norm"]) for r in S.rollout(5, m)] for m in ("log", "idm")}
print("JSON" + json.dumps(out))
