"""Smoke test of run_v4.do_state + summarize_v4 on ONE USED state with 1.0 s rollouts (not evidence). usage: smoke_v4.py OUTDIR"""
import os, sys, json, gzip
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v4/scorer_gym")
import gym_env as g, run_v4 as r
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
orig = g.GymState.rollout
g.GymState.rollout = lambda self, *a, **k: orig(self, *a, **{**k, "horizon_s": 1.0})
st = g.select_states(20, 10)[1]
for cmd in ("main", "ttcoff"):
    d = os.path.join(out, "run_" + cmd); os.makedirs(d, exist_ok=True)
    res = r.do_state((0, st, cmd))
    with gzip.open(os.path.join(d, "states.jsonl.gz"), "wt") as f:
        f.write(json.dumps(res) + "\n")
    json.dump(dict(total_wall_s=res["wall_task"], workers=1, n_states=1, cmd=cmd), open(os.path.join(d, "run_meta.json"), "w"))
    print(cmd, "ok", res["wall_task"], flush=True)
