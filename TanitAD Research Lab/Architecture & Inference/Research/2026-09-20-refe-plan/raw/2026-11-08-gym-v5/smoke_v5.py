"""Smoke test of run_v5._do_state + summarize_v5 + next_block_v5 on ONE USED state (M-GYM-4 list idx 11) with 1 s rollouts (not evidence; no fresh state is touched).
usage: smoke_v5.py OUTDIR"""
import os, sys, json, gzip, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; sys.dont_write_bytecode = True
V = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v5/scorer_gym"
sys.path.insert(0, V)
import gym_env as g, run_v5 as r
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
orig = g.GymState.rollout
g.GymState.rollout = lambda self, *a, **k: orig(self, *a, **{**k, "horizon_s": 1.0})
st = g.load_pinned_states()[11]
for cmd, d, rng in (("main", "run_main", [0, 20]), ("ttcoff", "run_ttcoff", [0, 20]), ("ladder", "ladder_20_21", [20, 21])):
    dd = os.path.join(out, d); os.makedirs(dd, exist_ok=True)
    res = r._do_state(0 if cmd != "ladder" else 20, st, cmd, 0.20)
    with gzip.open(os.path.join(dd, "states.jsonl.gz"), "wt", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    json.dump(dict(total_wall_s=res["wall_task"], workers=1, n_states=1, n_errors=0, cmd=cmd, floor=0.20, range=rng), open(os.path.join(dd, "run_meta.json"), "w", encoding="utf-8"))
    print(cmd, "ok", res["wall_task"], flush=True)
