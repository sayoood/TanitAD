"""Smoke test of run_v6._do_state + next_block_v6 + summarize_v6 on ONE USED state (REWARD-1 list idx 11) with 1 s rollouts. Seed 20261111 is never touched (no state list of it exists or is read:
r2.load_states_v6 is not called; MGYM6_DRY=1). Not evidence. usage: smoke_v6.py OUTDIR"""
import os, sys, json, gzip
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ["MGYM6_DRY"] = "1"; sys.dont_write_bytecode = True
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scorer_gym")
sys.path.insert(0, V)
import gym_env as g, run_v6 as r, reward1 as r1, reward2 as r2
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
orig = g.GymState.rollout
g.GymState.rollout = lambda self, *a, **k: orig(self, *a, **{**k, "horizon_s": 1.0})
st = r1.load_states_v5()[11]
for cmd, d, rng in (("main", "run_main", [0, 20]), ("ttcoff", "run_ttcoff", [0, 20]), ("ladder", "ladder_20_21", [20, 21])):
    dd = os.path.join(out, d); os.makedirs(dd, exist_ok=True)
    res = r._do_state(0 if cmd != "ladder" else 20, st, cmd, 0.20, r2.CURVE_MIN_SEG_M)
    assert "error" not in res, res
    with gzip.open(os.path.join(dd, "states.jsonl.gz"), "wt", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    json.dump(dict(total_wall_s=res["wall_task"], workers=1, n_states=1, n_errors=0, cmd=cmd, seed=r2.SEED_V6, floor=0.20, curve_min_seg_m=r2.CURVE_MIN_SEG_M, range=rng),
              open(os.path.join(dd, "run_meta.json"), "w", encoding="utf-8"))
    print(cmd, "ok", res["wall_task"], flush=True)
S = g.GymState(st, fixes="on", q_free_floor=0.20, curve_min_seg_m=r2.CURVE_MIN_SEG_M)
f = lambda m: [(x["reward"], x["qprod"], x["v"], x["progress_norm"], x["hard"]) for x in S.rollout(5, m)]
d = {m: [f(m), f(m)] for m in ("log", "idm")}
for i in (1, 2):
    open(os.path.join(out, f"det{i}.txt"), "w", encoding="utf-8").write("JSON" + json.dumps(d) + "\n")
