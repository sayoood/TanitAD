"""DERIVATION (used states only): centreline-footprint overshoot (next 40 m) at all 20 main + 20 stall states of M-GYM-2.
usage: prescreen_dist.py OUT.json   (3 spawn workers)"""
import os, sys, json, multiprocessing as mp
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/pilot_v4/scorer_gym")
def work(a):
    import gym_env as g
    kind, idx = a
    sel = g.select_states if kind == "main" else g.select_states_stall
    S = g.GymState(sel(20, 10)[idx], fixes="on")
    ev = S.evaluator
    return dict(set=kind, idx=idx, centreline_overshoot=ev.centreline_overshoot_m, t0_overshoot=ev.t0_overshoot_m, human_slack=ev.map_slack_m)
if __name__ == "__main__":
    tasks = [("main", i) for i in range(20)] + [("stall", i) for i in range(20)]
    res = []
    with mp.get_context("spawn").Pool(3) as p:
        for r in p.imap_unordered(work, tasks):
            res.append(r); print(r, flush=True); json.dump(res, open(sys.argv[1], "w"))
    print("PRESCREEN_DONE", flush=True)
