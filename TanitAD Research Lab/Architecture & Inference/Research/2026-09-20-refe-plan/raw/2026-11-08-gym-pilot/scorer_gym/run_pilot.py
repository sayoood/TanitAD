"""M-GYM-1 driver. usage: run_pilot.py main|smoke|determinism --workers 6 --out DIR"""
import argparse, gzip, json, multiprocessing as mp, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gym_env as g

MODES = ("log", "idm")


def slim(logs):
    out = []
    for r in logs:
        out.append(dict(cycle=r["cycle"], hard=r["hard"], hard_step=r["hard_step"], reward=r["reward"],
                        progress=r["progress_norm"], v=r["v"], a=r["a_lon"], qprod=r["qprod"], q=r["q"],
                        active=r["active"], lead_gap=r["lead_gap"], conf_ttc=r["conf_ttc"], comf=r["comf_viol"]))
    return out


def do_state(args):
    idx, st, regress, arms = args
    t_all = time.perf_counter()
    S = g.GymState(st)
    res = dict(idx=idx, state=st, limit=S.limit, v0=float(S.ego0.dynamic_car_state.rear_axle_velocity_2d.x),
               n_obj0=len(S.obs0.tracked_objects.tracked_objects), build=dict(S.timer.t), arms={}, reg={},
               pdm_best=int(np.argmax(S.pdm_scores)), pdm_scores=[float(x) for x in S.pdm_scores])
    arm_list = list(range(len(S.proposals))) if arms is None else arms
    for mode in MODES:
        for k in arm_list:
            t0 = time.perf_counter()
            logs = S.rollout(k, mode)
            res["arms"][f"{mode}:{k}"] = dict(wall=time.perf_counter() - t0, ret=g.episode_return(logs), logs=slim(logs))
    res["timer_main"] = dict(S.timer.t)
    if regress:
        for pol, modes in (("slowest", ("log",)), ("fastest", MODES), ("stop", ("log",))):
            ft = S.first_traj_for(pol)
            for mode in modes:
                logs = S.rollout(-1, mode, policy=pol, first_traj=ft, timer=g.Timer())
                res["reg"][f"{pol}:{mode}"] = dict(ret=g.episode_return(logs), logs=slim(logs))
    res["wall_task"] = time.perf_counter() - t_all
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="F:/tanitad_scratch_mgym1/run")
    ap.add_argument("--n-states", type=int, default=20)
    ap.add_argument("--arms", type=str, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sts = g.select_states(20, 10)[: a.n_states] if a.cmd != "smoke" else g.select_states(20, 10)[:a.n_states]
    arms = None if a.arms is None else [int(x) for x in a.arms.split(",")]
    tasks = [(i, s, a.cmd == "main", arms) for i, s in enumerate(sts)]
    t0 = time.perf_counter()
    done = 0
    with mp.get_context("spawn").Pool(a.workers) as pool, gzip.open(os.path.join(a.out, "states.jsonl.gz"), "wt") as fo:
        for r in pool.imap_unordered(do_state, tasks):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            done += 1
            print(f"state {r['idx']} done wall={r['wall_task']:.1f}s ({done}/{len(tasks)})", flush=True)
    wall = time.perf_counter() - t0
    json.dump(dict(total_wall_s=wall, workers=a.workers, n_states=len(tasks)), open(os.path.join(a.out, "run_meta.json"), "w"))
    print("RUN_DONE", wall, flush=True)


if __name__ == "__main__":
    main()
