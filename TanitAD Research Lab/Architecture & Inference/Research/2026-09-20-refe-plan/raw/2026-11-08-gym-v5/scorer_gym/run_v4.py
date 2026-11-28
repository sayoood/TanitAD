"""M-GYM-4 driver.
usage: run_v4.py main|ttcoff --workers 4 --out DIR [--set fresh|lowkey] [--n-states 20] [--only 0,3]
  main   : fixes ON, 24 arms x {log, idm}, regression policies (slowest/fastest/stop), constant-speed ladder (log), shadow legacy flag.
  ttcoff : diagnostic truth run for the ttc guard bar: ttc_mode=off + shadow flags (legacy / corridor first step), 24 arms x {log, idm}, arms only.
State list: gym_env.select_states_v4() (seed 20261109); written to states_used.json before any rollout."""
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
                        active=r["active"], lead_gap=r["lead_gap"], conf_ttc=r["conf_ttc"], comf=r["comf_viol"],
                        progress_legacy=r.get("progress_norm_legacy"), reward_legacy=r.get("reward_legacy"),
                        comf_raw_jerk=r.get("comf_raw_jerk"), lon_jerk_max=r.get("lon_jerk_max"), mag_jerk_max=r.get("mag_jerk_max"),
                        ttc_shadow=r.get("ttc_shadow")))
    return out


def do_state(args):
    idx, st, cmd = args
    t_all = time.perf_counter()
    off = cmd == "ttcoff"
    S = g.GymState(st, fixes="on", ttc_mode="off" if off else None, ttc_shadow=True)
    ev = S.evaluator
    res = dict(idx=idx, state=st, limit=S.limit, v0=float(S.ego0.dynamic_car_state.rear_axle_velocity_2d.x),
               n_obj0=len(S.obs0.tracked_objects.tracked_objects), build=dict(S.timer.t), arms={}, reg={}, ladder={},
               pdm_best=int(np.argmax(S.pdm_scores)), pdm_scores=[float(x) for x in S.pdm_scores],
               map=dict(slack=ev.map_slack_m, unreliable=bool(ev.map_unreliable), t0_overshoot=ev.t0_overshoot_m,
                        centreline_overshoot=ev.centreline_overshoot_m, prescreen_flag=bool(ev.map_unreliable_prescreen)))
    for mode in MODES:
        for k in range(len(S.proposals)):
            t0 = time.perf_counter()
            logs = S.rollout(k, mode)
            res["arms"][f"{mode}:{k}"] = dict(wall=time.perf_counter() - t0, ret=g.episode_return(logs), logs=slim(logs))
    res["timer_main"] = dict(S.timer.t)
    if not off:
        for pol, modes in (("slowest", ("log",)), ("fastest", MODES), ("stop", ("log",))):
            ft = S.first_traj_for(pol)
            for mode in modes:
                logs = S.rollout(-1, mode, policy=pol, first_traj=ft, timer=g.Timer())
                res["reg"][f"{pol}:{mode}"] = dict(ret=g.episode_return(logs), logs=slim(logs))
        t_l = time.perf_counter()
        for name, vt in S.ladder_targets():
            logs = S.ladder_rollout(vt, "log")
            res["ladder"][name] = dict(vt=vt, ret=g.episode_return(logs), logs=slim(logs))
        res["ladder_wall"] = time.perf_counter() - t_l
    res["wall_task"] = time.perf_counter() - t_all
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "ttcoff"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", type=str, default=None)
    a = ap.parse_args()
    assert a.workers <= 4
    os.makedirs(a.out, exist_ok=True)
    sts = g.load_pinned_states()
    json.dump(sts, open(os.path.join(a.out, "states_used.json"), "w", encoding="utf-8"))
    idxs = list(range(len(sts))) if a.only is None else [int(x) for x in a.only.split(",")]
    tasks = [(i, sts[i], a.cmd) for i in idxs]
    t0 = time.perf_counter()
    done = 0
    with mp.get_context("spawn").Pool(a.workers) as pool, gzip.open(os.path.join(a.out, "states.jsonl.gz"), "wt", encoding="utf-8") as fo:
        for r in pool.imap_unordered(do_state, tasks):
            fo.write(json.dumps(r) + "\n"); fo.flush()
            done += 1
            print(f"state {r['idx']} done wall={r['wall_task']:.1f}s ({done}/{len(tasks)})", flush=True)
    json.dump(dict(total_wall_s=time.perf_counter() - t0, workers=a.workers, n_states=len(tasks), cmd=a.cmd, seed=g.SEED_V4),
              open(os.path.join(a.out, "run_meta.json"), "w", encoding="utf-8"))
    print("RUN_DONE", time.perf_counter() - t0, flush=True)


if __name__ == "__main__":
    main()
