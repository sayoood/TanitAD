"""REWARD-2 driver (PREREG_REWARD_2.md). Same as run_v5 (REWARD-1) with the curve-speed switch: `curve_min_seg_m = 0.05` is passed EXPLICITLY (--curve-min-seg-m is REQUIRED,
no default, and must equal 0.05) and recorded in every output (run_meta.json, every state record). Workers are started only while available RAM >= 7 GiB (ramguard).
usage: run_v6.py main|ttcoff|ladder --floor 0.20 --curve-min-seg-m 0.05 --workers 4 --out DIR [--range lo:hi]
  main   : states 0..19 (full-arm tier): fixes ON, 24 arms x {log, idm}, regression policies, constant-speed ladder.
  ttcoff : states 0..19, diagnostic truth run for the ttc guard bar.
  ladder : states lo..hi-1 of the fixed ordered list, only the six ladder rungs (log mode).
Failed builds go to errors.jsonl and are counted, never replaced."""
import argparse, gzip, json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gym_env as g
import reward1 as r1
import reward2 as r2
import ramguard
from run_v4 import slim, MODES


def do_state(args):
    idx, st, cmd, floor, curve_min_seg_m = args
    try:
        return _do_state(idx, st, cmd, floor, curve_min_seg_m)
    except Exception as e:
        return dict(idx=idx, error=repr(e), tb=traceback.format_exc()[-1500:], state=st, curve_min_seg_m=curve_min_seg_m)


def _do_state(idx, st, cmd, floor, curve_min_seg_m):
    assert curve_min_seg_m is not None, "REWARD-2 passes curve_min_seg_m explicitly"
    t_all = time.perf_counter()
    off = cmd == "ttcoff"
    S = g.GymState(st, fixes="on", ttc_mode="off" if off else None, ttc_shadow=True, q_free_floor=floor, curve_min_seg_m=curve_min_seg_m)
    ev = S.evaluator
    assert ev.q_free_floor == floor and ev.curve_min_seg_m == float(curve_min_seg_m)
    res = dict(idx=idx, state=st, limit=S.limit, floor=floor, curve_min_seg_m=ev.curve_min_seg_m,
               v0=float(S.ego0.dynamic_car_state.rear_axle_velocity_2d.x),
               n_obj0=len(S.obs0.tracked_objects.tracked_objects), build=dict(S.timer.t), arms={}, reg={}, ladder={},
               pdm_best=int(np.argmax(S.pdm_scores)), pdm_scores=[float(x) for x in S.pdm_scores],
               map=dict(slack=ev.map_slack_m, unreliable=bool(ev.map_unreliable), t0_overshoot=ev.t0_overshoot_m,
                        centreline_overshoot=ev.centreline_overshoot_m, prescreen_flag=bool(ev.map_unreliable_prescreen)))
    if cmd in ("main", "ttcoff"):
        for mode in MODES:
            for k in range(len(S.proposals)):
                t0 = time.perf_counter()
                logs = S.rollout(k, mode)
                res["arms"][f"{mode}:{k}"] = dict(wall=time.perf_counter() - t0, ret=g.episode_return(logs), logs=slim(logs))
        res["timer_main"] = dict(S.timer.t)
    if cmd == "main":
        for pol, modes in (("slowest", ("log",)), ("fastest", MODES), ("stop", ("log",))):
            ft = S.first_traj_for(pol)
            for mode in modes:
                logs = S.rollout(-1, mode, policy=pol, first_traj=ft, timer=g.Timer())
                res["reg"][f"{pol}:{mode}"] = dict(ret=g.episode_return(logs), logs=slim(logs))
    if cmd in ("main", "ladder"):
        t_l = time.perf_counter()
        for name, vt in S.ladder_targets():
            logs = S.ladder_rollout(vt, "log")
            res["ladder"][name] = dict(vt=vt, ret=g.episode_return(logs), logs=slim(logs))
        res["ladder_wall"] = time.perf_counter() - t_l
    res["wall_task"] = time.perf_counter() - t_all
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "ttcoff", "ladder"])
    ap.add_argument("--floor", type=float, required=True)
    ap.add_argument("--curve-min-seg-m", type=float, required=True, help="PREREG_REWARD_2 section 1: must be 0.05")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--range", type=str, default=None)
    ap.add_argument("--ram-log", type=str, default=None, help="file that receives the RAM_WAIT lines (default: stdout)")
    a = ap.parse_args()
    assert a.workers <= ramguard.MAX_WORKERS
    assert a.floor == r1.FLOOR and a.curve_min_seg_m == r2.CURVE_MIN_SEG_M, "REWARD-2 runs floor 0.20 and curve_min_seg_m 0.05 only"
    os.makedirs(a.out, exist_ok=True)
    sts = r2.load_states_v6()
    if a.cmd == "ladder":
        lo, hi = [int(x) for x in a.range.split(":")]
        assert r1.N_FULL <= lo < hi <= r1.CAP
    else:
        lo, hi = 0, r1.N_FULL
    json.dump([sts[i] for i in range(lo, hi)], open(os.path.join(a.out, "states_used.json"), "w", encoding="utf-8"))
    tasks = [(i, sts[i], a.cmd, a.floor, a.curve_min_seg_m) for i in range(lo, hi)]
    t0 = time.perf_counter()
    state = dict(done=0, nerr=0)

    def log(s):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        if a.ram_log:
            open(a.ram_log, "a", encoding="utf-8").write(line + "\n")
        else:
            print(line, flush=True)

    with gzip.open(os.path.join(a.out, "states.jsonl.gz"), "wt", encoding="utf-8") as fo, open(os.path.join(a.out, "errors.jsonl"), "w", encoding="utf-8") as fe:
        def on_result(r, task):
            if "error" in r:
                fe.write(json.dumps(r) + "\n"); fe.flush(); state["nerr"] += 1
                print(f"state {r['idx']} BUILD/RUN ERROR {r['error']}", flush=True)
            else:
                assert r["curve_min_seg_m"] == a.curve_min_seg_m
                fo.write(json.dumps(r) + "\n"); fo.flush()
            state["done"] += 1
            print(f"state {r['idx']} done wall={r.get('wall_task', 0):.1f}s ({state['done']}/{len(tasks)})", flush=True)
        ramguard.run_tasks(do_state, tasks, a.workers, on_result, label_fn=lambda t: f"{a.cmd} state {t[0]}", log=log)
    json.dump(dict(total_wall_s=time.perf_counter() - t0, workers=a.workers, n_states=len(tasks), n_errors=state["nerr"], cmd=a.cmd, seed=r2.SEED_V6,
                   floor=a.floor, curve_min_seg_m=a.curve_min_seg_m, range=[lo, hi], min_avail_ram_gib=ramguard.MIN_AVAIL_GIB),
              open(os.path.join(a.out, "run_meta.json"), "w", encoding="utf-8"))
    print("RUN_DONE", time.perf_counter() - t0, flush=True)


if __name__ == "__main__":
    main()
