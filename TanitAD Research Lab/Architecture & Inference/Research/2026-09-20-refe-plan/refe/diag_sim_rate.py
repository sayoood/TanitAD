"""What does running the per-frame teacher rollouts at nuPlan's 10 Hz (instead of the DB's 20 Hz) change?

The teacher's published closed-loop behaviour comes from nuPlan's simulation, which subsamples the
DB by 0.5 (10 Hz); REFe's navtrain rollouts were built at 20 Hz (twice the planner calls). This
runs BOTH rates on the same frames, concurrently (same machine load), and reports:
  * cost: s per tuple at each rate;
  * change: per-frame ADE / FDE / max |dyaw| between the 10 Hz and 20 Hz targets (20 poses, 5 Hz);
  * scale: each rate's ADE to the LOGGED expert future, so the change can be read against how far
    the teacher already is from the human drive; and the aug-search threshold TAU = 0.3 m.
Rows are joined on (log_name, token, step); a frame kept at one rate and dropped at the other is
counted, not hidden.

  python diag_sim_rate.py --log <log_name>[,<log_name>...] --frames 8 --shards 3 --out <scratch dir>
(--frames is PER LOG; --shards splits each arm's logs over that many processes)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def expert_future(log_name, token):
    """The LOGGED ego future at the 20 target times, in the frame of the logged pose at the token."""
    import navtrain_scenarios as NS
    from build_teacher_rollouts import ego_frame
    db = NS.find_db(log_name) if hasattr(NS, "find_db") else None
    if db is None:
        root = os.environ.get("REFE_NUPLAN_DB_ROOT") or os.path.join(
            os.environ["NUPLAN_DATA_ROOT"], "nuplan-v1.1", "splits", "trainval")
        db = os.path.join(root, log_name + ".db")
    sc = next(iter(NS.build_scenarios_for_log(db, [token])), None)
    if sc is None:
        return None
    st = sc.get_ego_state_at_iteration(0)
    px, py, pyaw = st.rear_axle.x, st.rear_axle.y, st.rear_axle.heading
    fut = list(sc.get_ego_future_trajectory(0, 4.0, 20))[:20]
    if len(fut) < 20:
        return None
    xs = np.array([p.rear_axle.x for p in fut]); ys = np.array([p.rear_axle.y for p in fut])
    return ego_frame(px, py, pyaw, xs, ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logs_file = os.path.join(a.out, "logs.txt")
    open(logs_file, "w", encoding="utf-8").write("\n".join(a.log.split(",")) + "\n")
    procs = {}
    for hz in ("20", "10"):
        for i in range(a.shards):
            d = os.path.join(a.out, f"hz{hz}" if a.shards == 1 else f"hz{hz}_s{i}")
            f = os.path.join(d, "targets_rank0.jsonl")
            if os.path.exists(f):
                os.remove(f)
            env = dict(os.environ, REFE_SIM_HZ=hz, PYTHONIOENCODING="utf-8")
            log = open(os.path.join(a.out, f"hz{hz}_s{i}.log"), "w", encoding="utf-8")
            cmd = [sys.executable, "build_teacher_rollouts.py", "--source", "navtrain", "--logs-file",
                   logs_file, "--limit-frames", str(a.frames), "--out", d, "--rank", "0"]
            if a.shards > 1:
                cmd += ["--log-shard", f"{i}/{a.shards}"]
            procs[(hz, i)] = (subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log,
                                               stderr=subprocess.STDOUT), log, time.time(), d)
    res, rows = {}, {"20": {}, "10": {}}
    busy = {"20": 0.0, "10": 0.0}
    for (hz, i), (p, log, t0, d) in procs.items():
        rc = p.wait(); log.close()
        txt = open(os.path.join(a.out, f"hz{hz}_s{i}.log"), encoding="utf-8").read()
        # the per-log timer is CUMULATIVE from the start of the log loop, so a shard's loop time is
        # its LAST value (summing them would count early logs several times)
        busy[hz] += max([float(x) for x in re.findall(r"kept +[0-9]+ +([0-9.]+)s", txt)] or [0.0])
        f = os.path.join(d, "targets_rank0.jsonl")
        rr = [json.loads(l) for l in open(f, encoding="utf-8")] if os.path.exists(f) else []
        rows[hz].update({(r["log_name"], r["token"], r["step"]): r for r in rr})
        r0 = res.setdefault(hz, {"rc": 0, "rows": 0})
        r0["rc"] = max(r0["rc"], rc)
        r0["rows"] += len(rr)
    for hz in ("20", "10"):
        # per-frame LOOP seconds (the per-log timer), so start-up does not dilute the ratio
        res[hz]["s_per_tuple"] = round(busy[hz] / max(res[hz]["rows"], 1), 3)
    # a merged bank per rate, for diag_sim_rate_pdm.py
    if a.shards > 1:
        for hz in ("20", "10"):
            d = os.path.join(a.out, f"hz{hz}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "targets_rank0.jsonl"), "w", encoding="utf-8") as fo:
                for k in sorted(rows[hz]):
                    fo.write(json.dumps(rows[hz][k]) + "\n")
    common = sorted(set(rows["20"]) & set(rows["10"]))
    per = []
    for k in common:
        a20 = np.array(rows["20"][k]["traj"]); a10 = np.array(rows["10"][k]["traj"])
        d = np.linalg.norm(a20[:, :2] - a10[:, :2], axis=1)
        dyaw = np.abs((a20[:, 2] - a10[:, 2] + math.pi) % (2 * math.pi) - math.pi)
        rec = {"token": k[1], "ade_10v20": float(d.mean()), "fde_10v20": float(d[-1]),
               "max_dyaw_deg": float(np.degrees(dyaw.max())),
               "goal_equal": rows["20"][k]["goal"] == rows["10"][k]["goal"],
               "ego_equal": rows["20"][k]["ego"] == rows["10"][k]["ego"]}
        try:
            ex = expert_future(k[0], k[1])
        except Exception as e:                       # the scale is context, never a gate
            ex, rec["expert_error"] = None, repr(e)[:120]
        if ex is not None:
            rec["ade_20_vs_log"] = float(np.linalg.norm(a20[:, :2] - ex, axis=1).mean())
            rec["ade_10_vs_log"] = float(np.linalg.norm(a10[:, :2] - ex, axis=1).mean())
        per.append(rec)

    def stat(key):
        v = [r[key] for r in per if key in r]
        return {"mean": float(np.mean(v)), "median": float(np.median(v)), "max": float(np.max(v))} if v else None

    out = {"arms": res, "joined": len(common),
           "only_20": len(set(rows["20"]) - set(rows["10"])),
           "only_10": len(set(rows["10"]) - set(rows["20"])),
           "speedup_20_over_10": (res["20"]["s_per_tuple"] / res["10"]["s_per_tuple"]
                                  if res["20"]["s_per_tuple"] and res["10"]["s_per_tuple"] else None),
           "ade_10v20_m": stat("ade_10v20"), "fde_10v20_m": stat("fde_10v20"),
           "max_dyaw_deg": stat("max_dyaw_deg"),
           "ade_20_vs_log_m": stat("ade_20_vs_log"), "ade_10_vs_log_m": stat("ade_10_vs_log"),
           "goal_equal_all": all(r["goal_equal"] for r in per),
           "ego_equal_all": all(r["ego_equal"] for r in per),
           "tau_m": 0.3, "per_frame": per}
    json.dump(out, open(os.path.join(a.out, "diag_sim_rate.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_frame"}, indent=1))
    ok = res["20"]["rc"] == 0 and res["10"]["rc"] == 0 and len(common) > 0
    print("ZZSIMRATE_" + ("MEASUREDZZ" if ok else "FAILZZ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
