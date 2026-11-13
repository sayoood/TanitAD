#!/usr/bin/env python3
"""RJ5 navtrain roll: re-roll the bank's rank-0 rows with the REGISTERED teacher protocol and ONE exposed knob, the
positive-jerk limit (R0 = 1.0 registered, RJ5 = 5.0). Runs in the driverl-eval / box-2 teacher env, CPU only, one thread.

It does NOT re-implement the bank's construction: it calls `build_teacher_rollouts.make_row` (the bank's own row builder),
`navtrain_scenarios.build_scenarios_for_log` and `build_controller` unchanged, with the stub camera index below (cameras only decide
whether a row is kept; the row's `traj` does not depend on them). The knob hook is `build_teacher_rollouts.TEACHER_OVERRIDES`
(TF-N2 reroll patch, empty by default). Controls built in:
  * every row stores `knobs`, read back from the BUILT planner after the roll; a mismatch with the request is an ERROR row;
  * the R0 arm stores `bank_dmax` = max |traj - bank traj| (60 numbers) so the identity control is read from the rows themselves.

  python rj5_roll.py --bank <bank_slim.jsonl[.gz]> --db-dir <dir with <log>.db> --out-dir <dir> --arms R0,RJ5
         --logs-file <file> [--shard i/N] [--limit-logs n] [--limit-tokens k] [--tag t]

Rows go to <out-dir>/rj5_<tag>.jsonl (append, resume-safe, flushed per row). A log without a DB is listed in <out-dir>/nodb_<tag>.txt.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

ARM_JERK = {"R0": None, "RJ5": 5.0}
REGISTERED = {"route_goal_min_speed_mps": 5.0, "route_goal_horizon_s": 12.0, "policy_interval_s": 0.0,
              "positive_jerk_limit_when_nonnegative_acc": 1.0}


def overrides_for(arm):
    j = ARM_JERK[arm]
    return {} if j is None else {"positive_jerk_limit_when_nonnegative_acc": float(j)}


def expected_knobs(arm):
    e = dict(REGISTERED)
    e.update(overrides_for(arm))
    return e


def knob_mismatch(built, expected, tol=1e-9):
    return [f"{k}: built {built.get(k)} != requested {v}" for k, v in expected.items()
            if built.get(k) is None or abs(built[k] - v) > tol]


def traj_sha(traj):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(traj, np.float64)).tobytes()).hexdigest()


def key_of(r):
    return (r["log_name"], r["token"], int(r.get("rank", 0)), r.get("arm"))


def read_jsonl(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_bank(path):
    by = {}
    for r in read_jsonl(path):
        if int(r.get("rank", 0)) != 0:
            continue
        by.setdefault(r["log_name"], {})[r["token"]] = r["traj"]
    return by


def pick_logs(by, logs_file, shard, limit_logs):
    logs = sorted(by)
    if logs_file:
        want = {x.strip() for x in open(logs_file, encoding="utf-8") if x.strip()}
        logs = [lg for lg in logs if lg in want]
    i, n = (int(x) for x in shard.split("/"))
    if not (n > 0 and 0 <= i < n):
        raise SystemExit(f"--shard {shard} invalid")
    logs = logs[i::n]                       # applied AFTER the named list: N shards partition exactly that list
    return logs[:limit_logs] if limit_logs else logs


def done_keys(path):
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                # a torn last line of a killed shard is re-done
                if "traj" in r:
                    out.add(key_of(r))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="R0,RJ5")
    ap.add_argument("--logs-file", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit-logs", type=int, default=0)
    ap.add_argument("--limit-tokens", type=int, default=0, help="first k tokens of each log (sorted); smoke only")
    ap.add_argument("--tokens-file", default=None, help="smoke: only these tokens (one per line)")
    ap.add_argument("--tag", default="s0")
    a = ap.parse_args(argv)
    arms = a.arms.split(",")
    for x in arms:
        if x not in ARM_JERK:
            raise SystemExit(f"unknown arm {x}; reachable arms: {sorted(ARM_JERK)}")
    if os.environ.get("DRIVERL_EVAL_ROUTE_LANE_RANK", "").strip():
        raise SystemExit("rank 0 must run the UNPATCHED route: unset DRIVERL_EVAL_ROUTE_LANE_RANK")
    if os.environ.get("REFE_SIM_HZ") != "10":
        raise SystemExit("REFE_SIM_HZ must be 10 (the bank's rate)")

    import build_teacher_rollouts as BTR
    import build_targets as BT
    import navtrain_scenarios as NS
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    if not hasattr(BTR, "TEACHER_OVERRIDES"):
        raise SystemExit("build_teacher_rollouts has no TEACHER_OVERRIDES: put the TF-N2 reroll patch first on PYTHONPATH")
    BTR.TEACHER_DEVICE = "cpu"

    by = load_bank(a.bank)
    want_tok = {x.strip() for x in open(a.tokens_file, encoding="utf-8") if x.strip()} if a.tokens_file else None
    logs = pick_logs(by, a.logs_file, a.shard, a.limit_logs)
    os.makedirs(a.out_dir, exist_ok=True)
    out_fp = os.path.join(a.out_dir, f"rj5_{a.tag}.jsonl")
    done = done_keys(out_fp)

    planners = {}
    for arm in arms:
        BTR.TEACHER_OVERRIDES = overrides_for(arm)      # REBOUND, not mutated: each planner's post-init hook captures its own dict
        planners[arm] = BTR.build_planner("cpu")
    print(f"RJ5_ROLL tag={a.tag} arms={arms} logs={len(logs)} done_rows={len(done)} out={out_fp}", flush=True)

    out = open(out_fp, "a", encoding="utf-8", newline="\n")
    nodb = []
    n_ok = n_err = 0
    t0 = time.time()
    for lg in logs:
        db = os.path.join(a.db_dir, lg + ".db")
        if not os.path.exists(db):
            nodb.append(lg)
            continue
        toks = sorted(by[lg])
        if want_tok is not None:
            toks = [t for t in toks if t in want_tok]
        if a.limit_tokens:
            toks = toks[:a.limit_tokens]
        todo = [t for t in toks if any((lg, t, 0, arm) not in done for arm in arms)]
        if not todo:
            continue
        try:
            scs = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(db, todo)}
        except Exception as exc:                       # noqa: BLE001 -- counted, never silent
            print(f"SCEN_FAIL {lg}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            for t in todo:
                for arm in arms:
                    out.write(json.dumps({"log_name": lg, "token": t, "rank": 0, "arm": arm, "error": "SCEN_FAIL"}) + "\n")
                    n_err += 1
            out.flush()
            continue
        for t in todo:
            sc = scs.get(t)
            for arm in arms:
                if (lg, t, 0, arm) in done:
                    continue
                if sc is None:
                    out.write(json.dumps({"log_name": lg, "token": t, "rank": 0, "arm": arm, "error": "token not built (margin guard)"}) + "\n")
                    n_err += 1
                    continue
                pl = planners[arm]
                t1 = time.perf_counter()
                try:
                    ts_i = int(sc.get_ego_state_at_iteration(0).time_point.time_us)
                    cams = {ch: [(ts_i, "stub")] for ch in BT.CAMERAS}
                    cam_ts = {ch: np.array([ts_i], dtype=np.int64) for ch in BT.CAMERAS}
                    controller = BTR.build_controller(sc, "cpu")
                    pl.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                        mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                    r = BTR.make_row(sc, pl, controller, 0, cams, cam_ts, 0)
                    if r is None:
                        raise RuntimeError("make_row returned None")
                    knobs = BTR.effective_knobs(pl)
                    bad = knob_mismatch(knobs, expected_knobs(arm))
                    if bad:
                        raise RuntimeError("KNOB-NOT-REACHED " + "; ".join(bad))
                    row = {"log_name": lg, "token": t, "rank": 0, "arm": arm, "step": r["step"], "sim_hz": r["sim_hz"],
                           "teacher_device": r["teacher_device"], "traj": r["traj"], "traj_world": r["traj_world"],
                           "goal": r["goal"], "origin_world": r["origin_world"], "knobs": knobs,
                           "sec": round(time.perf_counter() - t1, 3)}
                    if arm == "R0":
                        row["bank_dmax"] = float(np.abs(np.asarray(r["traj"], np.float64) - np.asarray(by[lg][t], np.float64)).max())
                    out.write(json.dumps(row) + "\n")
                    n_ok += 1
                except Exception as exc:               # noqa: BLE001
                    BTR._reset_planner_state(pl)
                    out.write(json.dumps({"log_name": lg, "token": t, "rank": 0, "arm": arm,
                                          "error": f"{type(exc).__name__}: {str(exc)[:200]}"}) + "\n")
                    n_err += 1
                out.flush()
        print(f"  {lg[:30]:30s} tokens {len(todo):4d}  ok {n_ok} err {n_err}  {time.time() - t0:8.1f}s", flush=True)
    out.close()
    with open(os.path.join(a.out_dir, f"nodb_{a.tag}.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(nodb) + ("\n" if nodb else ""))
    print(f"ZZRJ5_ROLL_DONE tag={a.tag} ok={n_ok} err={n_err} nodb_logs={len(nodb)} sec={time.time() - t0:.1f}", flush=True)
    return 0 if n_err == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
