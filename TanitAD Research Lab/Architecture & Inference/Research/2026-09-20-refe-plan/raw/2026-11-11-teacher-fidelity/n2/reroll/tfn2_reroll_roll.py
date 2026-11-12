#!/usr/bin/env python3
"""TF-N2 re-roll: the registered DriveRL teacher protocol (eval/teacher_navtest_seam.roll, 10 Hz, CPU, rank 0, step 0, per-token
scenario) with its PROTOCOL KNOBS exposed behind explicit CLI flags, on navtrain held-out tokens. Runs inside eval/ through
`eval/run_p4.py --script tfn2_reroll_roll.py` (the driverl-eval venv, env_driverl()). CPU only: `S._init("cpu", ...)`.

Flags (every default leaves the build bit-identical to the registered protocol; the override dict stays EMPTY):
  --route-goal-min-speed <mps>   planner cfg route_goal_min_speed_mps        (registered 5.0)   -> feature builder goal distance
  --route-goal-horizon <s>       planner cfg route_goal_horizon_s            (registered 12.0)
  --policy-interval-s <s>        planner cfg policy_interval_s               (registered 0.0 = the policy runs EVERY 0.1 s step)
  --pos-jerk-limit <m/s^3>       positive jerk limit while a >= 0            (registered 1.0, release yaml engine_config)
Reachability: after S._init the BUILT planner's knobs are read back (BTR.effective_knobs) and compared with the request; after every
roll they are read again (the jerk limit is overwritten by planner.initialize and re-applied by a wrapper) and stored in the row as
`knobs`. A row whose knobs differ from the request is written as an error, never as a result.
  --samples <tokens.txt> --shard i/N --out <jsonl> --arm <label>
Row: {log, token, arm, knobs, traj20 [20,3], goal, buffer_s, history_clamped, v0, sec} or {log, token, arm, error}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import teacher_navtest_seam as S  # noqa: E402  (puts ../refe on sys.path)
import build_teacher_rollouts as BTR  # noqa: E402

REGISTERED = {"route_goal_min_speed_mps": 5.0, "route_goal_horizon_s": 12.0, "policy_interval_s": 0.0,
              "positive_jerk_limit_when_nonnegative_acc": 1.0}
FLAG_TO_KEY = {"route_goal_min_speed": "route_goal_min_speed_mps", "route_goal_horizon": "route_goal_horizon_s",
               "policy_interval_s": "policy_interval_s", "pos_jerk_limit": "positive_jerk_limit_when_nonnegative_acc"}


def overrides_from_args(a) -> dict:
    return {key: float(getattr(a, flag)) for flag, key in FLAG_TO_KEY.items() if getattr(a, flag) is not None}


def expected_knobs(ov: dict) -> dict:
    e = dict(REGISTERED)
    e.update(ov)
    return e


def knob_mismatch(built: dict, expected: dict, tol: float = 1e-9) -> list:
    return [f"{k}: built {built[k]} != requested {v}" for k, v in expected.items() if abs(built[k] - v) > tol]


def groups(keys):
    by = {}
    for lg, tok in keys:
        by.setdefault(lg, []).append(tok)
    return by


def run(keys, arm, expected, on_row):
    NS = S._S["NS"]
    planner = S._S["planner"]
    dbs = NS.index_dbs(db_root=os.path.dirname(S.TRAINVAL_DB))
    for lg, toks in sorted(groups(keys).items()):
        db = dbs.get(lg, "")
        if not os.path.exists(db):
            for tok in sorted(toks):
                on_row({"log": lg, "token": tok, "arm": arm, "error": "no local DB"})
            continue
        pos = S._log_positions(db)
        scs = {sc.scenario_name: sc for sc in NS.build_scenarios_for_log(db, sorted(toks))}
        for tok in sorted(toks):
            rec = {"log": lg, "token": tok, "arm": arm}
            try:
                t1 = time.time()
                g = S.roll(scs[tok], pos.get(tok))
                knobs = BTR.effective_knobs(planner)
                bad = knob_mismatch(knobs, expected)
                if bad:
                    raise RuntimeError("KNOB-NOT-REACHED " + "; ".join(bad))
                rec.update({"knobs": knobs, "traj20": np.asarray(g["traj20"], np.float64).tolist(), "goal": g["goal"],
                            "buffer_s": g["buffer_s"], "history_clamped": g["history_clamped"], "v0": g["v0"],
                            "sec": round(time.time() - t1, 2)})
            except Exception as e:                                               # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            on_row(rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="lines 'log:token'")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", default="1")
    ap.add_argument("--route-goal-min-speed", dest="route_goal_min_speed", type=float, default=None)
    ap.add_argument("--route-goal-horizon", dest="route_goal_horizon", type=float, default=None)
    ap.add_argument("--policy-interval-s", dest="policy_interval_s", type=float, default=None)
    ap.add_argument("--pos-jerk-limit", dest="pos_jerk_limit", type=float, default=None)
    a = ap.parse_args()
    ov = overrides_from_args(a)
    BTR.TEACHER_OVERRIDES.clear()
    if os.environ.get("TFN2_MUTATE_DROP_OVERRIDE") == "1":    # TEST ONLY (mutation arm): the request is kept, the override is dropped
        print("  MUTATION: override dropped before the build", flush=True)
    else:
        BTR.TEACHER_OVERRIDES.update(ov)                     # BEFORE S._init builds the planner
    S._init("cpu", int(a.threads), 0)
    exp = expected_knobs(ov)
    built = BTR.effective_knobs(S._S["planner"])
    # the jerk limit is only re-applied after planner.initialize, so the build-time check excludes it
    bad = knob_mismatch(built, {k: v for k, v in exp.items() if k != "positive_jerk_limit_when_nonnegative_acc"})
    if bad:
        print("ZZTFN2_KNOB_FAIL " + "; ".join(bad), flush=True)
        return 3
    print(f"  arm {a.arm}: overrides {ov}; built knobs {built}", flush=True)
    keys = [tuple(x.strip().split(":")) for x in open(a.samples, encoding="utf-8") if x.strip()]
    si, sn = (int(x) for x in a.shard.split("/"))
    keys = keys[si::sn]
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done.add((r["log"], r["token"]))
    keys = [k for k in keys if k not in done]
    if a.limit:
        keys = keys[:a.limit]
    n = [0]
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        def on(rec):
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            n[0] += 1
        run(keys, a.arm, exp, on)
    print("ZZTFN2_ROLL_DONE", a.arm, a.shard, n[0], round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
