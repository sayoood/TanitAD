#!/usr/bin/env python3
"""Full-runtime trafficsim reactivity: do non-ego agents move in response to the ego?

Reads the RUNTIME'S OWN `rollout.asl` (not a hand-built session) for each arm/repeat and
computes the same statistic gate 2 used, so the two are like-for-like:

    delta = between(GO, STOP) - between(GO, GO2)

`between(GO,STOP)` and `between(GO,GO2)` are both CROSS-RUN mean pairwise distances of the
same cardinality, so under the null (agents ignore the ego) STOP and GO2 are exchangeable and
delta == 0. Positive and CI-separated => reaction.

Estimator: paired episode-cluster bootstrap (taniteval/ci.py, B=2000), unit = AGENT.
NEVER overlapping_holdout_se.

Controls, both required and both reported:
  * FIDELITY   -- the ego inputs really differ between arms (from `traffic_request`).
                  If they do not, the test is void and says so rather than reporting a null.
  * REPLAY     -- returned poses vs the agents' own logged tracks (`traffic_session_request`).
                  If they match, trafficsim is replay and the construct collapses regardless.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")

from alpasim_utils.logs import async_read_pb_log  # noqa: E402

try:
    from taniteval.ci import paired_episode_cluster_bootstrap
    _CI_SRC = "taniteval.ci (/root/taniteval)"
except Exception as _e:  # pragma: no cover
    raise SystemExit(f"taniteval.ci unavailable ({_e}); refusing to vendor an estimator")

NEAR_EGO_M = 50.0


async def read_arm(asl: str) -> dict:
    """-> {'agents': {oid: {ts: (x,y)}}, 'ego': {ts: (x,y)}, 'logged': {oid: {ts:(x,y)}}}"""
    agents: dict = defaultdict(dict)
    ego: dict = {}
    logged: dict = defaultdict(dict)
    n_ret = n_req = 0
    async for e in async_read_pb_log(asl):
        which = [f.name for f, _ in e.ListFields()]
        if "traffic_session_request" in which:
            for ot in e.traffic_session_request.logged_object_trajectories:
                for ps in ot.trajectory.poses:
                    logged[ot.object_id][int(ps.timestamp_us)] = (ps.pose.vec.x, ps.pose.vec.y)
        if "traffic_request" in which:
            n_req += 1
            tr = e.traffic_request
            for u in tr.object_trajectory_updates:
                if u.object_id == "EGO":
                    for ps in u.trajectory.poses:
                        ego[int(ps.timestamp_us)] = (ps.pose.vec.x, ps.pose.vec.y)
        if "traffic_return" in which:
            n_ret += 1
            for u in e.traffic_return.object_trajectory_updates:
                if u.object_id == "EGO":
                    continue
                for ps in u.trajectory.poses:
                    agents[u.object_id][int(ps.timestamp_us)] = (ps.pose.vec.x, ps.pose.vec.y)
    return {"agents": {k: dict(v) for k, v in agents.items()}, "ego": ego,
            "logged": {k: dict(v) for k, v in logged.items()},
            "n_traffic_return": n_ret, "n_traffic_request": n_req, "asl": asl}


def find_asl(logdir: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(logdir, "rollouts", "*", "*", "rollout.asl")))
    return hits[0] if hits else None


def pairwise(runs_a: list, runs_b: list, keys: list) -> dict:
    """mean distance over all cross pairs, per (agent, ts) sample."""
    out: dict = defaultdict(list)
    for ra, rb in itertools.product(runs_a, runs_b):
        if ra is rb:
            continue
        for oid, ts in keys:
            pa = ra["agents"].get(oid, {}).get(ts)
            pb = rb["agents"].get(oid, {}).get(ts)
            if pa is None or pb is None:
                continue
            out[(oid, ts)].append(float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])))
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="/workspace/tsreact/<scene_id>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    loop = asyncio.new_event_loop()
    arms: dict = {}
    for arm in ("GO", "STOP", "GO2"):
        arms[arm] = []
        for r in range(1, args.repeats + 1):
            ld = os.path.join(args.base, f"{arm}_r{r}")
            asl = find_asl(ld)
            if not asl:
                print(f"  MISSING asl for {arm}_r{r}", flush=True)
                continue
            arms[arm].append(loop.run_until_complete(read_arm(asl)))
            a = arms[arm][-1]
            print(f"  {arm}_r{r}: agents={len(a['agents'])} returns={a['n_traffic_return']}",
                  flush=True)
    for arm in ("GO", "STOP", "GO2"):
        if len(arms[arm]) < 1:
            raise SystemExit(f"arm {arm} has no runs; cannot test")

    # ---- keys present in EVERY run of every arm (like-for-like by construction)
    allruns = arms["GO"] + arms["STOP"] + arms["GO2"]
    common = None
    for r in allruns:
        ks = {(oid, ts) for oid, d in r["agents"].items() for ts in d}
        common = ks if common is None else (common & ks)
    common = sorted(common)
    print(f"common (agent,ts) samples: {len(common)}", flush=True)

    # ---- CONTROL 1: FIDELITY -- did the arms actually differ at the ego?
    ego_go, ego_stop = arms["GO"][0]["ego"], arms["STOP"][0]["ego"]
    shared_ts = sorted(set(ego_go) & set(ego_stop))
    sep = [float(np.hypot(ego_go[t][0] - ego_stop[t][0], ego_go[t][1] - ego_stop[t][1]))
           for t in shared_ts]
    ego_go2 = arms["GO2"][0]["ego"]
    ts2 = sorted(set(ego_go) & set(ego_go2))
    sep_floor = [float(np.hypot(ego_go[t][0] - ego_go2[t][0], ego_go[t][1] - ego_go2[t][1]))
                 for t in ts2]
    fidelity = {
        "ego_GO_vs_STOP_mean_m": round(float(np.mean(sep)), 4) if sep else None,
        "ego_GO_vs_STOP_max_m": round(float(np.max(sep)), 4) if sep else None,
        "ego_GO_vs_GO2_mean_m": round(float(np.mean(sep_floor)), 6) if sep_floor else None,
        "n_shared_ts": len(shared_ts),
        "intervention_reached_model": bool(sep and float(np.max(sep)) > 5.0),
    }
    print(f"FIDELITY ego GO-vs-STOP mean={fidelity['ego_GO_vs_STOP_mean_m']} "
          f"max={fidelity['ego_GO_vs_STOP_max_m']} | GO-vs-GO2 "
          f"mean={fidelity['ego_GO_vs_GO2_mean_m']}", flush=True)

    # ---- CONTROL 2: REPLAY -- are returned poses just the logged tracks?
    vs_log, frac_same = [], []
    ref = arms["GO"][0]
    for oid, d in ref["agents"].items():
        lg = ref["logged"].get(oid, {})
        for ts, p in d.items():
            q = lg.get(ts)
            if q is None:
                continue
            dist = float(np.hypot(p[0] - q[0], p[1] - q[1]))
            vs_log.append(dist)
            frac_same.append(1.0 if dist < 0.1 else 0.0)
    replay = {
        "vs_logged_mean_m": round(float(np.mean(vs_log)), 4) if vs_log else None,
        "frac_within_0p1m_of_logged": round(float(np.mean(frac_same)), 4) if frac_same else None,
        "n_compared": len(vs_log),
        "is_replay": bool(frac_same and float(np.mean(frac_same)) > 0.9),
    }
    print(f"REPLAY vs_logged_mean={replay['vs_logged_mean_m']} "
          f"frac_same={replay['frac_within_0p1m_of_logged']} "
          f"is_replay={replay['is_replay']}", flush=True)

    # ---- the statistic, on the two strata
    ego_ref = arms["GO"][0]["ego"]

    def stratum(keys, near_only):
        if not near_only:
            return keys
        out = []
        for oid, ts in keys:
            e = ego_ref.get(ts)
            p = arms["GO"][0]["agents"].get(oid, {}).get(ts)
            if e is None or p is None:
                continue
            if np.hypot(p[0] - e[0], p[1] - e[1]) <= NEAR_EGO_M:
                out.append((oid, ts))
        return out

    results = {}
    for name, near in (("all", False), ("near_ego_50m", True)):
        keys = stratum(common, near)
        btw = pairwise(arms["GO"], arms["STOP"], keys)
        flr = pairwise(arms["GO"], arms["GO2"], keys)
        shared = sorted(set(btw) & set(flr))
        if len(shared) < 5:
            results[name] = {"error": f"only {len(shared)} paired samples"}
            continue
        a = np.array([btw[k] for k in shared])
        b = np.array([flr[k] for k in shared])
        eid = np.array([k[0] for k in shared])          # cluster = AGENT
        ci = paired_episode_cluster_bootstrap(a, b, eid, n_boot=args.n_boot, seed=0)
        results[name] = {
            "n_samples": len(shared), "n_agents": int(len(set(eid))),
            "between_GO_STOP_mean_m": round(float(a.mean()), 4),
            "floor_GO_GO2_mean_m": round(float(b.mean()), 4),
            "delta_m": round(float(a.mean() - b.mean()), 4),
            "ci": ci,
            "ci_half_width_m": round(float((ci["hi"] - ci["lo"]) / 2), 4),
            "separated": bool(ci.get("separated")),
            "reactive": bool(ci.get("separated") and (a.mean() - b.mean()) > 0),
        }
        r = results[name]
        print(f"[{name}] n={r['n_samples']} agents={r['n_agents']} "
              f"between={r['between_GO_STOP_mean_m']} floor={r['floor_GO_GO2_mean_m']} "
              f"delta={r['delta_m']} CI=[{ci['lo']},{ci['hi']}] "
              f"sep={r['separated']} REACTIVE={r['reactive']}", flush=True)

    payload = {
        "base": args.base, "repeats": args.repeats,
        "estimator": f"paired_episode_cluster_bootstrap, unit=AGENT, B={args.n_boot}",
        "ci_source": _CI_SRC,
        "n_runs": {k: len(v) for k, v in arms.items()},
        "control_fidelity": fidelity,
        "control_replay": replay,
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
