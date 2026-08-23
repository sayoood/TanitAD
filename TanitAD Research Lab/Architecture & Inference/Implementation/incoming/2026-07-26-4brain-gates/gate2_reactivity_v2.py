#!/usr/bin/env python3
"""
GATE 2 v2 -- DO REACTIVE AGENTS ACTUALLY REACT?  (corrected measurement)

v1 had two measurement defects, both found by its own control and both fixed here:

 (1) It averaged divergence over ALL returned objects. On the probe scene that is
     75 agents of which 62 are PARKED. A reaction by the 2-3 agents that actually
     interact with the ego is diluted into invisibility.
     -> v2 scores DYNAMIC agents, and reports the near-ego subset separately.

 (2) It used ONE run per arm, so per-session nondeterminism was indistinguishable
     from reaction. v1's control proved this matters: PROCEED-vs-PROCEED2 (identical
     inputs, identical seed) diverged as much as YIELD-vs-PROCEED.
     -> v2 runs R repeats per arm and tests BETWEEN-arm spread against WITHIN-arm
        spread, paired, with the episode-cluster bootstrap over AGENTS.

It also adds the control v1 was missing entirely:

 (3) RETURNED-vs-LOGGED. If the service returns each agent's own logged track, then
     trafficsim is replay with extra steps and T1-T4's `Y_outcome` collapses --
     regardless of whether the arms differ. This is the "replay with extra steps"
     clause of the gate and it must be measured, not assumed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
sys.path.insert(0, "/workspace")

import grpc  # noqa: E402
from alpasim_grpc.v0 import common_pb2, traffic_pb2, traffic_pb2_grpc  # noqa: E402
from gate2_reactivity import build_session, ego_update, traj_to_grpc  # noqa: E402,F401

try:
    from taniteval.ci import paired_episode_cluster_bootstrap
    _CI = "taniteval.ci"
except Exception:
    _CI = "VENDORED"

    def paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0, alpha=0.05, reduce="mean"):
        a = np.asarray(a, float); b = np.asarray(b, float); eid = np.asarray(eid)
        uniq = np.unique(eid); idx = {u: np.where(eid == u)[0] for u in uniq}
        rng = np.random.default_rng(seed)
        pt = float(a.mean() - b.mean()); d = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([idx[u] for u in pick])
            d.append(float(a[sel].mean() - b[sel].mean()))
        d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta": round(pt, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                "separated": bool(lo > 0 or hi < 0), "p_delta_gt0": round(float((d > 0).mean()), 4),
                "n_agents": int(len(uniq)), "n_boot": n_boot,
                "estimator": "paired_episode_cluster_bootstrap(VENDORED, unit=AGENT)"}


def run_once(stub, ds, sid, arm, handover, query, seed, timeout=900):
    import uuid
    su = "g2v2-%s-%s" % (arm, uuid.uuid4().hex[:8])
    req, (ep, eq, ets) = build_session(ds, sid, handover, su, seed=seed)
    stub.start_session(req, timeout=timeout)
    out = {}
    try:
        for tq in query:
            r = stub.simulate(traffic_pb2.TrafficRequest(
                session_uuid=su, time_query_us=int(tq),
                object_trajectory_updates=[ego_update(arm, ep, eq, ets, handover, tq)]),
                timeout=timeout)
            snap = {}
            for u in r.object_trajectory_updates:
                best = None
                for ps in u.trajectory.poses:
                    if best is None or abs(int(ps.timestamp_us) - int(tq)) < abs(best[0] - int(tq)):
                        best = (int(ps.timestamp_us), ps.pose.vec.x, ps.pose.vec.y)
                if best is not None:
                    snap[u.object_id] = (best[1], best[2])
            out[int(tq)] = snap
    finally:
        try:
            stub.close_session(traffic_pb2.TrafficSessionCloseRequest(session_uuid=su), timeout=60)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sceneset", default="/workspace/alpa-invest/alpasim/data/nre-artifacts/"
                                          "scenesets/986fec83193b1baf3d5121f09462e248")
    ap.add_argument("--scene-prefix", default="clipgt-00169207")
    ap.add_argument("--addr", default="localhost:6200")
    ap.add_argument("--handover-s", type=float, default=2.0)
    ap.add_argument("--step-s", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--near-m", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/workspace/gate2_reactivity_v2.json")
    a = ap.parse_args()

    from alpasim_runtime.scene_loader import ArtifactSceneProvider
    prov = ArtifactSceneProvider.from_path(a.sceneset, smooth_trajectories=True)
    sid = [s for s in sorted(prov.scene_ids) if s.startswith(a.scene_prefix)][0]
    ds = prov.get_data_source(sid)
    ets = np.asarray(ds.rig.trajectory.timestamps_us, dtype=np.int64)
    E = np.asarray(ds.rig.trajectory.positions, dtype=np.float64)[:, :2]
    t0, t1 = int(ets[0]), int(ets[-1])
    handover = t0 + int(a.handover_s * 1e6)
    query = list(range(handover + int(a.step_s * 1e6), t1, int(a.step_s * 1e6)))

    # ---- which agents are DYNAMIC, and where were they logged?
    objs = ds.traffic_objects
    logged = {}
    dynamic = set()
    for k in objs:
        o = objs[k]
        P = np.asarray(o.trajectory.positions, dtype=np.float64)
        if len(P) < 2:
            continue
        T = np.asarray(o.trajectory.timestamps_us, dtype=np.float64)
        logged[str(k)] = (T, P[:, :2])
        if float(np.linalg.norm(P[-1, :2] - P[0, :2])) > 3.0:
            dynamic.add(str(k))
    print("scene %s | n_query=%d | dynamic agents=%d of %d" % (sid, len(query), len(dynamic), len(logged)))

    ch = grpc.insecure_channel(a.addr, options=[("grpc.max_send_message_length", 256 << 20),
                                                ("grpc.max_receive_message_length", 256 << 20)])
    grpc.channel_ready_future(ch).result(timeout=180)
    stub = traffic_pb2_grpc.TrafficServiceStub(ch)

    runs = {"PROCEED": [], "YIELD": []}
    for arm in ("PROCEED", "YIELD"):
        for r in range(a.repeats):
            runs[arm].append(run_once(stub, ds, sid, arm, handover, query, a.seed))
            print("  %s repeat %d done" % (arm, r), flush=True)

    def near(oid, tq):
        T, P = logged[oid]
        i = int(np.clip(np.searchsorted(T, tq), 0, len(P) - 1))
        j = int(np.clip(np.searchsorted(ets, tq), 0, len(E) - 1))
        return float(np.linalg.norm(P[i] - E[j]))

    # ---- per (agent,time): within-arm spread vs between-arm distance
    rows = []
    for tq in query:
        oids = set(runs["PROCEED"][0].get(tq, {}))
        for r in itertools.chain(*runs.values()):
            oids &= set(r.get(tq, {}))
        for oid in oids:
            if oid == "EGO" or oid not in dynamic:
                continue
            pa = np.array([r[tq][oid] for r in runs["PROCEED"]])
            pb = np.array([r[tq][oid] for r in runs["YIELD"]])
            # LIKE-FOR-LIKE. Both statistics are mean PAIRWISE distances between
            # INDIVIDUAL samples, so they share a scale and are equal under the null.
            # (Comparing |mean_a - mean_b| against within-arm pairwise distances is
            #  biased: averaging R samples shrinks the former by ~sqrt(R). v2's first
            #  pass made exactly that error and produced a spurious "between < within".)
            wp = ([np.linalg.norm(x - y) for x, y in itertools.combinations(pa, 2)] +
                  [np.linalg.norm(x - y) for x, y in itertools.combinations(pb, 2)])
            cp = [np.linalg.norm(x - y) for x in pa for y in pb]
            within = float(np.mean(wp)) if wp else 0.0
            between = float(np.mean(cp)) if cp else 0.0
            # replay control: does the returned pose equal the LOGGED pose?
            T, P = logged[oid]
            i = int(np.clip(np.searchsorted(T, tq), 0, len(P) - 1))
            rep = float(np.linalg.norm(pa.mean(0) - P[i]))
            rows.append({"agent": oid, "t": tq, "within": within,
                         "between": between, "vs_logged": rep, "dist_ego": near(oid, tq)})

    def summarize(rr, tag):
        if not rr:
            return {"tag": tag, "n": 0}
        b = np.array([x["between"] for x in rr]); w = np.array([x["within"] for x in rr])
        v = np.array([x["vs_logged"] for x in rr]); ag = [x["agent"] for x in rr]
        pb_ = paired_episode_cluster_bootstrap(b, w, ag, n_boot=2000, seed=0)
        return {"tag": tag, "n": len(rr), "n_agents": len(set(ag)),
                "between_mean_m": round(float(b.mean()), 4), "between_p90_m": round(float(np.percentile(b, 90)), 4),
                "between_max_m": round(float(b.max()), 4),
                "within_mean_m": round(float(w.mean()), 4), "within_p90_m": round(float(np.percentile(w, 90)), 4),
                "vs_logged_mean_m": round(float(v.mean()), 4), "vs_logged_max_m": round(float(v.max()), 4),
                "frac_returned_equals_logged_within_0p1m": round(float((v < 0.1).mean()), 4),
                "paired_between_minus_within": pb_}

    allr = [r for r in rows]
    nearr = [r for r in rows if r["dist_ego"] <= a.near_m]
    res = {"scene_id": sid, "repeats": a.repeats, "n_query": len(query),
           "handover_us": handover, "step_s": a.step_s, "near_m": a.near_m,
           "estimator_module": _CI, "resampling_unit": "AGENT",
           "n_dynamic_agents": len(dynamic),
           "dynamic_all": summarize(allr, "dynamic agents, all distances"),
           "dynamic_near_ego": summarize(nearr, "dynamic agents within %.0f m of ego" % a.near_m)}
    for key in ("dynamic_all", "dynamic_near_ego"):
        s = res[key]
        if s.get("n"):
            p = s["paired_between_minus_within"]
            s["REACTIVE"] = bool(p["separated"] and p["delta"] > 0)
            s["IS_REPLAY"] = bool(s["frac_returned_equals_logged_within_0p1m"] > 0.95)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
