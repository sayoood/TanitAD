#!/usr/bin/env python3
"""
GATE 2 -- DO REACTIVE AGENTS ACTUALLY REACT?

The decisive property for every tactical problem (T1-T4): if non-ego agents merely
replay their logged tracks, then `Y_outcome` -- "roll the policy and score the
simulated consequence" -- collapses to log-replay and is no longer non-circular.
A null result here is DECISIVE and must be reported as such.

THE TEST
    Same scene, same seed, same session construction. Two arms whose ONLY difference
    is the ego's behaviour after handover:
        PROCEED  ego follows its logged ground-truth trajectory
        YIELD    ego stops dead at the handover pose and stays there
    Then measure whether the NON-EGO agents' returned trajectories differ.

THE CONTROL THAT MAKES IT ADMISSIBLE
    A third arm, PROCEED2, is byte-identical in construction to PROCEED (same seed).
    It measures the stochastic floor of the model. The A-vs-B divergence is only
    evidence of reactivity if it EXCEEDS the A-vs-A' floor. Without this control a
    diffusion/sampling model's own noise reads as "reaction".

This talks to the trafficsim gRPC service DIRECTLY -- no renderer, no physics, no
driver, no GPU rendering. That isolates exactly the question being asked.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")

import grpc  # noqa: E402
from alpasim_grpc.v0 import common_pb2, traffic_pb2, traffic_pb2_grpc  # noqa: E402


def pose_at(t_us, x, y, z, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    return common_pb2.PoseAtTime(
        pose=common_pb2.Pose(vec=common_pb2.Vec3(x=float(x), y=float(y), z=float(z)),
                             quat=common_pb2.Quat(w=float(qw), x=float(qx), y=float(qy), z=float(qz))),
        timestamp_us=int(t_us))


def traj_to_grpc(positions, quats, ts_us):
    return common_pb2.Trajectory(poses=[
        pose_at(t, p[0], p[1], p[2], q[0], q[1], q[2], q[3])
        for p, q, t in zip(positions, quats, ts_us)])


def build_session(ds, scene_id, handover_us, session_uuid, seed=7):
    ego = ds.rig.trajectory
    ep = np.asarray(ego.positions, dtype=np.float64)
    eq = np.asarray(ego.quaternions, dtype=np.float64)
    ets = np.asarray(ego.timestamps_us, dtype=np.int64)
    vc = ds.rig.vehicle_config
    ego_aabb = common_pb2.AABB(size_x=float(getattr(vc, "length", 4.8)),
                               size_y=float(getattr(vc, "width", 2.0)),
                               size_z=float(getattr(vc, "height", 1.6)))
    logged = [traffic_pb2.ObjectTrajectory(
        object_id="EGO", trajectory=traj_to_grpc(ep, eq, ets), aabb=ego_aabb,
        is_static=False, label_class="automobile")]
    to = ds.traffic_objects
    for k in to:
        o = to[k]
        p = np.asarray(o.trajectory.positions, dtype=np.float64)
        q = np.asarray(o.trajectory.quaternions, dtype=np.float64)
        t = np.asarray(o.trajectory.timestamps_us, dtype=np.int64)
        if len(p) == 0:
            continue
        logged.append(traffic_pb2.ObjectTrajectory(
            object_id=str(k), trajectory=traj_to_grpc(p, q, t),
            aabb=common_pb2.AABB(size_x=float(o.aabb.x), size_y=float(o.aabb.y),
                                 size_z=float(o.aabb.z)),
            is_static=bool(o.is_static), label_class=str(o.label_class)))
    return traffic_pb2.TrafficSessionRequest(
        session_uuid=session_uuid, scene_id=scene_id, random_seed=seed,
        logged_object_trajectories=logged, handover_time_us=int(handover_us)), (ep, eq, ets)


def ego_update(arm, ep, eq, ets, handover_us, upto_us):
    """The ONLY difference between the two arms.

    The FULL trajectory is sent, not a prefix truncated at `upto_us`: the service
    requires the ego update to cover the queried timestamp AND its forward sample
    window ("EGO trajectory does not cover required simulation timestamp ..."), and
    the CATK service conditions on ego FUTURE poses by design (trafficsim README).
    `upto_us` is therefore unused and kept only for call-site symmetry.
    """
    p = ep.copy(); q = eq.copy(); t = ets.copy()
    if arm == "YIELD":
        frozen = t >= handover_us
        if frozen.any():
            i0 = int(np.argmax(frozen))
            p[frozen] = p[i0]          # stop dead at the handover pose
            q[frozen] = q[i0]
    return traffic_pb2.ObjectTrajectoryUpdate(
        object_id="EGO", trajectory=traj_to_grpc(p, q, t))


def run_arm(stub, ds, scene_id, arm, handover_us, query_us, seed, timeout=600):
    su = "gate2-%s-%s" % (arm, uuid.uuid4().hex[:8])
    req, (ep, eq, ets) = build_session(ds, scene_id, handover_us, su, seed=seed)
    stub.start_session(req, timeout=timeout)
    out = {}
    try:
        for tq in query_us:
            r = stub.simulate(traffic_pb2.TrafficRequest(
                session_uuid=su, time_query_us=int(tq),
                object_trajectory_updates=[ego_update(arm, ep, eq, ets, handover_us, tq)]),
                timeout=timeout)
            snap = {}
            for u in r.object_trajectory_updates:
                best = None
                for ps in u.trajectory.poses:
                    if best is None or abs(int(ps.timestamp_us) - int(tq)) < abs(best[0] - int(tq)):
                        best = (int(ps.timestamp_us), ps.pose.vec.x, ps.pose.vec.y, ps.pose.vec.z)
                if best is not None:
                    snap[u.object_id] = best
            out[int(tq)] = snap
    finally:
        try:
            stub.close_session(traffic_pb2.TrafficSessionCloseRequest(session_uuid=su), timeout=60)
        except Exception:
            pass
    return out


def divergence(a, b, exclude=("EGO",)):
    """Per-agent, per-time L2 between two arms' returned non-ego poses."""
    d = []
    per_agent = {}
    for tq in sorted(set(a) & set(b)):
        for oid in set(a[tq]) & set(b[tq]):
            if oid in exclude:
                continue
            pa, pb = a[tq][oid], b[tq][oid]
            v = float(np.linalg.norm(np.array(pa[1:3]) - np.array(pb[1:3])))
            d.append(v)
            per_agent.setdefault(oid, []).append(v)
    if not d:
        return {"n": 0}
    d = np.asarray(d)
    mx = {k: float(np.max(v)) for k, v in per_agent.items()}
    return {"n": int(d.size), "n_agents": len(per_agent),
            "mean_m": round(float(d.mean()), 4), "p50_m": round(float(np.percentile(d, 50)), 4),
            "p90_m": round(float(np.percentile(d, 90)), 4), "max_m": round(float(d.max()), 4),
            "n_agents_gt_0p5m": int(sum(1 for v in mx.values() if v > 0.5)),
            "n_agents_gt_2m": int(sum(1 for v in mx.values() if v > 2.0)),
            "frac_poses_gt_0p5m": round(float((d > 0.5).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sceneset", default="/workspace/alpa-invest/alpasim/data/nre-artifacts/"
                                          "scenesets/986fec83193b1baf3d5121f09462e248")
    ap.add_argument("--scene-prefix", default="clipgt-00169207")
    ap.add_argument("--addr", default="localhost:6200")
    ap.add_argument("--handover-s", type=float, default=2.0)
    ap.add_argument("--step-s", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/workspace/gate2_reactivity.json")
    a = ap.parse_args()

    from alpasim_runtime.scene_loader import ArtifactSceneProvider
    prov = ArtifactSceneProvider.from_path(a.sceneset, smooth_trajectories=True)
    sid = [s for s in sorted(prov.scene_ids) if s.startswith(a.scene_prefix)][0]
    ds = prov.get_data_source(sid)
    ets = np.asarray(ds.rig.trajectory.timestamps_us, dtype=np.int64)
    t0, t1 = int(ets[0]), int(ets[-1])
    handover = t0 + int(a.handover_s * 1e6)
    query = list(range(handover + int(a.step_s * 1e6), t1, int(a.step_s * 1e6)))
    print("scene", sid)
    print("t0=%d t1=%d handover=%d  n_query=%d" % (t0, t1, handover, len(query)))

    ch = grpc.insecure_channel(a.addr, options=[("grpc.max_send_message_length", 256 << 20),
                                                ("grpc.max_receive_message_length", 256 << 20)])
    grpc.channel_ready_future(ch).result(timeout=180)
    stub = traffic_pb2_grpc.TrafficServiceStub(ch)
    md = stub.get_metadata(common_pb2.Empty(), timeout=60)
    print("service metadata:", str(md).replace("\n", " ")[:200])

    res = {"scene_id": sid, "handover_us": handover, "n_query": len(query),
           "seed": a.seed, "step_s": a.step_s}
    arms = {}
    for arm in ("PROCEED", "YIELD", "PROCEED2"):
        ts = time.time()
        arms[arm] = run_arm(stub, ds, sid, "PROCEED" if arm == "PROCEED2" else arm,
                            handover, query, a.seed)
        print("arm %-9s done in %.1fs  (%d timesteps)" % (arm, time.time() - ts, len(arms[arm])))

    res["divergence_YIELD_vs_PROCEED"] = divergence(arms["PROCEED"], arms["YIELD"])
    res["control_PROCEED2_vs_PROCEED"] = divergence(arms["PROCEED"], arms["PROCEED2"])
    # did the agents move at all relative to their own logged tracks?
    res["n_objects_returned"] = len(next(iter(arms["PROCEED"].values()), {}))

    fl = res["control_PROCEED2_vs_PROCEED"]
    sg = res["divergence_YIELD_vs_PROCEED"]
    floor = fl.get("p90_m", 0.0) if fl.get("n") else 0.0
    res["verdict"] = {
        "stochastic_floor_p90_m": floor,
        "signal_p90_m": sg.get("p90_m"),
        "signal_exceeds_floor": bool(sg.get("p90_m", 0) > max(floor, 0.05) * 3),
        "reactive": bool(sg.get("n_agents_gt_0p5m", 0) > 0 and sg.get("p90_m", 0) > max(floor, 0.05) * 3),
    }
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
