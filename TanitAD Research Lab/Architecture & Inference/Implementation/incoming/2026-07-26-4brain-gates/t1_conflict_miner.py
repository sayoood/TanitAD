#!/usr/bin/env python3
"""
T1 -- YIELD vs PROCEED AT AN UNPROTECTED CONFLICT: the decision-point miner.

Mines, per AlpaSim scene, the frames at which a tracked agent's path crosses the ego's
path with both arrival times finite and close -- i.e. a genuine right-of-way conflict.

    decision point   first frame where a tracked agent's path intersects the ego's
                     forward corridor within HORIZON s, with TTC_ego and TTC_agent
                     both finite and |TTC_ego - TTC_agent| <= CONFLICT_WINDOW
    option set       {yield, proceed}
    Y_expert         did the human ego reach the conflict point FIRST?
                     (a pure geometry+track fact -- replay-safe, but WEAK: the
                     expert's choice is one sample of a bimodal decision)
    Y_outcome        NOT computable here. It is a *simulated consequence* and needs
                     the full closed-loop stack with trafficsim ON. See GATE_RESULTS.md.

Runs on `tanitad-eval` (needs the scenes). Read-only. No GPU.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")

SSROOT = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets"
OUT = "/workspace/t1_conflict_points.json"

HORIZON_S = 15.0          # spec: 5-15 s
CONFLICT_WINDOW_S = 4.0   # both parties arrive within this of each other
MIN_MOVE_M = 3.0          # ignore parked cars
CROSS_ANGLE_MIN_DEG = 20.0  # a crossing, not a follow


def seg_intersect(p1, p2, q1, q2):
    """2D segment intersection; returns (t, u, point) or None."""
    r = p2 - p1
    s = q2 - q1
    d = r[0] * s[1] - r[1] * s[0]
    if abs(d) < 1e-9:
        return None
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / d
    u = (qp[0] * r[1] - qp[1] * r[0]) / d
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u, p1 + t * r
    return None


def path_cross(A, ta, B, tb):
    """First crossing between polyline A (times ta) and polyline B (times tb).

    Returns (xy, t_ego_arrival_us, t_agent_arrival_us, angle_deg) or None.
    NOTE: this is a *spatial* crossing; the two may pass through it at different
    times, which is exactly the quantity T1 is about.
    """
    best = None
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            r = seg_intersect(A[i], A[i + 1], B[j], B[j + 1])
            if r is None:
                continue
            t, u, pt = r
            te = ta[i] + t * (ta[i + 1] - ta[i])
            tg = tb[j] + u * (tb[j + 1] - tb[j])
            va = A[i + 1] - A[i]
            vb = B[j + 1] - B[j]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1e-6 or nb < 1e-6:
                continue
            ang = abs(math_deg(np.arccos(np.clip(float(va @ vb) / (na * nb), -1, 1))))
            if best is None or te < best[1]:
                best = (pt, te, tg, ang)
    return best


def math_deg(r):
    return float(r) * 180.0 / np.pi


def main():
    from alpasim_runtime.scene_loader import ArtifactSceneProvider

    scenes = {}
    for sd in sorted(glob.glob(SSROOT + "/*")):
        if not os.path.isdir(sd):
            continue
        try:
            prov = ArtifactSceneProvider.from_path(sd, smooth_trajectories=True)
        except Exception:
            continue
        for sid in sorted(prov.scene_ids):
            scenes.setdefault(sid, prov)
    print("scenes:", len(scenes), flush=True)

    all_cp = []
    per_scene = {}
    for i, (sid, prov) in enumerate(sorted(scenes.items()), 1):
        try:
            ds = prov.get_data_source(sid)
            ego = ds.rig.trajectory
            E = np.asarray(ego.positions, dtype=np.float64)[:, :2]
            te = np.asarray(ego.timestamps_us, dtype=np.float64)
            objs = ds.traffic_objects
            found = []
            for k in objs:
                o = objs[k]
                P = np.asarray(o.trajectory.positions, dtype=np.float64)
                if len(P) < 2:
                    continue
                P = P[:, :2]
                if float(np.linalg.norm(P[-1] - P[0])) < MIN_MOVE_M:
                    continue                      # parked / static
                tb = np.asarray(o.trajectory.timestamps_us, dtype=np.float64)
                c = path_cross(E, te, P, tb)
                if c is None:
                    continue
                pt, t_ego, t_agent, ang = c
                if ang < CROSS_ANGLE_MIN_DEG:
                    continue                      # a follow, not a crossing
                gap_s = (t_agent - t_ego) / 1e6
                if abs(gap_s) > CONFLICT_WINDOW_S:
                    continue                      # not a genuine conflict
                # decision point: HORIZON before the EARLIER arrival
                t_first = min(t_ego, t_agent)
                t_dp_us = t_first - HORIZON_S * 1e6
                idx = int(np.searchsorted(te, max(t_dp_us, te[0])))
                idx = min(max(idx, 0), len(E) - 2)
                ttc_ego = (t_ego - te[idx]) / 1e6
                ttc_agent = (t_agent - te[idx]) / 1e6
                if not (0 < ttc_ego <= HORIZON_S and 0 < ttc_agent <= HORIZON_S):
                    continue
                v0 = float(np.linalg.norm(E[min(idx + 1, len(E) - 1)] - E[idx]) /
                           max((te[min(idx + 1, len(te) - 1)] - te[idx]) / 1e6, 1e-3))
                found.append({
                    "scene_id": sid, "agent_id": str(k),
                    "label_class": str(o.label_class),
                    "t_dp_idx": idx,
                    "conflict_xy": [round(float(pt[0]), 2), round(float(pt[1]), 2)],
                    "ttc_ego_s": round(ttc_ego, 2),
                    "ttc_agent_s": round(ttc_agent, 2),
                    "gap_s": round(float(gap_s), 2),
                    "cross_angle_deg": round(ang, 1),
                    "v0_mps": round(v0, 2),
                    "dist_to_conflict_m": round(float(np.linalg.norm(pt - E[idx])), 1),
                    # Y_expert: did the EGO reach the conflict point first?
                    "Y_expert_ego_first": bool(t_ego < t_agent),
                })
            per_scene[sid] = len(found)
            all_cp.extend(found)
            if found:
                print("[%2d/%2d] %s  conflicts=%d" % (i, len(scenes), sid[8:20], len(found)), flush=True)
            try:
                ds.clear_cache()
            except Exception:
                pass
        except Exception as e:
            print("ERR", sid[:20], repr(e)[:120], flush=True)

    json.dump(all_cp, open(OUT, "w"), indent=1)
    nsc = len({c["scene_id"] for c in all_cp})
    first = sum(1 for c in all_cp if c["Y_expert_ego_first"])
    print("=" * 70)
    print("T1 conflict points: %d over %d scenes (of %d)" % (len(all_cp), nsc, len(scenes)))
    print("Y_expert ego-first: %d / %d  (majority-class rate %.4f)"
          % (first, len(all_cp), max(first, len(all_cp) - first) / max(len(all_cp), 1)))
    if all_cp:
        print("median |gap| s   : %.2f" % float(np.median([abs(c["gap_s"]) for c in all_cp])))
        print("median TTC_ego s : %.2f" % float(np.median([c["ttc_ego_s"] for c in all_cp])))
        print("median cross ang : %.1f deg" % float(np.median([c["cross_angle_deg"] for c in all_cp])))
        cls = {}
        for c in all_cp:
            cls[c["label_class"]] = cls.get(c["label_class"], 0) + 1
        print("agent classes    :", cls)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
