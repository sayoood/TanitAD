#!/usr/bin/env python3
"""The STRATEGIC family's ground truth, derived from the HD map — not from the ego's yaw.

What was actually missing
-------------------------
The closed-loop harness already reports a STRATEGIC block, but on the night
clip its headline (``route_head_eq_logged``) came out **1.0000** — a
constant-predictor tie, correctly self-flagged as ``degenerate``.  The reason
is now MEASURED (``results/junction_00040136.json``): the ego enters four
junctions and at **every one of them the map offers exactly ONE lane-level
continuation**.  There was nothing to choose, so agreeing with the label was
free.

The missing instrument is therefore not "a route classifier".  It is the
**option set**: at each pose, which continuations does the map admit, and
which did the ego take?  Without it there is no way to tell a scored 1.0 that
means "the model chose correctly among four" from a scored 1.0 that means
"there was one road and everybody drove down it".

This module emits that option set, per pose, from ``map.xodr`` + the clipgt
ego track.  Its outputs are the labels a strategic head is scored against, and
the ``admissible`` mask that says which poses may be scored at all.

Design rules that follow from the programme's standing constraints
------------------------------------------------------------------
1. **The class comes from the MAP, not from the realised trajectory.**  A
   route label read off the ego's own future yaw is circular: it cannot
   distinguish "took the left branch" from "drifted left on a curving road".
   Here the class of a manoeuvre is the heading change of the *connecting
   road* relative to the *incoming road*, both read from the OpenDRIVE
   reference lines.
2. **Only poses with >=2 options are scoreable.**  Everything else is a
   constant-predictor tie and is excluded by ``admissible``, not averaged in.
   This is the single change that makes the family able to discriminate.
3. **The cluster is the DECISION EVENT, not the pose.**  Every pose inside one
   approach to one junction carries the same label, so pooling poses inflates
   n by ~50x.  ``event_id`` is emitted for exactly this reason, and the
   paired episode-cluster bootstrap must resample events.
4. **The alternatives are emitted too.**  A confusion over the option set is
   the strategic analogue of the manoeuvre confusion matrix, and the branching
   factor is the metric's own discriminability.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from junction_probe import (clipgt_analysis, dist_to_junctions,  # noqa: E402
                            fit_rigid_2d, polyline_dist, wrap_deg,
                            xodr_analysis)

# map-derived route classes.  Deliberately the same 4-way vocabulary the
# programme's `route_from_future_v21` uses, so the two are comparable.
ROUTE = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT", 3: "UTURN"}
STRAIGHT_DEG = 25.0        # |dyaw| below this is road following, not a turn
UTURN_DEG = 150.0


def classify_branch(dyaw_deg: float) -> int:
    a = wrap_deg(dyaw_deg)
    if abs(a) >= UTURN_DEG:
        return 3
    if a > STRAIGHT_DEG:
        return 0
    if a < -STRAIGHT_DEG:
        return 2
    return 1


def travel_heading(road, lane_id, at="exit", chord_m=6.0, step=0.5):
    """Heading of TRAVEL on ``road`` in lane ``lane_id``, degrees.

    ⛔ **Do not compute this from ``planView`` headings.**  MEASURED
    2026-08-03 on scene 7c72937c, road 35: the road carries a ``laneOffset``
    of **10.495 m** at ``s=0``, decaying to 0 by ``s=30``.  Its *reference
    line* therefore sweeps from -112.18 deg to -71.67 deg — a 40.5 deg
    turn — while the *lane centreline the car actually drives* runs dead
    straight at **-112.43 deg**.  Reading ``ref_pose(s)[2]`` gave a branch
    angle of +51.49 deg for a manoeuvre the ego drove at +123.53 deg, and the
    self-consistency control fired on every left turn in the shortlist until
    this was fixed.  The reference line in this corpus is a construction
    curve, not the driven line.

    So: sample the LANE CENTRELINE (reference + laneOffset + inner widths, all
    of which ``Road.sample_lane`` already applies) and take a chord over the
    last/first ``chord_m`` metres.  A chord rather than a single segment
    because the 0.5 m sampling makes one segment noisy on tight arcs.

    OpenDRIVE's other invariant still applies: lanes with a NEGATIVE id run
    along increasing ``s``, positive ids run against it — so for a positive
    lane the polyline is reversed and the road's "exit" is at ``s = 0``.
    """
    pts, _w = road.sample_lane(lane_id, step=step)
    if len(pts) < 2:
        return None
    if lane_id is not None and lane_id > 0:
        pts = pts[::-1]
    k = max(1, min(len(pts) - 1, int(round(chord_m / step))))
    d = (pts[-1] - pts[-1 - k]) if at == "exit" else (pts[k] - pts[0])
    return math.degrees(math.atan2(float(d[1]), float(d[0])))


def branch_class(m, incoming_road_id, incoming_lane, connecting_road_id, connecting_lane):
    """Class of taking ``connecting_road`` out of ``incoming_road``.

    Read off the reference lines, so it is a property of the ROAD NETWORK and
    is identical no matter how the ego actually drove — which is the point: a
    route label derived from the ego's own future yaw cannot separate "took the
    left branch" from "drifted left on a curving road".
    """
    inc = m.roads.get(incoming_road_id)
    con = m.roads.get(connecting_road_id)
    if inc is None or con is None:
        return None, None
    h_in = travel_heading(inc, incoming_lane, at="exit")
    h_out = travel_heading(con, connecting_lane, at="exit")
    d = wrap_deg(h_out - h_in)
    return classify_branch(d), round(float(d), 2)


def _realised_on_branch(m, con_road_id, con_lane, Pw, yaw):
    """Ego heading change measured over EXACTLY the connecting road's own span.

    ⚠️ This exists because the first control compared two different spans and
    fired on every left turn.  MEASURED on 7c72937c junction 149: the map says
    the branch turns +51.49 deg, and the ego's heading changed +126.25 deg —
    but over the whole *junction surface* span, which also contains 40.5 deg of
    the approach road's own arc (road 35 is a single ``arc`` primitive) and
    37.3 deg of the exit road's arc (road 14, likewise).  Neither of those is
    part of the branch decision.  Projecting the connecting road's two
    endpoints onto the ego path isolates the manoeuvre itself.
    """
    r = m.roads.get(con_road_id)
    if r is None or len(Pw) < 2:
        return None
    lane = con_lane
    if lane is None:
        ids = r.driving_lane_ids()
        lane = ids[0] if ids else None
    if lane is None:
        return None
    fwd = lane < 0
    a = r.lane_center_xy(0.0 if fwd else r.length, lane)
    b = r.lane_center_xy(r.length if fwd else 0.0, lane)
    if a is None or b is None:
        return None
    i0 = int(np.linalg.norm(Pw - np.array(a[:2]), axis=1).argmin())
    i1 = int(np.linalg.norm(Pw - np.array(b[:2]), axis=1).argmin())
    if i1 == i0:
        return None
    return round(float(wrap_deg(yaw[i1] - yaw[i0])), 2)


def strategic_gt(clipgt_dir, xodr_path, pose_record, horizon_m=60.0, step=1.0):
    ca = clipgt_analysis(clipgt_dir)
    xa = xodr_analysis(xodr_path, pose_record, step=step)
    m, surf, road2j = xa["map"], xa["surf"], xa["road2j"]

    Pc = ca["_ego_xy"]
    Pm = xa["Pm"]
    n = len(Pc)
    best = None
    for off in range(0, max(1, len(Pm) - n + 1)):
        R, t, res = fit_rigid_2d(Pc, Pm[off:off + n])
        rms = float(np.sqrt((res ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, off, R, t)
    rms, off, R, t = best
    Pw = Pc @ R.T + t
    arc = ca["_arc"]

    cls = xa["centerlines"]
    keys = list(cls)
    D = np.stack([polyline_dist(Pw, cls[k]) for k in keys])
    lane = [keys[i] for i in D.argmin(0)]
    road = [k.split(":")[0] for k in lane]
    lid = [int(k.split(":")[1]) for k in lane]
    dj, which = dist_to_junctions(Pw, surf)
    inside = dj <= 0.0

    # ---- enumerate decision EVENTS (one per junction approach)
    events, i = [], 0
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j0 = i
        while i + 1 < n and inside[i + 1] and which[i + 1] == which[j0]:
            i += 1
        j1 = i
        jid = which[j0]
        j = m.junctions.get(jid)

        # ⛔ do NOT take the modal snapped junction road.  MEASURED on scene
        # 4a2db3b3 junction 56: the connecting roads all leave the same entry
        # point, so their lane centrelines overlap for the first metres and the
        # nearest-lane snap flip-flops 15 -> 20 -> 13 -> 32.  The mode picked
        # road 15 (STRAIGHT) while the ego actually drove 14 -> 13 -> 12 and
        # turned 163 deg.  Resolve TOPOLOGICALLY instead: the branch taken is
        # the one that lands on the road the ego is on when it leaves.
        entry_road = next((road[k] for k in range(j0 - 1, -1, -1)
                           if road2j.get(road[k]) is None), None)
        exit_road = next((road[k] for k in range(j1 + 1, n)
                          if road2j.get(road[k]) is None), None)

        def _links_to(rid, other):
            r = m.roads.get(rid)
            if r is None or other is None:
                return False
            return any(l is not None and l[0] == "road" and l[1] == other
                       for l in (r.pred, r.succ))

        taken_road, how = None, "unresolved"
        if j is not None:
            cand = [c for c in j.connections
                    if _links_to(c["connectingRoad"], exit_road)]
            narrowed = [c for c in cand if c["incomingRoad"] == entry_road]
            if narrowed:
                taken_road, how = narrowed[0]["connectingRoad"], "exit+entry-link"
            elif cand:
                taken_road, how = cand[0]["connectingRoad"], "exit-link"
            else:
                # geometric fallback: which connecting road does the ego path
                # actually COVER?  Overlap at the entry cannot fake coverage of
                # a whole branch.
                ego_seg = Pw[j0:j1 + 1]
                best_cov = 0.0
                for r in m.roads.values():
                    if r.junction != jid:
                        continue
                    for ln in r.driving_lane_ids():
                        pts, _w = r.sample_lane(ln, step=0.5)
                        if len(pts) < 2 or len(ego_seg) < 2:
                            continue
                        cov = float((polyline_dist(pts, ego_seg) <= 2.0).mean())
                        if cov > best_cov:
                            best_cov, taken_road = cov, r.rid
                how = f"coverage={best_cov:.2f}" if taken_road else "unresolved"

        # independent cross-check of the topological pick
        cov_taken = None
        if taken_road is not None and (j1 - j0) >= 1:
            rr = m.roads.get(taken_road)
            if rr is not None:
                covs = []
                for ln in rr.driving_lane_ids():
                    pts, _w = rr.sample_lane(ln, step=0.5)
                    if len(pts) >= 2:
                        covs.append(float((polyline_dist(pts, Pw[j0:j1 + 1]) <= 2.0).mean()))
                cov_taken = round(max(covs or [0.0]), 3)

        inc_r = inc_l = None
        taken_lane = None
        if j is not None and taken_road is not None:
            for c in j.connections:
                if c["connectingRoad"] != taken_road:
                    continue
                if entry_road is not None and c["incomingRoad"] != entry_road and \
                        any(cc["connectingRoad"] == taken_road and
                            cc["incomingRoad"] == entry_road for cc in j.connections):
                    continue
                if c["laneLinks"]:
                    inc_r, inc_l = c["incomingRoad"], c["laneLinks"][0][0]
                    taken_lane = c["laneLinks"][0][1]
                else:
                    inc_r = c["incomingRoad"]
                break
        opts = []
        if j is not None and inc_r is not None:
            for c in j.connections:
                if c["incomingRoad"] != inc_r:
                    continue
                to_lane = None
                if c["laneLinks"]:
                    hit = [b for a, b in c["laneLinks"] if inc_l is None or a == inc_l]
                    if not hit:
                        continue
                    to_lane = hit[0]
                if c["connectingRoad"] not in [o["road"] for o in opts]:
                    k, d = branch_class(m, inc_r, inc_l, c["connectingRoad"], to_lane)
                    opts.append({"road": c["connectingRoad"], "lane": to_lane,
                                 "class": k, "class_name": ROUTE.get(k),
                                 "branch_dyaw_deg": d})
        taken_cls, taken_dyaw = (branch_class(m, inc_r, inc_l, taken_road, taken_lane)
                                 if inc_r and taken_road else (None, None))
        events.append({
            "event_id": f"{Path(clipgt_dir).parent.name}|J{jid}|{j0}",
            "junction_id": jid,
            "entry_pose": int(j0), "exit_pose": int(j1),
            "entry_arc_m": round(float(arc[j0]), 2),
            "incoming_road": inc_r, "incoming_lane": inc_l,
            "entry_road": entry_road, "exit_road": exit_road,
            "connecting_road_taken": taken_road,
            "resolved_by": how,
            "coverage_of_taken_road_by_ego_path": cov_taken,
            "route_gt_class": taken_cls,
            "route_gt_name": ROUTE.get(taken_cls),
            "route_gt_branch_dyaw_deg": taken_dyaw,
            "n_options": len(opts),
            "options": sorted(opts, key=lambda o: (o["class"] is None, o["class"])),
            "option_classes": sorted({o["class_name"] for o in opts if o["class_name"]}),
            "realised_heading_change_over_junction_span_deg": round(
                float(wrap_deg(ca["_yaw_deg"][j1] - ca["_yaw_deg"][j0])), 2),
            "realised_heading_change_on_branch_deg": _realised_on_branch(
                m, taken_road, taken_lane, Pw, ca["_yaw_deg"]),
            "SCOREABLE": bool(len(opts) >= 2),
        })
        i += 1

    # ---- per-pose labels: the NEXT decision event ahead of each pose
    per_pose = []
    for i in range(n):
        nxt = next((e for e in events if e["entry_pose"] >= i), None)
        if nxt is None:
            per_pose.append({"pose": i, "has_decision_ahead": False,
                             "admissible": False,
                             "reason": "no junction ahead inside this clip"})
            continue
        d = float(nxt["entry_arc_m"] - arc[i])
        adm = bool(nxt["SCOREABLE"] and 0.0 <= d <= horizon_m)
        per_pose.append({
            "pose": i, "has_decision_ahead": True,
            "event_id": nxt["event_id"],
            "dist_to_decision_point_m": round(d, 2),
            "n_options": nxt["n_options"],
            "route_gt_class": nxt["route_gt_class"],
            "route_gt_name": nxt["route_gt_name"],
            "admissible": adm,
            "reason": ("ok" if adm else
                       ("single-option junction: a constant predictor ties"
                        if not nxt["SCOREABLE"] else
                        f"decision point {d:.1f} m away, outside the {horizon_m:.0f} m horizon")),
        })

    # ---- MANDATORY component-vs-family self-consistency control ------------
    # The map-derived class and the ego's realised heading change are two
    # independent descriptions of the SAME manoeuvre.  They must agree.  The
    # first version of `branch_class` read the reference-line heading without
    # the lane-sign travel direction and labelled a -0.07 deg traversal
    # "UTURN (-176.82 deg)"; this control is what caught it, so it now gates
    # the output instead of living in a comment.
    checks = []
    for e in events:
        real = e["realised_heading_change_on_branch_deg"]
        if e["route_gt_branch_dyaw_deg"] is None or real is None:
            continue
        # only meaningful where the traversal is not truncated by the clip end
        truncated = e["exit_pose"] >= n - 2
        err = abs(wrap_deg(e["route_gt_branch_dyaw_deg"] - real))
        checks.append({"event_id": e["event_id"],
                       "map_branch_dyaw_deg": e["route_gt_branch_dyaw_deg"],
                       "realised_on_branch_deg": real,
                       "realised_over_junction_span_deg":
                           e["realised_heading_change_over_junction_span_deg"],
                       "abs_err_deg": round(float(err), 2),
                       "truncated_by_clip_end": truncated})
        e["selfconsistency_abs_err_deg"] = round(float(err), 2)
        e["selfconsistency_truncated"] = truncated
    live = [c for c in checks if not c["truncated_by_clip_end"]]
    worst = max([c["abs_err_deg"] for c in live] or [0.0])
    control = {
        "name": "map-branch-class vs realised-heading (component vs family)",
        "n_events_checked": len(checks),
        "n_untruncated": len(live),
        "worst_abs_err_deg_untruncated": round(float(worst), 2),
        "tolerance_deg": 35.0,
        "PASS": bool(len(live) == 0 or worst <= 35.0),
        "note": ("truncated traversals are excluded because the realised heading change "
                 "is only a partial arc when the clip ends mid-junction; they are "
                 "reported, never scored."),
        "per_event": checks,
    }

    n_adm = sum(p["admissible"] for p in per_pose)
    scoreable_events = [e for e in events if e["SCOREABLE"]]
    return {
        "SELFCONSISTENCY_CONTROL": control,
        "ADMISSIBLE": control["PASS"],
        "tool": "strategic_gt.py", "evidence_class": "MEASURED",
        "clipgt": str(clipgt_dir), "xodr": str(xodr_path),
        "horizon_m": horizon_m,
        "alignment_rms_m": round(rms, 5),
        "n_poses": n,
        "n_decision_events": len(events),
        "n_SCOREABLE_events": len(scoreable_events),
        "branching_factor": {
            "max": max([e["n_options"] for e in events] or [0]),
            "values": [e["n_options"] for e in events],
        },
        "n_admissible_poses": n_adm,
        "admissible_pose_fraction": round(n_adm / max(n, 1), 4),
        "route_gt_class_share_over_events": {
            ROUTE[c]: sum(1 for e in events if e["route_gt_class"] == c)
            for c in range(4)},
        "EFFECTIVE_N_FOR_THE_CI": len(scoreable_events),
        "effective_n_note": (
            "the paired episode-cluster bootstrap must resample DECISION EVENTS, not "
            "poses: every pose approaching one junction carries the identical label, so "
            "a pose-level n overstates the sample by roughly the approach length in "
            "poses. On this scene that is "
            f"{n_adm} poses vs {len(scoreable_events)} events."),
        "events": events,
        "per_pose": per_pose,
    }


def _ci_module():
    """The programme's own estimator, or None when taniteval is not installed."""
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "taniteval"
        if (cand / "taniteval" / "ci.py").exists():
            sys.path.insert(0, str(cand))
            break
    try:
        from taniteval import ci
        return ci
    except Exception:                                            # noqa: BLE001
        return None


def score_strategic(events_by_scene, predictions, n_boot=2000, seed=0):
    """Score a strategic head against the map-derived option sets.

    ``events_by_scene``  {scene_id: [event, ...]} from :func:`strategic_gt`.
    ``predictions``      {event_id: {"road": <connecting road id>,
                                     "class": <0..3, optional>}}

    ⚠️ **The value is per DECISION EVENT; the resampling CLUSTER is the SCENE.**
    Two events inside one clip share a map, a driver and a traffic culture, so
    they are not independent; and every pose approaching one junction carries
    the identical label, so a pose-level n overstates the sample by ~50x. On
    ``7c72937c`` that is 100 admissible poses but **2** events.

    Nothing is scored where ``n_options < 2`` — that is a constant-predictor
    tie, and averaging it in is exactly how ``route_head_eq_logged`` reached
    1.0000 on a clip with no branch.
    """
    ci = _ci_module()
    rows, eids = [], []
    for sid, evs in events_by_scene.items():
        for e in evs:
            if not e.get("SCOREABLE"):
                continue
            p = predictions.get(e["event_id"])
            rows.append({
                "event_id": e["event_id"], "scene": sid,
                "n_options": e["n_options"],
                "gt_road": e["connecting_road_taken"],
                "gt_class": e["route_gt_class"],
                "pred_road": (p or {}).get("road"),
                "pred_class": (p or {}).get("class"),
                "has_prediction": p is not None,
            })
            eids.append(sid)
    if not rows:
        return {"n_events": 0,
                "reason": ("no SCOREABLE decision event: every junction in this set offers "
                           "one lane-level continuation, so a constant predictor ties and "
                           "no strategic accuracy is defined. This is a property of the "
                           "SCENES, not a missing instrument.")}

    choice = [float(r["pred_road"] == r["gt_road"]) for r in rows]
    klass = [float(r["pred_class"] == r["gt_class"]) for r in rows]
    chance = [1.0 / max(r["n_options"], 1) for r in rows]

    def agg(v):
        if ci is None:
            return {"mean": round(float(np.mean(v)), 4), "n": len(v),
                    "estimator": "MEAN ONLY -- taniteval not importable, NO CI"}
        return ci.episode_cluster_bootstrap(v, eids, n_boot=n_boot, seed=seed)

    conf = {}
    for r in rows:
        conf.setdefault(ROUTE.get(r["gt_class"], "?"), {})
        k = ROUTE.get(r["pred_class"], "none")
        conf[ROUTE.get(r["gt_class"], "?")][k] = \
            conf[ROUTE.get(r["gt_class"], "?")].get(k, 0) + 1

    by_branch = {}
    for r in rows:
        by_branch.setdefault(r["n_options"], []).append(
            float(r["pred_road"] == r["gt_road"]))

    return {
        "route_choice_accuracy": agg(choice),
        "route_class_accuracy": agg(klass),
        "chance_floor": round(float(np.mean(chance)), 4),
        "chance_floor_note": ("mean of 1/n_options. Quoting an accuracy without this is "
                              "meaningless: a set of 2-option junctions has a 0.50 floor."),
        "route_choice_confusion_gt_x_pred": conf,
        "accuracy_by_branching_factor": {
            int(k): {"n": len(v), "acc": round(float(np.mean(v)), 4)}
            for k, v in sorted(by_branch.items())},
        "n_events": len(rows),
        "n_scenes": len(set(eids)),
        "n_events_without_a_prediction": sum(1 for r in rows if not r["has_prediction"]),
        "CLUSTER": "scene (values are per decision EVENT, never per pose)",
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clipgt", required=True)
    ap.add_argument("--xodr", required=True)
    ap.add_argument("--pose-record", required=True)
    ap.add_argument("--horizon-m", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = strategic_gt(a.clipgt, a.xodr, a.pose_record, horizon_m=a.horizon_m)
    txt = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"wrote {a.out}")
    slim = {k: v for k, v in rep.items() if k != "per_pose"}
    print(json.dumps(slim, indent=1, default=str))


if __name__ == "__main__":
    main()
