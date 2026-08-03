#!/usr/bin/env python3
"""The FOUR METRIC FAMILIES for a closed-loop rollout — plus ADE, never instead of it.

⛔ BINDING (Sayed, 2026-08-02): every eval reports LONGITUDINAL, LATERAL, TACTICAL and
STRATEGIC **in addition to** ADE, per family, never pooled, each with its estimator and
CI on the same windows. An ADE-only table is an incomplete result. This module is that
instrument for closed-loop rollouts.

| family | what is computed here |
|---|---|
| ADE | plan vs the log's own motion from the matched point, per horizon + `ade_0_2s`; and the closed-loop `dist_to_gt_traj` |
| LONGITUDINAL | target-speed error, along-track error per horizon, and distance-keeping (headway / time-gap / TTC to the lead annotated agent) |
| LATERAL | heading error, **curvature error**, **yaw-rate error**, cross-track |
| TACTICAL | planned vs executed vs logged manoeuvre over the 5-way class set, the model's own `maneuver_head` where it exposes one, and the confusion matrices |
| STRATEGIC | route command vs the model's route head, route-corridor departure, and progress along the logged route |

Class boundaries are NOT invented here: manoeuvres come from
`scripts/refb_labels.classify_maneuver_v2` and routes from `route_target_v21` — the
programme's own labellers, so a closed-loop manoeuvre is the same object as a training
label.

Estimator: `taniteval.ci.episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`.
⚠️ The clusters here are the **rollout starts**, i.e. disjoint segments of ONE clip, not
40 independent val episodes. Stated on every interval this module emits.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DT = 0.1
WP_STEPS = (5, 10, 15, 20)
HORIZ = ("0_5s", "1s", "1_5s", "2s")
MAN_NAMES = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")
ROUTE_NAMES = ("left", "straight", "right", "unknown")
LEAD_HALF_W = 2.0        # m: an agent within this lateral band counts as in-lane
LEAD_MAX_X = 80.0        # m: beyond this it is not a distance-keeping target
EGO_LEN = 4.7            # m
CORRIDOR_M = 2.0         # m: |cross-track| beyond this = off the intended route


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_rollouts(path):
    return json.loads(Path(path).read_text())


def gt_arrays(gt):
    xy = np.array([[g["x"], g["y"]] for g in gt], float)
    yaw = np.array([g["yaw"] for g in gt], float)
    v = np.zeros(len(gt))
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1) / DT
    v[:-1] = d
    v[-1] = d[-1] if len(d) else 0.0
    return xy, yaw, v, np.stack([xy[:, 0], xy[:, 1], yaw, v], 1)


def ego_frame(delta_xy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * delta_xy[0] - s * delta_xy[1], s * delta_xy[0] + c * delta_xy[1]])


def cross_track(p, xy):
    """Signed perpendicular distance from p to the logged polyline (+left)."""
    seg = xy[1:] - xy[:-1]
    L2 = (seg ** 2).sum(1).clip(1e-9)
    t = (((p[None, :2] - xy[:-1]) * seg).sum(1) / L2).clip(0, 1)
    proj = xy[:-1] + t[:, None] * seg
    d = np.linalg.norm(proj - p[None, :2], axis=1)
    i = int(np.argmin(d))
    n = seg[i] / math.sqrt(L2[i])
    r = p[:2] - proj[i]
    return float(n[0] * r[1] - n[1] * r[0]), i, float(t[i])


def plan_poses(plan, v0, n=21):
    knots = np.vstack([[0.0, 0.0], np.asarray(plan, float)])
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    tq = np.arange(n) * DT
    x = np.interp(tq, ts, knots[:, 0])
    y = np.interp(tq, ts, knots[:, 1])
    d = np.diff(np.stack([x, y], 1), axis=0, prepend=np.zeros((1, 2)))
    yaw = np.arctan2(d[:, 1], np.maximum(d[:, 0], 1e-6))
    v = np.linalg.norm(d, axis=1) / DT
    v[0] = v0
    return np.stack([x, y, yaw, v], 1)


def classify(poses):
    import torch
    from refb_labels import classify_maneuver_v2
    return int(classify_maneuver_v2(torch.from_numpy(np.asarray(poses, np.float32))[None])[0])


def route_gt_at(gtp, i):
    """Logged strategic route at tick i, via the programme's own v2.1 derivation.

    `route_from_future_v21` is adaptive-horizon and **never-straight-by-default**: by
    Sayed's 2026-07-20 ruling a wide road sweep is ROAD FOLLOWING, not a route event, so
    it returns ROUTE_UNKNOWN with `reason='road_following'` rather than inventing a
    class. We keep the reason AND the graded `net_dyaw`, because "this clip contains no
    junction-scale strategic event" is a measurement, not a missing metric — and the
    graded signal still supports a strategic read where the discrete class refuses one.
    """
    import torch
    from refb_labels import route_from_future_v21
    try:
        r = route_from_future_v21(torch.from_numpy(gtp.astype(np.float32)), int(i))
        return (int(r["route"]), bool(r["valid"]), str(r.get("reason", "?")),
                float(r.get("net_dyaw", 0.0)))
    except Exception as e:                                   # noqa: BLE001
        return 3, False, f"error:{type(e).__name__}", 0.0


def actors_at(tracks, frac, ego):
    """Annotated agents at normalised clip time `frac`, in the ego frame."""
    out = []
    if tracks is None:
        return out
    for i in range(len(tracks)):
        T = tracks.pose_at(i, frac)
        if T is None:
            continue
        r = ego_frame(T[:2, 3] - np.array(ego[:2]), ego[3])
        out.append({"id": tracks.ids[i], "xy": [float(T[0, 3]), float(T[1, 3])],
                    "yaw": float(math.atan2(T[1, 0], T[0, 0])),
                    "rig": [float(r[0]), float(r[1])],
                    "dist": float(np.linalg.norm(r))})
    return out


def per_step_metrics(rec, gt, tracks=None):
    xy, gyaw, gv, gtp = gt_arrays(gt)
    N = len(gt)
    steps = rec["steps"]
    ego_poses = np.array([[s["ego"][0], s["ego"][1], s["ego"][3], s["v"]] for s in steps])
    ts0, ts1 = gt[0]["ts_us"], gt[-1]["ts_us"]
    out = []
    for k, st in enumerate(steps):
        e = st["ego"]
        plan = np.array(st["plan"], float)
        i = int(st["i_gt"])
        trunc = (i + WP_STEPS[-1]) >= N
        gtc = np.stack([ego_frame(xy[min(i + h, N - 1)] - xy[i], gyaw[i]) for h in WP_STEPS])
        de = np.linalg.norm(plan - gtc, axis=1)
        lon = np.abs(plan[:, 0] - gtc[:, 0])
        lat = np.abs(plan[:, 1] - gtc[:, 1])
        ct, ci, _ = cross_track(np.array(e[:2]), xy)

        j = min(i + WP_STEPS[-1], N - 1)
        dyaw_gt = wrap(gyaw[j] - gyaw[i])
        arc_gt = float(np.linalg.norm(np.diff(xy[i:j + 1], axis=0), axis=1).sum())
        pp = plan_poses(plan, st["v"])
        dyaw_pl = wrap(float(pp[-1, 2]))
        arc_pl = float(np.linalg.norm(np.diff(pp[:, :2], axis=0), axis=1).sum())
        kap_gt = dyaw_gt / max(arc_gt, 1.0)
        kap_pl = dyaw_pl / max(arc_pl, 1.0)
        yr_exec = st["v"] / 2.7 * math.tan(st["steer"])
        yr_gt = wrap(gyaw[min(i + 1, N - 1)] - gyaw[i]) / DT

        man_plan = classify(pp)
        man_gt = classify(np.column_stack([xy[i:j + 1, 0], xy[i:j + 1, 1],
                                           gyaw[i:j + 1], gv[i:j + 1]])) if j > i else 0
        man_exec = None
        if k + 20 < len(steps):
            sub = ego_poses[k:k + 21]
            man_exec = classify(sub)
        ex = st.get("extra", {})
        man_head = int(np.argmax(ex["maneuver_logits"])) if "maneuver_logits" in ex else None
        route_head = int(np.argmax(ex["s_route_logits"])) if "s_route_logits" in ex else None
        rgt, rok, rreason, rnet = route_gt_at(gtp, i)

        frac = float(np.clip((st["t_us"] - ts0) / max(ts1 - ts0, 1.0), 0.0, 1.0))
        acts = actors_at(tracks, frac, e)
        lead_idx, headway, tgap, ttc = -1, None, None, None
        cand = [(a["rig"][0], n) for n, a in enumerate(acts)
                if a["rig"][0] > 0 and abs(a["rig"][1]) < LEAD_HALF_W and a["rig"][0] < LEAD_MAX_X]
        if cand:
            xlead, lead_idx = min(cand)
            headway = float(xlead - EGO_LEN)
            tgap = headway / max(st["v"], 0.1)
        out.append({
            "k": k, "i_gt": i, "trunc": bool(trunc),
            "de": de.tolist(), "ade": float(de.mean()), "de2s": float(de[-1]),
            "lon": lon.tolist(), "lat": lat.tolist(),
            "lon_ade": float(lon.mean()), "lat_ade": float(lat.mean()),
            "cross_track": float(ct), "dist_to_gt": float(abs(ct)),
            "speed_err": float(st["v_target"] - gv[i]),
            "speed_track_err": float(st["v"] - gv[i]),
            "v_gt": float(gv[i]), "v_ego": float(st["v"]), "v_target": float(st["v_target"]),
            "heading_err": float(wrap(dyaw_pl - dyaw_gt)),
            "curv_err": float(abs(kap_pl - kap_gt)), "curv_plan": float(kap_pl),
            "curv_gt": float(kap_gt),
            "yawrate_err": float(abs(yr_exec - yr_gt)),
            "man_plan": man_plan, "man_gt": man_gt, "man_exec": man_exec,
            "man_head": man_head, "route_gt": rgt, "route_valid": rok,
            "route_reason": rreason, "route_net_dyaw": rnet, "route_head": route_head,
            "corridor_departure": int(abs(ct) > CORRIDOR_M),
            "headway": headway, "time_gap": tgap, "ttc": ttc,
            "lead_idx": lead_idx, "actors": acts,
        })
    return out


def rollout_summary(rec, gt, tracks=None):
    m = per_step_metrics(rec, gt, tracks)
    xy, gyaw, gv, gtp = gt_arrays(gt)
    e0, e1 = rec["steps"][0]["ego"], rec["steps"][-1]["ego"]
    driven = float(np.linalg.norm(np.array(e1[:2]) - np.array(e0[:2])))
    i0, i1 = rec["steps"][0]["i_gt"], rec["steps"][-1]["i_gt"]
    gt_dist = float(np.linalg.norm(np.diff(xy[i0:i1 + 1], axis=0), axis=1).sum()) if i1 > i0 else 0.0
    return m, {"start": rec["start_frame"], "n": len(m), "driven_m": driven,
               "gt_dist_m": gt_dist,
               "progress_rel": (driven / gt_dist) if gt_dist > 1 else None,
               "max_cross_track": float(max(abs(x["cross_track"]) for x in m)),
               "corridor_departure_rate": float(np.mean([x["corridor_departure"] for x in m]))}


# --------------------------------------------------------------------------------- #
def _ci(vals, eid, reduce="mean"):
    from taniteval.ci import episode_cluster_bootstrap
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return {"n": 0, "reason": "no finite values"}
    r = episode_cluster_bootstrap(v[ok], list(np.asarray(eid)[ok]), reduce=reduce)
    r["n_used"] = int(ok.sum())
    r["n_total"] = int(v.size)
    return r


def _paired(a, b, eid):
    from taniteval.ci import paired_episode_cluster_bootstrap
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return {"n": 0, "reason": "no jointly finite windows"}
    r = paired_episode_cluster_bootstrap(a[ok], b[ok], list(np.asarray(eid)[ok]))
    r["n_used"] = int(ok.sum())
    return r


def collect(path, tracks=None, drop_truncated=True):
    d = load_rollouts(path)
    gt = d["gt"]
    rows, eids, summ = [], [], []
    for rec in d["rollouts"]:
        m, s = rollout_summary(rec, gt, tracks)
        summ.append(s)
        for x in m:
            if drop_truncated and x["trunc"]:
                continue
            rows.append(x)
            eids.append(rec["start_frame"])
    return d, rows, np.array(eids), summ


def families(rows, eids, summ=None):
    g = lambda k: [r[k] for r in rows]
    fam = {}
    fam["ADE"] = {
        "ade_0_2s": _ci(g("ade"), eids),
        **{f"de_{h}": _ci([r["de"][i] for r in rows], eids) for i, h in enumerate(HORIZ)},
        "dist_to_gt_traj_m": _ci(g("dist_to_gt"), eids),
        "note": "plan vs the log's own motion from the matched point; `dist_to_gt_traj_m` "
                "is the CLOSED-LOOP deviation of the driven path from the logged path.",
    }
    fam["LONGITUDINAL"] = {
        "target_speed_err_ms": _ci(g("speed_err"), eids),
        "abs_target_speed_err_ms": _ci([abs(x) for x in g("speed_err")], eids),
        "executed_speed_err_ms": _ci(g("speed_track_err"), eids),
        "along_track_ade_m": _ci(g("lon_ade"), eids),
        **{f"along_track_{h}_m": _ci([r["lon"][i] for r in rows], eids)
           for i, h in enumerate(HORIZ)},
    }
    hw = [r["headway"] for r in rows if r["headway"] is not None]
    hwe = [e for r, e in zip(rows, eids) if r["headway"] is not None]
    if hw:
        fam["LONGITUDINAL"]["headway_m"] = _ci(hw, hwe)
        fam["LONGITUDINAL"]["time_gap_s"] = _ci(
            [r["time_gap"] for r in rows if r["headway"] is not None], hwe)
        fam["LONGITUDINAL"]["lead_present_rate"] = float(len(hw) / max(len(rows), 1))
    else:
        fam["LONGITUDINAL"]["distance_keeping"] = {
            "n": 0, "reason": "no annotated agent inside the in-lane window "
                              f"(|y|<{LEAD_HALF_W} m, 0<x<{LEAD_MAX_X} m) at any tick"}
    fam["LATERAL"] = {
        "heading_err_rad": _ci([abs(x) for x in g("heading_err")], eids),
        "curvature_err_1pm": _ci(g("curv_err"), eids),
        "yawrate_err_rads": _ci(g("yawrate_err"), eids),
        "cross_track_abs_m": _ci([abs(x) for x in g("cross_track")], eids),
        "cross_track_signed_m": _ci(g("cross_track"), eids),
        "lateral_ade_m": _ci(g("lat_ade"), eids),
    }
    cm_pg = np.zeros((5, 5), int)
    cm_pe = np.zeros((5, 5), int)
    for r in rows:
        cm_pg[r["man_gt"], r["man_plan"]] += 1
        if r["man_exec"] is not None:
            cm_pe[r["man_plan"], r["man_exec"]] += 1
    fam["TACTICAL"] = {
        "manoeuvre_plan_eq_logged": _ci([float(r["man_plan"] == r["man_gt"]) for r in rows], eids),
        "manoeuvre_exec_eq_plan": _ci(
            [float(r["man_exec"] == r["man_plan"]) if r["man_exec"] is not None else np.nan
             for r in rows], eids),
        "plan_class_share": {MAN_NAMES[c]: round(float(np.mean([r["man_plan"] == c for r in rows])), 4)
                             for c in range(5)},
        "logged_class_share": {MAN_NAMES[c]: round(float(np.mean([r["man_gt"] == c for r in rows])), 4)
                               for c in range(5)},
        "confusion_logged_x_planned": cm_pg.tolist(),
        "confusion_planned_x_executed": cm_pe.tolist(),
        "class_names": list(MAN_NAMES),
    }
    if rows and rows[0]["man_head"] is not None:
        fam["TACTICAL"]["head_eq_plan"] = _ci(
            [float(r["man_head"] == r["man_plan"]) for r in rows], eids)
        fam["TACTICAL"]["head_eq_logged"] = _ci(
            [float(r["man_head"] == r["man_gt"]) for r in rows], eids)
        fam["TACTICAL"]["head_class_share"] = {
            MAN_NAMES[c]: round(float(np.mean([r["man_head"] == c for r in rows])), 4)
            for c in range(5)}
    else:
        fam["TACTICAL"]["maneuver_head"] = {
            "n": 0, "reason": "this arm exposes no maneuver_head logits at the deploy path"}

    vr = [r for r in rows if r["route_valid"]]
    ver = [e for r, e in zip(rows, eids) if r["route_valid"]]
    has_head = any(r["route_head"] is not None for r in rows)
    reasons = {}
    for r in rows:
        reasons[r["route_reason"]] = reasons.get(r["route_reason"], 0) + 1
    fam["STRATEGIC"] = {
        "route_corridor_departure_rate": _ci(g("corridor_departure"), eids),
        "route_logged_share": {ROUTE_NAMES[c]: round(float(np.mean([r["route_gt"] == c for r in rows])), 4)
                               for c in range(4)},
        "route_label_valid_rate": round(float(np.mean([r["route_valid"] for r in rows])), 4),
        "route_derivation_reason_share": {k: round(v / len(rows), 4) for k, v in reasons.items()},
        "logged_net_dyaw_rad": _ci(g("route_net_dyaw"), eids),
    }
    # discrete route agreement, only where a route judgement EXISTS
    if not vr:
        fam["STRATEGIC"]["route_head_eq_logged"] = {
            "n": 0,
            "reason": ("no VALID logged route class on any window: "
                       f"route_from_future_v21 returned ROUTE_UNKNOWN on {len(rows)}/{len(rows)} "
                       f"windows (reasons {reasons}). Under Sayed's 2026-07-20 ruling a wide "
                       "road sweep is ROAD FOLLOWING, not a route event — this 20 s clip "
                       "contains no junction-scale strategic decision, so the discrete "
                       "strategic class is genuinely undefined here. This is a property of "
                       "the SCENE, not a missing instrument: the graded `logged_net_dyaw_rad`, "
                       "the corridor-departure rate and the route progress below ARE the "
                       "strategic read for this clip.")}
    elif has_head:
        fam["STRATEGIC"]["route_head_eq_logged"] = _ci(
            [float(r["route_head"] == r["route_gt"]) for r in vr], ver)
        # ⛔ a perfect score against a single-class label is not skill. Say so on the
        # number itself, not in a footnote nobody reads.
        n_lab = len({r["route_gt"] for r in vr})
        n_pred = len({r["route_head"] for r in vr})
        if n_lab < 2 or n_pred < 2:
            fam["STRATEGIC"]["route_head_eq_logged"]["degenerate"] = True
            fam["STRATEGIC"]["route_head_eq_logged"]["degenerate_note"] = (
                f"NOT SKILL: the logged route takes {n_lab} distinct value(s) and the "
                f"head predicts {n_pred} on these windows. A constant predictor scores "
                "the same. This clip contains no junction-scale strategic decision, so "
                "the discrete strategic metric cannot discriminate here — read "
                "`route_head_side_eq_graded_proxy` and `route_corridor_departure_rate` "
                "instead, and get a scene WITH a junction before quoting a strategic "
                "accuracy.")
    else:
        fam["STRATEGIC"]["route_head_eq_logged"] = {
            "n": 0, "reason": "this arm exposes no strategic route logits at the deploy path"}
    if has_head:
        fam["STRATEGIC"]["route_head_share"] = {
            ROUTE_NAMES[c]: round(float(np.mean([r["route_head"] == c for r in rows])), 4)
            for c in range(3)}
        # graded proxy: does the head's LATERAL intent agree with the sign of the
        # logged cumulative heading change? Deadband 0.087 rad (5 deg) over the
        # available future. Labelled a PROXY — it is not the trained target.
        def sgn(x, dead=0.087):
            return 1 if x > dead else (2 if x < -dead else 0)   # left / right / straight
        gt_side = [sgn(r["route_net_dyaw"]) for r in rows]
        hd = [{0: 1, 1: 0, 2: 2}.get(r["route_head"], 0) for r in rows]  # head -> l/s/r idx
        fam["STRATEGIC"]["route_head_side_eq_graded_proxy"] = _ci(
            [float(a == b) for a, b in zip(gt_side, hd)], eids)
        fam["STRATEGIC"]["route_head_side_note"] = (
            "PROXY, not the trained target: agreement between the route head's lateral "
            "side and sign(cumulative logged heading change) with a 5 deg deadband. "
            "Reported because the discrete route class is UNKNOWN on this clip.")
    else:
        fam["STRATEGIC"]["route_head"] = {
            "n": 0, "reason": "this arm exposes no strategic route logits at the deploy path"}
    if summ:
        pr = [s["progress_rel"] for s in summ if s["progress_rel"] is not None]
        se = [s["start"] for s in summ if s["progress_rel"] is not None]
        if pr:
            fam["STRATEGIC"]["route_progress_rel"] = _ci(pr, se)
            fam["STRATEGIC"]["route_progress_note"] = (
                "straight-line distance the ego covered divided by the logged path length "
                "over the same tick span; one value per rollout, so the cluster IS the "
                "window here (n_windows == n_episodes by construction).")
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm A rollouts json")
    ap.add_argument("--b", default=None, help="arm B rollouts json (paired)")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from gsplat_renderer import ActorTracks
    tr = ActorTracks(args.tracks) if args.tracks and Path(args.tracks).exists() else None

    dA, rA, eA, sA = collect(args.a, tr)
    res = {"arm_A": {"name": dA["arm"], "condition": dA["condition"], "ckpt": dA["ckpt"],
                     "f_eff": dA["f_eff"], "n_windows": len(rA),
                     "n_clusters": int(len(set(eA.tolist()))),
                     "rollouts": sA, "families": families(rA, eA, sA)}}
    res["estimator_note"] = (
        "episode-cluster bootstrap over ROLLOUT STARTS. The clusters are disjoint "
        "segments of ONE clip, not independent episodes — the interval is the right "
        "estimator for the resampling unit available, and the unit is stated here so "
        "it is never mistaken for the 40-episode val bootstrap.")
    res["within_sim_note"] = (
        "WITHIN-SIM RELATIVE. REF-C open-loop ADE is 1.5157 on these NuRec "
        "reconstructions vs 0.4728 on real footage (3.21x OOD). Orderings survive; "
        "absolute rates do not.")

    if args.b:
        dB, rB, eB, sB = collect(args.b, tr)
        res["arm_B"] = {"name": dB["arm"], "condition": dB["condition"], "ckpt": dB["ckpt"],
                        "f_eff": dB["f_eff"], "n_windows": len(rB),
                        "n_clusters": int(len(set(eB.tolist()))),
                        "rollouts": sB, "families": families(rB, eB, sB)}
        # pair on identical (start, k) windows only — the paired estimator is invalid
        # the moment the two arms are not scored on the same windows.
        key = lambda rows, eids: {(int(e), int(r["k"])): r for r, e in zip(rows, eids)}
        KA, KB = key(rA, eA), key(rB, eB)
        common = sorted(set(KA) & set(KB))
        pe = [c[0] for c in common]
        pair = {}
        for lab, fn in (("ade_0_2s", lambda r: r["ade"]),
                        ("dist_to_gt_traj_m", lambda r: r["dist_to_gt"]),
                        ("abs_target_speed_err_ms", lambda r: abs(r["speed_err"])),
                        ("along_track_ade_m", lambda r: r["lon_ade"]),
                        ("heading_err_rad", lambda r: abs(r["heading_err"])),
                        ("curvature_err_1pm", lambda r: r["curv_err"]),
                        ("yawrate_err_rads", lambda r: r["yawrate_err"]),
                        ("cross_track_abs_m", lambda r: abs(r["cross_track"])),
                        ("manoeuvre_plan_eq_logged", lambda r: float(r["man_plan"] == r["man_gt"])),
                        ("route_corridor_departure_rate", lambda r: float(r["corridor_departure"]))):
            pair[lab] = _paired([fn(KA[c]) for c in common], [fn(KB[c]) for c in common], pe)
        res["paired_A_minus_B"] = pair
        res["paired_n_windows"] = len(common)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:6000])


if __name__ == "__main__":
    main()
