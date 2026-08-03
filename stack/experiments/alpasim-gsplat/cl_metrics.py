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


def actors_at(tracks, frac, ego, keep=None, dfrac=None):
    """Annotated agents at normalised clip time `frac`, in the ego frame.

    `keep` — restrict to these track INDICES. Used to score only the agents the renderer
    actually draws: crediting distance-keeping to a lead the model cannot see would be a
    metric that measures the annotation rather than the policy.
    `dfrac` — normalised-time step used to finite-difference each track's world position
    into a speed, which is what makes TTC computable at all.
    """
    out = []
    if tracks is None:
        return out
    for i in range(len(tracks)):
        if keep is not None and i not in keep:
            continue
        T = tracks.pose_at(i, frac)
        if T is None:
            continue
        r = ego_frame(T[:2, 3] - np.array(ego[:2]), ego[3])
        spd = None
        if dfrac:
            Tp = tracks.pose_at(i, min(1.0, frac + dfrac))
            Tm = tracks.pose_at(i, max(0.0, frac - dfrac))
            if Tp is not None and Tm is not None:
                dt = (min(1.0, frac + dfrac) - max(0.0, frac - dfrac))
                if dt > 0:
                    spd = float(np.linalg.norm(Tp[:2, 3] - Tm[:2, 3]) / (dt * _CLIP_SPAN_S[0]))
        out.append({"id": tracks.ids[i], "idx": i,
                    "xy": [float(T[0, 3]), float(T[1, 3])],
                    "yaw": float(math.atan2(T[1, 0], T[0, 0])),
                    "rig": [float(r[0]), float(r[1])],
                    "v": spd,
                    "dist": float(np.linalg.norm(r))})
    return out


# clip duration in seconds, set once per rollout file so `actors_at` can turn a
# normalised-time difference into a real speed. A module-level cell rather than a global
# constant because it is a property of the clip, not of the metric.
_CLIP_SPAN_S = [20.0]


def per_step_metrics(rec, gt, tracks=None, lead_ref=None, keep_tracks=None):
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
        # ⚠️ CORRECTED 2026-08-03: this read only `s_route_logits`. flagship-v1 emits that
        # name; refc-base emits `route_logits` — on 450/450 steps of all six conditions.
        # The panel therefore published REF-C's STRATEGIC route head as
        # {"n":0,"reason":"this arm exposes no strategic route logits at the deploy path"},
        # which is FALSE: the head was there the whole time and a key-name mismatch
        # deleted a whole metric family for one of the two arms. Probe both names.
        rl = ex.get("s_route_logits", ex.get("route_logits"))
        route_head = int(np.argmax(rl)) if rl is not None else None
        rgt, rok, rreason, rnet = route_gt_at(gtp, i)

        frac = float(np.clip((st["t_us"] - ts0) / max(ts1 - ts0, 1.0), 0.0, 1.0))
        acts = actors_at(tracks, frac, e, keep=keep_tracks, dfrac=0.005)
        lead_idx, headway, tgap, ttc, v_lead = -1, None, None, None, None
        cand = [(a["rig"][0], n) for n, a in enumerate(acts)
                if a["rig"][0] > 0 and abs(a["rig"][1]) < LEAD_HALF_W and a["rig"][0] < LEAD_MAX_X]
        if cand:
            xlead, lead_idx = min(cand)
            headway = float(xlead - EGO_LEN)
            tgap = headway / max(st["v"], 0.1)
            v_lead = acts[lead_idx].get("v")
            # TTC is only defined while CLOSING. An opening gap has no time-to-collision
            # and must not be recorded as a large-but-finite one, which would make a
            # mean over it meaningless.
            if v_lead is not None:
                closing = st["v"] - v_lead
                if closing > 0.1 and headway > 0:
                    ttc = float(headway / closing)
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
            "nav": (int(st["nav"]) if st.get("nav") is not None else None),
            "man_plan": man_plan, "man_gt": man_gt, "man_exec": man_exec,
            "man_head": man_head, "route_gt": rgt, "route_valid": rok,
            "route_reason": rreason, "route_net_dyaw": rnet, "route_head": route_head,
            "corridor_departure": int(abs(ct) > CORRIDOR_M),
            "headway": headway, "time_gap": tgap, "ttc": ttc, "v_lead": v_lead,
            "lead_idx": lead_idx, "actors": acts,
        })
        # --- CONSTRUCTED lead geometry (synth_actor.LeadGeometry) ------------------
        # Recorded on EVERY run including `empty`, so the control carries the matched
        # COUNTERFACTUAL headway and the with-vs-without pairing has something to
        # compare. `lead_ref` names which constructed condition's geometry to score.
        L = st.get("lead")
        if L and lead_ref and lead_ref in L:
            e = L[lead_ref]
            tg = e.get("time_gap_s")
            out[-1].update({
                "synth_headway": e.get("headway_m"), "synth_time_gap": tg,
                "synth_ttc": e.get("ttc_s"), "synth_x": e.get("x"),
                "synth_y": e.get("y"), "synth_v_lead": e.get("v_lead"),
                "synth_tg_below_1s": (float(tg < 1.0) if tg is not None else np.nan),
                "synth_tg_below_0_5s": (float(tg < 0.5) if tg is not None else np.nan),
                # bumper-to-bumper headway <= 0 means the ego has driven INTO the lead.
                # The lead is a rendered object with no physics, so nothing stops it —
                # which is precisely why it has to be counted rather than assumed away.
                #
                # ⚠️ CORRECTED 2026-08-03 — THIS TEST HAD NO LATERAL GATE, so it counted a
                # longitudinal PASS as a collision. MEASURED on the banked rollouts: of
                # flagship-v1/cutin's 13 "collisions" the median |y| was 13.795 m and
                # NONE were in-lane; flagship-v1/lead8's 33 had median |y| 8.877 m, also
                # 0 % in-lane. A car 13 m to the side is not a collision. `synth_collision`
                # is kept UNGATED under its old name for continuity and is no longer the
                # quotable number; `synth_collision_inlane` is, and the precision of the
                # ungated detector against the gated truth is reported next to it.
                "synth_collision": (float(e["headway_m"] <= 0.0)
                                    if e.get("headway_m") is not None else np.nan),
                "synth_collision_inlane": (
                    float(e["headway_m"] <= 0.0 and abs(e["y"]) <= LEAD_HALF_W)
                    if (e.get("headway_m") is not None and e.get("y") is not None)
                    else np.nan),
                "synth_collision_abs_y": (abs(e["y"]) if e.get("y") is not None else np.nan),
                # is the lead actually in the ego's lane at this tick? On a curving road
                # a vehicle 15 m ahead is genuinely offset in the ego frame, so this is
                # reported, not asserted.
                "synth_inlane": (float(abs(e["y"]) <= 1.8)
                                 if e.get("y") is not None else np.nan),
            })
    return out


def rollout_summary(rec, gt, tracks=None, lead_ref=None, keep_tracks=None):
    m = per_step_metrics(rec, gt, tracks, lead_ref, keep_tracks)
    xy, gyaw, gv, gtp = gt_arrays(gt)
    e0, e1 = rec["steps"][0]["ego"], rec["steps"][-1]["ego"]
    driven = float(np.linalg.norm(np.array(e1[:2]) - np.array(e0[:2])))
    i0, i1 = rec["steps"][0]["i_gt"], rec["steps"][-1]["i_gt"]
    gt_dist = float(np.linalg.norm(np.diff(xy[i0:i1 + 1], axis=0), axis=1).sum()) if i1 > i0 else 0.0
    s = {"start": rec["start_frame"], "n": len(m), "driven_m": driven,
         "gt_dist_m": gt_dist,
         "progress_rel": (driven / gt_dist) if gt_dist > 1 else None,
         "max_cross_track": float(max(abs(x["cross_track"]) for x in m)),
         "corridor_departure_rate": float(np.mean([x["corridor_departure"] for x in m]))}
    hw = [x["synth_headway"] for x in m if x.get("synth_headway") is not None]
    if hw:
        s["synth_min_headway_m"] = round(float(min(hw)), 3)
        s["synth_mean_headway_m"] = round(float(np.mean(hw)), 3)
    return m, s


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


def per_class_pr(y_true, y_pred, names):
    """Per-class PRECISION, RECALL, F1 and BOTH denominators.

    ⛔ BINDING (this programme, 2026-08-03): a rate is never reported without the price it
    pays. A recall-only frontier published a "brake_stop 0.026 -> 0.503 free win" that was
    really 0.0719 -> 0.4248 recall bought with precision 0.2340 -> 0.1711 — 380 fires for
    153 true cases. So precision, recall, support (n true) and n_pred (n fires) travel
    together here, plus the majority-class baseline that any constant predictor achieves.
    """
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    out, f1s = {}, []
    for c, nm in enumerate(names):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else 0.0
        f1s.append(f1)
        out[nm] = {"precision": (round(prec, 4) if prec is not None else None),
                   "recall": (round(rec, 4) if rec is not None else None),
                   "f1": round(f1, 4), "support_n_true": tp + fn, "n_fires": tp + fp}
    supp = np.bincount(y_true, minlength=len(names))
    out["_macro_f1"] = round(float(np.mean(f1s)), 4)
    out["_accuracy"] = round(float((y_true == y_pred).mean()), 4) if y_true.size else None
    out["_majority_class_baseline_acc"] = (
        round(float(supp.max() / supp.sum()), 4) if supp.sum() else None)
    out["_n"] = int(y_true.size)
    return out


def _paired(a, b, eid):
    from taniteval.ci import paired_episode_cluster_bootstrap
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return {"n": 0, "reason": "no jointly finite windows"}
    r = paired_episode_cluster_bootstrap(a[ok], b[ok], list(np.asarray(eid)[ok]))
    r["n_used"] = int(ok.sum())
    return r


def collect(path, tracks=None, drop_truncated=True, lead_ref=None, keep_tracks=None):
    d = load_rollouts(path)
    gt = d["gt"]
    # real clip duration, so `actors_at` can finite-difference a normalised-time step
    # into m/s instead of assuming a nominal 20 s.
    try:
        _CLIP_SPAN_S[0] = max(1e-3, (gt[-1]["ts_us"] - gt[0]["ts_us"]) / 1e6)
    except Exception:                                            # noqa: BLE001
        pass
    rows, eids, summ = [], [], []
    for rec in d["rollouts"]:
        m, s = rollout_summary(rec, gt, tracks, lead_ref, keep_tracks)
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
        # closest approach to the REAL annotated lead, and the unsafe-gap rates. These
        # are the distance-keeping numbers the family is actually about; the mean headway
        # alone cannot distinguish "never got close" from "got close and recovered".
        fam["LONGITUDINAL"]["min_headway_m"] = _ci(
            hw, hwe, reduce=lambda v: float(np.min(v)))
        fam["LONGITUDINAL"]["frac_time_gap_below_1s"] = _ci(
            [float(r["time_gap"] < 1.0) for r in rows if r["headway"] is not None], hwe)
        fam["LONGITUDINAL"]["frac_time_gap_below_0_5s"] = _ci(
            [float(r["time_gap"] < 0.5) for r in rows if r["headway"] is not None], hwe)
        ttcs = [(r["ttc"] if r["ttc"] is not None else np.nan)
                for r in rows if r["headway"] is not None]
        n_ttc = int(np.isfinite(ttcs).sum())
        fam["LONGITUDINAL"]["ttc_s_when_closing"] = (
            _ci(ttcs, hwe) if n_ttc else
            {"n": 0, "reason": "the ego never closed on the lead at > 0.1 m/s on any "
                               "tick where a lead was present — TTC is undefined, not "
                               "missing"})
        fam["LONGITUDINAL"]["ttc_defined_rate"] = round(n_ttc / max(len(hw), 1), 4)
        fam["LONGITUDINAL"]["real_lead_note"] = (
            "REAL annotated lead from the scene's own sequence_tracks, restricted to the "
            "tracks the renderer actually draws when --renderable-from is given. Headway "
            f"is bumper-to-bumper (EGO_LEN={EGO_LEN} m), in-lane band |y| < {LEAD_HALF_W} m, "
            f"search window 0 < x < {LEAD_MAX_X} m.")
    else:
        fam["LONGITUDINAL"]["distance_keeping"] = {
            "n": 0, "reason": "no annotated agent inside the in-lane window "
                              f"(|y|<{LEAD_HALF_W} m, 0<x<{LEAD_MAX_X} m) at any tick"}
    # distance-keeping against the CONSTRUCTED lead. Present on every arm and every
    # condition — in `empty` it is the counterfactual, which is exactly what makes the
    # with-vs-without contrast a paired one.
    sh = [r.get("synth_headway") for r in rows]
    if any(x is not None for x in sh):
        ok = [i for i, x in enumerate(sh) if x is not None]
        oe = [eids[i] for i in ok]
        fam["LONGITUDINAL"]["synth_lead_headway_m"] = _ci([sh[i] for i in ok], oe)
        fam["LONGITUDINAL"]["synth_lead_time_gap_s"] = _ci(
            [rows[i]["synth_time_gap"] for i in ok], oe)
        # closest approach, not the average: `min` is not one of taniteval's named
        # reducers, so it is passed as a callable (the API explicitly allows one).
        fam["LONGITUDINAL"]["synth_lead_min_headway_m"] = _ci(
            [sh[i] for i in ok], oe, reduce=lambda v: float(np.min(v)))
        fam["LONGITUDINAL"]["synth_lead_frac_time_gap_below_1s"] = _ci(
            [rows[i]["synth_tg_below_1s"] for i in ok], oe)
        fam["LONGITUDINAL"]["synth_lead_ttc_s"] = _ci(
            [(rows[i]["synth_ttc"] if rows[i]["synth_ttc"] is not None else np.nan)
             for i in ok], oe)
        fam["LONGITUDINAL"]["synth_lead_collision_rate_UNGATED_DO_NOT_QUOTE"] = _ci(
            [rows[i]["synth_collision"] for i in ok], oe)
        fam["LONGITUDINAL"]["synth_lead_collision_rate_inlane"] = _ci(
            [rows[i]["synth_collision_inlane"] for i in ok], oe)
        # what the ungated detector was PAYING: of the ticks it called a collision, how
        # many were actually in-lane? This is the precision that the ungated rate hid.
        fires = [i for i in ok if rows[i]["synth_collision"] == 1.0]
        true_in = [i for i in fires if rows[i]["synth_collision_inlane"] == 1.0]
        ys = [rows[i]["synth_collision_abs_y"] for i in fires
              if np.isfinite(rows[i].get("synth_collision_abs_y", np.nan))]
        fam["LONGITUDINAL"]["synth_lead_collision_precision"] = {
            "n_fires": len(fires), "n_true_inlane": len(true_in),
            "precision": (round(len(true_in) / len(fires), 4) if fires else None),
            "median_abs_y_of_fires_m": (round(float(np.median(ys)), 3) if ys else None),
            "note": "precision of the UNGATED headway<=0 test against the in-lane truth "
                    f"(|y| <= {LEAD_HALF_W} m). The ungated rate counts a longitudinal "
                    "PASS as a collision; quote `synth_lead_collision_rate_inlane`."}
        fam["LONGITUDINAL"]["synth_lead_inlane_rate"] = _ci(
            [rows[i]["synth_inlane"] for i in ok], oe)
        fam["LONGITUDINAL"]["synth_lead_note"] = (
            "CONSTRUCTED lead (synth_actor.py). Geometry is computed on every run, so "
            "the `empty` control carries the matched COUNTERFACTUAL headway rather than "
            "a missing value. Headway is bumper-to-bumper using the rendered cuboid's "
            "own measured extent.")
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
        # PRECISION alongside recall, per class, with both denominators.
        hrows = [r for r in rows if r["man_head"] is not None]
        fam["TACTICAL"]["head_vs_logged_per_class_PR"] = per_class_pr(
            [r["man_gt"] for r in hrows], [r["man_head"] for r in hrows], MAN_NAMES)
        fam["TACTICAL"]["plan_vs_logged_per_class_PR"] = per_class_pr(
            [r["man_gt"] for r in rows], [r["man_plan"] for r in rows], MAN_NAMES)
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
        # ⛔ NAV-ECHO GUARD (added 2026-08-03 after it fired on the first scene it saw).
        # The harness FEEDS a nav command to the policy. If the route head is a
        # deterministic function of that input, `route_head_eq_logged` measures the echo
        # of the model's own conditioning, not a strategic decision — and it scores
        # perfectly, because the nav command was derived from the same log the route
        # label is derived from. MEASURED on scene 7c72937c: flagship-v1's head is an
        # exact bijection of nav (nav=1 -> head=0 on 369/369, nav=0 -> head=1 on 81/81)
        # and scored route_head_eq_logged = 1.0000 [1.0000, 1.0000]. REF-C's is not a
        # function of nav and scored 0.2605. The guard is computed, not assumed.
        nav_map = {}
        circular = None
        if all(r.get("nav") is not None for r in rows) and rows:
            for r in rows:
                nav_map.setdefault(int(r["nav"]), set()).add(r["route_head"])
            circular = all(len(v) == 1 for v in nav_map.values()) and len(nav_map) >= 1
        fam["STRATEGIC"]["route_head_nav_echo_check"] = {
            "head_is_deterministic_function_of_nav": circular,
            "nav_to_head_map": {str(k): sorted(x for x in v if x is not None)
                                for k, v in nav_map.items()},
            "n": len(rows),
            "verdict": ("CIRCULAR — route_head_eq_logged above reproduces the nav command "
                        "the policy was GIVEN and is NOT evidence of strategic skill; do "
                        "not quote it" if circular else
                        "not an echo — the head is not a function of nav on these windows"
                        if circular is False else "not evaluated (nav missing)")}
        if circular:
            for k in ("route_head_eq_logged", "route_head_side_eq_graded_proxy"):
                if isinstance(fam["STRATEGIC"].get(k), dict):
                    fam["STRATEGIC"][k]["CIRCULAR_NAV_ECHO"] = True
                    fam["STRATEGIC"][k]["do_not_quote"] = (
                        "the route head is a deterministic function of the nav input")
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
        # PRECISION alongside recall on the graded proxy, so a head that wins by always
        # saying `straight` is visible as such rather than scoring as skill.
        fam["STRATEGIC"]["route_head_side_per_class_PR"] = per_class_pr(
            gt_side, hd, ("straight", "left", "right"))
        if vr:
            fam["STRATEGIC"]["route_head_vs_logged_per_class_PR"] = per_class_pr(
                [r["route_gt"] for r in vr], [r["route_head"] for r in vr], ROUTE_NAMES)
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
    ap.add_argument("--renderable-from", default=None,
                    help="actor_map.json — restrict the REAL-lead search to the tracks "
                         "the renderer actually draws. Without it the distance-keeping "
                         "metric can credit an agent the model never saw.")
    ap.add_argument("--lead-ref", default=None,
                    help="constructed condition whose lead geometry to score "
                         "(lead25/lead15/lead8/cutin/behind). Applied to BOTH arms so "
                         "the empty control gets the matched counterfactual.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from gsplat_renderer import ActorTracks
    tr = ActorTracks(args.tracks) if args.tracks and Path(args.tracks).exists() else None
    keep = None
    if args.renderable_from and Path(args.renderable_from).exists():
        am = json.loads(Path(args.renderable_from).read_text())
        keep = {int(x["best_track"]) for x in am["per_track"] if x["accepted"]}

    dA, rA, eA, sA = collect(args.a, tr, lead_ref=args.lead_ref, keep_tracks=keep)
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

    res["lead_ref"] = args.lead_ref
    res["renderable_restricted"] = (None if keep is None else
                                    {"source": args.renderable_from, "n_tracks": len(keep)})
    if args.b:
        dB, rB, eB, sB = collect(args.b, tr, lead_ref=args.lead_ref, keep_tracks=keep)
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
                        ("lateral_ade_m", lambda r: r["lat_ade"]),
                        ("executed_speed_err_ms", lambda r: r["speed_track_err"]),
                        ("abs_executed_speed_err_ms", lambda r: abs(r["speed_track_err"])),
                        ("synth_lead_headway_m", lambda r: r.get("synth_headway", np.nan)),
                        ("synth_lead_time_gap_s", lambda r: r.get("synth_time_gap", np.nan)),
                        ("synth_lead_frac_tg_below_1s",
                         lambda r: r.get("synth_tg_below_1s", np.nan)),
                        ("synth_lead_collision_rate_inlane",
                         lambda r: r.get("synth_collision_inlane", np.nan)),
                        # REAL annotated lead — the distance-keeping contrast
                        ("real_lead_headway_m",
                         lambda r: (r["headway"] if r["headway"] is not None else np.nan)),
                        ("real_lead_time_gap_s",
                         lambda r: (r["time_gap"] if r["time_gap"] is not None else np.nan)),
                        ("real_lead_frac_tg_below_1s",
                         lambda r: (float(r["time_gap"] < 1.0)
                                    if r["time_gap"] is not None else np.nan)),
                        ("real_lead_ttc_s_when_closing",
                         lambda r: (r["ttc"] if r.get("ttc") is not None else np.nan)),
                        ("manoeuvre_head_eq_logged",
                         lambda r: (float(r["man_head"] == r["man_gt"])
                                    if r["man_head"] is not None else np.nan)),
                        ("route_head_eq_logged",
                         lambda r: (float(r["route_head"] == r["route_gt"])
                                    if (r["route_head"] is not None and r["route_valid"])
                                    else np.nan)),
                        ("route_corridor_departure_rate", lambda r: float(r["corridor_departure"]))):
            va = [fn(KA[c]) for c in common]
            vb = [fn(KB[c]) for c in common]
            va = [np.nan if v is None else v for v in va]
            vb = [np.nan if v is None else v for v in vb]
            pair[lab] = _paired(va, vb, pe)
        res["paired_A_minus_B"] = pair
        res["paired_n_windows"] = len(common)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:6000])


if __name__ == "__main__":
    main()
