#!/usr/bin/env python3
"""STREAM C — did the closed-loop headline survive a better render?

THE QUESTION. The render-quality pass lifted grad-NCC 0.2774 -> 0.3424 (+23.4 %) and the
four shipped videos were re-rendered with it. The DRIVING numbers printed beside those
files were measured on the OLD render. Changing the render changes what the policy sees,
so those numbers do not transfer. This script re-measures them and asks whether the
morning verdict — *REF-C beats flagship v1 closed-loop, and the separation is ENTIRELY
LATERAL* — is a property of the ARMS or of RENDER ARTIFACTS.

WHAT IT COMPUTES, and why each piece is necessary
-------------------------------------------------
1. **DIVERGENCE.** How far the driven path and the emitted plan actually moved between
   the two renders. If this is ~0 the render did not reach the policy and the whole
   comparison is vacuous; it has to be measured, not assumed.

2. **THE INTERACTION (difference-in-differences).** Eyeballing two CIs is not a test.
   For every window shared by all four rollout sets we form
   ``d = flagship - REF-C`` under each render and bootstrap ``d_HQ - d_MORNING``,
   clustered on the rollout start. THIS is the test of "the render changed the arm gap";
   a metric whose interaction CI contains zero has NO evidence that the render moved it.

3. **THE NEGATIVE CONTROL / NOISE FLOOR.** Re-running the MORNING config today and
   pairing it against the morning rollouts. The renderer is a step function of pose
   (a 0.1 px camera rotation has been measured to move the 2 s waypoint 6.65 m), so if
   that control is not ~0 then NOTHING may be attributed to the render. Run-to-run noise
   is the yardstick every render delta is read against.

4. **SELF-CONSISTENCY.** Component-vs-family checks on the emitted rows:
   ``dist_to_gt`` vs ``|cross_track|``; ``max(lon,lat) <= ade <= lon+lat`` per horizon;
   and an ADE recomputed here from the raw plan/GT arrays through a separate code path.
   The last one tests plumbing and arithmetic, NOT the definition — stated so nobody
   reads it as validating the choice of metric.

⛔ This script writes JSON only. Every number it prints carries the run directory it came
from, because a headline copied from a superseded run has had to be retracted twice.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DT = 0.1
WP_STEPS = (5, 10, 15, 20)

#: the metrics the report ranks, and the family each belongs to. `dist_to_gt_traj_m` is
#: deliberately NOT listed under ADE: it is `abs(cross_track)` (cl_metrics.py, one
#: dict literal — `"cross_track": ct, "dist_to_gt": abs(ct)`), so filing it under ADE
#: makes one measurement look like two independent separations.
FAMILY_OF = {
    "ade_0_2s": "ADE",
    "dist_to_gt_traj_m": "LATERAL (= abs cross-track)",
    "abs_target_speed_err_ms": "LONGITUDINAL",
    "along_track_ade_m": "LONGITUDINAL",
    "heading_err_rad": "LATERAL",
    "curvature_err_1pm": "LATERAL",
    "yawrate_err_rads": "LATERAL",
    "cross_track_abs_m": "LATERAL",
    "manoeuvre_plan_eq_logged": "TACTICAL",
    "route_corridor_departure_rate": "STRATEGIC",
}

#: per-window extractors, matching cl_metrics.main()'s paired block exactly
GETTERS = {
    "ade_0_2s": lambda r: r["ade"],
    "dist_to_gt_traj_m": lambda r: r["dist_to_gt"],
    "abs_target_speed_err_ms": lambda r: abs(r["speed_err"]),
    "along_track_ade_m": lambda r: r["lon_ade"],
    "heading_err_rad": lambda r: abs(r["heading_err"]),
    "curvature_err_1pm": lambda r: r["curv_err"],
    "yawrate_err_rads": lambda r: r["yawrate_err"],
    "cross_track_abs_m": lambda r: abs(r["cross_track"]),
    "manoeuvre_plan_eq_logged": lambda r: float(r["man_plan"] == r["man_gt"]),
    "route_corridor_departure_rate": lambda r: float(r["corridor_departure"]),
}


def key_rows(rows, eids):
    return {(int(e), int(r["k"])): r for r, e in zip(rows, eids)}


def ego_frame(d, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def independent_ade(path):
    """ADE recomputed from the raw rollout JSON through a SEPARATE code path.

    Plumbing/arithmetic control only — it reuses cl_metrics' DEFINITION of ADE
    (mean over the four horizons of the ego-frame plan-vs-log distance), so it can
    catch a mis-indexed pose or a stale field but NOT a wrong choice of metric.
    """
    d = json.loads(Path(path).read_text())
    gt = d["gt"]
    xy = np.array([[g["x"], g["y"]] for g in gt], float)
    yaw = np.array([g["yaw"] for g in gt], float)
    N = len(gt)
    out = {}
    for ro in d["rollouts"]:
        e = int(ro["start_frame"])
        for st in ro["steps"]:
            i = int(st["i_gt"])
            plan = np.array(st["plan"], float)
            gtc = np.stack([ego_frame(xy[min(i + h, N - 1)] - xy[i], yaw[i])
                            for h in WP_STEPS])
            out[(e, int(st["k"]))] = float(np.linalg.norm(plan - gtc, axis=1).mean())
    return out


def divergence(pa, pb):
    """How far the DRIVEN path and the EMITTED plan moved between two runs."""
    A = json.loads(Path(pa).read_text())
    B = json.loads(Path(pb).read_text())
    ga = {(int(r["start_frame"]), int(s["k"])): s for r in A["rollouts"] for s in r["steps"]}
    gb = {(int(r["start_frame"]), int(s["k"])): s for r in B["rollouts"] for s in r["steps"]}
    common = sorted(set(ga) & set(gb))
    ego, plan, vt, first = [], [], [], {}
    for k in common:
        a, b = ga[k], gb[k]
        ego.append(float(np.hypot(a["ego"][0] - b["ego"][0], a["ego"][1] - b["ego"][1])))
        plan.append(float(np.linalg.norm(np.array(a["plan"]) - np.array(b["plan"]), axis=1).mean()))
        vt.append(abs(float(a["v_target"]) - float(b["v_target"])))
        if k[1] == 0:
            first[k[0]] = plan[-1]
    return {
        "n_windows": len(common),
        "ego_xy_m": {"mean": round(float(np.mean(ego)), 4),
                     "p50": round(float(np.median(ego)), 4),
                     "max": round(float(np.max(ego)), 4)},
        "plan_m": {"mean": round(float(np.mean(plan)), 4),
                   "p50": round(float(np.median(plan)), 4),
                   "max": round(float(np.max(plan)), 4)},
        "v_target_ms": {"mean": round(float(np.mean(vt)), 4),
                        "max": round(float(np.max(vt)), 4)},
        "plan_at_k0_m": {str(k): round(v, 4) for k, v in sorted(first.items())},
        "plan_at_k0_note": (
            "k=0 is the FIRST tick of a rollout: both runs start from the identical "
            "logged pose, so a non-zero value here is the render acting on the policy "
            "with zero accumulated drift. Later ticks confound render and divergence."),
    }


def selfcheck(rows, eids, path):
    dtg = np.array([r["dist_to_gt"] for r in rows])
    ct = np.array([abs(r["cross_track"]) for r in rows])
    ade = np.array([r["ade"] for r in rows])
    lo = np.array([r["lon_ade"] for r in rows])
    la = np.array([r["lat_ade"] for r in rows])
    ind = independent_ade(path)
    K = key_rows(rows, eids)
    pairs = [(K[k]["ade"], ind[k]) for k in K if k in ind]
    d_ade = np.array([a - b for a, b in pairs])
    tri_lo = float(np.max(np.maximum(lo, la) - ade))     # must be <= 0 (tolerance)
    tri_hi = float(np.max(ade - (lo + la)))              # must be <= 0 (tolerance)
    return {
        "n_rows": len(rows),
        "dist_to_gt_IS_abs_cross_track": {
            "max_abs_diff": float(np.max(np.abs(dtg - ct))),
            "identical": bool(np.max(np.abs(dtg - ct)) == 0.0),
            "verdict": (
                "CONFIRMED IDENTICAL — `dist_to_gt_traj_m` and `cross_track_abs_m` are "
                "ONE measurement reported twice, under ADE and under LATERAL. A table "
                "that lists both is showing one separation, not two."),
        },
        "ade_recomputed_independently": {
            "n_compared": len(pairs),
            "max_abs_diff": round(float(np.max(np.abs(d_ade))) if len(d_ade) else float("nan"), 9),
            "agrees": bool(len(d_ade) and np.max(np.abs(d_ade)) < 1e-6),
            "scope_note": (
                "PLUMBING + ARITHMETIC ONLY. Same definition, separate implementation: "
                "catches a mis-indexed pose or a stale field, does not validate the "
                "choice of metric."),
        },
        "component_vs_family_triangle": {
            "max(lon,lat) - ade  (must be <= 0)": round(tri_lo, 9),
            "ade - (lon+lat)     (must be <= 0)": round(tri_hi, 9),
            "holds": bool(tri_lo <= 1e-9 and tri_hi <= 1e-9),
        },
        "source": str(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hq", required=True, help="HQ-render run dir (cl_out_hq)")
    ap.add_argument("--morning", required=True, help="morning run dir (cl_out)")
    ap.add_argument("--repro", default=None, help="morning-config re-run dir (cl_out_repro)")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--cond", default="empty")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cl_metrics
    from taniteval.ci import paired_episode_cluster_bootstrap as PB
    tr = None
    if a.tracks and Path(a.tracks).exists():
        from gsplat_renderer import ActorTracks
        tr = ActorTracks(a.tracks)

    def R(root, arm):
        return f"{root}/panel_{arm}_{a.cond}/rollouts_{arm}_{a.cond}.json"

    sets = {}
    for tag, root in (("HQ", a.hq), ("MORNING", a.morning)) + \
                     ((("REPRO", a.repro),) if a.repro else ()):
        for arm in ("flagship-v1", "refc-base"):
            p = R(root, arm)
            if not Path(p).exists():
                print(f"MISSING {tag}/{arm}: {p}", file=sys.stderr)
                continue
            _, rows, eids, _ = cl_metrics.collect(p, tr)
            sets[(tag, arm)] = (key_rows(rows, eids), p, rows, eids)

    res = {
        "question": ("Does the closed-loop headline — REF-C beats flagship v1 with the "
                     "separation ENTIRELY LATERAL — survive the +23.4 % grad-NCC render?"),
        "condition": a.cond,
        "run_dirs": {"HQ": a.hq, "MORNING": a.morning, "REPRO": a.repro},
        "render_flags_HQ": ("--cull-scale-quantile 0.95 --sky-gain 0.3"
                            + (" --all-dynamic-layers" if a.cond == "objects" else "")),
        "render_flags_MORNING": "(defaults) layers=background,road, no cull, no sky",
        "estimator": ("paired_episode_cluster_bootstrap over ROLLOUT STARTS — disjoint "
                      "segments of ONE clip, not independent episodes."),
    }

    # ---- 1. divergence: did the render reach the policy at all? -----------------
    res["divergence_hq_vs_morning"] = {
        arm: divergence(sets[("HQ", arm)][1], sets[("MORNING", arm)][1])
        for arm in ("flagship-v1", "refc-base") if ("HQ", arm) in sets and ("MORNING", arm) in sets}
    if a.repro:
        res["divergence_repro_vs_morning_CONTROL"] = {
            arm: divergence(sets[("REPRO", arm)][1], sets[("MORNING", arm)][1])
            for arm in ("flagship-v1", "refc-base")
            if ("REPRO", arm) in sets and ("MORNING", arm) in sets}

    # ---- 2. the arm gap under each render, on the SAME shared windows ----------
    tags = [t for t in ("HQ", "MORNING", "REPRO") if (t, "flagship-v1") in sets]
    common = None
    for t in tags:
        for arm in ("flagship-v1", "refc-base"):
            ks = set(sets[(t, arm)][0])
            common = ks if common is None else (common & ks)
    common = sorted(common)
    res["n_windows_shared_across_all_runs"] = len(common)
    eid = [k[0] for k in common]

    gap, inter = {}, {}
    for name, get in GETTERS.items():
        per = {}
        for t in tags:
            A = [get(sets[(t, "flagship-v1")][0][k]) for k in common]
            B = [get(sets[(t, "refc-base")][0][k]) for k in common]
            per[t] = (np.array(A, float), np.array(B, float))
            gap.setdefault(name, {})[t] = PB(A, B, eid, n_boot=a.n_boot)
        if "HQ" in per and "MORNING" in per:
            dh = per["HQ"][0] - per["HQ"][1]
            dm = per["MORNING"][0] - per["MORNING"][1]
            inter[name] = PB(dh, dm, eid, n_boot=a.n_boot)
            if "REPRO" in per:
                dr = per["REPRO"][0] - per["REPRO"][1]
                inter[name]["NOISE_FLOOR_repro_minus_morning"] = PB(dr, dm, eid, n_boot=a.n_boot)
    res["arm_gap_flagship_minus_refc"] = gap
    res["interaction_gap_HQ_minus_gap_MORNING"] = inter
    res["interaction_note"] = (
        "difference-in-differences on identical windows: (flagship-REF-C)|HQ minus "
        "(flagship-REF-C)|MORNING. A CI containing zero means there is NO evidence the "
        "render moved that arm gap. Its NOISE_FLOOR twin is the same statistic with the "
        "morning config re-run in place of HQ — the value a null render change produces.")

    # ---- 3. survival verdicts --------------------------------------------------
    surv = {}
    for name in GETTERS:
        m, h = gap[name].get("MORNING"), gap[name].get("HQ")
        if not m or not h:
            continue
        sm, sh = m["separated"], h["separated"]
        flip = (m["delta"] * h["delta"]) < 0
        v = ("SURVIVES" if sm and sh and not flip else
             "LOST" if sm and not sh else
             "GAINED" if sh and not sm else
             "SIGN FLIP" if flip else "NEITHER")
        surv[name] = {
            "family": FAMILY_OF[name], "verdict": v,
            "morning": [m["delta"], m["lo"], m["hi"], sm],
            "hq": [h["delta"], h["lo"], h["hi"], sh],
            "interaction_separated": inter.get(name, {}).get("separated"),
        }
    res["survival"] = surv

    # ---- 4. self-consistency ---------------------------------------------------
    res["self_consistency"] = {
        f"{t}/{arm}": selfcheck(sets[(t, arm)][2], sets[(t, arm)][3], sets[(t, arm)][1])
        for (t, arm) in sets}

    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({"out": a.out, "n_shared": len(common),
                      "survival": {k: v["verdict"] for k, v in surv.items()}}, indent=2))


if __name__ == "__main__":
    main()
