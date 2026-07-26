#!/usr/bin/env python3
"""
S1 -- BRANCH SELECTION AT A MULTI-OPTION JUNCTION: the end-to-end slice.

  decision-point miner   -> gate1_connectivity_probe.py (runs on the eval pod, needs the scenes)
  option-set constructor -> succ(lane) from trajdata.VectorMap, variable arity
  non-circular target    -> map INTERSECT realised path (this file consumes it)
  metric                 -> branch_accuracy (open-loop-choice surface)
  estimator              -> paired episode-cluster bootstrap, unit = SCENE (taniteval/ci.py)
  firewall               -> blind_conditioning_baseline on BOTH goal variants

Variants (spec sec 2.1):
  E  goal = route polyline in the ego frame, truncated at 30 m, 6 points
  H  goal = a single goal point at 150-200 m, which names no branch

Usage:
  python s1_slice.py --dp s1_decision_points.json --out S1_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blind_conditioning_baseline import run_firewall  # noqa: E402

# taniteval is the canonical estimator source; vendored fallback only if absent.
_CI = None
for _p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "taniteval"),
           "/root/taniteval", "/workspace/taniteval"):
    _c = os.path.abspath(_p)
    if os.path.isdir(_c):
        sys.path.insert(0, _c)
        break
try:
    from taniteval.ci import paired_episode_cluster_bootstrap
    _CI = "taniteval.ci"
except Exception:
    _CI = "VENDORED-FALLBACK"

    def paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0, alpha=0.05, reduce="mean"):
        a = np.asarray(a, float); b = np.asarray(b, float)
        eid = np.asarray(eid)
        uniq = np.unique(eid)
        idx = {u: np.where(eid == u)[0] for u in uniq}
        rng = np.random.default_rng(seed)
        point = float(a.mean() - b.mean())
        d = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([idx[u] for u in pick])
            d.append(float(a[sel].mean() - b[sel].mean()))
        d = np.array(d)
        lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"delta": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                "ci95": round(float((hi - lo) / 2), 4), "p_delta_gt0": round(float((d > 0).mean()), 4),
                "separated": bool(lo > 0 or hi < 0), "n_windows": int(a.size),
                "n_episodes": int(len(uniq)), "n_boot": n_boot,
                "estimator": "paired_episode_cluster_bootstrap(VENDORED)"}

WRAP = lambda x: math.atan2(math.sin(x), math.cos(x))          # noqa: E731


def route_polyline(dp, trunc_m):
    """Ego-frame realised path truncated at `trunc_m`. This IS how a route label is
    minted off a recorded clip, which is exactly why it must face the firewall."""
    pts = np.asarray(dp["future_xy_egoframe"], dtype=float)
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < trunc_m:
        return None
    k = int(np.searchsorted(cum, trunc_m))
    return pts[:k + 1]


def goal_point(dp, lo_m, hi_m):
    """A single goal point at [lo,hi] m along the realised path. Names no branch."""
    pts = np.asarray(dp["future_xy_egoframe"], dtype=float)
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < lo_m:
        return None
    j = int(np.argmin(np.abs(cum - min(hi_m, cum[-1]))))
    return pts[j]


def _pt_to_polyline_dist(pts, poly):
    """min distance from each point in `pts` [N,2] to polyline `poly` [M,2]."""
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or len(poly) < 2:
        return np.full(len(pts), np.inf)
    p0, p1 = poly[:-1], poly[1:]
    seg = p1 - p0
    L2 = np.maximum((seg ** 2).sum(-1), 1e-12)
    d = pts[:, None, :] - p0[None]
    t = np.clip((d * seg[None]).sum(-1) / L2[None], 0.0, 1.0)
    proj = p0[None] + t[..., None] * seg[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=-1).min(1)


def build_features(dp, variant):
    """X_cond = EXACTLY what a blind predictor would receive at inference:
    ego state + goal + the map-derived option-set geometry. No pixels, no history.

    The decisive blind feature is the GEOMETRIC COMPATIBILITY of the goal with each
    option's CENTRELINE. A bearing-only summary is a weak blind test, and a weak blind
    test fails UNSAFE -- it under-states acc_blind and admits a circular label.
    """
    opts = dp["option_geom_egoframe"]
    cls = dp.get("option_centerlines_egoframe") or [[] for _ in opts]
    K = len(opts)
    if K < 2:
        return None
    if variant == "E":
        rp = route_polyline(dp, 30.0)
        if rp is None:
            return None
        gvec = rp[-1]
        gh = math.atan2(*(rp[-1] - rp[max(0, len(rp) - 3)])[::-1])
        gpoly = rp
    elif variant == "H":
        gp = goal_point(dp, 150.0, 200.0)
        if gp is None:
            return None
        gvec = gp
        gh = math.atan2(gp[1], gp[0])
        gpoly = np.asarray([gp])
    elif variant == "NOGOAL":
        gvec = gh = gpoly = None
    else:
        raise ValueError(variant)

    feats = []
    for o, cl in zip(opts, cls):
        b, dist, hd = o["bearing_rad"], o["dist_m"], o["heading_delta_rad"]
        row = [math.sin(b), math.cos(b), dist / 50.0, math.sin(hd), math.cos(hd),
               dp["v0_mps"] / 20.0, K / 4.0]
        if gvec is None:
            row += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            gb = math.atan2(gvec[1], gvec[0])
            ox, oy = dist * math.cos(b), dist * math.sin(b)
            # --- centreline-based compatibility: how close does the goal/route
            #     actually run to THIS branch's centreline?
            if len(cl) >= 2:
                dd = _pt_to_polyline_dist(np.asarray(gpoly, dtype=float), cl)
                d_mean = float(np.mean(dd)); d_min = float(np.min(dd))
            else:
                d_mean = d_min = 50.0
            row += [math.cos(WRAP(b - gb)),
                    math.cos(WRAP(hd - gh)),
                    float(np.hypot(ox - gvec[0], oy - gvec[1])) / 50.0,
                    abs(WRAP(b - gb)),
                    min(d_mean, 50.0) / 50.0,          # <- the decisive blind feature
                    min(d_min, 50.0) / 50.0]
        feats.append(row)
    return np.asarray(feats, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="s1_decision_points.json")
    ap.add_argument("--out", default="S1_RESULTS.json")
    ap.add_argument("--min-heading-sep-deg", type=float, default=0.0,
                    help="keep only option sets whose branch headings diverge by >= this")
    a = ap.parse_args()

    dps = json.load(open(a.dp))
    res = [d for d in dps if d.get("target_branch") is not None]

    def hsep(d):
        h = [g["heading_delta_rad"] for g in d["option_geom_egoframe"]]
        return math.degrees(max(h) - min(h))

    if a.min_heading_sep_deg > 0:
        res = [d for d in res if hsep(d) >= a.min_heading_sep_deg]

    out = {
        "problem": "S1 branch selection at a multi-option junction",
        "surface": "open-loop-choice",
        "target": "map INTERSECT realised path -- the successor lane the ego's realised "
                  "future path first enters (a map fact intersected with a trajectory; "
                  "touches no model input)",
        "estimator_module": _CI,
        "resampling_unit": "AlpaSim scene",
        "n_decision_points_total": len(dps),
        "n_with_resolved_target": len(res),
        "min_heading_sep_deg": a.min_heading_sep_deg,
        "coverage": {},
        "class_balance": {},
        "firewall": {},
        "power": {},
    }

    # ---------------- coverage + class balance
    scenes = sorted({d["scene_id"] for d in res})
    arity = {}
    for d in res:
        arity[d["n_options"]] = arity.get(d["n_options"], 0) + 1
    cls = {}
    for d in res:
        cls[d["target_branch"]] = cls.get(d["target_branch"], 0) + 1
    maj = max(cls.values()) / max(len(res), 1) if res else 0.0
    out["coverage"] = {
        "n_scenes_with_dp": len(scenes),
        "n_decision_points": len(res),
        "dp_per_scene": round(len(res) / max(len(scenes), 1), 2),
        "arity_hist": dict(sorted(arity.items())),
        "median_heading_sep_deg": round(float(np.median([hsep(d) for d in res])), 1) if res else None,
        "median_horizon_to_branch_s": round(float(np.median([d["horizon_s"] for d in res])), 2) if res else None,
        "median_future_path_len_m": round(float(np.median([d["future_path_len_m"] for d in res])), 1) if res else None,
    }
    out["class_balance"] = {
        "target_index_counts": dict(sorted(cls.items())),
        "majority_class_rate": round(maj, 4),
        "mean_chance_rate": round(float(np.mean([1.0 / d["n_options"] for d in res])), 4) if res else None,
    }

    # ---------------- THE FIREWALL, on every goal variant
    for variant, label in (("E", "EASY  goal = route polyline truncated at 30 m"),
                           ("H", "HARD  goal = single point at 150-200 m"),
                           ("NOGOAL", "CONTROL  no goal at all (option geometry + ego state only)")):
        groups, clusters, kept = [], [], []
        for d in res:
            f = build_features(d, variant)
            if f is None:
                continue
            groups.append((f, int(d["target_branch"])))
            clusters.append(d["scene_id"])
            kept.append(d)
        r = run_firewall(groups, clusters, label)
        r["variant"] = variant
        r["goal_availability"] = "%d/%d decision points" % (len(groups), len(res))

        # --- a SECOND, zero-training deterministic blind attack.
        # The reported acc_blind must be the STRONGEST attack found, never the
        # first one tried: a weak blind test fails unsafe.
        det_ok, det_n = 0, 0
        if variant in ("E", "H"):
            for d in kept:
                cls_ = d.get("option_centerlines_egoframe") or []
                if len(cls_) != len(d["options"]):
                    continue
                if variant == "E":
                    g = route_polyline(d, 30.0)
                else:
                    gp = goal_point(d, 150.0, 200.0)
                    g = None if gp is None else np.asarray([gp])
                if g is None:
                    continue
                dd = []
                for cl in cls_:
                    dd.append(1e9 if len(cl) < 2
                              else float(np.mean(_pt_to_polyline_dist(np.asarray(g, float), cl))))
                det_n += 1
                det_ok += int(int(np.argmin(dd)) == int(d["target_branch"]))
        if det_n:
            det_acc = det_ok / det_n
            r["acc_blind_deterministic_argmin_dist"] = round(det_acc, 4)
            r["acc_blind_learned_mlp"] = r["acc_blind"]
            if det_acc > r["acc_blind"]:
                r["acc_blind"] = round(det_acc, 4)
                r["acc_blind_source"] = "deterministic argmin distance(goal, option centreline)"
            else:
                r["acc_blind_source"] = "learned option scorer (leave-one-scene-out)"
            r["refused"] = bool(r["acc_blind"] >= 0.98 * r["acc_ceiling"])
            r["verdict"] = ("REFUSED (target is recoverable from conditioning alone)"
                            if r["refused"] else "ADMITTED")
        if r.get("n", 0) > 0:
            pb = paired_episode_cluster_bootstrap(
                r.pop("correct_blind"), r.pop("correct_major"), clusters, n_boot=2000, seed=0)
            r["blind_vs_majority_paired"] = pb
        else:
            r.pop("correct_blind", None); r.pop("correct_major", None)
        out["firewall"][variant] = r

    # ---------------- power statement
    ncl = len(scenes)
    out["power"] = {
        "resampling_unit": "AlpaSim scene (episode-cluster)",
        "n_clusters_available": ncl,
        "bar_single_arm": 40,
        "bar_two_arm": 200,
        "meets_single_arm_bar": bool(ncl >= 40),
        "meets_two_arm_bar": bool(ncl >= 200),
        "shortfall_single_arm_x": round(40 / max(ncl, 1), 2),
        "shortfall_two_arm_x": round(200 / max(ncl, 1), 2),
    }
    json.dump(out, open(a.out, "w"), indent=1)

    # ---------------- report
    print("=" * 78)
    print("S1 SLICE  --  branch selection")
    print("=" * 78)
    print("decision points: %d total, %d with a resolved target, over %d scenes"
          % (len(dps), len(res), len(scenes)))
    print("coverage:", json.dumps(out["coverage"]))
    print("class balance:", json.dumps(out["class_balance"]))
    print()
    print("FIREWALL (blind_conditioning_baseline, leave-one-scene-out):")
    for v, r in out["firewall"].items():
        if r.get("n", 0) == 0:
            print("  %-7s n=0  %s" % (v, r.get("goal_availability", "")))
            continue
        pb = r.get("blind_vs_majority_paired", {})
        print("  %-7s n=%-3d clusters=%-3d acc_blind=%.4f  acc_major=%.4f  chance=%.4f  -> %s"
              % (v, r["n"], r["n_clusters"], r["acc_blind"], r["acc_major"], r["acc_chance"], r["verdict"]))
        print("          blind-minus-majority paired delta %.4f [%.4f, %.4f] separated=%s (%s)"
              % (pb.get("delta", 0), pb.get("lo", 0), pb.get("hi", 0),
                 pb.get("separated"), pb.get("estimator", "")))
        print("          goal availability: %s" % r["goal_availability"])
    print()
    print("POWER:", json.dumps(out["power"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
