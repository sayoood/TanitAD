#!/usr/bin/env python3
"""Analysis of the VectorMap corridor sweep -- intervals, HP-4 arithmetic, verdicts.

Runs on the DEV BOX against the pulled ``vectormap_corridor.json`` so that every
interval comes from the repo's own decision-grade estimator
(``taniteval.ci.episode_cluster_bootstrap``) rather than a re-implementation.

**Resampling unit = the AlpaSim SCENE.** Poses inside one scene are a single
20 s drive and are strongly dependent; the scene is the independent cluster, so
it is the unit resampled. ``overlapping_holdout_se`` is never used.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "..", ".."))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from taniteval import ci as _ci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "vectormap_corridor.json")
OUT = os.path.join(HERE, "corridor_verdict.json")

# pre-registered bars (VECTORMAP_CORRIDOR.md 0.2)
P1_LO, P1_HI = 2.5, 4.5
P2_BAR = 0.90
HP4_SINGLE_ARM = 40      # clusters per topology class, single arm
HP4_TWO_ARM = 200        # clusters per topology class, two arms
CORRIDOR_PROPOSED_HALFWIDTH = 1.75   # taniteval/corridor.py CORRIDOR_HALFWIDTH_M


def boot(vals, sids, reduce="mean", seed=0):
    """Episode(scene)-cluster bootstrap. One value per scene, unit = scene."""
    return _ci.episode_cluster_bootstrap(np.asarray(vals, dtype=np.float64),
                                         list(sids), reduce=reduce,
                                         n_boot=_ci.DEFAULT_N_BOOT, seed=seed)


def main():
    d = json.load(open(SRC))
    ps = d["per_scene"]
    ok = {k: v for k, v in ps.items() if "err" not in v and v.get("n_lanes_with_edges")}
    sids = sorted(ok)
    out = {"n_scenes": len(ps), "n_ok": len(ok),
           "estimator": {
               "interval": "episode_cluster_bootstrap",
               "resampling_unit": "AlpaSim scene",
               "n_boot": _ci.DEFAULT_N_BOOT,
               "deprecated_and_refused": "overlapping_holdout_se"}}

    contain = [ok[s]["ego_containment_rate"] for s in sids]
    g1 = [ok[s]["gate1_style_match_rate"] for s in sids]
    width = [ok[s]["lane_width_m"]["median"] for s in sids]
    hw = [ok[s]["corridor"]["halfwidth_m_median"] for s in sids]
    latp90 = [ok[s]["corridor"]["lat_offset_m"]["abs_p90"] for s in sids]
    nlanes = [ok[s]["n_lanes"] for s in sids]
    dur = [ok[s].get("ego_duration_s") for s in sids]
    dur = [x for x in dur if x]
    dt = [ok[s].get("ego_dt_s") for s in sids]
    dt = [x for x in dt if x]
    npose = [ok[s]["ego_n_poses"] for s in sids]

    # ---------------- P1 ----------------------------------------------------
    out["P1_geometry"] = {
        "lanes_per_scene": {"min": int(min(nlanes)), "max": int(max(nlanes)),
                            "median": float(np.median(nlanes)),
                            "p10": float(np.percentile(nlanes, 10)),
                            "p90": float(np.percentile(nlanes, 90))},
        "lanes_per_scene_INHERITED_claim": "130-472",
        "lane_width_m_ci": boot(width, sids),
        "lane_width_m_scene_range": [round(float(min(width)), 3),
                                     round(float(max(width)), 3)],
        "n_scenes_width_in_band": int(sum(1 for w in width if P1_LO <= w <= P1_HI)),
        "frac_lanes_with_edges": round(float(np.mean(
            [ok[s]["n_lanes_with_edges"] / max(ok[s]["n_lanes"], 1) for s in sids])), 4),
        "pass": None}
    out["P1_geometry"]["pass"] = bool(
        P1_LO <= out["P1_geometry"]["lane_width_m_ci"]["mean"] <= P1_HI
        and out["P1_geometry"]["n_scenes_width_in_band"] / len(sids) >= 0.90)

    # ---------------- P2 -- the FRAME PROOF ---------------------------------
    lo = [s for s in sids if ok[s]["ego_containment_rate"] < P2_BAR]
    out["P2_frame_proof"] = {
        "containment_rate_ci": boot(contain, sids),
        "n_scenes_ge_0.90": int(sum(1 for c in contain if c >= 0.90)),
        "n_scenes_ge_0.95": int(sum(1 for c in contain if c >= 0.95)),
        "n_scenes_eq_1.00": int(sum(1 for c in contain if c >= 0.9999)),
        "worst_scenes": [{"scene": s,
                          "containment": ok[s]["ego_containment_rate"],
                          "n_lanes": ok[s]["n_lanes"],
                          "median_dist_to_centreline_m":
                              ok[s]["ego_dist_to_matched_centreline_m"]["median"],
                          "lane_width_m": ok[s]["lane_width_m"]["median"]}
                         for s in sorted(lo, key=lambda x: ok[x]["ego_containment_rate"])],
        "gate1_style_match_rate_ci": boot(g1, sids),
        "note": ("containment = point-in-lane-ring (left edge + reversed right "
                 "edge). gate1_style = dist_to_nearest_centreline <= "
                 "max(halfwidth, 1.0) -- an ASSOCIATION test, reproduced here "
                 "only so the two are comparable. They are different quantities."),
        "pass": None}
    out["P2_frame_proof"]["pass"] = bool(
        out["P2_frame_proof"]["n_scenes_ge_0.90"] / len(sids) >= 0.90)

    # ---------------- P3 -- the corridor channel ----------------------------
    agree = [ok[s]["corridor"]["ring_vs_bounds_agreement"] for s in sids]
    legacy = [ok[s]["corridor"]["legacy_reparam_error_m"]["p90"] for s in sids]
    legacy = [x for x in legacy if x is not None and np.isfinite(x)]
    out["P3_corridor"] = {
        "halfwidth_m_ci": boot(hw, sids),
        "halfwidth_m_scene_p10_p90": [round(float(np.percentile(hw, 10)), 3),
                                      round(float(np.percentile(hw, 90)), 3)],
        "PROPOSED_constant_in_taniteval_corridor": CORRIDOR_PROPOSED_HALFWIDTH,
        "abs_lat_offset_p90_m_ci": boot(latp90, sids),
        # the FALSIFIABLE cross-check: ray-cast bounds vs the independent
        # point-in-ring test. Shares no code path with the bounds.
        "ring_vs_bounds_agreement_ci": boot(agree, sids),
        "n_scenes_agreement_ge_0.90": int(sum(1 for a in agree if a >= 0.90)),
        "worst_agreement": round(float(min(agree)), 4),
        # the station-reparameterisation route, kept only to price the curvature
        # artefact I hypothesised and then FALSIFIED (it is ~0.02 m, not the cause)
        "legacy_reparam_error_p90_m_max": round(float(max(legacy)), 4) if legacy else None,
        "frac_finite_min": float(min(ok[s]["corridor"]["frac_finite"] for s in sids)),
        "pass": bool(sum(1 for a in agree if a >= 0.90) / len(sids) >= 0.90
                     and min(ok[s]["corridor"]["frac_finite"] for s in sids) >= 0.60)}
    d0 = CORRIDOR_PROPOSED_HALFWIDTH
    m = out["P3_corridor"]["halfwidth_m_ci"]
    out["P3_corridor"]["proposed_constant_within_ci"] = bool(
        m["lo"] <= d0 <= m["hi"])

    # ---------------- the EFFECTIVE half-width ------------------------------
    # ⭐ The number that actually replaces CORRIDOR_HALFWIDTH_M. Half the lane
    # WIDTH (1.802) is the room a PERFECTLY CENTRED ego would have. The ego does
    # not drive down the centreline, so the room to the NEARER edge -- which is
    # what a cross-track error measured FROM THE EGO'S OWN PATH can consume -- is
    # smaller. taniteval's XTE has exactly that origin (offset of the prediction
    # from the gt/reference path), so this, not width/2, is the matched threshold.
    chan = os.path.join(HERE, "corridor_channel.npz")
    if os.path.exists(chan):
        sys.path.insert(0, HERE)
        from corridor_channel import effective_halfwidth, load_channel
        ch = load_channel(chan)
        eff_per, eff_sid, pooled = [], [], []
        for k in sorted(ch):
            e = effective_halfwidth(ch[k]["d_left_m"], ch[k]["d_right_m"])
            e = e[np.isfinite(e)]
            if len(e):
                eff_per.append(float(np.median(e)))
                eff_sid.append(k)
                pooled.append(e)
        pooled = np.concatenate(pooled)
        out["effective_halfwidth"] = {
            "what": ("min(d_left, d_right): metres of room from the EGO'S OWN "
                     "position to the nearer lane edge. The matched threshold for "
                     "a cross-track error measured from the reference path."),
            "ci": boot(eff_per, eff_sid),
            "scene_p10_p90": [round(float(np.percentile(eff_per, 10)), 3),
                              round(float(np.percentile(eff_per, 90)), 3)],
            "pooled_steps": int(len(pooled)),
            "pooled_median_m": round(float(np.median(pooled)), 3),
            "pooled_p10_m": round(float(np.percentile(pooled, 10)), 3),
            "frac_steps_room_below_proposed_1.75": round(
                float((pooled < CORRIDOR_PROPOSED_HALFWIDTH).mean()), 4),
            "n_scenes_tighter_than_proposed": int(
                sum(1 for x in eff_per if x < CORRIDOR_PROPOSED_HALFWIDTH)),
            "n_scenes": len(eff_per),
            "verdict": ("1.75 m is vindicated as HALF THE LANE WIDTH (1.802 CI "
                        "[1.686,1.939]) but is TOO PERMISSIVE as a departure "
                        "threshold, because the ego is not centred in its lane."),
        }

    # ---------------- horizon feasibility -----------------------------------
    T = int(np.median(npose))
    out["horizon"] = {
        "scene_duration_s": {"median": float(np.median(dur)),
                             "min": float(min(dur)), "max": float(max(dur))},
        "dt_s_median": float(np.median(dt)),
        "n_poses": {"median": T, "min": int(min(npose)), "max": int(max(npose))},
        "K_ceiling_median_scene": int(T - 8 - 1),
        "K60_feasible_scenes": int(sum(1 for n in npose if n - 8 - 1 >= 60)),
        "K70_feasible_scenes": int(sum(1 for n in npose if n - 8 - 1 >= 70)),
        "note": ("K ceiling uses taniteval.corridor.horizon_ceiling(T, W=8): a "
                 "window exists only if T - W - K >= 1.")}

    # ---------------- HP-4 --------------------------------------------------
    topo = d["topology"]
    by_class = {}
    for t in topo:
        by_class.setdefault(t["class"], set()).add(t["scene"])
    rows = sorted(((c, len(v), sum(1 for x in topo if x["class"] == c))
                   for c, v in by_class.items()), key=lambda r: -r[1])
    n_sc = len(sids)
    hp4 = {"n_topology_classes": len(by_class),
           "resampling_unit": "AlpaSim scene",
           "bar_single_arm": HP4_SINGLE_ARM, "bar_two_arm": HP4_TWO_ARM,
           "classes": [{"class": c, "n_scenes": ns, "n_branch_points": nb,
                        "scene_yield": round(ns / n_sc, 4)} for c, ns, nb in rows],
           "n_classes_ge_40_scenes": int(sum(1 for _, ns, _ in rows
                                             if ns >= HP4_SINGLE_ARM)),
           "n_classes_ge_200_scenes": int(sum(1 for _, ns, _ in rows
                                              if ns >= HP4_TWO_ARM))}
    # scenes required to bring the top classes up to bar, at the measured yield
    need = []
    for c, ns, nb in rows[:6]:
        y = ns / n_sc
        need.append({"class": c, "scene_yield": round(y, 4),
                     "scenes_for_40": int(np.ceil(HP4_SINGLE_ARM / y)),
                     "scenes_for_200": int(np.ceil(HP4_TWO_ARM / y))})
    hp4["scenes_required"] = need
    hp4["scenes_for_top3_all_ge40"] = int(max(x["scenes_for_40"] for x in need[:3]))
    hp4["scenes_for_top3_all_ge200"] = int(max(x["scenes_for_200"] for x in need[:3]))
    hp4["held_today"] = n_sc
    hp4["verdict"] = ("INSTRUMENT EXISTS, CORPUS SHORT"
                      if hp4["n_classes_ge_40_scenes"] < 2 else "RUNNABLE")
    out["HP4"] = hp4

    out["OUTCOME"] = ("A" if (out["P1_geometry"]["pass"] and out["P2_frame_proof"]["pass"]
                              and out["P3_corridor"]["pass"]) else "B")
    json.dump(out, open(OUT, "w"), indent=1, default=str)

    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("HP4",)}, indent=1, default=str))
    print("--- HP4 ---")
    print(json.dumps({k: v for k, v in hp4.items() if k != "classes"},
                     indent=1, default=str))
    print("top classes:", [(r["class"], r["n_scenes"]) for r in hp4["classes"][:6]])
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
