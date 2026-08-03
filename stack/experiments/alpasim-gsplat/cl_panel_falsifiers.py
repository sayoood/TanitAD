#!/usr/bin/env python3
"""ADVERSARIAL falsifiers for a closed-loop panel — the checks the panel's own
self-consistency block does NOT run.

Written 2026-08-03 during the adversarial verification of `results/closedloop-hq-render/`.
Four of the six checks below FIRED on that panel, so none of them is hypothetical.

Each check answers a question of the form "is this separation what the table says it is?"

1. `family_identity` — WHICH REPORTED METRICS ARE THE SAME MEASUREMENT?
   The panel already knows `dist_to_gt_traj_m == abs(cross_track)`. It does NOT know that
   `route_corridor_departure_rate`, filed under STRATEGIC, is `1[|cross_track| > 2 m]` —
   i.e. a THIRD copy of the same lateral number, cited as non-lateral evidence.
   MEASURED on that panel: identity holds on 450/450 and 437/437 windows, max|Δ| = 0.

2. `ade_is_longitudinal` — IS "ADE SEPARATES" AN INDEPENDENT FINDING?
   `max(lon,lat) <= ade <= lon+lat` passes trivially when one component dominates.
   MEASURED: lon/ade = 0.9941 and corr(ade, lon) = 0.99978, so ADE, along-track ADE and
   target-speed error were one longitudinal effect reported three times.

3. `cluster_overlap` — ARE THE BOOTSTRAP CLUSTERS INDEPENDENT?
   Every panel emits "paired_episode_cluster_bootstrap over ROLLOUT STARTS — disjoint
   segments of ONE clip". MEASURED: 50 ticks x 0.1 s = 4.9 s rollouts, starts 17 frames
   = 1.7 s apart -> ~65 % temporal overlap between neighbours, and overlapping `i_gt`
   reference windows. The clusters are NOT disjoint and the intervals are
   anti-conservative by an unquantified amount.

4. `disjoint_triples` — DOES THE SEPARATION SURVIVE ON NON-OVERLAPPING CLUSTERS?
   The falsifier for (3): re-run the paired bootstrap on each set of starts that really
   is disjoint. MEASURED on the morning render: `cross_track_abs_m` (+1.1705 [+0.0296,
   +2.2438], the interval that licensed "all four lateral separations SURVIVE") is NOT
   separated on the middle third, and `yawrate_err_rads` is not separated on the last.

5. `majority_baseline` — IS THE TACTICAL METRIC BEATING A CONSTANT PREDICTOR?
   `per_class_pr` already computes `_majority_class_baseline_acc`; nothing reads it.
   MEASURED: both arms scored BELOW it (0.2400 / 0.2998 vs 0.4578 / 0.4188), so
   "separates under neither render" was true and uninformative.

6. `unrendered_lead` — IS DISTANCE-KEEPING SCORED AGAINST AGENTS THAT WERE DRAWN?
   `cl_metrics --renderable-from` exists precisely to stop this and the scoring script
   omits it. MEASURED on the `empty` panel, where ZERO dynamic layers are rendered:
   lead_present_rate 0.4156, headway 37.57 m, min_headway -4.65 m — distance-keeping to
   agents that appear in no frame.

Usage:
    python cl_panel_falsifiers.py --rollouts A.json B.json ... [--panel HQ_a_vs_b.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

GETTERS = {
    "ade_0_2s": lambda r: r["ade"],
    "cross_track_abs_m": lambda r: abs(r["cross_track"]),
    "heading_err_rad": lambda r: abs(r["heading_err"]),
    "curvature_err_1pm": lambda r: r["curv_err"],
    "yawrate_err_rads": lambda r: r["yawrate_err"],
    "abs_target_speed_err_ms": lambda r: abs(r["speed_err"]),
    "along_track_ade_m": lambda r: r["lon_ade"],
    "route_corridor_departure_rate": lambda r: float(r["corridor_departure"]),
}


def family_identity(rows, corridor_m=2.0):
    ct = np.array([abs(r["cross_track"]) for r in rows])
    out = {}
    dtg = np.array([r["dist_to_gt"] for r in rows])
    out["dist_to_gt_IS_abs_cross_track"] = {
        "max_abs_diff": float(np.max(np.abs(dtg - ct))), "n": len(rows)}
    cd = np.array([r["corridor_departure"] for r in rows], float)
    recon = (ct > corridor_m).astype(float)
    out["corridor_departure_IS_thresholded_cross_track"] = {
        "n_agree": int((recon == cd).sum()), "n": len(rows),
        "max_abs_diff": float(np.max(np.abs(recon - cd))),
        "verdict": ("CONFIRMED IDENTICAL — `route_corridor_departure_rate` is "
                    f"1[|cross_track| > {corridor_m} m]. It is a LATERAL measurement "
                    "wearing a STRATEGIC label; it may not be cited as evidence that a "
                    "separation is not lateral.")
        if np.max(np.abs(recon - cd)) == 0 else "not identical"}
    return out


def ade_is_longitudinal(rows):
    ade = np.array([r["ade"] for r in rows])
    lon = np.array([r["lon_ade"] for r in rows])
    lat = np.array([r["lat_ade"] for r in rows])
    return {"mean_ade": float(ade.mean()), "mean_lon": float(lon.mean()),
            "mean_lat": float(lat.mean()),
            "lon_over_ade": float(lon.mean() / max(ade.mean(), 1e-9)),
            "corr_ade_lon": float(np.corrcoef(ade, lon)[0, 1]),
            "note": ("if lon/ade ~ 1 then 'ADE separates', 'along-track separates' and "
                     "'target-speed error separates' are ONE finding, not three.")}


def cluster_overlap(d, dt=0.1):
    spans = []
    for r in d["rollouts"]:
        s = r["steps"]
        spans.append((int(r["start_frame"]),
                      float(s[0]["t_us"]) / 1e6, float(s[-1]["t_us"]) / 1e6,
                      int(s[0]["i_gt"]), int(s[-1]["i_gt"])))
    ov = []
    for a, b in combinations(spans, 2):
        lo, hi = max(a[1], b[1]), min(a[2], b[2])
        if hi > lo:
            ov.append({"starts": [a[0], b[0]], "overlap_s": round(hi - lo, 3),
                       "frac_of_shorter": round((hi - lo) / min(a[2] - a[1], b[2] - b[1]), 4)})
    return {"n_clusters": len(spans), "n_overlapping_pairs": len(ov),
            "max_overlap_frac": (max(x["frac_of_shorter"] for x in ov) if ov else 0.0),
            "spans_s": [[s[0], round(s[1], 2), round(s[2], 2)] for s in spans],
            "i_gt_windows": [[s[0], s[3], s[4]] for s in spans],
            "verdict": ("⛔ CLUSTERS OVERLAP — the episode-cluster bootstrap assumes "
                        "independent clusters. Any interval from these is "
                        "ANTI-CONSERVATIVE and the label 'disjoint segments of ONE clip' "
                        "is wrong.") if ov else "clusters are disjoint in clip time"}


def disjoint_triples(spans):
    """Greedy partition of the rollout starts into sets with no pairwise time overlap."""
    order = sorted(spans, key=lambda s: s[1])
    groups: list[list] = []
    for s in order:
        for g in groups:
            if all(min(s[2], t[2]) <= max(s[1], t[1]) for t in g):
                g.append(s)
                break
        else:
            groups.append([s])
    return [[int(x[0]) for x in g] for g in groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--pair", nargs=2, default=None,
                    help="two rollout jsons (arm A, arm B) for the disjoint-cluster "
                         "falsifier on the paired delta")
    ap.add_argument("--panel", default=None, help="a scored panel json, for check 5/6")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import cl_metrics
    res = {}
    for p in a.rollouts:
        d, rows, eids, _ = cl_metrics.collect(p)
        res[Path(p).name] = {
            "family_identity": family_identity(rows, cl_metrics.CORRIDOR_M),
            "ade_is_longitudinal": ade_is_longitudinal(rows),
            "cluster_overlap": cluster_overlap(d)}

    if a.pair:
        from taniteval.ci import paired_episode_cluster_bootstrap as PB
        dA, rA, eA, _ = cl_metrics.collect(a.pair[0])
        dB, rB, eB, _ = cl_metrics.collect(a.pair[1])
        K = lambda rows, e: {(int(x), int(r["k"])): r for r, x in zip(rows, e)}
        KA, KB = K(rA, eA), K(rB, eB)
        common = sorted(set(KA) & set(KB))
        eid = np.array([c[0] for c in common])
        spans = [(int(r["start_frame"]), float(r["steps"][0]["t_us"]) / 1e6,
                  float(r["steps"][-1]["t_us"]) / 1e6, 0, 0) for r in dA["rollouts"]]
        groups = disjoint_triples(spans)
        out = {"disjoint_groups": groups, "metrics": {}}
        for name, g in GETTERS.items():
            A = np.array([g(KA[c]) for c in common])
            B = np.array([g(KB[c]) for c in common])
            full = PB(A, B, list(eid), n_boot=a.n_boot)
            per = {}
            for gi, grp in enumerate(groups):
                m = np.isin(eid, grp)
                if m.sum() < 2:
                    continue
                per[str(grp)] = PB(A[m], B[m], list(eid[m]), n_boot=a.n_boot)
            fragile = full["separated"] and any(not v["separated"] for v in per.values())
            out["metrics"][name] = {
                "all_clusters": full, "disjoint_groups": per,
                "FRAGILE": fragile,
                "note": ("separated on all clusters but NOT on at least one disjoint "
                         "subset — the separation depends on the overlap"
                         if fragile else "")}
        res["disjoint_cluster_falsifier"] = out

    if a.panel:
        j = json.loads(Path(a.panel).read_text())
        pan = {"renderable_restricted": j.get("renderable_restricted"), "arms": {}}
        for side in ("arm_A", "arm_B"):
            if side not in j:
                continue
            f = j[side]["families"]
            t = f.get("TACTICAL", {})
            L = f.get("LONGITUDINAL", {})
            pr = t.get("plan_vs_logged_per_class_PR", {})
            pan["arms"][j[side]["name"]] = {
                "condition": j[side].get("condition"),
                "majority_baseline": {
                    "plan_acc": pr.get("_accuracy"),
                    "majority_class_baseline_acc": pr.get("_majority_class_baseline_acc"),
                    "BELOW_BASELINE": (pr.get("_accuracy") is not None
                                       and pr.get("_majority_class_baseline_acc") is not None
                                       and pr["_accuracy"] < pr["_majority_class_baseline_acc"])},
                "unrendered_lead": {
                    "lead_present_rate": L.get("lead_present_rate"),
                    "headway_mean": (L.get("headway_m") or {}).get("mean"),
                    "min_headway": (L.get("min_headway_m") or {}).get("mean"),
                    "WARNING": ("distance-keeping was scored WITHOUT --renderable-from; "
                                "on an `empty` condition no agent is drawn, so any "
                                "non-zero lead_present_rate is a measurement of the "
                                "ANNOTATION, not of the policy")
                    if (j.get("renderable_restricted") is None
                        and L.get("lead_present_rate")) else None}}
        res["panel_checks"] = pan

    txt = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
