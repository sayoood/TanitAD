#!/usr/bin/env python3
"""The CUT-IN subset: score only the windows that actually contain a renderable cut-in.

WHY A SUBSET, AND WHY IT IS REPORTED SEPARATELY
The whole-clip panel answers "does the arm behave differently when real traffic is
drawn". It does NOT answer the question the closed-loop run actually named, which is
about a specific manoeuvre — a vehicle entering the ego lane at close range. Those ticks
are a MINORITY of the clip, so a whole-clip contrast dilutes them by construction.

⛔ STATE THE DENOMINATOR. This is the rule that killed the "brake_stop free win": a gain
that lives only on windows the label cannot represent is not a gain. So this script
reports, for every contrast, BOTH n_cutin_windows and n_all_windows, and the cut-in
windows are defined from the SCENE's own annotation (scene_geometry.py's cut-in events,
restricted to RENDERABLE tracks) — never from the rollouts themselves, which would be
circular.

A window (start, k) is a CUT-IN window when its absolute ego tick `i_gt` lies in
[k_start, k_end + POST_TICKS] of a renderable cut-in event: the entry itself plus the
reaction window in which a policy could still respond to it.

The contrast is the same paired episode-cluster bootstrap used everywhere else, on
identical (start, k) windows, with the empty-road control as the matched counterfactual.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POST_TICKS = 20          # 2.0 s of reaction window after the entry completes


def cutin_tick_set(geom_json, renderable_only=True):
    """Absolute 10 Hz ego ticks covered by a cut-in event, from the scene annotation."""
    g = json.loads(Path(geom_json).read_text())
    s = g["summary"] if "summary" in g else g
    ticks, ev = set(), []
    for c in s.get("cutins_ALL", []):
        if renderable_only and not c.get("renderable"):
            continue
        ev.append(c)
        for k in range(int(c["k_start"]), int(c["k_end"]) + POST_TICKS + 1):
            ticks.add(k)
    return ticks, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm/condition A rollouts json")
    ap.add_argument("--b", required=True, help="arm/condition B rollouts json (paired)")
    ap.add_argument("--geometry", required=True, help="scene_geometry json WITH --actor-map")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--renderable-from", default=None)
    ap.add_argument("--label", default="A_minus_B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cl_metrics as M
    from gsplat_renderer import ActorTracks

    tr = ActorTracks(a.tracks) if a.tracks and Path(a.tracks).exists() else None
    keep = None
    if a.renderable_from and Path(a.renderable_from).exists():
        am = json.loads(Path(a.renderable_from).read_text())
        keep = {int(x["best_track"]) for x in am["per_track"] if x["accepted"]}

    ticks, events = cutin_tick_set(a.geometry)
    dA, rA, eA, _ = M.collect(a.a, tr, lead_ref=None, keep_tracks=keep)
    dB, rB, eB, _ = M.collect(a.b, tr, lead_ref=None, keep_tracks=keep)

    key = lambda rows, eids: {(int(e), int(r["k"])): r for r, e in zip(rows, eids)}
    KA, KB = key(rA, eA), key(rB, eB)
    common = sorted(set(KA) & set(KB))
    incut = [c for c in common if int(KA[c]["i_gt"]) in ticks]

    METRICS = (
        ("ADE", "ade_0_2s", lambda r: r["ade"]),
        ("ADE", "dist_to_gt_traj_m", lambda r: r["dist_to_gt"]),
        ("LONGITUDINAL", "abs_target_speed_err_ms", lambda r: abs(r["speed_err"])),
        ("LONGITUDINAL", "along_track_ade_m", lambda r: r["lon_ade"]),
        ("LONGITUDINAL", "real_lead_headway_m",
         lambda r: (r["headway"] if r["headway"] is not None else np.nan)),
        ("LONGITUDINAL", "real_lead_time_gap_s",
         lambda r: (r["time_gap"] if r["time_gap"] is not None else np.nan)),
        ("LONGITUDINAL", "real_lead_frac_tg_below_1s",
         lambda r: (float(r["time_gap"] < 1.0) if r["time_gap"] is not None else np.nan)),
        ("LONGITUDINAL", "real_lead_ttc_s_when_closing",
         lambda r: (r["ttc"] if r.get("ttc") is not None else np.nan)),
        ("LATERAL", "heading_err_rad", lambda r: abs(r["heading_err"])),
        ("LATERAL", "curvature_err_1pm", lambda r: r["curv_err"]),
        ("LATERAL", "yawrate_err_rads", lambda r: r["yawrate_err"]),
        ("LATERAL", "cross_track_abs_m", lambda r: abs(r["cross_track"])),
        ("LATERAL", "lateral_ade_m", lambda r: r["lat_ade"]),
        ("TACTICAL", "manoeuvre_plan_eq_logged",
         lambda r: float(r["man_plan"] == r["man_gt"])),
        ("TACTICAL", "manoeuvre_head_eq_logged",
         lambda r: (float(r["man_head"] == r["man_gt"])
                    if r["man_head"] is not None else np.nan)),
        ("STRATEGIC", "route_corridor_departure_rate",
         lambda r: float(r["corridor_departure"])),
        ("STRATEGIC", "route_head_eq_logged",
         lambda r: (float(r["route_head"] == r["route_gt"])
                    if (r["route_head"] is not None and r["route_valid"]) else np.nan)),
    )

    def contrast(keys):
        pe = [c[0] for c in keys]
        out = {}
        for fam, lab, fn in METRICS:
            va = [fn(KA[c]) for c in keys]
            vb = [fn(KB[c]) for c in keys]
            va = [np.nan if v is None else v for v in va]
            vb = [np.nan if v is None else v for v in vb]
            r = M._paired(va, vb, pe)
            r["family"] = fam
            out[lab] = r
        return out

    res = {
        "what": f"CUT-IN WINDOW SUBSET — {a.label}",
        "A": {"file": a.a, "arm": dA["arm"], "condition": dA["condition"]},
        "B": {"file": a.b, "arm": dB["arm"], "condition": dB["condition"]},
        "denominator": {
            "n_all_paired_windows": len(common),
            "n_cutin_windows": len(incut),
            "cutin_share": round(len(incut) / max(len(common), 1), 4),
            "n_renderable_cutin_events": len(events),
            "n_cutin_ticks": len(ticks),
            "post_ticks_reaction_window": POST_TICKS,
            "n_clusters_all": len({c[0] for c in common}),
            "n_clusters_cutin": len({c[0] for c in incut}),
            "note": "cut-in ticks come from the SCENE annotation (scene_geometry.py, "
                    "renderable tracks only), never from the rollouts.",
        },
        "estimator": "paired episode-cluster bootstrap (taniteval.ci) over rollout "
                     "starts — disjoint segments of ONE clip, not independent episodes.",
        "paired_A_minus_B_CUTIN_WINDOWS": contrast(incut) if incut else
            {"n": 0, "reason": "no paired window overlaps a renderable cut-in event"},
        "paired_A_minus_B_ALL_WINDOWS": contrast(common),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["denominator"], indent=1))
    for lab, v in res["paired_A_minus_B_CUTIN_WINDOWS"].items():
        if isinstance(v, dict) and "delta" in v:
            sep = "*SEP*" if (v["lo"] > 0 or v["hi"] < 0) else ""
            print("  %-14s %-30s d=%+.4f [%+.4f,%+.4f] n=%-4s %s"
                  % (v["family"], lab, v["delta"], v["lo"], v["hi"],
                     v.get("n_used", "?"), sep))


if __name__ == "__main__":
    main()
