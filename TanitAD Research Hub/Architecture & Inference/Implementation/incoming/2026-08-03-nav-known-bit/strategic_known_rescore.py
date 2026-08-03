#!/usr/bin/env python3
"""P3 — re-score the junction scene's FOUR FAMILIES with the nav `known` bit visible.

WHAT IS NEW HERE, and it is one thing: the banked rollouts record `nav` but not whether
that `nav` was a JUDGEMENT or a CONFESSION. `nav_command_v21` maps both `ROUTE_STRAIGHT`
and `ROUTE_UNKNOWN` onto `NAV_FOLLOW`, so a `follow` token can mean either. This script
re-derives the companion bit from the banked GT poses with `refb_labels.nav_input_v22`
(a pure function of the poses — no model, no render, no pod) and re-cuts the STRATEGIC
family by it.

⛔ ALL FOUR FAMILIES ARE REPORTED, PER FAMILY, NEVER POOLED. An ADE horizon sweep is one
row of four. Where a family cannot be computed the reason and the n travel with it.

⚠️ FIVE METRICS ARE DEGENERATE BY CONSTRUCTION IN OPEN LOOP and are marked, not quoted:
`cross_track_abs_m`, `cross_track_signed_m`, `dist_to_gt_traj_m`, `executed_speed_err_ms`
and `route_corridor_departure_rate` are ~0 because the ego IS the logged path, and
`manoeuvre_exec_eq_plan` collapses onto `manoeuvre_plan_eq_logged`. That list is
`openloop_drive.py`'s own and is reproduced rather than re-derived.

Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` via `cl_metrics._paired`,
clusters = the run's 9 disjoint contiguous segments of ONE clip. ⛔ Never
`overlapping_holdout_se`.

⭐ The STRATEGIC family's echo verdict is supplied from a REAL MANIPULATION — the nav
sweep in `nav_sweep_junction.py`, pixels held fixed — via
`taniteval.strategic_optionset.conditioning_echo_control`. With no sweep the verdict is
`None` (UNTESTED), never a pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "stack" / "experiments" / "alpasim-gsplat"))
sys.path.insert(0, str(ROOT / "stack" / "scripts"))
sys.path.insert(0, str(ROOT / "stack"))
sys.path.insert(0, str(ROOT / "taniteval"))

DEGENERATE_IN_OPEN_LOOP = ("cross_track_abs_m", "cross_track_signed_m",
                           "dist_to_gt_traj_m", "executed_speed_err_ms",
                           "route_corridor_departure_rate", "manoeuvre_exec_eq_plan")


def nav_known_by_tick(gt) -> dict[int, float]:
    """k -> nav_known in {0.0, 1.0}, re-derived from the BANKED poses.

    Uses `closedloop_drive.gt_poses_xyv` so the pose format is the labeller's own, and
    `refb_labels.nav_input_v22` so the bit is the one a trainer would be fed."""
    import torch
    from closedloop_drive import gt_poses_xyv          # noqa: E402
    from refb_labels import nav_input_v22             # noqa: E402
    T = np.stack([np.eye(4) for _ in gt])
    for i, g in enumerate(gt):
        c, s = np.cos(g["yaw"]), np.sin(g["yaw"])
        T[i, 0, 0], T[i, 0, 1], T[i, 1, 0], T[i, 1, 1] = c, -s, s, c
        T[i, 0, 3], T[i, 1, 3], T[i, 2, 3] = g["x"], g["y"], g["z"]
    p = torch.from_numpy(gt_poses_xyv(list(T))).float()
    out = {}
    for i in range(len(gt)):
        try:
            _nav, known = nav_input_v22(p, i)
            out[i] = float(known)
        except Exception:                              # noqa: BLE001
            out[i] = float("nan")
    return out


def echo_from_sweep(sweep_path: Path) -> dict:
    """`conditioning_echo_control` fed from the manipulation, not from a contingency
    table. One row per tick: {nav_value: predicted route class}."""
    from taniteval.strategic_optionset import conditioning_echo_control  # noqa: E402
    d = json.loads(Path(sweep_path).read_text())
    rows = []
    for t in d["ticks"]:
        row = {}
        for nv in ("0", "1", "2"):          # ⛔ nav=3 is OUT OF VOCABULARY, excluded
            rl = t["sweep"][nv].get("route_logits")
            row[int(nv)] = None if rl is None else int(np.argmax(rl))
        rows.append(row)
    out = conditioning_echo_control(rows)
    out["sweep_source"] = str(sweep_path)
    out["nav_values_swept"] = [0, 1, 2]
    out["nav3_excluded_reason"] = (
        "_ROUTE_TO_NAV never emits NAV_STRAIGHT=3, so that embedding row is untrained "
        "and a sweep over it is an out-of-vocabulary probe, not a condition.")
    return out


def strategic_by_known(rows, eids, known) -> dict:
    """`route_head_eq_logged` cut by the companion bit, plus what the cut costs."""
    import cl_metrics as clm                            # noqa: E402
    k = np.array([known.get(int(r["i_gt"]), np.nan) for r in rows], float)
    have = np.array([(r.get("route_head") is not None and r.get("route_valid"))
                     for r in rows])
    acc = np.array([(float(r["route_head"] == r["route_gt"])
                     if (r.get("route_head") is not None and r.get("route_valid"))
                     else np.nan) for r in rows], float)
    out = {"n_windows": len(rows),
           "nav_known_1_count": int(np.nansum(k == 1.0)),
           "nav_known_0_count": int(np.nansum(k == 0.0)),
           "nav_known_0_share_of_follow": None,
           "route_scorable_count": int(have.sum())}
    nav = np.array([int(r["nav"]) for r in rows])
    fol = nav == 0
    if fol.sum():
        out["nav_known_0_share_of_follow"] = round(
            float(np.nansum((k == 0.0) & fol) / fol.sum()), 4)
        out["n_follow_windows"] = int(fol.sum())
    for lab, m in (("known_1", k == 1.0), ("known_0", k == 0.0)):
        n = int((m & have).sum())
        out[lab] = ({"n": 0, "reason": "no scorable window in this stratum"} if n == 0
                    else clm._ci(np.where(m, acc, np.nan), eids))
    return out


def run(a_path: Path, b_path: Path, sweep_a: Path | None, sweep_b: Path | None) -> dict:
    import cl_metrics as clm                            # noqa: E402
    dA, rA, eA, sA = clm.collect(str(a_path))
    dB, rB, eB, sB = clm.collect(str(b_path))
    known = nav_known_by_tick(dA["gt"])

    res = {"scene": dA["scene"], "condition": dA["condition"],
           "arm_A": dA["arm"], "arm_B": dB["arm"],
           "ckpt_A": dA["ckpt"], "ckpt_B": dB["ckpt"],
           "n_windows_A": len(rA), "n_windows_B": len(rB),
           "n_clusters": int(len(set(eA.tolist()))),
           "cluster_note": dA.get("cluster_note"),
           "degenerate_in_open_loop": list(DEGENERATE_IN_OPEN_LOOP),
           "nav_known_summary": {
               "n_ticks": len(known),
               "known_1": int(sum(1 for v in known.values() if v == 1.0)),
               "known_0": int(sum(1 for v in known.values() if v == 0.0)),
               "source": "refb_labels.nav_input_v22 on the BANKED gt poses"},
           "families_A": clm.families(rA, eA, sA),
           "families_B": clm.families(rB, eB, sB),
           "strategic_by_known_A": strategic_by_known(rA, eA, known),
           "strategic_by_known_B": strategic_by_known(rB, eB, known)}

    for lab, sw in (("A", sweep_a), ("B", sweep_b)):
        res[f"conditioning_echo_control_{lab}"] = (
            echo_from_sweep(sw) if sw else
            {"ECHO": None, "reason": "NO SWEEP SUPPLIED — UNTESTED, not a pass."})

    key = lambda rows, eids: {(int(e), int(r["k"])): r for r, e in zip(rows, eids)}
    KA, KB = key(rA, eA), key(rB, eB)
    common = sorted(set(KA) & set(KB))
    pe = [c[0] for c in common]
    pair = {}
    for lab, fn in (
            ("ADE:ade_0_2s", lambda r: r["ade"]),
            ("LONGITUDINAL:abs_target_speed_err_ms", lambda r: abs(r["speed_err"])),
            ("LONGITUDINAL:along_track_ade_m", lambda r: r["lon_ade"]),
            ("LONGITUDINAL:real_lead_headway_m",
             lambda r: (r["headway"] if r["headway"] is not None else np.nan)),
            ("LONGITUDINAL:real_lead_time_gap_s",
             lambda r: (r["time_gap"] if r["time_gap"] is not None else np.nan)),
            ("LONGITUDINAL:real_lead_ttc_s_when_closing",
             lambda r: (r["ttc"] if r.get("ttc") is not None else np.nan)),
            ("LATERAL:heading_err_rad", lambda r: abs(r["heading_err"])),
            ("LATERAL:curvature_err_1pm", lambda r: r["curv_err"]),
            ("LATERAL:yawrate_err_rads", lambda r: r["yawrate_err"]),
            ("LATERAL:lateral_ade_m", lambda r: r["lat_ade"]),
            ("TACTICAL:manoeuvre_plan_eq_logged",
             lambda r: float(r["man_plan"] == r["man_gt"])),
            ("TACTICAL:manoeuvre_head_eq_logged",
             lambda r: (float(r["man_head"] == r["man_gt"])
                        if r["man_head"] is not None else np.nan)),
            ("STRATEGIC:route_head_eq_logged",
             lambda r: (float(r["route_head"] == r["route_gt"])
                        if (r["route_head"] is not None and r["route_valid"])
                        else np.nan))):
        pair[lab] = clm._paired([fn(KA[c]) for c in common],
                                [fn(KB[c]) for c in common], pe)
    res["paired_A_minus_B"] = pair
    res["paired_n_windows"] = len(common)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--sweep-a", default=None)
    ap.add_argument("--sweep-b", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = run(Path(args.a), Path(args.b),
            Path(args.sweep_a) if args.sweep_a else None,
            Path(args.sweep_b) if args.sweep_b else None)
    Path(args.out).write_text(json.dumps(r, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
