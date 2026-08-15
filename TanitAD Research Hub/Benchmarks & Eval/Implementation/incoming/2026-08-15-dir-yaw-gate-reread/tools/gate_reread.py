"""DIR_YAW_RAD 0.15 -> 0.10 re-read — 0-GPU, over BANKED artifacts only.

⛔ WHAT THIS CAN AND CANNOT DO, stated up front because the limit IS the finding.

`taniteval.hierarchy` banked only the THRESHOLDED direction classes (`traj_dir` /
`gt_dir`) until 2026-08-06; the continuous net yaw was discarded at write time
(`hierarchy.py:610-620`, the fix). ⇒ **no coherence kappa in a pre-2026-08-06 panel can
be re-read at another gate** — that is a re-run, not a recompute, and this tool refuses
to fake it.

⭐ BUT ONE HALF OF THE AUDIT *IS* EXACTLY RECOVERABLE, and nobody had noticed:
`results/windows_<arm>.pt` banks `head_deg` per window, written by
`bench.py:399` as `driving_diagnostic.net_heading_change_deg(ep.poses, last)` =
``|wrap(poses[last+K_MAX,2] - poses[last,2])| * 180/pi`` with ``K_MAX = 20`` steps.
`hierarchy.py:557` forms its gate input as
``gt_net = wrap_to_pi(fut[:, GOAL_H-1, 2] - pl[:, 2])`` with ``GOAL_H = K_MAX = 20``.
**Same poses, same 2 s horizon, same wrap — `head_deg * pi/180` IS `|gt_net|`, exactly.**
The sign is gone (the dump takes `.abs()`), so L/S/R classes and therefore kappa cannot be
rebuilt — but every quantity that depends only on the MAGNITUDE can:

  * ``frac_gt_turning`` at any gate  (== 1 - P(straight) of `gt_dir`)
  * ``median_abs_net_yaw_rad`` / ``p90_abs_net_yaw_rad``  (the mis-scaling evidence)

and those are exactly the numbers R-2026-08-06-yawgate used to call the gate mis-scaled —
measured there on 39 OOD clips and 880 OOD-val windows only. This tool re-measures them on
every canonical-val arm that has a window dump, which is where 11 of the 13 unswept panels
live.

VERIFICATION (not an assumption): where a banked hierarchy panel exists for the same arm
and the same window count, `frac_gt_turning(0.15)` computed here is checked against
``1 - consistency.distributions.gt_dir[route_straight]/n`` from that panel. A match is
bit-level proof that the reconstructed quantity is the instrument's own.

Usage:
    python gate_reread.py --repo <repo-root> --out <results.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import lzma
import math
import os

GATES = (0.15, 0.10, 0.06, 0.04, 0.02, 0.01)
PUBLISHED_GATE = 0.15
PROPOSED_GATE = 0.10

#: `four_families.tactical_family` publishes this ladder as a WORD
#: (`maneuver_consistency_verdict`, four_families.py:888-890). ⚠️ It is NOT the
#: threshold `hierarchy._gate_sensitivity` tests in `verdict_stable` (that one uses
#: 0.2). Two different boundaries, one of them published and untested.
VERDICT_LADDER = ((0.1, "DECORATIVE"), (0.4, "WEAK"), (float("inf"), "SUBSTANTIAL"))
GATE_SENS_VERDICT_THRESHOLD = 0.2


def verdict(k):
    if k is None:
        return None
    for hi, name in VERDICT_LADDER:
        if k < hi:
            return name
    return "SUBSTANTIAL"


def _load_json(path):
    if path.endswith(".xz"):
        return json.loads(lzma.open(path).read().decode("utf-8"))
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# 1. Inventory every banked panel that carries a gate-dependent kappa          #
# --------------------------------------------------------------------------- #
def inventory_panels(repo):
    pats = ["TanitAD Research Hub/**/hier*.json", "TanitAD Research Hub/**/hier*.json.xz",
            "TanitAD Research Hub/**/hierarchy_*.json", "TanitAD Research Hub/**/*4fam*.json",
            "TanitAD Research Hub/**/fourfam*.json", "stack/experiments/**/hier*.json"]
    out, seen = [], set()
    for pat in pats:
        for p in sorted(glob.glob(os.path.join(repo, pat), recursive=True)):
            rel = os.path.relpath(p, repo).replace("\\", "/")
            if rel in seen:
                continue
            seen.add(rel)
            try:
                d = _load_json(p)
            except Exception as exc:                       # pragma: no cover
                out.append({"path": rel, "error": str(exc)})
                continue
            cons = d.get("consistency") or {}
            mvt = cons.get("maneuver_vs_trajectory") or {}
            tac = (d.get("four_families") or {}).get("tactical") or {}
            k = mvt.get("kappa", tac.get("maneuver_vs_trajectory_kappa"))
            if k is None:
                continue
            gs = cons.get("gate_sensitivity") or {}
            swept = bool(gs) and gs.get("status") != "UNAVAILABLE"
            row = {
                "path": rel,
                "n_windows": d.get("n_windows") or tac.get("n_windows"),
                "n_episodes": d.get("n_episodes"),
                "kappa_at_0.15": k,
                "verdict_at_0.15": tac.get("maneuver_consistency_verdict") or verdict(k),
                "kappa_turn_subset_at_0.15": mvt.get("kappa_turn_subset"),
                "n_turn_active": mvt.get("n_turn_active"),
                "gate_swept": swept,
            }
            if swept:
                pg = gs.get("per_gate", {})
                row["kappa_at_0.10"] = (pg.get("0.10") or {}).get(
                    "maneuver_vs_trajectory_kappa")
                row["verdict_at_0.10"] = verdict(row["kappa_at_0.10"])
                row["traj_vs_gt_kappa_0.15"] = (pg.get("0.15") or {}).get(
                    "trajectory_vs_gt_kappa")
                row["traj_vs_gt_kappa_0.10"] = (pg.get("0.10") or {}).get(
                    "trajectory_vs_gt_kappa")
                row["frac_gt_turning_0.15"] = (pg.get("0.15") or {}).get("frac_gt_turning")
                row["frac_gt_turning_0.10"] = (pg.get("0.10") or {}).get("frac_gt_turning")
                ks = [v.get("maneuver_vs_trajectory_kappa") for v in pg.values()
                      if v.get("maneuver_vs_trajectory_kappa") is not None]
                row["verdict_stable_published_ladder"] = len({verdict(x) for x in ks}) == 1
                row["verdicts_across_sweep"] = sorted({verdict(x) for x in ks})
                row["verdict_stable_as_reported"] = gs.get("verdict_stable")
            else:
                row["kappa_at_0.10"] = None
                row["recompute_blocked_by"] = (
                    "panel banked only the thresholded traj_dir/gt_dir; the continuous "
                    "net yaw was discarded at write time (pre-2026-08-06 instrument). "
                    "A 0.10 read is a GPU RE-RUN, not a recompute.")
            out.append(row)
    return out


# --------------------------------------------------------------------------- #
# 2. Recompute the GT turn-magnitude half from the per-window dumps            #
# --------------------------------------------------------------------------- #
def turn_magnitude_from_dumps(repo):
    import numpy as np
    import torch

    rows = []
    for p in sorted(glob.glob(os.path.join(repo, "taniteval/results/windows_*.pt"))):
        arm = os.path.basename(p)[len("windows_"):-len(".pt")]
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:                            # pragma: no cover
            rows.append({"arm": arm, "error": str(exc)})
            continue
        if "head_deg" not in d:
            rows.append({"arm": arm, "skipped": "no head_deg (legacy dump)"})
            continue
        a = np.abs(np.asarray(d["head_deg"], dtype=float)) * math.pi / 180.0
        rows.append({
            "arm": arm,
            "n_windows": int(a.size),
            "n_episodes": len(set(d["eid"])) if "eid" in d else None,
            "median_abs_net_yaw_rad": round(float(np.median(a)), 4),
            "p90_abs_net_yaw_rad": round(float(np.percentile(a, 90)), 4),
            "max_abs_net_yaw_rad": round(float(a.max()), 4),
            "frac_turning": {f"{g:.2f}": round(float((a > g).mean()), 4) for g in GATES},
            "gate_over_median_ratio": (round(PUBLISHED_GATE / float(np.median(a)), 1)
                                       if float(np.median(a)) > 0 else None),
        })
    return rows


# --------------------------------------------------------------------------- #
# 3. Bit-level cross-check: reconstructed frac vs the panel's own gt_dir hist  #
# --------------------------------------------------------------------------- #
def crosscheck(repo, dumps):
    """`frac_gt_turning(0.15)` from head_deg must equal 1 - P(straight) of the panel's
    OWN `consistency.distributions.gt_dir`, when the window sets coincide."""
    pairs = [
        ("flagship-30k",
         "stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/hier_flagship-30k.json"),
        ("flagship-speed",
         "stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/hier_flagship-speed.json"),
        ("flagship-nospeed",
         "stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/hier_flagship-nospeed.json"),
        ("refa-dinov2",
         "stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/hier_refa-dinov2.json"),
        ("refa-dynin-30k",
         "stack/experiments/pod-rescue-20260802/pod3/root/taniteval/results/hier_refa-dynin-30k.json"),
    ]
    by_arm = {r.get("arm"): r for r in dumps}
    out = []
    for arm, rel in pairs:
        path = os.path.join(repo, rel)
        if arm not in by_arm or not os.path.exists(path):
            out.append({"arm": arm, "status": "one side absent"})
            continue
        d = _load_json(path)
        hist = ((d.get("consistency") or {}).get("distributions") or {}).get("gt_dir")
        if not hist:
            out.append({"arm": arm, "status": "panel has no gt_dir histogram"})
            continue
        n = sum(hist.values())
        panel_frac = round(1.0 - hist.get("route_straight", 0) / n, 4)
        mine = by_arm[arm]["frac_turning"]["0.15"]
        out.append({
            "arm": arm, "panel": rel, "panel_n": n,
            "dump_n": by_arm[arm]["n_windows"],
            "frac_turning_0.15_from_panel_gt_dir": panel_frac,
            "frac_turning_0.15_from_head_deg": mine,
            "abs_diff": round(abs(panel_frac - mine), 6),
            "EXACT": abs(panel_frac - mine) < 1e-6,
        })
    return out


# --------------------------------------------------------------------------- #
# 4. The OTHER swept artifact — the Alpamayo comparison (not a `hier` panel)   #
# --------------------------------------------------------------------------- #
A2_AUDIT = ("TanitAD Research Hub/Benchmarks & Eval/Research/2026-08-05-alpamayo2-super/"
            "comparison/a2_gate_audit.json")
A2_SERIES = {
    "alpamayo_declared_lateral": "vs_driven_kappa",
    "flagship_declared_maneuver": "vs_driven_kappa",
    "flagship_executed_vs_gt": "kappa",
    "alpamayo_executed_vs_gt": "kappa",
}


def alpamayo_sweep(repo):
    path = os.path.join(repo, A2_AUDIT)
    if not os.path.exists(path):
        return {"status": "ABSENT", "path": A2_AUDIT}
    d = _load_json(path)
    rows = []
    for block, field in A2_SERIES.items():
        b = d.get(block) or {}
        sw = b.get("sweep", b)
        k15 = (sw.get("0.15") or {}).get(field)
        k10 = (sw.get("0.10") or {}).get(field)
        if k15 is None or k10 is None:
            continue
        rows.append({
            "series": f"{block}.{field}", "n": b.get("n"),
            "kappa_at_0.15": k15, "kappa_at_0.10": k10,
            "delta": round(k10 - k15, 4),
            "verdict_at_0.15": verdict(k15), "verdict_at_0.10": verdict(k10),
            "verdict_moved": verdict(k15) != verdict(k10),
            "crosses_gate_sens_0.2_line": (k15 >= 0.2) != (k10 >= 0.2),
        })
    return {"status": "SWEPT", "path": A2_AUDIT,
            "n_paired_clips": (d.get("turn_magnitude") or {}).get("n"),
            "series": rows}


# --------------------------------------------------------------------------- #
# 5. Boundary proximity for the UNSWEPT kappas, using the MEASURED move band   #
# --------------------------------------------------------------------------- #
def boundary_risk(unswept, moves):
    """⛔ NOT a prediction of the 0.10 value — that needs a GPU re-run. This asks only:
    is the distance from this kappa to its nearest verdict boundary INSIDE the range of
    0.15->0.10 moves we have actually MEASURED? If yes, the verdict word is not
    established at 0.10 and must not be quoted as if it were."""
    if not moves:
        return []
    lo, hi = min(moves), max(moves)
    out = []
    for p in unswept:
        k = p["kappa_at_0.15"]
        up = [b for b in (0.1, 0.4) if b > k]
        dn = [b for b in (0.1, 0.4) if b <= k]
        d_up = round(min(up) - k, 4) if up else None
        d_dn = round(k - max(dn), 4) if dn else None
        at_risk = bool((d_up is not None and hi >= d_up)
                       or (d_dn is not None and lo <= -d_dn))
        out.append({
            "path": p["path"], "kappa_at_0.15": k, "verdict_at_0.15": p["verdict_at_0.15"],
            "dist_to_next_boundary_up": d_up, "dist_to_next_boundary_down": d_dn,
            "measured_move_band": [round(lo, 4), round(hi, 4)],
            "verdict_could_move_at_0.10": at_risk,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    panels = inventory_panels(a.repo)
    dumps = turn_magnitude_from_dumps(a.repo)
    checks = crosscheck(a.repo, dumps)

    a2 = alpamayo_sweep(a.repo)

    swept = [p for p in panels if p.get("gate_swept")]
    unswept = [p for p in panels if "kappa_at_0.15" in p and not p.get("gate_swept")]
    moved = [p for p in swept
             if p.get("verdict_at_0.10") and p["verdict_at_0.10"] != p["verdict_at_0.15"]]
    a2_moved = [s for s in a2.get("series", []) if s["verdict_moved"]]
    moves = ([p["kappa_at_0.10"] - p["kappa_at_0.15"] for p in swept
              if p.get("kappa_at_0.10") is not None]
             + [s["delta"] for s in a2.get("series", [])])
    if swept and swept[0].get("traj_vs_gt_kappa_0.10") is not None:
        moves.append(swept[0]["traj_vs_gt_kappa_0.10"] - swept[0]["traj_vs_gt_kappa_0.15"])
    risk = boundary_risk(unswept, moves)

    res = {
        "_evidence_class": "MEASURED (ours) — 0 GPU, banked artifacts only",
        "_question": "does DIR_YAW_RAD 0.15 -> 0.10 move any published VERDICT?",
        "_verdict_ladder": "four_families.py:888-890 — <0.1 DECORATIVE, <0.4 WEAK, >=0.4 SUBSTANTIAL",
        "_gate_sensitivity_threshold": GATE_SENS_VERDICT_THRESHOLD,
        "gate_independent_fields": [
            "consistency.commanded_route_vs_maneuver (agreement, kappa, turn subset) — "
            "BOTH sides gate-free: route_nav is an argmax, man_dir is MAN2DIR[man_pred] "
            "(hierarchy.py:869-870,887). RETRACTION_LOG R-2026-08-06-yawgate lists it in "
            "the blast radius; from source it is NOT gate-fed.",
            "consistency.distributions.route_follow / route_commanded / maneuver_dir",
        ],
        "gate_dependent_fields": [
            "consistency.maneuver_vs_trajectory.{agreement,kappa,agreement_turn_subset,kappa_turn_subset}",
            "consistency.commanded_route_vs_trajectory.{same four}",
            "consistency.distributions.{trajectory_dir,gt_dir}",
            "four_families.tactical.{maneuver_vs_trajectory_kappa,maneuver_vs_trajectory_agreement,"
            "maneuver_consistency_verdict}",
            "legacy_overlapping_holdout_se.consistency.* (deprecated block, same inputs)",
        ],
        "instrument_gaps": [
            "_gate_sensitivity sweeps maneuver_vs_trajectory_kappa and trajectory_vs_gt_kappa "
            "ONLY. kappa_turn_subset is gate-dependent and is NOT swept — so the one number "
            "sitting ON a boundary (0.2005 on the 880-window panel) cannot be re-read even "
            "on the swept panel.",
            "verdict_stable tests kappa >= 0.2, but the PUBLISHED verdict word uses the "
            "0.1/0.4 ladder. The field named verdict_stable does not test the verdict that "
            "is published.",
        ],
        "n_panels_with_gate_dependent_kappa": len(panels),
        "n_swept": len(swept),
        "n_unswept_recompute_impossible": len(unswept),
        "n_verdicts_moved_0.15_to_0.10_hier_panels": len(moved),
        "n_verdicts_moved_0.15_to_0.10_alpamayo": len(a2_moved),
        "measured_move_band_0.15_to_0.10": [round(min(moves), 4), round(max(moves), 4)],
        "n_measured_series": len(moves),
        "panels": panels,
        "alpamayo_comparison": a2,
        "unswept_boundary_risk": risk,
        "n_unswept_whose_verdict_could_move": sum(
            1 for r in risk if r["verdict_could_move_at_0.10"]),
        "gt_turn_magnitude_recomputed": dumps,
        "crosscheck_head_deg_is_the_gate_input": checks,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"[gate-reread] panels={len(panels)} swept={len(swept)} "
          f"unswept={len(unswept)} hier_verdicts_moved={len(moved)} "
          f"a2_verdicts_moved={len(a2_moved)} "
          f"unswept_at_risk={res['n_unswept_whose_verdict_could_move']} -> {a.out}")
    print(f"  measured 0.15->0.10 move band over {len(moves)} series: "
          f"{res['measured_move_band_0.15_to_0.10']}")
    for s in a2.get("series", []):
        print(f"  A2 {s['series']:44s} {s['kappa_at_0.15']} -> {s['kappa_at_0.10']} "
              f"{s['verdict_at_0.15']} -> {s['verdict_at_0.10']}"
              f"{'  ** VERDICT MOVED **' if s['verdict_moved'] else ''}")
    for r in risk:
        if r["verdict_could_move_at_0.10"]:
            print(f"  AT RISK k={r['kappa_at_0.15']} ({r['verdict_at_0.15']}) "
                  f"up={r['dist_to_next_boundary_up']} dn={r['dist_to_next_boundary_down']} "
                  f"{r['path']}")
    for c in checks:
        print("  crosscheck", c.get("arm"), c.get("EXACT"), c.get("abs_diff"))


if __name__ == "__main__":
    main()
