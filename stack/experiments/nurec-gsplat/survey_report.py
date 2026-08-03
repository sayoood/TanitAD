#!/usr/bin/env python3
"""Rank the stage-1 NuRec survey and pick junction-decision candidates.

The selection criterion is NOT "the scene has a junction".  MEASURED on
00040136: the ego is inside a junction for 46 of its 202 poses and the
strategic family is still degenerate, because every traversal is a
straight-through pass with one lane-level continuation.  What a strategic
benchmark needs is a scene where the ego **turns at a junction, with clip
either side of the turn**, so that the decision is (i) taken inside the
window and (ii) observable before and after.

Tiers, in the order a benchmark wants them:

  T1  a COMPLETE traversal (>=10 poses before entry AND >=10 after exit)
      with |heading change through| >= 60 deg          — a real turn
  T2  same completeness, 25 deg <= |change| < 60 deg   — a slip road / merge
  T3  complete traversal, |change| < 25 deg            — straight-through
      (this is where 00040136 sits)
  T4  a traversal that is TRUNCATED by the clip boundary
  T5  intersection polygons present but never entered
  T6  no intersection polygon at all
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

T1_DEG, T2_DEG = 60.0, 25.0
MARGIN = 10          # poses required before entry and after exit


def tier(rec):
    if rec.get("error"):
        return "ERR"
    tr = rec.get("traversals") or []
    if not tr:
        return "T5" if rec.get("n_intersection_areas", 0) else "T6"
    comp = [t for t in tr
            if t["poses_before_entry"] >= MARGIN and t["poses_after_exit"] >= MARGIN]
    if not comp:
        return "T4"
    best = max(abs(t["heading_change_through_deg"]) for t in comp)
    return "T1" if best >= T1_DEG else ("T2" if best >= T2_DEG else "T3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/tmp/nurec_survey/stage1.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    recs = []
    for line in Path(a.jsonl).read_text().splitlines():
        if line.strip():
            recs.append(json.loads(line))
    # a resumed run can append a scene twice; keep the last
    dedup = {r["scene_id"]: r for r in recs}
    recs = list(dedup.values())
    for r in recs:
        r["tier"] = tier(r)

    tiers = Counter(r["tier"] for r in recs)
    ok = [r for r in recs if not r.get("error")]
    turn = np.array([r.get("best_complete_turn_deg", 0.0) for r in ok])

    night = dedup.get("00040136-e651-4abd-991d-0655ccda9430")
    rep = {
        "tool": "survey_report.py", "evidence_class": "MEASURED",
        "n_scenes_scored": len(recs),
        "n_errors": int(tiers.get("ERR", 0)),
        "tier_counts": dict(sorted(tiers.items())),
        "tier_legend": {
            "T1": f"complete traversal, |dyaw| >= {T1_DEG} deg  (A REAL TURN)",
            "T2": f"complete traversal, {T2_DEG} <= |dyaw| < {T1_DEG} deg",
            "T3": f"complete traversal, |dyaw| < {T2_DEG} deg  (straight-through)",
            "T4": "traversal truncated by the clip boundary",
            "T5": "intersection polygons present but never entered",
            "T6": "no intersection polygon at all",
            "margin_poses_each_side": MARGIN,
        },
        "best_complete_turn_deg_distribution": {
            "n": int(len(turn)),
            "frac_zero": round(float((turn <= 1e-9).mean()), 4),
            "p50": round(float(np.percentile(turn, 50)), 2),
            "p75": round(float(np.percentile(turn, 75)), 2),
            "p90": round(float(np.percentile(turn, 90)), 2),
            "p99": round(float(np.percentile(turn, 99)), 2),
            "max": round(float(turn.max()), 2),
        },
        "n_poses_hist": dict(Counter(r.get("n_poses") for r in ok).most_common(6)),
        "intersection_area_count_hist": dict(
            Counter(r.get("n_intersection_areas", 0) for r in ok).most_common(10)),
    }
    if night:
        rep["reference_scene_00040136"] = {
            "tier": night["tier"],
            "best_complete_turn_deg": night.get("best_complete_turn_deg"),
            "percentile_of_best_complete_turn": round(float(
                (turn <= night.get("best_complete_turn_deg", 0.0)).mean() * 100), 2),
            "n_traversals": len(night.get("traversals") or []),
        }

    def key(r):
        comp = [t for t in (r.get("traversals") or [])
                if t["poses_before_entry"] >= MARGIN and t["poses_after_exit"] >= MARGIN]
        best = max([abs(t["heading_change_through_deg"]) for t in comp] or [0.0])
        margin = max([min(t["poses_before_entry"], t["poses_after_exit"])
                      for t in comp] or [0])
        return (best, margin, len(comp))

    cands = sorted((r for r in ok if r["tier"] in ("T1", "T2")), key=key, reverse=True)
    rep["candidates"] = []
    for r in cands[:a.top]:
        comp = [t for t in r["traversals"]
                if t["poses_before_entry"] >= MARGIN and t["poses_after_exit"] >= MARGIN]
        rep["candidates"].append({
            "scene_id": r["scene_id"], "tier": r["tier"],
            "n_poses": r["n_poses"], "path_length_m": r["path_length_m"],
            "heading_net_change_deg": r["heading_net_change_deg"],
            "best_complete_turn_deg": r["best_complete_turn_deg"],
            "n_complete_traversals": len(comp),
            "n_traversals": len(r["traversals"]),
            "traversals": comp,
            "dist_to_nearest_intersection_m": r.get("dist_to_nearest_intersection_m"),
        })
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"wrote {a.out}")
    print(json.dumps({k: v for k, v in rep.items() if k != "candidates"}, indent=1))
    print("\nTOP CANDIDATES")
    for c in rep["candidates"][:a.top]:
        t = c["traversals"][0]
        print(f"  {c['scene_id']}  {c['tier']}  turn={c['best_complete_turn_deg']:7.2f} deg  "
              f"net={c['heading_net_change_deg']:7.2f}  n_trav={c['n_traversals']}  "
              f"cat={t['category']}  span={t['pose_span']} "
              f"pre={t['poses_before_entry']} post={t['poses_after_exit']}")


if __name__ == "__main__":
    main()
