#!/usr/bin/env python3
"""Compare OUR navhard CV row against the PUBLISHED one — the external check on the harness.

Published values are LITERALS, transcribed from the banked primary (never from a summary):
[N2] Cao et al., *Pseudo-Simulation for Autonomous Driving*, **arXiv 2506.04218v3 (20 Mar 2026)**,
Table 2 "navhard leaderboard. Snapshot from 03/2026", column **CV [8]**.
Library key `2506.04218`, sha256 a8431697ffa276068375c937cc2f90a2c8059a17b285fba0f0c8ba41eb877e81
(verified against library.json). The leaderboard figure 11.4816 is INHERITED
(NAVSIM_PROTOCOL.md §6.2, PUB-LB snapshot 2026-08-23) and is NOT re-fetched here (no downloads).

Pre-registered readings (SPEC_NAVHARD.md): |100*EPDMS - 11.4| <= 0.05 reproduces the paper's printed
value; each sub-score within +-0.05 reproduces that term; a miss > 0.5 on the combined EPDMS is a
HARNESS DEFECT to diagnose first.
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd

PUB = {  # [N2]v3 Table 2, CV column: (stage 1, stage 2)
    "NC": (88.8, 83.2), "DAC": (42.8, 59.1), "DDC": (70.6, 76.5), "TLC": (99.3, 98.0),
    "EP": (77.5, 71.3), "TTC": (87.3, 81.1), "LK": (78.6, 47.9), "HC": (97.1, 97.1), "EC": (60.4, 61.9),
}
PUB_EPDMS = 11.4
PUB_LB_EPDMS = 11.4816
LONG = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "DDC": "driving_direction_compliance",
        "TLC": "traffic_light_compliance", "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
        "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}
SUM = {"stage_one": "extended_pdm_score_stage_one", "stage_two": "extended_pdm_score_stage_two",
       "combined": "extended_pdm_score_combined"}

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--note", default="")
a = ap.parse_args()
df = pd.read_csv(sorted(glob.glob(a.csv))[-1], index_col=0)
row = {k: df[df.token == v].iloc[0] for k, v in SUM.items()}
out = {"source": "[N2] arXiv 2506.04218v3 Table 2 (banked primary, library key 2506.04218), column CV",
       "csv": sorted(glob.glob(a.csv))[-1], "note": a.note, "submetrics": {}, "epdms": {}}
worst = 0.0
for sh, (p1, p2) in PUB.items():
    ours1 = 100 * float(row["stage_one"][LONG[sh] + "_stage_one"])
    ours2 = 100 * float(row["stage_two"][LONG[sh] + "_stage_two"])
    out["submetrics"][sh] = {"stage_one": {"ours": round(ours1, 2), "published": p1, "delta": round(ours1 - p1, 2)},
                             "stage_two": {"ours": round(ours2, 2), "published": p2, "delta": round(ours2 - p2, 2)}}
    worst = max(worst, abs(ours1 - p1), abs(ours2 - p2))
comb = 100 * float(row["combined"]["score"])
out["epdms"] = {"ours_x100": round(comb, 4), "published_paper": PUB_EPDMS, "delta_vs_paper": round(comb - PUB_EPDMS, 4),
                "reproduces_paper_within_0.05": abs(comb - PUB_EPDMS) <= 0.05,
                "published_leaderboard_INHERITED": PUB_LB_EPDMS, "delta_vs_leaderboard": round(comb - PUB_LB_EPDMS, 4),
                "stage_one_x100": round(100 * float(row["stage_one"]["score"]), 4),
                "stage_two_x100": round(100 * float(row["stage_two"]["score"]), 4),
                "harness_defect_threshold_0.5": abs(comb - PUB_EPDMS) > 0.5}
out["worst_submetric_abs_delta"] = round(worst, 2)
a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["epdms"], indent=1))
print("worst submetric |delta| =", out["worst_submetric_abs_delta"])
for sh, v in out["submetrics"].items():
    print(f"  {sh:4s} S1 {v['stage_one']['ours']:7.2f} vs {v['stage_one']['published']:5.1f} "
          f"({v['stage_one']['delta']:+.2f})   S2 {v['stage_two']['ours']:7.2f} vs {v['stage_two']['published']:5.1f} "
          f"({v['stage_two']['delta']:+.2f})")
