#!/usr/bin/env python3
"""Collect the four-family readings of the five selectors (families6.py JSONs under raw/families/) into raw/families_table.json + families_table.md.
Intervals are families6's own PER-ARM episode-cluster bootstrap (136 logs, 2000 resamples) -- UNPAIRED, so reading a difference off two of them is conservative."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "raw", "families")
SEL = ("P2_pick", "ORACLE_EP", "ORACLE_TTC", "ORACLE_EP_TTC", "ORACLE_best64")
SPEC = (("LONGITUDINAL", "longitudinal", (("speed_mae_mps", "speed_mae_mps"), ("along_mae_m", "along_mae_m"), ("accel_mae_mps2", "accel_mae_mps2"),
                                         ("target_speed_acc.within_1.0_mps", "target_speed_within_1.0mps"), ("ego_progress.progress_ratio_mean", "progress_ratio_mean (human=1)"))),
        ("LATERAL", "lateral", (("heading_mae_deg", "heading_mae_deg"), ("yaw_rate_mae_degps", "yaw_rate_mae_degps"), ("curvature_mae_1pm", "curvature_mae_1pm"),
                                ("cross_mae_m", "cross_track_mae_m"), ("cross_final_mae_m", "cross_track_final_mae_m"))))
TAC = (("lateral_decision", "accuracy"), ("longitudinal_decision", "accuracy"), ("maneuver_5way_collapsed", "accuracy"), ("goal_setting", "goal_point_error_m"))
out = {"_what": "four metric families for the 5 selectors on the full navtest (families6.py, unchanged). STRATEGIC PAUSED (PI 2026-09-19 item 25) and UNAVAILABLE in NAVSIM. "
                "Longitudinal distance-keeping UNAVAILABLE: no lead-agent track supplied (families6 does not pass `lead=`), n = 0.",
       "intervals": "families6 per-arm episode-cluster bootstrap, 136 logs, 2,000 resamples; UNPAIRED", "selectors": {}}
for s in SEL:
    d = json.load(open(os.path.join(D, f"{s}_families.json"), encoding="utf-8"))["families"]["refcv6"]
    r = {}
    for fam_name, fk, items in SPEC:
        comp = d[fk]["ci"]["components"]
        for key, label in items:
            c = comp.get(key)
            r[f"{fam_name}.{label}"] = None if c is None else [c["mean"], c["lo"], c["hi"]]
    for blk, m in TAC:
        c = d["tactical"][blk]["ci"].get(m)
        r[f"TACTICAL.{blk}.{m}"] = None if c is None else [c["mean"], c["lo"], c["hi"]]
    out["selectors"][s] = r
json.dump(out, open(os.path.join(HERE, "..", "raw", "families_table.json"), "w", encoding="utf-8"), indent=1)
keys = list(out["selectors"]["P2_pick"].keys())
with open(os.path.join(HERE, "..", "raw", "families_table.md"), "w", encoding="utf-8") as f:
    f.write("| component | " + " | ".join(SEL) + " |\n|---|" + "---|" * len(SEL) + "\n")
    for k in keys:
        f.write(f"| {k} | " + " | ".join("n/a" if out["selectors"][s][k] is None else f"{out['selectors'][s][k][0]:.3f} [{out['selectors'][s][k][1]:.3f}, {out['selectors'][s][k][2]:.3f}]" for s in SEL) + " |\n")
print(open(os.path.join(HERE, "..", "raw", "families_table.md"), encoding="utf-8").read())
