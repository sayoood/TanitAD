#!/usr/bin/env python3
"""Compact printer for p1_diag_<scope>.json (used to build P1_DIAG.md)."""
import json
import sys

d = json.load(open(sys.argv[1]))


def f(t):
    return "%.3f [%.3f, %.3f]" % tuple(t)


print(d["scope"], "cruise", d["n_cruise"], "slow picks", d["n_pop_slow_picks"], "logs", d["n_logs_pop"], "with fast", d["n_popf"],
      "slowshare", f(d["slow_share_cruise"]))
for k, v in d["Q1"].items():
    print("Q1", k, f(v) if isinstance(v, list) else v)
for c, o in d["comparators"].items():
    print("==", c, "dlog agg total", f(o["dlog_agg_total"]))
    print(" terms", {k: f(v) for k, v in o["dlog_agg_by_term"].items()})
    print(" dominant", o["dominant_term_counts"])
    print(" dlogit", {k: f(v) for k, v in o["dlogit_pick_minus_f"].items()})
    print(" pdms pick", f(o["true_pdms_pick"]), "f", f(o["true_pdms_f"]), "f-pick", f(o["true_pdms_f_minus_pick"]),
          "share f>=pick", f(o["share_f_pdms_ge_pick"]), "share f>pick", f(o["share_f_pdms_gt_pick"]))
    print(" v4/v0 pick", f(o["v4_over_v0_pick"]), "f", f(o["v4_over_v0_f"]))
    for h, x in o["pred_vs_true"].items():
        print("  %-4s pred pick/f %.3f/%.3f true pick/f %.3f/%.3f | pred diff %s true diff %s" % (
            h, x["pred_pick"][0], x["pred_f"][0], x["true_pick"][0], x["true_f"][0], f(x["pred_diff_pick_minus_f"]), f(x["true_diff_pick_minus_f"])))
    print("  lane pred pick/f %.3f/%.3f" % (o["pred_lane"]["pick"][0], o["pred_lane"]["f"][0]))
print("Q3", {k: (f(v) if isinstance(v, list) else v) for k, v in d["Q3_human_nearest"].items()})
for row in d["Q4_calibration_by_speed_bin_cruise_all_valid_proposals"]:
    print("Q4", row["bin"], row["n"], " ".join("%s=%.3f" % (k, v) for k, v in row.items() if k not in ("bin", "n") and v is not None))
print("spearman", {k: (round(v[0], 3) if v[0] is not None else None, v[1]) for k, v in d["Q4_within_token_spearman_vs_speed"].items()})
