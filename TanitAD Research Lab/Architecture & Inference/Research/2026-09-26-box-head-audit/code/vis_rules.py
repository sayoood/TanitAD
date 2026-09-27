#!/usr/bin/env python3
"""vis_rules.py -- the proposed refcv7 camera-visibility target rule VIS-1, and its effect, from vis_zbuf *_boxes.json.

VIS-1 (one function for TRAINING targets AND EVAL scoring), applied to rows that already pass the trainer's
visible_target_filter (120 deg field AND the decode box):
  POSITIVE  vis_frac >= 0.30 AND n_vis >= 100 px     (vis_frac = visible AND in-image share of the full silhouette)
  IGNORE    0.05 <= vis_frac < 0.30, OR (vis_frac >= 0.30 AND n_vis < 100 px)
  DROPPED   vis_frac < 0.05 (the camera cannot see it: background for the loss, absent for the metric)
Prints positives / ignores / dropped per window, by class and by range, and a threshold sensitivity grid.
"""
import collections
import json
import sys

import numpy as np

T_POS, T_IGN, PX_MIN = 0.30, 0.05, 100


def label(r, t_pos=T_POS, t_ign=T_IGN, px=PX_MIN):
    v = r["vis_frac"]
    if v != v:
        return "dropped"
    if v >= t_pos and r["n_vis"] >= px:
        return "positive"
    if v >= t_ign:
        return "ignore"
    return "dropped"


def main():
    out = {"rule": {"positive": f"vis_frac >= {T_POS} and n_vis >= {PX_MIN} px", "ignore": f"{T_IGN} <= vis_frac "
                    f"< {T_POS}, or vis_frac >= {T_POS} with n_vis < {PX_MIN} px", "dropped": f"vis_frac < {T_IGN}"},
           "sets": {}}
    for p in sys.argv[1:]:
        rows = [r for r in json.load(open(p)) if r["target"]]
        # windows = ALL labelled windows of the run (the sibling summary's n_windows), not only those with targets
        nw = int(json.load(open(p.replace("_boxes.json", ".json")))["n_windows"])
        lab = collections.Counter(label(r) for r in rows)
        by_cls = collections.defaultdict(collections.Counter)
        by_rng = collections.defaultdict(collections.Counter)
        for r in rows:
            by_cls[r["cls"]][label(r)] += 1
            k = next(f"{lo}-{hi}m" for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90), (90, 1e9))
                     if lo <= r["range"] < hi)
            by_rng[k][label(r)] += 1
        grid = {}
        for tp in (0.2, 0.3, 0.4, 0.5):
            for px in (50, 100, 200):
                grid[f"vis>={tp}&px>={px}"] = round(sum(1 for r in rows if label(r, tp, T_IGN, px) == "positive") / max(nw, 1), 3)
        s = {"n_labelled_windows": nw, "n_targets_filter_only": len(rows),
             "targets_per_window_filter_only": len(rows) / max(nw, 1),
             "positives_per_window": lab["positive"] / max(nw, 1), "ignores_per_window": lab["ignore"] / max(nw, 1),
             "dropped_per_window": lab["dropped"] / max(nw, 1), "counts": dict(lab),
             "by_class": {c: dict(v) for c, v in sorted(by_cls.items())},
             "by_range": {k: dict(v) for k, v in by_rng.items()},
             "sensitivity_positives_per_window": grid}
        out["sets"][p.split("/")[-1]] = s
        print(p, json.dumps({k: v for k, v in s.items() if k not in ("by_class", "by_range")}), flush=True)
        for c, v in s["by_class"].items():
            print("   ", c, v)
        for c, v in s["by_range"].items():
            print("   ", c, v)
    json.dump(out, open("vis_rules.json", "w"), indent=1)


if __name__ == "__main__":
    main()
