#!/usr/bin/env python3
"""compare_v9_v10.py -- per-field label-change table between the banked v9 release and the v10 (band 2-6 s) release.

Rows are aligned by (sid, k) and checked to carry the same clock. For each field: the share of rows whose value
differs (NaN equals NaN), among (A) all rows and (B) the full-band rows of v9 (``h_obs_s >= 8``, the population of
the truncation test in ``AUDIT_MODEL.md`` F8: 5.3 % lat-a, 7.3 % lat-b, 15.4 % lon on eval139, INHERITED), and for
the class fields the share among rows where v9 holds a label (not IGNORE = -100).
"""
import argparse
import json
import sys

import numpy as np

IGN = -100
LAT_C = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg", "turn_r_arc_m",
         "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m")
LON_C = ("lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m", "lead_gap_m",
         "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s")
SPD = ("v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
CLS = ("lat_cls_a", "lat_cls_b", "lon_cls")
ALLOWED = ("lat_allowed_a", "lat_allowed_b", "lon_allowed")
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT", "SPEED_BAND",
          "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET", "REACT_ON_ONCOMING",
          "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW",
          "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")


def differ(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f":
        return ~((a == b) | (np.isnan(a) & np.isnan(b)))
    return a != b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9", required=True)
    ap.add_argument("--v10", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z9, z10 = np.load(a.v9), np.load(a.v10)
    assert (z9["clip__sid"] == z10["clip__sid"]).all() and (z9["row__k"] == z10["row__k"]).all()
    assert np.array_equal(z9["row__now_s"], z10["row__now_s"], equal_nan=True), "clock differs"
    N = len(z9["row__k"])
    full = z9["row__h_obs_s"] >= 8.0 - 1e-9
    res = {"n_rows": int(N), "n_rows_v9_fullband": int(full.sum()), "fields": {}}

    def add(name, d, defined9=None, defined10=None):
        r = {"changed_all": float(d.mean()), "n_changed_all": int(d.sum()), "changed_fullband": float(d[full].mean()) if full.any() else None}
        if defined9 is not None:
            m = defined9
            r["v9_defined"] = float(m.mean())
            r["v10_defined"] = float(defined10.mean())
            r["changed_among_v9_defined"] = float(d[m].mean()) if m.any() else None
            r["changed_among_v9_defined_fullband"] = float(d[m & full].mean()) if (m & full).any() else None
            both = m & defined10
            r["changed_among_both_defined"] = float(d[both].mean()) if both.any() else None
            r["defined_to_ignore"] = int((m & ~defined10).sum())
            r["ignore_to_defined"] = int((~m & defined10).sum())
        res["fields"][name] = r

    for f in CLS:
        x, y = z9["row__" + f], z10["row__" + f]
        add(f, differ(x, y), x != IGN, y != IGN)
    for f in ALLOWED:
        add(f, differ(z9["row__" + f], z10["row__" + f]))
    gy9, gw9, gy10, gw10 = (z[f"row__goal_{c}"].astype(np.uint64) for z in (z9, z10) for c in ("y", "w"))
    add("goal_y(22-bit mask)", gy9 != gy10)
    add("goal_w(22-bit mask)", gw9 != gw10)
    res["goal_bits"] = {}
    for i, g in enumerate(GOAL22):
        b = np.uint64(1) << np.uint64(i)
        w9, w10 = (gw9 & b) != 0, (gw10 & b) != 0
        y9, y10 = (gy9 & b) != 0, (gy10 & b) != 0
        res["goal_bits"][g] = {"w_changed": float((w9 != w10).mean()), "y_changed": float((y9 != y10).mean()),
                               "w_v9": float(w9.mean()), "w_v10": float(w10.mean()), "pos_v9": float((y9 & w9).mean()), "pos_v10": float((y10 & w10).mean())}
    for f in LAT_C + LON_C + SPD:
        x, y = z9["row__" + f], z10["row__" + f]
        d = differ(x, y)
        add(f, d, np.isfinite(x), np.isfinite(y))
    add("h_obs_s", differ(z9["row__h_obs_s"], z10["row__h_obs_s"]))
    cons_cells = LAT_C + LON_C + SPD
    anyc = np.zeros(N, bool)
    for f in cons_cells:
        anyc |= differ(z9["row__" + f], z10["row__" + f])
    res["any_constraint_cell_changed"] = float(anyc.mean())
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for k in CLS + ("goal_y(22-bit mask)", "goal_w(22-bit mask)"):
        r = res["fields"][k]
        print(k, {q: (round(v, 4) if isinstance(v, float) else v) for q, v in r.items()})


if __name__ == "__main__":
    main()
