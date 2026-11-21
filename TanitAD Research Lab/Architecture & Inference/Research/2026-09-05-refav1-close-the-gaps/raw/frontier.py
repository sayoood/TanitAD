#!/usr/bin/env python3
"""The refav1 cost-geometry FRONTIER, re-derived from the RECORDS, plus the
per-METRIC inference-seed floors. Zero GPU.

⛔ WHY THIS RE-DERIVES RATHER THAN QUOTES. `M51`: a name is not provenance.
`W_KAPPA` is read from each record's own `cost.weights`, the tier from
`arms[<arm>].tier`, and the GT arm is `ol` -- never a run whose NAME contains
"oracle". Two of this package's inputs (`lonseam`, `ccosh_w000`) are arms whose
levers are provably inert, and they are LABELLED as such here rather than being
quoted as lever results.

⛔ AND IT PRINTS TURN RECALL BESIDE ADE ON EVERY ROW. This package exists
because the best-ADE arm executes ZERO turns and ADE cannot see it.

ASCII output only (cp1252 dev box).
"""
import glob
import json
import os
import sys

FAM = "four_families"


def row(path):
    d = json.load(open(path, encoding="utf-8"))
    cl = (d.get("arms") or {}).get("cl")
    if cl is None:
        return None
    ff = cl[FAM]
    lat, tac, lon = ff["lateral"], ff["tactical"], ff["longitudinal"]
    pc = (tac.get("lateral_decision") or {}).get("per_class") or {}
    cost = d.get("cost") or {}
    w = cost.get("weights") or {}
    out = dict(
        name=os.path.basename(path)[4:-5],
        arm=d.get("arm"),
        tier=cl.get("tier"),
        metric=cost.get("metric"),
        wk=w.get("W_KAPPA"),
        wj=w.get("W_JERK"),
        wv=w.get("W_VEND"),
        wkg=cost.get("w_kappa_by_goal"),
        ade=cl["legacy_epmean_row"]["ade_m"],
        curv=lat["curvature_mae_1pm"],
        head=lat["heading_mae_deg"],
        cross=lat["cross_mae_m"],
        spd=lon["speed_mae_mps"],
        n=ff.get("_grid", {}).get("n_windows") if isinstance(ff.get("_grid"), dict) else None,
    )
    for k in ("lane_keep", "turn_left", "turn_right"):
        out[k] = (pc.get(k) or {}).get("recall")
        out[k + "_n"] = (pc.get(k) or {}).get("n_true")
    # the OTHER arms, which must be identical across levers on one grid
    for a in ("ha", "ha0", "ha0_ext", "ol"):
        aa = (d.get("arms") or {}).get(a)
        out["ADE_" + a] = (aa or {}).get("legacy_epmean_row", {}).get("ade_m")
    return out


def table(rows, title):
    print("=" * 118)
    print(title)
    print("=" * 118)
    hdr = ("%-16s %-6s %11s %9s %9s %8s %8s %8s %8s %8s"
           % ("arm", "metric", "W_KAPPA", "ADE_m", "curvMAE", "turnL", "turnR",
              "laneK", "spdMAE", "headMAE"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(x, w=8, p=4):
            return ("%*.*f" % (w, p, x)) if isinstance(x, (int, float)) else "%*s" % (w, "-")
        print("%-16s %-6s %11s %s %s %s %s %s %s %s"
              % (r["name"], str(r["metric"]), str(r["wk"]),
                 f(r["ade"], 9), f(r["curv"], 9, 5), f(r["turn_left"]),
                 f(r["turn_right"]), f(r["lane_keep"]), f(r["spd"]),
                 f(r["head"], 8, 2)))
    print()
    ns = {r["turn_left_n"] for r in rows if r["turn_left_n"] is not None}
    print("  n_true(turn_left)=%s  n_true(turn_right)=%s  n_true(lane_keep)=%s"
          % (sorted(ns),
             sorted({r["turn_right_n"] for r in rows if r["turn_right_n"] is not None}),
             sorted({r["lane_keep_n"] for r in rows if r["lane_keep_n"] is not None})))
    # ⛔ THE GRID CONTROL. `ha`/`ha0`/`ha0_ext`/`ol` do not depend on any cost
    # lever, so a difference means the arms are NOT on one window grid and no
    # pairing below is admissible.
    print("  GRID CONTROL (must be constant across every row):")
    for a in ("ha", "ha0", "ha0_ext", "ol"):
        vals = sorted({round(r["ADE_" + a], 6) for r in rows
                       if isinstance(r.get("ADE_" + a), float)})
        ok = "OK" if len(vals) == 1 else "*** DIFFERS ***"
        print("    ADE_%-8s %s  %s" % (a, vals, ok))
    print()


def floor(a, b, label):
    """Per-METRIC inference-seed floor from one replicate pair."""
    print("=" * 118)
    print("SEED FLOOR: %s" % label)
    print("  A = %s   B = %s" % (a["arm"], b["arm"]))
    print("=" * 118)
    print("  CONTROL - the non-planner arms MUST be identical (same grid):")
    ctl_ok = True
    for k in ("ha", "ha0", "ha0_ext", "ol"):
        x, y = a.get("ADE_" + k), b.get("ADE_" + k)
        same = (x == y)
        ctl_ok &= bool(same)
        print("    %-8s %.4f vs %.4f  identical=%s" % (k, x, y, same))
    print("    CONTROL %s" % ("PASSED" if ctl_ok else "*** FAILED - VOID ***"))
    print()
    print("    %-22s %10s %10s %10s" % ("statistic", "seed A", "seed B", "abs diff"))
    keys = [("ADE_m", "ade"), ("LAT curv_MAE", "curv"), ("LAT heading_MAE", "head"),
            ("LAT cross_MAE", "cross"), ("LON speed_MAE", "spd"),
            ("TACpc lane_keep_rec", "lane_keep"),
            ("TACpc turn_left_rec", "turn_left"),
            ("TACpc turn_right_rec", "turn_right")]
    out = {}
    for name, k in keys:
        x, y = a.get(k), b.get(k)
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        out[name] = abs(x - y)
        print("    %-22s %10.5f %10.5f %10.5f" % (name, x, y, abs(x - y)))
    print()
    return out


def main(argv):
    dev = "C:/Users/Admin/refav1_margin/p4out"
    rows = [r for r in (row(p) for p in sorted(glob.glob(dev + "/rec_*.json"))) if r]
    rows.sort(key=lambda r: (r["wk"] if isinstance(r["wk"], float) else 0.0, r["ade"]))
    table(rows, "DEV-BOX p4 PANEL (40 windows / 8 episodes, stride 16, ccos, T1 'cl')")

    by = {r["name"]: r for r in rows}
    floors = {}
    if "ccos_argmax" in by and "ccos_seed1" in by:
        floors["ccos"] = floor(by["ccos_argmax"], by["ccos_seed1"],
                               "ccos_argmax vs ccos_seed1 (W_KAPPA=0, no cap)")
    if "combined" in by and "combined_seed1" in by:
        floors["combined"] = floor(by["combined"], by["combined_seed1"],
                                   "combined vs combined_seed1 (cap + ladder)")

    thor = argv[1] if len(argv) > 1 else None
    if thor and os.path.isdir(thor):
        trows = [r for r in (row(p) for p in sorted(glob.glob(thor + "/rec_*.json"))) if r]
        if trows:
            trows.sort(key=lambda r: r["ade"])
            table(trows, "THOR p4 PANEL -- same ckpt/labels/episodes/grid, DIFFERENT GPU. "
                         "⛔ CEM is not bit-reproducible across GPUs: pair WITHIN this table only.")
            tb = {r["name"]: r for r in trows}
            if "T_wk15" in tb and "T_wk15_s1" in tb:
                floors["thor_wk15"] = floor(tb["T_wk15"], tb["T_wk15_s1"],
                                            "T_wk15 vs T_wk15_s1 (THOR-LOCAL, the A3 pair's floor)")

    if floors:
        print("=" * 118)
        print("BINDING PER-METRIC FLOOR = max over the available replicate pairs")
        print("=" * 118)
        allk = sorted({k for f in floors.values() for k in f})
        print("    %-22s %s %10s" % ("statistic",
                                     "".join("%12s" % n for n in floors),
                                     "BINDING"))
        for k in allk:
            vals = [floors[n].get(k) for n in floors]
            m = max(v for v in vals if v is not None)
            print("    %-22s %s %10.5f"
                  % (k, "".join(("%12.5f" % v) if v is not None else "%12s" % "-"
                                for v in vals), m))
        print()
        print("  ⭐ turn_left / turn_right recall floors of EXACTLY 0.00000 mean both")
        print("     replicates reproduced the recall to four decimals => ANY turn-recall")
        print("     change on this rig is above its own inference-seed noise floor.")
        print("  ⛔ A floor is quoted for the exact STATISTIC, never for the family.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
