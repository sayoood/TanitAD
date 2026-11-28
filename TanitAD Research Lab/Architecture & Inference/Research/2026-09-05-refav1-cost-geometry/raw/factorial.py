"""THE 2x2x2 FACTORIAL over {W_KAPPA} x {kamm cap} x {seed-kappa ladder}, in one table.

Every cell is an arm that was actually run; a missing cell is printed as ABSENT with
its reason and is never silently dropped. Rows carry the four binding families, the
per-class tactical recalls, and the `assert_feasible` safety rows re-derived from the
dump at report time -- so the floor, the rate and the families in one claim can never
come from different generations of different files (the failure this package
retracted).

The MAIN EFFECTS at the bottom are simple marginal contrasts, NOT model fits, and are
printed beside the per-metric inference-seed floor so a difference smaller than the
rig's own run-to-run noise cannot read as an effect.  ASCII only.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\stack")
from tanitad.refs import feasible_decode as FD          # noqa: E402

P4 = "C:/Users/Admin/refav1_margin/p4out"
DT, VMIN = 0.2, 2.0

#: (tag, W_KAPPA, cap, ladder)  -- the eight cells
CELLS = [("ccos_argmax", 0, 0, 0),
         ("wk15",        1, 0, 0),
         ("kamm07",      0, 1, 0),
         ("l3ladder",    0, 0, 1),
         ("wk15_ladder", 1, 0, 1),
         ("combined",    0, 1, 1),
         ("best",        1, 1, 0),        # sibling agent's arm (queueJ.sh)
         ("bestlad",     1, 1, 1)]        # mine

M = [("ADE", ("intervals", "metrics", "ade_dense_m", "mean"), 4),
     ("LONspd", ("four_families", "longitudinal", "speed_mae_mps"), 4),
     ("LONacc", ("four_families", "longitudinal", "accel_mae_mps2"), 4),
     ("LAThead", ("four_families", "lateral", "heading_mae_deg"), 3),
     ("LATyaw", ("four_families", "lateral", "yaw_rate_mae_degps"), 3),
     ("LATcurv", ("four_families", "lateral", "curvature_mae_1pm"), 5),
     ("LATcross", ("four_families", "lateral", "cross_mae_m"), 4),
     ("TAClatK", ("four_families", "tactical", "lateral_decision", "kappa"), 4),
     ("TACm5", ("four_families", "tactical", "maneuver_5way_collapsed", "accuracy"), 4),
     ("TACgFDE", ("four_families", "tactical", "goal_setting", "goal_point_error_m"), 4)]

PC = [("recL", "turn_left"), ("recR", "turn_right"), ("recLK", "lane_keep")]

#: per-metric inference-seed floor, MEASURED on ccos_argmax vs ccos_seed1
#: (raw/seed_floor_ext_ccos.txt). A blank means the statistic has no measured floor
#: on that pair and is quoted WITHOUT a sufficiency claim.
FLOOR = {"ADE": 0.0607, "LONspd": 0.0038, "LONacc": 0.0061, "LAThead": 1.1458,
         "LATyaw": 0.5771, "LATcurv": 0.00066, "LATcross": 0.0710,
         "TAClatK": 0.0973, "TACm5": 0.0500, "TACgFDE": 0.3440,
         "recL": 0.0000, "recR": 0.0000, "recLK": 0.1429,
         "kamm": 0.03704, "maxk": 0.0000, "maxa": 0.11418}


def dig(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return d


def feas(dumpdir, key="cl"):
    out = []
    for f in sorted(glob.glob(dumpdir + "/ep*.npz")):
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        a = z[key]
        if "v0" in z.files:
            a = a[z["v0"] >= VMIN]
        if len(a):
            out.append(a)
    if not out:
        return None
    t = torch.from_numpy(np.concatenate(out, 0)).to(torch.float64)
    t = torch.cat([torch.zeros(t.shape[0], 1, 2, dtype=t.dtype), t], dim=1)
    r = FD.assert_feasible(t, dt=DT)
    r["_n"] = int(t.shape[0])
    return r


def load(tag):
    rec = "%s/rec_%s.json" % (P4, tag)
    dmp = "%s/dump_%s" % (P4, tag)
    if not os.path.exists(rec):
        return None, "record absent (not run, or still running)"
    d = json.load(open(rec, encoding="utf-8"))
    n_ep = len(glob.glob(dmp + "/ep*.npz"))
    if n_ep != 8:
        return None, "dump has %d/8 episode files - a DIFFERENT panel, not paired" % n_ep
    row = {"_arm": d["arm"], "_wk": d["cost"]["weights"]["W_KAPPA"],
           "_mu": dig(d, ("refav1", "manifest", "plan_cfg", "kamm_mu")),
           "_lad": dig(d, ("refav1", "manifest", "goal_rule", "seed_kappa_ladder")),
           "_seed": dig(d, ("refav1", "manifest", "plan_cfg", "seed"))}
    cl = d["arms"]["cl"]
    for name, ks, _ in M:
        row[name] = dig(cl, ks)
    for short, cls in PC:
        row[short] = dig(cl, ("four_families", "tactical", "lateral_decision",
                              "per_class", cls, "recall"))
    r = feas(dmp)
    g = feas(dmp, "g")
    if r:
        row["kamm"], row["maxk"], row["maxa"] = (r["kamm_over_rate"],
                                                 r["max_abs_kappa"], r["max_abs_accel"])
        row["peakg"], row["_nfeas"] = r["peak_g_max"], r["_n"]
    row["_gctl"] = g["kamm_over_rate"] if g else float("nan")
    return row, None


def fmt(v, nd=4):
    try:
        return ("%%.%df" % nd) % float(v)
    except (TypeError, ValueError):
        return "  --  "


def main():
    rows, missing = {}, {}
    for tag, wk, cap, lad in CELLS:
        r, why = load(tag)
        if r is None:
            missing[tag] = why
        else:
            rows[tag] = r

    print("=" * 118)
    print("THE 2x2x2 FACTORIAL: W_KAPPA (0 / 15.11245) x kamm cap (off / mu=0.7) x "
          "seed-kappa ladder (off / on)")
    print("refav1 ckpt_ep3 step 21109, p4 panel, 40 windows / 8 episodes, T1 for `cl`; "
          "feasibility n = 27 at v0 >= 2 m/s")
    print("=" * 118)
    print()
    hdr = ("cell  W  C  L | %-13s | " % "arm") + " ".join("%-8s" % n for n, _, _ in M)
    print(hdr)
    print("-" * len(hdr))
    for i, (tag, wk, cap, lad) in enumerate(CELLS, 1):
        if tag not in rows:
            print("%-4d  %d  %d  %d | %-13s | ABSENT: %s" % (i, wk, cap, lad, tag,
                                                             missing[tag]))
            continue
        r = rows[tag]
        print("%-4d  %d  %d  %d | %-13s | " % (i, wk, cap, lad, tag) +
              " ".join("%-8s" % fmt(r[n], nd) for n, _, nd in M))
    print()
    hdr2 = ("cell  W  C  L | %-13s | %-8s %-8s %-8s | %-8s %-8s %-8s %-8s | %s"
            % ("arm", "recLK", "recL", "recR", "kamm_over", "max|k|", "max|a|",
               "peak_g", "GTctl"))
    print(hdr2)
    print("-" * len(hdr2))
    for i, (tag, wk, cap, lad) in enumerate(CELLS, 1):
        if tag not in rows:
            print("%-4d  %d  %d  %d | %-13s | ABSENT" % (i, wk, cap, lad, tag))
            continue
        r = rows[tag]
        print("%-4d  %d  %d  %d | %-13s | %-8s %-8s %-8s | %-8s %-8s %-8s %-8s | %s"
              % (i, wk, cap, lad, tag, fmt(r["recLK"]), fmt(r["recL"]), fmt(r["recR"]),
                 fmt(r.get("kamm")), fmt(r.get("maxk")), fmt(r.get("maxa"), 3),
                 fmt(r.get("peakg"), 3), fmt(r.get("_gctl"))))
    print()
    print("CONTROL: GTctl is the ground-truth path's kamm_over_rate on the same "
          "windows and MUST read 0.0000 in every row.")
    print("         `recL` = turn_left recall (n_true = 11); `recR` = turn_right "
          "(n_true = 8); `recLK` = lane_keep (n_true = 21).")
    print()

    # ---- provenance: the cell coding must match the RECORD, not the tag ------ #
    print("PROVENANCE CHECK - the cell coding is asserted against each record's own "
          "manifest, never against its name:")
    bad = 0
    for tag, wk, cap, lad in CELLS:
        if tag not in rows:
            continue
        r = rows[tag]
        okw = (float(r["_wk"]) > 0) == bool(wk)
        okc = (r["_mu"] == 0.7) == bool(cap)
        okl = (isinstance(r["_lad"], (list, tuple)) and len(r["_lad"]) == 5) == bool(lad)
        flag = "OK " if (okw and okc and okl) else "MISMATCH"
        bad += 0 if (okw and okc and okl) else 1
        print("   %-8s %-13s W_KAPPA=%-10s kamm_mu=%-6s ladder=%s seed=%s"
              % (flag, tag, r["_wk"], r["_mu"],
                 "5-rung" if okl == bool(lad) and lad else r["_lad"], r["_seed"]))
    print("   %s" % ("ALL CELLS MATCH THEIR RECORDS" if bad == 0 else
                     "%d MISMATCH - the table is NOT a factorial" % bad))
    print()

    # ---- main effects -------------------------------------------------------- #
    print("MAIN EFFECTS (marginal contrasts over the cells present; NOT a model fit).")
    print("Each is the mean over available pairs that differ ONLY in that lever, with")
    print("the per-metric inference-seed floor beside it. |effect| <= floor is NOT an")
    print("effect on this rig.")
    print()
    print("  READ THE COLUMNS `pairs used` AND `floor` BEFORE ANY EFFECT:")
    print("  * `pairs used` < 4 means the factorial is INCOMPLETE and the marginal is")
    print("    taken over a SUBSET of the design -- it is not yet balanced.")
    print("  * every floor comes from ONE seed pair (ccos_argmax vs ccos_seed1,")
    print("    raw/seed_floor_ext_ccos.txt). A floor of 0.00000 therefore means")
    print("    'that pair happened to agree exactly', NOT 'this statistic is noiseless'.")
    print("    `floor=0` verdicts are UNDERPOWERED and any effect resting on one is")
    print("    quotable only when it is large and mechanistically explained -- which")
    print("    is true of recL/recR (0.3636 -> 0.0000 is a whole class disappearing)")
    print("    and is NOT true of a small maxk shift.")
    print("  * these marginals are POINT ESTIMATES with no interval. The decision-grade")
    print("    intervals are the paired episode-cluster bootstrap in pd_final.md.")
    print()
    PAIRS = {"W_KAPPA": [("ccos_argmax", "wk15"), ("kamm07", "best"),
                         ("l3ladder", "wk15_ladder"), ("combined", "bestlad")],
             "cap":     [("ccos_argmax", "kamm07"), ("wk15", "best"),
                         ("l3ladder", "combined"), ("wk15_ladder", "bestlad")],
             "ladder":  [("ccos_argmax", "l3ladder"), ("wk15", "wk15_ladder"),
                         ("kamm07", "combined"), ("best", "bestlad")]}
    names = [n for n, _, _ in M] + [s for s, _ in PC] + ["kamm", "maxk"]
    print("%-9s %-10s %10s %10s  %-9s %s" % ("lever", "metric", "effect", "floor",
                                             "verdict", "pairs used"))
    print("-" * 92)
    for lever, prs in PAIRS.items():
        avail = [(a, b) for a, b in prs if a in rows and b in rows]
        for nm in names:
            ds = []
            for a, b in avail:
                try:
                    ds.append(float(rows[b][nm]) - float(rows[a][nm]))
                except (TypeError, ValueError, KeyError):
                    pass
            if not ds:
                continue
            eff = sum(ds) / len(ds)
            fl = FLOOR.get(nm)
            if fl is None:
                v = "no floor"
            elif abs(eff) <= fl:
                v = "IN FLOOR"
            else:
                v = "%.1fx floor" % (abs(eff) / fl) if fl > 0 else "floor=0"
            print("%-9s %-10s %10.5f %10s  %-9s %d/%d"
                  % (lever, nm, eff, ("%.5f" % fl) if fl is not None else "--", v,
                     len(ds), len(prs)))
        print()
    if missing:
        print("CELLS STILL ABSENT (named, not dropped):")
        for t, w in missing.items():
            print("   %-13s %s" % (t, w))


if __name__ == "__main__":
    main()
