#!/usr/bin/env python3
"""THE SEED TABLE: both frontier curves, per rung, with the ACROSS-INFERENCE-SEED
spread beside the measured seed floors.

⛔ The spread printed here answers "would another INFERENCE RUN say this?" -- the
question a sampling planner actually poses -- and NOT "would another draw of
EPISODES say this?" (the paired bootstrap, banked separately as INDICATIVE) and NOT
"would another TRAINING RUN say this?" (unpriced, one checkpoint).

⛔ THE TURN GATE `|dyaw| > 0.15` IS NEVER USED: the ground truth itself fails it 3
of 9 (D-TURNGATE). The recall is the four-family TACTICAL block's own per-class
lateral recall, `turn_left` against its printed `n_true`.

Floors (D-REFAV1-CG-SEEDFLOOR): curvature 0.00200, ADE 0.1035, heading 1.1458,
turn_left recall EXACTLY 0.00000. Straight floor `ha0` curvature = 0.040083.
ASCII only.
"""
import json
import os
import sys
from collections import defaultdict

P4 = r"C:\Users\Admin\refav1_margin\p4out"
OUT = r"C:\Users\Admin\wkfront\out"
CURV_FLOOR = 0.00200
HA0_STRAIGHT = 0.040083


def find_rec(tag):
    for d in (OUT, P4):
        p = os.path.join(d, "rec_%s.json" % tag)
        if os.path.exists(p):
            return p
    return None


def read(tag):
    p = find_rec(tag)
    if p is None:
        return None
    d = json.load(open(p, encoding="utf-8"))
    cl = d["arms"]["cl"]
    ff = cl["four_families"]
    lat, tac = ff["lateral"], ff["tactical"]
    per = ((tac.get("lateral_decision") or {}).get("per_class") or {})

    def rec_of(name):
        c = per.get(name) or {}
        return c.get("recall"), c.get("n_true")

    iv = (cl.get("intervals", {}).get("metrics") or {}).get("ade_dense_m") or {}
    ha0 = d["arms"].get("ha0", {}).get("four_families", {}).get("lateral", {})
    return {
        "path": p,
        "n_windows": d.get("n_windows"),
        "n_episodes": d.get("n_episodes"),
        "curv": lat.get("curvature_mae_1pm"),
        "head": lat.get("heading_mae_deg"),
        "ade": iv.get("mean"),
        "ade_lo": iv.get("lo"), "ade_hi": iv.get("hi"),
        "tl": rec_of("turn_left"), "tr": rec_of("turn_right"),
        "lk": rec_of("lane_keep"),
        "ha0_curv": ha0.get("curvature_mae_1pm"),
        "wkg": (d.get("cost") or {}).get("w_kappa_by_goal"),
        "wk": ((d.get("cost") or {}).get("weights") or {}).get("W_KAPPA"),
    }


def main():
    # groups: label -> [tags]; the tags are inference-seed replicates of ONE rung
    groups = defaultdict(list)
    for spec in sys.argv[1:]:
        # ⚠️ separator is '|', not '=': rung labels contain '=' (e.g. "W_KAPPA=7")
        # and partitioning on '=' silently COLLAPSED two rungs onto one key.
        label, _, tags = spec.partition("|")
        groups[label] = tags.split(",")
    print("floors: curvature %.5f | turn_left recall 0.00000 (exact) | "
          "ha0 straight floor curvature %.6f" % (CURV_FLOOR, HA0_STRAIGHT))
    print()
    hdr = ("%-18s %-10s %5s %5s | %-9s %-9s | %-16s | %-7s %-7s | %s"
           % ("rung", "arm", "nwin", "neps", "curv", "vs ha0", "turn_L recall",
              "head", "ADE", "record"))
    print(hdr)
    print("-" * len(hdr))
    for label, tags in groups.items():
        curvs, tls = [], []
        for t in tags:
            r = read(t)
            if r is None:
                print("%-18s %-10s  MISSING (not yet written)" % (label, t))
                continue
            f = r["ha0_curv"] if r["ha0_curv"] else HA0_STRAIGHT
            d = r["curv"] - f
            tl, ntl = r["tl"]
            curvs.append(r["curv"])
            if tl is not None:
                tls.append(tl)
            print("%-18s %-10s %5s %5s | %.6f %+.6f | %.4f of %-6s | %7.3f %7.4f | %s"
                  % (label, t, r["n_windows"], r["n_episodes"], r["curv"], d,
                     tl if tl is not None else float("nan"),
                     ntl if ntl is not None else "?", r["head"], r["ade"],
                     os.path.basename(r["path"])))
        if len(curvs) >= 2:
            spread = max(curvs) - min(curvs)
            mean = sum(curvs) / len(curvs)
            marg = HA0_STRAIGHT - mean
            print("   %-15s SEEDS n=%d  curv mean %.6f  SPREAD %.6f (%.2fx the "
                  "0.00200 floor)  margin under straight %+.6f (%.2fx floor)  "
                  "%s" % ("^ across", len(curvs), mean, spread,
                          spread / CURV_FLOOR, marg, marg / CURV_FLOOR,
                          "MARGIN > SPREAD -> holds across seeds"
                          if marg > spread else
                          "SPREAD >= MARGIN -> NOT resolved across seeds"))
            if tls:
                print("   %-15s turn_left recall across seeds: %s  min %.4f  "
                      "%s" % ("^ across", ["%.4f" % x for x in tls], min(tls),
                              "ABOVE the 0.00000 floor at EVERY seed"
                              if min(tls) > 0 else
                              "ZERO at at least one seed -> turn NOT retained"))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
