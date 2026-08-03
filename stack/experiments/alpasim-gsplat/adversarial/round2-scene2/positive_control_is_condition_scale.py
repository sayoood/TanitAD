#!/usr/bin/env python3
"""The scene-2 panel's negative control is CROSS-ARM. That is the wrong control.

`objects vs empty` asks whether ONE policy changes when the render changes. Showing that
two DIFFERENT policies separate (flagship minus refc, effect ~2.8 m ADE) does not show
that the panel can see a *condition* effect on a *fixed* policy, which is a far smaller
signal. The right control is condition-vs-empty on a scene where a reaction is known to
be producible — and it already exists in the banked scene-1 panel, where a CONSTRUCTED
lead was attached at 25 / 15 / 8 m and `behind`.

This script counts, per (arm, condition), how many of the same paired metrics separated
against the same empty control. It also reports the duplicate-metric and circular-metric
adjustments to the scene-2 counts.
"""
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve()
GS = HERE.parents[2]


def sep_counts(paired):
    scored = {k: v for k, v in paired.items() if isinstance(v, dict) and "separated" in v}
    return scored, [k for k, v in scored.items() if v["separated"]]


def dup_pairs(scored):
    out = []
    for a, b in itertools.combinations(scored, 2):
        va, vb = scored[a], scored[b]
        if (abs(va["delta"] - vb["delta"]) < 1e-9 and abs(va["lo"] - vb["lo"]) < 1e-9
                and abs(va["hi"] - vb["hi"]) < 1e-9):
            out.append((a, b))
    return out


def main():
    print("== SCENE 1 (00040136): CONDITION vs EMPTY, the control that matches the design ==")
    p1 = json.loads((GS / "results" / "cutin" / "CUTIN_PANEL.json").read_text())
    for arm, conds in p1["arms"].items():
        for cond, blk in conds.items():
            pv = blk.get("paired_vs_empty")
            if not pv:
                continue
            sc, sep = sep_counts(pv)
            print(f"  {arm:12s}/{cond:7s}  {len(sep):2d}/{len(sc)}  {sep if sep else ''}")
    print()
    print("== SCENE 2 (7c72937c): the four contrasts, with the denominator audited ==")
    for f in sorted((GS / "results" / "scene2-realclose" / "metrics").glob("*_vs_*.json")):
        d = json.loads(f.read_text())
        pr = d.get("paired_A_minus_B")
        if pr is None:
            continue
        sc, sep = sep_counts(pr)
        dups = dup_pairs(sc)
        circ = [k for k in sep if k == "route_head_eq_logged"]
        n_adj = len(sc) - len(dups) - len(circ)
        s_adj = len(sep) - sum(1 for a, b in dups if a in sep and b in sep) - len(circ)
        print(f"  {f.name}")
        print(f"     as reported {len(sep)}/{len(sc)}   duplicates={dups}   circular_counted={circ}")
        print(f"     DISTINCT, NON-CIRCULAR: {s_adj}/{n_adj}")


if __name__ == "__main__":
    main()
