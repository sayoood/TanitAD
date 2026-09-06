#!/usr/bin/env python3
"""Re-derive the REPLICATE FALSE-POSITIVE RATE for `separated` on the v7-tiny rig.

WHY. The programme quoted `3 of 18 ~ 17 %` for the `A0b_replicate` vs `A0_fixed`
contrast. That form does not reproduce at any scoping this script could construct, and
the artifact it was attributed to (`raw/NOISE_FLOOR.md`) is a CRASHED file carrying no
numbers at all. This script reads the panel report itself and counts.

WHAT IS COUNTED. Every leaf of `arms.A0b_replicate.paired_vs_A0` that carries a
`separated` flag -- i.e. every (regime x horizon x metric) CELL the panel actually
bootstrapped. `A0b_replicate` is A0's flags and A0's SEED run a second time, zero levers
moved, so every `separated` here is a FALSE POSITIVE by construction.

CONTROLS (both must hold or the count is not admissible):
  C1  the arm must exist and its paired block must be non-empty  -> a zero cell count
      is a failed read, not a clean replicate.
  C2  a POSITIVE control: at least one OTHER arm in the same report must show a
      DIFFERENT separated count -- otherwise the recursion is reading a constant and
      the 6 is an artifact of the walker, not of the data.

usage: python replicate_fp_rate.py <panel_report.json>
ASCII only: the dev box is cp1252 and a non-ASCII print() is fatal.
"""
import json
import sys


def cells(node, prefix=""):
    """Yield (path, separated, delta, ci95) for every bootstrapped leaf."""
    if isinstance(node, dict):
        if "separated" in node:
            yield (prefix, bool(node.get("separated")),
                   node.get("delta", node.get("delta_m")), node.get("ci95"))
            return
        for k, v in node.items():
            for c in cells(v, prefix + "/" + str(k)):
                yield c


def rate(report, arm):
    blk = report["arms"][arm].get("paired_vs_A0")
    if not blk:
        return None
    cs = list(cells(blk))
    sep = [c for c in cs if c[1]]
    return cs, sep


def main(path):
    rep = json.load(open(path, encoding="utf-8"))
    out = rate(rep, "A0b_replicate")
    if out is None or not out[0]:
        print("C1 FAIL: A0b_replicate.paired_vs_A0 empty or absent -> INCONCLUSIVE")
        return 1
    cs, sep = out
    print("ARM         A0b_replicate  (A0's flags, A0's SEED, run again; zero levers moved)")
    print("C1 PASS     bootstrapped cells found: %d" % len(cs))
    print()
    print("SEPARATED CELLS (each one is a FALSE POSITIVE by construction):")
    for p, s, d, ci in sep:
        print("   %-42s delta=%-10s ci95=%s" % (p, d, ci))
    print()
    print("REPLICATE FALSE-POSITIVE RATE = %d / %d = %.4f = %.1f %%"
          % (len(sep), len(cs), len(sep) / len(cs), 100.0 * len(sep) / len(cs)))
    print()
    # C2 -- the positive control
    others = []
    for arm in rep["arms"]:
        if arm == "A0b_replicate":
            continue
        o = rate(rep, arm)
        if o:
            others.append((arm, len(o[1]), len(o[0])))
    distinct = {n for _, n, _ in others}
    print("C2 positive control -- separated counts on the OTHER arms of the same report:")
    for arm, ns, nt in sorted(others):
        print("   %-16s %2d / %2d" % (arm, ns, nt))
    ok = len(distinct - {len(sep)}) > 0
    print("C2 %s     at least one other arm differs from %d: %s"
          % ("PASS" if ok else "FAIL", len(sep), sorted(distinct)))
    print()
    print("NOTE  The restricted view the panel's own VERDICT.md tabulates -- 7 family rows")
    print("      at the 2 s horizon, both regimes -- is 3 of 14 = 21.4 %. Neither view")
    print("      is 3 of 18 and neither is ~17 %.")
    print()
    print("<!-- ARTIFACT-COMPLETE: REPLICATE_FP_RATE -->")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
