"""PREREG_V8F_DAG_OACC section 4: is the outcome table COMPLETE over its cells, and does every cell map to EXACTLY ONE outcome?
Cells = controls failed (2) x, for EACH of the two seeds, nine independent coordinates (512 per seed):
    PA        PRIMARY-ACC passes (accel MAE, DAG-OACC - DAG, < 0 and separated)
    KEEP      KEEP passes (ADE6 DAG-OACC - DAG, upper bound <= +F/2)
    D1, D3    Read A's D1 (ADE6 separated-better, <= -2F) and D3 (each family has a separated-better row)
    LON       a LONGITUDINAL row of DAG-OACC - P10 is separated-worse (so LON-CLEAR fails)
    LAT       a LATERAL row (heading, yaw-rate, curvature, cross-track, L1, L2) is separated-worse
    TAC, STR  a TACTICAL / a STRATEGIC row is separated-worse
    M1        the crawl share is separated-worse (outside D2)
D2 is DERIVED (no non-crawl row separated-worse: not LON, LAT, TAC, STR), as build_report.verdict defines it.
Each cell must be placed in EXACTLY the row the PREREG's order gives ("Applied in this order: REFUSED -> NO EFFECT -> COSTS POSITION -> DOMINANT -> LONGITUDINAL FIXED -> NOT FIXED", first match wins).
The expected outcome below is written independently of dagoacc_read.decide (a flat if-chain over the PREREG's printed conditions on the RAW coordinates); decide() is then run on the flags
seed_flags_of would derive, for every cell, and must agree.
    python enumerate_cells.py     prints the counts.  rc 0: complete and agreeing | rc 1: decide() disagrees with the independent table (a BUG) | rc 3: decide() AGREES but the PREREG table leaves cells unplaced (the PREREG gap).
"""
import itertools
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("DAGOACC_READ_DIR", HERE))
import dagoacc_read as DRD  # noqa: E402

COORDS = ("PA", "KEEP", "D1", "D3", "LON", "LAT", "TAC", "STR", "M1")


def seed_cells():
    return [dict(zip(COORDS, bits)) for bits in itertools.product((False, True), repeat=len(COORDS))]


def flags_of(c):
    """What seed_flags_of derives from rows with these coordinates (D2: no non-crawl row separated-worse; LON-CLEAR: no longitudinal row worse; lateral-or-crawl: LAT or M1)."""
    return {"PA": c["PA"], "KEEP": c["KEEP"], "D1": c["D1"], "D2": not (c["LON"] or c["LAT"] or c["TAC"] or c["STR"]), "D3": c["D3"], "LON_CLEAR": not c["LON"], "LATCRAWL_WORSE": c["LAT"] or c["M1"]}


def expected(cf, c0, c1):
    """The PREREG's section 4 conditions, in the PREREG's order, from the cell's RAW coordinates.  UNPLACED = no condition applies."""
    if cf:
        return "REFUSED"
    if not (c0["PA"] and c1["PA"]):
        return "NO EFFECT"
    if not (c0["KEEP"] and c1["KEEP"]):
        return "COSTS POSITION"
    clean = lambda c: not (c["LON"] or c["LAT"] or c["TAC"] or c["STR"])                    # noqa: E731  D2
    if all(c["D1"] and c["D3"] and clean(c) for c in (c0, c1)):
        return "DOMINANT"
    if not (c0["LON"] or c1["LON"]) and (c0["LAT"] or c0["M1"] or c1["LAT"] or c1["M1"]):
        return "LONGITUDINAL FIXED"
    if c0["LON"] or c1["LON"]:
        return "NOT FIXED"
    return "UNPLACED"


def gap_reason(c0, c1):
    """For an UNPLACED cell: which of the not-DOMINANT reasons hold (D1 short / D3 short / a TACTICAL or STRATEGIC row worse) at either seed."""
    r = []
    if not (c0["D1"] and c1["D1"]):
        r.append("D1 short")
    if not (c0["D3"] and c1["D3"]):
        r.append("D3 short")
    if c0["TAC"] or c1["TAC"] or c0["STR"] or c1["STR"]:
        r.append("a TACTICAL / STRATEGIC row separated-worse")
    return " + ".join(r)


def proj(c):
    """Everything expected() reads from a seed's raw coordinates (so a group of cells with the same derived flags is interchangeable for it)."""
    return (c["PA"], c["KEEP"], c["D1"], c["D3"], not (c["LON"] or c["LAT"] or c["TAC"] or c["STR"]), not c["LON"], c["LAT"] or c["M1"])


def main():
    cells = seed_cells()
    groups = {}                                           # derived decide() input -> raw cells that give it (decide sees only the flags and M1)
    for c in cells:
        f = flags_of(c)
        groups.setdefault((tuple(sorted(f.items())), c["M1"]), []).append(c)
    inv_bad = [k for k, g in groups.items() if len({proj(c) for c in g}) != 1]
    glist = list(groups.items())
    cnt, bad, multi, gap, alt_diff, n = Counter(), [], 0, Counter(), 0, 0
    for cf in (False, True):
        for k0, g0 in glist:
            for k1, g1 in glist:
                w = len(g0) * len(g1)
                n += w
                c0, c1 = g0[0], g1[0]
                sf = {0: dict(k0[0]), 1: dict(k1[0])}
                m1 = {0: k0[1], 1: k1[1]}
                d = DRD.decide(["C-O1"] if cf else [], sf, None, m1)
                want = expected(cf, c0, c1)
                cnt[d["OUTCOME"]] += w
                if d["OUTCOME"] != want:
                    bad.append(((cf, k0, k1), d["OUTCOME"], want))
                    continue
                ms = d["matching_rows_in_table_order"]
                if (ms[0] if ms else "UNPLACED") != d["OUTCOME"]:
                    bad.append(((cf, k0, k1), "first match is not the outcome", ms))
                multi += w * (len(ms) > 1)
                if d["OUTCOME"] == "UNPLACED":
                    gap[gap_reason(c0, c1)] += w
                if not cf:
                    for rd in ({"lf": "both"}, {"dom_crawl": True}):
                        if DRD.decide([], sf, rd, m1)["OUTCOME"] != d["OUTCOME"]:
                            alt_diff += w
                            break
    unplaced = cnt.get("UNPLACED", 0)
    print("cells:", n, "| outcome counts:", dict(cnt), f"| {len(glist)} distinct decide() inputs per seed (the other raw coordinates give identical flags: expected() is checked to read nothing beyond them: {not inv_bad})")
    print("cells that match more than one row (first match wins, as the PREREG orders):", multi)
    print("unplaced cells:", unplaced, "| disagreements with the independent table:", len(bad) + len(inv_bad))
    for b in bad[:10]:
        print("  DISAGREE", b)
    if gap:
        print("UNPLACED cells by reason (PRIMARY-ACC and KEEP pass at both seeds, LON-CLEAR at both, no lateral / crawl row worse, NOT DOMINANT):")
        for k, v in sorted(gap.items(), key=lambda kv: -kv[1]):
            print(f"   {v:7d}  {k}")
    print("cells whose outcome changes under an alternative reading of (a) LF at both seeds or (b) DOMINANT needing the crawl rows to pass (descriptive):", alt_diff)
    reached = set(cnt) - {"UNPLACED"}
    print("every table row reached by at least one cell:", set(DRD.OUTCOME_NAMES) == reached, sorted(set(DRD.OUTCOME_NAMES) - reached))
    if bad or inv_bad or set(DRD.OUTCOME_NAMES) != reached:
        return 1
    return 3 if unplaced else 0


if __name__ == "__main__":
    sys.exit(main())
