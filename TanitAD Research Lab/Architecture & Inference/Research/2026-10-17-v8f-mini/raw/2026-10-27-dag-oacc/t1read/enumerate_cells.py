"""PREREG_V8F_DAG_OACC section 4 AS AMENDED BY AMENDMENT 1 (A1-2026-10-27): is the outcome table COMPLETE over its cells, and does every cell map to EXACTLY ONE outcome?
Cells = controls failed (2) x, for EACH of the two seeds, nine independent coordinates (512 per seed):
    PA        PRIMARY-ACC passes (accel MAE, DAG-OACC - DAG, < 0 and separated)
    KEEP      KEEP passes (ADE6 DAG-OACC - DAG, upper bound <= +F/2)
    D1, D3    Read A's D1 (ADE6 separated-better, <= -2F) and D3 (each family has a separated-better row)
    LON       a LONGITUDINAL row of DAG-OACC - P10 is separated-worse (so LON-CLEAR fails)
    LAT       a LATERAL row (heading, yaw-rate, curvature, cross-track, L1, L2) is separated-worse
    TAC, STR  a TACTICAL / a STRATEGIC row is separated-worse
    M1        the crawl share is separated-worse (outside D2)
D2 is DERIVED (no non-crawl row separated-worse: not LON, LAT, TAC, STR), as build_report.verdict defines it; L2 is a LATERAL row, so it sits inside LAT and inside D2.
Each cell must be placed in EXACTLY the row the table gives ("Applied in this order: REFUSED -> NO EFFECT -> COSTS POSITION -> DOMINANT -> LONGITUDINAL FIXED -> NOT FIXED", first match wins).
expected() is the AMENDED table (Amendment 1), expected_unamended() is the PREREG's table as printed before the amendment; both are written independently of dagoacc_read.decide (flat if-chains over the printed conditions on the
RAW coordinates).  decide() is then run on the flags seed_flags_of would derive, for every cell, and must agree with expected(); decide_unamended() must agree with expected_unamended() (it is the descriptive beside field).
    python enumerate_cells.py     prints the counts.  rc 0: complete (0 unplaced, every row reached, exactly one row per non-REFUSED cell) and agreeing | rc 1: decide() disagrees with the independent table, or a row is
    never reached, or a non-REFUSED cell matches two rows (a BUG) | rc 3: decide() AGREES but cells are unplaced (the table is not complete: it was before Amendment 1, with 255)
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
    """The section-4 table AS AMENDED (Amendment 1), in its order, from the cell's RAW coordinates.  UNPLACED = no condition applies."""
    if cf:
        return "REFUSED"
    if not (c0["PA"] and c1["PA"]):
        return "NO EFFECT"
    if not (c0["KEEP"] and c1["KEEP"]):
        return "COSTS POSITION"
    clean = lambda c: not (c["LON"] or c["LAT"] or c["TAC"] or c["STR"])                    # noqa: E731  D2
    crawl_pass = not (c0["M1"] or c1["M1"])                                                 # L2 worse is a LAT row, so it already breaks clean(); M1 is the part outside D2
    if all(c["D1"] and c["D3"] and clean(c) for c in (c0, c1)) and crawl_pass:
        return "DOMINANT"
    if not (c0["LON"] or c1["LON"]):                                                        # LON-CLEAR at both seeds, and (the branch above) not DOMINANT
        return "LONGITUDINAL FIXED"
    return "NOT FIXED"                                                                      # PRIMARY-ACC and KEEP pass; a longitudinal row worse at >= 1 seed


def expected_unamended(cf, c0, c1):
    """The PREREG's section 4 as printed BEFORE Amendment 1 (default reading: LF needs a lateral or crawl row worse at >= 1 seed; DOMINANT is Read A's D1-D3 alone).  UNPLACED = no condition applies."""
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


def not_dominant_reason(c0, c1):
    """For a LONGITUDINAL FIXED cell: which of DOMINANT's clauses fail at either seed (D1 short / D3 short / a LATERAL row worse / a TACTICAL or STRATEGIC row worse / the crawl share M1 worse)."""
    r = []
    if not (c0["D1"] and c1["D1"]):
        r.append("D1 short")
    if not (c0["D3"] and c1["D3"]):
        r.append("D3 short")
    if c0["LAT"] or c1["LAT"]:
        r.append("a LATERAL row worse")
    if c0["TAC"] or c1["TAC"] or c0["STR"] or c1["STR"]:
        r.append("a TACTICAL / STRATEGIC row worse")
    if c0["M1"] or c1["M1"]:
        r.append("crawl open (M1 worse)")
    return " + ".join(r)


def proj(c):
    """Everything expected() and expected_unamended() read from a seed's raw coordinates (so a group of cells with the same derived flags is interchangeable for them)."""
    return (c["PA"], c["KEEP"], c["D1"], c["D3"], not (c["LON"] or c["LAT"] or c["TAC"] or c["STR"]), not c["LON"], c["LAT"] or c["M1"], c["M1"])


def main():
    cells = seed_cells()
    groups = {}                                           # derived decide() input -> raw cells that give it (decide sees only the flags and M1)
    for c in cells:
        f = flags_of(c)
        groups.setdefault((tuple(sorted(f.items())), c["M1"]), []).append(c)
    inv_bad = [k for k, g in groups.items() if len({proj(c) for c in g}) != 1]
    glist = list(groups.items())
    cnt, bad, multi_refused, multi_other, n = Counter(), [], 0, 0, 0
    reasons, changed, former_unplaced, former_unplaced_now = Counter(), Counter(), 0, Counter()
    for cf in (False, True):
        for k0, g0 in glist:
            for k1, g1 in glist:
                w = len(g0) * len(g1)
                n += w
                c0, c1 = g0[0], g1[0]
                sf = {0: dict(k0[0]), 1: dict(k1[0])}
                m1 = {0: k0[1], 1: k1[1]}
                fail = ["C-O1"] if cf else []
                d = DRD.decide(fail, sf, m1)
                want = expected(cf, c0, c1)
                cnt[d["OUTCOME"]] += w
                if d["OUTCOME"] != want:
                    bad.append(((cf, k0, k1), d["OUTCOME"], want))
                    continue
                ms = d["matching_rows_in_table_order"]
                if (ms[0] if ms else "UNPLACED") != d["OUTCOME"]:
                    bad.append(((cf, k0, k1), "first match is not the outcome", ms))
                if len(ms) > 1:
                    if cf:
                        multi_refused += w                  # REFUSED wins by the order: the controls are checked before the outcome
                    else:
                        multi_other += w
                        bad.append(((cf, k0, k1), "a non-REFUSED cell matches more than one row", ms))
                un = DRD.decide_unamended(fail, sf, m1)
                want_u = expected_unamended(cf, c0, c1)
                if un["OUTCOME"] != want_u:
                    bad.append(((cf, k0, k1), "decide_unamended disagrees with the independent unamended table", (un["OUTCOME"], want_u)))
                if not cf:
                    if un["OUTCOME"] != d["OUTCOME"]:
                        changed[(un["OUTCOME"], d["OUTCOME"])] += w
                    if un["OUTCOME"] == "UNPLACED":
                        former_unplaced += w
                        former_unplaced_now[d["OUTCOME"]] += w
    for r0 in cells:                                       # the "why not DOMINANT" breakdown needs the RAW coordinates (the derived flags merge LAT / TAC / STR), so it walks every raw pair of the independent table
        for r1 in cells:
            if expected(False, r0, r1) == "LONGITUDINAL FIXED":
                rs = not_dominant_reason(r0, r1).split(" + ")
                for x in rs:
                    reasons[x] += 1
                reasons["(exactly one reason: " + rs[0] + ")" if len(rs) == 1 else "(two or more reasons)"] += 1
    unplaced = cnt.get("UNPLACED", 0)
    print("cells:", n, "| outcome counts:", dict(cnt), f"| {len(glist)} distinct decide() inputs per seed (the other raw coordinates give identical flags: expected() is checked to read nothing beyond them: {not inv_bad})")
    print("non-REFUSED cells that match more than one row (the rows are mutually exclusive once a control has not failed):", multi_other, "| REFUSED cells that also match another row (REFUSED wins by the order):", multi_refused)
    print("unplaced cells:", unplaced, "| disagreements with the independent table:", len(bad) + len(inv_bad))
    for b in bad[:10]:
        print("  DISAGREE", b)
    print(f"the gap before Amendment 1: {former_unplaced} cells were UNPLACED in the unamended table; now:", dict(former_unplaced_now))
    print("cells whose word changes, unamended -> amended (descriptive; the unamended word decides nothing):", sum(changed.values()), {f"{a} -> {b}": v for (a, b), v in sorted(changed.items())})
    print("LONGITUDINAL FIXED cells by which of DOMINANT's clauses fail at either seed (descriptive; a cell can fail several):")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   {v:7d}  {k}")
    reached = set(cnt) - {"UNPLACED"}
    print("every table row reached by at least one cell:", set(DRD.OUTCOME_NAMES) == reached, sorted(set(DRD.OUTCOME_NAMES) - reached))
    if bad or inv_bad or multi_other or set(DRD.OUTCOME_NAMES) != reached:
        return 1
    return 3 if unplaced else 0


if __name__ == "__main__":
    sys.exit(main())
