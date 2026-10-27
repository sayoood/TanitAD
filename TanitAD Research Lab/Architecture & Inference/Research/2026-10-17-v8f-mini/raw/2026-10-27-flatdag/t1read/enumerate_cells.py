"""PREREG_V8F_FLATDAG section 4: is the outcome table COMPLETE over its cells?
Cells = controls failed (2) x BLOCKED (2) x ADE6 DAG - FLAT-DAG state at seed 0 (4) x state at seed 1 (4) x a family row separated-worse (2) = 128.
The per-seed states are exhaustive for one ADE6 row: MET (separated, point <= -2F), SHORT (separated-better, point above -2F), NS (not separated), WORSE (separated-worse).
Each cell must be placed in EXACTLY the row the PREREG's order gives ("Applied in this order: REFUSED -> BLOCKED -> FLAT BETTER -> HIERARCHY BEATS FLAT -> NOT ESTABLISHED, first match wins").
The expected outcome below is written independently of flatdag_read.decide (a flat if-chain over the PREREG's printed conditions); decide() is then run on a real-shaped row for every cell and must agree.
    python enumerate_cells.py            prints the counts; exits 1 if any cell is unplaced, doubly placed in a way that changes the first match, or decide() disagrees with the independent table
"""
import itertools
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("FLATDAG_READ_DIR", HERE))
import flatdag_read as FR  # noqa: E402

TWO_F = 0.2415
ROW = {"MET": {"delta": -0.30, "lo": -0.40, "hi": -0.20, "separated": True},
       "SHORT": {"delta": -0.10, "lo": -0.15, "hi": -0.05, "separated": True},
       "NS": {"delta": -0.30, "lo": -0.60, "hi": 0.05, "separated": False},            # a point past the bar that is NOT separated is not a pass
       "WORSE": {"delta": 0.20, "lo": 0.05, "hi": 0.35, "separated": True}}


def expected(cf, bl, st0, st1, fam):
    """The PREREG's section 4 conditions, in the PREREG's order, from the cell's coordinates."""
    if cf:
        return "REFUSED"
    if bl:
        return "BLOCKED"
    if st0 == "WORSE" or st1 == "WORSE":
        return "FLAT BETTER"
    if st0 == "MET" and st1 == "MET" and not fam:
        return "HIERARCHY BEATS FLAT"
    return "NOT ESTABLISHED"


def cells():
    return list(itertools.product((0, 1), (0, 1), ("MET", "SHORT", "NS", "WORSE"), ("MET", "SHORT", "NS", "WORSE"), (0, 1)))


def main():
    cnt, bad, multi = Counter(), [], 0
    for cf, bl, st0, st1, fam in cells():
        d = FR.decide(["C-F2"] if cf else [], ["BUILD-CUT-LEVER"] if bl else [], {0: dict(ROW[st0]), 1: dict(ROW[st1])}, {0: ["curvature MAE"] if fam else [], 1: []})
        want = expected(cf, bl, st0, st1, fam)
        cnt[d["OUTCOME"]] += 1
        if d["OUTCOME"] != want:
            bad.append(((cf, bl, st0, st1, fam), d["OUTCOME"], want))
        if d["matching_rows_in_table_order"][0] != d["OUTCOME"] or d["matching_rows_in_table_order"][-1] != "NOT ESTABLISHED":
            bad.append(((cf, bl, st0, st1, fam), "first match is not the outcome / NOT ESTABLISHED is not the catch-all", d["matching_rows_in_table_order"]))
        multi += len(d["matching_rows_in_table_order"]) > 1
    print("cells:", len(cells()), "| outcome counts:", dict(cnt))
    print("cells that match more than one row (first match wins, as the PREREG orders):", multi)
    print("unplaced cells:", cnt.get("UNPLACED", 0), "| disagreements with the independent table:", len(bad))
    for b in bad[:10]:
        print("  DISAGREE", b)
    every_row_reached = set(FR.OUTCOME_NAMES) == set(cnt)
    print("every table row reached by at least one cell:", every_row_reached, sorted(set(FR.OUTCOME_NAMES) - set(cnt)))
    return 1 if (bad or cnt.get("UNPLACED") or not every_row_reached) else 0


if __name__ == "__main__":
    sys.exit(main())
