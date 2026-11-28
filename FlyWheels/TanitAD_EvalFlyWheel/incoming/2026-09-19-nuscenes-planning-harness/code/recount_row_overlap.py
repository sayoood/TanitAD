"""How much of W6's nuScenes patch was ACTUALLY re-read by W4 - the instrument behind RESULT.md F12.

⛔ WHY THIS EXISTS. W6 reported "15 of my rows share a (paper, table) with rows W4 extracted, and
the page agrees 15/15, 0 disagreements" and offered it as the independent control its own pypdf
token check could not be. W4 could not reproduce it. This script re-derives every count, and the
retraction is F12 in RESULT.md.

Three keys, in increasing strength. Only the third would be a re-read of the same measurement:

  (1) LOOSE  (paper, table with the sub-table parenthetical stripped)  -- what W6 computed
  (2) EXACT  (paper, table as printed in the row)
  (3) CELL   (paper, page, system)          <- the only one that double-reads a NUMBER
  (4) VALUE  (paper, the full value-set)    <- stronger still: the same numbers, read twice

⚠️ TWO TRAPS, BOTH LIVE IN THIS FILE'S HISTORY:

  * **Key loosening.** `'Tab. 1 (ID-3)'.split(' (')[0]` -> `'Tab. 1'` merges distinct sub-tables,
    which inflated the exactly-shared count 5 -> 15. A join key normalised for convenience became
    the claim's denominator.
  * **Self-agreement after the merge.** W4 has since MERGED the 39 rows, so the reference file now
    CONTAINS them: run unfiltered, this script reads 39/39 and every count above looks perfect.
    That is W6's rows compared with themselves. The filter below is therefore load-bearing, and
    both numbers are printed so the trap is visible rather than avoided silently.

    python recount_row_overlap.py
"""
import json

W4_PATH = "D:/Projects/TanitAD/products/P7-TanitEval/benchmarks/published_results.json"
MINE_PATH = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
             "2026-09-19-nuscenes-planning-harness/code/published_results_nuscenes.patch.json")

W4 = json.load(open(W4_PATH, encoding="utf-8"))["results"]
MINE = json.load(open(MINE_PATH, encoding="utf-8"))["rows"]
MY_IDS = {r["id"] for r in MINE}

merged = sum(1 for r in W4 if r["id"] in MY_IDS)
print("rows in W4's file:", len(W4), "| of which are MINE (already merged):", merged)
w4 = [r for r in W4 if r.get("source") and r["source"].get("library_key")
      and r["id"] not in MY_IDS]
print("W4's OWN rows used as the reference:", len(w4))
if merged and not w4:
    raise SystemExit("refusing: nothing left to compare against")


def strip(t):
    return t.split(" (")[0]


def shared(keyfn):
    ref = {}
    for r in w4:
        ref.setdefault(keyfn(r), set()).add(r["source"]["page"])
    hit = [r for r in MINE if keyfn(r) in ref]
    agree = [r for r in hit if r["source"]["page"] in ref[keyfn(r)]]
    return hit, agree


loose_hit, loose_ok = shared(lambda r: (r["source"]["library_key"], strip(r["source"]["table"])))
exact_hit, exact_ok = shared(lambda r: (r["source"]["library_key"], r["source"]["table"]))
cells = {(r["source"]["library_key"], r["source"]["page"], r["system"]) for r in w4}
cell_hit = [r["id"] for r in MINE
            if (r["source"]["library_key"], r["source"]["page"], r["system"]) in cells]
vals = {(r["source"]["library_key"], json.dumps(r.get("values"), sort_keys=True)) for r in w4}
val_hit = [r["id"] for r in MINE
           if (r["source"]["library_key"], json.dumps(r.get("values"), sort_keys=True)) in vals]
papers = {r["source"]["library_key"] for r in w4}

print("(1) LOOSE (paper, table-family):  %2d shared, %2d agree on page" % (len(loose_hit), len(loose_ok)))
print("(2) EXACT (paper, table):         %2d shared, %2d agree on page  -- tables: %s"
      % (len(exact_hit), len(exact_ok),
         sorted({(r["source"]["library_key"], r["source"]["table"]) for r in exact_hit})))
print("(3) CELL  (paper, page, system):  %2d double-read  %s" % (len(cell_hit), cell_hit))
print("(4) VALUE (paper, value-set):     %2d double-read  %s" % (len(val_hit), val_hit))
print("    my rows on a paper W4 also cites: %d of %d"
      % (sum(1 for r in MINE if r["source"]["library_key"] in papers), len(MINE)))
print()
print("VERDICT: a page number is fixed by the paper's LAYOUT, so (1) and (2) agreeing says the two")
print("of us opened the same page for the same table. (3) and (4) are what would make it a re-read")
print("of a NUMBER, and both are 0. ⇒ consistency check, NOT an independent verification.")
