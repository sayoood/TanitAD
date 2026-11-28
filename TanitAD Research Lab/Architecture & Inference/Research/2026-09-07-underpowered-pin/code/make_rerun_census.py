"""Simulate the census AFTER a re-run against the corrected frozenset.

vocab_fill_census.py line 78 is  `t in UNDER or n < 30`, so once MERGE (84) and
TAKE_EXIT_R (133) join TACTICAL_GOAL_UNDERPOWERED their status flips
HEALTHY -> UNDERPOWERED even though their counts did not move. This file proves
the circularity test tolerates that (it is expected) while still refusing an
unexplained divergence.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../2026-09-07-underpowered-pin/code
BUNDLE = os.path.dirname(HERE)
src = os.path.join(BUNDLE, "raw", "vocab_fill_census.json")
dst = os.path.join(BUNDLE, "raw", "vocab_fill_census.SIMULATED_RERUN.json")

CORRECTED = {
    "TAKE_EXIT_R", "MERGE", "TRAFFIC_LIGHT_REACT_YELLOW", "YIELD_FOR_TURN_L",
    "LANE_CHANGE_L", "OVERTAKE_VEHICLE", "TAKE_EXIT_L", "YIELD_FOR_TURN_R",
    "TRAFFIC_LIGHT_REACT", "LANE_CHANGE_R",
}

with io.open(src, encoding="utf-8") as fh:
    doc = json.load(fh)

toks = doc["fields"]["g_tac.goals"]["tokens"]
flipped = []
for t, rec in toks.items():
    # reproduce line 78 exactly, with the CORRECTED set as UNDER
    want = "UNDERPOWERED" if (t in CORRECTED or rec["n"] < 30) else "HEALTHY"
    if rec["n"] and rec["status"] != want:
        flipped.append((t, rec["n"], rec["status"], want))
        rec["status"] = want

if not flipped:
    print("NO STATUS FLIPPED -- the simulation is inert, it proves nothing")
    sys.exit(2)

with io.open(dst, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=1)

print("status flips a re-run would produce (counts UNCHANGED):")
for t, n, was, now in flipped:
    print("  %-28s n=%-5d %s -> %s" % (t, n, was, now))
print("wrote", dst)
