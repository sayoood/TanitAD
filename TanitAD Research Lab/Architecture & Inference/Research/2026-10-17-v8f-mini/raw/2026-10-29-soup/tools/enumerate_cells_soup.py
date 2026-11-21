#!/usr/bin/env python3
"""PREREG_V8F_SOUP section 10: the outcome table is complete and exclusive.
(1) every assignment of the composed booleans (controls x {H, MV, AD, DM} x 2 seeds = 512 cells) matches exactly one row;
(2) 200,000 seeded random PAIR-level assignments (5 guards, R-MOVE, R-KEEP, R-NOBASE per pair and seed; Read A D1 / D2 / D3 per decisive soup and seed)
    composed by the reader's own per_seed_booleans each match exactly one row, and every row is reached;
(3) the per-seed AND: a miss in either pair (or on either soup, for DM) is a miss at the seed.  Exit 1 on any failure."""
import itertools
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "read"))
import soup_read as R  # noqa: E402

import numpy as np  # noqa: E402

fail = 0
cnt = Counter()
for ctrl in (True, False):
    for bits in itertools.product((False, True), repeat=8):
        b = {s: dict(zip(("H", "MV", "AD", "DM"), bits[4 * s:4 * s + 4])) for s in R.SEEDS}
        d = R.decide(ctrl, b)
        if len(d["matching_rows_in_table_order"]) != 1:
            fail += 1
        cnt[d["OUTCOME"]] += 1
print("(1) composed cells", sum(cnt.values()), dict(cnt), "multi/none", fail)
rng = np.random.default_rng(0)
cnt2, fail2 = Counter(), 0
for _ in range(200000):
    pairs, dm = {}, {}
    for s in R.SEEDS:
        pairs[s] = {}
        for p in R.PAIRS:
            g = {k: bool(rng.random() < 0.9) for k in R.GUARDS}
            f = {k: bool(rng.random() < 0.5) for k in ("R-MOVE", "R-KEEP", "R-NOBASE", "R-CRAWL")}
            pairs[s][p] = {"guards": g, "flags": f}
        dm[s] = {x: bool(all(rng.random() < 0.8 for _ in ("D1", "D2", "D3"))) for x in ("SOUP", "SOUPR")}
    d = R.decide(bool(rng.random() < 0.95), R.per_seed_booleans(pairs, dm))
    if len(d["matching_rows_in_table_order"]) != 1:
        fail2 += 1
    cnt2[d["OUTCOME"]] += 1
print("(2) random pair-level", sum(cnt2.values()), dict(cnt2), "multi/none", fail2, "all rows reached", set(cnt2) == set(R.OUTCOME_NAMES))
ok3 = 0
base = {"guards": {g: True for g in R.GUARDS}, "flags": {"R-MOVE": True, "R-KEEP": True, "R-NOBASE": True, "R-CRAWL": True}}
for p in R.PAIRS:
    for k in ("R-MOVE", "R-KEEP", "R-NOBASE"):
        pr = {s: {q: base for q in R.PAIRS} for s in R.SEEDS}
        pr[0] = {q: ({"guards": base["guards"], "flags": dict(base["flags"], **{k: False})} if q == p else base) for q in R.PAIRS}
        b = R.per_seed_booleans(pr, {s: {"SOUP": True, "SOUPR": True} for s in R.SEEDS})
        ok3 += int(not (b[0]["MV"] and b[0]["AD"]) and b[1]["MV"] and b[1]["AD"])
for x in ("SOUP", "SOUPR"):
    b = R.per_seed_booleans({s: {q: base for q in R.PAIRS} for s in R.SEEDS}, {0: {"SOUP": x != "SOUP", "SOUPR": x != "SOUPR"}, 1: {"SOUP": True, "SOUPR": True}})
    ok3 += int(not b[0]["DM"] and b[1]["DM"])
print("(3) per-seed AND cases", ok3, "of 8")
bad = fail or fail2 or set(cnt2) != set(R.OUTCOME_NAMES) or set(cnt) != set(R.OUTCOME_NAMES) or ok3 != 8
print("ENUMERATE-SOUP", "FAIL" if bad else "OK")
sys.exit(1 if bad else 0)
