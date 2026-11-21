"""D-LAT-AGREE: decompose the lateral disagreements BY CAUSE, with counts.

PI: "regarding the lateral contradiction, give it to the data flywheel to check
it and fix it BEFORE WE MOVE." Blocker on v7f / refc_v3 / refa_v1 / refd.

Two separate questions, and conflating them is what made 41.7 % look alarming:

  Q1 IS THE FLAG RIGHT?  No. `alpamayo.lateral.agree` was computed against a
     GOAL-token scan for TURN_/YIELD_FOR_TURN_ prefixes, defaulting to
     "straight" — so NUDGE_L/NUDGE_R (ACTIONS, never turn goals) were compared
     as straight. Localised in source at s2_geom_emit_v7.py:786-789 and fixed.
     ⚠️ `side_of` was NOT at fault; it handles the NUDGE suffix correctly and
     was never called with a NUDGE class. Fixing it would have been a no-op.

  Q2 DO THE SOURCES ACTUALLY CONTRADICT?  Only sometimes, and the shape matters
     enormously. "Alpamayo says straight, geometry says a 1 m nudge" is a
     SENSITIVITY difference between two instruments; "Alpamayo says left,
     geometry says right" is a real contradiction. This script separates them BY
     MEASUREMENT rather than by assertion.

Classes (a record is in exactly one):
  BOTH_AGREE          both name the same side, or both say straight
  HARD_CONTRADICTION  both name a side and the sides are OPPOSITE
  ONE_SIDED_ALPAMAYO  Alpamayo names a side, geometry says straight
  ONE_SIDED_GEOMETRY  geometry names a side, Alpamayo says straight

⭐ CONTROL (must read a KNOWN value or the decomposition is not trustworthy):
   side=straight + LANE_KEEP is unambiguously agreement. It must land in
   BOTH_AGREE, 100 % of it, and its shipped `agree` must already be True.
"""
import collections
import gzip
import hashlib
import json
from pathlib import Path

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus")
BLOB = REL / "labels" / "s2_labels_v7.jsonl.gz"
PINNED_MD5 = "ee44875916ae7c0ac002c6716b9658ea"      # THE RELEASE blob

raw = BLOB.read_bytes()
md5 = hashlib.md5(raw).hexdigest()
print(f"blob md5 {md5}  ({'PINNED RELEASE' if md5 == PINNED_MD5 else '⛔ NOT THE RELEASE'})")
assert md5 == PINNED_MD5, "six copies exist across three roots; this is not the release"
rows = [json.loads(x) for x in gzip.open(BLOB, "rt", encoding="utf-8") if x.strip()]
print(f"records {len(rows)}")


def side_of(cls: str) -> str:
    t = (cls or "").upper()
    if t.endswith(("_L", "_LEFT")):
        return "left"
    if t.endswith(("_R", "_RIGHT")):
        return "right"
    return "straight"


def classify(alp_side: str, geom_cls: str) -> str:
    g = side_of(geom_cls)
    if alp_side == g:
        return "BOTH_AGREE"
    if alp_side != "straight" and g != "straight":
        return "HARD_CONTRADICTION"
    return "ONE_SIDED_ALPAMAYO" if g == "straight" else "ONE_SIDED_GEOMETRY"


cls_counts = collections.Counter()
cell = collections.Counter()
shipped_agree = collections.Counter()
control = collections.Counter()
n = 0
for r in rows:
    lat = ((r.get("alpamayo") or {}).get("lateral") or {})
    alp = lat.get("side") or lat.get("alpamayo_side") or lat.get("value")
    ag = lat.get("agree")
    gc = (r.get("a_tac") or {}).get("lat")
    gc = gc.get("value") if isinstance(gc, dict) else gc
    if alp is None or gc is None or ag is None:
        continue
    n += 1
    k = classify(alp, gc)
    cls_counts[k] += 1
    cell[(alp, gc, k)] += 1
    shipped_agree[(k, bool(ag))] += 1
    if alp == "straight" and gc == "LANE_KEEP":
        control[(k, bool(ag))] += 1

print(f"records with both fields: {n}\n")
print("DECOMPOSITION BY CAUSE")
for k, v in cls_counts.most_common():
    print(f"  {k:20s} {v:5d}  {v/n*100:5.2f}%")
hard = cls_counts["HARD_CONTRADICTION"]
one = cls_counts["ONE_SIDED_ALPAMAYO"] + cls_counts["ONE_SIDED_GEOMETRY"]
print(f"\n  -> TRUE contradictions (both name a side, opposite): {hard} = {hard/n*100:.2f}%")
print(f"  -> sensitivity differences (one says straight):      {one} = {one/n*100:.2f}%")
print(f"  -> shipped `agree`=False count:  "
      f"{sum(v for (k, a), v in shipped_agree.items() if not a)} = "
      f"{sum(v for (k, a), v in shipped_agree.items() if not a)/n*100:.2f}%")

print("\nSHIPPED FLAG vs TRUTH (this is the defect, in numbers)")
for k in sorted(cls_counts):
    t = shipped_agree[(k, True)]
    f = shipped_agree[(k, False)]
    want = "True" if k == "BOTH_AGREE" else "False"
    bad = f if k == "BOTH_AGREE" else t
    print(f"  {k:20s} shipped True {t:5d} / False {f:5d}  (should be all {want}) "
          f"-> WRONG on {bad}")

print("\n⭐ CONTROL — side=straight + LANE_KEEP must be BOTH_AGREE and shipped True")
for (k, a), v in sorted(control.items()):
    print(f"   class={k} shipped_agree={a}  n={v}")
ok = list(control) == [("BOTH_AGREE", True)]
print(f"   CONTROL {'PASSES' if ok else '⛔ FAILS'} — "
      f"{'the decomposition reads the known cell correctly' if ok else 'do not trust the table above'}")

print("\nPER-CELL (alpamayo_side, geometry_class) -> class")
for (a, g, k), v in sorted(cell.items(), key=lambda x: -x[1]):
    print(f"  {a:8s} {g:10s} {k:20s} {v:5d}")

json.dump({"blob_md5": md5, "n": n, "classes": dict(cls_counts),
           "hard_contradiction": hard, "one_sided": one,
           "cells": {f"{a}|{g}|{k}": v for (a, g, k), v in cell.items()},
           "control_passes": ok},
          open(Path(__file__).with_name("lat_decomposition.json"), "w"), indent=1)
