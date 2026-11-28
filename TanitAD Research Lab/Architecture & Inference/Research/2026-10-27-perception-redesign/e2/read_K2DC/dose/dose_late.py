"""Addendum 4 committed statistic over readings at steps >= 3000 (13 firings) and after step 250 (companion to dose_diag.py).
usage: python -I dose_late.py < dc_rows.jsonl   (same bootstrap as dose_diag.py: firing bootstrap, 20000 draws, seed 0)"""
import json
import math
import random
import sys

rows = [json.loads(line) for line in sys.stdin if line.strip()]
rows = [r for r in rows if "dc_share_read" in r]
for lab, sel in (("steps >= 3000", [r for r in rows if r["step"] >= 3000]), ("steps > 250", [r for r in rows if r["step"] > 250])):
    vals = [r["dc_share_read"] for r in sel]
    pos = [v for v in vals if v > 0]
    lg = [math.log(v) for v in pos]
    m = sum(lg) / len(lg)
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(lg) for _ in lg) / len(lg) for _ in range(20000))
    lo, hi = math.exp(boots[500]), math.exp(boots[19499])
    verdict = "CI upper < 0.15: dose held at or under the band top" if hi < 0.15 else ("CI lower > 0.15: ABOVE the band" if lo > 0.15 else "not resolved by n = %d" % len(lg))
    print("%s: n %d (non-positive excluded %d) geometric-mean %.4f firing-bootstrap 95%% CI [%.4f, %.4f] -> %s" % (lab, len(lg), len(vals) - len(pos), math.exp(m), lo, hi, verdict))
    print("  readings:", [(r["step"], round(r["dc_share_read"], 4)) for r in sel])
