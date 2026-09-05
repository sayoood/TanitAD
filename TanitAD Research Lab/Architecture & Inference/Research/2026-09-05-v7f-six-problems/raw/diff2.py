#!/usr/bin/env python3
"""Full key-by-key config diff between two v7tiny arms + metrics history keys."""
import json, sys, os

ROOT = "/home/nvidia/v7tiny"

def load(n):
    p = os.path.join(ROOT, n, "config.json")
    c = json.load(open(p))
    return c.get("args", c), c

def diff(a_name, b_name):
    A, ca = load(a_name)
    B, cb = load(b_name)
    ka, kb = set(A), set(B)
    print("#### %s (%d args) vs %s (%d args)" % (a_name, len(A), b_name, len(B)))
    onlyA = sorted(ka - kb); onlyB = sorted(kb - ka)
    print("  keys only in %s (%d): %s" % (a_name, len(onlyA), onlyA))
    print("  keys only in %s (%d): %s" % (b_name, len(onlyB), onlyB))
    print("  SHARED KEYS WITH DIFFERENT VALUES:")
    n = 0
    for k in sorted(ka & kb):
        if A[k] != B[k]:
            print("    %-28s %r  ->  %r" % (k, A[k], B[k]))
            n += 1
    print("  -> %d shared-key differences" % n)
    # argv if present
    for nm, c in ((a_name, ca), (b_name, cb)):
        if "argv" in c:
            print("  ARGV[%s]: %s" % (nm, " ".join(map(str, c["argv"]))[:600]))
    print()

for pair in [("postrain30k", "postrain30k_freeze"),
             ("postrain30k", "splitp30k"),
             ("postrain30k_freeze", "splitp30k")]:
    diff(*pair)

# metrics history keys
for n in ("postrain30k", "postrain30k_freeze", "splitp30k"):
    p = os.path.join(ROOT, n, "metrics.json")
    if os.path.exists(p):
        j = json.load(open(p))
        h = j.get("history")
        print("METRICS %s: top=%s" % (n, sorted(j.keys())))
        if isinstance(h, list) and h:
            print("   history rows=%d  last-row keys=%s" % (len(h), sorted(h[-1].keys())))
        elif isinstance(h, dict):
            print("   history dict keys=%s" % sorted(h.keys())[:60])
