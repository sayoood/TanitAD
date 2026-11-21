"""Compact verdict summary of raw/RECOVER_READ.json (for the Master Mind's decision; RESULTS.md is written separately).
python -I summarize_read.py [path]"""
import json
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "E:/Projects/TanitAD-artifacts/dagland/recover_read/raw/RECOVER_READ.json"
j = json.load(open(P, encoding="utf-8"))


def short(d, keys):
    return {k: d.get(k) for k in keys if k in d}


print("mode", j.get("mode"), "| tier", j.get("tier"), "| refused", j.get("refused"))
print("LABEL", j.get("label"))
print("PRIMARY", short(j.get("primary", {}), ("n_seeds_separated_better", "mean_share_closed", "share_closed", "PASS")))
g = j.get("guards", {})
print("GUARDS PASS", g.get("PASS"))
for s, c in (g.get("per_seed") or {}).items():
    print("  guard", s, json.dumps(c)[:400])
f = j.get("floor", {})
print("FLOOR", json.dumps(f)[:500])
print("FREEZE", json.dumps(short(j.get("frame_freeze", {}), ("reading", "n_seeds_separated_better", "mean_share_closed", "share_closed", "note")))[:500])
ra = j.get("read_a_R_vs_P10", {})
print("READ_A R vs P10", json.dumps({k: v for k, v in ra.items() if not isinstance(v, dict)})[:300])
for s, c in ra.items():
    if isinstance(c, dict):
        print("  ", s, json.dumps(short(c, ("D1", "D2", "D3", "D3_three_family", "PASS", "worse_rows", "dominant")))[:500])
co = j.get("composite", {})
print("COMPOSITE", json.dumps({k: v for k, v in co.items() if not isinstance(v, dict)})[:600])
for s, c in co.items():
    if isinstance(c, dict):
        print("  ", s, json.dumps(short(c, ("D1", "D2", "D3", "PASS", "worse_rows", "ade6", "dominant")))[:600])
for s in ("s0", "s1"):
    rows = (j.get("seeds", {}).get(s, {}) or {}).get("R_minus_C0", {})
    for k in ("M1_crawl_share", "ADE6", "ade6"):
        if k in rows:
            print("  R-C0", s, k, json.dumps(rows[k])[:200])
    m1 = [k for k in rows if "M1" in k or "crawl" in k.lower()]
    if m1:
        print("  R-C0", s, "M1 keys", m1[:4], json.dumps(rows[m1[0]])[:200])
