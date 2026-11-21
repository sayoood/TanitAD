"""Compact table of a pair_describe.py JSON: per metric, both seeds: base, treat, delta, CI, separated.
usage: python pair_table.py <json> [metric-substring ...]"""
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
want = sys.argv[2:]
print(f"{d['treatment']} vs {d['base']}  ({d['what']})")
s0, s1 = d["seeds"]["s0"]["pair"], d["seeds"]["s1"]["pair"]
keys = [k for k in s0 if isinstance(s0[k], dict) and "family" in s0[k]]
if want:
    keys = [k for k in keys if any(w in k for w in want)]
first = s0[keys[0]] if keys else {}
print("paired keys:", sorted(k for k in first if k not in ("family", "base", "treat", "n")))


def cell(r):
    if not r:
        return "-"
    delta = r.get("delta")
    lo, hi = (r.get("ci") or [None, None])[:2]
    sep = "BETTER" if r.get("sep_better") else ("WORSE" if r.get("sep_worse") else "ns")
    fmt = lambda x: "None" if x is None else f"{x:+.4f}"
    return f"{r['base']:.4f}->{r['treat']:.4f} d={fmt(delta)} [{fmt(lo)},{fmt(hi)}] {sep}"


for k in keys:
    print(f"{s0[k]['family'][:5]:5s} {k:28s} s0 {cell(s0.get(k))} | s1 {cell(s1.get(k))}")
