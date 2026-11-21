import json, sys, collections
d = json.load(open(sys.argv[1], encoding="utf-8"))
P = d["params"]; B = d["buffers"]
print("dvb", d["dvb"], "state keys", d["state_dict"])
def under(n, m): return n == m or n.startswith(m + ".")
MAP = ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
dec = [p for p in P if under(p[0], "core.decoder")]
mp = [p for p in P if any(under(p[0], m) for m in MAP)]
print("planner(core.decoder) params", len(dec), sum(p[2] for p in dec))
print("map params", len(mp), sum(p[2] for p in mp))
print("planner+map", len(dec) + len(mp), sum(p[2] for p in dec + mp))
print("total", len(P), sum(p[2] for p in P))
print("frozen under planner,map", len(P) - len(dec) - len(mp), sum(p[2] for p in P) - sum(p[2] for p in dec + mp))
print("frozen under planner", len(P) - len(dec), sum(p[2] for p in P) - sum(p[2] for p in dec))
print("frozen under map", len(P) - len(mp), sum(p[2] for p in P) - sum(p[2] for p in mp))
print("decoder buffers", [(b[0], b[2]) for b in B if under(b[0], "core.decoder")])
tops = collections.OrderedDict()
for n, s, k in P:
    p = n.split(".")
    t = ".".join(p[:2]) if p[0] in ("core", "_perception", "_map_hires") else p[0]
    e = tops.setdefault(t, [0, 0]); e[0] += 1; e[1] += k
for t, (c, k) in tops.items():
    print(f"  {t:36s} {c:5d} {k:12d}")
g = collections.OrderedDict()
for n, s, k in dec + mp:
    gg = n.rsplit(".", 1)[0]
    g[gg] = g.get(gg, 0) + k
print("leaf groups planner+map", len(g))
json.dump(g, open(sys.argv[2], "w"), indent=1)
