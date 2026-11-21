"""Dump the nearfield block (definition, table, ci rows) of EVAL jsons side by side. Read-only."""
import json, sys

paths = sys.argv[1:]
E = [json.load(open(p, encoding="utf-8")) for p in paths]
nf0 = E[0]["nearfield"]
print("definition:", json.dumps(nf0.get("definition"), indent=1)[:3000])
print("lift_valid:", json.dumps(nf0.get("lift_valid"))[:600])
print("unobs_flags:", json.dumps(nf0.get("unobs_flags"))[:600])
print("edge_plane:", nf0.get("edge_plane"))
print("table keys:", list(nf0["table"].keys()))
for k in nf0["table"]:
    print("TABLE", k, [json.dumps(e["nearfield"]["table"].get(k))[:700] for e in E])
print()
for k in nf0["ci"]:
    row = []
    for e in E:
        v = e["nearfield"]["ci"].get(k)
        if isinstance(v, dict):
            row.append("%s [%s, %s] n=%s" % (v.get("point"), v.get("lo"), v.get("hi"), v.get("n")))
        else:
            row.append(repr(v)[:80])
    print("CI %-36s %s" % (k, "  ||  ".join(row)))
