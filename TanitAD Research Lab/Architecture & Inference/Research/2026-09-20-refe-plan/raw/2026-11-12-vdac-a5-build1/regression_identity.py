"""regression identity: v2 on the S1a inputs vs s1a_read.json.  usage: regress.py <v2 json> <v1 json>"""
import json, sys, math

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
SKIP_TOP = {"tool", "wall_s", "reader_sha256", "manifest_sha256", "audit"}
bad, n = [], [0]


def cmp(x, y, path):
    if isinstance(y, dict):
        if not isinstance(x, dict):
            bad.append((path, "type")); return
        for k in y:
            if path == "" and k in SKIP_TOP:
                continue
            if k not in x:
                bad.append((path + "/" + k, "missing in v2")); continue
            cmp(x[k], y[k], path + "/" + k)
    elif isinstance(y, list):
        if not isinstance(x, list) or len(x) != len(y):
            bad.append((path, "list len")); return
        for i, (p, q) in enumerate(zip(x, y)):
            cmp(p, q, f"{path}[{i}]")
    else:
        n[0] += 1
        if isinstance(y, float) and math.isnan(y):
            ok = isinstance(x, float) and math.isnan(x)
        else:
            ok = (x == y) and type(x) == type(y)
        if not ok:
            bad.append((path, f"{x!r} != {y!r}"))


cmp(a, b, "")
extra = sorted(set(a) - set(b))
print("compared leaf values:", n[0])
print("mismatches:", len(bad))
for p, m in bad[:20]:
    print("  ", p, m)
print("v2-only top-level keys:", extra)
print("audit equal except paths:", {k: (a['audit'].get(k) == b['audit'].get(k)) for k in b['audit']})
print("verdict v2:", a["verdict"], "| v1:", b["verdict"])
sys.exit(1 if bad else 0)
