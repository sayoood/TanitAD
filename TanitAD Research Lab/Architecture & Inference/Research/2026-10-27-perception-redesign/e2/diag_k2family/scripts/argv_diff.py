"""Parse r9e1/r9e2 argv.json files into flag->value dicts and print every difference.
Read-only; usage: python -I argv_diff.py <dir>"""
import json, sys, os

D = sys.argv[1]
ARMS = ["r9e2_K0", "r9e2_K2M1N20", "r9e2_K2DC", "r9e1_C0", "r9e1_M1N20"]


def parse(argv):
    out = {}
    order = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            vals = []
            j = i + 1
            while j < len(argv) and not (argv[j].startswith("--") and len(argv[j]) > 2 and not argv[j][2].isdigit()):
                vals.append(argv[j])
                j += 1
            key = a
            v = " ".join(vals) if vals else "<flag>"
            if key in out:
                out[key] = out[key] + " || " + v
            else:
                out[key] = v
                order.append(key)
            i = j
        else:
            out.setdefault("<positional>", "")
            out["<positional>"] += " " + a
            i += 1
    return out, order


data = {}
meta = {}
for arm in ARMS:
    p = os.path.join(D, arm + ".argv.json")
    j = json.load(open(p, encoding="utf-8"))
    meta[arm] = {k: v for k, v in j.items() if k != "argv"}
    data[arm] = parse(j["argv"])

for arm in ARMS:
    print("META", arm, json.dumps(meta[arm])[:1500])
print()


def diff(a, b):
    da, _ = data[a]
    db, _ = data[b]
    keys = sorted(set(da) | set(db))
    print("=== DIFF %s -> %s" % (a, b))
    n = 0
    for k in keys:
        va = da.get(k, "<absent>")
        vb = db.get(k, "<absent>")
        if va != vb:
            n += 1
            print("  %-40s %s  |  %s" % (k, va[:110], vb[:110]))
    print("  n_diff =", n)
    print()


diff("r9e2_K0", "r9e2_K2M1N20")
diff("r9e2_K2M1N20", "r9e2_K2DC")
diff("r9e2_K0", "r9e2_K2DC")
diff("r9e1_C0", "r9e1_M1N20")
diff("r9e1_M1N20", "r9e2_K2M1N20")
diff("r9e1_C0", "r9e2_K0")
