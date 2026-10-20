"""Compare the re-derived C1 analysis (spc_analyse.py at the pinned tip, run on C1's inputs) with the chain's banked RESULT_SPC.json, key by key."""
import json, sys, hashlib
a = json.load(open(sys.argv[1], encoding="utf-8"))
b = json.load(open(sys.argv[2], encoding="utf-8"))
diffs = []
def walk(x, y, p=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                diffs.append(("KEY", p + "/" + k)); continue
            walk(x[k], y[k], p + "/" + k)
    elif x != y:
        diffs.append(("VALUE", p, str(x)[:60], str(y)[:60]))
walk(a, b)
out = {"rederived": sys.argv[1], "chain": sys.argv[2], "n_diffs": len(diffs), "diffs": diffs, "dict_equal": a == b,
       "concordance_equal": a.get("concordance_reported_only") == b.get("concordance_reported_only")}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_diffs", "dict_equal", "concordance_equal")}))
