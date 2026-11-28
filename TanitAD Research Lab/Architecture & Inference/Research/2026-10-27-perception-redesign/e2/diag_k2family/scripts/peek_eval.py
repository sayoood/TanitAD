"""Print the structure of an EVAL json (keys, ckpt provenance, map blocks). Read-only."""
import json, sys


def walk(d, pre="", depth=0, maxdepth=3):
    if depth > maxdepth:
        return
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                n = len(v)
                print("%s%s: <%s n=%d>" % ("  " * depth, k, type(v).__name__, n))
                if isinstance(v, dict):
                    walk(v, pre + k + ".", depth + 1, maxdepth)
            else:
                s = repr(v)
                print("%s%s: %s" % ("  " * depth, k, s[:160]))


for p in sys.argv[1:]:
    print("#####", p)
    d = json.load(open(p, encoding="utf-8"))
    walk(d, maxdepth=int(__import__("os").environ.get("MAXD", "2")))
