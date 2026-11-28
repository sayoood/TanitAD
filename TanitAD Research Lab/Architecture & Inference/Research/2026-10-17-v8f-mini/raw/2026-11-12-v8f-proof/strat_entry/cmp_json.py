"""cmp_json.py <a.json> <b.json> : every value of the two result files must be equal, except strings that carry the STRATEGIC entry, file hashes of the reader itself
and time stamps. Prints the differing paths (at most 20) and ZZ CMP-JSON EQUAL | DIFFERENT."""
import io
import json
import math
import sys

OLD = "PAUSED (PI item 25)"
SKIP_KEYS = ("sha256", "time", "date", "elapsed", "out_dir", "argv", "reader", "script", "tool", "host", "cwd")
diffs, n_num, n_skip, n_strat = [], 0, 0, 0


def walk(a, b, path):
    global n_num, n_skip, n_strat
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(path + "/" + k + " (in one file only)")
            else:
                walk(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(path + " (length %d against %d)" % (len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + "[%d]" % i)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            diffs.append(path)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if path.lower().rsplit("/", 1)[-1].startswith(("elapsed", "time", "wall")):      # run time of the reader, not a result
            n_skip += 1
            return
        n_num += 1
        if not (a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))):
            diffs.append(path + " (%r against %r)" % (a, b))
    elif isinstance(a, str) and isinstance(b, str):
        if a != b:
            if OLD in a or OLD in b:
                n_strat += 1
            elif any(s in path.lower().rsplit("/", 1)[-1] for s in SKIP_KEYS):
                n_skip += 1
            else:
                diffs.append(path + " (strings differ)")
    elif a != b:
        diffs.append(path + " (types differ)")


A = json.load(io.open(sys.argv[1], encoding="utf-8"))
B = json.load(io.open(sys.argv[2], encoding="utf-8"))
walk(A, B, "")
for d in diffs[:20]:
    print("   differs:", d[:200])
print("numbers compared", n_num, "| strings with the STRATEGIC entry", n_strat, "| skipped provenance strings", n_skip, "| differing", len(diffs))
print("ZZ CMP-JSON", "EQUAL" if not diffs and n_num > 0 else "DIFFERENT", sys.argv[1].replace("\\", "/").rsplit("/", 1)[-1])
sys.exit(0 if not diffs and n_num > 0 else 1)
