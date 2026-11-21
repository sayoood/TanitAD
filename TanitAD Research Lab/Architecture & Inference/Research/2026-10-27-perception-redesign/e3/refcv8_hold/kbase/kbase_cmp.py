"""kbase_cmp.py -- MM 4a3d05, 2026-11-08: independent of the agent's cmp.py. sha256 of the banked K-BASE table against the registered prefix, then a recursive compare with the re-derived table."""
import glob
import hashlib
import json
import math

A = glob.glob("C:/Users/Admin/qland/hold_r8pf/stage/**/FROZEN_GAINS_RETIME_train_pf.json", recursive=True)
assert len(A) == 1, A
B = "E:/Projects/TanitAD-artifacts/dagland/e3/refcv8_hold/kbase/FROZEN_GAINS_RETIME_train_pf.REDERIVED.json"
ha, hb = (hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (A[0], B))
print("banked", A[0][-70:], ha)
print("rederived", hb)
print("banked sha starts 2aacc9ff:", ha.startswith("2aacc9ff"))
stats = {"num": 0, "maxdiff": 0.0, "diff_keys": []}


def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                stats["diff_keys"].append(f"{path}/{k} ONE-SIDED")
            else:
                walk(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            stats["diff_keys"].append(f"{path} LEN {len(a)} {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        stats["num"] += 1
        if not (a == b or (math.isnan(a) and math.isnan(b))):
            stats["maxdiff"] = max(stats["maxdiff"], abs(a - b))
            stats["diff_keys"].append(path)
    elif a != b:
        stats["diff_keys"].append(f"{path} STR {str(a)[:60]} | {str(b)[:60]}")


walk(json.load(open(A[0], encoding="utf-8")), json.load(open(B, encoding="utf-8")), "")
print("numeric leaves", stats["num"], "max abs diff", stats["maxdiff"])
print("differing keys", len(stats["diff_keys"]))
for k in stats["diff_keys"][:14]:
    print("  ", k[:170])
