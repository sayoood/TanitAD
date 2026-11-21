"""per-test base-vs-new table from the batch summaries (markdown on stdout).  usage: compare.py [new_dir] [base_dir]   (defaults: logs/batch_new vs logs/batch_base; the *_long and *_env dirs override per test)"""
import json, sys
from pathlib import Path
L = Path("C:/Users/Admin/hn34/logs")
new_d, base_d = (sys.argv[1] if len(sys.argv) > 1 else "batch_new"), (sys.argv[2] if len(sys.argv) > 2 else "batch_base")


def load(d):
    p = L / d / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


new, base = load(new_d), {**load(base_d), **load("batch_base_rest")}   # pass3: the base batch's remainder
over_new = {**load("batch_new_env"), **load("batch_new_gpu")}   # pass3: CUDA tests run under the lock
over_base = {**load("batch_base_long"), **load("batch_base_env"), **load("batch_base_gpu")}


def cell(r):
    if r is None:
        return "not run"
    tok = (r.get("token_lines") or [""])[-1][:60]
    return f"rc {r['rc']} ({r['secs']} s) {tok}".strip()


keys = list(new) + [k for k in base if k not in new]
print("| test | base (unedited export) | new (edited tree) | verdict |")
print("|---|---|---|---|")
bad = 0
for k in keys:
    n, b = new.get(k), base.get(k)
    nn, bb = over_new.get(k, n), over_base.get(k, b)
    note = ""
    if k in over_new or k in over_base:
        note = " (env-fixed / long-timeout rerun)"
    if nn is None:
        v = "NEW-ONLY (no base counterpart)" if False else "not run"
    elif nn["rc"] == 0:
        v = "pass" if bb is None or bb["rc"] == 0 else "new passes, base fails"
        if bb is None:
            v = "pass (no base counterpart: a NEW test, or the base run has not reached it)"
    else:
        if bb is not None and bb["rc"] not in (0,):
            v = "same failure on base (environmental) - compare tails"
        else:
            v = "NEW FAILURE"
            bad += 1
    print(f"| `{k}` | {cell(bb)} | {cell(nn)} | {v}{note} |")
print(f"\nnew failures: {bad}")
