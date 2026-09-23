"""Does a speed lever change what the model learns? Read from the smokes' own metrics.jsonl.

For each (reference, candidate) pair of smokes run with the same seed and data order: the
per-step training loss of both, the relative difference per step, its max and mean, and --
for the in-run evals -- the median relative deviation over the eval metrics (count fields
excluded) with the largest four named. Also reports each smoke's dedup telemetry
(`trunk_frame_slots` / `trunk_frames_computed`) as the distinct values its rows carry.

usage: smoke_tracking.py <out.json> <smoke_root> <refTag>:<candTag> [...]
"""
import json
import os
import sys


def load(path):
    tr, ev = {}, {}
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        st = r.get("step")
        if any(k.startswith("eval_") for k in r):
            ev[st] = r
        elif isinstance(st, int) and "loss" in r:
            tr[st] = r
    return tr, ev


def eval_dev(a, c, st):
    ks = [k for k in a[st] if k.startswith("eval_") and isinstance(a[st][k], (int, float))
          and isinstance(c[st].get(k), (int, float)) and "_n_" not in k
          and not k.startswith("eval_n") and abs(a[st][k]) > 1e-9]
    d = sorted(((abs(c[st][k] - a[st][k]) / abs(a[st][k]), k) for k in ks), reverse=True)
    med = sorted(x for x, _ in d)[len(d) // 2] if d else None
    return {"n_metrics": len(ks), "median_rel": med,
            "top4": [[k, round(x, 5)] for x, k in d[:4]],
            "eval_loss": [a[st].get("eval_loss"), c[st].get("eval_loss")]}


out, root, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
res = {"_what": "same-seed smoke pairs: per-step training-loss and in-run eval deviations",
       "_evidence_class": "MEASURED (ours, Thor metrics.jsonl)", "pairs": {}, "dedup": {}}
cache = {}
for p in pairs:
    ra, rc = p.split(":")
    for tag in (ra, rc):
        if tag not in cache:
            cache[tag] = load(os.path.join(root, tag, "metrics.jsonl"))
            rows = cache[tag][0].values()
            res["dedup"][tag] = sorted({(r.get("trunk_frame_slots"), r.get("trunk_frames_computed"))
                                        for r in rows}, key=str)
    (ta, ea), (tc, ec) = cache[ra], cache[rc]
    steps = sorted(set(ta) & set(tc))
    rel = [(tc[s]["loss"] - ta[s]["loss"]) / ta[s]["loss"] for s in steps]
    res["pairs"][p] = {
        "steps": len(steps),
        "loss_ref": [round(ta[s]["loss"], 3) for s in steps],
        "loss_cand": [round(tc[s]["loss"], 3) for s in steps],
        "rel_diff_pct": [round(100 * x, 3) for x in rel],
        "max_abs_rel": round(max(abs(x) for x in rel), 5),
        "mean_abs_rel": round(sum(abs(x) for x in rel) / len(rel), 5),
        "evals": {str(st): eval_dev(ea, ec, st) for st in sorted(set(ea) & set(ec))}}
    print(p, res["pairs"][p]["max_abs_rel"], res["pairs"][p]["mean_abs_rel"])
json.dump(res, open(out, "w"), indent=1)
