"""Read smoke metrics.jsonl files the SAME way and project the production step time.

Per smoke, from the artifact only (metrics.jsonl `elapsed_s`, logged every step):
  * plain    -- median step-to-step delta over steps that carry NO conflict reading and follow
                no in-run eval (steps <= 3 are warm-up and excluded);
  * conflict -- median extra seconds on a step that carries a conflict reading, over plain;
  * eval     -- extra seconds on the step that follows an in-run eval, over plain, per eval
                batch (the smoke evaluates EVAL_BATCHES=2; the launch evaluates 8 every 500);
  * window   -- the pre-registered read: (elapsed[last] - elapsed[10th row]) / steps.
Projection for the launch line: plain + conflict/CE + eval_per_batch * 8 / 500; samples/s uses
the smoke's OWN batch, read from the `--batch` in its config.json (never assumed).

usage: step_budget.py <out.json> <tag>=<metrics.jsonl> [...]
"""
import json
import os
import statistics
import sys


def load(path):
    rows, evals, conflict_steps = [], set(), set()
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        st = r.get("step")
        if any(k.startswith("eval_") for k in r) and isinstance(st, int):
            evals.add(st)
            continue
        if isinstance(st, int) and "elapsed_s" in r:
            rows.append(r)
            if any(k.startswith("cd_") and k != "cd_deferred" for k in r):
                conflict_steps.add(st)
    rows.sort(key=lambda r: r["step"])
    return rows, evals, conflict_steps


def _batch(metrics_path):
    argv = json.load(open(os.path.join(os.path.dirname(metrics_path), "config.json"),
                          encoding="utf-8"))["argv"]
    return int(argv[argv.index("--batch") + 1])


def analyse(path, ce=10, launch_eval_batches=8, launch_eval_every=500, smoke_eval_batches=2):
    rows, evals, cds = load(path)
    batch = _batch(path)
    by = {r["step"]: r for r in rows}
    deltas = {}
    for r in rows:
        p = by.get(r["step"] - 1)
        if p is not None:
            deltas[r["step"]] = r["elapsed_s"] - p["elapsed_s"]
    # an in-run eval runs AFTER its step's train row is logged, so its cost lands in the NEXT
    # step's delta -- which in these smokes is also a conflict-reading step (eval at 20, cd at 21)
    after_eval = {e + 1 for e in evals}
    plain = [d for s, d in deltas.items() if s > 3 and s not in cds and s not in after_eval]
    conf = [d for s, d in deltas.items() if s > 3 and s in cds and s not in after_eval]
    p_med = statistics.median(plain) if plain else float("nan")
    c_extra = (statistics.median(conf) - p_med) if conf else float("nan")
    ev = [d - p_med - (c_extra if s in cds else 0.0)
          for s, d in deltas.items() if s > 3 and s in after_eval]
    e_extra = max(ev) if ev else float("nan")
    w = rows[9:] if len(rows) >= 12 else []
    window = ((w[-1]["elapsed_s"] - w[0]["elapsed_s"]) / max(1, w[-1]["step"] - w[0]["step"])
              if w else float("nan"))
    eval_per_batch = e_extra / smoke_eval_batches if ev else 0.0
    proj = p_med + (c_extra if conf else 0.0) / ce + eval_per_batch * launch_eval_batches / launch_eval_every
    return {"batch": batch, "rows": len(rows), "eval_steps": sorted(evals),
            "conflict_steps": sorted(cds),
            "plain_s_median": round(p_med, 3), "plain_n": len(plain),
            "plain_min_max": [round(min(plain), 3), round(max(plain), 3)] if plain else None,
            "conflict_extra_s": round(c_extra, 3), "conflict_n": len(conf),
            "eval_extra_s_smoke": round(e_extra, 3) if ev else None,
            "window_s_per_step": round(window, 3),
            "peak_cuda_max_mem_gb": max((r.get("cuda_max_mem_gb", 0.0) for r in rows), default=0.0),
            "loss_first6": [round(float(r.get("loss", float("nan"))), 2) for r in rows[:6]],
            "projected_launch_s_per_step_CE10": round(proj, 3),
            "projected_samples_per_s_CE10": round(batch / proj, 4)}


out, pairs = sys.argv[1], sys.argv[2:]
res = {"_what": "smoke step budget, read identically for every smoke (see docstring)",
       "_evidence_class": "MEASURED (ours, Thor metrics.jsonl); projection ESTIMATED from it"}
for p in pairs:
    tag, path = p.split("=", 1)
    res[tag] = analyse(path)
    print(tag, json.dumps(res[tag]))
json.dump(res, open(out, "w"), indent=1)
