#!/usr/bin/env python3
"""Per-scene throughput from seam-agent call logs (``*.calls.jsonl``: one ``event: call`` per scored token,
with a wall-clock ``t``). Any python.

    python throughput.py --calls a.calls.jsonl [b.calls.jsonl ...] --out tp.json

Per log: n calls, span (first->last call), seconds/scene in that span, and init seconds (initialize
event -> first call). Across logs: the window in which ALL logs were emitting (max first .. min last),
the calls inside it and the aggregate scenes/s. The init (scene-loader) phase is excluded from s/scene
on purpose -- it is a per-process constant, reported separately.
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    per = {}
    for p in a.calls:
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        ts = sorted(r["t"] for r in recs if r.get("event") == "call")
        t_init = next((r["t"] for r in recs if r.get("event") == "initialize"), None)
        if len(ts) < 2:
            continue
        per[p] = {"n_calls": len(ts), "first": ts[0], "last": ts[-1], "span_s": round(ts[-1] - ts[0], 1),
                  "s_per_scene": round((ts[-1] - ts[0]) / (len(ts) - 1), 3),
                  "init_s": (round(ts[0] - t_init, 1) if t_init else None), "_ts": ts}
    rep = {"per_log": {k: {kk: vv for kk, vv in v.items() if kk != "_ts"} for k, v in per.items()}}
    if len(per) > 1:
        lo = max(v["first"] for v in per.values())
        hi = min(v["last"] for v in per.values())
        if hi > lo:
            n = sum(sum(1 for t in v["_ts"] if lo <= t <= hi) for v in per.values())
            rep["all_concurrent_window"] = {"n_logs": len(per), "window_s": round(hi - lo, 1), "calls": n,
                                            "aggregate_scenes_per_s": round(n / (hi - lo), 3),
                                            "per_process_s_per_scene": round(len(per) * (hi - lo) / n, 3)}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
