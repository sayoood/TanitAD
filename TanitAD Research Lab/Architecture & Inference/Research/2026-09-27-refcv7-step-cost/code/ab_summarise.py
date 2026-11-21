#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-process A/B (step_profiler --ab LEVER): the lever alternates OFF/ON in blocks of K steps after
the warm-up, in ONE process. Estimator: block-PAIRED difference -- for each adjacent (OFF block,
ON block) pair, median(compute_s | ON) - median(compute_s | OFF); reported as the median over pairs
with a percentile bootstrap 90 % CI over PAIRS (not over steps: steps inside a block share drift).
compute_s = wall - data_wait (the dev box loads in the main process; the boundary sync makes the
GPU idle while it loads). Conflict steps (k % 10 == 0) and logged/final steps are excluded -- the
conflict probe's own cost is measured separately and is unaffected by these levers except where
stated. A NEGATIVE difference = the lever SAVES time."""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path


def main(job_dir: str, out: str | None = None):
    jd = Path(job_dir)
    prof = json.loads(next(jd.glob("profile_*.json")).read_text(encoding="utf-8"))
    ab = prof.get("ab") or {}
    K, warm, steps = int(ab["block"]), int(ab["warm"]), int(prof["args"]["steps"])
    log_every = int(prof["args"]["log_every"])
    rows = [json.loads(l) for l in (jd / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    blocks = {}
    for r in rows:
        k = int(r["step_index"])
        if k < warm or k >= steps - 1 or k % 10 == 0 or (k + 1) % log_every == 0:
            continue
        if "wall" not in r:
            continue
        b = (k - warm) // K
        blocks.setdefault(b, []).append((int(r.get("ab_on", 0)), r["wall"] - r.get("data_wait", 0.0)))
    pairs = []
    for b in sorted(blocks):
        if b % 2 == 0 and (b + 1) in blocks:
            off = [v for on, v in blocks[b] if not on]
            onv = [v for on, v in blocks[b + 1] if on]
            if off and onv:
                pairs.append(statistics.median(onv) - statistics.median(off))
    allon = [v for b in blocks.values() for on, v in b if on]
    alloff = [v for b in blocks.values() for on, v in b if not on]
    res = {"lever": ab.get("lever"), "block": K, "warm": warm, "n_pairs": len(pairs),
           "n_on_steps": len(allon), "n_off_steps": len(alloff),
           "median_on_s": round(statistics.median(allon), 4) if allon else None,
           "median_off_s": round(statistics.median(alloff), 4) if alloff else None,
           "paired_diff_median_s": round(statistics.median(pairs), 4) if pairs else None,
           "paired_diffs_s": [round(x, 4) for x in pairs],
           "batch": int(prof["args"]["batch"]), "peak_mem_gb": prof.get("peak_mem_gb")}
    if len(pairs) >= 3:
        rng = random.Random(0)
        bs = sorted(statistics.median([rng.choice(pairs) for _ in pairs]) for _ in range(4000))
        res["ci90_paired_diff_s"] = [round(bs[200], 4), round(bs[3800], 4)]
    if res["median_off_s"]:
        res["rel_saving_vs_off"] = (round(-res["paired_diff_median_s"] / res["median_off_s"], 4)
                                    if res["paired_diff_median_s"] is not None else None)
    if out:
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "paired_diffs_s"}))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
