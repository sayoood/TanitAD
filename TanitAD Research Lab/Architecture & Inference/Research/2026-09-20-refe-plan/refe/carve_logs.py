#!/usr/bin/env python3
"""B8 (raw/2026-10-18-r0e-carvefeed): build the CARVE LOG LIST that `scorer_finetune.py --epo-con-feed carve --epo-carve-logs <file> --epo-carve-sha256 <sha>` reads.

THE DEFINITION IS SO-CARVE'S, NOT NEW (raw/2026-10-18-so-carve/PREREG_SOCARVE.md s.5; so_carve_analysis.py `carve_draws`): the pool's logs are sorted by name (np.unique), `rng = np.random.default_rng(20260927)`,
the carve is the FIRST draw `rng.choice(n_logs, size=m, replace=False)`, m = 13 (the val pool's log count in that package).  What changes is the POOL: SO-CARVE drew from the 30 logs of a harness pool of R0's states (so its
carve was in-sample, F1); here the pool is the RL-train index's logs and the trainer CUTS the carve out of the training index, which is what makes it held out.  Whole logs, never sets: a set-level split would leak
neighbouring frames of one drive into both sides.

Refusals (exit 4, nothing written): the pool overlaps --rlval-logs (the pool must already exclude RL-val, as the trainer's own filter does), the carve overlaps --navtest-logs, m is not in [1, n_logs - 1].
    python carve_logs.py --train-logs F --rlval-logs F [--navtest-logs F] [--m 13] [--seed 20260927] --out carve_logs.txt     (writes <out>, <out>.sha256 and <out>.manifest.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

SEED, M_PRIMARY = 20260927, 13            # SO-CARVE: SEED, M_PRIMARY (so_carve_analysis.py:10-11)


def draw(pool_logs, m: int = M_PRIMARY, seed: int = SEED) -> list:
    """SO-CARVE's first carve draw over the pool's logs (sorted unique names) -> the m carved log names, sorted"""
    logs = sorted(set(pool_logs))
    if not 1 <= m <= len(logs) - 1:
        raise ValueError(f"m = {m} is not in [1, {len(logs) - 1}] for a pool of {len(logs)} logs")
    idx = np.random.default_rng(seed).choice(len(logs), size=m, replace=False)
    return sorted(logs[int(i)] for i in idx)


def build(pool_logs, rlval_logs, navtest_logs=(), m: int = M_PRIMARY, seed: int = SEED, counts: dict | None = None) -> dict:
    """-> {"logs", "text", "sha256", "manifest"}; raises ValueError on any overlap"""
    pool, val, nav = set(pool_logs), set(rlval_logs), set(navtest_logs)
    if pool & val:
        raise ValueError(f"{len(pool & val)} pool log(s) are RL-val logs (the pool must exclude RL-val)")
    carve = draw(pool, m, seed)
    if set(carve) & nav:
        raise ValueError(f"{len(set(carve) & nav)} carve log(s) are navtest logs")
    text = "\n".join(carve) + "\n"
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    man = {"definition": "SO-CARVE carve_draws, first draw, over the sorted unique pool logs", "seed": seed, "m": m, "pool_logs": len(pool), "carve_logs": len(carve), "rlval_logs": len(val), "navtest_logs": len(nav),
           "pool_overlaps_navtest": len(pool & nav), "carve_overlaps_rlval": 0, "carve_overlaps_navtest": 0, "sha256": sha}
    if counts:
        man["carve_sets"], man["pool_sets"] = sum(int(counts.get(x, 0)) for x in carve), sum(int(counts.get(x, 0)) for x in pool)
    return {"logs": carve, "text": text, "sha256": sha, "manifest": man}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-logs", required=True, help="the RL-train index's log names, whitespace separated")
    ap.add_argument("--rlval-logs", required=True)
    ap.add_argument("--navtest-logs", default="")
    ap.add_argument("--counts", default="", help="optional JSON {log: n_sets}: the manifest then reports the carve's set count")
    ap.add_argument("--m", type=int, default=M_PRIMARY)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rd = lambda p: open(p, encoding="utf-8").read().split() if p else []          # noqa: E731
    try:
        r = build(rd(a.train_logs), rd(a.rlval_logs), rd(a.navtest_logs), a.m, a.seed, json.load(open(a.counts)) if a.counts else None)
    except ValueError as e:
        print("REFUSED: " + str(e))
        return 4
    open(a.out, "wb").write(r["text"].encode("utf-8"))
    open(a.out + ".sha256", "w", encoding="utf-8").write(r["sha256"] + "\n")
    json.dump(r["manifest"], open(a.out + ".manifest.json", "w", encoding="utf-8"), indent=1)
    print(f"carve: {len(r['logs'])} of {r['manifest']['pool_logs']} logs, sha256 {r['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
