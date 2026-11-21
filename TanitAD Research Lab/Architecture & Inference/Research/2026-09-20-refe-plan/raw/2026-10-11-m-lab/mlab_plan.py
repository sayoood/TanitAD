#!/usr/bin/env python3
"""M-LAB step 1: index the navtest metric cache, draw the seeded, disjoint token samples, and extract the banked
64-proposal sets + NAVSIM census PDMS for each sample (a known-answer control for every timed run).

    python mlab_plan.py            (any python with numpy; reads, never writes, the cache/proposals)

Outputs (in this directory): index.json (token -> log, cache path), plan/<run>.json + plan/<run>.npz.
Seeds are fixed (20261011) so the samples are reproducible.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
PROPS = "E:/Projects/TanitAD/data/refe_navtest/proptable/navtest_final/proposals.npz"
CENSUS = os.path.join(PKG, "raw", "2026-10-04-lane-discipline", "lane_census.jsonl")
SEED = 20261011


def build_index():
    p = os.path.join(HERE, "index.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    idx = {}
    for f in glob.glob(f"{CACHE}/*/*/*/metric_cache.pkl"):
        f = f.replace("\\", "/")
        parts = f.split("/")
        tok, log = parts[-2], parts[-4]
        idx[tok] = {"log": log, "path": f}
    json.dump(idx, open(p, "w", encoding="utf-8"))
    return idx


def main():
    os.makedirs(os.path.join(HERE, "plan"), exist_ok=True)
    idx = build_index()
    P = np.load(PROPS, allow_pickle=True)
    ptok = [str(t) for t in P["token"]]
    pidx = {t: i for i, t in enumerate(ptok)}
    census = {}
    for line in open(CENSUS, encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:
            census[r["token"]] = np.asarray(r["pdms"][1:65], np.float64)        # 0 = pdm_closed, 1..64 = hypotheses
    toks = sorted(t for t in idx if t in pidx and t in census)
    print(f"cache tokens {len(idx)}, with proposals+census {len(toks)}")
    rng = random.Random(SEED)
    pool = toks[:]
    rng.shuffle(pool)
    runs = {}
    # (1) cached-parts arm: three disjoint random samples of 240 sets (tokens spread over all logs)
    for r in (1, 2, 3):
        runs[f"cached_r{r}"] = pool[(r - 1) * 240: r * 240]
    # (2) parts-build arm: log-grouped (the offline relabeller's regime, one scenario build per log); three disjoint log
    #     sets, >= 215 tokens each so >= 200 survive REFe's 5 s future guard. Logs are drawn in a seeded random order.
    bylog = {}
    for t in toks:
        bylog.setdefault(idx[t]["log"], []).append(t)
    logs = sorted(bylog)
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(logs)
    used = set()
    for r in (1, 2, 3):
        sel, n = [], 0
        for lg in logs:
            if lg in used:
                continue
            if n >= 225:
                break
            ts = sorted(bylog[lg])
            if n + len(ts) > 260:                      # keep the run bounded
                ts = ts[: 260 - n]
            sel += ts
            n += len(ts)
            used.add(lg)
        runs[f"build_r{r}"] = sel
    # (3) supplementary: one token per scenario build (no per-log amortisation), 100 distinct logs per run, disjoint
    rest = [lg for lg in logs if lg not in used]
    for r in (1, 2, 3):
        rr = random.Random(SEED + 10 + r)
        lgs = rr.sample([lg for lg in logs], 100)
        runs[f"single_r{r}"] = [rr.choice(sorted(bylog[lg])) for lg in lgs]
    for name, tl in runs.items():
        ptk = np.asarray([pidx[t] for t in tl])
        np.savez(os.path.join(HERE, "plan", name + ".npz"), token=np.asarray(tl),
                 proposals=P["proposals"][ptk].astype(np.float32),
                 census_pdms=np.stack([census[t] for t in tl]))
        json.dump({"name": name, "n": len(tl), "tokens": tl, "logs": sorted({idx[t]["log"] for t in tl}),
                   "paths": {t: idx[t]["path"] for t in tl}}, open(os.path.join(HERE, "plan", name + ".json"), "w"), indent=0)
        print(name, len(tl), "tokens", len({idx[t]['log'] for t in tl}), "logs")
    print("proposals ckpt:", str(P["ckpt"]), "| rule:", str(P["rule"]), "| repair_last_heading:", bool(P["repair_last_heading"]))


if __name__ == "__main__":
    sys.exit(main())
