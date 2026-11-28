#!/usr/bin/env python3
"""prof_labeller.py -- where the P6 online labeller's 0.79-0.85 CPU-s per set goes, and which exact levers cut it.

Runs IN-PROCESS (one core, the cost a pool worker pays per job) on REAL D9 parts files under the teacher venv:
  1. per-set breakdown at chunks=2 (P6's argv): read bytes / lzma.decompress / pickle.loads / executed_plans8 / score_plans, per chunk
  2. cProfile of online_labeller._job over the same sets (top functions by cumulative time)
  3. EXACT levers, each checked BIT-IDENTICAL against the unmodified _job's labels:
       raw      parts stored DECOMPRESSED (the same pickle bytes, no lzma) -> pickle.loads only
       colo     both chunks of a set on the same worker (2nd chunk hits the LRU: one load per set instead of two)
       obj      parts kept as live objects (an in-worker cache that never re-parses): score cost only
  usage: python prof_labeller.py --cache <metric_cache dir> --refe <refe dir> --n-sets 24 [--out prof.json]
Proposals: deterministic straight-ish trajectories (diag_p6_labeller_identity.proposals), or --props-npy from a real model dump.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import lzma
import os
import pickle
import pstats
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--refe", default="/root/tad/speed/refe")
    ap.add_argument("--n-sets", type=int, default=24)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--props-npy", default="")
    ap.add_argument("--raw-dir", default="/root/tad/speed/mc_raw", help="where the decompressed copies of the sampled parts go")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sys.path.insert(0, a.refe)
    sys.path.insert(0, str(Path(a.refe).parent / "code"))
    import online_labeller as OL
    import diag_p6_labeller_identity as DI
    OL._pdm_init(a.cache, lru=8)
    PT = OL._W["PT"]
    files = []
    for d in sorted(Path(a.cache).iterdir()):
        if d.is_dir():
            for f in sorted(d.glob("*.pkl.xz")):
                if "__lane_rank_" not in f.name:
                    files.append(f)
                    break
        if len(files) >= a.n_sets:
            break
    props = np.load(a.props_npy) if a.props_npy else DI.proposals(len(files), seed=0)
    M = props.shape[1]
    cuts = [slice(c * M // a.chunks, (c + 1) * M // a.chunks) for c in range(a.chunks)]
    rows, ref_labels = [], []
    # ---- 1. breakdown (each chunk pays its own load: P6's pool gives the two chunks to different workers)
    for i, f in enumerate(files):
        r = {"file": f"{f.parent.name}/{f.name}", "xz_bytes": f.stat().st_size}
        labs = []
        for c, sl in enumerate(cuts):
            t0 = time.process_time(); b = f.read_bytes(); t1 = time.process_time()
            raw = lzma.decompress(b); t2 = time.process_time()
            obj = pickle.loads(raw); t3 = time.process_time()
            parts = tuple(obj["parts"]) if isinstance(obj, dict) and obj.get("format") == OL.PARTS_FORMAT else OL.load_parts(f)
            p8 = PT.executed_plans8(props[i, sl], repair=True); t4 = time.process_time()
            lab = OL.labels_from_targets(PT.score_plans(parts, p8)); t5 = time.process_time()
            labs.append(lab)
            for k, v in (("read", t1 - t0), ("lzma", t2 - t1), ("unpickle", t3 - t2), ("plans8", t4 - t3), ("score", t5 - t4)):
                r[k] = r.get(k, 0.0) + v
        r["raw_bytes"] = len(raw)
        r["total"] = sum(r[k] for k in ("read", "lzma", "unpickle", "plans8", "score"))
        ref_labels.append(np.concatenate(labs, 0))
        rows.append(r)
        os.makedirs(Path(a.raw_dir) / f.parent.name, exist_ok=True)
        (Path(a.raw_dir) / f.parent.name / f.name[:-3]).write_bytes(raw)          # <token>.pkl (decompressed)
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("read", "lzma", "unpickle", "plans8", "score", "total", "xz_bytes", "raw_bytes")}
    print("BREAKDOWN per set (CPU-s, chunks=%d):" % a.chunks, json.dumps(agg))
    # ---- 2. the unmodified _job, as the pool runs it, must give the same labels; cProfile it
    OL._W["lru"].clear()
    pr = cProfile.Profile()
    t0 = time.process_time()
    pr.enable()
    job_labels = []
    for i, f in enumerate(files):
        meta = {"log_name": f.parent.name, "token": f.name[:-len(".pkl.xz")], "aug": None}
        outs = []
        for sl in cuts:
            OL._W["lru"].clear()                                    # the sibling chunk runs on ANOTHER worker in P6: no LRU hit
            outs.append(OL._job("pdm", meta, props[i, sl])["labels"])
        job_labels.append(np.concatenate(outs, 0))
    pr.disable()
    t_job = (time.process_time() - t0) / len(files)
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(35)
    prof_txt = s.getvalue()
    same_job = all(np.array_equal(x, y) for x, y in zip(job_labels, ref_labels))
    print(f"_job per set {t_job:.4f} CPU-s; labels == breakdown path: {same_job}")
    # ---- 3a. raw (decompressed) parts: same bytes -> same objects -> same labels
    t0 = time.process_time(); raw_labels = []
    for i, f in enumerate(files):
        outs = []
        for sl in cuts:
            obj = pickle.loads((Path(a.raw_dir) / f.parent.name / f.name[:-3]).read_bytes())
            parts = tuple(obj["parts"]) if isinstance(obj, dict) and obj.get("format") == OL.PARTS_FORMAT else None
            outs.append(OL.labels_from_targets(PT.score_plans(parts, PT.executed_plans8(props[i, sl], repair=True))))
        raw_labels.append(np.concatenate(outs, 0))
    t_raw = (time.process_time() - t0) / len(files)
    # ---- 3b. colocated chunks (one load per set, both chunks scored)
    t0 = time.process_time(); colo_labels = []
    for i, f in enumerate(files):
        OL._W["lru"].clear()
        meta = {"log_name": f.parent.name, "token": f.name[:-len(".pkl.xz")], "aug": None}
        colo_labels.append(np.concatenate([OL._job("pdm", meta, props[i, sl])["labels"] for sl in cuts], 0))
    t_colo = (time.process_time() - t0) / len(files)
    # ---- 3c. chunks=1 (one job per set)
    t0 = time.process_time(); one_labels = []
    for i, f in enumerate(files):
        OL._W["lru"].clear()
        meta = {"log_name": f.parent.name, "token": f.name[:-len(".pkl.xz")], "aug": None}
        one_labels.append(OL._job("pdm", meta, props[i])["labels"])
    t_one = (time.process_time() - t0) / len(files)
    res = {"n_sets": len(files), "chunks": a.chunks, "breakdown_mean_cpu_s": agg, "job_cpu_s_per_set": t_job,
           "raw_parts_cpu_s_per_set": t_raw, "raw_bit_identical": all(np.array_equal(x, y) for x, y in zip(raw_labels, ref_labels)),
           "colocated_chunks_cpu_s_per_set": t_colo, "colocated_bit_identical": all(np.array_equal(x, y) for x, y in zip(colo_labels, ref_labels)),
           "chunks1_cpu_s_per_set": t_one, "chunks1_bit_identical": all(np.array_equal(x, y) for x, y in zip(one_labels, ref_labels)),
           "job_matches_breakdown": same_job, "rows": rows}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print(prof_txt[:6000])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        with open(a.out + ".prof.txt", "w") as f:
            f.write(prof_txt)


if __name__ == "__main__":
    main()
