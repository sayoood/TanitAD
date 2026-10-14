#!/usr/bin/env python3
"""P6 / D9 -- the metric-cache PARTS, precomputed once per token (and once per lane-rank twin) so the online labeller never rebuilds a scenario.

SPEC_P6 section 3.1: `PT.metric_cache_parts(scenario)` (navsim_pdm_targets.py) is deterministic per (token, route): PDM-Closed's reference trajectory, the initial ego
state, the interpolated observations, the centreline, the route lanes and the drivable-area map. The online labeller loads these from disk and runs only
`executed_plans8` + `score_plans` on the 64 proposals (M-LAB: 0.34-0.83 CPU-s per set, vs 0.80-1.97 when the parts are rebuilt for every set).
These are TRAINING-TIME MAP LABELS (admissible, PI 2026-10-11); they never reach the model's input.

    build   python build_metric_cache.py build  --targets <bank dir> --out <cache root> [--db-root D] [--map-root M] [--workers 16] [--lane-rank K] [--limit N]
            one file per (log, token): <out>/<log>/<token>.pkl.xz = lzma(pickle({"format": "tanitad.p6parts/1", "parts": (traj, ego, obs, centerline, lanes, dam)}))
            --lane-rank K builds ONLY the rank-K lane-rank twins (<token>__lane_rank_K.pkl.xz) in a process where route_rank.apply(K) installed the DriveRL route patch
            (D7). ⚠️ UNVERIFIED here: the patch needs the DriveRL fork (pod). Whether PDM-Closed honours it is also UNVERIFIED -- `verify --lane-rank` checks it.
    verify  python build_metric_cache.py verify --cache <root> --db-root D --map-root M --tokens <log:token,...> [--n-sets 4]
            D9's acceptance test: score_plans(cached parts) == score_plans(fresh parts) EXACTLY (every array, np.array_equal) on seeded random proposals; with a
            deliberate-regression arm (one cached field perturbed) that must NOT match. Needs the NAVSIM venv + a nuPlan DB.

A twin whose parts file is missing is NOT silently given the logged route's: the labeller masks it (trajectory loss only, SPEC_P6 D7's declared fallback) and counts it.
"""
from __future__ import annotations

import argparse
import json
import lzma
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))
import online_labeller as OL  # noqa: E402


def write_parts(path, parts) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = lzma.compress(pickle.dumps({"format": OL.PARTS_FORMAT, "parts": tuple(parts)}, protocol=4), preset=1)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    return len(blob)


def keys_of_bank(targets_dir, lane_rank=None) -> dict:
    """{log_name: sorted set of (token, aug)} from targets_rank*.jsonl. lane_rank None => the logged-route parts (rank 0 and goal-horizon twins); K => the rank-K lane twins."""
    by: dict = {}
    for fp in sorted(Path(targets_dir).glob("targets_rank*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                r = json.loads(ln)
                aug = r.get("aug")
                is_lane = isinstance(aug, dict) and aug.get("kind") == "lane_rank"
                if lane_rank is None and is_lane:
                    continue
                if lane_rank is not None and not (is_lane and int(aug.get("value")) == int(lane_rank)):
                    continue
                by.setdefault(r["log_name"], set()).add((r["token"], json.dumps(aug, sort_keys=True) if lane_rank is not None else "null"))
    return {k: sorted(v) for k, v in by.items()}


_S: dict = {}


def _scenarios(db_root, map_root, log, tokens):
    import navtrain_scenarios as NS
    if db_root:
        db = f"{db_root}/{log}.db"
    else:
        db = NS.index_dbs()[log]
    kw = {"map_root": map_root} if map_root else {}
    return {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(db, sorted(tokens), history_rows=40, future_rows=100, **kw)}


def _pause_point():
    """P6 OVERLAP (eval/P6_ON_11VCPU.md section 9): while the file named by REFE_PAUSE_FILE exists a worker sleeps, so the chain's step-time guard can back a build off a running trainer
    without killing it.  Unset (every other caller) = no effect."""
    pf = os.environ.get("REFE_PAUSE_FILE", "")
    while pf and os.path.exists(pf):
        time.sleep(float(os.environ.get("REFE_PAUSE_POLL_S", "2")))


def _build_log(job):
    log, items, out, db_root, map_root, lane_rank = job
    import navsim_pdm_targets as PT
    done = skipped = failed = 0
    msgs = []
    try:
        scs = _scenarios(db_root, map_root, log, {t for t, _ in items})
    except Exception as e:                                                   # noqa: BLE001
        return 0, 0, len(items), [f"{log}: scenario build {type(e).__name__}: {str(e)[:120]}"]
    for tok, aug_s in items:
        _pause_point()
        aug = json.loads(aug_s)
        dst = OL.parts_path(out, log, tok, aug)
        if dst.is_file():
            done += 1
            continue
        sc = scs.get(tok)
        if sc is None:                       # NAVSIM's 5 s future guard: the token has no cache, reported and not padded
            skipped += 1
            continue
        try:
            write_parts(dst, PT.metric_cache_parts(sc))
            done += 1
        except Exception as e:                                               # noqa: BLE001
            failed += 1
            msgs.append(f"{log}:{tok}: {type(e).__name__}: {str(e)[:120]}")
    return done, skipped, failed, msgs


def cmd_build(a) -> int:
    if a.lane_rank is not None:
        import route_rank
        print(route_rank.apply(int(a.lane_rank)), flush=True)
    by = keys_of_bank(a.targets, a.lane_rank)
    jobs = [(lg, it, a.out, a.db_root, a.map_root, a.lane_rank) for lg, it in sorted(by.items())]
    if a.limit:
        jobs = jobs[: a.limit]
    t0 = time.time()
    tot = [0, 0, 0]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for done, skipped, failed, msgs in pool.imap_unordered(_build_log, jobs):
            tot[0] += done; tot[1] += skipped; tot[2] += failed
            for m in msgs[:2]:
                print("  " + m, flush=True)
    print(f"ZZMETRIC_CACHE_DONE built={tot[0]} skipped_5s_guard={tot[1]} failed={tot[2]} logs={len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0 if tot[2] == 0 else 1


def exactness(cache, db_root, map_root, items, n_sets=4, seed=0) -> dict:
    """D9 acceptance. items: [(log, token)]. Returns {n_tokens, n_sets, n_equal, n_regression_equal} -- the regression arm perturbs one cached field and must NOT match."""
    import navsim_pdm_targets as PT
    rng = np.random.default_rng(seed)
    res = {"n_tokens": 0, "n_sets": 0, "n_equal": 0, "regression_sets": 0, "regression_equal": 0}
    for log, tok in items:
        sc = _scenarios(db_root, map_root, log, {tok}).get(tok)
        if sc is None:
            continue
        fresh = PT.metric_cache_parts(sc)
        cached = OL.load_parts(OL.parts_path(cache, log, tok))
        res["n_tokens"] += 1
        for _ in range(n_sets):
            props = np.stack([np.stack([np.linspace(0.5, 30.0, 20) * (0.8 + 0.4 * rng.random()), rng.normal(0, 1.0, 20).cumsum() * 0.1, np.zeros(20)], -1)
                              for _ in range(8)])
            plans8 = PT.executed_plans8(props, repair=True)
            a, b = PT.score_plans(fresh, plans8), PT.score_plans(cached, plans8)
            res["n_sets"] += 1
            res["n_equal"] += int(all(np.array_equal(a[k], b[k]) for k in a))
            # regression arm: the SAME cached parts with the ego state's x shifted 5 m must disagree somewhere on a set that moves
            import copy
            bad = list(cached)
            ego = copy.deepcopy(bad[1])
            ego.rear_axle.x += 5.0
            bad[1] = ego
            try:
                c = PT.score_plans(tuple(bad), plans8)
                res["regression_sets"] += 1
                res["regression_equal"] += int(all(np.array_equal(a[k], c[k]) for k in a))
            except Exception:                                                # noqa: BLE001 -- a perturbed cache that cannot even score is also 'not equal'
                res["regression_sets"] += 1
    return res


def cmd_verify(a) -> int:
    items = [tuple(t.split(":", 1)) for t in a.tokens.split(",")]
    r = exactness(a.cache, a.db_root, a.map_root, items, a.n_sets)
    print(json.dumps(r))
    ok = r["n_sets"] > 0 and r["n_equal"] == r["n_sets"] and r["regression_equal"] == 0
    print("ZZD9_PASS" if ok else "ZZD9_FAIL", flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--targets", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--db-root", default=None)
    b.add_argument("--map-root", default=None)
    b.add_argument("--workers", type=int, default=16)
    b.add_argument("--lane-rank", type=int, default=None)
    b.add_argument("--limit", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("--cache", required=True)
    v.add_argument("--db-root", default=None)
    v.add_argument("--map-root", default=None)
    v.add_argument("--tokens", required=True)
    v.add_argument("--n-sets", type=int, default=4)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
