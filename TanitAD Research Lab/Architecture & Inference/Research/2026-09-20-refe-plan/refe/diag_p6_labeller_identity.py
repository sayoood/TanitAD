#!/usr/bin/env python3
"""G-LIVE-LAB's instruments, runnable by hand on any box that has the NAVSIM stack + a D9 cache (SPEC_P6 section 5).

  identity    D1 (a): the ONLINE labeller's labels == the OFFLINE `onpolicy_relabel_pdm.label_log` body (fresh parts) on the SAME proposals, bit for bit, >= 64 sets;
              and the stale-proposal arm (REFE_LAB_MUTATE=stale) must be REFUSED
  throughput  sets/s of the pool at the profile's worker count on distinct tokens, vs the 1.5 x 4.49 sets/s bar (a 2-worker pool must FAIL it)
    python diag_p6_labeller_identity.py --cache C --db-root D --map-root M --tokens log:tok,... [--python <navsim venv python>] [--workers 4] [--n-sets 64]
Prints ZZLAB_IDENTITY_PASS / ZZLAB_THROUGHPUT <sets/s> and writes --out json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))
import online_labeller as OL  # noqa: E402

NEED_SETS_S = 4.49          # SPEC_P6 section 3.3
MARGIN = 1.5                # SPEC_P6 section 5 (G-LIVE-LAB)


def proposals(n_sets: int, seed: int = 0, m: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((n_sets, m, 20, 3))
    for s in range(n_sets):
        for k in range(m):
            v = 2.0 + 10.0 * rng.random()
            out[s, k, :, 0] = np.cumsum(np.full(20, v * 0.2))
            out[s, k, :, 1] = np.cumsum(rng.normal(0, 0.15, 20))
            out[s, k, :, 2] = np.cumsum(rng.normal(0, 0.01, 20))
    return out


def fresh_labels(db_root, map_root, log, token, props) -> np.ndarray:
    """The OFFLINE path: exactly onpolicy_relabel_pdm.label_log's body (fresh scenario build -> metric_cache_parts -> executed_plans8 -> score_plans)."""
    import navsim_pdm_targets as PT
    import build_metric_cache as BM
    sc = BM._scenarios(db_root, map_root, log, {token})[token]
    return OL.labels_from_targets(PT.score_plans(PT.metric_cache_parts(sc), PT.executed_plans8(props, repair=True)))


def run_labeller(cache, workers, python, items, props, stale=False):
    lab = OL.OnlineLabeller(backend="pdm", workers=workers, metric_cache=cache, python=python, n_prop=props.shape[1], timeout_s=600)
    if stale:
        lab.mutate = "stale"
    outs = []
    # warm-up (NOT timed): start every worker (each imports the NAVSIM stack, ~seconds) and warm its parts cache, so the throughput is the pool's steady state
    wm = [{"log_name": lg, "token": tk, "rank": 0, "aug": None, "step": 0} for lg, tk in items[:max(workers, 1)]]
    lab.collect(lab.submit(-1, 0, wm, props[:len(wm)]), 0)
    t0 = time.perf_counter()
    h = None
    for i in range(0, len(items), 4):                                    # micro-batches of 4 sets, like the trainer
        metas = [{"log_name": lg, "token": tk, "rank": 0, "aug": None, "step": 0} for lg, tk in items[i:i + 4]]
        prev = h
        h = lab.submit(i // 4, 0, metas, props[i:i + 4])
        if prev is not None:
            outs.append(lab.collect(prev, 0))
    outs.append(lab.collect(h, 0))
    wall = time.perf_counter() - t0
    lab.close()
    return outs, wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--db-root", default=None)
    ap.add_argument("--map-root", default=None)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--python", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-sets", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-fresh", action="store_true", help="do not build fresh parts (this interpreter has no NAVSIM stack); compare to --compare-labels instead")
    ap.add_argument("--save-labels", default=None)
    ap.add_argument("--compare-labels", default=None)
    a = ap.parse_args()
    base = [tuple(t.split(":", 1)) for t in a.tokens.split(",")]
    items = [base[i % len(base)] for i in range(a.n_sets)]
    props = proposals(a.n_sets)
    res = {"n_sets": a.n_sets, "distinct_tokens": len(set(items)), "workers": a.workers, "python": a.python or sys.executable}
    outs, wall = run_labeller(a.cache, a.workers, a.python, items, props)
    lab_online = np.concatenate([o[0] for o in outs])
    mask = np.concatenate([o[1] for o in outs])
    res["labelled"] = int(mask.sum())
    if a.save_labels:
        np.save(a.save_labels, lab_online)
    if a.skip_fresh:
        fresh = np.load(a.compare_labels)                                 # the labels run 1 proved identical to the offline path
    else:
        fresh = np.stack([fresh_labels(a.db_root, a.map_root, lg, tk, props[i]) for i, (lg, tk) in enumerate(items[: min(a.n_sets, 64)])])
    n = fresh.shape[0]
    res["identity_vs"] = "saved labels of the fresh-verified run" if a.skip_fresh else "fresh parts (onpolicy_relabel_pdm.label_log body)"
    res["identity_equal_sets"] = int(sum(np.array_equal(lab_online[i], fresh[i]) for i in range(n)))
    res["identity_ok"] = bool(res["identity_equal_sets"] == n and n >= min(a.n_sets, 64) and mask[:n].all())
    res["throughput_sets_s"] = round(a.n_sets / wall, 3)
    res["throughput_bar_sets_s"] = round(MARGIN * NEED_SETS_S, 3)
    res["throughput_ok"] = bool(res["throughput_sets_s"] >= res["throughput_bar_sets_s"])
    try:                                                                  # D1 (c): a one-step-stale proposal must be refused
        run_labeller(a.cache, 2, a.python, items[:8], props[:8], stale=True)
        res["stale_refused"] = False
    except RuntimeError as e:
        res["stale_refused"] = "IDENTITY VIOLATED" in str(e)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("ZZLAB_IDENTITY_PASS" if res["identity_ok"] and res["stale_refused"] else "ZZLAB_IDENTITY_FAIL")
    print(f"ZZLAB_THROUGHPUT {res['throughput_sets_s']}")
    return 0 if res["identity_ok"] and res["stale_refused"] else 1


if __name__ == "__main__":
    sys.exit(main())
