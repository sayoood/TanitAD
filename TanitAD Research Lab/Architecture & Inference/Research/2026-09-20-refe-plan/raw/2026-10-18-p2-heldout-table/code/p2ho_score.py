#!/usr/bin/env python3
"""P2 held-out table: score every dumped proposal of every token with the PDM scorer that LABELLED the held-out sets (and P2's training sets).

Why this scorer and not W3's harness (the landed full-table pipeline scored navtest through W3's `run_v1.py cmd_score`): W3's harness needs the navtest metric cache and
OpenScene `navsim_logs/test`; a navtrain held-out token has neither on this box (`data/openscene/navsim_logs` holds `test` only).  The only held-out-capable instrument is
`refe/navsim_pdm_targets.py` = NAVSIM v1.1 @ 3e8291b's OWN MetricCacheProcessor steps + PDMSimulator + PDMScorer on a scenario rebuilt from the nuPlan DB.  It is the scorer that
produced the banked held-out labels and P2's 174,912 training labels, so a table scored with it is on the labels' instrument.  Its difference from W3's navtest instrument is
MEASURED (raw/2026-10-04-lane-discipline/validate_pdm_targets.json: 300 navtest tokens / 19,500 plans, PDMS exact on 94.73 %, mean |d| 0.0008, argmax-PDMS identical 100 %)
and is re-measured on P2's own landed proposals by p2ho_fidelity.py (control B).

Per log: scenarios (history_rows=40, future_rows=100: the labeller's call, onpolicy_relabel_pdm.py:47-48) -> metric_cache_parts -> score_plans(parts, proposals[64,8,3]).
Output order (nc, dac, ep, ttc, c, ddc) + pdms, exactly the labels' heads.  Resumable per log (parts/<log>.npz).

    navsim-crun python p2ho_score.py --dump <dump.npz> --plan <plan.json> --out <scores.npz> [--tree <PK>]
    env: PYTHONPATH=<navsim v1.1>;<tree>/refe  NUPLAN_MAPS_ROOT=.../navsim/data/maps  NUPLAN_MAP_VERSION=nuplan-maps-v1.0  REFE_SIM_HZ=10  OMP_NUM_THREADS=1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

SUBS = ("nc", "dac", "ep", "ttc", "c", "ddc")


def score_log(NS, PT, db, tokens, P8_by_tok, log):
    out = {}
    scs = {s.scenario_name: s for s in NS.build_scenarios_for_log(db, sorted(tokens), history_rows=40, future_rows=100)}
    for t in sorted(tokens):
        sc = scs.get(t)
        if sc is None:
            out[t] = None
            continue
        res = PT.score_plans(PT.metric_cache_parts(sc), P8_by_tok[t])
        out[t] = (np.stack([np.asarray(res[k], np.float64) for k in SUBS], -1), np.asarray(res["pdms"], np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", default="C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.tree, "refe"))
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    PT._setup()
    D = np.load(a.dump)
    plan = json.load(open(a.plan, encoding="utf-8"))
    tok = [str(t) for t in D["token"]]
    P8 = D["proposals"]
    assert P8.shape[1:] == (64, 8, 3), P8.shape
    by_log = {}
    for i, t in enumerate(tok):
        by_log.setdefault(plan["token_log"][t], []).append(t)
    idx = {t: i for i, t in enumerate(tok)}
    partdir = a.out[:-4] + ".parts"
    os.makedirs(partdir, exist_ok=True)
    t0, done = time.time(), 0
    for log in sorted(by_log):
        pp = os.path.join(partdir, f"{log}.npz")
        want = sorted(by_log[log])
        if os.path.exists(pp):
            z = np.load(pp)
            if sorted(str(x) for x in z["token"]) == want:
                print(f"  {log}: already scored ({len(want)})", flush=True)
                done += len(want)
                continue
        res = score_log(NS, PT, plan["db"][log], want, {t: P8[idx[t]] for t in want}, log)
        bad = [t for t in want if res[t] is None]
        if bad:
            print(f"  {log}: scenario builder skipped {len(bad)} tokens {bad[:3]}", flush=True)
        good = [t for t in want if res[t] is not None]
        tmp = pp[:-4] + f".tmp{os.getpid()}.npz"
        np.savez(tmp, token=np.array(good), sub=np.stack([res[t][0] for t in good]) if good else np.zeros((0, 64, 6)),
                 pdms=np.stack([res[t][1] for t in good]) if good else np.zeros((0, 64)), skipped=np.array(bad))
        os.replace(tmp, pp)
        done += len(want)
        print(f"  {log}: {len(good)}/{len(want)} scored, {done}/{len(tok)} total, {time.time() - t0:.0f} s", flush=True)
    # merge in the dump's token order
    sub = np.full((len(tok), 64, 6), np.nan)
    pdms = np.full((len(tok), 64), np.nan)
    for log in by_log:
        z = np.load(os.path.join(partdir, f"{log}.npz"))
        for j, t in enumerate(z["token"]):
            sub[idx[str(t)]], pdms[idx[str(t)]] = z["sub"][j], z["pdms"][j]
    valid = np.isfinite(sub).all((1, 2)) & np.isfinite(pdms).all(1)
    tmp = a.out[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, token=np.array(tok), sub=sub, pdms=pdms, valid_token=valid, sub_names=np.array(SUBS))
    os.replace(tmp, a.out)
    print(f"ZZP2HOSCORE_{'OK' if valid.all() else 'FAIL'} {int(valid.sum())}/{len(tok)} {time.time() - t0:.0f} s", flush=True)
    return 0 if valid.all() else 1


if __name__ == "__main__":
    sys.exit(main())
