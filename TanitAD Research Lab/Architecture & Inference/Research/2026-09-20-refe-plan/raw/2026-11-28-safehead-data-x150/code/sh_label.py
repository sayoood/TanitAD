#!/usr/bin/env python3
"""CPU stage: TRACKED NAVSIM v1.1 PDM labels for all 64 proposals of every dumped token (navsim venv; NAVSIM's own simulator + scorer).

Same call the programme's labeller makes (onpolicy_relabel_pdm.label_log): scenario at the token with history 40 / future 100 rows
(navtrain_scenarios.build_scenarios_for_log) -> navsim_pdm_targets.metric_cache_parts -> score_plans(parts, plans8) where plans8 are the
EXECUTED plans on NAVSIM's [64, 8, 3] grid = the dump's `props8` (A7 last-heading repair, to_navsim) -- the very poses the harness scores for a pick.
LQR tracking is inside PDMSimulator.simulate_proposals (the "tracked" frame), not a separate step.

    sh_label.py --dump <log>.dump.npz --db <log>.db --out <log>.lab.npz [--workers 6] [--only-first N]
Output: token [n], tr_nc/tr_dac/tr_ddc/tr_ttc/tr_ep/tr_c/tr_pdms float32 [n,64], plus failed tokens in <out>.report.json (never silently dropped).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh_lib as L  # noqa: E402

_S: dict = {}


def work(job):
    log, db, items = job                              # items: [(token, props8 [64,8,3])]
    import navsim_pdm_targets as PT
    import navtrain_scenarios as NS
    out, fails = [], []
    try:
        scs = {sc.scenario_name: sc for sc in NS.build_scenarios_for_log(db, [t for t, _ in items], map_root=L.MAPS, history_rows=L.HIST_ROWS, future_rows=L.FUT_ROWS)}
    except Exception as e:                            # noqa: BLE001
        return [], [(t, f"scenario build {type(e).__name__}: {str(e)[:120]}") for t, _ in items]
    for tok, p8 in items:
        sc = scs.get(tok)
        if sc is None:
            fails.append((tok, "scenario builder skipped the token (5 s future guard)"))
            continue
        try:
            t = PT.score_plans(PT.metric_cache_parts(sc), np.asarray(p8, np.float64))
            out.append((tok, {k: np.asarray(t[v], np.float32) for k, v in L.LAB_MAP.items()}))
        except Exception:                             # noqa: BLE001
            fails.append((tok, traceback.format_exc()[-300:]))
    return out, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=6)
    a = ap.parse_args()
    d = np.load(a.dump)
    toks, p8 = [str(t) for t in d["token"]], d["props8"]
    log = os.path.basename(a.dump).split(".dump.npz")[0]
    jobs = [(log, a.db, [(toks[i], p8[i]) for i in range(j, min(j + a.chunk, len(toks)))]) for j in range(0, len(toks), a.chunk)]
    t0 = time.time()
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(work, jobs, chunksize=1)
    else:
        res = [work(j) for j in jobs]
    got, fails = {}, []
    for o, f in res:
        for tok, v in o:
            got[tok] = v
        fails += f
    if fails and a.workers > 1:                       # RAM-pressure failures (ArrayMemoryError under parallel workers) get ONE serial retry; real failures stay failures
        ix = {t: i for i, t in enumerate(toks)}
        retry = [(t, p8[ix[t]]) for t, _ in fails]
        o2, f2 = work((log, a.db, retry))
        for tok, v in o2:
            got[tok] = v
        print(f"retry: {len(o2)}/{len(retry)} recovered", flush=True)
        fails = f2
    keep = [t for t in toks if t in got]
    arrs = {"token": np.array(keep)}
    for k in L.LAB_KEYS:
        arrs[k] = np.stack([got[t][k] for t in keep]) if keep else np.zeros((0, L.N_PROP), np.float32)
    L.save_npz_det(a.out, **arrs)
    json.dump({"log": log, "asked": len(toks), "labelled": len(keep), "failed": fails, "seconds": round(time.time() - t0, 1)}, open(a.out + ".report.json", "w"), indent=0)
    print(f"ZZLABEL {log} labelled {len(keep)}/{len(toks)} failed {len(fails)} {time.time() - t0:.0f} s", flush=True)
    return 0 if keep else 1


if __name__ == "__main__":
    sys.exit(main())
