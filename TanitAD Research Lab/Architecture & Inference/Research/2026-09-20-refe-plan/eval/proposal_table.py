#!/usr/bin/env python3
"""SPEC_NAVTEST E-6 (AMENDMENT 3): score EVERY proposal of one checkpoint on a token subset.

    python eval/proposal_table.py --ckpt <snap.pt> --name sub200_ep011 --tokens <subset.json> [--workers 6]

1 dump   the seam re-run with --dump-proposals (driverl venv, GPU): all M proposals on NAVSIM's grid
         and the scorer's raw six logits, from ONE forward pass per token
   G1    the re-run's seam against the LANDED seam `seams/refe_<name>.npz` (per-token max |d pose|)
2 score  one single-proposal seam per proposal index k, each scored by `score_navtest_refe.py`
         UNCHANGED -- W3's harness, so EP is normalised against the PDM-Closed reference exactly as
         for a real submission and guards C1-C3 run on every column of the table
   G3    all M runs PASS with every token valid
3 table  PDMS + the six sub-scores per (token, proposal) -> proptable/<name>/table.npz
   G2    the table at the pick reproduces the landed per-token score wherever the poses are equal

Resumable: a stage whose output exists and passed is not re-run. The readout is
`eval/selection_readout.py`. Prints ZZPROPTABLE_OK <name> <N> <M> or ZZPROPTABLE_FAIL <name> <why>.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_checkpoint as EC                          # noqa: E402  (one source for venv + env)

SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
# the scorer head's component order (refe/planner.py aggregate): NC, DAC, EP, TTC, C, DDC -- the
# SAME order as SUB, asserted by name below so a reorder on either side cannot pass silently
HEAD_ORDER = ("NC", "DAC", "EP", "TTC", "C", "DDC")
SUB_SHORT = dict(zip(SUB, HEAD_ORDER))


def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]),
                            tuple(float(r[c]) for c in SUB))
    return rows


def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--frames", default=f"{EC.DATA}/frames")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="score only these indices (smoke)")
    ap.add_argument("--reuse-dump", action="store_true",
                    help="the dump was written by the landed eval's own seam run (eval_checkpoint.py); "
                         "skip the re-run -- G1 then checks the dump against that seam")
    a = ap.parse_args()
    assert tuple(SUB_SHORT[c] for c in SUB) == HEAD_ORDER
    t0 = time.time()
    wd = os.path.join(EC.DATA, "proptable", a.name)
    os.makedirs(os.path.join(wd, "logs"), exist_ok=True)
    dump = os.path.join(wd, "proposals.npz")
    rerun_seam = os.path.join(EC.DATA, "seams", f"refe_{a.name}_dumprun.npz")
    landed_seam = os.path.join(EC.DATA, "seams", f"refe_{a.name}.npz")
    gates = {}

    # 1 dump -------------------------------------------------------------------------------------
    dlog = os.path.join(wd, "logs", "1_dump.log")
    reuse = a.reuse_dump and os.path.exists(dump)
    if not reuse and not (os.path.exists(dump) and os.path.exists(dlog)
                          and "ZZSEAM_OK" in open(dlog, encoding="utf-8", errors="replace").read()):
        rc, txt = EC.run([EC.DRIVERL_PY, "refe_navtest_seam.py", "--ckpt", a.ckpt, "--frames", a.frames,
                          "--tokens", a.tokens, "--out", rerun_seam, "--arm", f"REFe_{a.name}_dumprun",
                          "--dump-proposals", dump], HERE, EC.env_driverl(), dlog)
        if "ZZSEAM_OK" not in txt or not os.path.exists(dump):
            print(f"ZZPROPTABLE_FAIL {a.name} dump (see {dlog})"); return 1
    D = np.load(dump)
    tok = [str(t) for t in D["token"]]
    P, L, pick = D["proposals"], D["logits"], D["pick"]
    N, M = P.shape[0], P.shape[1]
    print(f"  dump: N={N} M={M} poses {P.shape} logits {L.shape}  {time.time() - t0:.0f} s", flush=True)
    # G1 -- the re-run against the landed seam
    if os.path.exists(landed_seam):
        S0 = np.load(landed_seam)
        t0_ = [str(t) for t in S0["token"]]
        assert t0_ == tok, "token order differs between the landed seam and the re-run"
        dmax = np.abs(S0["poses"].astype(np.float64) - P[np.arange(N), pick].astype(np.float64)).reshape(N, -1).max(1)
        gates["G1"] = {"landed_seam": landed_seam, "max_abs_pose_diff": float(dmax.max()),
                       "tokens_identical": int((dmax == 0).sum()), "tokens_differ_gt_1e-4": int((dmax > 1e-4).sum()),
                       "pass": bool((dmax <= 1e-4).all())}
    else:
        dmax = None
        gates["G1"] = {"landed_seam": None, "pass": None, "note": "no landed seam to compare"}
    print(f"  G1 {gates['G1']}", flush=True)

    # 2 score ------------------------------------------------------------------------------------
    def seam_k(k):
        return os.path.join(EC.DATA, "seams", "proptable", a.name, f"refe_{a.name}_p{k:02d}.npz")

    def label_k(k):
        return f"refe_{a.name}_p{k:02d}"

    def csv_k(k):
        return f"{EC.DATA}/score/{label_k(k)}/{label_k(k)}.csv"

    os.makedirs(os.path.dirname(seam_k(0)), exist_ok=True)
    samp = D["sampling"]
    for k in range(M):
        if not os.path.exists(seam_k(k)):
            np.savez(seam_k(k), token=D["token"], fingerprint=D["fingerprint"],
                     poses=P[:, k].astype(np.float32), sampling=samp,
                     arm=np.array(f"REFe_{a.name}_p{k:02d}"))

    def score_one(k):
        log = os.path.join(wd, "logs", f"p{k:02d}.log")
        if os.path.exists(csv_k(k)) and os.path.exists(log):
            st = status_of(open(log, encoding="utf-8", errors="replace").read())
            if st and st.get("status") == "PASS":
                return k, st
        rc, txt = EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", label_k(k), "--seam", seam_k(k),
                          "--tokens", a.tokens, "--out", f"{EC.DATA}/score"],
                         HERE, dict(os.environ, PYTHONIOENCODING="utf-8"), log)
        return k, status_of(txt)

    ks = list(range(M)) if a.only is None else list(a.only)
    t1 = time.time()
    status = {}
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for i, (k, st) in enumerate(ex.map(score_one, ks)):
            status[k] = st
            if (i + 1) % 8 == 0 or i + 1 == len(ks):
                npass = sum(1 for s in status.values() if s and s.get("status") == "PASS")
                print(f"    scored {i + 1}/{len(ks)}  pass {npass}  {time.time() - t1:.0f} s", flush=True)
    bad = [k for k in ks if not (status.get(k) and status[k].get("status") == "PASS"
                                 and status[k].get("csv_valid_rows") == N)]
    gates["G3"] = {"runs": len(ks), "pass": len(ks) - len(bad), "failed": bad[:20], "ok": not bad}
    print(f"  G3 {gates['G3']}", flush=True)
    if a.only is not None:
        print(f"ZZPROPTABLE_SMOKE {a.name} {len(ks) - len(bad)}/{len(ks)}"); return 0 if not bad else 1
    if bad:
        print(f"ZZPROPTABLE_FAIL {a.name} G3 {bad[:10]}"); return 1

    # 3 table ------------------------------------------------------------------------------------
    pdms = np.full((N, M), np.nan)
    sub = np.full((N, M, 6), np.nan)
    valid = np.zeros((N, M), dtype=bool)
    for k in range(M):
        rows = read_csv(csv_k(k))
        for i, t in enumerate(tok):
            v, s, c = rows[t]
            valid[i, k], pdms[i, k], sub[i, k] = v, s, c
    # G2 -- the table at the pick against the LANDED per-token score
    landed_csv = f"{EC.DATA}/score/refe_{a.name}/refe_{a.name}.csv"
    if os.path.exists(landed_csv) and dmax is not None:
        lr = read_csv(landed_csv)
        same = dmax == 0
        dd = np.array([abs(pdms[i, pick[i]] - lr[t][1]) for i, t in enumerate(tok)])
        gates["G2"] = {"landed_csv": landed_csv, "tokens_compared": int(same.sum()),
                       "max_abs_score_diff_identical_poses": float(dd[same].max()) if same.any() else None,
                       "max_abs_score_diff_all": float(dd.max()),
                       "pass": bool(same.any() and dd[same].max() <= 1e-9)}
    else:
        gates["G2"] = {"pass": None, "note": "no landed csv or no G1"}
    print(f"  G2 {gates['G2']}", flush=True)
    np.savez(os.path.join(wd, "table.npz"), token=np.array(tok), pdms=pdms, sub=sub, valid=valid,
             logits=L, proposals=P, pick=pick, sub_names=np.array(SUB), head_order=np.array(HEAD_ORDER))
    json.dump({"name": a.name, "ckpt": a.ckpt, "tokens": a.tokens, "N": N, "M": M, "gates": gates,
               "seconds": round(time.time() - t0, 1)}, open(os.path.join(wd, "gates.json"), "w"), indent=1)
    ok = all(g.get("pass") is not False for g in (gates["G1"], gates["G2"])) and gates["G3"]["ok"]
    print(f"ZZPROPTABLE_{'OK' if ok else 'FAIL'} {a.name} {N} {M}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
