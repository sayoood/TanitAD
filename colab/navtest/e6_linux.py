#!/usr/bin/env python3
"""SPEC_NAVTEST E-6 on Linux: score EVERY proposal of one proposal dump with W3's unchanged harness.

The Linux twin of `eval/proposal_table.py` stages 2-3 (which call `score_navtest_refe.py`, a Windows launcher). The
stage-2 seams, labels, the CSV reader, the table fields and G2/G3 are the SAME code shape; the only change is the
launcher (`linux_run_v1.py score`) and that the dump / landed seam / landed CSV are passed explicitly instead of
being derived from the dev box's DATA tree.

    python e6_linux.py --dump proposals.npz --name sub200_ep016 --tokens toks.json --work <dir> --paths paths.json \
        [--workers 8] [--landed-seam seam.npz] [--landed-csv seam.csv] [--windowspath-shim] [--only 0 1]

Writes <work>/table.npz (pdms [N,M], sub [N,M,6], valid, logits, proposals, pick, sub_names, head_order, rule) and
<work>/gates.json; per-proposal harness outputs under <work>/score/refe_<name>_pNN/. Resumable: a proposal whose
counts.json says PASS is not re-run. Prints ZZE6_OK <name> <N> <M> or ZZE6_FAIL <name> <why>.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
HEAD_ORDER = ("NC", "DAC", "EP", "TTC", "C", "DDC")


def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-name", default="navtest")
    ap.add_argument("--windowspath-shim", action="store_true")
    ap.add_argument("--landed-seam", default=None)
    ap.add_argument("--landed-csv", default=None)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(a.work, "logs"), exist_ok=True)
    os.makedirs(os.path.join(a.work, "seams"), exist_ok=True)
    D = np.load(a.dump)
    tok = [str(t) for t in D["token"]]
    P, L, pick = D["proposals"], D["logits"], D["pick"]
    N, M = P.shape[0], P.shape[1]
    gates = {}
    dmax = None
    if a.landed_seam and os.path.exists(a.landed_seam):
        S0 = np.load(a.landed_seam)
        assert [str(t) for t in S0["token"]] == tok, "token order differs between the landed seam and the dump"
        dmax = np.abs(S0["poses"].astype(np.float64) - P[np.arange(N), pick].astype(np.float64)).reshape(N, -1).max(1)
        gates["G1"] = {"landed_seam": a.landed_seam, "max_abs_pose_diff": float(dmax.max()),
                       "tokens_identical": int((dmax == 0).sum()), "pass": bool((dmax <= 1e-4).all())}
    samp = D["sampling"]

    def seam_k(k):
        return os.path.join(a.work, "seams", f"refe_{a.name}_p{k:02d}.npz")

    def label_k(k):
        return f"refe_{a.name}_p{k:02d}"

    def counts_k(k):
        return os.path.join(a.work, "score", label_k(k), f"{label_k(k)}.counts.json")

    def csv_k(k):
        return os.path.join(a.work, "score", label_k(k), f"{label_k(k)}.csv")

    for k in range(M):
        if not os.path.exists(seam_k(k)):
            np.savez(seam_k(k), token=D["token"], fingerprint=D["fingerprint"], poses=P[:, k].astype(np.float32),
                     sampling=samp, arm=np.array(f"REFe_{a.name}_p{k:02d}"))

    def status(k):
        try:
            return json.load(open(counts_k(k), encoding="utf-8"))
        except Exception:                                  # noqa: BLE001
            return None

    def score_one(k):
        st = status(k)
        if st and st.get("status") == "PASS":
            return k, st, 0.0
        cmd = [sys.executable, os.path.join(HERE, "linux_run_v1.py"), "score", "--label", label_k(k),
               "--arm", f"SEAM:{seam_k(k)}", "--tokens", a.tokens, "--out", os.path.join(a.work, "score"),
               "--cache-name", a.cache_name, "--paths", a.paths] + (["--windowspath-shim"] if a.windowspath_shim else [])
        t = time.time()
        with open(os.path.join(a.work, "logs", f"p{k:02d}.log"), "w") as fh:
            subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        return k, status(k), time.time() - t

    ks = list(range(M)) if a.only is None else list(a.only)
    st_all, secs = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for i, (k, st, dt) in enumerate(ex.map(score_one, ks)):
            st_all[k], secs[k] = st, round(dt, 1)
            if (i + 1) % 8 == 0 or i + 1 == len(ks):
                npass = sum(1 for s in st_all.values() if s and s.get("status") == "PASS")
                print(f"    scored {i + 1}/{len(ks)} pass {npass} {time.time() - t0:.0f} s", flush=True)
    bad = [k for k in ks if not (st_all.get(k) and st_all[k].get("status") == "PASS"
                                 and st_all[k].get("csv_valid_rows") == N)]
    gates["G3"] = {"runs": len(ks), "pass": len(ks) - len(bad), "failed": bad[:20], "ok": not bad,
                   "run_seconds": secs}
    if a.only is not None or bad:
        json.dump({"name": a.name, "N": N, "M": M, "gates": gates, "seconds": round(time.time() - t0, 1)},
                  open(os.path.join(a.work, "gates.json"), "w"), indent=1)
        print(f"ZZE6_{'SMOKE' if a.only is not None and not bad else 'FAIL'} {a.name} {len(ks) - len(bad)}/{len(ks)}")
        return 0 if not bad else 1
    pdms = np.full((N, M), np.nan)
    sub = np.full((N, M, 6), np.nan)
    valid = np.zeros((N, M), dtype=bool)
    for k in range(M):
        rows = read_csv(csv_k(k))
        for i, t in enumerate(tok):
            v, s, c = rows[t]
            valid[i, k], pdms[i, k], sub[i, k] = v, s, c
    if a.landed_csv and os.path.exists(a.landed_csv) and dmax is not None:
        lr = read_csv(a.landed_csv)
        same = dmax == 0
        dd = np.array([abs(pdms[i, pick[i]] - lr[t][1]) for i, t in enumerate(tok)])
        gates["G2"] = {"landed_csv": a.landed_csv, "tokens_compared": int(same.sum()),
                       "max_abs_score_diff_identical_poses": float(dd[same].max()) if same.any() else None,
                       "pass": bool(same.any() and dd[same].max() <= 1e-9)}
    np.savez(os.path.join(a.work, "table.npz"), token=np.array(tok), pdms=pdms, sub=sub, valid=valid, logits=L,
             proposals=P, pick=pick, sub_names=np.array(SUB), head_order=np.array(HEAD_ORDER),
             rule=np.array(str(D["rule"]) if "rule" in D.files else "v2_shape"))
    json.dump({"name": a.name, "dump": a.dump, "tokens": a.tokens, "N": N, "M": M, "workers": a.workers,
               "gates": gates, "seconds": round(time.time() - t0, 1)},
              open(os.path.join(a.work, "gates.json"), "w"), indent=1)
    ok = gates["G3"]["ok"] and all(g.get("pass") is not False for k_, g in gates.items() if k_ in ("G1", "G2"))
    print(f"ZZE6_{'OK' if ok else 'FAIL'} {a.name} {N} {M} {time.time() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
