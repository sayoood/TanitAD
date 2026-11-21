"""S1b chain driver (detached; idle between worker exits, no model turns).  Stages:
  1. IDENTITY  : 20 seeded logs of the 214 navtrain-in-val logs, ALL candidates, 3 single-thread workers  -> raw/identity/, then analyze_identity.py -> raw/identity_s1b.json
  2. SEQ SAMPLE: the 123 S1-provisional S3 logs, per-log seeded permutation, score until 4 pass the filter (cap 60, then cap 120 for unfilled logs if budget allows)
                 -> raw/seq/, then make_filtered_fetch_list.py -> raw/fetch_list_s2_s4_filtered.{json,txt}
Markers: state.json (stage, times), STAGE1_DONE, STAGE2_DONE, CHAIN_DONE, CHAIN_FAIL.  Kill by explicit PID only (pid in chain.pid).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = f"{ROOT}/code"
NV = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
TAN = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
S1 = "C:/Users/Admin/par18/unseen_s1"
W = 3
ID_SEED, ID_N = 20261019, 20
SEQ_SEED = 20261020
BELOW_NORMAL, NO_WINDOW = 0x00004000, 0x08000000


def state(**kw):
    p = f"{ROOT}/state.json"
    s = json.load(open(p)) if os.path.exists(p) else {}
    s.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(s, open(p, "w"), indent=1)


def lpt(logs, weights, w):
    bins = [[] for _ in range(w)]
    load = [0.0] * w
    for lg in sorted(logs, key=lambda l: -weights[l]):
        j = int(np.argmin(load))
        bins[j].append(lg)
        load[j] += weights[lg]
    return bins


def run_workers(mode, out, bins, tag, extra_args):
    os.makedirs(out, exist_ok=True)
    for attempt in range(8):
        procs = []
        for i, b in enumerate(bins):
            todo = [l for l in b if not os.path.exists(f"{out}/{l}.done")]
            if not todo:
                continue
            lf = f"{ROOT}/logs/{tag}_w{i}_a{attempt}.log"
            open(f"{ROOT}/logs/{tag}_w{i}.logs", "w").write("\n".join(todo))
            cmd = [NV, "-u", f"{CODE}/s1b_worker.py", "--mode", mode, "--logs", "@" + f"{ROOT}/logs/{tag}_w{i}.logs", "--out", out] + extra_args
            procs.append((i, subprocess.Popen(cmd, stdout=open(lf, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=BELOW_NORMAL | NO_WINDOW,
                                              env=dict(os.environ, PYTHONPATH="", PYTHONUNBUFFERED="1"))))
        if not procs:
            return True
        state(stage=tag, attempt=attempt, worker_pids=[p.pid for _, p in procs])
        rcs = [p.wait() for _, p in procs]
        print(tag, "attempt", attempt, "rcs", rcs, flush=True)
        if all(os.path.exists(f"{out}/{l}.done") for b in bins for l in b):
            return True
        if any(rc not in (0, 75) for rc in rcs):
            print("worker failure", rcs, flush=True)
    return all(os.path.exists(f"{out}/{l}.done") for b in bins for l in b)


def main():
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    open(f"{ROOT}/chain.pid", "w").write(str(os.getpid()))
    t0 = time.time()
    state(stage="start", started=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid())
    try:
        # ---- stage 1: identity ----------------------------------------------------------------------------------------------------
        per214 = json.load(open(f"{S1}/raw/identity_214_per_log.json"))
        names = sorted(per214)
        rng = np.random.default_rng(ID_SEED)
        sel = [names[i] for i in sorted(rng.choice(len(names), ID_N, replace=False).tolist())]
        json.dump({"seed": ID_SEED, "n": ID_N, "population": "214 navtrain-in-val logs (sorted names)", "logs": sel, "n_candidates": {l: per214[l]["n_cand"] for l in sel},
                   "n_known": {l: per214[l]["n_known"] for l in sel}}, open(f"{ROOT}/raw/identity_logs.json", "w"), indent=1)
        bins = lpt(sel, {l: per214[l]["n_cand"] for l in sel}, W)
        ok = run_workers("all", f"{ROOT}/raw/identity", bins, "identity", ["--extras"])
        if not ok:
            raise RuntimeError("identity workers did not finish")
        subprocess.run([TAN, "-u", f"{CODE}/analyze_identity.py"], stdout=open(f"{ROOT}/logs/analyze_identity.log", "w"), stderr=subprocess.STDOUT, check=True)
        open(f"{ROOT}/STAGE1_DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        state(stage="stage1_done", stage1_wall_s=round(time.time() - t0))
        # ---- stage 2: the filtered S3 sample (sequential acceptance) --------------------------------------------------------------------
        s3 = json.load(open(f"{ROOT}/raw/s3_provisional_logs.json"))["logs"]
        U = json.load(open(f"{S1}/raw/U_candidates_per_log.json"))["per_log"]
        wts = {l: min(60, U[l]["n_candidates"]) for l in s3}
        bins2 = lpt(s3, wts, W)
        out2 = f"{ROOT}/raw/seq"
        ok = run_workers("seq", out2, bins2, "seq1", ["--seed", str(SEQ_SEED), "--n-accept", "4", "--cap", "60"])
        if not ok:
            raise RuntimeError("seq pass 1 did not finish")
        # pass 2: the logs that did not reach 4 accepted get a larger cap, while the budget (2 CPU-h in total for the stage) allows
        cpu1 = 0.0
        unfilled = []
        for l in s3:
            d = json.load(open(f"{out2}/{l}.done"))
            cpu1 += d["cpu_s"]
            if d["n_accepted"] < 4:
                unfilled.append(l)
        state(stage="seq_pass1_done", seq1_cpu_s=round(cpu1), seq1_unfilled=len(unfilled))
        if unfilled and cpu1 < 5400:
            for l in unfilled:
                os.remove(f"{out2}/{l}.done")
            ok = run_workers("seq", out2, lpt(unfilled, {l: 1.0 for l in unfilled}, W), "seq2", ["--seed", str(SEQ_SEED), "--n-accept", "4", "--cap", "120"])
            if not ok:
                raise RuntimeError("seq pass 2 did not finish")
        subprocess.run([TAN, "-u", f"{CODE}/make_filtered_fetch_list.py"], stdout=open(f"{ROOT}/logs/make_filtered_fetch_list.log", "w"), stderr=subprocess.STDOUT, check=True)
        open(f"{ROOT}/STAGE2_DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        open(f"{ROOT}/CHAIN_DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        state(stage="done", total_wall_s=round(time.time() - t0))
        return 0
    except Exception as e:  # noqa: BLE001
        open(f"{ROOT}/CHAIN_FAIL", "w").write(f"{type(e).__name__}: {e}")
        state(stage="FAILED", error=str(e)[:300])
        raise


if __name__ == "__main__":
    sys.exit(main())
