#!/usr/bin/env python3
"""Measure 5 effectiveness run: the v4 labellers on the dev box's CPUs, streaming behind the GPU forward.

N copies of the REAL CLI `refe/onpolicy_label_v4.py --slow-copies --slow-factors 0.75 --slow-frac 1.0` (C2, every
sample labelled with its copies; the arms choose per sample which set they use) poll <m5>/queue and write <m5>/sets.
The driver keeps N alive, and stops them -- by explicit PID, only when every chunk is `.done_*` (so no worker is
mid-sample) -- once the forward has written <m5>/queue/FORWARD_DONE_<split> for every split in --splits.

    python eval/m5_label.py --workers 12 [--splits train,val] [--scale-to 16 --scale-when-file F --scale-when-token T]

--scale-to: start more workers (new names m5w<N>..) once the file F contains the text T -- the coordinator's CPU-share
rule (2026-09-27): keep >= 4 logical cores free until M6's verdict exists, then go up.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_slow_labels_v4 as V  # noqa: E402  worker_env (the pod's thread discipline) and the labeller path

M5 = "D:/Projects/TanitAD/data/refe_m5"


def state(q):
    names = os.listdir(q)
    return (sum(1 for n in names if n.startswith("props_r0_") and n.endswith(".jsonl")),
            sum(1 for n in names if n.startswith("props_r0_") and ".jsonl.m5w" in n),
            sum(1 for n in names if n.startswith("props_r0_") and ".jsonl.done_" in n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--m5", default=M5)
    ap.add_argument("--scale-to", type=int, default=0)
    ap.add_argument("--scale-when-file", default="")
    ap.add_argument("--scale-when-token", default="")
    a = ap.parse_args()
    q, out, logs = os.path.join(a.m5, "queue"), os.path.join(a.m5, "sets"), os.path.join(a.m5, "logs")
    for d in (q, out, logs):
        os.makedirs(d, exist_ok=True)
    procs: dict = {}
    t0 = time.time()

    def launch(w):
        cmd = [V.EC.DRIVERL_PY, V.V4, "--queue", q, "--out", out, "--rank", "0", "--worker", f"m5w{w}",
               "--poll-s", "10", "--slow-copies", "--slow-factors", "0.75", "--slow-frac", "1.0"]
        procs[w] = V.run_cli(cmd, os.path.join(logs, f"label_m5w{w}.log"), V.worker_env())

    for w in range(a.workers):
        launch(w)
        time.sleep(4)                      # stagger the first claims (the exFAT rename race)
    need = [f"FORWARD_DONE_{s}" for s in a.splits.split(",") if s]
    last = 0.0
    scaled = not (a.scale_to > a.workers and a.scale_when_file and a.scale_when_token)
    while True:
        if not scaled:
            try:
                hit = a.scale_when_token in open(a.scale_when_file, encoding="utf-8", errors="replace").read()
            except OSError:
                hit = False
            if hit:
                for w in range(a.workers, a.scale_to):
                    launch(w)
                    time.sleep(4)
                print(f"  {time.strftime('%H:%M:%S')} SCALED to {a.scale_to} workers ('{a.scale_when_token}' seen)",
                      flush=True)
                scaled = True
        pend, claimed, done = state(q)
        for w, p in list(procs.items()):
            if p.poll() is not None:       # a worker that died: restart it (it resumes its own in-flight chunk)
                print(f"  worker m5w{w} exited rc {p.returncode}; restarting", flush=True)
                launch(w)
        fwd_done = all(os.path.exists(os.path.join(q, n)) for n in need)
        if time.time() - last > 120:
            n_sets = sum(sum(1 for _ in open(f, encoding="utf-8")) for f in glob.glob(os.path.join(out, "onpolicy_*.jsonl")))
            print(f"  {time.strftime('%H:%M:%S')} chunks pending {pend} claimed {claimed} done {done}; set lines "
                  f"{n_sets}; forward done {fwd_done}; {time.time() - t0:.0f} s", flush=True)
            last = time.time()
        if fwd_done and pend == 0 and claimed == 0:
            break
        time.sleep(15)
    for w, p in procs.items():
        if p.poll() is None:
            p.terminate()                  # idle: every chunk is .done_, so no sample is in flight
    for p in procs.values():
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
    st = [json.load(open(s)) for s in glob.glob(os.path.join(q, "status_r0_m5w*.json"))]
    tot = {k: sum(x.get(k, 0) for x in st) for k in ("written", "skipped_ndiff", "failed", "selfcheck_failed",
                                                      "slow_sets", "slow_copies")}
    json.dump({"workers": a.workers, "workers_final": len(procs), "seconds": round(time.time() - t0, 1), **tot},
              open(os.path.join(a.m5, "label_summary.json"), "w"), indent=1)
    print(f"ZZM5_LABEL_DONE {json.dumps(tot)} in {time.time() - t0:.0f} s")
    return 0 if tot["failed"] == 0 and tot["selfcheck_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
