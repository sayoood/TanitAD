#!/usr/bin/env python3
"""Measure 5: take over the running v4 labellers from eval/m5_label.py with FEWER workers (RAM rule, coordinator
2026-09-28 00:38: keep >= 5 GB of system RAM free; 16 workers held ~14.6 GB and left 3.1 GB).

Precondition: the m5_label.py driver has been stopped by explicit PID (so it no longer restarts workers); its worker
processes keep running. This script:
  1. retires the named workers: kills each one's launcher + interpreter by explicit PID (matched on `--worker <name>`
     in the command line) and RELEASES its claimed chunk (props_*.jsonl.<name> -> props_*.jsonl) so a kept worker
     claims it. A released chunk is labelled again from its first sample by the claimant: the retired worker's
     partial lines stay in its own file, duplicates carry the same key, and m5_finetune_eval.data keeps one line per key;
  2. supervises the kept workers: a kept worker that dies has its chunk released and is relaunched (the same CLI as
     m5_label.py, log appended, never overwritten);
  3. once FORWARD_DONE_<split> exists for every split and no chunk is pending or claimed, stops the kept workers by
     explicit PID and writes <m5>/label_summary.json -- from the STATUS files (failed / selfcheck_failed) AND from the
     SETS files (distinct keys labelled) against the keys in the chunks (coverage).

    python eval/m5_label_finish.py --retire m5w10,m5w11,m5w12,m5w13,m5w14,m5w15 [--splits train,val]
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
import validate_slow_labels_v4 as V  # noqa: E402

M5 = "D:/Projects/TanitAD/data/refe_m5"


def workers_by_name() -> dict:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
          "'onpolicy_label_v4' -and $_.CommandLine -match '--worker m5w' } | ForEach-Object { \"$($_.ProcessId) "
          "$($_.CommandLine)\" }")
    o = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    res: dict = {}
    for line in o.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        toks = parts[1].split()
        if "--worker" in toks:
            name = toks[toks.index("--worker") + 1].strip('"')
            res.setdefault(name, []).append(int(parts[0]))
    return res


def kill(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True)


def release(q: str, name: str) -> list:
    out = []
    for c in glob.glob(os.path.join(q, f"props_r0_*.jsonl.{name}")):
        dst = c[: -(len(name) + 1)]
        try:
            os.rename(c, dst)
            out.append(os.path.basename(dst))
        except OSError as e:
            print(f"  release {c} failed: {e}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retire", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--m5", default=M5)
    a = ap.parse_args()
    q, out, logs = os.path.join(a.m5, "queue"), os.path.join(a.m5, "sets"), os.path.join(a.m5, "logs")
    retire = [x for x in a.retire.split(",") if x]
    W = workers_by_name()
    print(f"  {time.strftime('%H:%M:%S')} live workers: {sorted(W)}", flush=True)
    for name in retire:
        for pid in sorted(W.get(name, []), reverse=True):
            kill(pid)
        time.sleep(2)
        print(f"  retired {name} (PIDs {W.get(name)}); released {release(q, name)}", flush=True)
    keep = sorted(n for n in W if n not in retire)
    need = [f"FORWARD_DONE_{s}" for s in a.splits.split(",") if s]
    t0, last = time.time(), 0.0

    def relaunch(name):
        cmd = [V.EC.DRIVERL_PY, V.V4, "--queue", q, "--out", out, "--rank", "0", "--worker", name, "--poll-s", "10",
               "--slow-copies", "--slow-factors", "0.75", "--slow-frac", "1.0"]
        lf = open(os.path.join(logs, f"label_{name}.log"), "a", encoding="utf-8")
        subprocess.Popen(cmd, cwd=V.REFE, env=V.worker_env(), stdout=lf, stderr=subprocess.STDOUT)

    while True:
        names = os.listdir(q)
        pend = sum(1 for n in names if n.startswith("props_r0_") and n.endswith(".jsonl"))
        claimed = sum(1 for n in names if n.startswith("props_r0_") and ".jsonl.m5w" in n)
        done = sum(1 for n in names if n.startswith("props_r0_") and ".jsonl.done_" in n)
        W = workers_by_name()
        for name in keep:
            if name not in W:
                rel = release(q, name)
                print(f"  {time.strftime('%H:%M:%S')} kept worker {name} is gone; released {rel}; relaunching", flush=True)
                relaunch(name)
        fwd = all(os.path.exists(os.path.join(q, n)) for n in need)
        if time.time() - last > 120:
            print(f"  {time.strftime('%H:%M:%S')} pending {pend} claimed {claimed} done {done}; forward done {fwd}; "
                  f"workers {len(W)}", flush=True)
            last = time.time()
        if fwd and pend == 0 and claimed == 0:
            break
        time.sleep(15)
    W = workers_by_name()
    for name, pids in W.items():
        for pid in sorted(pids, reverse=True):
            kill(pid)                                  # idle: every chunk is .done_, so no sample is in flight
    st = [json.load(open(s)) for s in glob.glob(os.path.join(q, "status_r0_m5w*.json"))]
    tot = {k: sum(x.get(k, 0) for x in st) for k in ("failed", "selfcheck_failed", "skipped_ndiff")}
    keys_q, keys_s = set(), set()
    for c in glob.glob(os.path.join(q, "props_r0_*.jsonl.done_*")):
        for line in open(c, encoding="utf-8"):
            r = json.loads(line)
            keys_q.add(f"{r['log_name']}|{r['token']}|{int(r['step'])}|{int(r['rank'])}")
    n_lines = 0
    for f in glob.glob(os.path.join(out, "onpolicy_r0_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_lines += 1
            keys_s.add(f"{r['log_name']}|{r['token']}|{int(r['step'])}|{int(r['rank'])}")
    summ = {"driver": "eval/m5_label.py (16 workers) then eval/m5_label_finish.py (kept " + ",".join(keep) + ")",
            "retired": retire, "seconds_finisher": round(time.time() - t0, 1), **tot,
            "status_note": "status files count per worker PROCESS lifetime (a restarted worker restarts its counters); "
                           "coverage below is from the files themselves",
            "keys_in_chunks": len(keys_q), "keys_labelled": len(keys_q & keys_s), "set_lines": n_lines,
            "keys_unlabelled": len(keys_q - keys_s)}
    json.dump(summ, open(os.path.join(a.m5, "label_summary.json"), "w"), indent=1)
    print(f"ZZM5_LABEL_DONE {json.dumps(summ)}")
    return 0 if tot["failed"] == 0 and tot["selfcheck_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
