#!/usr/bin/env python3
"""Measure 5r (eval/PREREG_MEASURE5R.md, blob a8241136): every labelling run, under ONE RAM-aware supervisor.

Queues, in priority order (a worker always takes the highest-priority queue with an unclaimed chunk):
  A  G-R2  flag OFF == v4: the first 2 chunks of M5's queue, the real CLI with M5's flags (NO --repair-last-heading)
           -> <m5r>/gr2/sets; compared byte for byte (timing fields excluded) with M5's lines by eval/m5r_gates.py
  B  G-R3/G-R4  W3's 200 navtest tokens (validate_slow_labels_v4.py's prepped chunks), C2 + --repair-last-heading
           -> <navtest v4_labels>/v5rep
  C  the v5 labels: M5's 1,900 navtrain props chunks (copied, unchanged), C2 + --repair-last-heading -> <m5r>/sets
RAM RULE (coordinator, 2026-09-28): the dev box keeps >= 5 GB free (another session's refav1 eval runs). A worker is
ADDED only when free RAM >= 6.3 GB and >= 120 s after the previous launch (a worker settles at ~0.93 GB, MEASURED);
a worker is RETIRED -- the newest one, killed by explicit PID (its process tree), its claimed chunk renamed back into
the queue -- when free RAM stays < 4.7 GB for 40 s. A killed worker's partial lines stay in its own file; the re-claimed
chunk is labelled again and m5_finetune_eval.data keeps one line per key.

    python eval/m5r_label.py [--max-workers 3] [--add-at-gb 6.3] [--retire-below-gb 4.7] [--a-first-add-gb 5.0]
                             [--a-first-retire-gb 4.0]
Coordinator rules (2026-09-28 ~07:00): at most 3 workers (`--max-workers`); the 2nd and later workers, and any worker on
queue B or C, are added at >= 6.3 GB free and retired below 4.7 GB for 40 s; ONE exception: a SINGLE worker on queue A
(G-R2) may start at >= 5.0 GB and is retired only below 4.0 GB for 40 s.
GATE BEFORE C (default ON): queue C (1,900 samples, hours of RAM-capped CPU) is labelled only after queue B is complete
AND `eval/m5r_gates.py gr34` PASSES (G-R3 + G-R4). PREREG_MEASURE5R §5: if G-R4 fails, v4r is NOT tested -- labelling C
would then only take RAM from the refcv7 battery. A failing gate stops the supervisor (label_summary.json says why).
RESUMABLE: on start the supervisor ADOPTS the labellers of its queues that are still alive (found by command line;
polled by PID), releases the claims of workers that are gone, and names new workers after the highest index in use. A
retired or crashed worker loses at most its one claimed chunk (the chunk is re-queued and labelled again in full).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_slow_labels_v4 as V  # noqa: E402  (worker_env, run_cli, the navtest prep paths, the v4 CLI path)

M5 = "D:/Projects/TanitAD/data/refe_m5"
M5R = "D:/Projects/TanitAD/data/refe_m5r"
C2 = ["--slow-copies", "--slow-factors", "0.75", "--slow-frac", "1.0"]
REP = ["--repair-last-heading"]


def queues():
    return [
        {"name": "A_gr2", "q": f"{M5R}/gr2/queue", "out": f"{M5R}/gr2/sets", "flags": C2},
        {"name": "B_gr3", "q": f"{V.WD}/queue_v5rep", "out": f"{V.WD}/v5rep", "flags": C2 + REP},
        {"name": "C_v5", "q": f"{M5R}/queue", "out": f"{M5R}/sets", "flags": C2 + REP},
    ]


def base_name(p):
    b = os.path.basename(p)
    return b[: b.index(".jsonl") + len(".jsonl")]


def setup() -> dict:
    Q = {x["name"]: x for x in queues()}
    made = {}
    m5_chunks = sorted(glob.glob(f"{M5}/queue/props_r0_*.jsonl.done_*"))
    for name, src in (("A_gr2", [c for c in m5_chunks if base_name(c).endswith(("10000000.jsonl", "10000001.jsonl"))]),
                      ("C_v5", m5_chunks),
                      ("B_gr3", sorted(glob.glob(f"{V.Q}/props_r0_*.jsonl*")))):
        q = Q[name]["q"]
        os.makedirs(Q[name]["out"], exist_ok=True)
        if os.path.isdir(q) and os.listdir(q):
            made[name] = f"exists ({len(os.listdir(q))} entries)"
            continue
        os.makedirs(q, exist_ok=True)
        seen = set()
        for c in src:
            b = base_name(c)
            if b in seen:
                continue
            seen.add(b)
            shutil.copyfile(c, os.path.join(q, b))
        made[name] = f"{len(seen)} chunks copied"
    return made


def state(q: str) -> tuple:
    names = os.listdir(q) if os.path.isdir(q) else []
    pend = sum(1 for n in names if n.startswith("props_r0_") and n.endswith(".jsonl"))
    claimed = sum(1 for n in names if n.startswith("props_r0_") and ".jsonl.r" in n and ".done_" not in n)
    done = sum(1 for n in names if n.startswith("props_r0_") and ".jsonl.done_" in n)
    return pend, claimed, done


def free_gb() -> float:
    o = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                       capture_output=True, text=True).stdout.strip()
    return int(o) / 2 ** 20 if o.isdigit() else -1.0


def release(q: str, wname: str) -> list:
    out = []
    for c in glob.glob(os.path.join(q, f"props_r0_*.jsonl.{wname}")):
        try:
            os.rename(c, c[: -(len(wname) + 1)])
            out.append(os.path.basename(c))
        except OSError:
            pass
    return out


def labellers() -> list:
    """(pid, worker, queue) of every live onpolicy_label_v4.py process (the venv launcher AND its child)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
          "'onpolicy_label_v4.py' } | ForEach-Object { \"$($_.ProcessId)`t$([int][double]::Parse((Get-Date "
          "($_.CreationDate.ToUniversalTime()) -UFormat %s)))`t$($_.CommandLine)\" }")
    o = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    out = []
    for line in o.splitlines():
        if "\t" not in line:
            continue
        pid, ct, cmd = line.split("\t", 2)
        tk = cmd.split()
        try:
            w = tk[tk.index("--worker") + 1].strip('"')
            q = tk[tk.index("--queue") + 1].strip('"')
        except (ValueError, IndexError):
            continue
        out.append((int(pid), w, q, float(ct) if ct.strip().lstrip("-").isdigit() else 0.0))
    return out


class Adopted:
    """A worker started by an earlier supervisor run: liveness by its PIDs."""

    def __init__(self, pids):
        self.pids, self.returncode = sorted(pids), None
        self.pid = self.pids[0]

    def poll(self):
        alive = {x[0] for x in labellers()}
        if not any(p in alive for p in self.pids):
            self.returncode = "gone"
            return self.returncode
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--add-at-gb", type=float, default=6.3)
    ap.add_argument("--retire-below-gb", type=float, default=4.7)
    ap.add_argument("--a-first-add-gb", type=float, default=5.0)
    ap.add_argument("--a-first-retire-gb", type=float, default=4.0)
    ap.add_argument("--no-gate-before-c", action="store_true")
    a = ap.parse_args()
    gate_c = {"state": "open" if a.no_gate_before_c else "pending"}      # pending -> open | blocked
    logs = f"{M5R}/logs"
    os.makedirs(logs, exist_ok=True)
    print(f"  setup: {json.dumps(setup())}", flush=True)
    QS = queues()
    live: dict = {}                 # name -> {"p": Popen | Adopted, "queue": dict, "t": launch time}
    n_launch, last_launch, low_since, t0, last_print = 0, 0.0, None, time.time(), 0.0
    # ---- ADOPT the still-alive labellers of our queues; release the claims of the gone ones
    byq = {os.path.normcase(os.path.abspath(x["q"])): x for x in QS}
    found: dict = {}
    for pid, w, q, ct in labellers():
        x = byq.get(os.path.normcase(os.path.abspath(q)))
        if x is not None:
            e = found.setdefault(w, [x, [], ct])
            e[1].append(pid)
            e[2] = min(e[2], ct) if ct else e[2]
    for w, (x, pids, ct) in found.items():
        live[w] = {"p": Adopted(pids), "queue": x, "t": ct or time.time()}      # t = the TRUE launch time
        print(f"  ADOPTED {w} on {x['name']} (PIDs {pids})", flush=True)
    for x in QS:
        for c in glob.glob(os.path.join(x["q"], "props_r0_*.jsonl.r*")):
            wn = c.rsplit(".", 1)[1]
            if ".done_" not in c and wn not in live:
                print(f"  released a stale claim of {wn}: {release(x['q'], wn)}", flush=True)
    idx = [int(w[1:]) for w in live if w[1:].isdigit()]
    for f in glob.glob(f"{M5R}/logs/label_r*.log"):
        b = os.path.basename(f)[len("label_r"):-len(".log")]
        if b.isdigit():
            idx.append(int(b))
    n_launch = max(idx) + 1 if idx else 0
    hist = []
    while True:
        # reap exited workers (with --once a worker exits when its queue is empty)
        for wn, w in list(live.items()):
            if w["p"].poll() is not None:
                rel = release(w["queue"]["q"], wn)
                print(f"  {time.strftime('%H:%M:%S')} worker {wn} ({w['queue']['name']}) exited rc {w['p'].returncode}"
                      + (f"; released {rel}" if rel else ""), flush=True)
                live.pop(wn)
        st = {x["name"]: state(x["q"]) for x in QS}
        if gate_c["state"] == "pending" and st["B_gr3"][0] == 0 and st["B_gr3"][1] == 0 and \
                not any(w["queue"]["name"] == "B_gr3" for w in live.values()):
            print(f"  {time.strftime('%H:%M:%S')} queue B complete -- running G-R3/G-R4 before queue C", flush=True)
            import eval_checkpoint as EC
            rc = subprocess.call([V.EC.DRIVERL_PY, os.path.join(HERE, "m5r_gates.py"), "gr34"], cwd=HERE,
                                 env=EC.env_driverl(), stdout=open(f"{logs}/gate_gr34_supervisor.log", "w"),
                                 stderr=subprocess.STDOUT)
            gate_c["state"] = "open" if rc == 0 else "blocked"
            print(f"  {time.strftime('%H:%M:%S')} G-R3/G-R4 rc {rc} -> queue C {gate_c['state'].upper()}", flush=True)
        if gate_c["state"] == "blocked" and not live:
            print("  G-R3/G-R4 FAILED: queue C is NOT labelled (PREREG_MEASURE5R §5: v4r is not tested)", flush=True)
            break
        if all(s[0] == 0 and s[1] == 0 for s in st.values()) and not live:
            break
        fg = free_gb()
        now = time.time()
        # RETIRE on sustained low RAM: the newest worker whose threshold is crossed. The OLDEST worker, when it is on
        # queue A (the coordinator's single-worker exception), has the lower threshold; every other worker 4.7 GB.
        oldest = min(live, key=lambda k: live[k]["t"]) if live else None

        def thr_of(k):
            return a.a_first_retire_gb if (k == oldest and live[k]["queue"]["name"] == "A_gr2") else a.retire_below_gb
        viol = [k for k in live if 0 <= fg < thr_of(k)]
        if viol:
            low_since = low_since or now
            if now - low_since >= 40:
                wn = max(viol, key=lambda k: live[k]["t"])
                w = live.pop(wn)
                for kp in (w["p"].pids if isinstance(w["p"], Adopted) else [w["p"].pid]):
                    subprocess.run(["taskkill", "/PID", str(kp), "/T", "/F"], capture_output=True)
                time.sleep(2)
                rel = release(w["queue"]["q"], wn)
                print(f"  {time.strftime('%H:%M:%S')} RAM {fg:.2f} GB < {a.retire_below_gb} for 40 s: RETIRED {wn} "
                      f"(PID {w['p'].pid}); released {rel}", flush=True)
                low_since, last_launch = None, now
        else:
            low_since = None
        # ADD a worker to the highest-priority queue with an unclaimed chunk (queue A: one worker at most)
        on_a = sum(1 for w in live.values() if w["queue"]["name"] == "A_gr2")
        if len(live) < a.max_workers and now - last_launch >= 120:
            for x in QS:
                if st[x["name"]][0] == 0 or (x["name"] == "A_gr2" and on_a >= 1):
                    continue
                if x["name"] == "C_v5" and gate_c["state"] != "open":
                    continue
                need = a.a_first_add_gb if (x["name"] == "A_gr2" and not live) else a.add_at_gb
                if fg < need:
                    break
                if True:
                    wn = f"r{n_launch}"
                    n_launch += 1
                    cmd = [V.EC.DRIVERL_PY, V.V4, "--queue", x["q"], "--out", x["out"], "--rank", "0", "--worker", wn,
                           "--once", "--poll-s", "10"] + x["flags"]
                    p = V.run_cli(cmd, os.path.join(logs, f"label_{wn}.log"), V.worker_env())
                    live[wn] = {"p": p, "queue": x, "t": now}
                    last_launch = now
                    print(f"  {time.strftime('%H:%M:%S')} ADDED {wn} -> {x['name']} (free RAM {fg:.2f} GB; "
                          f"{len(live)} live)", flush=True)
                    break
        if now - last_print >= 120:
            hist.append((now, st["C_v5"][2]))
            print(f"  {time.strftime('%H:%M:%S')} free RAM {fg:.2f} GB; workers {len(live)} "
                  f"{sorted((w['queue']['name'] for w in live.values()))}; " +
                  "; ".join(f"{k} pend {s[0]} claimed {s[1]} done {s[2]}" for k, s in st.items()), flush=True)
            json.dump({"at": time.strftime("%H:%M:%S"), "free_gb": round(fg, 2), "workers": len(live),
                       "queues": {k: {"pending": s[0], "claimed": s[1], "done": s[2]} for k, s in st.items()}},
                      open(f"{M5R}/label_status.json", "w"), indent=1)
            last_print = now
        time.sleep(20)
    summ = {"seconds": round(time.time() - t0, 1), "launches": n_launch, "gate_before_c": gate_c["state"]}
    for x in QS:
        stt = [json.load(open(s)) for s in glob.glob(os.path.join(x["q"], "status_r0_r*.json"))]
        tot = {k: sum(s.get(k, 0) for s in stt) for k in ("written", "failed", "selfcheck_failed", "skipped_ndiff")}
        keys_q, keys_s = set(), set()
        for c in glob.glob(os.path.join(x["q"], "props_r0_*.jsonl.done_*")):
            for line in open(c, encoding="utf-8"):
                r = json.loads(line)
                keys_q.add(f"{r['log_name']}|{r['token']}|{int(r['step'])}|{int(r['rank'])}")
        lv = set()
        for f in glob.glob(os.path.join(x["out"], "onpolicy_r0_*.jsonl")):
            for line in open(f, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys_s.add(f"{r['log_name']}|{r['token']}|{int(r['step'])}|{int(r['rank'])}")
                lv.add((r.get("label_version"), r.get("repair")))
        summ[x["name"]] = {**tot, "keys_in_chunks": len(keys_q), "keys_labelled": len(keys_q & keys_s),
                           "label_version_repair": sorted(str(v) for v in lv)}
    json.dump(summ, open(f"{M5R}/label_summary.json", "w"), indent=1)
    print(f"ZZM5R_LABEL_DONE {json.dumps(summ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
