"""GPU/host gate for the dev-box RTX 4060 (the PI's desktop, shared with the NavSim agent).

Rules (brief, binding): memory.used < 4300 MiB; no OTHER python compute process on the GPU;
free host RAM >= 8 GB. Prints a one-line verdict and exits 0 (PASS) / 3 (WAIT).
The verdict is ALSO written as JSON to --out so a caller asserts on the ARTIFACT, not $?.

⛔ FIXED 2026-09-24 ~03:55 Berlin (MEASURED self-deadlock, reported by the Master Mind at 03:42).
run_battery's `gate_wait()` called `gate()` without the caller's pid. The caller had created a CUDA
context in its seed-0 roll, so its OWN pid sat in `python_compute`: the gate waited on itself
(11 WAIT rows in `raw/battery_step1000.log`), blocked the sibling NavSim stream as well, and would
have given up only after 12 h. Two fixes, both pinned by `test_gpu_gate.py`:
  * `evaluate()` is the pure decision, and it always drops the caller's own pid(s) from the
    python-compute list (`gate()` adds `os.getpid()` by default);
  * run_battery never holds a CUDA context while it waits: every GPU stage runs in a subprocess.
"""
import argparse, csv, io, json, os, subprocess, sys, time


def _q(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout


def facts():
    """What the machine says right now: (used MiB, [(pid, process_name)], free RAM GB)."""
    used = int(_q(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
               .strip().splitlines()[0])
    apps = list(csv.reader(io.StringIO(_q(["nvidia-smi",
                                           "--query-compute-apps=pid,process_name,used_memory",
                                           "--format=csv,noheader"]))))
    pids = []
    for row in apps:
        if not row:
            continue
        try:
            pids.append((int(row[0].strip()), row[1].strip() if len(row) > 1 else ""))
        except ValueError:
            pass
    ps = _q(["powershell.exe", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"])
    try:
        free_gb = int(ps.strip().splitlines()[-1]) / 1024 / 1024
    except Exception:
        free_gb = -1.0
    return used, pids, free_gb


def evaluate(used_mib, apps, free_gb, *, self_pids=(), max_used_mib=4300, min_free_ram_gb=8.0):
    """The PURE gate decision. `self_pids` are the caller's own process ids: a process never waits
    on itself."""
    own = {int(p) for p in (self_pids or ()) if p is not None}
    py = [(int(p), n) for p, n in apps if "python" in str(n).lower() and int(p) not in own]
    ok = used_mib < max_used_mib and not py and free_gb >= min_free_ram_gb
    return {"ok": bool(ok), "gpu_used_mib": used_mib, "python_compute": py,
            "self_pids_excluded": sorted(own), "free_ram_gb": round(float(free_gb), 2),
            "rule": f"used<{max_used_mib} MiB, no other python compute, free RAM>={min_free_ram_gb} GB",
            "t": time.strftime("%Y-%m-%dT%H:%M:%S")}


def gate(max_used_mib=4300, min_free_ram_gb=8.0, self_pid=None, self_pids=None):
    used, pids, free_gb = facts()
    own = set(self_pids or ())
    own.add(os.getpid() if self_pid is None else int(self_pid))
    return evaluate(used, pids, free_gb, self_pids=own, max_used_mib=max_used_mib,
                    min_free_ram_gb=min_free_ram_gb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--wait", action="store_true", help="re-check every 60 s until PASS")
    ap.add_argument("--max-wait-s", type=int, default=6 * 3600)
    a = ap.parse_args()
    t0 = time.time()
    while True:
        g = gate()
        print(json.dumps(g), flush=True)
        if g["ok"] or not a.wait or time.time() - t0 > a.max_wait_s:
            break
        time.sleep(60)
    if a.out:
        json.dump(g, open(a.out, "w"), indent=1)
    sys.exit(0 if g["ok"] else 3)
