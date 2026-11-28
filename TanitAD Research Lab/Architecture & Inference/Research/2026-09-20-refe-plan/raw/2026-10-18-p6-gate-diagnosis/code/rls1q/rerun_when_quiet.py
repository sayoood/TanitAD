"""Detached: wait for a QUIET dev box, then run refe/test_rls1_queue.py in full form, battery-style (run1.py = battery.run's own code), as two PAIRS, each pair
FIX+test-fix (trees/fix_rls1q) || pristine BASE (trees/base_full) concurrently (<= 2 processes, as the battery).  Writes rerun/RESULTS.json and rerun/DONE.
Quiet = commit headroom >= 20 GB AND physical free >= 6 GB on 3 consecutive 30 s polls (the 2026-10-18 failures came with 4.8-7 GB headroom and MSYS fork errors).
Gives up after 4 h with RESULTS.json {"quiet": false}.  The box state is logged every 30 s to rerun/env.tsv."""
import ctypes
import json
import os
import shutil
import subprocess
import time

D = "C:/Users/Admin/p6gate/rls1q_diag"
O = f"{D}/rerun"
os.makedirs(O, exist_ok=True)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TREES = {"fixq": "C:/Users/Admin/p6gate/trees/fix_rls1q", "base": "C:/Users/Admin/p6gate/trees/base_full"}


class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong)]


def env_row():
    m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    r = {"t": time.strftime("%H:%M:%S"), "Cfree_MB": shutil.disk_usage("C:/").free // 2**20, "commit_avail_MB": m.ullAvailPageFile // 2**20, "phys_avail_MB": m.ullAvailPhys // 2**20}
    with open(f"{O}/env.tsv", "a") as f:
        f.write("\t".join(f"{k}={v}" for k, v in r.items()) + "\n")
    return r


res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "quiet": False, "runs": {}}
t0, ok = time.time(), 0
while time.time() - t0 < 4 * 3600:
    r = env_row()
    ok = ok + 1 if r["commit_avail_MB"] >= 20000 and r["phys_avail_MB"] >= 6000 else 0
    if ok >= 3:
        res["quiet"], res["quiet_at"] = True, r
        break
    time.sleep(30)
if res["quiet"]:
    for pair in (1, 2):
        ps = [subprocess.Popen([PY, f"{D}/run1.py", TREES[t], "refe/test_rls1_queue.py", O, f"{t}_p{pair}", "1500"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
              for t in ("fixq", "base")]
        while any(p.poll() is None for p in ps):
            env_row(); time.sleep(30)
        for t, p in zip(("fixq", "base"), ps):
            out = p.stdout.read()
            res["runs"][f"{t}_p{pair}"] = out.strip().splitlines()[-1] if out.strip() else "NO OUTPUT"
        json.dump(res, open(f"{O}/RESULTS.json", "w"), indent=1)
json.dump(res, open(f"{O}/RESULTS.json", "w"), indent=1)
open(f"{O}/DONE", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
