"""Peak RSS of rl_val_eval.load_scores (OLD) vs rls1_checks.load_scores_light (NEW) on a REAL banked scores npz truncated to its first N sets.
usage: measure_loader.py old|new N      (prints one JSON line).  A watchdog aborts the process if physical RAM + swap headroom drops below 3 GB."""
import os, sys, time, json, threading
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, PKG + "/refe"); sys.path.insert(0, PKG + "/eval")
import numpy as np, psutil
which, N = sys.argv[1], int(sys.argv[2])
SRC = "C:/Users/Admin/AppData/Local/Temp/rls1dl/final/scores_update250_B.npz"
TMP = f"C:/Users/Admin/AppData/Local/Temp/rls1dl/trunc_{N}.npz"
z = np.load(SRC)
if N < len(z["token"]):
    np.savez(TMP, **{k: (z[k][:N] if z[k].ndim else z[k]) for k in z.files})
    path = TMP
else:
    path = SRC
P = psutil.Process(); peak = [0]; aborted = [False]
def sampler():
    while True:
        rss = P.memory_info().rss
        peak[0] = max(peak[0], rss)
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        if vm.available + (sw.total - sw.used) < 3 * 2**30:
            aborted[0] = True; print(json.dumps({"which": which, "N": N, "ABORTED_AT_RSS_MB": round(rss/2**20), "peak_rss_mb": round(peak[0]/2**20)}), flush=True); os._exit(3)
        time.sleep(0.01)
threading.Thread(target=sampler, daemon=True).start()
import rls1_checks as RC, rl_val_eval as RV
base = P.memory_info().rss
t0 = time.time()
d = (RV.load_scores if which == "old" else RC.load_scores_light)(path)
dt = time.time() - t0
print(json.dumps({"which": which, "N": N, "sets": len(d), "seconds": round(dt, 2), "baseline_rss_mb": round(base/2**20), "end_rss_mb": round(P.memory_info().rss/2**20), "peak_rss_mb": round(peak[0]/2**20), "peak_commit_mb": round(getattr(P.memory_info(), "peak_pagefile", 0)/2**20),
                  "file_mb": round(os.path.getsize(path)/2**20, 1)}), flush=True)
