"""Peak-RSS / wall-time probe of the RL-S1 monitor's check_fn on the REAL banked RL-val inputs (dev box, Windows).
Each phase prints peak working set (psutil peak_wset) and current RSS.  Usage: measure_rss.py <step0|update250> [--sampler]"""
import os, sys, time, json, threading
os.environ.setdefault("OMP_NUM_THREADS", "1")
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, PKG + "/refe"); sys.path.insert(0, PKG + "/eval")
import psutil
P = psutil.Process()
peak = [0]
def sampler():
    while True:
        peak[0] = max(peak[0], P.memory_info().rss)
        time.sleep(0.02)
threading.Thread(target=sampler, daemon=True).start()
def mark(name, t0):
    mi = P.memory_info()
    print(f"{name:34s} t={time.time()-t0:7.2f}s rss={mi.rss/2**20:8.1f} MB  peak_wset(os)={getattr(mi,'peak_wset',0)/2**20:8.1f} MB  sampled_peak={peak[0]/2**20:8.1f} MB", flush=True)
D = "C:/Users/Admin/AppData/Local/Temp/rls1dl"
tag = sys.argv[1]
t0 = time.time(); mark("python+psutil baseline", t0)
import rls1_checks as RC
mark("import rls1_checks", t0)
import rl_val_eval, rl_reward
mark("import rl_val_eval, rl_reward", t0)
chk = RC.RlValChecker("R0", D + "/final", D + "/x/pdm_val", D + "/x/terms_val", D + "/x/sets_val", PKG + "/raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000/rl_val_scenes.tsv")
mark("RlValChecker.__init__ (load)", t0)
rec = chk(tag, {"B": D + f"/final/scores_{tag}_B.npz"})
mark(f"check_fn({tag})", t0)
print(json.dumps({k: rec[k] for k in ("tag", "g3_ok", "g5_ok", "k4_breach", "pdms_delta_x100", "k5_breach", "n_sets")}))
