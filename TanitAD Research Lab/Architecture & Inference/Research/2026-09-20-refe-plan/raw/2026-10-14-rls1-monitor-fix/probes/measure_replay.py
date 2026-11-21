"""REAL-data proof: rls1_checks.py `replay` over R0's banked sft.jsonl / scores (B) and C0's (A), compared BYTE FOR BYTE with the kill_replay.json the pod's (old) code wrote."""
import os, sys, json, time, threading, shutil, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, PKG + "/refe"); sys.path.insert(0, PKG + "/eval")
import psutil
D = "C:/Users/Admin/AppData/Local/Temp/rls1dl"
OUT = "C:/Users/Admin/AppData/Local/Temp/rls1dl/replay_out"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
rows = open("C:/Users/Admin/AppData/Local/Temp/rls1dl_r0/final/sft.jsonl", encoding="utf-8").read().replace("/workspace/data/rls1/arms/R0/scores", D + "/final")
open(OUT + "/sft.jsonl", "w", encoding="utf-8", newline="\n").write(rows)
P = psutil.Process(); peak = [0]
def sampler():
    while True:
        peak[0] = max(peak[0], P.memory_info().rss); time.sleep(0.02)
threading.Thread(target=sampler, daemon=True).start()
import rls1_checks as RC
t0 = time.time()
rc = RC.main(["replay", "--arm", "R0", "--out", OUT, "--refdir", D + "/final", "--pdm", D + "/x/pdm_val", "--terms", D + "/x/terms_val", "--sets", D + "/x/sets_val",
              "--tsv", PKG + "/raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000/rl_val_scenes.tsv"])
dt = time.time() - t0
new = open(OUT + "/kill_replay.json", "rb").read(); pod = open(D + "/final/kill_replay.json", "rb").read()
print(json.dumps({"rc": rc, "seconds": round(dt, 1), "peak_rss_mb": round(peak[0] / 2**20), "new_bytes": len(new), "pod_bytes": len(pod), "BYTE_IDENTICAL": new == pod,
                  "new_sha256": hashlib.sha256(new).hexdigest()[:16], "pod_sha256": hashlib.sha256(pod).hexdigest()[:16]}))
if new != pod:
    import difflib
    for l in list(difflib.unified_diff(pod.decode().splitlines(), new.decode().splitlines(), "pod", "new", lineterm="", n=0))[:30]: print(l)
