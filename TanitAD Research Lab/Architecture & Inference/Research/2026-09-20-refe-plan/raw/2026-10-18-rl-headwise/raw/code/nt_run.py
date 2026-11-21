#!/usr/bin/env python3
"""RL-1 step 4b: score the composite seam with W3's harness (CPU only), FREEPOINT's call UNCHANGED except --out (own dir; the banked score store is not written).
    python nt_run.py --variant V1 --work C:/Users/Admin/par18/rl/nav
Writes <work>/score_logs/refe_rl1_<V>.status.json and the marker <work>/DONE_<V> (first line OK or FAIL)."""
import argparse, json, os, subprocess, sys, time

SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="V1")
ap.add_argument("--work", required=True)
a = ap.parse_args()
label = f"refe_rl1_{a.variant}"
seam = f"{a.work}/seam_{a.variant}.npz"
out = f"{a.work}/score"
logd = f"{a.work}/score_logs"
os.makedirs(logd, exist_ok=True)
os.makedirs(out, exist_ok=True)


def free_mb():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


status = {}
for att in range(1, 4):
    while free_mb() < 4000:
        time.sleep(60)
    lp = f"{logd}/{label}.attempt{att}.log"
    t0 = time.time()
    subprocess.call([PY_D, "score_e.py", "--label", label, "--seam", seam, "--out", out], cwd=SP,
                    env=dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1"), stdout=open(lp, "w"), stderr=subprocess.STDOUT)
    lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()
    status = next((json.loads(l) for l in reversed(lines) if l.startswith("{") and '"status"' in l), {})
    status["attempt"], status["seconds"] = att, round(time.time() - t0)
    json.dump(status, open(f"{logd}/{label}.status.json", "w"), indent=1)
    if status.get("status") == "PASS" and os.path.exists(f"{out}/{label}/{label}.csv"):
        break
    time.sleep(60)
ok = status.get("status") == "PASS" and os.path.exists(f"{out}/{label}/{label}.csv")
open(f"{a.work}/DONE_{a.variant}", "w").write(("OK" if ok else "FAIL") + "\n" + json.dumps(status) + "\n")
print("ZZNTRUN", "OK" if ok else "FAIL", flush=True)
