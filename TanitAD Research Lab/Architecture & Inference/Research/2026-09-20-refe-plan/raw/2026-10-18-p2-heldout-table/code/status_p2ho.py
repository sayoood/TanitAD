#!/usr/bin/env python3
"""Zero-model-turn status check for the P2 held-out table job.  Prints markers, heartbeat age, stage, progress, guard, GPU-lock holder.
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I C:/Users/Admin/p2heldout/code/status_p2ho.py [tag]
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
wd = ROOT if tag == "main" else os.path.join(ROOT, f"_{tag}")
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"


def rd(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"_unreadable": repr(e)[:120]}


print("markers:", [m for m in ("P2HO_DONE", "P2HO_FAILED") if os.path.exists(os.path.join(wd, m))] or "none yet")
st = rd(os.path.join(wd, "STATUS.json"))
hb = st.get("heartbeat")
age = None
if hb:
    try:
        age = round(time.time() - time.mktime(time.strptime(hb[:19], "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        pass
pid = st.get("pid")
alive = None
if pid:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)
    alive = f'"{pid}"' in r.stdout.decode("utf-8", "replace")
print(json.dumps({"stage": st.get("stage"), "state": st.get("state"), "pid": pid, "pid_alive": alive, "heartbeat_age_s": age, "result": st.get("result"), "guard": st.get("guard"),
                  "progress": st.get("progress"), "n_units": st.get("n_units"), "n_tokens_planned": st.get("n_tokens_planned"), "fidelity_A": st.get("fidelity_A"),
                  "dgdump": st.get("dgdump")}, indent=1, default=str))
try:
    print("gpu lock:", open(LOCK, encoding="utf-8").read()[:300])
except OSError:
    print("gpu lock: none")
for m in ("P2HO_DONE", "P2HO_FAILED"):
    p = os.path.join(wd, m)
    if os.path.exists(p):
        print(f"--- {m} ---")
        print(open(p, encoding="utf-8").read()[:1500])
