#!/usr/bin/env python3
"""Zero-model-turn status check for SR-TTC S0.  Prints markers, pid liveness, heartbeat age, stage / state, batches done, sets banked, guard, GPU-lock holder, free disk on C:, and a
measured ETA (average wall time per finished batch x batches left; ESTIMATED).
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I C:/Users/Admin/srttc_s0/code/status_s0.py [tag]
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

SRC = "C:/Users/Admin/srttc_s0"
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
wd = f"{SRC}/run" if tag == "main" else f"{SRC}/run/_{tag}"
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"


def rd(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"_unreadable": repr(e)[:120]}


print("markers:", [m for m in ("S0_DONE", "S0_FAILED", "FINALIZED.txt") if os.path.exists(os.path.join(wd, m))] or "none yet")
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
states = sorted(glob.glob(f"{wd}/state/unit_*.json"))
whens = []
for p in states:
    w = rd(p).get("when")
    if w:
        try:
            whens.append(time.mktime(time.strptime(w[:19], "%Y-%m-%dT%H:%M:%S")))
        except ValueError:
            pass
eta = None
n_total, n_done = st.get("n_batches"), len(states)
tokens_left = None
ps = st.get("progress") or {}
if st.get("n_tokens_planned") and ps.get("tokens_banked") is not None:
    tokens_left = st["n_tokens_planned"] - ps["tokens_banked"]
if len(whens) >= 2 and tokens_left is not None and ps.get("tokens_banked"):
    per_tok = (whens[-1] - whens[0]) / max(ps["tokens_banked"] - 0, 1)
    eta = f"~{round(per_tok * tokens_left / 60)} min of dump+staging at the average pace so far (ESTIMATED; excludes time waiting for an idle pod)"
print(json.dumps({"stage": st.get("stage"), "state": st.get("state"), "pid": pid, "pid_alive": alive, "heartbeat_age_s": age, "result": st.get("result"), "guard": st.get("guard"),
                  "progress": st.get("progress"), "units_banked_files": n_done, "phase_L": st.get("phase_L"), "phase_S": st.get("phase_S"), "resumed_units": st.get("resumed_units"), "n_tokens_planned": st.get("n_tokens_planned"), "n_sets_planned": st.get("n_sets_planned"),
                  "dgdump": st.get("dgdump"), "last_dump_seconds": st.get("last_dump_seconds"), "eta": eta, "c_free_gib": round(shutil.disk_usage("C:/").free / 2 ** 30, 1), "d_free_gib": round(shutil.disk_usage("D:/").free / 2 ** 30, 1)}, indent=1, default=str))
try:
    print("gpu lock:", open(LOCK, encoding="utf-8").read()[:300])
except OSError:
    print("gpu lock: none")
for m in ("S0_DONE", "S0_FAILED"):
    p = os.path.join(wd, m)
    if os.path.exists(p):
        print(f"--- {m} ---")
        print(open(p, encoding="utf-8").read()[:1500])
