#!/usr/bin/env python3
"""Waits for the main driver (full/EXTRA_DONE.txt, i.e. nothing else is running), then re-runs the gate_full tests whose fix landed AFTER they ran
(test_rl_r0c: layout-independent mutation snippet; test_rls1_chain: MODEL_REGISTRY.md added above the package), ONE at a time, and writes RERUN_DONE.txt.
Deleting a summary entry while the main driver runs is useless (lane_runner re-dumps its in-memory dict), hence the wait."""
import json, os, subprocess, sys, time
F = "C:/Users/Admin/b8carve/full"
while not os.path.exists(F + "/EXTRA_DONE.txt"):
    time.sleep(30)
    if os.path.exists(F + "/RERUN_ABORT.txt"):
        sys.exit(0)
RERUN = ["refe/test_rl_r0c.py", "refe/test_rls1_chain.py"]
sp = F + "/gate_summary.json"
d = json.load(open(sp))
for t in RERUN:
    d.pop(os.path.basename(t), None)
d.pop("_done", None)
json.dump(d, open(sp, "w"), indent=1)
env = dict(os.environ, PYTHONIOENCODING="utf-8", LANE_TIMEOUT="3000")
subprocess.run(["C:/Python314/python.exe", "C:/Users/Admin/b8carve/lane_runner.py", F + "/gate_full", F + "/gate_logs", sp] + RERUN, env=env, stdin=subprocess.DEVNULL)
open(F + "/RERUN_DONE.txt", "w").write("done\n")
