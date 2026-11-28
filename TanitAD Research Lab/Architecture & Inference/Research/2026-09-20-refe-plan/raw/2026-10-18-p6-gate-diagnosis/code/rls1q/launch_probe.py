#!/usr/bin/env python3
"""Launch s1probe.py with the env battery.py's run() builds (lines copied verbatim), cwd = the refe dir, stdin DEVNULL.  Usage: launch_probe.py <fix|base> <N> <log> [scen]"""
import os, subprocess, sys
TREES = {"fix": "C:/Users/Admin/p6gate/trees/fix_full", "base": "C:/Users/Admin/p6gate/trees/base_full", "fixq": "C:/Users/Admin/p6gate/trees/fix_rls1q"}
PKREL = os.path.join("TanitAD Research Lab", "Architecture & Inference", "Research", "2026-09-20-refe-plan")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
tree, n, log = sys.argv[1:4]; scen = sys.argv[4] if len(sys.argv) > 4 else "s1"
refe = os.path.join(TREES[tree], PKREL, "refe")
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
for k in [k for k in env if k.upper().startswith("P6_")]:
    env.pop(k)
env["PATH"] = "C:/Program Files/Git/usr/bin;" + env.get("PATH", "")
if len(sys.argv) > 5: env["S1PROBE_DELAY"] = sys.argv[5]
with open(log, "wb") as f:
    p = subprocess.run([PY, "C:/Users/Admin/p6gate/rls1q_diag/s1probe.py", refe, n, scen], cwd=refe, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=280)
print(tree, "rc", p.returncode); print(open(log, encoding="utf-8").read().splitlines()[-1])
