"""Launch one headrow_probe.py stage DETACHED (survives the launching session): CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW, stdin from NUL, stdout+stderr appended to a log.
CPU only: CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=8.  Prints the PID.   usage: python launch_probe.py <stage> [--fraction 10] [extra args]   (log: <run>/f<fraction>/<stage>.log)
Kill by explicit PID only.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
RUN = "C:/Users/Admin/par18/probe/run"

stage = sys.argv[1]
extra = sys.argv[2:]
frac = "10"
if "--fraction" in extra:
    frac = extra[extra.index("--fraction") + 1]
out = f"{RUN}/f{frac}"
os.makedirs(out, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONIOENCODING="utf-8")
log = open(f"{out}/{stage}.log", "ab")
p = subprocess.Popen([PY, "-I", f"{HERE}/headrow_probe.py", stage, *extra], cwd=HERE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                     creationflags=0x00000008 | 0x00000200 | 0x08000000, close_fds=True)
print(p.pid)
