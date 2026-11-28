"""Dump every arm of one seed and run the four-family panel. One command."""
from __future__ import annotations
import os, subprocess, sys

PY = r"C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ROOT = r"C:/Users/Admin/p1gate"
seed = sys.argv[1] if len(sys.argv) > 1 else "0"
stride = sys.argv[2] if len(sys.argv) > 2 else "4"
arms = ["off", "head", "shuf"]
env = dict(os.environ, PYTHONIOENCODING="utf-8")

dumps = []
for a in arms:
    run = f"{ROOT}/runs/{a}_s{seed}"
    npz = f"{ROOT}/runs/dump_{a}_s{seed}.npz"
    if not os.path.exists(run + "/ckpt.pt"):
        print(f"[finish] SKIP {a}: no ckpt.pt")
        continue
    if not os.path.exists(npz):
        rc = subprocess.call([PY, "-u", ROOT + "/dump_eval.py", "--run", run,
                              "--out", npz, "--stride", stride], env=env)
        if rc != 0:
            print(f"[finish] dump {a} FAILED rc={rc}")
            sys.exit(rc)
    dumps.append(f"{a}={npz}")

panel = f"{ROOT}/runs/panel_s{seed}.json"
rc = subprocess.call([PY, "-u", ROOT + "/analyze.py", "--dumps", *dumps,
                      "--out", panel], env=env)
if rc:
    sys.exit(rc)
md = subprocess.run([PY, ROOT + "/make_tables.py", panel], env=env,
                    capture_output=True, text=True, encoding="utf-8")
open(f"{ROOT}/runs/tables_s{seed}.md", "w", encoding="utf-8").write(md.stdout)
print(md.stdout)
