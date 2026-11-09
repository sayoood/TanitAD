"""Detached launch of the REWARD-2 chain (pattern of pilot_v5/launch_chain_v5.py): own process group, no window, stdin DEVNULL, CPU only.
usage: launch_chain_v6.py [--dry]     (--dry: F:/tanitad_scratch/mgym6dry, stops before the draw, cannot draw)"""
import subprocess, os, sys
F = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
dry = "--dry" in sys.argv
W = r"F:\tanitad_scratch\mgym6dry" if dry else r"F:\tanitad_scratch\mgym6"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
os.makedirs(W, exist_ok=True)
args = [r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe", "-I", r"E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v6\scorer_gym\chain_v6.py"] + (["--dry", "--workers", "2"] if dry else [])
p = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=open(W + r"\chain.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, creationflags=F, env=env)
print("launched pid", p.pid)
