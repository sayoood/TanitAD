"""Detached launch of the REWARD-1 chain (pattern of pilot_v4/launch_chain.py): own process group, no window, stdin DEVNULL, CPU only."""
import subprocess, os
F = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
os.makedirs(r"F:\tanitad_scratch\mgym5", exist_ok=True)
p = subprocess.Popen([r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe", "-I", r"E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v5\scorer_gym\chain_v5.py"],
                     stdin=subprocess.DEVNULL, stdout=open(r"F:\tanitad_scratch\mgym5\chain.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, creationflags=F, env=env)
print("launched pid", p.pid)
