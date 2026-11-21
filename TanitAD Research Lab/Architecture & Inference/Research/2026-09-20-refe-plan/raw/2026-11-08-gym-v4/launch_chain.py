"""Detached launch of the M-GYM-4 chain (pattern of tsf_bundle2_20261107/a6/launch_detached_a6.py)."""
import subprocess, os
F = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
os.makedirs(r"F:\tanitad_scratch\mgym4", exist_ok=True)
subprocess.Popen([r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe", "-I", r"E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v4\scorer_gym\chain_v4.py"],
                 stdin=subprocess.DEVNULL, stdout=open(r"F:\tanitad_scratch\mgym4\chain.log", "w"), stderr=subprocess.STDOUT, creationflags=F, env=env)
