import subprocess, os
F = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
PY = r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
D = r"F:\tanitad_scratch\mgym4\explore"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
subprocess.Popen([PY, "-I", r"E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v4\explore\prescreen_dist.py", D + r"\prescreen_dist.json"],
                 stdin=subprocess.DEVNULL, stdout=open(D + r"\prescreen_dist.log", "w"), stderr=subprocess.STDOUT, creationflags=F, env=env)
