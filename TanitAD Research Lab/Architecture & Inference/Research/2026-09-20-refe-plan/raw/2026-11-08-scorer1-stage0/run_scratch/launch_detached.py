"""Detached launch of the Stage 0 chain (pattern of pilot_v5/launch_chain_v5.py): own process group, no window, stdin DEVNULL."""
import subprocess
F = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
p = subprocess.Popen([r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe", r"F:\tanitad_scratch\stage0_20261108\chain_stage0.py"],
                     cwd=r"F:\tanitad_scratch\stage0_20261108", stdin=subprocess.DEVNULL,
                     stdout=open(r"F:\tanitad_scratch\stage0_20261108\chain_stdout.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, creationflags=F)
print("pid", p.pid)
