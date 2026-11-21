# exec wrapper: run ufunc_probe.py in the VM's NAVSIM venv and print its line; also report the G-H lever state
import subprocess, os, json
r = subprocess.run(["/content/p0/venv_navsim/bin/python", "/content/p0/code/navtest/ufunc_probe.py"], capture_output=True, text=True, cwd="/content/p0")
print(r.stdout.strip()[-3000:] or r.stderr[-1000:])
print("GHX", sorted(os.listdir("/content/p0/out/ghx")))
