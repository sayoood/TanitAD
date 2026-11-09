import subprocess, sys, json
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
D = "F:/TanitAD_p6video"
p = subprocess.Popen([PY, "-u", D + "/run_all.py"], cwd=D, stdin=subprocess.DEVNULL, stdout=open(D + "/run_all.log", "a", encoding="utf-8"),
                     stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
json.dump({"pid": p.pid}, open(D + "/run_all.pid.json", "w"))
print(p.pid)
