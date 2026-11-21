# VM side, run with `colab exec -f` (short): unpack the uploaded code bundle and start vm_p0.py DETACHED with the
# args in /content/p0_args.json, so no client-side exec timeout can interrupt the job (RUNNER.md 9, trap 4).
# ASCII only (colab-cli reads this file with the dev box's codec).
import json
import os
import subprocess
import tarfile

os.makedirs("/content/p0/code", exist_ok=True)
with tarfile.open("/content/p0_code.tgz") as tf:
    tf.extractall("/content/p0/code")
args = json.load(open("/content/p0_args.json"))
log = open("/content/p0/runner.log", "a")
p = subprocess.Popen(["python3", "/content/p0/code/navtest/vm_p0.py"] + args, stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True, cwd="/content/p0")
open("/content/p0/runner.pid", "w").write(str(p.pid))
print("ZZLAUNCHED", p.pid, json.dumps(args))
