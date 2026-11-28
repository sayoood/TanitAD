#!/usr/bin/env python3
"""queue.py <queue.json> <logdir> : run the tests of the queue ONE AT A TIME at BELOW_NORMAL priority; before each wait (script, not model turns) until free RAM >= 8 GB; each under a timeout; kill ONLY the
test's own process tree by explicit PID.  queue.json = [{"id":..., "tree": "<pkg root>", "cmd": ["refe/test_x.py", "args"...], "env": {..}, "timeout": 2400}].  Appends one line per run to <logdir>/SUMMARY.txt,
writes <logdir>/<id>.log, and <logdir>/QUEUE_DONE when finished.  Nothing else."""
import json, os, re, subprocess, sys, time
q = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"


def free_gb():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], capture_output=True, text=True)
    return int(r.stdout.strip()) / 1048576


for job in q:
    waited = 0
    while free_gb() < 8.0:
        time.sleep(20)
        waited += 20
    env = dict(os.environ, PYTHONIOENCODING="utf-8", **job.get("env", {}))
    env.pop("PYTHONPATH", None) if not job.get("keep_pythonpath") else None
    # a WMI-detached process has Git for Windows' dirs NOT on PATH and no HOME (the interactive Git-bash environment has both): put them back, exactly as the 2026-10-16 readiness harness did.
    # (Without it test_sft4r_chain / test_p2r_chain fail in `last_exit`'s grep|tail|cut -- MEASURED in the first queue attempt, logs qfinal_attempt1.)
    env["PATH"] = os.pathsep.join(["C:/Program Files/Git/usr/bin".replace("/", os.sep), "C:/Program Files/Git/bin".replace("/", os.sep), env.get("PATH", "")])
    env.setdefault("HOME", env.get("USERPROFILE", "C:" + os.sep + "Users" + os.sep + "Admin"))
    t0 = time.time()
    log = os.path.join(out, job["id"] + ".log")
    with open(log, "wb") as f:
        p = subprocess.Popen([PY, "-X", "utf8", *job["cmd"]], stdout=f, stderr=subprocess.STDOUT, cwd=job["tree"], env=env, creationflags=0x00004000)
        try:
            rc = p.wait(timeout=job.get("timeout", 2400))
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            rc = "TIMEOUT"
    secs = int(time.time() - t0)
    txt = open(log, encoding="utf-8", errors="replace").read()
    toks = re.findall(r"^(ZZ[A-Za-z0-9_\-]*(?:PASS|SKIP|FAIL)[A-Za-z0-9_\-]*[^\n]*)$", txt, re.M)
    skips = len(re.findall(r"ZZTEST_SKIP", txt))
    last = txt.strip().splitlines()[-1][:140] if txt.strip() else ""
    with open(os.path.join(out, "SUMMARY.txt"), "a", encoding="utf-8") as s:
        s.write(f"{job['id']:34s} rc={rc} secs={secs} ram_wait={waited}s skips={skips} tok=[{(toks[-1] if toks else 'NONE')[:80]}] last=[{last}]\n")
open(os.path.join(out, "QUEUE_DONE"), "w").write("done\n")
