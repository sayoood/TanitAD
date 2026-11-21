"""Run test files ONE AT A TIME (one child python), CPU only, RAM-gated; write a RESUMABLE JSON summary + one log per file.
usage: suite_runner.py <cfg.json>   cfg = {"plan": <refe-plan layout root>, "out": <out.json>, "logs": <logs dir>, "list": <list file: one `refe/test_x.py[:arg1,arg2]` or `eval/test_y.py` per line>}
A file already in <out.json> is skipped (resume).  The marker <out.json>.DONE is written when the whole list has run."""
import json
import os
import re
import subprocess
import sys
import time

import psutil

PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
cfg = json.load(open(sys.argv[1], encoding="utf-8"))               # {"plan":..., "out":..., "logs":..., "list":...}: paths with spaces and & travel in a file, not on a command line
plan, out_json, logdir, listfile = cfg["plan"], cfg["out"], cfg["logs"], cfg["list"]
os.makedirs(logdir, exist_ok=True)
files = [ln.strip() for ln in open(listfile, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", TEMP=r"C:\Users\Admin\par18\hnfp3\tmp", TMP=r"C:\Users\Admin\par18\hnfp3\tmp")
env["PATH"] = r"C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;" + env.get("PATH", "")   # a WMI-launched process has a bare PATH: the chain tests drive bash scripts that need grep / sed / cat
res = json.load(open(out_json, encoding="utf-8")) if os.path.isfile(out_json) else {}
skipped = []
CNT = re.compile(r"(ZZTEST_PASS|ZZLG_FULL_PASS|ZZLG_STATIC_PASS|ZZP6CHAIN_PASS)\s+(\d+)")
for spec in files:
    if spec in res:
        continue
    f, args = (spec.split(":", 1) + [""])[:2] if ":" in spec else (spec, "")
    args = [a for a in args.split(",") if a]
    sub, name = f.split("/", 1)
    waited = 0
    while (psutil.virtual_memory().available < 2 * 2**30 or psutil.disk_usage("C:" + chr(92)).free < 1.5 * 2**30) and waited < 900:
        time.sleep(15)
        waited += 15
    if psutil.disk_usage("C:" + chr(92)).free < 1.0 * 2**30:                 # C: nearly full (other writers): a test that dies of ENOSPC reads as a regression -> do NOT record it; a resume retries it
        skipped.append(spec)
        print("SKIPPED_NO_DISK", spec, flush=True)
        continue
    t0 = time.time()
    try:
        cp = subprocess.run([PY, name] + args, cwd=os.path.join(plan, sub), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3000)
        rc, so, se = cp.returncode, cp.stdout, cp.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = "TIMEOUT", (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")), ""
    text = so + "\n" + se
    with open(os.path.join(logdir, spec.replace("/", "__").replace(":", "_").replace(",", "_") + ".log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text[-300000:])
    lines = [ln for ln in text.splitlines() if ln.strip()]
    counts = [(m.group(1), int(m.group(2))) for m in CNT.finditer(so)]
    res[spec] = {"rc": rc, "secs": round(time.time() - t0, 1), "counts": counts, "skips": sum(1 for ln in lines if ln.startswith("ZZTEST_SKIP")), "tail": [ln[:200] for ln in lines[-4:]]}
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print(spec, rc, res[spec]["secs"], counts, flush=True)
if not skipped:
    open(out_json + ".DONE", "w").write("done" + chr(10))
else:
    print("NOT DONE: skipped for lack of disk on C:", skipped, flush=True)
