#!/usr/bin/env python3
"""Sequential test lane: python lane_runner.py <tree_root> <log_dir> <summary.json> test1 [test2 ...]
A test spec is  <relpath-from-PK>[::args joined by +]  e.g.  refe/test_launch_gate.py::--static+--no-banked
Runs each with cwd = its own directory (tests use HERE), UTF-8, no stdin; logs per test; writes the summary after every test."""
import json, os, shutil, subprocess, sys, time
root, logdir, summ = sys.argv[1], sys.argv[2], sys.argv[3]
PK = os.path.join(root, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-09-20-refe-plan")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
os.makedirs(logdir, exist_ok=True)
res = {}
if os.path.exists(summ):
    res = json.load(open(summ))
for spec in sys.argv[4:]:
    rel, _, args = spec.partition("::")
    args = [x for x in args.split("+") if x]
    name = os.path.basename(rel) + ("".join("_" + x.strip("-") for x in args))
    if name in res and res[name].get("rc") == 0:
        continue
    path = os.path.join(PK, rel)
    log = os.path.join(logdir, name + ".log")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env["PATH"] = "C:/Program Files/Git/bin;C:/Program Files/Git/usr/bin;" + env.get("PATH", "")  # Git bash first: a WSL bash.exe on PATH breaks the *_under_bash checks
    tdir = os.path.join("C:/Users/Admin/b8carve/tmp", os.path.basename(root), name)   # per-test TEMP (cleaned after): the shared C: drive filled to 0 free with two lanes in parallel
    shutil.rmtree(tdir, ignore_errors=True); os.makedirs(tdir, exist_ok=True)
    env["TEMP"] = env["TMP"] = tdir
    t0 = time.time()
    TMO = float(os.environ.get("LANE_TIMEOUT", "2700"))
    with open(log, "wb") as f:
        try:
            p = subprocess.run([PY, path] + args, cwd=os.path.dirname(path), stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=TMO)
        except subprocess.TimeoutExpired:
            f.write(b"\nLANE_TIMEOUT: killed by lane_runner\n"); p = type("P", (), {"returncode": 124})()
    shutil.rmtree(tdir, ignore_errors=True)
    txt = open(log, "rb").read().decode("utf-8", "replace")
    tok = [ln for ln in txt.splitlines() if ln.startswith("ZZ")]
    res[name] = {"rc": p.returncode, "secs": round(time.time() - t0, 1), "tokens": tok[-3:], "tail": txt.strip().splitlines()[-3:] if p.returncode else []}
    json.dump(res, open(summ, "w"), indent=1)
json.dump(dict(res, _done=True), open(summ, "w"), indent=1)
