"""The test matrix of C2-GATE-SIBLING-LEAK, one python at a time, CPU only, detached.  Appends one line per run to <evidence>/matrix_results.jsonl and writes <evidence>/logs/<job>.log.

    python -B -I run_matrix.py <work dir (c2leak)> [only-substring]

Trees under <work>: base/ (the tip files), mine/ (the fix), nl/ (mine + the earlier NOT-landed attempt's test file).  Writes nothing outside <work> and the temp dir (C:).  When the last job ends it runs
<work>/code/finalize.py, so the deliverable is complete without anyone polling.
"""
import json
import os
import subprocess
import sys
import time

work = os.path.abspath(sys.argv[1])
only = sys.argv[2] if len(sys.argv) > 2 else ""
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
ev = os.path.join(work, "evidence")
logs = os.path.join(ev, "logs")
os.makedirs(logs, exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_SITE": os.path.join(work, "pytest_site")})
RUNNER_PY = os.path.join(work, "code", "run_order.py")
GATE_TEST = "test_srttc_c2_gate.py"
R1_TEST = "test_srttc_r1.py"
TIMEOUT = 1800


def sh(args, log=None, timeout=TIMEOUT):
    p = subprocess.Popen([PY, "-B", "-I", RUNNER_PY, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
    try:
        out, _ = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()                                                                    # this child only, by its own handle: never a name-pattern kill
        out, _ = p.communicate()
        rc = "TIMEOUT"
    text = out.decode("utf-8", "replace")
    if log:
        open(log, "wb").write(out)
    return rc, text


def list_ids(tree, runner, fname):
    rc, text = sh(["--list", tree, runner, fname])
    line = [ln for ln in text.splitlines() if ln.startswith("IDS ")]
    if not line:
        raise SystemExit(f"cannot list {tree} {runner} {fname}: {text[-500:]}")
    return json.loads(line[-1][4:])


JOBS = []
for order in ("default", "first", "last"):
    for variant in ("mine", "base"):
        JOBS.append((variant, "unittest", GATE_TEST, order))
for order in ("default", "first", "last", "reversed"):
    for variant in ("mine", "base"):
        JOBS.append((variant, "pytest", GATE_TEST, order))
for runner in ("unittest", "pytest"):
    for variant in ("mine", "base"):
        JOBS.append((variant, runner, R1_TEST, "default"))
JOBS.append(("nl", "unittest", GATE_TEST, "default"))                                 # the earlier attempt's tests against the fixed gate (cross-check)
JOBS.append(("nlbase", "unittest", GATE_TEST, "default"))                             # the earlier attempt's gate AND tests (c2_not_landed/): reproduces its 42 / 43

ids_cache = {}
done = set()
res_path = os.path.join(ev, "matrix_results.jsonl")
if os.path.exists(res_path):
    for ln in open(res_path, encoding="utf-8"):
        try:
            r = json.loads(ln)
            done.add((r["variant"], r["runner"], r["file"], r["order"]))
        except Exception:                                                           # noqa: BLE001
            pass
for variant, runner, fname, order in JOBS:
    if only and only not in f"{variant}_{runner}_{fname}_{order}":
        continue
    if (variant, runner, fname, order) in done:
        continue
    tree = os.path.join(work, variant)
    key = (variant, runner, fname)
    if key not in ids_cache:
        ids_cache[key] = list_ids(tree, runner, fname)
        idf = os.path.join(ev, f"ids_{variant}_{runner}_{fname[:-3]}.json")
        json.dump(ids_cache[key], open(idf, "w"))
    idf = os.path.join(ev, f"ids_{variant}_{runner}_{fname[:-3]}.json")
    name = f"{variant}_{runner}_{fname[:-3]}_{order}"
    t0 = time.time()
    rc, text = sh(["--run", tree, runner, fname, order, idf], log=os.path.join(logs, name + ".log"))
    line = [ln for ln in text.splitlines() if ln.startswith("RESULT ")]
    rec = json.loads(line[-1][7:]) if line else {"runner": runner, "file": fname, "order": order, "error": "no RESULT line", "tail": text[-400:]}
    rec.update({"variant": variant, "rc": rc, "wall_s": round(time.time() - t0, 1), "listed_default": len(ids_cache[key])})
    with open(res_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] {name}: rc={rc} ran={rec.get('ran')} passed={rec.get('passed')} failed={rec.get('failed')} errors={rec.get('errors')} ({rec['wall_s']} s)", flush=True)
if not only:
    fin = os.path.join(work, "code", "finalize.py")
    if os.path.exists(fin):
        subprocess.run([sys.executable, "-B", "-I", fin, work], check=False)
print("MATRIX DONE", flush=True)
