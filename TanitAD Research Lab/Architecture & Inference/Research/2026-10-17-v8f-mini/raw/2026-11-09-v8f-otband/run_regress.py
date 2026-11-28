"""OTBAND regression runner (CPU, one pytest at a time). For every test file that names v8f_mini / v8f_mini_train / v8f_mini_data (except test_v8f_otband.py): run it on the OTBAND tree; every file with a failure is
re-run on the BASE tree (ladder commit 1faa1a06, F:/TanitAD-scratch/ladder_tree, read-only) and the two failure sets are compared. Result: logs/regress_summary.json (+ one log per file under logs/reg/).
A failure that the base tree has too is export-limited or pre-existing (not ours); a failure only on the OTBAND tree is a regression. Usage: python run_regress.py [name ...]"""
import json
import os
import re
import subprocess
import sys
import time

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = HERE + "/logs/reg"
SUM = HERE + "/logs/regress_summary.json"
TREES = {"otband": "F:/TanitAD-scratch/hwm_bands/b/tree", "base": "F:/TanitAD-scratch/ladder_tree"}
SCR = "F:/TanitAD-scratch/hwm_bands/b/regress_tmp"


def files(tree):
    out = []
    for fn in sorted(os.listdir(tree + "/stack/tests")):
        if fn.startswith("test_") and fn.endswith(".py") and "test_v8f_otband" not in fn:
            if "v8f_mini" in open(tree + "/stack/tests/" + fn, encoding="utf-8", errors="replace").read():
                out.append("tests/" + fn)
    return out


def run(which, f, timeout=5400):
    tree = TREES[which]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", TMP=SCR, TEMP=SCR,
               PYTHONPATH=";".join(p.replace("/", "\\") for p in (tree + "/stack", tree + "/stack/scripts", tree + "/taniteval")))
    name = os.path.basename(f)[:-3]
    bt = f"{SCR}/bt_{which}_{name}"
    t0 = time.time()
    try:
        r = subprocess.run([PY, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", "--deselect", "tests/test_sigreg_gloo.py::test_sigreg_gloo_two_process", f"--basetemp={bt}", f],
                           cwd=tree + "/stack", env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        out, rc = r.stdout, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", 124
    open(f"{LOGS}/{which}_{name}.log", "w", encoding="utf-8").write(out)
    bad = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)))
    last = out.strip().splitlines()[-1] if out.strip() else ""
    subprocess.run(["cmd.exe", "/c", "rmdir", "/s", "/q", bt.replace("/", "\\")], capture_output=True)
    return {"rc": rc, "failed": bad, "last": last, "seconds": round(time.time() - t0)}


os.makedirs(LOGS, exist_ok=True)
os.makedirs(SCR, exist_ok=True)
only = sys.argv[1:]
res = json.load(open(SUM)) if os.path.exists(SUM) else {}
for f in files(TREES["otband"]):
    n = os.path.basename(f)[:-3]
    if only and n not in only:
        continue
    if n in res and res[n].get("otband", {}).get("rc") is not None and not only:
        continue
    res[n] = {"otband": run("otband", f)}
    if res[n]["otband"]["rc"] != 0:
        res[n]["base"] = run("base", f)
        a, b = set(res[n]["otband"]["failed"]), set(res[n]["base"]["failed"])
        res[n]["only_in_otband"], res[n]["only_in_base"] = sorted(a - b), sorted(b - a)
    json.dump(res, open(SUM, "w"), indent=1)
    print(n, res[n]["otband"]["last"], res[n].get("only_in_otband"), flush=True)
json.dump(res, open(SUM, "w"), indent=1)
open(HERE + "/logs/regress.done", "w").write("done\n")
