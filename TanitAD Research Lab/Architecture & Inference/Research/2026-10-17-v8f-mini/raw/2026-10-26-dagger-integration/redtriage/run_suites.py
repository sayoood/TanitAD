"""Run the six red suites on redtriage/pristine and redtriage/dagger, ONE pytest at a time, each with the tree's own stack on PYTHONPATH.

  python run_suites.py [--only SUITE] [--trees pristine,dagger] [--tag TAG]

Outputs in redtriage/out: <tree>__<suite>.out (pytest -rfE), .xml (junit), .probe.json (where tanitad etc. were loaded from, recorded INSIDE the run), runner.log
"""
import json
import os
import shutil
import subprocess
import sys
import time

RT = "E:/Projects/TanitAD-artifacts/redtriage"
OUT = RT + "/out"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SUITES = ["test_v8f_speedavg", "test_v8f_lora_ar2", "test_v8f_lora", "test_v8f_kindx_s1", "test_v8f_kscore", "test_v8f_oacc"]


def log(s):
    line = time.strftime("%H:%M:%S ") + s
    print(line, flush=True)
    with open(OUT + "/runner.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def wait_slot():
    for _ in range(1440):
        r = subprocess.run([PY, "C:/Users/Admin/orcread/pytest_runs.py"], capture_output=True, text=True)
        if r.stdout.strip().isdigit() and int(r.stdout.strip()) < 2:
            return
        time.sleep(15)
    raise SystemExit("no pytest slot")


def run(tree, suite, tag):
    tdir = f"{RT}/{tree}"
    pp = os.pathsep.join([f"{tdir}/stack", f"{tdir}/taniteval", f"{tdir}/stack/scripts", RT])
    base = f"{tag}{tree}__{suite}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONPATH": pp,
           "TMP": f"{RT}/tmp", "TEMP": f"{RT}/tmp", "ZZ_PROBE_OUT": f"{OUT}/{base}.probe.json"}
    env.pop("PYTHONHOME", None)
    bt = f"{RT}/bt_{tree}"
    wait_slot()
    free = shutil.disk_usage("E:/").free
    if free < 700 * 2**20:
        raise SystemExit(f"DISK GUARD: {free / 2**20:.0f} MiB free on E:")
    t = time.time()
    with open(f"{OUT}/{base}.out", "w", encoding="utf-8") as fh:
        p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "zz_probe", "-rfE", "--basetemp", bt, f"--junitxml={OUT}/{base}.xml", f"stack/tests/{suite}.py"],
                           cwd=tdir, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=3 * 3600)
    shutil.rmtree(bt, ignore_errors=True)
    tail = open(f"{OUT}/{base}.out", encoding="utf-8", errors="replace").read().strip().splitlines()[-1:]
    log(f"{tree:9s} {suite:20s} rc={p.returncode} [{time.time() - t:.0f}s] {tail}")


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    trees = sys.argv[sys.argv.index("--trees") + 1].split(",") if "--trees" in sys.argv else ["pristine", "dagger"]
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    os.makedirs(OUT, exist_ok=True)
    log(f"runner start: trees={trees} only={only} tag={tag!r}")
    for s in SUITES:
        if only and s != only:
            continue
        for tr in trees:
            run(tr, s, tag)
    log("runner done")


main()
