"""Re-run the package's two test files on the STAGED code in a deploy layout: <deploy>/code/ = a copy of out/<repo path>/code/*.py, <deploy>/ref/ = the landed ref files (sha256-checked against the staged code/REFS.json).
Records the counts, the tail of each run and the sha256 of every tested file in run/deploy_tests/deploy_tests.json.   usage: run_deploy_tests.py <deploy_dir>
CPU only (CUDA_VISIBLE_DEVICES=-1), one test process at a time.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = "C:/Users/Admin/par18/privgeom"
BASE = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-privgeom-safe"
STAGED = f"{ROOT}/out/{BASE}/code"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    dep = sys.argv[1]
    if os.path.isdir(dep):
        shutil.rmtree(dep)
    os.makedirs(f"{dep}/code")
    os.makedirs(f"{dep}/ref")
    tested = {}
    for n in sorted(os.listdir(STAGED)):
        if n.endswith(".py"):
            shutil.copyfile(f"{STAGED}/{n}", f"{dep}/code/{n}")
            tested[n] = sha(f"{dep}/code/{n}")
    refs = json.load(open(f"{STAGED}/REFS.json", encoding="utf-8"))["sha256"]
    for n, h in refs.items():
        shutil.copyfile(f"{ROOT}/ref/{n}", f"{dep}/ref/{n}")
        if sha(f"{dep}/ref/{n}") != h:
            raise SystemExit(f"ref {n} does not hash to REFS.json")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1")
    res = {}
    for tf in ("test_pg_geom.py", "test_pg_lib.py"):
        r = subprocess.run([PY, "-B", "-W", "ignore", f"{dep}/code/{tf}"], capture_output=True, text=True, env=env, cwd=f"{dep}/code")
        out = r.stdout + r.stderr
        m = re.search(r"Ran (\d+) tests? in", out)
        res[tf] = {"rc": r.returncode, "ran": int(m.group(1)) if m else None, "ok": ("\nOK" in out) and r.returncode == 0, "tail": out[-700:]}
        print(tf, res[tf]["rc"], res[tf]["ran"], res[tf]["ok"], flush=True)
    os.makedirs(f"{ROOT}/run/deploy_tests", exist_ok=True)
    with open(f"{ROOT}/run/deploy_tests/deploy_tests.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"layout": "<deploy>/code (copy of the STAGED code) + <deploy>/ref (landed files, sha256-checked against code/REFS.json)", "tested_sha256": tested, "ref_sha256": refs, "results": res,
                   "all_ok": all(v["ok"] for v in res.values())}, f, indent=1)
    return 0 if all(v["ok"] for v in res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
