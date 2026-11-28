#!/usr/bin/env python3
"""Generic Thor gate runner. plan_thor.json = {"archive": <tar>, "overlay_md5": {target: md5},
"stages": {name: {"overlay": [targets], "stack": [files rel. to stack/], "taniteval": [files rel. to taniteval/]}}}.
Each stage: one tree extracted from the archive + its overlay (md5-verified after writing); one pytest process per
file with junit; stages run concurrently; progress and end state are FILES (<stage>.json, ALLDONE).
THOR_GATE_WORKERS=N (default 1 = the serial order, unchanged) runs up to N pytest processes at once WITHIN each stage. Every
per-file command, junit path, <stage>.json record and its file ORDER stay as in the serial run (only "s" timings differ);
with N>1 each worker slot also gets its own TMPDIR so concurrent pytest basetemp rotation cannot touch another slot's dirs."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

PY = "/home/nvidia/venvs/tanitad-train/bin/python"
R = PYT = plan = OUT = None                          # set by main(): <gate dir>, the --no-deps pytest side folder


def n_workers():
    try:
        return max(1, int(os.environ.get("THOR_GATE_WORKERS", "1")))
    except ValueError:
        return 1


def env_for(tree, tmpdir=None):
    e = dict(os.environ)
    e.update({"PYTHONPATH": f"{tree}/stack:{tree}/stack/scripts:{tree}/taniteval:{PYT}", "OMP_NUM_THREADS": "4",
              "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
              "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})
    # 2026-09-27: huggingface_hub's offline-mode error text dumps the request headers, incl. the
    # Bearer token, into the failing trunk tests' output. No token enters the gate environment.
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        e.pop(k, None)
    e["HF_TOKEN_PATH"] = "/nonexistent/hf_token"
    if tmpdir is not None:
        e["TMPDIR"] = str(tmpdir)
    return e


def junit_path(name, sub, f):
    return OUT / name / (sub + "__" + f.replace("/", "__") + ".xml")


def run_file(tree, sub, f, x, tmpdir=None):
    t0 = time.time()
    try:
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", f"--junitxml={x}", f],
                           cwd=tree / sub, env=env_for(tree, tmpdir), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=2400)
        rc, tail = r.returncode, (r.stdout[-300:] + r.stderr[-200:])
    except subprocess.TimeoutExpired:
        rc, tail = "TIMEOUT", ""
    return {"cwd": sub, "file": f, "rc": rc, "s": round(time.time() - t0, 1), "junit": x.exists(), "tail": tail}


def run_stage(name, st):
    rec = {"stage": name, "overlay": st["overlay"], "files": [], "started": time.time()}
    try:
        tree = R / name
        tree.mkdir()
        with tarfile.open(R / plan["archive"]) as tf:
            tf.extractall(tree)
        for t in st["overlay"]:
            d = tree / t
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes((R / "overlay" / t).read_bytes())
            assert hashlib.md5(d.read_bytes()).hexdigest() == plan["overlay_md5"][t], t
        probe = subprocess.run([PY, "-c", "import tanitad, taniteval; print(tanitad.__file__); print(taniteval.__file__)"],
                               cwd=tree, env=env_for(tree), capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=300)
        rec["import_probe"] = probe.stdout.strip().splitlines()
        assert len(rec["import_probe"]) == 2 and all(str(tree) in ln for ln in rec["import_probe"]), probe.stdout + probe.stderr
        (OUT / name).mkdir(exist_ok=True)
        jobs = [(sub, f) for sub in ("stack", "taniteval") for f in st[sub]]
        n = n_workers()
        if n <= 1:
            for sub, f in jobs:
                rec["files"].append(run_file(tree, sub, f, junit_path(name, sub, f)))
                (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
        else:
            paths = [junit_path(name, sub, f) for sub, f in jobs]
            assert len(set(paths)) == len(paths), "two files map to one junit path"
            wtmp = R / "wtmp" / name
            slots = [wtmp / f"w{k}" for k in range(n)]
            for d in slots:
                d.mkdir(parents=True, exist_ok=True)
            free, done, lock = list(slots), {}, threading.Lock()

            def work(i):
                with lock:
                    d = free.pop()
                try:
                    res = run_file(tree, jobs[i][0], jobs[i][1], paths[i], d)
                finally:
                    with lock:
                        free.append(d)
                with lock:                        # the progress record lists finished files in PLAN order, as the serial run does
                    done[i] = res
                    rec["files"] = [done[k] for k in sorted(done)]
                    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))

            try:
                with ThreadPoolExecutor(max_workers=n) as ex:
                    for fut in [ex.submit(work, i) for i in range(len(jobs))]:
                        fut.result()
            finally:
                shutil.rmtree(wtmp, ignore_errors=True)
        rec["done"] = True
    except Exception as ex:
        rec["error"] = repr(ex)
    rec["ended"] = time.time()
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))


def main():
    global R, PYT, plan, OUT
    R, PYT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    plan = json.load(open(R / "plan_thor.json"))
    OUT = R / "results"
    OUT.mkdir(exist_ok=True)
    with tarfile.open(R / "overlay.tar") as tf:
        tf.extractall(R)
    th = [threading.Thread(target=run_stage, args=(n, s)) for n, s in plan["stages"].items()]
    for t in th:
        t.start()
    for t in th:
        t.join()
    (OUT / "ALLDONE").write_text("done\n")


if __name__ == "__main__":
    main()
