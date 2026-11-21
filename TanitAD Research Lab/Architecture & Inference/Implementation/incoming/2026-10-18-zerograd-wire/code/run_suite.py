"""Shard-run a pytest directory in parallel and merge per-test outcomes into one JSON.
usage: run_suite.py TREE_STACK OUT_JSON NSHARDS [extra files to include/exclude via env ONLY=a,b]"""
import json, os, subprocess, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
stack = Path(sys.argv[1]); out = Path(sys.argv[2]); n = int(sys.argv[3])
only = [x for x in os.environ.get("ONLY", "").split(",") if x]
files = sorted(p for p in (stack / "tests").glob("test_*.py") if not only or p.name in only)
files.sort(key=lambda p: -p.stat().st_size)
shards = [[] for _ in range(n)]
load = [0] * n
for f in files:
    i = load.index(min(load)); shards[i].append(f); load[i] += f.stat().st_size + 20000
work = out.parent / (out.stem + "_shards"); work.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
t0 = time.time(); procs = []
for i, sh in enumerate(shards):
    if not sh: continue
    xml = work / ("s%d.xml" % i)
    log = open(work / ("s%d.log" % i), "wb")
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(xml), "-o", "junit_family=xunit1"] + [str(f) for f in sh]
    procs.append((i, subprocess.Popen(cmd, cwd=str(stack), env=env, stdout=log, stderr=subprocess.STDOUT), xml))
res = {}
for i, p, xml in procs:
    p.wait()
    try:
        root = ET.parse(xml).getroot()
    except Exception as e:
        res["SHARD_%d_NO_XML" % i] = "error:%s" % e; continue
    for tc in root.iter("testcase"):
        nid = (tc.get("classname") or "") + "::" + (tc.get("name") or "")
        o = "passed"
        for ch in tc:
            if ch.tag in ("failure", "error"): o = "FAILED" if ch.tag == "failure" else "ERROR"
            elif ch.tag == "skipped": o = "skipped"
        res[nid] = o
out.write_text(json.dumps({"n_files": len(files), "seconds": round(time.time() - t0, 1), "results": res}, indent=1))
c = {}
for v in res.values(): c[v] = c.get(v, 0) + 1
print("DONE", out, json.dumps(c), "files", len(files), "seconds", round(time.time() - t0, 1))
