"""One source mutation at a time on the ACTRSP tree; run the test file; expect the named tests RED and the named controls GREEN; restore byte for byte (sha256 re-checked).
usage: python mutate_rsp.py <mutations.json> [<id> ...]   (ONE pytest process at a time, never parallel)"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
tree, py = cfg["tree"], cfg["py"]
env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join([os.path.join(tree, "stack"), os.path.join(tree, "taniteval"), os.path.join(tree, "stack", "scripts")])
env["CUDA_VISIBLE_DEVICES"] = "-1"
env["OMP_NUM_THREADS"] = "2"
env["PYTHONDONTWRITEBYTECODE"] = "1"
ids = sys.argv[2:] or list(cfg["muts"])
out_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "mutation_results_rsp_ALL.json")
rows = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else []
rows = [r for r in rows if r["id"] not in ids]
for mid in ids:
    m = cfg["muts"][mid]
    path = os.path.join(tree, m["file"])
    raw = open(path, "rb").read()
    h0 = hashlib.sha256(raw).hexdigest()
    old, new = m["old"].encode("utf-8"), m["new"].encode("utf-8")
    n = raw.count(old)
    if n != 1:
        rows.append({"id": mid, "result": "NOT-APPLIED", "occurrences": n})
        print("RESULT", json.dumps(rows[-1]), flush=True)
        continue
    try:
        open(path, "wb").write(raw.replace(old, new))
        bt = os.path.join(cfg["basetemp"], "mut_" + mid)
        r = subprocess.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", bt, m["tests"]], cwd=tree, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=3000, stdin=subprocess.DEVNULL)
        out = r.stdout
        failed = sorted(set(re.findall(r"^(?:FAILED|ERROR)\s+\S*?::(\S+?)(?:\[|\s|$)", out, flags=re.M)))
        tail = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed)", ln)][-1:]
        missing = [w for w in m.get("must_fail", []) if w not in failed]
        controls_red = [w for w in m.get("must_pass", []) if w in failed]
        res = "RED" if (failed and not missing and not controls_red) else "GREEN-SURVIVED" if not failed else "RED-BUT-CHECK"
        rows.append({"id": mid, "what": m.get("what"), "result": res, "rc": r.returncode, "must_fail": m.get("must_fail", []), "expected_not_failed": missing, "controls_checked": m.get("must_pass", []), "controls_that_went_red": controls_red,
                     "failed_all": failed, "summary": tail})
    finally:
        open(path, "wb").write(raw)
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == h0, "restore failed " + path
        shutil.rmtree(os.path.join(cfg["basetemp"], "mut_" + mid), ignore_errors=True)
    print("RESULT", json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(out_path, "w", encoding="utf-8"), indent=1)
