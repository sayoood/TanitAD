"""Run one source mutation at a time on the ACTSENS working tree, run the named test file, expect RED, restore the file byte for byte.

usage: python mutate_actsens.py <mutations.json> <id> [<id> ...]
JSON: {"tree": "...", "py": "...", "basetemp": "...", "muts": {"M1": {"file": "stack/...", "old": "...", "new": "...", "tests": "stack/tests/x.py", "must_fail": ["test_name", ...]}}}
Prints one RESULT line per mutation (RED = the expected tests failed; GREEN = a mutation survived). The file is always restored in a finally block and its sha256 re-checked.
"""
import hashlib
import json
import os
import re
import subprocess
import shutil
import sys

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
tree, py = cfg["tree"], cfg["py"]
env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join([os.path.join(tree, "stack"), os.path.join(tree, "taniteval")])
env["CUDA_VISIBLE_DEVICES"] = "-1"
env.pop("MSYS_NO_PATHCONV", None)
ids = sys.argv[2:] or list(cfg["muts"])
rows = []
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
        r = subprocess.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", bt, m["tests"]], cwd=tree, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=3000)
        out = r.stdout
        failed = sorted(set(re.findall(r"^(?:FAILED|ERROR)\s+\S*?::(\S+?)(?:\[|\s|$)", out, flags=re.M)))
        tail = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed)", ln)][-1:]
        want = m.get("must_fail", [])
        missing = [w for w in want if w not in failed]
        res = "RED" if (failed and not missing) else "GREEN-SURVIVED"
        rows.append({"id": mid, "result": res, "rc": r.returncode, "failed": failed, "expected_not_failed": missing, "summary": tail})
    finally:
        open(path, "wb").write(raw)
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == h0, "restore failed " + path
        shutil.rmtree(os.path.join(cfg["basetemp"], "mut_" + mid), ignore_errors=True)
    print("RESULT", json.dumps(rows[-1]), flush=True)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "mutation_results_" + os.path.basename(sys.argv[1])), "w", encoding="utf-8"), indent=1)
