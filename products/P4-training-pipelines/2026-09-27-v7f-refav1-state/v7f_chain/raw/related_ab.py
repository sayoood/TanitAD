"""The 7 other test files that reference v6_chain: run them on the ORIGINAL chain and on the PATCHED chain
(same isolated copy, one pytest at a time) and compare the failed/error test-id SETS.
usage: python related_ab.py <orig_chain> <out_json>"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

STACK = Path("<V7F_CHAIN>/stack")
F = STACK / "scripts" / "v6_chain.py"
orig = Path(sys.argv[1]).read_bytes()
patched = F.read_bytes()
p_sha = hashlib.sha256(patched).hexdigest()
FILES = ["tests/test_launch_closure_audit.py", "tests/test_nav_v6stack.py", "tests/test_pod_git_drift.py",
         "tests/test_runbook_commands.py", "tests/test_v6_dump_sw_latents.py", "tests/test_v6_gstr_port.py",
         "tests/test_v6_st_launch_fixes.py"]
env = dict(os.environ)
env.update(PYTHONPATH=f"{STACK};{STACK.parent}", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4",
           PYTHONUTF8="1", MSYS_NO_PATHCONV="1")


def run(tag):
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfEs", *FILES],
                       cwd=str(STACK), env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    bad = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", p.stdout, re.M)))
    summ = [ln for ln in p.stdout.strip().splitlines() if re.search(r"\d+ passed", ln)][-1:]
    return {"tag": tag, "rc": p.returncode, "summary": summ, "bad": bad}


res = {}
try:
    F.write_bytes(orig)
    res["orig"] = run("orig")
finally:
    F.write_bytes(patched)
assert hashlib.sha256(F.read_bytes()).hexdigest() == p_sha, "RESTORE FAILED"
res["patched"] = run("patched")
res["same_bad_set"] = res["orig"]["bad"] == res["patched"]["bad"]
res["patched_sha256"] = p_sha
Path(sys.argv[2]).write_text(json.dumps(res, indent=1), encoding="utf-8")
for k in ("orig", "patched"):
    print(k, res[k]["rc"], res[k]["summary"], len(res[k]["bad"]))
print("same failed/error set:", res["same_bad_set"])
