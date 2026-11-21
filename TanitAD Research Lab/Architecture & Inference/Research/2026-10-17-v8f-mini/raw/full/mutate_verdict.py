"""Mutation check for v8f_mini_full_verdict.py (PREREG_V8F_MINI Amendment 4): each mutation must turn test_v8f_mini_full.py RED and the
clean source must be GREEN. Restores the file after every mutation (binary I/O)."""
import json
import os
import subprocess
import sys

TREE = sys.argv[1]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SRC = os.path.join(TREE, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini", "code",
                   "v8f_mini_full_verdict.py")
MUT = {
    "w1_any_seed_not_both": ('ok = all(c["separated"] and c["mean"] < 0 and abs(c["mean"]) > 2 * Fmax for c in ci)',
                             'ok = any(c["separated"] and c["mean"] < 0 and abs(c["mean"]) > 2 * Fmax for c in ci)'),
    "w2_no_2F_rule": ('and abs(c["mean"]) > 2 * Fmax for c in ci)', 'for c in ci)'),
    "w3_sign_dropped": ('c["separated"] and c["mean"] < 0 and', 'c["separated"] and'),
    "p1_holdout_unchecked": ('if {d.get("excluded_sha256") for d in S.values()} != {HOLDOUT_SHA}:', 'if False:'),
    "p2_train_list_unchecked": ('if {d["train_sha256"] for d in S.values()} != {S[("M1", 0)]["train_sha256"]}:', 'if False:'),
    "p3_missing_run_skipped": ('raise SystemExit(f"REFUSED: {p} is missing (the run did not finish)")', 'continue'),
    "k1_all_not_any": ('k1_pass = any(bool(k1[arm]["K1_pass"]) for arm in ARMS)', 'k1_pass = all(bool(k1[arm]["K1_pass"]) for arm in ARMS)'),
}


def suite():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v8f_mini_full.py", "-k", "verdict"],
                       cwd=os.path.join(TREE, "stack"), capture_output=True, text=True, env=env)
    return r.returncode, r.stdout.strip().splitlines()[-1]


orig = open(SRC, "rb").read()
res = {}
try:
    res["clean"] = suite()
    for k, (old, new) in MUT.items():
        s = orig.decode("utf-8")
        assert s.count(old) == 1, (k, s.count(old))
        open(SRC, "wb").write(s.replace(old, new).encode("utf-8"))
        res[k] = suite()
        open(SRC, "wb").write(orig)
finally:
    open(SRC, "wb").write(orig)
out = {k: {"rc": v[0], "tail": v[1], "verdict": ("GREEN" if v[0] == 0 else "RED")} for k, v in res.items()}
ok = out["clean"]["verdict"] == "GREEN" and all(v["verdict"] == "RED" for k, v in out.items() if k != "clean")
out["ALL_MUTATIONS_RED_AND_CLEAN_GREEN"] = ok
print(json.dumps({k: (v["verdict"] if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=1)
sys.exit(0 if ok else 1)
