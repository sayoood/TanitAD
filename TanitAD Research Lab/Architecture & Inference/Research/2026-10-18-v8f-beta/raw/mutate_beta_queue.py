"""Mutation check for the rung-beta box-2 queue (PREREG_V8F_BETA Amendment 1): each mutation must turn test_v8f_beta_box2.py RED and
the clean queue must be GREEN. Restores the queue after every mutation (binary I/O)."""
import json
import os
import subprocess
import sys

q, stack, out = sys.argv[1:4]
BS = chr(92).encode()
o = open(q, "rb").read()
M = {"unknown_flag": (b"--plan beta --families", b"--plan beta --families --no-such-flag"),
     "eval_dir_is_the_k1_dir": (b'set "EVB=%RUNSM%' + BS + b'eval_beta"', b'set "EVB=%RUNSM%' + BS + b'eval"'),
     "seed_not_passed": (b"call :run %2", b"call :run %1")}


def suite():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v8f_beta_box2.py"], cwd=stack,
                       capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    return r.returncode, r.stdout.strip().splitlines()[-1]


res = {"clean": suite()}
try:
    for k, (a, b) in M.items():
        assert o.count(a) >= 1, k
        open(q, "wb").write(o.replace(a, b))
        res[k] = suite()
        open(q, "wb").write(o)
finally:
    open(q, "wb").write(o)
r = {k: {"rc": v[0], "tail": v[1], "verdict": "GREEN" if v[0] == 0 else "RED"} for k, v in res.items()}
r["ALL_MUTATIONS_RED_AND_CLEAN_GREEN"] = r["clean"]["verdict"] == "GREEN" and all(v["verdict"] == "RED" for k, v in r.items() if k != "clean")
json.dump(r, open(out, "w", newline="\n"), indent=1)
print(json.dumps({k: (v["verdict"] if isinstance(v, dict) else v) for k, v in r.items()}))
sys.exit(0 if r["ALL_MUTATIONS_RED_AND_CLEAN_GREEN"] else 1)
