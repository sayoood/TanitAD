"""Mutation check: flip the floor comparison in obd_read.cell_delta (the line tagged FLOORCMP: `above = abs(pt) > floor`) to `abs(pt) <= floor` and, separately, drop it
(`above = True`); require the tests to go RED under each.  Writes tests/mutation_floor.log (pytest output of both) and tests/mutation_floor.json.
Temp copy: two levels deep under F:/TanitAD-scratch/obd_read_build/work/ (deleted by this script, literal path)."""
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
W = "F:/TanitAD-scratch/obd_read_build/work/mut"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
src = open(R + "/obd_read.py", encoding="utf-8").read()
old = "above = abs(pt) > floor    # FLOORCMP"
assert src.count(old) == 1, "mutation target not found exactly once"
MUTS = {"flipped": "above = abs(pt) <= floor    # MUTATED: comparison flipped", "dropped": "above = True    # MUTATED: floor comparison dropped"}
os.makedirs(W, exist_ok=True)
res, log = {}, []
for name, new in MUTS.items():
    mut = f"{W}/obd_read_{name}.py"
    open(mut, "w", encoding="utf-8").write(src.replace(old, new))
    env = dict(os.environ, OBD_READ_SRC=mut, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="F:/TanitAD-scratch/otband_read/tree/stack;F:/TanitAD-scratch/otband_read/tree/taniteval")
    p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", f"{W}/bt_{name}", R + "/test_obd_read.py"], env=env, capture_output=True, encoding="utf-8")
    log.append(f"===== mutation {name} (rc {p.returncode}) =====\n" + p.stdout + p.stderr)
    failed = [l for l in p.stdout.splitlines() if l.startswith("FAILED")]
    res[name] = {"rc": p.returncode, "n_failed": len(failed), "failed": failed, "red": p.returncode != 0}
open(HERE + "/mutation_floor.log", "w", encoding="utf-8").write("\n".join(log))
json.dump(res, open(HERE + "/mutation_floor.json", "w"), indent=1)
shutil.rmtree(W)          # literal path above, created by this script
print(json.dumps({k: (v["rc"], v["n_failed"]) for k, v in res.items()}))
assert all(v["red"] for v in res.values()), "a mutation did not turn the tests red"
