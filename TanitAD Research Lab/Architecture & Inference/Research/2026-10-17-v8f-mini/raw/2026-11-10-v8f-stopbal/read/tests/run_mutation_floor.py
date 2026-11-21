"""Mutation check: drop the floor comparison from stopbal_read.cell_delta (the line tagged FLOORCMP) and require the tests to go RED.
Writes tests/mutation_floor.log (pytest output under the mutation) and tests/mutation_floor.json.  Temp copy: two levels deep under F:/TanitAD-scratch/stopbal_read/work/ (deleted at the end)."""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
W = "F:/TanitAD-scratch/stopbal_read/work/mut"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
src = open(R + "/stopbal_read.py", encoding="utf-8").read()
old = "above = abs(pt) > floor    # FLOORCMP"
assert src.count(old) == 1, "mutation target not found exactly once"
os.makedirs(W, exist_ok=True)
mut = W + "/stopbal_read_mut.py"
open(mut, "w", encoding="utf-8").write(src.replace(old, "above = True    # MUTATED: floor comparison dropped"))
env = dict(os.environ, STOPBAL_READ_SRC=mut, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", R + "/test_stopbal_read.py"], env=env, capture_output=True, encoding="utf-8")
open(HERE + "/mutation_floor.log", "w", encoding="utf-8").write(p.stdout + p.stderr)
failed = [l for l in p.stdout.splitlines() if l.startswith("FAILED")]
json.dump({"rc": p.returncode, "failed": failed, "red": p.returncode != 0}, open(HERE + "/mutation_floor.json", "w"), indent=1)
shutil.rmtree(W)          # literal path above, created by this script
print("rc", p.returncode, failed)
assert p.returncode != 0 and any("subfloor" in f or "sub_floor" in f for f in failed), "the mutation did not turn the sub-floor test red"
