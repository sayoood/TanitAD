"""Mutation check: replace the floor rule (the line tagged FLOORMAX in stopperc_s1_read.py: max of the two replicate gaps) by the NULL_rep gap alone, and require the tests to go RED.
Writes tests/mutation_floor_s1.log and .json.  Temp copy: two levels deep under F:/TanitAD-scratch/stopperc_read/ (deleted at the end by its literal path)."""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
W = "F:/TanitAD-scratch/stopperc_read/mut"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
src = open(R + "/stopperc_s1_read.py", encoding="utf-8").read()
old = "return max(gap_rep, gap_rep2)  # FLOORMAX"
assert src.count(old) == 1, "mutation target not found exactly once"
os.makedirs(W, exist_ok=True)
mut = W + "/stopperc_s1_read_mut.py"
open(mut, "w", encoding="utf-8", newline="\n").write(src.replace(old, "return gap_rep  # MUTATED: NULL_rep gap alone"))
env = dict(os.environ, STOPPERC_READ_SRC=mut, STOPPERC_THEORY=os.path.dirname(os.path.dirname(os.path.dirname(R))), CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", W + "/pt", R + "/test_stopperc_s1_read.py"], env=env, capture_output=True, encoding="utf-8", errors="replace")
open(HERE + "/mutation_floor_s1.log", "w", encoding="utf-8").write(p.stdout + p.stderr)
failed = [l for l in p.stdout.splitlines() if l.startswith("FAILED")]
json.dump({"rc": p.returncode, "failed": failed, "red": p.returncode != 0}, open(HERE + "/mutation_floor_s1.json", "w"), indent=1)
shutil.rmtree(W)          # literal path above, created by this script
print("rc", p.returncode, failed)
assert p.returncode != 0 and any("between" in f for f in failed), "the mutation did not turn the between-the-floors tests red"
