"""Mutation check of the combination rule's clause 'at least one of them at each init' (obd_read_seed3.combine4, the line tagged INITCLAUSE):
  flipped: `each_init = not ((0 in held) and (1 in held))`   dropped: `each_init = True`
Both must turn test_obd_read_seed3.py RED.  Expected shape (MEASURED by this runner, see mutation_initclause.json): the flip turns many tests red (every 3-of-4 majority);
the drop turns only the planted unbalanced-layout test red, because the real layout has two seeds per init, so every 3-of-4 majority already holds both inits.
Writes tests/mutation_initclause.log and .json.  Temp copy under F:/TanitAD-scratch/obd_seed3_read/work/mut (deleted by this script, literal path)."""
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.dirname(HERE)
READ = os.path.join(os.path.dirname(R3), "read")
W = "F:/TanitAD-scratch/obd_seed3_read/work/mut"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
src = open(R3 + "/obd_read_seed3.py", encoding="utf-8").read()
old = "each_init = (0 in held) and (1 in held)       # INITCLAUSE"
assert src.count(old) == 1, "mutation target not found exactly once"
MUTS = {"flipped": "each_init = not ((0 in held) and (1 in held))       # MUTATED: clause flipped", "dropped": "each_init = True       # MUTATED: clause dropped"}
os.makedirs(W, exist_ok=True)
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=";".join(["F:/TanitAD-scratch/otband_read/tree/stack", "F:/TanitAD-scratch/otband_read/tree/taniteval"]))
pc = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", f"{W}/bt_clean", R3 + "/test_obd_read_seed3.py", R3 + "/test_run_dumps_seed3.py"],
                    env=env0, capture_output=True, encoding="utf-8")
open(HERE + "/pytest_clean.log", "w", encoding="utf-8").write(pc.stdout + pc.stderr)
assert pc.returncode == 0, "the clean tests are not green"
res, log = {}, []
for name, new in MUTS.items():
    mut = f"{W}/obd_read_seed3_{name}.py"
    open(mut, "w", encoding="utf-8").write(src.replace(old, new))
    env = dict(os.environ, OBD_SEED3_SRC=mut, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=";".join([R3, READ, "F:/TanitAD-scratch/otband_read/tree/stack", "F:/TanitAD-scratch/otband_read/tree/taniteval"]))
    p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", f"{W}/bt_{name}", R3 + "/test_obd_read_seed3.py"], env=env, capture_output=True, encoding="utf-8")
    log.append(f"===== mutation {name} (rc {p.returncode}) =====\n" + p.stdout + p.stderr)
    failed = [l for l in p.stdout.splitlines() if l.startswith("FAILED")]
    res[name] = {"rc": p.returncode, "n_failed": len(failed), "failed": failed, "red": p.returncode != 0}
open(HERE + "/mutation_initclause.log", "w", encoding="utf-8").write("\n".join(log))
json.dump(res, open(HERE + "/mutation_initclause.json", "w"), indent=1)
shutil.rmtree(W)          # literal path above, created by this script
print(json.dumps({k: (v["rc"], v["n_failed"]) for k, v in res.items()}))
assert all(v["red"] for v in res.values()), "a mutation did not turn the tests red"
