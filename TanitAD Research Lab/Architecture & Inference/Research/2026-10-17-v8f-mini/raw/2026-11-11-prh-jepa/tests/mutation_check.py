"""Mutation check: flip the separation test (lo > 0 -> lo < 0 in is_sep_pos) in a scratch copy of prh_seed.py and require test_prh_seed.py to go RED.
Writes MUTATION_PRH_SEED.json beside this file."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PRH = os.path.dirname(HERE)
PY = sys.executable


def run(src_dir):
    env = dict(os.environ, PRH_SRC_DIR=src_dir, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.join(PRH, "test_prh_seed.py")], capture_output=True, text=True, encoding="utf-8", env=env)
    return r.returncode, (r.stdout or "")[-600:]


tmp = tempfile.mkdtemp(prefix="prh_mut_", dir="F:/TanitAD-scratch/prh_seed/work")
src = open(os.path.join(PRH, "prh_seed.py"), encoding="utf-8").read()
needle = "    return lo > 0\n\n\ndef is_sep_any"
assert needle in src
open(os.path.join(tmp, "prh_seed.py"), "w", encoding="utf-8").write(src.replace(needle, "    return lo < 0\n\n\ndef is_sep_any"))
rc_clean, out_clean = run(PRH)
rc_mut, out_mut = run(tmp)
res = {"clean_rc": rc_clean, "mutated_rc": rc_mut, "mutation": "is_sep_pos: lo > 0 -> lo < 0", "clean_tail": out_clean, "mutated_tail": out_mut,
       "mutation_detected": bool(rc_clean == 0 and rc_mut != 0)}
json.dump(res, open(os.path.join(HERE, "MUTATION_PRH_SEED.json"), "w", encoding="utf-8"), indent=1)
shutil.rmtree(tmp)
print(res["mutation_detected"], rc_clean, rc_mut)
