"""Mutation checks for prh_tac: flip the Q1 separation rule (and, separately, the outcome rule) in scratch copies and require test_prh_tac.py to go RED.
Writes MUTATION_PRH_TAC.json beside this file."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PRH = os.path.dirname(HERE)


def run(src_dir):
    env = dict(os.environ, PRH_SRC_DIR=src_dir, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.join(PRH, "test_prh_tac.py")], capture_output=True, text=True, encoding="utf-8", env=env)
    return r.returncode, (r.stdout or "")[-400:]


MUTS = {"q1_separation_flipped": ("return bool(P.is_sep_pos(lo, hi))", "return not bool(P.is_sep_pos(lo, hi))"),
        "outcome_any_instead_of_all": ('"TAC-LEAN" if all(q1_flags)', '"TAC-LEAN" if any(q1_flags)'),
        "label_shuffle_is_identity": ("out[rows[c]] = Y[src[np.arange(len(rows[c])) % len(src)]]", "out[rows[c]] = Y[rows[c]]")}
res = {"clean": run(PRH), "mutations": {}}
for name, (old, new) in MUTS.items():
    tmp = tempfile.mkdtemp(prefix="prh_mut_tac_", dir="F:/TanitAD-scratch/prh_seed/work")
    src = open(os.path.join(PRH, "prh_tac.py"), encoding="utf-8").read()
    assert old in src, name
    open(os.path.join(tmp, "prh_tac.py"), "w", encoding="utf-8").write(src.replace(old, new))
    for f in ("prh_seed.py", "prh_seed_v2.py"):
        shutil.copy(os.path.join(PRH, f), tmp)
    res["mutations"][name] = run(tmp)
    shutil.rmtree(tmp)
res["all_detected"] = bool(res["clean"][0] == 0 and all(v[0] != 0 for v in res["mutations"].values()))
json.dump(res, open(os.path.join(HERE, "MUTATION_PRH_TAC.json"), "w", encoding="utf-8"), indent=1)
print(res["all_detected"], res["clean"][0], {k: v[0] for k, v in res["mutations"].items()})
