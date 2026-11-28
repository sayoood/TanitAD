"""Mutation checks for v2: flip the admissibility rule (clause 3 separation and clause 4 shuffle) in scratch copies and require test_prh_seed_v2.py to go RED.
Writes MUTATION_PRH_SEED_V2.json beside this file."""
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
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", os.path.join(PRH, "test_prh_seed_v2.py")], capture_output=True, text=True, encoding="utf-8", env=env)
    return r.returncode, (r.stdout or "")[-400:]


MUTS = {"clause3_separation_dropped": ('"3_mix50_minus_floor_separated": bool(P.is_sep_pos(*v["diff50_ci"]))', '"3_mix50_minus_floor_separated": True'),
        "clause4_shuf_flipped": ('v["SHUF"] < v["FLOOR"]', 'v["SHUF"] > v["FLOOR"]'),
        "clause2_monotone_flipped": ('v["MIX25"] > v["MIX50"] > v["MIX75"]', 'v["MIX25"] < v["MIX50"] < v["MIX75"]')}
res = {"clean": run(PRH), "mutations": {}}
for name, (old, new) in MUTS.items():
    tmp = tempfile.mkdtemp(prefix="prh_mut2_", dir="F:/TanitAD-scratch/prh_seed/work")
    src = open(os.path.join(PRH, "prh_seed_v2.py"), encoding="utf-8").read()
    assert old in src, name
    open(os.path.join(tmp, "prh_seed_v2.py"), "w", encoding="utf-8").write(src.replace(old, new))
    shutil.copy(os.path.join(PRH, "prh_seed.py"), tmp)
    res["mutations"][name] = run(tmp)
    shutil.rmtree(tmp)
res["all_detected"] = bool(res["clean"][0] == 0 and all(v[0] != 0 for v in res["mutations"].values()))
json.dump(res, open(os.path.join(HERE, "MUTATION_PRH_SEED_V2.json"), "w", encoding="utf-8"), indent=1)
print(res["all_detected"], res["clean"][0], {k: v[0] for k, v in res["mutations"].items()})
