"""mutate_check.py -- one source mutation per committed rule; the unit tests must go RED on every one (and GREEN on the unmutated copy).  Writes out/MUTATIONS.json.  Usage: python mutate_check.py"""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
W = HERE + "/mut"
MUTS = [
    ("ds_read.py", '    if not reach_ok:\n        return ["NO-REACH"]', '    if False:\n        return ["NO-REACH"]', "NO-REACH no longer precedes the labels"),
    ("ds_read.py", "if gb and cb:", "if gb or cb:", "MIXED needs a gain AND a cost at both seeds"),
    ("ds_read.py", "if len(gb) >= 2:", "if len(gb) >= 1:", "DATA-HELPS needs >= 2 families"),
    ("ds_read.py", "if c0 | c1:", "if c0 & c1:", "a one-seed cost is SEEDS-DISAGREE"),
    ("ds_read.py", "if set(per_seed) != {0, 1}:", "if False:", "no label without both seeds"),
    ("ds_read.py", '"cost" if costs else ("gain" if gains else "none")', '"gain" if gains else ("cost" if costs else "none")', "family cost wins over family gain"),
    ("ds_read.py", '"label": "NOT-YET (seed 1 absent)"', '"label": "DATA-HELPS"', "the seed-0 read is never a label"),
    ("ds_read.py", "                n += 1\n", "                n += 0\n", "K-RE counts the compared fields"),
    ("theory/obd/read/obd_read.py", "above = abs(pt) > floor", "above = True", "the cell rule's floor comparison (FLOORCMP)"),
    ("ds_t0.py", "r >= bar and d[\"separated\"] and d[\"point\"] > 0", "r >= -bar and d[\"separated\"] and d[\"point\"] > 0", "the reach bar is +5 %"),
    ("ds_t0.py", 'sign_ok = d["point"] < 0', "sign_ok = True", "the sign rule of the T label"),
    ("ds_t0.py", 'above = floor is not None and abs(d["point"]) > floor', "above = True", "the stage-A T floor"),
    ("ds_lib.py", 'if sha256_file(p) != r["sha256"]:', "if False:", "the checkpoint sha256 against the bank record"),
    ("ds_lib.py", 'if r.get("steps") != STEPS[stage]:', "if False:", "the stage's step count"),
    ("ds_lib.py", "return None if f >= need else", "return None if f >= 0 else", "the F: 25 GB bar"),
    ("ds_pull.py", "    if got != want:\n        os.remove(p)", "    if False:\n        os.remove(p)", "the pulled copy's sha256 against the banker's"),
    ("ds_run.py", '"--tac-groups", "all-latcons", "--tac-alpha", "1"', '"--tac-groups", "all-latcons"', "the all-latcons oracle eval's flags"),
    ("ds_t0.py", "        t2 = (v[\"t_cons\"] * sd + m - mr) / sr\n", "        t2 = v[\"t_cons\"]\n", "GTFIX: T's code is re-expressed in the reference frame"),
    ("ds_t0.py", "        if bad.any():\n            _refuse(f\"{arm}_s{s}: re-expressed", "        if False:\n            _refuse(f\"{arm}_s{s}: re-expressed", "GTFIX: a genuine relabel is refused after re-expression"),
    ("ds_t0.py", "        if not np.array_equal(pa[has], pr[has]):\n", "        if False:\n", "GTFIX: a GT presence change is refused"),
]
FILES = [f for f in os.listdir(HERE) if f.endswith(".py") and f not in ("mutate_check.py", "make_pins.py")] + ["READ_PINS.json"]
def prep(n):
    d = f"{W}/{n}"
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in FILES:
        shutil.copy(f"{HERE}/{f}", d)
    shutil.copytree(HERE + "/theory", d + "/theory")
    return d
def run(d):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", TMP=W, TEMP=W)
    return subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={d}/bt", "test_ds_read.py"], cwd=d, env=env, capture_output=True, encoding="utf-8").returncode
res = {}
os.makedirs(W, exist_ok=True)
d = prep("base"); res["unmutated"] = {"rc": run(d), "expect": 0}
for i, (f, old, new, what) in enumerate(MUTS):
    d = prep(f"m{i}")
    s = open(f"{d}/{f}", encoding="utf-8").read()
    assert old in s, (f, old)
    open(f"{d}/{f}", "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
    res[f"m{i}: {what}"] = {"rc": run(d), "expect": "nonzero"}
ok = res["unmutated"]["rc"] == 0 and all(v["rc"] != 0 for k, v in res.items() if k != "unmutated")
os.makedirs(HERE + "/out", exist_ok=True)
json.dump({"all_red": ok, "results": res}, open(HERE + "/out/MUTATIONS.json", "w"), indent=1)
print("MUTATIONS all red:", ok, sum(1 for k, v in res.items() if k != "unmutated" and v["rc"] != 0), "of", len(MUTS))
shutil.rmtree(W)
