"""Mutation check for PREREG_V8F_MINI Amendment 4's --exclude: each source mutation must turn the v8F-mini tests RED; the clean
source must be GREEN. Restores the file after every mutation (binary I/O)."""
import json
import os
import subprocess
import sys

TREE = sys.argv[1]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SRC = os.path.join(TREE, "stack", "scripts", "v8f_mini_train.py")
NL = chr(10)
MUT = {
    "m1_filter_not_applied": ("        clips_all = [c for c in clips_all if c not in ex]" + NL, ""),
    "m2_no_dvb_entry": (' "exclude": "data:n_excluded",', ""),
    "m3_no_leak_check": ('    if data_info.get("excluded_in_train", 0):' + NL +
                         "        bad.append(f\"exclude {a.exclude!r}: {data_info['excluded_in_train']} excluded clips are still in the TRAIN set\")" + NL, ""),
    "m4_slice_never_taken": ("        excluded = sorted(clips_all, key=D.sha12)[xlo:xhi]" + NL, ""),
    "m5_count_check_removed": ('    if data_info.get("n_excluded", 0) != data_info.get("n_excluded_declared", 0):' + NL +
                               "        bad.append(f\"exclude {a.exclude!r}: {data_info.get('n_excluded', 0)} clips excluded, \"" + NL +
                               "                   f\"{data_info.get('n_excluded_declared', 0)} declared\")" + NL, ""),
}


def suite():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v8f_mini.py", "-k",
                        "exclude or every_flag or end_to_end or declared_vs_built"], cwd=os.path.join(TREE, "stack"),
                       capture_output=True, text=True, env=env)
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
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=1)
sys.exit(0 if ok else 1)
