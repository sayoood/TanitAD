"""Mutation runner (ROLLADD build): applies ONE mutation at a time IN PLACE to the build tree (originals are saved under mutants/orig and restored in a finally block and by `restore`), runs the whole
test_v8f_rolladd.py with -x (CPU, one python process, commit-gated), requires a FAILURE (rc != 0), restores the file. One line per mutant; MUTANTS_RESULT.json at the end.
usage: python run_mutants.py [M1 M2 ...]        python run_mutants.py restore"""
import json
import os
import shutil
import subprocess
import sys
import time

BASE = "F:/TanitAD-scratch/rolladd"
TREE = BASE + "/tree"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
D = TREE + "/stack/tanitad/data/v8f_dagger.py"
T = TREE + "/stack/scripts/v8f_mini_train.py"
ORIG = BASE + "/mutants/orig"
M = {
    "M1_no_append": (D, "out = append_rows(batch, rows, upd)", "out = dict(batch)"),
    "M2_logged_rows_appended_instead_of_swapped": (D, "        src = upd.get(k, v)\n", "        src = v\n"),
    "M3_replace_instead_of_append_under_add": (D, "out = append_rows(batch, rows, upd)", "out.update(upd)"),
    "M4_n_appended_not_counted": (D, "                self.last[\"n_appended\"] = int(out[\"p10f\"].shape[0]) - B\n", "                self.last[\"n_appended\"] = 0\n"),
    "M5_trainer_never_passes_the_mode": (T, "kin_mode=a.kin_mode, mode=a.dagger_mode)", "kin_mode=a.kin_mode)"),
    "M6_default_mode_is_add": (D, "DEFAULT_MODE = \"swap\"", "DEFAULT_MODE = \"add\""),
    "M7_add_at_p0_not_refused": (D, "        if self.mode == \"add\" and not self.active:", "        if False:"),
    "M8_launch_stage_not_refused_beside_add": (T, "(\"--launch-intent-w\", a.launch_intent_w > 0.0),", "(\"--launch-intent-w\", False),"),
    "M9_flag_check_entry_deleted": (T, "    \"dagger_mode\": \"dagger:dagger_mode\",\n", ""),
    "M10_dict_entries_not_appended": (D, "torch.cat([vv, src[kk][rows]], 0) if torch.is_tensor(vv)", "vv if True or torch.is_tensor(vv)"),
    "M11_swap_mode_no_longer_swaps": (D, "                out.update(upd)", "                pass"),
    "M12_trainer_counts_no_appended_rows": (T, "                        c_[\"appended\"] += dag.last[\"n_appended\"]\n", "                        c_[\"appended\"] += 0\n"),
}
if sys.argv[1:] == ["restore"]:
    for f, p in ((D, "v8f_dagger.py"), (T, "v8f_mini_train.py")):
        if os.path.exists(f"{ORIG}/{p}"):
            shutil.copyfile(f"{ORIG}/{p}", f)
    print("restored")
    sys.exit(0)
os.makedirs(ORIG, exist_ok=True)
for f, p in ((D, "v8f_dagger.py"), (T, "v8f_mini_train.py")):
    if not os.path.exists(f"{ORIG}/{p}"):
        shutil.copyfile(f, f"{ORIG}/{p}")
only = set(sys.argv[1:])
out = {}
for name, (path, old, new) in M.items():
    if only and name.split("_")[0] not in only:
        continue
    orig = open(path, encoding="utf-8", newline="").read()
    assert orig.count(old) == 1, (name, orig.count(old))
    open(path, "w", encoding="utf-8", newline="").write(orig.replace(old, new))
    subprocess.run([PY, "-I", "F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py", "5.5", "240"], check=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TMP=BASE + "/tmp/tt", TEMP=BASE + "/tmp/tt", PYTHONPATH=TREE.replace("/", "\\") + "\\stack")
    t0 = time.time()
    try:
        p = subprocess.run([PY, "-m", "pytest", "-q", "-x", "stack/tests/test_v8f_rolladd.py", "--basetemp=" + BASE + "/tmp/ptm", "-p", "no:cacheprovider"], cwd=TREE, env=env, capture_output=True, text=True, timeout=1500)
        tail = [l for l in p.stdout.splitlines() if l.startswith(("FAILED", "ERROR")) or " passed" in l or " failed" in l][-3:]
        rc = p.returncode
    finally:
        open(path, "w", encoding="utf-8", newline="").write(orig)
        shutil.rmtree(BASE + "/tmp/ptm", ignore_errors=True)
    out[name] = {"rc": rc, "RED": rc != 0, "seconds": round(time.time() - t0), "tail": tail}
    print(name, "RED" if rc != 0 else "!! STILL GREEN !!", tail[:2], flush=True)
    prev = json.load(open(BASE + "/mutants/MUTANTS_RESULT.json")) if os.path.exists(BASE + "/mutants/MUTANTS_RESULT.json") else {}
    prev.update(out)
    json.dump(prev, open(BASE + "/mutants/MUTANTS_RESULT.json", "w"), indent=1)
