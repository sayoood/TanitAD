"""Mutation runner (KINCACHE build): copies tree2's stack/ and taniteval/taniteval/ once, applies ONE mutation at a time, runs stack/tests/test_v8f_kincache.py WHOLE-FILE (-x), requires a FAILURE; M0 (no mutation) must be GREEN.
usage: python run_mutants_kincache.py [M1 M2 ...]   (no args = all). Result: mutants_kincache/MUTANTS_RESULT.json"""
import json
import os
import shutil
import subprocess
import sys
import time

K = "F:/TanitAD-scratch/kinlab"
SRC = K + "/tree2"
DST = K + "/mutants_kincache/tree"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
if not os.path.isdir(DST + "/stack"):
    shutil.copytree(SRC + "/stack", DST + "/stack", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "experiments"))
    shutil.copytree(SRC + "/taniteval/taniteval", DST + "/taniteval/taniteval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
D = DST + "/stack/tanitad/data/v8f_dagger.py"
T = DST + "/stack/scripts/v8f_mini_train.py"
F8 = '("label_kind", "bank_json_sha256", "labeller_sha256", "dagger_sha256", "n_records", "eligible_index_sha256", "records_sha256", "clip_store_sha256")'


def drop(field):
    fs = ["label_kind", "bank_json_sha256", "labeller_sha256", "dagger_sha256", "n_records", "eligible_index_sha256", "records_sha256", "clip_store_sha256"]
    fs.remove(field)
    return "(" + ", ".join(f'"{x}"' for x in fs) + ")"


FIELDS = {"M2": "label_kind", "M3": "bank_json_sha256", "M4": "labeller_sha256", "M5": "dagger_sha256", "M6": "n_records", "M7": "eligible_index_sha256", "M8": "records_sha256", "M9": "clip_store_sha256"}
M = {
    "M0_control_no_mutation": (D, "LABEL_CACHE_VERSION = ", "LABEL_CACHE_VERSION = "),
    "M1_header_check_skipped": (D, "    if diff:\n        raise BankRefused", "    if False:\n        raise BankRefused"),
}
for k, f in FIELDS.items():
    M[f"{k}_header_field_{f}_ignored"] = (D, "LABEL_CACHE_FIELDS = " + F8, "LABEL_CACHE_FIELDS = " + drop(f))
M.update({
    "M10_arrays_sha256_ignored": (D, '    if header.get("arrays_sha256") != _cache_arrays_sha256(arrays):', "    if False:"),
    "M11_cache_loaded_without_the_flag": (T, 'getattr(a, "dagger_label_cache", "") or "", None, None, None', '(getattr(a, "dagger_label_cache", "") or os.path.join(a.dagger_bank, "label_cache.npz")), None, None, None'),
    "M12_write_not_atomic": (D, '    part = path + ".part"', "    part = path"),
    "M13_cache_sha_not_recorded_in_the_built_levers": (D, '"dagger_label_cache_sha256": (getattr(self.bank, "label_cache", None) or {}).get("sha256")', '"dagger_label_cache_sha256": None'),
    "M14_cache_path_not_recorded_in_the_built_levers": (D, '"dagger_label_cache": (getattr(self.bank, "label_cache", None) or {}).get("path")', '"dagger_label_cache": None'),
    "M15_existing_cache_never_read": (T, "            cache_hit = os.path.exists(cache_path)", "            cache_hit = False"),
    "M16_flag_parsed_not_attached_to_the_bank": (T, '            bank.label_cache = {"path": os.path.abspath(cache_path), "sha256": cache_sha}', "            pass"),
    "M17_p_zero_not_refused": (T, "        if a.dagger_p <= 0:\n            bad.append(f\"--dagger-label-cache", "        if False:\n            bad.append(f\"--dagger-label-cache"),
    "M18_human_label_with_cache_not_refused": (T, '    if (getattr(a, "dagger_label_cache", "") or "") and a.dagger_label not in ("speed_v1", "kin_v1"):', "    if False:"),
    "M19_gate_does_not_rehash_the_cache_file": (T, '        if lv.get("dagger_label_cache_sha256") is None or lv.get("dagger_label_cache_sha256") != disk_c:', "        if False:"),
    "M20_trainer_passes_the_dagger_file_as_the_labeller": (T, "labeller_path=DR.__file__, dagger_path=DG.__file__)", "labeller_path=DG.__file__, dagger_path=DG.__file__)"),
    "M21_trainer_passes_the_labeller_file_as_the_dagger": (T, "labeller_path=DR.__file__, dagger_path=DG.__file__)", "labeller_path=DR.__file__, dagger_path=DR.__file__)"),
    "M22_clip_store_pose_not_hashed": (D, '        _hash_array(st, np.asarray(read_clip(nm)["pose"], np.float64))', "        pass"),
    "M23_record_arrays_not_hashed": (D, '("own", np.asarray(a["p10f"])[:, -1, :3])):', '("own", np.zeros(1))):'),
    "M24_write_after_read_rewrites_the_cache": (T, "        if cache_path and not cache_hit:", "        if cache_path:"),
    "M25_part_file_read_when_path_absent": (T, "            cache_hit = os.path.exists(cache_path)", '            cache_hit = os.path.exists(cache_path) or os.path.exists(cache_path + ".part")'),
})
only = set(sys.argv[1:])
os.makedirs(K + "/mutants_kincache", exist_ok=True)
resf = K + "/mutants_kincache/MUTANTS_RESULT.json"
out = json.load(open(resf)) if (only and os.path.exists(resf)) else {}
for name, spec in M.items():
    path, old, new = spec[:3]
    if only and name.split("_")[0] not in only:
        continue
    orig = open(path, encoding="utf-8", newline="").read()
    assert orig.count(old) == 1, (name, orig.count(old))
    open(path, "w", encoding="utf-8", newline="").write(orig.replace(old, new))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TMP=K + "/tmp", TEMP=K + "/tmp", PYTHONPATH=DST + "/stack")
    t0 = time.time()
    try:
        subprocess.run([PY, "-I", K.replace("kinlab", "strat1") + "/read/seeds/wait_commit.py", "5.5", "60"], capture_output=True, timeout=700)
        argv = [PY, "-m", "pytest", "-q", *([] if name.startswith("M0_") else ["-x"]), "stack/tests/test_v8f_kincache.py", "--basetemp=" + K + "/tmp/ptmk", "-p", "no:cacheprovider"]
        p = subprocess.run(argv, cwd=DST, env=env, capture_output=True, text=True, timeout=1500)
        tail = [ln for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR")) or " passed" in ln or " failed" in ln][-3:]
        rc = p.returncode
    finally:
        open(path, "w", encoding="utf-8", newline="").write(orig)
        shutil.rmtree(K + "/tmp/ptmk", ignore_errors=True)
    if name.startswith("M0_"):
        print(name, "GREEN (control, as required)" if rc == 0 else "!! CONTROL RED !!", tail[:2], flush=True)
    else:
        print(name, "RED" if rc != 0 else "!! STILL GREEN !!", tail[:2], flush=True)
    out[name] = {"rc": rc, "RED": rc != 0, "seconds": round(time.time() - t0), "tail": tail}
    json.dump(out, open(resf, "w"), indent=1)
