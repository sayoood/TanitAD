"""mutate_reps2.py -- MUTANTS of the patched checker (dev box, CPU): each mutation of pod_copy/dscale10k_check_reps2.py must turn test_reps2.py RED (rc != 0); the unmutated file must stay GREEN (control).
usage: python -I mutate_reps2.py   (writes tmp/mutants/*.py, tmp/MUTANTS_REPS2.json; one pytest run per mutant, whole file)"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "pod_copy", "dscale10k_check_reps2.py")
OUT = os.path.join(HERE, "tmp", "mutants")
os.makedirs(OUT, exist_ok=True)
src = open(SRC, encoding="utf-8").read()
MUT = [
    ("M16_mint_ignores_the_reference", "put(\"C-ARGV\", argv_diff_problems(base, toks, arm, stage, ue, rep_ref))", "put(\"C-ARGV\", argv_diff_problems(base, toks, arm, stage, ue, None))"),
    ("M1_rep_check_not_called", "        bad += rep_ref_problems(toks, rep_ref, arm)\n", "        pass\n"),
    ("M2_seed_not_pinned", '    pinned = {"--seed": [str(ARMS[arm]["seed"])], "--out"', '    pinned = {"--seed": a.get("--seed"), "--out"'),
    ("M3_out_not_pinned", 'pinned = {"--seed": [str(ARMS[arm]["seed"])], "--out": [ro.replace(f"/runs/{REP_REF_ARM}/", f"/runs/{arm}/")]}', 'pinned = {"--seed": [str(ARMS[arm]["seed"])], "--out": a.get("--out")}'),
    ("M4_ue_s3_missing", '    "UE_s3": dict(set="U", stage_b=False, seed=3, dagger_seed=None, steps="ue"),', ""),
    ("M5_ue_s3_wrong_seed", 'dict(set="U", stage_b=False, seed=3, dagger_seed=None, steps="ue"),      # REPS2: the seed-3', 'dict(set="U", stage_b=False, seed=2, dagger_seed=None, steps="ue"),      # REPS2: the seed-3'),
    ("M6_ref_not_bound_in_run", ', "rep_ref_argv": g("rep_ref_argv")}', "}"),
    ("M7_ref_not_bound_in_mint", ',\n             "rep_ref_argv": getattr(a, "rep_ref_argv", "")}', "}"),
    ("M8_stage_b_allowed", '        if stage != "A":\n            bad.append(f"{arm} is a stage-A-only arm (stage {stage} refused)")\n', ""),
    ("M9_missing_flag_allowed", '''    for f in r:
        if f not in a:
            bad.append(f"{arm}: flag {f} of {REP_REF_ARM}'s argv is missing")''', "    pass"),
    ("M10_extra_flag_allowed", '''        if f not in r:
            bad.append(f"{arm}: flag {f} {a[f]} is not in {REP_REF_ARM}'s argv (only --seed and --out may differ)")''', "        if f not in r:\n            pass"),
    ("M11_value_diff_allowed", '''        elif a[f] != r[f]:
            bad.append(f"{arm}: flag {f}: {a[f]} differs from {REP_REF_ARM}'s {r[f]} (only --seed and --out may differ)")''', "        elif a[f] != r[f]:\n            pass"),
    ("M12_no_reference_allowed", '''    if ref_toks is None:
        return [f"{arm}: no reference argv (--rep-ref-argv = {REP_REF_ARM}.A.json): the seed-only rule cannot be checked"]''', "    if ref_toks is None:\n        return []"),
    ("M13_um_added_to_arms", '    "B_s3": dict(set="B"', '    "UM_s2": dict(set="U", stage_b=False, seed=2, dagger_seed=None, steps="ue"),\n    "B_s3": dict(set="B"'),
    ("M14_old_arm_changed", '    "UE_s1": dict(set="U", stage_b=False, seed=1,', '    "UE_s1": dict(set="U", stage_b=False, seed=9,'),
    ("M15_reference_seed_unchecked", '''    if r.get("--seed") != ["0"]:
        bad.append(f"the reference argv has --seed {r.get('--seed')}, not ['0'] ({REP_REF_ARM})")''', "    pass"),
]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")


def run(path):
    e = dict(env, REPS2_CHECKER=path)
    r = subprocess.run([sys.executable, "-I", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", os.path.join(HERE, "test_reps2.py"), "--basetemp", os.path.join(HERE, "tmp", "pytest_mut")], env=e, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, (r.stdout.strip().split("\n") or [""])[-1]


res = {}
rc, last = run(SRC)
res["CONTROL_unmutated"] = {"rc": rc, "last": last, "green_as_required": rc == 0}
for name, old, new in MUT:
    assert src.count(old) == 1, (name, src.count(old))
    p = os.path.join(OUT, name + ".py")
    open(p, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
    rc, last = run(p)
    res[name] = {"rc": rc, "last": last, "red_as_required": rc != 0}
ok = res["CONTROL_unmutated"]["green_as_required"] and all(v["red_as_required"] for k, v in res.items() if k != "CONTROL_unmutated")
json.dump({"all_ok": ok, "results": res}, open(os.path.join(HERE, "tmp", "MUTANTS_REPS2.json"), "w"), indent=1)
print(json.dumps({k: (v.get("red_as_required", v.get("green_as_required"))) for k, v in res.items()}), "ALL-OK" if ok else "SOME-NOT-RED")
sys.exit(0 if ok else 1)
