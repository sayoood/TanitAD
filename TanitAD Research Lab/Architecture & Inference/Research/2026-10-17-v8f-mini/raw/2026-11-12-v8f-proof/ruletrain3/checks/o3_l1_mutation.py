"""o3 mutation test for the linear form: the line of rule_own.own_term that selects the linear form (`per = knot_distance(pl - cap).mean(1)`) is replaced on a copy of the tree by the squared computation, so `l1` silently computes `sq`.
Checks, for the control (the real tree) and the mutation:
 (a) o2_o3_l1.py in-process (term by hand, exact zeros, partial, knots on the cap): control PASS, mutation FAIL
 (b) the trainer for 3 steps with --w-rule-own 1.0 --rule-own-form l1: its `total` trace must DIFFER from the --rule-own-form sq run of the same argv (equal = the form did nothing = RED)
 (c) the pytest of the unit test file (control green, mutation red on the mutation test's own subject is inside the file; here the mutated tree must fail test_l1_cap_by_hand_and_term_is_the_mean_distance_in_metres)
Usage: o3_l1_mutation.py"""
import json
import os
import shutil
import subprocess

S = "F:/TanitAD-scratch/v8proof/ruletrain3/work"
CK = "F:/TanitAD-scratch/v8proof/ruletrain3/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
w = lambda *p: "\\".join(p).replace("/", "\\")
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--exclude", "1:3"]
TINY = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1", "--seed", "0"]


def envf(root):
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                TMP=w(S, "tmp"), TEMP=w(S, "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))


dst = f"{S}/tmp/r3mut_ML1"
if not os.path.exists(dst):
    shutil.copytree(f"{S}/tree_patch/stack", f"{dst}/stack", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(f"{S}/tree_patch/taniteval", f"{dst}/taniteval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
p = f"{dst}/stack/tanitad/data/rule_own.py"
s = open(p, encoding="utf-8", newline="").read()
old = "        per = knot_distance(pl - cap).mean(1)"
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, "        per = ((pl - cap) ** 2).sum(-1).mean(1)"))
res = {}


def trainer(root, extra, out):
    return subprocess.run([PY, f"{root}/stack/scripts/v8f_mini_train.py", *TINY, *G, *extra, "--out", out], env=envf(root), capture_output=True, text=True, encoding="utf-8")


def trace(out):
    return [json.loads(line)["total"] for line in open(f"{out}/train_log.jsonl", encoding="utf-8")]


for name, root in (("control", f"{S}/tree_patch"), ("MUT_l1_computes_sq", dst)):
    rec = {}
    r = subprocess.run([PY, f"{CK}/o2_o3_l1.py", root, f"{S}/tmp/mut_o2o3_{name}.json", "l1"], env=envf(root), capture_output=True, text=True, encoding="utf-8")
    rec["a_inprocess_rc"] = r.returncode
    rec["a_inprocess_pass"] = ("O2_O3 PASS" in r.stdout)
    rec["a_tail"] = (r.stdout + r.stderr)[-200:].replace("\n", " ")
    ts = {}
    for form in ("sq", "l1"):
        o = f"{S}/tmp/r3mut_run_{name}_{form}"
        rr = trainer(root, ["--w-rule-own", "1.0", "--rule-own-form", form, "--steps", "3", "--log-every", "1"], o)
        ts[form] = trace(o) if rr.returncode == 0 else None
        rec[f"b_rc_{form}"] = rr.returncode
    rec["b_totals_sq"], rec["b_totals_l1"] = ts["sq"], ts["l1"]
    rec["b_l1_differs_from_sq"] = bool(ts["sq"] and ts["l1"] and ts["sq"] != ts["l1"])
    t = subprocess.run([PY, "-m", "pytest", "-q", "-x", "stack/tests/test_v8f_rule_own_l1.py", "-k", "cap_by_hand or mutation_deleting", f"--basetemp={S}/pt/mut_{name}", "-p", "no:cacheprovider"], cwd=root, env=envf(root),
                       capture_output=True, text=True, encoding="utf-8")
    rec["c_pytest_rc"] = t.returncode
    rec["c_tail"] = t.stdout[-300:].replace("\n", " ")
    rec["GREEN"] = bool(rec["a_inprocess_pass"] and rec["b_l1_differs_from_sq"] and rec["c_pytest_rc"] == 0)
    rec["RED_in_all_three"] = bool((not rec["a_inprocess_pass"]) and (not rec["b_l1_differs_from_sq"]) and rec["c_pytest_rc"] != 0)
    res[name] = rec
    print(name, json.dumps({k: v for k, v in rec.items() if k not in ("a_tail", "c_tail")}))
res["PASS"] = bool(res["control"]["GREEN"] and res["MUT_l1_computes_sq"]["RED_in_all_three"])
json.dump(res, open(f"{CK}/o3_l1_mutation.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("O3_MUT", "PASS" if res["PASS"] else "FAIL")
