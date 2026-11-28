"""t3 mutation test: the reach checks must go RED when the relabel is removed. Two mutations on copies of the tree (each a one-line deletion):
  M1  the loop line `b["gt"] = rr_gt` of v8f_mini_train.py is deleted (the relabelled target never reaches the loss)
  M2  the line `new[idx] = out.plan` of rule_relabel.py is replaced by `pass` (the stage computes the rule and returns the logged target)
For the control (the real tree) and each mutation: (a) t2_t3.py (module level, in-process), (b) the trainer for 3 steps with --rule-relabel on (rc and message), (c) the trainer selfcheck-only (rc).
Expected: control green everywhere; M1: (a) green (the module is intact), (b) RED (ZZRULE-NOT-APPLIEDZZ, rc != 0), (c) green (the probe calls the stage directly); M2: (a) RED, (b) the loop guard passes (stage returns logged gt) so the
check is the train-log ade equal to the flag-off run: RED by comparison, (c) RED if the probe sees the rule changed a target but no loss movement (rc != 0).
Usage: t3_mutation.py"""
import json
import os
import shutil
import subprocess

S = "C:/Users/Admin/v8proof_c/ruletrain"
CK = "F:/TanitAD-scratch/v8proof/ruletrain/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
w = lambda *p: "\\".join(p).replace("/", "\\")
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--exclude", "1:3"]
TINY = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1", "--seed", "0"]


def envf(root):
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
                TMP=w(S, "tmp"), TEMP=w(S, "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))


def mutate(name, rel, old, new):
    dst = f"{S}/tmp/mut_{name}_v2"
    if not os.path.exists(dst):
        shutil.copytree(f"{S}/tree_patch/stack", f"{dst}/stack", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(f"{S}/tree_patch/taniteval", f"{dst}/taniteval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    p = f"{dst}/{rel}"
    s = open(p, encoding="utf-8", newline="").read()
    assert s.count(old) == 1, (rel, s.count(old))
    open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    return dst


M1 = mutate("M1", "stack/scripts/v8f_mini_train.py", '                b["gt"] = rr_gt\n', "                pass\n")
M2 = mutate("M2", "stack/tanitad/data/rule_relabel.py", "    new[idx] = out.plan\n", "    pass\n")
res = {}


def trainer(root, extra, out):
    r = subprocess.run([PY, f"{root}/stack/scripts/v8f_mini_train.py", *TINY, *G, *extra, "--out", out], env=envf(root), capture_output=True, text=True, encoding="utf-8")
    return r


def ade_trace(out):
    return [json.loads(l)["ade"] for l in open(f"{out}/train_log.jsonl", encoding="utf-8")]


off = trainer(f"{S}/tree_patch", ["--steps", "3", "--log-every", "1", "--rule-relabel", "off"], f"{S}/tmp/mut_off")
ade_off = ade_trace(f"{S}/tmp/mut_off")
for name, root in (("control", f"{S}/tree_patch"), ("M1", M1), ("M2", M2)):
    rec = {}
    r = subprocess.run([PY, f"{CK}/t2_t3.py", root, f"{S}/tmp/mut_t2t3_{name}.json"], env=envf(root), capture_output=True, text=True, encoding="utf-8")
    rec["a_t2_t3_rc"] = r.returncode
    rec["a_tail"] = ((r.stdout or "") + (r.stderr or ""))[-300:].replace("\n", " ")
    out = f"{S}/tmp/mut_run_{name}"
    r = trainer(root, ["--steps", "3", "--log-every", "1", "--rule-relabel", "on"], out)
    rec["b_trainer_rc"] = r.returncode
    rec["b_trainer_msg"] = ((r.stdout or "") + (r.stderr or ""))[-260:].replace("\n", " ")
    try:
        ade_on = ade_trace(out)
        rec["b_ade_on_equals_off_all_steps"] = ade_on == ade_off
    except Exception as e:      # noqa: BLE001  the run was refused before a log line
        rec["b_ade_on_equals_off_all_steps"] = None
        rec["b_no_log"] = type(e).__name__
    r = trainer(root, ["--rule-relabel", "on", "--selfcheck-only"], f"{S}/tmp/mut_sc_{name}")
    rec["c_selfcheck_rc"] = r.returncode
    rec["c_msg"] = ((r.stdout or "") + (r.stderr or ""))[-260:].replace("\n", " ")
    rec["green_a"] = rec["a_t2_t3_rc"] == 0
    rec["green_b"] = rec["b_trainer_rc"] == 0 and rec["b_ade_on_equals_off_all_steps"] is False
    rec["green_c"] = rec["c_selfcheck_rc"] == 0
    res[name] = rec
    print(name, json.dumps(rec))
res["PASS"] = bool(res["control"]["green_a"] and res["control"]["green_b"] and res["control"]["green_c"]
                   and res["M1"]["green_a"] and not res["M1"]["green_b"] and res["M1"]["green_c"]
                   and not res["M2"]["green_a"] and not res["M2"]["green_b"] and not res["M2"]["green_c"])
json.dump(res, open(f"{CK}/t3_mutation.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("T3MUT", "PASS (every mutation caught by at least one check, control green)" if res["PASS"] else "FAIL")
