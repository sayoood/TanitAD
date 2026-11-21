"""o3 / w3 mutation tests: each line that applies a lever is deleted on a copy of the tree and the checks must go RED; the control (the real tree) stays green.
  MA1  rule_own.py: the line `T["total"] = T["total"] + W * term` of add_own_term is replaced by `pass`   (flag A: the term never reaches the total)
  MA2  v8f_mini_train.py: the loop line `ro_i = ROW.add_own_term(...)` is replaced by `pass`                (flag A: the loop never calls it)
  MB1  rule_own.py: the line `T["total"] = T["total"] + weight_extra(...)` is replaced by `pass`           (flag B: the weight never reaches the total)
For the control and each mutation: (a) the in-process reach check (o2_o3.py for A, w2_w4.py for B), (b) the trainer for 3 steps with the flag on (rc, message, and the `total` trace against the flag-off run of the same
argv otherwise: equal = the lever did nothing = RED), (c) the trainer selfcheck-only (rc).
Expected: control green everywhere; MA1 red in a, b and c; MA2 green in a and c (they call the helper directly) and red in b (the loop guard / the call is gone); MB1 red in a and b, green in c (flag B has no probe of its own).
Usage: o3_w3_mutation.py"""
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


def mutate(name, rel, old, new):
    dst = f"{S}/tmp/r3mut3_{name}"
    if not os.path.exists(dst):
        shutil.copytree(f"{S}/tree_patch/stack", f"{dst}/stack", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(f"{S}/tree_patch/taniteval", f"{dst}/taniteval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    p = f"{dst}/{rel}"
    s = open(p, encoding="utf-8", newline="").read()
    assert s.count(old) == 1, (rel, s.count(old))
    open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    return dst


MA1 = mutate("MA1", "stack/tanitad/data/rule_own.py", '    T["total"] = T["total"] + W * term\n', "    pass\n")
MA2 = mutate("MA2", "stack/scripts/v8f_mini_train.py", '                    ro_i = ROW.add_own_term(T, out["traj30"], ro_in, RO.w_own)\n', "                    pass\n")
MB1 = mutate("MB1", "stack/tanitad/data/rule_own.py", '    T["total"] = T["total"] + weight_extra(traj30, gt, gt_ok, changed, W, w_traj, w_acc, launch, launch_w)\n', "    pass\n")
res = {}


def trainer(root, extra, out):
    return subprocess.run([PY, f"{root}/stack/scripts/v8f_mini_train.py", *TINY, *G, *extra, "--out", out], env=envf(root), capture_output=True, text=True, encoding="utf-8")


def trace(out, key):
    return [json.loads(line)[key] for line in open(f"{out}/train_log.jsonl", encoding="utf-8")]


refs = {}
for kind, flags_off in (("A", []), ("B", ["--rule-relabel", "on"])):
    trainer(f"{S}/tree_patch", ["--steps", "3", "--log-every", "1", *flags_off], f"{S}/tmp/r3mut_ref_{kind}")
    refs[kind] = trace(f"{S}/tmp/r3mut_ref_{kind}", "total")
FLAGS = {"A": ["--w-rule-own", "1.0"], "B": ["--rule-relabel", "on", "--rule-relabel-weight", "4"]}
SCRIPT = {"A": "o2_o3.py", "B": "w2_w4.py"}
for name, root, kind in (("control_A", f"{S}/tree_patch", "A"), ("control_B", f"{S}/tree_patch", "B"), ("MA1", MA1, "A"), ("MA2", MA2, "A"), ("MB1", MB1, "B")):
    rec = {"kind": kind}
    r = subprocess.run([PY, f"{CK}/{SCRIPT[kind]}", root, f"{S}/tmp/r3mut_json_{name}.json"], env=envf(root), capture_output=True, text=True, encoding="utf-8")
    rec["a_rc"] = r.returncode
    rec["a_tail"] = ((r.stdout or "") + (r.stderr or ""))[-200:].replace("\n", " ")
    out = f"{S}/tmp/r3mut_run_{name}"
    r = trainer(root, ["--steps", "3", "--log-every", "1", *FLAGS[kind]], out)
    rec["b_trainer_rc"] = r.returncode
    rec["b_trainer_msg"] = ((r.stdout or "") + (r.stderr or ""))[-260:].replace("\n", " ")
    try:
        tt = trace(out, "total")
        rec["b_total_equals_flag_off_all_steps"] = tt == refs[kind]
    except Exception as e:      # noqa: BLE001  the run was refused before a log line
        rec["b_total_equals_flag_off_all_steps"] = None
        rec["b_no_log"] = type(e).__name__
    r = trainer(root, [*FLAGS[kind], "--selfcheck-only"], f"{S}/tmp/r3mut_sc_{name}")
    rec["c_selfcheck_rc"] = r.returncode
    rec["c_msg"] = ((r.stdout or "") + (r.stderr or ""))[-260:].replace("\n", " ")
    rec["green_a"] = rec["a_rc"] == 0
    rec["green_b"] = rec["b_trainer_rc"] == 0 and rec["b_total_equals_flag_off_all_steps"] is False
    rec["green_c"] = rec["c_selfcheck_rc"] == 0
    res[name] = rec
    print(name, json.dumps(rec))
g = lambda n, *x: [res[n][k] for k in x]      # noqa: E731
res["PASS"] = bool(all(g("control_A", "green_a", "green_b", "green_c")) and all(g("control_B", "green_a", "green_b", "green_c"))
                   and g("MA1", "green_a", "green_b", "green_c") == [False, False, False]
                   and g("MA2", "green_a", "green_b", "green_c") == [True, False, True]
                   and g("MB1", "green_a", "green_b", "green_c") == [False, False, True])
json.dump(res, open(f"{CK}/o3_w3_mutation.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("MUT", "PASS (every mutation caught by at least one check, control green)" if res["PASS"] else "FAIL")
