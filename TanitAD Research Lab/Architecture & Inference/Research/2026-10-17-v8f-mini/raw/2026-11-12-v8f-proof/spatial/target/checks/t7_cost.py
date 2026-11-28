"""m7 cost: seconds per training step of the tiny synthetic model on the CPU (batch 8, d 32, hmax 16, 2 threads), 30 steps, interleaved, 3 repetitions, best and mean of the
repetitions: tree_base (no lever), then tree_patch with K = 10 W = 0.3 for the flat predictor (grid arm G) and for the cell predictor (arm P), and residual alone.
A CPU ratio of a whole step on a tiny model, not a pod number. Usage: m7_cost.py"""
import json
import os
import subprocess

S = "C:/Users/Admin/v8proof_c/target"
CK = "F:/TanitAD-scratch/v8proof/spatial/target/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
w = lambda *p: "\\".join(p).replace("/", "\\")
GRIDARM = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16"]
CELL = ["--wm-l1-pred", "cell"]
EGOV = ["--wm-l1-act", "speed"]
FUT = ["--wm-l1-futtok-k", "10", "--w-wm-futtok", "1.0"]
VARIANTS = {"P": ("tree_base", GRIDARM + CELL), "PE": ("tree_patch", GRIDARM + CELL + EGOV), "PF": ("tree_patch", GRIDARM + CELL + FUT), "PEF": ("tree_patch", GRIDARM + CELL + EGOV + FUT)}
STEPS, REPS = 30, 3


def run(name, tree, extra, rep):
    root = f"{S}/{tree}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", MW_NORESUME="1",
               TMP=w(S, "tmp"), TEMP=w(S, "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
    out = f"{S}/tmp/t7/run"                                           # ONE folder per variant, overwritten by every repetition
    r = subprocess.run([PY, f"{CK}/t_worker.py", root, out, str(STEPS), json.dumps(extra)], env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(l) for l in open(f"{out}/run/train_log.jsonl", encoding="utf-8")][-STEPS:]      # the log is opened in append mode: this repetition = the last STEPS rows
    s = [x["sec"] for x in rows]
    d = [b - a for a, b in zip(s, s[1:])][4:]                            # per-step differences, the first 5 steps dropped
    return sum(d) / len(d), all(x["total"] == x["total"] and abs(x["total"]) < 1e30 for x in rows), [rows[0]["total"], rows[-1]["total"]]


t = {k: [] for k in VARIANTS}
fin = {}
for rep in range(REPS):
    for name, (tree, extra) in VARIANTS.items():
        sec, ok, tot = run(name, tree, extra, rep)
        t[name].append(sec)
        fin[name] = {"all_finite": ok, "total_first_last": tot}
        print(rep, name, round(sec, 4), ok, flush=True)
res = {"steps": STEPS, "reps": REPS, "s_per_step": t, "best": {k: min(v) for k, v in t.items()}, "mean": {k: sum(v) / len(v) for k, v in t.items()}, "finite": fin}
for v in ("PE", "PF", "PEF"):
    res[f"ratio_{v}_over_P_best"] = min(t[v]) / min(t["P"])
    res[f"ratio_{v}_over_P_mean"] = sum(t[v]) / sum(t["P"])
json.dump(res, open(f"{CK}/t7_cost.json", "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k.startswith("ratio") or k == "best"}, indent=1))
