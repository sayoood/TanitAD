"""t8: --wm-l1-diag on against off, 40 trainer steps (tiny synthetic model, CPU, same seed), tree_patch, for PE, PF and PEF (cell predictor): every weight of the final checkpoint equal
(max abs difference must be exactly 0.0), and the logged fields common to both runs equal. Usage: t8_diag.py"""
import json
import os
import subprocess

import torch

S = "C:/Users/Admin/v8proof_c/target"
CK = "F:/TanitAD-scratch/v8proof/spatial/target/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
STEPS = 40
w = lambda *p: "\\".join(p).replace("/", "\\")
GRIDARM = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--wm-l1-pred", "cell"]
ARMS = {"PE": ["--wm-l1-act", "speed"], "PF": ["--wm-l1-futtok-k", "10", "--w-wm-futtok", "1.0"], "PEF": ["--wm-l1-act", "speed", "--wm-l1-futtok-k", "10", "--w-wm-futtok", "1.0"]}
root = f"{S}/tree_patch"
env = dict(os.environ, MW_NORESUME="1", CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           TMP=w(S, "tmp"), TEMP=w(S, "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
res = {"steps": STEPS}
for arm, lev in ARMS.items():
    outs = {}
    for diag in ("off", "on"):
        out = f"{S}/tmp/t8/{diag}"                      # two folders reused by every arm (disk), no resume
        r = subprocess.run([PY, f"{CK}/t_worker.py", root, out, str(STEPS), json.dumps(GRIDARM + lev + ["--wm-l1-diag", diag])], env=env, capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr[-3000:]
        outs[diag] = out
    ld = lambda o: [json.loads(l) for l in open(f"{o}/run/train_log.jsonl", encoding="utf-8")][-STEPS:]      # the log is opened in append mode: this arm = the last STEPS rows
    a, b = ld(outs["off"]), ld(outs["on"])
    skip = ("sec", "sec_per_step", "elapsed")
    common = sorted({k for r in a for k in r if k not in skip and isinstance(r[k], (int, float))} & {k for r in b for k in r if isinstance(r[k], (int, float))})
    ca, cb = (torch.load(f"{o}/run/ckpt_last.pt", weights_only=False) for o in (outs["off"], outs["on"]))
    wd = max(float((ca["model"][k].float() - cb["model"][k].float()).abs().max()) for k in ca["model"])
    ld_ = max(abs(x[k] - y[k]) for x, y in zip(a, b) for k in common)
    extra = sorted({k for r in b for k in r} - {k for r in a for k in r})
    res[arm] = {"argv_extra": lev, "keys_equal": list(ca["model"]) == list(cb["model"]), "n_keys": len(ca["model"]), "max_abs_diff_final_weights": wd, "max_abs_diff_common_logged_fields": ld_,
                "steps_logged": [len(a), len(b)], "keys_only_with_diag": extra, "PASS": wd == 0.0 and ld_ == 0.0 and list(ca["model"]) == list(cb["model"]) and len(a) == len(b) == STEPS}
    print(arm, json.dumps(res[arm]), flush=True)
res["PASS"] = all(v["PASS"] for k, v in res.items() if k != "steps")
json.dump(res, open(f"{CK}/t8_diag.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("T8", "PASS" if res["PASS"] else "FAIL")
