"""c1 identity (ACTCONS, --w-act-cons at its default 0.0): work/tree_patch against work/tree_base (= the tree of LEADCOND, leadcond/work/tree_patch).
Shapes G, LS (G + --lead-sup on --w-lead-sup 1.0), LSFT (LS fine-tune from the tiny LS base), LSOWN (LSFT + --w-rule-own 4.0 --rule-own-form l1). Same seed, 100 steps:
state-dict key list, init tensors, loss trace (every logged field but wall time), final weights, checkpoint cfg key set: bit for bit.
Usage: a1_identity.py [steps] [shape,shape]"""
import json
import os
import subprocess
import sys

import torch

S = "F:/TanitAD-scratch/v8proof/actcons/work"
CK = "F:/TanitAD-scratch/v8proof/actcons/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16"]
BASE_CKPT = "F:/TanitAD-scratch/v8proof/leadcond/work/tmp/base_ls/ckpt_last.pt"      # tiny LS-shape checkpoint (read only)
LS = G + ["--lead-sup", "on", "--w-lead-sup", "1.0"]
FT = ["--init-ckpt", BASE_CKPT, "--lr", "1e-4", "--warmup", "2", "--seed", "100"]
SHAPES = {"G": G, "LS": LS, "LSFT": LS + FT, "LSOWN": LS + FT + ["--w-rule-own", "4.0", "--rule-own-form", "l1"]}      # the new flags --w-act-cons / --act-cons-max are absent in all four
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
if ONLY:
    SHAPES = {k: v for k, v in SHAPES.items() if k in ONLY}
res = {"steps": STEPS}
w = lambda *p: "\\".join(p).replace("/", "\\")
for name, extra in SHAPES.items():
    outs = {}
    for t in ("tree_base", "tree_patch"):
        root = f"{S}/{t}"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   PYTHONDONTWRITEBYTECODE="1", TMP=w(S, "tmp"), TEMP=w(S, "tmp"),
                   PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
        out = f"{S}/tmp/ac1/{name}_{t}"
        r = subprocess.run([PY, f"{CK}/t_worker.py", root, out, str(STEPS), json.dumps(extra)], env=env, capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr[-3000:]
        outs[t] = out
    a, b = outs["tree_base"], outs["tree_patch"]
    ka, kb = (json.load(open(f"{x}/init_keys.json", encoding="utf-8")) for x in (a, b))
    sa, sb = (torch.load(f"{x}/init_state.pt") for x in (a, b))
    init_equal = list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    la, lb = ([json.loads(l) for l in open(f"{x}/run/train_log.jsonl", encoding="utf-8")] for x in (a, b))
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("sec", "sec_per_step", "elapsed")} for r in rows]
    fields = sorted({k for r in la for k in r if k not in ("sec", "sec_per_step", "elapsed") and isinstance(r[k], (int, float))})
    maxdiff = max((abs(x[k] - y[k]) for x, y in zip(la, lb) for k in fields), default=None)
    ca, cb = (torch.load(f"{x}/run/ckpt_last.pt", weights_only=False) for x in (a, b))
    final_equal = list(ca["model"]) == list(cb["model"]) and all(torch.equal(ca["model"][k], cb["model"][k]) for k in ca["model"])
    wdiff = max(float((ca["model"][k].float() - cb["model"][k].float()).abs().max()) for k in ca["model"])
    res[name] = {"argv_extra": extra, "keys_equal": ka["keys"] == kb["keys"], "n_keys": len(ka["keys"]), "n_params": [ka["n_params"], kb["n_params"]],
                 "init_tensors_equal": init_equal, "steps_logged": [len(la), len(lb)], "loss_trace_equal": strip(la) == strip(lb),
                 "log_fields_compared": fields, "max_abs_diff_all_logged_fields": maxdiff,
                 "max_abs_diff_total": max(abs(x["total"] - y["total"]) for x, y in zip(la, lb)),
                 "final_weights_equal": final_equal, "max_abs_diff_final_weights": wdiff,
                 "ckpt_cfg_keys_equal": sorted(ca["cfg"]) == sorted(cb["cfg"]), "total_first_last": [la[0]["total"], la[-1]["total"]],
                 "tanitad_files": [ka["tanitad_file"], kb["tanitad_file"]]}
    print(name, json.dumps({k: v for k, v in res[name].items() if k != "log_fields_compared"}))
ok = all(v["keys_equal"] and v["init_tensors_equal"] and v["loss_trace_equal"] and v["final_weights_equal"] and v["ckpt_cfg_keys_equal"]
         and v["steps_logged"] == [STEPS, STEPS] and v["max_abs_diff_all_logged_fields"] == 0.0 and v["max_abs_diff_final_weights"] == 0.0
         for k, v in res.items() if k != "steps")
res["PASS"] = bool(ok)
json.dump(res, open(f"{CK}/a1_identity{'_' + '_'.join(ONLY) + '_final_tree' if ONLY else ''}.json", "w", encoding="utf-8"), indent=1)
print("T1", "PASS" if ok else "FAIL")
