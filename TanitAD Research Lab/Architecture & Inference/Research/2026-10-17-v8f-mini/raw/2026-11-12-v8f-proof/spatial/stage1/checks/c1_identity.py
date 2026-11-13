"""c1 identity: default flags, same seed, tree_base vs tree_patch. State-dict key list, init tensors, 200-step loss trace (every logged field but wall time), final weights: bit-for-bit."""
import json
import os
import subprocess
import sys

import torch

S = "F:/TanitAD-scratch/v8proof/spatial/stage1"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
res = {"steps": STEPS}
for arm in ("M2b", "M1"):
    outs = {}
    for t in ("tree_base", "tree_patch"):
        root = f"{S}/{t}"
        w = lambda *p: "\\".join(p).replace("/", "\\")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4",
                   PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
        out = f"{S}/tmp/c1/{arm}_{t}"
        r = subprocess.run([PY, f"{S}/checks/c1_worker.py", root, out, arm, str(STEPS)], env=env, capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr[-2000:]
        outs[t] = out
    a, b = outs["tree_base"], outs["tree_patch"]
    ka, kb = (json.load(open(f"{x}/init_keys.json", encoding="utf-8")) for x in (a, b))
    sa, sb = (torch.load(f"{x}/init_state.pt") for x in (a, b))
    init_equal = all(torch.equal(sa[k], sb[k]) for k in sa) and list(sa) == list(sb)
    la, lb = ([json.loads(l) for l in open(f"{x}/run/train_log.jsonl", encoding="utf-8")] for x in (a, b))
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("sec", "sec_per_step", "elapsed")} for r in rows]
    trace_equal = strip(la) == strip(lb)
    ca, cb = (torch.load(f"{x}/run/ckpt_last.pt", weights_only=False)["model"] for x in (a, b))
    final_equal = list(ca) == list(cb) and all(torch.equal(ca[k], cb[k]) for k in ca)
    cfa, cfb = (torch.load(f"{x}/run/ckpt_last.pt", weights_only=False)["cfg"] for x in (a, b))
    res[arm] = {"keys_equal": ka["keys"] == kb["keys"], "n_keys": len(ka["keys"]), "n_params": [ka["n_params"], kb["n_params"]],
                "init_tensors_equal": init_equal, "steps_logged": [len(la), len(lb)], "loss_trace_equal": trace_equal,
                "max_abs_total_diff": max(abs(x["total"] - y["total"]) for x, y in zip(la, lb)),
                "final_weights_equal": final_equal, "ckpt_cfg_keys_equal": sorted(cfa) == sorted(cfb),
                "total_first_last": [la[0]["total"], la[-1]["total"]]}
    print(arm, json.dumps(res[arm]))
ok = all(v["keys_equal"] and v["init_tensors_equal"] and v["loss_trace_equal"] and v["final_weights_equal"] and v["ckpt_cfg_keys_equal"] and v["steps_logged"] == [STEPS, STEPS]
         for k, v in res.items() if k != "steps")
res["PASS"] = bool(ok)
json.dump(res, open(f"{S}/checks/c1_identity.json", "w", encoding="utf-8"), indent=1)
print("C1", "PASS" if ok else "FAIL")
