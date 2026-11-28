"""Tiny end-to-end fine-tune smoke of ACTCONS on the CPU (tiny synthetic world, tiny LS-shape base checkpoint). Runs the trainer of tree_patch as a subprocess:
 (1) --w-act-cons 1.0, 20 steps from the base checkpoint (--init-ckpt, fresh optimiser, lr 1e-4, warm-up): rc 0, every logged value finite, act_cons and act_cons_n in the log (term > 0, n > 0),
     checkpoint cfg EQUAL to the base checkpoint cfg (no new field), state-dict keys EQUAL to the base checkpoint keys, strict load into a model built from that cfg, selfcheck levers and probe, summary keys;
 (2) the same argv with the flag at its default (W = 0): no act_cons keys in log / selfcheck / summary; at step 1 total(W=1) - total(W=0) == 1.0 x act_cons (the term reaches the total);
     the weights after 20 steps DIFFER between W = 1 and W = 0 (control: the a1 identity check can read non-zero);
 (3) MUTATION: a single-file copy of the trainer with the total line deleted: at step 1 total(W=1) - total(W=0) reads 0.0 (RED: the step-1 check fails), act_cons still logged;
 (4) the unchanged v8f_t1_eval.py of tree_patch loads the fine-tuned checkpoint (strict) and runs (finite dumps).
Usage: a4_smoke_finetune.py"""
import json
import os
import subprocess

import numpy as np
import torch

import ac_common as C

AC = C.AC_ROOT
CK = f"{AC}/checks"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
w = lambda *p: "\\".join(p).replace("/", "\\")  # noqa: E731
root = f"{AC}/work/tree_patch"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           TMP=w(AC, "work", "tmp"), TEMP=w(AC, "work", "tmp"), PYTHONPATH=";".join([w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]))
G = ["--arm", "M2b", "--grid", "pool4x10", "--kin-mode", "yaw", "--wm-state", "grid", "--wm-cell-dim", "16", "--exclude", "1:3"]
TINY = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--threads", "1"]
LSF = ["--lead-sup", "on", "--w-lead-sup", "1.0"]
BASE = C.BASE_CKPT
O = f"{AC}/work/tmp/a4"
os.makedirs(O, exist_ok=True)
W = "F:/TanitAD-scratch/v8proof/leadcond/work/tmp/world_c3"                      # the synthetic world written by the leadcond c3 check (eval split, tracks, grid, join); read only


def train(tag, extra, steps, script=None, log_every=5):
    out = f"{O}/{tag}"
    cmd = [PY, script or f"{root}/stack/scripts/v8f_mini_train.py", *TINY, *G, *LSF, "--init-ckpt", BASE, "--lr", "1e-4", "--warmup", "100", "--seed", "0", "--steps", str(steps),
           "--log-every", str(log_every), "--ckpt-every", "100", *extra, "--out", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
    with open(f"{CK}/a4_{tag}.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write((r.stdout or "")[-2000:] + "\nSTDERR\n" + (r.stderr or "")[-2500:] + f"\nRC {r.returncode}\n")
    rows = [json.loads(line) for line in open(f"{out}/train_log.jsonl", encoding="utf-8")] if r.returncode == 0 else []
    return r.returncode, rows, out


res = {}
rc1, rows1, out1 = train("ft_LA", ["--w-act-cons", "1.0"], 20)
bck = torch.load(BASE, map_location="cpu", weights_only=False)
ck = torch.load(f"{out1}/ckpt_last.pt", map_location="cpu", weights_only=False)
fin = all(bool(np.isfinite(x[k])) for x in rows1 for k in x if isinstance(x[k], (int, float)))
sc = json.load(open(f"{out1}/selfcheck.json", encoding="utf-8"))
sm = json.load(open(f"{out1}/summary.json", encoding="utf-8"))
res["LA_finetune"] = {"rc": rc1, "n_log": len(rows1), "all_logged_values_finite": fin, "act_cons_in_every_row": all(("act_cons" in x and "act_cons_n" in x) for x in rows1),
                      "act_cons_min": min(x["act_cons"] for x in rows1), "act_cons_n_min": min(x["act_cons_n"] for x in rows1),
                      "ckpt_cfg_equal_to_base_cfg": ck["cfg"] == bck["cfg"], "cfg_diff": {k: (bck["cfg"].get(k), ck["cfg"].get(k)) for k in set(bck["cfg"]) | set(ck["cfg"]) if bck["cfg"].get(k) != ck["cfg"].get(k)},
                      "state_keys_equal_to_base": list(ck["model"]) == list(bck["model"]), "init_ckpt_fresh_keys": (ck.get("init_ckpt") or {}).get("fresh_keys"),
                      "selfcheck_levers": {k: v for k, v in sc["levers"].items() if "act_cons" in k}, "selfcheck_probe": sc.get("act_cons_probe"), "summary": {k: v for k, v in sm.items() if "act_cons" in k},
                      "weights_moved_from_base": max(float((ck["model"][k].float() - bck["model"][k].float()).abs().max()) for k in ck["model"])}
# strict load into a model built from the checkpoint cfg (the loader of the eval tools)
sys_path = [w(root, "stack"), w(root, "taniteval"), w(root, "stack", "scripts")]
C.use_tree("patch")
from tanitad.models import v8f_mini as M  # noqa: E402
mdl = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
mdl.load_state_dict(ck["model"], strict=True)
res["LA_finetune"]["strict_load_into_model_built_from_ckpt_cfg"] = True
# (2) the default-flag control
rc0, rows0, out0 = train("ft_L0", [], 20)
s0 = json.load(open(f"{out0}/selfcheck.json", encoding="utf-8"))
ck0 = torch.load(f"{out0}/ckpt_last.pt", map_location="cpu", weights_only=False)
res["L0_default_control"] = {"rc": rc0, "no_act_cons_keys_in_log": not any("act_cons" in k for x in rows0 for k in x), "no_act_cons_in_selfcheck": "act_cons_probe" not in s0 and not [k for k in s0["levers"] if "act_cons" in k],
                             "ckpt_cfg_equal_to_base_cfg": ck0["cfg"] == bck["cfg"], "state_keys_equal_to_base": list(ck0["model"]) == list(bck["model"]),
                             "weights_differ_W1_vs_W0_max_abs": max(float((ck["model"][k].float() - ck0["model"][k].float()).abs().max()) for k in ck0["model"])}
# step-1 comparison: total(W=1) - total(W=0) == act_cons
_, a1, _ = train("s1_W1", ["--w-act-cons", "1.0"], 1, log_every=1)
_, a0, _ = train("s1_W0", [], 1, log_every=1)
d1 = a1[0]["total"] - a0[0]["total"]
res["step1_total_moves_by_W_times_term"] = {"total_W1": a1[0]["total"], "total_W0": a0[0]["total"], "delta": d1, "act_cons": a1[0]["act_cons"], "abs_diff": abs(d1 - a1[0]["act_cons"]), "ok": bool(abs(d1 - a1[0]["act_cons"]) < 1e-3 and a1[0]["act_cons"] > 0)}
# (3) mutation: the total line deleted, in a single-file copy of the trainer
src = open(f"{root}/stack/scripts/v8f_mini_train.py", encoding="utf-8", newline="").read()
old = 'T["total"] = tot_ac + a.w_act_cons * ac_r["term"]'
assert src.count(old) == 1
os.makedirs(f"{AC}/work/mut/stack/scripts", exist_ok=True)
mp = f"{AC}/work/mut/stack/scripts/v8f_mini_train.py"
open(mp, "w", encoding="utf-8", newline="").write(src.replace(old, 'T["total"] = tot_ac'))
_, m1, _ = train("s1_mutW1", ["--w-act-cons", "1.0"], 1, script=mp, log_every=1)
dm = m1[0]["total"] - a0[0]["total"]
res["mutation_total_line_deleted"] = {"delta_vs_W0": dm, "act_cons_still_logged": m1[0]["act_cons"], "red": bool(abs(dm - m1[0]["act_cons"]) > 1e-3)}
# (4) the unchanged eval tool loads the checkpoint
ev = f"{O}/ev_LA"
os.makedirs(ev, exist_ok=True)
e = subprocess.run([PY, f"{root}/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", f"{out1}/ckpt_last.pt", "--tag", "LA", "--tracks", f"{W}/native_tracks.pt", "--eval-cache", W, "--eval-grid", f"{W}/eval_grid_windows.json",
                    "--eval-join", f"{W}/join_eval.npz", "--out-dir", ev, "--max-windows", "4", "--batch", "4", "--n-boot", "20", "--device", "cpu", "--no-analyze"], env=env, capture_output=True, text=True, encoding="utf-8")
dumps = {f: np.load(f"{ev}/dump_LA/{f}", allow_pickle=True) for f in sorted(os.listdir(f"{ev}/dump_LA"))} if e.returncode == 0 else {}
ev_fin = bool(dumps) and all(np.isfinite(np.asarray(v[k], dtype=np.float64)).all() for v in dumps.values() for k in v.files if np.asarray(v[k]).dtype.kind == "f")
res["t1_eval_loads_and_runs"] = {"rc": e.returncode, "n_dump_files": len(dumps), "finite": ev_fin, "tail": (e.stdout or e.stderr or "")[-200:].replace("\n", " | ")}
la, l0, st, mu = res["LA_finetune"], res["L0_default_control"], res["step1_total_moves_by_W_times_term"], res["mutation_total_line_deleted"]
res["PASS"] = bool(la["rc"] == 0 and la["all_logged_values_finite"] and la["act_cons_in_every_row"] and la["act_cons_min"] > 0 and la["act_cons_n_min"] > 0 and la["ckpt_cfg_equal_to_base_cfg"] and la["state_keys_equal_to_base"]
                   and la["init_ckpt_fresh_keys"] in (None, []) and la["selfcheck_probe"]["pass"] and la["selfcheck_levers"] == {"w_act_cons": 1.0, "act_cons_max": 1.5}
                   and l0["rc"] == 0 and l0["no_act_cons_keys_in_log"] and l0["no_act_cons_in_selfcheck"] and l0["ckpt_cfg_equal_to_base_cfg"] and l0["state_keys_equal_to_base"] and l0["weights_differ_W1_vs_W0_max_abs"] > 0.0
                   and st["ok"] and mu["red"] and res["t1_eval_loads_and_runs"]["rc"] == 0 and res["t1_eval_loads_and_runs"]["finite"])
C.dump(f"{CK}/a4_smoke_finetune.json", res)
print(json.dumps(res, default=str)[:3500])
print("A4", "PASS" if res["PASS"] else "FAIL")
